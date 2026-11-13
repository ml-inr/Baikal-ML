"""Hit selection before the fit: causality clique, triggers, scanfit.

Ports of the reco-muon task chain of BARS v2.3.2 up to the fit, in its order
(`programs/reco-muon/reco.cc`, with `UseScanfitMask 1`, `UseGroupMask 0`,
`UseMulticlusterGroupMask 0` as in both the DATA and MC resource files):

    BCausalityClique -> BLocalTriggerSelect -> BGlobalTriggerSelect
      -> BInitializeCausScanGrid -> BScanFitClique -> BOfflineTriggerCut

The channel mask filter that precedes them is not reproduced: there is no DQM or
section mask in the HDF5 files, so every channel counts as on.

All functions take hits in BARS' pulse order -- by channel, then by time -- which
`fit.fit_event` establishes.  The order matters: clique enumeration prunes by the
largest clique found so far.

Two BARS quirks are reproduced on purpose, because they decide which vertices
survive:

* `BARS::Muon::z_dist_range` skips the hits that ARE in the clique, so the
  60 m z-distance requirement is applied to the hits outside it -- every hit of
  the event, noise included, not only the clique-masked ones.
* the scanfit time limit (`Scanfit.MaxScanTime`) is a wall-clock guard on the C++
  code and is not reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np
from numba import njit

from .cliques import enumerate_cliques, select_from_max_clique
from .config import FitConfig
from .physics import C_BAIKAL, C_VACUUM, K_SCANFIT

#: Channels per section and per string of a GVD cluster.  BLocalTriggerSelect
#: looks for neighbours inside a section, BGlobalTriggerSelect counts strings.
#: (assumed: 3 sections of 12 OMs per 36-OM string; the string divisor 36 is the
#: one the whole repository uses.)
CHANNELS_PER_SECTION = 12
CHANNELS_PER_STRING = 36

#: BARS __MAX_HITS_PER_EVENT__: larger events are dropped by the clique filter.
MAX_HITS = 2048

#: BScanFitClique::MinimizeMEst constants
MEST_TOLERANCE = 0.4
MEST_MAX_ITER = 1000
MEST_FREEPAR = 2
MEST_STEP = 2.0
MEST_OUT_OF_RANGE = 1e9
MEST_RANGE = 30.0          # fRangeX0Y0: max deviation from the initial (x0, y0), m
#: BARS::Caus::MuDirectionalTest: maximum transverse OM separation, m (3 * 22)
DIRECTIONAL_RMAX = 66.0


# ── BCausalityClique ───────────────────────────────────────────────────────────
def causality_graph(t: np.ndarray, pos: np.ndarray, channel: np.ndarray,
                    window: float, long_window: float | None,
                    enabled: np.ndarray | None = None) -> np.ndarray:
    """BARS::Caus::CausalityTest for every pair, as a symmetric boolean matrix.

    `long_window=None` is the conventional test (type 0); a number switches on the
    long-range condition (type 1) that BCausalityClique uses.  Same-channel pairs
    are never causal.  A pair is skipped when either hit sits on a channel the
    channel mask switches off, which leaves those hits isolated -- the same effect
    as `BCausalityClique::Filter`'s two `continue`s.
    """
    dt = np.abs(t[:, None] - t[None, :])
    d_r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    causal = (dt - d_r / C_BAIKAL) < window
    if long_window is not None:
        causal &= ~(np.abs(dt - d_r / C_VACUUM) > long_window)
    causal &= channel[:, None] != channel[None, :]
    if enabled is not None:
        causal &= enabled[:, None] & enabled[None, :]
    np.fill_diagonal(causal, False)
    return causal


def causality_clique_mask(t, pos, channel, cfg: FitConfig, enabled=None):
    """BCausalityClique::Filter: union of the near-largest causality cliques.

    Returns `(mask, overflow)`.
    """
    n = len(t)
    if n < cfg.clique_min_size:                  # BFilter::Process, fMinChanNum
        return np.zeros(n, dtype=bool), False
    adj = causality_graph(t, pos, channel, cfg.clique_caus_window,
                          cfg.clique_long_window, enabled)
    cliques, _, overflow = enumerate_cliques(adj, cfg.clique_min_size)
    return select_from_max_clique(cliques, cfg.clique_tolerance), overflow


# ── triggers ───────────────────────────────────────────────────────────────────
def local_trigger(q, t, channel, mask, cfg: FitConfig) -> bool:
    """BLocalTriggerSelect::TestEvent on the masked hits.

    A hit above the high threshold with a hit on an adjacent channel of the same
    section, at or above the low threshold, within the time window.
    """
    use = mask & ~(q < cfg.lt_low_threshold)
    ht = np.flatnonzero(use & (q > cfg.lt_high_threshold))
    if ht.size == 0:
        return False
    lt = np.flatnonzero(use)
    for i in ht:
        ch = channel[i]
        local = ch % CHANNELS_PER_SECTION
        for nb in (ch - 1, ch + 1):
            if nb == ch - 1 and local == 0:
                continue
            if nb == ch + 1 and local + 1 >= CHANNELS_PER_SECTION:
                continue
            near = lt[channel[lt] == nb]
            if near.size and np.any(np.abs(t[i] - t[near]) < cfg.lt_time_window):
                return True
    return False


def channel_string_counts(channel: np.ndarray, mask: np.ndarray,
                          enabled: np.ndarray | None = None) -> tuple[int, int]:
    """Distinct channels and strings among the hits that pass `mask`.

    `enabled` is the channel mask: both trigger tasks skip channels it switches
    off.  Hits selected by a clique are always on an enabled channel, so this only
    matters if the function is called with some other hit set.
    """
    use = mask if enabled is None else (mask & enabled)
    ch = channel[use]
    return (len(np.unique(ch)), len(np.unique(ch // CHANNELS_PER_STRING)))


def passes_channel_trigger(channel, mask, cfg: FitConfig, enabled=None) -> bool:
    """BGlobalTriggerSelect / BOfflineTriggerCut: enough channels and strings."""
    n_ch, n_str = channel_string_counts(channel, mask, enabled)
    return n_ch >= cfg.min_channels and n_str >= cfg.min_strings


# ── BInitializeCausScanGrid ────────────────────────────────────────────────────
def scan_grid(theta_step_deg: float, phi_step_deg: float):
    """Vertex directions of the full-sky grid, in BARS' order and float32 steps.

    `for (float phi = 0; phi < 2*pi; phi += step)` outside, theta inside, with
    `pi` a float -- the accumulation is done in float32 to land on the same
    number of vertices.  Returns float64 arrays (theta, phi) of the direction of
    motion.
    """
    f32 = np.float32
    pi = f32(np.pi)
    two_pi = f32(2) * pi
    t_step = f32(theta_step_deg * np.pi / 180)
    p_step = f32(phi_step_deg * np.pi / 180)
    thetas, phis = [], []
    phi = f32(0)
    while phi < two_pi:
        theta = f32(0)
        while theta < pi:
            thetas.append(theta)
            phis.append(phi)
            theta = f32(theta + t_step)
        phi = f32(phi + p_step)
    return np.asarray(thetas, dtype=np.float64), np.asarray(phis, dtype=np.float64)


def vertex_rotations(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """TRotation of each vertex: Ry(pi - theta) Rz(-phi), shape (V, 3, 3).

    In the rotated frame the direction of motion is -z.
    """
    a = np.pi - theta
    ca, sa = np.cos(a), np.sin(a)
    cp, sp = np.cos(-phi), np.sin(-phi)
    rz = np.zeros((len(theta), 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1] = cp, -sp, sp, cp
    rz[:, 2, 2] = 1
    ry = np.zeros((len(theta), 3, 3))
    ry[:, 0, 0], ry[:, 0, 2], ry[:, 2, 0], ry[:, 2, 2] = ca, sa, -sa, ca
    ry[:, 1, 1] = 1
    return ry @ rz


# ── BScanFitClique ─────────────────────────────────────────────────────────────
@njit(cache=True)
def _mest(x0, y0, x0_init, y0_init, xr, yr, zr, t, sigma):
    """BARS::Muon::MEstimator_X0Y0 inside the BScanFitClique loss lambda."""
    if abs(x0 - x0_init) > MEST_RANGE or abs(y0 - y0_init) > MEST_RANGE:
        return MEST_OUT_OF_RANGE
    n = t.shape[0]
    t_th = np.empty(n)
    t0 = 0.0
    for i in range(n):
        rho = np.sqrt((xr[i] - x0) ** 2 + (yr[i] - y0) ** 2)
        t_th[i] = -zr[i] / C_VACUUM + rho / C_VACUUM * K_SCANFIT
        t0 += t[i] - t_th[i]
    t0 /= n
    m = 0.0
    for i in range(n):
        d = (t_th[i] + t0 - t[i]) / sigma
        m += log(1.0 + d * d / 2.0)
    if n > MEST_FREEPAR:
        m /= n - MEST_FREEPAR
    return m


@njit(cache=True)
def _dnrm2(a, b):
    """gslcblas cblas_dnrm2 for two elements (scaled sum of squares)."""
    scale = 0.0
    ssq = 1.0
    for x in (a, b):
        if x != 0.0:
            ax = abs(x)
            if scale < ax:
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax)
                scale = ax
            else:
                ssq += (ax / scale) * (ax / scale)
    return scale * np.sqrt(ssq)


@njit(cache=True)
def _compute_center(x1, center):
    """GSL compute_center: sum the corners with daxpy, then dscal by 1/P."""
    center[0] = 0.0
    center[1] = 0.0
    for i in range(3):
        center[0] += 1.0 * x1[i, 0]
        center[1] += 1.0 * x1[i, 1]
    alpha = 1.0 / 3.0
    center[0] *= alpha
    center[1] *= alpha


@njit(cache=True)
def _simplex_size(x1, center):
    """GSL compute_size, returning the squared size S2."""
    ss = 0.0
    for i in range(3):
        t = _dnrm2(x1[i, 0] + -1.0 * center[0], x1[i, 1] + -1.0 * center[1])
        ss += t * t
    return ss / 3.0


@njit(cache=True)
def minimize_mest(x0_init, y0_init, xr, yr, zr, t, sigma):
    """BScanFitClique::MinimizeMEst: GSL nmsimplex2 on the M-estimator.

    Port of `gsl_multimin_fminimizer_nmsimplex2` for two parameters (P = 3
    corners), driven as BScanFitClique drives it: step 2 m, stop when the simplex
    size is below 0.4, at most 1000 iterations.  Returns `(ok, x0, y0, fval)`;
    `ok` is False when the iteration limit is reached (GSL_CONTINUE), which BARS
    treats as a failed clique.  (assumed: GSL source recalled, not checked
    against the library; see the README.)
    """
    x1 = np.empty((3, 2))
    y1 = np.empty(3)
    x1[0, 0], x1[0, 1] = x0_init, y0_init
    x1[1, 0], x1[1, 1] = x0_init + MEST_STEP, y0_init
    x1[2, 0], x1[2, 1] = x0_init, y0_init + MEST_STEP
    for i in range(3):
        y1[i] = _mest(x1[i, 0], x1[i, 1], x0_init, y0_init, xr, yr, zr, t, sigma)
    center = np.empty(2)
    _compute_center(x1, center)
    s2 = _simplex_size(x1, center)
    xc = np.empty(2)
    xc2 = np.empty(2)
    p = 3.0

    ok = False
    it = 0
    while True:
        it += 1
        # highest, second highest, lowest
        dhi = y1[0]
        dlo = y1[0]
        hi = 0
        lo = 0
        ds_hi = y1[1]
        s_hi = 1
        for i in range(1, 3):
            val = y1[i]
            if val < dlo:
                dlo = val
                lo = i
            elif val > dhi:
                ds_hi = dhi
                s_hi = hi
                dhi = val
                hi = i
            elif val > ds_hi:
                ds_hi = val
                s_hi = i

        # reflection, coeff -1: xc = (1-c) P/(P-1) center + (P c - 1)/(P-1) x_hi
        val = _corner(-1.0, x1, center, hi, xc, p, x0_init, y0_init,
                      xr, yr, zr, t, sigma)
        if val < y1[lo]:
            val2 = _corner(-2.0, x1, center, hi, xc2, p, x0_init, y0_init,
                           xr, yr, zr, t, sigma)
            if val2 < y1[lo]:
                s2 = _update(x1, y1, center, s2, hi, xc2, val2, p)
            else:
                s2 = _update(x1, y1, center, s2, hi, xc, val, p)
        elif val > y1[s_hi]:
            if val <= y1[hi]:
                s2 = _update(x1, y1, center, s2, hi, xc, val, p)
            val2 = _corner(0.5, x1, center, hi, xc2, p, x0_init, y0_init,
                           xr, yr, zr, t, sigma)
            if val2 <= y1[hi]:
                s2 = _update(x1, y1, center, s2, hi, xc2, val2, p)
            else:
                # contract_by_best
                for i in range(3):
                    if i != lo:
                        x1[i, 0] = 0.5 * (x1[i, 0] + x1[lo, 0])
                        x1[i, 1] = 0.5 * (x1[i, 1] + x1[lo, 1])
                        y1[i] = _mest(x1[i, 0], x1[i, 1], x0_init, y0_init,
                                      xr, yr, zr, t, sigma)
                _compute_center(x1, center)
                s2 = _simplex_size(x1, center)
        else:
            s2 = _update(x1, y1, center, s2, hi, xc, val, p)

        if s2 > 0:
            size = np.sqrt(s2)
        else:
            s2 = _simplex_size(x1, center)
            size = np.sqrt(s2)
        if size < MEST_TOLERANCE:
            ok = True
            break
        if it >= MEST_MAX_ITER:
            break

    best = 0
    for i in range(1, 3):
        if y1[i] < y1[best]:
            best = i
    return ok, x1[best, 0], x1[best, 1], y1[best]


@njit(cache=True)
def _corner(coeff, x1, center, corner, out, p, x0_init, y0_init, xr, yr, zr, t,
            sigma):
    alpha = (1.0 - coeff) * p / (p - 1.0)
    beta = (p * coeff - 1.0) / (p - 1.0)
    out[0] = alpha * center[0] + beta * x1[corner, 0]
    out[1] = alpha * center[1] + beta * x1[corner, 1]
    return _mest(out[0], out[1], x0_init, y0_init, xr, yr, zr, t, sigma)


@njit(cache=True)
def _update(x1, y1, center, s2, i, x, val, p):
    """GSL update_point: replace corner i, update centre and squared size."""
    d0 = x[0] + -1.0 * x1[i, 0]
    d1 = x[1] + -1.0 * x1[i, 1]
    m0 = x1[i, 0] + -1.0 * center[0]
    m1 = x1[i, 1] + -1.0 * center[1]
    d = _dnrm2(d0, d1)
    xmcd = 0.0
    xmcd += m0 * d0
    xmcd += m1 * d1
    s2 += (2.0 / p) * xmcd + ((p - 1.0) / p) * (d * d / p)
    alpha = 1.0 / p
    center[0] += -alpha * x1[i, 0]
    center[1] += -alpha * x1[i, 1]
    center[0] += alpha * x[0]
    center[1] += alpha * x[1]
    x1[i, 0] = x[0]
    x1[i, 1] = x[1]
    y1[i] = val
    return s2


@dataclass
class ScanResult:
    """What BScanFitClique leaves behind for BRecoMuonProducer."""

    theta: np.ndarray                  # (V,) direction of motion, rad
    phi: np.ndarray                    # (V,)
    passed: np.ndarray                 # (V,) bool, vertex passed the cut
    mest: np.ndarray                   # (V,) M-estimator of the chosen clique
    selected: list = field(default_factory=list)  # per vertex: bool mask or None
    best: int = -1                     # index of the best vertex, -1 if none
    max_clique_size: int = 0
    overflow: bool = False

    @property
    def mask(self) -> np.ndarray | None:
        return None if self.best < 0 else self.selected[self.best]


def scanfit(t: np.ndarray, pos: np.ndarray, clique_mask: np.ndarray,
            cfg: FitConfig) -> ScanResult:
    """BScanFitClique::Filter over the full-sky grid."""
    theta, phi = scan_grid(cfg.scan_theta_step_deg, cfg.scan_phi_step_deg)
    n_v = len(theta)
    rot = vertex_rotations(theta, phi)                      # (V, 3, 3)
    res = ScanResult(theta=theta, phi=phi, passed=np.zeros(n_v, dtype=bool),
                     mest=np.full(n_v, np.nan), selected=[None] * n_v)

    idx = np.flatnonzero(clique_mask)                       # event indices
    m = len(idx)
    t_c = np.ascontiguousarray(t[idx])
    r_c = np.einsum("vij,nj->vni", rot, pos[idx])           # (V, m, 3)
    z_all = np.einsum("vj,nj->vn", rot[:, 2, :], pos)       # (V, N)
    dt = t_c[:, None] - t_c[None, :]
    diag = np.arange(m)
    w = cfg.scan_caus_window

    vertex_cliques = []
    max_size = 0
    for v in range(n_v):
        # BARS::Caus::MuDirectionalTest for all pairs of clique-masked hits
        dz = r_c[v, :, None, 2] - r_c[v, None, :, 2]
        d_r = np.hypot(r_c[v, :, None, 0] - r_c[v, None, :, 0],
                       r_c[v, :, None, 1] - r_c[v, None, :, 1])
        adj = (-K_SCANFIT * d_r - dz) / C_VACUUM - w <= dt
        adj &= dt <= (K_SCANFIT * d_r - dz) / C_VACUUM + w
        adj &= d_r <= DIRECTIONAL_RMAX
        adj[diag, diag] = False
        cliques, size, overflow = enumerate_cliques(adj, cfg.scan_min_clique_size)
        vertex_cliques.append(cliques)
        max_size = max(max_size, size)
        res.overflow |= overflow
    res.max_clique_size = max_size
    min_hits = max_size - cfg.scan_clique_gap

    for v in range(n_v):
        best_mest = 1e6
        chosen = None
        for clique in vertex_cliques[v]:
            if clique.sum() < min_hits:
                continue
            in_clique = np.zeros(len(t), dtype=bool)
            in_clique[idx[clique]] = True
            # z_dist_range: over the hits NOT in the clique (BARS quirk)
            z_out = z_all[v, ~in_clique]
            z_dist = (z_out.max() - z_out.min()) if z_out.size else -2e6
            if z_dist < cfg.scan_zdist_min:
                continue
            xr = np.ascontiguousarray(r_c[v, clique, 0])
            yr = np.ascontiguousarray(r_c[v, clique, 1])
            zr = np.ascontiguousarray(r_c[v, clique, 2])
            ok, _, _, fval = minimize_mest(xr.mean(), yr.mean(), xr, yr, zr,
                                           t_c[clique], cfg.scan_time_sigma)
            if not ok or fval > best_mest:
                continue
            best_mest = fval
            chosen = in_clique
        if chosen is not None:
            res.passed[v] = True
            res.mest[v] = best_mest
            res.selected[v] = chosen

    good = np.flatnonzero(res.passed)
    if good.size:
        # first vertex with the strictly smallest mest
        res.best = int(good[np.argmin(res.mest[good])])
    return res
