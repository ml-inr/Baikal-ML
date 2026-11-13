"""The track fit itself: a port of `bmuon/BMuonCandidate.cc` (BARS v2.3.2).

Only what the reco-muon chain calls is ported: quality functions of type 2 (time
chi2) and 4 (time chi2 plus a charge-weighted distance term), `Refit` with all
five parameters free, `RefitExclusive`, `RecollectHits`, `OrderImpulses`,
`GetFirstApproximation`.

The candidate's state is kept exactly as BARS keeps it, including three side
effects that change the numbers and are therefore reproduced, not fixed:

* `fSumCharge` is never reset by `SetImpulseArray`.  `RefitExclusive` calls it
  once per hit, so by the final fit the normalisation of the type-4 charge term,
  `fSumCharge / (N - 6)`, is inflated about N-fold and the charge term nearly
  vanishes.
* `fX0, fY0` are measured from the hottest hit's OM (`fHotOMLocation`) but are
  only recomputed by `SetDirection`, `SetRefPoint`, the first approximation and
  the end of a fit -- not when adding or removing a hit moves the hottest OM.  A
  fit started after such a move starts from a displaced track.
* the fit parameters are (x0, y0, theta, phi, t) about the hottest OM, and the
  track point is rebuilt from the angles of the direction *vector*, so a theta
  parameter that wanders negative flips the x0 axis (`BHelperFunctions::GetRefXYZ`).

Minimisation is Minuit2 through iminuit, set up as `BMuonCandidate::Refit` sets
ROOT's `Minuit2Minimizer("Minuit2", "MINIMIZE")`: combined Migrad/Simplex,
strategy 1, tolerance 5, precision 1e-7, at most 1000 calls, errordef 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from iminuit import Minuit
from numba import njit

from .physics import (
    direction_of, om_light_cos, om_sensitivity, propagation_time, ref_xyz,
    time_at_pca, track_distance, vector_angles, x0y0_of,
)

# BMuonCandidate::getModQuality and Reset
JITTER = 3.0            # fJitter, ns
A_ZERO = 50.0           # aZero: charge saturation, p.e.
D_ONE = 1.0             # dOne, m
TYPE4_WEIGHT = 0.3

# BMuonCandidate::RefitBase
STEPS = (1.0, 1.0, 0.01, 0.01, 0.01)
POINT_MARGIN = 100.0                  # m, limits on x0 and y0
ANGLE_MARGIN = 50.0 * pi / 180        # rad, limits on theta and phi
PARAM_NAMES = ("ax0", "ay0", "bTheta", "bPhi", "t0")

# BMuonCandidate::Refit
MAX_CALLS = 1000
TOLERANCE = 5.0
PRECISION = 1e-7

# Strings per channel number (BStaticConfig::GetChannel().GetString())
CHANNELS_PER_STRING = 36


@njit(cache=True)
def quality_terms(t0, rx, ry, rz, dx, dy, dz, pos, t, q, fit_type, charge_norm,
                  out):
    """Per-hit getModQuality for types 2 and 4, written into `out`."""
    for i in range(t.shape[0]):
        px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
        d = t0 + propagation_time(rx, ry, rz, dx, dy, dz, px, py, pz) - t[i]
        qual = d * d / (JITTER * JITTER)
        if fit_type == 4:
            dist = track_distance(rx, ry, rz, dx, dy, dz, px, py, pz)
            sens = om_sensitivity(om_light_cos(rx, ry, rz, dx, dy, dz, px, py, pz))
            qc = q[i] / sens
            a = A_ZERO * qc / sqrt(A_ZERO * A_ZERO + qc * qc)
            dd = sqrt(D_ONE * D_ONE + dist * dist)
            qual += TYPE4_WEIGHT * a * dd / charge_norm
        out[i] = qual


@njit(cache=True)
def fit_function(par, hot, pos, t, q, fit_type, charge_norm):
    """BMuonCandidate::trackFitFunction."""
    dx, dy, dz = direction_of(par[2], par[3])
    rx, ry, rz = ref_xyz(hot[0], hot[1], hot[2], par[0], par[1], dx, dy, dz)
    out = np.empty(t.shape[0])
    quality_terms(par[4], rx, ry, rz, dx, dy, dz, pos, t, q, fit_type,
                  charge_norm, out)
    return out.sum()


def error_deg(variance: float) -> float:
    """BARS' `sqrt(err[i]) * 180/pi`, NaN included.

    A valid Minuit minimum can still come with a covariance whose diagonal is
    negative (iminuit reports one that has been made positive definite, among
    others).  In C++ `sqrt` of a negative is NaN and the fit carries on; Python
    raises, so the branch is spelled out here.
    """
    return sqrt(variance) * 180 / pi if variance >= 0 else float("nan")


def charge_norm(sum_charge: float, n_hits: int) -> float:
    """`fSumCharge / (fImpulses_id.size() - 6)`, with the unsigned size_t."""
    denom = n_hits - 6
    if denom == 0:
        return float("inf")
    if denom < 0:
        denom += 2 ** 64
    return sum_charge / denom


@dataclass
class MinuitOutcome:
    values: np.ndarray
    cov: np.ndarray            # 5x5, zeros when the minimum is invalid
    n_calls: int
    cov_status: int            # ROOT Minuit2Minimizer::CovMatrixStatus
    min_status: int            # ROOT Minuit2Minimizer::Status
    valid: bool


def _make_minuit(fcn, start, errors, limits, strategy: int) -> Minuit:
    m = Minuit(fcn, np.asarray(start, dtype=float), name=PARAM_NAMES)
    m.errordef = Minuit.LEAST_SQUARES
    m.errors = errors
    m.limits = limits
    m.tol = TOLERANCE
    m.precision = PRECISION
    m.strategy = strategy
    m.print_level = 0
    return m


def _outcome(m: Minuit, n_calls: int) -> MinuitOutcome:
    fmin = m.fmin
    if fmin.has_accurate_covar:
        cov_status = 3
    elif fmin.has_made_posdef_covar:
        cov_status = 2
    elif fmin.has_covariance and fmin.has_posdef_covar:
        cov_status = 1
    elif fmin.has_covariance:
        cov_status = 0
    else:
        cov_status = -1
    status = 0
    if fmin.has_made_posdef_covar:
        status = 1
    if fmin.hesse_failed:
        status = 2
    if fmin.is_above_max_edm:
        status = 3
    if fmin.has_reached_call_limit:
        status = 4
    if not fmin.is_valid and status == 0:
        status = 5
    cov = np.zeros((5, 5))
    if fmin.is_valid and m.covariance is not None:
        cov = np.array(m.covariance)
    return MinuitOutcome(values=np.array(m.values), cov=cov, n_calls=n_calls,
                         cov_status=cov_status, min_status=status,
                         valid=fmin.is_valid)


def minuit2_minimize(fcn, start, limits) -> MinuitOutcome:
    """ROOT's Minuit2 "MINIMIZE" (CombinedMinimumBuilder) through iminuit.

    Migrad at strategy 1; if its minimum is invalid, Simplex from the original
    seed at strategy 2 and Migrad again from there, the three sharing one call
    budget, falling back to the Simplex result.  The budget sharing across
    separate iminuit objects is an approximation of ROOT's single counter.
    """
    m = _make_minuit(fcn, start, STEPS, limits, strategy=1)
    m.migrad(ncall=MAX_CALLS, iterate=1)
    n_calls = m.fmin.nfcn
    if m.fmin.is_valid:
        return _outcome(m, n_calls)
    s = _make_minuit(fcn, start, STEPS, limits, strategy=2)
    s.simplex(ncall=max(MAX_CALLS - n_calls, 1))
    n_calls += s.fmin.nfcn
    if not s.fmin.is_valid:
        return _outcome(s, n_calls)
    m2 = _make_minuit(fcn, np.array(s.values), np.array(s.errors), limits,
                      strategy=2)
    m2.migrad(ncall=max(MAX_CALLS - n_calls, 1), iterate=1)
    n_calls += m2.fmin.nfcn
    if not m2.fmin.is_valid:
        return _outcome(s, n_calls)
    return _outcome(m2, n_calls)


class Candidate:
    """BMuonCandidate: a set of hits and a track, with BARS' bookkeeping."""

    def __init__(self, q: np.ndarray, t: np.ndarray, pos: np.ndarray,
                 channel: np.ndarray) -> None:
        # event arrays, in BARS pulse order
        self.q, self.t, self.pos, self.channel = q, t, pos, channel
        self.ids: list[int] = []
        self.sum_charge = 0.0
        self.hot_charge = 0.0
        self.hot = np.zeros(3)
        self.ref_point = np.zeros(3)
        self.direction = np.array([1.0, 0.0, 0.0])
        self.ref_time = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.theta_deg = 0.0
        self.phi_deg = 0.0
        self.theta_err_deg = 0.0
        self.phi_err_deg = 0.0
        self.n_calls = 0
        self.cov_status = 0
        self.min_status = 0

    # ── hit bookkeeping ────────────────────────────────────────────────────
    @property
    def n_hits(self) -> int:
        return len(self.ids)

    def add_impulse(self, i: int) -> None:
        self.ids.append(int(i))
        self.sum_charge += self.q[i]
        if self.q[i] > self.hot_charge:
            self.hot_charge = self.q[i]
            self.hot = self.pos[i].copy()

    def _recompute_hot(self) -> None:
        self.hot_charge = 0.0
        for i in self.ids:
            if self.q[i] > self.hot_charge:
                self.hot_charge = self.q[i]
                self.hot = self.pos[i].copy()

    def remove_impulse(self, k: int) -> None:
        """RemoveImpulse(k): k is a position in the hit list."""
        self.sum_charge -= self.q[self.ids[k]]
        del self.ids[k]
        self._recompute_hot()

    def set_impulse_array(self, ids: list[int]) -> None:
        """SetImpulseArray(vector<Int_t>): adds to fSumCharge without resetting."""
        self.ids = list(ids)
        self.sum_charge += float(self.q[self.ids].sum()) if self.ids else 0.0
        self._recompute_hot()

    def order_impulses(self, min_ampl: float = 0.0) -> None:
        """OrderImpulses: by time, dropping q <= min_ampl.

        Placement is by the number of strictly earlier hits, into a vector
        resized in place; equal times collide and the unfilled slot keeps its
        old entry.  Reproduced as written.
        """
        sel = [i for i in self.ids if self.q[i] > min_ampl]
        out = self.ids[:len(sel)]
        times = self.t[sel]
        for j, i in enumerate(sel):
            out[int((times[j] > times).sum())] = i
        self.ids = out

    # ── track bookkeeping ──────────────────────────────────────────────────
    def _update_x0y0(self) -> None:
        self.x0, self.y0 = x0y0_of(*self.hot, *self.ref_point, *self.direction)

    def set_direction(self, d) -> None:
        self.direction = np.asarray(d, dtype=float).copy()
        self._update_x0y0()

    def set_ref_point(self, p) -> None:
        self.ref_point = np.asarray(p, dtype=float).copy()
        self._update_x0y0()

    def set_ref_time(self, t0: float) -> None:
        self.ref_time = float(t0)

    def _set_angles_deg(self) -> None:
        th, ph = vector_angles(*self.direction)
        self.theta_deg = th * 180 / pi
        self.phi_deg = ph * 180 / pi
        if self.phi_deg < 0:
            self.phi_deg += 360

    def first_approximation(self) -> None:
        """GetFirstApproximation(0): start at the hottest hit, direction from pairs."""
        if not self.ids:
            self.ref_time = 0.0
            self.ref_point = np.zeros(3)
            return
        max_q, sel = 0.0, -1
        for i in self.ids:
            if self.q[i] > max_q:
                max_q, sel = self.q[i], i
        self.direction = self._direction_estimate()
        self._set_angles_deg()
        self.ref_point = self.pos[sel] - np.array([0.0, 0.0, 1.0])
        self.ref_time = float(self.t[sel])
        self._update_x0y0()

    def _direction_estimate(self) -> np.ndarray:
        """getTrackDirectionEstimate: charge-weighted inter-string hit vectors."""
        ids = self.ids
        if len(ids) < 2:
            return np.array([0.0, 0.0, 1.0])
        strings = self.channel[ids] // CHANNELS_PER_STRING
        s = np.zeros(3)
        same_string = True
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if strings[a] == strings[b]:
                    continue
                same_string = False
                qa, qb = self.q[ids[a]], self.q[ids[b]]
                s += qa * qb / (qa + qb) * (self.pos[ids[b]] - self.pos[ids[a]])
        if same_string:
            s = np.array([0.0, 0.0, self.pos[ids[1], 2] - self.pos[ids[0], 2]])
        th, ph = vector_angles(*s)
        return np.array(direction_of(th, ph))

    # ── quality ────────────────────────────────────────────────────────────
    def _arrays(self):
        ids = np.asarray(self.ids, dtype=np.int64)
        return (np.ascontiguousarray(self.pos[ids]),
                np.ascontiguousarray(self.t[ids]),
                np.ascontiguousarray(self.q[ids]))

    def quality_terms(self, fit_type: int) -> np.ndarray:
        """getModQuality of every hit at the current track."""
        pos, t, q = self._arrays()
        out = np.empty(len(t))
        quality_terms(self.ref_time, *self.ref_point, *self.direction, pos, t, q,
                      fit_type, charge_norm(self.sum_charge, self.n_hits), out)
        return out

    def quality(self, fit_type: int) -> float:
        """GetQuality(type)."""
        return float(self.quality_terms(fit_type).sum())

    # ── fitting ────────────────────────────────────────────────────────────
    def refit(self, fit_type: int) -> None:
        """Refit(type, 3): all five parameters free, Minuit2 with limits."""
        hot = self.hot.copy()
        t_start = time_at_pca(*hot, *self.ref_point, *self.direction,
                              self.ref_time)
        th0, ph0 = vector_angles(*self.direction)
        start = [self.x0, self.y0, th0, ph0, t_start]
        limits = [(self.x0 - POINT_MARGIN, self.x0 + POINT_MARGIN),
                  (self.y0 - POINT_MARGIN, self.y0 + POINT_MARGIN),
                  (max(th0 - ANGLE_MARGIN, -pi), min(th0 + ANGLE_MARGIN, pi)),
                  (ph0 - ANGLE_MARGIN, ph0 + ANGLE_MARGIN),
                  (None, None)]
        pos, t, q = self._arrays()
        norm = charge_norm(self.sum_charge, self.n_hits)

        def fcn(par):
            return fit_function(par, hot, pos, t, q, fit_type, norm)

        res = minuit2_minimize(fcn, start, limits)
        p = res.values
        self.n_calls = res.n_calls
        self.cov_status = res.cov_status
        self.min_status = res.min_status
        self.direction = np.array(direction_of(p[2], p[3]))
        self.ref_point = np.array(ref_xyz(*hot, p[0], p[1], *self.direction))
        self._set_angles_deg()
        self.theta_err_deg = error_deg(res.cov[2, 2])
        self.phi_err_deg = error_deg(res.cov[3, 3])
        self.ref_time = float(p[4])
        self.x0, self.y0 = float(p[0]), float(p[1])

    def refit_exclusive(self, fit_type: int) -> None:
        """RefitExclusive: drop the one hit whose removal best improves Q/(N-5).

        The C++ loop is `for (i=0; i<GetNHits(); i++)` with GetNHits() read after
        the removal, i.e. N-1: the last hit of the list is never tried.
        """
        if self.n_hits <= 7:
            return
        best_q = self.quality(fit_type) / (self.n_hits - 5)
        best = (self.direction.copy(), self.ref_point.copy(), self.ref_time,
                list(self.ids))
        ref = best
        k = 0
        while k < self.n_hits:
            self.set_direction(ref[0])
            self.set_ref_point(ref[1])
            self.set_ref_time(ref[2])
            self.set_impulse_array(ref[3])
            self.remove_impulse(k)
            self.refit(fit_type)
            qpndf = self.quality(fit_type) / (self.n_hits - 5)
            if qpndf < best_q:
                best = (self.direction.copy(), self.ref_point.copy(),
                        self.ref_time, list(self.ids))
                best_q = qpndf
            k += 1
        self.set_direction(best[0])
        self.set_ref_point(best[1])
        self.set_ref_time(best[2])
        self.set_impulse_array(best[3])

    def recollect_hits(self, tres_min: float, tres_max: float, rho_cut: float,
                       min_charge: float, enabled=None) -> None:
        """RecollectHits: add pulses on unused channels close to the track.

        `enabled` is the channel mask, which the source checks before anything
        else; a switched-off channel never comes back.
        """
        rp, d, t0 = self.ref_point, self.direction, self.ref_time
        for i in range(len(self.q)):
            if self.q[i] < min_charge:
                continue
            if enabled is not None and not enabled[i]:
                continue
            if i in self.ids or self.channel[i] in self.channel[self.ids]:
                continue
            p = self.pos[i]
            tres = self.t[i] - (propagation_time(*rp, *d, *p) + t0)
            rho = track_distance(*rp, *d, *p)
            if tres_min < tres < tres_max and rho < rho_cut:
                self.add_impulse(i)

    # ── per-hit views of the final track ──────────────────────────────────
    def residuals(self) -> np.ndarray:
        """GetResidualTime for every hit."""
        pos, t, _ = self._arrays()
        rp, d = self.ref_point, self.direction
        return np.array([t[k] - (self.ref_time + propagation_time(*rp, *d, *pos[k]))
                         for k in range(len(t))])

    def distances(self) -> np.ndarray:
        """GetDistToOM for every hit."""
        pos, _, _ = self._arrays()
        rp, d = self.ref_point, self.direction
        return np.array([track_distance(*rp, *d, *pos[k]) for k in range(len(pos))])

    def n_strings(self) -> int:
        return len(np.unique(self.channel[self.ids] // CHANNELS_PER_STRING))

    def n_oms(self) -> int:
        return len(np.unique(self.channel[self.ids]))
