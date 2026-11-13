"""One event through the BARS v2.3.2 reco-muon chain, plus derived variables.

`fit_event` takes the raw pulses of one event -- all of them, noise included, as
BARS does -- and returns a `TrackFit`.  Field names are snake_case; `BARS_NAMES`
maps each to the BRecoMuon / BRecoMuonBDTEval field it reproduces.

What is and is not reproduced is listed in the package README.  In short: the
task chain, the fit and its bookkeeping are ported line by line; the channel masks
(DQM, section) are not, because the HDF5 files do not carry them; coordinates are
the HDF5 cluster-centred ones, which moves `ev_center_z` and nothing else.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .candidate import CHANNELS_PER_STRING, Candidate
from .config import FitConfig
from .physics import (
    C_BAIKAL, direction_of, phit_light_path, phit_perpendicular, phit_table,
    propagation_time,
)
from .scanfit import (
    MAX_HITS, ScanResult, causality_clique_mask, local_trigger,
    passes_channel_trigger, scanfit,
)

# BRecoMuonProducer::Filter: RecollectHits(-10, 10, 100, 0.03, ...)
RECOLLECT_TRES = (-10.0, 10.0)        # ns
RECOLLECT_RHO = 100.0                 # m
RECOLLECT_CHARGE = 0.03               # p.e.
# BRecoMuonProducer::GetVarsZ
EV_CENTER_MIN_CHARGE = 2.0            # p.e., pulses with q < 2 are skipped
# BRecoMuonBDTEval::EvalVars
EVAL_CHARGE_CAP = 100.0               # p.e.
EVAL_PAIR_WINDOW = 20.0               # ns, CausalityTest(..., 20, 0)
EVAL_MIN_HITS = 5

STATUSES = ("ok", "too_many_hits", "no_clique", "no_local_trigger",
            "no_global_trigger", "no_offline_trigger")

#: snake_case field -> BARS field (BRecoMuon unless noted).
BARS_NAMES = {
    "theta_rec": "fThetaRec", "phi_rec": "fPhiRec",
    "theta_err": "fThetaErr", "phi_err": "fPhiErr",
    "func_value": "fFuncValue", "time_chi2": "fTimeChi2",
    "charge_term": "fChargeTerm",
    "n_hits": "fNHits", "n_strings": "fNStrings", "n_oms": "fNOMs",
    "n_calls": "fNCalls", "cov_matrix_status": "fCovMatrixStatus",
    "min_status": "fMinStatus",
    "scf_max_theta": "fScfMaxTheta", "scf_min_theta": "fScfMinTheta",
    "scf_theta": "fScfTheta", "scf_phi": "fScfPhi",
    "scf_n_vtx_passed": "fScfNVtx_passed",
    "func_value_scf": "fFuncValueScf", "time_chi2_scf": "fTimeChi2Scf",
    "theta_init": "fThetaInit", "phi_init": "fPhiInit",
    "n_hits_init": "fNHitsInit", "func_value_init": "fFuncValueInit",
    "time_chi2_init": "fTimeChi2Init",
    "z_dist": "fZDist", "n_triplets": "fNTriplets",
    "ev_center_z": "fEvCenterZ", "p_hit": "fPHit",
    "q_acc_20": "BRecoMuonBDTEval::fQacc_20",
    "q_tot_20": "BRecoMuonBDTEval::fQtot_20",
    "t_res_30_med": "BRecoMuonBDTEval::fTRes_30_med",
    "log_phit_30_min": "BRecoMuonBDTEval::fLogPHit_30_min",
}


@dataclass
class TrackFit:
    """Result of one event.  Scalars are NaN unless `status == "ok"`.

    Angles: `theta_rec`, `phi_rec`, `*_err` in degrees; `scf_*theta`, `scf_phi`
    in radians -- the units BARS writes them in.  `theta_rec` is the polar angle
    of the direction of motion: below 90 is up-going.
    """

    status: str
    theta_rec: float = np.nan
    phi_rec: float = np.nan
    theta_err: float = np.nan
    phi_err: float = np.nan
    func_value: float = np.nan
    time_chi2: float = np.nan
    charge_term: float = np.nan
    n_hits: float = np.nan
    n_strings: float = np.nan
    n_oms: float = np.nan
    n_calls: float = np.nan
    cov_matrix_status: float = np.nan
    min_status: float = np.nan
    scf_max_theta: float = np.nan
    scf_min_theta: float = np.nan
    scf_theta: float = np.nan
    scf_phi: float = np.nan
    scf_n_vtx_passed: float = np.nan
    # trial 0 before its fit, and the best trial after it
    func_value_scf: float = np.nan
    time_chi2_scf: float = np.nan
    theta_init: float = np.nan
    phi_init: float = np.nan
    n_hits_init: float = np.nan
    func_value_init: float = np.nan
    time_chi2_init: float = np.nan
    z_dist: float = np.nan
    n_triplets: float = np.nan
    ev_center_z: float = np.nan
    p_hit: float = np.nan
    q_acc_20: float = np.nan
    q_tot_20: float = np.nan
    t_res_30_med: float = np.nan
    log_phit_30_min: float = np.nan
    # the track: point, unit direction of motion, time at the point
    ref_x: float = np.nan
    ref_y: float = np.nan
    ref_z: float = np.nan
    dir_x: float = np.nan
    dir_y: float = np.nan
    dir_z: float = np.nan
    ref_time: float = np.nan
    # selection diagnostics
    n_hits_clique: int = 0
    n_hits_scanfit: int = 0
    clique_overflow: bool = False
    # per hit, in the caller's hit order (not in DataFrames)
    clique_mask: np.ndarray | None = field(default=None, repr=False)
    scanfit_mask: np.ndarray | None = field(default=None, repr=False)
    fit_hits: np.ndarray | None = field(default=None, repr=False)
    residuals: np.ndarray | None = field(default=None, repr=False)

    def scalars(self) -> dict:
        """Everything except the per-hit arrays, for a DataFrame row."""
        out = asdict(self)
        for k in ("clique_mask", "scanfit_mask", "fit_hits", "residuals"):
            out.pop(k)
        return out


def enabled_channels(channel: np.ndarray, channel_mask) -> np.ndarray | None:
    """Per-hit view of a channel mask: True where the hit's channel is usable.

    `channel_mask` is indexed by channel id (True = `BChannelMask::kOn`), as
    `masks.dqm_mask_from_solution` returns it.  None means "no mask", which is
    what BARS reconstructs MC with.
    """
    if channel_mask is None:
        return None
    mask = np.asarray(channel_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("channel_mask must be one-dimensional, indexed by channel")
    if channel.size and int(channel.max()) >= mask.size:
        raise ValueError(f"channel_mask covers {mask.size} channels but the event "
                         f"has channel {int(channel.max())}")
    return mask[channel]


def fit_event(q, t, x, y, z, channel, config: FitConfig | None = None,
              channel_mask=None) -> TrackFit:
    """Fit one event from all its raw pulses.

    Arrays may be in any order; they are put into BARS' pulse order (by channel,
    then time) internally and per-hit results are returned in the caller's order.

    `channel_mask` is BARS' channel mask, indexed by channel id (True = usable);
    see `masks`.  It removes hits from the track selection exactly where BARS
    does -- the causality graph, the first trial and `RecollectHits` -- while the
    derived variables that BARS computes over every pulse of the event
    (`q_tot_20`, `ev_center_z`, the paired-hit test) still see all of them.
    """
    cfg = config or FitConfig()
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    pos_in = np.column_stack([x, y, z]).astype(np.float64)
    ch_in = np.asarray(channel, dtype=np.int64)
    n = len(q)

    order = np.lexsort((t, ch_in))
    qb, tb, pb, cb = q[order], t[order], np.ascontiguousarray(pos_in[order]), \
        ch_in[order]

    def to_caller(mask_b: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        out[order] = mask_b
        return out

    if n > MAX_HITS:
        return TrackFit(status="too_many_hits")

    enabled = enabled_channels(cb, channel_mask)
    clique, overflow = causality_clique_mask(tb, pb, cb, cfg, enabled)
    fit = TrackFit(status="no_clique", n_hits_clique=int(clique.sum()),
                   clique_overflow=bool(overflow), clique_mask=to_caller(clique))
    if not clique.any():
        return fit
    if not local_trigger(qb, tb, cb, clique, cfg):
        fit.status = "no_local_trigger"
        return fit
    if not passes_channel_trigger(cb, clique, cfg, enabled):
        fit.status = "no_global_trigger"
        return fit

    scan = scanfit(tb, pb, clique, cfg)
    scan_mask = scan.mask if scan.mask is not None else np.zeros(n, dtype=bool)
    fit.n_hits_scanfit = int(scan_mask.sum())
    fit.scanfit_mask = to_caller(scan_mask)
    fit.clique_overflow |= scan.overflow
    if not passes_channel_trigger(cb, scan_mask, cfg, enabled):
        fit.status = "no_offline_trigger"
        return fit

    cand, scf = reconstruct(qb, tb, pb, cb, scan, scan_mask, cfg, enabled)
    fit.status = "ok"
    for k, v in scf.items():
        setattr(fit, k, v)
    for k, v in producer_variables(cand, qb, tb, pb, cfg).items():
        setattr(fit, k, v)
    for k, v in eval_variables(cand, qb, tb, pb, cb, scan_mask).items():
        setattr(fit, k, v)
    fit.fit_hits = order[np.asarray(cand.ids, dtype=np.int64)]
    fit.residuals = cand.residuals()
    return fit


def reconstruct(q, t, pos, channel, scan: ScanResult, scan_mask: np.ndarray,
                cfg: FitConfig, enabled=None) -> tuple[Candidate, dict]:
    """BRecoMuonProducer::Filter with UseScanfitData and DoSingleClusterRefit.

    The single-cluster refits (`*_clHM`, `*_clLM`) are skipped: with one cluster
    they are diagnostics that feed nothing below.
    """
    n = len(q)
    n_trials = cfg.n_candidate_trials
    trials = [Candidate(q, t, pos, channel) for _ in range(n_trials)]
    vertex_of = [-1] * n_trials
    for i in range(n):
        if scan_mask[i] and (enabled is None or enabled[i]):
            trials[0].add_impulse(i)
    trials[0].order_impulses(0.0)

    passed = np.flatnonzero(scan.passed)
    for v in passed:
        n_better = int(np.sum(scan.mest[passed] <= scan.mest[v])) - 1
        if n_better < n_trials:
            if n_better != 0:
                for i in np.flatnonzero(scan.selected[v]):
                    trials[n_better].add_impulse(i)
            vertex_of[n_better] = int(v)

    thetas = scan.theta[passed]
    scf = {"scf_n_vtx_passed": len(passed),
           "scf_max_theta": float(thetas.max()) if len(passed) else -1000.0,
           "scf_min_theta": float(thetas.min()) if len(passed) else 1000.0,
           "scf_theta": np.nan, "scf_phi": np.nan}

    best_k, best_qual = 0, 1e10
    for k in range(min(n_trials, len(passed))):
        v = vertex_of[k]
        if v == -1:
            continue
        cand = trials[k]
        cand.first_approximation()
        cand.set_direction(direction_of(scan.theta[v], scan.phi[v]))
        if k == 0:
            scf["func_value_scf"] = cand.quality(cfg.prefit_type)
            scf["time_chi2_scf"] = cand.quality(2)
        cand.refit(2)
        if cand.n_hits > 5:
            qual = cand.quality(2) / (cand.n_hits - 5)
            if qual < best_qual:
                best_k, best_qual = k, qual
    if vertex_of[0] != -1:
        scf["scf_theta"] = float(scan.theta[vertex_of[0]])
        scf["scf_phi"] = float(scan.phi[vertex_of[0]])

    best = trials[best_k]
    scf.update(n_hits_init=best.n_hits, theta_init=best.theta_deg,
               phi_init=best.phi_deg,
               func_value_init=best.quality(cfg.prefit_type),
               time_chi2_init=best.quality(2))
    best.refit_exclusive(cfg.prefit_type)
    best.recollect_hits(*RECOLLECT_TRES, RECOLLECT_RHO, RECOLLECT_CHARGE, enabled)
    best.refit(cfg.prefit_type)
    return best, scf


def producer_variables(cand: Candidate, q, t, pos, cfg: FitConfig) -> dict:
    """The BRecoMuon fields BRecoMuonProducer fills after the final fit."""
    ids = np.asarray(cand.ids, dtype=np.int64)
    ch = cand.channel[ids]
    k = (pos[ids] - cand.ref_point) @ cand.direction
    same_string = (ch[:, None] // CHANNELS_PER_STRING
                   == ch[None, :] // CHANNELS_PER_STRING)
    has_top = ((ch[None, :] - ch[:, None] == 1) & same_string).any(axis=1)
    has_bot = ((ch[None, :] - ch[:, None] == -1) & same_string).any(axis=1)
    func_value = cand.quality(cfg.prefit_type)
    time_chi2 = cand.quality(2)
    return {
        "theta_rec": cand.theta_deg, "phi_rec": cand.phi_deg,
        "theta_err": cand.theta_err_deg, "phi_err": cand.phi_err_deg,
        "func_value": func_value, "time_chi2": time_chi2,
        "charge_term": func_value - time_chi2,
        "n_hits": cand.n_hits, "n_strings": cand.n_strings(),
        "n_oms": cand.n_oms(),
        "n_calls": cand.n_calls, "cov_matrix_status": cand.cov_status,
        "min_status": cand.min_status,
        "z_dist": float(k.max() - k.min()),
        "n_triplets": int((has_top & has_bot).sum()),
        "ev_center_z": ev_center_z(q, t, pos),
        "p_hit": p_hit_product(cand),
        "ref_x": cand.ref_point[0], "ref_y": cand.ref_point[1],
        "ref_z": cand.ref_point[2],
        "dir_x": cand.direction[0], "dir_y": cand.direction[1],
        "dir_z": cand.direction[2], "ref_time": cand.ref_time,
    }


def ev_center_z(q, t, pos) -> float:
    """BRecoMuonProducer::GetVarsZ, second value: charge-weighted z of q >= 2 pulses.

    In the coordinates of the input -- cluster-centred for the HDF5 files, while
    BARS writes absolute detector z.  -1000 when all such pulses share one z or
    one time (the histogram RMS test in the source).
    """
    use = ~(q < EV_CENTER_MIN_CHARGE)
    if not use.any():
        return -1000.0
    z, tt, w = pos[use, 2], t[use], q[use]
    if np.std(z) == 0 or np.std(tt) == 0:
        return -1000.0
    return float((w * z).sum() / w.sum())


def p_hit_product(cand: Candidate) -> float:
    """BMuonCandidate::GetPHitProd(2): geometric mean of P(>=2 p.e.) over fit hits.

    As in the source, a single zero or NaN probability makes the result 1.
    """
    table = phit_table()
    rp, d = cand.ref_point, cand.direction
    total = 0.0
    for i in cand.ids:
        p = phit_perpendicular(*rp, *d, *cand.pos[i], table, 2)
        if p == 0 or np.isnan(p):
            total = 0.0
            break
        total += np.log10(p)
    return float(10 ** (total / cand.n_hits)) if cand.n_hits else 0.0


def eval_variables(cand: Candidate, q, t, pos, channel, scan_mask) -> dict:
    """BRecoMuonBDTEval::EvalVars: q_acc_20, q_tot_20, t_res_30_med, log_phit_30_min.

    `scan_mask` plays the role of the event mask the analysis tree passes
    (`ScanfitMask`).  Charges are capped at 100 p.e. first.
    """
    if cand.n_hits < EVAL_MIN_HITS:
        return {"q_acc_20": 0.0, "q_tot_20": 0.0, "t_res_30_med": 1000.0,
                "log_phit_30_min": -20.0}
    qc = np.minimum(q, EVAL_CHARGE_CAP)
    # causPaired: a pulse on an adjacent channel of the same string, causally
    # connected under the conventional test with a 20 ns window.
    adjacent = ((np.abs(channel[:, None] - channel[None, :]) == 1)
                & (channel[:, None] // CHANNELS_PER_STRING
                   == channel[None, :] // CHANNELS_PER_STRING))
    dt = np.abs(t[:, None] - t[None, :])
    d_r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    paired = (adjacent & (dt - d_r / C_BAIKAL < EVAL_PAIR_WINDOW)).any(axis=1)

    above2 = qc > 2.0
    q_tot = float(qc[above2].sum())
    q_acc = float(qc[above2 & scan_mask].sum())

    table = phit_table()
    rp, d, t0 = cand.ref_point, cand.direction, cand.ref_time
    tres, min_log = [], 0.0
    for i in np.flatnonzero((qc > 3.0) & paired):
        tres.append(abs(t[i] - (propagation_time(*rp, *d, *pos[i]) + t0)))
        # a hit at the track can have P = 0, and log10(0) = -inf is the answer BARS
        # gets too (C++ log10 of zero); numpy would only warn about it
        with np.errstate(divide="ignore"):
            log_p = np.log10(phit_light_path(*rp, *d, *pos[i], table))
        min_log = min(min_log, log_p)
    # GetMedianVec: 0 for fewer than two entries
    med = float(np.median(tres)) if len(tres) > 1 else 0.0
    return {"q_acc_20": q_acc, "q_tot_20": q_tot, "t_res_30_med": med,
            "log_phit_30_min": float(min_log)}
