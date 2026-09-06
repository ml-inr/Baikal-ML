"""Per-event interpretable features for the group-structure study.

Nothing here uses the nu-classifier: no score, no embedding, no manifold distance. Nothing
uses MC truth. The same code runs on simulation and on data, from the sig-noise-filtered hits.

The backbone is a Cherenkov track fit (PLAN.md §2.1). For a relativistic track through a point
`p0` on the line with direction `u`, light reaches a module at `r` at

    t_pred = t0 + [ s + K*d ] / c,   s = (r-p0).u,   d = |(r-p0) - s*u|
    K = sqrt(n^2 - 1) = 0.9364      (n = 1.37, doc/data_format.md)

`p0` must lie *on the track*: shifting it along `u` is absorbed by `t0`, but shifting it
perpendicular changes every `d`. The hit centroid does not lie on the track, so the fit runs in
three fixed-grid stages -- direction, then perpendicular offset, then direction again.

Direction comes from timing, never from the shape of the hit cloud: modules sit on vertical
strings, so a PCA axis is near-vertical whatever the track did.
"""

from __future__ import annotations

import numpy as np

# --- physical constants, all sourced, none tuned -------------------------------------------
C_VAC = 0.299792458          # m/ns
N_WATER = 1.37               # doc/data_format.md
K_CHERENKOV = float(np.sqrt(N_WATER ** 2 - 1.0))     # 0.9364
V_LIGHT = C_VAC / N_WATER    # 0.21882 m/ns
Q_CLIP = 100.0               # p.e., the bound the classifier itself applies
STRING_DIVISOR = 36          # channel // 36 -> string, as io.py:_count_sig_hits_strings

# --- fixed grids, built once at import, never tuned on data --------------------------------
N_DIRECTIONS = 500
OFFSET_STEPS = 7             # 7x7 perpendicular offsets
N_LOCAL = 50
# Refinement rounds: (perpendicular-offset half-range in m, direction radius in degrees).
# Fixed a priori and shrinking; a single coarse pass leaves a few degrees of grid error, which
# over a 300 m lever arm is tens of ns of residual -- the synthetic-track test catches it.
REFINE_ROUNDS = ((60.0, 10.0), (20.0, 3.0), (7.0, 1.0), (2.5, 0.3), (1.0, 0.1))
# Five rounds, not three: direction and offset are coupled, so alternating them has to
# iterate. Three left ~1 deg of angle error on an exact synthetic track (4.5 ns of
# residual where there should be none); five leave well under the 25 ns tolerance.
ON_TRACK_NS = 25.0           # a third of the 15.0 m module step (68.6 ns of light in water)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """`n` directions spread evenly over the sphere. Deterministic."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


DIRECTIONS = _fibonacci_sphere(N_DIRECTIONS)

_unit = np.linspace(-1.0, 1.0, OFFSET_STEPS)
OFFSET_UNIT = np.array([(x, y) for x in _unit for y in _unit])        # (49, 2), scaled per round

_g = np.arange(N_LOCAL, dtype=np.float64) + 0.5
_rad = np.sqrt(_g / N_LOCAL)
_ang = np.pi * (1.0 + 5.0 ** 0.5) * _g
LOCAL_UNIT = np.stack([_rad * np.cos(_ang), _rad * np.sin(_ang)], axis=1)   # (50, 2), scaled

def _perp_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `u`."""
    seed = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def _residuals(pos: np.ndarray, t: np.ndarray, p0: np.ndarray,
               dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time residuals for many directions at once. Returns (rms per dir, resid (n, n_dir))."""
    r = pos - p0                                     # (n, 3)
    s = r @ dirs.T                                   # (n, G)
    d = np.sqrt(np.maximum((r ** 2).sum(1)[:, None] - s ** 2, 0.0))
    resid = t[:, None] - (s + K_CHERENKOV * d) / C_VAC
    resid = resid - resid.mean(0)                    # t0 in closed form, per direction
    return np.sqrt((resid ** 2).mean(0)), resid


def fit_track(pos: np.ndarray, t: np.ndarray, q: np.ndarray) -> dict:
    """Cherenkov track fit: a global direction scan, then rounds of shrinking refinement.

    One coarse pass is not enough. 500 directions over the sphere are ~9 deg apart and the
    local pass leaves ~3 deg, which over a 300 m lever arm is tens of ns of residual on a
    track that has none. `REFINE_ROUNDS` alternates offset and direction at shrinking scales.
    """
    # stage 1 -- direction over the whole sphere, anchored at the charge-weighted centroid
    p0 = (q[:, None] * pos).sum(0) / q.sum()
    rms_grid, _ = _residuals(pos, t, p0, DIRECTIONS)
    best = int(np.argmin(rms_grid))
    u = DIRECTIONS[best]
    # contrast is defined on this grid alone: how much better the best direction is than a
    # typical one. An event with no track structure fits every direction about equally badly.
    med = float(np.median(rms_grid))
    contrast = (med - float(rms_grid[best])) / med if med > 0 else 0.0

    resid = None
    for offset_m, radius_deg in REFINE_ROUNDS:
        e1, e2 = _perp_basis(u)
        # perpendicular offset of p0 at the current direction
        cand = p0 + (offset_m * OFFSET_UNIT[:, 0:1]) * e1 + (offset_m * OFFSET_UNIT[:, 1:2]) * e2
        r = pos[:, None, :] - cand[None, :, :]
        s = r @ u
        d = np.sqrt(np.maximum((r ** 2).sum(-1) - s ** 2, 0.0))
        res = t[:, None] - (s + K_CHERENKOV * d) / C_VAC
        res = res - res.mean(0)
        p0 = cand[int(np.argmin(np.sqrt((res ** 2).mean(0))))]

        # direction again, locally, at the improved p0
        scale = np.tan(np.radians(radius_deg))
        local = u + (scale * LOCAL_UNIT[:, 0:1]) * e1 + (scale * LOCAL_UNIT[:, 1:2]) * e2
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        local = np.vstack([u[None, :], local])
        rms_local, resid_local = _residuals(pos, t, p0, local)
        k = int(np.argmin(rms_local))
        u, resid, fit_rms = local[k], resid_local[:, k], float(rms_local[k])

    absr = np.abs(resid)
    on = absr < ON_TRACK_NS
    q_sum = float(q.sum())
    return {
        "fit_rms": fit_rms,
        "fit_contrast": float(contrast),
        "fit_zenith": float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0)))),
        "fit_azimuth": float(np.degrees(np.arctan2(u[1], u[0]))),
        "frac_on_track": float(on.mean()),
        "q_offtrack_frac": float(q[~on].sum() / q_sum) if q_sum > 0 else 0.0,
        "q_weighted_residual": float((q * resid).sum() / q_sum) if q_sum > 0 else 0.0,
        "_u": u, "_p0": p0, "_resid": resid,
    }


def _perp_distance(pos: np.ndarray, p0: np.ndarray, u: np.ndarray) -> np.ndarray:
    r = pos - p0
    s = r @ u
    return np.sqrt(np.maximum((r ** 2).sum(1) - s ** 2, 0.0))


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and r^2, or (nan, nan) when x has no spread."""
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    sx, sy = x - x.mean(), y - y.mean()
    denom = float((sx ** 2).sum())
    if denom == 0:
        return np.nan, np.nan
    slope = float((sx * sy).sum() / denom)
    ss = float((sy ** 2).sum())
    r2 = float(((sx * sy).sum() ** 2) / (denom * ss)) if ss > 0 else np.nan
    return slope, r2


COLUMNS = [
    # track fit
    "fit_rms", "fit_contrast", "fit_zenith", "fit_azimuth", "frac_on_track",
    "q_offtrack_frac", "q_weighted_residual",
    # dimensionless
    "slowness", "extent_m", "q_vs_d_slope", "q_vs_d_r2", "string_slope_spread",
    "n_strings_fitted",
    # repeated hits
    "dt_repeat_median", "q_repeat_over_first",
    # multiplicity and geometry
    "n_hits", "n_modules", "n_strings", "hits_per_module", "hits_per_string_max",
    "hits_per_string_mean", "z_span", "xy_span", "z_c", "r_cyl", "z_first", "z_last",
    "dz_signed", "elongation", "planarity", "r_vert",
    # timing
    "t_span", "t_span_core", "t_std", "dtdz_slope", "track_likeness",
    "causality_violation_frac", "spearman_tz",
    # charge
    "q_total", "q_mean", "q_std", "q_median", "q_iqr", "q_max", "q_frac_max",
    "frac_q_below_2", "q_asymmetry", "centroid_shift",
    # sig-noise context (secondary, flagged)
    "prob_mean", "prob_min", "n_raw_hits", "survival_frac",
]


def event_features(h: np.ndarray, p: np.ndarray, c: np.ndarray, n_raw: int) -> tuple:
    """One event. `h` = (n,5) raw hits [q,t,x,y,z], `p` = sig-noise probs, `c` = channel ids."""
    q_raw, t, x, y, z = (h[:, i].astype(np.float64) for i in range(5))
    q = np.clip(q_raw, 0.0, Q_CLIP)
    pos = np.stack([x, y, z], axis=1)
    n = len(h)

    fit = fit_track(pos, t, np.maximum(q, 1e-6))
    d_perp = _perp_distance(pos, fit["_p0"], fit["_u"])

    # --- dimensionless ---------------------------------------------------------------------
    diff = pos[:, None, :] - pos[None, :, :]
    extent = float(np.sqrt((diff ** 2).sum(-1)).max())
    t_span = float(t.max() - t.min())
    slowness = t_span / (extent / V_LIGHT) if extent > 0 else np.nan

    q_vs_d_slope, q_vs_d_r2 = _linfit(d_perp, np.log(np.maximum(q, 0.05)))

    strings = c // STRING_DIVISOR
    slopes = []
    for s_id in np.unique(strings):
        m = strings == s_id
        if m.sum() >= 3:
            sl, _ = _linfit(z[m], t[m])
            if np.isfinite(sl):
                slopes.append(sl)
    string_slope_spread = float(np.std(slopes)) if len(slopes) >= 2 else np.nan

    # --- repeated hits ---------------------------------------------------------------------
    gaps, q_first, q_rep = [], 0.0, 0.0
    for ch in np.unique(c):
        m = c == ch
        if m.sum() == 1:
            q_first += float(q[m][0])
            continue
        tt = np.sort(t[m])
        gaps.extend(np.diff(tt).tolist())
        order = np.argsort(t[m])
        qq = q[m][order]
        q_first += float(qq[0])
        q_rep += float(qq[1:].sum())
    dt_repeat_median = float(np.median(gaps)) if gaps else np.nan
    q_repeat_over_first = (q_rep / q_first) if q_first > 0 else np.nan

    # --- multiplicity and geometry ---------------------------------------------------------
    n_modules = int(np.unique(c).size)
    n_strings = int(np.unique(strings).size)
    per_string = np.bincount(np.unique(strings, return_inverse=True)[1])
    cov = np.cov((pos - pos.mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1] if np.all(np.isfinite(cov)) else np.zeros(3)
    ev = np.maximum(ev, 0.0)
    tot_ev = float(ev.sum())
    elongation = float(ev[0] / tot_ev) if tot_ev > 0 else np.nan
    planarity = float((ev[0] + ev[1]) / tot_ev) if tot_ev > 0 else np.nan
    order_t = np.argsort(t)
    sx, sy, sz = x.std(), y.std(), z.std()

    # --- timing ----------------------------------------------------------------------------
    t_lo, t_hi = np.percentile(t, [5, 95])
    dtdz_slope, track_likeness = _linfit(z, t)
    dist = np.sqrt((diff ** 2).sum(-1))
    dt = np.abs(t[:, None] - t[None, :])
    iu = np.triu_indices(n, 1)
    causality = float((dt[iu] < dist[iu] / C_VAC).mean()) if n > 1 else np.nan
    rt, rz = np.argsort(np.argsort(t)), np.argsort(np.argsort(z))
    spearman = float(np.corrcoef(rt, rz)[0, 1]) if n > 2 and np.ptp(rz) > 0 else np.nan

    # --- charge ----------------------------------------------------------------------------
    q_total, q_max = float(q.sum()), float(q.max())
    half = n // 2
    early = q[order_t][:half].sum()
    late = q[order_t][half:].sum()
    q_asym = float((early - late) / (early + late)) if (early + late) > 0 else np.nan
    cw = (q[:, None] * pos).sum(0) / q_total if q_total > 0 else pos.mean(0)
    centroid_shift = float(np.linalg.norm(cw - pos.mean(0)))

    values = {
        "fit_rms": fit["fit_rms"], "fit_contrast": fit["fit_contrast"],
        "fit_zenith": fit["fit_zenith"], "fit_azimuth": fit["fit_azimuth"],
        "frac_on_track": fit["frac_on_track"], "q_offtrack_frac": fit["q_offtrack_frac"],
        "q_weighted_residual": fit["q_weighted_residual"],
        "slowness": slowness, "extent_m": extent,
        "q_vs_d_slope": q_vs_d_slope, "q_vs_d_r2": q_vs_d_r2,
        "string_slope_spread": string_slope_spread, "n_strings_fitted": len(slopes),
        "dt_repeat_median": dt_repeat_median, "q_repeat_over_first": q_repeat_over_first,
        "n_hits": n, "n_modules": n_modules, "n_strings": n_strings,
        "hits_per_module": n / n_modules, "hits_per_string_max": int(per_string.max()),
        "hits_per_string_mean": float(per_string.mean()),
        "z_span": float(z.max() - z.min()),
        "xy_span": float(np.hypot(x.max() - x.min(), y.max() - y.min())),
        "z_c": float(z.mean()), "r_cyl": float(np.hypot(x.mean(), y.mean())),
        "z_first": float(z[order_t[0]]), "z_last": float(z[order_t[-1]]),
        "dz_signed": float(z[order_t[-1]] - z[order_t[0]]),
        "elongation": elongation, "planarity": planarity,
        "r_vert": float(sz / (np.sqrt(sx * sx + sy * sy) + 1e-3)),
        "t_span": t_span, "t_span_core": float(t_hi - t_lo), "t_std": float(t.std()),
        "dtdz_slope": dtdz_slope, "track_likeness": track_likeness,
        "causality_violation_frac": causality, "spearman_tz": spearman,
        "q_total": q_total, "q_mean": float(q.mean()), "q_std": float(q.std()),
        "q_median": float(np.median(q)),
        "q_iqr": float(np.subtract(*np.percentile(q, [75, 25]))),
        "q_max": q_max, "q_frac_max": q_max / q_total if q_total > 0 else 0.0,
        "frac_q_below_2": float((q < 2.0).mean()), "q_asymmetry": q_asym,
        "centroid_shift": centroid_shift,
        "prob_mean": float(p.mean()), "prob_min": float(p.min()),
        "n_raw_hits": int(n_raw), "survival_frac": n / n_raw if n_raw > 0 else np.nan,
    }
    return tuple(values[k] for k in COLUMNS)
