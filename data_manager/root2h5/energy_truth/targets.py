"""Energy targets for a muon track relative to one Baikal-GVD cluster.

The muon energy stored by the simulation (`fMuonEnergy`, mirrored into the HDF5 as
`muons_prty/individ[:, 6]`) is given at the MC reference point: the closest approach to the
centre of the WHOLE detector. For a single-cluster event that point can be hundreds of
metres from the cluster that actually saw the light, and the value is an extrapolation that
is negative for ~20% of nue2 tracks -- the muon was already dead there. Neither the number
nor its sign can be used at the cluster without propagating the muon, and propagating it
accurately needs the individual stochastic interactions rather than an average dE/dx: over
100-200 m the stochastic term is tens of percent with a heavy tail.

Two targets are produced, deliberately kept side by side because they fail in different
places:

  E_ca      the muon energy at its closest approach to the cluster centre. A physical
            quantity, defined for every track including those that pass outside the
            instrumented volume, and the natural input to a later unfolding to neutrino
            energy.

  E_dep, L_path over a sensitive cylinder (per radius), from which <dE/dx> = E_dep / L_path
            follows. This is what the light actually measures: normalising by path length
            removes the geometry (a track through the centre deposits more than one clipping
            the edge at equal energy), leaving a quantity proportional to E above ~1 TeV.
            Below roughly a TeV it saturates at the ionisation plateau (~0.24 GeV/m) and
            stops carrying energy information at all -- a physical limit, not a defect, and
            one the uncertainty head is expected to express.

E_dep and L_path are stored separately rather than as their ratio: the ratio is always
recoverable from the pair, the reverse is not, and L_path is needed to interpret the
uncertainty, since over a short segment <dE/dx> is a noisy estimator of energy (it depends
on whether a catastrophic interaction happened to occur there).

Conventions, all verified against the data (see reference/energy.C and the extraction
notes): (theta, phi) is the direction of MOTION, interactions increase along +direction,
coordinates are global metres, energies GeV.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Geometry of one cluster, measured from the hit envelope in baikal_mc_merged.h5:
# x/y span ~119 m (radius 60 m, matching the 60 m used in energy.C) and z span 529.6 m,
# i.e. 36 OMs at 15 m spacing centred on the cluster centre.
CLUSTER_RADIUS_M = 60.0
CLUSTER_Z_HALF_M = 265.0

# Continuous loss term. The interaction chain holds only discrete losses above the
# generator's threshold; ionisation and sub-threshold radiative losses are absorbed into
# this constant, following energy.C which uses 0.24 GeV/m. Negligible over metres, but 72
# GeV over 300 m -- irrelevant for a 100 TeV muon, decisive for a 100 GeV one.
IONISATION_GEV_PER_M = 0.24

# Margin added to the cluster on all sides to define the sensitive volume. energy.C uses a
# single parameter (70 m) applied both radially and in z, tuned on nuE1 events, and reports
# only weak dependence; the scan keeps that construction and brackets the author's value.
R_LIMITS_M = (30.0, 50.0, 70.0, 90.0, 110.0)


def direction_from_angles(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Unit vectors of motion from zenith/azimuth in degrees. Shape (n, 3)."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def closest_approach(ref: np.ndarray, direction: np.ndarray,
                     centre: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closest approach of the track to a point.

    Returns (s_ca, b_impact): the signed distance along the track from the reference point
    to the closest-approach point, and the perpendicular distance there. s_ca is the
    analogue of CorDist in energy.C -- its sign decides whether losses are added or
    subtracted when propagating, and its magnitude says how far the extrapolation was
    stretched, which is what limits the accuracy of the ionisation approximation.
    """
    delta = centre - ref
    s_ca = np.einsum("...i,...i->...", delta, direction)
    perp = delta - s_ca[..., None] * direction
    return s_ca, np.linalg.norm(perp, axis=-1)


def energy_at(e_ref: float, s_int: np.ndarray, e_int: np.ndarray, s: float,
              ionisation: float = IONISATION_GEV_PER_M) -> float:
    """Muon energy at signed distance `s` along the track from the reference point.

    Moving forward the muon loses the discrete interactions it passes plus the continuous
    term; moving backward both are added back, because the muon had more energy earlier:

        E(s) = E_ref - a*s - sign(s) * sum(e_i for s_i between 0 and s)

    This reproduces energy.C, where interactions between the entry point and the reference
    point are added when the entry lies behind it and the ionisation correction carries the
    same sign as the displacement.
    """
    if s >= 0.0:
        in_between = (s_int > 0.0) & (s_int <= s)
        return float(e_ref - ionisation * s - e_int[in_between].sum())
    in_between = (s_int > s) & (s_int <= 0.0)
    return float(e_ref - ionisation * s + e_int[in_between].sum())


def cylinder_segment(ref: np.ndarray, direction: np.ndarray, centre: np.ndarray,
                     radius: float, z_half: float) -> Tuple[float, float] | None:
    """Entry/exit of the track through a vertical cylinder, as signed distances from ref.

    Returns None when the track misses the volume. energy.C leaves the entry point at the
    origin in that case, which silently produces a meaningless correction; here the caller
    is told explicitly so those tracks can be marked rather than mis-valued.
    """
    d = ref - centre
    dx, dy, dz = direction[0], direction[1], direction[2]
    a = dx * dx + dy * dy
    if a <= 1e-12:                                   # exactly vertical: always inside radially
        s_lo, s_hi = -np.inf, np.inf
    else:
        b = 2.0 * (d[0] * dx + d[1] * dy)
        c = d[0] * d[0] + d[1] * d[1] - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        root = np.sqrt(disc)
        s_lo, s_hi = (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)

    if abs(dz) < 1e-12:                              # horizontal: inside iff z within range
        if abs(d[2]) > z_half:
            return None
        z_lo, z_hi = -np.inf, np.inf
    else:
        z_lo = (centre[2] - z_half - ref[2]) / dz
        z_hi = (centre[2] + z_half - ref[2]) / dz
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo

    s_in, s_out = max(s_lo, z_lo), min(s_hi, z_hi)
    return (float(s_in), float(s_out)) if s_out > s_in else None


def deposited(s_int: np.ndarray, e_int: np.ndarray, s_in: float,
              s_out: float, ionisation: float = IONISATION_GEV_PER_M) -> Tuple[float, float, int]:
    """Energy deposited inside a segment and its length: (E_dep, L_path, n_interactions)."""
    inside = (s_int >= s_in) & (s_int <= s_out)
    length = s_out - s_in
    return float(e_int[inside].sum() + ionisation * length), float(length), int(inside.sum())


def death_point(e_ref: float, s_int: np.ndarray, e_int: np.ndarray,
                ionisation: float = IONISATION_GEV_PER_M) -> float:
    """Distance along the track at which the muon's energy reaches zero, going forward.

    Beyond it the muon does not exist, so any energy computed there is an extrapolation of a
    particle that is already gone. Returned as a signed distance from the reference point,
    on the same scale as s_int.
    """
    forward = np.sort(s_int[s_int > 0.0])
    energy, s_prev = e_ref, 0.0
    for s_i in forward:
        at_i = energy - ionisation * (s_i - s_prev)
        if at_i <= 0.0:
            return s_prev + energy / ionisation
        energy = at_i - float(e_int[s_int == s_i].sum())
        if energy <= 0.0:
            return float(s_i)
        s_prev = float(s_i)
    return s_prev + energy / ionisation if energy > 0.0 else s_prev


def target_noise_floor(s_eval: float, s_first_int: float, s_death: float,
                       ionisation: float = IONISATION_GEV_PER_M) -> float:
    """Upper bound on the error the continuous term introduces at `s_eval`.

    Interactions exist only where the muon lived, so the stochastic part of the propagation
    is exact; what can be invented is the ionisation charged over stretches where the muon is
    not there -- beyond its death, or before it was born. The birth point is unknown (the
    vertex is not stored, see doc/mc_energy_truth.md), so the first interaction is used as a
    conservative left bound: the muon may well have existed earlier without interacting, which
    makes this a maximum rather than an estimate.

    Nothing is clipped on the strength of this number. Clipping forward at zero would destroy
    information -- a negative energy encodes how far past death the point lies -- and clipping
    backward at E_nu is useless, since inelasticity lets the muon carry a small fraction of
    the neutrino energy so the bound almost never binds. The value is stored instead, to be
    used as a known noise floor in a heteroscedastic loss or as a selection variable.
    """
    outside = max(0.0, s_first_int - s_eval) + max(0.0, s_eval - s_death)
    return ionisation * outside


def targets_for_track(ref: np.ndarray, direction: np.ndarray, e_ref: float,
                      s_int: np.ndarray, e_int: np.ndarray, centre: np.ndarray,
                      r_limits: Tuple[float, ...] = R_LIMITS_M) -> dict:
    """All energy targets for one track against one cluster.

    `s_int` are the interaction positions already projected onto the track direction; the
    extractor sorts them, so no ordering assumption is made here.
    """
    s_ca, b_impact = closest_approach(ref, direction, centre)
    s_first = float(s_int.min()) if len(s_int) else 0.0
    s_last = float(s_int.max()) if len(s_int) else 0.0
    s_death = death_point(e_ref, s_int, e_int)
    out = {
        "E_ca": energy_at(e_ref, s_int, e_int, float(s_ca)),
        "s_ca": float(s_ca),
        "b_impact": float(b_impact),
        "s_first_int": s_first,
        "s_last_int": s_last,
        "s_death": s_death,
        "sigma_target_min": target_noise_floor(float(s_ca), s_first, s_death),
        "E_dep": np.zeros(len(r_limits), dtype=np.float32),
        "L_path": np.zeros(len(r_limits), dtype=np.float32),
        "n_int_in": np.zeros(len(r_limits), dtype=np.int32),
    }
    for i, r_lim in enumerate(r_limits):
        seg = cylinder_segment(ref, direction, centre,
                               CLUSTER_RADIUS_M + r_lim, CLUSTER_Z_HALF_M + r_lim)
        if seg is None:                              # track misses the volume: no segment
            out["E_dep"][i] = np.nan
            out["L_path"][i] = 0.0
            continue
        e_dep, length, n_in = deposited(s_int, e_int, seg[0], seg[1])
        out["E_dep"][i], out["L_path"][i], out["n_int_in"][i] = e_dep, length, n_in
    return out
