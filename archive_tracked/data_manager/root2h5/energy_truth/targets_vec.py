"""Vectorised computation of the energy targets for a whole part at once.

`targets.py` computes one muon at a time and reads clearly, but the per-muon Python call
overhead dominates: ~530k muons/min, which is fine for nue2 (7.7M muons) and hopeless for
muatm (~2.5 billion muons, ~80 h). This module produces the identical numbers with array
operations over all muons of a part, and is checked against the reference implementation by
`test_targets_vec.py`.

The one part that looks like it needs a loop is the death point — walking forward through the
chain until the energy runs out. It does not: the muon's energy decreases monotonically along
the track, so "still alive" is always a *prefix* of the forward interactions. Counting the
prefix length per track replaces the walk, and the death point follows in closed form from the
last surviving interaction.

Conventions are those of `targets.py`: (theta, phi) is the direction of motion, interactions
are sorted along the track, distances are signed from the MC reference point, energies GeV.
"""
from __future__ import annotations

import numpy as np

from targets import (CLUSTER_RADIUS_M, CLUSTER_Z_HALF_M, IONISATION_GEV_PER_M,
                     R_LIMITS_M, direction_from_angles)


def _segment_sum(values: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum `values` per group id (groups are muon indices of each interaction)."""
    if len(values) == 0:
        return np.zeros(n_groups)
    return np.bincount(group, weights=values, minlength=n_groups)[:n_groups]


def compute_targets(theta_deg, phi_deg, ref_xyz, e_ref, n_inter,
                    int_xyz, int_energy, centres_of_muon,
                    r_limits=R_LIMITS_M, ionisation=IONISATION_GEV_PER_M):
    """All targets for every muon of a part.

    Args:
        theta_deg, phi_deg, e_ref, n_inter: (n_mu,) per-muon track truth
        ref_xyz: (n_mu, 3) reference points, global metres
        int_xyz: (n_inter, 3), int_energy: (n_inter,) chain, grouped by muon and sorted
        centres_of_muon: (n_mu, 3) centre of the cluster each muon is evaluated against

    Returns a dict of (n_mu,) arrays plus (n_mu, len(r_limits)) arrays for the cylinder scan,
    matching the column meanings documented in doc/hdf5_format.md.
    """
    n_mu = len(e_ref)
    # Everything below runs in double precision. The stored chain is float32, and the death
    # point is obtained from a running sum over the *whole part*: at ~1e6 GeV accumulated the
    # float32 step is already a few tens of MeV, which is metres of muon range once the
    # per-track offset is subtracted back off. The per-muon reference never accumulates across
    # tracks and so never saw this.
    e_ref = np.asarray(e_ref, dtype=np.float64)
    int_xyz = np.asarray(int_xyz, dtype=np.float64)
    int_energy = np.asarray(int_energy, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    centres_of_muon = np.asarray(centres_of_muon, dtype=np.float64)
    # The angles matter for the same reason: b_impact is a difference of ~250 m quantities
    # that lands on ~4 m, so a float32 direction is worth tens of microns of impact parameter.
    dirs = direction_from_angles(np.asarray(theta_deg, dtype=np.float64),
                                 np.asarray(phi_deg, dtype=np.float64))
    offsets = np.concatenate([[0], np.cumsum(n_inter)]).astype(np.int64)
    group = np.repeat(np.arange(n_mu), n_inter)          # muon index of each interaction

    # projection of every interaction onto its own track, in one pass
    s_int = np.einsum("ij,ij->i", int_xyz - ref_xyz[group], dirs[group]) if len(group) else np.zeros(0)

    # closest approach to the cluster centre
    delta = centres_of_muon - ref_xyz
    s_ca = np.einsum("ij,ij->i", delta, dirs)
    perp = delta - s_ca[:, None] * dirs
    b_impact = np.linalg.norm(perp, axis=1)

    # chain extent; empty tracks keep 0 as in the reference implementation
    has = n_inter > 0
    s_first = np.zeros(n_mu)
    s_last = np.zeros(n_mu)
    if len(s_int):
        idx = np.where(has)[0]
        s_first[idx] = np.minimum.reduceat(s_int, offsets[idx])
        s_last[idx] = np.maximum.reduceat(s_int, offsets[idx])

    # --- energy at the closest approach -------------------------------------------------
    # forward: subtract what lies in (0, s_ca]; backward: add back what lies in (s_ca, 0]
    s_ca_of_int = s_ca[group] if len(group) else np.zeros(0)
    fwd = (s_int > 0.0) & (s_int <= s_ca_of_int) & (s_ca_of_int >= 0.0)
    bwd = (s_int > s_ca_of_int) & (s_int <= 0.0) & (s_ca_of_int < 0.0)
    sum_fwd = _segment_sum(int_energy * fwd, group, n_mu) if len(group) else np.zeros(n_mu)
    sum_bwd = _segment_sum(int_energy * bwd, group, n_mu) if len(group) else np.zeros(n_mu)
    e_ca = e_ref - ionisation * s_ca - sum_fwd + sum_bwd

    # --- death point --------------------------------------------------------------------
    # Energy after passing interaction j: R_j = E_ref - a*s_j - (sum of e up to and including j).
    # R decreases monotonically along the track, so {R > 0} is a prefix of the forward chain;
    # its length is all the sequential walk ever determined. The muon then dies a further
    # R_last/a metres beyond the last interaction it survived.
    # With no surviving interaction the muon simply runs out of range; a non-positive e_ref
    # yields a negative death point, which correctly reads as "already dead before the
    # reference point" and must not be clamped.
    # Walk state before the fatal interaction: either the last interaction the muon
    # survived, or -- if it survives none -- the reference point itself. Unifying the two
    # matters: a muon that reaches its very first interaction alive and is killed by it dies
    # AT that interaction, not where its range would have run out.
    s_prev = np.zeros(n_mu)
    r_prev = e_ref.astype(np.float64).copy()
    s_next = np.full(n_mu, np.inf)

    n_fwd = np.zeros(n_mu, dtype=np.int64)
    if len(group):
        is_fwd = s_int > 0.0
        cum_all = np.cumsum(int_energy * is_fwd)
        starts = offsets[np.where(has)[0]]
        base = np.zeros(n_mu)
        base[has] = cum_all[starts] - (int_energy * is_fwd)[starts]
        R = e_ref[group] - ionisation * s_int - (cum_all - base[group])
        alive = is_fwd & (R > 0.0)

        n_alive = np.bincount(group, weights=alive, minlength=n_mu).astype(np.int64)
        n_before = np.bincount(group, weights=~is_fwd, minlength=n_mu).astype(np.int64)
        n_fwd[:] = np.bincount(group, weights=is_fwd, minlength=n_mu).astype(np.int64)

        surv = n_alive > 0                       # survived at least one forward interaction
        last = offsets[:-1][surv] + n_before[surv] + n_alive[surv] - 1
        s_prev[surv] = s_int[last]
        r_prev[surv] = R[last]

        nxt_exists = n_alive < n_fwd             # a forward interaction the muon did not survive
        nx = offsets[:-1][nxt_exists] + n_before[nxt_exists] + n_alive[nxt_exists]
        s_next[nxt_exists] = s_int[nx]

    # free range from the last surviving point; if that reaches the next interaction, the
    # interaction is what kills the muon
    s_free = s_prev + r_prev / ionisation
    s_death = np.where(s_free > s_next, s_next, s_free)
    # With no forward interaction at all the reference walk never enters its loop and falls
    # through to `s_prev`, which is zero -- so a muon already dead at the reference point
    # reports 0 here rather than a negative range. Matching that exactly matters: the nue2
    # twin was written with the reference implementation.
    if len(group):
        no_fwd = n_fwd == 0
    else:
        no_fwd = np.ones(n_mu, dtype=bool)
    s_death = np.where(no_fwd & (e_ref <= 0), 0.0, s_death)

    # --- deposited energy over the sensitive cylinder, per radius -----------------------
    n_r = len(r_limits)
    e_dep = np.full((n_mu, n_r), np.nan)
    l_path = np.zeros((n_mu, n_r))
    n_in = np.zeros((n_mu, n_r), dtype=np.int32)
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    d0 = ref_xyz - centres_of_muon
    a_quad = dx * dx + dy * dy
    for k, r_lim in enumerate(r_limits):
        radius, z_half = CLUSTER_RADIUS_M + r_lim, CLUSTER_Z_HALF_M + r_lim
        b_quad = 2.0 * (d0[:, 0] * dx + d0[:, 1] * dy)
        c_quad = d0[:, 0] ** 2 + d0[:, 1] ** 2 - radius ** 2
        disc = b_quad ** 2 - 4.0 * a_quad * c_quad
        vertical = a_quad <= 1e-12
        ok_radial = vertical | (disc >= 0.0)
        root = np.sqrt(np.where(disc > 0, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            s_lo = np.where(vertical, -np.inf, (-b_quad - root) / (2.0 * a_quad))
            s_hi = np.where(vertical, np.inf, (-b_quad + root) / (2.0 * a_quad))
            z_a = (centres_of_muon[:, 2] - z_half - ref_xyz[:, 2]) / dz
            z_b = (centres_of_muon[:, 2] + z_half - ref_xyz[:, 2]) / dz
        horizontal = np.abs(dz) < 1e-12
        z_lo = np.where(horizontal, -np.inf, np.minimum(z_a, z_b))
        z_hi = np.where(horizontal, np.inf, np.maximum(z_a, z_b))
        ok_z = ~horizontal | (np.abs(d0[:, 2]) <= z_half)
        s_in, s_out = np.maximum(s_lo, z_lo), np.minimum(s_hi, z_hi)
        good = ok_radial & ok_z & (s_out > s_in)

        length = np.where(good, s_out - s_in, 0.0)
        if len(group):
            inside = (s_int >= s_in[group]) & (s_int <= s_out[group]) & good[group]
            summed = _segment_sum(int_energy * inside, group, n_mu)
            counted = np.bincount(group, weights=inside, minlength=n_mu).astype(np.int32)
        else:
            summed = np.zeros(n_mu)
            counted = np.zeros(n_mu, dtype=np.int32)
        e_dep[:, k] = np.where(good, summed + ionisation * length, np.nan)
        l_path[:, k] = length
        n_in[:, k] = counted

    outside = np.maximum(0.0, s_first - s_ca) + np.maximum(0.0, s_ca - s_death)
    return dict(E_ca=e_ca, s_ca=s_ca, b_impact=b_impact, s_first_int=s_first,
                s_last_int=s_last, s_death=s_death,
                sigma_target_min=ionisation * outside,
                E_dep=e_dep, L_path=l_path, n_int_in=n_in)
