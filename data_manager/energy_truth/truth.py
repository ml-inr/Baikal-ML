"""Geometry and energy propagation along a simulated muon track.

Pure functions, no I/O. This module is the single implementation of the model documented in
doc/hdf5_energy_truth.md section 4, shared by the companion-file builder and by any downstream
analysis, so that "the energy of this muon at that point" has exactly one answer in the
project.

Everything is expressed in the track coordinate `s`: the signed distance in metres from the
reference point, positive in the muon's direction of motion,

    s(r) = (r - r_ref) . d

Two conventions that are easy to get wrong and are therefore stated once here: angles are
radians (as `muons_prty/individ` stores them, unlike `prime_prty`, which is degrees), and
positions are in the global array frame (unlike hit coordinates, which are cluster-centred).

All internal arithmetic is float64. The inputs are float32, and a float32 cumulative sum over
a chain reaching ~1e6 GeV already produced a wrong result in this project once; the promotion
is not cosmetic.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np

# Muons are ultrarelativistic here (beta = 0.9999967 at 41 GeV), so the vacuum speed of light
# is the right constant for converting a flight time into a path length -- not c/n.
C_M_PER_NS = 0.299792458

# Continuous ionisation loss in water. The stochastic part of dE/dx is NOT modelled by a b*E
# term, because the stochastic losses are known individually as showers and adding both would
# count them twice. Status: assumed -- a standard value, not fitted to this MC.
IONISATION_GEV_PER_M = 0.24

# The sentinel value ROOT writes for a muon that does not yet exist at the reference point.
SENTINEL_E_REF = -1e-3

# Status codes of `entry_energy`, mirrored in the companion file's attributes.
STATUS_PROPAGATED = 0        # propagated from the reference point to the entry point
STATUS_BORN_INSIDE = 1       # the trajectory begins inside the volume; energy at birth
STATUS_DIED_BEFORE = 2       # the muon was already dead at the entry point
STATUS_NO_CROSSING = 3       # the trajectory never enters the volume
STATUS_SENTINEL = 4          # sentinel muon, propagated from e_bundle_reg at the track start
STATUS_SENTINEL_INSIDE = 5   # sentinel muon born inside the volume; energy is e_bundle_reg

# Used to make a per-muon searchsorted into a single global one: shifting each muon's chain
# into its own numeric band makes the concatenated array globally sorted. Track coordinates
# stay well inside +-1e3 m, so a 1e7 band is four orders of magnitude of headroom.
_BAND = 1e7


def direction_from_angles(theta_rad: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    """Unit vectors of the direction of motion, shape (n, 3).

    Convention check that comes for free with the data: neutrino-induced muons have
    theta < 90 degrees and travel upward, atmospheric muons have theta > 90 degrees and travel
    downward, so cos(theta) is the z component of the motion and not of the arrival direction.
    """
    th = np.asarray(theta_rad, dtype=np.float64)
    ph = np.asarray(phi_rad, dtype=np.float64)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)


def s_along(points: np.ndarray, ref: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Track coordinate of points, given each point's own reference and direction."""
    v = np.asarray(points, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return np.einsum("...i,...i->...", v, np.asarray(d, dtype=np.float64))


def track_start(t_first_muon: np.ndarray) -> np.ndarray:
    """Where the simulated trajectory begins, in track coordinate.

    The clock's zero is the moment the muon passes the reference point, so a positive time
    places the start behind it. For sentinel muons the time is negative and the start lies
    ahead -- which is the same statement as "the muon did not exist at the reference point",
    and the two agree in 100.00 % of muons of all three productions.
    """
    return -C_M_PER_NS * np.asarray(t_first_muon, dtype=np.float64)


def shower_times(s: np.ndarray, t_first_muon_per_shower: np.ndarray) -> np.ndarray:
    """Time at which the muon reaches each shower, zero at the muon's birth."""
    return np.asarray(t_first_muon_per_shower, dtype=np.float64) + np.asarray(s, np.float64) / C_M_PER_NS


class Chain(NamedTuple):
    """A muon's showers, sorted along the track, with running energy sums.

    Built once and evaluated many times: a single muon is asked for its energy at up to eight
    points (entry and exit of three volumes, closest approach, birth).
    """
    s: np.ndarray          # (n_showers,) sorted within each muon
    cum: np.ndarray        # (n_showers,) energy summed within the muon, inclusive
    starts: np.ndarray     # (n_muons + 1,)
    key: np.ndarray        # (n_showers,) s shifted into per-muon bands, globally sorted
    at_ref: np.ndarray     # (n_muons,) energy summed over showers at s <= 0


def build_chain(shower_s: np.ndarray, shower_energy: np.ndarray,
                starts: np.ndarray) -> Chain:
    """Sort each muon's showers along its track and precompute the running sums."""
    s = np.asarray(shower_s, dtype=np.float64)
    e = np.asarray(shower_energy, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    n_mu = len(starts) - 1
    counts = np.diff(starts)
    owner = np.repeat(np.arange(n_mu), counts)

    if np.abs(s).max(initial=0.0) >= _BAND / 2:
        raise ValueError("track coordinates exceed the banding headroom of build_chain")

    order = np.lexsort((s, owner))
    s = s[order]
    e = e[order]

    # Per-muon inclusive cumulative sum, obtained from the global one by subtracting the
    # value just before each muon's first shower.
    total = np.cumsum(e)
    base = np.concatenate([[0.0], total])[starts[:-1]]
    cum = total - np.repeat(base, counts)

    key = s + owner * _BAND
    chain = Chain(s=s, cum=cum, starts=starts, key=key, at_ref=np.zeros(n_mu))
    return chain._replace(at_ref=_cumulative_at(chain, np.zeros(n_mu)))


def _cumulative_at(chain: Chain, s: np.ndarray) -> np.ndarray:
    """Energy of all showers of each muon at track coordinate <= s."""
    n_mu = len(chain.starts) - 1
    if len(chain.cum) == 0:                      # a production can have showerless muons only
        return np.zeros(n_mu)
    target = np.asarray(s, dtype=np.float64) + np.arange(n_mu) * _BAND
    idx = np.searchsorted(chain.key, target, side="right")
    # idx counts showers of *all* preceding muons too; only those of this muon are wanted,
    # which is what comparing against the muon's own first index does.
    has_any = idx > chain.starts[:-1]
    return np.where(has_any, chain.cum[np.clip(idx - 1, 0, len(chain.cum) - 1)], 0.0)


def energy_at(chain: Chain, e_ref: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Muon energy at track coordinate s, propagated from the reference point.

        E(s) = e_ref - a * s - (C(s) - C(0))

    where C is the running shower sum. The single expression covers both directions: going
    backwards, s is negative and C(s) - C(0) is negative, so both terms add energy, which is
    what "the muon had more energy earlier" means. E is strictly decreasing in s, so a
    non-positive value means the muon is dead at that point, and the crossing is its death.
    """
    e0 = np.asarray(e_ref, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return e0 - IONISATION_GEV_PER_M * s - (_cumulative_at(chain, s) - chain.at_ref)


def energy_between(chain: Chain, e_known: np.ndarray, s_known: np.ndarray,
                   s: np.ndarray) -> np.ndarray:
    """Energy at s, given the energy at some other point s_known rather than at s = 0."""
    e_known = np.asarray(e_known, dtype=np.float64)
    s_known = np.asarray(s_known, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    delta_ion = IONISATION_GEV_PER_M * (s - s_known)
    delta_showers = _cumulative_at(chain, s) - _cumulative_at(chain, s_known)
    return e_known - delta_ion - delta_showers


def cylinder_crossing(ref: np.ndarray, d: np.ndarray, centre: np.ndarray,
                      radius: float, z_half: float) -> Tuple[np.ndarray, np.ndarray]:
    """Where the track line enters and leaves a vertical cylinder.

    Returns (s_in, s_out) in track coordinates, NaN where the line misses the cylinder. The
    cylinder is the finite one: the radial condition and the end caps are intersected, so a
    track passing beside the cluster at the right radius but above its top is correctly
    reported as a miss.
    """
    ref = np.asarray(ref, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    delta = ref - centre

    # Radial interval, from |delta_xy + s * d_xy| = radius.
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2.0 * (delta[..., 0] * d[..., 0] + delta[..., 1] * d[..., 1])
    c = delta[..., 0] ** 2 + delta[..., 1] ** 2 - radius ** 2
    disc = b ** 2 - 4.0 * a * c

    vertical = a <= 1e-12                       # d has no horizontal component
    safe_a = np.where(vertical, 1.0, a)
    safe_disc = np.where(disc > 0.0, disc, 0.0)
    root = np.sqrt(safe_disc)
    r_lo = (-b - root) / (2.0 * safe_a)
    r_hi = (-b + root) / (2.0 * safe_a)
    # A vertical track is inside the radial band everywhere, or nowhere.
    inside_band = c <= 0.0
    r_lo = np.where(vertical, np.where(inside_band, -np.inf, np.nan), r_lo)
    r_hi = np.where(vertical, np.where(inside_band, np.inf, np.nan), r_hi)
    r_lo = np.where(~vertical & (disc <= 0.0), np.nan, r_lo)
    r_hi = np.where(~vertical & (disc <= 0.0), np.nan, r_hi)

    # Cap interval, from |ref_z + s * d_z - centre_z| = z_half.
    dz = d[..., 2]
    horizontal = np.abs(dz) <= 1e-12
    safe_dz = np.where(horizontal, 1.0, dz)
    z1 = (-z_half - delta[..., 2]) / safe_dz
    z2 = (z_half - delta[..., 2]) / safe_dz
    z_lo = np.minimum(z1, z2)
    z_hi = np.maximum(z1, z2)
    within_caps = np.abs(delta[..., 2]) <= z_half
    z_lo = np.where(horizontal, np.where(within_caps, -np.inf, np.nan), z_lo)
    z_hi = np.where(horizontal, np.where(within_caps, np.inf, np.nan), z_hi)

    s_in = np.maximum(r_lo, z_lo)
    s_out = np.minimum(r_hi, z_hi)
    miss = ~(s_in < s_out)                      # NaN-safe: a NaN comparison is False
    return np.where(miss, np.nan, s_in), np.where(miss, np.nan, s_out)


def entry_energy(chain: Chain, e_ref: np.ndarray, s_start: np.ndarray,
                 e_bundle_reg: np.ndarray, s_in: np.ndarray,
                 s_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Muon energy where it enters a volume, with a status saying what the number means.

    The order of the tests is the substance of this function. "Died before entry" cannot be
    decided from a negative `e_ref` alone -- a muon dead at the reference point may well have
    been alive earlier, upstream of it -- so it is decided from the propagated energy, and
    therefore last.
    """
    e_ref = np.asarray(e_ref, dtype=np.float64)
    s_start = np.asarray(s_start, dtype=np.float64)
    e_bundle_reg = np.asarray(e_bundle_reg, dtype=np.float64)
    s_in = np.asarray(s_in, dtype=np.float64)
    s_out = np.asarray(s_out, dtype=np.float64)

    sentinel = e_ref == np.float32(SENTINEL_E_REF)
    # The trajectory, not the line: a line that crosses the volume upstream of where the muon
    # begins is not a crossing the muon ever makes.
    no_crossing = np.isnan(s_in) | (s_start > s_out)
    born_inside = ~no_crossing & (s_start > s_in)

    # Where the energy is taken from, and where it is taken at.
    e_known = np.where(sentinel, e_bundle_reg, e_ref)
    s_known = np.where(sentinel, s_start, 0.0)
    s_target = np.where(born_inside, s_start, s_in)

    with np.errstate(invalid="ignore"):
        energy = energy_between(chain, e_known, s_known, np.where(no_crossing, 0.0, s_target))

    status = np.where(sentinel, STATUS_SENTINEL, STATUS_PROPAGATED)
    status = np.where(born_inside,
                      np.where(sentinel, STATUS_SENTINEL_INSIDE, STATUS_BORN_INSIDE), status)
    dead = ~no_crossing & (energy <= 0.0)
    status = np.where(dead, STATUS_DIED_BEFORE, status)
    status = np.where(no_crossing, STATUS_NO_CROSSING, status)
    energy = np.where(no_crossing | dead, np.nan, energy)
    return energy, status.astype(np.int8)


def to_cluster_frame(xyz: np.ndarray, cluster_centre: np.ndarray) -> np.ndarray:
    """Shift global coordinates into the frame the hit coordinates use."""
    return np.asarray(xyz, dtype=np.float64) - np.asarray(cluster_centre, dtype=np.float64)
