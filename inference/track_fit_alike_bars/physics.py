"""Constants and track geometry, ported from BARS v2.3.2.

Every function names the BARS routine it reproduces.  BARS carries two copies of
most formulas -- `butils/BHelperFunctions.cc` (used inside the fit) and
`bmuon/BMuonNamespace.cc` (used by RecollectHits and the derived variables) -- and
the copies differ in ways that change numbers, so the differences are kept:

* the hit-probability routines use different distances: `GetPHitChan` the
  perpendicular distance to the track, `PHitChannel` that distance divided by
  sin(theta_c), i.e. the light path;
* `c_baikal` (BMuonNamespace) is 218826621 m/s, `c / 1.37` (BHelperFunctions) is
  218826611.7 m/s.  The relative difference is 4e-8, so one value is used.

Units: metres, nanoseconds, radians unless a name says otherwise.

Track convention (BARS): a track is a point `ref`, a unit `direction` of motion and
the time `t0` at which the muon passes `ref`.  `theta` is the polar angle of the
direction of motion, so an up-going muon has theta < 90 degrees.
"""
from __future__ import annotations

from functools import lru_cache
from math import atan2, cos, exp, sin, sqrt, tan
from pathlib import Path

import numpy as np
from numba import njit

# ── constants (bbase/BPhysNamespace.h, butils/BHelperFunctions.cc) ─────────────
C_VACUUM = 0.299792458          # m/ns, BARS::Phys::c
C_BAIKAL = 0.218826621          # m/ns, BARS::Phys::c_baikal
N_GROUP = 1.37                  # BARS::Phys::n, BHelperFunctions::GetWRefraction
C_WATER = C_VACUUM / N_GROUP    # BHelperFunctions::GetCWater
CHER_RAD = 0.719888             # BARS::Phys::cher_rad (41.25 deg)

# The fit and the pHit routines take the Cherenkov angle from cos = 1/n with the
# GROUP index (43.1 deg), not CHER_RAD.  Both are in BARS; both are kept.
COS_C = 1.0 / N_GROUP
SIN_C = sqrt(1.0 - COS_C * COS_C)
TAN_C = SIN_C / COS_C

#: Scanfit light-flight factor, BARS::Muon::LightFlightTime and MuDirectionalTest:
#: n / sin(theta_c) - 1 / tan(theta_c) with theta_c = CHER_RAD.
K_SCANFIT = N_GROUP / sin(CHER_RAD) - 1.0 / tan(CHER_RAD)

#: Angular sensitivity of the Hamamatsu 10" OM, polynomial in cos of the angle
#: between +z and the direction from the OM to the emission point.
#: BHelperFunctions::GetAmplAngularCorrection.  The PMT looks down: cos = -1
#: gives 0.999, cos = +1 gives 0.014.
SENS_COEF = (0.3082, -0.54192, 0.19831, 0.04912)

#: Effective area of a 10" PMT, cm^2 (25.4^2 * pi / 4).
SEFF_10INCH = 506.7074791

PHIT_TABLE_PATH = Path(__file__).with_name("quant10HE.data")


@lru_cache(maxsize=1)
def phit_table() -> np.ndarray:
    """`nAvg` as BARS reads it from `config/quant10HE.data`, indexed by metres.

    BARS reads the file with `ifile >> dd >> nAvg` in a loop of 1000 and stops at
    the first `dd == floor(dist + 0.5)`.  The file starts with a three-number
    header (0, 1, 1001) that the loop consumes as data, so:

    * round(dist) == 0 matches the header pair (0, 1) and gives nAvg = 1;
    * round(dist) == 1001 matches (1001, 0) and gives nAvg = 0;
    * 1..997 give the table row;
    * anything else leaves the loop exhausted on row 997.

    The returned array holds that mapping for indices 0..1001; `lookup_navg` clips
    larger distances to row 997.  Copied from BARS master; identical to v2.3.2.
    """
    rows = np.loadtxt(PHIT_TABLE_PATH, skiprows=3)
    table = rows[:, 1]
    out = np.empty(1002)
    out[0] = 1.0
    out[1:998] = table[1:998]
    out[998:1001] = table[997]
    out[1001] = 0.0
    return out


# ── numba kernels on one hit ───────────────────────────────────────────────────
@njit(cache=True)
def om_sensitivity(cos_angle: float) -> float:
    c = cos_angle
    return (SENS_COEF[0] + c * SENS_COEF[1] + c * c * SENS_COEF[2]
            + c * c * c * SENS_COEF[3])


@njit(cache=True)
def propagation_time(rx, ry, rz, dx, dy, dz, px, py, pz) -> float:
    """Time from the track point r to the first Cherenkov photon at OM p.

    BHelperFunctions::GetPropagationTime (== BARS::Muon::PropagationTime).  The
    emission point may lie behind r, so the result can be negative.  Note the
    source clamps cos^2 > 1 to cos = +1, sign lost; kept.
    """
    ax, ay, az = px - rx, py - ry, pz - rz
    mod_am = sqrt(ax * ax + ay * ay + az * az)
    if mod_am == 0.0:
        return 0.0
    mod_s = sqrt(dx * dx + dy * dy + dz * dz)
    cos_a = (ax * dx + ay * dy + az * dz) / (mod_am * mod_s)
    if cos_a * cos_a > 1.0:
        cos_a = 1.0
    sin_a = sqrt(1.0 - cos_a * cos_a)
    d_muon = mod_am * cos_a - mod_am * sin_a / TAN_C
    d_light = mod_am * sin_a / SIN_C
    return d_muon / C_VACUUM + d_light / C_WATER


@njit(cache=True)
def _angle_cos(ax, ay, az, bx, by, bz) -> float:
    """cos of TVector3::Angle, including its clamping and zero-vector rule."""
    ptot2 = (ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz)
    if ptot2 <= 0.0:
        return 1.0
    arg = (ax * bx + ay * by + az * bz) / sqrt(ptot2)
    if arg > 1.0:
        arg = 1.0
    if arg < -1.0:
        arg = -1.0
    return arg


@njit(cache=True)
def track_distance(rx, ry, rz, dx, dy, dz, px, py, pz) -> float:
    """Perpendicular distance from OM p to the track line.

    BHelperFunctions::GetTrackDistanceToPoint (== BARS::Muon::TrackDistanceToPoint).
    """
    ax, ay, az = px - rx, py - ry, pz - rz
    mod_am = sqrt(ax * ax + ay * ay + az * az)
    if mod_am == 0.0:
        return 0.0
    c = _angle_cos(ax, ay, az, dx, dy, dz)
    return mod_am * sqrt(1.0 - c * c)


@njit(cache=True)
def om_light_cos(rx, ry, rz, dx, dy, dz, px, py, pz) -> float:
    """cos of the angle between +z and the direction OM -> emission point.

    BHelperFunctions::GetOMlightAngle (== BARS::Muon::OMlightAngle), returned as a
    cosine because every caller takes cos of it.
    """
    ax, ay, az = px - rx, py - ry, pz - rz
    mod_am = sqrt(ax * ax + ay * ay + az * az)
    c = _angle_cos(ax, ay, az, dx, dy, dz)
    s = sqrt(1.0 - c * c)
    d_muon = mod_am * c - mod_am * s / TAN_C
    ex = rx + dx * d_muon - px
    ey = ry + dy * d_muon - py
    ez = rz + dz * d_muon - pz
    return _angle_cos(0.0, 0.0, 1.0, ex, ey, ez)


@njit(cache=True)
def lookup_navg(dist: float, table: np.ndarray) -> float:
    """`nAvg` for a distance, reproducing BARS' read loop (see `phit_table`)."""
    idx = int(np.floor(dist + 0.5))
    if idx < 0 or idx >= table.shape[0]:
        return table[997]
    return table[idx]


@njit(cache=True)
def phit_light_path(rx, ry, rz, dx, dy, dz, px, py, pz, table) -> float:
    """P(>=1 p.e.) at OM p.  BARS::Muon::PHitChannel -- used for fLogPHit_30_min.

    The distance is the light path, perpendicular distance / sin(theta_c).
    """
    dist = track_distance(rx, ry, rz, dx, dy, dz, px, py, pz) / SIN_C
    n_avg = lookup_navg(dist, table)
    if dist == 0.0:
        return 1.0
    sens = om_sensitivity(om_light_cos(rx, ry, rz, dx, dy, dz, px, py, pz))
    mu = sens * n_avg * SEFF_10INCH / dist
    return 1.0 - exp(-mu)


@njit(cache=True)
def phit_perpendicular(rx, ry, rz, dx, dy, dz, px, py, pz, table, npe) -> float:
    """P(>=npe p.e.) at OM p.  BMuonCandidate::GetPHitChan -- used for fPHit.

    The distance is the perpendicular one, not the light path.
    """
    dist = track_distance(rx, ry, rz, dx, dy, dz, px, py, pz)
    n_avg = lookup_navg(dist, table)
    if dist == 0.0:
        return 1.0
    sens = om_sensitivity(om_light_cos(rx, ry, rz, dx, dy, dz, px, py, pz))
    mu = sens * n_avg * SEFF_10INCH / dist
    # 1 - poisson_cdf(npe - 1, mu)
    term = exp(-mu)
    cdf = term
    for k in range(1, npe):
        term *= mu / k
        cdf += term
    return 1.0 - cdf


# ── track parametrisation used by the fit (BHelperFunctions) ──────────────────
@njit(cache=True)
def vector_angles(dx, dy, dz):
    """(theta, phi) of a vector as TVector3::Theta / TVector3::Phi return them."""
    if dx == 0.0 and dy == 0.0:
        phi = 0.0
        theta = 0.0 if dz == 0.0 else atan2(0.0, dz)
    else:
        phi = atan2(dy, dx)
        theta = atan2(sqrt(dx * dx + dy * dy), dz)
    return theta, phi


@njit(cache=True)
def ref_xyz(cx, cy, cz, x0, y0, dx, dy, dz):
    """BHelperFunctions::GetRefXYZ: centre + Rz(phi) Ry(theta) (x0, y0, 0).

    The angles are those of the direction vector (TVector3::Theta/Phi), not the
    raw fit parameters: a negative theta parameter flips the x0 axis.
    """
    th, ph = vector_angles(dx, dy, dz)
    ct, st, cp, sp = cos(th), sin(th), cos(ph), sin(ph)
    return (cx + x0 * ct * cp - y0 * sp,
            cy + x0 * ct * sp + y0 * cp,
            cz - x0 * st)


@njit(cache=True)
def x0y0_of(cx, cy, cz, px, py, pz, dx, dy, dz):
    """BHelperFunctions::CalculateX0 / CalculateY0 of the line (p, d) about c."""
    # PCA of the line to the centre (GetPCAPoint)
    ax, ay, az = cx - px, cy - py, cz - pz
    mod = sqrt(ax * ax + ay * ay + az * az)
    kat = mod * _angle_cos(ax, ay, az, dx, dy, dz)
    qx, qy, qz = px + kat * dx, py + kat * dy, pz + kat * dz
    # rotate by Ry(-theta) Rz(-phi) and subtract the rotated centre
    th, ph = vector_angles(dx, dy, dz)
    ct, st, cp, sp = cos(th), sin(th), cos(ph), sin(ph)
    ux, uy = qx - cx, qy - cy
    uz = qz - cz
    # Rz(-phi): (x cp + y sp, -x sp + y cp, z); then Ry(-theta)
    rx = ux * cp + uy * sp
    ry = -ux * sp + uy * cp
    x0 = rx * ct - uz * st
    return x0, ry


@njit(cache=True)
def time_at_pca(cx, cy, cz, px, py, pz, dx, dy, dz, t_at_p) -> float:
    """BHelperFunctions::CalculateTimeX0Y0: time at the PCA to c."""
    ax, ay, az = cx - px, cy - py, cz - pz
    mod = sqrt(ax * ax + ay * ay + az * az)
    kat = mod * _angle_cos(ax, ay, az, dx, dy, dz)
    return t_at_p + kat / C_VACUUM


@njit(cache=True)
def direction_of(theta, phi):
    return sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)
