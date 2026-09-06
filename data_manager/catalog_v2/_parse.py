"""Part-key parsing and ROOT-filename helpers for the catalog_v2 builders.

Self-contained: the original `data_manager.catalog._parse` module is gone, so
the five helpers are implemented directly here.

Part-key conventions (top group / `raw/ev_starts/<part_key>/data`):
    exp       : part_s2020_c01_r0027                     (s<season>_c<cc>_r<rrrr>)
    exp_reco  : part_s2020_c01_r0039                     (same shape as exp)
    mc_reco   : part_2020_cl1_run22101_scl_nu_MC_s19-21  (root stem with 'part_')
    mc_merged : part_10225                               (opaque index; not parseable)

ROOT-filename conventions:
    exp       : s2020_c01_r0027.root
    exp_reco  : 2020_cl1_run100_scl_nu_DATA2020.root
    mc_reco   : 2020_cl1_run10000_scl_nu_MC_s19-21.root
"""

import re

__all__ = [
    "parse_exp_part_key",
    "parse_mc_part_key",
    "exp_root_filename",
    "exp_reco_root_filename",
    "mc_root_filename_pattern",
]

# part_s2020_c01_r0027  ->  season=2020, cluster=1, run=27
_EXP_PART_RE = re.compile(r"^part_s(\d+)_c(\d+)_r(\d+)$")

# part_2020_cl1_run22101_scl_nu_MC_s19-21  ->  season=2020, cluster=1, run=22101
_MC_RECO_PART_RE = re.compile(r"_cl(\d+)_run(\d+)")
_SEASON_RE = re.compile(r"part_(\d+)_cl")


def parse_exp_part_key(part_key: str) -> tuple[int, int, int]:
    """Parse an exp / exp_reco part key into (season, cluster, run) ints.

    >>> parse_exp_part_key("part_s2020_c01_r0027")
    (2020, 1, 27)
    """
    m = _EXP_PART_RE.match(part_key)
    if not m:
        raise ValueError(f"Cannot parse exp part key: {part_key!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_mc_part_key(part_key: str) -> tuple[int, int, int]:
    """Parse a reco-style MC part key into (season, cluster, run) ints.

    Targets the mc_reco naming (``part_<season>_cl<cluster>_run<run>_...``).
    The flat mc_merged keys (e.g. ``part_10225``) carry no season/cluster/run
    and raise ValueError — those builders derive the fields from the HDF5
    payload instead.

    >>> parse_mc_part_key("part_2020_cl1_run22101_scl_nu_MC_s19-21")
    (2020, 1, 22101)
    """
    clrun = _MC_RECO_PART_RE.search(part_key)
    season = _SEASON_RE.search(part_key)
    if not clrun or not season:
        raise ValueError(f"Cannot parse mc part key: {part_key!r}")
    return int(season.group(1)), int(clrun.group(1)), int(clrun.group(2))


def exp_root_filename(season: int, cluster: int, run: int) -> str:
    """Build the exp ROOT filename: ``s2020_c01_r0027.root``."""
    return f"s{season}_c{int(cluster):02d}_r{int(run):04d}.root"


def exp_reco_root_filename(season: int, cluster: int, run: int) -> str:
    """Build the exp_reco ROOT filename: ``2020_cl1_run100_scl_nu_DATA2020.root``."""
    return f"{season}_cl{int(cluster)}_run{int(run)}_scl_nu_DATA{season}.root"


def mc_root_filename_pattern(season: int, cluster: int, run: int) -> str:
    """Build a glob pattern for an mc_reco ROOT file.

    The trailing simulation tag (e.g. ``_scl_nu_MC_s19-21``) can vary between
    productions, so it is matched with a wildcard:
    ``2020_cl1_run10000_*.root``.
    """
    return f"{season}_cl{int(cluster)}_run{int(run)}_*.root"
