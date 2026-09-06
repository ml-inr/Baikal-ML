"""The five sources: where they live, what they contain, what is broken in them.

Every exclusion here is a measured result, and each carries its measurement in the
comment.  An unexplained cut is one that gets copied into the next analysis by
someone who cannot check it -- that has already happened once in this project with a
cluster exclusion that turned out to be refuted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .schema import EXP_RECO_COLUMNS, MC_RECO_COLUMNS

#: Project root, found by walking up to the marker file.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").exists())
H5DIR = ROOT / "data_manager/data/h5datasets"
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"

#: The sig-noise model whose probabilities selected the hits.  The batch size it ran
#: at is part of the identity of that selection, not a speed setting -- see
#: doc/sig_noise_batch_size.md.  Recorded per run in `run_info.json`.
SN_TAG = "k_nsol_labelneq0_da_hs128_k0p0001"


@dataclass(frozen=True)
class Source:
    """One HDF5 source and everything that differs about it."""

    name: str
    stem: str                        # HDF5 file stem, which is not the source name
    groups: tuple[str, ...]          # top-level HDF5 groups
    has_prime: bool                  # MC truth of the primary particle
    reco_columns: tuple[str, ...]    # empty when there is no `reco_prty`
    has_header: bool                 # experimental `header_prty`

    #: Runs excluded with evidence.  Key is the `part_key`.
    bad_runs: tuple[str, ...] = ()
    #: Clusters excluded with evidence.
    bad_clusters: tuple[int, ...] = ()
    #: Minimum `n_gt_sig_hits` from the probabilities file; None disables the cut.
    #: Not expressible in SQL, so it is applied after reading and always reported.
    fragment_cut: int | None = None
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def h5(self) -> Path:
        return H5DIR / f"{self.stem}.h5"

    @property
    def probs(self) -> Path:
        return H5DIR / f"{self.stem}_probs_{SN_TAG}.h5"

    @property
    def has_reco(self) -> bool:
        return bool(self.reco_columns)


SOURCES: dict[str, Source] = {
    "mc_merged": Source(
        name="mc_merged", stem="baikal_mc_merged",
        groups=("muatm_2020", "nuatm_2020", "nue2_2020"),
        has_prime=True, reco_columns=(), has_header=False,
        notes={"probs_coverage":
               "sig-noise probabilities exist for 10,100 of the 20,004 HDF5 parts; "
               "the rest were never scored and simply do not appear."},
    ),
    "exp_full": Source(
        name="exp_full", stem="exp_full",
        groups=("exp_full",),
        has_prime=False, reco_columns=(), has_header=True,
        bad_runs=("part_s2020_c02_r0020", "part_s2020_c02_r0249"),
        notes={"bad_runs":
               "channels 222/224/225/227 (one string of cluster 2) emit up to "
               "3169.8 p.e. against a median hit of 0.97; 36-78% of their hits "
               "exceed 100 p.e. where a normal channel sits at 0.0066%. Events "
               "touching them score above 0.8 8.7x more often. These are the only "
               "two runs where those channels fire.",
               "cluster_1_refuted":
               "cluster 1 is not in exp_full at all, and the old cluster 1/4 "
               "exclusion was refuted -- do not reintroduce it."},
    ),
    "mc_reco": Source(
        name="mc_reco", stem="baikal_mc_reco",
        groups=("muatm", "nuatm_conv", "nuatm_prompt", "nue2"),
        has_prime=True, reco_columns=MC_RECO_COLUMNS, has_header=False,
        fragment_cut=5,
        notes={"fragments":
               "the converter ran with split_multi: true, so a physical event that "
               "lit several clusters is stored as one row per cluster, each carrying "
               "the reconstruction of the WHOLE event beside the hits of one piece. "
               "Those pieces have 0-5 true signal hits and a median of one raw hit, "
               "against 66 for whole events."},
    ),
    "exp_reco": Source(
        name="exp_reco", stem="exp_reco",
        groups=("exp_reco",),
        has_prime=False, reco_columns=EXP_RECO_COLUMNS, has_header=True,
        bad_clusters=(1,),
        bad_runs=("part_s2020_c02_r0020", "part_s2020_c02_r0249"),
        notes={"bad_runs":
               "same four channels as in exp_full, measured here independently and "
               "worse: in part_s2020_c02_r0249 channels 222/224/225/227 have a "
               "MEDIAN charge of 28,754 p.e. and a maximum of 277,734, against a "
               "median of 1.00 for every other channel in the same run; 85% of their "
               "hits exceed 100 p.e. against 0.06%. Only r0249 exists in exp_reco -- "
               "r0020 is listed because it is the same physical fault and costs "
               "nothing if the run is absent.",
               "cluster_1":
               "t_span median 17,600 ns against 4,900 ns in every other cluster; the "
               "sig-noise filter finds a median of ZERO signal hits, only 30% of "
               "events reach scoring at all, and what survives is accepted at 6.9% "
               "against 0.94-1.18%. An exp_reco fault; says nothing about exp_full.",
               "subset":
               "exp_reco.h5 is an exact subset of exp_reco_full_2020.h5: 370 of 1818 "
               "parts, 14.4M of 72.1M events, byte-identical where they overlap. The "
               "subset is capped at ~52 runs per cluster, so cluster coverage ranges "
               "14% (c7) to 56% (c4) and the cluster mix is distorted. Reweighting "
               "to the full file moves the high-score rate 0.290% -> 0.271% once "
               "cluster 1 is excluded."},
    ),
    "exp": Source(
        name="exp", stem="exp",
        groups=("exp",),
        has_prime=False, reco_columns=(), has_header=True,
        notes={"legacy": "the 25k-capped experimental set, ~650k events. Superseded "
                         "by exp_full; kept because old predictions reference it."},
    ),
}

#: Catalog `data_class` -> HDF5 group.  They agree everywhere except `mc_reco`,
#: whose production has no year suffix in the file while the catalog adds one.
_MC_RECO_CLASS_TO_GROUP = {
    "muatm_2020": "muatm", "nuatm_conv_2020": "nuatm_conv",
    "nuatm_prompt_2020": "nuatm_prompt", "nue2_2020": "nue2",
}


def h5_group(source: str, data_class: str) -> str:
    if source == "mc_reco":
        return _MC_RECO_CLASS_TO_GROUP.get(data_class, data_class)
    return data_class


def describe(name: str) -> str:
    """One paragraph per source, including why each exclusion exists."""
    spec = SOURCES[name]
    lines = [f"{spec.name}  ({spec.h5.name})",
             f"  groups        {', '.join(spec.groups)}",
             f"  prime_prty    {'yes' if spec.has_prime else 'no'}",
             f"  reco_prty     {len(spec.reco_columns) or 'no'}"]
    if spec.bad_runs:
        lines.append(f"  bad runs      {', '.join(spec.bad_runs)}")
    if spec.bad_clusters:
        lines.append(f"  bad clusters  {list(spec.bad_clusters)}")
    if spec.fragment_cut is not None:
        lines.append(f"  fragment cut  n_gt_sig_hits > {spec.fragment_cut}")
    for key, text in spec.notes.items():
        lines.append(f"  [{key}] {text}")
    return "\n".join(lines)
