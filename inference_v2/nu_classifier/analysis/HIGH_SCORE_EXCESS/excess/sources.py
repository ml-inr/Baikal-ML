"""The four sources, and everything that differs between them.

Each entry answers, in one place: which HDF5 groups it has, what the prediction
database carries, which events the model already saw, and which events are known
to be broken.  Every exclusion below is a measured result, not a precaution, and
each carries the measurement in its comment -- an unexplained cut is one that
gets copied into the next analysis by someone who cannot check it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inference_v2.shared.catalog_query import MC_RECO_PTYPE_TO_DATA_CLASS
from inference_v2.shared.reco_schema import EXP_RECO_COLUMNS, MC_RECO_COLUMNS

from . import paths

#: The h8s3 quality cut, in the `predictions` table's own terms.
MIN_SN_HITS = 8
MIN_SN_STRINGS = 3
QUALITY = f"p.n_sn_hits >= {MIN_SN_HITS} AND p.n_sn_strings >= {MIN_SN_STRINGS}"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    stem: str                      # HDF5 file stem, which is not the DB name
    groups: tuple[str, ...]        # top-level HDF5 groups
    has_splits: bool               # DB carries part_key / local_idx / flags
    has_prime: bool                # MC truth of the primary particle
    reco_columns: tuple[str, ...]  # empty when there is no `reco_prty`
    train_filter: str | None       # SQL removing what the model was fitted on
    flag_masks: tuple[str, ...] = ()   # SQL for faults recorded in `splits`
    run_blacklist: tuple[str, ...] = ()
    cluster_blacklist: tuple[int, ...] = ()
    fragment_cut: int | None = None

    @property
    def h5(self) -> Path:
        return paths.h5(self.stem)

    @property
    def probs(self) -> Path:
        return paths.probs(self.stem)

    @property
    def preds(self) -> Path:
        return paths.preds(self.name)

    @property
    def has_reco(self) -> bool:
        return bool(self.reco_columns)


SOURCES: dict[str, SourceSpec] = {
    # Labelled training source.  `used_for_labels` removes the events whose
    # labels the model was fitted on; their scores are biased.
    "mc_merged": SourceSpec(
        name="mc_merged", stem="baikal_mc_merged",
        groups=("muatm_2020", "nuatm_2020", "nue2_2020"),
        has_splits=True, has_prime=True, reco_columns=(),
        train_filter="NOT s.used_for_labels",
    ),
    # Unlabelled domain-adaptation target.  Two separate removals, because they
    # answer different questions and one may be wanted without the other:
    #   `was_da_target` -- what the model adapted to (training exclusion);
    #   `excluded`      -- runs s2020_c02_r0020 and r0249, where channels
    #                      222/224/225/227 emit up to 3169.8 p.e. against a
    #                      median hit of 0.97, and events touching them score
    #                      above 0.8 8.7x more often.
    # Cluster 1 is not in exp_full at all, and the old cluster 1/4 exclusion is
    # refuted -- do not reintroduce it here (doc: exp_blacklist).
    "exp_full": SourceSpec(
        name="exp_full", stem="exp_full",
        groups=("exp_full",),
        has_splits=True, has_prime=False, reco_columns=(),
        train_filter="NOT s.was_da_target",
        flag_masks=("NOT s.excluded",),
    ),
    # Never used in training, so nothing to exclude on that count.
    # `fragment_cut` drops multi-cluster fragments: the converter ran with
    # `split_multi: true`, so a physical event that lit several clusters is
    # stored as one row per cluster, each carrying the reconstruction of the
    # *whole* event beside the hits of one piece.  Those pieces have 0-5 true
    # signal hits and a median of one raw hit, against 66 for whole events.
    "mc_reco": SourceSpec(
        name="mc_reco", stem="baikal_mc_reco",
        groups=("muatm", "nuatm_conv", "nuatm_prompt", "nue2"),
        has_splits=False, has_prime=True, reco_columns=MC_RECO_COLUMNS,
        train_filter=None, fragment_cut=5,
    ),
    # Cluster 1 is broken here: t_span median 17,600 ns against 4,900 ns
    # elsewhere, the sig-noise filter finds a median of zero signal hits, and
    # what survives is accepted at 6.9% against 0.94-1.18%.  This is an
    # exp_reco fault and says nothing about cluster 1 of exp_full.
    "exp_reco": SourceSpec(
        name="exp_reco", stem="exp_reco",
        groups=("exp_reco",),
        has_splits=False, has_prime=False, reco_columns=EXP_RECO_COLUMNS,
        train_filter=None,
        cluster_blacklist=(1,),
        run_blacklist=("part_s2020_c02_r0020", "part_s2020_c02_r0249"),
    ),
}

#: Catalog `data_class` -> HDF5 group.  They agree everywhere except `mc_reco`,
#: whose production has no year suffix in the file while the catalog adds one.
_DATA_CLASS_TO_GROUP = {
    "mc_reco": {v: k for k, v in MC_RECO_PTYPE_TO_DATA_CLASS.items()},
}


def h5_group(source: str, data_class: str) -> str:
    return _DATA_CLASS_TO_GROUP.get(source, {}).get(data_class, data_class)


def describe() -> str:
    """One line per source, for putting at the top of a notebook run."""
    lines = []
    for spec in SOURCES.values():
        cuts = []
        if spec.train_filter:
            cuts.append("training")
        if spec.flag_masks:
            cuts.append("flagged bad runs")
        if spec.cluster_blacklist:
            cuts.append(f"clusters {list(spec.cluster_blacklist)}")
        if spec.run_blacklist:
            cuts.append(f"{len(spec.run_blacklist)} runs")
        if spec.fragment_cut is not None:
            cuts.append(f"fragments (n_gt_sig_hits > {spec.fragment_cut})")
        lines.append(f"{spec.name:<10} groups={len(spec.groups)}  "
                     f"prime={'yes' if spec.has_prime else 'no ':<3} "
                     f"reco={len(spec.reco_columns) or 'no':<3}  "
                     f"removed: {', '.join(cuts) or 'nothing'}")
    return "\n".join(lines)
