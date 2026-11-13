"""The DQM channel mask BARS applies to experimental data, and where to get it.

BARS builds the mask in `BApplyDQMMask` out of two things: a per-run DQM solution
file (`j{run}.fit.solution.txt`, `DQM_FIT_SOLUTION`, dqm-version v1.3) and the
per-event section mask (`BSectionStatus`).  The Gantt histograms are read only
when `Mask.UseGantt` is on, and it is **0** in the 2020 resource files, so for
2020 the solution file alone decides every channel whose section is active.

The mask is not in the HDF5 files, so a fit meant to reproduce BARS on
experimental data has to be handed one:

* `exp_reco` parts have their ROOT file next to them, and its `ChannelMaskDQM`
  container *is* the mask BARS used, per event -- exact by construction;
* `exp_full` and `exp` have no ROOT file locally, but the solution file of every
  one of their runs is present, which gives the run-level part of the mask;
* the MC sources are reconstructed **without** a DQM mask (`Mask.UseDQMChannelMask`
  is 0 in their resource files), so `None` is the faithful answer for them.

Measured on `c01/r0027`: the solution file reproduces the ROOT container exactly,
the same 40 channels, event after event.  Where the section mask varies inside a
run the ROOT container switches off further whole sections, which the solution
file cannot know; the size of that residue is measured by `tests/test_fit.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import uproot

#: `BChannelMask::EMode`; anything else means "not usable by the reconstruction".
ON = 1
#: `BApplyDQMMask` fails a channel on these solution codes (in the file's legend:
#: 0 excluded by configuration, 1 empty data, 4 bad data, 7 LED+bad, 10 luminescent
#: layer + bad).  Codes 2/3/5/6/8/9 are good or normal data and stay on.
FAIL_CODES = frozenset({0, 1, 4, 7, 10})
#: A cluster is 24 sections of 12 channels, or 27 in the versions that have one more
#: string -- the two sizes `BDQMChannelMask` accepts.
N_CHANNELS = (288, 324)

EVENT_ID_BRANCH = "BJointHeader./BJointHeader.fEventIDCC"
MASK_BRANCH = "ChannelMaskDQM./ChannelMaskDQM.fFlag"
#: `fFlag` is a pointer to the flag array, so uproot needs the layout told to it:
#: a jagged array of int32 behind a four-byte (shape) header.  Use it as
#: `tree[MASK_BRANCH].array(FLAGS, ...)`.
FLAGS = uproot.AsJagged(uproot.AsDtype(">i4"), header_bytes=1)

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Where the ROOT files of the reco sources and the DQM solutions are unpacked.
RECO_ROOT = REPO_ROOT / "data_manager/data/exp_reco_root/exp_reco_files"
SOLUTION_ROOT = REPO_ROOT / "data_manager/data/exp_root"

_PART = re.compile(r"part_s(\d+)_c(\d+)_r(\d+)")


def part_run(part_key: str) -> tuple[int, int, int]:
    """`part_s2020_c04_r0100` -> (2020, 4, 100)."""
    m = _PART.match(part_key)
    if not m:
        raise ValueError(f"unexpected part key: {part_key!r}")
    return tuple(int(g) for g in m.groups())


def reco_root_path(part_key: str, root: Path = RECO_ROOT) -> Path:
    """ROOT file of an `exp_reco` part, e.g. `2020_cl4_run100_scl_nu_DATA2020.root`."""
    season, cluster, run = part_run(part_key)
    return Path(root) / f"{season}_cl{cluster}_run{run}_scl_nu_DATA{season}.root"


def solution_path(season: int, cluster: int, run: int,
                  root: Path = SOLUTION_ROOT) -> Path | None:
    """DQM solution file of a run, or None when that run's DQM is not unpacked."""
    path = (Path(root) / f"dqm_s{season}_c{cluster:02d}" / f"{run:04d}"
            / f"j{run:04d}.fit.solution.txt")
    return path if path.exists() else None


def dqm_mask_from_solution(path: str | Path) -> np.ndarray:
    """Run-level channel mask from a `j{run}.fit.solution.txt`.

    Reads the `### For channels` block -- everything after it, exactly as
    `BDQMChannelMask` does -- and returns a boolean array, True where the channel
    is usable.  The per-event section part of the mask is not in this file.
    """
    lines = Path(path).read_text().splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("### For channels")), None)
    if start is None:
        raise ValueError(f"{path}: no '### For channels' block")
    codes = [int(tok) for line in lines[start + 1:] for tok in line.split()]
    if len(codes) not in N_CHANNELS:
        raise ValueError(f"{path}: {len(codes)} channels, expected {N_CHANNELS}")
    return np.array([code not in FAIL_CODES for code in codes], dtype=bool)


def dqm_masks_from_root(path: str | Path,
                        event_ids) -> dict[int, np.ndarray]:
    """Channel masks of a BARS ROOT file, keyed by `BJointHeader` event id.

    Only the requested events are read: the ids of the whole file are read first
    (one small branch), then one entry at a time from the mask branch.  An id the
    file does not contain is left out of the result.
    """
    want = {int(e) for e in np.atleast_1d(event_ids)}
    out: dict[int, np.ndarray] = {}
    with uproot.open(path) as f:
        tree = f["Events"]
        ids = tree[EVENT_ID_BRANCH].array(library="np")
        entry_of = {int(v): i for i, v in enumerate(ids)}
        branch = tree[MASK_BRANCH]
        for eid in sorted(want):
            entry = entry_of.get(eid)
            if entry is None:
                continue
            flags = branch.array(FLAGS, entry_start=entry, entry_stop=entry + 1,
                                 library="np")[0]
            out[eid] = np.asarray(flags) == ON
    return out


def root_channel_masks(tree, **window) -> list[np.ndarray]:
    """The `ChannelMaskDQM` masks of a slice of an open `Events` tree, in order.

    `window` is passed to uproot, e.g. `entry_start=0, entry_stop=100`.
    """
    return [np.asarray(f) == ON for f in tree[MASK_BRANCH].array(FLAGS, **window)]


def dqm_masks(source: str, part_key: str, event_ids=None):
    """Mask(s) for one part of a source, in the form that source allows.

    Returns one of:

    * a `{event id: mask}` dict -- an `exp_reco` part, read from its ROOT file;
    * a `(n_channels,)` array -- an `exp_full`/`exp` part, from the run's DQM
      solution file, the same for every event of the run;
    * `None` -- no mask: an MC source (which BARS reconstructs without one), or an
      experimental run whose files are not unpacked locally.
    """
    if source == "exp_reco":
        return dqm_masks_from_root(reco_root_path(part_key), event_ids)
    if source in ("exp_full", "exp"):
        # only these two need the part key parsed: an MC part is named differently,
        # and asking `part_run` about it would raise instead of answering None
        season, cluster, run = part_run(part_key)
        path = solution_path(season, cluster, run)
        return None if path is None else dqm_mask_from_solution(path)
    return None
