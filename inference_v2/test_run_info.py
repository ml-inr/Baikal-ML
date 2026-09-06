"""Checks on per-source provenance writing.

The property that matters: recording one source must never erase another's record. That is
the bug this replaced — an exp run overwrote the MC run's entry, leaving no way to tell from
metadata how 94M MC events had been scored.

Usage:
    python inference_v2/test_run_info.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from inference_v2.shared.run_info import UNKNOWN, probs_batch_size, write_run_info  # noqa: E402


def test_two_sources_coexist(tmp: Path) -> None:
    p = tmp / "run_info.json"
    write_run_info(p, "mc_merged", checkpoint="ckpt.pth", n_events_total=94_363_339,
                   sn_batch_size=256)
    write_run_info(p, "exp_full", checkpoint="ckpt.pth", n_events_total=8_231_582,
                   sn_batch_size=256)
    info = json.loads(p.read_text())
    assert set(info["runs"]) == {"mc_merged", "exp_full"}, info["runs"].keys()
    assert info["runs"]["mc_merged"]["n_events_total"] == 94_363_339
    assert info["runs"]["exp_full"]["n_events_total"] == 8_231_582
    assert info["checkpoint"] == "ckpt.pth"
    print("  two sources coexist                        ok")


def test_rerun_does_not_touch_the_other(tmp: Path) -> None:
    p = tmp / "rerun.json"
    write_run_info(p, "mc_merged", n_events_total=1)
    write_run_info(p, "exp_full", n_events_total=2)
    write_run_info(p, "mc_merged", n_events_total=3)
    runs = json.loads(p.read_text())["runs"]
    assert runs["mc_merged"]["n_events_total"] == 3
    assert runs["exp_full"]["n_events_total"] == 2, "rerunning one source clobbered another"
    print("  rerunning one source leaves the other      ok")


def test_legacy_flat_file_is_migrated(tmp: Path) -> None:
    """A pre-existing flat record must survive, not be silently dropped."""
    p = tmp / "legacy.json"
    p.write_text(json.dumps({
        "checkpoint": "old.pth", "source": "exp_full",
        "h5_path": "exp_full.h5", "threshold": 0.8, "n_events_total": 7_938_765,
    }, indent=2))
    write_run_info(p, "mc_merged", n_events_total=42)
    info = json.loads(p.read_text())
    assert set(info["runs"]) == {"exp_full", "mc_merged"}, info["runs"].keys()
    assert info["runs"]["exp_full"]["n_events_total"] == 7_938_765, "legacy record lost"
    assert info["runs"]["exp_full"]["migrated_from_flat_record"] is True
    assert "source" not in {k for k in info if k != "runs"}, "flat keys left at top level"
    print("  legacy flat record migrated, not lost      ok")


def test_batch_size_comes_from_the_probs_file(tmp: Path) -> None:
    """The recorded batch must be the one the probs file was made with."""
    stamped, bare = tmp / "stamped.h5", tmp / "bare.h5"
    with h5py.File(stamped, "w") as f:
        f.attrs["sig_noise_batch_size"] = 256
        f.create_dataset("x", data=np.zeros(1))
    with h5py.File(bare, "w") as f:
        f.create_dataset("x", data=np.zeros(1))

    assert probs_batch_size(str(stamped)) == 256
    assert probs_batch_size(str(bare)) == UNKNOWN, "a missing attribute must not be guessed"
    assert probs_batch_size(None) is None
    assert probs_batch_size(str(tmp / "absent.h5")) == UNKNOWN
    print("  batch size read from probs file            ok")


if __name__ == "__main__":
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for t in (test_two_sources_coexist, test_rerun_does_not_touch_the_other,
                  test_legacy_flat_file_is_migrated, test_batch_size_comes_from_the_probs_file):
            try:
                t(tmp)
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {t.__name__}: {exc}")
    print("\n" + ("ALL PASSED" if not failures else f"{failures} FAILED"))
    sys.exit(1 if failures else 0)
