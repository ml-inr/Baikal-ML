"""Per-source provenance for a checkpoint's prediction directory.

One checkpoint directory holds a separate DuckDB per source (`mc_merged`, `exp_full`, …),
but `run_info.json` used to be a single flat dict that each run overwrote, so only the last
run left a record. Scoring 94M MC events and 8M experimental ones and then being unable to
say from the metadata how the MC side selected its hits is exactly the failure this file
exists to prevent — see doc/sig_noise_batch_size.md for what that cost.

Layout written here:

    {
      "schema": 2,
      "checkpoint": "...",
      "runs": {
        "mc_merged": {"h5_path": ..., "sn_probs_source": ..., "sn_batch_size": 256, ...},
        "exp_full":  {...}
      }
    }

A pre-existing flat file is migrated into `runs` under its own `source` rather than
discarded, so nothing already recorded is lost.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

SCHEMA = 2
UNKNOWN = "unknown"

# Keys that belonged to a single run in the old flat layout.
_RUN_KEYS = {"source", "h5_path", "npy_dir", "threshold", "min_hits", "min_strings",
             "n_events_total", "sn_probs_source", "sn_batch_size", "elapsed_s", "written"}


def probs_batch_size(probs_h5: Optional[str]) -> Any:
    """The sig-noise batch size a probs file was written with.

    This is the number that defines the hit selection, so a run that reads precomputed
    probabilities must record it — the batch it would have used itself is irrelevant.
    Returns UNKNOWN rather than a plausible guess when the file predates the attribute.
    """
    if not probs_h5:
        return None
    try:
        import h5py
        with h5py.File(probs_h5, "r") as f:
            bs = f.attrs.get("sig_noise_batch_size")
        return int(bs) if bs is not None else UNKNOWN
    except Exception:
        return UNKNOWN


def _migrate(info: dict) -> dict:
    """Fold a legacy flat record into `runs`, keyed by the source it described."""
    if "runs" in info or not (info.keys() & _RUN_KEYS):
        info.setdefault("runs", {})
        return info
    legacy_source = info.get("source", "unknown_source")
    run = {k: info.pop(k) for k in list(info) if k in _RUN_KEYS}
    run.setdefault("migrated_from_flat_record", True)
    info["runs"] = {legacy_source: run}
    return info


def write_run_info(path: Path, source: str, checkpoint: Optional[str] = None,
                   **fields: Any) -> None:
    """Record one source's run, leaving every other source's record untouched."""
    path = Path(path)
    info = json.loads(path.read_text()) if path.exists() else {}
    info = _migrate(info)

    info["schema"] = SCHEMA
    if checkpoint:
        info["checkpoint"] = checkpoint

    run = dict(info["runs"].get(source, {}))
    run.update({k: v for k, v in fields.items() if v is not None})
    run["source"] = source
    run["written"] = datetime.now().isoformat(timespec="seconds")
    info["runs"][source] = run

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
