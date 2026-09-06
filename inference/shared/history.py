"""Prediction run history logging to prediction_history.csv."""

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_COLUMNS = [
    "timestamp",
    "checkpoint",
    "script",
    "source",
    "db_file",
    "threshold",
    "min_hits",
    "min_strings",
    "n_events_new",
    "n_events_skipped",
    "is_successful",
    "error_msg",
]


def append_history_row(
    history_path: str,
    *,
    checkpoint: str,
    script: str,
    source: str,
    db_file: str,
    threshold: float,
    min_hits: int,
    min_strings: int,
    n_events_new: int,
    n_events_skipped: int,
    is_successful: bool,
    error_msg: str = "",
    timestamp: str | None = None,
) -> None:
    """Append one row to prediction_history.csv, creating the file if needed."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    row = {
        "timestamp":        timestamp or datetime.now().isoformat(timespec="seconds"),
        "checkpoint":       checkpoint,
        "script":           script,
        "source":           source,
        "db_file":          db_file,
        "threshold":        threshold,
        "min_hits":         min_hits,
        "min_strings":      min_strings,
        "n_events_new":     n_events_new,
        "n_events_skipped": n_events_skipped,
        "is_successful":    is_successful,
        "error_msg":        error_msg,
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.debug(f"History written: {history_path}")
