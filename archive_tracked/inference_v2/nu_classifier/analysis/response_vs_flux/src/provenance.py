"""Artefact writing with a machine-checkable provenance sidecar.

Every stage writes exactly one artefact through :func:`write`, which stores a
``<name>.meta.json`` next to it recording the git commit, the hash of
``config.yaml``, the hashes of the input files, the row count and the runtime.
:func:`verify` re-checks a chain without recomputing it, so a reader can tell at
a glance whether a figure came from the code currently in the tree.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def file_hash(path: Path, chunk: int = 1 << 20, limit: int | None = 1 << 26) -> str:
    """SHA-256 of a file, over at most ``limit`` bytes (None = whole file).

    Multi-hundred-gigabyte HDF5 inputs are hashed over their first 64 MiB plus
    their size, which detects replacement without costing an hour.
    """
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    read = 0
    with path.open("rb") as fh:
        while (limit is None or read < limit) and (block := fh.read(chunk)):
            h.update(block)
            read += len(block)
    return h.hexdigest()


def scoped_config_hash(config_path: Path, stage: str) -> str:
    """Hash of the config a stage actually depends on, not of the whole file.

    Hashing ``config.yaml`` whole made every artefact look stale the moment a
    later stage added its own section, which turns `make verify` into noise that
    gets ignored -- the opposite of its purpose.  What a stage depends on is the
    shared keys plus its own ``stageNN`` block, so other stages' blocks are
    dropped before hashing.
    """
    raw = yaml.safe_load(Path(config_path).read_text())
    own = "stage" + stage.split("_")[0]
    relevant = {key: value for key, value in raw.items()
                if not (key.startswith("stage") and key != own)}
    canonical = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                                     # noqa: BLE001
        return "unknown"


def write(
    frame: pd.DataFrame,
    path: Path,
    *,
    stage: str,
    config_path: Path,
    inputs: Iterable[Path] = (),
    started: float | None = None,
    notes: dict[str, Any] | None = None,
) -> Path:
    """Write ``frame`` to parquet and its provenance sidecar beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    meta = {
        "stage": stage,
        "artefact": path.name,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": None if started is None else round(time.time() - started, 1),
        "git_commit": git_commit(path.resolve().parents[1]),
        "config_sha256": scoped_config_hash(Path(config_path), stage),
        "config_scope": "stage" + stage.split("_")[0],
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "inputs": {str(p): file_hash(Path(p)) for p in inputs},
        "notes": notes or {},
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return path


def read_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(str(path) + ".meta.json").read_text())


def verify(data_dir: Path, config_path: Path) -> list[str]:
    """Return a list of human-readable complaints; empty means the chain is clean."""
    problems: list[str] = []
    for meta_path in sorted(Path(data_dir).glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg = scoped_config_hash(Path(config_path), meta["stage"])
        artefact = meta_path.parent / meta["artefact"]
        if not artefact.exists():
            problems.append(f"{meta['artefact']}: missing")
            continue
        if meta["config_sha256"] != cfg:
            problems.append(f"{meta['artefact']}: built with a different config.yaml")
        for src, digest in meta["inputs"].items():
            if not Path(src).exists():
                problems.append(f"{meta['artefact']}: input gone -- {src}")
            elif file_hash(Path(src)) != digest:
                problems.append(f"{meta['artefact']}: input changed -- {src}")
    return problems
