#!/usr/bin/env python3
"""Small dependency-aware launcher for muon-energy training runs.

The queue is intentionally conservative: one job per GPU at a time, a GPU is
considered free only when its memory use is low, and a job starts only after its
declared predecessor has finished successfully.  It is meant to survive a
long-running shell session and to leave a machine-readable status file behind.

Usage:
    nohup python scripts/run_energy_queue.py \
        --queue src/training/configs/track_energy_reconstruction/energy_queue.yaml \
        > trained_models/track_energy_reconstruction/energy_queue.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = (
    PROJECT_ROOT
    / "trained_models/track_energy_reconstruction/energy_queue_status.json"
)
LOG_DIR = PROJECT_ROOT / "trained_models/track_energy_reconstruction/queue_logs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_usage(gpu: int) -> tuple[int, int] | None:
    """Return (used MiB, utilization %) for a physical GPU, or None."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        used, util = [int(x.strip()) for x in out.split(",")]
        return used, util
    except Exception:
        return None


def _save_status(status: Dict[str, Any], path: Path = STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def _launch(job: Dict[str, Any], log_dir: Path = LOG_DIR) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(job["id"])
    log_path = log_dir / f"{run_name}.log"
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    cmd = [
        sys.executable,
        "-m",
        "src.training.mu_energy_trainer",
        "--config",
        str(PROJECT_ROOT / job["config"]),
    ]
    log = open(log_path, "a", buffering=1)
    log.write(f"\n\n===== {_now()} launching {run_name} on GPU {job['gpu']} =====\n")
    log.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Keep the file handle alive with the process object so the queue owns it.
    proc._queue_log = log  # type: ignore[attr-defined]
    return proc


def _dependency_done(
    dep: str,
    job_states: Dict[str, Dict[str, Any]],
) -> bool:
    if dep in job_states:
        return job_states[dep]["status"] == "done"
    # A file dependency is useful for externally produced checkpoints.
    path = PROJECT_ROOT / dep
    return path.exists()


def _dependency_failed(dep: str, job_states: Dict[str, Dict[str, Any]]) -> bool:
    return dep in job_states and job_states[dep]["status"] in (
        "failed",
        "blocked",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--status", default=str(STATUS_PATH))
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    args = ap.parse_args()

    status_path = Path(args.status)
    if not status_path.is_absolute():
        status_path = PROJECT_ROOT / status_path
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = PROJECT_ROOT / queue_path
    queue = yaml.safe_load(queue_path.read_text())
    jobs: List[Dict[str, Any]] = list(queue["jobs"])
    poll_seconds = int(queue.get("poll_seconds", args.poll_seconds))
    free_memory_mib = int(queue.get("free_memory_mib", 1000))
    max_util = int(queue.get("max_utilization", 5))

    states: Dict[str, Dict[str, Any]] = {
        j["id"]: {
            "status": "pending",
            "pid": None,
            "gpu": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "error": None,
        }
        for j in jobs
    }
    running: Dict[str, subprocess.Popen] = {}

    log_dir.mkdir(parents=True, exist_ok=True)
    _save_status(
        {"updated_at": _now(), "jobs": states, "queue": str(queue_path)},
        status_path,
    )
    print(f"{_now()} queue started: {len(jobs)} jobs, poll={poll_seconds}s")

    while True:
        # Reap finished processes.
        for job_id, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            states[job_id]["status"] = "done" if rc == 0 else "failed"
            states[job_id]["finished_at"] = _now()
            states[job_id]["returncode"] = rc
            if rc != 0:
                states[job_id]["error"] = f"exit code {rc}"
            running.pop(job_id)
            log = getattr(proc, "_queue_log", None)
            if log is not None:
                log.close()
            print(f"{_now()} {job_id}: {'done' if rc == 0 else 'FAILED'} rc={rc}")

        # Mark blocked jobs whose dependencies failed.
        for job in jobs:
            job_id = job["id"]
            if states[job_id]["status"] != "pending":
                continue
            deps = list(job.get("depends_on", []) or [])
            if any(_dependency_failed(d, states) for d in deps):
                states[job_id]["status"] = "blocked"
                states[job_id]["error"] = "dependency failed"
                print(f"{_now()} {job_id}: blocked by failed dependency")

        # Start at most one pending job per GPU.
        busy_gpus = {int(s["gpu"]) for s in states.values()
                     if s["status"] == "running" and s["gpu"] is not None}
        for job in jobs:
            job_id = job["id"]
            if states[job_id]["status"] != "pending":
                continue
            gpu = int(job["gpu"])
            if gpu in busy_gpus:
                continue
            deps = list(job.get("depends_on", []) or [])
            if not all(_dependency_done(d, states) for d in deps):
                continue
            usage = _gpu_usage(gpu)
            if usage is None:
                continue
            used, util = usage
            if used > free_memory_mib or util > max_util:
                continue
            try:
                proc = _launch(job, log_dir)
            except Exception as exc:
                states[job_id]["status"] = "failed"
                states[job_id]["error"] = f"launch: {exc}"
                print(f"{_now()} {job_id}: LAUNCH FAILED {exc}")
                continue
            running[job_id] = proc
            busy_gpus.add(gpu)
            states[job_id].update({
                "status": "running",
                "pid": proc.pid,
                "gpu": gpu,
                "started_at": _now(),
                "error": None,
            })
            print(f"{_now()} {job_id}: started pid={proc.pid} gpu={gpu}")

        _save_status(
            {
                "updated_at": _now(),
                "queue": str(queue_path),
                "jobs": states,
            },
            status_path,
        )

        if not running and all(
            states[j["id"]]["status"] in ("done", "failed", "blocked")
            for j in jobs
        ):
            print(f"{_now()} queue finished")
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
