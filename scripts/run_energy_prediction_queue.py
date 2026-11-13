#!/usr/bin/env python3
"""Run muon-energy test-part predictions for completed checkpoints.

The training queue only launches trainers.  This launcher is deliberately
separate: predictions can start while later training jobs are still running,
and each checkpoint gets its own DuckDB under the usual ``preds/<run>@best_model``
directory.  One prediction process is allowed per GPU lane.

Usage:
    nohup python scripts/run_energy_prediction_queue.py \
        --queue src/training/configs/track_energy_reconstruction/energy_prediction_queue.yaml \
        --status trained_models/track_energy_reconstruction/prediction_queue_status.json \
        --log-dir trained_models/track_energy_reconstruction/prediction_queue_logs \
        > trained_models/track_energy_reconstruction/prediction_queue.log 2>&1 &
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save_status(path: Path, jobs: Dict[str, Dict[str, Any]], queue: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"updated_at": _now(), "queue": queue, "jobs": jobs},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _launch(job: Dict[str, Any], log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['id']}.log"
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    cmd = [
        sys.executable,
        "-m",
        "inference.track_energy_reconstruction.predict_full",
        "--checkpoint",
        str(PROJECT_ROOT / job["checkpoint"]),
        "--npy-dir",
        str(PROJECT_ROOT / job.get(
            "npy_dir",
            "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit",
        )),
        "--subset",
        str(job.get("subset", "test")),
        "--parts-file",
        str(PROJECT_ROOT / job.get(
            "parts_file",
            "data_manager/mu_energy_ds_builder/part_split/parts_test.yaml",
        )),
        "--output-dir",
        str(PROJECT_ROOT / job.get(
            "output_dir", "inference/track_energy_reconstruction/preds"
        )),
        "--batch-size",
        str(int(job.get("batch_size", 512))),
        "--device",
        "cuda:0",
    ]
    log = open(log_path, "a", buffering=1)
    log.write(
        f"\n\n===== {_now()} launching {job['id']} on GPU {job['gpu']} =====\n"
    )
    log.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._queue_log = log  # type: ignore[attr-defined]
    return proc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = PROJECT_ROOT / queue_path
    queue = yaml.safe_load(queue_path.read_text())
    jobs: List[Dict[str, Any]] = list(queue["jobs"])
    poll = int(queue.get("poll_seconds", args.poll_seconds))
    status_path = Path(args.status)
    if not status_path.is_absolute():
        status_path = PROJECT_ROOT / status_path
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir

    states: Dict[str, Dict[str, Any]] = {
        j["id"]: {
            "status": "pending",
            "pid": None,
            "gpu": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "error": None,
            "db": None,
        }
        for j in jobs
    }
    running: Dict[str, subprocess.Popen] = {}
    _save_status(status_path, states, str(queue_path))
    print(f"{_now()} prediction queue started: {len(jobs)} jobs, poll={poll}s")

    while True:
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

        busy = {
            int(s["gpu"])
            for s in states.values()
            if s["status"] == "running" and s["gpu"] is not None
        }
        for job in jobs:
            job_id = job["id"]
            if states[job_id]["status"] != "pending":
                continue
            gpu = int(job["gpu"])
            if gpu in busy:
                continue
            ready_file = job.get("ready_file")
            if ready_file and not (PROJECT_ROOT / ready_file).exists():
                # Training is still running; keep the job pending.
                continue
            checkpoint = PROJECT_ROOT / job["checkpoint"]
            if not checkpoint.exists():
                states[job_id]["status"] = "failed"
                states[job_id]["error"] = f"missing checkpoint: {checkpoint}"
                print(f"{_now()} {job_id}: missing checkpoint")
                continue
            try:
                proc = _launch(job, log_dir)
            except Exception as exc:
                states[job_id]["status"] = "failed"
                states[job_id]["error"] = f"launch: {exc}"
                print(f"{_now()} {job_id}: launch failed: {exc}")
                continue
            running[job_id] = proc
            busy.add(gpu)
            run_name = Path(job["checkpoint"]).parent.name
            states[job_id].update({
                "status": "running",
                "pid": proc.pid,
                "gpu": gpu,
                "started_at": _now(),
                "error": None,
                "db": str(
                    PROJECT_ROOT
                    / job.get(
                        "output_dir",
                        "inference/track_energy_reconstruction/preds",
                    )
                    / f"{run_name}@best_model"
                    / "mc_merged_h8s3_thr0p8.duckdb"
                ),
            })
            print(f"{_now()} {job_id}: started pid={proc.pid} gpu={gpu}")

        _save_status(status_path, states, str(queue_path))
        if not running and all(
            s["status"] in ("done", "failed") for s in states.values()
        ):
            print(f"{_now()} prediction queue finished")
            break
        time.sleep(poll)


if __name__ == "__main__":
    main()
