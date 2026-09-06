"""Stage 02 -- one row per reconstructed event: score, reconstruction, quality flags.

Everything the analysis needs, joined once:

* the classifier score and the sig-noise hit counts, from the prediction DB;
* the 25 ``BRecoMuon`` reconstruction scalars, from ``reco_prty``.  The column
  names come from ``inference/shared_utils.py``, which is the authority; the
  mapping was additionally confirmed against value ranges (integer-valued
  ``nHits``/``nStrings``/``nOMs``, ``scfPhi`` spanning exactly 0..2π);
* ``cluster`` and ``run``, because the prefilter work excludes cluster 1 of
  ``exp_reco`` and that claim has to be checkable here;
* ``max_q``, the largest charge in the event.  The prefilter notebooks drop
  events above 10^4 p.e. ("Bad Qmax Found!"), and this is the only quantity
  among them that cannot be read from a derived file -- it needs the raw hits;
* ``n_gt_sig_hits`` for MC, which is how the prefilter work identifies the
  multi-cluster artefacts: an event split across clusters leaves a fragment in
  each, and the fragment carries the reconstruction of the *whole* event while
  its hits are only a piece of it.

Units, which are not uniform and have to be remembered: ``thetaRec``, ``phiRec``,
``thetaErr``, ``phiErr`` are in **degrees**, while ``scfMaxTheta``, ``scfMinTheta``,
``scfTheta``, ``scfPhi`` are in **radians**.

Usage:
    nohup python stages/02_event_table.py > /tmp/stage02.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
ROOT = (HERE / "../../../..").resolve()
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT))

import provenance                                              # noqa: E402
from inference.shared_utils import EXP_RECO_COL_NAMES          # noqa: E402

LOG = logging.getLogger("stage02")
RDCC = {"rdcc_nbytes": 128 * 1024 * 1024, "rdcc_nslots": 100_003}
PART_RE = re.compile(r"part_s(\d+)_c(\d+)_r(\d+)$")
_CFG: dict | None = None


def part_summary(task: tuple) -> pd.DataFrame:
    """Per-event reconstruction, identity and max charge for one part."""
    source, group, part, h5_path, probs_path, threshold = task
    with h5py.File(h5_path, "r", **RDCC) as src:
        node = src[group]
        starts = node["raw"]["ev_starts"][part]["data"][:].astype(np.int64)
        charge = node["raw"]["data"][part]["data"][:, 0]
        reco = node["reco_prty"][part]["data"][:]
        # BRecoMuon writes evCenterZ in absolute detector coordinates while the
        # hits are stored cluster-centred.  The two reco files disagree about
        # where that absolute origin is -- 361.7 m for the experimental cluster,
        # 0.4 m for the simulated one -- so a threshold on the raw value means
        # different things in the two samples.  Carrying the centre lets every
        # later step compare them in one frame.
        centre_z = float(np.atleast_2d(
            node["clusters_centers"][part]["data"][:])[0, 2])
        header = (node["header_prty"][part]["data"][:]
                  if "header_prty" in node else None)
    n_events = len(starts) - 1
    if n_events <= 0:
        return pd.DataFrame()
    max_q = np.maximum.reduceat(charge, starts[:-1]) if len(charge) else np.zeros(n_events)
    frame = pd.DataFrame({name: reco[:, i].astype(np.float32)
                          for i, name in enumerate(EXP_RECO_COL_NAMES)})
    frame["max_q"] = max_q.astype(np.float32)
    frame["cluster_center_z"] = np.float32(centre_z)
    frame["evCenterZ_rel"] = (frame.evCenterZ - centre_z).astype(np.float32)
    frame["local_idx"] = np.arange(n_events, dtype=np.int64)
    frame["part_key"] = part
    frame["source"] = source
    frame["group"] = group
    if header is not None:                       # experimental: real identity
        frame["season"] = header[:, 0].astype(np.int32)
        frame["cluster"] = header[:, 1].astype(np.int32)
        frame["run"] = header[:, 2].astype(np.int32)
        frame["event_id"] = header[:, 3].astype(np.int64)
    else:                                        # MC: cluster from the part name
        match = re.search(r"cl(\d+)", part)
        frame["season"] = 2020
        frame["cluster"] = int(match.group(1)) if match else -1
        frame["run"] = int(re.search(r"run(\d+)", part).group(1))
        frame["event_id"] = np.arange(n_events, dtype=np.int64)
    if probs_path is not None:                   # MC: ground-truth signal hits
        with h5py.File(probs_path, "r", **RDCC) as pf:
            key = f"{group}/n_gt_sig_hits/{part}/data"
            frame["n_gt_sig_hits"] = (pf[key][:].astype(np.int32) if key in pf
                                      else np.full(n_events, -1, dtype=np.int32))
    return frame


def _init(cfg: dict) -> None:
    global _CFG
    _CFG = cfg


def build(cfg: dict, source: str, groups: list[str], h5_path: Path,
          probs_path: Path | None, workers: int) -> pd.DataFrame:
    threshold = float(cfg["sig_noise_threshold"])
    tasks = []
    with h5py.File(h5_path, "r") as src:
        for group in groups:
            for part in sorted(src[group]["raw"]["ev_starts"].keys()):
                tasks.append((source, group, part, str(h5_path),
                              str(probs_path) if probs_path else None, threshold))
    LOG.info("%s: %d parts", source, len(tasks))
    frames, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(part_summary, t) for t in tasks]
        for future in as_completed(futures):
            frames.append(future.result())
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                LOG.info("  %d/%d parts, %d events", done, len(tasks),
                         sum(len(f) for f in frames))
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def attach_scores(table: pd.DataFrame, source: str, db: Path,
                  catalog: Path) -> pd.DataFrame:
    """Join the classifier score through the catalog identity of each event."""
    con = duckdb.connect(str(db), read_only=True)
    con.execute(f"ATTACH '{catalog}' AS cat (READ_ONLY)")
    if source == "exp_reco":
        scores = con.execute("""
            SELECT e.cluster, CAST(e.run AS INTEGER) AS run, e.event_id,
                   p.score, p.n_sn_hits, p.n_sn_strings
            FROM predictions p JOIN cat.events e ON p.event_fk = e.id
            WHERE e.source = 'exp_reco'""").df()
        merged = table.merge(scores, on=["cluster", "run", "event_id"], how="left")
        if len(merged) != len(table):
            raise SystemExit(f"join changed the row count: {len(table)} -> {len(merged)}")
    else:
        scores = con.execute("""
            SELECT e.data_class, e.run AS part_key, e.event_id,
                   p.score, p.n_sn_hits, p.n_sn_strings
            FROM predictions p JOIN cat.events e ON p.event_fk = e.id
            WHERE e.source = 'mc_reco'""").df()
        # part names repeat across MC classes -- part_2020_cl1_run10000_... exists
        # in muatm and in nuatm_conv alike -- so joining on the part alone
        # multiplies rows and silently mixes classes.  The data class must be in
        # the key.  This is the same trap that broke an earlier feature build.
        scores = scores.rename(columns={"event_id": "local_idx"})
        scores["group"] = scores.data_class.str.replace("_2020", "", regex=False)
        merged = table.merge(scores.drop(columns="data_class"),
                             on=["group", "part_key", "local_idx"], how="left")
        if len(merged) != len(table):
            raise SystemExit(f"join changed the row count: {len(table)} -> {len(merged)}")
    con.close()
    LOG.info("%s: %d of %d events have a score (%.1f%%)", source,
             int(merged.score.notna().sum()), len(merged),
             100 * merged.score.notna().mean())
    return merged


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    root = (HERE / cfg["paths"]["root"]).resolve()
    preds = root / cfg["paths"]["preds"] / cfg["model"]
    catalog = root / cfg["paths"]["catalog"]
    workers = int(cfg["stage02"]["workers"])

    for source, groups, h5_key, probs_key, db in (
            ("exp_reco", [cfg["exp_reco_group"]], "exp_reco", None,
             "exp_reco_thr0p8.duckdb"),
            ("mc_reco", cfg["mc_reco_groups"], "mc_reco", "mc_reco",
             "mc_reco_thr0p8.duckdb")):
        table = build(cfg, source, groups, root / cfg["paths"]["h5"][h5_key],
                      root / cfg["paths"]["probs"][probs_key] if probs_key else None,
                      workers)
        table = attach_scores(table, source, preds / db, catalog)
        provenance.write(table, HERE / "data" / f"02_{source}_events.parquet",
                         stage="02_event_table", config_path=HERE / "config.yaml",
                         inputs=[root / cfg["paths"]["h5"][h5_key]], started=started,
                         notes={"events": int(len(table)),
                                "with_score": int(table.score.notna().sum())})
        LOG.info("%s written: %d events", source, len(table))
    LOG.info("stage 02 done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
