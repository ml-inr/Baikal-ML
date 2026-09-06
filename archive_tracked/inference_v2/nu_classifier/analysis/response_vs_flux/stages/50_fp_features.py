"""Stage 50 -- features for every event test 5 will see, with no sampling weights.

See PROTOCOL.md, test 5, for the design and the criteria.  This stage only builds
the table.

**Why the existing feature table cannot be reused.**
``excess_mechanism/features.duckdb`` is a stratified sample, not a census: events
above xi = 0.8 were taken nearly whole, everything below was thinned to 200,000
of 23.2 million muatm and 200,000 of 3.2 million experimental events.  Almost the
whole false-positive sample lives *below* 0.8 -- 26,630 of 34,382 muatm and 7,676
of 10,896 experimental events -- so that table covers 23% of the target and 32%
of the test.  It was also built with the fitter that fails the synthetic-track
oracle, and mixing events fitted by broken and corrected code would make a cut on
``fit_zenith`` behave differently on the two halves.

**Sampling, and why no weights appear anywhere.**  The two populations that must
not carry sampling error -- the cut's target and its blind test -- are taken
whole.  Everything else is uniform, and only *fractions* of those are ever
reported; a uniform sample estimates a fraction without bias.

Rows carry ``from_target`` and ``from_uniform`` so the two selections stay
separable afterwards.  Merging them would over-represent false positives in the
"muatm in general" denominator by a factor of about 700.

Usage:
    nohup python stages/50_fp_features.py > /tmp/stage50.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import features as feat                                        # noqa: E402
import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage50")
RDCC = {"rdcc_nbytes": 256 * 1024 * 1024, "rdcc_nslots": 100_003}
QUALITY = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3"
COLUMNS = "p.event_fk, p.score, s.part_key, s.local_idx"
_CFG: h5io.Config | None = None


def _uniform(con, where: str, target: int) -> tuple[pd.DataFrame, int]:
    """Uniform subsample by hash residue: deterministic, and one pass over 94M rows."""
    total = con.execute(f"SELECT count(*) FROM predictions p "
                        f"JOIN splits s USING (event_fk) WHERE {where}").fetchone()[0]
    modulus = max(total // target, 1)
    frame = con.execute(f"""
        SELECT {COLUMNS} FROM predictions p JOIN splits s USING (event_fk)
        WHERE {where} AND hash(p.event_fk) % {modulus} = 0""").df()
    return frame, int(total)


def select(cfg: h5io.Config) -> tuple[pd.DataFrame, dict]:
    """Everything the cut search needs, with the target and the test taken whole."""
    model = cfg.path("preds") / cfg["model"]
    settings = cfg["stage50"]
    score_min = float(settings["score_min"])
    frames, census = [], {}

    def add(frame: pd.DataFrame, data_class: str, source: str,
            from_target: bool, from_uniform: bool) -> None:
        frame = frame.assign(data_class=data_class, source=source,
                             from_target=from_target, from_uniform=from_uniform)
        frames.append(frame)

    with duckdb.connect(str(model / "mc_merged_thr0p8.duckdb"), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        target = con.execute(f"""
            SELECT {COLUMNS} FROM predictions p JOIN splits s USING (event_fk)
            WHERE s.data_class = 'muatm_2020' AND {QUALITY}
              AND p.score > {score_min}""").df()
        add(target, "muatm_2020", "mc", True, False)
        LOG.info("muatm above %.2f: %d (whole)", score_min, len(target))

        reference, total = _uniform(con, f"s.data_class = 'muatm_2020' AND {QUALITY}",
                                    int(settings["muatm_reference"]))
        add(reference, "muatm_2020", "mc", False, True)
        census["muatm_quality_total"] = total
        LOG.info("muatm reference: %d of %d (uniform)", len(reference), total)

        for data_class in ("nuatm_2020", "nue2_2020"):
            where = (f"s.data_class = '{data_class}' AND {QUALITY} "
                     f"AND p.score > {score_min}")
            frame, total = _uniform(con, where, int(settings["neutrino_sample"]))
            add(frame, data_class, "mc", False, True)
            census[f"{data_class}_above"] = total
            census[f"{data_class}_quality_total"] = int(con.execute(
                f"SELECT count(*) FROM predictions p JOIN splits s USING (event_fk)"
                f" WHERE s.data_class = '{data_class}' AND {QUALITY}").fetchone()[0])
            LOG.info("%s above %.2f: %d of %d (uniform)", data_class, score_min,
                     len(frame), total)

    with duckdb.connect(str(model / "exp_full_thr0p8.duckdb"), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        accepted = con.execute(f"""
            SELECT {COLUMNS} FROM predictions p JOIN splits s USING (event_fk)
            WHERE {QUALITY} AND p.score > {score_min}""").df()
        add(accepted, "exp_full", "exp", True, False)
        LOG.info("exp above %.2f: %d (whole)", score_min, len(accepted))
        reference, total = _uniform(con, QUALITY, int(settings["exp_reference"]))
        add(reference, "exp_full", "exp", False, True)
        census["exp_quality_total"] = total
        LOG.info("exp reference: %d of %d (uniform)", len(reference), total)

    selection = pd.concat(frames, ignore_index=True)
    # one row per event; an event drawn by both selections keeps both flags
    selection = (selection.groupby(["source", "data_class", "event_fk"], as_index=False)
                 .agg(score=("score", "first"), part_key=("part_key", "first"),
                      local_idx=("local_idx", "first"),
                      from_target=("from_target", "max"),
                      from_uniform=("from_uniform", "max")))
    return selection, census


def _init(config_dir: str) -> None:
    global _CFG
    _CFG = h5io.load_config(Path(config_dir))


def process_part(task: tuple) -> pd.DataFrame:
    """Features for every selected event of one part."""
    source, h5_group, part, frame = task
    cfg = _CFG
    threshold = float(cfg["sig_noise_threshold"])
    with h5py.File(cfg.path("h5", source), "r", **RDCC) as src, \
         h5py.File(cfg.path("probs", source), "r", **RDCC) as pf:
        starts = src[f"{h5_group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = src[f"{h5_group}/raw/data/{part}/data"][:].astype(np.float32)
        channels = src[f"{h5_group}/raw/channels/{part}/data"][:]
        probs = pf[f"{h5_group}/probs/{part}/data"][:].astype(np.float32)
    rows, keys = [], []
    for record in frame.itertuples():
        lo, hi = int(starts[record.local_idx]), int(starts[record.local_idx + 1])
        keep = probs[lo:hi] > threshold
        hits = data[lo:hi][keep].astype(np.float64)
        if len(hits) < 8:
            continue
        rows.append(feat.event_features(hits, probs[lo:hi][keep],
                                        channels[lo:hi][keep], n_raw=hi - lo))
        keys.append((record.event_fk, record.score, record.data_class, source,
                     part, bool(record.from_target), bool(record.from_uniform)))
    if not rows:
        return pd.DataFrame()
    meta = pd.DataFrame(keys, columns=["event_fk", "score", "data_class", "source",
                                       "part_key", "from_target", "from_uniform"])
    return pd.concat([meta, pd.DataFrame(rows, columns=feat.COLUMNS)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data, config_path = HERE / "data", HERE / "config.yaml"

    selection, census = select(cfg)
    groups = {"exp": "exp_full"}
    selection["h5_group"] = [groups.get(s, c) for s, c
                             in zip(selection.source, selection.data_class)]
    tasks = [(source, group, part, sub) for (source, group, part), sub
             in selection.groupby(["source", "h5_group", "part_key"], sort=True)]
    tasks.sort(key=lambda t: -len(t[3]))
    if args.smoke:
        tasks = tasks[:10]
    workers = int(cfg["stage50"]["workers"])
    LOG.info("%d events over %d parts, %d workers",
             len(selection), len(tasks), workers)

    frames, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(HERE),)) as pool:
        futures = [pool.submit(process_part, task) for task in tasks]
        for future in as_completed(futures):
            frames.append(future.result())
            done += 1
            if done % 500 == 0 or done == len(tasks):
                LOG.info("%d/%d parts, %d events, %.0f s", done, len(tasks),
                         sum(len(f) for f in frames), time.time() - started)
    table = pd.concat([f for f in frames if len(f)], ignore_index=True)
    LOG.info("computed %d events; target %d, uniform %d", len(table),
             int(table.from_target.sum()), int(table.from_uniform.sum()))
    provenance.write(table, data / "50_fp_features.parquet", stage="50_fp_features",
                     config_path=config_path,
                     inputs=[cfg.path("h5", "mc"), cfg.path("h5", "exp")],
                     started=started,
                     notes={"features": len(feat.COLUMNS), "census": census})
    LOG.info("stage 50 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
