"""Fit a sample manifest and store the tracks in a DuckDB catalogue.

Reads the manifest and nothing else: no checkpoint, no prediction database, no
score.  The fit's own hit selection decides what it uses -- charge, causal clique,
scanfit -- and what it returns are the variables a cut can be placed on.

Resumable by block.  A block is marked done only after its rows are committed,
and the rows themselves are keyed by `event_fk` with INSERT OR IGNORE, so a
restart skips what is finished and re-running a block can neither duplicate nor
corrupt anything.  Ten hours of work has to survive being interrupted.

Tasks are blocks, one part's worth of them at a time, and the pool is asked for
more as they finish: `fit_events` parallelises *across parts*, so a runner that
handed it one part would leave all but one worker idle.

Run:
    python -m inference.track_fit_alike_bars.run_fit --manifest sample/manifest.parquet
    python -m inference.track_fit_alike_bars.run_fit --manifest ... --limit 4  # smoke
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # before h5py, and shared

import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from dataclasses import fields  # noqa: E402
from pathlib import Path  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from inference.track_fit_alike_bars import masks  # noqa: E402
from inference.track_fit_alike_bars.config import FitConfig  # noqa: E402
from inference.track_fit_alike_bars.fit import TrackFit  # noqa: E402
from inference.track_fit_alike_bars.frame import fit_events  # noqa: E402

log = logging.getLogger("run_fit")
ROOT = Path(__file__).resolve().parents[2]

#: What a fit row carries: the scalars of `TrackFit` and nothing else.  Taken
#: from the dataclass by type, not by name, so a field added to the port is
#: stored (or left out) by what it is -- the per-hit arrays are not storable and
#: are skipped by the same rule.
SQL_TYPE = {"float": "DOUBLE", "int": "INTEGER", "bool": "BOOLEAN", "str": "VARCHAR"}
FIT_COLUMNS = tuple(f.name for f in fields(TrackFit) if f.type in SQL_TYPE)

FITS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fits (
    event_fk BIGINT, source VARCHAR, part_key VARCHAR, local_idx BIGINT,
    block INTEGER, run_id VARCHAR,
""" + "".join(f"    {name} {SQL_TYPE[types[1]]},\n"
              for name, types in ((f.name, (f.name, f.type))
                                  for f in fields(TrackFit)
                                  if f.type in SQL_TYPE)) \
    + """    fitted_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (event_fk)
);
CREATE TABLE IF NOT EXISTS blocks (
    source VARCHAR, data_class VARCHAR, part_key VARCHAR, block INTEGER,
    run_id VARCHAR,
    n_events INTEGER, n_ok INTEGER, seconds DOUBLE, error VARCHAR,
    done_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (source, data_class, part_key, block)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY, started_at TIMESTAMP, finished_at TIMESTAMP,
    manifest VARCHAR, manifest_hash VARCHAR, port_hash VARCHAR,
    fit_config VARCHAR, workers INTEGER
);
"""


#: The two drivers.  They decide *which* events get fitted, not how -- the port
#: does that -- so a change here does not invalidate rows already written, and the
#: hash is of the reconstruction alone.
DRIVERS = ("run_fit.py", "make_sample.py")


def port_hash() -> str:
    """Fingerprint of the port's code.

    A fit catalogue is worthless if nobody can say which reconstruction produced
    it: change the port and every stored row describes something else.  The hash
    covers the modules that fit and deliberately not the drivers, so re-running
    with a better runner does not look like a different reconstruction.
    """
    digest = hashlib.sha1()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name in DRIVERS:
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def fit_config_of(config) -> str:
    """The `FitConfig` fields, as a string -- the second half of the fingerprint."""
    values = getattr(config, "__dataclass_fields__", None)
    if values is None:
        return str(config)
    return ",".join(f"{name}={getattr(config, name)!r}" for name in values)


def _block_rows(manifest: str, payload: dict) -> pd.DataFrame:
    """The events of one block, read by the worker that is about to fit them.

    In the worker, not in the parent: reading a part at a time left the parent
    parsing a 44-million-row parquet while sixty-four workers sat idle, and the
    run ground to a stop.  Sixty-four workers reading their own blocks is the
    same number of queries spread over the pool instead of serialised in front
    of it.
    """
    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT local_idx, event_fk FROM read_parquet(?) "
            "WHERE source = ? AND data_class = ? AND part_key = ? AND block = ? "
            "ORDER BY local_idx",
            [manifest, payload["source"], payload["data_class"],
             payload["part_key"], payload["block"]]).df()
    finally:
        connection.close()


def _fit_block(payload: dict) -> pd.DataFrame:
    """One block, fitted in this process.  The unit of work and of resumption."""
    rows = _block_rows(payload["manifest"], payload)
    local_idx = rows.local_idx.to_numpy(np.int64)
    frame = pd.DataFrame({
        "source": payload["source"],
        "data_class": payload["data_class"],
        "part_key": payload["part_key"],
        "local_idx": local_idx,
        "event_fk": rows.event_fk.to_numpy(np.int64),
    }).set_index("event_fk")
    provider = None
    if payload["source"] == "exp_full":
        # BARS masks channels per run; without it the fit sees hits BARS never had.
        channel_mask = masks.dqm_masks("exp_full", payload["part_key"])
        provider = lambda rows: channel_mask                             # noqa: E731
    fits = fit_events(frame[["source", "data_class", "part_key", "local_idx"]],
                      n_workers=1, channel_mask=provider)
    fits = fits.reset_index().rename(columns={"index": "event_fk"})
    # `fit_events` returns the fit and nothing else: the addressing of the row has
    # to be put back, and the index it carries is the `event_fk`.
    fits["source"] = payload["source"]
    fits["part_key"] = payload["part_key"]
    fits["block"] = payload["block"]
    fits["local_idx"] = local_idx
    return fits


def block_list(manifest: Path) -> pd.DataFrame:
    """Every block the manifest holds, as keys only -- one query, eight thousand rows.

    The events themselves are not here: a payload of `local_idx` per block is
    gigabytes for a whole sample, and it is the workers that need them anyway.
    """
    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT source, data_class, part_key, block, count(*) AS n "
            "FROM read_parquet(?) GROUP BY 1, 2, 3, 4 ORDER BY part_key, block",
            [str(manifest)]).df()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None,
                        help="по умолчанию рядом с манифестом, fits.duckdb")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="остановиться после стольких блоков (проверка)")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    manifest = args.manifest.resolve()
    db = args.db or manifest.parent / "fits.duckdb"
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    started = time.time()

    connection = duckdb.connect(str(db))
    connection.execute(FITS_SCHEMA)
    done = {(s, c, p, b) for s, c, p, b in connection.execute(
        "SELECT source, data_class, part_key, block FROM blocks").fetchall()}
    manifest_hash = hashlib.sha1(manifest.read_bytes()).hexdigest()[:12]
    connection.execute(
        "INSERT OR REPLACE INTO runs (run_id, started_at, manifest, manifest_hash,"
        " port_hash, fit_config, workers) VALUES (?, now(), ?, ?, ?, ?, ?)",
        [run_id, manifest.name, manifest_hash, port_hash(),
         fit_config_of(FitConfig()), args.workers])
    log.info("прогон %s: манифест %s (%s), порт %s, воркеров %s, готовых блоков %s",
             run_id, manifest.name, manifest_hash, port_hash(), args.workers,
             f"{len(done):,}")

    remaining = connection.execute(
        "SELECT count(*) FROM (SELECT DISTINCT source, data_class, part_key, block "
        "FROM read_parquet(?))", [str(manifest)]).fetchone()[0] - len(done)
    if args.limit:
        remaining = min(remaining, args.limit)
    log.info("к фиту: %s блоков (готовых уже %s)", f"{remaining:,}", f"{len(done):,}")

    todo = block_list(manifest)
    todo = todo[~todo.apply(lambda r: (r.source, r.data_class, r.part_key,
                                       int(r.block)) in done, axis=1)]
    if args.limit:
        todo = todo.head(args.limit)
    queue = iter([{"manifest": str(manifest), "source": r.source,
                   "data_class": r.data_class, "part_key": r.part_key,
                   "block": int(r.block)}
                  for r in todo.itertuples(index=False)])

    events = ok = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}

        def fill() -> None:
            for payload in queue:
                pending[pool.submit(_fit_block, payload)] = payload
                if len(pending) >= args.workers:
                    return

        fill()
        bar = tqdm(total=remaining, unit=" блок", desc="фит")
        while pending:
            # `wait(FIRST_COMPLETED)`, not `as_completed(list(pending))`: the
            # latter takes a snapshot, so the next batch is only submitted once
            # the whole current one has finished -- and one slow block then idles
            # sixty-three workers while it grinds.  Which is exactly what the
            # neutrino parts did: five minutes of full speed, then twenty minutes
            # of one core.
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                payload = pending.pop(future)
                try:
                    fits = future.result()
                except Exception as exc:                    # noqa: BLE001
                    # A fourteen-hour run does not die of one unreadable part.
                    # The block is written down as failed, so a restart skips it
                    # instead of meeting the same wall again, and the count stays
                    # visible in the catalogue rather than in a lost terminal.
                    log.error("блок %s/%s #%s упал: %s", payload["data_class"],
                              payload["part_key"], payload["block"], exc,
                              exc_info=True)
                    connection.execute(
                        "INSERT OR REPLACE INTO blocks (source, data_class,"
                        " part_key, block, run_id, n_events, n_ok, seconds, error)"
                        " VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
                        [payload["source"], payload["data_class"],
                         payload["part_key"], payload["block"], run_id,
                         len(payload["local_idx"]), time.time() - started,
                         f"{type(exc).__name__}: {exc}"[:400]])
                    connection.commit()
                    failed += 1
                    bar.update(1)
                    fill()
                    continue
                fits["run_id"] = run_id
                connection.register("_block", fits)
                back = ("event_fk, source, part_key, local_idx, block, run_id, "
                        + ", ".join(FIT_COLUMNS))
                connection.execute(
                    f"INSERT OR IGNORE INTO fits ({back}) SELECT {back} FROM _block")
                connection.unregister("_block")
                n_ok = int((fits["status"] == "ok").sum())
                connection.execute(
                    "INSERT OR REPLACE INTO blocks (source, data_class, part_key,"
                    " block, run_id, n_events, n_ok, seconds) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [payload["source"], payload["data_class"], payload["part_key"],
                     payload["block"], run_id, len(fits), n_ok,
                     time.time() - started])
                connection.commit()
                events += len(fits)
                ok += n_ok
                bar.update(1)
                bar.set_postfix(соб=f"{events:,}", сошлось=f"{ok / max(events, 1):.0%}")
                fill()
        bar.close()
    connection.execute("UPDATE runs SET finished_at = now() WHERE run_id = ?",
                       [run_id])
    total = connection.execute("SELECT count(*) FROM fits").fetchone()[0]
    connection.close()
    log.info("готово: %s событий за %.1f ч, сошлось %.0f%%, упало блоков %s,"
             " всего в базе %s", f"{events:,}", (time.time() - started) / 3600,
             100 * ok / max(events, 1), failed, f"{total:,}")


if __name__ == "__main__":
    main()
