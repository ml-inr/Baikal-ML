"""Build the sample the fit run consumes.

The sample is decided here and written down: which parts, which event windows,
which events are dropped for sitting in the checkpoint's training set or in its
domain-adaptation target, and with which seed.  `run_fit.py` reads the manifest
and nothing else -- no checkpoint, no prediction database, no score.  That split
is the point: the fit has nothing to do with a model, but the sample definition
does, and one file in between keeps the two apart.

Output, under `sample/` by default:

    manifest.parquet   one row per event to fit -- source, data_class, part_key,
                       block, local_idx, event_fk
    manifest.json      the seed, the checkpoint, the filters, the sizes: what
                       makes this a sample rather than a pile of rows

Events are taken as the h5 stores them, in contiguous windows -- so the scored
events come along at their natural rate, about a twentieth for `exp_full`, and
nothing is selected on score.  Windows are spread through their part rather than
one per part, because conditions drift within a run and one window would carry
whatever drift it happened to land on.

Run:
    python -m inference.track_fit_alike_bars.make_sample --config <yaml>
    python -m inference.track_fit_alike_bars.make_sample --config <yaml> \
        --only exp_full --parts 2 --blocks 1 --events 2000   # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from inference.reader import training
from inference.reader.registry import open_checkpoint

log = logging.getLogger("make_sample")

ROOT = Path(__file__).resolve().parents[2]

COLUMNS = ("source", "data_class", "part_key", "block", "local_idx", "event_fk")
SCHEMA = pa.schema([("source", pa.dictionary(pa.int32(), pa.string())),
                    ("data_class", pa.dictionary(pa.int32(), pa.string())),
                    ("part_key", pa.dictionary(pa.int32(), pa.string())),
                    ("block", pa.int32()),
                    ("local_idx", pa.int64()),
                    ("event_fk", pa.int64())])


def part_sizes(catalog: Path, source: str, data_class: str) -> pd.DataFrame:
    """Events per part, from the catalogue.  No predictions are involved."""
    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        return connection.execute(
            """
            SELECT l.part_key, count(*) AS n
            FROM events e JOIN h5_locations l ON l.event_fk = e.id
            WHERE e.source = ? AND e.data_class = ?
            GROUP BY l.part_key ORDER BY l.part_key
            """, [source, data_class]).df()
    finally:
        connection.close()


def part_locations(connection, source: str, data_class: str,
                   part_key: str) -> pd.DataFrame:
    """`(local_idx, event_fk)` of one part, on an open connection.

    `part_key` alone does not identify a part: the MC converters name their file
    chunks `part_1000` in every class, so the same key exists under `muatm_2020`,
    `nuatm_2020` and `nue2_2020`, and each holds its own events.  The class is
    what makes the key unique, and it is also what the reader needs to find the
    hits -- without it three events answer to one `local_idx`.

    One query per part, on a connection the caller keeps open: the catalogue is
    a single file, and reconnecting per part costs more than the query.
    """
    frame = connection.execute(
        """
        SELECT l.local_idx, l.event_fk
        FROM h5_locations l JOIN events e ON e.id = l.event_fk
        WHERE e.source = ? AND e.data_class = ? AND l.part_key = ?
        ORDER BY l.local_idx
        """, [source, data_class, part_key]).df()
    if not frame.local_idx.is_unique:
        raise RuntimeError(
            f"{part_key}: local_idx repeats within one class -- the class filter "
            f"is not doing its job, and a fit addressed by (part, local_idx) "
            f"would read the wrong hits")
    return frame


def exclusions(checkpoint_name: str, exclude: dict) -> pd.DataFrame:
    """Every `(part_key, local_idx)` the checkpoint has already seen.

    Two sources, both from the training config rather than from the predictions:
    the labelled MC the model was fitted on, and the unlabelled domain-adaptation
    target.  Strictness is `all` -- every event in the dataset, not just the
    training half -- because an evaluation that leaks the held-out half of its own
    training set is not one this project wants to be able to report.
    """
    checkpoint = open_checkpoint(checkpoint_name)
    frames = []
    for flag, domain, dataset in ((exclude.get("source_training"),
                                   "source", "source_domain"),
                                  (exclude.get("da_target"), "target",
                                   "target_domain")):
        if not flag:
            continue
        frame = training.identity(checkpoint.training_dataset(domain),
                                  domain=domain, strictness="all")
        frame = frame[["part_key", "local_idx"]].copy()
        frame["part_key"] = frame["part_key"].astype(str)
        frames.append(frame)
        log.info("исключаем %s: %s событий", dataset, f"{len(frame):,}")
    if not frames:
        return pd.DataFrame({"part_key": [], "local_idx": []})
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["part_key", "local_idx"]))


def windows(available: np.ndarray, blocks: int, size: int,
            rng: np.random.Generator) -> list[tuple[int, int]]:
    """`blocks` disjoint windows of `size` events, spread through the part.

    Disjoint and spread: overlapping windows would double-count events, and the
    windows all sitting in one stretch of the part would carry its drift.  The
    offsets are drawn from a grid, so two runs with the same seed and the same
    part land on the same windows.
    """
    if size <= 0 or len(available) < size:
        return []
    slots = len(available) // size
    take = min(blocks, slots)
    picks = rng.choice(slots, size=take, replace=False)
    out = []
    for slot in sorted(picks):
        chunk = available[slot * size:(slot + 1) * size]
        out.append((int(chunk[0]), int(chunk[-1]) + 1))
    return out


def build_domain(name: str, spec: dict, *, catalog: Path, excluded: pd.DataFrame,
                 rng: np.random.Generator, writer: pq.ParquetWriter,
                 overrides: dict) -> dict:
    """One domain's share of the sample, written straight into the parquet."""
    source, data_class = spec["source"], spec["data_class"]
    n_events = overrides.get("events") or spec["events"]
    n_parts = overrides.get("parts") or spec["parts"]
    blocks = overrides.get("blocks") or spec["blocks_per_part"]

    sizes = part_sizes(catalog, source, data_class)
    drop = (excluded[excluded.part_key.isin(set(sizes.part_key))]
            .groupby("part_key")["local_idx"].apply(np.asarray).to_dict())
    block_size = max(1, int(round(n_events / (n_parts * blocks))))

    usable = []
    for part, n in zip(sizes.part_key, sizes.n):
        if len(available_index(int(n), drop.get(part))) >= block_size:
            usable.append(part)
    if not usable:
        raise RuntimeError(f"{name}: ни одного парта с {block_size} доступных событий")
    chosen = (rng.choice(usable, size=min(n_parts, len(usable)), replace=False)
              if len(usable) > n_parts else np.asarray(usable))

    log.info("%s: партов %s из %s, блок %s событий по %s блоков на парт",
             name, len(chosen), len(usable), f"{block_size:,}", blocks)
    written = blocks_made = 0
    t0 = time.time()
    # The bar is not decoration: a domain is thousands of parts, each a query and
    # a parquet write, and without it ten minutes of work looks like a hang.
    connection = duckdb.connect(str(catalog), read_only=True)
    progress = tqdm(sorted(chosen), unit=" парт", desc=f"манифест {name}")
    for part in progress:
        n_events_part = int(sizes.loc[sizes.part_key == part, "n"].iloc[0])
        available = available_index(n_events_part, drop.get(part))
        picked = windows(available, blocks, block_size, rng)
        if not picked:
            continue
        places = part_locations(connection, source, data_class, part)
        # position of an event in the part's own table, by its local_idx
        index = pd.Series(np.arange(len(places), dtype=np.int64),
                          index=places.local_idx.to_numpy())
        for number, (lo, hi) in enumerate(picked):
            inside = available[(available >= lo) & (available < hi)]
            frame = pd.DataFrame({
                "source": np.repeat(source, len(inside)),
                "data_class": np.repeat(data_class, len(inside)),
                "part_key": np.repeat(part, len(inside)),
                "block": np.full(len(inside), number, dtype=np.int32),
                "local_idx": inside.astype(np.int64),
                "event_fk": places.event_fk.to_numpy()[
                    index.reindex(inside).to_numpy()],
            })
            writer.write_table(pa.Table.from_pandas(frame, schema=SCHEMA,
                                                    preserve_index=False))
            written += len(frame)
            blocks_made += 1
        progress.set_postfix(соб=f"{written:,}", блоков=blocks_made)
    progress.close()
    connection.close()
    log.info("%s: %s событий в %s блоках за %.0f с", name, f"{written:,}",
             blocks_made, time.time() - t0)
    return {"domain": name, "source": source, "data_class": data_class,
            "events": written, "blocks": blocks_made, "block_size": block_size,
            "parts_chosen": len(chosen), "parts_available": len(usable)}


def available_index(n_events: int, dropped) -> np.ndarray:
    """`local_idx` of a part that are not excluded, in order."""
    everything = np.arange(n_events, dtype=np.int64)
    if dropped is None or not len(dropped):
        return everything
    return np.setdiff1d(everything, np.asarray(dropped, dtype=np.int64),
                        assume_unique=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", default=None,
                        help="один домен из конфига, для проверки")
    parser.add_argument("--parts", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--events", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    config = yaml.safe_load(args.config.read_text())
    catalog = ROOT / config["catalog"]
    out_dir = args.out or ROOT / config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config["seed"])
    excluded = exclusions(config["checkpoint"], config.get("exclude", {}))
    overrides = {"parts": args.parts, "blocks": args.blocks, "events": args.events}

    domains = config["domains"]
    if args.only:
        domains = {args.only: domains[args.only]}
    path = out_dir / "manifest.parquet"
    done = []
    with pq.ParquetWriter(path, SCHEMA, compression="zstd") as writer:
        for name, spec in domains.items():
            done.append(build_domain(name, spec, catalog=catalog, excluded=excluded,
                                     rng=rng, writer=writer, overrides=overrides))
    meta = {"seed": config["seed"], "checkpoint": config["checkpoint"],
            "exclude": config.get("exclude", {}), "catalog": config["catalog"],
            "manifest": path.name, "domains": done,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2,
                                                      ensure_ascii=False))
    log.info("манифест: %s (%s событий)", path,
             f"{sum(d['events'] for d in done):,}")


if __name__ == "__main__":
    main()
