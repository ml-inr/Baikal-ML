#!/usr/bin/env python
"""Mark every scored event as training-or-not and reference-or-test.

Writes a `splits` table into a run's prediction databases so that later steps join it
instead of re-deriving the same conditions. Nothing downstream may filter training events
by hand: the model scores the events it was fitted on optimistically, so an unmarked
training event silently biases whatever it is used for.

The table has one row per scored event:

    event_fk | data_class | part_key | local_idx | used_for_labels | was_da_target
             | excluded   | role

  * `used_for_labels` — the event's labels were used in training (MC);
  * `was_da_target`   — the event was fed to the domain discriminator as an unlabelled
                        target (recorded, not excluded: no labels were involved);
  * `excluded`        — the event sits in a part whose exclusion is backed by a measured
                        mechanism (see `splits_config.yaml`; only `verified` entries apply);
  * `role`            — `reference` or `test`, a property of the *part*, assigned by a hash
                        of the part name so that adding statistics later cannot move a part
                        from one side of the split to the other.

This is the live version of a script that had been archived with the pre-rename tree; the
logic is unchanged, the root is now found from this file's own location, and the
per-event membership tests are vectorised (the archived version looped in Python over
every scored event).

Usage:

    python .../build_splits.py                                  # the run named in the config
    python .../build_splits.py --model <preds-dir-name>         # another run
    python .../build_splits.py --model <name> --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CLAUDE.md").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_splits")

#: Particle-type codes of the source dataset, as written by the NPY builder.
PARTICLE_TYPE_NAMES = {0: "muatm_2020", 1: "nuatm_2020", 2: "nue2_2020"}

SPLITS_TABLE = """
CREATE TABLE IF NOT EXISTS splits (
    event_fk        BIGINT PRIMARY KEY,
    data_class      VARCHAR,
    part_key        VARCHAR,
    local_idx       BIGINT,
    used_for_labels BOOLEAN,
    was_da_target   BOOLEAN,
    excluded        BOOLEAN,
    role            VARCHAR
)
"""


def load_config(path: Path, model: str | None = None) -> dict:
    """The config, with the preds paths resolved for the chosen run."""
    config = yaml.safe_load(path.read_text())
    if model:
        config["model_name"] = model
    for source, template in config["predictions"].items():
        config["predictions"][source] = template.format(model_name=config["model_name"])
    return config


def part_is_reference(data_class: str, part_key: str,
                      reference_fraction: float, seed: int) -> bool:
    """Reference membership decided by hashing the part name.

    Deterministic and independent of how many parts a run happens to process, so adding
    statistics later cannot silently move a part from one side of the split to the other.
    """
    if reference_fraction <= 0:
        return False
    digest = hashlib.md5(f"{seed}:{data_class}:{part_key}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2 ** 64
    return position < reference_fraction


def verified_excluded_runs(config: dict) -> set[str]:
    """Runs whose exclusion is backed by a measured mechanism.

    Only `verified` entries are applied. A `refuted` one stays in the config so that nobody
    reinstates it from an old note, but it must not silently remove data.
    """
    exclusions = config.get("exclusions", {})
    return {entry["key"] for entry in exclusions.get("runs", [])
            if entry.get("status") == "verified"}


def load_labelled_training_events(dataset_dir: Path) -> pd.DataFrame:
    """MC events whose labels the model was trained on, as (class, part, local index)."""
    part_keys = np.load(dataset_dir / "h5_part_keys.npy", allow_pickle=True).astype(str)
    local_ids = np.load(dataset_dir / "h5_local_event_ids.npy").astype("int64")
    particle_types = np.load(dataset_dir / "particle_types.npy")
    return pd.DataFrame({
        "data_class": [PARTICLE_TYPE_NAMES[int(code)] for code in particle_types],
        "part_key": part_keys,
        "local_idx": local_ids,
    })


def load_da_target_events(dataset_dir: Path) -> pd.DataFrame:
    """Experimental events fed to the domain discriminator, without labels."""
    part_keys = np.load(dataset_dir / "exp_h5_part_keys.npy", allow_pickle=True).astype(str)
    local_ids = np.load(dataset_dir / "exp_h5_local_event_ids.npy").astype("int64")
    return pd.DataFrame({"part_key": part_keys, "local_idx": local_ids})


def read_scored_events(connection: duckdb.DuckDBPyConnection,
                       catalog_path: str) -> pd.DataFrame:
    """Every scored event with the identity the datasets also carry."""
    connection.execute(f"ATTACH '{catalog_path}' AS catalog (READ_ONLY)")
    events = connection.execute("""
        SELECT p.event_fk, e.data_class, l.part_key, l.local_idx
        FROM predictions p
        JOIN catalog.h5_locations l ON l.event_fk = p.event_fk
        JOIN catalog.events e       ON e.id       = p.event_fk
    """).df()
    connection.execute("DETACH catalog")
    return events


def _mark_membership(events: pd.DataFrame, keys: pd.DataFrame,
                     columns: list[str]) -> pd.Series:
    """True where (columns) of the event appear in `keys`."""
    marked = events[columns].merge(keys[columns].drop_duplicates().assign(_hit=True),
                                   on=columns, how="left")
    return marked["_hit"].fillna(False).to_numpy()


def build_splits(database_path: str, catalog_path: str, config: dict,
                 labelled_training: pd.DataFrame | None,
                 da_target: pd.DataFrame | None) -> pd.DataFrame:
    connection = duckdb.connect(database_path)
    connection.execute("DROP TABLE IF EXISTS splits")
    connection.execute(SPLITS_TABLE)

    events = read_scored_events(connection, catalog_path)
    logger.info(f"  {len(events):,} scored events")

    events["used_for_labels"] = (
        _mark_membership(events, labelled_training, ["data_class", "part_key", "local_idx"])
        if labelled_training is not None else False)
    events["was_da_target"] = (
        _mark_membership(events, da_target, ["part_key", "local_idx"])
        if da_target is not None else False)

    # Role is a property of the part, so reference and test never share one. Training
    # events are removed event by event instead: a part that gave some events to training
    # still holds plenty the model never saw, and discarding those costs real statistics
    # (82% of the events in nue2's training parts).
    excluded_runs = verified_excluded_runs(config)
    events["excluded"] = events["part_key"].isin(excluded_runs)
    if excluded_runs:
        logger.info(f"  {int(events.excluded.sum()):,} events in {len(excluded_runs)} "
                    f"excluded runs")

    seed = config["random_seed"]
    fractions = config["reference_part_fraction"]
    parts = events[["data_class", "part_key"]].drop_duplicates()
    role_of_part = {
        (data_class, part_key): (
            "reference" if part_is_reference(data_class, part_key,
                                             fractions.get(data_class, 0.0), seed)
            else "test")
        for data_class, part_key in parts.itertuples(index=False)
    }
    events["role"] = pd.MultiIndex.from_frame(
        events[["data_class", "part_key"]]).map(role_of_part)

    connection.register("new_splits", events[
        ["event_fk", "data_class", "part_key", "local_idx",
         "used_for_labels", "was_da_target", "excluded", "role"]])
    connection.execute("INSERT INTO splits SELECT * FROM new_splits")
    connection.unregister("new_splits")
    connection.close()
    return events


def summarise(events: pd.DataFrame) -> None:
    logger.info(f"  {'class':14s} {'events':>12s} {'labelled':>11s} {'da target':>11s} "
                f"{'excluded':>10s} {'reference':>11s} {'test':>12s}")
    for data_class, group in events.groupby("data_class"):
        logger.info(f"  {data_class:14s} {len(group):>12,} "
                    f"{int(group.used_for_labels.sum()):>11,} "
                    f"{int(group.was_da_target.sum()):>11,} "
                    f"{int(group.excluded.sum()):>10,} "
                    f"{int((group.role == 'reference').sum()):>11,} "
                    f"{int((group.role == 'test').sum()):>12,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(HERE / "splits_config.yaml"))
    parser.add_argument("--model", default=None,
                        help="directory under inference/nu_classifier/preds/; "
                             "overrides model_name in the config")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the config and check the databases, then stop")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config(Path(arguments.config), arguments.model)
    catalog_path = str(ROOT / config["catalog"])
    logger.info(f"model: {config['model_name']}")

    for label, key in (("mc", "mc"), ("exp", "exp")):
        path = ROOT / config["predictions"][key]
        if not path.exists():
            raise SystemExit(f"{path} does not exist - run the predictions first")
        logger.info(f"  {label}: {path}")
    if arguments.dry_run:
        return

    logger.info("Monte Carlo:")
    labelled_training = load_labelled_training_events(ROOT / config["mc_training_dataset"])
    logger.info(f"  {len(labelled_training):,} events were used with labels in training")
    mc_events = build_splits(str(ROOT / config["predictions"]["mc"]),
                             catalog_path, config, labelled_training, None)
    summarise(mc_events)

    logger.info("Experimental:")
    da_target = load_da_target_events(ROOT / config["exp_da_target_dataset"])
    logger.info(f"  {len(da_target):,} events were the domain-adaptation target")
    exp_events = build_splits(str(ROOT / config["predictions"]["exp"]),
                              catalog_path, config, None, da_target)
    summarise(exp_events)


if __name__ == "__main__":
    main()
