"""Event catalog retriever (DuckDB v2).

Usage:
    from data_manager.catalog_v2.retriever import EventCatalog

    cat = EventCatalog()
    ev  = cat.get(source='exp_reco', season=2020, cluster=1, run='1', event_id=137)

    hits  = ev.load_hits()    # (n_hits, 5) float32
    reco  = ev.load_reco()    # dict of reco scalars (exp_reco only)
    info  = ev.info           # dict of identity fields

    df = cat.query_df(source='exp_reco', season=2020, cluster=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import h5py
import numpy as np

from data_manager.catalog_v2.schema import open_catalog

CATALOG_V2_PATH = Path(__file__).resolve().parents[1] / "catalog_v2.duckdb"


def _h5_top_key(source: str, data_class: str) -> str:
    if source in ("exp", "exp_reco"):
        return source
    if source == "mc_merged":
        return data_class
    raise ValueError(f"Unknown source: {source!r}")


@dataclass
class Event:
    id          : int
    source      : str
    data_class  : str
    season      : int
    cluster     : int
    run         : str
    event_id    : int
    feature_hash: str | None

    _h5_path  : str = field(repr=False, default="")
    _part_key : str = field(repr=False, default="")
    _local_idx: int = field(repr=False, default=-1)

    @property
    def info(self) -> dict[str, Any]:
        return {
            "source":      self.source,
            "data_class":  self.data_class,
            "season":      self.season,
            "cluster":     self.cluster,
            "run":         self.run,
            "event_id":    self.event_id,
        }

    def load_hits(self) -> np.ndarray:
        """Return raw hit features as (n_hits, 5) float32: [amp, t, x, y, z]."""
        top = _h5_top_key(self.source, self.data_class)
        with h5py.File(self._h5_path, "r") as f:
            ev_starts = f[top]["raw"]["ev_starts"][self._part_key]["data"][:]
            s, e = int(ev_starts[self._local_idx]), int(ev_starts[self._local_idx + 1])
            return f[top]["raw"]["data"][self._part_key]["data"][s:e].astype(np.float32)

    def load_reco(self) -> dict[str, float]:
        if self.source != "exp_reco":
            raise ValueError(f"load_reco() only available for 'exp_reco', not {self.source!r}")
        from inference.shared_utils import EXP_RECO_COL_NAMES
        top = _h5_top_key(self.source, self.data_class)
        with h5py.File(self._h5_path, "r") as f:
            row = f[top]["reco_prty"][self._part_key]["data"][self._local_idx]
        return dict(zip(EXP_RECO_COL_NAMES, row.tolist()))

    def load_probs(self, probs_h5_path: str | Path) -> np.ndarray:
        with h5py.File(probs_h5_path, "r") as f:
            ev_starts = f[self.data_class]["ev_starts"][self._part_key]["data"][:]
            s, e = int(ev_starts[self._local_idx]), int(ev_starts[self._local_idx + 1])
            return f[self.data_class]["probs"][self._part_key]["data"][s:e].astype(np.float32)


class EventCatalog:
    def __init__(self, catalog_path: str | Path = CATALOG_V2_PATH) -> None:
        self._conn = open_catalog(catalog_path, read_only=True)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EventCatalog":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def get(
        self,
        source    : str,
        season    : int,
        cluster   : int,
        run       : str | int,
        event_id  : int,
        data_class: str | None = None,
    ) -> Event:
        if data_class is None:
            data_class = source
        run = str(run)

        row = self._conn.execute(
            """SELECT e.id, e.source, e.data_class, e.season, e.cluster, e.run,
                      e.event_id, e.feature_hash,
                      h.h5_path, h.part_key, h.local_idx
               FROM events e
               JOIN h5_locations h ON h.event_fk = e.id
               WHERE e.source=? AND e.data_class=?
                 AND e.season=? AND e.cluster=? AND e.run=? AND e.event_id=?
               LIMIT 1""",
            [source, data_class, season, cluster, run, event_id],
        ).fetchone()

        if row is None:
            raise KeyError(
                f"Event not found: source={source!r}, data_class={data_class!r}, "
                f"season={season}, cluster={cluster}, run={run}, event_id={event_id}"
            )
        return _row_to_event(row)

    def query(self, **filters) -> list[Event]:
        allowed = {"source", "data_class", "season", "cluster", "run", "event_id"}
        bad = set(filters) - allowed
        if bad:
            raise ValueError(f"Unknown filter keys: {bad}")

        clauses = [f"e.{k}=?" for k in filters]
        values  = [str(v) if k == "run" else v for k, v in filters.items()]

        sql = (
            "SELECT e.id, e.source, e.data_class, e.season, e.cluster, e.run, "
            "e.event_id, e.feature_hash, h.h5_path, h.part_key, h.local_idx "
            "FROM events e JOIN h5_locations h ON h.event_fk = e.id"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
        )
        rows = self._conn.execute(sql, values).fetchall()
        return [_row_to_event(r) for r in rows]

    def query_df(self, **filters):
        import pandas as pd
        events = self.query(**filters)
        return pd.DataFrame([e.info | {"id": e.id} for e in events])

    def summary(self) -> dict:
        rows = self._conn.execute(
            "SELECT source, data_class, COUNT(*) as n FROM events GROUP BY source, data_class"
        ).fetchall()
        return {(r[0], r[1]): r[2] for r in rows}


def _row_to_event(row: tuple) -> Event:
    return Event(
        id           = row[0],
        source       = row[1],
        data_class   = row[2],
        season       = row[3],
        cluster      = row[4],
        run          = row[5],
        event_id     = row[6],
        feature_hash = row[7],
        _h5_path     = row[8],
        _part_key    = row[9],
        _local_idx   = row[10],
    )
