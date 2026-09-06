"""Building the selection query and streaming it out grouped by part.

One query per pass, never one per part.  Because `event_fk` is contiguous inside a
part, ordering by it makes DuckDB emit rows already grouped by part, so the reader
gets part structure without paying for it.  Measured on `mc_merged` (25.07M eligible
events, 22,404 parts):

    one query per part, filtered on part_key       159 ms x 22,404 = 59 min
    one query per part, filtered on event_fk        71 ms x 22,404 = 27 min
    one streaming query ordered by event_fk         7.3 s + 28.6M rows/s = 8 s

`fetch_record_batch` hands back an Arrow stream, so the 25M rows are never
materialised at once.
"""
from __future__ import annotations

from typing import Iterator

import duckdb
import pandas as pd

from . import predicates
from .registry import Checkpoint
from .spec import CATALOG, Source

#: Rows pulled from DuckDB at a time.  Only affects the Arrow stream, not memory of
#: the hits, which is governed separately by the byte budget.
BATCH_ROWS = 200_000

#: Columns always fetched.  `part_key` and `local_idx` are the address into HDF5;
#: `data_class` picks the HDF5 group.
BASE_COLUMNS = (
    "p.event_fk",
    "e.data_class",
    "e.cluster",
    "e.run",
    "l.part_key",
    "l.local_idx",
)


def connect(checkpoint: Checkpoint, source: str, threads: int = 8
            ) -> duckdb.DuckDBPyConnection:
    """Read-only connection to one prediction database, catalog attached as `cat`."""
    con = duckdb.connect(str(checkpoint.database(source)), read_only=True)
    con.execute(f"PRAGMA threads={threads}")
    # DuckDB draws its own progress bar on a terminal, which in a notebook shows up as
    # unexplained stripes next to whatever the caller prints.
    con.execute("PRAGMA disable_progress_bar")
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    return con


def prediction_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Columns of this checkpoint's `predictions` table.

    Read rather than assumed: the nu-classifier writes
    `(event_fk, score, n_sn_hits, n_sn_strings)` and the prefilter writes
    `(event_fk, score, n_hits)`.
    """
    return [row[0] for row in con.execute("DESCRIBE predictions").fetchall()]


def build_sql(con: duckdb.DuckDBPyConnection, source: Source,
              checkpoint: Checkpoint, where,
              columns=None) -> tuple[str, dict[str, pd.DataFrame]]:
    """The streaming query and any frames that must be registered before running it."""
    resolved = predicates.resolve_all(where, source, checkpoint)
    available = prediction_columns(con)
    wanted = [c for c in available if c != "event_fk"]
    if columns is not None:
        wanted = [c for c in wanted if c in set(columns)]
    select = ", ".join([*BASE_COLUMNS, *(f"p.{c}" for c in wanted)])
    joins = " ".join([
        "JOIN cat.events e ON e.id = p.event_fk",
        "JOIN cat.h5_locations l ON l.event_fk = p.event_fk",
        *resolved.joins,
    ])
    clause = " AND ".join(resolved.where) if resolved.where else "TRUE"
    sql = (f"SELECT {select} FROM predictions p {joins} "
           f"WHERE {clause} ORDER BY p.event_fk")
    return sql, resolved.tables


def iter_parts(source: Source, checkpoint: Checkpoint, where=None, columns=None,
               batch_rows: int = BATCH_ROWS
               ) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield `(data_class, part_key, rows)` for every part touched by the selection.

    Rows arrive ordered by `event_fk`, which groups them by part; this buffers until
    the part changes, so a part split across two Arrow batches still comes out whole.
    """
    con = connect(checkpoint, source.name)
    try:
        sql, tables = build_sql(con, source, checkpoint, where, columns)
        for alias, frame in tables.items():
            con.register(alias, frame)
        reader = con.execute(sql).fetch_record_batch(batch_rows)

        buffer: list[pd.DataFrame] = []
        current: tuple[str, str] | None = None
        for batch in reader:
            frame = batch.to_pandas()
            if frame.empty:
                continue
            keys = list(zip(frame["data_class"], frame["part_key"]))
            boundaries = [0]
            boundaries += [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
            boundaries.append(len(keys))
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                piece = frame.iloc[start:stop]
                key = keys[start]
                if current is not None and key != current:
                    yield (*current, _joined(buffer))
                    buffer = []
                current = key
                buffer.append(piece)
        if current is not None and buffer:
            yield (*current, _joined(buffer))
    finally:
        con.close()


def _joined(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    if len(pieces) == 1:
        return pieces[0].reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)
