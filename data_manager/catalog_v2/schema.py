"""DuckDB schema for the event catalog (v2)."""

import duckdb
from pathlib import Path

_DDL = [
    "CREATE SEQUENCE IF NOT EXISTS events_id_seq START 1",
    """CREATE TABLE IF NOT EXISTS events (
        id           BIGINT  DEFAULT nextval('events_id_seq') PRIMARY KEY,
        source       VARCHAR NOT NULL,
        data_class   VARCHAR NOT NULL,
        season       INTEGER NOT NULL,
        cluster      INTEGER NOT NULL,
        run          VARCHAR NOT NULL,
        event_id     BIGINT  NOT NULL,
        feature_hash VARCHAR,
        UNIQUE(source, data_class, season, cluster, run, event_id)
    )""",
    """CREATE TABLE IF NOT EXISTS h5_locations (
        event_fk  BIGINT  NOT NULL REFERENCES events(id),
        h5_path   VARCHAR NOT NULL,
        part_key  VARCHAR NOT NULL,
        local_idx INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS root_locations (
        event_fk  BIGINT  NOT NULL REFERENCES events(id),
        root_path VARCHAR NOT NULL,
        local_idx INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS npy_locations (
        event_fk  BIGINT  NOT NULL REFERENCES events(id),
        npy_dir   VARCHAR NOT NULL,
        npy_tag   VARCHAR NOT NULL,
        local_idx INTEGER NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_events_key  ON events(source, data_class, season, cluster, run, event_id)",
    "CREATE INDEX IF NOT EXISTS idx_h5_event    ON h5_locations(event_fk)",
    "CREATE INDEX IF NOT EXISTS idx_root_event  ON root_locations(event_fk)",
    "CREATE INDEX IF NOT EXISTS idx_npy_event   ON npy_locations(event_fk)",
    "CREATE INDEX IF NOT EXISTS idx_npy_tag     ON npy_locations(npy_tag)",
]


def open_catalog(path: str | Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=read_only)


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for stmt in _DDL:
        conn.execute(stmt)
    conn.commit()


def create_hash_index(conn: duckdb.DuckDBPyConnection) -> None:
    """Build the feature_hash index. Call after bulk inserts are complete."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_hash ON events(feature_hash)")
    conn.commit()
