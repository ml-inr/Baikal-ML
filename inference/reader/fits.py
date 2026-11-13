"""The fit catalogue of `track_fit_alike_bars`, read by `event_fk`.

`track_fit_alike_bars` fits the BARS-like reconstruction over a sample taken
straight from the h5 files -- no score, no cut, no checkpoint -- and writes one row
per event.  With the prediction databases that makes a third joinable thing beside
the reader: for one `event_fk`, a score from the classifier and a track from the
reconstruction.

Two properties of the catalogue decide how it is used, and both bite silently:

* **the sample is natural.**  It is contiguous windows of h5 parts, so scored
  events come along at their own rate -- about a twentieth for `exp_full`.  An
  event with no score is not a missing row: it is an event that never passed the
  sig-noise cut the scoring run applies before writing anything (`--min-hits 5`
  for `exp_full`, with `--min-strings 0`, so presence in the prediction database
  is exactly the five-hit cut).  Its `n_sn_hits` is below the cut, not unknown.
* **`status` is the only column that always holds a value.**  Every fit variable
  is NULL unless the fit converged, and `status` says why it did not
  (`no_clique`, `no_offline_trigger`, `no_local_trigger`, `no_global_trigger`).
  Filtering on a variable drops the failures implicitly; usually right, always
  worth saying out loud.

    from inference import fits
    catalogue = fits.open()
    catalogue.columns
    tracks = catalogue.frame(where="status = 'ok'")
    annotated = catalogue.attach(events)     # fit_* columns onto a reader frame

Which events the sample holds -- parts, windows, and what was dropped for sitting
in the checkpoint's training set -- is written down beside the catalogue, in
`manifest.json` and `manifest.parquet`.  Nothing here re-derives it; the domain of
a row comes from that manifest, because `part_key` alone does not name one (the MC
converters reuse `part_1000` in every class, and the same `local_idx` then means
three different events).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

#: Catalogue of the run described in `track_fit_alike_bars/sample_config.yaml`.
DEFAULT_PATH = ROOT / "inference/track_fit_alike_bars/sample/fits.duckdb"

#: Where the sample definition lives: `(event_fk, data_class)` for every row, plus
#: the seed and the filters that produced it.
DEFAULT_MANIFEST = DEFAULT_PATH.parent / "manifest.parquet"


@dataclass(frozen=True)
class Fits:
    """One fit catalogue, read-only.  One row per event, keyed by `event_fk`."""

    path: Path = DEFAULT_PATH
    manifest: Path = DEFAULT_MANIFEST

    def __post_init__(self) -> None:
        if not Path(self.path).exists():
            raise FileNotFoundError(
                f"нет каталога фитов {self.path} -- его собирает "
                f"`python -m inference.track_fit_alike_bars.run_fit`")

    def __repr__(self) -> str:
        return f"<Fits {Path(self.path).name}>"

    def connection(self) -> duckdb.DuckDBPyConnection:
        """A read-only connection with the catalogue attached as `fitness`.

        One connection per call, so nothing has to be closed by the caller and two
        cells of a notebook cannot fight over a lock.
        """
        connection = duckdb.connect()
        # DuckDB turns its own progress bar on when it is connected from a notebook;
        # it draws a widget into every cell that queries, which is nobody's output.
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(f"ATTACH '{self.path}' AS fitness (READ_ONLY)")
        return connection

    def manifest_sql(self) -> str:
        """The sample definition as a SQL table expression, for joins.

        Carries `event_fk`, `data_class` and the rest of the sample's columns.  The
        catalogue itself has no domain: `part_key` cannot supply it, because the MC
        converters reuse `part_1000` in every class.
        """
        return f"read_parquet('{self.manifest}')"

    def columns(self) -> tuple[str, ...]:
        """Columns of the catalogue, in table order."""
        with self.connection() as connection:
            return tuple(row[0] for row in
                         connection.execute("DESCRIBE fitness.fits").fetchall())

    def frame(self, *, where: str | None = None, event_fks=None,
              columns: str | list[str] | None = None,
              limit: int | None = None) -> pd.DataFrame:
        """Rows by `event_fk`.  `where` is SQL over the catalogue's own columns.

        `columns='*'` keeps everything; naming a few is what a caller wants when
        the selection is large, because a fit row holds forty variables.
        """
        if columns is None or columns == "*":
            select = "*"
        else:
            names = (columns if isinstance(columns, list)
                     else [c.strip() for c in columns.split(",")])
            # `event_fk` comes back either way: it is the key of every row and the
            # index of the frame, and a caller asking for three variables does not
            # mean to drop it.
            select = ", ".join(["event_fk"] + [n for n in names if n != "event_fk"])
        sql = f"SELECT {select} FROM fitness.fits"
        conditions = [f"({where})"] if where else []
        if event_fks is not None:
            ids = ", ".join(str(int(v)) for v in event_fks)
            conditions.append(f"event_fk IN ({ids})")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.connection() as connection:
            return connection.execute(sql).df().set_index("event_fk", drop=False)

    def attach(self, events: pd.DataFrame, *, prefix: str = "fit_",
               where: str | None = None) -> pd.DataFrame:
        """`events` with the fit columns joined on, keeping its rows and order.

        A left join: an event outside the sample gets NULLs, an event in the
        sample whose fit failed gets `fit_status` naming the failure.  The two are
        different, and the status column is what tells them apart.
        """
        if "event_fk" not in events.columns:
            raise KeyError("attach: у кадра нет колонки event_fk, соединять не по чему")
        where_sql = f" AND ({where})" if where else ""
        with self.connection() as connection:
            connection.register("_events", events[["event_fk"]])
            tracks = connection.execute(
                f"SELECT f.* FROM fitness.fits f "
                f"JOIN _events e ON e.event_fk = f.event_fk{where_sql}").df()
        tracks = tracks.set_index("event_fk").add_prefix(prefix)
        return events.join(tracks, on="event_fk")

    def summary(self) -> pd.DataFrame:
        """Fit statuses per source -- what the sample holds and what converged."""
        with self.connection() as connection:
            return connection.execute(
                """
                SELECT source, status, count(*) AS events
                FROM fitness.fits GROUP BY 1, 2 ORDER BY source, events DESC
                """).df()


def open_fits(path: Path | str | None = None,
              manifest: Path | str | None = None) -> Fits:
    """The catalogue at `path`, or the one this repository's run wrote.

    Not `open`: that name is the reader's own factory, and two `open`s in one
    namespace is how a caller ends up reading events when they meant tracks.
    """
    if path is None:
        return Fits()
    return Fits(path=Path(path),
                manifest=Path(manifest) if manifest else DEFAULT_MANIFEST)
