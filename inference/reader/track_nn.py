"""Angle / track-line / uncertainty predictions, joined by ``event_fk``.

``inference/angle_track_uncertainty`` writes one row per event with a direction,
a point on the predicted track line and two raw uncertainty scales.  That makes
it a fourth joinable thing beside the reader, the classifier prediction DBs and
the BARS fit catalogue: for one ``event_fk``, a classifier score, a BARS track
and a track-NN track.

Two properties decide how it is used:

* **presence is a selection.**  A row exists only for an event the track model
  actually processed: by default ``prob >= 0.70`` on the canonical sig-noise
  probabilities, at least eight hits on at least two strings, and no gate.  An
  event with no row is not "unknown"; it failed that selection.  Left joins are
  therefore the honest default, and ``coverage()`` reports how many rows were
  matched.
* **the calibrated radii are usually NULL.**  ``sigma_angle_deg`` and
  ``sigma_anchor_m`` are raw model scales.  ``angle_r68_deg`` / ``angle_r95_deg``
  and their line counterparts are populated only for the unchanged release-SN
  ``p >= 0.70`` MC selection on CPU; for experimental data, canonical-SN rows,
  gate rows and changed thresholds they are SQL NULL.  NULL does not mean the
  event has no uncertainty; it means no calibration is claimed.

Example::

    from inference import reader
    track = reader.open_track_nn(
        "inference/angle_track_uncertainty/preds/.../exp_full_canonical_p070_track.duckdb")
    track.columns()
    annotated = track.attach(events)             # track_* columns, left join
    track.coverage(events)                       # matched / total, by class

The table also carries its own ``n_hits`` / ``n_strings`` at the track model's
selection threshold and ``selected_threshold``; these are not the classifier's
``n_sn_hits`` / ``n_sn_strings`` and should not be compared as if they were.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

#: The prediction table written by ``inference.angle_track_uncertainty.store``.
TABLE = "predictions"


@dataclass(frozen=True)
class TrackPredictions:
    """One track-NN prediction database, read-only, keyed by ``event_fk``."""

    path: Path

    def __post_init__(self) -> None:
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"нет track-NN predictions DB: {self.path}")

    def __repr__(self) -> str:
        return f"<TrackPredictions {Path(self.path).name}>"

    def connection(self) -> duckdb.DuckDBPyConnection:
        """A fresh read-only connection with the DB attached as ``track``."""
        connection = duckdb.connect()
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(f"ATTACH '{self.path}' AS track (READ_ONLY)")
        return connection

    def columns(self) -> tuple[str, ...]:
        """Columns of the prediction table, in table order."""
        with self.connection() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    f"DESCRIBE track.{TABLE}"
                ).fetchall()
            )

    def frame(
        self,
        *,
        where: str | None = None,
        event_fks=None,
        columns: str | list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Rows selected by SQL ``where`` and/or an explicit ``event_fk`` list.

        The frame is indexed by ``event_fk`` and keeps that column.  ``where`` is
        SQL over the prediction table's own columns, not over the caller's event
        frame.
        """
        if columns is None or columns == "*":
            select = "*"
        else:
            names = (
                columns
                if isinstance(columns, list)
                else [name.strip() for name in columns.split(",")]
            )
            select = ", ".join(
                ["event_fk"] + [name for name in names if name != "event_fk"]
            )
        sql = f"SELECT {select} FROM track.{TABLE}"
        conditions = [f"({where})"] if where else []
        if event_fks is not None:
            ids = ", ".join(str(int(value)) for value in event_fks)
            conditions.append(f"event_fk IN ({ids})")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.connection() as connection:
            return (
                connection.execute(sql)
                .df()
                .set_index("event_fk", drop=False)
            )

    def attach(
        self,
        events: pd.DataFrame,
        *,
        prefix: str = "track_",
        where: str | None = None,
        columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """``events`` with the track columns joined on, keeping rows and order.

        A left join: an event outside the track selection gets NULLs.  Columns
        are prefixed, so ``n_hits`` here becomes ``track_n_hits`` and cannot
        overwrite the classifier's ``n_sn_hits`` by accident.  ``where`` filters
        the track table before the join; filtered-out events also become NULLs.
        ``columns`` limits the attached columns; ``event_fk`` is always kept as
        the join key and is not prefixed.
        """
        if "event_fk" not in events.columns:
            raise KeyError(
                "attach: у кадра нет колонки event_fk, соединять не по чему"
            )
        if not prefix:
            raise ValueError("attach: prefix не может быть пустым")
        collisions = [name for name in events.columns if name.startswith(prefix)]
        if collisions:
            raise ValueError(
                f"attach: в кадре уже есть колонки с префиксом {prefix!r}: "
                f"{collisions[:5]}"
            )
        if columns is None or columns == "*":
            select = "t.*"
        else:
            names = (
                columns
                if isinstance(columns, list)
                else [name.strip() for name in columns.split(",")]
            )
            select = ", ".join(
                ["t.event_fk"]
                + [f"t.{name}" for name in names if name != "event_fk"]
            )
        where_sql = f" AND ({where})" if where else ""
        with self.connection() as connection:
            connection.register("_events", events[["event_fk"]])
            tracks = connection.execute(
                f"SELECT {select} FROM track.{TABLE} t "
                f"JOIN _events e ON e.event_fk = t.event_fk{where_sql}"
            ).df()
        tracks = tracks.set_index("event_fk").add_prefix(prefix)
        return events.join(tracks, on="event_fk")

    def coverage(self, events: pd.DataFrame) -> pd.DataFrame:
        """Matched/total event counts, split by ``data_class`` when available.

        ``n_missing`` counts events in ``events`` for which the track DB has no
        row -- i.e. events the track selection did not keep.  This is a coverage
        report, not an error: a NULL left join is the intended representation.
        """
        if "event_fk" not in events.columns:
            raise KeyError("coverage: у кадра нет колонки event_fk")
        keys = ["event_fk"]
        if "data_class" in events.columns:
            keys.append("data_class")
        with self.connection() as connection:
            connection.register("_events", events[keys])
            group = ", ".join(f"e.{key}" for key in keys if key != "event_fk")
            group_sql = f", {group}" if group else ""
            group_by = f"GROUP BY {group}" if group else ""
            return connection.execute(
                f"""
                SELECT {group or "NULL AS data_class"},
                       COUNT(*) AS n_events,
                       COUNT(t.event_fk) AS n_matched,
                       COUNT(*) - COUNT(t.event_fk) AS n_missing
                FROM _events e
                LEFT JOIN track.{TABLE} t ON t.event_fk = e.event_fk
                {group_by}
                ORDER BY 1
                """
            ).df()

    def summary(self) -> pd.DataFrame:
        """Rows by class, selection mode and calibration scope."""
        with self.connection() as connection:
            return connection.execute(
                f"""
                SELECT data_class, sn_mode, selection_op, gate_applied,
                       selected_threshold, uncertainty_calibration_scope,
                       COUNT(*) AS events
                FROM track.{TABLE}
                GROUP BY ALL
                ORDER BY data_class, sn_mode, gate_applied, selected_threshold
                """
            ).df()


def open_track_nn(path: Path | str) -> TrackPredictions:
    """The track-NN database at ``path``.

    Not named ``open``: that is the reader's own factory, and a second ``open``
    in the same namespace is how a caller ends up reading events when they meant
    tracks.
    """
    return TrackPredictions(path=Path(path))
