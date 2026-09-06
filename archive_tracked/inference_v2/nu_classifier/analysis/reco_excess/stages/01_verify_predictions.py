"""Stage 01 -- verify the prediction databases before measuring anything with them.

Stage 00 checked the sig-noise probabilities.  This checks what was built on top:
that every source part reached the database, that no event is there twice, that
the scores are usable numbers, and that the per-event hit counts stored beside
them agree with an independent recomputation from the probability file.

The last check is the one that matters most.  ``n_sn_hits`` and ``n_sn_strings``
define the h8s3 selection the whole analysis is counted in; if the writer and the
reader disagree about them -- a different threshold, a different string divisor --
every rate downstream is wrong in a way no plot would reveal.

Coverage is checked against the source h5 rather than against the catalog: the
catalog is itself a derived product, and a part missing from both would agree
with itself.

The two sources are keyed differently in the catalog and the check has to follow
that rather than assume it.  For ``mc_reco`` the ``run`` column holds the whole
part name and ``cluster`` is unused; for ``exp_reco`` a part is a *(cluster, run)*
pair with ``run`` a bare number, so 370 parts appear as 260 distinct run values.
Comparing part names against run numbers made every experimental part look
missing -- a failure of the check, not of the data.

Usage:
    python stages/01_verify_predictions.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import provenance                                              # noqa: E402

LOG = logging.getLogger("verify")


def _parse_exp_part(part: str) -> tuple[int, int, int]:
    """``part_s2020_c07_r0465`` -> ``(2020, 7, 465)``."""
    import re
    m = re.match(r"part_s(\d+)_c(\d+)_r(\d+)$", part)
    if m is None:
        raise ValueError(f"unparsable experimental part name: {part}")
    return int(m[1]), int(m[2]), int(m[3])

STRING_DIVISOR = 36
SAMPLE_PARTS = 12


def source_parts(path: Path, groups: list[str]) -> dict[str, set[str]]:
    with h5py.File(path, "r") as handle:
        return {g: set(handle[g]["raw"]["ev_starts"].keys()) for g in groups}


def check(name: str, db: Path, catalog: Path, source_h5: Path, probs_h5: Path,
          groups: list[str], class_of: dict[str, str], threshold: float) -> pd.DataFrame:
    rows = []
    con = duckdb.connect(str(db), read_only=True)
    con.execute(f"ATTACH '{catalog}' AS cat (READ_ONLY)")
    total, distinct = con.execute(
        "SELECT count(*), count(DISTINCT event_fk) FROM predictions").fetchone()
    bad_score = con.execute(
        "SELECT count(*) FROM predictions WHERE score IS NULL OR score < 0 "
        "OR score > 1 OR NOT isfinite(score)").fetchone()[0]
    in_source = source_parts(source_h5, groups)

    for group in groups:
        data_class = class_of[group]
        if name == "exp_reco":
            # part_s2020_c07_r0465 -> (7, "465"), matching the catalog's keying
            db_parts = {f"part_s{int(season)}_c{int(cluster):02d}_r{int(run):04d}"
                        for season, cluster, run in con.execute(
                            "SELECT DISTINCT e.season, e.cluster, e.run FROM predictions p "
                            "JOIN cat.events e ON p.event_fk = e.id WHERE e.source = ?",
                            [name]).fetchall()}
        else:
            db_parts = {r[0] for r in con.execute(
                "SELECT DISTINCT e.run FROM predictions p JOIN cat.events e "
                "ON p.event_fk = e.id WHERE e.source = ? AND e.data_class = ?",
                [name, data_class]).fetchall()}
        n_events = con.execute(
            "SELECT count(*) FROM predictions p JOIN cat.events e ON p.event_fk = e.id "
            "WHERE e.source = ? AND e.data_class = ?", [name, data_class]).fetchone()[0]
        missing = in_source[group] - db_parts
        problems = []
        if missing:
            # a part with no event surviving the preselection is legitimately absent
            empty = 0
            with h5py.File(source_h5, "r") as src, h5py.File(probs_h5, "r") as pf:
                for part in sorted(missing):
                    starts = src[f"{group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
                    probs = pf[f"{group}/probs/{part}/data"][:]
                    counts = (np.add.reduceat((probs > threshold).astype(np.int64),
                                              starts[:-1]) if len(starts) > 1
                              else np.zeros(0, dtype=np.int64))
                    if not (counts >= 5).any():
                        empty += 1
            if empty < len(missing):
                problems.append(f"{len(missing) - empty} source parts absent from the DB "
                                f"with events that should have passed")
            else:
                LOG.info("  %s: %d parts absent, all empty after preselection",
                         group, empty)
        rows.append({"source": name, "group": group, "data_class": data_class,
                     "source_parts": len(in_source[group]), "db_parts": len(db_parts),
                     "events": n_events, "problems": "; ".join(problems),
                     "status": "ok" if not problems else "FAILED"})
        LOG.info("%s/%s: %d parts, %d events %s", name, group, len(db_parts),
                 n_events, rows[-1]["status"])

    # independent recomputation of the stored hit counts
    mismatch = checked = 0
    with h5py.File(source_h5, "r") as src, h5py.File(probs_h5, "r") as pf:
        for group in groups:
            parts = sorted(in_source[group])[:SAMPLE_PARTS]
            for part in parts:
                starts = src[f"{group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
                chan = src[f"{group}/raw/channels/{part}/data"][:]
                probs = pf[f"{group}/probs/{part}/data"][:]
                mask = probs > threshold
                counts = np.add.reduceat(mask.astype(np.int64), starts[:-1])
                strings = np.array([
                    np.unique(chan[starts[i]:starts[i + 1]][
                        mask[starts[i]:starts[i + 1]]] // STRING_DIVISOR).size
                    for i in range(len(starts) - 1)])
                if name == "exp_reco":
                    season, cluster, run = _parse_exp_part(part)
                    stored = con.execute(
                        "SELECT e.event_id, p.n_sn_hits, p.n_sn_strings FROM predictions p "
                        "JOIN cat.events e ON p.event_fk = e.id "
                        "WHERE e.source = ? AND e.cluster = ? AND e.run = ?",
                        [name, cluster, str(run)]).df()
                    # the catalog's event_id is the reco physics id from
                    # header_prty, not the row position -- map it explicitly
                    physical = src[f"{group}/header_prty/{part}/data"][:, 3]
                    position = {int(v): i for i, v in enumerate(physical)}
                    stored = stored.assign(
                        event_id=[position.get(int(v), -1) for v in stored.event_id])
                else:
                    stored = con.execute(
                        "SELECT e.event_id, p.n_sn_hits, p.n_sn_strings FROM predictions p "
                        "JOIN cat.events e ON p.event_fk = e.id "
                        "WHERE e.source = ? AND e.data_class = ? AND e.run = ?",
                        [name, class_of[group], part]).df()
                if not len(stored):
                    continue
                idx = stored.event_id.to_numpy()
                ok = (idx >= 0) & (idx < len(counts))
                idx = idx[ok]
                checked += len(idx)
                mismatch += int((counts[idx] != stored.n_sn_hits.to_numpy()[ok]).sum()
                                + (strings[idx] != stored.n_sn_strings.to_numpy()[ok]).sum())
    con.close()
    # a check that silently checks nothing is worse than no check
    if checked == 0:
        rows.append({"source": name, "group": "(all)", "data_class": "",
                     "source_parts": -1, "db_parts": -1, "events": total,
                     "problems": "hit-count recomputation matched zero events -- "
                                 "the lookup key is wrong, the check did not run",
                     "status": "FAILED"})
        return pd.DataFrame(rows)
    rows.append({"source": name, "group": "(all)", "data_class": "",
                 "source_parts": -1, "db_parts": -1, "events": total,
                 "problems": (f"{total - distinct} duplicate event_fk; " if total != distinct else "")
                             + (f"{bad_score} unusable scores; " if bad_score else "")
                             + (f"{mismatch} hit-count mismatches of {checked}"
                                if mismatch else f"hit counts verified on {checked} events"),
                 "status": "ok" if (total == distinct and not bad_score and not mismatch)
                           else "FAILED"})
    LOG.info("%s: %d rows, %d distinct, %d unusable scores, %d/%d hit-count mismatches",
             name, total, distinct, bad_score, mismatch, checked)
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    root = (HERE / cfg["paths"]["root"]).resolve()
    preds = root / cfg["paths"]["preds"] / cfg["model"]
    catalog = root / cfg["paths"]["catalog"]
    threshold = float(cfg["sig_noise_threshold"])

    mc_classes = {g: (f"{g}_2020") for g in cfg["mc_reco_groups"]}
    report = pd.concat([
        check("mc_reco", preds / "mc_reco_thr0p8.duckdb", catalog,
              root / cfg["paths"]["h5"]["mc_reco"], root / cfg["paths"]["probs"]["mc_reco"],
              cfg["mc_reco_groups"], mc_classes, threshold),
        check("exp_reco", preds / "exp_reco_thr0p8.duckdb", catalog,
              root / cfg["paths"]["h5"]["exp_reco"], root / cfg["paths"]["probs"]["exp_reco"],
              [cfg["exp_reco_group"]], {cfg["exp_reco_group"]: "exp_reco"}, threshold),
    ], ignore_index=True)
    failed = report[report.status != "ok"]
    provenance.write(report, HERE / "data" / "01_prediction_verification.parquet",
                     stage="01_verify_predictions", config_path=HERE / "config.yaml",
                     inputs=[], started=started, notes={"failures": int(len(failed))})
    if len(failed):
        LOG.error("FAILED:\n%s", failed[["source", "group", "problems"]].to_string(index=False))
        raise SystemExit(1)
    LOG.info("prediction databases verified in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
