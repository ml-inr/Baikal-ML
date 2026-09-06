"""Shared helpers for domain-gap analysis (exp vs out-of-training MC-muon).

All loaders read the per-checkpoint prediction DuckDBs + catalog_v2 and return
tidy pandas frames with `score`, `n_sn_hits`, `n_sn_strings`, `cluster`.

The MC-muon loader excludes nu-classifier *training* events (identified via the
NPY dataset back-links), so the MC suppression curve reflects generalisation,
not memorisation.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[4]
PREDS_DIR  = ROOT / "inference_v2/nu_classifier/preds"
CATALOG    = ROOT / "data_manager/catalog_v2.duckdb"
NPY_TRAIN  = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"

DEFAULT_CKPT = "260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"
EXP_FULL_TRAIN_NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    return c


def _training_exp_keys(npy_dir: str | Path = EXP_FULL_TRAIN_NPY) -> set[str]:
    """Set of 'part_key|local_idx' strings for exp events used in DA training
    (from the exp_full NPY back-links). Used to hold out a test set."""
    npy_dir = Path(npy_dir)
    pk = np.asarray(np.load(npy_dir / "exp_h5_part_keys.npy", allow_pickle=True), dtype=str)
    li = np.asarray(np.load(npy_dir / "exp_h5_local_event_ids.npy"), dtype=np.int64)
    return set(np.char.add(np.char.add(pk, "|"), li.astype(str)).tolist())


# ── Loaders ────────────────────────────────────────────────────────────────

def load_exp_scores(
    ckpt: str = DEFAULT_CKPT,
    source: str = "exp_full",
    thr_tag: str = "0p8",
    exclude_training: bool = False,
    train_npy_dir: str | Path = EXP_FULL_TRAIN_NPY,
) -> pd.DataFrame:
    """Scored exp events: event_fk, score, n_sn_hits, n_sn_strings, cluster.

    exclude_training: drop events used as the DA target (via exp_full NPY
    back-links) → out-of-training test set. Requires source='exp_full'.
    """
    db = PREDS_DIR / ckpt / f"{source}_thr{thr_tag}.duckdb"
    if not db.exists():
        raise FileNotFoundError(db)
    c = _con()
    c.execute(f"ATTACH '{db}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    loc = ", l.part_key, l.local_idx" if exclude_training else ""
    join = ("JOIN cat.h5_locations l ON l.event_fk = pr.event_fk"
            if exclude_training else "")
    df = c.execute(f"""
        SELECT pr.event_fk, pr.score, pr.n_sn_hits, pr.n_sn_strings, e.cluster{loc}
        FROM p.predictions pr
        JOIN cat.events e ON e.id = pr.event_fk
        {join}
    """).df()
    c.close()

    if exclude_training:
        train = _training_exp_keys(train_npy_dir)
        key = df["part_key"].astype(str) + "|" + df["local_idx"].astype(str)
        df = df[~key.isin(train)].drop(columns=["part_key", "local_idx"]).copy()
    return df


def load_exp_meta(
    ckpt: str = DEFAULT_CKPT,
    source: str = "exp_full",
    thr_tag: str = "0p8",
    score_min: float | None = None,
) -> pd.DataFrame:
    """Scored exp events with full metadata: event_fk, score, n_sn_hits,
    n_sn_strings, cluster, run (for per-run/cluster localisation)."""
    db = PREDS_DIR / ckpt / f"{source}_thr{thr_tag}.duckdb"
    if not db.exists():
        raise FileNotFoundError(db)
    c = _con()
    c.execute(f"ATTACH '{db}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    where = f"WHERE pr.score > {score_min}" if score_min is not None else ""
    df = c.execute(f"""
        SELECT pr.event_fk, pr.score, pr.n_sn_hits, pr.n_sn_strings,
               e.cluster, e.run
        FROM p.predictions pr
        JOIN cat.events e ON e.id = pr.event_fk
        {where}
    """).df()
    c.close()
    return df


def _training_muatm_keys() -> set[str]:
    """Set of 'part_key|local_idx' strings for muatm training events."""
    pk  = np.load(NPY_TRAIN / "h5_part_keys.npy", allow_pickle=True)
    le  = np.load(NPY_TRAIN / "h5_local_event_ids.npy")
    lab = np.load(NPY_TRAIN / "labels.npy")
    m = np.asarray(lab, dtype=float) < 0.5                      # muatm = label 0
    pk_m = np.asarray(pk[m], dtype=str)
    le_m = np.asarray(le[m], dtype=np.int64)
    return set(np.char.add(np.char.add(pk_m, "|"), le_m.astype(str)).tolist())


def load_mc_muatm_scores(
    ckpt: str = DEFAULT_CKPT,
    thr_tag: str = "0p8",
    exclude_training: bool = True,
) -> pd.DataFrame:
    """Scored MC muatm events. If exclude_training, drop nu-classifier train set.

    Returns score, n_sn_hits, n_sn_strings, cluster (+ part_key/local_idx kept
    internally only when excluding training).
    """
    db = PREDS_DIR / ckpt / f"mc_merged_thr{thr_tag}.duckdb"
    if not db.exists():
        raise FileNotFoundError(db)
    c = _con()
    c.execute(f"ATTACH '{db}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    # event_fk is PK/indexed → joins are fast; we fetch part_key/local_idx so we
    # can exclude training events without scanning the full 651M h5_locations.
    df = c.execute("""
        SELECT pr.event_fk, pr.score, pr.n_sn_hits, pr.n_sn_strings,
               e.cluster, l.part_key, l.local_idx
        FROM p.predictions pr
        JOIN cat.events e        ON e.id       = pr.event_fk AND e.data_class = 'muatm_2020'
        JOIN cat.h5_locations l  ON l.event_fk = pr.event_fk
    """).df()
    c.close()

    if exclude_training:
        train = _training_muatm_keys()
        key = df["part_key"].astype(str) + "|" + df["local_idx"].astype(str)
        df = df[~key.isin(train)].copy()

    return df.drop(columns=["part_key", "local_idx"], errors="ignore")


# ── Suppression / survival curves ──────────────────────────────────────────

def default_thresholds() -> np.ndarray:
    """Threshold grid, denser near 1 where the tail matters."""
    return np.unique(np.concatenate([
        np.linspace(0.0, 0.9, 46),
        np.linspace(0.9, 0.999, 60),
    ]))


def survival_curve(scores: np.ndarray, thresholds: np.ndarray) -> dict:
    """Survival fraction f(ξ)=N(score>ξ)/N with Poisson (sqrt) error.

    Returns dict: frac, err, n_pass (per threshold), n_total.
    For zero survivors, frac=0 (mark as upper limit downstream).
    """
    s = np.asarray(scores)
    n = len(s)
    n_pass = np.array([(s > t).sum() for t in thresholds], dtype=float)
    frac = n_pass / max(n, 1)
    err  = np.sqrt(np.maximum(n_pass, 1)) / max(n, 1)   # ~Poisson on the count
    return {"frac": frac, "err": err, "n_pass": n_pass, "n_total": n}


def add_curve(ax, thresholds, surv: dict, color: str, label: str,
              lw: float = 2.0, fill: bool = True) -> None:
    """Plot one survival curve (log y) with a Poisson error band on an axes."""
    f = surv["frac"]; e = surv["err"]; m = f > 0
    ax.plot(thresholds[m], f[m], color=color, lw=lw,
            label=f"{label} (N={surv['n_total']:,})")
    if fill:
        ax.fill_between(thresholds[m], np.clip(f[m] - e[m], 1e-12, 1.0),
                        f[m] + e[m], color=color, alpha=0.2)


def gap_table(exp_scores, mc_scores, thresholds=(0.8, 0.9, 0.95, 0.99)) -> pd.DataFrame:
    """Domain-gap summary: exp vs MC-muon survival + ratio at key thresholds."""
    exp = np.asarray(exp_scores); mc = np.asarray(mc_scores)
    ne, nm = len(exp), len(mc)
    rows = []
    for t in thresholds:
        fe = (exp > t).sum() / ne
        km = (mc > t).sum()
        fm = km / nm
        rows.append({
            "threshold": t,
            "exp_total": ne, "mc_total": nm,
            "mc_n_pass": int(km),
            "exp_n_pass": int(fe * ne),
            "exp_survive": fe,
            "mc_muon_survive": fm if km > 0 else np.nan,
            "gap (exp/mc)": (fe / fm) if km > 0 else np.inf,
            "mc_suppression": (1.0 / fm) if km > 0 else np.inf,
        })
    return pd.DataFrame(rows)
