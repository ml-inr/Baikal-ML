"""Fitting and testing helpers for the group-structure notebook.

Two things here are load-bearing and easy to get wrong elsewhere, so they live in one place:

**Splits are by part or run, never by event.** Neighbouring events in one run share a detector
state, a geometry and a calibration; an event-level split leaks that across the boundary and
every score comes out optimistic.

**The null for "are these two samples one population" is a permutation, not 0.5.** A tree fitted
on a few thousand events reaches an AUC above 0.5 on held-out data by chance alone. Permuting
the labels and refitting the identical pipeline measures how much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 20260825


def part_split(frame: pd.DataFrame, test_fraction: float = 0.3,
               seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks that never split one part across the boundary."""
    parts = np.sort(frame.part_key.unique())
    rng = np.random.default_rng(seed)
    test_parts = set(rng.permutation(parts)[:max(1, int(round(len(parts) * test_fraction)))])
    is_test = frame.part_key.isin(test_parts).to_numpy()
    return ~is_test, is_test


def _clean(x: pd.DataFrame) -> np.ndarray:
    """Trees cannot take NaN. Fill with a value outside every feature's range and record it.

    Not the median: a NaN here means "undefined for this event" (no repeated hits, one string
    only), which is information. A sentinel lets the tree split on it explicitly.
    """
    return x.to_numpy(dtype=np.float64, na_value=-999.0)


def fit_tree(frame: pd.DataFrame, features: list[str], label: np.ndarray,
             max_depth: int = 4, min_leaf: int = 50,
             seed: int = SEED) -> dict:
    """One shallow tree, split by part, scored on held-out parts."""
    train, test = part_split(frame, seed=seed)
    x, y = _clean(frame[features]), label.astype(int)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                 class_weight="balanced", random_state=seed)
    clf.fit(x[train], y[train])
    p_test = clf.predict_proba(x[test])[:, 1]
    auc = roc_auc_score(y[test], p_test) if len(np.unique(y[test])) > 1 else np.nan
    used = [features[i] for i in np.argsort(clf.feature_importances_)[::-1]
            if clf.feature_importances_[i] > 0.01]
    return {"clf": clf, "auc": float(auc), "train": train, "test": test,
            "importances": pd.Series(clf.feature_importances_, index=features)
                             .sort_values(ascending=False),
            "top_features": used,
            "rules": export_text(clf, feature_names=list(features), decimals=3,
                                 max_depth=max_depth),
            "n_train": int(train.sum()), "n_test": int(test.sum())}


def permutation_null(frame: pd.DataFrame, features: list[str], label: np.ndarray,
                     n: int = 200, **kw) -> np.ndarray:
    """AUCs from refitting the identical pipeline with the labels shuffled."""
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        out.append(fit_tree(frame, features, rng.permutation(label), **kw)["auc"])
    return np.array([a for a in out if np.isfinite(a)])


def apply_rules(clf: DecisionTreeClassifier, frame: pd.DataFrame,
                features: list[str]) -> np.ndarray:
    """Probabilities from a tree fitted elsewhere — the cross-domain transfer test."""
    return clf.predict_proba(_clean(frame[features]))[:, 1]


def novelty(reference: pd.DataFrame, target: pd.DataFrame, features: list[str],
            contamination: float = 0.05, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Isolation forest fitted on `reference`, scored on `reference` (held out) and `target`.

    Returns (reference scores, target scores). Lower is more anomalous.
    """
    train, test = part_split(reference, seed=seed)
    forest = IsolationForest(n_estimators=300, contamination=contamination,
                             random_state=seed, n_jobs=8)
    forest.fit(_clean(reference[features])[train])
    return (forest.score_samples(_clean(reference[features])[test]),
            forest.score_samples(_clean(target[features])))


def concentration(frame: pd.DataFrame, selected: np.ndarray, by: str) -> pd.DataFrame:
    """Is a selected subset concentrated in some run/cluster? The H3 check.

    Reports the selected share per level against the overall share, so a level holding twice
    its due is visible whatever its size.
    """
    tot = frame.groupby(by).size().rename("n_total")
    sel = frame[selected].groupby(by).size().rename("n_selected")
    out = pd.concat([tot, sel], axis=1).fillna(0)
    out["share_selected"] = out.n_selected / out.n_selected.sum()
    out["share_total"] = out.n_total / out.n_total.sum()
    out["ratio"] = out.share_selected / out.share_total.replace(0, np.nan)
    return out.sort_values("ratio", ascending=False)
