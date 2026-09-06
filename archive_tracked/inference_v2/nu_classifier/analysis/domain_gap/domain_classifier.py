import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Domain classifier — exp-hi vs MC muon (tabular features)

    Train a simple classical ML classifier to separate **high-score exp** events
    from **MC muatm** events, using per-event aggregate statistics over the
    SN-filtered signal hits. Two questions:

    1. **Separability / domain gap** — how high is the cross-validated AUC?
       AUC ≈ 0.5 ⇒ domains overlap; AUC → 1 ⇒ strong domain shift.
    2. **Feature importance** — which event statistics differ most between exp
       and MC (permutation importance).

    MC is taken in the **same score region** (score > `NU_THR`) by default, so
    separability reflects domain shift, not signal-vs-background.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import duckdb

    _here = Path(__file__).resolve().parent
    for p in (str(_here), str(_here.parents[3])):   # dir + repo root
        if p not in sys.path:
            sys.path.insert(0, p)

    import archive_tracked.inference_v2.nu_classifier.analysis.domain_gap._common as C
    import archive_tracked.inference_v2.nu_classifier.analysis.domain_gap._features as F
    from inference_v2.shared.model_utils import load_sn_model
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        predict_flat as sn_predict_flat,
    )

    return C, F, duckdb, load_sn_model, mo, np, pd, plt, sn_predict_flat


@app.cell
def _(C):
    # ── Config ────────────────────────────────────────────────────────────
    CKPT = C.DEFAULT_CKPT
    NU_THR = 0.8                 # "high score" cut for exp and MC
    MC_MODE = "score_matched"    # "score_matched" (MC score>NU_THR) | "all" (any score)
    EXCLUDE_TRAINING = False     # drop nu-classifier train events from MC muon set
    MAX_PER_CLASS = 4000         # balance: subsample each class to <= this
    SN_THRESHOLD = 0.8
    AMP_CLIP = 100.0             # clip charge at Q=100 PE as the nu-classifier does
                                 # (only used when FILTER_ANOMALIES is False)
    # ── Anomaly filter (drop late-hit / high-charge events) ───────────────
    # When on, features are computed RAW and events with t_max>cut or
    # q_max>cut are dropped. Survivors have q_max<=100, so for them raw == the
    # model-clipped view (clipping a hit-set whose max is <=100 is a no-op) —
    # hence AMP_CLIP is irrelevant here.
    FILTER_ANOMALIES = True
    T_MAX_CUT = 1500.0
    Q_MAX_CUT = 100.0
    SN_DEVICE = "cuda:4"
    SEED = 42
    return (
        AMP_CLIP,
        CKPT,
        EXCLUDE_TRAINING,
        FILTER_ANOMALIES,
        MAX_PER_CLASS,
        MC_MODE,
        NU_THR,
        Q_MAX_CUT,
        SEED,
        SN_DEVICE,
        SN_THRESHOLD,
        T_MAX_CUT,
    )


@app.cell
def _(
    C,
    CKPT,
    EXCLUDE_TRAINING,
    MAX_PER_CLASS,
    MC_MODE,
    NU_THR,
    SEED,
    duckdb,
    np,
):
    # ── Select event_fks for the two classes ─────────────────────────────
    rng = np.random.default_rng(SEED)

    ce = duckdb.connect(str(C.PREDS_DIR / CKPT / "exp_full_thr0p8.duckdb"), read_only=True)
    ce.execute("PRAGMA disable_progress_bar")
    exp_fks_all = ce.execute(
        "SELECT event_fk FROM predictions WHERE score > ?", [NU_THR]
    ).df()["event_fk"].to_numpy()
    ce.close()

    cm = duckdb.connect(str(C.PREDS_DIR / CKPT / "mc_merged_thr0p8.duckdb"), read_only=True)
    cm.execute("PRAGMA disable_progress_bar")
    cm.execute(f"ATTACH '{C.CATALOG}' AS cat (READ_ONLY)")
    _where = "e.data_class='muatm_2020'" + (f" AND pr.score>{NU_THR}" if MC_MODE == "score_matched" else "")
    mc_fks_all = cm.execute(f"""
        SELECT pr.event_fk FROM predictions pr
        JOIN cat.events e ON e.id = pr.event_fk
        WHERE {_where}
    """).df()["event_fk"].to_numpy()
    cm.close()

    if EXCLUDE_TRAINING:
        from archive_tracked.inference_v2.nu_classifier.analysis.domain_gap._common import _training_muatm_keys  # noqa
        # (kept simple: training exclusion handled at feature stage if needed)

    n = min(len(exp_fks_all), len(mc_fks_all), MAX_PER_CLASS)
    exp_fks = rng.choice(exp_fks_all, n, replace=False)
    mc_fks  = rng.choice(mc_fks_all,  n, replace=False)
    print(f"exp-hi available={len(exp_fks_all):,}  MC available={len(mc_fks_all):,}  -> {n:,}/class")
    return exp_fks, mc_fks


@app.cell
def _(SN_DEVICE, load_sn_model):
    # ── Load sig-noise model (GPU) ────────────────────────────────────────
    sn_nn, _sn_cfg, sn_dev = load_sn_model(device=SN_DEVICE)
    print(f"SN model on {sn_dev}")
    return sn_dev, sn_nn


@app.cell
def _(
    AMP_CLIP,
    F,
    FILTER_ANOMALIES,
    Q_MAX_CUT,
    SN_THRESHOLD,
    T_MAX_CUT,
    exp_fks,
    mc_fks,
    pd,
    sn_dev,
    sn_nn,
    sn_predict_flat,
):
    # ── Compute tabular features for both classes ─────────────────────────
    # If filtering: compute RAW (amp_clip=None) so true q_max/t_max are visible
    # for flagging; survivors (q_max<=100) equal the model-clipped view anyway.
    # If not filtering: clip at AMP_CLIP to mirror the model's forward().
    _clip = None if FILTER_ANOMALIES else AMP_CLIP
    feat_exp = F.compute_features(exp_fks, "exp_full", sn_nn, sn_dev,
                                  sn_predict_flat, sn_threshold=SN_THRESHOLD, amp_clip=_clip)
    feat_mc  = F.compute_features(mc_fks, "muatm_2020", sn_nn, sn_dev,
                                  sn_predict_flat, sn_threshold=SN_THRESHOLD, amp_clip=_clip)
    feat_exp["domain"] = 1   # exp
    feat_mc["domain"]  = 0   # MC
    # add dimensionless ratio features (q_concentration, t_concentration, ...)
    data = F.add_derived_features(pd.concat([feat_exp, feat_mc], ignore_index=True))

    if FILTER_ANOMALIES:
        keep = (data.t_max <= T_MAX_CUT) & (data.q_max <= Q_MAX_CUT)
        n_exp_drop = int(((~keep) & (data.domain == 1)).sum())
        n_mc_drop  = int(((~keep) & (data.domain == 0)).sum())
        print(f"anomaly filter (t_max>{T_MAX_CUT:g} | q_max>{Q_MAX_CUT:g}): "
              f"dropped exp={n_exp_drop:,}  MC={n_mc_drop:,}")
        data = data[keep]

    data = data.dropna(subset=F.FEATURE_COLS_ALL)
    X = data[F.FEATURE_COLS_ALL].to_numpy()
    y = data["domain"].to_numpy()
    print(f"dataset: {X.shape}  ({len(F.FEATURE_COLS_ALL)} features)  "
          f"exp={int(y.sum()):,}  MC={int((1 - y).sum()):,}")
    # # Filtering
    # feat_exp = feat_exp[(feat_exp.q_max <= Q_MAX_CUT) & (feat_exp.t_max <= T_MAX_CUT)]
    # feat_mc = feat_mc[(feat_mc.q_max <= Q_MAX_CUT) & (feat_mc.t_max <= T_MAX_CUT)]
    return X, data, y


@app.cell
def _(SEED, X, plt, y):
    # ── Train + cross-validated AUC + ROC ─────────────────────────────────
    # Gradient boosting (HistGradientBoosting) — better feature handling than RF
    # and more trustworthy importances (RF impurity inflates correlated features).
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)

    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5))
    ax_roc.plot(fpr, tpr, color="purple", lw=2, label=f"AUC = {auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("FPR (MC kept)"); ax_roc.set_ylabel("TPR (exp kept)")
    ax_roc.set_title(f"exp-hi vs MC muon — domain separability\nAUC={auc:.4f} "
                     f"({'large gap' if auc > 0.8 else 'overlap' if auc < 0.6 else 'moderate'})")
    ax_roc.legend(); ax_roc.grid(alpha=0.3)
    print(f"Cross-validated AUC = {auc:.4f}")
    fig_roc
    return (clf,)


@app.cell
def _(F, SEED, X, clf, np, plt, y):
    # ── Permutation feature importance ────────────────────────────────────
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=SEED)
    clf.fit(Xtr, ytr)
    perm = permutation_importance(clf, Xte, yte, n_repeats=20,
                                  random_state=SEED, n_jobs=-1, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)
    names = np.array(F.FEATURE_COLS_ALL)[order]

    fig_imp, ax_imp = plt.subplots(figsize=(7, 6))
    ax_imp.barh(range(len(names)), perm.importances_mean[order],
                xerr=perm.importances_std[order], color="teal")
    ax_imp.set_yticks(range(len(names))); ax_imp.set_yticklabels(names, fontsize=9)
    ax_imp.set_xlabel("permutation importance (Δ AUC)")
    ax_imp.set_title("Which statistics separate exp from MC")
    ax_imp.grid(alpha=0.3, axis="x")
    fig_imp.tight_layout()
    fig_imp
    return


@app.cell
def _(F, data, np, plt, y):
    # ── Univariate separation: single-feature AUC ─────────────────────────
    # How well each feature ALONE separates exp vs MC (marginal difference).
    # Complements permutation importance: a feature can rank high in the
    # multivariate model yet be near-identical marginally (interaction /
    # correlation) — that shows up as low single-feature AUC here.
    from sklearn.metrics import roc_auc_score as _auc

    uni = np.array([max(a, 1 - a) for a in
                    (_auc(y, data[c].values) for c in F.FEATURE_COLS_ALL)])
    o_uni = np.argsort(uni)
    fig_uni, ax_uni = plt.subplots(figsize=(7, 7))
    ax_uni.barh(range(len(F.FEATURE_COLS_ALL)), uni[o_uni] - 0.5, left=0.5,
                color="darkorange")
    ax_uni.set_yticks(range(len(F.FEATURE_COLS_ALL)))
    ax_uni.set_yticklabels(np.array(F.FEATURE_COLS_ALL)[o_uni], fontsize=8)
    ax_uni.axvline(0.5, color="black", lw=1)
    ax_uni.set_xlabel("single-feature AUC (0.5 = identical, → 1 = fully separated)")
    ax_uni.set_title("Univariate domain separation per feature")
    ax_uni.grid(alpha=0.3, axis="x")
    fig_uni.tight_layout()
    fig_uni
    return


@app.cell
def _(data, np, plt):
    # ── Distributions of the top discriminating features ──────────────────
    #top = list(np.array(F.FEATURE_COLS_ALL)[order][::-1])#[:6])
    topplot = ['q_max', 'q_mean', 'q_std', 't_max', 't_min', 't_std', 't_per_hit', 'x_std']
    fig_d, axes_d = plt.subplots(3, 3, figsize=(15, 8))
    axes_d = axes_d.flatten()
    for ax_d, col in zip(axes_d, topplot):
        e = data[data.domain == 1][col].to_numpy()
        m = data[data.domain == 0][col].to_numpy()
        allv = np.concatenate([e, m])
        lo, hi = np.percentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax_d.hist(m, bins=bins, density=True, histtype="step", lw=2,
                  color="tomato", label="MC muon")
        ax_d.hist(e, bins=bins, density=True, histtype="step", lw=2,
                  color="steelblue", label="exp-hi")
        ax_d.set_xlabel(col); ax_d.set_ylabel("density"); ax_d.legend(fontsize=8)
    fig_d.suptitle("Top discriminating features: exp-hi vs MC muon", y=1.01)
    fig_d.tight_layout()
    fig_d
    return


if __name__ == "__main__":
    app.run()
