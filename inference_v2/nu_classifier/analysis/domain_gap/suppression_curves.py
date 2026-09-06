import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Suppression curves — exp vs out-of-training MC muon

    Survival fraction `N(score>ξ)/N` vs score threshold ξ, for **exp** and for
    **out-of-training MC muatm** (nu-classifier training events excluded via the
    NPY back-links). The vertical gap between the curves is the MC↔exp domain
    mismatch as a function of threshold.

    Also split by an **energy proxy** (`n_sn_hits`) and by **cluster**.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    _here = Path(__file__).resolve().parent
    for p in (str(_here), str(_here.parents[3])):
        if p not in sys.path:
            sys.path.insert(0, p)
    import _common as C
    import _features as F
    from inference_v2.shared.model_utils import load_sn_model
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        predict_flat as sn_predict_flat,
    )

    return C, F, load_sn_model, mo, np, plt, sn_predict_flat


@app.cell
def _():
    # ── Config ────────────────────────────────────────────────────────────
    CKPT = "260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"
    EXP_SOURCE = "exp_full"          # exp_full | exp | exp_reco
    EXCLUDE_TRAINING = True          # drop nu-classifier train events from MC muon set
    NSN_BINS = [(5, 10), (10, 20), (20, 40), (40, 10_000)]   # energy-proxy bins

    # ── Anomaly filtering (t_max / q_max) ─────────────────────────────────
    # Features are computed only for events with score > FEATURE_SCORE_MIN
    # (anomalies live in the high-score region); flagged event_fks are then
    # dropped from the survival curves. Lower FEATURE_SCORE_MIN to extend the
    # clean curve to smaller ξ — at extra feature-extraction cost.
    T_MAX_CUT = 1500.0
    Q_MAX_CUT = 10.0
    FEATURE_SCORE_MIN = 0.7
    SN_THRESHOLD = 0.8
    SN_DEVICE = "cuda:4"
    return (
        CKPT,
        EXCLUDE_TRAINING,
        EXP_SOURCE,
        FEATURE_SCORE_MIN,
        NSN_BINS,
        Q_MAX_CUT,
        SN_DEVICE,
        SN_THRESHOLD,
        T_MAX_CUT,
    )


@app.cell
def _(C, CKPT, EXCLUDE_TRAINING, EXP_SOURCE):
    # ── Load scored events (slow: catalog join, ~1-2 min) ─────────────────
    exp = C.load_exp_scores(ckpt=CKPT, source=EXP_SOURCE)
    mc = C.load_mc_muatm_scores(ckpt=CKPT, exclude_training=EXCLUDE_TRAINING)
    print(f"exp ({EXP_SOURCE}): {len(exp):,}   "
          f"MC muatm (exclude_training={EXCLUDE_TRAINING}): {len(mc):,}")
    exp.head()
    return exp, mc


@app.cell
def _(C, exp, mc):
    # ── Domain-gap summary table ──────────────────────────────────────────
    gap = C.gap_table(exp.score.values, mc.score.values)
    print(gap.to_string(index=False))
    gap
    return


@app.cell
def _(C, exp, mc, plt):
    # ── Overall overlay ───────────────────────────────────────────────────
    thr = C.default_thresholds()
    surv_exp = C.survival_curve(exp.score.values, thr)
    surv_mc = C.survival_curve(mc.score.values, thr)

    fig_all, ax_all = plt.subplots(figsize=(8, 5))
    C.add_curve(ax_all, thr, surv_exp, "steelblue", "exp")
    C.add_curve(ax_all, thr, surv_mc, "tomato", "MC muatm (out-of-train)")
    floor = 1.0 / surv_mc["n_total"]
    ax_all.axhline(floor, color="tomato", ls=":", lw=1, alpha=0.6,
                   label=f"MC 1-event floor (1/{surv_mc['n_total']:,})")
    ax_all.set_yscale("log")
    ax_all.set_xlabel("score threshold ξ")
    ax_all.set_ylabel("survival  N(score>ξ)/N")
    ax_all.set_title("Suppression: exp vs out-of-training MC muon")
    ax_all.legend(fontsize=8)
    ax_all.grid(alpha=0.3)
    fig_all
    return surv_mc, thr


@app.cell
def _(SN_DEVICE, load_sn_model):
    # ── SN model (for t_max / q_max feature extraction) ───────────────────
    sn_nn, _sn_cfg, sn_dev = load_sn_model(device=SN_DEVICE)
    print(f"SN model on {sn_dev}")
    return sn_dev, sn_nn


@app.cell
def _(
    EXP_SOURCE,
    F,
    FEATURE_SCORE_MIN,
    Q_MAX_CUT,
    SN_THRESHOLD,
    T_MAX_CUT,
    exp,
    mc,
    sn_dev,
    sn_nn,
    sn_predict_flat,
):
    # ── Flag t_max/q_max anomalies (only above FEATURE_SCORE_MIN) ──────────
    def _anomaly_fks(df, group):
        fks = df[df.score > FEATURE_SCORE_MIN].event_fk.to_numpy()
        if len(fks) == 0:
            return set()
        feat = F.compute_features(fks, group, sn_nn, sn_dev, sn_predict_flat,
                                  sn_threshold=SN_THRESHOLD, amp_clip=None)
        bad = feat[(feat.t_max > T_MAX_CUT) | (feat.q_max > Q_MAX_CUT)]
        return set(bad.event_fk.astype(int))

    exp_bad = _anomaly_fks(exp, EXP_SOURCE)
    mc_bad = _anomaly_fks(mc, "muatm_2020")
    exp_clean = exp[~exp.event_fk.isin(exp_bad)]
    mc_clean = mc[~mc.event_fk.isin(mc_bad)]
    print(f"exp flagged (score>{FEATURE_SCORE_MIN}): {len(exp_bad):,}/{len(exp):,}"
          f"  -> clean {len(exp_clean):,}")
    print(f"MC  flagged (score>{FEATURE_SCORE_MIN}): {len(mc_bad):,}/{len(mc):,}"
          f"  -> clean {len(mc_clean):,}")
    return exp_clean, mc_clean


@app.cell
def _():
    return


@app.cell
def _(C, Q_MAX_CUT, T_MAX_CUT, exp, exp_clean, mc_clean, plt, thr):
    # ── Suppression after anomaly removal ─────────────────────────────────
    surv_exp_raw = C.survival_curve(exp.score.values, thr)
    surv_exp_cln = C.survival_curve(exp_clean.score.values, thr)
    surv_mc_cln = C.survival_curve(mc_clean.score.values, thr)

    fig_cln, ax_cln = plt.subplots(figsize=(8, 5))
    C.add_curve(ax_cln, thr, surv_exp_raw, "lightblue", "exp (raw)", fill=False)
    C.add_curve(ax_cln, thr, surv_exp_cln, "steelblue", "exp (anomalies removed)")
    C.add_curve(ax_cln, thr, surv_mc_cln, "tomato", "MC muon (clean)")
    ax_cln.set_yscale("log")
    ax_cln.set_xlabel("score threshold ξ")
    ax_cln.set_ylabel("survival  N(score>ξ)/N")
    ax_cln.set_title(f"Suppression after removing t_max>{T_MAX_CUT:g} & q_max>{Q_MAX_CUT:g}")
    ax_cln.legend(fontsize=8)
    ax_cln.grid(alpha=0.3)
    fig_cln
    return


@app.cell
def _(C, NSN_BINS, exp, mc, np, plt, thr):
    # ── Split by energy proxy (n_sn_hits) ─────────────────────────────────
    fig_e, axes_e = plt.subplots(1, len(NSN_BINS), figsize=(5 * len(NSN_BINS), 4),
                                 sharey=True)
    axes_e = np.atleast_1d(axes_e)
    for ax_e, (lo, hi) in zip(axes_e, NSN_BINS):
        e_sub = exp[(exp.n_sn_hits >= lo) & (exp.n_sn_hits < hi)].score.values
        m_sub = mc[(mc.n_sn_hits >= lo) & (mc.n_sn_hits < hi)].score.values
        C.add_curve(ax_e, thr, C.survival_curve(e_sub, thr), "steelblue", "exp", fill=False)
        C.add_curve(ax_e, thr, C.survival_curve(m_sub, thr), "tomato", "MC mu", fill=False)
        ax_e.set_yscale("log")
        ax_e.set_xlabel("ξ")
        ax_e.set_title(f"n_sn_hits ∈ [{lo}, {hi})")
        ax_e.legend(fontsize=7)
        ax_e.grid(alpha=0.3)
    axes_e[0].set_ylabel("survival")
    fig_e.suptitle("Suppression by energy proxy (n_sn_hits)", y=1.03)
    fig_e.tight_layout()
    fig_e
    return


@app.cell
def _(C, exp, np, plt, surv_mc, thr):
    # ── Split by exp cluster (MC muon = single reference) ─────────────────
    clusters = sorted(exp.cluster.unique())
    ncol = 3
    nrow = (len(clusters) + ncol - 1) // ncol
    fig_c, axes_c = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharey=True)
    axes_c = np.atleast_1d(axes_c).flatten()
    for ax_c, cl in zip(axes_c, clusters):
        e_cl = exp[exp.cluster == cl].score.values
        C.add_curve(ax_c, thr, C.survival_curve(e_cl, thr), "steelblue", f"exp c{cl}", fill=False)
        C.add_curve(ax_c, thr, surv_mc, "tomato", "MC mu (all)", fill=False)
        ax_c.set_yscale("log")
        ax_c.set_xlabel("ξ")
        ax_c.set_title(f"cluster {cl}")
        ax_c.legend(fontsize=7)
        ax_c.grid(alpha=0.3)
    for ax_c in axes_c[len(clusters):]:
        ax_c.set_visible(False)
    fig_c.suptitle("Suppression by exp cluster", y=1.01)
    fig_c.tight_layout()
    fig_c
    return


@app.cell
def _(mo):
    mo.md("""
    ## Score distributions (for sharing)

    The same information as the survival curves, shown as plain score
    histograms — more intuitive at a glance. exp sits systematically higher
    in the tail than out-of-training MC muon.
    """)
    return


@app.cell
def _(exp_clean, mc_clean, np, plt):
    # ── Score distributions: exp vs MC muon (linear + log) ────────────────
    bins_sd = np.linspace(0, 1, 51)
    fig_sd, axes_sd = plt.subplots(1, 1, figsize=(10, 4.5))
    for ax_sd, logy in zip([axes_sd], [True]):
        ax_sd.hist(mc_clean.score.values, bins=bins_sd, density=True, histtype="step",
                   lw=2, color="tomato", label=f"MC muatm (N={len(mc_clean):,})")
        ax_sd.hist(exp_clean.score.values, bins=bins_sd, density=True, histtype="step",
                   lw=2, color="steelblue", label=f"exp (N={len(exp_clean):,})")
        for x in (0.8, 0.9):
            ax_sd.axvline(x, color="black", ls="--", lw=1, alpha=0.4)
        ax_sd.set_xlabel("nu-classifier score")
        ax_sd.set_ylabel("density" + (" (log)" if logy else ""))
        ax_sd.set_title("log y" if logy else "linear y")
        ax_sd.legend(fontsize=8)
        if logy:
            ax_sd.set_yscale("log")
    fig_sd.suptitle("Score distributions: exp vs out-of-training MC muon", y=1.02)
    fig_sd.tight_layout()
    fig_sd
    return


@app.cell
def _(exp, exp_clean, mc, mc_clean, np, plt):
    NSN_BINS_local_ = [(5,8), (8, 10), (10, 20), (20, 10_000)]
    # ── Score distributions split by energy proxy (n_sn_hits) ─────────────
    bins_sb = np.linspace(0, 1, 201)
    fig_sb, axes_sb = plt.subplots(1, len(NSN_BINS_local_), figsize=(5 * len(NSN_BINS_local_), 4),
                                   sharey=True)
    axes_sb = np.atleast_1d(axes_sb)
    for ax_sb, (lo_b, hi_b) in zip(axes_sb, NSN_BINS_local_):
        e_sb = exp[(exp.n_sn_hits >= lo_b) & (exp.n_sn_hits < hi_b) & (exp.score > 0.8)].score.values
        m_sb = mc[(mc.n_sn_hits >= lo_b) & (mc.n_sn_hits < hi_b) & (mc.score > 0.8)].score.values
        ax_sb.hist(m_sb, bins=bins_sb, density=True, histtype="step", lw=2,
                   color="tomato", label=f"MC ({len(m_sb):,})")
        ax_sb.hist(e_sb, bins=bins_sb, density=True, histtype="step", lw=2,
                   color="steelblue", label=f"exp ({len(e_sb):,})")
        ax_sb.axvline(0.8, color="black", ls="--", lw=1, alpha=0.4)
        ax_sb.set_yscale("log")
        ax_sb.set_xlabel("score")
        ax_sb.set_title(f"n_sn_hits ∈ [{lo_b}, {hi_b})")
        ax_sb.legend(fontsize=7)
        ax_sb.set_xlim(0.8, 1.0)
    axes_sb[0].set_ylabel("density (log)")
    fig_sb.suptitle("Score distributions by energy proxy (n_sn_hits)", y=1.03)
    fig_sb.tight_layout()
    fig_sb

    NSN_BINS_local_ = [(5,8), (8, 10), (10, 20), (20, 10_000)]
    # ── Score distributions split by energy proxy (n_sn_hits) ─────────────
    bins_sb = np.linspace(0, 1, 51)
    fig_sb_clean, axes_sb = plt.subplots(1, len(NSN_BINS_local_), figsize=(5 * len(NSN_BINS_local_), 4),
                                   sharey=True)
    axes_sb = np.atleast_1d(axes_sb)
    for ax_sb, (lo_b, hi_b) in zip(axes_sb, NSN_BINS_local_):
        e_sb = exp_clean[(exp_clean.n_sn_hits >= lo_b) & (exp_clean.n_sn_hits < hi_b) & (exp_clean.score > 0.8)].score.values
        m_sb = mc_clean[(mc_clean.n_sn_hits >= lo_b) & (mc_clean.n_sn_hits < hi_b) & (mc_clean.score > 0.8)].score.values
        ax_sb.hist(m_sb, bins=bins_sb, density=True, histtype="step", lw=2,
                   color="tomato", label=f"MC ({len(m_sb):,})")
        ax_sb.hist(e_sb, bins=bins_sb, density=True, histtype="step", lw=2,
                   color="steelblue", label=f"exp ({len(e_sb):,})")
        ax_sb.axvline(0.8, color="black", ls="--", lw=1, alpha=0.4)
        ax_sb.set_yscale("log")
        ax_sb.set_xlabel("score")
        ax_sb.set_title(f"n_sn_hits ∈ [{lo_b}, {hi_b})")
        ax_sb.legend(fontsize=7)
        ax_sb.set_xlim(0.8, 1.0)
    axes_sb[0].set_ylabel("density (log)")
    fig_sb_clean.suptitle("Score distributions by energy proxy (n_sn_hits)", y=1.03)
    fig_sb_clean.tight_layout()

    fig_sb, fig_sb_clean
    return


@app.cell
def _(C, exp_clean, mc_clean):
    # ── Domain-gap summary table ──────────────────────────────────────────
    gap_clean = C.gap_table(exp_clean.score.values, mc_clean.score.values)
    print(gap_clean.to_string(index=False))
    gap_clean
    return


app._unparsable_cell(
    r"""
    0.2%
    """,
    name="_"
)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TODO:
    1) В DANN заменить BCE loss классфикатора доменов на wasserstein loss (как в W-ганах)
    2) Аугментация послеимпульсами
    3) Спектральная норма фич.
    """)
    return


if __name__ == "__main__":
    app.run()
