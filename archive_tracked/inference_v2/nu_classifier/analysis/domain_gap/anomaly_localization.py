import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Anomaly localisation & filtering (exp-hi)

    Two exp-only anomalies found via the domain analysis, both inflating the
    time/charge features that drive the domain gap:

    1. **Late-hit cluster** — a separate population with `t_max > 1500` ns
       (drives `t_range`, `t_std`).
    2. **High charge** — many more events with `q_max > 100` p.e. than MC.

    Here we locate them (per cluster / run) and define a filter.
    Reference: MC muon (score-matched).
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
    for p in (str(_here), str(_here.parents[3])):
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
    NU_THR = 0.8
    T_MAX_CUT = 1500.0     # late-hit anomaly threshold (ns)
    Q_MAX_CUT = 100.0      # high-charge anomaly threshold (p.e.)
    SN_THRESHOLD = 0.8
    MAX_MC = 4000
    SN_DEVICE = "cuda:4"
    SEED = 42
    return (
        CKPT,
        MAX_MC,
        NU_THR,
        Q_MAX_CUT,
        SEED,
        SN_DEVICE,
        SN_THRESHOLD,
        T_MAX_CUT,
    )


@app.cell
def _(SN_DEVICE, load_sn_model):
    sn_nn, _sn_cfg, sn_dev = load_sn_model(device=SN_DEVICE)
    print(f"SN model on {sn_dev}")
    return sn_dev, sn_nn


@app.cell
def _(C, CKPT, F, NU_THR, SN_THRESHOLD, sn_dev, sn_nn, sn_predict_flat):
    # ── exp-hi: raw features (amp_clip=None to see true q_max) + metadata ──
    meta = C.load_exp_meta(ckpt=CKPT, source="exp_full", score_min=NU_THR)
    fexp = F.compute_features(meta.event_fk.values, "exp_full", sn_nn, sn_dev,
                              sn_predict_flat, sn_threshold=SN_THRESHOLD, amp_clip=None)
    exp_df = fexp.merge(meta, on="event_fk")
    print(f"exp-hi: {len(exp_df):,}  clusters={sorted(exp_df.cluster.unique())}")
    return (exp_df,)


@app.cell
def _(
    C,
    CKPT,
    F,
    MAX_MC,
    NU_THR,
    SEED,
    SN_THRESHOLD,
    duckdb,
    np,
    sn_dev,
    sn_nn,
    sn_predict_flat,
):
    # ── MC muon reference (score-matched) ─────────────────────────────────
    cm = duckdb.connect(str(C.PREDS_DIR / CKPT / "mc_merged_thr0p8.duckdb"), read_only=True)
    cm.execute("PRAGMA disable_progress_bar")
    cm.execute(f"ATTACH '{C.CATALOG}' AS cat (READ_ONLY)")
    mc_fks = cm.execute(f"""
        SELECT pr.event_fk FROM predictions pr
        JOIN cat.events e ON e.id = pr.event_fk
        WHERE pr.score > {NU_THR} AND e.data_class = 'muatm_2020'
    """).df()["event_fk"].to_numpy()
    cm.close()
    if len(mc_fks) > MAX_MC:
        mc_fks = np.random.default_rng(SEED).choice(mc_fks, MAX_MC, replace=False)
    mc_df = F.compute_features(mc_fks, "muatm_2020", sn_nn, sn_dev,
                               sn_predict_flat, sn_threshold=SN_THRESHOLD, amp_clip=None)
    print(f"MC-hi: {len(mc_df):,}")
    return (mc_df,)


@app.cell
def _(Q_MAX_CUT, T_MAX_CUT, exp_df, mc_df, np, plt):
    # ── Anomalous feature distributions ───────────────────────────────────
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax_dist, col, cut, xlim in [
        (axes_dist[0], "t_max", T_MAX_CUT, (-200, 3000)),
        (axes_dist[1], "q_max", Q_MAX_CUT, (0, 400)),
    ]:
        bins_d = np.linspace(*xlim, 60)
        ax_dist.hist(np.clip(mc_df[col], *xlim), bins=bins_d, density=True,
                     histtype="step", lw=2, color="tomato", label=f"MC muon ({len(mc_df):,})")
        ax_dist.hist(np.clip(exp_df[col], *xlim), bins=bins_d, density=True,
                     histtype="step", lw=2, color="steelblue", label=f"exp-hi ({len(exp_df):,})")
        ax_dist.axvline(cut, color="black", ls="--", lw=1.5, label=f"cut={cut:g}")
        ax_dist.set_yscale("log"); ax_dist.set_xlabel(col)
        ax_dist.set_ylabel("density (log)"); ax_dist.legend(fontsize=8)
    fig_dist.suptitle("Anomalous features: exp-hi vs MC muon", y=1.02)
    fig_dist.tight_layout()
    fig_dist
    return


@app.cell
def _(Q_MAX_CUT, T_MAX_CUT, exp_df, plt):
    # ── Joint structure: t_max vs q_max by cluster ────────────────────────
    fig_sc, ax_sc = plt.subplots(figsize=(7.5, 6))
    for cl, sub in exp_df.groupby("cluster"):
        ax_sc.scatter(sub.t_max, sub.q_max, s=8, alpha=0.4, label=f"c{cl}", rasterized=True)
    ax_sc.axvline(T_MAX_CUT, color="black", ls="--", lw=1)
    ax_sc.axhline(Q_MAX_CUT, color="black", ls="--", lw=1)
    ax_sc.set_xlabel("t_max (ns)"); ax_sc.set_ylabel("q_max (p.e.)")
    ax_sc.set_yscale("log")
    ax_sc.set_title("exp-hi: t_max vs q_max by cluster\n(late-hit + high-charge cluster, top-right)")
    ax_sc.legend(fontsize=8, markerscale=2)
    fig_sc.tight_layout()
    fig_sc
    return


@app.cell
def _(Q_MAX_CUT, T_MAX_CUT, exp_df, pd, plt):
    # ── Localisation by cluster and run ───────────────────────────────────
    def _rates(d):
        return pd.Series({"n": len(d),
                          "tmax%": 100 * (d.t_max > T_MAX_CUT).mean(),
                          "qmax%": 100 * (d.q_max > Q_MAX_CUT).mean()})

    by_cl = exp_df.groupby("cluster").apply(_rates, include_groups=False)
    by_run = exp_df.groupby(["cluster", "run"]).apply(_rates, include_groups=False)
    by_run = by_run[by_run.n >= 30].sort_values("tmax%", ascending=False)
    print("=== anomaly rate by cluster ===")
    print(by_cl.round(1).to_string())
    print(f"\n=== top runs by t_max>{T_MAX_CUT:g}% (n>=30) ===")
    print(by_run.head(10).round(1).to_string())

    fig_loc, axes_loc = plt.subplots(1, 2, figsize=(13, 4))
    axes_loc[0].bar(by_cl.index.astype(str), by_cl["tmax%"], color="purple")
    axes_loc[0].set_title(f"% t_max>{T_MAX_CUT:g} per cluster"); axes_loc[0].set_xlabel("cluster")
    axes_loc[1].bar(by_cl.index.astype(str), by_cl["qmax%"], color="darkorange")
    axes_loc[1].set_title(f"% q_max>{Q_MAX_CUT:g} per cluster"); axes_loc[1].set_xlabel("cluster")
    for a in axes_loc:
        a.set_ylabel("% of exp-hi")
    fig_loc.tight_layout()
    fig_loc
    return


@app.cell
def _(T_MAX_CUT, exp_df, np, plt):
    # ── Filter: t_max < T_MAX_CUT — and its effect on the exp-hi sample ────
    bad = exp_df.t_max > T_MAX_CUT
    print(f"flagged t_max>{T_MAX_CUT:g}: {int(bad.sum()):,} / {len(exp_df):,} "
          f"({100 * bad.mean():.1f}% of exp-hi)")
    print("concentration by (cluster, run):")
    print(exp_df[bad].groupby(["cluster", "run"]).size()
          .sort_values(ascending=False).head(8).to_string())

    bins_f = np.linspace(0.8, 1.0, 41)
    fig_flt, ax_flt = plt.subplots(figsize=(8, 4.5))
    ax_flt.hist(exp_df.score, bins=bins_f, histtype="step", lw=2, color="grey",
                label=f"all exp-hi ({len(exp_df):,})")
    ax_flt.hist(exp_df[~bad].score, bins=bins_f, histtype="step", lw=2, color="steelblue",
                label=f"after t_max<{T_MAX_CUT:g} ({int((~bad).sum()):,})")
    ax_flt.set_xlabel("nu-classifier score"); ax_flt.set_ylabel("events")
    ax_flt.set_title(f"exp-hi score: effect of t_max<{T_MAX_CUT:g} filter")
    ax_flt.legend()
    fig_flt.tight_layout()
    fig_flt
    return


@app.cell
def _(mo):
    mo.md("""
    ## Filter recipe

    - **Late-hit anomaly** (`t_max > 1500`): cleanly localised to **cluster 2,
      runs 20 & 249** (≈0 % elsewhere, 0 % in MC). Either an event-level cut
      `t_max < 1500` or excluding those two runs removes it.
    - **High charge** (`q_max > 100`): the extreme part overlaps the same c2
      runs; the rest is a softer ~20 %-vs-8 % baseline across all clusters and
      is **already clipped at 100 PE inside the model**, so it does not need a
      hard data cut — it shows up in the charge-*shape* features instead.
    """)
    return


if __name__ == "__main__":
    app.run()
