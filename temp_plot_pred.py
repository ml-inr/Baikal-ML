import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import duckdb
    import numpy as np
    import matplotlib.pyplot as plt

    return duckdb, np, plt


@app.cell
def _():
    DB  = "inference_v2/nu_classifier/preds/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model/exp_reco_thr0p8.duckdb"
    CAT = "data_manager/catalog_v2.duckdb"
    return CAT, DB


@app.cell
def _(CAT, DB, duckdb, np, plt):
    conn = duckdb.connect(DB, read_only=True)
    conn.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

    # Pull score + full metadata in one join
    df = conn.execute("""
        SELECT p.score, p.n_sn_hits, p.n_sn_strings,
               e.cluster, e.season, e.run AS run_num, e.event_id,
               l.part_key, l.h5_path, l.local_idx
        FROM predictions p
        JOIN cat.events       e ON e.id       = p.event_fk
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk
    """).df()

    conn.execute("DETACH cat")
    conn.close()

    print(f"Total events in DB:        {len(df):,}")

    # ── Blacklist: exclude cluster 4 (channel 71 broken across entire cluster) ──
    # channel 198 broken in part_s2020_c04_r0117 is a subset of cluster 4 → already covered
    df_bl = df[~df.cluster.isin([1,4])].copy()
    print(f"After blacklist (no c4):   {len(df_bl):,}  "
          f"(removed {len(df)-len(df_bl):,} cluster-4 events)")

    # ── Three cut levels ──────────────────────────────────────────────────
    cuts = {
        "No cuts":  df_bl,
        "SN ≥8-2":  df_bl[(df_bl.n_sn_hits >= 8)  & (df_bl.n_sn_strings >= 2)],
        "SN ≥10-3": df_bl[(df_bl.n_sn_hits >= 10) & (df_bl.n_sn_strings >= 3)],
    }

    for label, sub in cuts.items():
        hi = (sub.score > 0.8).sum()
        print(f"  {label:12s}: {len(sub):>9,} events  |  score>0.8: {hi:>7,}  ({100*hi/len(sub):.3f}%)")

    # ── Plot (log scale only) ─────────────────────────────────────────
    bins   = np.linspace(0, 1, 51)
    colors = ["steelblue", "tomato", "seagreen"]
    styles = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, sub), color, ls in zip(cuts.items(), colors, styles):
        ax.hist(sub.score, bins=bins, density=True,
                histtype="step", linewidth=2,
                color=color, linestyle=ls,
                label=f"{label}  (n={len(sub):,})")
    ax.axvline(0.8, color="black", lw=1.2, ls="--", alpha=0.6, label="thr=0.8")
    ax.set_yscale("log")
    ax.set_xlabel("nu-classifier score")
    ax.set_ylabel("density (log)")
    ax.set_title("Score distribution (log scale)")
    ax.legend(fontsize=8)

    plt.suptitle(
        "exp_reco  |  nu-classifier 260508_1724  |  blacklist: clusters 1,4 excluded",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    return df_bl, fig


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(df_bl):
    df_bl[df_bl['score']>0.98][['cluster', 'run_num']].value_counts()
    return


@app.cell
def _(df_bl):
    df_bl.columns
    return


if __name__ == "__main__":
    app.run()
