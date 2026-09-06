import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Afterpulse augmentation check

    Runs the **real training collate** (`nu_classifier_collate_fn`) on the MC
    source dataset with the same afterpulse config as E3/E4 (per-event
    sequential 1..3 injections, Q~U(5,100), random OM, random time), then:

    - identifies the injected hits by length diff (afterpulse-only aug, no
      rotation/noise/normalization → extra hits are appended at the end);
    - **3D-visualises ~10 events** (bubble size = charge Q, colour = time),
      afterpulses marked with a red star;
    - checks that afterpulse **charge** and **time** are actually random and
      that P(1)>P(2)>P(3).

    Expectation at p=0.25: ~2–3 of 10 events carry an afterpulse.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)

    _root = Path(__file__).resolve().parents[4]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.data.nu_classifier_dataset.dataset import NuClassifierNpyDataset
    from src.data.nu_classifier_dataset.collate import nu_classifier_collate_fn

    return NuClassifierNpyDataset, Path, mo, np, nu_classifier_collate_fn, plt


@app.cell
def _(Path):
    # ── Config (mirror E3/E4 afterpulse) ──────────────────────────────────
    ROOT_DIR = Path(__file__).resolve().parents[4]
    SOURCE_NPY_DIR = ROOT_DIR / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
    AP_PROB = 0.25              # E3b rate; try 0.02 for E3a
    AP_MAX = 3
    Q_RANGE = [5.0, 100.0]
    MAX_HITS = 500
    N_SAMPLE = 400             # events pulled for the display pool
    N_VIZ = 10                 # events shown in the 3D grid
    N_AGG = 5000               # events for the randomness histograms
    SEED = 1
    _ = Path
    return AP_MAX, AP_PROB, N_AGG, N_SAMPLE, Q_RANGE, SEED, SOURCE_NPY_DIR


@app.cell
def _(NuClassifierNpyDataset, Path, SEED, SOURCE_NPY_DIR, np):
    # ── OM-position pool (mirrors trainer._build_om_pool) + source dataset ─
    _feats = np.load(Path(SOURCE_NPY_DIR) / "features.npy", mmap_mode="r")
    _rng = np.random.default_rng(SEED)
    _idx = np.sort(_rng.choice(len(_feats), min(1_000_000, len(_feats)), replace=False))
    om_pool = np.unique(np.round(np.asarray(_feats[_idx, 2:5], np.float32) / 2.0) * 2.0,
                        axis=0).astype(np.float32)
    print(f"OM pool: {len(om_pool):,} unique positions")

    ds = NuClassifierNpyDataset(npy_dir=SOURCE_NPY_DIR, max_hits=500, max_events=1_000_000, seed=SEED)
    print(f"source dataset: {len(ds):,} events")
    return ds, om_pool


@app.cell
def _(AP_MAX, AP_PROB, Q_RANGE, ds, np, nu_classifier_collate_fn, om_pool):
    # ── Helper: inject afterpulses via the REAL collate, keep event order ──
    def inject(indices, seed):
        items = [ds[int(i)] for i in indices]
        orig = [int(it["features"].shape[0]) for it in items]
        aug = {"afterpulse": {"enabled": True, "prob": AP_PROB, "max_afterpulses": AP_MAX,
                              "q_range": Q_RANGE, "om_pool": om_pool}}
        np.random.seed(seed)  # collate's afterpulse draw uses global numpy RNG
        out = nu_classifier_collate_fn(items, normalization_config=None,
                                       augmentation_config=aug, shuffle_batch=False,
                                       device="cpu")
        feats = out["features"].numpy()        # (B, max_len, 5) RAW units (no norm)
        lens  = out["lengths"].numpy()
        evs = []
        for k in range(len(items)):
            n0, n1 = orig[k], int(lens[k])
            evs.append({"hits": feats[k, :n1], "n_real": n0, "n_ap": n1 - n0})
        return evs

    return (inject,)


@app.cell
def _(N_SAMPLE, SEED, inject, np):
    # ── Sample events; report afterpulse fraction; pick a display set ─────
    rng = np.random.default_rng(SEED)
    samp_idx = rng.choice(1_000_000, N_SAMPLE, replace=False)
    events = inject(samp_idx, seed=SEED)
    n_with = sum(e["n_ap"] > 0 for e in events)
    print(f"{n_with}/{N_SAMPLE} events got an afterpulse "
          f"({100*n_with/N_SAMPLE:.1f}%, expect ~{100*0.25:.0f}% for p=0.25)")

    ap_events = [e for e in events if e["n_ap"] > 0]
    no_events = [e for e in events if e["n_ap"] == 0]
    # display 10: a few with afterpulses + rest clean, interleaved
    show = (ap_events[:3] + no_events[:7])[:10]
    rng.shuffle(show)
    print(f"display set: {sum(e['n_ap']>0 for e in show)} with afterpulse, "
          f"{sum(e['n_ap']==0 for e in show)} clean")
    return events, show


@app.cell
def _(np, om_pool, plt, show):
    # ── 3D grid: fired OMs (size=Q, colour=time), afterpulses = red star ──
    # Body wrapped in a local function so loop vars stay cell-local (marimo).
    def _grid():
        def _size(q):
            return 15 + np.clip(q, 0, 150) * 1.6
        fig = plt.figure(figsize=(22, 9))
        for k, e in enumerate(show):
            ax = fig.add_subplot(2, 5, k + 1, projection="3d")
            ax.scatter(om_pool[:, 0], om_pool[:, 1], om_pool[:, 2],
                       s=1.5, c="lightgrey", alpha=0.08)           # cluster backdrop
            h = e["hits"]; n0 = e["n_real"]; real = h[:n0]
            ax.scatter(real[:, 2], real[:, 3], real[:, 4],
                       s=_size(real[:, 0]), c=real[:, 1], cmap="viridis",
                       alpha=0.9, edgecolors="k", linewidths=0.2)
            if e["n_ap"] > 0:
                ap = h[n0:]
                ax.scatter(ap[:, 2], ap[:, 3], ap[:, 4],
                           s=_size(ap[:, 0]) + 40, c=ap[:, 1], cmap="viridis",
                           marker="*", edgecolors="red", linewidths=1.8,
                           vmin=real[:, 1].min(), vmax=real[:, 1].max())
            ax.set_title(f"ev{k}  nh={n0}  ap={e['n_ap']}",
                         color="red" if e["n_ap"] else "black", fontsize=9)
            ax.tick_params(labelsize=5)
        fig.suptitle("Events from the training loader (★ red = injected afterpulse; "
                     "size∝Q, colour=time)", y=1.0)
        fig.tight_layout()
        return fig
    fig_g = _grid()
    fig_g
    return


@app.cell
def _(events):
    # ── Detail: print the afterpulse hits of the first afterpulse event ───
    def _detail():
        e = next((x for x in events if x["n_ap"] > 0), None)
        if e is None:
            print("no afterpulse event in the sample — rerun / raise N_SAMPLE")
            return
        h = e["hits"]; n0 = e["n_real"]; real = h[:n0]; ap = h[n0:]
        print(f"event with {e['n_ap']} afterpulse(s), {n0} real hits")
        print(f"  real-hit time range: [{real[:,1].min():.0f}, {real[:,1].max():.0f}] ns")
        print(f"  real-hit Q range:    [{real[:,0].min():.1f}, {real[:,0].max():.1f}] p.e.")
        for j, a in enumerate(ap):
            print(f"  afterpulse #{j+1}: Q={a[0]:.1f} p.e. (∈[5,100]?),  "
                  f"t={a[1]:.0f} ns (∈ event range?),  xyz=({a[2]:.1f},{a[3]:.1f},{a[4]:.1f})")
    _detail()
    return


@app.cell
def _(N_AGG, inject, np, plt):
    # ── Randomness check over many events ─────────────────────────────────
    def _agg_plot():
        import collections as _c
        rng2 = np.random.default_rng(123)
        agg = inject(rng2.choice(1_000_000, N_AGG, replace=False), seed=123)
        ap_q, ap_reltime, n_ap_list = [], [], []
        for e in agg:
            n_ap_list.append(e["n_ap"])
            if e["n_ap"] > 0:
                h = e["hits"]; n0 = e["n_real"]
                t = h[:n0, 1]; tlo, thi = float(t.min()), float(t.max())
                for a in h[n0:]:
                    ap_q.append(float(a[0]))
                    ap_reltime.append((float(a[1]) - tlo) / (thi - tlo) if thi > tlo else 0.5)
        cnt = _c.Counter(n_ap_list)
        fig, axr = plt.subplots(1, 3, figsize=(16, 4))
        axr[0].hist(np.array(ap_q), bins=30, color="steelblue", edgecolor="white")
        axr[0].set_title(f"Afterpulse Q  (expect U[5,100])\nn={len(ap_q):,}")
        axr[0].set_xlabel("Q (p.e.)")
        axr[1].hist(np.array(ap_reltime), bins=30, color="seagreen", edgecolor="white")
        axr[1].set_title("Afterpulse relative time (t−tmin)/(tmax−tmin)\n(expect U[0,1])")
        axr[1].set_xlabel("relative time")
        ks = [0, 1, 2, 3]
        vals = [100 * cnt.get(k, 0) / len(n_ap_list) for k in ks]
        axr[2].bar([str(k) for k in ks], vals, color="tomato")
        for kk, v in zip(ks, vals):
            axr[2].text(kk, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
        axr[2].set_title("n afterpulses / event  (P(1)>P(2)>P(3))")
        axr[2].set_xlabel("n afterpulses"); axr[2].set_ylabel("% events")
        fig.tight_layout()
        return fig
    fig_r = _agg_plot()
    fig_r
    return


if __name__ == "__main__":
    app.run()
