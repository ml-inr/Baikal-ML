"""Quick test: run sig-noise model on a small MC sample and plot prob histogram
plus feature distributions before/after normalization."""

from pathlib import Path
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#from sig_noise_model_v2 import load_model, predict_flat, MODEL_NORM_MEAN, MODEL_NORM_STD
from sig_noise_model_v3 import load_model, predict_flat, MODEL_NORM_MEAN, MODEL_NORM_STD



H5_PATH = Path(__file__).resolve().parents[2] / "data_manager/data/h5datasets/baikal_mc_merged.h5"
OUT_PATH = Path(__file__).resolve().parent / "test_prob_hist.png"

N_EVENTS = 5000  # events per particle type

def load_sample(h5, ptype, part="part_1100", n_events=N_EVENTS):
    ev_starts = h5[f"{ptype}/raw/ev_starts/{part}/data"][:n_events + 1].astype("int64")
    data_raw  = h5[f"{ptype}/raw/data/{part}/data"][:int(ev_starts[-1])].astype("float32")
    gt_labels = h5[f"{ptype}/raw/labels/{part}/data"][:int(ev_starts[-1])]
    return data_raw, ev_starts, (gt_labels != 0)

model, _, device = load_model(device="auto")
# MODEL_DIR = Path(__file__).resolve().parent
# DEFAULT_CHECKPOINT = MODEL_DIR / "best_mc_2020.ckpt"
# state_dict = torch.load(str(DEFAULT_CHECKPOINT), map_location=device, weights_only=True)
print(f"Model on {device}")

FEATURE_NAMES = ["amplitude", "time", "x", "y", "z"]

results = {}
with h5py.File(str(H5_PATH), "r") as h5:
    for ptype in ["muatm_2020", "nuatm_2020", "nue2_2020"]:
        data, ev_starts, gt_sig = load_sample(h5, ptype)
        n_ev = len(ev_starts) - 1
        prob = predict_flat(model, data, ev_starts, batch_size=128, device=device, normalize=True)
        data_norm = (data - MODEL_NORM_MEAN) / MODEL_NORM_STD
        results[ptype] = {"prob": prob, "gt_sig": gt_sig, "n_events": n_ev,
                          "data_raw": data, "data_norm": data_norm}
        print(f"{ptype}: {n_ev} events, {len(prob)} hits | "
              f"prob min={prob.min():.4f} max={prob.max():.4f} mean={prob.mean():.4f} | "
              f"GT signal fraction: {gt_sig.mean():.4f}")

colors = {"muatm_2020": "tab:blue", "nuatm_2020": "tab:orange", "nue2_2020": "tab:green"}

# ── Figure 1: prob histograms ─────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
#bins_prob = np.linspace(0, 1, 51)
for ax, (ptype, res) in zip(axes, results.items()):
    prob   = res["prob"]
    gt_sig = res["gt_sig"]
    ax.hist(prob[~gt_sig], bins=100, alpha=0.6, label="GT noise",   color="gray",          density=True)
    ax.hist(prob[ gt_sig], bins=100, alpha=0.7, label="GT signal",  color=colors[ptype],   density=True)
    ax.axvline(0.5, color="red", ls="--", lw=1.2, label="thr=0.5")
    ax.set_title(ptype)
    ax.set_xlabel("sig prob")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
fig1.suptitle(f"Sig-noise model: per-hit probability histograms (N≈{N_EVENTS} events/type)")
fig1.tight_layout()
out1 = OUT_PATH
fig1.savefig(str(out1), dpi=120)
print(f"Saved: {out1}")

# ── Figure 2: feature distributions raw vs normed ────────────────────
n_features = len(FEATURE_NAMES)
fig2, axes2 = plt.subplots(n_features, 2, figsize=(12, 3 * n_features))

# Use nue2 as representative sample
ptype_ref = "nue2_2020"
res_ref = results[ptype_ref]
data_raw  = res_ref["data_raw"]
data_norm = res_ref["data_norm"]

for fi, fname in enumerate(FEATURE_NAMES):
    raw_vals  = data_raw[:, fi]
    norm_vals = data_norm[:, fi]

    # clip outliers for display (1st–99th percentile)
    r_lo, r_hi = np.percentile(raw_vals,  [0.5, 99.5])
    n_lo, n_hi = np.percentile(norm_vals, [0.5, 99.5])

    ax_r = axes2[fi, 0]
    ax_n = axes2[fi, 1]

    ax_r.hist(raw_vals.clip(r_lo, r_hi),  bins=80, color="steelblue", alpha=0.8, density=True)
    ax_r.set_title(f"{fname} — raw")
    ax_r.set_xlabel("value")
    ax_r.axvline(MODEL_NORM_MEAN[fi], color="red", ls="--", lw=1.2,
                 label=f"norm_mean={MODEL_NORM_MEAN[fi]:.3g}")
    ax_r.legend(fontsize=7)

    ax_n.hist(norm_vals.clip(n_lo, n_hi), bins=80, color="darkorange", alpha=0.8, density=True)
    ax_n.set_title(f"{fname} — normalized")
    ax_n.set_xlabel("value")
    ax_n.axvline(0, color="red", ls="--", lw=1.2, label="0")
    ax_n.legend(fontsize=7)

fig2.suptitle(f"Feature distributions ({ptype_ref}, part_1054, N≈{N_EVENTS} events)")
fig2.tight_layout()
out2 = OUT_PATH.with_name("test_feature_hist.png")
fig2.savefig(str(out2), dpi=120)
print(f"Saved: {out2}")
