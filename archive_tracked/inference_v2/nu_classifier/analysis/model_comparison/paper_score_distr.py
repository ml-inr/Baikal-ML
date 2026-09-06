#!/usr/bin/env python
"""Paper figure for section 5.3: score distributions, base network vs fine-tuned.

Supersedes the score-distribution half of `paper_figs.py`, which did NOT apply the
exclusions the paper claims:
  * Monte Carlo events seen during training are removed via the NPY back-links
    (part_key|local_idx), and
  * the fine-tuning background sample is removed from the experimental data.

Both panels are drawn on the SAME events: the two models' prediction DBs are joined on
event_fk, so any difference between the panels is the network, not the sampling.

Outputs figures/paper_score_distr_base_vs_ft.png and prints the per-class counts that
belong in the figure caption.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "axes.titlesize": 18,
                     "legend.fontsize": 15, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "figure.dpi": 120})
PTYPE = {"muatm_2020": "steelblue", "nuatm_2020": "forestgreen",
         "nue2_2020": "darkorange", "exp_reco": "crimson"}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PREDS = ROOT / "inference_v2/nu_classifier/preds"
CAT = ROOT / "data_manager/catalog_v2.duckdb"
MC_TRAIN_NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
FT_BG_FKS_PATH = (ROOT / "inference_v2/nu_classifier/exp_finetuning/exp_bg_datasets/"
                  "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
                  "@da_checkpoint_epoch_010_ood2p5_rv0p7/exp_bg_event_fks.npy")
BASE = "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010"
FT = ("260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
      "@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model")
CUT = "{t}.n_sn_hits >= 8 AND {t}.n_sn_strings >= 3"
N_MUATM = 2_000_000


def main() -> None:
    train_pk = np.load(MC_TRAIN_NPY / "h5_part_keys.npy", allow_pickle=True).astype(str)
    train_li = np.load(MC_TRAIN_NPY / "h5_local_event_ids.npy").astype(np.int64)
    ft_fks = np.load(FT_BG_FKS_PATH).astype(np.int64)
    print(f"exclusions: MC training events {len(train_pk):,}, "
          f"fine-tuning background {len(ft_fks):,}", flush=True)

    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PREDS / BASE / 'mc_merged_thr0p8.duckdb'}' AS mb (READ_ONLY)")
    c.execute(f"ATTACH '{PREDS / FT / 'mc_merged_thr0p8.duckdb'}' AS mf (READ_ONLY)")
    c.execute(f"ATTACH '{PREDS / BASE / 'exp_full_thr0p8.duckdb'}' AS eb (READ_ONLY)")
    c.execute(f"ATTACH '{PREDS / FT / 'exp_full_thr0p8.duckdb'}' AS ef (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    c.execute("CREATE TEMP TABLE trainkey AS SELECT * FROM (SELECT UNNEST(?) AS k)",
              [[f"{a}|{b}" for a, b in zip(train_pk, train_li)]])
    c.execute("CREATE TEMP TABLE ftfk AS SELECT UNNEST(?) AS fk", [ft_fks.tolist()])

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls in ["muatm_2020", "nuatm_2020", "nue2_2020"]:
        samp = f"USING SAMPLE {N_MUATM}" if cls == "muatm_2020" else ""
        d = c.execute(f"""
            SELECT b.score AS sb, f.score AS sf
            FROM mb.predictions b
            JOIN mf.predictions f ON f.event_fk = b.event_fk
            JOIN cat.events ev ON ev.id = b.event_fk
            JOIN cat.h5_locations l ON l.event_fk = b.event_fk
            LEFT JOIN trainkey t ON t.k = l.part_key || '|' || CAST(l.local_idx AS VARCHAR)
            WHERE {CUT.format(t='b')} AND {CUT.format(t='f')}
              AND ev.data_class = '{cls}' AND t.k IS NULL
            {samp}""").df()
        out[cls] = (d.sb.to_numpy(), d.sf.to_numpy())
        print(f"  {cls:12s} out-of-training: {len(d):>9,}", flush=True)

    d = c.execute(f"""
        SELECT b.score AS sb, f.score AS sf
        FROM eb.predictions b
        JOIN ef.predictions f ON f.event_fk = b.event_fk
        JOIN cat.events ev ON ev.id = b.event_fk
        LEFT JOIN ftfk t ON t.fk = b.event_fk
        WHERE {CUT.format(t='b')} AND {CUT.format(t='f')}
          AND NOT (ev.cluster = 2 AND ev.run IN ('20', '249'))
          AND t.fk IS NULL""").df()
    out["exp"] = (d.sb.to_numpy(), d.sf.to_numpy())
    print(f"  {'experimental':12s} test set:        {len(d):>9,}", flush=True)
    c.close()

    style = [("muatm_2020", PTYPE["muatm_2020"], "MC EAS", "-"),
             ("nuatm_2020", PTYPE["nuatm_2020"], r"MC $\nu_\mu^{atm}$", "-"),
             ("nue2_2020", PTYPE["nue2_2020"], r"MC $\nu_\mu^{cosm}$", "-"),
             ("exp", PTYPE["exp_reco"], "Experimental", "--")]
    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for i, title in enumerate(["Base network", "After fine-tuning"]):
        for key, colour, label, ls in style:
            ax[i].hist(out[key][i], bins=bins, density=True, histtype="step",
                       lw=1.8, color=colour, ls=ls, label=label)
        ax[i].set_yscale("log")
        ax[i].set_xlabel(r"$\xi$")
        ax[i].set_title(title)
        ax[i].set_xlim(0, 1)
        ax[i].minorticks_on()
        ax[i].grid(True, which="major", alpha=0.35)
        ax[i].grid(True, which="minor", alpha=0.18, ls=":")
        ax[i].tick_params(which="major", length=6)
        ax[i].tick_params(which="minor", length=3)
    ax[0].set_ylabel("Density")
    ax[0].legend()
    fig.tight_layout()
    dst = HERE / "figures/paper_score_distr_base_vs_ft.png"
    fig.savefig(dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
