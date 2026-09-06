"""Event displays for the accepted populations: what the classifier actually takes.

The discriminating projection is depth against arrival time. Light from a down-going muon
sweeps downward, so depth falls with time; from an up-going neutrino it rises. The events the
classifier accepts have no consistent sweep at all, which is why their direction is
ill-determined -- and that is visible by eye.

For simulated muons the true zenith is known and printed on the panel, so a down-going muon
that looks up-going can be seen directly.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = ROOT / "inference_v2/nu_classifier/preds" / MODEL
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
H5 = {"mc": ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5",
      "exp": ROOT / "data_manager/data/h5datasets/exp_full.h5"}
PROBS = {"mc": ROOT / ("data_manager/data/h5datasets/baikal_mc_merged_probs_"
                       "k_nsol_labelneq0_da_hs128_k0p0001.h5"),
         "exp": ROOT / ("data_manager/data/h5datasets/exp_full_probs_"
                        "k_nsol_labelneq0_da_hs128_k0p0001.h5")}
SN_THR, STRING_DIVISOR = 0.8, 36
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
STRING_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
                  "#c9268a", "#9a8b1f", "#1f9a8b", "#8b1f2a"]


def pick(source: str, h5_group: str, where: str, part_filter: str, limit: int):
    db = "mc_merged_thr0p8.duckdb" if source == "mc" else "exp_full_thr0p8.duckdb"
    con = duckdb.connect(str(PREDS / db), read_only=True)
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    truth_join = ("LEFT JOIN truth t USING (event_fk)" if source == "mc" else "")
    truth_col = ("t.zenith_deg" if source == "mc" else "NULL")
    rows = con.execute(f"""
        SELECT l.part_key, l.local_idx, p.score, {truth_col} AS true_zenith
        FROM predictions p JOIN splits s USING (event_fk)
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk {truth_join}
        WHERE {where} AND l.part_key {part_filter}
        ORDER BY hash(p.event_fk) LIMIT {limit}""").fetchall()
    con.close()
    return [(h5_group, *r) for r in rows]


def read_hits(source: str, h5_group: str, part: str, idx: int):
    with h5py.File(H5[source], "r") as f, h5py.File(PROBS[source], "r") as pf:
        starts = f[f"{h5_group}/raw/ev_starts/{part}/data"][idx:idx + 2].astype(np.int64)
        a, b = int(starts[0]), int(starts[1])
        data = f[f"{h5_group}/raw/data/{part}/data"][a:b].astype(np.float32)
        chan = f[f"{h5_group}/raw/channels/{part}/data"][a:b].astype(np.int32)
        prob = pf[f"{h5_group}/probs/{part}/data"][a:b].astype(np.float32)
    m = prob > SN_THR
    return data[m], chan[m]


def draw(ax, hits, chans, title, subtitle):
    t = hits[:, 1] - hits[:, 1].min()
    z, q = hits[:, 4], np.clip(hits[:, 0], 0, 100)
    strings = chans // STRING_DIVISOR
    for k, s in enumerate(np.unique(strings)):
        m = strings == s
        ax.scatter(t[m], z[m], s=12 + 55 * np.sqrt(q[m] / q.max()),
                   color=STRING_COLOURS[k % len(STRING_COLOURS)], alpha=0.85,
                   edgecolors="none", zorder=3)
    if len(t) > 2 and np.ptp(t) > 0:                       # общий тренд глубины со временем
        k = np.polyfit(t, z, 1)
        xs = np.array([t.min(), t.max()])
        ax.plot(xs, np.polyval(k, xs), color=MUTED, lw=1.4, ls="--", zorder=2)
        arrow = "свет идёт ВНИЗ" if k[0] < 0 else "свет идёт ВВЕРХ"
    else:
        arrow = ""
    ax.set_facecolor(SURFACE); ax.grid(color="#e5e4e0", lw=0.7); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.set_title(title, fontsize=9.5, fontweight="bold", color=INK, pad=4)
    ax.text(0.03, 0.03, subtitle + ("\n" + arrow if arrow else ""), transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="bottom",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1.5))


if __name__ == "__main__":
    quality = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3"
    mc_ok = f"{quality} AND NOT s.used_for_labels"
    exp_ok = f"{quality} AND NOT s.excluded"
    groups = [
        ("данные, принято (скор > 0.8)", "exp", "exp_full",
         f"{exp_ok} AND p.score > 0.8", "= 'part_s2020_c05_r0022'", 5),
        ("моделирование, мюон принят (скор > 0.8)", "mc", "muatm_2020",
         f"{mc_ok} AND s.data_class = 'muatm_2020' AND p.score > 0.8",
         "IN ('part_13232','part_11454','part_12428','part_38325','part_14027')", 5),
        ("моделирование, мюон отвергнут — как выглядит настоящий трек", "mc", "muatm_2020",
         f"{mc_ok} AND s.data_class = 'muatm_2020' AND p.score < 0.01",
         "= 'part_13232'", 5),
        ("моделирование, атм. нейтрино принято — как выглядит цель", "mc", "nuatm_2020",
         f"{mc_ok} AND s.data_class = 'nuatm_2020' AND p.score > 0.99",
         "IS NOT NULL", 5),
    ]
    fig, axes = plt.subplots(len(groups), 5, figsize=(19.5, 13.0))
    fig.patch.set_facecolor(SURFACE)
    for row, (label, source, h5g, where, pf, n) in enumerate(groups):
        chosen = pick(source, h5g, where, pf, n)
        print(f"{label}: {len(chosen)} событий")
        for col in range(5):
            ax = axes[row, col]
            if col >= len(chosen):
                ax.axis("off"); continue
            g, part, idx, score, true_z = chosen[col]
            hits, chans = read_hits(source, g, part, idx)
            sub = f"скор {score:.3f}, хитов {len(hits)}"
            if true_z is not None and not pd.isna(true_z):
                sub += f"\nистинный зенит {true_z:.0f}°"
            draw(ax, hits, chans, label if col == 0 else "", sub)
            if col == 0:
                ax.set_ylabel("глубина z, м", color=MUTED, fontsize=9)
            if row == len(groups) - 1:
                ax.set_xlabel("время от первого хита, нс", color=MUTED, fontsize=9)
    fig.suptitle("События глазами: глубина против времени. Размер точки — заряд, "
                 "цвет — струна, пунктир — общий ход глубины",
                 x=0.006, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    out = HERE / "figures/event_display.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nsaved: {out}")
