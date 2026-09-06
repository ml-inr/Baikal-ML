"""Excess versus classifier threshold, as measured and with repeated-hit events removed.

"Repeated-hit events removed" means dropping the WHOLE event when any module fired twice
(n_sig_hits > n_channels), applied identically to both samples.

This is not the same operation as the input ablation, where repeated hits were removed *inside*
an event while the event stayed in the sample. There the network's input changed at fixed
selection; here the selection itself changes, and the effect has the opposite sign. Both are
legitimate and answer different questions.

Weights: groups 6 and 8 (accepted at 0.8) are exhaustive, groups 5 and 7 are quota samples, so
every event carries the weight that restores the full population.

Usage:
    python inference_v2/nu_classifier/analysis/excess_mechanism/excess_vs_threshold.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = HERE.parents[1] / "preds" / MODEL
FULL_SIZE = {5: 23_159_967, 6: 7_751, 7: 3_177_637, 8: 3_044}
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
BLUE, ORANGE, MUTED, INK, SURFACE = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#fcfcfb"


def load() -> pd.DataFrame:
    frame = duckdb.connect(str(HERE / "features.duckdb"), read_only=True).execute(
        "SELECT source, event_fk, group_id, n_hits, n_modules, hits_per_module "
        "FROM features").df()
    score = {}
    for src, db in (("mc", "mc_merged_thr0p8.duckdb"), ("exp", "exp_full_thr0p8.duckdb")):
        con = duckdb.connect(str(PREDS / db), read_only=True)
        con.register("_keys", pd.DataFrame({"event_fk": frame[frame.source == src].event_fk}))
        for fk, s in con.execute("SELECT p.event_fk, p.score FROM predictions p "
                                 "JOIN _keys USING (event_fk)").fetchall():
            score[(src, fk)] = s
        con.close()
    frame["score"] = [score.get((s, k), np.nan) for s, k in zip(frame.source, frame.event_fk)]
    frame["weight"] = frame.group_id.map(
        {k: FULL_SIZE[k] / (frame.group_id == k).sum() for k in FULL_SIZE})
    return frame


def check(frame: pd.DataFrame) -> None:
    """A silently missing score or a mis-read feature would corrupt every number below."""
    missing = int(frame.score.isna().sum())
    assert missing == 0, f"{missing} events have no score -- they would drop from the numerator"
    # hits_per_module = n_sig_hits / n_channels, so it cannot be below one
    assert (frame.hits_per_module >= 1.0 - 1e-9).all(), "hits_per_module < 1 -- wrong quantity"
    direct = frame.n_hits > frame.n_modules
    derived = frame.hits_per_module > 1.0 + 1e-9
    assert (direct == derived).all(), "feature disagrees with n_hits and n_modules"
    assert set(frame.group_id) == {1, 2, 3, 4, 5, 6, 7, 8}
    print(f"checks passed: {len(frame):,} events, no missing scores, "
          f"hits_per_module consistent with n_hits/n_modules")


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    mc = frame[frame.group_id.isin([5, 6])]
    exp = frame[frame.group_id.isin([7, 8])]
    clean_mc = (mc.n_hits == mc.n_modules).values        # no module fired twice
    clean_exp = (exp.n_hits == exp.n_modules).values
    print(f"events removed: simulation {100*(1-np.average(clean_mc, weights=mc.weight)):.1f}%, "
          f"experiment {100*(1-np.average(clean_exp, weights=exp.weight)):.1f}%")
    rows = []
    for xi in THRESHOLDS:
        acc_mc, acc_exp = (mc.score > xi).values, (exp.score > xi).values
        rate_mc = np.average(acc_mc, weights=mc.weight)
        rate_exp = np.average(acc_exp, weights=exp.weight)
        rate_mc_clean = np.average(acc_mc[clean_mc], weights=mc.weight.values[clean_mc])
        rate_exp_clean = np.average(acc_exp[clean_exp], weights=exp.weight.values[clean_exp])
        rows.append({"threshold": xi,
                     "accepted mc %": 100 * rate_mc, "accepted exp %": 100 * rate_exp,
                     "excess as measured": rate_exp / rate_mc,
                     "accepted mc, no repeats %": 100 * rate_mc_clean,
                     "accepted exp, no repeats %": 100 * rate_exp_clean,
                     "excess without repeats":
                         rate_exp_clean / rate_mc_clean if rate_mc_clean > 0 else np.nan})
    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.grid(color="#e5e4e0", lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        for side in ("left", "bottom"): ax.spines[side].set_color("#c9c8c3")
        ax.tick_params(colors=MUTED, length=0)
    ax = axes[0]
    ax.plot(table.threshold, table["excess as measured"], "o-", color=ORANGE, lw=2.4, ms=7,
            label="данные как есть")
    ax.plot(table.threshold, table["excess without repeats"], "s-", color=BLUE, lw=2.4, ms=7,
            label="события с повторными хитами удалены\nиз обеих выборок")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.2)
    ax.set_xlabel("порог классификатора ξ", color=MUTED)
    ax.set_ylabel("избыток: доля принятых exp / доля принятых MC", color=MUTED)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="upper left")
    ax.set_title("избыток в зависимости от порога", fontsize=12, fontweight="bold", color=INK)
    ax = axes[1]
    for column, colour, style, label in [
            ("accepted mc %", BLUE, "o-", "MC, как есть"),
            ("accepted exp %", ORANGE, "o-", "exp, как есть"),
            ("accepted mc, no repeats %", BLUE, "s--", "MC, без повторов"),
            ("accepted exp, no repeats %", ORANGE, "s--", "exp, без повторов")]:
        ax.plot(table.threshold, table[column], style, color=colour, lw=2,
                alpha=0.65 if "--" in style else 1.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("порог классификатора ξ", color=MUTED)
    ax.set_ylabel("доля принятых, %", color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    ax.set_title("сами доли принятых", fontsize=12, fontweight="bold", color=INK)
    fig.suptitle("Избыток до и после удаления событий с повторными хитами на модуле",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK, y=1.02)
    plt.tight_layout()
    out = HERE / "figures/excess_vs_threshold_norepeat.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return out


if __name__ == "__main__":
    events = load()
    check(events)
    table = build_table(events)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved: {make_figure(table)}")
