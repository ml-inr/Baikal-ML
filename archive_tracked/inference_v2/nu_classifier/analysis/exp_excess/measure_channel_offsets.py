"""Per-channel hit-time offsets, measured identically on MC and experimental data.

The question this answers: are the per-channel offsets seen in experimental data an
instrumental defect, or geometry? The decisive comparison is against MC, which has no
calibration errors by construction — so anything MC reproduces cannot be a calibration fault.

Result (2026-08-25, DATA_QUALITY.md §7): 97-99% of the per-channel spread is a monotone
function of the module's position in its string, the same curve in MC and data (r = 0.989 to
0.998). The residual left for the instrument is 9-11 ns, about 2 m of light travel.

Usage:
    python inference_v2/nu_classifier/analysis/exp_excess/measure_channel_offsets.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MC    = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
MC_P  = ROOT / ("data_manager/data/h5datasets/"
                "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")
EXP   = ROOT / "data_manager/data/h5datasets/exp_full.h5"
EXP_P = ROOT / ("data_manager/data/h5datasets/"
                "exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")

THRESHOLD = 0.8            # the sig-noise cut the whole pipeline uses
MIN_HITS_PER_CHANNEL = 200
MODULES_PER_STRING = 36    # STRING_DIVISOR, as in nu_classifier_ds_builder/io.py
EXCLUDED = ("r0020", "r0249")   # the two verified-bad c02 runs, config.yaml


def per_channel(h5: Path, probs: Path, group: str, parts: list[str],
                max_events: int = 200_000) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean of (t - median t of the event) per channel, over signal hits."""
    total: dict[int, float] = defaultdict(float)
    count: dict[int, int] = defaultdict(int)
    seen = 0
    with h5py.File(h5, "r") as src, h5py.File(probs, "r") as pf:
        g = src[group]
        for part in parts:
            ev = g[f"raw/ev_starts/{part}/data"][:].astype(np.int64)
            t  = g[f"raw/data/{part}/data"][:, 1].astype(np.float64)
            ch = g[f"raw/channels/{part}/data"][:].astype(np.int32)
            pr = pf[f"{group}/probs/{part}/data"][:].astype(np.float32)
            for i in range(len(ev) - 1):
                s, e = int(ev[i]), int(ev[i + 1])
                m = pr[s:e] > THRESHOLD
                if m.sum() < 5:
                    continue
                te, ce = t[s:e][m], ch[s:e][m]
                for c, v in zip(ce, te - np.median(te)):
                    total[int(c)] += float(v)
                    count[int(c)] += 1
                seen += 1
                if seen >= max_events:
                    break
            if seen >= max_events:
                break
    chans = np.array([c for c in total if count[c] >= MIN_HITS_PER_CHANNEL])
    return chans, np.array([total[c] / count[c] for c in chans]), seen


def profile(chans: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Mean offset as a function of module position in string."""
    pos = chans % MODULES_PER_STRING
    return np.array([means[pos == k].mean() if (pos == k).any() else np.nan
                     for k in range(MODULES_PER_STRING)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-parts", type=int, default=6)
    ap.add_argument("--exp-parts-per-cluster", type=int, default=3)
    a = ap.parse_args()

    with h5py.File(MC_P, "r") as f:
        mc_parts = sorted(f["muatm_2020/probs"].keys())[:a.mc_parts]
    mc_ch, mc_mean, mc_seen = per_channel(MC, MC_P, "muatm_2020", mc_parts)
    mc_prof = profile(mc_ch, mc_mean)
    mc_resid = mc_mean - mc_prof[mc_ch % MODULES_PER_STRING]
    print(f"MC muatm: {mc_seen:,} events, {len(mc_ch)} channels — "
          f"raw std {mc_mean.std():.1f} ns, residual {np.nanstd(mc_resid):.1f} ns")

    with h5py.File(EXP_P, "r") as f:
        all_parts = sorted(f["exp_full/probs"].keys())
    print(f"\n{'cluster':>8} {'channels':>9} {'raw std':>9} {'residual':>9} "
          f"{'explained':>10} {'r with MC':>10}")
    for cluster in ("c02", "c03", "c04", "c05", "c06", "c07"):
        parts = [p for p in all_parts
                 if f"_{cluster}_" in p and not any(r in p for r in EXCLUDED)]
        parts = parts[:a.exp_parts_per_cluster]
        if not parts:
            continue
        ch, mean, _ = per_channel(EXP, EXP_P, "exp_full", parts, max_events=120_000)
        if len(ch) < 50:
            print(f"{cluster:>8}   too little statistics")
            continue
        prof = profile(ch, mean)
        resid = mean - prof[ch % MODULES_PER_STRING]
        ok = ~np.isnan(prof) & ~np.isnan(mc_prof)
        print(f"{cluster:>8} {len(ch):>9} {mean.std():>9.1f} {np.nanstd(resid):>9.1f} "
              f"{100 * (1 - np.nanvar(resid) / mean.var()):>9.0f}% "
              f"{np.corrcoef(prof[ok], mc_prof[ok])[0, 1]:>+10.3f}")


if __name__ == "__main__":
    main()
