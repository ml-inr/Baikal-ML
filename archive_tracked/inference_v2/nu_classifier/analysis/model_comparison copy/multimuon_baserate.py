#!/usr/bin/env python
"""Is muon multiplicity a useful handle on the classifier's false positives?

Report 2026-07-09 §4 stated "79% of high-score muatm are multi-muon (n_muons>=2)" and
recommended an anti-multi-muon cut plus a multi-task n_muons head. That number was quoted
without the base rate, and the sample it came from was itself drawn with score>0.5. This
script measures the base rates properly, in the population the classifier actually sees.

Three populations of muatm, MC truth n_muons = prime_prty[:, 4]:
  (a) unconditional        — random events, no selection at all;
  (b) h8s3, any score      — what the classifier is trained and evaluated on
                             (n_sn_hits>=8, n_sn_strings>=3);
  (c) h8s3, score>0.8      — the false positives we want to fix.

The enrichment that matters is (c)/(b): (c)/(a) is confounded because the h8s3 cut itself
selects bright, high-multiplicity events.

Outputs tables/multimuon_baserate.csv. No GPU.
"""
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
CAT = ROOT / "data_manager/catalog_v2.duckdb"
PRED = ROOT / ("inference_v2/nu_classifier/preds/"
               "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010")
RDCC = dict(rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)
N_PARTS_UNCOND = 12
N_H8S3 = 40000
N_HIGH = 20000


def unconditional(f) -> np.ndarray:
    """n_muons for whole random parts — no selection whatsoever."""
    parts = sorted(f["muatm_2020"]["raw"]["data"].keys())
    rng = np.random.default_rng(0)
    sel = rng.choice(len(parts), N_PARTS_UNCOND, replace=False)
    return np.concatenate([f[f"muatm_2020/prime_prty/{parts[i]}/data"][:, 4] for i in sel])


def from_db(f, extra: str, n: int) -> np.ndarray:
    """n_muons for h8s3 muatm events sampled from the prediction DB."""
    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PRED / 'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    df = c.execute(f"""SELECT pr.score, l.part_key, l.local_idx
        FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE pr.n_sn_hits>=8 AND pr.n_sn_strings>=3 AND ev.data_class='muatm_2020' {extra}
        ORDER BY random() LIMIT {n}""").df()
    c.close()

    out = np.full(len(df), np.nan)
    for pk, idx in df.groupby("part_key").groups.items():
        pp = f[f"muatm_2020/prime_prty/{pk}/data"][:, 4]
        loc = df.loc[idx, "local_idx"].to_numpy()
        ok = loc < len(pp)
        out[np.array(idx)[ok]] = pp[loc[ok]]
    return out[~np.isnan(out)]


with h5py.File(MCH5, "r", **RDCC) as f:
    pops = {
        "(a) unconditional": unconditional(f),
        "(b) h8s3, any score": from_db(f, "", N_H8S3),
        "(c) h8s3, score>0.8": from_db(f, "AND pr.score>0.8", N_HIGH),
    }

rows = []
for name, nm in pops.items():
    rows.append(dict(population=name, n=len(nm), frac_ge2=float((nm >= 2).mean()),
                     frac_ge3=float((nm >= 3).mean()), median=float(np.median(nm))))
tab = pd.DataFrame(rows)
tab.to_csv(HERE / "tables/multimuon_baserate.csv", index=False)
print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

b = tab[tab.population == "(b) h8s3, any score"].iloc[0]
c = tab[tab.population == "(c) h8s3, score>0.8"].iloc[0]
print(f"\nenrichment of the false positives vs the population the classifier sees:")
print(f"  n_muons>=2 : {c.frac_ge2 / b.frac_ge2:.2f}x")
print(f"  n_muons>=3 : {c.frac_ge3 / b.frac_ge3:.2f}x")
print(f"  median n_muons {c['median']:.0f} vs {b['median']:.0f}")
print("\n<1 means the false positives are the LOW-multiplicity tail, not multi-muon bundles.")
