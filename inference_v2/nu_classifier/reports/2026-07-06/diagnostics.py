#!/usr/bin/env python
"""Regenerate all evidentiary tables/figures for the 2026-07-06 report.

Consolidates the ad-hoc queries run during the E1/E2/E3b post-training analysis
into reproducible artifacts under tables/ and figures/. Representative model:
E1 @ epoch 10 (baseline, out-of-training test preds). h8s3 = default topology cut
(n_sn_hits>=8 & n_sn_strings>=3), applied at analysis time via the DB columns.

Run:  python inference_v2/nu_classifier/reports/2026-07-06/diagnostics.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TAB = HERE / "tables"; TAB.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

CAT = ROOT / "data_manager/catalog_v2.duckdb"
PRED = ROOT / ("inference_v2/nu_classifier/preds/"
               "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010")
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
MC_NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"


def _con():
    c = duckdb.connect(); c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    return c


def t1_stat_validity():
    """excess + Poisson error per score threshold (h5s0, all events)."""
    c = _con()
    Nexp = c.execute("select count(*) from e.predictions").fetchone()[0]
    Nmu = c.execute("select count(*) from m.predictions p join cat.events ev "
                    "on ev.id=p.event_fk and ev.data_class='muatm_2020'").fetchone()[0]
    rows = []
    for t in [0.5, 0.8, 0.9, 0.95, 0.99]:
        ke = c.execute(f"select count(*) from e.predictions where score>{t}").fetchone()[0]
        km = c.execute("select count(*) from m.predictions p join cat.events ev "
                       f"on ev.id=p.event_fk and ev.data_class='muatm_2020' where score>{t}").fetchone()[0]
        fe, fm = ke/Nexp, km/Nmu
        ex = fe/fm if km else np.nan
        rel = np.sqrt(1/ke + 1/km) if (ke and km) else np.nan
        rows.append(dict(threshold=t, N_exp=Nexp, k_exp=ke, N_mu=Nmu, k_mu=km,
                         exp_frac=fe, mu_frac=fm, excess=ex, rel_err=rel,
                         sigma_from_1=((ex-1)/(ex*rel)) if (km and ex) else np.nan))
    c.close()
    df = pd.DataFrame(rows); df.to_csv(TAB/"excess_stat_validity.csv", index=False)
    print("wrote excess_stat_validity.csv"); return df


def t2_excess_vs_cut():
    """excess of exp>0.8 vs MC-muon under progressively tighter topology cuts."""
    c = _con(); SC = 0.8; rows = []
    for mh, ms in [(5, 0), (8, 0), (5, 3), (8, 3), (10, 3), (12, 4)]:
        cut = f"n_sn_hits>={mh} and n_sn_strings>={ms}"
        Ne, ke = c.execute(f"select count(*), count(*) filter(where score>{SC}) "
                           f"from e.predictions where {cut}").fetchone()
        Nm, km = c.execute("select count(*), count(*) filter(where score>%g) from m.predictions p "
                           "join cat.events ev on ev.id=p.event_fk and ev.data_class='muatm_2020' "
                           f"where {cut}" % SC).fetchone()
        ex = (ke/Ne)/(km/Nm) if (ke and km) else np.nan
        rel = np.sqrt(1/ke + 1/km) if (ke and km) else np.nan
        rows.append(dict(cut=f"h{mh}s{ms}", min_hits=mh, min_strings=ms,
                         exp_N=Ne, k_exp=ke, mu_N=Nm, k_mu=km, excess=ex, rel_err=rel))
    c.close()
    df = pd.DataFrame(rows); df.to_csv(TAB/"excess_vs_topology_cut.csv", index=False)
    print("wrote excess_vs_topology_cut.csv"); return df


def t3_topology_by_population():
    """nhits/nstrings of each score/class population under h8s3."""
    c = _con(); CUT = "n_sn_hits>=8 and n_sn_strings>=3"
    def desc(sql, label):
        df = c.execute(sql).df()
        h, s = df.n_sn_hits.to_numpy(), df.n_sn_strings.to_numpy()
        if not len(h): return dict(population=label, N=0)
        return dict(population=label, N=len(h), nhits_med=np.median(h), nhits_mean=round(h.mean(),1),
                    nhits_p90=np.percentile(h,90), nstr_med=np.median(s), nstr_mean=round(s.mean(),2),
                    frac_str_ge5=round(np.mean(s>=5),3))
    rows = [
        desc(f"select n_sn_hits,n_sn_strings from e.predictions where {CUT} and score>0.8", "exp score>0.8 (false nu)"),
        desc(f"select n_sn_hits,n_sn_strings from e.predictions where {CUT} and score<0.2", "exp score<0.2 (bg)"),
        desc(f"select n_sn_hits,n_sn_strings from e.predictions where {CUT}", "exp all"),
        desc(f"select p.n_sn_hits,p.n_sn_strings from m.predictions p join cat.events ev on ev.id=p.event_fk "
             f"and ev.data_class in ('nuatm_2020','nue2_2020') where {CUT} and score>0.8", "MC nu score>0.8 (real)"),
        desc(f"select p.n_sn_hits,p.n_sn_strings from m.predictions p join cat.events ev on ev.id=p.event_fk "
             f"and ev.data_class='muatm_2020' where {CUT} and score>0.8", "MC muon score>0.8 (false)"),
        desc(f"select p.n_sn_hits,p.n_sn_strings from m.predictions p join cat.events ev on ev.id=p.event_fk "
             f"and ev.data_class='muatm_2020' where {CUT}", "MC muon all"),
    ]
    c.close()
    df = pd.DataFrame(rows); df.to_csv(TAB/"topology_by_population_h8s3.csv", index=False)
    print("wrote topology_by_population_h8s3.csv"); return df


def t4_muatm_fp_vs_zenith():
    """KEY: MC muatm false-neutrino rate vs TRUE zenith (prime_prty[:,0]), h8s3."""
    c = _con()
    df = c.execute("""
        SELECT pr.score, l.part_key, l.local_idx
        FROM m.predictions pr
        JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020'
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE pr.n_sn_hits>=8 AND pr.n_sn_strings>=3
    """).df()
    c.close()
    theta = np.full(len(df), np.nan, np.float32)
    with h5py.File(MCH5, "r") as f:
        g = f["muatm_2020/prime_prty"]
        for pk, idx in df.groupby("part_key").groups.items():
            key = f"{pk}/data"
            if key in g:
                th = g[key][:, 0]; li = df.loc[idx, "local_idx"].to_numpy()
                ok = li < len(th); theta[np.array(idx)[ok]] = th[li[ok]]
    df["theta"] = theta; df = df[np.isfinite(df.theta)]
    edges = [90, 95, 105, 120, 150, 180]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = df[(df.theta >= a) & (df.theta < b)]
        if not len(s): continue
        k = int((s.score > 0.8).sum()); fr = k/len(s)
        rows.append(dict(theta_lo=a, theta_hi=b, N=len(s), k_hi=k, frac_gt0p8=fr,
                         err=np.sqrt(max(k,1))/len(s)))
    tab = pd.DataFrame(rows); tab.to_csv(TAB/"mc_muatm_fp_vs_zenith.csv", index=False)
    print("wrote mc_muatm_fp_vs_zenith.csv")
    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ctr = (tab.theta_lo + tab.theta_hi)/2
    ax.errorbar(ctr, tab.frac_gt0p8, yerr=tab.err, fmt="o-", lw=2, color="tab:red", capsize=4)
    ax.axvline(90, color="grey", ls=":", label="horizon (θ=90°)")
    ax.set_xlabel("true zenith θ (deg)  [muatm are down-going, θ>90]")
    ax.set_ylabel("frac(score>0.8) — false-neutrino rate")
    ax.set_title("MC muatm: false-neutrino rate concentrates at the horizon (h8s3)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"mc_muatm_fp_vs_zenith.png", dpi=130)
    print("wrote figures/mc_muatm_fp_vs_zenith.png"); return tab


def t5_feature_shift():
    """Per-feature normalized MC vs exp distribution shift (filtered hits, NPY)."""
    feats = ["amplitude", "time", "x", "y", "z"]
    nstd = [60., 1370., 40., 40., 150.]; nmean = [2., 0., 0., 0., 0.]
    def stats(path, n=3_000_000):
        f = np.load(path, mmap_mode="r")
        idx = np.linspace(0, len(f)-1, min(n, len(f))).astype(np.int64)
        x = np.asarray(f[idx], np.float32)
        return {feats[i]: (float((x[:,i].mean()-nmean[i])/nstd[i]),
                           float(x[:,i].std()/nstd[i])) for i in range(5)}
    mc = stats(MC_NPY/"features.npy"); ex = stats(EXP_NPY/"exp_features.npy")
    rows = [dict(feature=k, mc_norm_mean=round(mc[k][0],3), mc_norm_std=round(mc[k][1],3),
                 exp_norm_mean=round(ex[k][0],3), exp_norm_std=round(ex[k][1],3),
                 std_ratio_exp_mc=round(ex[k][1]/mc[k][1],2),
                 mean_shift=round(ex[k][0]-mc[k][0],3)) for k in feats]
    df = pd.DataFrame(rows); df.to_csv(TAB/"feature_shift_mc_vs_exp.csv", index=False)
    print("wrote feature_shift_mc_vs_exp.csv"); return df


if __name__ == "__main__":
    for fn in (t1_stat_validity, t2_excess_vs_cut, t3_topology_by_population,
               t4_muatm_fp_vs_zenith, t5_feature_shift):
        print(f"\n=== {fn.__name__} ===")
        print(fn().to_string(index=False))
