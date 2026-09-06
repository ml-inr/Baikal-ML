#!/usr/bin/env python
"""Assemble & characterize the fine-tune background set: exp events selected by
`encoder-OOD (kNN dist to MC embedding manifold) AND rvert<0.7`, to be labelled 0
in a fine-tune of E1 (poor man's SNGP). This is a SIZING/characterization pass BEFORE
touching the trainer — it reports how many events pass at MC p99 / p99.9 OOD thresholds
and their score/rvert/Q distributions.

Populations (h8s3, out-of-training preds; exp drops bad runs c02 r20/r249):
  - MC reference   (muatm+nu): defines the in-distribution encoder manifold (kNN fit).
  - MC calibration (muatm+nu): its kNN-to-ref distances give the p99 / p99.9 thresholds.
  - exp sample                : scored for OOD; the cut is applied here.

encoder-OOD = mean kNN(k=20) distance to the MC reference embeddings (E1@ep10 encoder).
Outputs tables/finetune_set.csv (per-exp-event), tables/finetune_summary.csv,
figures/finetune_set.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings

CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR = 0.8
CUT = 'pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
DEV = 'cuda:0'
KNN = 20; RVCUT = 0.7
N_MC_REF = 32000; N_MC_CAL = 8000; N_EXP = 120000
EXP_POOL_H8S3 = 480186   # total h8s3 out-of-training exp in this pred DB (for extrapolation)


def read_hits_feats(h5path, probs_path, df):
    """Per event: raw filtered-hit array (prob>THR), rvert, Q_mean(clip100), n_filt.
    Returns feats(list|None), rvert, qmean, nfilt arrays aligned to df rows."""
    feats = [None]*len(df)
    rvert = np.full(len(df), np.nan); qmean = np.full(len(df), np.nan)
    nfilt = np.full(len(df), 0, dtype=np.int32)
    f = h5py.File(h5path,'r',**RDCC); fp = h5py.File(probs_path,'r',**RDCC)
    for (base,pk), idx in df.groupby(['base','part_key']).groups.items():
        try:
            es = f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds = f[f'{base}/raw/data/{pk}/data']
            pr = fp[f'{base}/probs/{pk}/data']
        except KeyError:
            continue
        for row, l in zip(np.array(idx), df.loc[idx,'local_idx'].to_numpy()):
            if l+1 >= len(es): continue
            s,e = int(es[l]), int(es[l+1]); p = pr[s:e].astype(np.float32); m = p > THR
            if m.sum() < 2: continue
            hh = ds[s:e].astype(np.float32)[m]
            order = np.argsort(hh[:,1])          # sort by time (col 1)
            hh = hh[order]
            feats[row] = hh
            sx,sy,sz = hh[:,2].std(), hh[:,3].std(), hh[:,4].std()
            rvert[row] = sz/(np.sqrt(sx*sx+sy*sy)+1e-3)
            qmean[row] = np.clip(hh[:,0],0,100).mean(); nfilt[row] = len(hh)
    f.close(); fp.close(); return feats, rvert, qmean, nfilt


def embed(model, norm, df, h5, pr, tag):
    feats, rvert, qmean, nfilt = read_hits_feats(h5, pr, df)
    keep = [i for i,x in enumerate(feats) if x is not None]
    fl = [feats[i] for i in keep]
    sc, emb = predict_scores_and_embeddings(model, fl, norm, batch_size=512, device=DEV)
    sub = df.iloc[keep].copy().reset_index(drop=True)
    sub['rvert'] = rvert[keep]; sub['qmean'] = qmean[keep]; sub['nfilt'] = nfilt[keep]
    sub['emb_score'] = sc
    print(f'{tag:10s}: {len(sub):>6} events embedded (emb {emb.shape})', flush=True)
    return sub, emb.astype(np.float32)


def main():
    c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

    def q_mc(n):
        return c.execute(f"""SELECT pr.score, ev.data_class AS base, l.part_key, l.local_idx
          FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
          JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
          WHERE {CUT} AND ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020')
          ORDER BY random() LIMIT {n}""").df()

    def q_exp(n):
        return c.execute(f"""SELECT pr.score, 'exp_full' AS base, l.part_key, l.local_idx
          FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
          JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
          WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))
          ORDER BY random() LIMIT {n}""").df()

    mc = q_mc(N_MC_REF + N_MC_CAL); exp = q_exp(N_EXP)
    c.close()

    model, norm, _ = load_model(str(CKPT), device=DEV)
    mc_sub, mc_emb = embed(model, norm, mc, MCH5, MCPROBS, 'mc')
    exp_sub, exp_emb = embed(model, norm, exp, EXH5, EXPROBS, 'exp')

    # split MC into reference (fit kNN) and calibration (threshold)
    rng = np.random.default_rng(42); perm = rng.permutation(len(mc_sub))
    n_ref = min(N_MC_REF, len(mc_sub)-1000)
    ref_i, cal_i = perm[:n_ref], perm[n_ref:]
    ref_emb = mc_emb[ref_i]; cal_emb = mc_emb[cal_i]

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=KNN).fit(ref_emb)
    cal_ood = nn.kneighbors(cal_emb)[0].mean(1)
    exp_ood = nn.kneighbors(exp_emb)[0].mean(1)
    exp_sub['ood'] = exp_ood

    thr = {'p99': float(np.percentile(cal_ood, 99)), 'p99.9': float(np.percentile(cal_ood, 99.9))}
    print(f'\nMC calibration OOD kNN dist: median={np.median(cal_ood):.2f} '
          f'p99={thr["p99"]:.2f} p99.9={thr["p99.9"]:.2f}', flush=True)

    exp_sub.to_csv(HERE/'tables/finetune_set.csv', index=False)

    # ---- apply cuts, report ----
    N = len(exp_sub); scale = EXP_POOL_H8S3 / N
    n_hi = int((exp_sub.score > 0.8).sum())          # score>0.8 = the false positives we target
    rows = []
    print(f'\nexp sample N={N} (h8s3, out-of-training); pool={EXP_POOL_H8S3}, scale x{scale:.2f}')
    print(f'score>0.8 in sample: {n_hi} ({n_hi/N:.2e})\n')
    hdr = f'{"threshold":>10} | {"OOD-only":>22} | {"OOD & rvert<0.7":>22} | {"recall(score>0.8)":>17}'
    print(hdr); print('-'*len(hdr))
    for name, t in thr.items():
        ood_m = exp_sub.ood > t
        cut_m = ood_m & (exp_sub.rvert < RVCUT)
        n_ood = int(ood_m.sum()); n_cut = int(cut_m.sum())
        # recall of the actual false positives (score>0.8) by the full cut
        fp = exp_sub.score > 0.8
        rec = float((cut_m & fp).sum() / max(fp.sum(), 1))
        sel = exp_sub[cut_m]
        rows.append(dict(threshold=name, ood_thr=round(t,2),
            n_ood_sample=n_ood, n_cut_sample=n_cut,
            n_ood_pool=int(round(n_ood*scale)), n_cut_pool=int(round(n_cut*scale)),
            frac_cut=n_cut/N, recall_fp=round(rec,3),
            sel_score_med=round(float(sel.score.median()),3) if len(sel) else np.nan,
            sel_frac_score_gt08=round(float((sel.score>0.8).mean()),3) if len(sel) else np.nan,
            sel_rvert_med=round(float(sel.rvert.median()),3) if len(sel) else np.nan,
            sel_qmean_med=round(float(sel.qmean.median()),2) if len(sel) else np.nan,
            sel_nfilt_med=int(sel.nfilt.median()) if len(sel) else 0))
        print(f'{name:>10} | {n_ood:>7} ({int(round(n_ood*scale)):>7} pool) | '
              f'{n_cut:>7} ({int(round(n_cut*scale)):>7} pool) | {rec:>17.3f}')
    tab = pd.DataFrame(rows); tab.to_csv(HERE/'tables/finetune_summary.csv', index=False)
    print('\n'+tab.to_string(index=False))

    # ---- figure: OOD dist (exp vs MC calib), and selected-set score/rvert ----
    fig, ax = plt.subplots(1, 3, figsize=(17,5))
    lo,hi = 0, np.percentile(np.concatenate([exp_ood, cal_ood]), 99.5)
    bins = np.linspace(lo, hi, 60)
    ax[0].hist(cal_ood, bins=bins, density=True, histtype='step', lw=2, color='tab:red', label='MC calib')
    ax[0].hist(exp_ood, bins=bins, density=True, histtype='step', lw=2, color='black', label='exp')
    for name,t in thr.items():
        ax[0].axvline(t, ls='--', color='grey'); ax[0].text(t, ax[0].get_ylim()[1]*0.9, name, rotation=90, fontsize=8)
    ax[0].set_xlabel(f'encoder-OOD (mean kNN dist, k={KNN})'); ax[0].set_ylabel('density')
    ax[0].set_title('encoder-OOD: exp vs MC'); ax[0].legend(); ax[0].grid(alpha=0.3)

    cut_p99 = (exp_sub.ood > thr['p99']) & (exp_sub.rvert < RVCUT)
    ax[1].hist(exp_sub.score, bins=np.linspace(0,1,50), histtype='step', lw=2, color='grey', label='all exp', density=True)
    if cut_p99.sum():
        ax[1].hist(exp_sub[cut_p99].score, bins=np.linspace(0,1,50), histtype='step', lw=2, color='tab:blue', label='selected (p99∧rvert)', density=True)
    ax[1].set_xlabel('nu-classifier score'); ax[1].set_ylabel('density'); ax[1].set_title('score of selected fine-tune set'); ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].scatter(exp_sub.rvert, exp_sub.ood, s=3, alpha=0.15, color='grey', label='all exp')
    if cut_p99.sum():
        ax[2].scatter(exp_sub[cut_p99].rvert, exp_sub[cut_p99].ood, s=5, alpha=0.5, color='tab:blue', label='selected')
    ax[2].axvline(RVCUT, ls='--', color='k'); ax[2].axhline(thr['p99'], ls='--', color='tab:red')
    ax[2].set_xlim(0,3); ax[2].set_xlabel('rvert'); ax[2].set_ylabel('encoder-OOD'); ax[2].set_title('rvert vs OOD (cut corner)'); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE/'figures/finetune_set.png', dpi=130)
    print('\nsaved figures/finetune_set.png')


if __name__ == '__main__':
    main()
