#!/usr/bin/env python
"""Composite OOD analysis of the exp neutrino-like excess (E1@ep10 embeddings).

The A/B decomposition showed the excess is dominated by a per-angle domain gap (B),
i.e. verticality does NOT explain it. Here we characterize that gap directly in the
model's 128-dim encoder embedding space: is exp high-score (false-ν) OOD relative to
the MC manifold, and does OOD-ness add information beyond (verticality, score)?

Populations (h8s3, out-of-training preds): MC muatm, MC ν (nuatm+nue2), exp>0.8,
exp<0.2. For each: filtered hits (prob>0.8) -> encoder embedding. OOD metrics vs the
MC manifold: class-conditional Mahalanobis, kNN distance, domain-classifier AUC.

Outputs tables/composite_ood.csv, figures/composite_ood_umap.png,
figures/composite_ood_mahalanobis.png.
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
CUT = 'n_sn_hits>=8 and n_sn_strings>=3'
DEV = 'cuda:3'


def read_filtered_hits(h5path, probs_path, base_of, df):
    """Return list of (n_filt,5) raw filtered-hit arrays (prob>THR); NaN-skip drops."""
    feats = [None]*len(df); rvert = np.full(len(df), np.nan)
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
            feats[row] = hh
            sx,sy,sz = hh[:,2].std(), hh[:,3].std(), hh[:,4].std()
            rvert[row] = sz/(np.sqrt(sx*sx+sy*sy)+1e-3)
    f.close(); fp.close(); return feats, rvert


def main():
    c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

    def q_mc(where, n):
        d = c.execute(f"""SELECT pr.score, ev.data_class AS base, l.part_key, l.local_idx
          FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
          JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
          WHERE {CUT} AND {where} ORDER BY random() LIMIT {n}""").df()
        return d

    def q_exp(where, n):
        d = c.execute(f"""SELECT pr.score, 'exp_full' AS base, l.part_key, l.local_idx
          FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
          JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
          WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) AND {where}
          ORDER BY random() LIMIT {n}""").df()
        return d

    pops = {
        'mc_muon':   (q_mc("ev.data_class='muatm_2020'", 3000),               MCH5, MCPROBS),
        'mc_nu':     (q_mc("ev.data_class IN ('nuatm_2020','nue2_2020')",3000),MCH5, MCPROBS),
        'exp_hi':    (q_exp("pr.score>0.8", 2500),                            EXH5, EXPROBS),
        'exp_lo':    (q_exp("pr.score<0.2", 3000),                            EXH5, EXPROBS),
    }
    c.close()

    model, norm, _ = load_model(str(CKPT), device=DEV)
    frames = []
    for name,(df,h5,pr) in pops.items():
        feats, rvert = read_filtered_hits(h5, pr, None, df)
        keep = [i for i,x in enumerate(feats) if x is not None]
        fl = [feats[i] for i in keep]
        emb_sc = predict_scores_and_embeddings(model, fl, norm, batch_size=512, device=DEV)
        sc, emb = emb_sc
        sub = df.iloc[keep].copy().reset_index(drop=True)
        sub['pop'] = name; sub['rvert'] = rvert[keep]; sub['emb_score'] = sc
        frames.append((sub, emb))
        print(f'{name:9s}: {len(sub):>5} events, emb {emb.shape}')

    meta = pd.concat([f[0] for f in frames], ignore_index=True)
    E = np.vstack([f[1] for f in frames]).astype(np.float64)
    meta['idx'] = np.arange(len(meta))

    # ---- class-conditional Mahalanobis to MC manifold (shared cov) ----
    is_mc = meta['pop'].isin(['mc_muon','mc_nu']).to_numpy()
    Emc = E[is_mc]
    mu_mu = E[meta['pop'].to_numpy()=='mc_muon'].mean(0)
    mu_nu = E[meta['pop'].to_numpy()=='mc_nu'].mean(0)
    cov = np.cov(Emc.T) + 1e-3*np.eye(E.shape[1]); P = np.linalg.pinv(cov)
    def maha(x, m): d = x-m; return np.einsum('ij,jk,ik->i', d, P, d)
    meta['maha'] = np.minimum(maha(E, mu_mu), maha(E, mu_nu))   # min over MC class centroids

    # ---- kNN distance to MC embeddings (k=20) ----
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=20).fit(Emc)
    dist,_ = nn.kneighbors(E); meta['knn'] = dist.mean(1)

    # ---- domain classifier: exp vs MC-muon in embedding space (5-fold AUC) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    dm = meta['pop'].isin(['exp_hi','exp_lo','mc_muon'])
    y = meta.loc[dm,'pop'].str.startswith('exp').astype(int).to_numpy()
    Xd = E[meta.loc[dm,'idx'].to_numpy()]
    auc = cross_val_score(LogisticRegression(max_iter=2000), Xd, y, cv=5, scoring='roc_auc').mean()

    # ---- summary table ----
    rows=[]
    for p in ['mc_muon','mc_nu','exp_lo','exp_hi']:
        s = meta[meta['pop']==p]
        rows.append(dict(pop=p, N=len(s), rvert_med=round(s.rvert.median(),2),
                         maha_med=round(s.maha.median(),1), knn_med=round(s.knn.median(),2)))
    tab = pd.DataFrame(rows); tab.to_csv(HERE/'tables/composite_ood.csv', index=False)
    print('\n'+tab.to_string(index=False))
    print(f'\nDomain-classifier AUC (exp vs MC-muon, embeddings) = {auc:.3f}')
    with open(HERE/'tables/composite_ood.csv','a') as f:
        f.write(f'# domain_clf_auc_exp_vs_mcmuon={auc:.3f}\n')

    # is exp_hi OOD beyond verticality? compare maha of exp_hi vs mc_nu (real nu) & mc_muon
    print(f"\nMahalanobis(exp_hi) med={meta[meta['pop']=='exp_hi'].maha.median():.1f} "
          f"vs mc_nu={meta[meta['pop']=='mc_nu'].maha.median():.1f} "
          f"mc_muon={meta[meta['pop']=='mc_muon'].maha.median():.1f} exp_lo={meta[meta['pop']=='exp_lo'].maha.median():.1f}")

    # ---- figures ----
    fig,ax=plt.subplots(figsize=(8,5))
    for p,cl in [('mc_muon','tab:red'),('mc_nu','tab:green'),('exp_lo','tab:blue'),('exp_hi','black')]:
        s=meta[meta['pop']==p]
        ax.hist(np.log10(s.maha+1), bins=40, histtype='step', lw=2, color=cl, density=True, label=p)
    ax.set_xlabel('log10 Mahalanobis dist to MC manifold'); ax.set_ylabel('density')
    ax.set_title(f'Composite OOD: embedding-space Mahalanobis (domain-clf AUC={auc:.2f})')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE/'figures/composite_ood_mahalanobis.png',dpi=130)
    print('saved figures/composite_ood_mahalanobis.png')

    try:
        import umap
        red = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=1).fit_transform(E)
        fig2,ax2=plt.subplots(figsize=(8,7))
        for p,cl in [('mc_muon','tab:red'),('mc_nu','tab:green'),('exp_lo','tab:blue'),('exp_hi','black')]:
            mk=meta['pop'].to_numpy()==p
            ax2.scatter(red[mk,0],red[mk,1],s=4,alpha=0.4,c=cl,label=p)
        ax2.legend(); ax2.set_title('UMAP of encoder embeddings (E1@ep10)')
        fig2.tight_layout(); fig2.savefig(HERE/'figures/composite_ood_umap.png',dpi=130)
        print('saved figures/composite_ood_umap.png')
    except Exception as ex:
        print('UMAP skipped:', type(ex).__name__, ex)


if __name__ == '__main__':
    main()
