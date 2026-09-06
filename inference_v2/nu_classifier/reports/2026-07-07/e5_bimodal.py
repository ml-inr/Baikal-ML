#!/usr/bin/env python
"""Is E5's high-score cluster the CLEAN confident (vertical/real-ν) population, with
near-horizontal ambiguous events pushed down? (θ-loss working as designed)
For E5 vs E1 (@ep5): exp rvert vs score bin (does >0.95 become vertical?) and MC muatm
true-θ + false-rate vs score bin (does θ-loss push horizontal muons out of the top?).
Outputs tables/e5_bimodal_{exp,muatm}.csv, figures/e5_bimodal.png.
"""
from __future__ import annotations
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
CAT = ROOT/'data_manager/catalog_v2.duckdb'
PREDS = ROOT/'inference_v2/nu_classifier/preds'
E5 = '260707_0226_da_nu_classifier_exp_full_E5_horizon_lambda0.01@da_checkpoint_epoch_005'
E1 = '260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_005'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=8 and n_sn_strings>=3'
SBINS=[(0.6,0.8),(0.8,0.9),(0.9,0.95),(0.95,1.01)]

def rvert_for(dbdir, df):
    """read exp hits, return rvert per row (df has part_key, local_idx)."""
    rv=np.full(len(df),np.nan); f=h5py.File(EXH5,'r',**RDCC); fp=h5py.File(EXPROBS,'r',**RDCC)
    for pk,idx in df.groupby('part_key').groups.items():
        try: es=f[f'exp_full/raw/ev_starts/{pk}/data'][:]; ds=f[f'exp_full/raw/data/{pk}/data']; pr=fp[f'exp_full/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            h=ds[s:e].astype(np.float32)[m]
            rv[row]=h[:,4].std()/(np.sqrt(h[:,2].std()**2+h[:,3].std()**2)+1e-3)
    f.close(); fp.close(); return rv

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar'); c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

# ---- EXP: rvert vs score bin, E5 vs E1 ----
exp_rows=[]
for tag,d in [('E5',E5),('E1',E1)]:
    c.execute(f"ATTACH '{PREDS/d/'exp_full_thr0p8.duckdb'}' AS p (READ_ONLY)")
    for lo,hi in SBINS:
        df=c.execute(f"SELECT l.part_key,l.local_idx FROM p.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) AND pr.score>={lo} AND pr.score<{hi} ORDER BY random() LIMIT 3000").df()
        rv=rvert_for(d,df); rv=rv[np.isfinite(rv)]
        exp_rows.append(dict(model=tag, sbin=f'{lo}-{hi}', N=len(rv), rvert_med=np.median(rv) if len(rv) else np.nan))
    c.execute("DETACH p")
exp=pd.DataFrame(exp_rows); exp.to_csv(HERE/'tables/e5_bimodal_exp.csv',index=False)
print('== EXP rvert (verticality) by score bin ==\n'+exp.to_string(index=False))

# ---- MC muatm: true theta + count vs score bin, E5 vs E1 ----
mu_rows=[]
fg=h5py.File(MCH5,'r'); g=fg['muatm_2020/prime_prty']
for tag,d in [('E5',E5),('E1',E1)]:
    c.execute(f"ATTACH '{PREDS/d/'mc_merged_thr0p8.duckdb'}' AS p (READ_ONLY)")
    for lo,hi in SBINS:
        df=c.execute(f"SELECT l.part_key,l.local_idx FROM p.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020' JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND pr.score>={lo} AND pr.score<{hi}").df()
        th=np.full(len(df),np.nan)
        for pk,idx in df.groupby('part_key').groups.items():
            k=f'{pk}/data'
            if k in g:
                t=g[k][:,0]; li=df.loc[idx,'local_idx'].to_numpy(); ok=li<len(t); th[np.array(idx)[ok]]=t[li[ok]]
        th=th[np.isfinite(th)]
        mu_rows.append(dict(model=tag, sbin=f'{lo}-{hi}', N_muatm=len(th),
                            theta_med=np.median(th) if len(th) else np.nan,
                            frac_horizon=np.mean(th<110) if len(th) else np.nan))
    c.execute("DETACH p")
fg.close(); c.close()
mu=pd.DataFrame(mu_rows); mu.to_csv(HERE/'tables/e5_bimodal_muatm.csv',index=False)
print('\n== MC muatm false-positives by score bin (true theta) ==\n'+mu.to_string(index=False))

fig,ax=plt.subplots(1,2,figsize=(13,5))
xb=[f'{lo}-{hi}' for lo,hi in SBINS]; x=np.arange(len(xb))
for tag,cl in [('E5','tab:blue'),('E1','tab:orange')]:
    s=exp[exp.model==tag]; ax[0].plot(x,[s[s.sbin==b].rvert_med.values[0] for b in xb],'o-',lw=2,color=cl,label=tag)
ax[0].set_xticks(x); ax[0].set_xticklabels(xb); ax[0].set_xlabel('exp score bin'); ax[0].set_ylabel('rvert median (↑=vertical)')
ax[0].set_title('exp verticality vs score — does E5 top cluster go vertical?'); ax[0].legend(); ax[0].grid(alpha=0.3)
for tag,cl in [('E5','tab:blue'),('E1','tab:orange')]:
    s=mu[mu.model==tag]; ax[1].plot(x,[s[s.sbin==b].N_muatm.values[0] for b in xb],'s-',lw=2,color=cl,label=tag)
ax[1].set_xticks(x); ax[1].set_xticklabels(xb); ax[1].set_xlabel('score bin'); ax[1].set_ylabel('# muatm false-pos'); ax[1].set_yscale('log')
ax[1].set_title('MC muon false-positives vs score'); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/e5_bimodal.png',dpi=130); print('saved figures/e5_bimodal.png')
