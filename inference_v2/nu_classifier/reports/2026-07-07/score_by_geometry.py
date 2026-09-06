#!/usr/bin/env python
"""Fix the geometry, look at the FULL score distribution (no score cut).
Select near-horizontal events (rvert<0.7) — same selection on exp and muatm — and
compare their p(score) distributions. Hypothesis: near-horizontal muatm -> ~U(0,1)
(model genuinely uncertain on these seen-but-ambiguous events); near-horizontal exp ->
pulled toward p=1 (OOD brightness tips it to ν). Also exp near-horizontal & bright.
Outputs tables/score_by_geometry.csv, figures/score_by_geometry.png.
"""
from __future__ import annotations
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=8 and n_sn_strings>=3'
NSAMP=25000; RV_HORIZ=0.7

def read_rv_q(h5path, probs_path, df):
    rv=np.full(len(df),np.nan); qm=np.full(len(df),np.nan)
    f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            h=ds[s:e].astype(np.float32)[m]
            rv[row]=h[:,4].std()/(np.sqrt(h[:,2].std()**2+h[:,3].std()**2)+1e-3)
            qm[row]=np.clip(h[:,0],0,100).mean()
    f.close(); fp.close(); return rv, qm

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
exp=c.execute(f"SELECT pr.score,'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) ORDER BY random() LIMIT {NSAMP}").df()
mu=c.execute(f"SELECT pr.score,'muatm_2020' base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020' JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} ORDER BY random() LIMIT {NSAMP}").df()
c.close()
for d,h5,prb in [(exp,EXH5,EXPROBS),(mu,MCH5,MCPROBS)]:
    d['rvert'],d['qmean']=read_rv_q(h5,prb,d)
exp=exp.dropna(subset=['rvert']); mu=mu.dropna(subset=['rvert'])
pd.concat([exp.assign(dom='exp'),mu.assign(dom='muatm')]).to_csv(HERE/'tables/score_by_geometry.csv',index=False)
qbright=mu.qmean.quantile(0.9)   # 'bright' threshold from muatm dist
print(f'exp N={len(exp)}, muatm N={len(mu)}; horiz rvert<{RV_HORIZ}; bright qmean>{qbright:.1f}')

def summ(lab,s):
    s=np.asarray(s); return f'{lab:32s} N={len(s):>6} | frac>0.5={np.mean(s>0.5):.3f} frac>0.8={np.mean(s>0.8):.4f} frac>0.95={np.mean(s>0.95):.4f} median={np.median(s):.3f}'
sel={'exp ALL (h8s3)':exp.score,'exp horizontal (rvert<0.7)':exp[exp.rvert<RV_HORIZ].score,
     'exp horiz & bright':exp[(exp.rvert<RV_HORIZ)&(exp.qmean>qbright)].score,
     'muatm ALL (h8s3)':mu.score,'muatm horizontal (rvert<0.7)':mu[mu.rvert<RV_HORIZ].score}
print(); [print(summ(k,v)) for k,v in sel.items()]

fig,ax=plt.subplots(1,2,figsize=(14,5)); b=np.linspace(0,1,41)
for lab,v,cl in [('exp horizontal','black'),('muatm horizontal','tab:red')]:
    pass
ax[0].hist(exp[exp.rvert<RV_HORIZ].score,bins=b,density=True,histtype='step',lw=2,color='black',label=f'exp horiz (n={ (exp.rvert<RV_HORIZ).sum()})')
ax[0].hist(mu[mu.rvert<RV_HORIZ].score,bins=b,density=True,histtype='step',lw=2,color='tab:red',label=f'muatm horiz (n={(mu.rvert<RV_HORIZ).sum()})')
ax[0].hist(exp[(exp.rvert<RV_HORIZ)&(exp.qmean>qbright)].score,bins=b,density=True,histtype='step',lw=2,color='tab:blue',ls='--',label=f'exp horiz&bright (n={((exp.rvert<RV_HORIZ)&(exp.qmean>qbright)).sum()})')
ax[0].set_yscale('log'); ax[0].set_xlabel('score'); ax[0].set_ylabel('density'); ax[0].set_title('Score distribution at FIXED geometry (near-horizontal)'); ax[0].legend(); ax[0].grid(alpha=0.3)
# cumulative to see the pull to 1
for lab,v,cl in [('exp horiz',exp[exp.rvert<RV_HORIZ].score,'black'),('muatm horiz',mu[mu.rvert<RV_HORIZ].score,'tab:red'),('exp horiz&bright',exp[(exp.rvert<RV_HORIZ)&(exp.qmean>qbright)].score,'tab:blue')]:
    xs=np.sort(v); ax[1].plot(xs,1-np.arange(len(xs))/len(xs),color=cl,lw=2,label=lab)
ax[1].set_yscale('log'); ax[1].set_xlabel('score cut'); ax[1].set_ylabel('fraction above cut'); ax[1].set_title('Survival: pull toward p=1?'); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/score_by_geometry.png',dpi=130); print('\nsaved figures/score_by_geometry.png')
