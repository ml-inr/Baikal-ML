#!/usr/bin/env python
"""How does score depend on brightness for NEAR-HORIZONTAL events — same for exp and
muatm (=> excess is abundance of bright-horizontal) or steeper for exp (=> per-event
OOD: exp bright-horizontal score higher)? Also decompose the exp_hi/muatm_hi excess
into abundance vs per-event factors. Larger sample; muatm from boosted 6.6M preds.
Outputs tables/score_vs_brightness.csv, figures/score_vs_brightness.png.
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
NSAMP=40000; RV_HORIZ=0.7

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
            rv[row]=h[:,4].std()/(np.sqrt(h[:,2].std()**2+h[:,3].std()**2)+1e-3); qm[row]=np.clip(h[:,0],0,100).mean()
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
exp=exp.dropna(subset=['rvert','qmean']); mu=mu.dropna(subset=['rvert','qmean'])
pd.concat([exp.assign(dom='exp'),mu.assign(dom='muatm')]).to_csv(HERE/'tables/score_vs_brightness.csv',index=False)

eh=exp[exp.rvert<RV_HORIZ]; mh=mu[mu.rvert<RV_HORIZ]
print(f'horizontal (rvert<{RV_HORIZ}): exp N={len(eh)}, muatm N={len(mh)}')
# --- score vs brightness among horizontal ---
qbins=[0,3,5,8,15,100]
print(f'\n{"Qbin":>10} | {"exp N":>6} {"exp f>0.8":>9} {"exp f>0.5":>9} {"exp<s>":>7} | {"mu N":>6} {"mu f>0.8":>9} {"mu f>0.5":>9} {"mu<s>":>7}')
rows=[]
for lo,hi in zip(qbins[:-1],qbins[1:]):
    ee=eh[(eh.qmean>=lo)&(eh.qmean<hi)].score; mm=mh[(mh.qmean>=lo)&(mh.qmean<hi)].score
    rows.append(dict(qlo=lo,qhi=hi,exp_N=len(ee),exp_f08=np.mean(ee>0.8),exp_f05=np.mean(ee>0.5),exp_s=ee.mean(),
                     mu_N=len(mm),mu_f08=np.mean(mm>0.8) if len(mm) else np.nan,mu_f05=np.mean(mm>0.5) if len(mm) else np.nan,mu_s=mm.mean() if len(mm) else np.nan))
    print(f'{lo:>4}-{hi:>4} | {len(ee):>6} {np.mean(ee>0.8):>9.4f} {np.mean(ee>0.5):>9.4f} {ee.mean():>7.3f} | {len(mm):>6} {(np.mean(mm>0.8) if len(mm) else np.nan):>9.4f} {(np.mean(mm>0.5) if len(mm) else np.nan):>9.4f} {(mm.mean() if len(mm) else np.nan):>7.3f}')
pd.DataFrame(rows).to_csv(HERE/'tables/score_vs_brightness_binned.csv',index=False)
# correlation
from scipy.stats import spearmanr
print(f'\nSpearman corr(score, qmean) among horizontal: exp={spearmanr(eh.score,eh.qmean)[0]:.3f}  muatm={spearmanr(mh.score,mh.qmean)[0]:.3f}')

# --- Q1 decomposition of the exp/muatm excess at >0.8 ---
def frac08(d): return np.mean(d.score>0.8)
Ph_e=(exp.rvert<RV_HORIZ).mean(); Ph_m=(mu.rvert<RV_HORIZ).mean()
print(f'\n== excess decomposition (>0.8) ==')
print(f'  P(horizontal): exp={Ph_e:.3f} muatm={Ph_m:.3f} -> abundance factor {Ph_e/Ph_m:.2f}')
print(f'  frac>0.8 among horizontal: exp={frac08(eh):.4f} muatm={frac08(mh):.4f} -> per-horiz factor {frac08(eh)/frac08(mh) if frac08(mh)>0 else np.nan:.2f}')
print(f'  overall frac>0.8: exp={frac08(exp):.4f} muatm={frac08(mu):.4f} -> total excess {frac08(exp)/frac08(mu) if frac08(mu)>0 else np.nan:.2f}')

fig,ax=plt.subplots(1,2,figsize=(14,5))
qc=[(lo+hi)/2 for lo,hi in zip(qbins[:-1],qbins[1:])]
b=pd.DataFrame(rows)
ax[0].plot(qc,b.exp_f05,'o-',color='black',lw=2,label='exp horiz'); ax[0].plot(qc,b.mu_f05,'s-',color='tab:red',lw=2,label='muatm horiz')
ax[0].set_xlabel('Q_mean (brightness)'); ax[0].set_ylabel('frac(score>0.5)'); ax[0].set_title('score vs brightness (horizontal) — same slope?'); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_xscale('log')
ax[1].plot(qc,b.exp_s,'o-',color='black',lw=2,label='exp horiz'); ax[1].plot(qc,b.mu_s,'s-',color='tab:red',lw=2,label='muatm horiz')
ax[1].set_xlabel('Q_mean (brightness)'); ax[1].set_ylabel('mean score'); ax[1].set_title('mean score vs brightness (horizontal)'); ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_xscale('log')
fig.tight_layout(); fig.savefig(HERE/'figures/score_vs_brightness.png',dpi=130); print('saved figures/score_vs_brightness.png')
