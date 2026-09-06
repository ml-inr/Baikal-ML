#!/usr/bin/env python
"""Investigate & visualise the genuine ν candidates on the FULL exp_full: the super-
confident FT events (score>0.9, where exp/muatm rises to 4-51×, the bimodal peak) crossed
with verticality (rvert) and up/down direction (corr(t,z)). Real atmospheric-ν candidates =
high FT score AND vertical (survive the verticality cut), ideally up-going (corr>0).

Outputs tables/nu_candidates_full.csv (all super-confident, characterised) and
figures/nu_candidates_full.png (event displays of the top ones: 3D hits, colour=time)."""
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[4]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
FT='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'; SCORE_MIN=0.9; N_SHOW=12

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PREDS/FT/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
df=c.execute(f"""SELECT pr.event_fk, pr.score, ev.season, ev.cluster, ev.run, l.part_key, l.local_idx
  FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND pr.score>{SCORE_MIN} AND NOT(ev.cluster=2 AND ev.run IN('20','249'))
  ORDER BY pr.score DESC""").df()
c.close()
print(f'super-confident FT exp (score>{SCORE_MIN}, full exp_full): {len(df)}')

f=h5py.File(EXH5,'r',**RDCC); fp=h5py.File(EXPROBS,'r',**RDCC)
hits=[None]*len(df); rvert=np.full(len(df),np.nan); qm=np.full(len(df),np.nan); nh=np.zeros(len(df),int); corr=np.full(len(df),np.nan)
for pk,idx in df.groupby('part_key').groups.items():
    try: es=f[f'exp_full/raw/ev_starts/{pk}/data'][:]; ds=f[f'exp_full/raw/data/{pk}/data']; pr=fp[f'exp_full/probs/{pk}/data']
    except KeyError: continue
    for row,l in zip(np.array(idx), df.loc[idx,'local_idx'].to_numpy()):
        if l+1>=len(es): continue
        s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
        if m.sum()<2: continue
        hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]; hits[row]=hh
        sx,sy,sz=hh[:,2].std(),hh[:,3].std(),hh[:,4].std()
        rvert[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3); qm[row]=np.clip(hh[:,0],0,100).mean(); nh[row]=len(hh)
        t,z=hh[:,1],hh[:,4]
        corr[row]=np.corrcoef(t,z)[0,1] if (t.std()>1e-6 and z.std()>1e-6) else 0.0
f.close(); fp.close()
df['rvert']=rvert; df['qmean']=qm; df['nfilt']=nh; df['corr_tz']=corr
df['vertical']=df.rvert>0.9; df['up_going']=df.corr_tz>0.3   # down-going muon: corr<0
df=df.dropna(subset=['rvert']).reset_index(drop=True)
df.to_csv(HERE/'tables/nu_candidates_full.csv',index=False)

print(f'\nof {len(df)} super-confident: vertical(rvert>0.9)={int(df.vertical.sum())}, up-going-like(corr>0.3)={int(df.up_going.sum())}, both={int((df.vertical&df.up_going).sum())}')
print(f'\n== score>0.99 (bimodal peak, ~51x over bg) ==')
top=df[df.score>0.99]
print(top[['event_fk','score','rvert','corr_tz','qmean','nfilt','cluster','run']].to_string(index=False))
print(f'\n== all super-confident, top {N_SHOW} by score ==')
print(df[['score','rvert','corr_tz','qmean','nfilt','cluster','run','vertical','up_going']].head(N_SHOW).to_string(index=False))

# event displays: top N_SHOW by score
show=df.head(N_SHOW)
ncol=4; nrow=int(np.ceil(len(show)/ncol))
fig=plt.figure(figsize=(4.6*ncol,4.2*nrow))
for i,(_,r) in enumerate(show.iterrows()):
    hh=hits[df.index[df.event_fk==r.event_fk][0]]
    axp=fig.add_subplot(nrow,ncol,i+1,projection='3d')
    t=hh[:,1]; q=np.clip(hh[:,0],0,100)
    axp.scatter(hh[:,2],hh[:,3],hh[:,4],c=t,cmap='viridis',s=12+5*q,alpha=0.9)
    flag=('V' if r.vertical else 'h')+('↑' if r.up_going else ('↓' if r.corr_tz<-0.3 else '·'))
    axp.set_title(f"s={r.score:.3f} rv={r.rvert:.2f} {flag}\nQ={r.qmean:.1f} n={int(r.nfilt)} c{int(r.cluster)}r{int(r.run)}",fontsize=8)
    axp.set_xlabel('x',fontsize=7); axp.set_ylabel('y',fontsize=7); axp.set_zlabel('z',fontsize=7); axp.tick_params(labelsize=6)
fig.suptitle(f'Super-confident FT ν candidates (score>{SCORE_MIN}, full exp_full) — colour=time, size~charge; V=vertical ↑=up-going',fontsize=12)
fig.tight_layout(); fig.savefig(HERE/'figures/nu_candidates_full.png',dpi=140)
print('\nsaved figures/nu_candidates_full.png')
