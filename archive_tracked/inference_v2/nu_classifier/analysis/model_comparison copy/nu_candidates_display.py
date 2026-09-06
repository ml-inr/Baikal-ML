#!/usr/bin/env python
"""Find and event-display the genuine ν candidates that SURVIVE the fine-tuned model AND
the verticality cut — i.e. the exp events the FT model still scores high AND that are
vertical (high rvert), NOT the horizontal false-ν that the fine-tune / directional cuts
remove. These are the handful of real atmospheric-ν candidates.

Selection: out-of-training exp, FT score > SCORE_MIN, rvert > RVERT_MIN (vertical), sorted
by score. Event display = 3D hit positions (x,y,z) coloured by hit time, sized by charge.
Outputs tables/nu_candidates.csv and figures/nu_candidates_display.png.
"""
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
FT='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='n_sn_hits>=8 AND n_sn_strings>=3'
SCORE_MIN=0.5; RVERT_MIN=0.9; N_READ=800; N_SHOW=9

# candidate pool: top-score out-of-training exp
c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PREDS/FT/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
df=c.execute(f"""SELECT pr.event_fk, pr.score, ev.season, ev.cluster, ev.run, l.part_key, l.local_idx
  FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND pr.score>{SCORE_MIN} AND NOT(ev.cluster=2 AND ev.run IN('20','249'))
  ORDER BY pr.score DESC LIMIT {N_READ}""").df()
c.close()
print(f'candidate pool (FT score>{SCORE_MIN}, out-of-training): {len(df)}')

# read filtered hits, compute rvert / Q / n
f=h5py.File(EXH5,'r',**RDCC); fp=h5py.File(EXPROBS,'r',**RDCC)
hits_all=[None]*len(df); rvert=np.full(len(df),np.nan); qm=np.full(len(df),np.nan); nh=np.zeros(len(df),int)
for pk,idx in df.groupby('part_key').groups.items():
    try: es=f[f'exp_full/raw/ev_starts/{pk}/data'][:]; ds=f[f'exp_full/raw/data/{pk}/data']; pr=fp[f'exp_full/probs/{pk}/data']
    except KeyError: continue
    for row,l in zip(np.array(idx), df.loc[idx,'local_idx'].to_numpy()):
        if l+1>=len(es): continue
        s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
        if m.sum()<2: continue
        hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]; hits_all[row]=hh
        sx,sy,sz=hh[:,2].std(),hh[:,3].std(),hh[:,4].std()
        rvert[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3); qm[row]=np.clip(hh[:,0],0,100).mean(); nh[row]=len(hh)
f.close(); fp.close()
df['rvert']=rvert; df['qmean']=qm; df['nfilt']=nh
cand=df[(df.rvert>RVERT_MIN)].dropna(subset=['rvert']).sort_values('score',ascending=False).reset_index(drop=True)
cand.to_csv(HERE/'tables/nu_candidates.csv',index=False)
print(f'vertical ν candidates (rvert>{RVERT_MIN}): {len(cand)}')
print(cand[['event_fk','score','rvert','qmean','nfilt','season','cluster','run']].head(N_SHOW).to_string(index=False))

# event displays for the top N_SHOW
show=cand.head(N_SHOW)
ncol=3; nrow=int(np.ceil(len(show)/ncol))
fig=plt.figure(figsize=(5*ncol,4.5*nrow))
for i,(_,r) in enumerate(show.iterrows()):
    hh=hits_all[df.index[df.event_fk==r.event_fk][0]]
    ax=fig.add_subplot(nrow,ncol,i+1,projection='3d')
    t=hh[:,1]; q=np.clip(hh[:,0],0,100)
    p=ax.scatter(hh[:,2],hh[:,3],hh[:,4],c=t,cmap='viridis',s=10+4*q,alpha=0.85)
    ax.set_title(f"score={r.score:.2f} rvert={r.rvert:.2f}\nQ={r.qmean:.1f} nhit={int(r.nfilt)} c{int(r.cluster)}r{int(r.run)}",fontsize=9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z'); ax.tick_params(labelsize=6)
fig.suptitle(f'Genuine ν candidates surviving FT (score>{SCORE_MIN}) AND verticality (rvert>{RVERT_MIN}) — colour=time, size~charge',fontsize=12)
fig.tight_layout(); fig.savefig(HERE/'figures/nu_candidates_display.png',dpi=140)
print('saved figures/nu_candidates_display.png')
