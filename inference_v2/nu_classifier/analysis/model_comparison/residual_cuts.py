#!/usr/bin/env python
"""Are the RESIDUAL fine-tuned false-neutrinos removable by a verticality (rvert) /
brightness (Q) cut, and at what neutrino-efficiency cost? (For the paper: 'NN halves the
excess; standard directional cuts handle the rest.')

Residual false-ν = out-of-training exp events the FINE-TUNED model still scores >0.8.
Compare their rvert / Q_mean to MC-ν (nue2) that the FT model also scores >0.8 (the signal
we must keep). Then scan a verticality cut rvert>t: fraction of residual excess removed vs
fraction of nue2 survivors lost.

rvert=std_z/std_xy on filtered hits (low = horizontal). Outputs tables/residual_cuts.csv,
figures/residual_cuts.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores

PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
FT_DIR='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
FT_CKPT=ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned/best_finetuned_model.pth'
EXP_TRAIN_NPY = ROOT/'data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='n_sn_hits>=8 AND n_sn_strings>=3'; DEV='cuda:1'
# Residual defined at the FT WORKING-POINT cut (μ-survival≈1e-3 → FT score≈0.40 from
# validation), NOT fixed 0.8 — the FT model compresses scores, so >0.8 is unrepresentative.
FT_WP=0.40


def read(h5,pr,df):
    feats=[None]*len(df); rvert=np.full(len(df),np.nan); qm=np.full(len(df),np.nan)
    f=h5py.File(h5,'r',**RDCC); fp=h5py.File(pr,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; p=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        idx=np.array(idx); locs=df.loc[idx,'local_idx'].to_numpy(); o=np.argsort(locs)
        for row,l in zip(idx[o],locs[o]):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); pp=p[s:e].astype(np.float32); m=pp>THR
            if m.sum()<2: continue
            hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]; feats[row]=hh
            sx,sy,sz=hh[:,2].std(),hh[:,3].std(),hh[:,4].std()
            rvert[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3); qm[row]=np.clip(hh[:,0],0,100).mean()
    f.close(); fp.close(); return feats,rvert,qm

MODEL,NORM,_=load_model(str(FT_CKPT),device=DEV)
c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PREDS/FT_DIR/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
# residual FT false-nu = out-of-training exp with FT score>0.8
exp=c.execute(f"""SELECT pr.score sc0,'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND pr.score>{FT_WP} AND NOT(ev.cluster=2 AND ev.run IN('20','249'))""").df()
# nue2 sample (FT-score fresh; precomputed probs cover 100% of nue2)
c.execute(f"ATTACH '{ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
nue=c.execute(f"""SELECT 'nue2_2020' base,l.part_key,l.local_idx FROM m.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND ev.data_class='nue2_2020' ORDER BY random() LIMIT 40000""").df()
# out-of-training exp exclusion
p=np.load(EXP_TRAIN_NPY/'exp_h5_part_keys.npy',allow_pickle=True); l=np.load(EXP_TRAIN_NPY/'exp_h5_local_event_ids.npy')
train=set(f"{a}|{b}" for a,b in zip(p.astype(str),l.astype(np.int64)))
k=exp.part_key.astype(str)+'|'+exp.local_idx.astype(str); exp=exp[~k.isin(train)].reset_index(drop=True)
c.close()
print(f"residual FT false-nu (exp score>0.8, out-of-training): {len(exp)}",flush=True)

ef,er,eq=read(EXH5,EXPROBS,exp); ekeep=[i for i,x in enumerate(ef) if x is not None]
exp=exp.iloc[ekeep].reset_index(drop=True); exp['rvert']=er[ekeep]; exp['qmean']=eq[ekeep]
nf,nr,nq=read(MCH5,MCPROBS,nue); nkeep=[i for i,x in enumerate(nf) if x is not None]
nsc=predict_scores(MODEL,[nf[i] for i in nkeep],NORM,batch_size=512,device=DEV,with_tqdm=False)
nue=nue.iloc[nkeep].reset_index(drop=True); nue['rvert']=nr[nkeep]; nue['qmean']=nq[nkeep]; nue['ftscore']=nsc
nue_surv=nue[nue.ftscore>FT_WP].reset_index(drop=True)   # nue2 the FT model keeps as signal at the WP
print(f"nue2 FT-survivors (score>0.8): {len(nue_surv)}/{len(nue)}",flush=True)

print(f"\nresidual exp false-nu:  rvert med={exp.rvert.median():.3f}  Q med={exp.qmean.median():.2f}")
print(f"nue2 FT-survivors:      rvert med={nue_surv.rvert.median():.3f}  Q med={nue_surv.qmean.median():.2f}")

# verticality-cut scan: keep rvert> t  (remove horizontal)
print(f"\n{'rvert>t':>8} {'excess removed':>14} {'nue2 kept':>10}")
rows=[]
for t in [0.0,0.4,0.55,0.7,0.9,1.1,1.4]:
    exc_removed=float((exp.rvert<t).mean())      # residual false-nu below cut = removed
    nue_kept=float((nue_surv.rvert>=t).mean())   # signal retained
    rows.append(dict(rvert_cut=t,excess_removed=exc_removed,nue2_kept=nue_kept))
    print(f"{t:>8.2f} {exc_removed:>14.3f} {nue_kept:>10.3f}")
pd.DataFrame(rows).to_csv(HERE/'tables/residual_cuts.csv',index=False)

fig,ax=plt.subplots(1,3,figsize=(18,5))
b=np.linspace(0,3,50)
ax[0].hist(exp.rvert,bins=b,density=True,histtype='step',lw=2,color='tab:red',label=f'residual exp false-ν (n={len(exp)})')
ax[0].hist(nue_surv.rvert,bins=b,density=True,histtype='step',lw=2,color='tab:green',label=f'nue2 FT-survivors (n={len(nue_surv)})')
ax[0].set_xlabel('rvert (verticality)');ax[0].set_ylabel('density');ax[0].set_title('verticality: residual false-ν vs kept ν');ax[0].legend();ax[0].grid(alpha=0.3)
bq=np.linspace(0,30,50)
ax[1].hist(np.clip(exp.qmean,0,30),bins=bq,density=True,histtype='step',lw=2,color='tab:red',label='residual exp false-ν')
ax[1].hist(np.clip(nue_surv.qmean,0,30),bins=bq,density=True,histtype='step',lw=2,color='tab:green',label='nue2 FT-survivors')
ax[1].set_xlabel('Q_mean (brightness)');ax[1].set_ylabel('density');ax[1].set_title('brightness');ax[1].legend();ax[1].grid(alpha=0.3)
r=pd.DataFrame(rows)
ax[2].plot(r.nue2_kept,r.excess_removed,'o-',color='k')
for _,rr in r.iterrows(): ax[2].annotate(f"{rr.rvert_cut:.2f}",(rr.nue2_kept,rr.excess_removed),fontsize=8)
ax[2].set_xlabel('nue2 signal kept');ax[2].set_ylabel('residual excess removed');ax[2].set_title('verticality-cut tradeoff');ax[2].grid(alpha=0.3)
fig.tight_layout();fig.savefig(HERE/'figures/residual_cuts.png',dpi=140);print('saved figures/residual_cuts.png')
