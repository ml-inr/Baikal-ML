#!/usr/bin/env python
"""Working-point SWEEP: does the fine-tune on OOD bg (FT, converged @ep8) reduce the exp
neutrino-like excess relative to E1 (@ep10) across a range of muon-suppression working
points — not just at the single μ-survival=1e-3 point compared before (and there E5 was
undertrained @ep5)?

Both models score the SAME h8s3 sample (fresh inference, matched events). For a grid of
MC-muatm survival fractions μ, each model's cut = quantile(its muatm scores, 1-μ); we
report exp survival (the excess) and MC-ν signal efficiency at that cut. Comparing E1 vs
E5 at matched μ (and at matched signal-eff) removes the score-compression artifact that
made the fixed-0.8 comparison misleading.

Note: sample includes training events, but E1 and E5 share the same training set (same
source NPY + seed), so contamination is identical → the E1-vs-E5 relative comparison is
unaffected. Outputs tables/finetune_validation.csv + figures/finetune_validation.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores

CAT = ROOT/'data_manager/catalog_v2.duckdb'
E1_PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
E1_CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
FT_CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned/best_finetuned_model.pth'  # FINETUNED (OOD bg)
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT = 'pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
DEV='cuda:3'
N_MU=140000; N_NU=30000; N_EXP=90000
MU_GRID=[1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4]


def read_filtered(h5path, probs_path, df):
    feats=[None]*len(df)
    f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        idx=np.array(idx); locs=df.loc[idx,'local_idx'].to_numpy(); o=np.argsort(locs)
        for row,l in zip(idx[o],locs[o]):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            hh=ds[s:e].astype(np.float32)[m]; feats[row]=hh[np.argsort(hh[:,1])]
    f.close(); fp.close()
    keep=[i for i,x in enumerate(feats) if x is not None]
    return [feats[i] for i in keep]


c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{E1_PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{E1_PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def qmc(dc,n):
    return c.execute(f"""SELECT ev.data_class base,l.part_key,l.local_idx FROM m.predictions pr
      JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND ev.data_class IN ({dc}) ORDER BY random() LIMIT {n}""").df()
mu=qmc("'muatm_2020'",N_MU); nu=qmc("'nuatm_2020','nue2_2020'",N_NU)
exp=c.execute(f"""SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) ORDER BY random() LIMIT {N_EXP}""").df()
c.close()

print('reading hits …',flush=True)
mu_f=read_filtered(MCH5,MCPROBS,mu); nu_f=read_filtered(MCH5,MCPROBS,nu); exp_f=read_filtered(EXH5,EXPROBS,exp)
print(f'muatm {len(mu_f)}, nu {len(nu_f)}, exp {len(exp_f)}',flush=True)

def score_all(ckpt):
    model,norm,_=load_model(str(ckpt),device=DEV)
    s_mu=predict_scores(model,mu_f,norm,batch_size=512,device=DEV,with_tqdm=False)
    s_nu=predict_scores(model,nu_f,norm,batch_size=512,device=DEV,with_tqdm=False)
    s_ex=predict_scores(model,exp_f,norm,batch_size=512,device=DEV,with_tqdm=False)
    return s_mu,s_nu,s_ex
print('scoring E1 …',flush=True); e1=score_all(E1_CKPT)
print('scoring FINETUNED …',flush=True); e5=score_all(FT_CKPT)

rows=[]
for mu_surv in MU_GRID:
    r={'mu_surv':mu_surv}
    for tag,(s_mu,s_nu,s_ex) in [('E1',e1),('FT',e5)]:
        cut=np.quantile(s_mu,1.0-mu_surv)
        r[f'{tag}_cut']=cut
        r[f'{tag}_sig_eff']=float((s_nu>cut).mean())
        r[f'{tag}_exp_frac']=float((s_ex>cut).mean())
    r['exp_ratio_FT_over_E1']=r['FT_exp_frac']/r['E1_exp_frac'] if r['E1_exp_frac']>0 else np.nan
    rows.append(r)
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/finetune_validation.csv',index=False)
pd.set_option('display.width',200,'display.max_columns',20)
print('\n=== E1 (ep10) vs FINETUNED at matched muon working points ===')
print(tab.to_string(index=False,float_format=lambda x:f'{x:.4g}'))
# also: excess at matched SIGNAL efficiency (interpolate exp_frac vs sig_eff)
print('\n(reading: exp_frac = fraction of exp surviving the cut = the excess; lower is better)')

fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].plot(tab.mu_surv,tab.E1_exp_frac,'o-',label='E1 (ep10)',color='tab:blue')
ax[0].plot(tab.mu_surv,tab.FT_exp_frac,'s-',label='FINETUNED',color='tab:red')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].invert_xaxis()
ax[0].set_xlabel('MC muon survival (working point; stricter →)'); ax[0].set_ylabel('exp survival fraction (excess)')
ax[0].set_title('exp excess vs working point'); ax[0].legend(); ax[0].grid(alpha=0.3,which='both')
# excess vs signal efficiency (the fair axis)
ax[1].plot(tab.E1_sig_eff,tab.E1_exp_frac,'o-',label='E1',color='tab:blue')
ax[1].plot(tab.FT_sig_eff,tab.FT_exp_frac,'s-',label='E5 horizon',color='tab:red')
ax[1].set_yscale('log'); ax[1].set_xlabel('MC-ν signal efficiency'); ax[1].set_ylabel('exp survival fraction (excess)')
ax[1].set_title('exp excess vs signal efficiency'); ax[1].legend(); ax[1].grid(alpha=0.3,which='both')
fig.tight_layout(); fig.savefig(HERE/'figures/finetune_validation.png',dpi=130)
print('saved figures/finetune_validation.png')
