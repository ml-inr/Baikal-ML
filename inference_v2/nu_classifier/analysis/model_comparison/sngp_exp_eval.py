#!/usr/bin/env python
"""Does the trained SNGP flag the exp false-ν as uncertain? For each population score with
the SNGP model → (mean-field score, predictive variance var=φᵀΣφ). If distance-awareness
works, the exp false-ν (events the classifier calls ν but are OOD) get HIGH variance / a
mean-field score shrunk toward 0.5, while real MC-ν keep low variance / high score. §8 of the
2026-07-08 report predicted only PARTIAL help (false-ν sit at the MC-ν edge) — test it.

Populations (h8s3): exp false-ν (E1 score>0.8), exp bulk (E1 score<0.2), MC nue2 (real ν),
MC muatm (background). Outputs tables/sngp_exp_eval.csv + figures/sngp_exp_eval.png."""
from pathlib import Path
import sys, duckdb, h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
E1='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
SNGP_CKPT=ROOT/'experiments/numu/sngp_nu_classifier_baseline/best_sngp_model.pth'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'; DEV='cuda:0'

MODEL,NORM,_=load_model(str(SNGP_CKPT),device=DEV); HEAD=MODEL.classifier
MEANS=torch.tensor(NORM['means'],dtype=torch.float32,device=DEV); STDS=torch.tensor(NORM['stds'],dtype=torch.float32,device=DEV)

@torch.no_grad()
def sngp_score_var(feats_list, bs=512):
    """Return (mean_field_score, predictive_variance) per event."""
    scores=[]; vars_=[]
    for i in range(0,len(feats_list),bs):
        batch=feats_list[i:i+bs]; b=len(batch); L=max(len(x) for x in batch)
        pad=torch.zeros(b,L,5,dtype=torch.float32,device=DEV); lengths=torch.zeros(b,dtype=torch.long,device=DEV)
        for j,x in enumerate(batch):
            n=min(len(x),500); pad[j,:n]=torch.from_numpy(x[:n]); lengths[j]=n
        mask=torch.arange(L,device=DEV)[None,:]<lengths[:,None]
        pad=torch.where(mask.unsqueeze(-1),(pad-MEANS)/(STDS+1e-8),pad)
        bd={'features':pad,'lengths':lengths,'mask':mask}
        if hasattr(MODEL,'_clip_amplitude'): bd=MODEL._clip_amplitude(bd,getattr(MODEL,'amp_clip',None))
        emb=MODEL.feature_extractor(sequences=bd['features'],lengths=bd['lengths'],mask=bd['mask'])
        phi=HEAD._phi(emb); logit=HEAD.beta(phi).squeeze(-1)
        var=torch.einsum('bi,ij,bj->b',phi,HEAD.covariance,phi).clamp_min(0.0)
        mf=logit/torch.sqrt(1.0+HEAD.mean_field_factor*var)
        scores.append(torch.sigmoid(mf).cpu().numpy()); vars_.append(var.cpu().numpy())
    return np.concatenate(scores),np.concatenate(vars_)

def read(h5,pr,df):
    feats=[None]*len(df)
    f=h5py.File(h5,'r',**RDCC); fpp=h5py.File(pr,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; p=fpp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); pp=p[s:e].astype(np.float32); m=pp>THR
            if m.sum()<2: continue
            hh=ds[s:e].astype(np.float32)[m]; feats[row]=hh[np.argsort(hh[:,1])]
    f.close(); fpp.close(); keep=[i for i,x in enumerate(feats) if x is not None]
    return df.iloc[keep].reset_index(drop=True),[feats[i] for i in keep]

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PREDS/E1/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PREDS/E1/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def qexp(w,n): return c.execute(f"SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) AND {w} ORDER BY random() LIMIT {n}").df()
def qmc(cls,n): return c.execute(f"SELECT '{cls}' base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND ev.data_class='{cls}' ORDER BY random() LIMIT {n}").df()
pops={'exp false-ν (E1>0.8)':qexp('pr.score>0.8',4000),'exp bulk (E1<0.2)':qexp('pr.score<0.2',4000),
      'MC nue2 (real ν)':qmc('nue2_2020',5000),'MC muatm (bg)':qmc('muatm_2020',5000)}
c.close()

res={}
for name,(h5,pr) in [('exp false-ν (E1>0.8)',(EXH5,EXPROBS)),('exp bulk (E1<0.2)',(EXH5,EXPROBS)),('MC nue2 (real ν)',(MCH5,MCPROBS)),('MC muatm (bg)',(MCH5,MCPROBS))]:
    sub,fl=read(h5,pr,pops[name]); sc,vr=sngp_score_var(fl); res[name]=(sc,vr)
    print(f'{name:24s}: N={len(sc):>5}  SNGP score med={np.median(sc):.3f}  var med={np.median(vr):.4f}  var p90={np.percentile(vr,90):.4f}',flush=True)

print("\n=== KEY: does SNGP flag exp false-ν (high var / shrunk score) vs real MC-ν? ===")
fn_s,fn_v=res['exp false-ν (E1>0.8)']; nu_s,nu_v=res['MC nue2 (real ν)']
print(f"  exp false-ν : SNGP score med={np.median(fn_s):.3f}  var med={np.median(fn_v):.4f}")
print(f"  MC real ν   : SNGP score med={np.median(nu_s):.3f}  var med={np.median(nu_v):.4f}")
print(f"  var ratio (false-ν / real-ν) = {np.median(fn_v)/max(np.median(nu_v),1e-9):.2f}x  (>>1 = SNGP flags them; ~1 = no help, edge-of-nu)")
rows=[dict(pop=k,n=len(v[0]),sngp_score_med=float(np.median(v[0])),var_med=float(np.median(v[1])),var_p90=float(np.percentile(v[1],90))) for k,v in res.items()]
pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent/'tables/sngp_exp_eval.csv',index=False)

fig,ax=plt.subplots(1,2,figsize=(14,5))
col={'exp false-ν (E1>0.8)':'tab:red','exp bulk (E1<0.2)':'grey','MC nue2 (real ν)':'tab:green','MC muatm (bg)':'tab:blue'}
for k,(sc,vr) in res.items():
    ax[0].hist(np.log10(vr+1e-4),bins=40,histtype='step',lw=2,density=True,color=col[k],label=k)
    ax[1].hist(sc,bins=40,histtype='step',lw=2,density=True,color=col[k],label=k)
ax[0].set_xlabel('log10 SNGP predictive variance'); ax[0].set_ylabel('density'); ax[0].set_title('Distance-aware variance'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel('SNGP mean-field score'); ax[1].set_ylabel('density'); ax[1].set_title('SNGP score'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent/'figures/sngp_exp_eval.png',dpi=140); print('saved figures/sngp_exp_eval.png')
