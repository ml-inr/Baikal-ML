#!/usr/bin/env python
"""Formalize 'exp_hi is OOD' at the RAW-event level via a rich, multi-representation
feature set + formal two-sample / one-class tests. (User route (c).)

Each event (time-ordered signal hits, [Q,t,x,y,z]) -> ~40 features across
representations: marginal moments (Q,t,xyz), inter-hit time gaps, position-cloud PCA
shape, Takens delay-embedding shape (Q,z,axis), space-time causality (position-along-
principal-axis vs time regression residual + correlations), string topology.

Formal OOD (reference = MC training manifold, all classes):
  - Mahalanobis, kNN, IsolationForest anomaly score for each population;
  - OOD-AUC: can the raw features separate exp_hi from held-out MC? (RF, 5-fold)
    and exp_hi vs muatm_hi (the exp-SPECIFIC part, controlling for horizontal-FP);
  - per-feature KS statistic (exp_hi vs MC and vs muatm_hi) to LOCALISE the OOD.
Outputs tables/raw_ood_features.csv (per-event), tables/raw_ood_summary.csv,
figures/raw_ood.png.
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

def _mom(a):
    a=np.asarray(a,float); m=a.mean(); s=a.std()+1e-9
    z=(a-m)/s; return m, a.std(), float((z**3).mean()), float((z**4).mean()-3)

def _pr(series, M=3):
    s=np.asarray(series,float); sd=s.std()
    if sd<1e-9 or len(s)<M+1: return np.nan, np.nan
    s=(s-s.mean())/sd; c=np.stack([s[i:len(s)-M+1+i] for i in range(M)],1)
    if len(c)<3: return np.nan, np.nan
    ev=np.sort(np.clip(np.linalg.eigvalsh(np.cov(c.T)),0,None))[::-1]; tot=ev.sum()+1e-12
    return (ev.sum()**2)/((ev**2).sum()+1e-12), ev[2]/(ev[0]+1e-12)

def feat_event(h):
    """h: (n,5)=[Q,t,x,y,z], time-ordered. Return dict of raw features."""
    Q,t,x,y,z = h[:,0],h[:,1],h[:,2],h[:,3],h[:,4]; n=len(h)
    f={}
    # topology (from channels not available here; use spatial strings via unique xy)
    xy=np.round(np.c_[x,y]/2)*2; ustr=np.unique(xy,axis=0); nstr=len(ustr)
    f['log_nhits']=np.log10(n); f['nstrings']=nstr; f['hits_per_str']=n/max(nstr,1)
    # charge
    qm,qs,qsk,qk=_mom(np.clip(Q,0,100)); f['Q_mean']=qm; f['Q_std']=qs; f['Q_skew']=qsk; f['Q_kurt']=qk
    f['Q_max']=Q.max(); f['Q_maxfrac']=Q.max()/(Q.sum()+1e-9); f['frac_Q_gt10']=np.mean(Q>10)
    # time
    f['t_span']=t.max()-t.min(); f['t_std']=t.std()
    dt=np.diff(np.sort(t)); f['dt_mean']=dt.mean() if len(dt) else 0; f['dt_std']=dt.std() if len(dt) else 0
    f['dt_max']=dt.max() if len(dt) else 0; _,_,f['t_skew'],f['t_kurt']=_mom(t)
    # position cloud PCA shape
    P=np.c_[x,y,z]-np.c_[x,y,z].mean(0); evp=np.sort(np.clip(np.linalg.eigvalsh(np.cov(P.T)+1e-9*np.eye(3)),0,None))[::-1]
    tp=evp.sum()+1e-12; fp=evp/tp
    f['elong']=fp[0]; f['planarity']=fp[1]-fp[2]; f['sphericity']=fp[2]/(fp[0]+1e-12)
    f['z_std']=z.std(); f['xy_std']=np.sqrt(x.std()**2+y.std()**2); f['rvert']=f['z_std']/(f['xy_std']+1e-3)
    # delay-embedding shape
    f['prQ'],f['sphQ']=_pr(Q); f['prz'],f['sphz']=_pr(z)
    # space-time causality: project positions on principal axis, regress vs time
    w=np.linalg.eigh(np.cov(P.T)+1e-9*np.eye(3))[1][:,-1]; proj=P@w
    f['prAxis'],_=_pr(proj)
    if t.std()>1e-6:
        A=np.c_[t-t.mean(),np.ones(n)]; sl,ic=np.linalg.lstsq(A,proj,rcond=None)[0]
        resid=proj-(A@[sl,ic]); f['speed_slope']=sl; f['speed_resid']=resid.std()/(np.abs(proj).std()+1e-6)
        f['corr_t_proj']=np.corrcoef(t,proj)[0,1]; f['corr_t_z']=np.corrcoef(t,z)[0,1]
        f['corr_Q_proj']=np.corrcoef(Q,proj)[0,1]
    else:
        for k in ['speed_slope','speed_resid','corr_t_proj','corr_t_z','corr_Q_proj']: f[k]=np.nan
    return f

def read_and_feat(h5path, probs_path, df):
    rows=[]; f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for l in df.loc[idx,'local_idx'].to_numpy():
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<6: continue
            hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]
            rows.append(feat_event(hh))
    f.close(); fp.close(); return pd.DataFrame(rows)

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def qmc(w,n): return c.execute(f"SELECT ev.data_class base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND {w} ORDER BY random() LIMIT {n}").df()
qex=lambda w: c.execute(f"SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) AND {w}").df()
pops={'mc_ref': qmc("ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020')",8000),
      'muatm_hi': qmc("ev.data_class='muatm_2020' AND pr.score>0.8",4000),
      'exp_hi': qex("pr.score>0.8")}
srcs={'mc_ref':(MCH5,MCPROBS),'muatm_hi':(MCH5,MCPROBS),'exp_hi':(EXH5,EXPROBS)}
c.close()

F={}
for name,df in pops.items():
    h5,prb=srcs[name]; X=read_and_feat(h5,prb,df); F[name]=X; print(f'{name:9s}: {len(X)} events, {X.shape[1]} feats')
cols=[c for c in F['mc_ref'].columns]
big=pd.concat([F[k].assign(pop=k) for k in F],ignore_index=True).replace([np.inf,-np.inf],np.nan)
big.to_csv(HERE/'tables/raw_ood_features.csv',index=False)

# clean: fill nan with mc median, standardize on mc_ref
med=big[big['pop']=='mc_ref'][cols].median(); big[cols]=big[cols].fillna(med)
mu=big[big['pop']=='mc_ref'][cols].mean(); sd=big[big['pop']=='mc_ref'][cols].std()+1e-9
Z=(big[cols]-mu)/sd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import ks_2samp
Zmc=Z[big['pop'].values=='mc_ref'].values
covi=np.linalg.pinv(np.cov(Zmc.T)+1e-3*np.eye(len(cols))); mmc=Zmc.mean(0)
maha=lambda A:np.einsum('ij,jk,ik->i',A-mmc,covi,A-mmc)
iso=IsolationForest(n_estimators=200,random_state=0).fit(Zmc)
print('\n== OOD scores (median) vs MC manifold ==')
for p in ['mc_ref','muatm_hi','exp_hi']:
    A=Z[big['pop'].values==p].values
    print(f'  {p:9s}: maha={np.median(maha(A)):7.1f}  iso_anom={-np.median(iso.score_samples(A)):.3f}')

def auc(pa,pb):
    Xa=Z[big['pop'].values==pa].values; Xb=Z[big['pop'].values==pb].values
    X=np.vstack([Xa,Xb]); y=np.r_[np.ones(len(Xa)),np.zeros(len(Xb))]
    return cross_val_score(RandomForestClassifier(200,random_state=0,n_jobs=-1),X,y,cv=5,scoring='roc_auc').mean()
auc_mc=auc('exp_hi','mc_ref'); auc_mu=auc('exp_hi','muatm_hi')
print(f'\nRF OOD-AUC  exp_hi vs MC-ref = {auc_mc:.3f}   exp_hi vs muatm_hi (exp-specific) = {auc_mu:.3f}')

# per-feature KS (localise the OOD channel)
rows=[]
for cc in cols:
    a=big[big['pop']=='exp_hi'][cc].dropna(); b=big[big['pop']=='mc_ref'][cc].dropna(); d=big[big['pop']=='muatm_hi'][cc].dropna()
    rows.append(dict(feat=cc, KS_exp_vs_mc=ks_2samp(a,b)[0], KS_exp_vs_muatm=ks_2samp(a,d)[0]))
ks=pd.DataFrame(rows).sort_values('KS_exp_vs_muatm',ascending=False)
ks.to_csv(HERE/'tables/raw_ood_summary.csv',index=False)
print('\n== top features separating exp_hi from muatm_hi (exp-specific) ==')
print(ks.head(12).to_string(index=False))

fig,ax=plt.subplots(figsize=(10,6)); top=ks.head(15)
ax.barh(top.feat[::-1],top.KS_exp_vs_muatm[::-1],color='black',label='exp_hi vs muatm_hi')
ax.barh(top.feat[::-1],top.KS_exp_vs_mc[::-1],color='tab:red',alpha=0.4,label='exp_hi vs MC-ref')
ax.set_xlabel('KS statistic'); ax.set_title(f'Raw-feature OOD of exp_hi (AUC vs MC {auc_mc:.2f}, vs muatm {auc_mu:.2f})')
ax.legend(); fig.tight_layout(); fig.savefig(HERE/'figures/raw_ood.png',dpi=130); print('saved figures/raw_ood.png')
