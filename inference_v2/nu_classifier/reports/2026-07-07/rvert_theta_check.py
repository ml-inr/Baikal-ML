#!/usr/bin/env python
"""How reliable is the rvert<0.7 cut as a near-horizontal selector?
MC muatm (GT theta from prime_prty[:,0]) — rvert vs theta relation, and the
efficiency/purity of rvert<0.7 for selecting near-horizon (theta<110). Plus rvert
distributions exp vs muatm. h8s3, broad score range. rvert = std_z/std_xy on filtered hits.
Outputs tables/rvert_theta.csv, figures/rvert_theta.png.
"""
from __future__ import annotations
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=8 and n_sn_strings>=3'
NMU=30000; NEX=20000; RVCUT=0.7; HORIZON=110.0

def read_rvert(h5path, probs_path, df):
    rv=np.full(len(df),np.nan)
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
    f.close(); fp.close(); return rv

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
mu=c.execute(f"SELECT 'muatm_2020' base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020' JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} ORDER BY random() LIMIT {NMU}").df()
ex=c.execute(f"SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) ORDER BY random() LIMIT {NEX}").df()
c.close()
mu['rvert']=read_rvert(MCH5,MCPROBS,mu); ex['rvert']=read_rvert(EXH5,EXPROBS,ex)
# GT theta for muatm
th=np.full(len(mu),np.nan)
with h5py.File(MCH5,'r') as f:
    g=f['muatm_2020/prime_prty']
    for pk,idx in mu.groupby('part_key').groups.items():
        k=f'{pk}/data'
        if k in g:
            t=g[k][:,0]; li=mu.loc[idx,'local_idx'].to_numpy(); ok=li<len(t); th[np.array(idx)[ok]]=t[li[ok]]
mu['theta']=th
mu=mu.dropna(subset=['rvert','theta']); ex=ex.dropna(subset=['rvert'])
mu.to_csv(HERE/'tables/rvert_theta.csv',index=False)
print(f'muatm N={len(mu)}, exp N={len(ex)}')

# correlation
print(f'\ncorr(rvert, theta) muatm: pearson={pearsonr(mu.rvert,mu.theta)[0]:.3f} spearman={spearmanr(mu.rvert,mu.theta)[0]:.3f}')
# rvert<0.7 selector quality for near-horizon (theta<110)
sel=mu.rvert<RVCUT; horiz=mu.theta<HORIZON
eff=(sel&horiz).sum()/max(horiz.sum(),1); pur=(sel&horiz).sum()/max(sel.sum(),1)
print(f'\nrvert<{RVCUT} as near-horizon(theta<{HORIZON}) selector:')
print(f'  efficiency (recall of horizon) = {eff:.3f}   purity (of selected) = {pur:.3f}')
print(f'  theta of rvert<{RVCUT}: median={mu[sel].theta.median():.1f} [p10={np.percentile(mu[sel].theta,10):.0f}, p90={np.percentile(mu[sel].theta,90):.0f}]')
print(f'  theta of rvert>{RVCUT}: median={mu[~sel].theta.median():.1f} [p10={np.percentile(mu[~sel].theta,10):.0f}, p90={np.percentile(mu[~sel].theta,90):.0f}]')
# profile: median theta per rvert bin
print(f'\n{"rvert bin":>12} | {"N":>6} {"theta med":>9} {"theta p10-p90":>14} {"frac horizon":>12}')
edges=[0,0.4,0.55,0.7,0.9,1.2,2,10]
rows=[]
for a,b in zip(edges[:-1],edges[1:]):
    s=mu[(mu.rvert>=a)&(mu.rvert<b)]
    if len(s):
        rows.append(dict(rvlo=a,rvhi=b,N=len(s),theta_med=s.theta.median(),frac_horizon=np.mean(s.theta<HORIZON)))
        print(f'{a:.2f}-{b:>4.1f} | {len(s):>6} {s.theta.median():>9.1f} {np.percentile(s.theta,10):>6.0f}-{np.percentile(s.theta,90):<7.0f} {np.mean(s.theta<HORIZON):>12.3f}')
pd.DataFrame(rows).to_csv(HERE/'tables/rvert_theta_profile.csv',index=False)

fig,ax=plt.subplots(1,3,figsize=(17,5))
ax[0].hist(mu.rvert,bins=np.linspace(0,3,60),density=True,histtype='step',lw=2,color='tab:red',label=f'muatm (n={len(mu)})')
ax[0].hist(ex.rvert,bins=np.linspace(0,3,60),density=True,histtype='step',lw=2,color='black',label=f'exp (n={len(ex)})')
ax[0].axvline(RVCUT,color='grey',ls='--',label=f'cut {RVCUT}'); ax[0].set_xlabel('rvert'); ax[0].set_ylabel('density'); ax[0].set_title('rvert distribution'); ax[0].legend(); ax[0].grid(alpha=0.3)
hb=ax[1].hexbin(mu.rvert,mu.theta,gridsize=40,bins='log',cmap='viridis',extent=(0,3,90,180))
ax[1].axvline(RVCUT,color='white',ls='--'); ax[1].axhline(HORIZON,color='white',ls=':'); ax[1].set_xlabel('rvert'); ax[1].set_ylabel('GT theta (deg)'); ax[1].set_title('rvert vs true zenith (muatm)'); fig.colorbar(hb,ax=ax[1])
b=pd.DataFrame(rows); rc=[(a+bb)/2 for a,bb in zip([r for r in b.rvlo],[r for r in b.rvhi])]
ax[2].plot(rc,b.theta_med,'o-',color='tab:red',lw=2); ax[2].axvline(RVCUT,color='grey',ls='--'); ax[2].axhline(HORIZON,color='grey',ls=':')
ax[2].set_xlabel('rvert'); ax[2].set_ylabel('median GT theta'); ax[2].set_title('profile: median theta vs rvert'); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/rvert_theta.png',dpi=130); print('saved figures/rvert_theta.png')
