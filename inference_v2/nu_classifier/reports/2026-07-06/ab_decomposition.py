"""A/B decomposition of the exp excess over MC-muatm.

frac(score>0.8 | rvert bin) reconstructed per domain via Bayes:
  f(bin) = [n_hi(bin)/N_total] / [n_all(bin)/N_all_sample]
using ALL high-score events (exact n_hi) + a large all-sample for P(bin).
rvert = std_z/std_xy over sig-noise-filtered hits (verticality; low=horizontal).

Decomposes  excess = Σ P_exp(b) f_exp(b) / Σ P_mc(b) f_mc(b)  into:
  (A) angular-distribution:  Σ P_exp(b) f_mc(b) / Σ P_mc(b) f_mc(b)
  (B) per-angle gap:         Σ P_mc(b) f_exp(b) / Σ P_mc(b) f_mc(b)
Outputs tables/ab_decomposition.csv and figures/ab_decomposition.png.
"""
import duckdb, h5py, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
CAT=ROOT/'data_manager/catalog_v2.duckdb'
PRED=ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=8 and n_sn_strings>=3'
BINS=np.array([0,0.5,0.7,1.0,1.5,10.0]); LAB=['<0.5','0.5-0.7','0.7-1.0','1.0-1.5','>1.5']

def rvert(h5path,probs_path,base,df):
    out=np.full(len(df),np.nan)
    f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for pk,idx in df.groupby('part_key').groups.items():
        try:
            es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            xyz=ds[s:e,2:5].astype(np.float32)[m]
            sx,sy,sz=xyz[:,0].std(),xyz[:,1].std(),xyz[:,2].std()
            out[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3)
    f.close(); fp.close(); return out

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

def domain(tag):
    if tag=='exp':
        base,h5,pr='exp_full',EXH5,EXPROBS
        extra="AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))"; cls=''
    else:
        base,h5,pr='muatm_2020',MCH5,MCPROBS
        extra=''; cls="AND ev.data_class='muatm_2020'"
    N,K=c.execute(f"SELECT count(*), count(*) filter(where pr.score>0.8) FROM {'e' if tag=='exp' else 'm'}.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk {cls} WHERE {CUT} {extra}").fetchone()
    hi=c.execute(f"SELECT l.part_key,l.local_idx FROM {'e' if tag=='exp' else 'm'}.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk {cls} JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} {extra} AND pr.score>0.8").df()
    al=c.execute(f"SELECT l.part_key,l.local_idx FROM {'e' if tag=='exp' else 'm'}.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk {cls} JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} {extra} ORDER BY random() LIMIT 6000").df()
    rhi=rvert(h5,pr,base,hi); ral=rvert(h5,pr,base,al)
    rhi=rhi[np.isfinite(rhi)]; ral=ral[np.isfinite(ral)]
    nhi,_=np.histogram(rhi,BINS); nal,_=np.histogram(ral,BINS)
    Pbin=nal/nal.sum()                       # P(rvert bin)
    # f(bin)=P(>0.8|bin): n_hi(bin) is the FULL high-score count in bin (we read all hi);
    # scale all-sample to N: expected all-in-bin = Pbin*N.  f = n_hi(bin)/(Pbin*N)
    fbin=np.where(Pbin>0, nhi/(Pbin*N+1e-9), 0.0)
    print(f'{tag}: N={N:,} K(>0.8)={K} overall={K/N:.4%} | hi read={len(rhi)} all read={len(ral)}')
    return dict(N=N,K=K,Pbin=Pbin,fbin=fbin,nhi=nhi,nal=nal)

ex=domain('exp'); mc=domain('mc')
c.close()

# decomposition
tot = (ex['Pbin']*ex['fbin']).sum() / ((mc['Pbin']*mc['fbin']).sum()+1e-12)
A = (ex['Pbin']*mc['fbin']).sum() / ((mc['Pbin']*mc['fbin']).sum()+1e-12)   # exp dist, mc rates
B = (mc['Pbin']*ex['fbin']).sum() / ((mc['Pbin']*mc['fbin']).sum()+1e-12)   # mc dist, exp rates
rows=[]
for i,l in enumerate(LAB):
    rows.append(dict(rvert_bin=l, P_exp=round(ex['Pbin'][i],3), P_mc=round(mc['Pbin'][i],3),
                     f_exp=ex['fbin'][i], f_mc=mc['fbin'][i],
                     ratio_f_exp_mc=round(ex['fbin'][i]/mc['fbin'][i],2) if mc['fbin'][i]>0 else np.nan))
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/ab_decomposition.csv',index=False)
print('\n'+tab.to_string(index=False))
print(f'\nTOTAL excess (reconstructed) = {tot:.2f}')
print(f'(A) angular-distribution only = {A:.2f}   (B) per-angle-gap only = {B:.2f}   A*B={A*B:.2f}')
with open(HERE/'tables/ab_decomposition.csv','a') as f:
    f.write(f'# total_excess={tot:.3f}, A_dist={A:.3f}, B_perangle={B:.3f}\n')

# figure
ctr=[0.35,0.6,0.85,1.25,2.0]
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(ctr, ex['fbin'], 'o-', color='black', lw=2, label='exp')
ax.plot(ctr, mc['fbin'], 's--', color='tab:red', lw=2, label='MC muatm')
ax.set_yscale('log'); ax.set_xlabel('rvert = std_z/std_xy  (low = horizontal)')
ax.set_ylabel('frac(score>0.8 | rvert)')
ax.set_title(f'exp vs MC-muatm false-neutrino rate by verticality\nexcess={tot:.1f}× = (A) dist {A:.1f}× x (B) per-angle {B:.1f}×')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(HERE/'figures/ab_decomposition.png',dpi=130)
print('saved figures/ab_decomposition.png')
