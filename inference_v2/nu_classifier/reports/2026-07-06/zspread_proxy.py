"""Lean: MC high-score + exp score buckets (proxy already validated: corr 0.81)."""
import duckdb, h5py, numpy as np
from pathlib import Path
ROOT=Path('/home/albert/Baikal2025')
CAT=ROOT/'data_manager/catalog_v2.duckdb'
PRED=ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=8 and n_sn_strings>=3'

def zproxy(h5path,probs_path,base,df):
    z=np.full(len(df),np.nan); rv=np.full(len(df),np.nan)
    f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for pk,idx in df.groupby('part_key').groups.items():
        try:
            es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']
            pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            xyz=ds[s:e,2:5].astype(np.float32)[m]
            sx,sy,sz=xyz[:,0].std(),xyz[:,1].std(),xyz[:,2].std()
            z[row]=sz; rv[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3)
    f.close(); fp.close(); return z,rv

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

def show(lab,df,base,h5,pr):
    df=df.reset_index(drop=True); z,rv=zproxy(h5,pr,base,df); df=df.assign(z_std=z,rvert=rv).dropna(subset=['z_std'])
    print(f'{lab:26s}: N={len(df):>4} | z_std med={df.z_std.median():5.1f} | rvert med={df.rvert.median():4.2f}')

# MC references (targeted)
mn=c.execute(f"SELECT pr.score,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020' JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND pr.score>0.8").df()
show('MC muon score>0.8 (false)', mn,'muatm_2020',MCH5,MCPROBS)
mnu=c.execute(f"SELECT pr.score,l.part_key,l.local_idx,ev.data_class FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class in ('nuatm_2020','nue2_2020') JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND pr.score>0.8 ORDER BY random() LIMIT 2000").df()
for cls in ['nuatm_2020','nue2_2020']:
    show(f'MC {cls} score>0.8 (real)', mnu[mnu.data_class==cls],cls,MCH5,MCPROBS)

# exp buckets
print()
def eq(where,n,samp=False):
    tail=f"USING SAMPLE {n}" if samp else f"ORDER BY random() LIMIT {n}"
    return c.execute(f"SELECT pr.score,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) AND {where} {tail}").df()
for lab,w,n,s in [('exp score>0.8',"pr.score>0.8",2000,False),('exp 0.6-0.8',"pr.score>0.6 and pr.score<0.8",2000,False),
                  ('exp 0.4-0.6',"pr.score>0.4 and pr.score<0.6",2000,False),('exp score<0.2',"pr.score<0.2",2500,True)]:
    show(lab, eq(w,n,s),'exp_full',EXH5,EXPROBS)
c.close()
print('\n(ref MC truth: horizon θ95-105 rvert~0.60, steep θ150-180 rvert~2.39; corr(rvert,θ)=0.83)')
