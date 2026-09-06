"""Does exp have MORE horizontal events than MC muatm overall? (explains excess?)"""
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
    rv=np.full(len(df),np.nan)
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
            rv[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3)
    f.close(); fp.close(); return rv

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

def summ(lab,df,base,h5,pr):
    df=df.reset_index(drop=True); rv=zproxy(h5,pr,base,df); rv=rv[np.isfinite(rv)]
    print(f'{lab:22s}: N={len(rv):>4} | rvert med={np.median(rv):4.2f} | frac<0.7(horiz)={np.mean(rv<0.7):.3f} | frac<0.5={np.mean(rv<0.5):.3f} | frac>1.5(vert)={np.mean(rv>1.5):.3f}')

# ALL events (any score), sampled — the INPUT verticality distribution
mu=c.execute(f"SELECT l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020' JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} ORDER BY random() LIMIT 3000").df()
summ('MC muatm ALL', mu,'muatm_2020',MCH5,MCPROBS)
ex=c.execute(f"SELECT l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) ORDER BY random() LIMIT 3000").df()
summ('EXP ALL', ex,'exp_full',EXH5,EXPROBS)
c.close()
print('\nIf EXP frac<0.7 >> MC muatm frac<0.7 -> exp is more horizontal -> (A) angular-distribution difference drives the excess.')
