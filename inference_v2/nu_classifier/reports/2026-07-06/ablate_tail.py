"""Does the exp high score CAUSALLY depend on the unused (tail) MC-PC dims?
Reconstruct embeddings keeping only top-K MC PCs, re-run the classifier head,
see if exp_hi score collapses (=> excess is tail-driven, reducing dim helps) or
stays high (=> decision is on the used axis, cutting dims = backfire)."""
import sys; from pathlib import Path
import duckdb, h5py, numpy as np, torch
ROOT=Path('/home/albert/Baikal2025'); sys.path.insert(0,str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings
PRED=ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CKPT=ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
CAT=ROOT/'data_manager/catalog_v2.duckdb'; RDCC=dict(rdcc_nbytes=64*1024*1024,rdcc_nslots=1_000_003); THR=0.8; DEV='cuda:3'
def read_hits(h5,pr,df):
    fs=[None]*len(df); f=h5py.File(h5,'r',**RDCC); fp=h5py.File(pr,'r',**RDCC)
    for (b,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{b}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{b}/raw/data/{pk}/data']; p=fp[f'{b}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); pp=p[s:e].astype(np.float32); m=pp>THR
            if m.sum()<2: continue
            fs[row]=ds[s:e].astype(np.float32)[m]
    f.close(); fp.close(); return fs
c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)"); c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)"); c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
mc=c.execute("SELECT ev.data_class base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE pr.n_sn_hits>=5 AND ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020') ORDER BY random() LIMIT 8000").df()
nu=c.execute("SELECT ev.data_class base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE pr.n_sn_hits>=5 AND ev.data_class IN ('nuatm_2020','nue2_2020') AND pr.score>0.8 ORDER BY random() LIMIT 2000").df()
ex=c.execute("SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE pr.n_sn_hits>=5 AND NOT(ev.cluster=2 AND ev.run IN ('20','249')) AND pr.score>0.8 ORDER BY random() LIMIT 1500").df()
c.close()
model,norm,_=load_model(str(CKPT),device=DEV)
def emb(df,h5,pr):
    fs=read_hits(h5,pr,df); k=[i for i,x in enumerate(fs) if x is not None]
    s,E=predict_scores_and_embeddings(model,[fs[i] for i in k],norm,batch_size=512,device=DEV); return E.astype(np.float64), s
Emc,_=emb(mc,MCH5,MCPROBS); Enu,snu=emb(nu,MCH5,MCPROBS); Eex,sex=emb(ex,EXH5,EXPROBS)
mu=Emc.mean(0); U,S,Vt=np.linalg.svd(Emc-mu,full_matrices=False)
head=model.classifier.to(DEV).eval()
def rescore(E,K):
    P=(E-mu)@Vt[:K].T; Erec=mu+P@Vt[:K]           # keep top-K MC PCs
    with torch.no_grad():
        lg=head(torch.tensor(Erec,dtype=torch.float32,device=DEV)).view(-1)
        return torch.sigmoid(lg).cpu().numpy()
print(f'baseline mean score: exp_hi={sex.mean():.3f}  MC_nu={snu.mean():.3f}')
print(f'{"K (top MC PCs)":>16} | exp_hi mean score | exp_hi frac>0.8 | MC_nu mean | MC_nu frac>0.8')
for K in [2,8,16,26,64,128]:
    se=rescore(Eex,K); sn=rescore(Enu,K)
    print(f'{K:>16} | {se.mean():.3f}            | {np.mean(se>0.8):.3f}          | {sn.mean():.3f}     | {np.mean(sn>0.8):.3f}')
