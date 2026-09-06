#!/usr/bin/env python
"""Would feeding the per-hit SN prob help the classifier flag suspicious noise hits? Only if
passed NOISE hits have a LOWER / borderline SN prob than real muon hits. Test on MC muatm
(truth labels): among the filtered hits (prob>0.8) of high-FT-score muatm events, compare the
SN-prob distribution of truth-noise hits vs truth-muon hits, and specifically for ISOLATED
noise hits (far z-outliers — the case-1 topology-flippers)."""
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
FT='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PREDS/FT/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
mu=c.execute(f"""SELECT l.part_key, l.local_idx FROM m.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND ev.data_class='muatm_2020' AND pr.score>0.8 ORDER BY random() LIMIT 6000""").df()
c.close()

noise_p=[]; muon_p=[]; iso_noise_p=[]; iso_muon_p=[]   # prob of passed hits by truth, and for z-outliers
f=h5py.File(MCH5,'r',**RDCC); fp=h5py.File(MCPROBS,'r',**RDCC)
for pk,idx in mu.groupby('part_key').groups.items():
    try:
        es=f[f'muatm_2020/raw/ev_starts/{pk}/data'][:]; ds=f[f'muatm_2020/raw/data/{pk}/data']
        lbl=f[f'muatm_2020/raw/labels/{pk}/data']; pr=fp[f'muatm_2020/probs/{pk}/data']
    except KeyError: continue
    for l in mu.loc[idx,'local_idx'].to_numpy():
        if l+1>=len(es): continue
        s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
        if m.sum()<2: continue
        pf=p[m]; hh=ds[s:e].astype(np.float32)[m]; lab=lbl[s:e][m]
        is_noise=lab<=0; z=hh[:,4]
        noise_p.extend(pf[is_noise].tolist()); muon_p.extend(pf[~is_noise].tolist())
        # isolated = the single largest z-outlier hit of the event
        j=int(np.argmax(np.abs(z-np.median(z))))
        (iso_noise_p if is_noise[j] else iso_muon_p).append(float(pf[j]))
f.close(); fp.close()
noise_p=np.array(noise_p); muon_p=np.array(muon_p); iso_noise_p=np.array(iso_noise_p); iso_muon_p=np.array(iso_muon_p)

def stats(a,name):
    if len(a)==0: print(f'  {name}: (none)'); return
    print(f'  {name}: N={len(a):>7}  median={np.median(a):.4f}  frac>0.9={np.mean(a>0.9):.3f}  frac>0.95={np.mean(a>0.95):.3f}  frac>0.99={np.mean(a>0.99):.3f}  frac<0.85={np.mean(a<0.85):.3f}')
print('=== SN prob of PASSED hits (prob>0.8) in high-FT-score muatm, by truth label ===')
stats(muon_p,  'truth-MUON hits ')
stats(noise_p, 'truth-NOISE hits')
print('\n=== the single largest z-outlier hit per event (the case-1 topology-flipper) ===')
stats(iso_muon_p,  'z-outlier is MUON ')
stats(iso_noise_p, 'z-outlier is NOISE')
print(f'\nInterpretation: if truth-NOISE probs ~ truth-MUON probs (both near 1) -> SN prob is')
print(f'NOT informative, feeding it will not help flag suspicious hits (user hypothesis).')
print(f'If NOISE probs sit lower/borderline -> SN prob IS informative and worth feeding.')
