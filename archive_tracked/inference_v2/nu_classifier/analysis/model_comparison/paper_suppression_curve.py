#!/usr/bin/env python
"""Paper-quality muon-suppression curve: E1 (base) vs FINETUNED, strictly OUT-OF-TRAINING,
down to muon survival 1e-5, with Poisson error bars.

For each model, scores come from its persisted prediction DBs (muatm_2020 = muon cut,
nue2_2020 = signal efficiency, exp_full = the neutrino-like excess), all h8s3. Training
events are removed via the DA-training NPY back-links (part_key|local_idx), identically for
both models. At each MC-muon survival μ: cut = quantile(muatm, 1-μ); we report exp survival
(excess) and MC-ν (nue2) signal efficiency at that cut, each with a Poisson error from its
survivor count. Curve limited to where ≥ a few muons define the cut (1e-5 with ~2020 stats
is statistics-limited — shown with errors, not claimed as a precise measurement).

Outputs tables/paper_suppression_curve.csv + figures/paper_suppression_curve.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[4]
PREDS = ROOT/'inference_v2/nu_classifier/preds'
CAT = ROOT/'data_manager/catalog_v2.duckdb'
MC_TRAIN_NPY = ROOT/'data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8'
EXP_TRAIN_NPY = ROOT/'data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8'
CUT='n_sn_hits>=8 AND n_sn_strings>=3'
MODELS={
 'E1 (base)':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010',
 'Fine-tuned':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model',
}
MU_GRID=[1e-1,1e-2,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6]


def train_keys(npy_dir, part_f, loc_f):
    p=np.load(npy_dir/part_f, allow_pickle=True); l=np.load(npy_dir/loc_f)
    return set(f"{a}|{b}" for a,b in zip(p.astype(str), l.astype(np.int64)))

print("loading training back-links (out-of-training exclusion)…", flush=True)
MC_TRAIN = train_keys(MC_TRAIN_NPY, 'h5_part_keys.npy', 'h5_local_event_ids.npy')
EXP_TRAIN = train_keys(EXP_TRAIN_NPY, 'exp_h5_part_keys.npy', 'exp_h5_local_event_ids.npy')
print(f"  MC train keys={len(MC_TRAIN):,}  exp train keys={len(EXP_TRAIN):,}", flush=True)


def load(ckpt_dir):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PREDS/ckpt_dir/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{PREDS/ckpt_dir/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    def mc(cls):
        d=c.execute(f"""SELECT pr.score, l.part_key, l.local_idx FROM m.predictions pr
          JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
          WHERE {CUT} AND ev.data_class='{cls}'""").df()
        key=d.part_key.astype(str)+'|'+d.local_idx.astype(str)
        return d.score.to_numpy()[~key.isin(MC_TRAIN).to_numpy()]
    d=c.execute(f"""SELECT pr.score, l.part_key, l.local_idx FROM e.predictions pr
      JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249'))""").df()
    key=d.part_key.astype(str)+'|'+d.local_idx.astype(str)
    ex=d.score.to_numpy()[~key.isin(EXP_TRAIN).to_numpy()]
    mu=mc('muatm_2020'); nu=mc('nue2_2020'); c.close(); return mu,nu,ex

S={}
for name,ck in MODELS.items():
    if not (PREDS/ck/'mc_merged_thr0p8.duckdb').exists():
        print(f"!! {name}: MC DB not ready yet ({ck}) — run again when predict_mc finishes"); sys.exit(1)
    S[name]=load(ck); print(f"{name}: out-of-training muatm={len(S[name][0]):,} nue2={len(S[name][1]):,} exp={len(S[name][2]):,}",flush=True)

rows=[]
for mu in MU_GRID:
    r={'mu_surv':mu,'muon_rejection':1.0/mu}
    for name in MODELS:
        m,n,e=S[name]; tag='E1' if name.startswith('E1') else 'FT'
        cut=np.quantile(m,1-mu); k_e=int((e>cut).sum()); k_n=int((n>cut).sum()); k_m=int((m>cut).sum())
        r[f'{tag}_exp_frac']=k_e/len(e); r[f'{tag}_exp_err']=np.sqrt(max(k_e,1))/len(e)
        r[f'{tag}_sig_eff']=k_n/len(n); r[f'{tag}_sig_err']=np.sqrt(max(k_n,1))/len(n)
        r[f'{tag}_n_mu_at_cut']=k_m   # cut reliability (<~10 => statistics-limited)
    rows.append(r)
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/paper_suppression_curve.csv',index=False)
pd.set_option('display.width',240,'display.max_columns',40)
print('\n=== E1 vs Fine-tuned — out-of-training, muon suppression to 1e-5 ===')
print(tab.to_string(index=False,float_format=lambda x:f'{x:.4g}'))

fig,ax=plt.subplots(1,2,figsize=(14,5.2))
for tag,name,cl in [('E1','E1 (base)','tab:blue'),('FT','Fine-tuned','tab:red')]:
    ax[0].errorbar(tab.muon_rejection,tab[f'{tag}_exp_frac'],yerr=tab[f'{tag}_exp_err'],fmt='o-',color=cl,capsize=3,label=name)
ax[0].set_xscale('log');ax[0].set_yscale('log')
ax[0].set_xlabel('muon rejection  (1 / muon survival)');ax[0].set_ylabel('exp survival fraction (excess)')
ax[0].set_title('Exp neutrino-like excess vs muon rejection (out-of-training)');ax[0].legend();ax[0].grid(alpha=0.3,which='both')
for tag,name,cl in [('E1','E1 (base)','tab:blue'),('FT','Fine-tuned','tab:red')]:
    ax[1].errorbar(tab[f'{tag}_sig_eff'],tab[f'{tag}_exp_frac'],yerr=tab[f'{tag}_exp_err'],xerr=tab[f'{tag}_sig_err'],fmt='o-',color=cl,capsize=3,label=name)
ax[1].set_yscale('log');ax[1].set_xlabel('MC-ν (nue2) signal efficiency');ax[1].set_ylabel('exp survival fraction (excess)')
ax[1].set_title('Excess vs signal efficiency');ax[1].legend();ax[1].grid(alpha=0.3,which='both')
fig.tight_layout();fig.savefig(HERE/'figures/paper_suppression_curve.png',dpi=140)
print('saved figures/paper_suppression_curve.png')
