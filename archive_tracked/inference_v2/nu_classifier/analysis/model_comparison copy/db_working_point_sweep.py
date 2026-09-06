#!/usr/bin/env python
"""Working-point sweep DOWN TO 1e-5 (and beyond), reading scores directly from the
persisted prediction DBs (fast, no h5). Compares E1 (ep10) vs E5 horizon (ep8) at matched
MC-muon survival working points: exp survival (the excess) and MC-nu (nue2) signal
efficiency at each. Answers whether the horizon θ-loss reduces the excess at strict cuts.

Each model uses its OWN prediction DBs (muatm cut + nue2 sig-eff + exp excess), all h8s3.
DBs include training events, but E1 and E5 share the training set, so the E1-vs-E5 relative
comparison is unaffected. Outputs tables/db_working_point_sweep.csv + figure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
PREDS = ROOT/'inference_v2/nu_classifier/preds'
CUT='n_sn_hits>=8 AND n_sn_strings>=3'
MODELS={
 'E1 (ep10)':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010',
 'E5 horizon (ep8)':'260707_0226_da_nu_classifier_exp_full_E5_horizon_lambda0.01@da_checkpoint_epoch_008',
}
MU_GRID=[1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]


def load_scores(ckpt_dir):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PREDS/ckpt_dir/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{PREDS/ckpt_dir/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{ROOT/'data_manager/catalog_v2.duckdb'}' AS cat (READ_ONLY)")
    mu=c.execute(f"SELECT pr.score FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND ev.data_class='muatm_2020'").df()['score'].to_numpy()
    nu=c.execute(f"SELECT pr.score FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND ev.data_class='nue2_2020'").df()['score'].to_numpy()
    ex=c.execute(f"SELECT pr.score FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249'))").df()['score'].to_numpy()
    c.close(); return mu,nu,ex

S={}
for name,ck in MODELS.items():
    S[name]=load_scores(ck)
    print(f"{name}: muatm={len(S[name][0]):,}  nue2={len(S[name][1]):,}  exp={len(S[name][2]):,}",flush=True)

rows=[]
for mu_surv in MU_GRID:
    r={'mu_surv':mu_surv}
    for name in MODELS:
        mu,nu,ex=S[name]
        cut=np.quantile(mu,1.0-mu_surv)
        tag='E1' if name.startswith('E1') else 'E5'
        r[f'{tag}_cut']=cut; r[f'{tag}_sig_eff']=float((nu>cut).mean()); r[f'{tag}_exp_frac']=float((ex>cut).mean())
        r[f'{tag}_n_mu_tail']=int((mu>cut).sum())   # events defining the cut (noise check)
    r['exp_ratio_E5/E1']=r['E5_exp_frac']/r['E1_exp_frac'] if r['E1_exp_frac']>0 else np.nan
    rows.append(r)
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/db_working_point_sweep.csv',index=False)
pd.set_option('display.width',220,'display.max_columns',30)
print('\n=== E1 vs E5 horizon at matched muon working points (from DBs) ===')
print(tab.to_string(index=False,float_format=lambda x:f'{x:.4g}'))
print('\n(exp_frac=excess, lower better; n_mu_tail = #muatm above cut = cut reliability, <10 is noisy)')

fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].plot(tab.mu_surv,tab.E1_exp_frac,'o-',label='E1',color='tab:blue')
ax[0].plot(tab.mu_surv,tab.E5_exp_frac,'s-',label='E5 horizon',color='tab:red')
ax[0].set_xscale('log');ax[0].set_yscale('log');ax[0].invert_xaxis()
ax[0].set_xlabel('MC muon survival (stricter →)');ax[0].set_ylabel('exp survival (excess)')
ax[0].set_title('exp excess vs working point');ax[0].legend();ax[0].grid(alpha=0.3,which='both')
ax[1].plot(tab.E1_sig_eff,tab.E1_exp_frac,'o-',label='E1',color='tab:blue')
ax[1].plot(tab.E5_sig_eff,tab.E5_exp_frac,'s-',label='E5 horizon',color='tab:red')
ax[1].set_yscale('log');ax[1].set_xlabel('MC-nu (nue2) signal efficiency');ax[1].set_ylabel('exp survival (excess)')
ax[1].set_title('exp excess vs signal efficiency');ax[1].legend();ax[1].grid(alpha=0.3,which='both')
fig.tight_layout();fig.savefig(HERE/'figures/db_working_point_sweep.png',dpi=130)
print('saved figures/db_working_point_sweep.png')
