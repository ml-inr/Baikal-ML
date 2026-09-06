#!/usr/bin/env python
"""Score-distribution histograms before (E1) and after (FT) the fine-tune, on the clean
out-of-training exp pools, with MC muatm (the background the classifier should mimic) for
reference. Shows the neutrino-like tail of exp shrinking toward the muon level after FT.
Outputs figures/score_hist_before_after.png."""
from pathlib import Path
import duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
E1='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
FT='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
CUT='n_sn_hits>=8 AND n_sn_strings>=3'

def scores(ckpt, which, n=None):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    db = 'exp_full' if which=='exp' else 'mc_merged'
    c.execute(f"ATTACH '{PREDS/ckpt/(db+'_thr0p8.duckdb')}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    if which=='exp':
        w=f"{CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249'))"
    else:
        w=f"{CUT} AND ev.data_class='muatm_2020'"
    lim=f"USING SAMPLE {n}" if n else ""
    s=c.execute(f"SELECT pr.score FROM p.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {w} {lim}").df()['score'].to_numpy()
    c.close(); return s

e1_exp=scores(E1,'exp'); ft_exp=scores(FT,'exp')
e1_mu =scores(E1,'muatm',500000); ft_mu=scores(FT,'muatm',500000)
print(f'E1 exp={len(e1_exp):,} FT exp={len(ft_exp):,} E1 muatm={len(e1_mu):,} FT muatm={len(ft_mu):,}')

b=np.linspace(0,1,60)
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
# panel A: exp before vs after
ax[0].hist(e1_exp,bins=b,density=True,histtype='step',lw=2,color='tab:blue',label=f'exp — E1 (base)')
ax[0].hist(ft_exp,bins=b,density=True,histtype='step',lw=2,color='tab:red',label=f'exp — Fine-tuned')
ax[0].set_yscale('log'); ax[0].set_xlabel('nu-classifier score'); ax[0].set_ylabel('density (log)')
ax[0].set_title('Exp score distribution: before vs after fine-tune'); ax[0].legend(); ax[0].grid(alpha=0.3,which='both')
# panel B: exp vs muatm, before (E1) and after (FT) — the excess tail vs the muon background
ax[1].hist(e1_mu, bins=b,density=True,histtype='step',lw=2,color='tab:blue',ls=':',label='muatm — E1')
ax[1].hist(e1_exp,bins=b,density=True,histtype='step',lw=2,color='tab:blue',label='exp — E1')
ax[1].hist(ft_mu, bins=b,density=True,histtype='step',lw=2,color='tab:red',ls=':',label='muatm — FT')
ax[1].hist(ft_exp,bins=b,density=True,histtype='step',lw=2,color='tab:red',label='exp — FT')
ax[1].set_yscale('log'); ax[1].set_xlabel('nu-classifier score'); ax[1].set_ylabel('density (log)')
ax[1].set_title('Exp vs MC muatm (exp tail should match muon background)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3,which='both')
fig.tight_layout(); fig.savefig(HERE/'figures/score_hist_before_after.png',dpi=140)
print('saved figures/score_hist_before_after.png')
# quick tail numbers
for t in [0.3,0.5,0.8]:
    print(f'  score>{t}: E1 exp {np.mean(e1_exp>t):.2e}, FT exp {np.mean(ft_exp>t):.2e}, E1 muatm {np.mean(e1_mu>t):.2e}, FT muatm {np.mean(ft_mu>t):.2e}')
