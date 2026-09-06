#!/usr/bin/env python
"""Paper figures for §5.3 (E1 base → fine-tuned):
 (A) EAS suppression vs equal-weighted ν efficiency, E1 vs FT (the ROC; FT preserves it).
 (B) Score distributions (MC EAS=muatm, atm ν=nuatm, cosmo ν=nue2, exp), E1 vs FT — the exp
     high-score excess collapses to the MC-EAS level after fine-tuning.
Outputs figures/paper_suppression_vs_eff.png, figures/paper_score_distr_E1FT.png,
tables/paper_suppression_vs_eff.csv."""
from pathlib import Path
import duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt

plt.rcParams.update({"font.size":20, "axes.labelsize":20, "axes.titlesize":18,
                     "legend.fontsize":15, "xtick.labelsize":16, "ytick.labelsize":16, "figure.dpi":120})
PTYPE = {"muatm_2020":"steelblue", "nuatm_2020":"forestgreen", "nue2_2020":"darkorange", "exp_reco":"crimson"}

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[4]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
MODELS={'base':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010',
        'Fine-tuned':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'}
SURV=[3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6]

def con(ck):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PREDS/ck/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{PREDS/ck/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)"); return c

# ---- (A) suppression vs equal-weighted efficiency (SQL grid) ----
rows=[]
for name,ck in MODELS.items():
    c=con(ck)
    def eff(cls,cut): return c.execute(f"SELECT avg(CASE WHEN pr.score>{cut} THEN 1.0 ELSE 0.0 END) FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND ev.data_class='{cls}'").fetchone()[0]
    for s in SURV:
        cut=c.execute(f"SELECT quantile_cont(pr.score,{1-s}) FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND ev.data_class='muatm_2020'").fetchone()[0]
        ea,ec=eff('nuatm_2020',cut),eff('nue2_2020',cut)
        rows.append(dict(model=name,supp=1.0/s,eff=0.5*(ea+ec),nuatm=ea,nue2=ec,cut=cut))
    c.close(); print(f'{name}: suppression-vs-eff done',flush=True)
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/paper_suppression_vs_eff.csv',index=False)

fig,ax=plt.subplots(figsize=(8,6.5))
for name,cl,lab in [('base','steelblue','Base network'),('Fine-tuned','crimson','After fine-tuning')]:
    t=tab[tab.model==name].sort_values('eff')
    ax.plot(t.eff,t.supp,'o-',color=cl,lw=1.8,label=lab)
ax.axhline(1e6,ls=':',color='grey'); ax.set_yscale('log')
ax.set_xlabel(r'$\nu$ efficiency'); ax.set_ylabel('EAS suppression factor')
ax.legend(); ax.minorticks_on(); ax.grid(True,which='major',alpha=0.35); ax.grid(True,which='minor',alpha=0.18,ls=':')
ax.tick_params(which='major',length=6); ax.tick_params(which='minor',length=3)
fig.tight_layout(); fig.savefig(HERE/'figures/paper_suppression_vs_eff.png'); print('saved paper_suppression_vs_eff.png')

# ---- (B) score distributions, E1 vs FT ----
def scores(ck):
    c=con(ck); out={}
    for lab,w in [('EAS (muatm)',"ev.data_class='muatm_2020' USING SAMPLE 2000000"),
                  ('atm ν (nuatm)',"ev.data_class='nuatm_2020'"),
                  ('cosmo ν (nue2)',"ev.data_class='nue2_2020'"),
                  ('experiment',"NOT(ev.cluster=2 AND ev.run IN('20','249')) AND ev.source='exp_full' USING SAMPLE 1000000")]:
        src='e.predictions' if 'exp' in w else 'm.predictions'
        out[lab]=c.execute(f"SELECT pr.score FROM {src} pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND {w}").df()['score'].to_numpy()
    c.close(); return out
S={n:scores(ck) for n,ck in MODELS.items()}
# (color, display label, linestyle) per S-key; experiment last (drawn on top, dashed)
STYLE=[('EAS (muatm)',PTYPE['muatm_2020'],'MC EAS','-'),
       ('atm ν (nuatm)',PTYPE['nuatm_2020'],r'MC $\nu_\mu^{atm}$','-'),
       ('cosmo ν (nue2)',PTYPE['nue2_2020'],r'MC $\nu_\mu^{cosm}$','-'),
       ('experiment',PTYPE['exp_reco'],'Experimental','--')]
TITLE={'base':'Base network','Fine-tuned':'After fine-tuning'}
b=np.linspace(0,1,51)
fig,ax=plt.subplots(1,2,figsize=(15,6),sharey=True)
for i,(name,_) in enumerate(MODELS.items()):
    for key,cl,lab,ls in STYLE:
        ax[i].hist(S[name][key],bins=b,density=True,histtype='step',lw=1.8,color=cl,ls=ls,label=lab)
    ax[i].set_yscale('log'); ax[i].set_xlabel(r'$\xi$'); ax[i].set_title(TITLE.get(name,name)); ax[i].set_xlim(0,1)
    ax[i].minorticks_on(); ax[i].grid(True,which='major',alpha=0.35); ax[i].grid(True,which='minor',alpha=0.18,ls=':')
    ax[i].tick_params(which='major',length=6); ax[i].tick_params(which='minor',length=3)
ax[0].set_ylabel('Density'); ax[0].legend()
fig.tight_layout(); fig.savefig(HERE/'figures/paper_score_distr_E1FT.png'); print('saved paper_score_distr_E1FT.png')
