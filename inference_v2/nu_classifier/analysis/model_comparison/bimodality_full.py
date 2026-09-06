#!/usr/bin/env python
"""Bimodality of the FT score distribution on the FULL exp_full (3.11M h8s3 good-run
events): does a super-confident peak of genuine ν candidates separate from the muon
background at high score? Compare FT exp vs FT MC-muatm (background) score shapes in the
high tail, and the exp/muatm ratio vs score (a rising ratio toward 1.0 = a super-confident
excess above background = the bimodality signature). E1 exp shown for 'before' reference.
Outputs figures/bimodality_full.png + tables/bimodality_full.csv."""
from pathlib import Path
import duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
FT='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
E1='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'; GOOD="NOT(ev.cluster=2 AND ev.run IN('20','249'))"

def q(ckpt, db, where, col='pr.score'):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PREDS/ckpt/(db+'_thr0p8.duckdb')}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    r=c.execute(f"SELECT {col} s FROM p.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {where}").df()['s'].to_numpy()
    c.close(); return r

def tot(ckpt, db, where):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PREDS/ckpt/(db+'_thr0p8.duckdb')}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    n=c.execute(f"SELECT count(*) FROM p.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {where}").fetchone()[0]
    c.close(); return n

ft_exp = q(FT,'exp_full', f"{CUT} AND {GOOD} AND pr.score>0.4");  ft_exp_tot = tot(FT,'exp_full', f"{CUT} AND {GOOD}")
e1_exp = q(E1,'exp_full', f"{CUT} AND {GOOD} AND pr.score>0.4");  e1_exp_tot = tot(E1,'exp_full', f"{CUT} AND {GOOD}")
ft_mu  = q(FT,'mc_merged', f"{CUT} AND ev.data_class='muatm_2020' AND pr.score>0.4"); ft_mu_tot = tot(FT,'mc_merged', f"{CUT} AND ev.data_class='muatm_2020'")
print(f'FT exp tot={ft_exp_tot:,} (>0.4: {len(ft_exp)}, >0.8: {int((ft_exp>0.8).sum())}, >0.9: {int((ft_exp>0.9).sum())}, >0.95: {int((ft_exp>0.95).sum())})')
print(f'E1 exp tot={e1_exp_tot:,} (>0.8: {int((e1_exp>0.8).sum())})')
print(f'FT muatm tot={ft_mu_tot:,} (>0.4: {len(ft_mu)}, >0.8: {int((ft_mu>0.8).sum())})')

# ratio table
print(f'\n{"score>":>7} {"FT exp N":>9} {"FT exp frac":>12} {"muatm frac":>11} {"exp/muatm":>10}')
rows=[]
for t in [0.5,0.7,0.8,0.9,0.95,0.97,0.99,0.995]:
    en=int((ft_exp>t).sum()); mn=int((ft_mu>t).sum()); ef=en/ft_exp_tot; mf=mn/ft_mu_tot
    rows.append(dict(score=t,ft_exp_N=en,ft_exp_frac=ef,muatm_frac=mf,ratio=ef/mf if mf>0 else np.nan))
    print(f'{t:>7.3f} {en:>9} {ef:>12.2e} {mf:>11.2e} {ef/mf if mf>0 else np.nan:>10.2f}')
pd.DataFrame(rows).to_csv(HERE/'tables/bimodality_full.csv',index=False)

fig,ax=plt.subplots(1,3,figsize=(18,5))
# (a) normalized densities in the tail, FT exp vs muatm vs E1 exp
b=np.linspace(0.4,1.0,61)
ax[0].hist(ft_exp,bins=b,density=True,histtype='step',lw=2,color='tab:red',label=f'FT exp (n>0.4={len(ft_exp)})')
ax[0].hist(ft_mu, bins=b,density=True,histtype='step',lw=2,color='grey',ls=':',label=f'FT muatm (bg)')
ax[0].hist(e1_exp,bins=b,density=True,histtype='step',lw=2,color='tab:blue',alpha=0.6,label=f'E1 exp (before)')
ax[0].set_yscale('log'); ax[0].set_xlabel('score'); ax[0].set_ylabel('density'); ax[0].set_title('High-tail score density'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3,which='both')
# (b) exp/muatm ratio vs score threshold (rising toward 1 = super-confident excess)
r=pd.DataFrame(rows)
ax[1].plot(r.score,r.ratio,'o-',color='tab:red'); ax[1].axhline(1,ls='--',color='grey')
ax[1].set_xlabel('score >'); ax[1].set_ylabel('exp/muatm survival ratio'); ax[1].set_title('Excess factor vs score (rise = super-confident ν peak?)'); ax[1].grid(alpha=0.3)
# (c) zoom [0.85,1.0] counts of FT exp (look for valley + peak)
bz=np.arange(0.85,1.001,0.01)
hc,_=np.histogram(ft_exp,bins=bz)
ax[2].step(bz[:-1],hc,where='post',color='tab:red',lw=2,label='FT exp counts')
# muatm scaled to same total in [0.85,1] for shape comparison
hm,_=np.histogram(ft_mu,bins=bz); hm=hm*(hc.sum()/max(hm.sum(),1))
ax[2].step(bz[:-1],hm,where='post',color='grey',ls=':',lw=2,label='muatm (shape, scaled)')
ax[2].set_xlabel('score'); ax[2].set_ylabel('FT exp count'); ax[2].set_title('Zoom [0.85,1.0]: valley then super-confident peak?'); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/bimodality_full.png',dpi=140)
print('\nsaved figures/bimodality_full.png')
# print the fine counts for [0.85,1.0]
print('\nFT exp fine counts [0.85,1.00]:')
for i in range(len(hc)):
    print(f'  [{bz[i]:.2f},{bz[i+1]:.2f}): {int(hc[i]):>3}  '+'#'*int(hc[i]))
