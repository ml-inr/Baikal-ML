#!/usr/bin/env python
"""Neutrino selection efficiency vs zenith angle Θ and vs primary energy, at a fixed EAS
(muatm) suppression working point, for E1 (base) and the fine-tuned model. ν = nuatm+nue2
(equal-weighted). GT Θ = prime_prty[:,0], GT energy = prime_prty[:,2] (GeV). Reproduces the
paper's Fig 5.3(b,c) for the E1→FT model pair. Outputs figures/eff_theta_energy.png +
tables/eff_theta_energy.csv."""
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt

plt.rcParams.update({"font.size":20, "axes.labelsize":20, "axes.titlesize":18,
                     "legend.fontsize":15, "xtick.labelsize":16, "ytick.labelsize":16, "figure.dpi":120})

ROOT = Path(__file__).resolve().parents[4]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003)
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
MODELS={'base':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010',
        'Fine-tuned':'260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'}
SUPP=1e6  # EAS suppression working point

def load(ck):
    c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
    c.execute(f"ATTACH '{PREDS/ck/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    cut=c.execute(f"SELECT quantile_cont(pr.score,{1-1/SUPP}) FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk WHERE {CUT} AND ev.data_class='muatm_2020'").fetchone()[0]
    nu=c.execute(f"""SELECT pr.score, ev.data_class base, l.part_key, l.local_idx FROM m.predictions pr
      JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND ev.data_class IN ('nuatm_2020','nue2_2020')""").df()
    c.close(); return cut, nu

def read_truth(nu):
    theta=np.full(len(nu),np.nan); energy=np.full(len(nu),np.nan)
    f=h5py.File(MCH5,'r',**RDCC)
    for (base,pk),idx in nu.groupby(['base','part_key']).groups.items():
        k=f'{base}/prime_prty/{pk}/data'
        if k not in f: continue
        pp=f[k]; loc=nu.loc[idx,'local_idx'].to_numpy(); ok=loc<pp.shape[0]
        arr=pp[:]
        theta[np.array(idx)[ok]]=arr[loc[ok],0]; energy[np.array(idx)[ok]]=arr[loc[ok],2]
    f.close(); return theta,energy

res={}
for name,ck in MODELS.items():
    cut,nu=load(ck); th,en=read_truth(nu)
    nu=nu.assign(theta=th,energy=en,passc=(nu.score>cut).astype(float)).dropna(subset=['theta','energy'])
    res[name]=(cut,nu)
    na=int((nu.base=='nuatm_2020').sum()); nc=int((nu.base=='nue2_2020').sum())
    print(f'{name}: cut={cut:.4f}  nu N={len(nu)} (nuatm={na}, nue2={nc})',flush=True)

def eff_binned(nu, col, edges):
    """equal-weighted (nuatm,nue2) efficiency per bin."""
    cen=0.5*(edges[:-1]+edges[1:]); e=[]; err=[]
    for a,b in zip(edges[:-1],edges[1:]):
        vals=[]
        for cls in ['nuatm_2020','nue2_2020']:
            s=nu[(nu.base==cls)&(nu[col]>=a)&(nu[col]<b)]
            if len(s)>=5: vals.append(s.passc.mean())
        if vals: e.append(np.mean(vals)); err.append(np.std(vals)/max(len(vals),1)**0.5 if len(vals)>1 else 0.0)
        else: e.append(np.nan); err.append(0.0)
    return cen,np.array(e),np.array(err)

# theta range: use GT zenith; energy: log10
allth=np.concatenate([res[n][1].theta.to_numpy() for n in MODELS])
th_lo,th_hi=np.nanpercentile(allth,[1,99]); th_edges=np.linspace(th_lo,th_hi,13)
en_edges=np.linspace(1.0,7.0,13)  # log10(E/GeV)
fig,ax=plt.subplots(1,2,figsize=(15,6))
rows=[]
for name,cl,lab in [('base','steelblue','Base network'),('Fine-tuned','crimson','After fine-tuning')]:
    cut,nu=res[name]
    cth,eth,erth=eff_binned(nu,'theta',th_edges)
    le=np.log10(np.clip(nu.energy,1,None)); nu=nu.assign(logE=le)
    cen,een,eren=eff_binned(nu,'logE',en_edges)
    ax[0].errorbar(cth,eth,yerr=erth,fmt='o-',color=cl,lw=1.8,capsize=2,label=lab)
    ax[1].errorbar(cen,een,yerr=eren,fmt='o-',color=cl,lw=1.8,capsize=2,label=lab)
    for c,e in zip(cth,eth): rows.append(dict(model=name,var='theta',x=c,eff=e))
    for c,e in zip(cen,een): rows.append(dict(model=name,var='logE',x=c,eff=e))
for a in ax:
    a.legend(); a.set_ylim(0,1.02); a.minorticks_on()
    a.grid(True,which='major',alpha=0.35); a.grid(True,which='minor',alpha=0.18,ls=':')
    a.tick_params(which='major',length=6); a.tick_params(which='minor',length=3)
ax[0].set_xlabel(r'zenith angle $\Theta$ [deg]'); ax[0].set_ylabel(r'$\nu$ efficiency'); ax[0].set_title('Efficiency vs zenith angle')
ax[1].set_xlabel(r'$\log_{10}(E/\mathrm{GeV})$'); ax[1].set_ylabel(r'$\nu$ efficiency'); ax[1].set_title('Efficiency vs energy')
pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent/'tables/eff_theta_energy.csv',index=False)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent/'figures/eff_theta_energy.png',dpi=140)
print('saved figures/eff_theta_energy.png')
