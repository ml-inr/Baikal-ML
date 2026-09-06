#!/usr/bin/env python
"""Replot the distance-to-nu diagnostic from the cached per-event table
(tables/dist_to_nu.csv, produced by build_dist_to_nu.py). No GPU needed — tweak styling
here freely. Panel (a): distance to the MC-nu manifold per population. Panel (b): distance
to the EAS manifold vs distance to the nu manifold, experiment coloured by score."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

plt.rcParams.update({"font.size":20, "axes.labelsize":20, "axes.titlesize":18,
                     "legend.fontsize":14, "xtick.labelsize":16, "ytick.labelsize":16, "figure.dpi":120})
HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE/'tables/dist_to_nu.csv')
P = {k: d[d['pop'] == k] for k in d['pop'].unique()}
NU, EAS = 'forestgreen', 'steelblue'
KEYS = {'nu':'MC $\\nu$', 'eas':'MC EAS', 'bulk':'exp bulk', 'false':'exp false-$\\nu$'}

fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))

# (a) distance to simulated neutrinos
b = np.linspace(0, 13, 70)
for key, cl, lab in [(KEYS['nu'],NU,r'MC $\nu$'), (KEYS['eas'],EAS,'MC EAS'),
                     (KEYS['false'],'crimson',r'experiment ($\xi > 0.8$)'), (KEYS['bulk'],'0.55','experiment (random)')]:
    ax[0].hist(P[key].d_nu, bins=b, density=True, histtype='step', lw=1.8, color=cl, label=lab)
ax[0].set_xlabel(r'distance to simulated $\nu$,  $d_\nu$'); ax[0].set_ylabel('Density')
ax[0].set_title('Distance to simulated neutrinos'); ax[0].legend()
ax[0].minorticks_on(); ax[0].grid(True,which='major',alpha=0.35); ax[0].grid(True,which='minor',alpha=0.18,ls=':')
ax[0].tick_params(which='major',length=6); ax[0].tick_params(which='minor',length=3)

# (b) d_mu vs d_nu, exp coloured by score, faint MC references, class medians
ax[1].scatter(P[KEYS['eas']].d_mu, P[KEYS['eas']].d_nu, s=5, alpha=0.06, color=EAS, edgecolors='none')
ax[1].scatter(P[KEYS['nu']].d_mu,  P[KEYS['nu']].d_nu,  s=5, alpha=0.12, color=NU,  edgecolors='none')
bulk = P[KEYS['bulk']].sort_values('score')
scat = ax[1].scatter(bulk.d_mu, bulk.d_nu, c=bulk.score, s=np.where(bulk.score>0.8,20,5),
                     alpha=0.5, cmap='viridis', vmin=0, vmax=1, edgecolors='none')
ax[1].plot(P[KEYS['nu']].d_mu.median(),   P[KEYS['nu']].d_nu.median(),   '*', ms=20, color=NU,  mec='k', label=r'MC $\nu$ (median)')
ax[1].plot(P[KEYS['eas']].d_mu.median(),  P[KEYS['eas']].d_nu.median(),  '*', ms=20, color=EAS, mec='k', label='MC EAS (median)')
ax[1].plot(P[KEYS['false']].d_mu.median(),P[KEYS['false']].d_nu.median(),'X', ms=14, color='crimson', mec='k', label=r'experiment $\xi{>}0.8$ (median)')
ax[1].set_xlabel(r'distance to simulated EAS,  $d_\mu$'); ax[1].set_ylabel(r'distance to simulated $\nu$,  $d_\nu$')
ax[1].set_title(r'Distance to simulated $\nu$ vs EAS'); ax[1].legend(loc='upper right')
ax[1].minorticks_on(); ax[1].grid(True,which='major',alpha=0.35); ax[1].grid(True,which='minor',alpha=0.18,ls=':')
ax[1].tick_params(which='major',length=6); ax[1].tick_params(which='minor',length=3)
cb = fig.colorbar(scat, ax=ax[1]); cb.set_label(r'classifier score $\xi$')

fig.tight_layout(); fig.savefig(HERE/'figures/nu_classifier_dist_to_nu.png'); print('saved figures/nu_classifier_dist_to_nu.png')
