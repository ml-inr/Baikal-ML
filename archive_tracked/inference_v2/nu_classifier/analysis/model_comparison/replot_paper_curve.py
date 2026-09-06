#!/usr/bin/env python
"""Re-plot the paper suppression curve from the saved CSV (no re-query), adding the
'expected' line where exp is suppressed exactly like MC muatm (exp survival = muon
survival = μ → no neutrino-like excess). On the exp-survival vs muon-rejection panel that
is the diagonal y = 1/x (slope -1 in log-log). Any curve ABOVE it has an excess."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
tab = pd.read_csv(HERE/'tables/paper_suppression_curve.csv')

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
# --- panel 1: exp survival vs muon rejection, with the y=1/x expected (no-excess) line ---
xr = np.array([tab.muon_rejection.min(), tab.muon_rejection.max()])
ax[0].plot(xr, 1.0/xr, ls='--', color='grey', lw=1.8, zorder=1,
           label='expected: exp = muon (no excess)')
for tag, name, cl in [('E1','E1 (base)','tab:blue'), ('FT','Fine-tuned','tab:red')]:
    ax[0].errorbar(tab.muon_rejection, tab[f'{tag}_exp_frac'], yerr=tab[f'{tag}_exp_err'],
                   fmt='o-', color=cl, capsize=3, lw=1.8, zorder=3, label=name)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('muon rejection  (1 / muon survival)')
ax[0].set_ylabel('exp survival fraction (excess)')
ax[0].set_title('Exp neutrino-like excess vs muon rejection (out-of-training)')
ax[0].legend(); ax[0].grid(alpha=0.3, which='both')

# --- panel 2: the excess factor exp/μ directly (ratio=1 = no excess) ---
ax[1].axhline(1.0, ls='--', color='grey', lw=1.8, label='expected: exp/μ = 1 (no excess)')
for tag, name, cl in [('E1','E1 (base)','tab:blue'), ('FT','Fine-tuned','tab:red')]:
    ratio = tab[f'{tag}_exp_frac']/tab.mu_surv
    rerr  = tab[f'{tag}_exp_err']/tab.mu_surv
    ax[1].errorbar(tab.muon_rejection, ratio, yerr=rerr, fmt='o-', color=cl, capsize=3, lw=1.8, label=name)
ax[1].set_xscale('log'); ax[1].set_xlabel('muon rejection  (1 / muon survival)')
ax[1].set_ylabel('excess factor  exp survival / muon survival')
ax[1].set_title('Excess factor (=1 means exp behaves like muon background)')
ax[1].legend(); ax[1].grid(alpha=0.3, which='both')

fig.tight_layout(); fig.savefig(HERE/'figures/paper_suppression_curve.png', dpi=140)
print('saved figures/paper_suppression_curve.png (with expected lines)')
