#!/usr/bin/env python
"""Paper figure: the OOD-background selection for fine-tuning.
Reproduces the selection used to build the fine-tune background (encoder-OOD > 2.5 AND
rvert < 0.7) from the pre-computed per-event characterisation (analysis/finetune_set/
tables/finetune_set.csv: 120k random out-of-training exp events, h8s3, with encoder-OOD
= mean kNN(k=20) distance to the MC embedding manifold, verticality rvert = std_z/std_xy,
and classifier score). No GPU needed. Two panels:
 (a) encoder-OOD distribution of experimental events, with the MC in-distribution scale
     (calibration median 0.68, p99 4.10; Report 2026-07-08 §3) and the OOD>2.5 cut;
 (b) rvert vs encoder-OOD, coloured by classifier score, with the selection corner
     (rvert<0.7 AND OOD>2.5) — the near-horizontal, off-manifold events the base network
     over-confidently scores as neutrinos.
Outputs figures/nu_classifier_ood_selection.png."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CSV = HERE.parent/'finetune_set/tables/finetune_set.csv'
OOD_CUT, RV_CUT = 2.5, 0.7
MC_MED, MC_P99 = 0.68, 4.10   # MC calibration OOD (Report 2026-07-08 §3)

d = pd.read_csv(CSV)
sel = (d.ood > OOD_CUT) & (d.rvert < RV_CUT)
print(f'exp sample N={len(d)}; selected background {int(sel.sum())} '
      f'(recall of score>0.8: {((d.score>0.8)&sel).sum()}/{(d.score>0.8).sum()})', flush=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) encoder-OOD distribution
b = np.linspace(0, 8, 70)
ax[0].hist(d.ood, bins=b, color='0.7', label='all experimental events')
ax[0].hist(d.ood[sel], bins=b, color='tab:red', label='selected background')
ax[0].axvline(MC_MED, ls=':', color='tab:blue'); ax[0].text(MC_MED+0.05, ax[0].get_ylim()[1]*0.55, 'MC median', color='tab:blue', rotation=90, fontsize=8, va='top')
ax[0].axvline(MC_P99, ls=':', color='tab:green'); ax[0].text(MC_P99+0.05, ax[0].get_ylim()[1]*0.55, 'MC $p_{99}$', color='tab:green', rotation=90, fontsize=8, va='top')
ax[0].axvline(OOD_CUT, ls='--', color='k'); ax[0].text(OOD_CUT+0.05, ax[0].get_ylim()[1]*0.9, 'cut = 2.5', rotation=90, fontsize=9, va='top')
ax[0].set_yscale('log'); ax[0].set_xlabel('encoder-OOD  (mean kNN distance to MC manifold, $k{=}20$)')
ax[0].set_ylabel('events'); ax[0].set_title('(a) Off-manifold selection'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')

# (b) rvert vs OOD, coloured by classifier score (high-score plotted on top, larger)
ds = d.sort_values('score')
sizes = np.where(ds.score > 0.8, 22, 6)
sc = ax[1].scatter(ds.rvert, ds.ood, c=ds.score, s=sizes, alpha=0.55, cmap='viridis', vmin=0, vmax=1, edgecolors='none')
ax[1].add_patch(Rectangle((0, OOD_CUT), RV_CUT, 8-OOD_CUT, fill=False, ec='tab:red', lw=2, ls='--'))
ax[1].axvline(RV_CUT, ls=':', color='0.4'); ax[1].axhline(OOD_CUT, ls=':', color='0.4')
ax[1].text(RV_CUT*0.5, 6.8, 'selected\nbackground', color='tab:red', ha='center', fontsize=9, fontweight='bold')
ax[1].set_xlim(0, 3); ax[1].set_ylim(0, 8)
ax[1].set_xlabel('verticality  $r_\\mathrm{vert} = \\sigma_z/\\sigma_{xy}$'); ax[1].set_ylabel('encoder-OOD')
ax[1].set_title('(b) Selection corner'); ax[1].grid(alpha=0.3)
cb = fig.colorbar(sc, ax=ax[1]); cb.set_label('classifier score $\\xi$')

fig.tight_layout(); fig.savefig(HERE/'figures/nu_classifier_ood_selection.png', dpi=140)
print('saved figures/nu_classifier_ood_selection.png')
