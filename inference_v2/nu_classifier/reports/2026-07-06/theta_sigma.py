#!/usr/bin/env python
"""Fine theta-profile of MC muatm false-neutrino rate -> derive horizon sigma.

Uses the BOOSTED E1@ep10 muatm predictions (~6.6M muatm, ~3.2k score>0.8) so the
near-horizon falloff resolves at 2-3 deg. Fits frac>0.8(theta) ~ exp(-(theta-90)^2
/(2 sigma^2)); the fitted sigma sets the horizon-loss band width.
Outputs tables/mc_muatm_fp_vs_zenith_fine.csv, figures/mc_muatm_fp_vs_zenith_fine.png.
"""
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EDGES = [90,95,98,101,104,107,110,115,120,130,150,180]

c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
df = c.execute("""SELECT pr.score, l.part_key, l.local_idx
  FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='muatm_2020'
  JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE pr.n_sn_hits>=5""").df()
c.close()
theta = np.full(len(df), np.nan, np.float32)
with h5py.File(MCH5, 'r') as f:
    g = f['muatm_2020/prime_prty']
    for pk, idx in df.groupby('part_key').groups.items():
        k = f'{pk}/data'
        if k in g:
            t = g[k][:, 0]; li = df.loc[idx, 'local_idx'].to_numpy()
            ok = li < len(t); theta[np.array(idx)[ok]] = t[li[ok]]
df['theta'] = theta; df = df[np.isfinite(df.theta)]

rows = []
for a, b in zip(EDGES[:-1], EDGES[1:]):
    s = df[(df.theta >= a) & (df.theta < b)]
    if not len(s): continue
    k = int((s.score > 0.8).sum())
    rows.append(dict(theta_lo=a, theta_hi=b, ctr=(a+b)/2, N=len(s), k_hi=k,
                     frac_gt0p8=k/len(s), err=np.sqrt(max(k,1))/len(s)))
tab = pd.DataFrame(rows)
tab.to_csv(HERE/'tables/mc_muatm_fp_vs_zenith_fine.csv', index=False)

# Gaussian fit: ln f = ln A - (theta-90)^2 / (2 sigma^2), bins with k>=5, theta<130
fit = tab[(tab.k_hi >= 5) & (tab.ctr < 130)]
X = (fit.ctr.to_numpy()-90.0)**2; Y = np.log(fit.frac_gt0p8.to_numpy())
slope, intercept = np.polyfit(X, Y, 1)
sigma = float(np.sqrt(-1/(2*slope))); A = float(np.exp(intercept))
print(tab.to_string(index=False))
print(f'\nGaussian fit: sigma = {sigma:.1f} deg,  A = {A:.4f}')
with open(HERE/'tables/mc_muatm_fp_vs_zenith_fine.csv', 'a') as fh:
    fh.write(f'# gaussian_fit sigma_deg={sigma:.2f} A={A:.4f} (frac~A*exp(-(theta-90)^2/(2 sigma^2)))\n')

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(tab.ctr, tab.frac_gt0p8, yerr=tab.err, fmt='o', color='tab:red', capsize=3, label='MC muatm data')
tt = np.linspace(90, 140, 200)
ax.plot(tt, A*np.exp(-(tt-90)**2/(2*sigma**2)), 'k-', label=f'Gaussian fit σ={sigma:.1f}°')
ax.axvspan(90, 90+sigma, color='grey', alpha=0.15, label=f'±1σ band from horizon')
ax.set_yscale('log'); ax.set_xlabel('true zenith θ (deg)')
ax.set_ylabel('frac(score>0.8) — false-neutrino rate')
ax.set_title(f'MC muatm horizon confusion — σ={sigma:.1f}° (boosted stats, N={len(df):,})')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(HERE/'figures/mc_muatm_fp_vs_zenith_fine.png', dpi=130)
print('saved figures/mc_muatm_fp_vs_zenith_fine.png')
