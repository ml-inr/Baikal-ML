# Reconstruction quality cuts for reco events

The recommended selection for reconstructed events, what each condition maps to
in the HDF5 files, and what it actually does to the two samples. Applying the
list blind is unsafe, and the measurements below say why.

## The recommended cuts, as given

```
thetaRec      < 80
nHits         >= 8
nStrings      >= 2
covMatrixStatus == 3
scfMaxTheta   < 1.7
log10(pHit)   > -9          # questionable mapping with the slides
evCenterZ     < 220
zDist         > 70
pathLength    > 50
log10(funcValue / (nHits - 5)) < 1.1   # questionable mapping with the slides
nTriplets / nHits > 0.1
thetaErr      < 2.5
nCalls        < 350
```

Two are flagged at source as uncertain mappings. Two more differ between the
documented list and the code that applies it — see *Discrepancies* below.

## Where the quantities live — **verified**

`reco_prty` in `exp_reco.h5` has 25 columns, in `baikal_mc_reco.h5` 31 (the same
25 plus six MC-only vectors). The names come from `inference/shared_utils.py`
(`EXP_RECO_COL_NAMES`, `MC_RECO_COL_NAMES`), which is the authority; the mapping
was additionally confirmed against value ranges rather than assumed.

| idx | name | BRecoMuon field | idx | name | BRecoMuon field |
|---|---|---|---|---|---|
| 0 | `thetaRec` | `fThetaRec` | 13 | `covMatrixStatus` | `fCovMatrixStatus` |
| 1 | `phiRec` | `fPhiRec` | 14 | `scfMaxTheta` | `fScfMaxTheta` |
| 2 | `thetaErr` | `fThetaErr` | 15 | `scfMinTheta` | `fScfMinTheta` |
| 3 | `phiErr` | `fPhiErr` | 16 | `scfTheta` | `fScfTheta` |
| 4 | `funcValue` | `fFuncValue` | 17 | `scfPhi` | `fScfPhi` |
| 5 | `timeChi2` | `fTimeChi2` | 18 | `pHit` | `fPHit` |
| 6 | `chargeTerm` | `fChargeTerm` | 19 | `evCenterZ` | `fEvCenterZ` |
| 7 | `LLFit` | `fLLFit` | 20 | `zDist` | `fZDist` |
| 8 | `nHits` | `fNHits` | 21 | `nTriplets` | `fNTriplets` |
| 9 | `nStrings` | `fNStrings` | 22 | `nCalls` | `fNCalls` |
| 10 | `nOMs` | `fNOMs` | 23 | `classBDT` | `fClassBDT` |
| 11 | `pathLength` | `fPathLength` | 24 | `classBDTLowE` | `fClassBDTLowE` |
| 12 | `timeXYZRec` | `fTimeXYZRec` | 25–30 | `xyzRec_*`, `dirRec_*` | MC only |

## Three traps in the values themselves — **verified**

**Units are mixed.** `thetaRec`, `phiRec`, `thetaErr`, `phiErr` are in
**degrees** (ranges 0–180 and 0–360). `scfMaxTheta`, `scfMinTheta`, `scfTheta`
are in **radians** (maximum exactly π) and `scfPhi` spans 0–2π. So in the list
above `thetaRec < 80` is a cut in degrees and `scfMaxTheta < 1.7` one in radians.

**Three fields are not filled.** `LLFit` is identically 0; `classBDT` and
`classBDTLowE` are identically −2, in both files. A cut on any of them is
meaningless.

**`evCenterZ` is not on the same scale in the two files.** Median 435 in
`exp_reco` against 46 in `mc_reco` muatm (and 1.7 in `nuatm_conv`), with −1000
used as a sentinel. The recommended `evCenterZ < 220` therefore keeps 98.5% of
`mc_reco` and 9.1% of `exp_reco`. Until the origin of that difference is
established, this cut cannot be applied symmetrically.

## What the cuts do to each sample — **verified**

Measured on the full samples: 14,447,866 `exp_reco` events and 2,597,359
`mc_reco` **muatm** events (`analysis/reco_excess/stages/03_cuts_and_quality.py`).

Two things must be right before these numbers mean anything, and an earlier
version of this file got both wrong — the wrong numbers are quoted below so the
error is visible rather than silently replaced.

**Compare against muons, not against the whole simulation.** Three of the four
`mc_reco` classes are neutrinos, and they really are up-going: `thetaRec < 80`
keeps 99.1% of `nuatm_conv` and 87.8% of `nue2` against 2.1% of `muatm`. Measured
against the whole file, `thetaRec` and `scfMaxTheta` looked 10× and 22×
asymmetric; against `muatm` they are 0.77 and 0.64.

**Refer `evCenterZ` to the cluster centre.** BRecoMuon writes it in absolute
detector coordinates while the hits are stored cluster-centred, and the two files
disagree about that origin: the experimental cluster sits at z = 359.7–362.3
(twelve values, one per cluster), the simulated one at z = 0.4. Raw medians are
therefore 474 and 40, and `evCenterZ < 220` appeared to keep 9.1% of the data
against 98.5% of the simulation — a 10.8× asymmetry that is entirely a frame
mismatch. Referred to the cluster centre the medians are 74.3 and 46.0 and the
cut keeps 93.0% against 98.4%, a ratio of 1.06.

| cut | keeps in exp_reco | keeps in mc_reco muatm | ratio |
|---|---|---|---|
| `thetaRec < 80` | 0.0268 | 0.0206 | 0.77 |
| `nHits >= 8` | 0.7481 | 0.7525 | 1.01 |
| `nStrings >= 2` | 0.9985 | 0.9956 | 1.00 |
| `covMatrixStatus == 3` | 0.8467 | 0.9069 | 1.07 |
| `scfMaxTheta < 1.7` | 0.0125 | 0.0081 | 0.64 |
| `log10(pHit) > -9` | 0.9878 | 0.9918 | 1.00 |
| `evCenterZ − centre_z < 220` | 0.9296 | 0.9844 | 1.06 |
| `zDist > 70` | 0.9774 | 0.9745 | 1.00 |
| `pathLength > 50` | 0.9870 | 0.9885 | 1.00 |
| `log10(funcValue/(nHits−5)) < 1.1` | 0.4915 | 0.5598 | 1.14 |
| `nTriplets / nHits > 0.1` | 0.6984 | 0.7933 | 1.14 |
| `thetaErr < 2.5` | 0.9845 | 0.9677 | 0.98 |
| `nCalls < 350` | 0.9574 | 0.9806 | 1.02 |
| **all thirteen together** | **0.00105** | **0.00094** | **0.89** |

**The list is symmetric.** Every condition falls between 0.64 and 1.14 of parity
and together they keep 0.105% of the data against 0.094% of the muons. They can
be applied to both samples; they cost a factor of about a thousand in statistics.

> **Superseded claim, kept visible.** This file previously stated that the list
> keeps "17.0% of mc_reco and 0.013% of exp_reco, a factor of 1,300", and
> concluded it could not be applied symmetrically. Both halves were artefacts:
> the simulation side included the neutrino classes, and `evCenterZ` was compared
> across different coordinate origins.

## Discrepancies between the documented list and the code — **verified**

`get_mask` in `inference/prefilter_model/report.ipynb` and in
`test_exp_reco_mc_reco/plots_from_preds.ipynb` implements the list with two
differences:

* `log10(funcValue / (nHits - 5)) < **2.0**`, not the documented 1.1;
* `nStrings >= 2` in the report, `nStrings >= 3` in the current notebook.

## Known faults of the reco data

**Cluster 1 of `exp_reco` — exclude.** It accepts 6.9% of its h8s3 events where
every other cluster accepts 0.94–1.18%, and only 30% of it survives the
preselection against about 90% elsewhere. The prefilter work excludes it
(`BAD_RECO_PARTS`). Note that [claims_log](claims_log.md) records a *refuted*
cluster-1 exclusion — that check was made against `exp_full.h5`, and the same
note says the claim probably originated in `exp_reco`. Different file; the
refutation does not transfer, and the measurement here is unambiguous.

**Charges above 10⁴ p.e. — exclude.** The "Bad Qmax" fault. 0.20% of `exp_reco`
and 0.12% of `mc_reco`, but concentrated: 1.44% in cluster 2, exactly zero in
clusters 3 and 6.

**Multi-cluster fragments in `mc_reco` — exclude.** An event spanning several
clusters is stored once per cluster, and each fragment carries the reconstruction
of the *whole* event beside the hits of only one piece. The prefilter work
removes them with `n_signal_hits_gt > 5`; the equivalent here is
`n_gt_sig_hits > 5` from the probabilities file. 12% of `mc_reco` has no true
signal hit at all.

## Effect on the measured excess

With h8s3 and the three exclusions above, and `muatm` as the background
(`analysis/reco_excess/stages/04_reco_excess.py`):

| selection | events (exp / MC) | excess at ξ>0.5 | at ξ>0.8 |
|---|---|---|---|
| h8s3 only | 6,921,465 / 1,015,034 | 2.97 | 5.47 |
| without cluster 1 | 6,678,032 / 1,015,034 | 2.21 | 3.00 |
| without cluster 1 and bad charge | 6,660,634 / 1,015,002 | 2.17 | 2.95 |
| all three exclusions | 6,660,634 / 1,015,001 | 2.17 | **2.95** |
| up-going only (`thetaRec < 90`) | 312,497 / 47,508 | 2.07 | 2.71 |
| full BARS quality selection | 5,825 / 885 | 1.76 | 2.08 |

For comparison, `exp_full` against `mc_merged` gives **2.87** at ξ>0.8.

The BARS selection lowers the excess from 2.95 to 2.08 while costing a factor of
a thousand in statistics. It does not remove it.
