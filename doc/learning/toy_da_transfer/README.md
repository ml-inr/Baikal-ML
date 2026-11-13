# Toy surrogate-DA transfer on MNIST (distributional dirt + colour feature)

**Question.** A DANN is trained between slightly-dirty, grey two-digit images
(source) and dirtier, coloured two-digit images (target).  Does it make a
one-digit classifier, trained on the source domain, more robust on a held-out
coloured target?

**Short answer (2026-10-25).** Yes, when the domain shift is a *new feature*.
With dirt held nearly equal (source N(2.75, 0.25), target N(3.0, 0.25)) and a
target-only global colour gain (alpha 0.9) plus a wider target contrast
(sigma 0.45), the source-only model loses **11.4 pp** from source to target
(0.865 -> 0.751), while the DANN recovers **+8.60 +/- 1.77 pp** on the final
target (0.751 -> 0.837, 5 seeds).  The domain probes fall by 12-14 pp, the
paired colour probe by 13 pp, and the source accuracy is unchanged.  A
`lambda=0` run (critic trained, no adversarial term) changes nothing
(target +0.16 pp, probes unchanged, critic alive at 0.952), so the gain is
the adversarial removal of the colour/contrast feature, not extra domain
exposure.  Without colour the target is easier (source-only 0.847) and the
DA gain is only +2.54 pp; that line is the control.

## Data design

Five disjoint roles, split by original MNIST image before any augmentation:

| role | form | dirt level | contrast sigma | colour | pool | images |
|---|---|---|---:|---:|---|---:|
| source-main | 1 digit | N(2.75, 0.25) | 0.08 | no | MNIST train `[0:8k)` | 8 000 |
| source-DA | 2 digits | N(2.75, 0.25) | 0.08 | no | MNIST train `[8k:16k)` | 8 000 |
| source-test | 1 digit | N(2.75, 0.25) | 0.08 | no | MNIST train `[24k:28k)` | 4 000 |
| target-DA | 2 digits | N(3.00, 0.25) | 0.45 | alpha 0.9 | MNIST test `[0:8k)` | 8 000 |
| final-target | 1 digit | N(3.00, 0.25) | 0.45 | alpha 0.9 | MNIST test `[8k:10k)` | 2 000 |

The dirt level and the contrast factor are truncated normals (mean +/- 2
sigma).  Dirt `s` controls the horizontal squeeze `28(1-0.15s)`, a fixed
Gaussian blur, noise `0.08s`, salt `0.012s` and per-image gain `1 +/- 0.15s`.
Contrast is a separate per-image factor around the image mean.  Colour is a
target-only per-image RGB gain in `[1-alpha, 1+alpha]`; it tints the digit
and the background, and is label-preserving.  The source/target contrast
distributions and the colour are the *new domain feature*; the dirt is the
MC-like response that both domains share.

Augmentations are drawn on the fly per sample in every batch and shared by
both domains: rotation +/-15 deg, independent stretch `1 +/- 0.15`,
translation +/-2 px, shear +/-5 deg.  Two-digit images are built from two
dirtied digits: the black padding left by the dirt squeeze is cropped, the
two digits are concatenated, and the result is resized to 28x28, so the
width equals the one-digit images.

Evaluation uses `K=3` fixed draws per image (same draws for baseline and DA);
training draws fresh augmentations every batch.  See `figures/` for examples
(`classes.png`, `domains.png`, `construction.png`).

## Baikal analogy

| toy role | Baikal analogue | what the surrogate does not reproduce |
|---|---|---|
| source-main (1 digit, labelled) | MC events with labels for the main classification task | variable-length hit sequences; Cherenkov-light physics |
| source-DA (2 digits, unlabelled) | MC used as the unlabelled source domain | no direct Baikal counterpart (see below) |
| target-DA (2 digits, coloured) | experimental data as the unlabelled target domain | real detector systematics; colour stands in for a geometry/event-length feature |
| final-target (1 digit, coloured) | held-out experimental-like events for the main task | absolute rates; the real signal/background imbalance |
| dirt, both domains, target slightly higher | noise level and optical-module response spread, shared by MC and exp | the size of the real MC/exp response difference is chosen, not measured |
| contrast + colour, target only | new geometry or event time length, present only in exp | the two are not equivalent proxies: contrast alone makes the target easier |
| MNIST train / test pools | different dataset realisations (MC vs exp) | this is what A0/B0 correct (clean bias +0.65 pp) |

The split is deliberate.  On Baikal the shared response (noise, optical-module
response) differs modestly between MC and exp, while a new geometry or
event-length feature exists only in exp.  The toy keeps dirt nearly equal
between the domains and puts the shift into a target-only feature, so the
question it asks is the Baikal question: does the DANN remove the new feature
without destroying the shared response?

Structural gaps:

- The two-digit surrogate has no direct Baikal counterpart.  On Baikal the
  unlabelled source and target pools exist as such; on MNIST a separate
  unlabelled source pool has to be manufactured, hence two digits.  The
  transfer from the surrogate to the task is not one-to-one: probe2 falls
  14.2 pp against probe1's 12.0 pp.
- Absolute rates and class balance are not reproduced.  The toy is balanced
  over ten digits and scores one-vs-rest at ~10%; the Baikal task is a rare
  signal against a vast background, and the numbers here are not transferable.
- The analogy is structural, not quantitative: it carries the roles and the
  kind of shift, not the detector response or the numbers.

## Calibration (four models)

The shift was accepted only if both checks hold:

- **A** (source-trained): A(target) < A(source) — the target is harder.
- **B** (target-trained): B(target) >= A(source) + clean bias, where the
  clean bias is `B0(clean target) - A0(clean source)` and measures the MNIST
  train/test realisation difference.
- **A0/B0**: clean source/target models.

Colour line (dirt 2.75/3.0, contrast 0.08/0.45, colour alpha 0.9, 8k/8k,
10 epochs, 1 seed):

| model | source | target | notes |
|---|---:|---:|---|
| A | 0.859 | **0.779** | target 8.1 pp harder |
| B | — | **0.873** | target learnable |
| A0 clean | 0.978 | — | MNIST train |
| B0 clean | — | 0.9845 | MNIST test |

Clean bias **+0.65 pp**; check 1 passes (8.1 pp), check 2 passes
(B - A = +1.4 pp raw, +0.7 pp after the bias).  The no-colour line at the
same dirt/contrast has an easier target; it is the control.

## Main results (5 seeds, 10 epochs, lambda=0.2)

All numbers are mean over the last 3 epochs; +/- is the seed spread.

| line | scenario | source-test | final-target | delta target | probe 1-digit | probe 2-digit | paired colour | critic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| colour | baseline | 0.865 +/- 0.007 | 0.751 +/- 0.020 | — | 0.894 | 0.910 | 0.880 | — |
| colour | DA lambda=0 | 0.865 | 0.753 +/- 0.024 | +0.16 +/- 2.10 | 0.897 | 0.930 | 0.870 | **0.952** |
| colour | DA lambda=0.2 | 0.864 +/- 0.004 | **0.837 +/- 0.008** | **+8.60 +/- 1.77** | **0.774** | **0.768** | **0.750** | 0.666 |
| no colour | baseline | 0.865 +/- 0.007 | 0.847 +/- 0.017 | — | 0.664 | 0.701 | — | — |
| no colour | DA lambda=0 | 0.867 | 0.854 +/- 0.011 | +0.74 +/- 0.89 | 0.671 | 0.720 | — | **0.776** |
| no colour | DA lambda=0.2 | 0.868 +/- 0.006 | **0.872 +/- 0.007** | **+2.54 +/- 1.68** | 0.644 | 0.665 | — | 0.566 |

The colour line: the DA closes 8.6 of the 12.2 pp source-to-target gap; the
remaining 3.6 pp separate it from the target-supervised B (0.873).  The
source accuracy is preserved (-0.09 pp).  The in-sample probe2_train falls
from 0.935 to 0.803 while the held-out probe2 falls from 0.910 to 0.768, so
the encoder removes more of the training domain than of the held-out one.

The no-colour line: the target is easier (0.847 baseline), the DA gain is
+2.54 pp, and even the lambda=0 run gives +0.74 pp; the adversarial part is
about +1.8 pp.  This is the control showing that the large colour effect is
specific to the new feature.

## Lambda scan and lambda=0

Seed 0, colour line:

| lambda | target baseline | target DA | probe1 baseline -> DA | paired colour baseline -> DA | critic |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.769 | 0.752 | 0.910 -> 0.915 | 0.893 -> 0.897 | 0.952 |
| 0.1 | 0.769 | 0.846 | 0.910 -> 0.780 | 0.893 -> 0.753 | 0.694 |
| 0.2 | 0.769 | 0.851 | 0.910 -> 0.787 | 0.893 -> 0.738 | 0.639 |
| 0.3 | 0.769 | 0.851 | 0.910 -> 0.748 | 0.893 -> 0.730 | 0.616 |

The effect is lambda-insensitive over 0.1-0.3; 0.3 removes a little more of
the probe at a lower critic.  At lambda=0 the critic stays at 0.95 and the
probes do not move: the critic does not decay by itself.

## Oddities found

- **Contrast alone makes the target easier, not harder.**  A target contrast
  sigma of 0.45 without colour raises A(target) above A(source); the colour
  is the difficulty.  The contrast was kept as a second component of the new
  feature, but it is not the lever.
- **The target contrast tail is destructive.**  N(1, 0.45) truncated at
  +/-2 sigma reaches 0.1; some target images are nearly blank.  This caps B
  at 0.873 and leaves the DA residual (0.837) below it.
- **The lambda=0 DA slightly lowers the colour target** (0.751 vs 0.783 in
  seed 3, +0.3 in seed 1): extra domain exposure without the adversarial
  term does not help.
- **The paired colour probe is a clean readout**: 0.88 -> 0.75 with the
  adversarial DA, unchanged at lambda=0.
- **The no-colour target is easier than source** in most seeds; check 1 for
  that line is not robust.  It is a control, not a second positive result.

## Caveats

- The colour/contrast is a proxy for the real "new geometry / event length"
  domain feature (see *Baikal analogy*); MNIST cannot reproduce the detector
  response.
- The final target is 2 000 images x 3 draws; the 5-seed spread is 0.7-2 pp.
- The A/B calibration is one seed; the clean bias is +0.65 pp.
- The DA is trained on 2-digit surrogates and evaluated on 1-digit targets;
  the probe2 (2-digit) drops more than probe1 (1-digit), so the transfer is
  not one-to-one.
- The absolute accuracies are not Baikal numbers; the dirt levels are
  calibration choices.

## Files and reproduction

- `toy_da.py` — the experiment (data, augmentation, colour, DANN, probes).
- `calibrate_source.py` — the four-model A/B/A0/B0 calibration.
- `make_examples.py` — `figures/classes.png`, `domains.png`,
  `construction.png`.
- `analyze.py` — mean-of-last-N summary.
- `toy_da_walkthrough.ipynb` — executed English walkthrough: domains, DANN
  mechanics and the target-domain result.  It reads the saved JSONs and calls
  `toy_da.py`; no training is repeated.
- `runs/` — JSON results and logs (local).

Colour line, one seed:

```bash
python toy_da.py --epochs 10 --seeds 0 \
  --n-task 8000 --n-source 8000 --n-source-test 4000 \
  --n-target-train 8000 --n-target-test 2000 --eval-draws 3 \
  --source-dirt 2.75 --target-dirt 3.00 --contrast-target-sigma 0.45 \
  --color --color-mode global --color-alpha 0.90 \
  --da-lambda 0.2 --da-warmup-epochs 1 \
  --critic-steps 5 --critic-batch 512 --critic-hidden 128 \
  --critic-dropout 0 --critic-norm --out runs/main_color_l02_s0.json
```

The no-colour control drops `--color --color-mode global --color-alpha`.
The lambda=0 control uses `--da-lambda 0 --da-warmup-epochs 0`.
