# What the experimental excess is

Selection throughout: h8s3 (`n_sig_hits >= 8`, `n_sig_strings >= 3`), out-of-training,
excluding the two runs with a measured detector fault, score > 0.8. Model
`260816_2250_..._FIXED_sn256`, epoch 23. Status vocabulary from `doc/claims_log.md`.
Reproduced end to end by `excess.ipynb`.

**The order of argument is deliberate.** Identification first, from the events' own
properties; counting last, in the narrow role it can fill. The reverse order is a trap: "too
many to be neutrinos" presumes the muon prediction it is measured against, and it is
powerless against a population whose flux is unknown — the hypothesis most worth keeping
open.

> **Correction, 2026-08-21.** An earlier version of this document read the `r_vert` bands as
> zenith bands and concluded the excess "grows away from the horizon". That was wrong.
> Within the population the classifier scores high, `r_vert` and true zenith are **completely
> decoupled** — every band sits at 107° — so the binning is by hit-pattern shape, not by
> angle. The correction is worked through in §5 and the discarded reading is kept in the
> appendix.

## The observation

About **1,500 excess events** above score 0.8 in 604.9 hours of exposure — **twice** the muon
prediction of 1,546, at 26 sigma.

The prediction is built per shape band and summed rather than scaled globally, because the two
samples are not composed alike: the data holds **twice** the share of flat wide hit patterns
(`r_vert` below 0.4) and fewer tall ones, and that is true before any score is applied. The
simulation is therefore wrong in two compounding ways — composition, and the high-score rate
within a fixed composition. Scaling globally mixes them and gives 1,980 events at a ratio of
2.9; the per-band figure of 1,498 at a ratio of 2.0 is the conservative one and is used
throughout.

## 1. What the events are — `verified`

Medians at score > 0.8, against three references:

| | ordinary muons | muons the model gets wrong | **the excess** | genuine neutrinos |
|---|---|---|---|---|
| track-likeness | 0.902 | 0.140 | **0.135** | 0.991 |
| depth spread | 150 m | 61 m | **60 m** | 195 m |
| duration | 595 ns | 318 ns | **289 ns** | 754 ns |
| `r_vert` | 1.109 | 0.555 | **0.567** | 1.541 |
| hits | 13 | 10 | **11** | 9 |

In the model's own representation, distance to the reference manifolds:

| | to neutrinos | to muons |
|---|---|---|
| ordinary muons | 8.95 | **0.59** |
| genuine neutrinos | **0.38** | 8.04 |
| muons the model gets wrong | 3.90 | 6.13 |
| **the excess** | **3.84** | **6.16** |

Ordinary events sit on their own class's manifold. Both high-scoring populations sit on
neither — six times further from their own class than an ordinary event — and land on each
other to within 1.5%.

**The bulk of the excess is the classifier's own failure mode appearing on real data.** This
rests only on comparing events to each other and survives any revision of fluxes, exposures
or simulation rates.

## 2. The failure is geometric, and it happens at the horizon — `verified`

Fraction of MC muons scoring above 0.8, by **true** zenith:

| zenith | 90–95° | 95–100° | 100–105° | 105–110° | 110–120° | 120–140° | 140–180° |
|---|---|---|---|---|---|---|---|
| rate | 18.60% | 5.23% | 1.45% | 0.40% | 0.071% | 0.0059% | 0.0008% |

A muon within 5° of the horizon is called a neutrino 18.6% of the time; one from straight
above, essentially never. The simulation contains no down-going neutrinos and no up-going
muons, so "neutrino" and "up-going" coincide perfectly in training and the decision boundary
landed on the horizon.

Consequently the misread muons are a narrow, uniform population: median true zenith **107°**
with quartiles 102–112°, whatever else about them varies.

## 3. What the simulation gets wrong — `consistent`

The misread events are **low-energy, near-horizontal** muons: median primary energy 372 GeV
against 557 for all h8s3 muons, and 318 GeV against 860 for correctly-scored muons of similar
shape. Grazing angles are where the slant depth is largest and the flux hardest to model.

Charge expressed per signal hit, so multiplicity cannot masquerade as brightness:

| | events | hits | charge per hit |
|---|---|---|---|
| MC muons, all quality events | 23,167,718 | 13 | 3.58 |
| experimental, all quality events | 3,180,681 | 13 | **4.02** |
| MC muons, score > 0.8 | 7,751 | 10 | 4.69 |
| experimental, score > 0.8 | 3,044 | 11 | **6.39** |

**The simulation is 12% too faint across the whole quality sample and 36% too faint in the
events the classifier scores high**, at the same hit multiplicity. Whether that is the energy
spectrum of near-horizontal muons, bundle multiplicity, or light propagation, these data
cannot separate — and a charge-calibration offset in the data would look identical. Nothing
available to us excludes it.

The tempting single explanation of both this and the composition difference does not survive:
if the simulation were simply too faint, its events would lose marginal hits and carry fewer
of them. They do not — both samples sit at 13 hits.

## 3a. Where the excess sits in observable quantities — `verified`

Plotted in the plane of **track-likeness** (the `r²` of hit time against depth) against
**charge per signal hit** — two quantities the network never sees — the disagreement is
visible *before any score cut*: the data is over-represented above roughly 4 p.e. per hit and
under-represented below 3, at every value of track-likeness, with a matching deficit in the
most track-like column. Figures 3 and 4 of `excess.ipynb`.

**The excess is not a compact island in this plane.** It is a broad shift along the charge
axis. That is why purely physical cuts capture it only weakly: the best combination found
(`track_likeness < 0.3`, `z_span < 100 m`, `t_span < 400 ns`, charge per hit > 5) selects
53,185 experimental events against 38,832 predicted — a ratio of 1.37 and a purity of 27%,
against the classifier's 2.0 and 49%. It does reject neutrinos, as required: only 2% of
atmospheric neutrinos survive it.

### A separate and much larger discrepancy, found while looking — `verified`

Scanning observable quantities for the sharpest data-versus-simulation disagreement turned up
something the classifier does not react to at all. The quality cut demands at least 8 signal
hits; when the number of distinct modules is smaller, the same module fired more than once.
Binning by hits per module:

| hits per module | exp | predicted from muons | ratio | ν passing |
|---|---|---|---|---|
| exactly 1.0 | 909,883 | 1,749,012 | **0.52** | 98.41% |
| 1.001–1.2 | 1,506,478 | 1,293,182 | 1.16 | 1.53% |
| 1.2–1.5 | 741,022 | 137,981 | **5.37** | 0.06% |
| 1.5–2.0 | 23,265 | 505 | **46.1** | 0.00% |

Half a million experimental events carry repeated hits on a module that the simulation barely
produces. The median classifier score of these events is **0.0014** — the network ignores
them entirely — so this is a second, independent failure of the simulation, an order of
magnitude larger than the excess this document is about, and visible by counting alone.

**The rate is not the whole of it: the association is also wrong** — `verified` by direct
measurement, interpretation below is `assumed`. In simulation a repeated hit goes with a
bright event, as any light-based explanation requires: muon medians are 4.47 p.e. per hit when
a module repeats against 3.56 when none does, and astrophysical neutrinos 19.33 against 9.41.
In the data the association is absent — 4.22 against 3.97 p.e., track-likeness 0.85 against
0.86. Repeated hits in the data are common and carry no imprint of the event that produced
them. That pattern is what an instrumental artefact would look like and not what scattered
light does, but this is an inference from the MC/data contrast alone; no external anchor
(a pulse-shape study, a calibration run, the DAQ's own multi-hit record) has been consulted,
so the *instrumental* reading is `assumed`, not established.

It does not explain the excess: restricted to events with exactly one hit per module, where a
repeated hit cannot contribute at all, the excess is still **929 events, ratio 2.16** — a
higher ratio than the 1.97 of the full sample.

**The excesses of different subsets must not be added.** Measured separately: exactly 1.0
gives 929 at 2.16, `1.0 < r ≤ 1.2` gives 761 at 3.86, `> 1.2` gives 219 at 4.10, and "any
repeat" gives 969 at 3.78. The observed counts partition the sample (1,727 + 1,027 + 290 =
3,044) but the predictions do not (798.1 + 265.7 + 70.7 against 1,545.8 for the whole),
because each subset's prediction uses the muon selection efficiency measured *inside* that
subset — which is itself a correction for the composition difference this section reports.
"What share of the 1,498 excess events are repeated-hit events" is therefore not a quantity
this method defines. An earlier version of this section said "both halves carry it in
comparable number", which invited exactly that bad addition; the load-bearing claim is the
`== 1.0` row alone.

Worth stating because it is easy to misread: the ratio 46 rests on 505 predicted events from a
class the simulation reproduces badly, so it measures the size of the modelling gap rather
than an excess of anything physical.

## 4. Counting, in its one role — `consistent`, and narrow

The experimental sample carries the natural mixture, so no absolute normalisation is needed —
only the relative efficiency of the selection, which the simulation gives directly. A
neutrino reaching h8s3 scores above 0.8 with probability 0.990; a muon, 0.00033.

Explaining the bulk with known atmospheric neutrinos demands a mixture of order 10^-4 at
h8s3, two to three orders above the expected 10^-6.

**This argument assumes the muon simulation is right in rate** — the rival hypothesis §3 just
found wanting — **and can only speak about neutrinos whose flux is known.** It excludes known
atmospheric neutrinos as the explanation of the bulk, and nothing more.

## 5. `r_vert` is a shape variable, not an angle — `verified`

This is the correction announced above, and it matters because the earlier reading shaped the
conclusions.

**Across the whole muon sample** `r_vert` tracks zenith cleanly — median 0.507 at 90–100°
rising to 3.099 at 160–180°. That is what made the earlier reading tempting.

**Within the high-scoring population it carries no angular information at all:**

| `r_vert` band | events | true zenith, quartiles | median |
|---|---|---|---|
| 0.0–0.4 | 1,540 | 103–112° | 107° |
| 0.4–0.6 | 3,034 | 103–112° | 107° |
| 0.6–0.8 | 2,263 | 102–112° | 107° |
| 0.8–1.2 | 884 | 102–112° | 106° |
| > 1.2 | 30 | 102–111° | 105° |

Inversely, `r_vert` is flat against zenith in this population: 0.565 at 90–100°, 0.556 at
100–110°, 0.551 at 110–120°, 0.553 at 120–180°.

The apparent correlation in the full sample is a **composition effect** — the muon flux is
dominated by steeply down-going events, which also happen to give tall hit patterns. Once the
angular range is narrowed to 102–112° by the classifier itself, nothing is left.

What `r_vert` does separate here is the aspect ratio of the hit cloud, which is what it is
defined as:

| `r_vert` band | depth spread | horizontal spread | track-likeness | `q_total` |
|---|---|---|---|---|
| 0.0–0.4 | 45 m | 121 m | 0.069 | 48.7 |
| 0.4–0.6 | 60 m | 96 m | 0.137 | 49.8 |
| 0.6–0.8 | 75 m | 93 m | 0.188 | 51.6 |
| 0.8–1.2 | 90 m | 90 m | 0.213 | 59.3 |
| > 1.2 | 133 m | 93 m | 0.202 | 66.3 |

Flat wide patterns at one end, tall ones at the other, at constant true angle.

## 6. There is no direction measurement for the experimental excess — `verified`

Three routes, all closed:

* **`dt/dz`** is calibrated and works — 99.2% purity over 25.1M MC events, no error at all
  where the fit is good — but it fails on exactly this population, and the muons prove it:
  83% of high-scoring MC muons are called "upward" when every one of them travels downward.
* **`r_vert`** is decoupled from zenith here, per §5.
* **Standard reconstruction** (`exp_reco`) shares only 8 of our 29 runs and its selection is
  itself a strong shape cut; deliberately not used.

So no statement about the direction of the experimental excess events is available from this
work. Any such statement would be invention.

## 7. The elongated subset — unresolved

At `r_vert > 1.2`, after removing three events built on a causally impossible hit: 18 observed
against 3.8 predicted from muons, about **14 in excess**.

| | depth | duration | track-likeness |
|---|---|---|---|
| muons the model gets wrong | 60 m | 289 ns | 0.135 |
| **these 14** | **165 m** | **668 ns** | **0.444** |
| genuine neutrinos | 240 m | 858 ns | 0.991 |

**These are not "vertical" events.** The MC muons in the same band have true zenith 105° with
quartiles 102–111° — as horizontal as the rest. The label refers to the shape of the hit
pattern only.

Three things at once, none decisive: the shape is intermediate, the rate they would demand
(~7·10^-6) is within an order of magnitude of expectation unlike the bulk, and there are
fourteen of them. **These data cannot decide.**

## Appendix — refuted, and the limits of the methods

| claim | what killed it |
|---|---|
| the excess grows away from the horizon | `r_vert` is decoupled from zenith in this population (§5); the binning is by shape |
| bioluminescence bursts | intervals between high-score events are Poisson (0.099% shorter than 0.1% of the mean against 0.100% expected) |
| clusters 1 and 4 are faulty | channel 71 in c04 is normal by occupancy (0.889) and mean charge (1.019); c04 has no anomalous channel at all |
| under-modelled near-horizontal muons, simple version | at the flattest hit patterns data and simulation agree (1.06x) — though see §5 for what that binning actually means |
| direction from `dt/dz` for these events | 83% of high-scoring MC muons are called upward, and every one of them travels downward |

**A.1 — no absolute normalisation.** `muatm` carries `event_weight` identically 1 and no
generated livetime is recorded anywhere, so the simulation cannot be turned into an expected
rate. §4 sidesteps this by using the natural mixture and relative efficiencies.

**A.2 — kNN distances do not converge** with reference size: median 4.755 at 5% of the
reference against 3.887 at 100%, implying an intrinsic dimension near 15 of 128. Only
comparisons at a fixed reference mean anything, so the paper's `d_MC > 2.5` cut is tied to the
reference set it was measured against.

**A.3 — the detector fault that was found.** Four channels on one string of cluster 2 emit up
to 3,169.8 p.e. against a median hit of 0.97; no charge limit exists before the classifier, so
such a hit reaches the sig-noise network as 863 sigma and is called signal 100% of the time.
They fire in exactly two runs, which are excluded. Effect on the excess: about 1%.
Full record in `DATA_QUALITY.md`.
