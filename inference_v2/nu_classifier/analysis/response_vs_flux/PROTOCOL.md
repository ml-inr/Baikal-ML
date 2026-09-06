# Protocol — frozen before each test runs

The question: the classifier accepts ~3x more experimental events than simulated
atmospheric muons predict, and the deficit is in MC, not in the domain (see
`analysis/excess_mechanism`, and the domain-reweighting result: matching MC's
feature distribution to experiment moves the top band only 3.11 -> 2.62). Two
explanations remain, and they have opposite consequences for the paper.

Write `P(hits) = P(hits | true track) · P(true track)`.

* **(a) Flux deficit.** `P(true track)` is wrong: the generator underproduces
  some muon configurations. The detector response is fine, so the network's
  response function is validated, the neutrino simulation is untouched, and
  MC-derived signal efficiency transfers. Only the background estimate is broken.
* **(b) Response error.** `P(hits | true track)` is wrong in a
  topology-dependent way: real events are harder to reconstruct than simulated
  ones. The same mechanism that turns down-going muons into apparent up-going
  ones would turn genuine up-going neutrinos into apparent down-going ones, so
  recall falls at the same time as the false-positive rate rises. MC-derived
  efficiency does **not** transfer.

Marginal comparisons cannot separate them — they depend on both factors. Every
test below either conditions on the true track or fixes the flux.

Each section is written **before** the corresponding stage is run, and names the
observation that would kill the hypothesis. Results go to `RESULTS.md`; this
file is not edited to match them.

---

## Test 1 — internal redundancy (stages 30, 31)

Split the strings of a cluster into two interleaved sets A and B. Fit the
Cherenkov track on A's hits alone. For every module in B predict arrival time
and charge from the A-track, and form the residuals `dt = t_obs - t_pred` and
`log(q_obs / q_pred)`. Compare the residual distributions between data and MC,
**matched on the A-track parameters** (zenith, closest approach, A-side quality).

This measures `P(hits | track)` without truth and without the flux.

* **(a) predicts:** residual distributions agree. Population differs, response
  does not.
* **(b) predicts:** they disagree — a longer `dt` tail if scattering is
  mis-modelled, a distance-dependent shift in `log(q_obs/q_pred)` if absorption
  or PMT efficiency is.
* **Decisive form:** run it in bins of event difficulty and ask whether the
  response mismatch *tracks* the excess. Response agreeing in the hardest bin
  while the excess is 3x there is (a); mismatch growing with the excess is (b).
* **Falsified if:** residuals agree within statistics across every difficulty
  bin — then (b) is dead in this channel.
* **Known limitation, stated in advance:** on difficult events the A-fit is
  itself poor, so the anchor is loose and flux dependence leaks back in. The
  test is strongest where the anchor is tight; extending its conclusion to the
  difficult population is an assumption, not a measurement.

### 1s — cross-cluster anchor: **unavailable, see RESULTS stage 00**

The strong form anchors on a *different cluster* rather than a different half,
which removes the loose-anchor limitation entirely. It requires matching one
physical event across clusters. Stage 00 tested the three preconditions.
Outcome: no coincidence signal at any clock offset, with demonstrated
sensitivity. Independently, the timestamps this matching would rely on are not
trusted by the group (see RESULTS). **Test 1 therefore runs in the
half-detector form only**, and this fallback is recorded rather than silently
taken.

## Test 2 — is a flux reweighting sufficient? (stage 20)

Find a weight function `w(E, theta, multiplicity, impact)` over MC **truth**
variables that brings MC into agreement with data in the accepted region. Then
apply the *same* `w` to observables held out of the fit.

The test has power because it is over-determined: 4-5 truth variables against
70+ observables. A pure flux error must be repairable by one low-dimensional
function simultaneously everywhere.

* **(a) predicts:** such a `w` exists, is smooth and physically moderate, and
  fixes the held-out observables.
* **(b) predicts:** no `w` in truth variables works — the score can be matched
  but charges or timing then disagree.
* **Falsified if:** a smooth `w` reproduces every held-out observable, which
  would leave (b) with nothing to explain.

## Test 3 — reproduce the excess by perturbing the response (stage 40)

Overlay real detector noise, sampled from off-trigger windows of real runs, onto
MC events; re-run the sig-noise filter and the classifier. Repeat for physically
plausible response perturbations (time smearing, charge scaling, efficiency
dropout).

* **(b) is supported if:** some plausible perturbation reproduces **both** the
  3x excess **and** the observed feature-space differences. The conjunction is
  the requirement — matching the excess alone is easy and means nothing.
* **(b) is weakened if:** no plausible perturbation does both.

## Test 4 — run-to-run noise as a natural experiment (stage 10)

Lake Baikal's bioluminescence varies strongly between runs and seasons. Measure
the excess separately in each of the 29 clean runs and regress it on that run's
measured noise load. This uses variation that exists only in data and so cannot
be tuned away.

* **(b)-via-noise predicts:** the excess rises with noise load.
* **(a) predicts:** the excess is flat in noise load.
* **Falsified if:** flat — noise-driven (b) is then strongly disfavoured.
* **Confounding, stated in advance:** high-noise runs may also differ in trigger
  behaviour, so a *positive* correlation is ambiguous. A *null* is clean.

---

## Test 5 — what removes the false-positive muons, and what it costs (stages 50–52)

A different question from tests 1–4: not *why* the excess exists, but whether a
cut on hit-derived quantities can remove the muons the classifier wrongly
accepts, and what that costs.

**What the cut is trained to separate.** Both classes are atmospheric muons from
the same generator: those the classifier accepts (ξ > 0.5, false by definition —
all muatm is background) against those it confidently rejects (ξ < 0.01). The
only difference between them is that one set fools the network, so a
discriminator between them learns *confusability*, not "muon versus neutrino" —
which the classifier already does and which would say nothing new.

**Neutrinos never enter the design.** Their retention is therefore a prediction,
not an objective; a recall cost optimised against would be meaningless.

**Balance by subsampling, not by weights**, and in **five disjoint replicates**
of the negative class. Overlapping draws share events and make a cut look stabler
than it is. A cut counts only if it survives all five.

**The controls.**

* *muatm reference* — a uniform sample over all quality muatm, the denominator
  for "how much muon statistics survives in general".
* *exp reference* — a uniform sample over all quality experimental events. This
  is the control that decides the reading: the cut is trained on MC, so applied
  to experiment it might remove events simply because experiment differs from MC
  globally (a domain classifier reaches AUC 0.78 at every score). Without a
  low-score experimental control, "the cut removed confusable muons" cannot be
  distinguished from "the cut removes experimental events".

### Criteria, fixed before running

**On the experimental sample.** Compare two ratios: how much more strongly the
cut removes muatm-accepted than muatm-clean, and how much more strongly it
removes exp-accepted than exp-clean.

* the two agree → experimental accepted events are the same confusable-muon
  population, confirming the earlier conclusion;
* exp-accepted survives markedly better while muatm-accepted dies → they are
  **not** simply confusable muons, and the earlier conclusion needs revisiting;
* the cut removes exp uniformly across score → it is catching the domain shift,
  not confusability, and the whole comparison must be read differently.

**On recall.** Report the neutrino cost at the working points where 90% and 99%
of the false positives are removed, for nuatm and nue2 separately, and as
absolute efficiency over all quality neutrinos.

**Expectation, stated so it cannot be quietly abandoned.** Report 2026-08-10
established that the false positives are the *low-multiplicity* tail: a single
muon leaves the sparse track-like signature of a ν_μ charged-current event. So
the cut is expected to take neutrinos with it, perhaps almost entirely. If it
does, the answer is negative — this cannot be fixed with cuts — and that is a
result, not a failed search.

**On scale.** False positives are 1.5 per thousand of quality muatm, so even a
perfect cut costs at most 0.15% of muon statistics. Preserving muatm is
therefore nearly free, and the binding constraint is neutrino recall.
