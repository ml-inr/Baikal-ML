# Energy reconstruction — storyboard for the slide deck

Audience: mixed physics/ML, not familiar with this repository.  Main slides avoid
file formats and code paths; formulas and exact ID definitions live in the appendix.

## Slide 1 — What is the event?

**Purpose:** make the input and the target physical before any ML.

Visual: a real MC event display (actual hits, colour = time, size = charge,
true muon track) with the target cylinder and entry point overlaid.

Text:

- Baikal-GVD event = hits from Cherenkov light;
- one hit = charge, time, position `(Q,t,x,y,z)`;
- one event is a variable-length set of hits, not an image;
- energy dataset keeps events with exactly one muon track.

Target:

```text
t = log10(E_entry / 1 GeV)
E_entry = muon energy where the track enters
          r <= 90 m, |z| <= 295 m around its own cluster
```

Callout: **not** neutrino energy, **not** total deposited energy, **not** energy at
closest approach.

Figure: `figures/real_event_display.svg/.png` — a real MC event with the true
track, colour-coded hit time and charge-scaled hits.  The editable version is
`real_event_display_native.pptx`.

A schematic 3D drawing (`figures/target_definition_3d.*`) is kept only as a
fallback; it is redundant with the real event display and is not used on the
main slide.

## Slide 2 — Which events are used?

**Purpose:** show accepted and rejected topologies.

Visual: six small real MC event panels (editable in
`event_types_real_native.pptx`):

- single muon crosses the target volume — accepted; target is `E_entry`;
- muon born inside — target is energy at birth; special case (nuatm rows excluded
  because the vertex cascade is not recorded; nue2 uses birth energy);
- muon stops before reaching the volume — no `E_entry`;
- track misses the volume — no `E_entry`;
- two muons — violates single-track requirement;
- too few signal hits — `n_sig_hits >= 8`, `n_sig_strings >= 3` fails.

Figure: `event_types_real_native.pptx` — the editable slide; the same panels are
also rendered as `figures/event_types_real.svg/.png` (real MC events, not a
schematic).

## Slide 3 — Data roles and selection

**Purpose:** explain where labels come from and why there are three data streams.

Three columns:

| stream | role | labels |
|---|---|---|
| MC neutrino | energy training | energy |
| MC muon | DA source | domain only |
| experiment | DA target | domain only |

Funnel:

```text
signal-like hits (SN > 0.8)
  → >= 8 hits and >= 3 strings
  → exactly one muon track
  → valid energy truth
  → flat in log10 E (weights 1–20)
  → split by whole production parts
```

Key numbers: ≈1M events, ≈20M selected hits, 0.24 GeV–~100 PeV;
train/val/cal ≈ 900k/80k/20k; energy test = held-out parts with natural spectrum.

## Slide 4 — What does the network do?

**Purpose:** three tasks, one shared encoder.

```text
hits → transformer encoder ─→ μ  (energy)
                           ├→ σ  (uncertainty)
                           └→ D  (MC or experiment)
```

- **μ**: predicts `log10(E/GeV)`;
- **σ**: predicts an uncertainty scale; NLL penalises both wrong mean and dishonest
  uncertainty;
- **D**: domain discriminator; GRL makes encoder features less MC/exp-separable;
- an independent linear/MLP probe on held-out DA parts checks whether the domain gap
  actually shrank.

Main slide: icons and one-line explanations.  Formulas in appendix.

## Slide 5 — Training regimes

**Purpose:** show the experiment matrix without pretending every combination is a
separate physical model.

| | no DA | DA |
|---|---|---|
| mean only | `BASE` | `A` |
| joint σ, one optimizer | `B` | `AB` |
| joint σ, two optimizers (control) | `C` | `AC` |
| σ detached from encoder | `D` | `AD` |
| mean frozen, σ head only | `H` | `AH` |

Badges: `-G` Gaussian, `-T5` Student-t ν=5.  `C/AC` is a control.  `H/AH` is
post-hoc.  `D/AD` trains σ without updating the encoder.  `B/AB` is joint training
with one optimizer.

## Slide 6 — How we judge it

**Purpose:** separate energy quality, uncertainty quality, and domain adaptation.

- Energy: MAE in dex, bias, RMSE on held-out production parts.
- Uncertainty: NLL; coverage 68/95; calibration factors fitted on a calibration
  split and reported on val.
- DA: critic AUC and independent probe AUC on held-out domain parts.
- Test spectrum: natural; training spectrum: flattened, so raw event counts are not
  comparable without weights.
- Caveat: experiment has no energy truth; DA can only be checked indirectly.

## Appendix A — Formulas

```text
target:        t = log10(E_entry / 1 GeV)
Huber:        L_delta(r) = 0.5 r^2                 if |r| <= delta
                           delta (|r| - 0.5 delta)  otherwise
Gaussian NLL:  0.5 [ r^2 / sigma^2 + log sigma^2 + log(2 pi) ]
Student-t NLL: -lgamma((nu+1)/2) + lgamma(nu/2)
                + 0.5 log(nu pi) + log sigma
                + 0.5 (nu+1) log(1 + (r/sigma)^2 / nu)
domain BCE:    -[y log D(f) + (1-y) log(1-D(f))]
calibration:   c_q = quantile_q(|r| / sigma)
```

where `r = t - μ`, `σ` is the predicted scale, `ν = 5` for `-T5`.

## Appendix B — IDs and definitions

Map legacy names to IDs; define data roles, target cylinder, statuses 0/1/4/5,
single-track cut, flat selection, part-level test split, and the difference between
the independent probe and the critic.  See `doc/track_energy_experiment_ids.md`.

## Appendix C — real event display

One real selected MC event from the energy dataset, shown as x-y and z projections
of its Cherenkov hits, with the target cylinder marked.  This is explicitly an
**event display**, not the schematic used in the main story.

Figure: `figures/appendix_event_display.svg/.png`.
