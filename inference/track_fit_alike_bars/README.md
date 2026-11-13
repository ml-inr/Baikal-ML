# track_fit_alike_bars

A transparent re-implementation of the **BARS v2.3.2 muon-track reconstruction**, in
Python, for this project's own use: a per-event track fit whose input is the raw pulses
of an event and whose output is the set of reconstruction-quality variables the analysis
cuts on.

It is not a wrapper around BARS.  Every step of the chain is ported from the C++ sources
(line by line, with the BARS routine named in the docstring) and every ported piece is
checked against something outside the port — networkx, the GSL library, BARS' own
resource files, and BARS' own output on the same pulses.

The purpose is validation of the two-stage selection: apply quality cuts on this fit and
check that the false-positive score distributions of `muatm` and `exp` converge at high
score, losing as little neutrino exposure as possible.  The cuts and the success criterion
are a later decision; this package is only the fit.

## How to use it

Three ways in, depending on what you have at hand; everything else follows from
them.  All numbers below are in the units and frames of the last table.

### 1. An `event_fk` from the fit catalogue

```python
from inference import reader
from inference.track_fit_alike_bars import isolation

catalogue = reader.open_fits()          # sample/fits.duckdb, one row per fitted event
events = reader.open("exp_full", checkpoint=CKPT).events(where=[reader.H8S3])
tracks = catalogue.attach(events)       # fit_* columns; NaN outside the fitted sample
tracks = isolation.attach(tracks)       # iso_* columns: pnonhit and its counters
```

`isolation.attach` reads the pulses of each event from the HDF5 (once per part)
and the geometry of each event from its run's ROOT file.  A row whose fit did not
converge gets NaN -- there is no track to judge.

### 2. A frame of events, no catalogue

```python
from inference import reader
from inference.track_fit_alike_bars import fit_events, fit_hits, fit_event, masks

r = reader.open("exp_full", checkpoint=CKPT)
ev = r.events(where=[reader.H8S3])               # no hits are read
fits = fit_events(ev, n_workers=16)              # reads the pulses itself
ev.join(fits)                                    # same index as `ev`

# data: give the fit the channel mask BARS used
fits = fit_events(ev, n_workers=16, channel_mask=masks.dqm_mask_from_solution(txt))

# hits already in memory
for chunk in r.stream(where=[reader.H8S3], budget_mb=512):
    fits = fit_hits(chunk.hits)                  # all pulses: the sig-noise flag is a column

# one event
fit = fit_event(q, t, x, y, z, channel)
fit.theta_rec, fit.scf_max_theta, fit.time_chi2, fit.fit_hits, fit.residuals
```

`fit_events` needs a frame carrying `source`, `part_key`, `local_idx` (what
`reader.events()` returns) and `data_class` for the multi-group MC sources.
`fit_hits` takes any long table with columns `q, t, x, y, z, channel` and a group key.

`fit_events` and `fit_hits` report progress the way the reader does -- through a
callback, never by printing, because the package is a library:

```python
from inference.track_fit_alike_bars import tqdm_progress          # needs tqdm

with tqdm_progress("fit") as progress:                            # or your own callback
    fits = fit_events(ev[ev.score > 0.7], n_workers=16, progress=progress)
```

The callback receives a `FitProgress(stage, parts, events, total_parts, total_events,
part_key)`; the totals are known before the fit starts, so a bar can be sized from the
first call. Parts are reported as they finish, so the `part_key` in the description is a
part that is *done*, not one in flight.

`fit_event` returns a `TrackFit`.  Scalars are `NaN` unless `status == "ok"`, and `status`
is one of `ok`, `too_many_hits`, `no_clique`, `no_local_trigger`, `no_global_trigger`,
`no_offline_trigger` -- the places the BARS chain stops.  Per-hit results (`clique_mask`,
`scanfit_mask`, `fit_hits`, `residuals`) are arrays in the caller's hit order.  Angles:
`theta_rec`, `phi_rec` and their errors in degrees, the `scf_*` angles in radians, the
units BARS writes.  `theta_rec` is the polar angle of the direction of **motion**: below
90° is up-going.  `BARS_NAMES` maps every field to the `BRecoMuon` field it reproduces.

### 3. One track: the probability field and `pnonhit`

How likely a module near the track was to fire, and how well the track explains the
silence around it (BARS' `fPnonHit`; see `BMuonCandidate::GetPnonHitProd`).

```python
from inference.track_fit_alike_bars import geometry, isolation

om = geometry.oms("exp_full", "part_s2020_c04_r0031", local_idx=17)  # ОМ этого события
field = isolation.phit_field(track, om)             # P(>=1 p.e.) per OM, light distance
iso = isolation.pnonhit(track, om, fired=channels)  # pnonhit and counters
iso.pnonhit, iso.n_non_hit, iso.n_oms, iso.min_p_silent
```

`track` is a `TrackFit`, a row of `reader.open_fits().frame(...)`, or any mapping
with `ref_x/ref_y/ref_z/dir_x/dir_y/dir_z` (the catalogue's own column names).
`channels` are the channels that fired -- the HDF5 `raw/channels` of that event.

Three deliberate differences from BARS, each visible in the number:

* **the distance**: the light path (default, `distance="light"`) instead of the
  perpendicular distance of the legacy fit path (kept as
  `distance="perpendicular"` for comparisons with BARS' own output);
* **what counts as empty**: a channel that did not fire, where BARS excludes the
  channels its final track used -- a pulse the fit dropped reads as "fired" here
  and as "quiet" there;
* **no sentinel**: when nothing in the window is silent `pnonhit` is NaN, where
  BARS returns 1e-9, the worst value on its scale.

The check that these choices did not break the arithmetic: `validate_vs_bars`
now recomputes `pnonhit` in the strict BARS reading, at BARS' own track, and
matches `BRecoMuon.fPnonHit` to a median relative difference of ~3e-7 (the
float32 of the ROOT branch).  At *our* track the same comparison gives ~5e-4,
and that residue is the fit's own agreement with BARS, not the tool's.

`geometry.oms(source, part_key, local_idx=...)` returns the positions of every
OM in the hit frame.  Experimental geometry is stored per event and drifts by up
to ~1.5 m across a run, so `local_idx` is required; `snapshot=True` gives the
run's median table instead (for looking around, not for numbers).  The MC
detector is one static table: `geometry.oms("mc_merged")`.

### Module map

| module | what it is for |
|---|---|
| `frame.py` | `fit_events` / `fit_hits` / `read_event_hits` -- the tabular way in |
| `fit.py` | the BARS chain on one event (`fit_event`), `TrackFit`, `BARS_NAMES` |
| `candidate.py`, `scanfit.py`, `cliques.py`, `physics.py` | the stages: clique, triggers, scanfit, the Minuit fit, the light model |
| `config.py` | `FitConfig` -- BARS' thresholds out of the rc files |
| `masks.py` | the DQM channel mask, and where the ROOT files are |
| `geometry.py` | OM positions (from ROOT, the production geometry) in the h5 frame |
| `isolation.py` | the P(>=1 p.e.) field, `pnonhit`, and `attach` over a frame |
| `run_fit.py`, `make_sample.py` | the mass run and the sample manifest behind the catalogue |
| `validate_vs_bars.py`, `tests/` | comparisons with BARS itself and the self-checks |

### Units and frames

| quantity | units / frame |
|---|---|
| OM and hit coordinates (`om.xyz`, h5) | metres, cluster frame: `h5 = absolute - geometry.centre_of(...)` |
| `ref_x/y/z`, `dir_x/y/z` in the catalogue | the same frame; `dir_*` is a unit vector |
| `theta_rec`, `phi_rec`, `theta_err` | degrees; `scf_*` in radians, as BARS writes them |
| `ref_time`, `t`, `t_res_30_med` | nanoseconds |
| `pnonhit`, `phit_field` | dimensionless probabilities |
| charges (`q_*`, `charge_term`) | photoelectrons |

`theta_rec` is the polar angle of the direction of **motion**: below 90° is up-going.

## What is reproduced

`programs/reco-muon/reco.cc` with the 2020 resource files: the causality clique
(`BCausalityClique`, including the pruning quirks of `MaximumCliqueSearch`), both trigger
selections, the 6°-step scanfit over the whole sky (`BInitializeCausScanGrid`,
`BScanFitClique`, with the GSL `nmsimplex2` M-estimator), the offline trigger cut, and
`BRecoMuonProducer` with its 15 candidate trials, `RefitExclusive` and `RecollectHits`.
`FitConfig` carries the parameters of those files, and `FitConfig.from_bars_rc` reads them
from a BARS resource file directly — refusing a file that switches on a branch this port
does not have.

The fitted functional is BARS' own: `GetQuality(2)` (time χ², σ = 3 ns about the first
Cherenkov photon) and `GetQuality(4)` (`PrefitType 4`: the same χ² plus a charge-weighted
distance term), minimised by Minuit2 through iminuit with the settings `Refit` uses.

Fourth, a matter of arithmetic rather than of physics: BARS computes the angular
errors as `sqrt` of a covariance diagonal that a valid minimum can leave negative, and
C++ returns NaN where Python raises.  The port returns NaN (`candidate.error_deg`), so
such an event gets `theta_err = nan` and the analysis cuts it — in both implementations
for the same reason, since a NaN comparison is false in both.

Three behaviours of the C++ are reproduced on purpose, because they change the numbers:
`RefitExclusive` never tries the last hit of its list (`for (i=0; i<GetNHits(); i++)` with
`GetNHits()` read after the removal); `fSumCharge` is never reset by `SetImpulseArray`, so
the type-4 charge term is normalised by an inflated sum; and `z_dist_range` computes its
extent over the hits that are **not** in the clique.

## What is not

* **The BARS version is v2.3.2**, whose tree was extracted for the port.  `origin/v2.3.2`
  and the master checkout differ in `BRecoMuonProducer` and `BMuonNamespace`; the files
  the port uses that do differ are the ones whose behaviour the validation checks.
* **The channel masks.**  `exp_reco` is reconstructed with the DQM mask
  (`Mask.UseDQMChannelMask 1`); `masks` reads the per-event `ChannelMaskDQM` out of the
  run's ROOT file, or the run-level DQM solution file — see below.  MC is reconstructed
  with no mask at all, which is what BARS does (`UseDQMChannelMask 0`).  The *section*
  part of the mask (a per-event `BSectionStatus` file, not unpacked locally) cannot be
  reproduced; on the eleven runs where both parts are visible it touches 0–0.2% of hits.
* **`n_calls` and `covMatrixStatus` are not comparable.**  They count calls and states of
  ROOT's Minuit2, and the port minimises with iminuit.  A hypothesis that the difference
  comes from `Float_t` arithmetic (BARS stores its doubles as `float`) was tested by
  emulating the C++ types through the fit — the agreement did not improve, so it is
  **refuted**; the residual is the minimiser, not the arithmetic.  Any cut on `nCalls`
  has to be calibrated in the port, not copied from BARS.
* **`ev_center_z` is in the coordinates of the input.**  BARS writes it in the frame of
  `BGeomTel`: for the experimental files the cluster sits at z ≈ 360 and the two differ by
  exactly the cluster centre, so adding `clusters_centers` closes the comparison to 2e-5 m.
  For the MC files the cluster sits at z ≈ 0.4 and both frames give the same value to ~1 m
  (residual unexplained; the rows there are pieces of cluster-split events).  `fEvCenterZ <
  220` therefore means different things in the two frames, and the analysis tree's `-354`
  for data is not needed here.
* **External `BRecoMuon` fields** (energy, BDT scores, path length, RA/Dec) are not ported:
  the port stops where the quality cuts stop.

## How well it agrees with BARS

Measured by `validate_vs_bars.py` against BARS' own reconstruction of the same pulses.
`exp_reco` without and with the DQM mask the chain needs, and `mc_reco` muatm as the
control (no mask by construction).  2000 / 2000 / 1000 events.

| quantity | exp_reco, no mask | exp_reco, DQM mask | mc_reco muatm |
|---|---|---|---|
| scanfit mask identical | 93.4% | **100.0%** | **100.0%** |
| direction within 0.1° / 1° | 64.0% / 76.0% | 69.5% / 80.8% | 82.0% / 87.9% |
| direction, median difference | 0.039° | 0.033° | 0.015° |
| `n_hits`, `n_strings` equal | 87.7% / 91.5% | 93.2% / 94.2% | 96.0% / 96.6% |
| `scf_max_theta` equal | 97.0% | **100.0%** | **100.0%** |
| `time_chi2`, `func_value`, `z_dist`, median relative difference | 6e-4 / 1e-4 / 9e-5 | 4e-4 / 9e-5 / 7e-5 | 2e-4 / 2e-5 / 2e-5 |
| preliminary cuts: same decision | 85–100% | 85–100% | 94–100% |

The mask is the whole of the residual in the *selection*: with it, the scanfit mask and
`scf_max_theta` agree event for event, and both are also perfect on MC.  What is left is
the fit itself, and it is the minimiser: on a flat functional with 8–16 hits the two
implementations land in different local minima.  Refitting with tighter tolerances moves
us away from BARS, not towards it, so tolerance is not the cause.

Where they disagree, our χ² is *lower* — and that is a real minimum of the same
functional, not a different objective: `tests/truth_check.py` recomputes our own χ² at
BARS' track and reproduces BARS' `fTimeChi2` to a median relative difference of 1e-5, then
finds our track lower in 55% of 2000 `mc_reco` events.  The median *ratio* of the two
χ² values is nevertheless 1.0000 (quartiles 0.9998, 1.0002): both statistics are right,
because the median is set by the majority of events where the two fits agree to a part in
10⁴ and the sign is set by the minority where they do not.  Against MC truth (which neither fit sees)
the port is if anything better: on single-muon events, whose truth is unambiguous, its
direction is closer to the primary's in 57% of cases, and in 61% of the cases where the
two reconstructions disagree by more than a degree (median error 7.4° against 9.6°).

Stage by stage (`tests/stage_diagnostics.py`, on BARS' own ROOT file so the input is
exactly BARS'): after the scanfit the port reproduces `ScfTheta`, `FuncValueScf` and
`TimeChi2Scf` to 1e-3 in 100% of events (`scf_theta` exactly), the best trial's
`ThetaInit` in 91%, and the final `ThetaRec` in 86% (median relative difference 1e-4).
The chain is right up to the fit; the fit is right up to the minimiser.

## The channel mask

BARS applies a DQM channel mask to data.  It is not in the HDF5 files, and it matters: on
2000 `exp_reco` events it moves the scanfit-mask agreement from 93.4% to 100%.  `masks.py`
has the two sources:

* `dqm_masks_from_root(path, event_ids)` — the `ChannelMaskDQM` container of an `exp_reco`
  ROOT file, per event, keyed by `BJointHeader` event id (the id is `header_prty[:, 3]`).
  Exact by construction; needs the ~1800 ROOT files unpacked next to the h5.
* `dqm_mask_from_solution(txt)` — the run-level DQM solution file
  `dqm_s2020_cNN/NNNN/jNNNN.fit.solution.txt`, which is the file `BApplyDQMMask` reads
  (`DQM_FIT_SOLUTION`, dqm-version v1.3).  The Gantt histograms are only read when
  `Mask.UseGantt` is set, and it is 0 in the 2020 resource files, so this file alone
  decides every channel whose section is active.  It covers **all 29 runs of `exp_full.h5`**
  and all of `exp.h5`, but only 11 of the 370 `exp_reco` parts.

`masks.dqm_masks(source, part_key, event_ids)` dispatches on the source and returns a
`{event id: mask}` dict, one `(n_channels,)` array, or `None` for MC.

## On `exp_full`, where the fit will be used

There is no BARS reconstruction of `exp_full` to compare against, so
`tests/production_path.py` only asks whether the path works: the DQM solution file loads,
the fit runs, and it succeeds on 87% of the brightest 0.1% of events against 2% of random
ones.  The trigger stream is mostly noise — a random `exp_full` event has a median charge
per hit of 1 p.e. and no causal structure, and the fit says `no_clique` — while the bright
ones are the muons (and the LED/fault events the mask switches off).

## Self-checks and validation

```bash
python inference/track_fit_alike_bars/tests/test_fit.py           # 10 checks
python inference/track_fit_alike_bars/tests/test_isolation.py     # 6 checks, geometry + pnonhit
python inference/track_fit_alike_bars/tests/stage_diagnostics.py  # stage by stage, ROOT
python inference/track_fit_alike_bars/tests/production_path.py    # exp_full, mask from txt
python -m inference.track_fit_alike_bars.validate_vs_bars --source exp_reco \
    --n-parts 200 --per-part 10 --workers 16
```

`port_hash()` fingerprints every `.py` of this package, so adding `geometry.py` and
`isolation.py` changed the hash of *future* fit runs.  The catalogue already on disk
was produced by the code before them and stays valid: the fit itself was not touched,
which `test_fit.py` and a re-fit comparison confirm.

`validate_vs_bars.py` writes `validation/<YYMMDD_HHMM>/` with the per-event table, a
provenance sidecar and `summary.md`.  The sidecar (`.meta.json`) is not under git, like
every other `.json` here; `summary.md` and the code that produced it are.

The checks in `tests/test_fit.py` each anchor a ported piece to something outside it:
networkx and a transliteration of the C++ for the clique enumeration, `libgsl` itself for
the simplex (bit for bit), a literal emulation of the C++ stream read for the pHit table,
BARS' resource files for the defaults, and BARS' output for the chain.
