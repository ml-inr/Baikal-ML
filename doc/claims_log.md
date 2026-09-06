# Claims log

Every non-obvious factual claim this project relies on, with **what actually backs it** and how
far that goes. The point is that a reader can tell a proven statement from a plausible one
without re-deriving it — prose in `doc/` reads equally confident either way, and that has
already caused two wrong statements to be documented as established.

**Status vocabulary** (the distinction that matters most is the first two):

| status | meaning |
|---|---|
| `verified` | checked against an **external anchor** — physics, an independent source, code that writes the format, or a fact outside the data (e.g. a production's name stating its energy range) |
| `consistent` | the data agree with it, but the check could not distinguish it from a rival hypothesis. **Not proven.** |
| `assumed` | taken from a comment, a convention, or analogy. Treat as unverified. |
| `refuted` | was claimed, turned out false. Kept deliberately — see §3. |

Rules of use: a claim reaches `verified` only via something *outside* the dataset it describes;
comparing one field with another from the same file is `consistent` at best. When a claim
changes status, edit the row and add a line to §3 rather than quietly rewriting it.

---

## 1. Units and field semantics

| claim | evidence | status |
|---|---|---|
| `fInteractions.fEnergy` is in **TeV** | `⟨dE/dx⟩` vs the tabulated `a + bE` for muons in water agrees within 0.44–1.09 across five energy decades — a factor-1000 error would show as three orders. Independently, `energy.C` multiplies by 1000 in three places | `verified` |
| `fMuonEnergy` is in **GeV** | `energy.C` prints `GetMuonEnergy()` as GeV and pairs it with ionisation 0.24 GeV/m | `verified` |
| `fPrimaryParticleEnergy` is in **GeV** | the `nu2.100PeV` production maxes at 9.943e7, and 100 PeV = 1e8 GeV; nuatm then spans 20 GeV–97 TeV, a physical atmospheric spectrum | `verified` |
| `fSumEnergyBundleReg` is in **GeV** | shares a unit with `E_primary` (ratio ≤ 1 in 0.0000% of 61k events, reaching 0.999979 — a kinematic bound), and that unit is GeV by the row above | `verified` |
| `fSumEnergyBundleSurf` is filled **only for muatm** | identically 0 in 100% of nuatm and nue2 events; median 1737 GeV in muatm | `verified` |
| `fMuonEnergy` has three cases, `−0.001` being a sentinel for "born downstream" | tested on nue2 (15k tracks) and nuatm (982k): 0.0% of sentinel tracks have any interaction before the reference point, 100% of the other negatives do | `verified` |
| Showers are recorded only **above 0.1 GeV** | stored minimum is 0.1 GeV in both productions (the extractor already applies the TeV→GeV factor; reading it as TeV double-counts) | `verified` |
| `E_dep` accounts for the full loss, no systematic shortfall | mean `E_dep/L_path` over `a + bE` is 1.08–1.88 (nuatm). The earlier "0.44" compared a *median* with a *mean* of a skewed distribution | `verified` |
| Event-record coordinates are in **metres**, configuration-record in **centimetres** | `BReadMCGeomWout` applies `kCM_TO_M` to the channel table; event coordinates land in the right range for the detector | `consistent` |
| `fMaxDistance = 25000` means a **250 m** radius | read as cm by analogy with the channel table; but reconstructed entry points sit mostly *outside* 250 m, so either the unit or the meaning of the field is wrong | `refuted` |
| `Reg` is the energy on entering a volume of **several hundred metres**, with no sharp edge | walking each track back under `a+bE` until the energy reaches `Reg` gives entry points with median \|r\| 381 m, p95 773 m, only 14% inside 250 m, and no surface. Robust to `b` (±3% over a factor of two). The unit is settled; the exact geometry is not | `consistent` |

## 2. Formats, structure and pipeline

| claim | evidence | status |
|---|---|---|
| `.dat` / `.wout` record layout (17-word header, tracks, 4-word showers, channels, terminator) | transcribed from `Task_MAIN.cpp` / `ZDataReader.cpp`, then checked: parsing consumes exactly the declared length for 20,000/20,000 events; 5-word showers give 79%, no terminator 0% | `verified` |
| `E_muon_before_shower` exists in no production | showers are 4 words in 2020 (`v1.1`, `v3.0`, `v4.0`), in the generator `.dat`, and in the 2024 production | `verified` |
| `.dat` ↔ ROOT matching by content fingerprint | all 39,547 ROOT events found in the `.dat`; key unique for 1,000,000/1,000,000 | `verified` |
| ROOT holds all MC truth that exists upstream | channels, shower energies and coordinates agree digit-for-digit; ROOT additionally has noise and jitter | `verified` |
| `fMagic` encodes pulse origin: 1 = noise, −999999 = muon light, `k·10⁶+1` = k-th shower | `k` never exceeds `fInteractionN` (0 violations in 3341 events, maxima both 21); `mcread` reads the field as `n_source` with `icode_noise = 1` | `verified` |
| For **sentinel** tracks `Reg` is the muon energy at the neutrino vertex; in general `Reg` is taken at the detector's depth | full statistics (66.4M nuatm + 7.1M nue2): `corr(log(Reg/E_ν), log E_ν)` is **+0.03** for sentinels (nue2) against **−0.38** for alive tracks; sentinel ratio climbs to **0.769** at 10–100 PeV — the magnitude and trend of CC inelasticity — while alive tracks fall to 0.041, impossible for inelasticity. muatm cross-check: `Reg` = half of `Surf` and falls with zenith (`corr` with `1/cos θ` = +0.539) | `consistent` |
| `Reg` refers to a point upstream, and the first-detected-light point captures part of the way there | with the emission point taken exactly (shower coordinates via `fMagic`, Cherenkov geometry for muon light) the correction explains **40%** of `Reg − E_ref`, and crucially `corr(log(Reg−E_ref), log(correction)) = +0.71` — a residual correlation, not a mechanical gain from the shared `E_ref` term | `consistent` |
| The neutrino **vertex is not stored anywhere** in the ROOT files | all 83 leaves of the `Events` tree enumerated: the only position fields are `BMCEvent.fX/fY/fZ` (identically 0 for ν), `BMCTrack.fX/fY/fZ` (reference point), `MCEventSource.fMuonTracks.fPosition` (bitwise duplicate of the reference point) and `fInteractions`. Upstream `.dat`/`.wout` have no slot for it either | `verified` |
| `BMCTrack.fX/fY/fZ` is the closest approach of the track to the array centre | perpendicular to the track direction to 1e-6 (median \|B·d̂\|/\|B\| = 1.4e-6 nuatm, 5.5e-7 muatm; max 3.2e-6) | `verified` |
| `BMCTrack.fDelay` is **identically zero** and carries no information | 100% of tracks in both nuatm and muatm, the latter with bundles up to 347 muons. `fTime = fDelay + fFirstMuonTime` therefore equals the event-level time for every muon | `verified` |
| `fFirstMuonTime` is the muon's time **at the reference point** | `MCEventSource.fMuonTracks.fPosition` equals the reference point bitwise, and its `fTime` equals `fDelay + fFirstMuonTime` | `verified` |
| **`s_vertex = −c · fFirstMuonTime`** — the muon's birth point along the track | full statistics, 38M nuatm tracks: `s_vtx ≤ s_first` in **100.000%** and the sign is correct in **100.000%** for all three classes (sentinel positive, alive/dead negative). Energy cross-check on live tracks: `Reg` minus losses from the vertex predicts `E_ref` with median ratio 1.079 (p25 1.000). On nue2 the identity holds too — the apparent 44–51% "violations" are float32 ties where `s_vtx = s_first` exactly (absolute gap 0.00 m at p95) | `verified` |
| nue2 **records the vertex hadronic shower**, nuatm does **not** | at equal `E_ν` the first recorded interaction carries 45–74% of `E_ν` in nue2 against 0.1–2.3% in nuatm, and sits exactly at the vertex (gap 0.00 m) versus 31–70 m downstream. Holds across 100 GeV – 100 TeV, so it is not a visibility threshold: a 10 TeV vertex shower would be seen from anywhere | `verified` |
| Showers are recorded by **light yield**, not by energy | the generator's `readme_sim_gvd`: *"We stored information about all interactions in which number of P.El 0.001"*. Confirmed in data: the minimum recorded shower energy rises with distance to the nearest OM (0.10 GeV within 25 m, 1.77 GeV beyond 50 m), and no shower is recorded further than 107 m from any OM | `verified` |
| `fRunN`/`fEventN` unusable for neutrino productions | −1 in every event of nuatm and nue2; filled for muatm | `verified` |
| Season is identified by channel count (2016 → 7 clusters) | 288 channels per cluster, cross-checked against `clusters_centers` in the h5 | `verified` |
| `targets_vec.py` reproduces `targets.py` | all columns agree to 1e-14 on 16,449 muons; end-to-end the two builders give bit-identical indices and counts on 25 parts | `verified` |
| `energy.C`'s `E_in` equals our `energy_at(s_in)` | both add losses walking back from the reference point; read from the code, **not yet run side by side** | `assumed` |
| The sig-noise model's float padding mask is by design, not a wrapper bug — so its output legitimately depends on batch size | Grigory's training code, `/net/63/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/models/encoder.py:79` on cluster62, has the identical `(~mask).float()`, reached from the run's own `train_config` via `EncoderDomainAdaptation.forward`. Read from the code that produced the weights | `verified` |
| MC and exp hit selections were not mutually consistent | MC probs written at batch size 256, exp at 512. The **256-vs-512** difference specifically is 82 of 249,189 hits crossing the 0.8 cut (0.03%), measured on MC muons; the often-quoted 0.49% is 256 against a padding-free `bs=1` reference, a different comparison. The effect on the *event* selection, on experimental data, is being measured by `validate_exp_probs.py` | `verified` |

| Signal **strings** must be counted from channel ids, not from coordinates | `channel // 36` is what `io.py:_count_sig_hits_strings` uses, and that function produced `n_sig_strings` in the probs files and `n_sn_strings` in the predictions — i.e. the h8s3 selection itself. Coordinates are cluster-centred (`coords_are_cluster_centered` in the h5), so two strings at the same position in different clusters collide; channel ids are global. Measured on 101,777 MC events across all three classes the two agree in **every** case, because events do not span clusters — the coordinate version is right by accident, not by construction | `verified` |
| the per-hit muon index in `raw/labels` is destroyed for shower codes with `k >= 17` | every pulse word in the generator's binary format is a float32 (`read_wout.py`, doc/mc_binary_formats.md), and float32 is exact on integers only to 2^24 = 16,777,216; `k*10^6 + j` crosses that at k = 17. Measured on all three 2020 productions: every code below 2^24 has `j >= 1` (0.000%), every code above it is even and decodes to `j` = 0, 2, 12, 64. Shares affected: nue2 22.6%, nuatm 0.09%, muatm 0.05% of shower hits | `verified` |
| every shower lies at `s >= s_track_start` | 100.000% on nuatm; on nue2 one cascade in 69.4M sits 6.11 m upstream of its muon's start (a 6 PeV shower on a 932 TeV muon). The rule holds to 1 part in 7 million, not absolutely | `verified` (with a measured exception rate) |
| the mapping from ROOT entries to companion rows is correct | ROOT track reference points and `fMuonEnergy` reproduce `muons_prty/individ` **bit for bit** on every muon of every part built; a reference point is a continuous 3-vector, which no wrong mapping matches by accident | `verified` |
| ROOT order of the shower chain is recoverable from the old sorted npz | `lexsort((s, int_block, track))` on the old extraction reproduces the new extractor's ROOT order exactly on `nuatm/1000` (36,710 showers) | `verified` |
| the sentinel `-0.001` and the sign of `s_track_start` separate exactly | 100.00% on all three productions: every sentinel starts downstream, no other muon does. The boundary is `s <= 0` and not `s < 0` -- a muon beginning exactly at the reference point gives `-0.0`, which fails a strict test; one such case in 10.5 M muons checked | `verified` |
| Experimental data carries repeated hits on a module ~6x more often than the muon MC | `n_sig_hits / n_channels > 1.2` on the h8s3 selection: exp 670,590 / 3,180,681 = 21.08%, muatm 800,008 / 23,167,718 = 3.45%, nue2 1.87%, nuatm 0.04%. A direct count in the prediction DBs of the FIXED_sn256 model | `verified` (the count) |
| Those repeated hits are **instrumental**, not scattered light | In MC a repeated hit tracks event brightness, as light-based explanations require: muon median charge per hit 4.47 (repeat) vs 3.56 (no repeat), nue2 19.33 vs 9.41. In exp the association is absent: 4.22 vs 3.97, track-likeness 0.85 vs 0.86. This contrasts two datasets and no more — no pulse-shape study, calibration run or DAQ multi-hit record has been consulted, and "the MC does not reproduce it" does not by itself identify the cause | `assumed` |
| The excess is not an artefact of the repeated-hit discrepancy | Restricted to events with exactly one hit per module, where a repeated hit cannot contribute by construction, the per-shape-band count still gives 1,727 observed against 798.1 predicted: 929 excess, ratio 2.16, higher than the 1.97 of the full sample. Independent of any explanation of the repeated hits | `verified` |
| The excess **cannot** be apportioned between hits-per-module subsets | The per-band prediction uses the muon selection efficiency measured *within* each subset, so predictions are not additive: 798.1 + 265.7 + 70.7 = 1,134.5 against 1,545.8 for the whole, while observed counts do partition (1,727 + 1,027 + 290 = 3,044). Ratios per subset are meaningful; "share of the excess" is not defined by this method | `verified` |
| Per-channel hit-time offsets are **geometric**, not a calibration fault | The per-channel mean of (t − event median t), over sig-noise-selected hits, is a monotone function of `channel % 36` (module position in string) running +247 ns to −147 ns. Removing that profile leaves 9.0–10.8 ns in all six experimental clusters (99% explained). **MC reproduces the same profile at r = 0.989–0.998** while having no calibration errors by construction — an anchor outside the experimental data | `verified` |
| The experimental excess is the same population as the misread simulated muons | In a 48-quantity feature space using no classifier output, a discriminator separates groups 6 and 8 at 0.714 held out — but separates the *low-score* groups 5 and 7 at 0.731 under identical treatment and matched sizes. The pair above the score cut separates **less** well than the pair below it, so the separation is the generic simulation/data gap and not a property of the excess. `groups.ipynb` §4.1 | `verified` |
| The nu-classifier's score cut selects the same region of feature space in simulation and in data | A tree fitted on simulation to separate high- from low-scoring muons scores 0.988 applied unchanged to experimental data (0.991 in its own domain); the reverse transfer is 0.990 against 0.987. Both domains lean on the same four features in the same order. This refutes the domain-shift worry the study was built around | `verified` |
| Nothing in the accepted experimental events lies outside the simulation | Isolation forest on all simulated groups: 0.1% of group 8 and 0.0% of group 6 below the cut holding 1% of held-out simulation. Both groups are shifted toward the atypical in the bulk (90.7% and 81.0% below the reference median against an expected 50%), which the group 6 control shows is a property of being selected on a score, not of the data | `verified` |
| A Cherenkov track fit gives usable directions for these events | Fitted direction against MC truth, which the fit never sees: median error 1.0 deg for the 42% of events above `fit_contrast` 0.9, a few degrees for the 99.1% above 0.6. Direction is taken from timing, never from the hit-cloud PCA, whose axis is near-vertical whatever the track did | `verified` |

## 3. Refuted — and what the bad argument was

Kept because the failure modes repeat, not for bookkeeping.

- **`Reg` is in GeV** *(first version)* — argued from `Reg − fMuonEnergy` ≈ 61 "GeV" ≈ 255 m of
  ionisation path. **Circular**: it assumed the two shared a unit, which was the question. The
  conclusion happened to be right, the argument was worthless.
- **`Reg` is in TeV** — argued from the kinematic bound `Reg ≤ E_primary`. The bound is real but
  proves only that the two agree *with each other*; the missing step was checking `E_primary`
  itself. Documented as established for a day.
- **`E_muon_before_shower` is lost in the ROOT conversion** — inferred from `simGVD.yaml`
  declaring it while `BMCInteraction` does not. **The spec described a newer generator than any
  production**; `schema_version: 4` in the YAML is unrelated to `format_version: 4` in the data.
- **Sentinel tracks carry a muon-energy chain in `.dat`** — a descending run of numbers matched
  the hypothesis. They were **pulse times** on different OMs. Confirmation-shaped search; the
  parser that settles it was two steps away.
- **`Reg` is useless as a target** — followed from the TeV error (it implied `Reg/E_mu` ≈ 1839).
  With GeV the ratio is 1.84 and `Reg` is a live candidate again, notably for sentinel tracks.
- **`.dat` gives 25× more statistics, which is an advantage** — true but irrelevant: the extra
  events are those that failed the trigger.
- **`sum(trk_n_inter) == len(int_x)` validates interaction ordering** — it holds by
  construction. Ordering was later established geometrically (collinearity, max deviation
  1.3 mm).
- **Showers are recorded only above 100 GeV, so `E_dep` under-reports the loss by ~2×** — two
  compounding mistakes: the stored minimum (already in GeV) was read as TeV, inflating the
  threshold a thousandfold; and the "shortfall" was a median of a skewed distribution compared
  against a mean (`a + bE`). On means there is no shortfall. Found by a question about a
  bimodal histogram whose x-axis was wrong for the same unit reason.
- **The sig-noise wrapper's float mask was our own bug, so the weights are sound and every
  downstream artifact must be rebuilt** — argued from two measurements: with a boolean mask the
  output stops depending on batch size, and it matches a padding-free `bs=1` reference. Both are
  properties of *correct masking in the abstract*; neither can distinguish the rival hypothesis
  that the model was **trained** with the leak, which is what turned out to be true. The one
  number pointing that way (AUC very slightly *better* with the leak) was waved off as noise.
  Cost: a plan to recompute 176 GB of probabilities and retrain the classifier, plus 29
  deprecation markers on artifacts that were fine. The falsifying check — read the training
  code, which the run's own hydra config locates — took two minutes over `ssh`.

- **"The first-detected-light hypothesis is refuted"** — I called it dead on a +0.06 gain in
  `corr(log Reg, log E_pred)`, which is indeed a weak statistic when both sides share the
  dominant `E_ref` term. The right test is the residual: `corr(log(Reg−E_ref), log(correction))
  = +0.71`, i.e. the correction genuinely explains 40% of the gap. The hypothesis is
  *incomplete*, not wrong — the point sought lies further upstream.
- **The registration volume has a 250 m radius** — inferred from `fMaxDistance = 25000` read as
  centimetres, plus `Reg − fMuonEnergy` ≈ 255 m of ionisation path. Reconstructing the entry
  points directly (notebook §9) puts 86% of them *outside* 250 m, median 381 m. Both supporting
  arguments were weak: the first an assumed unit, the second an average over a broad
  distribution mistaken for a length.

- **"Both halves carry the excess in comparable number" (929 with one hit per module, 969 with
  repeats)** — stated in `FINDINGS.md` as though the two were complementary parts of the 1,498.
  They are not: 929 + 969 = 1,898, and the two subsets are not even complementary (the 969 is
  `> 1.0`, the 929 is `== 1.0`, which do partition — the failure is arithmetic, not overlap).
  The predictions are computed per subset with that subset's own muon efficiency, which is
  precisely a correction for the composition difference under discussion, so they do not sum.
  Caught by asking why three disjoint bins gave 1,909.5 against a total of 1,498.2. The
  surviving claim is the `== 1.0` row on its own.

- **"Per-channel time offsets of ±100 ns are present in the data and absent in MC, and
  contaminate every timing feature"** — written into the group-structure plan as confounder #1,
  and into DATA_QUALITY §7 as "cause not separable". The MC half was never measured; it was
  assumed because the offsets were found while looking at experimental data quality. Measuring
  it took ten minutes: MC has the same offsets, std 98.9 ns, and the profile matches the data's
  at r = 0.989. 97–99% of the pattern is module depth within the string; the instrumental
  residual is 9–11 ns. Caught by the user asking "have you actually measured this in MC?".
  The general shape of the error: a defect found while auditing one dataset was assumed to be
  a property of that dataset, with no control.

- **"`fit_contrast` above 0.25 means a track direction was found"** — read off a scrambled-times
  null, where a real track gave 0.900 against a 95th percentile of 0.234, and written into the
  notebook as a usable threshold. MC truth says there is no threshold: median zenith error falls
  smoothly from 18 deg at 0.25 to 1 deg above 0.9, and 99.1% of real events sit above 0.6. The
  null was real but answered a different question — it separates a real event from a scrambled
  one, not a fitted event from an unfitted one.
- **An AUC of 0.806 between groups 6 and 8** — produced by setting the neutrino-like subset
  aside from group 8 only. Removing it from both gives 0.704, and removing it from neither gives
  0.749. The asymmetry manufactured the difference; the inflated number looked like a result and
  would have been reported as one.

## 4. Open questions

- The geometry of the registration volume: notebook §9 rules out a sharp spherical shell and
  contradicts 250 m, but does not pin a boundary. Ask the production's author.
- Whether `E_in`/`E_out` from `targets.py` numerically match `energy.C` on shared events.
- What target the energy network should regress — deliberately open, see
  [mc_energy_truth.md](mc_energy_truth.md) §5.
