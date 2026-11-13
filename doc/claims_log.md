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
| In mc_merged the classes occupy **disjoint zenith hemispheres**: muatm above 90 deg, both neutrino classes at or below it | `prime_prty[:, 0]` over out-of-training parts: muatm spans 94.22-179.92 deg with 100.00% above 90; nuatm 0.16-90.00 and nue2 0.11-90.00 with 0.00% above 90, both capped at **exactly** 90.00 — the signature of a generation cut, not of a physical edge. Consequence for any MC-trained selection: no atmospheric muon and no neutrino share a zenith angle here, so a feature correlated with theta separates the classes perfectly in MC and cannot do so on data. Which end of the axis is the zenith is **not** established by this measurement — the generator was not read | `verified` (the disjointness; the angle convention is `assumed`) |
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
| `exp.h5` and `exp_full.h5` hold **disjoint detector runs** | the two files' part keys (`part_s{season}_c{cluster}_r{run}`, assigned upstream by the DAQ) were listed and intersected: 26 keys against 29, intersection empty. So no `exp_full` event was in the DA target set of any model trained on `exp.h5` | `verified` |
| Prefilter scores shift by **one float32 ulp** with inference batch size or read-window size | exp.h5, 649,976 events, one checkpoint: batch 512 vs 2048 moves 37.8% of scores, chunk 5,000 vs whole-part moves 5.3%; max \|Δ\| is 1.192e-07 in both — exactly `np.spacing(float32(1.0))`. The model pads each batch to its longest event, so composition fixes the order of a float reduction. Unlike the sig-noise model, nothing is *selected*, so this changes no hit and no count | `verified` |
| The prefilter NPY dataset records **no per-event back-link** to the HDF5 | `prefilter_npy_ds_builder/build.py` writes `offsets/labels/signal_hits/signal_strings/particle_types` and no `h5_part_keys.npy`; the directory confirms it. The finest available training exclusion is therefore the part list in its `build_config.yaml` (952 parts), which is what `reader.NOT_TRAINED_PARTS` uses | `verified` |
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

| The DQM channel mask BARS applies to data is reproducible from the run's DQM solution file | `BApplyDQMMask` reads `j{run}.fit.solution.txt` (`DQM_FIT_SOLUTION`, dqm-version v1.3) and consults the Gantt histograms only when `Mask.UseGantt` is 1, which is 0 in every 2020 resource file; the file's own legend defines the codes. Measured on `c01/r0027`: the solution reproduces the ROOT `ChannelMaskDQM` container exactly, the same 40 channels event after event. On the eleven `exp_reco` runs where both are unpacked, every channel the solution masks is masked in the container too | `verified` |
| The DQM mask is the whole residual in the port's hit selection | the scanfit mask of 2000 `exp_reco` events agrees with BARS in 93.4% of events without the mask and **100.0%** with it; `mc_reco` muatm, which BARS reconstructs with no mask (`UseDQMChannelMask 0`), agrees in 100.0% unmodified. `scf_max_theta` follows the same pattern (97.0% to 100.0%) | `verified` |
| The per-event *section* part of that mask costs at most 0.2% of hits | the solution file is run-level; the container additionally switches off whole sections per event, from a `BSectionStatus` file not unpacked locally. Over 200 events spread across each of the eleven runs where both parts are visible, the extra channels carry 0 hits in eight runs and 0.18-0.21% of hits in the other three (10-13% of events) | `verified` |
| `fEvCenterZ` is written in BARS' geometry frame, not in the frame of the h5 hits | on 2000 `exp_reco` events our charge-weighted z in hit coordinates differs from `fEvCenterZ` by 361 m, and adding `clusters_centers` closes it to 1.8e-5 m (the cluster sits at z = 359.8). The `-354` the analysis tree applies to data is not needed with these files | `verified` |
| The port reproduces the BARS chain but not the BARS minimiser | on BARS' own ROOT file (200 events) the port reproduces `ScfTheta`, `FuncValueScf` and `TimeChi2Scf` to 1e-3 in 100% of events, `ThetaInit` in 91% and `ThetaRec` in 86% (median relative difference 1e-4). Where the hit sets agree and the directions differ by more than a degree, the port's chi-squared is *lower* than BARS' in 69% of cases. `n_calls` counts calls of the other minimiser and does not agree (1%) | `verified` |
| The port's lower chi-square on disagreeing events is a *different* minimum of the **same** functional, and not a worse track | recomputing the port's own objective at BARS' track (from `xyzRec_*`/`dirRec_*`, shifted into the frame of the hits, the time profiled out on both) reproduces BARS' `fTimeChi2` to a median relative difference of 1e-5, so functional, hit set and frame are shared; the port's track is lower in 55% of 2000 `mc_reco` events and 76% of those where the directions differ by more than a degree. Against MC truth, on the 191 events with a single muon the port's direction is closer to the primary particle's in 57% (61% of the 56 that disagree by more than a degree; median error 7.4 against 9.6 deg) | `verified` |
| `Mask.UsePrefilter` does not affect the reconstruction | the `exp_reco` files come from `reco-muon_scl_nu_DATA.rc` (Prefilter 1, prescale 1000), but `BMuonStreamPrefilter` only returns `kCONTINUE`/`kTRUE` for the monitoring stream; neither the hit selection nor the fit reads its output | `verified` |

| Точная геометрия 2020 лежит локально в ROOT (`BGeomTel.fOMs.fPosition`) и совпадает с позициями хитов в h5 | `part_s2020_c04_r0031` (данные) и `muatm_2020/part_1000` (МС): макс \|xyz_ОМ − xyz_хита\| = **7.6 мкм** и **7.4 мкм**; у МС таблица статична (разброс по событиям 0 м, 2016 каналов) | `verified` |
| Экспериментальная геометрия хранится **на каждое событие** и плывёт по рану | ветка `fPosition` имеет форму (3, n_events, 288). На `c04_r0031` (3.04M событий, таблицы в 12 точках рана) пер-событийная таблица отличается от медианы рана на 0.07–0.46 м по медиане ОМ, худший ОМ **1.54 м**; первые 3000 событий дают лишь миллиметры — короткое окно дрейф не измеряет. Сверка с хитами h5: пер-событийная таблица 7.6 мкм, снапшот рана — до 0.51 м | `verified` |
| `fPnonHit` в BARS — геометрическое среднее P(молчит) по **неиспользованным** ОМ в 22 м, перпендикулярное расстояние, сентинел `1e-9` | прочитано в `bmuon/BMuonCandidate.cc:930-957` (`GetPnonHitProd`) и вызвано из `BRecoMuonProducer.cc:640`; «пустой» = включён маской ∧ ∉ `fUsedChans`, то есть импульс, выброшенный фитом, всё равно считается «пустым»; при нуле пустых — `pow(10,-9)`. Namespace-копия (`BMuonNamespace.cc:643`) отличается световым расстоянием и возвращает 0 | `verified` |
| Наша реализация `pnonhit` воспроизводит `fPnonHit` BARS, а видимое расхождение — расхождение самого фита | `validate_vs_bars.py`, 140 событий `exp_reco` (30 частей × 5), перпендикулярная модель и «пустой = не использованный`: на **треке самого BARS** медианное относительное расхождение **1.8–3.5e-7**, 59–74 % лучше 1e-6 (остаток — float32 ветки ROOT); на **нашем** треке то же сравнение даёт 4.6e-4–1.9e-3, что совпадает по порядку с расхождением `time_chi2` (8.5e-4) — то есть мерой фита (Minuit2 против iminuit), а не инструмента | `verified` |
| `wout` event words 5–7 carry the **primary vertex** in the 2025 files | `newout4_1.f` writes `graxis` there; the 2019–2023 v1.0 files have `(0,0,0)` (`nu2`/`nuatm` 1000). On sentinel tracks, the first `nu2` v3.0 event and events 5, 6, 8 of `nuatm` v4.0 (file 1226, batch 29) match `ref − c·fFirstMuonTime·d` to ≤2e-4 m. The formula is **not** valid for alive tracks (the first `nuatm` v4.0 event differs by ~180 m); the header field itself is the vertex | `verified` |
| The neutrino generator has two lineages: **4w** (atmospheric; no `nu_cascade`) and **7w/8w** (E⁻²; fills `nu_cascade`) | `munu4w_osc.f` has no `nu_cascade`; `munu7w_osc.f:495-502` and `munu8w_osc.f:436-443` fill it; `sim3h1.f:2143` adds the vertex cascade only if `nu_int=1`. Production logs: `nuatm` = Bartol/daemonflux (4w), `nu2` = power 2 (8w); data: `nuatm` has no vertex cascade, `nu2` does | `verified` (code path and logs; the exact build commit is `consistent`) |
| The inelasticity model is an **unversioned relative-path table**, and the samples disagree with it | `cc_over_y.ascii` (all repo copies identical, md5 `acf9954a…`) gives `<y>_ν = 0.479` at 30 GeV; the measured `nu2` ν is 0.502 and `nuatm` ν is 0.301 (`nuatm` ν̄ 0.231 vs the table's 0.351). `nuatm` v1.0 and v4.0 both show ~0.30; the production table is not preserved and `y` is not stored per event | `verified` (the numbers); the cause — a different production table/implementation — is `consistent` |

### Angle/track release (`release_v3`)

| claim | evidence | status |
|---|---|---|
| The released bundle, SN checkpoint and norms are the files named in the model card | SHA256 of `angle_track_uncertainty.pt` is `7cb8f77c…`, `signal_noise.ckpt` is `be9cc39b…`, norms match; both loaders verify the hashes and parameter counts (3,079,266 / 992,513) | `verified` |
| The angle/track backbone does **not** have the float-mask batch bug | `SetTransformerDirection.forward` uses `src_key_padding_mask=~full_mask` (bool). The same event alone vs padded next to a longer event: direction max \|Δ\| 9e-8, offset 3.8e-6, log_angle 0 — float reduction order only | `verified` |
| The bundled release SN classifier **does** have the float-mask batch bug | `EncoderTwoHead.forward` uses `key_pad = (~mask).float()`, an additive bias. The same hit probability changes by up to 0.177 when the event is padded next to a longer one; the p>=0.70 selection therefore moves with batch composition. Same mechanism as `doc/sig_noise_batch_size.md` | `verified` |
| Published 68%/95% radii apply only to unchanged release-SN p>=0.70 MC rows | Upstream `infer_rethresholded` deliberately leaves calibrated radii blank for gate rows; the model card ties them to the p>=0.70 pipeline. Our adapter emits NULL radii for canonical-SN, experimental and gate rows. The canonical/exp exclusion is a conservative policy, not a measured coverage result; no experimental truth exists | `consistent` (canonical/exp exclusion) |

## 2b. Energy-reconstruction experiments

| claim | evidence | status |
|---|---|---|
| DA lowers the held-out **domain-probe** AUC in every matched energy pair, but does not collapse it to chance | `analysis/reports/da_probe.csv` from `analysis/domain_probe.py`: 16 checkpoints (18 after AB finishes) on 5,000 `muatm` + 5,000 `exp_full` events held out from DA training; three MLP seeds and a linear probe, event-level and part-disjoint splits. Matched pairs: `A`/`BASE`, `AD-*`/`D-*`, `AC-*`/`C-*`; all MLP deltas are negative. Example part-disjoint MLP: `A` 0.599 vs `BASE` 0.665, `AD-T5` 0.626 vs `D-T5` 0.687. | `consistent` for the measured AUC direction; the probe is not a certificate of domain invariance and a residual AUC can reflect legitimate MC/data differences |
| Frozen-head `H`/`AH` probe values equal their base encoders, so they are not no-DA controls | `da_probe.csv` rows; `H-*` uses `BASE` as resume base, `AH-*` uses `A`; the probe reads encoder features, and the frozen head does not change them | `verified` (by construction and identical measured AUCs) |
| The working model beats the HGB handcrafted-feature baseline on the held-out test sample, but HGB is slightly better between 0.3 and 10 TeV | paired `event_fk` merge of the HGB `test.duckdb` and the AC-G test predictions: 418,153 events; MAE 0.5212 (HGB) vs 0.4748 (working), paired ΔMAE −0.0464 with block-bootstrap 95% interval [−0.0477, −0.0448]; by class 0.413→0.383 (atmospheric) and 0.556→0.505 (astrophysical); the binned curve crosses around 0.3 and 10 TeV. Generated by `doc/energy_training_schemes/build_acg_quality_deck.py`; notebook `analysis/non_neural_energy_baseline.ipynb` | `consistent` for this test sample, one checkpoint and one HGB configuration |
| Per-event sigma carries information beyond a sigma depending only on predicted energy | `analysis/reports/sigma_vs_binned.csv` from `analysis/sigma_vs_binned.py`: on `D-G/AC-G/B-G/H-G`, fit one scale per predicted-energy bin and one global scale for the learned sigma on half the test parts; on the other half Gaussian NLL is lower for the learned sigma by 0.070–0.129 (95% part-bootstrap CIs exclude zero) and within-bin Spearman(sigma, abs error) is 0.29–0.35. Coverage is calibrated to ~0.683/0.951 for both | `consistent` for this split-half test; Gaussian only and one split-half realization; it does not beat an equally flexible conditional baseline built on other *observable* covariates. MC `theta` is truth and cannot enter such a baseline |

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

- **The port's disagreement with BARS is `Float_t` arithmetic** — argued from BARS storing
  its `double` intermediates in `Float_t` members (`fRefTime`, `fX0`, `fSumCharge`) while the
  port works in double, and tested by emulating the C++ types through the whole fit. **The
  agreement did not move** (`ThetaInit` 90% before, 90% after), so the precision is not the
  cause; the residual is the minimiser. A plausible mechanism is not a measurement.

- **"The wout primary vertex is `(0,0,0)` for neutrino generation"** — true for the 2019–2023
  v1.0 files, generalized to the format. `newout4_1.f` writes `graxis` into words 5–7, and the
  2025 v3.0/v4.0 files carry the actual vertex there (first event of `nu2` v3.0:
  `(53.88, −229.34, −113.56)`, matching `ref − c·fFirstMuonTime·d`). The old claim came from
  looking only at v1.0.

## 4. Open questions

- The geometry of the registration volume: notebook §9 rules out a sharp spherical shell and
  contradicts 250 m, but does not pin a boundary. Ask the production's author.
- Whether `E_in`/`E_out` from `targets.py` numerically match `energy.C` on shared events.
- What target the energy network should regress — deliberately open, see
  [mc_energy_truth.md](mc_energy_truth.md) §5.
