# Is the excess a flux deficit or a detector-response error?

The nu-classifier accepts about three times more experimental events than
simulated atmospheric muons predict. Earlier work (`../excess_mechanism`)
established that the accepted events are mis-reconstructed muons of the kind MC
does produce, that no single domain property explains the surplus, and that even
the full multivariate domain shift does not: reweighting MC to the experimental
feature distribution moves the top band only from 3.11 to 2.62.

Two explanations survive, and the paper's conclusions differ between them —
under (a) only the background estimate is broken, under (b) the neutrino
efficiency measured on simulation does not transfer either. `PROTOCOL.md` states
both precisely and, for each test, the observation that would kill it.

## Running it

```bash
make smoke      # whole chain on a small subset, minutes
make all        # the real thing
make verify     # re-check provenance without recomputing
make catalogue  # what every file in data/ contains, in words
make test       # the synthetic-track oracle for the fitter
```

`config.yaml` holds every path, threshold, binning and seed. No constant is
defined inside a script.

## Layout

| | |
|---|---|
| `PROTOCOL.md` | hypotheses, discriminating observables, falsifiers — frozen before running |
| `RESULTS.md` | findings with confidence labels, append-only |
| `src/` | importable library, no side effects |
| `stages/` | numbered; each writes exactly one artefact plus a provenance sidecar |
| `notebooks/` | **read artefacts and plot; they never compute** |

## Reading order

Start at `notebooks/99_summary.ipynb` — a short overview of all four tests, with
pointers into the detail. Then, in the order the tests were run:

| notebook | test |
|---|---|
| `10_run_noise.ipynb` | test 4 — is the excess driven by water noise? |
| `20_flux_reweight.ipynb` | test 2 — can a flux error explain it? |
| `30_response.ipynb` | test 1 — is the detector response the same? |
| `40_jitter.ipynb` | test 3 — does the measured response error reproduce it? |

Every table a notebook displays goes through `src/present.py`, which renames the
columns, states what a row is, and names the stage and artefact the numbers came
from. The glossary lives in the same module, so a term is defined once and every
notebook agrees on it.
| `data/` | generated artefacts, each with `<name>.meta.json` |

## Rules this directory follows

1. Notebooks never compute, so they cannot drift from the scripts and rerun in
   seconds.
2. Every artefact carries git commit, config hash, input hashes, row count and
   runtime; `make verify` re-checks the chain.
3. `PROTOCOL.md` is written before a test runs and is not edited to match the
   outcome.
4. `RESULTS.md` is append-only; refutations are added in place.
5. Sampling is deterministic — `ORDER BY hash(...)`, never `USING SAMPLE`.
6. `make smoke` proves reproducibility cheaply before hours are spent.

## Stage status

| stage | test | state |
|---|---|---|
| `00_index` | preconditions for the cross-cluster anchor | done — anchor unavailable, see RESULTS |
| `10_run_noise` | test 4, noise as a natural experiment | done — predicted effect absent, design underpowered; per-cluster lead |
| `20_flux_reweight` | test 2, is a flux reweighting sufficient | done — a bounded flux reweighting cannot produce the excess |
| `30_split_response` | test 1, leave-one-out response comparison | done — experimental residuals wider in 66/66 matched cells |
| `31_response_compare` | test 1, matched comparison and calibration | done — MC lacks 28.7 ns of timing spread |
| `40_time_jitter` | test 3, does the measured jitter reproduce the excess | done — 30 ns gives 2.80, observed 2.87 |
| `41_jitter_validate` | does jittered MC also *look* like data | done — matched calibration 29.9 ns, closes the loop |
