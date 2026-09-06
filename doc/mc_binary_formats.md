# Reading the MC binaries: `.dat` and `.wout`

Both upstream formats are streams of 32-bit words wrapped in Fortran unformatted records. The
layout has to be exact — shift by one word and the coordinates silently become plausible
nonsense rather than an error. This document is the reference for
[read_wout.py](../data_manager/root2h5/energy_truth/read_wout.py), which implements it.

Where these files come from and where they live: [mc_provenance.md](mc_provenance.md).

## 1. Source of truth

The layout is transcribed from BARS, not inferred:

| what | file |
|---|---|
| record framing | `BARS/bars/programs/mcread/ZDataReader.cpp` |
| event layout (`.dat`) | `BARS/bars/programs/mcread/Task_MAIN.cpp` — `Task_MAIN::process` |
| event fields | `BARS/bars/bmcread/BMCEvent.cc` — `BMCEvent::ReadFromBinary` |
| field meanings | `BARS/bars/bmcread/BMCEvent.h`, `BARS/bars/simgvd/*.h`, `simGVD.yaml` |
| geometry, units | `BARS/bars/bgeom/BReadMCGeomWout.cc` (`kCM_TO_M`) |

An older Python sketch exists as `BARS/bars/simgvd/datfilelayout.py`; it is useful but reads
the channel block differently from `Task_MAIN.cpp`, so prefer the C++.

## 2. Record framing

```
[n_bytes][n_words][payload ... ][n_bytes]
```

`n_bytes / 4` counts everything between the two byte counts, i.e. `n_words` plus the payload
plus one trailing terminator word. The closing count is verified by `read_wout.py`, which is
what turns a wrong offset into a loud failure instead of garbage.

The **first record is the configuration**; every record after it is one event.

## 3. Layout

### Configuration record

```
[0] n_words   [1] endianness=100   [2] format_version   [3] mc_version
[4] n_channels   [5] n_strings
[6 ...]  channel table, 6 words per channel (coordinates in centimetres)
then:    date, generation_model, generation_condition, output_trigger,
         sine_cherenkov, sigma, exponent, background_flag, max_distance
```

`generation_model` follows `BMCArrayConfig::GenerationModel_t` (4 = `ReadFromExternalFile`
for muatm, 5 = `NeutrinoGenerationInVolume` for nuatm/nue2). Coordinates in this record are in
centimetres; whether `max_distance = 25000` follows the same unit is **unresolved** — reading it
as 250 m is contradicted by the data, see [mc_provenance.md](mc_provenance.md) §5.

### Event record

```
[0]  n_hit_channels        [1] theta_primary   [2] phi_primary   [3] A_primary
[4]  E_primary             [5..7] primary vertex (0,0,0 for neutrino generation)
[8]  t_first_muon          [9] E_bundle_surface   [10] E_bundle_reg
[11] muon cipher           [12] n_tracks
[13] run (−1)              [14] event (−1)        [15] weight
[16] n_showers of track 1

per track:  [n_showers][E_muon, t_muon, x, y, z][showers]
            shower = 4 words: (E, x, y, z)
            stride = 6 + 4 * n_showers
```

The one subtlety: a track's shower count sits in the word **before** its fields, so the count
of the *next* track is the word right after the current track's showers. Hence the stride
`6 + 4·n_showers` rather than `5 + 4·n_showers`.

After all tracks come `n_hit_channels` channels, and this is **the only place the two formats
differ**:

```
.dat    [number][_][_][n_pulses]  + n_pulses × (amplitude, time, magic)
.wout   [number][n_pulses]        + n_pulses × (amplitude, time, magic)
```

then a single terminator word closes the record.

## 4. What is *not* in these files

`simGVD::Shower` in `BARS/bars/simgvd/Shower.h` declares three members:

```cpp
float E_shower;                       // [TeV]
std::array<float,3> position_shower;  // [cm]
float E_muon_before_shower;           // [GeV]
```

The third one — the muon energy before each stochastic loss — would make the whole
extrapolation in [mc_energy_truth.md](mc_energy_truth.md) unnecessary, including for the 14–24%
of muons whose `fMuonEnergy` is the −0.001 sentinel.

**It does not exist in any available production.** A shower is 4 words everywhere:

| production | date | showers |
|---|---|---|
| 2020 `prod1/nu2.100PeV/wout/v1.1` | 2024-07 | 4 words |
| 2020 `nu2/wout/v3.0`, `nuatm/wout/v4.0` | 2025-05 | 4 words |
| 2020 `prod1/nuatm/simgvd/*.dat` | 2022-01 | 4 words |
| 2024 `nuatm/wout/v2.0` (135 strings) | 2025-05 | 4 words |

`simGVD.yaml` describes a newer generator than anything produced so far; `schema_version: 4`
in that file is the version of the schema document, unrelated to `format_version: 4` in the
data. If the collaboration ever reproduces MC with the current simGVD, this path becomes worth
revisiting.

## 5. Verification

The layout is not trusted on reading — it is checked:

- **Record lengths.** Parsing consumes exactly the declared number of words for
  **20,000 / 20,000** events of `nu_mu_1000.dat` and for every event of three `.wout` files.
  With 5 words per shower instead of 4 the rate drops to 79%, and without the terminator word
  to 0% — so the layout is pinned, not merely consistent.
- **Against ROOT.** For the same event, `.dat` channels are `1873, 1874` and ROOT's are
  `1873, 1874`; shower energies and coordinates agree to the last digit
  (`E=1.97206e-04, xyz=(−312.4595, 101.5164, 244.3206)`).
- **Channel numbers.** Over 3,000 events: none outside 1..2016, none out of ascending order.
- **Times.** `.dat` 167.951 / 122.931 / 124.854 ns against ROOT 166.056 / 119.722 / 124.187 —
  differing by the jitter `mcread` adds. ROOT additionally contains noise pulses
  (`fMagic = 1.0`, e.g. t = −2061.2 ns) that `.dat` does not have.

## 6. Traps

- **The tail of a record is hits, not physics.** After the showers come channel numbers,
  amplitudes, times and `fMagic` markers (−999999 and 1000000 for signal pulses). A descending
  run of values there is a sequence of **pulse times** on different OMs, not muon energies —
  times need not be monotonic, while channel numbers always are. This cost a day of work.
- **Offsets differ between the formats' documentation and code.** `Task_MAIN.cpp` indexes the
  payload after the leading word count; a parser written against the record start is off by
  one and still "works" for length checks while returning shifted coordinates.
- **Units are mixed inside one record**: shower energies are TeV, `E_muon` is GeV, coordinates
  are metres in the event records but centimetres in the configuration record.
- Comments in BARS disagree with the data on units — see
  [mc_provenance.md](mc_provenance.md) §6.

## 7. Usage

```bash
python3 data_manager/root2h5/energy_truth/read_wout.py FILE --events 3
python3 data_manager/root2h5/energy_truth/read_wout.py FILE --verify 20000
```

```python
from read_wout import read_file, iter_events

cfg, events = read_file("nu_mu_1000.dat", max_events=100)
for ev in iter_events("nu_mu_1000.wout"):
    for track in ev["tracks"]:
        track["e_muon"], track["showers"]      # (n, 4): energy, x, y, z
```

The format is inferred from the extension; pass `--format dat|wout` to override.
