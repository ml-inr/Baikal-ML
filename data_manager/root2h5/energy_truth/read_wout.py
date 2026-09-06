"""Reader for the Baikal MC binary files (`.dat` from the generator, `.wout` after mcread).

These are the files the ROOT productions are built from: the Fortran generator writes `.dat`,
`mcread` adds noise and a time jitter and applies the trigger to produce `.wout`, and
`bexport-mc` turns that into the ROOT trees we normally use. Everything is a stream of 32-bit
words wrapped in Fortran unformatted records, so the layout has to be known exactly -- one
misplaced word and the coordinates silently become garbage.

The layout below is not guessed: it is transcribed from `BARS/bars/programs/mcread/`
(`Task_MAIN.cpp` parses a `.dat` event, `ZDataReader.cpp` the record framing) and checked
against the ROOT files built from the very same data. See doc/mc_binary_formats.md.

The two formats differ in one place -- how a hit channel is laid out -- which is why the
format is a parameter rather than an assumption:

    .dat   channel = [number, _, _, n_pulses] + n_pulses x (amplitude, time, magic)
    .wout  channel = [number, n_pulses]       + n_pulses x (amplitude, time, magic)

Usage:
    python3 read_wout.py FILE [--events 5] [--format auto|dat|wout]

    from read_wout import read_file, iter_events
    cfg, events = read_file("nu_mu_1000.dat", max_events=100)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

WORDS_PER_SHOWER = 4          # (energy, x, y, z) -- verified on 2020 and 2024 productions
WORDS_PER_TRACK = 6
WORDS_PER_CHANNEL_CONFIG = 6  # per-channel entry in the configuration record
HEADER_WORDS = 17
CM_TO_M = 0.01


def _records(words: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield (start, length) of each Fortran record, both in words.

    A record is [n_bytes][payload][n_bytes]; the trailing count is verified, which is what
    makes a wrong offset fail loudly instead of returning plausible nonsense.
    """
    pos = 0
    while pos + 1 < len(words):
        n_bytes = int(words[pos])
        n_words = n_bytes // 4
        if n_words <= 0 or pos + 1 + n_words >= len(words):
            return
        if int(words[pos + 1 + n_words]) != n_bytes:
            raise ValueError(f"record at word {pos} is not closed by its byte count — "
                             f"the file is truncated or the offset is wrong")
        yield pos + 1, n_words
        pos += 2 + n_words


def parse_config(f: np.ndarray) -> dict:
    """Telescope configuration, the first record of the file.

    Channel coordinates in this record are centimetres (BReadMCGeomWout applies kCM_TO_M).
    Whether `max_distance` uses the same unit is unresolved -- reading 25000 as 250 m is
    contradicted by the entry points reconstructed from Reg -- so it is returned raw.
    """
    n_ch = int(f[4])
    tail = 6 + WORDS_PER_CHANNEL_CONFIG * n_ch     # generation parameters follow the channel table
    return dict(endianness=float(f[1]), format_version=float(f[2]), mc_version=float(f[3]),
                n_channels=n_ch, n_strings=int(f[5]), n_clusters=n_ch // 288,
                date=int(f[tail]), generation_model=float(f[tail + 1]),
                max_distance_raw=float(f[tail + 8]))


def parse_event(f: np.ndarray, fmt: str) -> dict:
    """One event record. `f` is the payload with the leading word count already stripped.

    Field meanings follow BMCEvent::ReadFromBinary. Note the two traps documented in
    doc/mc_binary_formats.md: for neutrino generation `run`/`event` are -1 and the primary
    vertex is (0,0,0), so neither can be used to identify an event.
    """
    n_ch, n_tracks = int(f[0]), int(f[12])
    ev = dict(n_hit_channels=n_ch, theta=float(f[1]), phi=float(f[2]), a_primary=float(f[3]),
              e_primary=float(f[4]), t_first_muon=float(f[8]), e_bundle_surface=float(f[9]),
              e_bundle_reg=float(f[10]), n_tracks=n_tracks, run=float(f[13]),
              event=float(f[14]), weight=float(f[15]))

    # Track block: [n_showers][E_mu, t, x, y, z][showers...], and the *next* track's shower
    # count is the word right after this block's showers -- which is why the count is read at
    # p-1 and the stride is 6 + 4*n_showers rather than 5 + 4*n_showers.
    p = HEADER_WORDS
    tracks = []
    for _ in range(n_tracks):
        n_shower = int(f[p - 1])
        e_muon, t_muon = float(f[p]), float(f[p + 1])
        x, y, z = (float(f[p + 2]), float(f[p + 3]), float(f[p + 4]))
        s0 = p + WORDS_PER_TRACK - 1
        showers = np.array(f[s0: s0 + WORDS_PER_SHOWER * n_shower],
                           dtype=np.float64).reshape(n_shower, WORDS_PER_SHOWER)
        p += WORDS_PER_TRACK + WORDS_PER_SHOWER * n_shower
        tracks.append(dict(e_muon=e_muon, t_muon=t_muon, position=(x, y, z), showers=showers))
    ev["tracks"] = tracks

    p -= 1
    channels = []
    step = 4 if fmt == "dat" else 2          # the one real difference between the two formats
    for _ in range(n_ch):
        number = int(f[p])
        n_pulses = int(f[p + step - 1])
        p += step
        pulses = np.array([f[p + i * 3: p + (i + 1) * 3] for i in range(n_pulses)],
                          dtype=np.float64).reshape(n_pulses, 3)
        p += 3 * n_pulses
        channels.append(dict(number=number, pulses=pulses))
    ev["channels"] = channels
    ev["_words_used"] = p + 1                # +1 for the trailing terminator word
    return ev


def detect_format(path: Path) -> str:
    return "dat" if path.suffix == ".dat" else "wout"


def iter_events(path: str | Path, fmt: str = "auto",
                max_events: int = 0) -> Iterator[dict]:
    """Yield events. The configuration record is consumed and skipped."""
    path = Path(path)
    if fmt == "auto":
        fmt = detect_format(path)
    words = np.fromfile(path, dtype="<u4")
    floats = words.view("<f4")
    for i, (start, n_words) in enumerate(_records(words)):
        if i == 0:
            continue
        payload = floats[start + 1: start + n_words]   # drop the leading word count
        ev = parse_event(payload, fmt)
        if ev["_words_used"] != n_words - 1:
            raise ValueError(f"event {i}: consumed {ev['_words_used']} words of {n_words - 1} — "
                             f"layout mismatch, do not trust this file")
        yield ev
        if max_events and i >= max_events:
            return


def read_file(path: str | Path, fmt: str = "auto", max_events: int = 0) -> tuple[dict, list]:
    """Configuration record plus a list of events."""
    path = Path(path)
    words = np.fromfile(path, dtype="<u4")
    floats = words.view("<f4")
    start, n_words = next(_records(words))
    cfg = parse_config(floats[start: start + n_words])
    return cfg, list(iter_events(path, fmt, max_events))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--events", type=int, default=3, help="how many events to print")
    ap.add_argument("--format", default="auto", choices=["auto", "dat", "wout"])
    ap.add_argument("--verify", type=int, default=0,
                    help="parse N events and only report whether the layout holds")
    args = ap.parse_args()

    cfg, _ = read_file(args.file, args.format, max_events=1)
    print(f"{args.file}")
    print(f"  format_version={cfg['format_version']:g}  mc_version={cfg['mc_version']:g}")
    print(f"  channels={cfg['n_channels']}  strings={cfg['n_strings']}  "
          f"clusters={cfg['n_clusters']}")

    if args.verify:
        n = sum(1 for _ in iter_events(args.file, args.format, args.verify))
        print(f"  layout verified on {n} events")
        return

    for i, ev in enumerate(iter_events(args.file, args.format, args.events), 1):
        print(f"\n  event {i}: theta={ev['theta']:.4f} phi={ev['phi']:.4f} "
              f"E_prim={ev['e_primary']:.5g} E_reg={ev['e_bundle_reg']:.5g} "
              f"weight={ev['weight']:.5g} tracks={ev['n_tracks']} channels={ev['n_hit_channels']}")
        for t in ev["tracks"]:
            print(f"     muon E={t['e_muon']:.5g} GeV at "
                  f"({t['position'][0]:.2f}, {t['position'][1]:.2f}, {t['position'][2]:.2f}) m, "
                  f"{len(t['showers'])} showers")
            for s in t["showers"][:3]:
                print(f"        shower E={s[0]:.6g} TeV at "
                      f"({s[1]:.2f}, {s[2]:.2f}, {s[3]:.2f})")
        for c in ev["channels"][:2]:
            for pu in c["pulses"][:2]:
                print(f"     channel {c['number']}: amplitude={pu[0]:.4g} "
                      f"time={pu[1]:.4f} ns magic={pu[2]:.6g}")


if __name__ == "__main__":
    main()
