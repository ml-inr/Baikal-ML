"""Stage 30 -- PROTOCOL test 1: is the detector response the same in data and MC?

The excess could be a flux deficit (a) or a response error (b).  Marginal
comparisons cannot separate them, because both factors enter.  What separates
them is a comparison **conditional on the track**: withhold one hit, fit the
Cherenkov track on the others, predict the withheld hit, and compare the
prediction error between data and MC.  Nothing about the withheld hit enters its
own prediction, so its residual measures ``P(hit | track)`` and nothing else.

If (a) holds, the residual distributions agree -- the population differs, the
response does not.  If (b) holds they diverge, and the divergence should grow
with event difficulty, because a response error bites the hard tail hardest.

Reading the withheld-hit residual:

``dt``
    observed minus predicted arrival time, in nanoseconds.  A longer tail in data
    means light arriving later than the track model allows, which is what extra
    scattering looks like.
``d``
    perpendicular distance from the fitted track to the withheld module.
``q``
    charge actually recorded there.  Together with ``d`` this probes the light
    yield without needing a yield model: at matched track and distance, the
    charge distribution must agree if the response does.
``anchor_rms``
    fit residual of the *other* hits.  Comparisons are made at matched anchor
    quality, so a looser fit in one sample cannot masquerade as a response
    difference.

Usage:
    nohup python stages/30_split_response.py > /tmp/stage30.log 2>&1 &
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402
import tracks                                                  # noqa: E402

LOG = logging.getLogger("stage30")
RDCC = {"rdcc_nbytes": 256 * 1024 * 1024, "rdcc_nslots": 100_003}
_CFG: h5io.Config | None = None


def select_events(cfg: h5io.Config) -> pd.DataFrame:
    """Events to process, per source and score band, sampled deterministically.

    Sampling is by hash residue rather than ``ORDER BY hash(...) LIMIT n``: the
    latter forces a full sort of a 94-million-row join for every band, which is
    what made the first version of this stage time out.  The residue filter is
    equally deterministic and reads the table once.  Accepted bands are taken
    whole -- there are only 7,752 MC and 3,220 experimental events in them.
    """
    model = cfg.path("preds") / cfg["model"]
    bands = cfg["stage30"]["bands"]
    per_band = int(cfg["stage30"]["sample_per_band"])
    min_hits = int(cfg["stage30"]["min_hits"])
    frames = []
    sources = (("mc", "mc_merged_thr0p8.duckdb", "s.data_class LIKE 'muatm%' AND "),
               ("exp", "exp_full_thr0p8.duckdb", ""))
    for source, db, extra in sources:
        with duckdb.connect(str(model / db), read_only=True) as con:
            con.execute("PRAGMA threads=8")
            for lo, hi in bands:
                started = time.time()
                # the modulus is taken from the band's own count.  Deriving it
                # from the total assumes events are spread evenly over the score,
                # and they are not -- that left the middle bands with 63 events.
                in_band = con.execute(f"""
                    SELECT count(*) FROM predictions p JOIN splits s USING (event_fk)
                    WHERE {extra} p.n_sn_hits >= {min_hits} AND p.n_sn_strings >= 3
                      AND p.score >= {lo} AND p.score < {hi}
                """).fetchone()[0]
                modulus = max(in_band // per_band, 1)
                frame = con.execute(f"""
                    SELECT p.event_fk, p.score, s.data_class, s.part_key,
                           s.local_idx, p.n_sn_hits, p.n_sn_strings
                    FROM predictions p JOIN splits s USING (event_fk)
                    WHERE {extra} p.n_sn_hits >= {min_hits}
                      AND p.n_sn_strings >= 3
                      AND p.score >= {lo} AND p.score < {hi}
                      AND hash(p.event_fk) % {modulus} = 0
                    LIMIT {per_band}
                """).df()
                frame["source"] = source
                frame["band_lo"], frame["band_hi"] = lo, hi
                frames.append(frame)
                LOG.info("%s band %.2f-%.2f: %d of %d events (modulus %d, %.0f s)",
                         source, lo, hi, len(frame), in_band, modulus,
                         time.time() - started)
    return pd.concat(frames, ignore_index=True)


def _init(config_dir: str) -> None:
    global _CFG
    _CFG = h5io.load_config(Path(config_dir))


def process_part(task: tuple) -> pd.DataFrame:
    """Leave-one-out residuals for every selected event of one part."""
    source, h5_group, part, frame = task
    cfg = _CFG
    threshold = float(cfg["sig_noise_threshold"])
    n_held = int(cfg["stage30"]["held_out_per_event"])
    min_hits = int(cfg["stage30"]["min_hits"])
    rows = []
    with h5py.File(cfg.path("h5", source), "r", **RDCC) as src, \
         h5py.File(cfg.path("probs", source), "r", **RDCC) as pf:
        starts = src[f"{h5_group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = src[f"{h5_group}/raw/data/{part}/data"][:].astype(np.float64)
        channels = src[f"{h5_group}/raw/channels/{part}/data"][:]
        probs = pf[f"{h5_group}/probs/{part}/data"][:].astype(np.float32)
    for record in frame.itertuples():
        lo, hi = starts[record.local_idx], starts[record.local_idx + 1]
        keep = probs[lo:hi] > threshold
        hits = data[lo:hi][keep]
        if len(hits) < min_hits:
            continue
        charge = np.clip(hits[:, 0], 0, tracks.Q_CLIP)
        order = np.argsort(hits[:, 1])          # time order, for even spacing
        hits, charge = hits[order], charge[order]
        chan = channels[lo:hi][keep][order]
        pos, times = hits[:, 2:5], hits[:, 1]
        step = max(len(hits) // n_held, 1)
        indices = np.arange(0, len(hits), step)[:n_held]
        out = tracks.leave_one_out(pos, times, charge, indices=indices)
        if len(out["dt"]) == 0:
            continue
        whole = tracks.fit_track(pos, times, charge)
        for k, index in enumerate(indices[:len(out["dt"])]):
            rows.append({
                "event_fk": record.event_fk, "source": source,
                "score": record.score, "band_lo": record.band_lo,
                "n_hits": len(hits),
                "n_strings": int(np.unique(chan // tracks.STRING_DIVISOR).size),
                "held_index": int(index),
                "dt": out["dt"][k], "d": out["d"][k], "q": out["q"][k],
                "anchor_rms": out["anchor_rms"][k],
                "event_fit_rms": whole["fit_rms"],
                "event_fit_zenith": whole["fit_zenith"],
                "event_fit_contrast": whole["fit_contrast"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data, config_path = HERE / "data", HERE / "config.yaml"

    selection = select_events(cfg)
    groups = {"exp": "exp_full"}
    selection["h5_group"] = [groups.get(s, c) for s, c
                             in zip(selection.source, selection.data_class)]
    tasks = [(source, group, part, sub)
             for (source, group, part), sub
             in selection.groupby(["source", "h5_group", "part_key"], sort=True)]
    tasks.sort(key=lambda t: -len(t[3]))
    if args.smoke:
        tasks = tasks[:8]
    workers = int(cfg["stage30"]["workers"])
    LOG.info("%d events over %d parts, %d workers",
             len(selection), len(tasks), workers)

    frames, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(HERE),)) as pool:
        futures = [pool.submit(process_part, task) for task in tasks]
        for future in as_completed(futures):
            frames.append(future.result())
            done += 1
            if done % 50 == 0 or done == len(tasks):
                rows = sum(len(f) for f in frames)
                LOG.info("%d/%d parts, %d residuals, %.0f s",
                         done, len(tasks), rows, time.time() - started)
    residuals = pd.concat(frames, ignore_index=True)
    LOG.info("total residuals: %d (%s)", len(residuals),
             residuals.groupby("source").size().to_dict())
    provenance.write(residuals, data / "30_loo_residuals.parquet",
                     stage="30_split_response", config_path=config_path,
                     inputs=[cfg.path("h5", "mc"), cfg.path("h5", "exp")],
                     started=started,
                     notes={"held_out_per_event": int(cfg["stage30"]["held_out_per_event"])})
    LOG.info("stage 30 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
