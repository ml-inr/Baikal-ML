"""ROOT -> HDF5 converter for full-statistics experimental data.

Differs from root2h5_exp.py:
  * Chunked reading (bounded memory) — handles multi-million-event runs.
  * Captures physical event IDs from BJointHeader into `header_prty`
    (= [season, cluster, run, fEventIDWR, fTimeCC_sec, fTimeCC_nsec]),
    so catalog_v2 build_exp can use a real physical event_id (column 3).
  * Per-event geometry (the detector drifts within a run, ~0.4 m).
  * No multi-cluster split: these ROOT files are per-cluster, every event is
    single-cluster with LOCAL channels 0-287 (channel//36 = string 0-7).
  * Vectorised chunk processing (coords via fancy-index, time-centering via
    bincount, within-event time sort via lexsort) — scales to ~10^8 events.

One HDF5 part is written per ROOT file: `{particle}/<key>/part_<prefix>/data`,
where prefix = root filename stem (e.g. 'part_s2020_c01_r0027').

Usage (from project root, inside conda env baikal25):
    python data_manager/root2h5/root2h5_exp_full.py \\
        --config data_manager/root2h5/root2h5_config_exp_full.yaml \\
        [--files s2020_c01_r0206.root ...]   # restrict to specific files (testing)
        [--workers N]                        # override NUM_WORKERS
"""

import argparse
import logging
import os
import re
import traceback
from multiprocessing import Lock, Pool

import awkward as ak
import h5py as h5
import numpy as np
import uproot as ur
import yaml

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36       # channel // 36 = local string id (0-7)

_PART_RE = re.compile(r"s(\d+)_c(\d+)_r(\d+)")

# globals set per worker (Lock and config are not re-picklable per call)
_LOCK = None
_H5_PATH = None
_CFG = None


# ── config ─────────────────────────────────────────────────────────────────

def read_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def parse_prefix(prefix: str) -> tuple[int, int, int]:
    """'s2020_c01_r0027' -> (season=2020, cluster=1, run=27)."""
    m = _PART_RE.search(prefix)
    if not m:
        raise ValueError(f"Cannot parse run prefix: {prefix!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ── chunk processing (vectorised) ──────────────────────────────────────────

def process_chunk(
    amp_ak: ak.Array,
    time_ak: ak.Array,
    chan_ak: ak.Array,
    geom: np.ndarray,          # (3, n_ev, n_om) float32
    hdr: dict,                 # each (n_ev,) — wr, cc, sec, nsec, season, cluster, run
    cl_center: np.ndarray,     # (3,) float
    cfg: dict,
) -> dict | None:
    """Process one chunk of events into compact per-event hit arrays.

    Returns a dict of arrays for the surviving (non-empty) events, or None if
    nothing survives. Event order within the chunk is preserved.
    """
    g = cfg["general"]
    n_ev_in = len(chan_ak)
    if n_ev_in == 0:
        return None

    pn_in = ak.num(chan_ak).to_numpy().astype(np.int64)     # hits per event
    keep0 = pn_in > 0                                       # drop empty events
    if not keep0.any():
        return None

    # restrict everything to non-empty events
    idx_keep0 = np.nonzero(keep0)[0]
    chan_ak = chan_ak[keep0]
    amp_ak  = amp_ak[keep0]
    time_ak = time_ak[keep0]
    geom_k  = geom[:, idx_keep0, :]                          # (3, K0, n_om)
    pn      = pn_in[keep0]                                   # (K0,)
    K0      = len(pn)

    # flatten hits
    flat_ev   = np.repeat(np.arange(K0), pn)                 # (H,)
    flat_chan = ak.flatten(chan_ak).to_numpy().astype(np.int64)
    flat_amp  = ak.flatten(amp_ak).to_numpy().astype(np.float32)
    flat_time = ak.flatten(time_ak).to_numpy().astype(np.float32)

    # per-event geometry lookup: coords[:, h] = geom_k[:, event_of_h, channel_of_h]
    coords = geom_k[:, flat_ev, flat_chan].astype(np.float32)   # (3, H)
    x, y, z = coords[0], coords[1], coords[2]

    # sort hits within each event by time (stable, vectorised)
    order = np.lexsort((flat_time, flat_ev))
    flat_ev   = flat_ev[order]
    flat_chan = flat_chan[order]
    flat_amp  = flat_amp[order]
    flat_time = flat_time[order]
    x, y, z   = x[order], y[order], z[order]

    # exclude hits with large time residuals (known reconstruction bug)
    if g["exclude_big_ts"]:
        t_thr = float(g["t_threshold"])
        hmask = (np.abs(flat_time) <= t_thr) & ~np.isnan(flat_time)
        flat_ev   = flat_ev[hmask]
        flat_chan = flat_chan[hmask]
        flat_amp  = flat_amp[hmask]
        flat_time = flat_time[hmask]
        x, y, z   = x[hmask], y[hmask], z[hmask]

    if len(flat_ev) == 0:
        return None

    # renumber events that still have >=1 hit -> contiguous 0..K-1
    present, flat_ev2 = np.unique(flat_ev, return_inverse=True)
    K  = len(present)
    pn2 = np.bincount(flat_ev2, minlength=K).astype(np.int64)

    # center times per event (subtract per-event mean)
    if g["center_times"]:
        sums  = np.bincount(flat_ev2, weights=flat_time.astype(np.float64), minlength=K)
        means = (sums / pn2).astype(np.float32)
        flat_time = flat_time - means[flat_ev2]

    # shift coords to cluster center (constant offset per axis == shifting geom)
    if g["shift_coords_to_cl_center"]:
        x = x - cl_center[0]
        y = y - cl_center[1]
        z = z - cl_center[2]

    # unique strings per event
    str_ids = (flat_chan // STRING_DIVISOR).astype(np.int64)
    key     = flat_ev2.astype(np.int64) * 64 + str_ids
    ukey    = np.unique(key)
    num_un_strings = np.bincount((ukey // 64).astype(np.int64), minlength=K).astype(np.int32)

    # header rows for present events (map present -> original kept0 indices)
    orig_idx = idx_keep0[present]
    season = hdr["season"][orig_idx].astype(np.int64)
    cluster = hdr["cluster"][orig_idx].astype(np.int64)
    run = hdr["run"][orig_idx].astype(np.int64)
    sec = hdr["sec"][orig_idx].astype(np.int64)
    nsec = hdr["nsec"][orig_idx].astype(np.int64)
    # physical event_id = CC timestamp (sec*1e9 + nsec): unique + monotonic per run,
    # always present (unlike fEventIDWR which wraps at 2^15 or is zero for some runs).
    ev_key = sec * 1_000_000_000 + nsec
    header_prty = np.stack([season, cluster, run, ev_key, sec, nsec], axis=1)   # (K,6)

    ev_starts = np.concatenate([[0], np.cumsum(pn2)]).astype(np.int64)
    data = np.stack([flat_amp, flat_time, x, y, z], axis=1).astype(np.float32)   # (H,5)

    return {
        "data":           data,
        "channels":       flat_chan.astype(np.int32),
        "labels":         np.zeros(len(flat_chan), dtype=np.int32),
        "ev_starts":      ev_starts,
        "num_un_strings": num_un_strings,
        "cluster_ids":    cluster.astype(np.int32),
        "header_prty":    header_prty,
        "ev_key":         ev_key,
    }


# ── per-file conversion ────────────────────────────────────────────────────

def convert_file(rf_path: str, cfg: dict) -> dict | None:
    """Read one ROOT file in chunks, return accumulated part arrays (or None)."""
    rp = cfg["root_paths"]
    g = cfg["general"]
    particle = cfg["input"]["particle"]
    prefix = os.path.basename(rf_path).split(".")[0]
    chunk_size = int(g["chunk_size"])
    cap = g.get("events_per_file_cap")
    cap = int(cap) if cap else None

    try:
        f = ur.open(rf_path)
    except Exception as e:
        logger.error(f"{prefix}: cannot open ({e}); skipping")
        return None
    if not any("Events" in k for k in f.keys(recursive=False)):
        logger.error(f"{prefix}: no 'Events' tree; skipping")
        f.close()
        return None

    t = f["Events"]
    n_total = t[rp["pulse_n"]].num_entries
    if cap is not None:
        n_total = min(n_total, cap)

    hdr_paths = rp["header"]
    geo_path = rp["geometry"]

    acc_data, acc_chan, acc_lab = [], [], []
    acc_nstr, acc_clu, acc_hdr, acc_key = [], [], [], []
    ev_offsets = [0]          # running cumulative hit count for ev_starts stitching
    cl_center = None
    n_out = 0

    for a in range(0, n_total, chunk_size):
        b = min(a + chunk_size, n_total)
        chan_ak = t[rp["channel"]].array(entry_start=a, entry_stop=b)
        amp_ak  = t[rp["amplitude"]].array(entry_start=a, entry_stop=b)
        time_ak = t[rp["time"]].array(entry_start=a, entry_stop=b)
        geom = np.asarray(ak.unzip(t[geo_path].array(entry_start=a, entry_stop=b)),
                          dtype=np.float32)                     # (3, n, n_om)

        if cl_center is None:
            cl_center = geom.mean(axis=(1, 2))                  # (3,) from first chunk

        hdr = {
            "sec":     t[hdr_paths["time_sec"]].array(entry_start=a, entry_stop=b, library="np"),
            "nsec":    t[hdr_paths["time_nsec"]].array(entry_start=a, entry_stop=b, library="np"),
            "season":  t[hdr_paths["season"]].array(entry_start=a, entry_stop=b, library="np"),
            "cluster": t[hdr_paths["cluster"]].array(entry_start=a, entry_stop=b, library="np"),
            "run":     t[hdr_paths["run"]].array(entry_start=a, entry_stop=b, library="np"),
        }

        res = process_chunk(amp_ak, time_ak, chan_ak, geom, hdr, cl_center, cfg)
        if res is None:
            continue

        # stitch ev_starts across chunks: shift by running hit offset, drop leading 0
        starts = res["ev_starts"][1:] + ev_offsets[-1]
        ev_offsets.extend(starts.tolist())

        acc_data.append(res["data"])
        acc_chan.append(res["channels"])
        acc_lab.append(res["labels"])
        acc_nstr.append(res["num_un_strings"])
        acc_clu.append(res["cluster_ids"])
        acc_hdr.append(res["header_prty"])
        acc_key.append(res["ev_key"])
        n_out += len(res["num_un_strings"])

    f.close()

    if n_out == 0:
        logger.warning(f"{prefix}: no events survived; skipping")
        return None

    ev_key_all = np.concatenate(acc_key)
    n_dup = len(ev_key_all) - len(np.unique(ev_key_all))
    if n_dup:
        logger.warning(f"{prefix}: {n_dup} duplicate timestamp event_ids (non-unique!)")
    ev_ids = np.array(
        [f"{particle}_{prefix}_{int(k)}" for k in ev_key_all],
        dtype="bytes",
    )

    return {
        "prefix":         prefix,
        "raw/data":       np.concatenate(acc_data, axis=0),
        "raw/channels":   np.concatenate(acc_chan),
        "raw/labels":     np.concatenate(acc_lab),
        "raw/ev_starts":  np.asarray(ev_offsets, dtype=np.int64),
        "raw/num_un_strings": np.concatenate(acc_nstr),
        "raw/cluster_ids":    np.concatenate(acc_clu),
        "header_prty":    np.concatenate(acc_hdr, axis=0),
        "ev_ids":         ev_ids,
        "clusters_centers": np.asarray(cl_center, dtype=np.float32),
        "n_events":       n_out,
    }


# ── writing ────────────────────────────────────────────────────────────────

_GZIP_KEYS = {"raw/data", "raw/channels", "raw/labels", "header_prty"}
_DTYPES = {
    "raw/data": np.float32, "raw/channels": np.int32, "raw/labels": np.int32,
    "raw/ev_starts": np.int64, "raw/num_un_strings": np.int32,
    "raw/cluster_ids": np.int32, "header_prty": np.int64,
}


def write_part(h5_path: str, particle: str, part: dict) -> None:
    """Write one part to the shared HDF5 file (caller holds the write lock)."""
    prefix = part["prefix"]
    with h5.File(h5_path, "a") as hf:
        for key in ["raw/data", "raw/channels", "raw/labels", "raw/ev_starts",
                    "raw/num_un_strings", "raw/cluster_ids", "header_prty", "ev_ids"]:
            ds_path = f"{particle}/{key}/part_{prefix}/data"
            if ds_path in hf:
                del hf[ds_path]
            if key == "ev_ids":
                hf.create_dataset(ds_path, data=part["ev_ids"])
            else:
                hf.create_dataset(
                    ds_path, data=part[key].astype(_DTYPES[key]),
                    compression="gzip" if key in _GZIP_KEYS else None,
                )
        cc_path = f"{particle}/clusters_centers/part_{prefix}/data"
        if cc_path in hf:
            del hf[cc_path]
        hf.create_dataset(cc_path, data=part["clusters_centers"])


# ── worker ─────────────────────────────────────────────────────────────────

def _init_worker(lock, h5_path, cfg):
    global _LOCK, _H5_PATH, _CFG
    _LOCK, _H5_PATH, _CFG = lock, h5_path, cfg


def _worker(rf_path: str) -> tuple[str, int]:
    prefix = os.path.basename(rf_path).split(".")[0]
    try:
        part = convert_file(rf_path, _CFG)
        if part is None:
            return prefix, 0
        with _LOCK:
            write_part(_H5_PATH, _CFG["input"]["particle"], part)
        logger.info(f"{prefix}: written {part['n_events']:,} events")
        return prefix, part["n_events"]
    except Exception:
        logger.error(f"{prefix}: FAILED\n{traceback.format_exc()}")
        return prefix, -1


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--files", nargs="*", default=None,
                        help="restrict to these ROOT filenames (basename); default = all")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = read_config(args.config)
    particle = cfg["input"]["particle"]
    root_dir = cfg["input"]["root_dir_path"]
    h5_path = os.path.join(cfg["output"]["h5_prefix"], cfg["output"]["h5_name"])
    n_workers = args.workers or int(cfg["multiprocessing"]["NUM_WORKERS"])

    all_roots = sorted(f for f in os.listdir(root_dir) if f.endswith(".root"))
    if args.files:
        wanted = set(args.files)
        all_roots = [f for f in all_roots if f in wanted]

    # drop entire clusters flagged as bad (e.g. c01 time-calibration bug)
    exclude_clusters = set(cfg["general"].get("exclude_clusters") or [])
    if exclude_clusters:
        kept = []
        for f in all_roots:
            try:
                _, cl, _ = parse_prefix(f.split(".")[0])
            except ValueError:
                logger.warning(f"Cannot parse cluster from {f}; keeping it")
                kept.append(f)
                continue
            if cl in exclude_clusters:
                logger.info(f"Excluding {f} (cluster {cl} in exclude_clusters={sorted(exclude_clusters)})")
            else:
                kept.append(f)
        all_roots = kept

    root_paths = [os.path.join(root_dir, f) for f in all_roots]

    logger.info(f"Output : {h5_path}")
    logger.info(f"Files  : {len(root_paths)}  | workers: {n_workers}")

    if not os.path.exists(h5_path):
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5.File(h5_path, "w"):
            pass

    lock = Lock()
    results = []
    if n_workers <= 1:
        _init_worker(lock, h5_path, cfg)
        for rp in root_paths:
            results.append(_worker(rp))
    else:
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(lock, h5_path, cfg)) as pool:
            results = pool.map(_worker, root_paths)

    # write global flag once
    with h5.File(h5_path, "a") as hf:
        flag = f"{particle}/coords_are_cluster_centered/data"
        if flag in hf:
            del hf[flag]
        hf.create_dataset(flag, data=cfg["general"]["shift_coords_to_cl_center"])

    total = sum(n for _, n in results if n > 0)
    failed = [p for p, n in results if n < 0]
    logger.info(f"Done. {total:,} events written across "
                f"{sum(1 for _, n in results if n > 0)} parts.")
    if failed:
        logger.warning(f"FAILED parts: {failed}")


if __name__ == "__main__":
    main()
