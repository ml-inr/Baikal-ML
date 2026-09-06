"""Stage 00 -- verify every input before any analysis is built on it.

This is not ceremony.  In this session a sig-noise run over ``baikal_mc_reco.h5``
wrote a 6 kB output and exited zero, because the file's groups are named
``muatm`` / ``nuatm_conv`` / ``nuatm_prompt`` / ``nue2`` while the script carried
a hardcoded list of the ``mc_merged`` names.  An empty result reported as success
is worse than a crash: it is discovered several steps later, once something has
been built on it.

What is checked, per source:

**Coverage** -- every part of the source h5 has a counterpart in the probs file.

**Byte-level agreement** -- for a sample of parts, ``ev_starts`` and ``channels``
in the probs file are compared element by element against the source.  The probs
builder documents that hit order is preserved; that claim is tested rather than
trusted, because every downstream index depends on it.

**Value sanity** -- probabilities finite and inside [0, 1].

**Internal consistency** -- the stored ``n_sig_hits_0.8`` and
``n_sig_strings_0.8`` are recomputed from the probabilities and the channel ids
and must agree exactly.  This catches a threshold or a string divisor applied
differently in the writer than in the reader.

**Provenance** -- the batch-size attribute must read 256.  The sig-noise mask
makes batch size part of the hit selection, so a file written at another value is
not interchangeable.

Usage:
    python stages/00_verify_inputs.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from zlib import crc32

import h5py
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import provenance                                              # noqa: E402

LOG = logging.getLogger("verify")
STRING_DIVISOR = 36            # channel // 36 -> string, as the pipeline defines it


def load_config() -> tuple[dict, Path]:
    raw = yaml.safe_load((HERE / "config.yaml").read_text())
    return raw, (HERE / raw["paths"]["root"]).resolve()


def check_source(name: str, source: Path, probs: Path, groups: list[str],
                 threshold: float, n_parts: int, expected_batch: int) -> pd.DataFrame:
    """All checks for one (source, probs) pair, one row per group."""
    rows = []
    with h5py.File(source, "r") as src, h5py.File(probs, "r") as pf:
        batch = pf.attrs.get("sig_noise_batch_size")
        for group in groups:
            problems: list[str] = []
            if group not in src:
                rows.append({"source": name, "group": group, "status": "MISSING IN SOURCE",
                             "problems": "group absent"})
                continue
            if group not in pf:
                rows.append({"source": name, "group": group, "status": "MISSING IN PROBS",
                             "problems": "group absent from probs file"})
                continue
            src_parts = sorted(src[f"{group}/raw/data"].keys())
            probs_parts = sorted(pf[f"{group}/probs"].keys())
            missing = set(src_parts) - set(probs_parts)
            extra = set(probs_parts) - set(src_parts)
            if missing:
                problems.append(f"{len(missing)} parts missing from probs")
            if extra:
                problems.append(f"{len(extra)} parts not in source")
            if batch is None or int(batch) != expected_batch:
                problems.append(f"batch attribute is {batch}, expected {expected_batch}")

            # byte-level agreement and value sanity on a deterministic sample
            order = np.argsort([crc32(p.encode()) for p in src_parts])
            sample = [src_parts[i] for i in order[:n_parts] if src_parts[i] in probs_parts]
            n_hits_total = n_events_total = 0
            bad_order = bad_channels = bad_values = bad_counts = 0
            for part in sample:
                s_starts = src[f"{group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
                p_starts = pf[f"{group}/ev_starts/{part}/data"][:].astype(np.int64)
                s_chan = src[f"{group}/raw/channels/{part}/data"][:]
                p_chan = pf[f"{group}/channels/{part}/data"][:]
                probs_arr = pf[f"{group}/probs/{part}/data"][:]
                n_hits_total += len(probs_arr)
                n_events_total += len(s_starts) - 1
                if len(s_starts) != len(p_starts) or not np.array_equal(s_starts, p_starts):
                    bad_order += 1
                if len(s_chan) != len(p_chan) or not np.array_equal(s_chan, p_chan):
                    bad_channels += 1
                if len(probs_arr) != len(s_chan):
                    bad_values += 1
                elif not np.all(np.isfinite(probs_arr)) or probs_arr.min() < 0 \
                        or probs_arr.max() > 1:
                    bad_values += 1
                # recompute the stored per-event counts
                key = f"{group}/n_sig_hits_{threshold}/{part}/data"
                if key in pf:
                    stored_hits = pf[key][:]
                    stored_str = pf[f"{group}/n_sig_strings_{threshold}/{part}/data"][:]
                    mask = probs_arr > threshold
                    recomputed_hits = np.add.reduceat(
                        mask.astype(np.int64), s_starts[:-1]) if len(s_starts) > 1 else []
                    strings = np.array([
                        np.unique(p_chan[s_starts[i]:s_starts[i + 1]][
                            mask[s_starts[i]:s_starts[i + 1]]] // STRING_DIVISOR).size
                        for i in range(len(s_starts) - 1)])
                    if not np.array_equal(np.asarray(recomputed_hits), stored_hits):
                        bad_counts += 1
                    elif not np.array_equal(strings, stored_str):
                        bad_counts += 1
            for label, count in (("ev_starts differ", bad_order),
                                 ("channels differ", bad_channels),
                                 ("bad probability values", bad_values),
                                 ("stored counts wrong", bad_counts)):
                if count:
                    problems.append(f"{label} in {count}/{len(sample)} parts")
            rows.append({
                "source": name, "group": group,
                "source_parts": len(src_parts), "probs_parts": len(probs_parts),
                "parts_checked": len(sample), "hits_checked": n_hits_total,
                "events_checked": n_events_total,
                "batch_attr": int(batch) if batch is not None else -1,
                "status": "ok" if not problems else "FAILED",
                "problems": "; ".join(problems)})
            LOG.info("%s/%s: %s%s", name, group, rows[-1]["status"],
                     "" if not problems else "  -- " + rows[-1]["problems"])
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg, root = load_config()
    threshold = float(cfg["sig_noise_threshold"])
    n_parts = int(cfg["stage00"]["parts_checked"])
    expected_batch = int(cfg["sig_noise_batch_size"])

    frames = [
        check_source("mc_reco", root / cfg["paths"]["h5"]["mc_reco"],
                     root / cfg["paths"]["probs"]["mc_reco"], cfg["mc_reco_groups"],
                     threshold, n_parts, expected_batch),
        check_source("exp_reco", root / cfg["paths"]["h5"]["exp_reco"],
                     root / cfg["paths"]["probs"]["exp_reco"], [cfg["exp_reco_group"]],
                     threshold, n_parts, expected_batch)]
    report = pd.concat(frames, ignore_index=True)
    failed = report[report.status != "ok"]
    provenance.write(report, HERE / "data" / "00_input_verification.parquet",
                     stage="00_verify_inputs", config_path=HERE / "config.yaml",
                     inputs=[], started=started,
                     notes={"failures": int(len(failed))})
    if len(failed):
        LOG.error("%d checks FAILED:\n%s", len(failed),
                  failed[["source", "group", "problems"]].to_string(index=False))
        raise SystemExit(1)
    LOG.info("all inputs verified in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
