"""Cut-flow diagnostic for the muon-energy dataset builder.

Runs the same pass-1 sig-noise metadata and target selection as ``build.py``, but writes
nothing except a JSON cut-flow report.  It answers which selection removes the most events
of a given class.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python -m data_manager.mu_energy_ds_builder.diagnose_cuts \
        --config data_manager/mu_energy_ds_builder/pilot_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from .build import (
    DEFAULT_PROBS_PATH,
    PARTICLE_ENCODE,
    PARTICLE_MAP,
    _load_parts,
    _read_targets,
)
from .io import read_part_metadata, read_part_metadata_from_probs

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--particle", default=None,
                    help="muatm | nuatm | nue2 (default: all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(args.config))
    parts = _load_parts(cfg["parts_file"])
    if args.particle:
        parts = {args.particle: parts[args.particle]}

    cuts = cfg.get("event_cuts", {})
    min_hits = int(cuts.get("min_hits", 8))
    min_strings = int(cuts.get("min_strings", 3))
    target_cfg = cfg.get("target", {})
    column = int(target_cfg.get("column", 1))
    valid_status = tuple(int(x) for x in target_cfg.get("valid_status", [0, 1, 4, 5]))
    born_inside = tuple(int(x) for x in target_cfg.get("born_inside_status", [1, 5]))
    exclude_nuatm = bool(target_cfg.get("exclude_nuatm_born_inside", True))

    source = str(cfg.get("sig_noise_source", "model")).lower()
    probs_h5_path = str(cfg.get("sig_noise_probs_path", DEFAULT_PROBS_PATH))
    threshold = float(cfg.get("sig_noise_threshold", 0.8))
    model_cfg = cfg.get("sig_noise_model", {})
    batch_size = int(model_cfg.get("batch_size", 256))
    probs_file = None
    if source == "model":
        from trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
            load_model,
        )
        model, _, device = load_model(device=model_cfg.get("device", "auto"))
        logger.info("Model on %s, threshold %.2f, h%d s%d", device, threshold,
                    min_hits, min_strings)
    else:
        model = None
        device = None
        probs_file = h5py.File(probs_h5_path, "r")
        logger.info("Probs file %s, threshold %.2f, h%d s%d", probs_h5_path,
                    threshold, min_hits, min_strings)

    report = {}
    with h5py.File(cfg["h5_path"], "r") as h5, \
            h5py.File(cfg["companion_path"], "r") as companion:
        for short, nums in parts.items():
            ptype = PARTICLE_MAP[short]
            qtype = PARTICLE_ENCODE[ptype]
            grp = h5[ptype]
            n_total = 0
            n_h8 = n_h8s = n_h8sv = n_final = 0
            n_single = n_valid = n_noexcl = 0
            h8_fail_hits = []
            h8_pass_hits = []
            h8_fail_e = []
            h8_pass_e = []
            status_counts = {}
            n_tracks_counts = {}
            t0 = time.time()

            for i, pn in enumerate(sorted(nums)):
                pk = f"part_{pn}"
                if source == "probs_h5":
                    meta = read_part_metadata_from_probs(
                        grp, probs_file[ptype], pk, threshold
                    )
                else:
                    meta = read_part_metadata(
                        grp, pk, model, batch_size, device, threshold
                    )
                if meta is None:
                    continue
                n_ev = len(meta["n_hits"])
                tar = _read_targets(companion, ptype, pk, n_ev, column)
                n_total += n_ev

                h8 = ((meta["n_sig_hits"] >= min_hits)
                      & (meta["n_sig_strings"] >= min_strings))
                single = tar["n_tracks"] == 1
                valid = (np.isin(tar["status"], valid_status)
                         & np.isfinite(tar["energy"]))
                excl = np.zeros(n_ev, dtype=bool)
                if exclude_nuatm and short == "nuatm":
                    excl = np.isin(tar["status"], born_inside)

                n_h8 += int(h8.sum())
                n_h8s += int((h8 & single).sum())
                n_h8sv += int((h8 & single & valid).sum())
                n_final += int((h8 & single & valid & ~excl).sum())
                n_single += int(single.sum())
                n_valid += int(valid.sum())
                n_noexcl += int((~excl).sum())

                h8_fail_hits.append(meta["n_sig_hits"][~h8])
                h8_pass_hits.append(meta["n_sig_hits"][h8])
                h8_fail_e.append(tar["energy"][~h8])
                h8_pass_e.append(tar["energy"][h8])
                for st, cnt in zip(*np.unique(tar["status"], return_counts=True)):
                    status_counts[int(st)] = status_counts.get(int(st), 0) + int(cnt)
                for nt, cnt in zip(*np.unique(tar["n_tracks"], return_counts=True)):
                    n_tracks_counts[int(nt)] = n_tracks_counts.get(int(nt), 0) + int(cnt)

                if (i + 1) % 10 == 0 or i == len(nums) - 1:
                    logger.info("  [%d/%d] %s (%.0fs)", i + 1, len(nums), pk,
                                time.time() - t0)

            def med(arrs):
                a = np.concatenate(arrs)
                a = a[np.isfinite(a)]
                return float(np.median(a)) if a.size else float("nan")

            report[short] = {
                "parts": len(nums),
                "total": n_total,
                "h8s3": n_h8,
                "h8s3_fraction": n_h8 / n_total if n_total else 0.0,
                "h8s3_and_single": n_h8s,
                "h8s3_single_valid": n_h8sv,
                "final": n_final,
                "fractions": {
                    "h8s3_of_total": n_h8 / n_total if n_total else 0.0,
                    "single_of_total": n_single / n_total if n_total else 0.0,
                    "valid_of_total": n_valid / n_total if n_total else 0.0,
                    "not_excluded_of_total": n_noexcl / n_total if n_total else 0.0,
                    "final_of_h8s3": n_final / n_h8 if n_h8 else 0.0,
                    "final_of_total": n_final / n_total if n_total else 0.0,
                },
                "losses_sequential": {
                    "h8s3": 1.0 - n_h8 / n_total if n_total else 0.0,
                    "single": 1.0 - n_h8s / n_h8 if n_h8 else 0.0,
                    "valid": 1.0 - n_h8sv / n_h8s if n_h8s else 0.0,
                    "exclusion": 1.0 - n_final / n_h8sv if n_h8sv else 0.0,
                },
                "status_counts": status_counts,
                "n_tracks_counts": n_tracks_counts,
                "h8s3_fail": {
                    "median_n_sig_hits": med(h8_fail_hits),
                    "median_energy_gev": med(h8_fail_e),
                },
                "h8s3_pass": {
                    "median_n_sig_hits": med(h8_pass_hits),
                    "median_energy_gev": med(h8_pass_e),
                },
            }
            logger.info("%s: %s -> final %s", short, f"{n_total:,}",
                        f"{n_final:,}")

    if probs_file is not None:
        probs_file.close()

    out = Path(cfg["output_dir"]) / "cutflow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
