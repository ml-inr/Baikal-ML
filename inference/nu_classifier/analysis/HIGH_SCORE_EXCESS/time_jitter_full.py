"""Полный тест отклика: джиттер времени -> SN-фильтр заново -> классификатор.

Адаптация архивного `response_vs_flux/stages/40_time_jitter.py` под текущий
h8s3 μ-only чекпоинт. В отличие от `time_jitter_classifier.py`, здесь заново
прогоняется sig-noise сеть на размытых временах: 30 нс меняют ~1.5% хитов
около порога и на ~20% — число качественных событий, поэтому фиксировать
старый отбор нельзя.

Считаем условную долю хвоста среди качественных событий (тех же определений,
что в prediction DB: prob > 0.8, n_sn_hits >= 8, n_sn_strings >= 3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # до импорта torch

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths  # noqa: E402
from inference.shared.model_utils import (  # noqa: E402
    load_model, load_sn_model, predict_scores,
)
from data_manager.nu_classifier_ds_builder.io import (  # noqa: E402
    _count_sig_hits_strings,
)
import time_jitter_classifier as tj  # noqa: E402

from trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (  # noqa: E402,E501
    predict_flat,
)


def process_part(h5_path, part, model, norm, sn_model, sn_dev, device,
                 sigmas, rng, with_probs, baseline=None):
    with h5py.File(h5_path, "r") as f:
        g = f[tj.MC_GROUP]
        starts = g[f"raw/ev_starts/{part}"]["data"][:].astype(np.int64)
        data = g[f"raw/data/{part}"]["data"][:].astype(np.float32)
        channels = g[f"raw/channels/{part}"]["data"][:]
    n_events = len(starts) - 1
    n_hits = (starts[1:] - starts[:-1]).astype(np.int32)
    z = rng.standard_normal(len(data)).astype(np.float32)

    out = {"part": part, "n_triggered": int(n_events)}
    for sigma in sigmas:
        if sigma == 0 and baseline is not None:
            for key in ("n_quality_0.0", "n_tail_0.0", "n_acc_0.0"):
                out[key] = baseline[key]
            continue
        perturbed = data.copy()
        if sigma > 0:
            perturbed[:, 1] += sigma * z
        probs = predict_flat(sn_model, perturbed, starts,
                             batch_size=256, device=sn_dev, normalize=True)
        mask = probs > tj.SN_THRESHOLD
        n_sn_h, n_sn_s = _count_sig_hits_strings(
            mask, channels, starts[:-1], n_hits, n_events)
        quality = (n_sn_h >= tj.MIN_HITS) & (n_sn_s >= tj.MIN_STRINGS)
        selected = np.flatnonzero(quality)
        feats = []
        for index in selected:
            lo, hi = int(starts[index]), int(starts[index + 1])
            ev_mask = mask[lo:hi]
            if not ev_mask.any():
                continue
            event = perturbed[lo:hi][ev_mask]
            if with_probs:
                event = np.column_stack([event, probs[lo:hi][ev_mask]])
            feats.append(event)
        if feats:
            scores = predict_scores(model, feats, norm, batch_size=512,
                                    device=device,
                                    feats_with_probs=with_probs,
                                    with_tqdm=False)
        else:
            scores = np.empty(0, dtype=np.float32)
        out[f"n_quality_{sigma}"] = int(len(scores))
        out[f"n_tail_{sigma}"] = int((scores > 0.5).sum())
        out[f"n_acc_{sigma}"] = int((scores >= tj.SN_THRESHOLD).sum())
    return out


def _summary(frame, sigmas):
    res = {}
    base = frame["n_tail_0.0"].sum()
    base_acc = frame["n_acc_0.0"].sum()
    base_q = frame["n_quality_0.0"].sum()
    for sigma in sigmas:
        if sigma == 0:
            continue
        q = frame[f"n_quality_{sigma}"].sum()
        t = frame[f"n_tail_{sigma}"].sum()
        a = frame[f"n_acc_{sigma}"].sum()
        res[str(sigma)] = {
            "quality_0": int(base_q), "quality_s": int(q),
            "quality_ratio": float(q / base_q),
            "tail_rate_0": float(base / base_q),
            "tail_rate_s": float(t / q),
            "tail_ratio": (float((t / q) / (base / base_q))
                           if base and q else float("nan")),
            "acc_rate_0": float(base_acc / base_q),
            "acc_rate_s": float(a / q),
            "acc_ratio": (float((a / q) / (base_acc / base_q))
                          if base_acc and q else float("nan")),
        }
    return res


def _bootstrap(frame, sigmas, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    vals = {f"{k}_{s}": [] for s in sigmas if s > 0
            for k in ("tail", "acc", "quality")}
    for _ in range(n_boot):
        b = frame.iloc[rng.integers(0, len(frame), len(frame))]
        q0, t0, a0 = (b["n_quality_0.0"].sum(), b["n_tail_0.0"].sum(),
                      b["n_acc_0.0"].sum())
        for s in sigmas:
            if s == 0:
                continue
            q = b[f"n_quality_{s}"].sum()
            vals[f"tail_{s}"].append(
                (b[f"n_tail_{s}"].sum() / q) / (t0 / q0)
                if q and t0 else np.nan)
            vals[f"acc_{s}"].append(
                (b[f"n_acc_{s}"].sum() / q) / (a0 / q0)
                if q and a0 else np.nan)
            vals[f"quality_{s}"].append(q / q0)
    return {k: [float(np.quantile(v, 0.16)),
                float(np.quantile(v, 0.84))] for k, v in vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=tj.CHECKPOINT)
    ap.add_argument("--parts", type=int, default=200)
    ap.add_argument("--sigmas", default="30")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--sn-device", default="cuda:1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse-zero", default=None,
                    help="JSON from a previous run: reuse its sigma=0 counts "
                         "and its part list, run only the new sigmas")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sigmas = [0.0] + [float(s) for s in args.sigmas.split(",") if s.strip()]
    h5_path = str(paths.h5("baikal_mc_merged"))
    probs_path = str(paths.probs("baikal_mc_merged"))
    model, norm, config = load_model(args.checkpoint, device=args.device)
    with_probs = config.get("model", {}).get("input_dim", 5) == 6
    sn_model, _, sn_dev = load_sn_model(device=args.sn_device)
    print(f"classifier on {args.device}, SN on {sn_dev}, "
          f"with_probs={with_probs}", flush=True)

    baseline = None
    if args.reuse_zero:
        prev = json.loads(Path(args.reuse_zero).read_text())
        baseline = {r["part"]: r for r in prev["per_part"]}
        parts = [r["part"] for r in prev["per_part"]]
        print(f"reusing sigma=0 from {args.reuse_zero}: {len(parts)} parts",
              flush=True)
    else:
        with h5py.File(h5_path, "r") as f:
            all_parts = sorted(f[tj.MC_GROUP]["raw/data"].keys())
        with h5py.File(probs_path, "r") as fp:
            have = set(fp[tj.MC_GROUP]["probs"].keys())
        all_parts = [p for p in all_parts if p in have]
        n_parts = 2 if args.smoke else min(args.parts, len(all_parts))
        idx = np.linspace(0, len(all_parts) - 1, n_parts).astype(int)
        parts = [all_parts[i] for i in idx]

    rng = np.random.default_rng(42)
    rows, t0 = [], time.time()
    for i, part in enumerate(parts, 1):
        rows.append(process_part(h5_path, part, model, norm, sn_model,
                                 sn_dev, args.device, sigmas, rng,
                                 with_probs, baseline.get(part)
                                 if baseline else None))
        if i % 5 == 0 or i == len(parts):
            elapsed = time.time() - t0
            print(f"  {i}/{len(parts)} parts, {elapsed:.0f}s "
                  f"(eta {elapsed/i*(len(parts)-i):.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    result = {
        "sigmas": sigmas, "n_parts": len(frame),
        "summary": _summary(frame, sigmas),
        "ratio_68ci": _bootstrap(frame, sigmas),
        "per_part": frame.to_dict(orient="records"),
        "observed": {"tail_gt0p5_clean": 2.177,
                     "acc_ge0p8_db": 2.680},
    }
    out = Path(args.out) if args.out else (
        paths.DATA / "time_jitter_full_h8s3.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"summary": result["summary"],
                      "ci": result["ratio_68ci"]}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
