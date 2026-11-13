"""Быстрый тест отклика: время хитов МК размывается на sigma нс, скор — заново.

Классификатор пересчитывается на уже отобранных SN-хитах (prob > 0.8 из
замороженного файла), без повторного прогона sig-noise сети. Это изолирует
отклик классификатора: если 30 нс размытия поднимают долю хвоста МК до
наблюдаемого отношения exp/MC, картина «ошибка отклика» подтверждается и на
текущей модели. Повторный прогон SN-фильтра может изменить часть отбора и
даст другую (обычно меньшую) величину — это ограничение теста, см. старый
stage 40.

Джиттер применяется к колонке t отобранных хитов; для всех sigma
используется один и тот же набор нормальных приращений (парный дизайн).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # до импорта torch

import duckdb
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths  # noqa: E402
from inference.shared.model_utils import load_model, predict_scores  # noqa: E402
from data_manager.nu_classifier_ds_builder.io import (  # noqa: E402
    _count_sig_hits_strings,
)

SN_THRESHOLD = 0.8
MIN_HITS = 8
MIN_STRINGS = 3
MC_GROUP = "muatm_2020"
CHECKPOINT = (
    "trained_models/classifier/261024_1202_da_nu_classifier_exp_full_E1_"
    "weights_z128_k10_dlr1e-2_lam0p003_delay2_h8s3_muOnlyDA_s42/"
    "best_da_model.pth")
PRED_DIR = (
    "inference/nu_classifier/preds/261024_1202_da_nu_classifier_exp_full_E1_"
    "weights_z128_k10_dlr1e-2_lam0p003_delay2_h8s3_muOnlyDA_s42@best_da_model")


def _score(feats, model, norm, device, with_probs):
    if not feats:
        return np.empty(0, dtype=np.float32)
    return predict_scores(model, feats, norm, batch_size=512, device=device,
                          feats_with_probs=with_probs, with_tqdm=False)


def process_part(h5_path, probs_path, part, model, norm, device, sigmas,
                 rng, with_probs):
    with h5py.File(h5_path, "r") as f, h5py.File(probs_path, "r") as fp:
        g = f[MC_GROUP]
        starts = g[f"raw/ev_starts/{part}"]["data"][:].astype(np.int64)
        data = g[f"raw/data/{part}"]["data"][:]
        channels = g[f"raw/channels/{part}"]["data"][:]
        probs = fp[MC_GROUP]["probs"][part]["data"][:].astype(np.float32)
    n_events = len(starts) - 1
    n_hits = (starts[1:] - starts[:-1]).astype(np.int32)
    mask = probs > SN_THRESHOLD
    n_sn_h, n_sn_s = _count_sig_hits_strings(
        mask, channels, starts[:-1], n_hits, n_events)
    quality = (n_sn_h >= MIN_HITS) & (n_sn_s >= MIN_STRINGS)
    selected = np.flatnonzero(quality)

    by_sigma = {float(s): [] for s in sigmas}
    for index in selected:
        lo, hi = int(starts[index]), int(starts[index + 1])
        ev_mask = mask[lo:hi]
        if not ev_mask.any():
            continue
        feats = data[lo:hi][ev_mask].astype(np.float32)
        # Один и тот же z на хит для всех sigma: сравнение парное.
        z = rng.standard_normal(len(feats)).astype(np.float32)
        for sigma in by_sigma:
            f = feats.copy()
            if sigma > 0:
                f[:, 1] += sigma * z
            if with_probs:
                f = np.column_stack([f, probs[lo:hi][ev_mask]])
            by_sigma[sigma].append(f)

    out = {"part": part, "n_quality": int(len(selected))}
    for sigma, feats in by_sigma.items():
        scores = _score(feats, model, norm, device, with_probs)
        out[f"n_tail_{sigma}"] = int((scores > 0.5).sum())
        out[f"n_acc_{sigma}"] = int((scores >= SN_THRESHOLD).sum())
    return out


def _ratios(frame, sigmas):
    res = {}
    base = frame["n_tail_0.0"].sum()
    base_acc = frame["n_acc_0.0"].sum()
    for sigma in sigmas:
        if sigma == 0:
            continue
        res[f"tail_{sigma}"] = {
            "rate_0": float(base / frame["n_quality"].sum()),
            "rate_s": float(frame[f"n_tail_{sigma}"].sum()
                            / frame["n_quality"].sum()),
            "ratio": float(frame[f"n_tail_{sigma}"].sum() / base)
                     if base else float("nan"),
        }
        res[f"acc_{sigma}"] = {
            "rate_0": float(base_acc / frame["n_quality"].sum()),
            "rate_s": float(frame[f"n_acc_{sigma}"].sum()
                            / frame["n_quality"].sum()),
            "ratio": float(frame[f"n_acc_{sigma}"].sum() / base_acc)
                     if base_acc else float("nan"),
        }
    return res


def _bootstrap(frame, sigmas, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(frame)
    vals = {f"{kind}_{s}": [] for s in sigmas if s > 0
            for kind in ("tail", "acc")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = frame.iloc[idx]
        base_t, base_a = b["n_tail_0.0"].sum(), b["n_acc_0.0"].sum()
        for s in sigmas:
            if s == 0:
                continue
            vals[f"tail_{s}"].append(
                b[f"n_tail_{s}"].sum() / base_t if base_t else np.nan)
            vals[f"acc_{s}"].append(
                b[f"n_acc_{s}"].sum() / base_a if base_a else np.nan)
    return {k: [float(np.nanquantile(v, 0.16)),
                float(np.nanquantile(v, 0.84))] for k, v in vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--parts", type=int, default=400)
    ap.add_argument("--sigmas", default="30")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-spotcheck", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sigmas = [0.0] + [float(s) for s in args.sigmas.split(",") if s.strip()]
    h5_path = str(paths.h5("baikal_mc_merged"))
    probs_path = str(paths.probs("baikal_mc_merged"))
    model, norm, config = load_model(args.checkpoint, device=args.device)
    with_probs = config.get("model", {}).get("input_dim", 5) == 6
    print(f"model input_dim={config.get('model', {}).get('input_dim', 5)} "
          f"with_probs={with_probs}", flush=True)

    with h5py.File(h5_path, "r") as f:
        all_parts = sorted(f[MC_GROUP]["raw/data"].keys())
    with h5py.File(probs_path, "r") as fp:
        have_probs = set(fp[MC_GROUP]["probs"].keys())
    all_parts = [p for p in all_parts if p in have_probs]
    n_parts = 3 if args.smoke else min(args.parts, len(all_parts))
    idx = np.linspace(0, len(all_parts) - 1, n_parts).astype(int)
    parts = [all_parts[i] for i in idx]
    print(f"{len(parts)} parts from {len(all_parts)}", flush=True)

    rng = np.random.default_rng(42)
    rows = []
    t0 = time.time()
    for i, part in enumerate(parts, 1):
        rows.append(process_part(h5_path, probs_path, part, model, norm,
                                 args.device, sigmas, rng, with_probs))
        if i % 25 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)} parts, {time.time()-t0:.0f}s",
                  flush=True)
    frame = pd.DataFrame(rows)

    result = {
        "sigmas": sigmas,
        "n_parts": len(frame),
        "n_quality": int(frame["n_quality"].sum()),
        "per_part": frame.to_dict(orient="records"),
        "ratios": _ratios(frame, sigmas),
        "ratio_68ci": _bootstrap(frame, sigmas),
        "observed": {
            "tail_gt0p5_clean": 2.177,
            "note": "exp clean tail / exp clean total vs mc tail / mc total; "
                    ">=0.8 not bad-run-corrected here",
            "acc_ge0p8_db": 2.680,
        },
        "note": (
            "Классификатор пересчитан на SN-хитах, выбранных по замороженным "
            "вероятностям без джиттера. Повторный SN-фильтр не выполнялся: "
            "джиттер может перевести часть хитов через порог, поэтому это "
            "оценка отклика классификатора, а не полный ответ пайплайна."),
    }
    out = Path(args.out) if args.out else (
        paths.DATA / "time_jitter_classifier_h8s3.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("n_quality", "ratios", "ratio_68ci")},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
