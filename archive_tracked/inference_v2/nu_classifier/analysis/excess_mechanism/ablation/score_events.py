"""Пересчёт скора nu-классификатора для контролируемо изменённых событий.

Идея: избыток не объясняется ни одним из 52 агрегатов (суррогат даёт 1.35 против 2.86 у
сети). Значит сеть отвечает на что-то похитовое. Единственный способ узнать, на что именно —
менять вход по одному свойству и смотреть, какая правка убирает избыток.

Первый шаг — воспроизвести сохранённые скоры. Без этого остальное бессмысленно.
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py, numpy as np, duckdb

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores   # noqa: E402

DEV = "cuda:0"
CKPT = ROOT / "experiments/numu/260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/best_da_model.pth"
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = ROOT / "inference_v2/nu_classifier/preds" / MODEL
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
H5 = {"mc": ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5",
      "exp": ROOT / "data_manager/data/h5datasets/exp_full.h5"}
PROBS = {"mc": ROOT / "data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5",
         "exp": ROOT / "data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"}
SN_THR = 0.8


def fetch(source: str, h5_group: str, where: str, limit: int, seed_col="p.event_fk"):
    """(event_fk, score, [hits]) для выборки событий, детерминированно."""
    dbf = "mc_merged_thr0p8.duckdb" if source == "mc" else "exp_full_thr0p8.duckdb"
    con = duckdb.connect(str(PREDS / dbf), read_only=True)
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    rows = con.execute(f"""
        SELECT l.part_key, l.local_idx, p.event_fk, p.score
        FROM predictions p JOIN splits s USING(event_fk)
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk
        WHERE {where}
        ORDER BY hash({seed_col}) LIMIT {limit}""").fetchall()
    con.close()
    by_part: dict[str, list] = {}
    for pk, idx, fk, sc in rows:
        by_part.setdefault(pk, []).append((idx, fk, sc))

    hits, fks, scores = [], [], []
    with h5py.File(H5[source], "r") as src, h5py.File(PROBS[source], "r") as pf:
        for pk, items in by_part.items():
            ev = src[f"{h5_group}/raw/ev_starts/{pk}/data"][:].astype(np.int64)
            data = src[f"{h5_group}/raw/data/{pk}/data"][:].astype(np.float32)
            prob = pf[f"{h5_group}/probs/{pk}/data"][:].astype(np.float32)
            for idx, fk, sc in items:
                s, e = int(ev[idx]), int(ev[idx + 1])
                m = prob[s:e] > SN_THR
                if m.sum() < 5:
                    continue
                hits.append(data[s:e][m]); fks.append(fk); scores.append(sc)
    return hits, np.array(fks), np.array(scores)


def score(hits, model, norm, batch_size=512):
    return predict_scores(model, hits, norm, batch_size=batch_size, device=DEV,
                          with_tqdm=False)


if __name__ == "__main__":
    model, norm, cfg = load_model(str(CKPT), device=DEV)
    print(f"модель загружена на {DEV}; входных признаков: "
          f"{cfg.get('model', {}).get('input_dim', '?')}")
    Q = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3"
    hits, fks, stored = fetch("mc", "muatm_2020",
                              f"{Q} AND NOT s.used_for_labels AND s.data_class='muatm_2020'",
                              2000)
    got = score(hits, model, norm)
    d = np.abs(got - stored)
    print(f"\nMC muatm, {len(hits)} событий:")
    print(f"  максимальное расхождение со сохранённым скором: {d.max():.2e}")
    print(f"  медианное: {np.median(d):.2e}")
    print("  ВОСПРОИЗВЕДЕНО" if d.max() < 1e-4 else "  НЕ ВОСПРОИЗВЕДЕНО")
    # зависимость от размера батча — та же проверка, что провалила sig-noise
    for bs in (64, 256, 512):
        s2 = score(hits[:500], model, norm, batch_size=bs)
        print(f"  batch={bs:>4}: max|delta| к batch=512 "
              f"{np.abs(s2 - score(hits[:500], model, norm, 512)).max():.2e}")
