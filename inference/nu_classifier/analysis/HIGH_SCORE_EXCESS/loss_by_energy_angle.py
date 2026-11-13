#!/usr/bin/env python
"""Разложение Δлосса (FIXED → weights) по ln E и углам θ, φ — по всем классам МС.

События — общие для mc_merged-баз обоих чекпоинтов (пересечение event_fk).
На событие берутся:

  * energy — та же величина, что флэттенили веса: `row_energy` компаньона
    (сумма положительных `e_ref`, иначе лучший `e_at_entry` по статусу),
    посчитанная векторизованно по парту. `SumEnergyBundle` из
    `muons_prty/aggregate` сюда НЕ годится: это другая величина (медианное
    отношение ~2.3, квартили 1.4–8.2, лог-корреляция 0.90), хотя первый
    столбец `muons_prty/individ[:, 6]` и совпадает с `e_ref` компаньона;
  * theta, phi — `prime_prty` (из него же собран `theta.npy` датасета — на
    первом парте класса печатается сверка);
  * для карты «куда ушла масса весов» — сам `weights.npy` датасета на тех же
    осях (`energy_cache.npy` и `theta.npy`): оси совпадают с Δ-картой точно,
    а не приблизительно.

Лосс — focal(gamma=2) с pos_weight тренера, без per-event весов; Δ — разность
средних по событиям бина (парная), доля бина в общем Δ.

Запуск: python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/loss_by_energy_angle.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import multiprocessing
import sys
import time
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").exists())
if str(ROOT) not in sys.path:                    # row_energy живёт в data_manager
    sys.path.insert(0, str(ROOT))
PREDS = ROOT / "inference/nu_classifier/preds"
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
MC_H5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
COMPANION = ROOT / "data_manager/data/h5datasets/baikal_mc_merged_energy_truth.h5"
DATASET = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"

RUNS = {
    "fix": "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model",
    "wgt": "261002_0804_da_nu_classifier_exp_full_E1_lambda0.01_DArepair_weights@best_da_model",
}
BOUND_NUATM, BOUND_NUE2 = 576_947_380, 643_643_317
POS_WEIGHT = 2_249_911 / 2_250_089
GAMMA = 2.0

CLASSES = [("muatm_2020", 0), ("nuatm_2020", 1), ("nue2_2020", 2)]
COS_THETA_EDGES = np.linspace(-1.0, 1.0, 11)
PHI_EDGES = np.linspace(0.0, 360.0, 9)


def class_code(fk: np.ndarray) -> np.ndarray:
    return ((fk > BOUND_NUATM).astype(np.int8) + (fk > BOUND_NUE2).astype(np.int8))


def focal_loss(fk: np.ndarray, score: np.ndarray) -> np.ndarray:
    """focal(gamma) с pos_weight тренера, без per-event весов."""
    positive = fk > BOUND_NUATM
    p = score.astype(np.float64)
    p_t = np.where(positive, p, 1.0 - p)
    loss = (1.0 - p_t) ** GAMMA * (-np.log(p_t))
    return np.where(positive, POS_WEIGHT, 1.0) * loss


def read_predictions(db: Path) -> dict[str, np.ndarray]:
    connection = duckdb.connect(str(db), read_only=True)
    columns = connection.execute(
        "SELECT event_fk, score FROM predictions").fetchnumpy()
    connection.close()
    return columns


def locate(fks: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(part_id, local_idx) для event_fk через catalog.h5_locations.

    Имена партов уникальны только внутри источника, поэтому список партов
    собирается именно по файлу mc_merged; в выборку отдаются целые id, а не
    строки — 89 млн строк со строками в pandas не помещаются по-человечески.
    """
    connection = duckdb.connect(str(CATALOG), read_only=True)
    paths = connection.execute("SELECT DISTINCT h5_path FROM h5_locations").df()
    matches = [p for p in paths["h5_path"] if str(p).endswith("baikal_mc_merged.h5")]
    assert len(matches) == 1, f"путей mc_merged в h5_locations: {matches}"
    parts = connection.execute(
        "SELECT DISTINCT part_key FROM h5_locations WHERE h5_path = ? ORDER BY 1",
        [matches[0]]).df()
    parts["part_id"] = np.arange(len(parts), dtype=np.int64)
    connection.register("_p", parts)
    connection.register("_q", pd.DataFrame({"query_idx": np.arange(len(fks)),
                                            "event_fk": fks}))
    frame = connection.execute("""
        SELECT q.query_idx, m.part_id, l.local_idx
        FROM _q q
        JOIN h5_locations l ON l.event_fk = q.event_fk
        JOIN _p m ON m.part_key = l.part_key
        ORDER BY q.query_idx
    """).df()
    connection.close()
    assert len(frame) == len(fks), f"нашлось {len(frame)} из {len(fks)}"
    return (frame["part_id"].to_numpy(), frame["local_idx"].to_numpy(),
            list(parts["part_key"]))


def part_rows(part_local: np.ndarray, parts: list[str]) -> list[np.ndarray]:
    """Индексы строк выборки для каждого парта (парты идут вплотную)."""
    order = np.argsort(part_local, kind="stable")
    boundaries = np.searchsorted(part_local[order], np.arange(len(parts) + 1))
    return [order[boundaries[i]:boundaries[i + 1]] for i in range(len(parts))]


def _physics_worker(payload):
    """theta/phi/energy для списка партов — в отдельном процессе.

    Файлы открываются внутри воркера: h5py-дескрипторы не переживают fork.
    Возвращаются и строки выборки, к которым относятся значения, — родитель
    раскладывает по ним, поэтому порядок чанков не важен.
    """
    gname, items = payload
    rows_out, theta_out, phi_out, energy_out, fallback = [], [], [], [], 0
    with h5py.File(MC_H5, "r") as store, h5py.File(COMPANION, "r") as companion:
        for part, local_ids, rows in items:
            prime = store[gname]["prime_prty"][part]["data"][local_ids]
            sums, need = part_energy(companion, gname, part)
            rows_out.append(rows)
            theta_out.append(prime[:, 0])
            phi_out.append(prime[:, 1])
            energy_out.append(sums[local_ids].astype(np.float32))
            fallback += int(need[local_ids].sum())
    return (np.concatenate(rows_out), np.concatenate(theta_out),
            np.concatenate(phi_out), np.concatenate(energy_out), fallback)


def read_class_physics(name: str, parts: np.ndarray, part_keys: list[str],
                       part_local: np.ndarray, local: np.ndarray, size: int,
                       workers: int = 12) -> tuple[dict[str, np.ndarray], int]:
    """Физика всех событий класса; парты читаются пулом процессов.

    Чанки набираются по числу событий, а не партов: парты отличаются по размеру
    на порядки, и по-партовое деление оставило бы половину воркеров без работы.
    """
    rows_of_part = part_rows(part_local, [part_keys[p] for p in parts])
    counts = np.array([len(r) for r in rows_of_part])
    running = np.zeros(workers, dtype=np.int64)
    buckets: list[list[int]] = [[] for _ in range(workers)]
    for slot in np.argsort(-counts):
        worker = int(np.argmin(running))
        buckets[worker].append(int(slot))
        running[worker] += counts[slot]
    payloads = [(name, [(part_keys[parts[s]], local[rows_of_part[s]],
                         rows_of_part[s]) for s in bucket])
                for bucket in buckets if bucket]
    with h5py.File(COMPANION, "r") as companion:
        check_energy(companion, name, part_keys[parts[0]], local[rows_of_part[0]])
    physics = {key: np.full(size, np.nan, dtype=np.float32)
               for key in ("theta", "phi", "energy")}
    fallback = 0
    with multiprocessing.get_context("fork").Pool(len(payloads)) as pool:
        for number, (rows, theta, phi, energy, n_fallback) in enumerate(
                pool.imap_unordered(_physics_worker, payloads), 1):
            physics["theta"][rows] = theta
            physics["phi"][rows] = phi
            physics["energy"][rows] = energy
            fallback += n_fallback
            print(f"    {name}: чанк {number}/{len(payloads)}", flush=True)
    return physics, fallback


def part_energy(companion: h5py.File, gname: str, part: str
                ) -> tuple[np.ndarray, np.ndarray]:
    """Энергия каждого события парта — векторизованный `row_energy` (см. add_weights).

    Возвращает (энергии, маска событий, ушедших в запасную ветку) для контроля,
    как часто она вообще работает.
    """
    group = companion[gname]
    mu_starts = group["mu_starts"][part]["data"][:].astype(np.int64)
    n_events = len(mu_starts) - 1
    e_ref = group["e_ref"][part]["data"][:]
    event_of_muon = np.repeat(np.arange(n_events), np.diff(mu_starts))
    alive = e_ref > 0.0
    sums = np.bincount(event_of_muon[alive],
                       weights=e_ref[alive].astype(np.float64),
                       minlength=n_events)
    need = sums == 0.0
    if need.any():
        entry = group["e_at_entry"][part]["data"][:, 0].astype(np.float64)
        status = group["e_at_entry_status"][part]["data"][:, 0]
        usable = np.isin(status, (0, 1, 4, 5)) & np.isfinite(entry)
        best = np.full(n_events, -np.inf)
        np.maximum.at(best, event_of_muon[usable], entry[usable])
        sums = np.where(need, np.where(np.isfinite(best), best, np.nan), sums)
    return sums, need


def check_energy(companion: h5py.File, gname: str, part: str,
                 local_idx: np.ndarray) -> None:
    """`part_energy` против `row_energy` — на одном парте, до прохода."""
    from data_manager.nu_classifier_ds_builder.add_weights import row_energy

    sample = np.sort(local_idx)[:5000]
    fast = part_energy(companion, gname, part)[0][sample]
    slow = row_energy(companion, gname, part, sample)
    both = ~np.isnan(fast) & ~np.isnan(slow)
    print(f"    {gname} {part}: part_energy vs row_energy — max|Δ| "
          f"{np.max(np.abs(fast[both] - slow[both])):.3e}, расхождение NaN-статуса "
          f"{int((np.isnan(fast) != np.isnan(slow)).sum())} (N={len(sample):,})",
          flush=True)


def bin_table(title: str, values: np.ndarray, loss_fix: np.ndarray,
              loss_wgt: np.ndarray, edges: np.ndarray) -> None:
    """Лосс по бинам `values` с рёбрами `edges`; доля Δ на бин."""
    delta = loss_wgt - loss_fix
    total = delta.mean()
    index = np.clip(np.searchsorted(edges, values, side="right") - 1,
                    0, len(edges) - 2)
    print(f"\n-- {title} (Δ(среднее) = {total:+.6f})")
    print(f"{'бин':>15s} {'N':>11s} {'loss_fix':>9s} {'loss_wgt':>9s} "
          f"{'Δ':>10s} {'±SE':>7s} {'Δотн.%':>8s} {'доляΔ%':>7s}")
    for i in range(len(edges) - 1):
        selected = index == i
        n = int(selected.sum())
        if n == 0:
            continue
        d = delta[selected]
        name = f"{edges[i]:.2f}..{edges[i + 1]:.2f}"
        print(f"{name:>15s} {n:>11,d} {loss_fix[selected].mean():>9.6f} "
              f"{loss_wgt[selected].mean():>9.6f} {d.mean():>+10.6f} "
              f"{d.std() / np.sqrt(n):>7.1e} "
              f"{100 * d.mean() / loss_fix[selected].mean():>+8.1f} "
              f"{100 * n * d.mean() / len(values) / total:>+7.1f}")


def weights_map(gname: str, code_value: int, ln_e_edges: np.ndarray) -> None:
    """Куда ушла масса весов: средний вес и доля массы по бинам lnE и cos θ.

    Оси — те же, на которых веса строились: `energy_cache.npy` (энергия
    компаньона) и `theta.npy`; обе сверяются с источниками на одном парте.
    """
    particle_types = np.load(DATASET / "particle_types.npy")
    weights = np.load(DATASET / "weights.npy")
    energy = np.load(DATASET / "energy_cache.npy")
    theta = np.load(DATASET / "theta.npy")
    part_keys = np.load(DATASET / "h5_part_keys.npy", allow_pickle=True).astype(str)
    local_ids = np.load(DATASET / "h5_local_event_ids.npy").astype(np.int64)

    selected = particle_types == code_value
    weight, energy, theta = weights[selected], energy[selected], theta[selected]
    keys, local = part_keys[selected], local_ids[selected]
    first = np.unique(keys)[0]
    rows = np.flatnonzero(keys == first)
    order = np.argsort(local[rows])              # h5py требует возрастающих индексов
    with h5py.File(MC_H5, "r") as store:
        prime_theta = store[gname]["prime_prty"][first]["data"][local[rows][order], 0]
    print(f"\n-- карта весов {gname}: N={int(selected.sum()):,}, "
          f"max|theta.npy − prime_prty| на {first}: "
          f"{np.nanmax(np.abs(theta[rows][order] - prime_theta)):.2e}°")
    finite = np.isfinite(energy) & (energy > 0)
    log_e = np.log10(energy[finite])
    index = np.clip(np.searchsorted(ln_e_edges, log_e, side="right") - 1,
                    0, len(ln_e_edges) - 2)
    weight_here = weight[finite]
    print(f"{'lnE(ГэВ)':>15s} {'N':>10s} {'доля массы w':>13s} {'средний w':>10s}")
    for i in range(len(ln_e_edges) - 1):
        band = index == i
        if not band.any():
            continue
        print(f"{ln_e_edges[i]:>6.2f}..{ln_e_edges[i + 1]:<6.2f} {int(band.sum()):>10,d} "
              f"{100 * weight_here[band].sum() / weight_here.sum():>12.1f}% "
              f"{weight_here[band].mean():>10.3f}")
    cos_theta = np.cos(np.deg2rad(theta[finite]))
    cos_index = np.clip(np.searchsorted(COS_THETA_EDGES, cos_theta, side="right") - 1,
                        0, len(COS_THETA_EDGES) - 2)
    print(f"{'cosθ':>15s} {'N':>10s} {'доля массы w':>13s} {'средний w':>10s}")
    for i in range(len(COS_THETA_EDGES) - 1):
        band = cos_index == i
        if not band.any():
            continue
        print(f"{COS_THETA_EDGES[i]:>6.2f}..{COS_THETA_EDGES[i + 1]:<6.2f} {int(band.sum()):>10,d} "
              f"{100 * weight_here[band].sum() / weight_here.sum():>12.1f}% "
              f"{weight_here[band].mean():>10.3f}")


def main() -> None:
    started = time.time()
    data = {tag: read_predictions(PREDS / run / "mc_merged_thr0p8.duckdb")
            for tag, run in RUNS.items()}
    for tag in RUNS:
        order = np.argsort(data[tag]["event_fk"])
        data[tag] = {key: value[order] for key, value in data[tag].items()}
    common = np.intersect1d(data["fix"]["event_fk"], data["wgt"]["event_fk"],
                            assume_unique=True)
    take = {tag: np.searchsorted(data[tag]["event_fk"], common) for tag in RUNS}
    code = class_code(common)
    loss = {tag: focal_loss(common, data[tag]["score"][take[tag]]) for tag in RUNS}
    print(f"общих событий {len(common):,} ({time.time() - started:.0f} с)", flush=True)

    print("раскладываю event_fk по партам через каталог...", flush=True)
    part_code, local_idx, part_keys = locate(common)
    print(f"партов: {len(part_keys):,} ({time.time() - started:.0f} с)", flush=True)

    for name, value in CLASSES:
        selected = code == value
        parts = np.unique(part_code[selected])
        print(f"\n{'=' * 78}\nкласс {name}", flush=True)
        physics, fallback = read_class_physics(
            name, parts, part_keys,
            np.searchsorted(parts, part_code[selected]),
            local_idx[selected], int(selected.sum()))
        finite = (np.isfinite(physics["energy"]) & (physics["energy"] > 0)
                  & np.isfinite(physics["theta"]))
        print(f"    физика есть у {finite.mean():.2%} событий, запасная ветка "
              f"энергии у {fallback:,} ({time.time() - started:.0f} с)", flush=True)

        log_e = np.log10(physics["energy"][finite])
        ln_e_edges = np.quantile(log_e, np.linspace(0, 1, 9))
        for label, values, edges in (("lnE (октили, ГэВ)", log_e, ln_e_edges),
                                     ("cos(theta)",
                                      np.cos(np.deg2rad(physics["theta"][finite])),
                                      COS_THETA_EDGES),
                                     ("phi (град)", physics["phi"][finite], PHI_EDGES)):
            bin_table(f"{name}: {label}", values,
                      loss["fix"][selected][finite], loss["wgt"][selected][finite],
                      edges)
        weights_map(name, value, ln_e_edges)
    print(f"\nвсего {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
