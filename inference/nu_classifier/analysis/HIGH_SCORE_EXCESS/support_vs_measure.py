"""Support vs measure: где лежит exp-хвост относительно muatm MC.

Операционное определение: в независимом (не энкодерном) пространстве
признаков события считаем расстояние до k-го соседа в muatm MC. Событие
«вне носителя», если расстояние больше порога, откалиброванного по
held-out MC. Доля таких событий в exp-хвосте даёт support-вклад; остаток
избытка — measure-вклад.

Признаки считаются из сырых хитов, без энкодера; по умолчанию — только из
хитов, принятых sig-noise моделью (prob > 0.8), то есть из того же набора,
который видит классификатор:
  * per-event: счётчики, заряд, время, геометрия облака;
  * cluster/string geometry: положения и наклоны струн, межструнные
    расстояния, PCA облака. Это нужно, потому что в exp струны плавают и
    наклоняются, а в MC не все комбинации представлены.

Часть HDF5 читается целиком только если в ней >= PART_WHOLE_MIN_EVENTS
выбранных событий и она <= PART_WHOLE_MAX_BYTES; иначе срезы читаются по
событию. Замерено холодным чтением: MC (56 МБ/часть, ~4 события) — 158 ev/s
по-событийно против 36 ev/s целиком; exp (2–8 ГБ/часть, ~480 событий) —
65 против 58 ev/s. `--features-in/--features-out` кэшируют таблицу parquet,
чтобы повторный kNN не читал HDF5.

Стандартизация по MC: признаки с СКО < 1e-9 считаются константными и не
масштабируются. Иначе `string_tilt_*` (в MC струны строго вертикальны,
СКО ~1e-14) делится на этот шум и все exp-события оказываются в ±8σ.

Режим `--representation embedding` берёт 128-D выход энкодера из
prediction DB (`embeddings`); HDF5 не читается. Это model-level анализ:
вопрос не «что в данных», а «что видит энкодер после DA и аугментаций».

Запуск (baikal25 python):
    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/support_vs_measure.py
    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/support_vs_measure.py \\
        --representation embedding
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
from pynndescent import NNDescent
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths, raw_features  # noqa: E402
from inference.shared.h5_hits import STRING_DIVISOR  # noqa: E402

DEFAULT_CHECKPOINT = (
    "261024_1202_da_nu_classifier_exp_full_E1_weights_z128_k10_"
    "dlr1e-2_lam0p003_delay2_h8s3_muOnlyDA_s42@best_da_model"
)
#: Два exp-рана с подтверждённым дефектом четырёх каналов; см. reader/spec.py.
BAD_EXP_RUNS = ("part_s2020_c02_r0020", "part_s2020_c02_r0249")
#: Непрерывные диапазоны event_fk в catalog_v2; см. doc / memory.
MC_MUATM_MAX = 576_947_380
#: Полная чистая exp-популяция (NOT_EXCLUDED) для данного каталога.
EXP_CLEAN_TOTAL = 3_180_681
BRIGHT_CHARGE = 5.0
#: Тот же порог, что у sig-noise модели и в prediction DB (thr0p8).
SN_THRESHOLD = 0.8
#: `paths.probs` takes the HDF5 stem, which differs from the source name.
PROBS_STEM = {"mc_merged": "baikal_mc_merged", "exp_full": "exp_full"}
#: Whole-part read only pays when the part holds many selected events.
#: Measured cold with 12 workers on disjoint parts:
#:   MC  (56 MB/part, ~4 events/part):  whole 36.5 ev/s, per-event 158.5 ev/s
#:   exp (2-8 GB/part, ~480 events/part): whole 65.1 ev/s, per-event 57.9 ev/s
#: So MC parts stay on per-event slices, exp parts are read whole. Break-even
#: by the MC curve is ~25 events; every exp part has >= 82, every MC part <= 17.
PART_WHOLE_MIN_EVENTS = 25
#: Cap for one whole part. exp_full parts run 2.2-29.8 GB (median 6.6 GB);
#: 12 workers x 8 GB = 96 GB, which fits in the ~144 GB available. Larger
#: parts (12 of 29) fall back to per-event slices.
PART_WHOLE_MAX_BYTES = 8 * 1024 ** 3
#: 5 float32 (data) + int32 (channels) + float32 (probs).
PART_BYTES_PER_HIT = 28


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _load_scores(checkpoint: str, tail_logit: float, low_logit: float,
                 low_sample: int, background: int, seed: int):
    """Выбрать exp tail/low и muatm tail/low/background из prediction DB."""
    pred = Path("inference/nu_classifier/preds") / checkpoint
    exp_db = pred / "exp_full_thr0p8.duckdb"
    mc_db = pred / "mc_merged_thr0p8.duckdb"
    s_tail, s_low = _sigmoid(tail_logit), _sigmoid(low_logit)

    con = duckdb.connect(str(exp_db), read_only=True)
    exp_tail = con.execute(
        "SELECT event_fk, score, n_sn_hits, n_sn_strings "
        "FROM predictions WHERE score > ?",
        [s_tail]).fetchdf()
    exp_low_all = con.execute(
        "SELECT event_fk, score, n_sn_hits, n_sn_strings "
        "FROM predictions WHERE score > ? AND score <= ?",
        [s_low, s_tail]).fetchdf()
    exp_total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()
    exp_low = exp_low_all.sample(
        n=min(low_sample, len(exp_low_all)), random_state=seed)

    con = duckdb.connect(str(mc_db), read_only=True)
    mc_tail = con.execute(
        "SELECT event_fk, score, n_sn_hits, n_sn_strings FROM predictions "
        "WHERE event_fk <= ? AND score > ?",
        [MC_MUATM_MAX, s_tail]).fetchdf()
    mc_low_all = con.execute(
        "SELECT event_fk, score, n_sn_hits, n_sn_strings FROM predictions "
        "WHERE event_fk <= ? AND score > ? AND score <= ?",
        [MC_MUATM_MAX, s_low, s_tail]).fetchdf()
    mc_total = con.execute(
        "SELECT COUNT(*) FROM predictions WHERE event_fk <= ?",
        [MC_MUATM_MAX]).fetchone()[0]
    mc_low_n = len(mc_low_all)
    mc_bg = con.execute(
        f"SELECT event_fk, score, n_sn_hits, n_sn_strings FROM predictions "
        f"WHERE event_fk <= ? AND score <= ? "
        f"USING SAMPLE {int(background)} ROWS (reservoir, {int(seed)})",
        [MC_MUATM_MAX, s_low]).fetchdf()
    con.close()
    mc_low = mc_low_all.sample(
        n=min(low_sample, len(mc_low_all)), random_state=seed)

    exp_tail = exp_tail.assign(grp="exp_tail")
    exp_low = exp_low.assign(grp="exp_low")
    mc_tail = mc_tail.assign(
        grp=np.where(mc_tail.event_fk.to_numpy() % 2 == 0,
                     "mc_tail_ref", "mc_tail_hold"))
    mc_low = mc_low.assign(
        grp=np.where(mc_low.event_fk.to_numpy() % 2 == 0,
                     "mc_low_ref", "mc_low_hold"))
    mc_bg = mc_bg.assign(grp="mc_bg")

    sel = pd.concat([exp_tail, exp_low, mc_tail, mc_low, mc_bg],
                    ignore_index=True)
    counts = {
        "exp_db_total": int(exp_total),
        "exp_clean_total": int(EXP_CLEAN_TOTAL),
        "mc_total_muatm": int(mc_total),
        "mc_tail": int(len(mc_tail)),
        "mc_low": int(mc_low_n),
        "exp_tail_db": int(len(exp_tail)),
        "exp_low_db": int(len(exp_low_all)),
        "exp_low_sample": int(len(exp_low)),
        "mc_low_sample": int(len(mc_low)),
        "mc_background_sample": int(len(mc_bg)),
    }
    return sel, counts, s_tail, s_low


def _map_locations(sel: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(paths.CATALOG), read_only=True)
    con.register("sel", sel[["event_fk", "grp", "score",
                            "n_sn_hits", "n_sn_strings"]])
    loc = con.execute("""
        SELECT s.event_fk, s.grp, s.score, s.n_sn_hits, s.n_sn_strings,
               h.h5_path, h.part_key, h.local_idx
        FROM sel s JOIN h5_locations h USING (event_fk)
    """).fetchdf()
    con.close()
    return loc


def _features(q, t, x, y, z, ch) -> dict:
    """Функции из УЖЕ выбранных хитов события; без энкодера.

    Выбор хитов делается в `_extract_part`: по умолчанию это принятые
    sig-noise моделью (prob > 0.8), то есть тот же набор, который видит
    классификатор. `bright`/`all` — только контрольные режимы.
    """
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ch = np.asarray(ch, dtype=np.int64)
    n = len(q)
    bright = q > BRIGHT_CHARGE
    nb = int(bright.sum())
    sid = ch // STRING_DIVISOR

    # --- per-event, все хиты ---
    q_mean, q_std = float(q.mean()), float(q.std())
    q95 = float(np.quantile(q, 0.95))
    t_std = float(t.std())
    t05, t95 = (float(v) for v in np.quantile(t, [0.05, 0.95]))
    x_std, y_std, z_std = float(x.std()), float(y.std()), float(z.std())
    radial = np.hypot(x, y)
    qb = q[bright]
    tb = t[bright]
    xb, yb, zb = x[bright], y[bright], z[bright]
    span = lambda v: float(v.max() - v.min()) if len(v) else 0.0
    counts_str = np.bincount(sid - sid.min())

    feats = {
        "n_hits": n,
        "n_bright": nb,
        "n_strings": int(len(np.unique(sid))),
        "n_strings_bright": int(len(np.unique(sid[bright]))) if nb else 0,
        "log_charge_total": np.log1p(q.sum()),
        "log_charge_bright": np.log1p(qb.sum()) if nb else 0.0,
        "log_brightest": np.log1p(qb.max()) if nb else 0.0,
        "log_charge_mean": np.log1p(q_mean),
        "log_charge_std": np.log1p(q_std),
        "log_charge_q95": np.log1p(q95),
        "log_time_span": np.log1p(span(t)),
        "log_time_span_bright": np.log1p(span(tb)) if nb >= 2 else 0.0,
        "time_std": t_std,
        "time_q05": t05,
        "time_q95": t95,
        "log_z_span": np.log1p(span(z)),
        "log_z_span_bright": np.log1p(span(zb)) if nb >= 2 else 0.0,
        "log_z_std": np.log1p(z_std),
        "log_x_std": np.log1p(x_std),
        "log_y_std": np.log1p(y_std),
        "log_radial_std": np.log1p(float(radial.std())),
        "log_xy_span": np.log1p(float(np.hypot(np.ptp(x), np.ptp(y)))),
        "hits_per_string_mean": float(counts_str.mean()),
        "hits_per_string_max": float(counts_str.max()),
        "log_hits_per_string_std": np.log1p(float(counts_str.std())),
    }

    # --- string geometry ---
    z_ref = float(np.median(z))
    sx, sy, sn, szspan, tilt, bx, by = [], [], [], [], [], [], []
    for s_id in np.unique(sid):
        m = sid == s_id
        # Для наклона предпочитаем яркие хиты, иначе все.
        use = (m & bright)
        if use.sum() < 2:
            use = m
        zz, xx, yy = z[use], x[use], y[use]
        sn.append(int(m.sum()))
        szspan.append(span(z[use]) if use.any() else 0.0)
        if len(zz) >= 2 and float(np.ptp(zz)) > 0:
            bx_i, ax_i = np.polyfit(zz, xx, 1)
            by_i, ay_i = np.polyfit(zz, yy, 1)
            sx.append(float(ax_i + bx_i * z_ref))
            sy.append(float(ay_i + by_i * z_ref))
            tilt.append(float(np.degrees(np.arctan(np.hypot(bx_i, by_i)))))
            bx.append(float(bx_i))
            by.append(float(by_i))
        else:
            sx.append(float(xx.mean()) if len(xx) else 0.0)
            sy.append(float(yy.mean()) if len(yy) else 0.0)
            tilt.append(np.nan)
            bx.append(np.nan)
            by.append(np.nan)

    sx, sy = np.asarray(sx), np.asarray(sy)
    tilt, bx, by = np.asarray(tilt), np.asarray(bx), np.asarray(by)
    pos = np.column_stack([sx, sy])
    if len(pos) >= 2:
        dmat = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(pos), 1)
        pdists = dmat[iu]
        pair_mean = float(pdists.mean())
        pair_min = float(pdists.min())
        pair_max = float(pdists.max())
        pair_std = float(pdists.std())
    else:
        pair_mean = pair_min = pair_max = pair_std = 0.0

    tilt_ok = tilt[np.isfinite(tilt)]
    string_pos_radial = np.hypot(sx, sy)
    feats.update({
        "n_strings_fit": int(np.isfinite(tilt).sum()),
        "string_fit_frac": (
            float(np.isfinite(tilt).sum() / len(tilt)) if tilt.size else 0.0),
        "string_n_mean": float(np.mean(sn)) if sn else 0.0,
        "string_n_std": float(np.std(sn)) if sn else 0.0,
        "string_n_max": float(np.max(sn)) if sn else 0.0,
        "string_zspan_mean": float(np.mean(szspan)) if szspan else 0.0,
        "string_zspan_std": float(np.std(szspan)) if szspan else 0.0,
        "string_tilt_mean": float(np.mean(tilt_ok)) if len(tilt_ok) else 0.0,
        "string_tilt_std": float(np.std(tilt_ok)) if len(tilt_ok) else 0.0,
        "string_tilt_max": float(np.max(tilt_ok)) if len(tilt_ok) else 0.0,
        "tilt_frac_gt3": float(np.mean(tilt_ok > 3.0)) if len(tilt_ok) else 0.0,
        "string_bx_mean": float(np.nanmean(bx)) if len(bx) else 0.0,
        "string_bx_std": float(np.nanstd(bx)) if len(bx) else 0.0,
        "string_by_mean": float(np.nanmean(by)) if len(by) else 0.0,
        "string_by_std": float(np.nanstd(by)) if len(by) else 0.0,
        "string_pos_x_std": float(sx.std()) if len(sx) else 0.0,
        "string_pos_y_std": float(sy.std()) if len(sy) else 0.0,
        "string_pos_radial_std": (
            float(string_pos_radial.std()) if len(sx) else 0.0),
        "string_pair_dist_mean": pair_mean,
        "string_pair_dist_min": pair_min,
        "string_pair_dist_max": pair_max,
        "string_pair_dist_std": pair_std,
    })

    # --- PCA облака хитов ---
    X = np.column_stack([x, y, z])
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / len(X)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    l1 = max(float(eigval[0]), 1e-12)
    l2 = max(float(eigval[1]), 1e-12)
    l3 = max(float(eigval[2]), 1e-12)
    feats.update({
        "log_pca_l1": np.log1p(l1),
        "log_pca_l2": np.log1p(l2),
        "log_pca_l3": np.log1p(l3),
        "pca_l2_l1": l2 / l1,
        "pca_l3_l1": l3 / l1,
        "pca_verticality": abs(float(eigvec[2, 0])),
    })
    return feats


#: Per-worker HDF5 handles. The data and probs files are huge; opening them
#: once per part (10k parts) dominated the wall time (measured: 483 events
#: over 475 parts took 500 s cold vs 2.25 s warm for the same work). The
#: worker processes persist across part tasks, so one handle per file per
#: worker is enough. Read-only; multiple processes may open the same file.
_H5_CACHE: dict[str, h5py.File] = {}


def _h5(path: str) -> h5py.File:
    f = _H5_CACHE.get(path)
    if f is None:
        f = h5py.File(path, "r")
        _H5_CACHE[path] = f
    return f


def _extract_part(task) -> list[dict]:
    h5_path, part_key, block, hit_selection, feature_set = task
    is_exp = str(h5_path).endswith("exp_full.h5")
    source = "exp_full" if is_exp else "mc_merged"
    group = "exp_full" if is_exp else "muatm_2020"
    probs_path = paths.probs(PROBS_STEM[source])
    rows = []
    f = _h5(h5_path)
    fp = _h5(probs_path)
    g = f[group]
    starts = g[f"raw/ev_starts/{part_key}"]["data"][:]
    data_ds = g[f"raw/data/{part_key}"]["data"]
    ch_ds = g[f"raw/channels/{part_key}"]["data"]
    prob_ds = fp[group]["probs"][part_key]["data"]

    # Whole-part sequential read: each HDF5 chunk is decompressed once, and
    # the selected events are sliced from RAM. Only worth it when the part
    # holds many events (see PART_WHOLE_MIN_EVENTS); otherwise per-event
    # slices read ~0.4 MB instead of the whole part.
    n_hits_part = int(data_ds.shape[0])
    part_bytes = n_hits_part * PART_BYTES_PER_HIT
    if part_bytes <= PART_WHOLE_MAX_BYTES and len(block) >= PART_WHOLE_MIN_EVENTS:
        part_data = data_ds[:]
        part_ch = ch_ds[:]
        part_prob = prob_ds[:]
    else:
        part_data = part_ch = part_prob = None

    for row in block.itertuples(index=False):
        local = int(row.local_idx)
        a, b = int(starts[local]), int(starts[local + 1])
        if part_data is not None:
            d = part_data[a:b]
            ch = part_ch[a:b]
            prob = part_prob[a:b]
        else:
            d = data_ds[a:b]
            ch = ch_ds[a:b]
            prob = prob_ds[a:b]
        if len(prob) != len(d):
            raise ValueError(
                f"{part_key}#{local}: hits={len(d)} probs={len(prob)}")
        sig = prob > SN_THRESHOLD
        raw_bright = d[:, 0] > BRIGHT_CHARGE
        n_sig_hits = int(sig.sum())
        n_sig_strings = (int(len(np.unique(
            ch[sig] // STRING_DIVISOR))) if sig.any() else 0)
        # Сверка с prediction DB: там те же probs и тот же порог.
        if n_sig_hits != int(row.n_sn_hits):
            raise ValueError(
                f"{part_key}#{local}: SN hits={n_sig_hits} "
                f"!= DB {row.n_sn_hits}")
        if n_sig_strings != int(row.n_sn_strings):
            raise ValueError(
                f"{part_key}#{local}: SN strings={n_sig_strings} "
                f"!= DB {row.n_sn_strings}")

        if hit_selection == "signal":
            selected = sig
        elif hit_selection == "bright":
            selected = raw_bright
        else:
            selected = np.ones(len(d), dtype=bool)
        if int(selected.sum()) == 0:
            continue
        dd = d[selected]
        cc = ch[selected]
        feats = _features(dd[:, 0], dd[:, 1], dd[:, 2],
                          dd[:, 3], dd[:, 4], cc)
        if feature_set == "expanded":
            feats.update(raw_features.event_features(dd, cc))
        feats.update({
            "n_hits_raw": int(len(d)),
            "n_bright_raw": int(raw_bright.sum()),
            "n_signal_hits": n_sig_hits,
            "n_signal_strings": n_sig_strings,
            "signal_fraction": float(sig.mean()),
        })
        rows.append({
            "event_fk": int(row.event_fk),
            "grp": row.grp,
            "score": float(row.score),
            **feats,
        })
    return rows


def _extract_features(loc: pd.DataFrame, workers: int,
                      hit_selection: str, feature_set: str) -> pd.DataFrame:
    tasks = []
    for (h5_path, part_key), block in loc.groupby(
            ["h5_path", "part_key"], sort=False):
        tasks.append((h5_path, part_key,
                      block[["event_fk", "grp", "score", "local_idx",
                             "n_sn_hits", "n_sn_strings"]].copy(),
                      hit_selection, feature_set))
    rows: list[dict] = []
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            rows.extend(_extract_part(task))
            if i % 200 == 0:
                print(f"  parts {i}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, part_rows in enumerate(
                    pool.map(_extract_part, tasks, chunksize=1), 1):
                rows.extend(part_rows)
                if i % 200 == 0:
                    print(f"  parts {i}/{len(tasks)} "
                          f"({len(rows):,} events)", flush=True)
    return pd.DataFrame(rows)


def _prepare_matrix(df, cols, reference):
    """Стандартизация по MC-референсу; константные признаки — без масштаба.

    В MC струны строго вертикальны: `string_tilt_*` имеет СКО ~1e-14, а в exp
    наклон ~0.25°. Деление на такое СКО раздувает exp до ±8σ и делает все
    exp-события «вне носителя» по геометрии. Признаки с СКО < 1e-9 считаем
    константными, оставляем им единичный масштаб и возвращаем их список,
    чтобы отчёт не выдавал этот артефакт за support.
    """
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    R = reference[cols].to_numpy(dtype=np.float64)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = R.mean(axis=0), R.std(axis=0)
    degenerate = sd < 1e-9
    sd[degenerate] = 1.0
    X = np.clip((X - mu) / sd, -8, 8)
    R = np.clip((R - mu) / sd, -8, 8)
    return (R.astype(np.float32), X.astype(np.float32),
            [cols[i] for i in np.flatnonzero(degenerate)])


def _components(S_e, S_m, o_e, o_m) -> dict:
    """Знаковое разложение избытка: outside + inside = E.

    outside_share/inside_share могут выходить за [0,1]: если outside-компонента
    больше всего избытка, inside-компонента отрицательна (дефицит внутри
    носителя). Не обрезаем.
    """
    den = S_e - S_m
    if den <= 0:
        return {"outside_component": float("nan"),
                "inside_component": float("nan"),
                "outside_share": float("nan"),
                "inside_share": float("nan")}
    c_out = S_e * o_e - S_m * o_m
    c_in = S_e * (1.0 - o_e) - S_m * (1.0 - o_m)
    return {"outside_component": float(c_out),
            "inside_component": float(c_in),
            "outside_share": float(c_out / den),
            "inside_share": float(c_in / den)}


def _load_embeddings(loc: pd.DataFrame, checkpoint: str) -> tuple:
    """128-D encoder embeddings for the selected events, from prediction DBs."""
    pred = Path("inference/nu_classifier/preds") / checkpoint
    frames = []
    for is_exp, db_name in ((True, "exp_full_thr0p8.duckdb"),
                            (False, "mc_merged_thr0p8.duckdb")):
        block = loc[loc.h5_path.str.endswith("exp_full.h5") == is_exp]
        if not len(block):
            continue
        db = pred / db_name
        con = duckdb.connect(str(db), read_only=True)
        con.register("sel", block[["event_fk", "grp", "score"]])
        frame = con.execute("""
            SELECT s.event_fk, s.grp, s.score, e.embedding
            FROM sel s JOIN embeddings e USING (event_fk)
        """).fetchdf()
        con.close()
        frame["source"] = "exp_full" if is_exp else "mc_merged"
        frames.append(frame)
        print(f"  embeddings {frame['source'].iloc[0]}: {len(frame):,}",
              flush=True)
    out = pd.concat(frames, ignore_index=True)
    emb = np.vstack([np.asarray(v, dtype=np.float32)
                     for v in out["embedding"]])
    cols = [f"emb_{i}" for i in range(emb.shape[1])]
    out = out.drop(columns="embedding")
    out[cols] = emb
    return out, cols


def _embedding_diagnostics(df, cols, pca_dims):
    """Диагностика коллапса: ранг, участие, объяснённая дисперсия."""
    ref = df[df.grp.isin(["mc_bg", "mc_tail_ref", "mc_low_ref"])]
    X = ref[cols].to_numpy(dtype=np.float64)
    std_raw = X.std(axis=0)
    dead = int((std_raw < 1e-6).sum())
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    total = eig.sum()
    participation = float(total ** 2 / (eig ** 2).sum()) if total > 0 else 0.0
    order = np.argsort(eig)[::-1]
    cum = np.cumsum(eig[order]) / total if total > 0 else np.zeros_like(eig)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    out = {
        "n_dims": int(len(cols)),
        "n_reference": int(len(X)),
        "dead_dims": dead,
        "participation_ratio": participation,
        "n_components_95pct": n95,
        "eigenvalues_top10": [float(v) for v in eig[order][:10]],
        "pca": {},
    }
    for n in pca_dims:
        p = PCA(n_components=n, random_state=42)
        p.fit(X)
        out["pca"][str(n)] = float(p.explained_variance_ratio_.sum())
    return out


def _evaluate(df, cols, k, q, seed):
    ref = df[df.grp.isin(["mc_bg", "mc_tail_ref", "mc_low_ref"])]
    _, Xall, degenerate = _prepare_matrix(df, cols, ref)
    idx = np.flatnonzero(df.grp.isin(
        ["mc_bg", "mc_tail_ref", "mc_low_ref"]).to_numpy())
    index = NNDescent(Xall[idx], n_neighbors=max(k, 50), metric="euclidean",
                      n_jobs=-1, random_state=seed)
    groups = ["exp_tail", "exp_low", "mc_tail_hold", "mc_low_hold"]
    dist = {}
    for grp in groups:
        mask = (df.grp == grp).to_numpy()
        Xg = Xall[mask]
        _, d = index.query(Xg, k=max(k, 50))
        dist[grp] = d[:, k - 1]
    return dist, degenerate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--tail-logit", type=float, default=0.0)
    ap.add_argument("--low-logit", type=float, default=-2.0)
    ap.add_argument("--background", type=int, default=50_000)
    ap.add_argument("--low-sample", type=int, default=10_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--representation", choices=("raw", "embedding"),
                    default="raw",
                    help="raw = hand-crafted features from hits; "
                         "embedding = 128-D encoder features from prediction DB")
    ap.add_argument("--hit-selection", choices=("signal", "bright", "all"),
                    default="signal",
                    help="which hits feed the raw features: SN>0.8 (default, "
                         "same as classifier input), q>5, or all raw hits")
    ap.add_argument("--feature-set", choices=("base", "expanded"),
                    default="expanded",
                    help="base = 61 columns of the first pass; expanded = "
                         "base + charge/time/light-curve/track/fit/occupancy "
                         "features from excess/raw_features.py")
    ap.add_argument("--features-in", default=None,
                    help="read the extracted feature table (parquet) instead "
                         "of touching HDF5 again")
    ap.add_argument("--features-out", default=None,
                    help="where to write the extracted features; default "
                         "data/features_{hit_selection}_{feature_set}_h8s3.parquet")
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-feature-group-out on the full expanded set")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sel, counts, s_tail, s_low = _load_scores(
        args.checkpoint, args.tail_logit, args.low_logit,
        args.low_sample, args.background, args.seed)
    loc = _map_locations(sel)
    bad = loc.part_key.isin(BAD_EXP_RUNS) & loc.grp.str.startswith("exp")
    loc = loc[~bad].copy()

    exp_tail_clean = int((loc.grp == "exp_tail").sum())
    S_e = exp_tail_clean / EXP_CLEAN_TOTAL
    S_m = counts["mc_tail"] / counts["mc_total_muatm"]

    print(f"selected events: {len(loc):,}", flush=True)
    print(f"exp tail clean: {exp_tail_clean:,}  S_e={S_e:.3e}", flush=True)
    print(f"mc tail: {counts['mc_tail']:,}  S_m={S_m:.3e}  "
          f"ratio={S_e / S_m:.3f}", flush=True)

    diagnostics = None
    if args.representation == "embedding":
        df, emb_cols = _load_embeddings(loc, args.checkpoint)
        print(f"embeddings: {len(df):,} events, {len(emb_cols)} dims",
              flush=True)
        ref = df[df.grp.isin(["mc_bg", "mc_tail_ref", "mc_low_ref"])]
        Xref = ref[emb_cols].to_numpy(dtype=np.float64)
        mu, sd = Xref.mean(axis=0), Xref.std(axis=0)
        sd[sd == 0] = 1.0
        Zref = (Xref - mu) / sd
        Zall = (df[emb_cols].to_numpy(dtype=np.float64) - mu) / sd
        pca_cols = {}
        for n in (32, 16):
            p = PCA(n_components=n, random_state=42).fit(Zref)
            cols = [f"pca{n}_{i}" for i in range(n)]
            df[cols] = p.transform(Zall)
            pca_cols[n] = cols
        subsets = {"embedding_all": emb_cols,
                   "pca32": pca_cols[32], "pca16": pca_cols[16]}
        diagnostics = _embedding_diagnostics(df, emb_cols, (32, 16))
        payload_features = {
            "embedding_all": emb_cols,
            "pca32": pca_cols[32],
            "pca16": pca_cols[16],
        }
    else:
        features_cache = None
        if args.features_in:
            df = pd.read_parquet(args.features_in)
            print(f"features from {args.features_in}: {len(df):,} events, "
                  f"{len(df.columns)} columns", flush=True)
        else:
            df = _extract_features(loc, args.workers, args.hit_selection,
                                   args.feature_set)
            print(f"features: {len(df):,} events, {len(df.columns)} columns "
                  f"(hit_selection={args.hit_selection}, "
                  f"feature_set={args.feature_set})", flush=True)
            features_cache = (
                Path(args.features_out) if args.features_out else
                paths.DATA / f"features_{args.hit_selection}_"
                             f"{args.feature_set}_h8s3.parquet")
            df.to_parquet(features_cache, index=False)
            print(f"features cached: {features_cache}", flush=True)
        event_cols = [
            "n_hits", "n_bright", "n_strings", "n_strings_bright",
            "log_charge_total", "log_charge_bright", "log_brightest",
            "log_charge_mean", "log_charge_std", "log_charge_q95",
            "log_time_span", "log_time_span_bright", "time_std",
            "time_q05", "time_q95",
            "log_z_span", "log_z_span_bright", "log_z_std",
            "log_x_std", "log_y_std", "log_radial_std", "log_xy_span",
            "hits_per_string_mean", "hits_per_string_max",
            "log_hits_per_string_std",
        ]
        string_cols = [
            "n_strings_fit", "string_fit_frac",
            "string_n_mean", "string_n_std", "string_n_max",
            "string_zspan_mean", "string_zspan_std",
            "string_tilt_mean", "string_tilt_std", "string_tilt_max",
            "tilt_frac_gt3",
            "string_bx_mean", "string_bx_std", "string_by_mean",
            "string_by_std",
            "string_pos_x_std", "string_pos_y_std", "string_pos_radial_std",
            "string_pair_dist_mean", "string_pair_dist_min",
            "string_pair_dist_max", "string_pair_dist_std",
        ]
        pca_cols = [
            "log_pca_l1", "log_pca_l2", "log_pca_l3",
            "pca_l2_l1", "pca_l3_l1", "pca_verticality",
        ]
        geom_cols = string_cols + pca_cols
        if args.feature_set == "expanded":
            g = raw_features.GROUPS
            event_ext = (event_cols + list(g["basic"])
                         + list(g["charge"]) + list(g["time"]))
            subsets = {
                "event": event_ext,
                "charge": list(g["charge"]),
                "time": list(g["time"]),
                "lightcurve": list(g["lightcurve"]),
                "charge_geometry": list(g["charge_geometry"]),
                "track": pca_cols + list(g["track"]),
                "fits": list(g["fits"]),
                "occupancy": list(g["occupancy"]),
                "strings": string_cols,
                "all": (event_ext + string_cols + pca_cols
                        + list(g["track"]) + list(g["fits"])
                        + list(g["lightcurve"]) + list(g["charge_geometry"])
                        + list(g["occupancy"])),
            }
            subsets["all_no_strings"] = [
                c for c in subsets["all"] if c not in string_cols]
            subsets["all_no_geometry"] = [
                c for c in event_ext + list(g["lightcurve"])
                + list(g["charge_geometry"]) + list(g["occupancy"])]
            payload_features = {k: v for k, v in subsets.items()}
        else:
            subsets = {"all": event_cols + geom_cols,
                       "event": event_cols, "geometry": geom_cols}
            payload_features = {"event": event_cols, "geometry": geom_cols}

    results = {}
    for name, cols in subsets.items():
        dist, degenerate = _evaluate(df, cols, args.k, args.threshold,
                                     args.seed)
        tau = float(np.quantile(dist["mc_tail_hold"], args.threshold / 100))
        tau_low = float(np.quantile(dist["mc_low_hold"],
                                    args.threshold / 100))
        o_e = float((dist["exp_tail"] > tau).mean())
        o_m = float((dist["mc_tail_hold"] > tau).mean())
        o_e_low = float((dist["exp_low"] > tau_low).mean())
        o_m_low = float((dist["mc_low_hold"] > tau_low).mean())
        comp = _components(S_e, S_m, o_e, o_m)
        results[name] = {
            **comp,
            "outside_exp_tail": o_e,
            "outside_mc_tail": o_m,
            "outside_exp_low": o_e_low,
            "outside_mc_low": o_m_low,
            "outside_share_gt1": bool(comp["outside_share"] > 1.0),
            "tau_tail": tau,
            "tau_low": tau_low,
            "constant_features": degenerate,
            "n_exp_tail": int((df.grp == "exp_tail").sum()),
            "n_mc_tail_hold": int((df.grp == "mc_tail_hold").sum()),
            "n_exp_low": int((df.grp == "exp_low").sum()),
            "n_mc_low_hold": int((df.grp == "mc_low_hold").sum()),
        }
        print(f"{name:14s}: outside={comp['outside_share']:.3f} "
              f"inside={comp['inside_share']:.3f} "
              f"(o_exp={o_e:.3f} o_mc={o_m:.3f}; "
              f"low {o_e_low:.3f}/{o_m_low:.3f})")

    # Leave-one-group-out: насколько outside-доля держится без каждой группы.
    results_loo = {}
    if args.loo and args.representation == "raw" \
            and args.feature_set == "expanded":
        g = raw_features.GROUPS
        loo_groups = {
            "event_base": event_cols,
            "basic": list(g["basic"]),
            "charge": list(g["charge"]),
            "time": list(g["time"]),
            "strings": string_cols,
            "pca": pca_cols,
            "track": list(g["track"]),
            "fits": list(g["fits"]),
            "lightcurve": list(g["lightcurve"]),
            "charge_geometry": list(g["charge_geometry"]),
            "occupancy": list(g["occupancy"]),
        }
        for name, cols in loo_groups.items():
            drop = set(cols)
            keep = [c for c in subsets["all"] if c not in drop]
            dist, _ = _evaluate(df, keep, args.k, args.threshold, args.seed)
            tau = float(np.quantile(dist["mc_tail_hold"],
                                    args.threshold / 100))
            tau_low = float(np.quantile(dist["mc_low_hold"],
                                        args.threshold / 100))
            o_e = float((dist["exp_tail"] > tau).mean())
            o_m = float((dist["mc_tail_hold"] > tau).mean())
            o_e_low = float((dist["exp_low"] > tau_low).mean())
            o_m_low = float((dist["mc_low_hold"] > tau_low).mean())
            comp = _components(S_e, S_m, o_e, o_m)
            results_loo[name] = {
                **comp,
                "outside_exp_tail": o_e,
                "outside_mc_tail": o_m,
                "outside_exp_low": o_e_low,
                "outside_mc_low": o_m_low,
                "tau_tail": tau,
                "tau_low": tau_low,
                "n_features": len(keep),
            }
            print(f"LOO -{name:16s}: outside={comp['outside_share']:.3f} "
                  f"inside={comp['inside_share']:.3f} "
                  f"(o_exp={o_e:.3f} o_mc={o_m:.3f}; "
                  f"low {o_e_low:.3f}/{o_m_low:.3f})", flush=True)

    # Чувствительность по k и порогу — на полном наборе.
    sensitivity = []
    for k in sorted({5, args.k, 50}):
        dist, _ = _evaluate(df, subsets[list(subsets)[0]], k,
                            args.threshold, args.seed)
        for thr in (95.0, 99.0, 99.9):
            tau = float(np.quantile(dist["mc_tail_hold"], thr / 100))
            o_e = float((dist["exp_tail"] > tau).mean())
            o_m = float((dist["mc_tail_hold"] > tau).mean())
            comp = _components(S_e, S_m, o_e, o_m)
            sensitivity.append({
                "k": k, "threshold": thr,
                "outside_exp_tail": o_e, "outside_mc_tail": o_m,
                "outside_share": comp["outside_share"],
                "inside_share": comp["inside_share"],
            })

    if args.representation == "embedding":
        suffix = "embedding_"
        hit_selection = "signal (encoder input)"
    elif args.feature_set == "expanded":
        suffix = f"{args.hit_selection}_expanded_"
        hit_selection = args.hit_selection
    else:
        suffix = f"{args.hit_selection}_"
        hit_selection = args.hit_selection
    out = Path(args.out) if args.out else (
        paths.DATA
        / f"support_vs_measure_{suffix}h8s3_muOnlyDA_e23_tail0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "representation": args.representation,
        "feature_set": args.feature_set,
        "features_cache": (str(features_cache)
                           if args.representation == "raw" and features_cache
                           else None),
        "hit_selection": hit_selection,
        "sn": {
            "threshold": SN_THRESHOLD,
            "model": paths.SN_TAG,
            "batch_size": paths.SN_BATCH,
            "note": "probabilities are frozen; batch size is part of the "
                    "hit selection; DB n_sn_hits/n_sn_strings cross-checked "
                    "against the probs slice",
        },
        "population": {
            "mc_class": "muatm via event_fk <= 576947380",
            "exp": "exp_full, bad runs excluded",
            "h8s3": "from prediction DB",
        },
        "selection": {
            "tail_logit": args.tail_logit,
            "low_logit": args.low_logit,
            "tail_score": s_tail,
            "low_score": s_low,
        },
        "counts": counts,
        "exp_tail_clean": exp_tail_clean,
        "S_e": S_e, "S_m": S_m,
        "features": payload_features,
        "embedding_diagnostics": diagnostics,
        "results": results,
        "results_loo": results_loo,
        "sensitivity": sensitivity,
        "note": (
            "Знаковое разложение: E=S_e-S_m=C_out+C_in; outside_share и "
            "inside_share не обрезаются и могут выходить за [0,1]. "
            "outside = kNN distance to muatm MC > held-out MC quantile; "
            "это операционный coverage-прокси, не measure-theoretic support. "
            "Raw features считаются на выбранных хитах: по умолчанию "
            "SN > 0.8 (тот же вход, что у классификатора)."),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {out}")


if __name__ == "__main__":
    main()
