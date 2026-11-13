"""Расширенные признаки события из уже выбранных хитов.

Дополняет `_features` в `support_vs_measure.py`, а не заменяет его: старые
колонки остаются, новые добавляются с собственными именами. Группы:
заряд, время, световая кривая, геометрия трека, фиты z/x/y(t), занятость
каналов и модулей, корреляции заряда с координатами.

Все признаки считаются по тем хитам, которые передал вызывающий код. В
основном анализе это хиты, принятые sig-noise моделью (prob > 0.8), то есть
тот же набор, который видит классификатор.

Устойчивость важнее полноты: при 1-2 хитах, нулевом разбросе и делении на
ноль возвращается 0.0 или NaN, но не исключение. NaN дальше заполняется
средним по MC-референсу в `_prepare_matrix`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

#: Хиты модуля: 36 каналов на струну, 12 модулей по 3 ФЭУ.
CHANNELS_PER_STRING = 36
CHANNELS_PER_MODULE = 3
#: Границы «яркого» хита; q_frac_* считаются по SN-выбранным хитам.
CHARGE_THRESHOLDS = (5.0, 20.0, 100.0)
#: Число равновременных бинов световой кривой.
LC_BINS = 8
#: Выше этого числа хитов диаметр облака заменяется диагональю bbox:
#: pdist квадратичен, а хвост ярких событий не должен стоить секунд.
DIAMETER_MAX_HITS = 512
#: Скорость света в м/нс — масштаб для beta и длины по времени.
C_M_PER_NS = 0.299_792_458


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b else default


def _entropy(p: np.ndarray, normalize: bool = True) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    if normalize and p.size > 1:
        h /= np.log(p.size)
    return h


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * x).sum() / (n * total)) - (n + 1.0) / n)


def _moments(x: np.ndarray) -> tuple[float, float, float]:
    """IQR, MAD и (skew, kurtosis); NaN при недостатке точек."""
    if x.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    q25, q75 = np.quantile(x, [0.25, 0.75])
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    sd = x.std()
    if sd <= 0:
        return float(q75 - q25), float(mad), 0.0, 0.0
    z = (x - x.mean()) / sd
    return (float(q75 - q25), float(mad),
            float((z ** 3).mean()), float((z ** 4).mean() - 3.0))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _linfit(t: np.ndarray, y: np.ndarray,
            w: np.ndarray | None = None) -> dict[str, float]:
    """Взвешенный МНК y = a + b t: наклон, ошибка, R², остатки."""
    n = t.size
    if n < 3:
        return {"slope": 0.0, "slope_err": 0.0, "r2": 0.0,
                "resid_rms": 0.0, "resid_q95": 0.0,
                "frac_5": 0.0, "frac_10": 0.0}
    if w is None:
        w = np.ones(n, dtype=np.float64)
    sw = w.sum()
    st = (w * t).sum()
    sy = (w * y).sum()
    stt = (w * t * t).sum()
    sty = (w * t * y).sum()
    den = sw * stt - st * st
    if sw <= 0 or den <= 1e-12:
        return {"slope": 0.0, "slope_err": 0.0, "r2": 0.0,
                "resid_rms": 0.0, "resid_q95": 0.0,
                "frac_5": 0.0, "frac_10": 0.0}
    slope = (sw * sty - st * sy) / den
    intercept = (stt * sy - st * sty) / den
    resid = y - (intercept + slope * t)
    ssres = float((w * resid * resid).sum())
    y_mean = sy / sw
    sstot = float((w * (y - y_mean) ** 2).sum())
    rms = np.sqrt(ssres / sw)
    dof = max(n - 2, 1)
    slope_err = float(np.sqrt((ssres / dof) * sw / den))
    return {
        "slope": float(slope),
        "slope_err": slope_err,
        "r2": float(1.0 - ssres / sstot) if sstot > 0 else 0.0,
        "resid_rms": float(rms),
        "resid_q95": float(np.quantile(np.abs(resid), 0.95)),
        "frac_5": float((np.abs(resid) < 5.0).mean()),
        "frac_10": float((np.abs(resid) < 10.0).mean()),
    }


def _light_curve(t_rel: np.ndarray, q: np.ndarray,
                 span: float) -> dict[str, float]:
    """Световая кривая в долях span и компактные статистики формы."""
    n = t_rel.size
    out: dict[str, float] = {}
    if span > 0:
        idx = np.minimum((t_rel / span * LC_BINS).astype(np.int64),
                         LC_BINS - 1)
    else:
        idx = np.zeros(n, dtype=np.int64)
    q_bin = np.bincount(idx, weights=q, minlength=LC_BINS).astype(np.float64)
    h_bin = np.bincount(idx, minlength=LC_BINS).astype(np.float64)
    q_frac = q_bin / q_bin.sum() if q_bin.sum() > 0 else q_bin
    h_frac = h_bin / h_bin.sum() if h_bin.sum() > 0 else h_bin
    for i in range(LC_BINS):
        out[f"lc_q_frac_{i}"] = float(q_frac[i])
        out[f"lc_h_frac_{i}"] = float(h_frac[i])
    out["lc_q_entropy"] = _entropy(q_frac)
    out["lc_h_entropy"] = _entropy(h_frac)
    out["lc_occupied_bins"] = float((h_bin > 0).sum())
    out["lc_peak_q_frac"] = float(q_frac.max())
    out["lc_peak_h_frac"] = float(h_frac.max())

    bins = np.arange(LC_BINS, dtype=np.float64)
    mean_bin = float((q_frac * bins).sum())
    var_bin = float((q_frac * (bins - mean_bin) ** 2).sum())
    sd_bin = np.sqrt(var_bin)
    out["lc_q_width"] = float(sd_bin)
    out["lc_q_skew"] = float(
        (q_frac * (bins - mean_bin) ** 3).sum() / sd_bin ** 3
        if sd_bin > 0 else 0.0)

    # Моменты накопленного заряда: t50/t90 в долях span.
    order = np.argsort(t_rel, kind="stable")
    cum = np.cumsum(q[order])
    total = cum[-1] if cum.size else 0.0
    if total > 0 and span > 0:
        ts = t_rel[order]
        t50 = float(ts[np.searchsorted(cum, 0.5 * total)])
        t90 = float(ts[np.searchsorted(cum, 0.9 * total)])
        out["lc_t50"] = t50 / span
        out["lc_t90"] = t90 / span
        out["lc_rise50"] = t50 / span
        out["lc_fall50_90"] = (t90 - t50) / span
        early = ts < t50
        out["lc_early_q_frac"] = _safe_div(
            float(q[order][early].sum()), float(total), 0.0)
        out["lc_late_q_frac"] = _safe_div(
            float(q[order][~early].sum()), float(total), 0.0)
    else:
        out["lc_t50"] = out["lc_t90"] = 0.0
        out["lc_rise50"] = out["lc_fall50_90"] = 0.0
        out["lc_early_q_frac"] = out["lc_late_q_frac"] = 0.0

    # Автокорреляция и грубая спектральная статистика бинов заряда.
    x = q_frac - q_frac.mean()
    den = float((x * x).sum())
    out["lc_ac1"] = float((x[1:] * x[:-1]).sum() / den) if den > 0 else 0.0
    out["lc_ac2"] = float((x[2:] * x[:-2]).sum() / den) if den > 0 else 0.0
    out["lc_zcr"] = float((np.diff(np.signbit(x))).mean()) if n > 1 else 0.0
    spec = np.abs(np.fft.rfft(x))
    if spec.size > 1 and spec.sum() > 0:
        freq = np.arange(spec.size, dtype=np.float64)
        centroid = float((freq * spec).sum() / spec.sum())
        out["lc_spec_centroid"] = centroid / (spec.size - 1)
        out["lc_spec_spread"] = float(
            np.sqrt((spec * (freq - centroid) ** 2).sum() / spec.sum())
            / (spec.size - 1))
    else:
        out["lc_spec_centroid"] = out["lc_spec_spread"] = 0.0
    return out


def event_features(data: np.ndarray, channels: np.ndarray) -> dict[str, float]:
    """Расширенные признаки одного события.

    Parameters
    ----------
    data:
        `(n, 5)` float32: q (p.e.), t (нс), x, y, z (м). Уже отобранные хиты.
    channels:
        `(n,)` целые номера каналов: струна = ch // 36, модуль = ch // 3.
    """
    data = np.asarray(data, dtype=np.float64)
    ch = np.asarray(channels, dtype=np.int64)
    q, t = data[:, 0], data[:, 1]
    x, y, z = data[:, 2], data[:, 3], data[:, 4]
    n = q.size
    if n == 0:
        raise ValueError("event_features: пустое событие")

    sid = ch // CHANNELS_PER_STRING
    mid = ch // CHANNELS_PER_MODULE
    radial = np.hypot(x, y)
    t0 = t - t.min()
    span = float(t0.max())

    feats: dict[str, float] = {}

    feats["n_channels"] = float(np.unique(ch).size)
    feats["n_modules"] = float(np.unique(mid).size)
    mod_counts = np.bincount(mid - mid.min())
    feats["hits_per_module_mean"] = float(mod_counts.mean())
    feats["hits_per_module_max"] = float(mod_counts.max())
    feats["n_multi_modules"] = float((mod_counts > 1).sum())
    ch_counts = np.bincount(ch - ch.min())
    feats["n_multi_channels"] = float((ch_counts > 1).sum())
    feats["channel_repeat_max"] = float(ch_counts.max())
    feats["channel_repeat_frac"] = float(1.0 - np.unique(ch).size / n)
    str_counts = np.bincount(sid - sid.min())
    feats["string_hit_entropy"] = _entropy(str_counts)
    feats["string_hits_gini"] = _gini(str_counts)

    # --- заряд: форма распределения в log1p, концентрация в сыром ---
    lq = np.log1p(q)
    q25, q50, q75 = np.quantile(q, [0.25, 0.5, 0.75])
    lq25, lq50, lq75 = np.log1p([q25, q50, q75])
    lq_iqr, lq_mad, lq_skew, lq_kurt = _moments(lq)
    feats.update({
        "q_sum_log": float(np.log1p(q.sum())),
        "q_log_min": float(lq.min()),
        "q_log_max": float(lq.max()),
        "q_log_median": float(lq50),
        "q_log_mean": float(lq.mean()),
        "q_log_std": float(lq.std()),
        "q_log_iqr": lq_iqr,
        "q_log_mad": lq_mad,
        "q_log_skew": lq_skew,
        "q_log_kurt": lq_kurt,
        "q_gini": _gini(q),
    })
    q_share = q / q.sum() if q.sum() > 0 else np.zeros_like(q)
    feats["q_top1_share"] = float(q_share.max())
    feats["q_top3_share"] = float(np.sort(q_share)[-3:].sum())
    feats["q_effective_n"] = float(1.0 / (q_share ** 2).sum()) if q.sum() > 0 else 0.0
    for thr in CHARGE_THRESHOLDS:
        feats[f"q_frac_gt{int(thr)}"] = float((q > thr).mean())

    str_q = np.bincount(sid - sid.min(), weights=q)
    ch_q = np.bincount(ch - ch.min(), weights=q)
    feats["string_charge_top_share"] = _safe_div(float(str_q.max()), float(str_q.sum()))
    feats["string_charge_entropy"] = _entropy(str_q)
    feats["channel_charge_top_share"] = _safe_div(float(ch_q.max()), float(ch_q.sum()))
    feats["channel_charge_entropy"] = _entropy(ch_q)
    feats["channel_effective_n_charge"] = (
        float(1.0 / ((ch_q / ch_q.sum()) ** 2).sum()) if ch_q.sum() > 0 else 0.0)

    # --- время: разброс, промежутки, асимметрия и связь с зарядом ---
    t_iqr, t_mad, t_skew, t_kurt = _moments(t0)
    tq25, tq75 = np.quantile(t0, [0.25, 0.75])
    order = np.argsort(t0, kind="stable")
    gaps = np.diff(t0[order])
    feats.update({
        "t_span_raw": span,
        "t_iqr": float(tq75 - tq25),
        "t_mad": t_mad,
        "t_skew": t_skew,
        "t_kurt": t_kurt,
        "t_q25": float(tq25),
        "t_q75": float(tq75),
        "t_hit_rate": _safe_div(n, span),
        "t_gap_mean": float(gaps.mean()) if gaps.size else 0.0,
        "t_gap_median": float(np.median(gaps)) if gaps.size else 0.0,
        "t_gap_std": float(gaps.std()) if gaps.size else 0.0,
        "t_gap_q90": float(np.quantile(gaps, 0.9)) if gaps.size else 0.0,
        "t_gap_frac_lt1": float((gaps < 1.0).mean()) if gaps.size else 0.0,
        "t_gap_frac_gt50": float((gaps > 50.0).mean()) if gaps.size else 0.0,
        "t_qmean": _safe_div(float((q * t0).sum()), float(q.sum())),
        "t_qstd": float(np.sqrt(np.average((t0 - np.average(t0, weights=q)) ** 2,
                                           weights=q))) if q.sum() > 0 else 0.0,
        "q_t_corr": _corr(q, t0),
        "logq_t_corr": _corr(lq, t0),
        "q_abst_corr": _corr(q, np.abs(t0 - np.median(t0))),
    })
    early = t0 < np.median(t0)
    feats["t_charge_asym"] = _safe_div(
        float(q[~early].sum() - q[early].sum()), float(q.sum()))
    feats["t_count_asym"] = _safe_div(
        float((~early).sum() - early.sum()), float(n))

    # --- световая кривая ---
    feats.update(_light_curve(t0, q, span))

    # --- заряд и геометрия ---
    z_med = float(np.median(z))
    r_med = float(np.median(radial))
    feats.update({
        "q_z_corr": _corr(q, z),
        "q_r_corr": _corr(q, radial),
        "q_absz_corr": _corr(q, np.abs(z - z_med)),
        "q_z_asym": _safe_div(
            float(q[z > z_med].sum() - q[z <= z_med].sum()), float(q.sum())),
        "q_r_asym": _safe_div(
            float(q[radial > r_med].sum() - q[radial <= r_med].sum()),
            float(q.sum())),
        "q_centroid_offset": float(np.linalg.norm(
            np.average(data[:, 2:5], axis=0, weights=q)
            - data[:, 2:5].mean(axis=0))),
    })
    bright = q > CHARGE_THRESHOLDS[0]
    if bright.sum() >= 2:
        bc = data[bright, 2:5].mean(axis=0)
        gc = data[:, 2:5].mean(axis=0)
        feats["bright_centroid_offset"] = float(np.linalg.norm(bc - gc))
    else:
        feats["bright_centroid_offset"] = 0.0

    # --- форма облака: bbox, PCA, диаметр ---
    pts = data[:, 2:5]
    bbox = pts.max(axis=0) - pts.min(axis=0)
    feats.update({
        "bbox_dx": float(bbox[0]),
        "bbox_dy": float(bbox[1]),
        "bbox_dz": float(bbox[2]),
        "bbox_diag": float(np.linalg.norm(bbox)),
    })
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / n
    eigval, eigvec = np.linalg.eigh(cov)
    order_e = np.argsort(eigval)[::-1]
    eigval, eigvec = np.clip(eigval[order_e], 0.0, None), eigvec[:, order_e]
    l1, l2, l3 = (max(float(v), 1e-12) for v in eigval)
    pc1 = eigvec[:, 0]
    proj = centered @ pc1
    track_length = float(proj.max() - proj.min())
    perp = np.linalg.norm(centered - np.outer(proj, pc1), axis=1)
    if n <= DIAMETER_MAX_HITS:
        diameter = float(pdist(pts, metric="euclidean").max()) if n > 1 else 0.0
    else:
        diameter = float(np.linalg.norm(bbox))
    path = (np.linalg.norm(np.diff(pts[order], axis=0), axis=1).sum()
            if n > 1 else 0.0)
    feats.update({
        "pca_participation_ratio": float(
            (l1 + l2 + l3) ** 2 / (l1 * l1 + l2 * l2 + l3 * l3)),
        "pca_pc1_var_frac": float(l1 / (l1 + l2 + l3)),
        "pca_linearity": float((l1 - l2) / l1),
        "pca_planarity": float((l2 - l3) / l1),
        "gyration_radius": float(np.sqrt(l1 + l2 + l3)),
        "track_length_pc1": track_length,
        "track_thickness_mean": float(perp.mean()),
        "track_thickness_max": float(perp.max()),
        "diameter_3d": diameter,
        "straightness": _safe_div(diameter, path),
        "angle_pc1_z_deg": float(np.degrees(np.arccos(
            np.clip(abs(float(pc1[2])), 0.0, 1.0)))),
        "sigma_z": float(z.std()),
        "sigma_radius": float(radial.std()),
        "z_verticality": _safe_div(float(z.std()), float(radial.std())),
    })

    # --- фиты координат по времени ---
    fit_z = _linfit(t0, z)
    fit_x = _linfit(t0, x)
    fit_y = _linfit(t0, y)
    fit_zq = _linfit(t0, z, q)
    feats.update({
        "fit_z_slope": fit_z["slope"],
        "fit_z_slope_err": fit_z["slope_err"],
        "fit_z_r2": fit_z["r2"],
        "fit_z_resid_rms": fit_z["resid_rms"],
        "fit_z_resid_q95": fit_z["resid_q95"],
        "fit_z_frac_5m": fit_z["frac_5"],
        "fit_z_frac_10m": fit_z["frac_10"],
        "fit_x_slope": fit_x["slope"],
        "fit_x_r2": fit_x["r2"],
        "fit_y_slope": fit_y["slope"],
        "fit_y_r2": fit_y["r2"],
        "fit_z_slope_q": fit_zq["slope"],
        "fit_z_r2_q": fit_zq["r2"],
    })
    speed = float(np.linalg.norm(
        [fit_z["slope"], fit_x["slope"], fit_y["slope"]]))
    feats["fit_speed"] = speed
    feats["fit_beta"] = speed / C_M_PER_NS
    feats["fit_angle_z_deg"] = float(np.degrees(np.arccos(
        np.clip(abs(fit_z["slope"]) / speed, 0.0, 1.0)))) if speed > 0 else 0.0
    feats["fit_length_time"] = speed * span
    feats["fit_length_ratio"] = _safe_div(speed * span, diameter)

    # --- повторы на канале ---
    sorder = np.lexsort((t0, ch))
    sch, st = ch[sorder], t0[sorder]
    same = sch[1:] == sch[:-1]
    dts = st[1:][same] - st[:-1][same]
    feats.update({
        "occ_channel_dt_median": float(np.median(dts)) if dts.size else 0.0,
        "occ_channel_dt_mean": float(dts.mean()) if dts.size else 0.0,
        "occ_channel_dt_frac_lt10": (
            float((dts < 10.0).mean()) if dts.size else 0.0),
    })
    return {name: float(value) for name, value in feats.items()}


#: Признаки, которые имеет смысл стандартизовать вместе, по смыслу.
GROUPS: dict[str, tuple[str, ...]] = {
    "basic": (
        "n_channels", "n_modules",
        "hits_per_module_mean", "hits_per_module_max", "n_multi_modules",
        "n_multi_channels", "channel_repeat_max", "channel_repeat_frac",
        "string_hit_entropy", "string_hits_gini",
    ),
    "charge": (
        "q_sum_log", "q_log_min", "q_log_max", "q_log_median", "q_log_mean",
        "q_log_std", "q_log_iqr", "q_log_mad", "q_log_skew", "q_log_kurt",
        "q_gini", "q_top1_share", "q_top3_share", "q_effective_n",
        "q_frac_gt5", "q_frac_gt20", "q_frac_gt100",
        "string_charge_top_share", "string_charge_entropy",
        "channel_charge_top_share", "channel_charge_entropy",
        "channel_effective_n_charge",
    ),
    "time": (
        "t_span_raw", "t_iqr", "t_mad", "t_skew", "t_kurt", "t_q25", "t_q75",
        "t_hit_rate", "t_gap_mean", "t_gap_median", "t_gap_std",
        "t_gap_q90", "t_gap_frac_lt1", "t_gap_frac_gt50", "t_qmean",
        "t_qstd", "t_charge_asym", "t_count_asym", "q_t_corr",
        "logq_t_corr", "q_abst_corr",
    ),
    "lightcurve": tuple(
        [f"lc_q_frac_{i}" for i in range(LC_BINS)]
        + [f"lc_h_frac_{i}" for i in range(LC_BINS)]
        + ["lc_q_entropy", "lc_h_entropy", "lc_occupied_bins",
           "lc_peak_q_frac", "lc_peak_h_frac", "lc_q_width", "lc_q_skew",
           "lc_t50", "lc_t90", "lc_rise50", "lc_fall50_90",
           "lc_early_q_frac", "lc_late_q_frac", "lc_ac1", "lc_ac2",
           "lc_zcr", "lc_spec_centroid", "lc_spec_spread"]
    ),
    "charge_geometry": (
        "q_z_corr", "q_r_corr", "q_absz_corr", "q_z_asym", "q_r_asym",
        "q_centroid_offset", "bright_centroid_offset",
    ),
    "track": (
        "bbox_dx", "bbox_dy", "bbox_dz", "bbox_diag",
        "pca_participation_ratio", "pca_pc1_var_frac", "pca_linearity",
        "pca_planarity", "gyration_radius", "track_length_pc1",
        "track_thickness_mean", "track_thickness_max", "diameter_3d",
        "straightness", "angle_pc1_z_deg", "sigma_z", "sigma_radius",
        "z_verticality",
    ),
    "fits": (
        "fit_z_slope", "fit_z_slope_err", "fit_z_r2", "fit_z_resid_rms",
        "fit_z_resid_q95", "fit_z_frac_5m", "fit_z_frac_10m",
        "fit_x_slope", "fit_x_r2", "fit_y_slope", "fit_y_r2",
        "fit_z_slope_q", "fit_z_r2_q", "fit_speed", "fit_beta",
        "fit_angle_z_deg", "fit_length_time", "fit_length_ratio",
    ),
    "occupancy": (
        "occ_channel_dt_median", "occ_channel_dt_mean",
        "occ_channel_dt_frac_lt10",
    ),
}
