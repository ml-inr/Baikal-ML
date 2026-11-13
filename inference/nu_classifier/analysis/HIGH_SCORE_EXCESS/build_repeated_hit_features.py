"""Признаки «пачек» — многократных срабатываний одного ОМ — для анализа exp против muatm МС.

Считает по одному ряду на событие: сколько импульсов пришлось на самый «болтливый» модуль,
какова его доля сигнальных по SN-модели, насколько он далёк от облака сигнальных хитов
события, и (для МС) чем эти импульсы были по правде. Пишет parquet + сайдкар провенанса.

Сэмпл фиксирован здесь: четыре рана exp_full и 189 партов muatm (~2x и ~21x по числу
событий относительно исходного сэмпла). Парты muatm выбраны систематически -- каждый
~56-й из списка партов, у которых есть файл вероятностей, без 2 обучающих партов
(`part_13295`, `part_63085`).

    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/build_repeated_hit_features.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from inference.reader.spec import SOURCES                      # noqa: E402

CHECKPOINT = ("261002_0804_da_nu_classifier_exp_full_E1_lambda0.01_DArepair_weights"
              "@best_da_model")
PRED = ROOT / f"inference/nu_classifier/preds/{CHECKPOINT}/exp_full_thr0p8.duckdb"
MC_PRED = ROOT / f"inference/nu_classifier/preds/{CHECKPOINT}/mc_merged_thr0p8.duckdb"
FITS = ROOT / "inference/track_fit_alike_bars/sample/fits.duckdb"
OUT = Path(__file__).with_name("repeated_hit_features")
OUT.mkdir(exist_ok=True)

EXP_PARTS = ["part_s2020_c04_r0031", "part_s2020_c06_r0024",
             "part_s2020_c02_r0055", "part_s2020_c07_r0465"]
MC_PARTS = [
    "part_1002", "part_10029", "part_1004", "part_10046", "part_10049",
    "part_1005", "part_10057", "part_1009", "part_10002", "part_10118",
    "part_10213", "part_10318", "part_10425", "part_10540", "part_10652",
    "part_10769", "part_1086", "part_10950", "part_11062", "part_11157",
    "part_11268", "part_11365", "part_1146", "part_11573", "part_11671",
    "part_11781", "part_11877", "part_11984", "part_12069", "part_12184",
    "part_12288", "part_12383", "part_12479", "part_1258", "part_12674",
    "part_12781", "part_12862", "part_12965", "part_1306", "part_1317",
    "part_13273", "part_13373", "part_13472", "part_13582", "part_13691",
    "part_13779", "part_13875", "part_13960", "part_1407", "part_14168",
    "part_14268", "part_14369", "part_1446", "part_21997", "part_22100",
    "part_22211", "part_22325", "part_22438", "part_22541", "part_22632",
    "part_22737", "part_22845", "part_22949", "part_23077", "part_23177",
    "part_23302", "part_23409", "part_23531", "part_23649", "part_23739",
    "part_23850", "part_23953", "part_24055", "part_24177", "part_24292",
    "part_24398", "part_24502", "part_24611", "part_24722", "part_24827",
    "part_24929", "part_25047", "part_25173", "part_25285", "part_25404",
    "part_25506", "part_25624", "part_25715", "part_25811", "part_25915",
    "part_26028", "part_26123", "part_26236", "part_26351", "part_26468",
    "part_26587", "part_26698", "part_26821", "part_26927", "part_35629",
    "part_35744", "part_35853", "part_35942", "part_36045", "part_36160",
    "part_36268", "part_36374", "part_36459", "part_36576", "part_36687",
    "part_36792", "part_36895", "part_36996", "part_37115", "part_37233",
    "part_37338", "part_37441", "part_37544", "part_37653", "part_37770",
    "part_37887", "part_37987", "part_38102", "part_38212", "part_38341",
    "part_38436", "part_38563", "part_38689", "part_38788", "part_38904",
    "part_39023", "part_39135", "part_39234", "part_39347", "part_39463",
    "part_39565", "part_39662", "part_39771", "part_39896", "part_40007",
    "part_40131", "part_40240", "part_40348", "part_40462", "part_61499",
    "part_61603", "part_61724", "part_61834", "part_61945", "part_62078",
    "part_62199", "part_62314", "part_62419", "part_62520", "part_62625",
    "part_62739", "part_62845", "part_62962", "part_63074", "part_63175",
    "part_63298", "part_63396", "part_63512", "part_63629", "part_63735",
    "part_63841", "part_63968", "part_64082", "part_64181", "part_64299",
    "part_64416", "part_64519", "part_64636", "part_64769", "part_64870",
    "part_65000", "part_65110", "part_65214", "part_65332", "part_65440",
    "part_65554", "part_65670", "part_65792", "part_65905", "part_66007",
    "part_66097", "part_66195", "part_66282", "part_66378",
]
SN_THRESHOLD = 0.8
CH_PER_STRING = 36
N_STRINGS = 8

COLUMNS = (
    "event_fk", "part_key", "local_idx", "n_hits", "n_sn", "n_strings",
    "n_max", "om_max", "n_rep_oms", "n_rep_pulses",
    "top_n", "top_qsum", "top_qmax", "top_tspan", "top_sn_share",
    "top_dist", "top_truth_noise", "truth_noise_share",
    "prime_theta", "prime_phi", "prime_e", "prime_muons",
)


def part_features(part, domain, exp):
    """Ряды по событиям одного парта, без score и фита (они присоединятся ниже)."""
    source = SOURCES["exp_full"] if exp else SOURCES["mc_merged"]
    group = "exp_full" if exp else "muatm_2020"
    with h5py.File(source.h5, "r") as f, h5py.File(source.probs, "r") as p:
        base = f"{group}/raw"
        starts = f[f"{base}/ev_starts"][part]["data"][:].astype(np.int64)
        ch = f[f"{base}/channels"][part]["data"][:].astype(np.int64)
        data = f[f"{base}/data"][part]["data"][:]
        probs = p[f"{group}/probs"][part]["data"][:]
        labels = f[f"{base}/labels"][part]["data"][:] if not exp else None
        prime = f[f"{group}/prime_prty/{part}/data"][:] if not exp else None
        event_fk = _event_fks(part, exp)

    n_events = len(starts) - 1
    q, t = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    x, y, z = (data[:, k].astype(np.float64) for k in (2, 3, 4))
    event_of = np.searchsorted(starts, np.arange(len(ch)), side="right") - 1
    is_sig = probs > SN_THRESHOLD
    is_noise = labels == 0 if labels is not None else None

    key = event_of * (288 if exp else 2048) + ch
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    ev_key, ch_key = uniq // (288 if exp else 2048), uniq % (288 if exp else 2048)
    first_hit = np.unique(inverse, return_index=True)[1]
    x_key, y_key, z_key = x[first_hit], y[first_hit], z[first_hit]
    qsum_key = np.bincount(inverse, weights=q, minlength=len(uniq))
    n_sig_key = np.bincount(inverse[is_sig], minlength=len(uniq))

    # событие: счётчики
    n_hits = np.diff(starts)
    n_sn = np.bincount(event_of[is_sig], minlength=n_events)
    string_key = event_of * N_STRINGS + ((ch // CH_PER_STRING) % N_STRINGS)
    n_strings = np.bincount(np.unique(string_key) // N_STRINGS, minlength=n_events)

    # самый «болтливый» модуль события
    order = np.lexsort((-counts, ev_key))
    first_ev = np.unique(ev_key[order], return_index=True)[1]
    best = order[first_ev]
    events = ev_key[order][first_ev]
    n_max = np.zeros(n_events, dtype=np.int64)
    n_max[events] = counts[best]
    rep_oms = np.zeros(n_events, dtype=np.int64)
    rep_oms[events] = np.bincount(ev_key[counts >= 2], minlength=n_events)[events]
    rep_pulses = np.zeros(n_events, dtype=np.int64)
    rep_pulses[events] = np.bincount(ev_key[counts >= 2], weights=counts[counts >= 2],
                                     minlength=n_events)[events].astype(np.int64)

    # хвост по событиям с повторами: заряд/время/правда/distance до облака
    top = {k: np.full(n_events, np.nan) for k in
           ("top_n", "top_qsum", "top_qmax", "top_tspan", "top_sn_share",
            "top_dist", "top_truth_noise")}
    top["om_max"] = np.full(n_events, -1, dtype=np.int64)
    top["top_n"][events] = counts[best]
    top["top_qsum"][events] = qsum_key[best]
    top["top_sn_share"][events] = n_sig_key[best] / counts[best]
    top["om_max"][events] = ch_key[best]
    best_of_event = np.full(n_events, -1, dtype=np.int64)
    best_of_event[events] = best
    loop_events = events[counts[best] >= 2]
    for k in loop_events:
        lo, hi = starts[k], starts[k + 1]
        sel = ch[lo:hi] == top["om_max"][k]
        top["top_qmax"][k] = q[lo:hi][sel].max()
        top["top_tspan"][k] = t[lo:hi][sel].max() - t[lo:hi][sel].min()
        if is_noise is not None:
            top["top_truth_noise"][k] = is_noise[lo:hi][sel].mean()
        cloud = is_sig[lo:hi] & ~sel
        if cloud.any():
            j = best_of_event[k]
            d = np.hypot(np.hypot(x[lo:hi] - x_key[j], y[lo:hi] - y_key[j]),
                         z[lo:hi] - z_key[j])
            top["top_dist"][k] = d[cloud].min()

    frame = pd.DataFrame(top)
    frame["event_fk"] = event_fk
    frame["part_key"] = part
    frame["local_idx"] = np.arange(n_events)
    frame["n_hits"] = n_hits
    frame["n_sn"] = n_sn
    frame["n_strings"] = n_strings
    frame["n_max"] = n_max
    frame["n_rep_oms"] = rep_oms
    frame["n_rep_pulses"] = rep_pulses
    if is_noise is not None:
        frame["truth_noise_share"] = np.bincount(
            event_of[is_noise], minlength=n_events) / np.maximum(n_hits, 1)
        frame["prime_theta"] = prime[:, 0]
        frame["prime_phi"] = prime[:, 1]
        frame["prime_e"] = prime[:, 2]
        frame["prime_muons"] = prime[:, 4]
    return frame


def _event_fks(part, exp):
    """event_fk каждого события парта, из каталога."""
    source = "exp_full" if exp else "mc_merged"
    group = "exp_full" if exp else "muatm_2020"
    with duckdb.connect(str(ROOT / "data_manager/catalog_v2.duckdb"),
                        read_only=True) as con:
        rows = con.execute("""
            SELECT l.local_idx, l.event_fk FROM h5_locations l
            JOIN events e ON e.id = l.event_fk
            WHERE e.source = ? AND e.data_class = ? AND l.part_key = ?
            ORDER BY l.local_idx
        """, [source, group, part]).fetch_df()
    if not rows.local_idx.is_unique:
        raise RuntimeError(f"{part}: local_idx повторился")
    if len(rows) and not np.array_equal(rows.local_idx.to_numpy(), np.arange(len(rows))):
        raise RuntimeError(f"{part}: local_idx не сплошной -- выравнивание кадра сломается")
    return rows.event_fk.to_numpy()


def attach_observables(frame, exp):
    """score и колонки фита к событию, по event_fk."""
    pred = PRED if exp else MC_PRED
    con = duckdb.connect()
    con.register("sample", frame[["event_fk"]])
    con.execute(f"ATTACH '{pred}' AS pred (READ_ONLY)")
    con.execute(f"ATTACH '{FITS}' AS fit (READ_ONLY)")
    scores = con.execute("""
        SELECT p.event_fk, p.score, p.n_sn_hits, p.n_sn_strings
        FROM pred.predictions p JOIN sample s USING (event_fk)
    """).df()
    fits = con.execute("""
        SELECT f.event_fk, f.status, f.theta_rec, f.phi_rec, f.n_hits,
               f.cov_matrix_status
        FROM fit.fits f JOIN sample s USING (event_fk)
    """).df().rename(columns={"status": "fit_status", "theta_rec": "fit_theta_rec",
                              "phi_rec": "fit_phi_rec", "n_hits": "fit_n_hits",
                              "cov_matrix_status": "fit_cov_matrix_status"})
    con.close()
    out = frame.merge(scores, on="event_fk", how="left").merge(fits, on="event_fk",
                                                               how="left")
    if exp:
        out["truth_noise_share"] = np.nan
        for c in ("prime_theta", "prime_phi", "prime_e", "prime_muons"):
            out[c] = np.nan
    return out


def main():
    started = time.time()
    frames = []
    for part in EXP_PARTS:
        frames.append(attach_observables(part_features(part, "exp", exp=True), exp=True))
        print(".", end="", flush=True)
    for part in MC_PARTS:
        frames.append(attach_observables(part_features(part, "muatm", exp=False), exp=False))
        print(".", end="", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    frame["domain"] = np.where(frame.part_key.str.startswith("part_s"), "exp", "muatm")

    # санитарная сверка: n_sn обязан совпасть с базой предсказаний (где она есть)
    both = frame.dropna(subset=["n_sn_hits"])
    if len(both):
        mismatch = int((both.n_sn.to_numpy() != both.n_sn_hits.to_numpy()).sum())
        print(f"\nсверка n_sn с базой: {len(both):,} событий, расхождений {mismatch}")
    ok = frame[frame.fit_status.eq("ok")] if "fit_status" in frame else frame
    print(f"событий: {len(frame):,} (exp {frame.domain.eq('exp').sum():,}, "
          f"muatm {frame.domain.eq('muatm').sum():,}); "
          f"с фитом ok: {len(ok):,}")

    path = OUT / "repeated_hit_features.parquet"
    frame.to_parquet(path, index=False)
    meta = {"script": str(Path(__file__).relative_to(ROOT)),
            "argv": sys.argv, "checkpoint": CHECKPOINT,
            "exp_parts": EXP_PARTS, "mc_parts": MC_PARTS,
            "columns": list(frame.columns), "events": len(frame),
            "events_exp": int(frame.domain.eq("exp").sum()),
            "events_muatm": int(frame.domain.eq("muatm").sum()),
            "sn_threshold": SN_THRESHOLD,
            "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
            "runtime_s": round(time.time() - started, 1),
            "written": datetime.now().isoformat(timespec="seconds")}
    (OUT / "repeated_hit_features.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"записано {path} ({path.stat().st_size / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
