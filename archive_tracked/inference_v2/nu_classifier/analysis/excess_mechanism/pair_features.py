"""Попарные признаки: насколько хиты события согласованы с одним световым фронтом.

Зачем. После добавления последовательностных признаков объяснено ×1.95 из ×2.86; остаток ×1.47
однороден по корзинам, но РАСТЁТ с числом хитов (2.8 при 8-9 до 7.5 при 30+). Больше хитов --
больше пар, то есть больше информации о взаимной согласованности. Значит остаток может сидеть
в попарных отношениях, которых среди 64 признаков нет: все они описывают событие целиком или
его последовательность, но не связи хит-хит.

Физика. Для пары хитов от одного светового фронта отношение |dt| / (dr / v_свет) равно единице:
свет прошёл ровно расстояние между модулями. Для рассеянного света оно больше, для причинно
несвязанных пар -- меньше единицы (сигнал пришёл быстрее, чем свет мог пройти), что физически
невозможно и указывает на два независимых источника.
"""
from __future__ import annotations
import argparse, logging, sys, time
from multiprocessing import Pool
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
H5 = {"mc": ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5",
      "exp": ROOT / "data_manager/data/h5datasets/exp_full.h5"}
PROBS = {"mc": ROOT / ("data_manager/data/h5datasets/baikal_mc_merged_probs_"
                       "k_nsol_labelneq0_da_hs128_k0p0001.h5"),
         "exp": ROOT / ("data_manager/data/h5datasets/exp_full_probs_"
                        "k_nsol_labelneq0_da_hs128_k0p0001.h5")}
SN_THR = 0.8
V_LIGHT = 0.299792458 / 1.37      # м/нс, скорость света в воде
MAX_PAIRS = 20000
COLS = ["pair_lc_med", "pair_lc_iqr", "pair_lc_frac_light", "pair_lc_frac_acausal",
        "pair_lc_frac_slow", "pair_dtdr_corr", "pair_lc_med_bright", "pair_lc_med_far",
        "pair_frac_same_str", "pair_dr_med"]
log = logging.getLogger("pair")


def pair_features(h: np.ndarray, c: np.ndarray) -> tuple:
    n = len(h)
    t = h[:, 1].astype(np.float64)
    pos = h[:, 2:5].astype(np.float64)
    q = np.clip(h[:, 0].astype(np.float64), 0, 100)
    i, j = np.triu_indices(n, 1)
    if len(i) > MAX_PAIRS:                      # у длинных событий пары прореживаются
        sel = np.linspace(0, len(i) - 1, MAX_PAIRS).astype(int)
        i, j = i[sel], j[sel]
    dr = np.linalg.norm(pos[i] - pos[j], axis=1)
    dt = np.abs(t[i] - t[j])
    ok = dr > 1e-6
    if ok.sum() < 3:
        return (np.nan,) * len(COLS)
    dr, dt, i, j = dr[ok], dt[ok], i[ok], j[ok]
    # 1 = хиты ровно на световом фронте; <1 физически невозможно; >1 -- рассеяние или два источника
    lc = dt / (dr / V_LIGHT)
    bright = (q[i] > np.median(q)) & (q[j] > np.median(q))
    far = dr > np.median(dr)
    same_str = (c[i] // 36) == (c[j] // 36)
    corr = float(np.corrcoef(dt, dr)[0, 1]) if np.ptp(dt) > 0 and np.ptp(dr) > 0 else np.nan
    return (
        float(np.median(lc)),
        float(np.subtract(*np.percentile(lc, [75, 25]))),
        float(np.mean((lc > 0.8) & (lc < 1.2))),
        float(np.mean(lc < 1.0)),
        float(np.mean(lc > 3.0)),
        corr,
        float(np.median(lc[bright])) if bright.sum() >= 3 else np.nan,
        float(np.median(lc[far])) if far.sum() >= 3 else np.nan,
        float(np.mean(same_str)),
        float(np.median(dr)),
    )


def _one_part(job):
    source, h5g, part, idx, fks = job
    rows, keep = [], []
    with h5py.File(H5[source], "r") as f, h5py.File(PROBS[source], "r") as pf:
        ev = f[f"{h5g}/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = f[f"{h5g}/raw/data/{part}/data"][:].astype(np.float32)
        ch = f[f"{h5g}/raw/channels/{part}/data"][:].astype(np.int32)
        pr = pf[f"{h5g}/probs/{part}/data"][:].astype(np.float32)
    for k in np.argsort(idx):
        s, e = int(ev[idx[k]]), int(ev[idx[k] + 1])
        m = pr[s:e] > SN_THR
        if m.sum() < 8:
            continue
        rows.append(pair_features(data[s:e][m], ch[s:e][m])); keep.append(int(fks[k]))
    return source, keep, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    con = duckdb.connect(str(HERE / "pair.duckdb"))
    con.execute(f"CREATE TABLE IF NOT EXISTS pair (source VARCHAR, event_fk BIGINT, "
                f"{', '.join(f'{c} DOUBLE' for c in COLS)}, PRIMARY KEY (source, event_fk))")
    done = {(s, k) for s, k in con.execute("SELECT source,event_fk FROM pair").fetchall()}
    src = duckdb.connect(str(HERE / "features.duckdb"), read_only=True)
    rows = src.execute("SELECT source, event_fk, part_key, group_id FROM features").fetchall()
    src.close()
    cat = duckdb.connect(str(CATALOG), read_only=True)
    cat.register("_need", pd.DataFrame({"event_fk": [int(r[1]) for r in rows]}))
    loc = dict(cat.execute("SELECT l.event_fk, l.local_idx FROM h5_locations l "
                           "JOIN _need n USING (event_fk)").fetchall())
    cat.close()
    GRP = {1: "nuatm_2020", 2: "nuatm_2020", 3: "nue2_2020", 4: "nue2_2020",
           5: "muatm_2020", 6: "muatm_2020", 7: "exp_full", 8: "exp_full"}
    jobs: dict = {}
    for s_, fk, pk, g in rows:
        if (s_, fk) in done:
            continue
        jobs.setdefault((s_, GRP[g], pk), ([], []))
        jobs[(s_, GRP[g], pk)][0].append(loc[fk]); jobs[(s_, GRP[g], pk)][1].append(fk)
    tasks = [(s, g, p, np.array(v[0]), np.array(v[1])) for (s, g, p), v in jobs.items()]
    tasks.sort(key=lambda t: -len(t[3]))
    log.info(f"{len(tasks):,} партов, {sum(len(t[3]) for t in tasks):,} событий")
    t0, n = time.time(), 0
    with Pool(a.workers) as pool:
        for i, (s_, keep, rr) in enumerate(pool.imap_unordered(_one_part, tasks, chunksize=1), 1):
            if rr:
                df = pd.DataFrame(rr, columns=COLS)
                df.insert(0, "event_fk", keep); df.insert(0, "source", s_)
                con.register("_t", df); con.execute("INSERT OR IGNORE INTO pair SELECT * FROM _t")
                con.unregister("_t"); n += len(rr)
            if i % 300 == 0 or i == len(tasks):
                log.info(f"  [{i}/{len(tasks)}] {n:,} событий, {time.time()-t0:.0f} c")
    log.info(f"готово: {con.execute('SELECT count(*) FROM pair').fetchone()[0]:,} строк")


if __name__ == "__main__":
    main()
