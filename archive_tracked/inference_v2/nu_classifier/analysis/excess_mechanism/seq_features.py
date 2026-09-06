"""Последовательностные признаки — то, чего не было в исходных 52.

Абляции показали: разрушение временнóго порядка хитов снимает 48% избытка, тогда как правки
значений (заряд, время, координаты) не дают ничего. Значит информация сидит в
последовательности, а все 52 исходных признака — агрегаты, порядок стирающие целиком. Отсюда
и провал суррогата: ×1.26–1.35 против ×2.86 у сети.

Здесь считаются величины, определённые ИМЕННО на упорядоченной по времени последовательности
хитов. Пишется отдельная таблица seq, ключ (source, event_fk), чтобы не пересобирать features.
"""
from __future__ import annotations
import argparse, logging, sys, time
from multiprocessing import Pool
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = ROOT / "inference_v2/nu_classifier/preds" / MODEL
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
H5 = {"mc": ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5",
      "exp": ROOT / "data_manager/data/h5datasets/exp_full.h5"}
PROBS = {"mc": ROOT / ("data_manager/data/h5datasets/baikal_mc_merged_probs_"
                       "k_nsol_labelneq0_da_hs128_k0p0001.h5"),
         "exp": ROOT / ("data_manager/data/h5datasets/exp_full_probs_"
                        "k_nsol_labelneq0_da_hs128_k0p0001.h5")}
SN_THR = 0.8
COLS = ["seq_depth_corr", "seq_depth_absc", "seq_reversals", "seq_longest_run",
        "seq_gap_med", "seq_gap_max", "seq_gap_cv", "seq_tight_frac",
        "seq_q_corr", "seq_r_corr", "seq_step_med", "seq_backstep_frac"]
log = logging.getLogger("seq")


def seq_features(h: np.ndarray) -> tuple:
    """Величины на упорядоченной по времени последовательности хитов."""
    n = len(h)
    t, z = h[:, 1].astype(np.float64), h[:, 4].astype(np.float64)
    q = np.clip(h[:, 0].astype(np.float64), 0, 100)
    r = np.hypot(h[:, 2].astype(np.float64), h[:, 3].astype(np.float64))
    idx = np.arange(n, dtype=np.float64)

    def corr(a, b):
        if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    dz = np.diff(z)
    sign = np.sign(dz)
    nz = sign[sign != 0]
    reversals = float(np.mean(nz[1:] != nz[:-1])) if len(nz) > 1 else np.nan
    # самая длинная монотонная по глубине серия, в долях длины события
    longest, cur = 1, 1
    for k in range(1, len(nz)):
        cur = cur + 1 if nz[k] == nz[k - 1] else 1
        longest = max(longest, cur)
    gaps = np.diff(t)
    gm = float(np.median(gaps)) if len(gaps) else np.nan
    return (
        corr(idx, z),                                   # ход глубины вдоль последовательности
        abs(corr(idx, z)) if np.isfinite(corr(idx, z)) else np.nan,
        reversals,                                      # доля разворотов по глубине
        longest / n,
        gm,
        float(gaps.max()) if len(gaps) else np.nan,
        float(gaps.std() / gm) if len(gaps) and gm and gm > 0 else np.nan,
        float(np.mean(gaps < 5.0)) if len(gaps) else np.nan,
        corr(idx, q),                                   # ход заряда вдоль последовательности
        corr(idx, r),                                   # ход радиуса
        float(np.median(np.abs(dz))) if len(dz) else np.nan,
        float(np.mean(dz < 0)) if len(dz) else np.nan,  # доля шагов «назад» по глубине
    )


def _one_part(job):
    source, h5g, part, idx, fks = job
    rows, keep = [], []
    with h5py.File(H5[source], "r") as f, h5py.File(PROBS[source], "r") as pf:
        ev = f[f"{h5g}/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = f[f"{h5g}/raw/data/{part}/data"][:].astype(np.float32)
        pr = pf[f"{h5g}/probs/{part}/data"][:].astype(np.float32)
    for j in np.argsort(idx):
        s, e = int(ev[idx[j]]), int(ev[idx[j] + 1])
        m = pr[s:e] > SN_THR
        if m.sum() < 8:
            continue
        rows.append(seq_features(data[s:e][m])); keep.append(int(fks[j]))
    return source, keep, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="seq.duckdb")  # отдельный файл: features.duckdb
                    # может быть занят чужим процессом (jupyter держит блокировку)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    con = duckdb.connect(str(HERE / a.out))
    con.execute(f"CREATE TABLE IF NOT EXISTS seq (source VARCHAR, event_fk BIGINT, "
                f"{', '.join(f'{c} DOUBLE' for c in COLS)}, PRIMARY KEY (source, event_fk))")
    done = {(s, k) for s, k in con.execute("SELECT source,event_fk FROM seq").fetchall()}
    log.info(f"уже посчитано: {len(done):,}")
    src_db = duckdb.connect(str(HERE / "features.duckdb"), read_only=True)
    rows = src_db.execute("SELECT source, event_fk, part_key, group_id FROM features").fetchall()
    src_db.close()
    # local_idx берём ТОЧЕЧНО по нужным ключам. Первая версия тянула
    # "SELECT event_fk, local_idx FROM h5_locations" целиком -- сотни миллионов строк в
    # питоновский словарь, и процесс висел, не дойдя до пула.
    cat = duckdb.connect(str(CATALOG), read_only=True)
    # регистрация датафрейма, а не executemany: 628 тысяч отдельных INSERT в duckdb
    # занимают минуты, и процесс висел, не дойдя до пула
    need = pd.DataFrame({"event_fk": [int(r[1]) for r in rows]})
    cat.register("_need", need)
    loc = dict(cat.execute("SELECT l.event_fk, l.local_idx FROM h5_locations l "
                           "JOIN _need n USING (event_fk)").fetchall())
    cat.close()
    log.info(f"local_idx получен для {len(loc):,} событий")
    GRP = {1: "nuatm_2020", 2: "nuatm_2020", 3: "nue2_2020", 4: "nue2_2020",
           5: "muatm_2020", 6: "muatm_2020", 7: "exp_full", 8: "exp_full"}
    jobs: dict = {}
    for src, fk, pk, g in rows:
        if (src, fk) in done:
            continue
        jobs.setdefault((src, GRP[g], pk), ([], []))
        jobs[(src, GRP[g], pk)][0].append(loc[fk]); jobs[(src, GRP[g], pk)][1].append(fk)
    tasks = [(s, g, p, np.array(v[0]), np.array(v[1])) for (s, g, p), v in jobs.items()]
    tasks.sort(key=lambda t: -len(t[3]))
    log.info(f"{len(tasks):,} партов, {sum(len(t[3]) for t in tasks):,} событий")
    t0, n = time.time(), 0
    with Pool(a.workers) as pool:
        for i, (src, keep, rr) in enumerate(pool.imap_unordered(_one_part, tasks, chunksize=1), 1):
            if rr:
                df = pd.DataFrame(rr, columns=COLS)
                df.insert(0, "event_fk", keep); df.insert(0, "source", src)
                con.register("_t", df); con.execute("INSERT OR IGNORE INTO seq SELECT * FROM _t")
                con.unregister("_t"); n += len(rr)
            if i % 200 == 0 or i == len(tasks):
                log.info(f"  [{i}/{len(tasks)}] {n:,} событий, {time.time()-t0:.0f} c")
    log.info(f"готово: {con.execute('SELECT count(*) FROM seq').fetchone()[0]:,} строк")


if __name__ == "__main__":
    main()
