"""Абляция входов nu-классификатора: что именно в exp-событиях сеть принимает вдвое чаще.

Постановка. Избыток (2.86) не объясняется ни одним из 52 агрегатов: суррогат, обученный на MC
и применённый к exp при подогнанном пороге, даёт 1.26-1.35. Значит сеть отвечает на что-то
похитовое. Единственный способ узнать, на что — менять вход по одному свойству и смотреть,
какая правка убирает избыток.

Чтение. Только целыми партами и только один раз: raw/data нарезан чанками (43215, 1), поэтому
чтение отдельного события распаковывает пять чанков. В MC принятые события размазаны по 1.4 на
парт (5373 парта на 7751 событие), в exp один ран даёт 307 принятых из 278 026 -- отсюда
асимметрия в числе читаемых партов.

Выборка. Все принятые события берутся целиком (вес 1), остальные прореживаются 1 к KEEP
(вес KEEP). Взвешенная доля принятых при этом остаётся несмещённой, включая события, которые
перешли порог после правки.
"""
from __future__ import annotations

import argparse, logging, sys, time
from pathlib import Path

import duckdb, h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores   # noqa: E402

CKPT = ROOT / ("experiments/numu/260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_"
               "FIXED_sn256/best_da_model.pth")
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = ROOT / "inference_v2/nu_classifier/preds" / MODEL
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
SRC = {"mc":  (ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5", "muatm_2020",
               ROOT / ("data_manager/data/h5datasets/baikal_mc_merged_probs_"
                       "k_nsol_labelneq0_da_hs128_k0p0001.h5"), "mc_merged_thr0p8.duckdb"),
       "exp": (ROOT / "data_manager/data/h5datasets/exp_full.h5", "exp_full",
               ROOT / ("data_manager/data/h5datasets/exp_full_probs_"
                       "k_nsol_labelneq0_da_hs128_k0p0001.h5"), "exp_full_thr0p8.duckdb")}
SN_THR, KEEP = 0.8, 8
SCORE_THR = 0.8   # переопределяется --score-thr
QUALITY = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3"
log = logging.getLogger("ablate")


def pick_parts(domain: str, n_parts: int) -> list[str]:
    h5, grp, probs, dbf = SRC[domain]
    con = duckdb.connect(str(PREDS / dbf), read_only=True)
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    where = ("NOT s.used_for_labels AND s.data_class='muatm_2020'" if domain == "mc"
             else "NOT s.excluded")
    # Парты берутся СЛУЧАЙНО, а не по числу принятых событий. Отбор по `ORDER BY hi DESC`
    # выглядит экономным -- меньше читать ради тех же принятых -- но смещает знаменатель:
    # в MC принятые размазаны по 1.4 на парт, и верхние 300 из 5373 дали долю 0.1127% вместо
    # истинных 0.0335%, то есть базовое отношение 0.79 вместо 2.86. На таком базисе абляции
    # бессмысленны. Записано здесь, потому что первый прогон был сделан именно так.
    rows = con.execute(f"""
        SELECT l.part_key
        FROM predictions p JOIN splits s USING(event_fk)
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk
        WHERE {QUALITY} AND {where}
        GROUP BY 1 ORDER BY hash(l.part_key) LIMIT {n_parts}""").fetchall()
    con.close()
    return [r[0] for r in rows]


def load_domain(domain: str, parts: list[str]):
    """Хиты, каналы, сохранённый скор и вес для выборки событий из указанных партов."""
    h5, grp, probs, dbf = SRC[domain]
    where = ("NOT s.used_for_labels AND s.data_class='muatm_2020'" if domain == "mc"
             else "NOT s.excluded")
    con = duckdb.connect(str(PREDS / dbf), read_only=True)
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    con.execute("CREATE TEMP TABLE _p (part_key VARCHAR)")
    con.executemany("INSERT INTO _p VALUES (?)", [(p,) for p in parts])
    rows = con.execute(f"""
        SELECT l.part_key, l.local_idx, p.score
        FROM predictions p JOIN splits s USING(event_fk)
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk
        JOIN _p ON _p.part_key = l.part_key
        WHERE {QUALITY} AND {where}""").fetchall()
    con.close()

    by_part: dict[str, list] = {}
    rng = np.random.default_rng(20260825)
    for pk, idx, sc in rows:
        keep = sc > SCORE_THR or rng.integers(KEEP) == 0
        if keep:
            by_part.setdefault(pk, []).append((idx, sc, 1.0 if sc > SCORE_THR else float(KEEP)))
    log.info(f"  {domain}: {len(rows):,} quality-событий в {len(parts)} партах, "
             f"после прореживания {sum(len(v) for v in by_part.values()):,}")

    hits, chans, stored, weight, hit_probs = [], [], [], [], []
    t0 = time.time()
    with h5py.File(h5, "r") as f, h5py.File(probs, "r") as pf:
        for i, (pk, items) in enumerate(by_part.items(), 1):
            ev = f[f"{grp}/raw/ev_starts/{pk}/data"][:].astype(np.int64)
            data = f[f"{grp}/raw/data/{pk}/data"][:].astype(np.float32)
            ch = f[f"{grp}/raw/channels/{pk}/data"][:].astype(np.int32)
            pr = pf[f"{grp}/probs/{pk}/data"][:].astype(np.float32)
            for idx, sc, w in items:
                s, e = int(ev[idx]), int(ev[idx + 1])
                m = pr[s:e] > SN_THR
                if m.sum() < 5:
                    continue
                hits.append(data[s:e][m]); chans.append(ch[s:e][m])
                hit_probs.append(pr[s:e][m]); stored.append(sc); weight.append(w)
            if i % 50 == 0 or i == len(by_part):
                log.info(f"    прочитано партов {i}/{len(by_part)}  ({time.time()-t0:.0f} c)")
    return hits, chans, np.array(stored), np.array(weight), hit_probs


# ---------------------------------------------------------------- правки входа
def q_map(hits, src_q, dst_q):
    """Похитовая замена заряда: квантиль в своём распределении -> тот же квантиль в чужом.

    Сетка квантилей строится ОДИН раз. В первой версии `np.linspace(0, 1, len(dst_q))`
    вызывался внутри цикла по событиям, то есть массив на 3.9 млн элементов создавался
    260 тысяч раз; прогон завис на пятнадцать минут именно здесь.
    """
    grid = np.linspace(0.0, 1.0, len(dst_q))
    n_src = max(len(src_q), 1)
    out = []
    for h in hits:
        g = h.copy()
        r = np.searchsorted(src_q, h[:, 0]) / n_src
        g[:, 0] = np.interp(r, grid, dst_q)
        out.append(g)
    return out


def _snap(hits, chans, geom):
    """Координаты каждого хита заменяются номинальным положением его канала в MC."""
    if not geom:
        return hits
    out = []
    for h, c in zip(hits, chans):
        g = h.copy()
        for i, ch in enumerate(c):
            # в exp нумерация внутрикластерная; кластер неизвестен на этом уровне, поэтому
            # пробуем все семь смещений и берём ближайшее -- расхождение геометрий 0.25 м,
            # а расстояние между кластерами сотни метров, так что выбор однозначен
            best = None
            for k in range(7):
                q = geom.get(int(ch) + 288 * k)
                if q is None:
                    continue
                d = float(np.sum((q - h[i, 2:5]) ** 2))
                if best is None or d < best[0]:
                    best = (d, q)
            if best is not None:
                g[i, 2:5] = best[1]
        out.append(g)
    return out


def _tighten(hits, probs, thr):
    """Ужесточённый порог sig-noise: убираем пограничные хиты."""
    out = []
    for h, p in zip(hits, probs):
        m = p > thr
        out.append(h[m] if m.sum() >= 5 else h)
    return out


ABLATIONS = {
    "как есть":                    lambda h, c, rng, **k: h,
    "заряды /1.15":                lambda h, c, rng, **k: [np.c_[g[:, 0] / 1.15, g[:, 1:]] for g in h],
    "заряды -> распределение MC":  lambda h, c, rng, **k: q_map(h, k["src_q"], k["dst_q"]),
    "время +N(0,5нс)":             lambda h, c, rng, **k: [np.c_[g[:, 0], g[:, 1] + rng.normal(0, 5, len(g)), g[:, 2:]] for g in h],
    "время округлено до 10нс":     lambda h, c, rng, **k: [np.c_[g[:, 0], np.round(g[:, 1] / 10) * 10, g[:, 2:]] for g in h],
    "порядок хитов перемешан":     lambda h, c, rng, **k: [g[rng.permutation(len(g))] for g in h],
    # np.unique(..., return_index=True) отдаёт индексы в порядке ОТСОРТИРОВАННЫХ каналов,
    # поэтому без np.sort событие переупорядочивается по номеру модуля вместо времени. В
    # первом прогоне так и было, и строка мерила переупорядочивание, а не удаление повторов.
    "только первый хит модуля":    lambda h, c, rng, **k: [g[np.sort(np.unique(cc, return_index=True)[1])] for g, cc in zip(h, c)],
    "порядок по номеру модуля":    lambda h, c, rng, **k: [g[np.argsort(cc, kind="stable")] for g, cc in zip(h, c)],
    "координаты -> номинал MC":   lambda h, c, rng, **k: _snap(h, c, k["geom"]),
    # Пороги подобраны по РЕАЛЬНОМУ распределению: у отобранных хитов вероятности лежат
    # между 0.80 и 0.94, медиана 0.883, выше 0.99 их ноль. Первая версия резала по 0.95 и
    # 0.99, из-за чего почти каждое событие откатывалось к исходному по правилу "меньше
    # пяти хитов -- не трогаем", и обе строки были пустышками, неотличимыми от нуля.
    "хиты sig-noise > 0.85":      lambda h, c, rng, **k: _tighten(h, k["probs"], 0.85),
    "хиты sig-noise > 0.90":      lambda h, c, rng, **k: _tighten(h, k["probs"], 0.90),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mc-parts", type=int, default=300)
    ap.add_argument("--exp-parts", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--score-thr", type=float, default=0.8,
                    help="порог приёмки; при 0.01 статистики в 70 раз больше")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    global SCORE_THR
    SCORE_THR = a.score_thr
    log.info(f"порог приёмки: {SCORE_THR}")

    t0 = time.time()
    log.info(f"загрузка модели на {a.device}")
    model, norm, _ = load_model(str(CKPT), device=a.device)
    log.info(f"  готово, {time.time()-t0:.0f} c; cuda доступна: {torch.cuda.is_available()}")

    data = {}
    for dom, n in (("exp", a.exp_parts), ("mc", a.mc_parts)):
        log.info(f"выбор партов: {dom}")
        parts = pick_parts(dom, n)
        data[dom] = load_domain(dom, parts)
        log.info(f"  {dom}: {len(data[dom][0]):,} событий загружено "
                 f"({time.time()-t0:.0f} c от старта)")

    # 10 000 квантилей описывают распределение заряда с запасом; хранить 3.9 млн точек и
    # интерполировать по ним на каждом событии незачем
    # номинальная геометрия: медианное положение каждого канала В СИМУЛЯЦИИ.
    # MC использует одну геометрию, exp пересчитывает её по ранам; расхождение между ними
    # 0.25 м по медиане и до 1.97 м, дрейф между exp-ранами 0.08 м. Нумерация каналов в MC
    # глобальная (cluster*288 + c), в exp внутрикластерная -- отсюда сдвиг при сборке.
    GEOM = {}
    mc_pos: dict[int, np.ndarray] = {}
    for h, c in zip(data["mc"][0], data["mc"][1]):
        for pos, ch in zip(h[:, 2:5], c):
            mc_pos.setdefault(int(ch), []).append(pos)
    mc_med = {ch: np.median(np.stack(v), axis=0) for ch, v in mc_pos.items() if len(v) >= 20}
    GEOM["mc"] = mc_med
    GEOM["exp"] = mc_med           # exp тоже приводится к номинальной геометрии MC
    log.info(f"номинальная геометрия: {len(mc_med)} каналов")

    pool_q = {}
    for d in ("mc", "exp"):
        allq = np.sort(np.concatenate([h[:, 0] for h in data[d][0]]))
        pool_q[d] = np.quantile(allq, np.linspace(0, 1, 10_001))
    log.info(f"похитовых зарядов: mc {len(pool_q['mc']):,}, exp {len(pool_q['exp']):,}")

    rng = np.random.default_rng(0)
    log.info("\n" + "=" * 78)
    log.info(f"{'правка':30s} {'MC':>10s} {'exp':>11s} {'отн.':>9s}"
             f"   |  {'exp при рабочей точке MC':>10s} {'отн.':>9s}")
    log.info("=" * 78)
    def wq(v, w, q):
        o = np.argsort(v)
        return float(np.interp(q, np.cumsum(w[o]) / w.sum(), v[o]))

    base_mc_rate = None
    for name, fn in ABLATIONS.items():
        sc, w, touched = {}, {}, {}
        for dom in ("mc", "exp"):
            hits, chans, stored, wd, prb = data[dom]
            kw = dict(src_q=pool_q[dom], dst_q=pool_q["mc"], probs=prb,
                      geom=GEOM.get(dom))
            mod = fn(hits, chans, np.random.default_rng(1), **kw)
            n_before = sum(len(x) for x in hits)
            n_after = sum(len(x) for x in mod)
            changed = sum(1 for x, y in zip(hits, mod)
                          if len(x) != len(y) or not np.array_equal(x, y))
            touched[dom] = (100.0 * changed / len(hits), 100.0 * n_after / n_before)
            sc[dom] = predict_scores(model, mod, norm, batch_size=a.batch_size,
                                     device=a.device, with_tqdm=False)
            w[dom] = wd
            if name == "как есть":
                d = np.abs(sc[dom] - stored)
                log.info(f"    сверка со сохранённым скором ({dom}): max|delta| {d.max():.2e}, "
                         f"медиана {np.median(d):.2e}")
        raw_mc = float(w["mc"][sc["mc"] > SCORE_THR].sum() / w["mc"].sum())
        raw_ex = float(w["exp"][sc["exp"] > SCORE_THR].sum() / w["exp"].sum())
        if base_mc_rate is None:
            base_mc_rate = raw_mc
        # рабочая точка выравнивается: порог подбирается по MC под ИСХОДНУЮ долю приёмки,
        # и тем же порогом отбирается exp. Без этого правка, сдвигающая общий масштаб
        # (перемешивание порядка поднимает приёмку впятеро), подделывает отношение.
        thr = wq(sc["mc"], w["mc"], 1.0 - base_mc_rate)
        m_rate = float(w["mc"][sc["mc"] >= thr].sum() / w["mc"].sum())
        e_rate = float(w["exp"][sc["exp"] >= thr].sum() / w["exp"].sum())
        ev_pct, hit_pct = touched["exp"]
        log.info(f"{name:30s} {100*raw_mc:9.3f}% {100*raw_ex:10.3f}% {raw_ex/raw_mc:8.2f}"
                 f"   |  {100*e_rate:9.3f}% {e_rate/m_rate:9.2f}"
                 f"   |  событий изменено {ev_pct:5.1f}%, хитов осталось {hit_pct:5.1f}%")
    log.info("=" * 78)
    log.info(f"всего {time.time()-t0:.0f} c")


if __name__ == "__main__":
    main()
