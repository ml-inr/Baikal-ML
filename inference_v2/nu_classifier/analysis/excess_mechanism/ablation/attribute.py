"""Интегральные градиенты: на что именно смотрит nu-классификатор.

Абляции показали, чего в избытке НЕТ: ни заряда, ни времени, ни координат, ни повторных
хитов, ни отбора sig-noise. Ни одна правка входа не убирает его больше чем на треть, и то
ценой разрушения события. Значит вопрос надо ставить не «что изменилось во входе», а «на что
отвечает сеть» -- и сравнить ответ между доменами.

Метод. Вход нормируется как (x-mean)/std, поэтому базовой точкой берётся нормированный ноль,
то есть «среднее событие той же длины»: структура маски и число хитов сохраняются, меняются
только значения. Атрибуция каждого входного числа -- интеграл градиента логита по пути от
базовой точки к событию, умноженный на смещение. Сумма атрибуций по всем числам равна
разности логитов между событием и базовой точкой; это свойство проверяется в прогоне.
"""
from __future__ import annotations

import argparse, logging, sys, time
from pathlib import Path

import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model            # noqa: E402
from ablate import CKPT, SRC, pick_parts, load_domain, QUALITY    # noqa: E402

CHANNELS = ["заряд", "время", "x", "y", "z"]
log = logging.getLogger("attr")


def integrated_gradients(model, hits, norm, device, steps=32, batch=64, max_hits=500,
                         clip=None):
    """Возвращает (атрибуция по каналам, логит события, логит базовой точки, сумма атрибуций)."""
    means = torch.tensor(norm["means"], dtype=torch.float32, device=device)
    stds = torch.tensor(norm["stds"], dtype=torch.float32, device=device)
    out_attr, out_logit, out_base, out_sum = [], [], [], []

    for start in range(0, len(hits), batch):
        chunk = hits[start:start + batch]
        lengths = torch.tensor([min(len(h), max_hits) for h in chunk],
                               dtype=torch.long, device=device)
        b, L = len(chunk), int(lengths.max())
        x = torch.zeros(b, L, 5, dtype=torch.float32, device=device)
        for i, h in enumerate(chunk):
            x[i, :lengths[i]] = torch.from_numpy(np.asarray(h[:lengths[i]], dtype=np.float32))
        mask = torch.arange(L, device=device)[None, :] < lengths[:, None]
        xn = torch.where(mask.unsqueeze(-1), (x - means) / (stds + 1e-8), x)
        # Модель обрезает амплитуду ВНУТРИ forward записью поверх среза features[:,:,0].
        # Срез нужен autograd для backward, поэтому при requires_grad это падает с
        # "modified by an inplace operation". Обрезаем сами, вне графа, и отключаем
        # внутреннее обрезание -- результат тот же, обрезание идёт до дифференцирования.
        if clip is not None:
            xn = torch.cat([xn[:, :, :1].clamp(max=clip), xn[:, :, 1:]], dim=2)

        # базовая точка -- нормированный ноль на валидных позициях
        base = torch.zeros_like(xn)
        total = torch.zeros_like(xn)
        for k in range(steps):
            alpha = (k + 0.5) / steps
            pt = (base + alpha * (xn - base)).detach().requires_grad_(True)
            logit = model({"features": pt, "lengths": lengths, "mask": mask}).sum()
            g, = torch.autograd.grad(logit, pt)
            total += g
        attr = (xn - base) * total / steps
        attr = torch.where(mask.unsqueeze(-1), attr, torch.zeros_like(attr))

        with torch.no_grad():
            l_ev = model({"features": xn, "lengths": lengths, "mask": mask}).flatten()
            l_bs = model({"features": base, "lengths": lengths, "mask": mask}).flatten()
        out_attr.append(attr.sum(dim=1).cpu().numpy())          # (b, 5) сумма по хитам
        out_logit.append(l_ev.cpu().numpy()); out_base.append(l_bs.cpu().numpy())
        out_sum.append(attr.sum(dim=(1, 2)).cpu().numpy())
    return (np.concatenate(out_attr), np.concatenate(out_logit),
            np.concatenate(out_base), np.concatenate(out_sum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mc-parts", type=int, default=400)
    ap.add_argument("--exp-parts", type=int, default=2)
    ap.add_argument("--score-thr", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-events", type=int, default=1500)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    import ablate
    ablate.SCORE_THR = a.score_thr
    t0 = time.time()
    model, norm, _ = load_model(str(CKPT), device=a.device)
    model.eval()
    clip = getattr(model, "amp_clip", None)
    model.amp_clip = None
    log.info(f"обрезание амплитуды вынесено наружу: clip={clip}")
    log.info(f"модель на {a.device}, порог {a.score_thr}")

    groups = {}
    for dom, n in (("exp", a.exp_parts), ("mc", a.mc_parts)):
        hits, chans, stored, w, prb = load_domain(dom, pick_parts(dom, n))
        hi = [h for h, s in zip(hits, stored) if s > a.score_thr][:a.max_events]
        lo = [h for h, s in zip(hits, stored) if s <= a.score_thr][:a.max_events]
        groups[(dom, "принятые")] = hi
        groups[(dom, "отвергнутые")] = lo
        log.info(f"  {dom}: принятых {len(hi):,}, отвергнутых {len(lo):,} "
                 f"({time.time()-t0:.0f} c)")

    log.info("\n" + "=" * 96)
    log.info(f"{'выборка':26s} {'N':>6} " + " ".join(f"{c:>11s}" for c in CHANNELS)
             + f" {'проверка':>10s}")
    log.info("=" * 96)
    for key, hits in groups.items():
        if not hits:
            continue
        attr, l_ev, l_bs, s = integrated_gradients(model, hits, norm, a.device, a.steps,
                                                  clip=clip)
        share = np.abs(attr).mean(axis=0)
        share = 100 * share / share.sum()
        # свойство полноты: сумма атрибуций должна равняться разности логитов
        err = np.abs(s - (l_ev - l_bs)) / np.maximum(np.abs(l_ev - l_bs), 1e-6)
        log.info(f"{key[0] + ', ' + key[1]:26s} {len(hits):>6,} "
                 + " ".join(f"{v:10.1f}%" for v in share)
                 + f" {np.median(err):10.1e}")
    log.info("=" * 96)
    log.info("Доля |атрибуции| по входным каналам. Последний столбец -- медианная "
             "относительная ошибка свойства полноты; больше ~0.1 означает, что шагов мало.")
    log.info(f"всего {time.time()-t0:.0f} c")


if __name__ == "__main__":
    main()
