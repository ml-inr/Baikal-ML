"""«Честная» задача владельца: {exp + exp_flip} против {nu + nu_flip}.

Классы по построению содержат обе ориентации, поэтому признак направления
нейтрализован. Классификатор обучается и тестируется (3-fold, стратифицированно)
на замороженных фичах энкодера (кэш probe_emb_latest.npz, чекпойнт
261010_1755/latest_da_checkpoint.pth). Для сравнения — диагональные и
перекрёстные квартанты и тривиальный контроль без модели (заряд).

Запуск: python .../honest_class_separation.py
"""
import sys, os, logging
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); os.chdir(REPO)
logging.basicConfig(level=logging.ERROR)

d = np.load(Path(__file__).parent / "probe_emb_latest.npz")
emb, emb_f, group = d["emb"], d["emb_f"], d["group"]

nu = np.flatnonzero(group >= 4)
exp = np.flatnonzero((group >= 2) & (group <= 3))
exp_nat, exp_tail = np.flatnonzero(group == 2), np.flatnonzero(group == 3)
nu_nat, nu_tail = np.flatnonzero(group == 4), np.flatnonzero(group == 5)


def cv_auc(X, y):
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs))


def mlp_auc(X, y, hidden=(32, 16), ln=True, steps=20000, seed=0, bs=256):
    torch.manual_seed(seed)
    layers, prev = [], X.shape[1]
    for h in hidden:
        layers.append(torch.nn.Linear(prev, h))
        if ln: layers.append(torch.nn.LayerNorm(h))
        layers.append(torch.nn.ReLU()); prev = h
    layers.append(torch.nn.Linear(prev, 1))
    m = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(steps):
        i = torch.randint(0, len(Xt), (bs,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[i]).ravel(), yt[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return roc_auc_score(y, m(Xt).numpy().ravel())


def rows(idx, flip):
    """Эмбеддинги строк idx; flip — булев скаляр или массив (не список!)."""
    flip = np.asarray(flip)
    if flip.ndim == 0:
        flip = np.full(len(idx), bool(flip))
    return np.where(flip[:, None], emb_f[idx], emb[idx])


def pair(a_idx, a_flip, b_idx, b_flip, tag, model="logreg"):
    X = np.concatenate([rows(a_idx, a_flip), rows(b_idx, b_flip)])
    y = np.concatenate([np.ones(len(a_idx)), np.zeros(len(b_idx))]).astype(int)
    auc = cv_auc(X, y) if model == "logreg" else mlp_auc(X, y)
    print(f"  {tag:<42} {auc:.4f}   ({model})")
    return auc


print("=== честная задача: {exp + exp_flip} vs {nu + nu_flip} ===")
pair(exp, False, nu, False, "только чистые (= C, сверка 0.9934)")
pair(exp, True, nu, True, "только зеркальные")
Xp = np.concatenate([emb[exp], emb_f[exp], emb[nu], emb_f[nu]])
yp = np.concatenate([np.zeros(2 * len(exp)), np.ones(2 * len(nu))]).astype(int)
groups = np.concatenate([np.arange(len(exp)), np.arange(len(exp)),
                         np.arange(len(nu)), np.arange(len(nu))])
aucs = []
for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=42).split(Xp, yp, groups):
    sc = StandardScaler().fit(Xp[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xp[tr]), yp[tr])
    aucs.append(roc_auc_score(yp[te], clf.decision_function(sc.transform(Xp[te]))))
print(f"  ПУЛ {{exp+flip}} vs {{nu+flip}}, logreg (grouped-CV): {np.mean(aucs):.4f}")

print("=== та же пуловая задача, нелинейный читатель (grouped 80/20, тест) ===")
rng = np.random.default_rng(0)
g_exp, g_nu = rng.permutation(len(exp)), rng.permutation(len(nu))
n_exp, n_nu = len(exp), len(nu)
te_mask = np.zeros(len(Xp), bool)
te_mask[np.concatenate([g_exp[:n_exp//5], 2*n_exp + g_nu[:n_nu//5]])] = True  # ν-строки со сдвигом 2*n_exp
tr_mask = ~te_mask
Xt = torch.tensor(Xp, dtype=torch.float32)
torch.manual_seed(0)
m = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.LayerNorm(32), torch.nn.ReLU(),
                        torch.nn.Linear(32, 16), torch.nn.LayerNorm(16), torch.nn.ReLU(),
                        torch.nn.Linear(16, 1))
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
yt = torch.tensor(yp, dtype=torch.float32)
itr = np.flatnonzero(tr_mask)
for _ in range(20000):
    i = itr[torch.randint(0, len(itr), (256,))]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[i]).ravel(), yt[i])
    opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    sc_mlp = m(Xt).numpy().ravel()
assert np.isfinite(sc_mlp).all(), "MLP разошёлся (nan) — чинить гиперпараметры"
print(f"  ПУЛ, MLP [32,16]+LN (тест по событиям): "
      f"{roc_auc_score(yp[te_mask], sc_mlp[te_mask]):.4f}")

print("=== квартанты (чистое/зеркало × exp/nu) ===")
pair(nu, False, exp, False, "ν чистые vs exp чистые (направление вкл.)")
pair(nu, True, exp, True, "ν зеркало vs exp зеркало (напр. вкл.)")
pair(nu, False, exp, True, "ν чистые vs exp ЗЕРКАЛО (напр. выкл.)")
pair(nu, True, exp, False, "ν ЗЕРКАЛО vs exp чистые (напр. выкл.)")
print("=== натуральные подгруппы, направление выключено ===")
pair(nu_nat, False, exp_nat, True, "ν-нат чистые vs exp-нат зеркало")
pair(nu_nat, True, exp_nat, False, "ν-нат зеркало vs exp-нат чистые")
print("=== контроль без модели: сырой заряд (флип-инвариантен) ===")
probe = np.load("data_manager/probe_sample/probe_sample.npz")
hits, offsets, g2 = probe["hits"], probe["offsets"], probe["group"]
sum_q = np.add.reduceat(hits[:, 0], offsets[:-1])
nu2 = np.flatnonzero(g2 >= 4); ex2 = np.flatnonzero((g2 >= 2) & (g2 <= 3))
auc_q = roc_auc_score([0] * len(nu2) + [1] * len(ex2), np.concatenate([sum_q[nu2], sum_q[ex2]]))
print(f"  суммарный заряд, {ex2.size}+{nu2.size} событий: {auc_q:.4f}  (флип его не меняет)")

print("=== ПОЛНАЯ задача игры, честный протокол ===")
mu = np.flatnonzero(group <= 1)
mc = np.concatenate([mu, nu])


def grouped_cv_auc(X, y, g, tag, model="logreg"):
    if model == "logreg":
        aucs = []
        for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=42).split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
        auc = float(np.mean(aucs))
    else:                                   # MLP, grouped 80/20, тест по событиям
        ev = np.unique(g)
        rng = np.random.default_rng(0)
        te_ev = set(rng.choice(ev, size=int(0.2 * len(ev)), replace=False).tolist())
        te = np.array([gi in te_ev for gi in g]); tr = ~te
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.LayerNorm(32),
                                torch.nn.ReLU(), torch.nn.Linear(32, 16),
                                torch.nn.LayerNorm(16), torch.nn.ReLU(),
                                torch.nn.Linear(16, 1))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
        itr = np.flatnonzero(tr)
        for _ in range(20000):
            i = itr[torch.randint(0, len(itr), (256,))]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[i]).ravel(), yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            sc_mlp = m(Xt).numpy().ravel()
        assert np.isfinite(sc_mlp).all()
        assert len(np.unique(y[te])) == 2, "тест из одного класса"
        auc = roc_auc_score(y[te], sc_mlp[te])
    print(f"  {tag:<48} {auc:.4f}   ({model})")
    return auc


def both_orientations(a, b, tag, model="logreg"):
    X = np.concatenate([emb[a], emb_f[a], emb[b], emb_f[b]])
    y = np.concatenate([np.zeros(2 * len(a)), np.ones(2 * len(b))]).astype(int)
    g = np.concatenate([np.arange(len(a)), np.arange(len(a)),
                        np.arange(len(b)) + len(a), np.arange(len(b)) + len(a)])
    return grouped_cv_auc(X, y, g, tag, model=model)


def mixed_once(a, b, tag):
    """Как в игре: по одному представлению на событие, случайная половина флипнута."""
    rng = np.random.default_rng(42)
    fa = np.zeros(len(a), bool); fa[rng.choice(len(a), len(a) // 2, replace=False)] = True
    fb = np.zeros(len(b), bool); fb[rng.choice(len(b), len(b) // 2, replace=False)] = True
    X = np.concatenate([rows(a, fa), rows(b, fb)])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))]).astype(int)
    g = np.concatenate([np.arange(len(a)), np.arange(len(b)) + len(a)])
    return grouped_cv_auc(X, y, g, tag)


both_orientations(exp, mu, "{mu+mu_flip} vs {exp+exp_flip}")
both_orientations(exp, nu, "{nu+nu_flip} vs {exp+exp_flip}   (сверка с 0.8329)")
both_orientations(exp, mc, "{mu+nu, обе} vs {exp, обе}   <-- ЗАДАЧА ИГРЫ")
both_orientations(exp, mc, "{mu+nu, обе} vs {exp, обе}, MLP", model="mlp")
mixed_once(exp, mc, "{mu+nu} vs {exp}, по одному пре-нию на событие")
