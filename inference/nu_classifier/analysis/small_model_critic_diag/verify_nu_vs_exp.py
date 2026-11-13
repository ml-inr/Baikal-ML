"""Три независимые проверки числа ν-vs-exp ≈ 0.99 из decompose_game_vs_hist.py.

1. Сырые скаляры из хитов, БЕЗ модели вообще (суммарный заряд, число хитов):
   если ν-vs-exp велико уже на сырых данных, число из фич — не артефакт пайплайна.
   Для сравнения те же скаляры для μ-vs-exp (известная величина 0.78-0.81).
2. Независимый путь фич: кэш epoch0_feats.npz (другой чекпойнт — линейный прогон,
   другая реализация нормализации), метки выровнены по блокам. Внутренняя сверка:
   μ-vs-exp должен воспроизвести 0.6611 из первого дневника.
3. Юнит-проверка _probe_embeddings(flip_z=True): ручная реализация
   (паддинг -> нормализация -> негация z на валидных) против выходов тренера.
"""
import sys, logging
from pathlib import Path
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import os; os.chdir(REPO)
logging.basicConfig(level=logging.ERROR)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROBE = "data_manager/probe_sample/probe_sample.npz"
CACHE = Path(__file__).parent / "epoch0_feats.npz"

def cv_auc(X, y):
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs))

print("=== 1) сырые скаляры, без модели ===")
d = np.load(PROBE)
hits, offsets, group = d["hits"], d["offsets"], d["group"]
n = len(group)
sum_q = np.add.reduceat(hits[:, 0], offsets[:-1])
n_hits = (offsets[1:] - offsets[:-1]).astype(float)
max_q = np.maximum.reduceat(hits[:, 0], offsets[:-1])
nu = np.flatnonzero(group >= 4)
exp_all = np.flatnonzero((group >= 2) & (group <= 3))
exp_tail = np.flatnonzero(group == 3)
mu = np.flatnonzero(group <= 1)
def scalar_auc(v, a, b):
    return roc_auc_score([0]*len(a) + [1]*len(b), np.concatenate([v[a], v[b]]))
for name, v in [("суммарный заряд", sum_q), ("число хитов", n_hits), ("макс. заряд", max_q)]:
    print(f"  {name:<15} ν-vs-exp(весь): {scalar_auc(v, nu, exp_all):.4f}   "
          f"ν-vs-exp(только хвост скopa): {scalar_auc(v, nu, exp_tail):.4f}   "
          f"μ-vs-exp(весь): {scalar_auc(v, mu, exp_all):.4f}")
print(f"  медианы: ν {np.median(sum_q[nu]):.0f}, μ {np.median(sum_q[mu]):.0f}, "
      f"exp {np.median(sum_q[exp_all]):.0f}, exp-хвост {np.median(sum_q[exp_tail]):.0f} п.е.")

print("=== 2) независимый путь фич (кэш линейного прогона, эпоха 0) ===")
c = np.load(CACHE)
src_all, tgt_all, src_lab = c["src"], c["tgt"], c["src_lab"]


def block_sizes(n_events, bs):
    sizes, left = [], n_events
    while left > 0:
        b = min(bs, left)
        sizes.append(b)
        left -= b
    return sizes


def clean_rows(F, sizes):
    """Строки чистой половины: укладка кэша [чистое|флип] на батч."""
    out, o = [], 0
    for b in sizes:
        out.append(F[o:o + b])
        o += 2 * b
    return np.concatenate(out)


SRC_BS, TGT_BS = 512, 256          # фактические размеры батчей загрузчиков
src_sz = block_sizes(4000, SRC_BS)
tgt_sz = block_sizes(8000, TGT_BS)

src_c = clean_rows(src_all, src_sz)
tgt_c = clean_rows(tgt_all, tgt_sz)
lab_c = clean_rows(src_lab.reshape(-1, 1), src_sz).ravel()
print(f"  событий: MC {len(src_c)} (ν {int((lab_c>.5).sum())}), exp {len(tgt_c)}")
nu2, mu2 = src_c[lab_c > .5], src_c[lab_c < .5]
mu_auc = cv_auc(np.concatenate([mu2, tgt_c]),
                np.concatenate([np.zeros(len(mu2)), np.ones(len(tgt_c))]))
nu_auc = cv_auc(np.concatenate([nu2, tgt_c]),
                np.concatenate([np.ones(len(nu2)), np.zeros(len(tgt_c))]))
print(f"  μ-vs-exp (сверка с 0.6611 из первого дневника): {mu_auc:.4f}")
print(f"  ν-vs-exp (натуральный состав, независимый путь): {nu_auc:.4f}")

print("=== 2б) подблоки образца пробы (кэш пробных фич) ===")
emb = np.load(Path(__file__).parent / "probe_emb_latest.npz")
E, G = emb["emb"], emb["group"]
gidx = {g: np.flatnonzero(G == g) for g in np.unique(G)}


def sub(a_groups, b_groups, tag):
    ia = np.concatenate([gidx[g] for g in a_groups])
    ib = np.concatenate([gidx[g] for g in b_groups])
    auc = cv_auc(np.concatenate([E[ia], E[ib]]),
                 np.concatenate([np.ones(len(ia)), np.zeros(len(ib))]))
    print(f"  {tag:<38} {auc:.4f}")


sub([4], [2], "ν нат vs exp нат")
sub([4], [2, 3], "ν нат vs exp весь")
sub([4, 5], [2, 3], "ν весь vs exp весь (как C)")
sub([5], [3], "ν хвост vs exp хвост")
sub([5], [2], "ν хвост vs exp нат")
sub([0], [2], "μ нат vs exp нат (как A)")

print("=== 3) юнит-проверка flip_z ===")
from src.training.da_nu_classifier_trainer_da_repair import NuClassifierDomainAdaptationTrainer
cfg = yaml.safe_load(open("trained_models/classifier/261010_1755_da_nu_classifier_exp_full_E1_small_d64_l2/da_config.yaml"))
cfg["data"]["device"] = "cpu"
cfg["data"]["source_domain"]["max_events"] = 20000
cfg["data"]["target_domain"]["max_events"] = 20000
cfg["logging"]["output_dir"] = "/tmp/diag_small_out"
cfg["logging"]["tensorboard"] = False
with NuClassifierDomainAdaptationTrainer(cfg) as t:
    t.prepare_data(); t.prepare_model()
    ck = torch.load("trained_models/classifier/261010_1755_da_nu_classifier_exp_full_E1_small_d64_l2/latest_da_checkpoint.pth",
                    map_location="cpu", weights_only=False)
    t.da_model.base_model.load_state_dict(ck["base_model_state_dict"])
    t.da_model.eval()
    means = torch.tensor(t.normalization_config["means"]); stds = torch.tensor(t.normalization_config["stds"])
    sl = slice(offsets[nu[0]], offsets[nu[0] + 3])
    h = hits[sl]; o = np.array([0, len(h)])
    got = t._probe_embeddings(h, o, flip_z=True)
    # ручная реализация: паддинг -> нормализация -> негация z на валидных
    x = torch.tensor(h[None, :, :], dtype=torch.float32)
    xn = (x - means) / (stds + 1e-8)
    xn[:, :, 4] = -xn[:, :, 4]
    lengths = torch.tensor([len(h)]); mask = torch.ones(1, len(h), dtype=torch.bool)
    with torch.no_grad():
        manual = t.da_model.base_model.get_feature_representation(
            {"features": xn, "lengths": lengths, "mask": mask}).numpy()
    print(f"  max|_probe_embeddings(flip) - ручная реализация| = {np.abs(got - manual).max():.2e}")
