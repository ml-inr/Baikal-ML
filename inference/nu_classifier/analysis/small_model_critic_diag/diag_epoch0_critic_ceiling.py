"""Почему критик маленькой модели стоит на 0.53: живой против offline по рецепту игры.

Замеры на чекпойнте эпохи 0 прогона 261010_1642_..._small_d64_l2_lin (задача игры:
MC-смесь с половинным z-флипом против exp с зеркалом; метрика — AUC).
Сравнивается:
  * живой критик из чекпойнта (eval);
  * тот же 65-параметровый линейный критик, обученный OFFLINE на замороженных
    фичах по рецепту игры (батчи 256+256, instance noise 0.05, AdamW 3e-4, wd 0.01);
  * свежий logreg (3-fold, стандартизация) — линейный потолок;
  * MLP [32,16]±LayerNorm (критик эталонного прогона) — нелинейный потолок;
  * проба тренера (μ против exp) как санитарная проверка воспроизводимости.

Запуск: python inference/nu_classifier/analysis/small_model_critic_diag/diag_epoch0_critic_ceiling.py
Фичи кэшируются в epoch0_feats.npz рядом со скриптом (в git не попадает).
"""
import sys, os, logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
logging.basicConfig(level=logging.ERROR)

from src.training.da_nu_classifier_trainer_da_repair import (  # noqa: E402
    NuClassifierDomainAdaptationTrainer,
)

RUN = "trained_models/classifier/261010_1642_da_nu_classifier_exp_full_E1_small_d64_l2_lin"
CONFIG = (RUN + "/da_config.yaml")   # снимок конфига внутри прогона
FEATS = Path(__file__).parent / "epoch0_feats.npz"
DEV = "cpu"


def build_cfg():
    cfg = yaml.safe_load(open(CONFIG))
    cfg["data"]["device"] = DEV
    cfg["data"]["source_domain"]["max_events"] = 40000
    cfg["data"]["target_domain"]["max_events"] = 40000
    cfg["logging"]["output_dir"] = "/tmp/diag_small_out"
    cfg["logging"]["tensorboard"] = False
    return cfg


def auc_of(model, S, T):
    with torch.no_grad():
        sa = model(S).numpy().ravel()
        sb = model(T).numpy().ravel()
    return roc_auc_score([0] * len(sa) + [1] * len(sb), np.concatenate([sa, sb]))


def train_torch_reader(S, T, hidden=None, ln=False, steps=15625, seed=0,
                       bs=256, noise=0.05, lr=3e-4, wd=0.01):
    torch.manual_seed(seed)
    layers, prev = [], S.shape[1]
    for h in (hidden or []):
        layers.append(torch.nn.Linear(prev, h))
        if ln:
            layers.append(torch.nn.LayerNorm(h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    model = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    z = torch.zeros(bs, 1)
    o = torch.ones(bs, 1)
    for _ in range(steps):
        fs = S[torch.randint(0, len(S), (bs,))]
        ft = T[torch.randint(0, len(T), (bs,))]
        if noise:
            fs = fs + torch.randn_like(fs) * noise
            ft = ft + torch.randn_like(ft) * noise
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(fs), z)
                + torch.nn.functional.binary_cross_entropy_with_logits(model(ft), o)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main():
    need_extract = not FEATS.exists()

    with NuClassifierDomainAdaptationTrainer(build_cfg()) as t:
        t.prepare_data()
        t.prepare_model()
        ck = torch.load(f"{RUN}/da_checkpoint_epoch_000.pth",
                        map_location="cpu", weights_only=False)
        t.da_model.base_model.load_state_dict(ck["base_model_state_dict"])
        t.da_model.domain_discriminator.load_state_dict(ck["domain_discriminator_state_dict"])
        t.da_model.eval()
        dev = t.device
        w = t.da_model.domain_discriminator.discriminator[0]
        print(f"live critic: ||w||={w.weight.norm().item():.4f}, bias={w.bias.item():.4f}")

        if need_extract:
            print("probe (trainer's own):",
                  {k: round(v, 4) for k, v in t._invariance_probe().items()})

            def extract(loader, labels=False):
                F, Y = [], []
                with torch.no_grad():
                    for b in loader:
                        for k in b:
                            if isinstance(b[k], torch.Tensor):
                                b[k] = b[k].to(dev)
                        f, m, l = b["features"], b["mask"], b["lengths"]
                        F.append(t.da_model.base_model.get_feature_representation(
                            {"features": f, "lengths": l, "mask": m}).cpu().numpy())
                        gf = f.clone()
                        gf[:, :, 4] = torch.where(m, -f[:, :, 4], f[:, :, 4])
                        F.append(t.da_model.base_model.get_feature_representation(
                            {"features": gf, "lengths": l, "mask": m}).cpu().numpy())
                        if labels:
                            Y.append(np.asarray(b["labels"]).reshape(-1))
                return (np.concatenate(F), np.concatenate(Y)) if labels else np.concatenate(F)

            src_all, src_lab = extract(t.source_val_loader, labels=True)
            tgt_all = extract(t.target_val_loader)
            np.savez(FEATS, src=src_all, tgt=tgt_all, src_lab=np.repeat(src_lab, 2))
        else:
            print("features: cache")

        d = np.load(FEATS)
        src_all, tgt_all = d["src"], d["tgt"]
        with torch.no_grad():
            sa = t.da_model.domain_discriminator(
                torch.tensor(src_all).to(dev)).cpu().numpy().ravel()
            sb = t.da_model.domain_discriminator(
                torch.tensor(tgt_all).to(dev)).cpu().numpy().ravel()
        print("LIVE critic, in-game: "
              f"{roc_auc_score([0]*len(sa)+[1]*len(sb), np.concatenate([sa, sb])):.4f}")

    S = torch.tensor(src_all, dtype=torch.float32)
    T = torch.tensor(tgt_all, dtype=torch.float32)

    for steps, tag in [(15625, "15 625 шагов"), (62500, "62 500 шагов")]:
        lin = train_torch_reader(S, T, hidden=None, steps=steps)
        print(f"OFFLINE linear (рецепт игры, {tag}): {auc_of(lin, S, T):.4f}")

    for ln in (True, False):
        mlp = train_torch_reader(S, T, hidden=[32, 16], ln=ln)
        print(f"OFFLINE MLP [32,16] LN={ln} (рецепт игры): {auc_of(mlp, S, T):.4f}")
    mlp = train_torch_reader(S, T, hidden=[32, 16], ln=True, steps=62500)
    print(f"OFFLINE MLP [32,16] LN=True (62 500 шагов): {auc_of(mlp, S, T):.4f}")

    X = np.concatenate([src_all, tgt_all])
    y = np.concatenate([np.zeros(len(src_all)), np.ones(len(tgt_all))])
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    print(f"FRESH logreg (3-fold, стандартизация): {np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
