d#!/usr/bin/env python
"""2D distributions in the (d_MC, r_vert) plane --- the plane the fine-tuning cut lives in.

d_MC is the mean distance to the 20 nearest events of a kNN reference. Unlike the deployed
selection, whose reference was drawn at random from the full Monte Carlo set and therefore
followed its natural abundances (~96% EAS), the reference here holds an EQUAL number of
events per class, so the neutrino region is sampled as densely as the EAS one.

Populations:
  exp            -- a natural random sample of exp_full, NOT selected by any cut
  nuatm / nue2 / muatm -- out-of-training MC

Each population is drawn twice: over all its events, and restricted to classifier score
> 0.8. NOTE the score played no part in building the fine-tuning sample --- that selection
was blind to it (`min_score_prefilter: 0.0`; only 2.07% of the selected events exceed 0.8).
The score cut here isolates the high-score excess, which is a different population.

Sources, all memory-mapped NPY (random per-event reads from the gzip-chunked HDF5 run at
~200 events/s and were the previous bottleneck):
  MC   : nu_classifier_TEST_dataset_h5s0_thr0.5 --- built from testds_parts.json, whose
         parts are DISJOINT from the training parts (verified: 0 overlap in all three
         classes). Stored at threshold 0.5, so hits are re-filtered here to prob > 0.8,
         which reproduces exactly the hit sample the model was trained on.
         n_sig_hits is recomputed after the re-filter; n_sig_strings is recovered from the
         unique (x, y) of the surviving hits (validated against the stored 0.8 counts on
         the training dataset: 100% exact agreement for tolerances 1-20 m).
  exp  : nu_classifier_dataset_exp_full_thr0.8 --- already at threshold 0.8, bad runs
         c02_r0020/r0249 excluded at build time.

Outputs tables/dmc_rvert_planes.csv and four figures: the full plane and the in-box
zoom, each for all events and for score>0.8 only. GPU: see DEV.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings

DEV = "cuda:0"
CKPT = (ROOT / "experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
        / "da_checkpoint_epoch_010.pth")
MCNPY = ROOT / "data_manager/datasets/nu_classifier_TEST_dataset_h5s0_thr0.5"
TRNPY = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
PRED = (ROOT / "inference_v2/nu_classifier/preds"
        / "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010")
CAT = ROOT / "data_manager/catalog_v2.duckdb"
EXPNPY = ROOT / "data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"
KNN, BS = 20, 512
N_REF_KEEP = 7_000
N_EVAL = {"muatm_2020": 600_000, "nuatm_2020": 600_000, "nue2_2020": 250_000}
N_EXP = 2_500_000        # capped by the pool; high-score exp events are ~0.06% of it
N_PLOT = 200_000         # per population and per view, enough for a 2D histogram
SCORE_CUT = 0.8          # applied to the EVALUATED populations, not to the reference
N_HISCORE_NUATM = 200_000  # ~3.4% of them land in the box -> ~6,800, comfortably over 5,000
MIN_HITS, MIN_STR, SN = 8, 3, 0.8
STRING_TOL = 10.0
OOD_CUT, RV_CUT = 2.5, 0.7
PTYPE = {"muatm_2020": 0, "nuatm_2020": 1, "nue2_2020": 2}
CLASSES = list(PTYPE)

T0 = time.time()


def log(msg: str) -> None:
    el = time.time() - T0
    print(f"[{int(el)//60:02d}:{int(el)%60:02d}] {msg}", flush=True)


def rvert_of(h: np.ndarray) -> float:
    sx, sy, sz = h[:, 2].std(), h[:, 3].std(), h[:, 4].std()
    return float(sz / (np.sqrt(sx * sx + sy * sy) + 1e-3))


def n_strings(h: np.ndarray) -> int:
    # NOTE: strings counted from cluster-centred (x, y). The pipeline counts them
    # from channel ids (io.py:_count_sig_hits_strings), which is what defines the
    # h8s3 selection; coordinates can merge two strings at the same position in
    # different clusters. Kept as-is here because published numbers used it — use
    # inference_v2/nu_classifier/compute_scalars.py for new work.
    return len(np.unique(np.round(h[:, 2:4].astype(np.float64) / STRING_TOL), axis=0))


def qmean_of(h: np.ndarray) -> float:
    """Mean hit charge, clipped at 100 p.e. as elsewhere in this analysis chain."""
    return float(np.clip(h[:, 0], 0, 100).mean())


def _read_and_measure(h5, probs, d, label):
    """read_filtered on a small (part_key, local_idx) frame -> hits, r_vert, q_mean."""
    from archive_tracked.inference_v2.nu_classifier.exp_finetuning.build_exp_bg_ood import read_filtered
    feats, _, _, rv = read_filtered(h5, True, probs, d, SN)
    keep = [i for i, x in enumerate(feats) if x is not None and n_strings(x) >= MIN_STR]
    hits = [feats[i] for i in keep]
    log(f"  {label}: read {len(d):,} -> {len(hits):,} after the string cut")
    return hits, rv[keep], np.asarray([qmean_of(h) for h in hits], np.float32)


def hiscore_mc_from_db(cls, train_keys, limit):
    """High-score events of a rare-at-high-score MC class, straight from the prediction DB.

    Muons score above 0.8 only ~0.03% of the time, so the TEST NPY yields ~130 of them;
    atmospheric neutrinos are plentiful at high score but the out-of-training TEST parts
    hold only ~29k events in total. In both cases the DB of the same checkpoint gives the
    (part_key, local_idx) of exactly the wanted events, and only those are read from HDF5.
    Training events are dropped via the NPY back-links.
    """
    import duckdb
    from archive_tracked.inference_v2.nu_classifier.exp_finetuning.build_exp_bg_ood import MCH5, MCPROBS

    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PRED / 'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    d = c.execute(f"""SELECT pr.score, ev.data_class AS base, l.part_key, l.local_idx
        FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE pr.n_sn_hits>={MIN_HITS} AND pr.n_sn_strings>={MIN_STR}
          AND ev.data_class='{cls}' AND pr.score>{SCORE_CUT}""").df()
    c.close()
    key = d.part_key.astype(str) + "|" + d.local_idx.astype(str)
    d = d[~key.isin(train_keys).to_numpy()].reset_index(drop=True)
    log(f"  high-score {cls} from DB: {len(d):,} out-of-training")
    if len(d) > limit:
        d = d.sample(n=limit, random_state=42).sort_values(
            ["part_key", "local_idx"]).reset_index(drop=True)
        log(f"  capped to {limit:,} (sorted by part for chunk locality)")
    return _read_and_measure(MCH5, MCPROBS, d, f"high-score {cls}")


def hiscore_exp_from_db():
    """Every high-score experimental event there is: the DB covers all of exp_full.

    3,110,199 events pass h8s3 after dropping the bad runs, and only 2,259 of them exceed
    the score cut, so this is the whole available sample --- the dataset, not the method,
    is what limits this population.
    """
    import duckdb
    from archive_tracked.inference_v2.nu_classifier.exp_finetuning.build_exp_bg_ood import EXH5, EXPROBS

    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PRED / 'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    d = c.execute(f"""SELECT pr.score, 'exp_full' AS base, l.part_key, l.local_idx
        FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE pr.n_sn_hits>={MIN_HITS} AND pr.n_sn_strings>={MIN_STR}
          AND pr.score>{SCORE_CUT}
          AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))
        ORDER BY l.part_key, l.local_idx""").df()
    c.close()
    log(f"  high-score exp from DB: {len(d):,} (this is the entire 2020 sample)")
    return _read_and_measure(EXH5, EXPROBS, d, "high-score exp")


def main() -> None:
    log(f"start; device={DEV}")

    # ---------- MC: out-of-training TEST dataset, re-filtered to prob > 0.8 ----------
    feats = np.load(MCNPY / "features.npy", mmap_mode="r")
    probs = np.load(MCNPY / "probs.npy", mmap_mode="r")
    offs = np.load(MCNPY / "offsets.npy")
    ptype = np.load(MCNPY / "particle_types.npy")
    log(f"MC TEST dataset: {len(ptype):,} events, {len(feats):,} hits (threshold 0.5)")

    keep_hit = np.asarray(probs) > SN
    nsh08 = np.add.reduceat(keep_hit.astype(np.int64), offs[:-1])
    nsh08[np.diff(offs) == 0] = 0
    log(f"re-filtered to prob>{SN}: events with >={MIN_HITS} hits: "
        f"{(nsh08 >= MIN_HITS).sum():,}")

    rng = np.random.default_rng(42)
    sel = {}
    for cls in CLASSES:
        cand = np.flatnonzero((ptype == PTYPE[cls]) & (nsh08 >= MIN_HITS))
        rng.shuffle(cand)
        need = N_REF_KEEP + N_EVAL[cls]
        picked, hits, rv, qm = [], [], [], []
        for i in cand:
            sl = slice(offs[i], offs[i + 1])
            h = np.asarray(feats[sl], dtype=np.float32)[np.asarray(keep_hit[sl])]
            if n_strings(h) < MIN_STR:
                continue
            picked.append(i)
            hits.append(h)
            rv.append(rvert_of(h))
            qm.append(qmean_of(h))
            if len(picked) >= need:
                break
        sel[cls] = (hits, np.asarray(rv, np.float32), np.asarray(qm, np.float32))
        log(f"  {cls:14s} candidates {len(cand):>9,} -> kept {len(picked):,}"
            f"{'  (POOL EXHAUSTED)' if len(picked) < need else ''}")

    model, norm, _ = load_model(str(CKPT), device=DEV)
    log("model loaded")

    def embed(hits):
        s, e = predict_scores_and_embeddings(model, hits, norm, batch_size=BS, device=DEV)
        return np.asarray(s, np.float32), e.astype(np.float32)

    ref_emb, ref_cls, raw, extra_hi = [], [], {}, {}
    for cls in CLASSES:
        hits, rv, qm = sel[cls]
        sc, e = embed(hits)
        ref_emb.append(e[:N_REF_KEEP])
        ref_cls.append(np.full(min(N_REF_KEEP, len(e)), cls))
        raw[cls] = (e[N_REF_KEEP:], rv[N_REF_KEEP:], sc[N_REF_KEEP:], qm[N_REF_KEEP:])
        n_hi = int((sc[N_REF_KEEP:] > SCORE_CUT).sum())
        log(f"embedded {cls}: {len(e):,} (ref {min(N_REF_KEEP,len(e)):,}, "
            f"eval {max(0,len(e)-N_REF_KEEP):,}) -> score>{SCORE_CUT}: {n_hi:,}"
            f" ({100*n_hi/max(1,len(e)-N_REF_KEEP):.3f}%)")

    # The high-score views need their own sourcing: muons are too rare at high score in the
    # TEST NPY (~130), the out-of-training nuatm pool there is only ~29k, and the exp NPY
    # caps at 200k events per run. The prediction DB of this checkpoint has all of them.
    tr_pk = np.load(TRNPY / "h5_part_keys.npy", allow_pickle=True).astype(str)
    tr_li = np.load(TRNPY / "h5_local_event_ids.npy").astype(np.int64)
    tr_pt = np.load(TRNPY / "particle_types.npy")

    def keys_for(cls):
        m = tr_pt == PTYPE[cls]
        return pd.Index([f"{a}|{b}" for a, b in zip(tr_pk[m], tr_li[m])])

    for cls, limit in (("muatm_2020", 200_000), ("nuatm_2020", N_HISCORE_NUATM)):
        h_hits, h_rv, h_qm = hiscore_mc_from_db(cls, keys_for(cls), limit)
        h_sc, h_e = embed(h_hits)
        kept = h_sc > SCORE_CUT
        extra_hi[cls] = (h_e[kept], h_rv[kept], h_qm[kept])
        log(f"high-score {cls}: embedded {len(h_e):,}, confirmed score>{SCORE_CUT}: "
            f"{int(kept.sum()):,}  (DB/forward-pass agreement {100*kept.mean():.1f}%)")

    x_hits, x_rv, x_qm = hiscore_exp_from_db()
    x_sc, x_e = embed(x_hits)
    kept = x_sc > SCORE_CUT
    extra_hi["exp"] = (x_e[kept], x_rv[kept], x_qm[kept])
    log(f"high-score exp: embedded {len(x_e):,}, confirmed score>{SCORE_CUT}: "
        f"{int(kept.sum()):,}  (DB/forward-pass agreement {100*kept.mean():.1f}%)")

    ref_emb = np.concatenate(ref_emb)
    ref_cls = np.concatenate(ref_cls)
    log(f"balanced reference: {len(ref_emb):,}, per class "
        f"{[int((ref_cls == k).sum()) for k in CLASSES]}")

    nn = NearestNeighbors(n_neighbors=KNN, n_jobs=-1).fit(ref_emb)
    log("kNN index built")


    # ---------- exp: natural sample, no selection cuts ----------
    e_feats = np.load(EXPNPY / "exp_features.npy", mmap_mode="r")
    e_offs = np.load(EXPNPY / "exp_offsets.npy")
    e_nsh = np.load(EXPNPY / "exp_n_sig_hits.npy")
    e_nss = np.load(EXPNPY / "exp_n_sig_strings.npy")
    cand = np.flatnonzero((e_nsh >= MIN_HITS) & (e_nss >= MIN_STR))
    pick = rng.choice(cand, size=min(N_EXP, len(cand)), replace=False)
    hits = [np.asarray(e_feats[e_offs[i]:e_offs[i + 1]], dtype=np.float32) for i in pick]
    rv = np.asarray([rvert_of(h) for h in hits], np.float32)
    qm = np.asarray([qmean_of(h) for h in hits], np.float32)
    sc, e = embed(hits)
    raw["exp"] = (e, rv, sc, qm)
    log(f"exp natural sample: {len(cand):,} pass h8s3 -> embedded {len(e):,}"
        f" -> score>{SCORE_CUT}: {int((sc > SCORE_CUT).sum()):,}")

    # ---------- d_MC only for what will be plotted (kNN over 3.2M would dominate) ----
    ORDER = ["exp", "nuatm_2020", "nue2_2020", "muatm_2020"]
    views = {}
    for k in ORDER:
        emb, rv, sc, qm = raw[k]
        for view, mask in (("all", np.ones(len(sc), bool)), ("hi", sc > SCORE_CUT)):
            if view == "hi" and k in extra_hi:      # rare class: use the DB-sourced set
                e2, rv2, qm2 = extra_hi[k]
                d = nn.kneighbors(e2)[0].mean(1)
                views[(k, view)] = (d, rv2, qm2, len(d))
                log(f"d_MC {k} [{view}]: plotted {len(d):,} (from prediction DB)")
                continue
            idx = np.flatnonzero(mask & ~np.isnan(rv))
            if len(idx) > N_PLOT:
                idx = rng.choice(idx, size=N_PLOT, replace=False)
            if len(idx) == 0:
                views[(k, view)] = (np.empty(0), np.empty(0), np.empty(0), 0)
                continue
            d = nn.kneighbors(emb[idx])[0].mean(1)
            views[(k, view)] = (d, rv[idx], qm[idx], int(mask.sum()))
            log(f"d_MC {k} [{view}]: plotted {len(idx):,} of {int(mask.sum()):,}")

    log(f"fraction inside the box (d_MC>{OOD_CUT} AND r_vert<{RV_CUT}):")
    rows = []
    for k in ORDER:
        for view in ("all", "hi"):
            d, rv, qm, ntot = views[(k, view)]
            if len(d) == 0:
                print(f"  {k:14s} [{view}] EMPTY", flush=True)
                continue
            inbox = (d > OOD_CUT) & (rv < RV_CUT)
            print(f"  {k:14s} [{view:3s}] pop={ntot:>9,}  plotted={len(d):>7,}  "
                  f"in box {inbox.sum():>7,} ({100*inbox.mean():6.2f}%)   "
                  f"d_MC med {np.median(d):5.2f}   r_vert med {np.median(rv):5.2f}   "
                  f"q_mean med {np.median(qm):5.2f}   q_mean med in box "
                  f"{(np.median(qm[inbox]) if inbox.any() else float('nan')):5.2f}",
                  flush=True)
            rows.append(pd.DataFrame(dict(pop=k, view=view, d_MC=d, r_vert=rv, q_mean=qm)))
    pd.concat(rows).to_csv(HERE / "tables/dmc_rvert_planes.csv", index=False)

    # ---------- figures ----------
    titles = {"exp": "Experimental", "nuatm_2020": r"MC $\nu_\mu^{atm}$",
              "nue2_2020": r"MC $\nu_\mu^{cosm}$", "muatm_2020": "MC EAS"}

    def draw(fname, view, xlim, ylim, only_box, suptitle):
        fig, ax = plt.subplots(2, 2, figsize=(13, 10))
        xb, yb = np.linspace(*xlim, 80), np.linspace(*ylim, 80)
        for a, k in zip(ax.ravel(), ORDER):
            d, rv, _qm, ntot = views[(k, view)]
            dd, rr = d, rv
            if only_box and len(d):
                m = (d > OOD_CUT) & (rv < RV_CUT)
                dd, rr = d[m], rv[m]
            if len(dd) == 0:
                a.set_title(f"{titles[k]} — no events")
                a.set_xlabel(r"$d_\mathrm{MC}$")
                a.set_ylabel(r"$r_\mathrm{vert}$")
                continue
            h = a.hist2d(np.clip(dd, *xlim), np.clip(rr, *ylim),
                         bins=[xb, yb], norm=LogNorm(), cmap="viridis")
            if not only_box:
                a.add_patch(plt.Rectangle((OOD_CUT, 0), xlim[1] - OOD_CUT, RV_CUT,
                                          fill=False, ec="red", lw=2))
            frac = 100 * ((d > OOD_CUT) & (rv < RV_CUT)).mean()
            a.set_title(f"{titles[k]} — {frac:.2f}% in box  (N={len(dd):,})", fontsize=12)
            a.set_xlabel(r"$d_\mathrm{MC}$")
            a.set_ylabel(r"$r_\mathrm{vert}$")
            fig.colorbar(h[3], ax=a, label="events")
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        fig.savefig(HERE / f"figures/{fname}", dpi=140)
        log(f"saved figures/{fname}")

    box = r"$d_\mathrm{MC}>2.5$, $r_\mathrm{vert}<0.7$"
    draw("dmc_rvert_planes.png", "all", (0, 8), (0, 3), False,
         rf"$(d_\mathrm{{MC}}, r_\mathrm{{vert}})$ plane, all events; selection box in red ({box})")
    draw("dmc_rvert_planes_inbox.png", "all", (OOD_CUT, 8), (0, RV_CUT), True,
         rf"All events, restricted to the selection box ({box})")
    draw("dmc_rvert_planes_hiscore.png", "hi", (0, 8), (0, 3), False,
         rf"Same, but only events with classifier score $\xi>{SCORE_CUT}$;"
         rf" selection box in red ({box})")
    draw("dmc_rvert_planes_hiscore_inbox.png", "hi", (OOD_CUT, 8), (0, RV_CUT), True,
         rf"Events with $\xi>{SCORE_CUT}$, restricted to the selection box ({box})")


if __name__ == "__main__":
    main()
