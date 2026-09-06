"""Build the exp-background NPY dataset for fine-tuning, selecting by ENCODER-OOD
(not by low score as build_exp_bg.py does).

Rationale: the exp neutrino-like excess is over-confident OOD extrapolation — exp
events far from the MC encoder manifold (near-horizontal + bright, a combination MC
lacks) that the classifier wrongly scores as ν. We select exactly those events as
hard background (label 0) so a fine-tune teaches the model to reject them. Selection:

    encoder-OOD > MC-p{PCT}  AND  rvert < RVCUT

encoder-OOD = mean kNN(k) distance to a MC reference embedding manifold (E1 encoder).
rvert = std_z/std_xy on filtered hits (verticality proxy) — a soft safety filter so we
never label a rare steep (potential-ν) event as background. See
inference_v2/nu_classifier/reports/2026-07-07/REPORT.md §6.

Output (prefix ``exp_bg_``, same schema as build_exp_bg.py, loadable by
NuClassifierExpNpyDataset(prefix="exp_bg")):
    exp_bg_features.npy      (total_sig_hits,5) float32   filtered hits (prob>sn_thr), time-sorted
    exp_bg_offsets.npy       (n_events+1,)      int64
    exp_bg_n_sig_hits.npy    (n_events,)        int32
    exp_bg_n_sig_strings.npy (n_events,)        int32
    exp_bg_event_fks.npy     (n_events,)        int64     catalog FK (traceability)
    exp_bg_scores.npy        (n_events,)        float32   original model score
    exp_bg_ood.npy           (n_events,)        float32   encoder-OOD kNN dist
    exp_bg_rvert.npy         (n_events,)        float32
    exp_bg_dataset_info.json

Speed: OOD strongly correlates with high score, so only score>--min-score exp events
are read/embedded (default 0.15 keeps all plausible OOD; low-score exp are in-dist
muons that cannot pass the cut). This cuts the read ~20x vs the full pool.

Usage (from project root):
    python inference_v2/nu_classifier/exp_finetuning/build_exp_bg_ood.py \\
        --ckpt        experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth \\
        --exp-preds   inference_v2/nu_classifier/preds/260705_0702_..._E1_lambda0.01@da_checkpoint_epoch_010 \\
        --ood-percentile 99 --rvert-cut 0.7 --device cuda:0
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
import duckdb, h5py, numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings

CAT = ROOT / "data_manager/catalog_v2.duckdb"
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
EXH5 = ROOT / "data_manager/data/h5datasets/exp_full.h5"
MCPROBS = ROOT / "data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
EXPROBS = ROOT / "data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
RDCC = dict(rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)
STRING_DIVISOR = 36


def read_filtered(h5path, chan_avail, probs_path, df, sn_thr):
    """Per event -> (feats[time-sorted], n_sig_hits, n_sig_strings, rvert). None if <2 hits."""
    feats = [None] * len(df)
    nsh = np.zeros(len(df), np.int32); nss = np.zeros(len(df), np.int32)
    rvert = np.full(len(df), np.nan)
    f = h5py.File(h5path, "r", **RDCC); fp = h5py.File(probs_path, "r", **RDCC)
    for (base, pk), idx in df.groupby(["base", "part_key"]).groups.items():
        try:
            es = f[f"{base}/raw/ev_starts/{pk}/data"][:]
            ds = f[f"{base}/raw/data/{pk}/data"]
            ch = f[f"{base}/raw/channels/{pk}/data"]
            pr = fp[f"{base}/probs/{pk}/data"]
        except KeyError:
            continue
        idx = np.array(idx); locs = df.loc[idx, "local_idx"].to_numpy()
        order_l = np.argsort(locs)                  # monotonic reads within part (chunk locality)
        for row, l in zip(idx[order_l], locs[order_l]):
            if l + 1 >= len(es):
                continue
            s, e = int(es[l]), int(es[l + 1])
            p = pr[s:e].astype(np.float32); m = p > sn_thr
            if m.sum() < 2:
                continue
            hh = ds[s:e].astype(np.float32)[m]
            cc = ch[s:e].astype(np.int32)[m]
            order = np.argsort(hh[:, 1])                # time-sort
            hh = hh[order]; cc = cc[order]
            feats[row] = hh
            nsh[row] = len(hh)
            nss[row] = len(np.unique(cc // STRING_DIVISOR))
            sx, sy, sz = hh[:, 2].std(), hh[:, 3].std(), hh[:, 4].std()
            rvert[row] = sz / (np.sqrt(sx * sx + sy * sy) + 1e-3)
    f.close(); fp.close()
    return feats, nsh, nss, rvert


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Pretrained DA checkpoint (the fine-tune 'original')")
    ap.add_argument("--exp-preds", required=True, help="Preds dir with exp_full_thr*.duckdb + mc_merged_thr*.duckdb")
    ap.add_argument("--catalog", default=str(CAT))
    ap.add_argument("--sn-threshold", type=float, default=0.8)
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--min-strings", type=int, default=3)
    ap.add_argument("--ood-percentile", type=float, default=99.0, help="MC OOD percentile for the threshold")
    ap.add_argument("--ood-threshold", type=float, default=None,
                    help="Absolute OOD cut; overrides --ood-percentile for selection (percentile still logged)")
    ap.add_argument("--rvert-cut", type=float, default=0.7)
    ap.add_argument("--min-score", type=float, default=0.0, help="Only read/embed exp with score> this (speed)")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--n-mc-ref", type=int, default=40000, help="MC reference events for the manifold")
    ap.add_argument("--max-exp", type=int, default=0, help="Cap candidate exp read (0 = all above min-score)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    thr_tag = str(args.sn_threshold).replace(".", "p")
    preds = Path(args.exp_preds)
    ckpt_name = Path(args.ckpt).parent.name + "@" + Path(args.ckpt).stem
    if args.ood_threshold is not None:
        ood_tag = ("ood%g" % args.ood_threshold).replace(".", "p")   # e.g. ood2p5
    else:
        ood_tag = ("oodp%g" % args.ood_percentile).replace(".", "p")  # e.g. oodp99
    rv_tag = ("rv%g" % args.rvert_cut).replace(".", "p")
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "exp_bg_datasets" / f"{ckpt_name}_{ood_tag}_{rv_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out]  {out_dir}", flush=True)

    # ── candidates ────────────────────────────────────────────────────────────
    c = duckdb.connect(); c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{preds / f'exp_full_thr{thr_tag}.duckdb'}' AS e (READ_ONLY)")
    c.execute(f"ATTACH '{preds / f'mc_merged_thr{thr_tag}.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{args.catalog}' AS cat (READ_ONLY)")
    CUT = f"pr.n_sn_hits>={args.min_hits} AND pr.n_sn_strings>={args.min_strings}"
    mc = c.execute(f"""SELECT ev.data_class AS base, l.part_key, l.local_idx
        FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE {CUT} AND ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020')
        ORDER BY random() LIMIT {args.n_mc_ref}""").df()
    exp_lim = f"ORDER BY random() LIMIT {args.max_exp}" if args.max_exp else ""
    exp = c.execute(f"""SELECT pr.event_fk, pr.score, 'exp_full' AS base, l.part_key, l.local_idx
        FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE {CUT} AND pr.score > {args.min_score}
          AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) {exp_lim}""").df()
    c.close()
    print(f"[cand] MC ref {len(mc):,}, exp score>{args.min_score} {len(exp):,}", flush=True)

    model, norm, _ = load_model(args.ckpt, device=args.device)

    # ── MC reference manifold + OOD threshold (leave-out calibration) ──────────
    t0 = time.time()
    mc_feats, *_ = read_filtered(MCH5, True, MCPROBS, mc, args.sn_threshold)
    mc_keep = [x for x in mc_feats if x is not None]
    _, mc_emb = predict_scores_and_embeddings(model, mc_keep, norm, batch_size=args.batch_size, device=args.device)
    mc_emb = mc_emb.astype(np.float32)
    print(f"[mc]   embedded {len(mc_emb):,} ({time.time()-t0:.0f}s)", flush=True)

    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(42); perm = rng.permutation(len(mc_emb))
    n_ref = max(len(mc_emb) - 3000, int(0.8 * len(mc_emb)))
    ref, cal = mc_emb[perm[:n_ref]], mc_emb[perm[n_ref:]]
    nn = NearestNeighbors(n_neighbors=args.knn).fit(ref)
    cal_ood = nn.kneighbors(cal)[0].mean(1)
    pct_thr = float(np.percentile(cal_ood, args.ood_percentile))
    ood_thr = float(args.ood_threshold) if args.ood_threshold is not None else pct_thr
    print(f"[thr]  MC calib N={len(cal)} median={np.median(cal_ood):.3f} "
          f"p{args.ood_percentile}={pct_thr:.3f}; using OOD cut = {ood_thr:.3f}"
          + (" (absolute)" if args.ood_threshold is not None else ""), flush=True)

    # ── exp: embed, OOD, cut ──────────────────────────────────────────────────
    t0 = time.time()
    ex_feats, ex_nsh, ex_nss, ex_rvert = read_filtered(EXH5, True, EXPROBS, exp, args.sn_threshold)
    keep = [i for i, x in enumerate(ex_feats) if x is not None]
    fl = [ex_feats[i] for i in keep]
    ex_score, ex_emb = predict_scores_and_embeddings(model, fl, norm, batch_size=args.batch_size, device=args.device)
    ex_ood = nn.kneighbors(ex_emb.astype(np.float32))[0].mean(1)
    print(f"[exp]  embedded {len(keep):,} ({time.time()-t0:.0f}s)", flush=True)

    kk = np.array(keep)
    rvert_k = ex_rvert[kk]
    sel_mask = (ex_ood > ood_thr) & (rvert_k < args.rvert_cut)
    sel = np.where(sel_mask)[0]
    print(f"[cut]  OOD>{ood_thr:.2f} & rvert<{args.rvert_cut}: {len(sel):,} / {len(keep):,} candidates", flush=True)
    if len(sel) == 0:
        print("no events selected — aborting"); return

    # ── assemble output ───────────────────────────────────────────────────────
    sel_feats = [fl[i] for i in sel]
    sel_rows = kk[sel]
    features_arr = np.concatenate(sel_feats, axis=0).astype(np.float32)
    n_sig_hits = ex_nsh[sel_rows].astype(np.int32)
    n_sig_strings = ex_nss[sel_rows].astype(np.int32)
    offsets = np.zeros(len(sel) + 1, dtype=np.int64)
    np.cumsum(n_sig_hits.astype(np.int64), out=offsets[1:])
    assert offsets[-1] == len(features_arr), (offsets[-1], len(features_arr))
    event_fks = exp["event_fk"].to_numpy()[sel_rows].astype(np.int64)
    scores = ex_score[sel].astype(np.float32)
    ood_sel = ex_ood[sel].astype(np.float32)
    rvert_sel = rvert_k[sel].astype(np.float32)

    np.save(out_dir / "exp_bg_features.npy", features_arr)
    np.save(out_dir / "exp_bg_offsets.npy", offsets)
    np.save(out_dir / "exp_bg_n_sig_hits.npy", n_sig_hits)
    np.save(out_dir / "exp_bg_n_sig_strings.npy", n_sig_strings)
    np.save(out_dir / "exp_bg_event_fks.npy", event_fks)
    np.save(out_dir / "exp_bg_scores.npy", scores)
    np.save(out_dir / "exp_bg_ood.npy", ood_sel)
    np.save(out_dir / "exp_bg_rvert.npy", rvert_sel)

    info = {
        "selection": "encoder-OOD (kNN to MC manifold) AND rvert<cut",
        "n_events": int(len(sel)),
        "n_sig_hits_total": int(offsets[-1]),
        "checkpoint": ckpt_name, "ckpt_path": args.ckpt,
        "ood_percentile": args.ood_percentile, "ood_threshold": ood_thr, "knn": args.knn,
        "rvert_cut": args.rvert_cut, "min_score_prefilter": args.min_score,
        "sn_threshold": args.sn_threshold, "min_hits": args.min_hits, "min_strings": args.min_strings,
        "n_mc_ref": int(len(mc_emb)), "n_exp_candidates": int(len(keep)),
        "exp_h5": str(EXH5), "catalog": args.catalog,
        "score_stats": {"mean": float(scores.mean()), "median": float(np.median(scores)),
                        "min": float(scores.min()), "max": float(scores.max()),
                        "frac_gt0.8": float((scores > 0.8).mean())},
        "rvert_median": float(np.median(rvert_sel)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "exp_bg_dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"[done] {len(sel):,} events, {int(offsets[-1]):,} hits, "
          f"score med {np.median(scores):.3f} (frac>0.8 {(scores>0.8).mean():.2f}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
