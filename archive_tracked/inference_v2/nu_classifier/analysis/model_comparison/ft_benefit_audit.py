"""Honest re-assessment of the fine-tuning benefit.

(1) Excess at FIXED MC-muon working points (not a fixed score threshold).
(2) How much of the 'hard' experimental population was actually removed from the test.
"""
import duckdb
import numpy as np
import pandas as pd

ROOT = "/home/albert/Baikal2025"
NC = f"{ROOT}/inference_v2/nu_classifier"
CSV = f"{NC}/analysis/model_comparison/tables/paper_suppression_curve_daexp.csv"
BASE = (f"{NC}/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
        f"@da_checkpoint_epoch_010/exp_full_thr0p8.duckdb")
FKS = (f"{NC}/exp_finetuning/exp_bg_datasets/260705_0702_da_nu_classifier_exp_full"
       f"_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7/exp_bg_event_fks.npy")

d = pd.read_csv(CSV)
print("=== (1) excess at FIXED MC-muon working points, held-out, same events both models ===")
print(f"{'mu-surv':>9} {'suppression':>12} {'base excess':>12} {'FT excess':>10} "
      f"{'base sig-eff':>13} {'FT sig-eff':>11} {'n_mu@cut':>9}")
for _, r in d.iterrows():
    be = r.E1_exp_frac / r.mu_surv
    fe = r.FT_exp_frac / r.mu_surv
    print(f"{r.mu_surv:9.0e} {r.muon_rejection:12,.0f} {be:11.2f}x {fe:9.2f}x "
          f"{100*r.E1_sig_eff:12.1f}% {100*r.FT_sig_eff:10.1f}% {r.E1_n_mu_at_cut:9,.0f}")

print("\n=== (2) was the hard population removed from the test? ===")
ft = np.load(FKS).astype(np.int64)
con = duckdb.connect()
con.execute(f"ATTACH '{BASE}' AS b (READ_ONLY)")
con.execute(f"ATTACH '{ROOT}/data_manager/catalog_v2.duckdb' AS cat (READ_ONLY)")
con.execute("CREATE TEMP TABLE ftfk AS SELECT UNNEST(?) AS fk", [ft.tolist()])

n_eval = con.execute(
    "SELECT count(*) FROM b.predictions WHERE n_sn_hits>=8 AND n_sn_strings>=3").fetchone()[0]
n_pool = con.execute("""
    SELECT count(*) FROM b.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
    WHERE pr.n_sn_hits>=8 AND pr.n_sn_strings>=3 AND pr.score>0.0
      AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))""").fetchone()[0]
print(f"  evaluation set (h8s3)                 : {n_eval:>10,}")
print(f"  removed as fine-tuning background     : {len(ft):>10,}  "
      f"({100*len(ft)/n_eval:.2f}% of the evaluation set)")
print(f"  screened at all (pool, as of 8 July)  : {480186:>10,}  "
      f"({100*480186/n_eval:.1f}% of the evaluation set)")
print(f"  NEVER screened -> hard events kept    : {n_eval-480186:>10,}  "
      f"({100*(n_eval-480186)/n_eval:.1f}%)")
rate = len(ft) / 480186
print(f"  cut-passing rate inside the pool      : {100*rate:.2f}%")
print(f"  -> expected cut-passing events still in the test (same rate): "
      f"{rate*(n_eval-480186):,.0f}  (~{rate*(n_eval-480186)/len(ft):.1f}x more than removed)")

hi = con.execute("""
    SELECT sum(CASE WHEN t.fk IS NULL THEN 1 ELSE 0 END) AS kept,
           sum(CASE WHEN t.fk IS NOT NULL THEN 1 ELSE 0 END) AS removed
    FROM b.predictions pr LEFT JOIN ftfk t ON t.fk = pr.event_fk
    WHERE pr.n_sn_hits>=8 AND pr.n_sn_strings>=3 AND pr.score>0.8""").fetchone()
print(f"\n  base-model high-score events (xi>0.8, base scores only -- no cross-model threshold):")
print(f"    kept in the test : {hi[0]:>6,}  ({100*hi[0]/(hi[0]+hi[1]):.0f}%)")
print(f"    removed          : {hi[1]:>6,}  ({100*hi[1]/(hi[0]+hi[1]):.0f}%)")
