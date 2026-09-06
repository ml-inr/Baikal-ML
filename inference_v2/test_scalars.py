"""Check the stored scalars against the implementation the excess study was built on.

Two kinds of check, and they are worth very different amounts.

**Hand-computed values** are the real oracle: a four-hit event small enough that every
quantity can be worked out on paper, independently of any code here. That is what actually
establishes correctness.

**Agreement with the earlier implementation** (frozen below from the deleted
`analysis/exp_excess_investigation/build_dataset.py`) proves only that the formulas were
transcribed consistently when they moved. That builder was written in the same effort as
this code and its outputs were never used for any published number, so it is a sibling
implementation, not an authority. Kept because a differential check still catches typos.

`n_sig_strings` gets a different oracle on purpose. The builder counts strings from rounded
cluster-centred coordinates, which can merge strings sitting at the same position in
different clusters; the pipeline counts them from channel ids, and the pipeline's count is
what the h8s3 selection is made of. So it is checked against
`io.py:_count_sig_hits_strings` — the function that actually produced `n_sn_strings`.

The three new quantities have no oracle, so they are checked against a deliberately
different computation (Python sets and explicit sorting rather than numpy reductions).

Usage:
    python inference_v2/test_scalars.py [--n-events 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_v2.nu_classifier.compute_scalars import COLUMNS, event_scalars  # noqa: E402

MC_H5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
PROBS = ROOT / ("data_manager/data/h5datasets/"
                "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")
THRESHOLD = 0.8


# Frozen from the deleted builder. A sibling implementation, not ground truth — see the
# module docstring. Do not "improve" it, including its coordinate-based string count, which
# is precisely what compute_scalars.py deliberately departs from.
STRING_TOL = 10.0


def _reference_event_scalars(h: np.ndarray, p: np.ndarray) -> dict:
    """Per-event quantities derived from the noise-filtered hits (raw units)."""
    q, t, x, y, z = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4]
    qc = np.clip(q, 0, 100)
    sx, sy, sz = x.std(), y.std(), z.std()
    return {
        "n_sig_hits": len(h),
        # NOTE: strings counted from cluster-centred (x, y). The pipeline counts them
        # from channel ids (io.py:_count_sig_hits_strings), which is what defines the
        # h8s3 selection; coordinates can merge two strings at the same position in
        # different clusters. Kept as-is here because published numbers used it — use
        # inference_v2/nu_classifier/compute_scalars.py for new work.
        "n_sig_strings": len(np.unique(np.round(h[:, 2:4].astype(np.float64) / STRING_TOL),
                                       axis=0)),
        "q_mean": float(qc.mean()), "q_std": float(qc.std()),
        "q_total": float(qc.sum()), "q_max": float(qc.max()),
        "r_vert": float(sz / (np.sqrt(sx * sx + sy * sy) + 1e-3)),
        "t_span": float(t.max() - t.min()),
        "z_span": float(z.max() - z.min()),
        "xy_span": float(np.hypot(x.max() - x.min(), y.max() - y.min())),
        "x_c": float(x.mean()), "y_c": float(y.mean()), "z_c": float(z.mean()),
        "prob_mean": float(p.mean()), "prob_min": float(p.min()),
    }


def _load_reference():
    return _reference_event_scalars


def _sample(n_events: int):
    """Real events from a real part, selected exactly as the pipeline selects them."""
    with h5py.File(MC_H5, "r") as src, h5py.File(PROBS, "r") as pf:
        part = sorted(pf["muatm_2020/probs"].keys())[0]
        ev   = src[f"muatm_2020/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = src[f"muatm_2020/raw/data/{part}/data"][:].astype(np.float32)
        chan = src[f"muatm_2020/raw/channels/{part}/data"][:].astype(np.int32)
        prob = pf[f"muatm_2020/probs/{part}/data"][:].astype(np.float32)

    out = []
    for i in range(len(ev) - 1):
        s, e = int(ev[i]), int(ev[i + 1])
        m = prob[s:e] > THRESHOLD
        if m.sum() < 5:
            continue
        out.append((data[s:e][m], prob[s:e][m], chan[s:e][m]))
        if len(out) >= n_events:
            break
    return out


def test_strings_match_the_pipeline(events) -> bool:
    """n_sig_strings must equal what the scoring pipeline computes, not the builder."""
    from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

    i_str = COLUMNS.index("n_sig_strings")
    n_bad = 0
    for h, p, c in events:
        ours = event_scalars(h, p, c)[i_str]
        # the pipeline's vectorised routine, run on this one event
        _, theirs = _count_sig_hits_strings(
            np.ones(len(c), dtype=bool), c,
            np.array([0], dtype=np.int64), np.array([len(c)], dtype=np.int32), 1)
        n_bad += int(ours != int(theirs[0]))
    ok = n_bad == 0
    print(f"\n  n_sig_strings vs the pipeline's own routine, {len(events):,} events: "
          f"{'ok' if ok else f'FAIL — {n_bad} differ'}")
    return ok


def test_matches_reference(events) -> bool:
    ref_fn = _load_reference()
    shared = [c for c in COLUMNS
              if c in ref_fn(*events[0][:2]) and c != "n_sig_strings"]
    idx = {c: COLUMNS.index(c) for c in shared}
    worst = {c: 0.0 for c in shared}
    n_bad = 0

    for h, p, c in events:
        ours = event_scalars(h, p, c)
        theirs = ref_fn(h, p)
        for col in shared:
            a, b = float(ours[idx[col]]), float(theirs[col])
            d = abs(a - b)
            worst[col] = max(worst[col], d)
            if d != 0.0:
                n_bad += 1

    print(f"  reference agreement on {len(shared)} shared quantities, {len(events):,} events:")
    for col in shared:
        print(f"    {col:14s} max|delta| {worst[col]:.3e}")
    ok = n_bad == 0
    print(f"  {'ok — bit-identical' if ok else f'FAIL — {n_bad} differing values'}")
    return ok


def test_new_quantities(events) -> bool:
    """The three additions, against an independent computation."""
    ok = True
    i_nch, i_frac, i_core = (COLUMNS.index(c) for c in ("n_channels", "q_frac_max",
                                                        "t_span_core"))
    d_nch = d_frac = d_core = 0.0
    for h, p, c in events:
        s = event_scalars(h, p, c)
        d_nch = max(d_nch, abs(s[i_nch] - len(set(c.tolist()))))

        qc = sorted(min(max(float(v), 0.0), 100.0) for v in h[:, 0])
        tot = sum(qc)
        expect = (qc[-1] / tot) if tot > 0 else 0.0
        d_frac = max(d_frac, abs(s[i_frac] - expect))

        t = np.sort(h[:, 1].astype(np.float64))
        lo, hi = np.interp([0.05, 0.95], np.linspace(0, 1, len(t)), t)
        d_core = max(d_core, abs(s[i_core] - (hi - lo)))

    # t_span_core is compared relatively: the module takes percentiles of float32 times
    # while this check promotes them to float64 first, so an exact match is not expected.
    # Times run to thousands of ns, making an absolute tolerance meaningless — a logic
    # error would still show as a large relative difference.
    scale = max(1.0, max(float(np.ptp(h[:, 1])) for h, _, _ in events))
    print("\n  new quantities vs an independent computation:")
    for name, d, tol in [("n_channels", d_nch, 0.0), ("q_frac_max", d_frac, 1e-6),
                         ("t_span_core", d_core, 1e-6 * scale)]:
        good = d <= tol
        ok &= good
        print(f"    {name:14s} max|delta| {d:.3e}  {'ok' if good else 'FAIL'}")
    return ok


def test_clip_and_guard_behave() -> bool:
    """The two defensive constants must actually be doing something."""
    ok = True
    h = np.array([[500.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 10.0, 0.0, 0.0, 5.0]], dtype=np.float32)
    p = np.array([0.9, 0.95], dtype=np.float32)
    c = np.array([1, 2], dtype=np.int32)
    s = event_scalars(h, p, c)
    q_max = s[COLUMNS.index("q_max")]
    if q_max != 100.0:
        ok = False
    print(f"\n  charge clipped at 100 p.e.: q_max={q_max}  {'ok' if q_max == 100.0 else 'FAIL'}")

    # all hits on one string: the denominator of r_vert would be zero without the guard
    flat = np.array([[1.0, 0.0, 3.0, 4.0, 0.0], [1.0, 1.0, 3.0, 4.0, 10.0]], dtype=np.float32)
    r = event_scalars(flat, p, c)[COLUMNS.index("r_vert")]
    finite = np.isfinite(r)
    ok &= bool(finite)
    print(f"  single-string event gives finite r_vert={r:.1f}  {'ok' if finite else 'FAIL'}")
    return ok


def test_hand_computed() -> bool:
    """The actual oracle: an event small enough to work out on paper.

    Four hits, two strings (channels 0,1 -> string 0; 36,37 -> string 1), charges 2/4/6/8,
    times 0/100/200/300, positions (0,0,-10) (0,0,10) (30,40,-10) (30,40,10).
    """
    h = np.array([[2.0,   0.0,  0.0,  0.0, -10.0],
                  [4.0, 100.0,  0.0,  0.0,  10.0],
                  [6.0, 200.0, 30.0, 40.0, -10.0],
                  [8.0, 300.0, 30.0, 40.0,  10.0]], dtype=np.float32)
    p_ = np.array([0.90, 0.85, 0.95, 0.99], dtype=np.float32)
    c  = np.array([0, 1, 36, 37], dtype=np.int32)

    expect = {
        "n_sig_hits": 4,
        "n_sig_strings": 2,          # channels 0,1 -> 0 ; 36,37 -> 1
        "n_channels": 4,
        "q_mean": 5.0,               # (2+4+6+8)/4
        "q_std": 5.0 ** 0.5,         # sqrt((9+1+1+9)/4)
        "q_total": 20.0,
        "q_max": 8.0,
        "q_frac_max": 0.4,           # 8/20
        "r_vert": 10.0 / (25.0 + 1e-3),   # sz=10, sqrt(15^2+20^2)=25
        "t_span": 300.0,
        "t_span_core": 270.0,        # linear percentiles: 285 - 15
        "z_span": 20.0,
        "xy_span": 50.0,             # hypot(30, 40)
        "x_c": 15.0, "y_c": 20.0, "z_c": 0.0,
        "prob_mean": (0.90 + 0.85 + 0.95 + 0.99) / 4,
        "prob_min": 0.85,
    }
    got = event_scalars(h, p_, c)
    ok = True
    print("\n  hand-computed four-hit event:")
    for name, want in expect.items():
        have = float(got[COLUMNS.index(name)])
        good = abs(have - want) <= 1e-5 * max(1.0, abs(want))
        ok &= good
        print(f"    {name:14s} expected {want:>12.6f}  got {have:>12.6f}  "
              f"{'ok' if good else 'FAIL'}")
    missing = set(COLUMNS) - set(expect)
    if missing:
        ok = False
        print(f"    FAIL — no hand-computed value for: {sorted(missing)}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=2000)
    a = ap.parse_args()

    ev = _sample(a.n_events)
    print(f"  sampled {len(ev):,} real events\n")
    results = [test_hand_computed(), test_matches_reference(ev),
               test_strings_match_the_pipeline(ev), test_new_quantities(ev),
               test_clip_and_guard_behave()]
    print("\n" + ("ALL PASSED" if all(results) else "FAILED"))
    sys.exit(0 if all(results) else 1)
