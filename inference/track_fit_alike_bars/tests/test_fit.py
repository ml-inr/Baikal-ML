"""Self-checking script for inference/track_fit_alike_bars.

    python inference/track_fit_alike_bars/tests/test_fit.py

Each check anchors a ported piece to something outside the port: networkx for
the clique listing, the GSL shared library for the simplex, a literal emulation of
the C++ stream read for the pHit table, BARS' resource file for the defaults, and
BARS' own reconstruction of the same pulses for the whole chain.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.track_fit_alike_bars import FitConfig, fit_event, read_event_hits  # noqa: E402
from inference.track_fit_alike_bars.cliques import enumerate_cliques  # noqa: E402
from inference.track_fit_alike_bars.physics import PHIT_TABLE_PATH, phit_table  # noqa: E402
from inference.track_fit_alike_bars.scanfit import (  # noqa: E402
    MEST_MAX_ITER, MEST_STEP, MEST_TOLERANCE, _mest, minimize_mest,
)

RESULTS: list[tuple[str, str]] = []


def check(name):
    """Register a check.  The function may return a short note for the report."""
    def wrap(fn):
        try:
            note = fn()
            RESULTS.append((name, f"ok{'' if not note else '  -- ' + note}"))
        except SkipTest as e:
            RESULTS.append((name, f"skipped: {e}"))
        except AssertionError as e:
            RESULTS.append((name, f"FAILED: {e}"))
        return fn
    return wrap


class SkipTest(Exception):
    pass


def random_graph(rng, n, density):
    a = rng.random((n, n)) < density
    a = np.triu(a, 1)
    return a | a.T


# ── cliques ────────────────────────────────────────────────────────────────────
def tomita_reference(adj, min_size, delta=1, limit=10_000):
    """Line-by-line transliteration of MaximumCliqueSearch.cc, Python sets."""
    n = len(adj)
    nb = [set(np.flatnonzero(adj[i])) for i in range(n)]
    st = {"found": [], "max": delta, "abort": False}

    def dead_end(rp):
        if len(st["found"]) == limit:
            st["abort"] = True
            return True
        return rp == 0 or rp < min_size or rp < st["max"] - delta

    def search(R, P, X):
        if dead_end(len(R) + len(P)):
            return
        if not P and not X:
            st["found"].append(frozenset(R))
            st["max"] = max(st["max"], len(R))
            return
        pivot, best = None, 0
        for i in range(n):
            if i in P or i in X:
                if pivot is None:
                    pivot = i
                c = len(P & nb[i])
                if c > best:
                    pivot, best = i, c
        for i in sorted(P - nb[pivot]):
            search(R | {i}, P & nb[i], X & nb[i])
            if st["abort"]:
                return
            P = P - {i}
            X = X | {i}

    P0 = {i for i in range(n) if len(nb[i]) >= min_size}
    if len(P0) >= min_size and P0:
        search(set(), P0, set())
    return st["found"]


@check("cliques: full listing equals networkx.find_cliques")
def _():
    import networkx as nx
    rng = np.random.default_rng(1)
    for _ in range(60):
        n = int(rng.integers(8, 30))
        adj = random_graph(rng, n, rng.uniform(0.2, 0.7))
        cl, _, _ = enumerate_cliques(adj, 1, 10**6)
        ours = {frozenset(np.flatnonzero(c)) for c in cl}
        g = nx.from_numpy_array(adj.astype(int))
        ref = {frozenset(c) for c in nx.find_cliques(g) if len(c) >= 2}
        assert ours == ref, f"n={n}: {len(ours)} vs {len(ref)} cliques"


@check("cliques: pruned listing equals the C++ transliteration, in order")
def _():
    rng = np.random.default_rng(2)
    for _ in range(60):
        n = int(rng.integers(10, 45))
        adj = random_graph(rng, n, rng.uniform(0.3, 0.8))
        for min_size in (7, 8):
            cl, mx, _ = enumerate_cliques(adj, min_size)
            ours = [frozenset(np.flatnonzero(c)) for c in cl]
            ref = tomita_reference(adj, min_size)
            assert ours == ref, f"n={n} min={min_size}"
            assert mx == (max(map(len, ref)) if ref else 0)


# ── GSL simplex ────────────────────────────────────────────────────────────────
class GslVector(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("stride", ctypes.c_size_t),
                ("data", ctypes.POINTER(ctypes.c_double)),
                ("block", ctypes.c_void_p), ("owner", ctypes.c_int)]


FCN = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(GslVector), ctypes.c_void_p)


class GslFunction(ctypes.Structure):
    _fields_ = [("f", FCN), ("n", ctypes.c_size_t), ("params", ctypes.c_void_p)]


def load_gsl():
    for name in ("gslcblas", "gsl"):
        path = ctypes.util.find_library(name)
        if path is None:
            raise SkipTest(f"lib{name} not found")
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    lib.gsl_set_error_handler_off()
    for fn, res, args in [
        ("gsl_multimin_fminimizer_alloc", ctypes.c_void_p,
         [ctypes.c_void_p, ctypes.c_size_t]),
        ("gsl_multimin_fminimizer_set", ctypes.c_int,
         [ctypes.c_void_p, ctypes.POINTER(GslFunction), ctypes.c_void_p,
          ctypes.c_void_p]),
        ("gsl_multimin_fminimizer_iterate", ctypes.c_int, [ctypes.c_void_p]),
        ("gsl_multimin_fminimizer_size", ctypes.c_double, [ctypes.c_void_p]),
        ("gsl_multimin_fminimizer_minimum", ctypes.c_double, [ctypes.c_void_p]),
        ("gsl_multimin_fminimizer_x", ctypes.POINTER(GslVector), [ctypes.c_void_p]),
        ("gsl_multimin_fminimizer_free", None, [ctypes.c_void_p]),
        ("gsl_multimin_test_size", ctypes.c_int, [ctypes.c_double, ctypes.c_double]),
        ("gsl_vector_calloc", ctypes.c_void_p, [ctypes.c_size_t]),
        ("gsl_vector_set", None, [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double]),
        ("gsl_vector_free", None, [ctypes.c_void_p]),
    ]:
        getattr(lib, fn).restype = res
        getattr(lib, fn).argtypes = args
    kind = ctypes.c_void_p.in_dll(lib, "gsl_multimin_fminimizer_nmsimplex2")
    return lib, kind


def gsl_mest(lib, kind, xr, yr, zr, t, sigma):
    """BScanFitClique::MinimizeMEst driven through the real GSL."""
    x0i, y0i = xr.mean(), yr.mean()

    def f(v, _):
        vec = v.contents
        return _mest(vec.data[0], vec.data[vec.stride], x0i, y0i, xr, yr, zr, t,
                     sigma)

    cb = FCN(f)
    func = GslFunction(cb, 2, None)
    step, start = lib.gsl_vector_calloc(2), lib.gsl_vector_calloc(2)
    lib.gsl_vector_set(step, 0, MEST_STEP)
    lib.gsl_vector_set(step, 1, MEST_STEP)
    lib.gsl_vector_set(start, 0, x0i)
    lib.gsl_vector_set(start, 1, y0i)
    s = lib.gsl_multimin_fminimizer_alloc(kind, 2)
    lib.gsl_multimin_fminimizer_set(s, ctypes.byref(func), start, step)
    it, status = 0, -2
    while True:
        it += 1
        status = lib.gsl_multimin_fminimizer_iterate(s)
        if status:
            break
        size = lib.gsl_multimin_fminimizer_size(s)
        status = lib.gsl_multimin_test_size(size, MEST_TOLERANCE)
        if not (status == -2 and it < MEST_MAX_ITER):
            break
    x = lib.gsl_multimin_fminimizer_x(s).contents
    out = (status == 0, x.data[0], x.data[x.stride],
           lib.gsl_multimin_fminimizer_minimum(s))
    lib.gsl_multimin_fminimizer_free(s)
    lib.gsl_vector_free(step)
    lib.gsl_vector_free(start)
    return out


@check("scanfit simplex: identical to libgsl nmsimplex2")
def _():
    lib, kind = load_gsl()
    rng = np.random.default_rng(3)
    for _ in range(300):
        m = int(rng.integers(7, 25))
        xr = rng.normal(0, 30, m)
        yr = rng.normal(0, 30, m)
        zr = rng.uniform(-250, 250, m)
        x0, y0 = rng.normal(0, 10, 2)
        rho = np.hypot(xr - x0, yr - y0)
        from inference.track_fit_alike_bars.physics import C_VACUUM, K_SCANFIT
        t = -zr / C_VACUUM + rho / C_VACUUM * K_SCANFIT + rng.normal(0, 5, m)
        t[rng.random(m) < 0.2] += rng.normal(0, 200)
        ours = minimize_mest(xr.mean(), yr.mean(), xr, yr, zr, t, 3.0)
        ref = gsl_mest(lib, kind, xr, yr, zr, t, 3.0)
        assert ours[0] == ref[0], (ours, ref)
        assert np.allclose(ours[1:], ref[1:], rtol=0, atol=1e-9), (ours, ref)


# ── pHit table ────────────────────────────────────────────────────────────────
@check("pHit table: equals a literal emulation of the C++ read loop")
def _():
    import re
    text = PHIT_TABLE_PATH.read_text()
    read_int = re.compile(r"\s*([+-]?\d+)")                       # `>> Int_t`
    read_double = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

    def cpp_lookup(dist):
        """for (i<1000) { ifile >> dd >> nAvg; if (floor(d+0.5)==dd) break; }"""
        pos, n_avg = 0, -1.0
        for _ in range(1000):
            m = read_int.match(text, pos)
            dd, pos = int(m.group(1)), m.end()
            m = read_double.match(text, pos)
            n_avg, pos = float(m.group(1)), m.end()
            if np.floor(dist + 0.5) == dd:
                break
        return n_avg

    table = phit_table()
    from inference.track_fit_alike_bars.physics import lookup_navg
    for dist in [0.2, 0.6, 1.4, 7.5, 21.9, 99.5, 500.2, 996.6, 997.4, 998.6,
                 1000.2, 1001.1, 1200.0]:
        assert lookup_navg(dist, table) == cpp_lookup(dist), dist


# ── configuration ─────────────────────────────────────────────────────────────
@check("config: defaults equal BARS' 2020 DATA and MC resource files")
def _():
    base = ROOT / "bars" / "config" / "2020" / "resources"
    files = [base / "reco-muon_scl_DATA.rc", base / "reco-muon_scl_MC.rc"]
    files = [f for f in files if f.exists()]
    if not files:
        raise SkipTest("bars/config/2020/resources not present")
    for f in files:
        assert FitConfig.from_bars_rc(f) == FitConfig(), f.name


# ── channel masks ─────────────────────────────────────────────────────────────
@check("masks: the DQM solution file sits inside the ROOT mask BARS used")
def _():
    """The solution file is the run-level part; the ROOT container adds sections.

    So every channel the solution switches off must also be off in the container,
    and the container may switch off more (whole sections, per event).  Checked on
    every `exp_reco` run whose DQM solution is unpacked locally.
    """
    import h5py
    from inference.reader.spec import SOURCES
    from inference.track_fit_alike_bars import masks

    spec = SOURCES["exp_reco"]
    if not spec.h5.exists():
        raise SkipTest("exp_reco.h5 not present")
    runs, events, all_on, extra = 0, 0, 0, []
    with h5py.File(spec.h5, "r") as f:
        g = f["exp_reco"]
        for part in sorted(g["reco_prty"].keys()):
            season, cluster, run = masks.part_run(part)
            solution = masks.solution_path(season, cluster, run)
            if solution is None:
                continue
            root = masks.reco_root_path(part)
            if not root.exists():
                raise AssertionError(f"{part}: ROOT file not unpacked, cannot check")
            # spread over the run: the section mask changes from event to event,
            # so the first few events of a run would understate its share
            header = g["header_prty"][part]["data"]
            take = np.unique(np.linspace(0, len(header) - 1, 20).astype(int))
            ids = header[take, 3]
            solved = masks.dqm_mask_from_solution(solution)
            for eid, mask in masks.dqm_masks_from_root(root, ids).items():
                if mask.all():
                    all_on += 1          # the first event of a run, before DQM
                    continue
                leaked = int(np.sum(~solved & mask))
                assert leaked == 0, (
                    f"{part} event {eid}: {leaked} channels the solution switches "
                    f"off are on in the ROOT mask")
                events += 1
                extra.append(int(np.sum(~mask & solved)))
            runs += 1
    assert events > 0, "no run had both files"
    return (f"{runs} runs, {events} events; the section part adds "
            f"{min(extra)}-{max(extra)} channels, {all_on} events all-on")


# ── the chain ─────────────────────────────────────────────────────────────────
PART = "part_s2020_c04_r0100"


@check("chain: result does not depend on the order hits are passed in")
def _():
    hits = read_event_hits("exp_reco", "exp_reco", PART, np.array([3]))[0]
    a = fit_event(*hits)
    perm = np.random.default_rng(4).permutation(len(hits[0]))
    b = fit_event(*(h[perm] for h in hits))
    assert a.status == b.status == "ok"
    assert a.scalars() == b.scalars() or np.allclose(
        [a.theta_rec, a.phi_rec, a.time_chi2], [b.theta_rec, b.phi_rec, b.time_chi2])
    assert np.array_equal(a.fit_hits, perm[b.fit_hits]) or \
        set(a.fit_hits) == set(perm[b.fit_hits])


@check("chain: the channel mask removes exactly the masked hits from the fit")
def _():
    hits = read_event_hits("exp_reco", "exp_reco", PART, np.array([3]))[0]
    mask = np.ones(300, dtype=bool)
    plain = fit_event(*hits)
    full = fit_event(*hits, channel_mask=mask)
    assert plain.status == full.status == "ok"
    assert np.array_equal(plain.fit_hits, full.fit_hits), "an all-on mask changed the fit"
    # switch off a channel the fit used: its hits must leave the fit
    used = set(hits[5][plain.fit_hits].tolist())
    victim = sorted(used)[0]
    mask[victim] = False
    masked = fit_event(*hits, channel_mask=mask)
    assert masked.status == "ok"
    assert victim not in set(hits[5][masked.fit_hits].tolist())
    assert masked.n_hits < plain.n_hits or masked.n_hits_clique < plain.n_hits_clique


@check("chain: a fit whose covariance has a negative variance survives, as in C++")
def _():
    """BARS does `sqrt(err[i])` and C++ returns NaN; Python raises instead.

    A valid Minuit minimum can carry a covariance that is not positive definite, so
    the branch is reachable in production, not hypothetical -- it stopped a real
    run of this port on `exp_full`.
    """
    import inference.track_fit_alike_bars.candidate as candidate
    from inference.track_fit_alike_bars.candidate import error_deg

    assert np.isnan(error_deg(-1e-6)), "a negative variance must give NaN"
    assert error_deg(0.0) == 0.0
    assert abs(error_deg(1.0) - 180 / np.pi) < 1e-12

    hits = read_event_hits("exp_reco", "exp_reco", PART, np.array([3]))[0]
    original = candidate.minuit2_minimize

    def broken(fcn, start, limits):
        out = original(fcn, start, limits)
        out.cov = np.array(out.cov)
        out.cov[2, 2] = -1.0
        out.cov[3, 3] = -2.0
        return out

    candidate.minuit2_minimize = broken
    try:
        fit = fit_event(*hits)
    finally:
        candidate.minuit2_minimize = original
    assert fit.status == "ok", fit.status
    assert np.isnan(fit.theta_err) and np.isnan(fit.phi_err)


@check("chain: agrees with BARS' own reconstruction on three events")
def _():
    import h5py
    from inference.reader.schema import EXP_RECO_COLUMNS
    path = ROOT / "data_manager/data/h5datasets/exp_reco.h5"
    if not path.exists():
        raise SkipTest("exp_reco.h5 not present")
    rows = [1, 3, 4]
    with h5py.File(path, "r") as f:
        reco = f["exp_reco/reco_prty"][PART]["data"][:max(rows) + 1]
    hits = read_event_hits("exp_reco", "exp_reco", PART, np.array(rows))
    for r, h in zip(rows, hits):
        bars = dict(zip(EXP_RECO_COLUMNS, reco[r]))
        fit = fit_event(*h)
        assert fit.status == "ok", (r, fit.status)
        assert fit.n_hits == bars["nHits"], (r, fit.n_hits, bars["nHits"])
        assert abs(fit.theta_rec - bars["thetaRec"]) < 0.1, r
        assert abs(fit.phi_rec - bars["phiRec"]) < 0.1, r
        assert abs(fit.time_chi2 - bars["timeChi2"]) < 0.01 * bars["timeChi2"], r
        assert fit.cov_matrix_status == bars["covMatrixStatus"], r


if __name__ == "__main__":
    width = max(len(n) for n, _ in RESULTS)
    for name, outcome in RESULTS:
        print(f"{name:<{width}}  {outcome}")
    failed = [n for n, o in RESULTS if o.startswith("FAILED")]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed or skipped")
    sys.exit(1 if failed else 0)
