"""Maximal-clique enumeration as BARS v2.3.2 does it.

Port of `bfilter/MaximumCliqueSearch.cc` (`ResolveCliquesTomita`, mode
`EnumerateMaximalCliques`) and `BARS::Caus::SelectFromMaxClique`.

This is not a plain Bron-Kerbosch listing.  BARS prunes in ways that make the
result depend on vertex order, and the pruning is reproduced exactly:

* only vertices of degree >= `min_size` enter the search (`initRPX`) -- so a clique
  of exactly `min_size` vertices, whose members have degree `min_size - 1`, is
  never found;
* a branch is dropped when |R| + |P| < `min_size`, or when it is smaller than the
  largest clique found **so far** minus `delta` (`isDeadEndConditionSatisfied`) --
  so which near-maximal cliques are listed depends on the order they are met;
* the search aborts after 10,000 cliques (`maximalCliqueVectorThreshold`) and
  keeps what it has, with an overflow flag.

The recursion is written as an explicit stack; the order of visits is the
recursive one (pivot = first vertex of P|X with the most neighbours in P, then
the candidates P \\ N(pivot) in increasing index).  `tests/test_fit.py` checks it
against a direct transliteration of the C++ and against networkx.
"""
from __future__ import annotations

import numpy as np
from numba import njit

#: CliquesEnumeration::maximalCliqueVectorThreshold
MAX_CLIQUES = 10_000
#: CliquesEnumeration::maximal_clique_size_delta
SIZE_DELTA = 1


@njit(cache=True)
def enumerate_cliques(adj: np.ndarray, min_size: int, delta: int = SIZE_DELTA,
                      max_cliques: int = MAX_CLIQUES):
    """Maximal cliques of a boolean adjacency matrix, in BARS' order.

    Returns `(cliques, max_size, overflow)`: `cliques` is a (k, n) boolean array
    in the order found, `max_size` the largest size found (0 if none -- BARS'
    `result::operator=`), `overflow` whether the 10,000 limit was hit.
    """
    n = adj.shape[0]
    capacity = 64
    found = np.zeros((capacity, n), dtype=np.bool_)
    n_found = 0
    max_size = delta          # CliquesEnumeration::maximum_clique_size initial value
    overflow = False

    p0 = np.zeros(n, dtype=np.bool_)
    n_p0 = 0
    for i in range(n):
        if adj[i].sum() >= min_size:
            p0[i] = True
            n_p0 += 1
    if n_p0 < min_size or n_p0 == 0:
        return found[:0], 0, False

    depth_max = n + 2
    R = np.zeros((depth_max, n), dtype=np.bool_)
    P = np.zeros((depth_max, n), dtype=np.bool_)
    X = np.zeros((depth_max, n), dtype=np.bool_)
    cand = np.zeros((depth_max, n), dtype=np.bool_)
    pos = np.zeros(depth_max, dtype=np.int64)
    P[0] = p0

    depth = 0
    entering = True
    while depth >= 0:
        if entering:
            entering = False
            if n_found == max_cliques:
                overflow = True
                break
            rp = R[depth].sum() + P[depth].sum()
            if rp == 0 or rp < min_size or rp < max_size - delta:
                depth -= 1
                continue
            if not P[depth].any() and not X[depth].any():
                if n_found == capacity:
                    capacity *= 2
                    grown = np.zeros((capacity, n), dtype=np.bool_)
                    grown[:n_found] = found[:n_found]
                    found = grown
                found[n_found] = R[depth]
                n_found += 1
                size = R[depth].sum()
                if size > max_size:
                    max_size = size
                depth -= 1
                continue
            # Tomita pivot: first vertex of P|X, replaced only by a strictly
            # larger |P & N(u)|.
            pivot = -1
            best = 0
            for i in range(n):
                if P[depth, i] or X[depth, i]:
                    if pivot < 0:
                        pivot = i
                    c = 0
                    for j in range(n):
                        if P[depth, j] and adj[i, j]:
                            c += 1
                    if c > best:
                        pivot = i
                        best = c
            for j in range(n):
                cand[depth, j] = P[depth, j] and not adj[pivot, j]
            pos[depth] = 0

        i = pos[depth]
        while i < n and not cand[depth, i]:
            i += 1
        if i == n:
            depth -= 1
            continue
        pos[depth] = i + 1
        child = depth + 1
        for j in range(n):
            R[child, j] = R[depth, j]
            P[child, j] = P[depth, j] and adj[i, j]
            X[child, j] = X[depth, j] and adj[i, j]
        R[child, i] = True
        P[depth, i] = False
        X[depth, i] = True
        depth = child
        entering = True

    if n_found == 0:
        return found[:0], 0, overflow
    return found[:n_found], max_size, overflow


def select_from_max_clique(cliques: np.ndarray, tolerance: int) -> np.ndarray:
    """Union of the cliques no more than `tolerance` smaller than the largest.

    BARS::Caus::SelectFromMaxClique.  The largest size is taken over the listed
    cliques, which is what BARS does.
    """
    n = cliques.shape[1]
    if cliques.shape[0] == 0:
        return np.zeros(n, dtype=bool)
    sizes = cliques.sum(axis=1)
    keep = sizes >= sizes.max() - tolerance
    return cliques[keep].any(axis=0)
