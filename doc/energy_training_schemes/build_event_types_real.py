#!/usr/bin/env python3
"""Real-data contact sheet for accepted/rejected energy events.

Each panel is a real MC event, not a drawing: grey OMs are reconstructed from
raw hit positions, signal hits are coloured by time and scaled by charge, and
the true muon track(s) come from ``muons_prty``.  The target cylinder is
overlaid for context.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).resolve().parent / "figures/event_types_real"
H5 = "data_manager/data/h5datasets/baikal_mc_merged.h5"
COMP = "data_manager/data/h5datasets/baikal_mc_merged_energy_truth.h5"
PROBS = "data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
PAPER = "#FAFAF8"
INK = "#263238"
MUTED = "#6B777D"
GRID = "#D9DEE1"
CLUSTER = "#E8EEF2"
SIGMA = "#E07A2E"
GOOD = "#2E7D4F"
BAD = "#C0392B"
ISO = 0.28

# Explicit real events.  These are checked against probs/status below.
EXAMPLES = [
    ("nue2_2020", "part_2050", 18841, "cross", True, "пересекает target"),
    ("nue2_2020", "part_2050", 16621, "born", None, "родился внутри"),
    ("nue2_2020", "part_2050", 2474, "stopped", False, "умер до target"),
    ("nue2_2020", "part_2050", 19047, "missed", False, "прошёл мимо"),
    ("muatm_2020", "part_1205", 0, "multi", False, "несколько мюонов"),
    ("nue2_2020", "part_2050", 0, "sparse", False, "< 8 signal hits"),
]


def p3(x, y, z):
    return x, z - ISO * y


def load(ptype, part, local):
    with h5py.File(H5, "r") as f:
        g = f[ptype]
        starts = g[f"raw/ev_starts/{part}/data"][:].astype(np.int64)
        s, e = int(starts[local]), int(starts[local + 1])
        raw = np.asarray(g[f"raw/data/{part}/data"][s:e])
        ch = np.asarray(g[f"raw/channels/{part}/data"][s:e])
        cid = int(g[f"raw/cluster_ids/{part}/data"][local])
        cc = np.asarray(g["clusters_centers/data"][:])[cid]
        ms = g[f"muons_prty/mu_starts/{part}/data"][:].astype(np.int64)
        mu = np.asarray(g[f"muons_prty/individ/{part}/data"][ms[local]:ms[local + 1]])
    with h5py.File(PROBS, "r") as f:
        prob = np.asarray(f[ptype]["probs"][part]["data"][s:e], dtype=float)
        ns = int(f[ptype]["n_sig_hits_0.8"][part]["data"][local])
        nss = int(f[ptype]["n_sig_strings_0.8"][part]["data"][local])
    with h5py.File(COMP, "r") as f:
        st = int(f[ptype]["e_at_entry_status"][part]["data"][local, 1])
        en = float(f[ptype]["e_at_entry"][part]["data"][local, 1])
    mask = prob > 0.8
    return dict(
        ptype=ptype, part=part, local=local, raw=raw, ch=ch, sig=raw[mask],
        prob=prob, status=st, energy=en, cc=cc, mu=mu,
        ns=ns, nss=nss,
    )


def cross_track(ref, d, radius=90.0):
    a = ref
    b = d
    roots = np.roots([
        b[0] ** 2 + b[1] ** 2,
        2 * (a[0] * b[0] + a[1] * b[1]),
        a[0] ** 2 + a[1] ** 2 - radius ** 2,
    ])
    vals = [float(r.real) for r in roots if abs(r.imag) < 1e-8]
    return min(vals) if vals else None


def draw_panel(ax, ev, kind, accepted, label):
    raw, sig, ch, mu = ev["raw"], ev["sig"], ev["ch"], ev["mu"]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-135, 135)
    ax.set_ylim(-350, 330)

    # target / cluster volumes
    ax.add_patch(Rectangle((-60, -265), 120, 530, facecolor=CLUSTER,
                           edgecolor="none", alpha=0.6, zorder=1))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(90 * np.cos(th), 295 - ISO * 90 * np.sin(th), color=SIGMA,
            ls=(0, (5, 4)), lw=1.2, zorder=2)
    th = np.linspace(0, np.pi, 150)
    ax.plot(90 * np.cos(th), -295 - ISO * 90 * np.sin(th), color=SIGMA,
            ls=(0, (5, 4)), lw=1.2, zorder=2)
    ax.plot([-90, -90], [-295, 295], color=SIGMA, ls=(0, (5, 4)), lw=1.2, zorder=2)
    ax.plot([90, 90], [-295, 295], color=SIGMA, ls=(0, (5, 4)), lw=1.2, zorder=2)
    ax.plot(60 * np.cos(th), 265 - ISO * 60 * np.sin(th), color=GRID, lw=0.9, zorder=2)
    ax.plot(60 * np.cos(th), -265 - ISO * 60 * np.sin(th), color=GRID, lw=0.9, zorder=2)

    # raw OMs
    x, y, z = raw[:, 2], raw[:, 3], raw[:, 4]
    u, v = p3(x, y, z)
    ax.scatter(u, v, s=7, c="#C9D6DD", edgecolors="#7F929E", linewidths=0.2,
               zorder=3)
    # strings from raw hit positions
    for st in np.unique(ch // 36):
        m = (ch // 36) == st
        xm, ym = raw[m, 2].mean(), raw[m, 3].mean()
        ax.plot([xm, xm], [-265 - ISO * ym, 265 - ISO * ym],
                color="#8FA1AC", lw=0.55, zorder=2)

    # signal hits, colour/time and size/charge
    if len(sig):
        t = sig[:, 1]
        q = sig[:, 0]
        t_rel = t - np.median(t)
        norm = mcolors.Normalize(vmin=float(t_rel.min()), vmax=float(t_rel.max()) + 1e-6)
        sizes = 12 + 28 * np.log10(np.maximum(q, 1.0))
        ax.scatter(*p3(sig[:, 2], sig[:, 3], sig[:, 4]), s=sizes, c=t_rel,
                   cmap="viridis", norm=norm, edgecolors="white",
                   linewidths=0.35, zorder=8)

    # true track(s) only where they help the topology; a dead/missing track is
    # not drawn as a full line through the target (that would contradict status 2/3).
    tracks_to_draw = mu if kind in ("cross", "born", "multi") else []
    for track in tracks_to_draw:
        theta, phi = float(track[0]), float(track[1])
        d = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        ref = track[2:5] - ev["cc"]
        roots = []
        a = ref
        b = d
        for rr in np.roots([
            b[0] ** 2 + b[1] ** 2,
            2 * (a[0] * b[0] + a[1] * b[1]),
            a[0] ** 2 + a[1] ** 2 - 90 ** 2,
        ]):
            if abs(rr.imag) < 1e-8:
                roots.append(float(rr.real))
        if roots:
            lo, hi = min(roots) - 75, max(roots) + 75
        else:
            lo, hi = -180, 180
        pts = ref[None, :] + np.array([lo, hi])[:, None] * d[None, :]
        uv = np.array([p3(*p) for p in pts])
        ax.plot(uv[:, 0], uv[:, 1], color=INK, lw=1.8, zorder=9)

    # entry marker for the accepted crossing
    if kind == "cross" and ev["status"] == 0 and np.isfinite(ev["energy"]):
        tr = mu[0]
        theta, phi = float(tr[0]), float(tr[1])
        d = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi), np.cos(theta)])
        ref = tr[2:5] - ev["cc"]
        te = cross_track(ref, d)
        if te is not None:
            pe = p3(*(ref + te * d))
            ax.add_patch(Circle(pe, radius=4.8, facecolor=SIGMA,
                                edgecolor="white", lw=1.0, zorder=12))

    # badge
    color = GOOD if accepted is True else (BAD if accepted is False else SIGMA)
    sym = "✓" if accepted is True else ("✗" if accepted is False else "*")
    ax.set_title(f"{label}", fontsize=10.5, fontweight="bold", pad=4)
    ax.text(0, -330, f"{sym} {_reason(kind, ev)}", ha="center", va="center",
            color=color, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, lw=0.9))


def _reason(kind, ev):
    if kind == "cross":
        return "есть E_entry"
    if kind == "born":
        return "E при рождении"
    if kind == "stopped":
        return "умер до target"
    if kind == "missed":
        return "нет пересечения"
    if kind == "multi":
        return "не один мюон"
    return f"n_sig={ev['ns']}, n_str={ev['nss']}"


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (ptype, part, local, kind, accepted, label) in zip(axes.flat, EXAMPLES):
        ev = load(ptype, part, local)
        draw_panel(ax, ev, kind, accepted, label)
        print(kind, ptype, part, local, "status", ev["status"],
              "sig", ev["ns"], "strings", ev["nss"], "n_mu", len(ev["mu"]))
    fig.patch.set_facecolor(PAPER)
    fig.suptitle("Реальные MC-события: что проходит отбор энергии",
                 fontsize=21, fontweight="bold", x=0.035, y=0.985, ha="left")
    fig.text(0.04, 0.912,
             "Каждая панель — реальное событие; цвет — время, размер — заряд. "
             "Штриховой цилиндр — target volume.",
             fontsize=11.0, color=MUTED, ha="left")
    fig.text(0.04, 0.015,
             "* born inside: target = энергия при рождении; nuatm born-inside исключены; "
             "nue2 — разрешены.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.075, wspace=0.04, hspace=0.20)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"), dpi=220, facecolor=PAPER)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
