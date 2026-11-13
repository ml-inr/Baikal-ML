#!/usr/bin/env python3
"""Render a real single-track MC event in the Baikal-GVD cluster.

The event is read from the test-parts NPY dataset (selected hits) and the raw
MC HDF5 (OM geometry and the true muon track).  Color encodes time and marker
area encodes charge; the target cylinder and true entry point are overlaid.

This is a data figure, not a schematic.  It is intended for the appendix or as
a cover image; the editable PowerPoint schematic remains a separate artifact.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
DATASET = Path(__file__).resolve().parents[2] / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_testparts"
H5 = Path(__file__).resolve().parents[2] / "data_manager/data/h5datasets/baikal_mc_merged.h5"
PAPER = "#FAFAF8"
INK = "#263238"
MUTED = "#6B777D"
GRID = "#D9DEE1"
CLUSTER = "#E8EEF2"
SIGMA = "#E07A2E"
BLUE = "#2F6DB3"
ISO = 0.28


def p3(x, y, z):
    return x, z - ISO * y


def load_event():
    offsets = np.load(DATASET / "offsets.npy", mmap_mode="r")
    features = np.load(DATASET / "features.npy", mmap_mode="r")
    energy = np.load(DATASET / "energy.npy", mmap_mode="r")
    status = np.load(DATASET / "energy_status.npy")
    nh = np.load(DATASET / "n_sig_hits.npy")
    keys = np.load(DATASET / "h5_part_keys.npy", allow_pickle=True)
    locals_ = np.load(DATASET / "h5_local_event_ids.npy", allow_pickle=True)

    candidates = np.flatnonzero(
        (status == 0) & np.isfinite(energy) & (nh >= 30) & (nh <= 100)
    )
    best = None
    for idx in candidates[:10000]:
        a, b = int(offsets[idx]), int(offsets[idx + 1])
        h = np.asarray(features[a:b])
        xy = h[:, 2:4]
        r = np.hypot(xy[:, 0], xy[:, 1])
        # central string lit + a real 8-string topology
        if r.min() >= 8.0 or np.unique(np.round(xy, 1), axis=0).shape[0] < 7:
            continue
        spread = float(np.std(xy[:, 0]) + np.std(xy[:, 1]))
        if best is None or spread > best[0]:
            best = (spread, int(idx))
    idx = best[1]
    a, b = int(offsets[idx]), int(offsets[idx + 1])
    sig = np.asarray(features[a:b])
    part = str(keys[idx])
    local = int(locals_[idx])

    with h5py.File(H5, "r") as f:
        g = f["nue2_2020"]
        starts = g[f"raw/ev_starts/{part}/data"][:].astype(np.int64)
        s, e = int(starts[local]), int(starts[local + 1])
        raw = np.asarray(g[f"raw/data/{part}/data"][s:e])
        ch = np.asarray(g[f"raw/channels/{part}/data"][s:e])
        cid = int(g[f"raw/cluster_ids/{part}/data"][local])
        cc = np.asarray(g["clusters_centers/data"][:])[cid]
        ms = g[f"muons_prty/mu_starts/{part}/data"][:].astype(np.int64)
        mu = np.asarray(g[f"muons_prty/individ/{part}/data"][ms[local]:ms[local + 1]])
    ref = mu[0, 2:5] - cc
    theta, phi = float(mu[0, 0]), float(mu[0, 1])
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return {
        "sig": sig,
        "raw": raw,
        "channels": ch,
        "part": part,
        "local": local,
        "energy": float(energy[idx]),
        "track_ref": ref,
        "track_dir": direction,
    }


def track_entry(ref, direction, radius=90.0):
    a = ref
    b = direction
    aa = b[0] ** 2 + b[1] ** 2
    bb = 2 * (a[0] * b[0] + a[1] * b[1])
    cc = a[0] ** 2 + a[1] ** 2 - radius ** 2
    roots = np.roots([aa, bb, cc])
    vals = [float(r.real) for r in roots if abs(r.imag) < 1e-8]
    # ``muons_prty`` stores the direction of travel.  The first crossing along
    # that direction is the entry point; the other root is the exit.
    return min(vals) if vals else None


def draw_cylinder(ax, radius, half_h, color, ls):
    # top full ellipse; bottom front half (front is the lower arc in this projection)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(radius * np.cos(th), half_h - ISO * radius * np.sin(th),
            color=color, ls=ls, lw=1.8, zorder=2)
    th = np.linspace(0, np.pi, 150)
    ax.plot(radius * np.cos(th), -half_h - ISO * radius * np.sin(th),
            color=color, ls=ls, lw=1.8, zorder=2)
    ax.plot([-radius, -radius], [-half_h, half_h], color=color, ls=ls, lw=1.8, zorder=2)
    ax.plot([radius, radius], [-half_h, half_h], color=color, ls=ls, lw=1.8, zorder=2)


def main() -> None:
    ev = load_event()
    sig = ev["sig"]
    raw = ev["raw"]
    ch = ev["channels"]
    q, t, x, y, z = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3], sig[:, 4]
    t_rel = t - np.median(t)
    s = 12 + 34 * np.log10(np.maximum(q, 1.0))

    fig = plt.figure(figsize=(14, 8.4), facecolor=PAPER)
    ax = fig.add_axes([0.05, 0.10, 0.62, 0.82])
    ax.set_facecolor(PAPER)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-135, 135)
    ax.set_ylim(-360, 365)

    # cluster volume and target volume
    ax.add_patch(Rectangle((-60, -265), 120, 530, facecolor=CLUSTER,
                           edgecolor="none", alpha=0.55, zorder=1))
    for r, h, col, ls in [(60, 265, GRID, "-"), (90, 295, SIGMA, (0, (5, 4)))]:
        draw_cylinder(ax, r, h, col, ls)

    # actual hit OMs (raw hits), then signal hits on top
    u, v = p3(raw[:, 2], raw[:, 3], raw[:, 4])
    ax.scatter(u, v, s=9, c="#C9D6DD", edgecolors="#7F929E", linewidths=0.25,
               zorder=4, label="hit OM")

    # strings from the raw hit positions
    for st in np.unique(ch // 36):
        m = (ch // 36) == st
        xm, ym = raw[m, 2].mean(), raw[m, 3].mean()
        zz = np.linspace(-265, 265, 2)
        ax.plot([xm, xm], zz - ISO * ym, color="#8FA1AC", lw=0.8, zorder=3)

    sc = ax.scatter(*p3(x, y, z), s=s, c=t_rel, cmap="viridis",
                    edgecolors="white", linewidths=0.4, zorder=8,
                    label="signal hit")

    # true muon track; long enough to enter the volume
    ref = ev["track_ref"]
    d = ev["track_dir"]
    tt = np.linspace(-500, 500, 2)
    pts = ref[None, :] + tt[:, None] * d[None, :]
    uv = np.array([p3(*p) for p in pts])
    ax.plot(uv[:, 0], uv[:, 1], color=INK, lw=3.4, zorder=9)
    ax.annotate("", xy=(uv[1, 0], uv[1, 1]), xytext=(uv[0, 0], uv[0, 1]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3),
                zorder=10)

    # true entry into the r=90 cylinder
    te = track_entry(ref, d, 90.0)
    if te is not None:
        pe = p3(*(ref + te * d))
        ax.add_patch(Circle(pe, radius=6.5, facecolor=SIGMA, edgecolor="white", lw=1.4, zorder=12))
        ax.annotate("E_entry", xy=pe, xytext=(70, 70), color=SIGMA,
                    fontsize=12, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=SIGMA, lw=1.2), zorder=20)
    else:
        pe = None

    ax.text(-128, 350, "Реальное MC-событие: цвет — время, размер — заряд",
            fontsize=17, fontweight="bold", ha="left")
    ax.text(-128, -345,
            "сигнальные хиты после sig-noise; серые кружки — OMs с raw-хитами\n"
            "штриховой цилиндр — target volume r ≤ 90 м, |z| ≤ 295 м",
            fontsize=9.5, color=MUTED, ha="left")
    # colorbar
    cax = fig.add_axes([0.73, 0.13, 0.22, 0.025])
    fig.colorbar(sc, cax=cax, orientation="horizontal")
    cax.set_xlabel("hit time − median, ns", fontsize=9, color=MUTED)
    cax.tick_params(labelsize=8, colors=MUTED)

    # right-side event info
    axr = fig.add_axes([0.73, 0.32, 0.25, 0.45])
    axr.axis("off")
    axr.text(0, 1.0,
             f"E_entry = {ev['energy']/1e3:.1f} TeV\n"
             f"status = 0\n"
             f"signal hits = {len(sig)}\n"
             f"raw hits = {len(raw)}\n"
             f"part = {ev['part']}",
             fontsize=11, color=INK, va="top", linespacing=1.5)
    axr.text(0, 0.53,
             "Цвет: время хита.\nРазмер: заряд в фотоэлектронах.\n"
             "Серая геометрия — реальные OMs события.",
             fontsize=9.5, color=MUTED, va="top", linespacing=1.45)

    fig.savefig(OUT / "real_event_display.svg", dpi=220, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / "real_event_display.png", dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("selected", ev["part"], ev["local"], "E", ev["energy"], "n_sig", len(sig))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
