#!/usr/bin/env python3
"""Build the two explanatory figures for the muon-energy training slides.

The figures are deliberately schematic.  They explain the *definition of the
target* and the event topologies accepted/rejected by the energy dataset; they
are not event displays and are not to scale.

Run from the repository root:
    python doc/energy_training_schemes/build_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"

INK = "#263238"
MUTED = "#6B777D"
GRID = "#D9DEE1"
CLUSTER = "#E8EEF2"
ENERGY = "#2F6DB3"
SIGMA = "#E07A2E"
DOMAIN = "#3A8A5A"
BAD = "#C0392B"
GOOD = "#2E7D4F"
PAPER = "#FAFAF8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
    }
)


def _box(ax, x, y, text, color=INK, size=12, weight="normal", ha="center"):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va="center",
        zorder=20,
    )


def _cluster(ax, mini=False):
    """Draw a schematic cluster and its sensitive target cylinder."""
    # target volume: r <= 90 m, |z| <= 295 m
    ax.add_patch(
        Rectangle(
            (-90, -295),
            180,
            590,
            fill=False,
            edgecolor=SIGMA,
            linestyle=(0, (5, 4)),
            linewidth=2.2,
            zorder=2,
        )
    )
    # cluster itself: r <= 60 m, |z| <= 265 m
    ax.add_patch(
        Rectangle(
            (-60, -265),
            120,
            530,
            facecolor=CLUSTER,
            edgecolor=GRID,
            linewidth=1.4,
            zorder=1,
        )
    )
    # 8 strings; fewer OMs in mini panels
    xs = np.linspace(-60, 60, 8)
    om_ys = np.linspace(-235, 235, 7 if not mini else 5)
    for x in xs:
        ax.plot([x, x], [-265, 265], color="#9AA9B2", lw=1.1, zorder=3)
        for y in om_ys:
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=4.2 if not mini else 3.2,
                    facecolor="#C9D6DD",
                    edgecolor="#8EA1AC",
                    lw=0.7,
                    zorder=4,
                )
            )
    if not mini:
        ax.annotate(
            "cluster\nr ≤ 60 m, |z| ≤ 265 m",
            xy=(-60, -265),
            xytext=(-190, -330),
            color=MUTED,
            fontsize=10.5,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=GRID, lw=1),
        )
        ax.annotate(
            "target volume\nr ≤ 90 m, |z| ≤ 295 m",
            xy=(90, 220),
            xytext=(112, 285),
            color=SIGMA,
            fontsize=10.5,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=SIGMA, lw=1),
        )


def _track(ax, start, end, color=INK, lw=4.0, label=None, arrow=True):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=8,
    )
    if arrow:
        ax.add_patch(
            FancyArrowPatch(
                (0.72 * start[0] + 0.28 * end[0], 0.72 * start[1] + 0.28 * end[1]),
                (0.55 * start[0] + 0.45 * end[0], 0.55 * start[1] + 0.45 * end[1]),
                arrowstyle="-|>",
                mutation_scale=16,
                color=color,
                lw=1.5,
                zorder=9,
            )
        )
    if label:
        ax.text(
            0.2 * start[0] + 0.8 * end[0],
            0.2 * start[1] + 0.8 * end[1] + 18,
            label,
            color=color,
            fontsize=14,
            fontweight="bold",
            ha="center",
            zorder=20,
        )


def _hits(ax, rng, start, end, n=55, mini=False):
    """Small deterministic hit cloud near the track inside the cluster."""
    t = rng.uniform(0.12, 0.78, n)
    x = start[0] + t * (end[0] - start[0]) + rng.normal(0, 13, n)
    y = start[1] + t * (end[1] - start[1]) + rng.normal(0, 20, n)
    keep = (np.abs(x) < 58) & (np.abs(y) < 260)
    ax.scatter(
        x[keep],
        y[keep],
        s=9 if mini else 16,
        c="#4C9BB8",
        alpha=0.75,
        linewidths=0,
        zorder=7,
    )


def target_definition() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(-195, 195)
    ax.set_ylim(-485, 470)
    ax.set_aspect("equal")
    ax.axis("off")
    _cluster(ax)

    start = (-175, 370)
    end = (135, -260)
    _track(ax, start, end, color=INK, lw=5)
    _hits(ax, np.random.RandomState(7), start, end)

    # crossing of the dotted target cylinder
    # parametrise the line and find x=+90
    t = (90 - start[0]) / (end[0] - start[0])
    entry = (90, start[1] + t * (end[1] - start[1]))
    ax.add_patch(
        Circle(entry, radius=8, facecolor=SIGMA, edgecolor="white", lw=2, zorder=12)
    )
    ax.annotate(
        "E_entry\nэнергия мюона\nна входе в target volume",
        xy=entry,
        xytext=(112, -15),
        color=SIGMA,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=SIGMA, lw=1.6),
        zorder=20,
    )

    # small direction-of-energy-loss cue
    ax.annotate(
        "энергия убывает\nвдоль трека",
        xy=(0.62 * start[0] + 0.38 * end[0], 0.62 * start[1] + 0.38 * end[1]),
        xytext=(-165, 120),
        color=MUTED,
        fontsize=10.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
        zorder=20,
    )

    _box(ax, -180, 350, "μ", color=INK, size=17, weight="bold", ha="left")
    _box(ax, -160, 324, "один мюонный трек", color=MUTED, size=10.5, ha="left")

    ax.text(
        -190,
        455,
        "Что мы называем энергией мюона?",
        fontsize=19,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        -190,
        -468,
        "Цель — энергия при входе в цилиндр r ≤ 90 м, |z| ≤ 295 м вокруг своего кластера.\n"
        "Это не энергия нейтрино и не полная энергия, выделенная в детекторе.\n"
        "Точки — черенковские хиты; линия — трек мюона (схематично).",
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="bottom",
        linespacing=1.45,
    )

    # top view inset
    inset = fig.add_axes([0.735, 0.18, 0.20, 0.24])
    inset.set_aspect("equal")
    inset.axis("off")
    inset.set_xlim(-125, 125)
    inset.set_ylim(-125, 125)
    inset.add_patch(Circle((0, 0), 60, fill=True, facecolor=CLUSTER, edgecolor=GRID, lw=1.5))
    inset.add_patch(Circle((0, 0), 90, fill=False, edgecolor=SIGMA, ls=(0, (4, 3)), lw=1.8))
    inset.plot([-115, 105], [95, -80], color=INK, lw=2.8, solid_capstyle="round")
    inset.add_patch(Circle((62, -27), radius=4, facecolor=SIGMA, edgecolor="white", lw=1))
    inset.text(-118, 105, "вид сверху", fontsize=9, color=MUTED, ha="left")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.07)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(
            OUT / f"target_definition.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor=PAPER,
        )
    plt.close(fig)


def _mini_cluster(ax):
    ax.set_xlim(-155, 155)
    ax.set_ylim(-375, 330)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (-90, -295),
            180,
            590,
            fill=False,
            edgecolor=SIGMA,
            linestyle=(0, (4, 3)),
            lw=1.4,
        )
    )
    ax.add_patch(
        Rectangle((-60, -265), 120, 530, facecolor=CLUSTER, edgecolor=GRID, lw=1)
    )
    for x in np.linspace(-60, 60, 5):
        ax.plot([x, x], [-265, 265], color="#9AA9B2", lw=0.9)
        for y in np.linspace(-210, 210, 4):
            ax.add_patch(Circle((x, y), radius=2.7, facecolor="#C9D6DD", edgecolor="#8EA1AC", lw=0.5))


def _badge(ax, accepted: bool | None, text: str):
    color = GOOD if accepted is True else (BAD if accepted is False else SIGMA)
    ax.add_patch(
        FancyBboxPatch(
            (-145, -350),
            300,
            32,
            boxstyle="round,pad=0.3,rounding_size=8",
            facecolor="white",
            edgecolor=color,
            lw=1.2,
        )
    )
    ax.text(
        0,
        -334,
        ("✓ " if accepted is True else ("✗ " if accepted is False else "* ")) + text,
        color=color,
        fontsize=7.4,
        ha="center",
        va="center",
        fontweight="bold",
    )


def _panel(ax, title, kind):
    _mini_cluster(ax)
    if kind == "cross":
        _track(ax, (-118, 300), (92, -280), lw=3.0)
        ax.add_patch(Circle((90, -160), radius=5, facecolor=SIGMA, edgecolor="white", lw=1))
        _badge(ax, True, "есть E_entry")
    elif kind == "inside":
        _track(ax, (-28, -70), (100, -265), lw=3.0)
        ax.add_patch(Circle((-28, -70), radius=5, facecolor=SIGMA, edgecolor="white", lw=1))
        _badge(ax, None, "E при рождении")
    elif kind == "stopped":
        _track(ax, (-118, 300), (-104, 55), lw=3.0)
        ax.add_patch(Circle((-104, 55), radius=5, facecolor=BAD, edgecolor="white", lw=1))
        _badge(ax, False, "умер до входа")
    elif kind == "miss":
        _track(ax, (-125, 320), (125, 300), lw=3.0)
        _badge(ax, False, "не пересекает объём")
    elif kind == "multi":
        _track(ax, (-118, 300), (70, -280), lw=3.0)
        _track(ax, (-70, 300), (118, -260), color="#6B777D", lw=2.4)
        _badge(ax, False, "не один мюон")
    elif kind == "sparse":
        rng = np.random.RandomState(11)
        ax.scatter(rng.uniform(-55, 55, 9), rng.uniform(-230, 230, 9), s=10, c="#4C9BB8")
        _badge(ax, False, "< 8 хитов / < 3 струны")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=5, color=INK)


def event_types() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.0))
    panels = [
        ("проходит объём", "cross"),
        ("родился внутри", "inside"),
        ("остановился раньше", "stopped"),
        ("прошёл мимо", "miss"),
        ("два мюона", "multi"),
        ("мало сигнальных\nхитов", "sparse"),
    ]
    for ax, (title, kind) in zip(axes.flat, panels):
        _panel(ax, title, kind)
    fig.suptitle(
        "Какие события попадают в обучение энергии",
        fontsize=20,
        fontweight="bold",
        x=0.06,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.06,
        0.935,
        "Схематично; не в масштабе.  Target volume показан штриховой линией; "
        "зелёный — корректный E_entry, красный — отбраковка, оранжевый — особый случай.",
        fontsize=10.5,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.06,
        0.012,
        "* born inside: target = E при рождении; nuatm-события исключены — в этой продукции не записан вершинный каскад.",
        fontsize=9.0,
        color=MUTED,
        ha="left",
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.85, bottom=0.065, wspace=0.06, hspace=0.22)
    for suffix in ("svg", "png"):
        fig.savefig(
            OUT / f"event_types.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor=PAPER,
        )
    plt.close(fig)


_ISO_Y = 0.28


def _p3(x, y, z):
    """Oblique projection: x stays horizontal, y shifts vertical."""
    return x, z - _ISO_Y * y


def _ring(ax, radius, z, half=False, **kwargs):
    theta = (
        np.linspace(0.0, np.pi, 140)
        if half else np.linspace(0.0, 2.0 * np.pi, 240)
    )
    x = radius * np.cos(theta)
    v = z - _ISO_Y * radius * np.sin(theta)
    ax.plot(x, v, **kwargs)


def _cylinder_sides(ax, radius, z0, z1, **kwargs):
    for x in (-radius, radius):
        ax.plot([x, x], [z0, z1], **kwargs)


def _track_entry(start, end, radius):
    """First t in [0,1] where the 3D track crosses the cylinder r=radius."""
    a = np.array(start, float)
    b = np.array(end, float) - a
    aa = b[0] ** 2 + b[1] ** 2
    bb = 2.0 * (a[0] * b[0] + a[1] * b[1])
    cc = a[0] ** 2 + a[1] ** 2 - radius ** 2
    roots = np.roots([aa, bb, cc])
    valid = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0.0 <= r.real <= 1.0]
    if not valid:
        return None
    return a + min(valid) * b


def target_definition_3d() -> None:
    """Isometric-style cluster; explanatory, not a real projection."""
    fig, ax = plt.subplots(figsize=(14, 9.2))
    ax.set_xlim(-250, 250)
    ax.set_ylim(-500, 450)
    ax.set_aspect("equal")
    ax.axis("off")

    # target cylinder: r = 90 m, |z| = 295 m
    _cylinder_sides(
        ax, 90, -295, 295, color=SIGMA, ls=(0, (5, 4)), lw=1.8, zorder=2
    )
    _ring(ax, 90, -295, half=True, color=SIGMA, ls=(0, (5, 4)), lw=1.8, zorder=2)
    _ring(ax, 90, 295, color=SIGMA, ls=(0, (5, 4)), lw=1.8, zorder=2)

    # cluster body: r = 60 m, |z| = 265 m
    ax.add_patch(
        Rectangle(
            (-60, -265), 120, 530, facecolor=CLUSTER, edgecolor="none",
            alpha=0.62, zorder=1,
        )
    )
    _cylinder_sides(ax, 60, -265, 265, color=GRID, lw=1.3, zorder=2)
    _ring(ax, 60, -265, half=True, color="#B5C3CB", lw=1.2, zorder=2)
    _ring(ax, 60, 265, color="#B5C3CB", lw=1.2, zorder=2)

    # 8 schematic strings / OMs; keep the exact OM positions for hit selection.
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    om_levels = np.linspace(-225, 225, 7)
    om_xyz: list[tuple[float, float, float]] = []
    for ang in angles:
        x = 60 * np.cos(ang)
        y = 60 * np.sin(ang)
        v0 = -265 - _ISO_Y * y
        v1 = 265 - _ISO_Y * y
        ax.plot([x, x], [v0, v1], color="#8FA1AC", lw=1.0, zorder=3)
        for z in om_levels:
            om_xyz.append((x, y, float(z)))
            ax.add_patch(
                Circle(
                    (x, z - _ISO_Y * y), radius=4.0,
                    facecolor="#C9D6DD", edgecolor="#7F929E", lw=0.6, zorder=4,
                )
            )
    # central string: the real cluster has 7 perimeter strings + 1 central
    for z in om_levels:
        om_xyz.append((0.0, 0.0, float(z)))
        ax.plot([0, 0], [-265, 265], color="#8FA1AC", lw=1.0, zorder=3)
        ax.add_patch(
            Circle((0, z), radius=4.0, facecolor="#C9D6DD",
                   edgecolor="#7F929E", lw=0.6, zorder=4)
        )

    # one muon track; entry point on the right side of the target cylinder
    start = (200.0, 20.0, 330.0)
    end = (-50.0, -35.0, -230.0)
    entry = _track_entry(start, end, 90.0)
    if entry is None:
        raise RuntimeError("track does not cross target cylinder")
    p0 = _p3(*start)
    p1 = _p3(*end)
    pe = _p3(*entry)
    ax.plot(
        [p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=5.0,
        solid_capstyle="round", zorder=9,
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.70 * p0[0] + 0.30 * p1[0], 0.70 * p0[1] + 0.30 * p1[1]),
            (0.56 * p0[0] + 0.44 * p1[0], 0.56 * p0[1] + 0.44 * p1[1]),
            arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.4, zorder=10,
        )
    )

    # Cherenkov hits are drawn on the OMs they belong to, not as a random cloud.
    # Pick the OMs closest to the track (a schematic subset of the real hit list).
    a3 = np.array(start, float)
    b3 = np.array(end, float) - a3
    distances = []
    for om in om_xyz:
        p3 = np.array(om, float)
        t = float(np.clip(np.dot(p3 - a3, b3) / np.dot(b3, b3), 0.0, 1.0))
        distances.append((np.linalg.norm(p3 - (a3 + t * b3)), om))
    distances.sort(key=lambda item: item[0])
    for _, (x, y, z) in distances[:26]:
        u, v = _p3(x, y, z)
        ax.add_patch(
            Circle((u, v), radius=6.2, facecolor="#4C9BB8",
                   edgecolor="white", lw=0.8, zorder=10)
        )

    # entry marker + right-side callout
    ax.add_patch(Circle(pe, radius=8, facecolor=SIGMA, edgecolor="white", lw=2, zorder=12))
    ax.annotate(
        "E_entry\nэнергия мюона\nна входе в target volume",
        xy=pe,
        xytext=(120, 85),
        color=SIGMA,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=SIGMA, lw=1.6),
        zorder=20,
    )

    # free label zones: left for geometry, right for the energy-loss cue
    ax.annotate(
        "target volume\nr ≤ 90 m, |z| ≤ 295 m",
        xy=(-90, 190),
        xytext=(-238, 355),
        color=SIGMA,
        fontsize=10.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=SIGMA, lw=1),
        zorder=20,
    )
    ax.annotate(
        "cluster: 8 strings (7 + central)\nr ≤ 60 m, |z| ≤ 265 m",
        xy=(-60, -180),
        xytext=(-238, -325),
        color=MUTED,
        fontsize=10.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=GRID, lw=1),
        zorder=20,
    )
    ax.annotate(
        "энергия убывает\nвдоль трека",
        xy=(0.30 * p0[0] + 0.70 * p1[0], 0.30 * p0[1] + 0.70 * p1[1]),
        xytext=(122, 250),
        color=MUTED,
        fontsize=10.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
        zorder=20,
    )

    ax.text(
        -242,
        435,
        "Что мы называем энергией мюона?",
        fontsize=19,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        -242,
        -475,
        "Схематично, не в масштабе.  Точки — черенковские хиты; линия — трек мюона.\n"
        "Цель — энергия при входе в цилиндр r ≤ 90 м, |z| ≤ 295 м вокруг своего кластера.",
        fontsize=10.8,
        color=MUTED,
        ha="left",
        va="bottom",
        linespacing=1.45,
    )

    # legend in its own box, clear of the footer
    legend = FancyBboxPatch(
        (102, -405), 142, 88,
        boxstyle="round,pad=0.4,rounding_size=8",
        facecolor="white", edgecolor=GRID, lw=1.0, zorder=18,
    )
    ax.add_patch(legend)
    ax.scatter([114], [-338], s=22, c="#4C9BB8", lw=0, zorder=19)
    ax.text(124, -338, "Cherenkov hit", fontsize=9.2, color=MUTED, va="center", zorder=20)
    ax.plot([106, 124], [-360, -360], color=INK, lw=3, zorder=19)
    ax.text(130, -360, "muon track", fontsize=9.2, color=MUTED, va="center", zorder=20)
    ax.plot([106, 124], [-382, -382], color=SIGMA, lw=1.6, ls=(0, (4, 3)), zorder=19)
    ax.text(130, -382, "target volume", fontsize=9.2, color=MUTED, va="center", zorder=20)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.03)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(
            OUT / f"target_definition_3d.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor=PAPER,
        )
    plt.close(fig)


def appendix_event_display() -> None:
    """A deliberately real (not schematic) selected MC event display."""
    import numpy as np

    dataset = (
        Path(__file__).resolve().parents[2]
        / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_testparts"
    )
    offsets = np.load(dataset / "offsets.npy", mmap_mode="r")
    features = np.load(dataset / "features.npy", mmap_mode="r")
    energy = np.load(dataset / "energy.npy", mmap_mode="r")
    status = np.load(dataset / "energy_status.npy", mmap_mode="r")
    n_hits = np.load(dataset / "n_sig_hits.npy", mmap_mode="r")

    # A representative single-track test event: status 0, moderate multiplicity,
    # chosen for spatially distinct hits so the display is not a stack of repeats.
    candidates = np.flatnonzero(
        (status == 0) & np.isfinite(energy) & (n_hits >= 30) & (n_hits <= 100)
    )
    best = None
    for idx in candidates[:10000]:
        a, b = int(offsets[idx]), int(offsets[idx + 1])
        h = np.asarray(features[a:b])
        xy = h[:, 2:4]
        r = np.hypot(xy[:, 0], xy[:, 1])
        # Prefer an event that actually lights the central string as well as the
        # perimeter, so the appendix does not illustrate a disabled-string part.
        if r.min() >= 8.0 or np.unique(np.round(xy, 1), axis=0).shape[0] < 5:
            continue
        spread = float(np.std(xy[:, 0]) + np.std(xy[:, 1]))
        if best is None or spread > best[0]:
            best = (spread, int(idx))
    event_idx = best[1] if best is not None else 0
    start, stop = int(offsets[event_idx]), int(offsets[event_idx + 1])
    hit = np.asarray(features[start:stop], dtype=float)
    q, t, x, y, z = hit[:, 0], hit[:, 1], hit[:, 2], hit[:, 3], hit[:, 4]
    r = np.hypot(x, y)
    t_rel = t - np.median(t)
    energy_gev = float(energy[event_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)

    # top view
    ax = axes[0]
    ax.add_patch(Circle((0, 0), 90, fill=False, edgecolor=SIGMA, ls=(0, (5, 4)), lw=1.8))
    ax.add_patch(Circle((0, 0), 60, fill=False, edgecolor=GRID, lw=1.4))
    sc = ax.scatter(x, y, c=t_rel, s=28 + 12 * np.log10(np.maximum(q, 1.0)),
                    cmap="viridis", edgecolors="white", linewidths=0.35, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title("вид сверху (x–y)", fontsize=12, fontweight="bold")
    ax.text(-105, 101, "target r ≤ 90 m", color=SIGMA, fontsize=9)
    ax.text(-105, 91, "cluster r ≤ 60 m", color=MUTED, fontsize=9)
    if r.min() < 8:
        k = int(np.argmin(r))
        ax.annotate(
            "central\nstring",
            xy=(x[k], y[k]),
            xytext=(20, -88),
            color=MUTED,
            fontsize=8.5,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9),
        )

    # side view
    ax = axes[1]
    ax.add_patch(Rectangle((0, -295), 90, 590, fill=False, edgecolor=SIGMA,
                           ls=(0, (5, 4)), lw=1.8))
    ax.add_patch(Rectangle((0, -265), 60, 530, fill=False, edgecolor=GRID, lw=1.4))
    sc2 = ax.scatter(r, z, c=t_rel, s=28 + 12 * np.log10(np.maximum(q, 1.0)),
                     cmap="viridis", edgecolors="white", linewidths=0.35, zorder=4)
    ax.set_xlim(0, 105)
    ax.set_ylim(-315, 315)
    ax.set_xlabel("r = sqrt(x² + y²), m")
    ax.set_ylabel("z, m")
    ax.set_title("вид сбоку (r–z)", fontsize=12, fontweight="bold")
    ax.text(5, 304, "target: r ≤ 90 m, |z| ≤ 295 m", color=SIGMA, fontsize=8.5, va="bottom")

    cbar = fig.colorbar(sc2, ax=axes, orientation="horizontal", fraction=0.045, pad=0.10)
    cbar.set_label("hit time − median, ns", fontsize=10)

    fig.suptitle("Appendix: real MC event display", fontsize=20, fontweight="bold",
                 x=0.06, y=0.98, ha="left")
    fig.text(
        0.06,
        0.91,
        f"selected single-track event; E_entry = {energy_gev:.3g} GeV; "
        f"status = {int(status[event_idx])}; {len(hit)} signal hits. "
        "The energy dataset stores hits, not a reconstructed track.",
        fontsize=10.5, color=MUTED, ha="left",
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.82, bottom=0.18, wspace=0.25)
    for suffix in ("svg", "png"):
        fig.savefig(
            OUT / f"appendix_event_display.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor=PAPER,
        )
    plt.close(fig)


if __name__ == "__main__":
    # Keep the older 2D versions; the 3D target is the main-deck version.
    target_definition()
    target_definition_3d()
    event_types()
    appendix_event_display()
    print("wrote", OUT)
