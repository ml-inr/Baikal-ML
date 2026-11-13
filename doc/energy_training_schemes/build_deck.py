#!/usr/bin/env python3
"""Build a native, fully editable PPTX slide for the muon-energy target.

The 3D-looking cluster is deliberately assembled from ordinary PowerPoint shapes:
ellipses, connectors, rectangles, text boxes and small circles.  Nothing is a
flattened picture, so every ring, string, OM, hit, label and the track can be
selected and edited in PowerPoint.

Run:
    python doc/energy_training_schemes/build_deck.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent
PPTX = OUT / "target_definition_3d_native.pptx"

INK = RGBColor(0x26, 0x32, 0x38)
MUTED = RGBColor(0x6B, 0x77, 0x7D)
GRID = RGBColor(0xD9, 0xDE, 0xE1)
CLUSTER = RGBColor(0xE8, 0xEE, 0xF2)
ENERGY = RGBColor(0x2F, 0x6D, 0xB3)
SIGMA = RGBColor(0xE0, 0x7A, 0x2E)
DOMAIN = RGBColor(0x3A, 0x8A, 0x5A)
PAPER = RGBColor(0xFA, 0xFA, 0xF8)

SW, SH = 13.333, 7.5
S = 0.009          # inches per metre
ISO = 0.28
CX, CY = 6.35, 3.80


def sx(x: float) -> float:
    return CX + x * S


def sy(x: float, y: float, z: float) -> float:
    return CY - z * S + ISO * y * S


def line(slide, x1, y1, x2, y2, color=INK, width=1.5, dash=None, name=None):
    shp = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    shp.line.color.rgb = color
    shp.line.width = Pt(width)
    if dash is not None:
        shp.line.dash_style = dash
    if name:
        shp.name = name
    return shp


def half_ellipse(slide, radius, z, color, width=1.4, dashed=False, name_prefix="arc"):
    """Front half of a horizontal circular rim, as native editable line segments."""
    theta = np.linspace(0.0, np.pi, 20)
    points = []
    for a in theta:
        x = radius * np.cos(a)
        y = radius * np.sin(a)
        points.append((sx(x), sy(x, y, z)))
    for i in range(len(points) - 1):
        line(
            slide, points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
            color, width, MSO_LINE.DASH if dashed else None,
            f"{name_prefix}_{i:02d}",
        )


def ellipse(slide, cx, cy, w, h, color, width=1.2, dashed=False, fill=None, name=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(cx - w / 2), Inches(cy - h / 2), Inches(w), Inches(h),
    )
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = color
    shp.line.width = Pt(width)
    if dashed:
        shp.line.dash_style = MSO_LINE.DASH
    if name:
        shp.name = name
    return shp


def rect(slide, x, y, w, h, fill, name=None, line_color=None, line_width=0.8):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(line_width)
    if name:
        shp.name = name
    return shp


def text(slide, x, y, w, h, value, color=INK, size=12, bold=False, align=PP_ALIGN.LEFT, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if name:
        box.name = name
    return box


def add_track_arrow(slide, x1, y1, x2, y2, color=INK):
    """Small native triangle at 35% along the track (separate editable shape)."""
    t = 0.35
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    size = 0.13
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ISOSCELES_TRIANGLE,
        Inches(x - size / 2), Inches(y - size / 2), Inches(size), Inches(size),
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    # The triangle points up by default; align it with the track.
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90.0
    shp.rotation = float(angle)
    shp.name = "muon_track_arrow"
    return shp


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER

    # --- target cylinder -------------------------------------------------
    h_target = 2 * 295 * S
    h_cluster = 2 * 265 * S
    cx = CX
    cy_top = CY - 295 * S
    cy_bottom = CY + 295 * S
    top_w = 2 * 90 * S
    top_h = 2 * 90 * ISO * S
    ellipse(slide, cx, cy_top, top_w, top_h, SIGMA, 1.6, dashed=True, name="target_top_ring")
    half_ellipse(slide, 90, -295, SIGMA, 1.6, True, "target_bottom_arc")
    line(slide, cx - 90 * S, cy_top, cx - 90 * S, cy_bottom, SIGMA, 1.6, MSO_LINE.DASH, "target_left_side")
    line(slide, cx + 90 * S, cy_top, cx + 90 * S, cy_bottom, SIGMA, 1.6, MSO_LINE.DASH, "target_right_side")

    # --- cluster body ----------------------------------------------------
    cluster_top = CY - 265 * S
    cluster_bottom = CY + 265 * S
    rect(slide, CX - 60 * S, cluster_top, 120 * S, h_cluster, CLUSTER, "cluster_body")
    ellipse(slide, CX, cluster_top, 120 * S, 2 * 60 * ISO * S, GRID, 1.1, name="cluster_top_ring")
    half_ellipse(slide, 60, -265, GRID, 1.0, False, "cluster_bottom_arc")
    line(slide, CX - 60 * S, cluster_top, CX - 60 * S, cluster_bottom, GRID, 1.0, name="cluster_left_side")
    line(slide, CX + 60 * S, cluster_top, CX + 60 * S, cluster_bottom, GRID, 1.0, name="cluster_right_side")

    # 7 perimeter strings + 1 central string; each OM is a native ellipse
    om_levels = np.linspace(-225, 225, 7)
    om_d = 0.075
    om_xyz: list[tuple[float, float, float]] = []
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 7, endpoint=False)):
        x = 60 * np.cos(ang)
        y = 60 * np.sin(ang)
        x0, y0 = sx(x), sy(x, y, -265)
        x1, y1 = sx(x), sy(x, y, 265)
        line(slide, x0, y0, x1, y1, RGBColor(0x8F, 0xA1, 0xAC), 0.9, name=f"string_perimeter_{i}")
        for j, z in enumerate(om_levels):
            om_xyz.append((x, y, float(z)))
            ellipse(
                slide, sx(x), sy(x, y, z), om_d, om_d,
                RGBColor(0x7F, 0x92, 0x9E), 0.4,
                fill=RGBColor(0xC9, 0xD6, 0xDD),
                name=f"om_perimeter_{i}_{j}",
            )
    # central string
    line(slide, CX, sy(0, 0, -265), CX, sy(0, 0, 265),
         RGBColor(0x8F, 0xA1, 0xAC), 0.9, name="string_central")
    for j, z in enumerate(om_levels):
        om_xyz.append((0.0, 0.0, float(z)))
        ellipse(
            slide, CX, sy(0, 0, z), om_d, om_d,
            RGBColor(0x7F, 0x92, 0x9E), 0.4,
            fill=RGBColor(0xC9, 0xD6, 0xDD),
            name=f"om_central_{j}",
        )

    # --- track -----------------------------------------------------------
    start = (200.0, 20.0, 330.0)
    end = (-50.0, -35.0, -230.0)
    # first crossing of r=90 along the segment
    a = np.array(start, float)
    b = np.array(end, float) - a
    roots = np.roots([
        b[0] ** 2 + b[1] ** 2,
        2 * (a[0] * b[0] + a[1] * b[1]),
        a[0] ** 2 + a[1] ** 2 - 90 ** 2,
    ])
    valid = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 <= r.real <= 1]
    entry = a + min(valid) * b
    tx0, ty0 = sx(start[0]), sy(*start)
    tx1, ty1 = sx(end[0]), sy(*end)
    ex, ey = sx(entry[0]), sy(*entry)
    line(slide, tx0, ty0, tx1, ty1, INK, 3.4, name="muon_track")
    add_track_arrow(slide, tx0, ty0, tx1, ty1)

    # --- Cherenkov hits on their OMs (native circles) -------------------
    a3 = np.array(start, float)
    b3 = np.array(end, float) - a3
    distances = []
    for om in om_xyz:
        p3 = np.array(om, float)
        t = float(np.clip(np.dot(p3 - a3, b3) / np.dot(b3, b3), 0.0, 1.0))
        distances.append((np.linalg.norm(p3 - (a3 + t * b3)), om))
    distances.sort(key=lambda item: item[0])
    n = 0
    for _, (x, y, z) in distances[:26]:
        d = 0.105
        ellipse(
            slide, sx(x), sy(x, y, z), d, d,
            RGBColor(0x3F, 0x8F, 0xAE), 0.7,
            fill=RGBColor(0x4C, 0x9B, 0xB8),
            name=f"cherenkov_hit_{n:02d}",
        )
        n += 1

    # --- entry marker and labels ----------------------------------------
    ellipse(slide, ex, ey, 0.15, 0.15, SIGMA, 1.4,
            fill=SIGMA, name="entry_marker_E_entry")

    text(slide, 0.55, 0.28, 8.5, 0.55,
         "Что мы называем энергией мюона?", INK, 26, bold=True,
         name="slide_title")
    text(slide, 0.55, 0.98, 2.9, 0.8,
         "target volume\nr ≤ 90 m, |z| ≤ 295 m", SIGMA, 11.5,
         name="label_target_volume")
    line(slide, 2.45, 1.35, cx - 90 * S, 1.85, SIGMA, 1.0, name="leader_target_volume")

    text(slide, 0.55, 5.88, 3.15, 0.75,
         "cluster: 8 strings (7 + central)\nr ≤ 60 m, |z| ≤ 265 m", MUTED, 9.8,
         name="label_cluster")
    line(slide, 3.05, 6.08, cx - 60 * S, 5.55, GRID, 1.0, name="leader_cluster")

    text(slide, 9.00, 1.20, 3.8, 0.8,
         "энергия убывает\nвдоль трека", MUTED, 11.0,
         name="label_energy_loss")
    line(slide, 9.15, 1.72, 8.15, 2.00, MUTED, 1.0, name="leader_energy_loss")

    text(slide, 7.42, 2.78, 5.3, 0.9,
         "E_entry\nэнергия мюона\nна входе в target volume", SIGMA, 13.0, bold=True,
         name="label_E_entry")
    line(slide, ex + 0.08, ey, 7.40, 3.05, SIGMA, 1.3, name="leader_E_entry")

    # legend: native shapes + text
    ellipse(slide, 9.10, 5.23, 0.10, 0.10, RGBColor(0x3F, 0x8F, 0xAE), 0.0,
            fill=RGBColor(0x4C, 0x9B, 0xB8), name="legend_hit_dot")
    text(slide, 9.28, 5.15, 1.8, 0.26, "Cherenkov hit", MUTED, 9.5, name="legend_hit_text")
    line(slide, 9.04, 5.53, 9.22, 5.53, INK, 2.0, name="legend_track_line")
    text(slide, 9.28, 5.43, 1.8, 0.26, "muon track", MUTED, 9.5, name="legend_track_text")
    line(slide, 9.04, 5.80, 9.22, 5.80, SIGMA, 1.3, MSO_LINE.DASH, "legend_target_line")
    text(slide, 9.28, 5.69, 2.0, 0.26, "target volume", MUTED, 9.5, name="legend_target_text")

    text(slide, 0.55, 6.68, 12.2, 0.72,
         "Схематично, не в масштабе.  Точки — черенковские хиты; линия — трек мюона.\n"
         "Цель — энергия при входе в цилиндр r ≤ 90 м, |z| ≤ 295 м вокруг своего кластера.",
         MUTED, 9.4, name="caption")

    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "Target definition: e_at_entry[:, 1] from the energy-truth companion; "
        "r <= 90 m, |z| <= 295 m around the row's own cluster; unit GeV. "
        "The model predicts log10(E/GeV). The event dataset keeps exactly one muon "
        "track and valid truth statuses 0, 1, 4, 5. Born-inside statuses have "
        "special semantics; nuatm born-inside rows are excluded. "
        "All visible elements are native PowerPoint shapes, not an embedded image."
    )

    OUT.mkdir(parents=True, exist_ok=True)
    prs.save(PPTX)
    print("wrote", PPTX, "shapes:", len(slide.shapes))


if __name__ == "__main__":
    build()
