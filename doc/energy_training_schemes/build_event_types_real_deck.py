#!/usr/bin/env python3
"""Native, editable PowerPoint slide of the accepted/rejected event sheet.

The six panels reuse the real MC events from ``build_event_types_real.py``.
Every visible element -- target cylinder, cluster, strings, OMs, hits, tracks,
labels and badges -- is a native PowerPoint shape, so the slide can be edited
without rerunning the plotting code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_event_types_real import (  # noqa: E402
    EXAMPLES,
    ISO,
    _reason,
    cross_track,
    load,
    p3,
)

OUT = HERE / "event_types_real_native.pptx"
SW, SH = 13.333, 7.5
S = 0.00478  # inches per metre; enlarged relative to the PNG contact sheet
COLUMN_X = (2.52, 6.80, 11.08)
ROW_Y = (2.3706, 5.7406)
LABEL_Y = (0.59, 3.96)
LABEL_H = 0.23

PAPER = RGBColor(0xFA, 0xFA, 0xF8)
INK = RGBColor(0x26, 0x32, 0x38)
MUTED = RGBColor(0x6B, 0x77, 0x7D)
GRID = RGBColor(0xD9, 0xDE, 0xE1)
CLUSTER = RGBColor(0xE8, 0xEE, 0xF2)
SIGMA = RGBColor(0xE0, 0x7A, 0x2E)
GOOD = RGBColor(0x2E, 0x7D, 0x4F)
BAD = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OM_FILL = RGBColor(0xC9, 0xD6, 0xDD)
OM_LINE = RGBColor(0x7F, 0x92, 0x9E)
STRING = RGBColor(0x8F, 0xA1, 0xAC)


def _text(slide, x, y, w, h, value, color=INK, size=12, bold=False,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
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


def _line(slide, x1, y1, x2, y2, color=INK, width=1.2, dash=None, name=None):
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


def _ellipse(slide, cx, cy, w, h, fill, line_color=None, width=0.5, dash=None,
             name=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(cx - w / 2), Inches(cy - h / 2), Inches(w), Inches(h)
    )
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(width)
        if dash is not None:
            shp.line.dash_style = dash
    if name:
        shp.name = name
    return shp


def _rect(slide, x, y, w, h, fill, name=None, line_color=None, width=0.5):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(width)
    if name:
        shp.name = name
    return shp


def _polyline(slide, points, color, width=1.2, dash=None, name=None):
    """Draw an open native freeform path through (x, y) points in inches.

    ``python-pptx`` rounds freeform local coordinates to integers, so the path is
    expressed in milli-inches (1000 units per inch) rather than inches.
    """
    units = 1000.0
    scale = Inches(1) / units
    builder = slide.shapes.build_freeform(
        round(points[0][0] * units), round(points[0][1] * units), scale=scale
    )
    builder.add_line_segments(
        [(round(x * units), round(y * units)) for x, y in points[1:]], close=False
    )
    shp = builder.convert_to_shape()
    shp.fill.background()
    shp.line.color.rgb = color
    shp.line.width = Pt(width)
    if dash is not None:
        shp.line.dash_style = dash
    if name:
        shp.name = name
    return shp


def _rgb(color):
    return RGBColor(*[int(round(255 * v)) for v in color[:3]])


def _px(cx, x):
    return cx + S * float(x)


def _py(cy, y, z):
    return cy - S * (float(z) - ISO * float(y))


def _clip_segment(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
    """Liang-Barsky clipping in projected panel coordinates."""
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - xmin, xmax - x1, y1 - ymin, ymax - y1)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            if t > u1:
                return None
            u0 = max(u0, t)
        else:
            if t < u0:
                return None
            u1 = min(u1, t)
    if u0 > u1:
        return None
    return (x1 + u0 * dx, y1 + u0 * dy, x1 + u1 * dx, y1 + u1 * dy)


def _draw_target(slide, cx, cy, prefix):
    ring_w = 2 * 90 * S
    ring_h = 2 * 90 * ISO * S
    _ellipse(slide, cx, cy - 295 * S, ring_w, ring_h, None, SIGMA, 1.2,
             MSO_LINE.DASH, f"{prefix}_target_top")
    theta = np.linspace(0, np.pi, 33)
    pts = [(_px(cx, 90 * np.cos(a)), _py(cy, 90 * np.sin(a), -295))
           for a in theta]
    _polyline(slide, pts, SIGMA, 1.2, MSO_LINE.DASH, f"{prefix}_target_bottom_arc")
    _line(slide, cx - 90 * S, cy - 295 * S, cx - 90 * S, cy + 295 * S,
          SIGMA, 1.2, MSO_LINE.DASH, f"{prefix}_target_left_side")
    _line(slide, cx + 90 * S, cy - 295 * S, cx + 90 * S, cy + 295 * S,
          SIGMA, 1.2, MSO_LINE.DASH, f"{prefix}_target_right_side")


def _draw_cluster(slide, cx, cy, prefix):
    _rect(slide, cx - 60 * S, cy - 265 * S, 120 * S, 530 * S, CLUSTER,
          f"{prefix}_cluster_body")
    _ellipse(slide, cx, cy - 265 * S, 120 * S, 2 * 60 * ISO * S, None,
             GRID, 0.9, name=f"{prefix}_cluster_top")
    theta = np.linspace(0, np.pi, 33)
    pts = [(_px(cx, 60 * np.cos(a)), _py(cy, 60 * np.sin(a), -265))
           for a in theta]
    _polyline(slide, pts, GRID, 0.9, name=f"{prefix}_cluster_bottom_arc")


def _draw_event(slide, ev, kind, accepted, label, index):
    col, row = index % 3, index // 3
    cx, cy = COLUMN_X[col], ROW_Y[row]
    prefix = f"event_{index + 1}"
    raw, sig, ch, mu = ev["raw"], ev["sig"], ev["ch"], ev["mu"]

    _draw_cluster(slide, cx, cy, prefix)
    _draw_target(slide, cx, cy, prefix)

    # strings, inferred from the raw-hit channel groups
    for j, st in enumerate(np.unique(ch // 36)):
        m = (ch // 36) == st
        xm = float(raw[m, 2].mean())
        ym = float(raw[m, 3].mean())
        _line(slide, _px(cx, xm), _py(cy, ym, -265),
              _px(cx, xm), _py(cy, ym, 265), STRING, 0.7,
              name=f"{prefix}_string_{j:02d}")

    # raw-hit OMs, deduplicated by position
    unique_xyz = np.unique(raw[:, 2:5], axis=0)
    for j, (x, y, z) in enumerate(unique_xyz):
        _ellipse(slide, _px(cx, x), _py(cy, y, z), 0.027, 0.027,
                 OM_FILL, OM_LINE, 0.25, name=f"{prefix}_om_{j:03d}")

    # signal hits: colour = time, diameter = charge
    if len(sig):
        q, t, x, y, z = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3], sig[:, 4]
        t_rel = t - np.median(t)
        vmin, vmax = float(t_rel.min()), float(t_rel.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps["viridis"]
        for j, (qq, tt, xx, yy, zz) in enumerate(zip(q, t_rel, x, y, z)):
            size = float(np.sqrt(12 + 28 * np.log10(max(float(qq), 1.0))) / 72.0)
            _ellipse(slide, _px(cx, xx), _py(cy, yy, zz), size, size,
                     _rgb(cmap(norm(tt))), WHITE, 0.45,
                     name=f"{prefix}_signal_hit_{j:03d}")

    # true track(s), clipped to the panel data rectangle
    tracks_to_draw = mu if kind in ("cross", "born", "multi") else []
    for j, track in enumerate(tracks_to_draw):
        theta, phi = float(track[0]), float(track[1])
        d = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        ref = track[2:5] - ev["cc"]
        roots = []
        for rr in np.roots([
            d[0] ** 2 + d[1] ** 2,
            2 * (ref[0] * d[0] + ref[1] * d[1]),
            ref[0] ** 2 + ref[1] ** 2 - 90 ** 2,
        ]):
            if abs(rr.imag) < 1e-8:
                roots.append(float(rr.real))
        if roots:
            lo, hi = min(roots) - 75, max(roots) + 75
        else:
            lo, hi = -180, 180
        pts = ref[None, :] + np.array([lo, hi])[:, None] * d[None, :]
        uv = np.array([p3(*p) for p in pts])
        clipped = _clip_segment(uv[0, 0], uv[0, 1], uv[1, 0], uv[1, 1],
                                -135, 135, -350, 330)
        if clipped is not None:
            u1, v1, u2, v2 = clipped
            _line(slide, _px(cx, u1), cy - S * v1,
                  _px(cx, u2), cy - S * v2, INK, 2.0,
                  name=f"{prefix}_track_{j:02d}")

    # true entry marker for the accepted crossing
    if kind == "cross" and ev["status"] == 0 and np.isfinite(ev["energy"]):
        tr = mu[0]
        theta, phi = float(tr[0]), float(tr[1])
        d = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        ref = tr[2:5] - ev["cc"]
        te = cross_track(ref, d)
        if te is not None:
            pe = ref + te * d
            _ellipse(slide, _px(cx, pe[0]), _py(cy, pe[1], pe[2]),
                     0.075, 0.075, SIGMA, WHITE, 0.9,
                     name=f"{prefix}_entry_marker")

    # title and accepted/rejected badge share one compact row above the event
    label_y = LABEL_Y[row]
    title_w = 1.70
    badge_w = 1.70
    _text(slide, cx - 1.82, label_y, title_w, LABEL_H, label, INK,
          9.5, True, PP_ALIGN.RIGHT, name=f"{prefix}_title")

    color = GOOD if accepted is True else (BAD if accepted is False else SIGMA)
    sym = "✓" if accepted is True else ("✗" if accepted is False else "*")
    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(cx + 0.12), Inches(label_y), Inches(badge_w), Inches(LABEL_H)
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = WHITE
    badge.line.color.rgb = color
    badge.line.width = Pt(0.9)
    badge.name = f"{prefix}_badge"
    tf = badge.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.margin_left = Inches(0.03)
    tf.margin_right = Inches(0.03)
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = f"{sym} {_reason(kind, ev)}"
    run.font.name = "Arial"
    run.font.size = Pt(8.0)
    run.font.bold = True
    run.font.color.rgb = color


def add_event_types_real_slide(prs: Presentation) -> None:
    """Append the native real-event contact sheet as one slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER

    _text(slide, 0.55, 0.07, 12.2, 0.31,
          "Реальные MC-события: что проходит отбор энергии",
          INK, 19.5, True, name="event_types_title")
    _text(slide, 0.55, 0.40, 12.2, 0.16,
          "Каждая панель — реальное событие; цвет — время, размер — заряд. "
          "Штриховой цилиндр — target volume.",
          MUTED, 9.2, name="event_types_subtitle")

    for index, example in enumerate(EXAMPLES):
        ptype, part, local, kind, accepted, label = example
        ev = load(ptype, part, local)
        _draw_event(slide, ev, kind, accepted, label, index)
        print(kind, ptype, part, local, "status", ev["status"],
              "sig", ev["ns"], "strings", ev["nss"], "n_mu", len(ev["mu"]))

    _text(slide, 0.55, 7.31, 12.2, 0.17,
          "* born inside: target = энергия при рождении; "
          "nuatm born-inside исключены; nue2 — разрешены.",
          MUTED, 8.0, name="event_types_caption")

    slide.notes_slide.notes_text_frame.text = (
        "Real MC events, one per panel, not a schematic.  Colour is the signal-hit "
        "time relative to the event median; marker diameter scales with charge.  "
        "Grey dots are OMs with raw hits; the dashed cylinder is the target volume "
        "r <= 90 m, |z| <= 295 m.  The true tracks come from muons_prty/individ.  "
        "The six examples are the same events used by figures/event_types_real.png."
    )


def build() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    add_event_types_real_slide(prs)
    slide = prs.slides[0]
    prs.save(OUT)
    print("wrote", OUT, "shapes", len(slide.shapes))


if __name__ == "__main__":
    build()
