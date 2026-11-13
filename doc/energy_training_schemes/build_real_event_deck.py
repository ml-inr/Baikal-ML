#!/usr/bin/env python3
"""Native, editable PowerPoint slide of a real MC event.

Data are the same as build_real_event.py: signal-hit time controls colour, hit
charge controls marker size, the true muon track and the target cylinder are
overlaid.  All visible elements are native PowerPoint shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
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
from build_real_event import load_event, track_entry, p3  # noqa: E402

OUT = HERE / "real_event_display_native.pptx"
INK = RGBColor(0x26, 0x32, 0x38)
MUTED = RGBColor(0x6B, 0x77, 0x7D)
GRID = RGBColor(0xD9, 0xDE, 0xE1)
CLUSTER = RGBColor(0xE8, 0xEE, 0xF2)
SIGMA = RGBColor(0xE0, 0x7A, 0x2E)
PAPER = RGBColor(0xFA, 0xFA, 0xF8)

SW, SH = 13.333, 7.5
S = 0.009
ISO = 0.28
CX, CY = 4.65, 3.78


def sx(x):
    return CX + x * S


def sy(x, y, z):
    return CY - z * S + ISO * y * S


def line(slide, x1, y1, x2, y2, color=INK, width=1.2, dash=None, name=None):
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


def ellipse(slide, cx, cy, w, h, fill, line_color=None, width=0.4, name=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(cx - w / 2), Inches(cy - h / 2), Inches(w), Inches(h)
    )
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


def rect(slide, x, y, w, h, fill, name=None, line_color=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
    if name:
        shp.name = name
    return shp


def text(slide, x, y, w, h, value, color=INK, size=12, bold=False, name=None):
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
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if name:
        box.name = name
    return box


def half_arc(slide, radius, z, color, width=1.4, dashed=False, prefix="arc"):
    theta = np.linspace(0, np.pi, 20)
    pts = [(sx(radius * np.cos(a)), sy(radius * np.cos(a), radius * np.sin(a), z))
           for a in theta]
    for i in range(len(pts) - 1):
        line(slide, *pts[i], *pts[i + 1], color, width,
             MSO_LINE.DASH if dashed else None, f"{prefix}_{i:02d}")


def rgb(c):
    return RGBColor(*[int(round(255 * v)) for v in c[:3]])


def build():
    ev = load_event()
    sig, raw, ch = ev["sig"], ev["raw"], ev["channels"]
    q, t, x, y, z = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3], sig[:, 4]
    t_rel = t - np.median(t)
    vmin, vmax = float(t_rel.min()), float(t_rel.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["viridis"]
    qmin, qmax = float(np.log10(max(q.min(), 1.0))), float(np.log10(max(q.max(), 1.0)))

    ref, d = ev["track_ref"], ev["track_dir"]
    te = track_entry(ref, d, 90.0)
    entry = ref + te * d if te is not None else None

    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER

    # target cylinder
    top_w = 2 * 90 * S
    top_h = 2 * 90 * ISO * S
    ellipse(slide, CX, CY - 295 * S, top_w, top_h, PAPER, SIGMA, 1.6, "target_top_ring")
    half_arc(slide, 90, -295, SIGMA, 1.6, True, "target_bottom_arc")
    line(slide, CX - 90 * S, CY - 295 * S, CX - 90 * S, CY + 295 * S,
         SIGMA, 1.6, MSO_LINE.DASH, "target_left_side")
    line(slide, CX + 90 * S, CY - 295 * S, CX + 90 * S, CY + 295 * S,
         SIGMA, 1.6, MSO_LINE.DASH, "target_right_side")

    # cluster volume
    rect(slide, CX - 60 * S, CY - 265 * S, 120 * S, 530 * S, CLUSTER, "cluster_body")
    ellipse(slide, CX, CY - 265 * S, 120 * S, 2 * 60 * ISO * S, PAPER, GRID, 1.1, "cluster_top_ring")
    half_arc(slide, 60, -265, GRID, 1.0, False, "cluster_bottom_arc")

    # actual raw-hit OMs and strings
    raw_xyz = raw[:, 2:5]
    for si, st in enumerate(np.unique(ch // 36)):
        m = (ch // 36) == st
        xm, ym = float(raw_xyz[m, 0].mean()), float(raw_xyz[m, 1].mean())
        line(slide, sx(xm), sy(xm, ym, -265), sx(xm), sy(xm, ym, 265),
             RGBColor(0x8F, 0xA1, 0xAC), 0.8, name=f"string_{int(st)}")
        # one OM per distinct raw hit position on this string
        seen = set()
        for zz in raw_xyz[m, 2]:
            key = round(float(zz), 1)
            if key in seen:
                continue
            seen.add(key)
            ellipse(slide, sx(xm), sy(xm, ym, float(zz)), 0.036, 0.036,
                    RGBColor(0xC9, 0xD6, 0xDD), RGBColor(0x7F, 0x92, 0x9E), 0.25,
                    f"om_string_{int(st)}_{len(seen):02d}")

    # signal hits: colour time, size charge
    for i, (qq, tt, xx, yy, zz) in enumerate(zip(q, t_rel, x, y, z)):
        frac = (np.log10(max(float(qq), 1.0)) - qmin) / max(qmax - qmin, 1e-9)
        dsize = 0.045 + 0.065 * frac
        ellipse(slide, sx(xx), sy(xx, yy, zz), dsize, dsize,
                rgb(cmap(norm(tt))), RGBColor(255, 255, 255), 0.45,
                f"signal_hit_{i:02d}")

    # true track and entry; draw only the visible part around the target volume
    aa = d[0] ** 2 + d[1] ** 2
    bb = 2 * (ref[0] * d[0] + ref[1] * d[1])
    cc = ref[0] ** 2 + ref[1] ** 2 - 90 ** 2
    roots = [float(r.real) for r in np.roots([aa, bb, cc]) if abs(r.imag) < 1e-8]
    t_lo, t_hi = min(roots) - 75.0, max(roots) + 75.0
    pts = ref[None, :] + np.array([t_lo, t_hi])[:, None] * d[None, :]
    uv = np.array([p3(*p) for p in pts])
    line(slide, sx(uv[0, 0]), sy(*pts[0]), sx(uv[1, 0]), sy(*pts[1]),
         INK, 3.2, name="true_muon_track")
    if entry is not None:
        ellipse(slide, sx(entry[0]), sy(*entry), 0.14, 0.14, SIGMA, SIGMA, 1.2,
                "entry_marker_E_entry")
        text(slide, 7.55, 2.75, 5.0, 0.8,
             "E_entry\nэнергия на входе в target volume", SIGMA, 12.5, True,
             "label_E_entry")
        line(slide, sx(entry[0]) + 0.07, sy(*entry), 7.53, 2.98, SIGMA, 1.2,
             name="leader_E_entry")

    # title / info / legend
    text(slide, 0.55, 0.28, 8.0, 0.55,
         "Реальное MC-событие: цвет — время, размер — заряд", INK, 20, True,
         "slide_title")
    text(slide, 8.55, 1.25, 4.2, 1.8,
         f"E_entry = {ev['energy']/1e3:.1f} TeV\n"
         f"status = 0\n"
         f"signal hits = {len(sig)}\n"
         f"raw hits = {len(raw)}\n"
         f"part = {ev['part']}", INK, 11.5, name="event_info")
    text(slide, 8.55, 3.25, 4.2, 1.2,
         "Цвет: время хита.\nРазмер: заряд в фотоэлектронах.\n"
         "Серые OMs — реальные модули с raw-хитами.", MUTED, 9.8, name="event_legend")

    # native colour bar
    nbar = 18
    x0, y0, bw, bh = 8.65, 5.60, 3.7, 0.16
    for i in range(nbar):
        c = rgb(cmap(i / (nbar - 1)))
        rect(slide, x0 + i * bw / nbar, y0, bw / nbar + 0.002, bh, c,
             f"colorbar_cell_{i:02d}")
    text(slide, x0, y0 + 0.22, 0.8, 0.25, f"{vmin:.0f} ns", MUTED, 8, name="colorbar_min")
    text(slide, x0 + bw - 0.8, y0 + 0.22, 0.8, 0.25, f"{vmax:.0f} ns", MUTED, 8,
         name="colorbar_max")
    text(slide, x0 + 1.1, y0 + 0.22, 1.8, 0.25, "hit time − median", MUTED, 8,
         name="colorbar_label")

    text(slide, 0.55, 6.85, 12.2, 0.50,
         "Реальные сигнальные хиты после sig-noise; трек и OMs — из MC truth.  "
         "Штриховой цилиндр — target volume r ≤ 90 м, |z| ≤ 295 м.",
         MUTED, 9.5, name="caption")

    slide.notes_slide.notes_text_frame.text = (
        "Real MC event from part_2050.  Hit colour is time relative to the event "
        "median; marker area scales with charge.  The true track and entry point "
        "come from muons_prty/individ; the target energy comes from "
        "e_at_entry[:, 1].  This slide complements the schematic geometric slide."
    )
    prs.save(OUT)
    print("wrote", OUT, "shapes", len(slide.shapes))


if __name__ == "__main__":
    build()
