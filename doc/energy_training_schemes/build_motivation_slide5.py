"""Build slide 5 (motivation) of MuEnergy_ML2026.pptx.

The slide has three jobs:
  1. show why energy changes the measurement (atmospheric vs astrophysical shape);
  2. state what the network actually reconstructs (E_entry, not E_nu);
  3. state the current working point and the next technical steps.

The left panel is a published IceCube spectrum, not a synthetic illustration:
IceCube Collaboration, ApJ 833 (2016) 3, Fig. 5 (arXiv:1607.08006).  It shows the
6-year through-going muon-neutrino measurement with the conventional atmospheric,
prompt-atmospheric and astrophysical components.  The crop was rendered from the
arXiv PDF and is stored as figures/icecube_6yr_spectrum.png.

The only numbers taken from our analysis are the current working-point metrics
shown at the bottom; they are the held-out values quoted on the later results
slide and are read from summary_metrics.csv.

Run from the repository root:
    python doc/energy_training_schemes/build_motivation_slide5.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "MuEnergy_ML2026.pptx"
FIG_DIR = ROOT / "doc/energy_training_schemes/figures"
CHART = FIG_DIR / "icecube_6yr_spectrum.png"
SUMMARY = (
    ROOT
    / "inference/track_energy_reconstruction/analysis/reports/summary_metrics.csv"
)

# Deck palette, taken from the existing slides.
INK = RGBColor(0x26, 0x32, 0x38)
MUTED = RGBColor(0x6B, 0x77, 0x7D)
BLUE = RGBColor(0x2F, 0x6D, 0xB3)
ORANGE = RGBColor(0xE0, 0x7A, 0x2E)
RULE = RGBColor(0xD9, 0xDE, 0xE1)
PANEL = RGBColor(0xF5, 0xF7, 0xF9)
LIGHT_BLUE = RGBColor(0xE8, 0xF1, 0xFB)

def add_text(
    slide,
    name: str,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: float,
    color: RGBColor = INK,
    bold: bool = False,
    italic: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
    line_spacing: float | None = None,
    space_after: float = 0.0,
):
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.name = f"motivation_{name}"
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor

    p = tf.paragraphs[0]
    p.alignment = align
    if line_spacing is not None:
        p.line_spacing = line_spacing
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def add_bullets(
    slide,
    name: str,
    lines: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: float,
    color: RGBColor = INK,
    space_after: float = 4.0,
):
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.name = f"motivation_{name}"
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP

    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.0
        run = p.add_run()
        run.text = f"•  {line}"
        run.font.name = "Arial"
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return box


def add_rule(slide, name: str, left: float, top: float, width: float):
    line = slide.shapes.add_connector(
        1, Inches(left), Inches(top), Inches(left + width), Inches(top)
    )
    line.name = f"motivation_{name}"
    line.line.color.rgb = RULE
    line.line.width = Pt(0.8)


def add_box(
    slide,
    name: str,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    fill: RGBColor,
    line: RGBColor,
    color: RGBColor,
    size: float,
    bold: bool = False,
):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    shape.name = f"motivation_{name}"
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(0.9)
    try:
        shape.adjustments[0] = 0.12
    except IndexError:
        pass
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.0)
    tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name = "Arial"
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return shape


def add_arrow(slide, name: str, left: float, top: float, width: float):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(0.13),
    )
    shape.name = f"motivation_{name}"
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xB8, 0xC4, 0xCC)
    shape.line.fill.background()
    return shape


def clear_previous_content(slide) -> None:
    for shape in list(slide.shapes):
        if shape.name.startswith("motivation_"):
            shape._element.getparent().remove(shape._element)


def build_slide(prs: Presentation) -> None:
    # Canonical working point used on the results slide; the ID is internal and
    # never appears on the slide.
    summary = pd.read_csv(SUMMARY)
    row = summary.loc[summary["id"] == "AC-G"].iloc[0]
    mae_text = f"MAE {row['mae']:.3f} log10(E/GeV)  ≈  factor 3"
    pct_text = (
        f"{100.0 * row['within_x2']:.0f}% within ×2  ·  "
        f"{100.0 * row['within_x5']:.0f}% within ×5"
    )

    slide = prs.slides[4]

    # The original title box was 6.26 in wide and wrapped; keep the deck's
    # 36 pt title but give it the full slide width.
    title = slide.shapes.title
    if title is None:
        for shape in slide.shapes:
            if shape.has_text_frame and "Why to reconstruct energy" in shape.text:
                title = shape
                break
    if title is None:
        raise RuntimeError("Slide 5 title shape not found")

    title.left = Inches(0.45)
    title.top = Inches(0.18)
    title.width = Inches(12.45)
    title.height = Inches(0.78)
    for p in title.text_frame.paragraphs:
        p.line_spacing = 0.9
        for r in p.runs:
            r.font.name = "Calibri Light"
            r.font.size = Pt(36)
            r.font.color.rgb = INK

    clear_previous_content(slide)

    # Tagline: the one-sentence reason the slide exists.
    add_text(
        slide,
        "tagline",
        "Energy turns the track channel from a count of ν candidates "
        "into a spectrum measurement.",
        0.47,
        0.99,
        12.0,
        0.36,
        size=15.5,
        color=BLUE,
        bold=True,
    )
    add_rule(slide, "rule_top", 0.47, 1.42, 12.4)

    # ---- left: published spectrum ---------------------------------------
    add_text(
        slide,
        "spectrum_heading",
        "WHY ENERGY  ·  TeV–PeV",
        0.47,
        1.56,
        4.2,
        0.22,
        size=8.5,
        color=MUTED,
        bold=True,
    )
    pic = slide.shapes.add_picture(
        str(CHART), Inches(0.47), Inches(1.78), width=Inches(4.18)
    )
    pic.name = "motivation_spectrum_chart"
    chart_bottom = 1.78 + pic.height / 914400.0
    add_text(
        slide,
        "spectrum_caption",
        "IceCube 6-yr νμ: atmospheric (blue) vs astrophysical (red).",
        0.47,
        chart_bottom + 0.03,
        4.18,
        0.18,
        size=7.8,
        color=MUTED,
        italic=True,
    )
    add_text(
        slide,
        "spectrum_source",
        "Source: IceCube Collaboration, ApJ 833 (2016) 3, Fig. 5.",
        0.47,
        chart_bottom + 0.21,
        4.18,
        0.16,
        size=7.4,
        color=MUTED,
        italic=True,
    )

    # ---- middle: what the network reconstructs ----------------------------
    add_text(
        slide,
        "reconstruct_heading",
        "WHAT WE RECONSTRUCT",
        4.95,
        1.56,
        3.9,
        0.22,
        size=8.5,
        color=MUTED,
        bold=True,
    )

    box_y = 1.93
    box_h = 0.42
    add_box(
        slide,
        "box_numu",
        "νμ",
        4.95,
        box_y,
        0.56,
        box_h,
        fill=PANEL,
        line=RULE,
        color=INK,
        size=10.2,
        bold=True,
    )
    add_arrow(slide, "arrow_1", 5.54, box_y + 0.145, 0.17)
    add_box(
        slide,
        "box_vertex",
        "ν vertex",
        5.74,
        box_y,
        0.87,
        box_h,
        fill=PANEL,
        line=RULE,
        color=INK,
        size=8.6,
    )
    add_arrow(slide, "arrow_2", 6.64, box_y + 0.145, 0.17)
    add_box(
        slide,
        "box_muon",
        "μ",
        6.84,
        box_y,
        0.5,
        box_h,
        fill=PANEL,
        line=RULE,
        color=INK,
        size=10.2,
        bold=True,
    )
    add_arrow(slide, "arrow_3", 7.37, box_y + 0.145, 0.17)
    add_box(
        slide,
        "box_entry",
        "E_entry",
        7.57,
        box_y,
        1.0,
        box_h,
        fill=LIGHT_BLUE,
        line=BLUE,
        color=BLUE,
        size=9.6,
        bold=True,
    )

    add_text(
        slide,
        "vertex_note",
        "often outside",
        5.74,
        box_y + box_h + 0.02,
        0.87,
        0.18,
        size=7.3,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        "entry_note",
        "instrumented volume",
        7.57,
        box_y + box_h + 0.02,
        1.0,
        0.18,
        size=7.3,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )

    add_text(
        slide,
        "equation",
        "Eν  =  E_entry  +  ΔE_prop  +  E_had",
        4.95,
        2.74,
        3.9,
        0.35,
        size=12.3,
        color=INK,
        bold=True,
    )
    add_text(
        slide,
        "entry_warning",
        "E_entry is a lower bound on Eν: the vertex and the hadronic "
        "energy are often outside the detector, and propagation loss is "
        "not known event-by-event.",
        4.95,
        3.08,
        3.9,
        0.86,
        size=9.8,
        color=MUTED,
        line_spacing=1.05,
    )
    add_text(
        slide,
        "unfolding",
        "Response matrix + unfolding  →  Eν spectrum",
        4.95,
        4.12,
        3.9,
        0.3,
        size=10.5,
        color=ORANGE,
        bold=True,
    )
    add_text(
        slide,
        "unfolding_note",
        "Needed to compare a reconstructed E_entry with a physical Eν flux.",
        4.95,
        4.43,
        3.9,
        0.3,
        size=8.6,
        color=MUTED,
        italic=True,
    )

    # ---- right: science case ---------------------------------------------
    add_text(
        slide,
        "enables_heading",
        "WHAT IT ENABLES",
        9.12,
        1.56,
        3.75,
        0.22,
        size=8.5,
        color=MUTED,
        bold=True,
    )
    add_bullets(
        slide,
        "enables_bullets",
        [
            "Astrophysical diffuse flux: γ, normalization, cutoff",
            "Atmospheric vs astrophysical: spectral shape separates them",
            "Sources: energy plus direction; multi-messenger links",
            "Flavor, BSM and detector systematics",
        ],
        9.12,
        1.93,
        3.75,
        2.9,
        size=11.2,
        space_after=8.0,
    )

    # ---- bottom: honest status and next steps ----------------------------
    add_rule(slide, "rule_bottom", 0.47, 5.05, 12.4)

    add_text(
        slide,
        "status_heading",
        "CURRENT WORKING POINT",
        0.47,
        5.18,
        4.2,
        0.2,
        size=8.5,
        color=MUTED,
        bold=True,
    )
    add_text(
        slide,
        "status_mae",
        mae_text,
        0.47,
        5.40,
        4.2,
        0.3,
        size=13.0,
        color=INK,
        bold=True,
    )
    add_text(
        slide,
        "status_pct",
        pct_text,
        0.47,
        5.72,
        4.2,
        0.24,
        size=10.5,
        color=INK,
    )
    add_text(
        slide,
        "status_caveat",
        "Broad bins only (≥0.5 decade); no event-by-event energy truth in data.",
        0.47,
        6.02,
        4.2,
        0.42,
        size=8.5,
        color=MUTED,
        line_spacing=1.05,
    )

    add_text(
        slide,
        "perspectives_heading",
        "PERSPECTIVES",
        5.05,
        5.18,
        7.75,
        0.2,
        size=8.5,
        color=MUTED,
        bold=True,
    )
    add_bullets(
        slide,
        "perspectives_bullets",
        [
            "Response matrix + unfolding: turn E_entry into an Eν spectrum",
            "Improve resolution and tails; control MC/data systematics",
            "Joint track–cascade bins; BSM channels",
        ],
        5.05,
        5.41,
        7.75,
        1.35,
        size=10.8,
        space_after=6.0,
    )

    notes = (
        "Motivation slide. The left panel is reproduced from IceCube "
        "Collaboration, ApJ 833 (2016) 3, Fig. 5 (arXiv:1607.08006): the "
        "6-year through-going muon-neutrino measurement, with conventional "
        "atmospheric (blue), prompt-atmospheric (green dashed limit) and "
        "astrophysical (red band) components. It is a published measurement, "
        "not an illustrative curve. The network reconstructs E_entry at the "
        "instrumented volume, not E_nu: "
        "E_nu = E_entry + Delta E_prop + E_had, and E_entry is a lower bound "
        "when the vertex is outside. A response matrix and unfolding are "
        "required. Current working point: MAE ~0.475 in log10(E/GeV) "
        "(factor ~3), 47% within a factor 2, 78% within a factor 5, so broad "
        "bins only. There is no energy truth in experimental data, and "
        "MC/data systematics remain."
    )
    slide.notes_slide.notes_text_frame.text = notes


def main() -> None:
    prs = Presentation(str(DECK))
    build_slide(prs)
    prs.save(str(DECK))
    print(f"updated {DECK} (slide 5) with {CHART}")


if __name__ == "__main__":
    main()
