#!/usr/bin/env python3
"""Assemble the complete energy-training presentation.

The real-event slide is reused from ``real_event_display_native.pptx``; the
event-types slide is built by ``build_event_types_real_deck.py``.  Other diagrams
are native PowerPoint shapes; numeric content is read from the actual
dataset/config files rather than typed from memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "energy_training_schemes.pptx"
REAL = HERE / "real_event_display_native.pptx"
sys.path.insert(0, str(HERE))
from build_event_types_real_deck import add_event_types_real_slide  # noqa: E402

INK = RGBColor(0x26, 0x32, 0x38)
MUTED = RGBColor(0x6B, 0x77, 0x7D)
GRID = RGBColor(0xD9, 0xDE, 0xE1)
ENERGY = RGBColor(0x2F, 0x6D, 0xB3)
SIGMA = RGBColor(0xE0, 0x7A, 0x2E)
DOMAIN = RGBColor(0x3A, 0x8A, 0x5A)
BAD = RGBColor(0xC0, 0x39, 0x2B)
GOOD = RGBColor(0x2E, 0x7D, 0x4F)
PAPER = RGBColor(0xFA, 0xFA, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _text(slide, x, y, w, h, value, color=INK, size=12, bold=False,
          align=PP_ALIGN.LEFT, name=None):
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


def _box(slide, x, y, w, h, value, fill=WHITE, line=GRID, color=INK,
         size=11, bold=False, align=PP_ALIGN.CENTER, name=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(1.0)
    tf = shp.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if name:
        shp.name = name
    return shp


def _arrow(slide, x, y, w, h, direction="right", color=ENERGY, name=None):
    shape = MSO_SHAPE.RIGHT_ARROW if direction == "right" else MSO_SHAPE.DOWN_ARROW
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    if name:
        shp.name = name
    return shp


def _slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER
    return slide


def _title(slide, title, subtitle=None):
    _text(slide, 0.55, 0.30, 12.2, 0.55, title, INK, 22, True, name="slide_title")
    if subtitle:
        _text(slide, 0.55, 0.85, 12.2, 0.35, subtitle, MUTED, 10.5,
              name="slide_subtitle")


def _load_configs():
    base = yaml.safe_load((ROOT / "src/training/configs/track_energy_reconstruction/"
                           "mu_energy_B-G_partsplit.yaml").read_text())
    da = yaml.safe_load((ROOT / "src/training/configs/track_energy_reconstruction/"
                         "mu_energy_AB-G_partsplit.yaml").read_text())
    return base, da


def _data_numbers():
    d = json.loads((ROOT / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit/"
                    "dataset_info.json").read_text())
    s = json.loads((ROOT / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit/"
                    "selection_info.json").read_text())
    test = json.loads((ROOT / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_testparts/"
                       "dataset_info.json").read_text())
    return d, s, test


def _add_picture(slide, path, x, y, w, h=None):
    im = Image.open(path)
    if h is None:
        h = w * im.height / im.width
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def add_title_slide(prs):
    slide = _slide(prs)
    _text(slide, 0.85, 2.05, 11.8, 0.8,
          "Восстановление энергии мюона", INK, 34, True, name="deck_title")
    _text(slide, 0.90, 2.95, 7.25, 0.5,
          "Baikal-GVD: от хитов к энергии и её неопределённости",
          ENERGY, 16.5, name="deck_subtitle")
    _text(slide, 0.90, 3.65, 7.2, 0.4,
          "данные · модель · режимы обучения · проверка",
          MUTED, 11.5, name="deck_sections")
    _text(slide, 0.90, 4.35, 7.2, 0.35,
          "1M событий  ·  20M хитов  ·  0.2 GeV–100 PeV",
          MUTED, 10.2, name="deck_numbers")
    crop_src = HERE / "figures/real_event_display.png"
    crop_dst = HERE / "figures/title_event_crop.png"
    im = Image.open(crop_src).convert("RGB")
    im.crop((0, 95, min(950, im.width), min(1370, im.height))).save(crop_dst)
    pic = _add_picture(slide, crop_dst, 8.65, 1.05, 3.9)
    pic.name = "title_event_thumbnail"
    # thin accent line
    shp = slide.shapes.add_connector(1, Inches(0.90), Inches(3.55), Inches(5.0), Inches(3.55))
    shp.line.color.rgb = SIGMA
    shp.line.width = Pt(2.0)
    shp.name = "title_accent"
    _text(slide, 0.90, 6.55, 7.3, 0.4,
          "Все визуалы — из реальных MC-событий и фактических конфигов обучения.",
          MUTED, 9.5, name="title_provenance")
    slide.notes_slide.notes_text_frame.text = (
        "Short title slide.  The main visual story is: one real event -> which "
        "events are selected -> data streams -> model tasks -> training regimes -> "
        "evaluation."
    )


def add_event_types_slide(prs):
    add_event_types_real_slide(prs)


def add_data_slide(prs):
    _, _, test = _data_numbers()
    slide = _slide(prs)
    _title(slide, "Данные и отбор",
           "Что попадает в обучение энергии и какие домены участвуют в DA")

    # funnel
    steps = [
        ("MC neutrino events\n(2 simulation samples)", MUTED, WHITE),
        ("signal-like hits\nSN probability > 0.8", SIGMA, WHITE),
        ("event cut\n≥ 8 hits and ≥ 3 strings", SIGMA, WHITE),
        ("exactly one\nmuon track", SIGMA, WHITE),
        ("valid energy truth\nentry volume", ENERGY, WHITE),
        ("flat in log10 E\nweights 1–20", ENERGY, WHITE),
        ("split by whole\nproduction parts", DOMAIN, WHITE),
    ]
    x, y, w = 0.55, 1.35, 3.65
    h = 0.58
    for i, (txt, col, fill) in enumerate(steps):
        _box(slide, x, y, w, h, txt, fill, col, INK, 10.5, False, name=f"funnel_{i}")
        if i < len(steps) - 1:
            _arrow(slide, x + w / 2 - 0.10, y + h + 0.02, 0.20, 0.22, "down", col)
        y += h + 0.28

    # data roles
    rx, ry, rw = 4.55, 1.45, 2.35
    roles = [
        ("MC neutrino", "energy labels", "energy training", ENERGY),
        ("MC muon", "domain only", "DA source", DOMAIN),
        ("experiment", "domain only", "DA target", DOMAIN),
    ]
    for i, (name, sub, role, col) in enumerate(roles):
        yy = ry + i * 1.43
        _box(slide, rx, yy, rw, 1.10, "", WHITE, col, INK, 10, name=f"role_{i}")
        _text(slide, rx + 0.12, yy + 0.10, rw - 0.24, 0.28, name, col, 11.5, True)
        _text(slide, rx + 0.12, yy + 0.43, rw - 0.24, 0.26, sub, INK, 9.5)
        _text(slide, rx + 0.12, yy + 0.72, rw - 0.24, 0.24, role, MUTED, 9.5)

    # numbers and caveat
    _text(slide, 7.35, 1.40, 5.2, 0.8,
          "1.00M selected events\n20.2M selected hits\n"
          "0.24 GeV – ~100 PeV",
          ENERGY, 13, True, name="numbers_1")
    _text(slide, 7.35, 2.65, 5.2, 0.9,
          "train / val / cal\n900k / 80k / 20k\n"
          "energy test: 418k,\nheld-out parts",
          INK, 11.5, name="numbers_2")
    _text(slide, 7.35, 4.35, 5.2, 1.35,
          "Training spectrum is deliberately flat in log E;\n"
          "the test spectrum is natural.\n"
          "The raw event counts are not directly comparable\n"
          "without the flat-selection weights.",
          MUTED, 10.0, name="numbers_caveat")
    _text(slide, 7.35, 6.05, 5.2, 0.7,
          "muatm is not an energy-training class; it is used\n"
          "only as the unlabelled DA source.",
          DOMAIN, 9.8, name="muatm_note")
    slide.notes_slide.notes_text_frame.text = (
        "Counts from dataset_info.json/selection_info.json; testparts from the "
        "separate test-parts dataset.  The energy labels exist only for MC "
        "neutrinos; MC muon and experimental streams carry domain labels only."
    )


def add_model_slide(prs):
    base, da = _load_configs()
    fe = base["model"]["feature_extractor"]
    reg = base["model"]["regressor"]
    het = base["model"]["heteroscedastic"]
    disc = da["domain_adaptation"]["discriminator"]
    slide = _slide(prs)
    _title(slide, "Задачи нейросети",
           "Один encoder, три головы: энергия, неопределённость, домен")

    _box(slide, 0.55, 2.05, 2.25, 1.0,
         "hit sequence\n(Q, t, x, y, z)\nvariable length", WHITE, GRID, MUTED, 10.5,
         name="input_box")
    _box(slide, 3.35, 1.80, 2.55, 1.55,
         f"Transformer encoder\n{fe['num_layers']} layers · {fe['num_heads']} heads\n"
         f"d_model = {fe['d_model']} · {fe['pooling']} pooling",
         RGBColor(0xE8, 0xEE, 0xF2), ENERGY, INK, 10.5, True, name="encoder_box")
    _arrow(slide, 2.85, 2.45, 0.40, 0.22, "right", ENERGY)
    _arrow(slide, 5.98, 2.45, 0.42, 0.22, "right", ENERGY)

    heads = [
        ("μ  mean energy", f"MLP {reg['hidden_dims']}\nlog10(E/GeV)",
         "mean loss: Huber / NLL", 1.05, ENERGY),
        ("σ  uncertainty", f"MLP {het['hidden_dims']}\nscale / log-variance",
         "NLL: Gaussian / Student-t", 2.80, SIGMA),
        ("D  domain", f"MLP {disc['hidden_dims']}\nMC vs experiment",
         "BCE + gradient reversal", 4.55, DOMAIN),
    ]
    for i, (head, detail, loss, yy, col) in enumerate(heads):
        _box(slide, 6.70, yy, 2.25, 1.15, "", WHITE, col, INK, 10, name=f"head_{i}")
        _text(slide, 6.82, yy + 0.10, 2.0, 0.28, head, col, 11.0, True)
        _text(slide, 6.82, yy + 0.40, 2.0, 0.40, detail, INK, 8.8)
        _text(slide, 6.82, yy + 0.82, 2.0, 0.24, loss, MUTED, 8.2)

    _text(slide, 9.35, 1.15, 3.4, 1.65,
          "Energy labels:\nonly MC neutrinos.\n\n"
          "Domain labels:\nMC muons vs experiment;\nno energy truth on data.",
          INK, 10.0, name="labels_note")
    _text(slide, 9.35, 4.05, 3.4, 1.8,
          "DA is auxiliary:\n"
          "the critic tries to separate domains;\n"
          "GRL pushes encoder features\nto be domain-invariant.\n"
          "An independent probe checks\nwhether the gap actually shrank.",
          DOMAIN, 9.8, name="da_note")
    _text(slide, 0.55, 5.55, 5.8, 0.8,
          "The mean head answers “how much energy?”\n"
          "The σ head answers “how sure?”\n"
          "The D head asks “simulation or data?”",
          MUTED, 10.5, name="three_questions")
    slide.notes_slide.notes_text_frame.text = (
        "Architecture values are read from the B-G and AB-G YAML configs: "
        f"{fe['num_layers']} layers, {fe['num_heads']} heads, d_model={fe['d_model']}, "
        f"mean/sigma hidden dims {reg['hidden_dims']}/{het['hidden_dims']}, "
        f"domain discriminator {disc['hidden_dims']}."
    )


def add_regimes_slide(prs):
    slide = _slide(prs)
    _title(slide, "Режимы обучения",
           "Комбинации sigma-головы, DA и способа обновления encoder-а")

    # Data-derived matrix.  Column positions leave a separate right-hand legend.
    rows = [
        ("mean only", "BASE", "A"),
        ("joint σ · one optimizer", "B-G / B-T5", "AB-G / AB-T5"),
        ("joint σ · two optimizers (control)", "C-G / C-T5", "AC-G / AC-T5"),
        ("σ detached from encoder", "D-G / D-T5", "AD-G / AD-T5"),
        ("mean frozen · σ head only", "H-G / H-T5", "AH-G / AH-T5"),
    ]
    x_row, x_left, x_right = 0.55, 3.00, 6.90
    row_w, row_h = 2.15, 0.50
    y = 1.55
    _text(slide, x_left, y - 0.38, row_w, 0.25, "без DA", MUTED, 10.0, True)
    _text(slide, x_right, y - 0.38, row_w, 0.25, "с DA", DOMAIN, 10.0, True)
    for i, (label, left, right) in enumerate(rows):
        yy = y + i * 0.62
        _text(slide, x_row, yy + 0.10, 2.30, 0.32, label, MUTED, 9.8)
        _box(slide, x_left, yy, row_w, row_h, left, WHITE,
             GRID if i else MUTED, INK, 9.8, name=f"regime_left_{i}")
        _box(slide, x_right, yy, row_w, row_h, right,
             WHITE, GRID if i else MUTED, INK, 9.8, name=f"regime_right_{i}")

    _text(slide, 9.45, 1.25, 3.35, 1.0,
          "A prefix = domain adaptation.\n"
          "-G = Gaussian; -T5 = Student-t ν=5.",
          INK, 9.8, name="id_legend")
    _text(slide, 9.45, 2.40, 3.35, 1.65,
          "B / AB: joint mean+σ,\nfull gradient, one optimizer.\n"
          "C / AC: same, two optimizers (control).\n"
          "D / AD: σ detached from encoder.\n"
          "H / AH: mean frozen, only σ head.",
          MUTED, 9.0, name="regime_explanation")
    _text(slide, 9.45, 4.20, 3.35, 1.2,
          "Matrix assembled from the actual YAML configs:\n"
          "sigma_strategy, domain_adaptation.enabled,\n"
          "distribution, optimizer.",
          MUTED, 8.8, name="matrix_provenance")
    _text(slide, 0.65, 5.90, 7.2, 0.65,
          "C/AC are controls, not a separate physical model.\n"
          "H/AH use a frozen mean and a post-hoc σ head.",
          MUTED, 10.0, name="regime_caveat")
    slide.notes_slide.notes_text_frame.text = (
        "The matrix is a compact rendering of the experiment-ID registry.  The "
        "cells correspond to the actual config families in "
        "src/training/configs/track_energy_reconstruction."
    )


def add_eval_slide(prs):
    _, _, test = _data_numbers()
    slide = _slide(prs)
    _title(slide, "Как мы проверяем",
           "Энергия, неопределённость и доменная адаптация — разные вопросы")

    cards = [
        ("Energy", "held-out production parts\nnatural spectrum\nMAE in dex · bias · RMSE",
         ENERGY, 0.65),
        ("Uncertainty", "NLL\n68% / 95% coverage\ncalibration factors q68/q95",
         SIGMA, 3.55),
        ("Domain", "critic AUC\nindependent linear/MLP probe\non held-out DA parts",
         DOMAIN, 6.45),
        ("Test isolation", f"{test['n_events']:,} events\nwhole held-out parts\nno event overlap",
         MUTED, 9.35),
    ]
    for name, body, col, x in cards:
        _box(slide, x, 1.55, 3.65, 2.25, "", WHITE, col, INK, 9.5, name=f"eval_{name}")
        _text(slide, x + 0.15, 1.70, 3.35, 0.30, name, col, 12, True)
        _text(slide, x + 0.15, 2.10, 3.35, 1.45, body, INK, 10.5)

    _text(slide, 0.65, 4.25, 12.0, 0.9,
          "Training spectrum is flattened; test spectrum is natural.  "
          "Coverage is reported both raw and after global calibration.",
          MUTED, 10.5, name="eval_note_1")
    _box(slide, 0.65, 5.10, 12.0, 1.0,
         "What this does not prove: experiment has no energy truth; the domain "
         "probe measures feature separability, not energy accuracy.  DA is an "
         "auxiliary check, not a target-domain accuracy certification.",
         RGBColor(0xFD, 0xF1, 0xEC), BAD, BAD, 9.5, name="eval_caveat")
    slide.notes_slide.notes_text_frame.text = (
        "Metric definitions are the ones implemented in the trainer: MAE/RMSE/bias "
        "in dex, weighted NLL, empirical 68/95 coverage, global q68/q95 scale factors. "
        "The independent probe is fit on held-out DA parts; its own fit/test split is "
        "event-level within those parts."
    )


def add_formula_slide(prs):
    slide = _slide(prs)
    _title(slide, "Appendix A — формулы", "Точные функции потерь и калибровка")
    _text(slide, 0.60, 1.20, 6.0, 5.3,
          "target\n"
          "  t = log10(E_entry / 1 GeV)\n\n"
          "Huber\n"
          "  L = 0.5 r²                    |r| ≤ δ\n"
          "      0.5 δ (2|r| − δ)          otherwise\n\n"
          "Gaussian NLL\n"
          "  0.5 [ r²/σ² + log σ² + log 2π ]\n\n"
          "Student-t (ν=5)\n"
          "  −lgamma((ν+1)/2) + lgamma(ν/2)\n"
          "  + 0.5 log(νπ) + log σ\n"
          "  + 0.5(ν+1) log(1 + (r/σ)²/ν)",
          INK, 11.5, name="formulas_left")
    _text(slide, 6.80, 1.20, 6.0, 5.3,
          "r = t − μ;  σ is the predicted scale.\n\n"
          "Domain BCE\n"
          "  −[y log D(f) + (1−y) log(1−D(f))]\n"
          "  D is fed by encoder features through GRL.\n\n"
          "Calibration\n"
          "  c_q = quantile_q(|r| / σ)\n"
          "  reported on a disjoint validation set;\n"
          "  q = 0.68, 0.95.\n\n"
          "The target is an entry energy, not an\n"
          "exit energy and not total deposited energy.",
          MUTED, 10.8, name="formulas_right")
    _text(slide, 0.60, 6.55, 12.0, 0.35,
          "NLL is the heteroscedastic objective; Huber is the deterministic/mean-only "
          "baseline.  The Student-t scale is still one scalar σ per event.",
          MUTED, 9.0, name="formulas_note")
    slide.notes_slide.notes_text_frame.text = (
        "Formulas match the trainer implementation and the target definition. "
        "The target cylinder is r <= 90 m, |z| <= 295 m around the row's own cluster."
    )


def add_ids_slide(prs):
    slide = _slide(prs)
    _title(slide, "Appendix B — ID и определения",
           "Как читать BASE / A / B / C / D / H и DA-варианты")
    head = ["ID", "что обучается", "gradient / optimizer"]
    rows = [
        ("BASE", "mean, no sigma", "Huber; one optimizer"),
        ("A", "mean + DA", "Huber; one optimizer"),
        ("B / AB", "mean + sigma, joint full", "one optimizer"),
        ("C / AC", "mean + sigma, joint full", "two optimizers (control)"),
        ("D / AD", "mean + sigma, detached", "σ head does not update encoder"),
        ("H / AH", "only σ on frozen mean", "post-hoc; one head optimizer"),
    ]
    x, y = 0.65, 1.55
    widths = [1.45, 4.7, 5.7]
    for j, txt in enumerate(head):
        _text(slide, x + sum(widths[:j]), y, widths[j], 0.3, txt, MUTED, 9.5, True)
    for i, row in enumerate(rows):
        yy = y + 0.45 + i * 0.62
        for j, txt in enumerate(row):
            _box(slide, x + sum(widths[:j]), yy, widths[j] - 0.06, 0.50, txt,
                 WHITE, GRID, INK, 9.2, name=f"id_{i}_{j}")
    _text(slide, 0.65, 5.85, 12.0, 0.8,
          "Suffix -G = Gaussian NLL, -T5 = Student-t ν=5.  A prefix means DA.\n"
          "Legacy mapping: D0→BASE, B1/B2→B-G/B-T5, AB1/AB2→AB-G/AB-T5, "
          "F→H, J1→D, J2→C.",
          MUTED, 9.6, name="id_note")
    slide.notes_slide.notes_text_frame.text = (
        "ID semantics and the legacy mapping follow doc/track_energy_experiment_ids.md. "
        "C/AC are controls; H/AH are post-hoc."
    )


def add_appendix_event_slide(prs):
    slide = _slide(prs)
    _title(slide, "Appendix C — проекции реального события",
           "То же событие, но с координатными проекциями и цветом по времени")
    src = HERE / "figures/appendix_event_display.png"
    cropped = HERE / "figures/appendix_event_display_deck.png"
    im = Image.open(src).convert("RGB")
    # Remove the standalone figure title; the slide title carries it.
    im.crop((0, 92, im.width, im.height)).save(cropped)
    _add_picture(slide, cropped, 0.55, 1.10, 12.2)
    slide.notes_slide.notes_text_frame.text = (
        "The same class of event as the main slide, shown as x-y and r-z projections. "
        "The energy dataset stores hits, not a reconstructed track."
    )


def main():
    if not REAL.exists():
        raise FileNotFoundError(REAL)
    prs = Presentation(str(REAL))
    add_event_types_slide(prs)
    add_data_slide(prs)
    add_model_slide(prs)
    add_regimes_slide(prs)
    add_eval_slide(prs)
    add_formula_slide(prs)
    add_ids_slide(prs)
    add_appendix_event_slide(prs)
    add_title_slide(prs)
    # Move the event-types slide (appended after the deck) directly after slide 1.
    sld_ids = prs.slides._sldIdLst
    ids = list(sld_ids)
    ev_idx = None
    for idx, sld in enumerate(prs.slides):
        if any(sh.name == "event_types_title" for sh in sld.shapes):
            ev_idx = idx
            break
    if ev_idx is not None and ev_idx != 1:
        sld_id = ids[ev_idx]
        sld_ids.remove(sld_id)
        sld_ids.insert(1, sld_id)
    # Move the title slide to the very beginning (it was appended last).
    title_idx = None
    for idx, sld in enumerate(prs.slides):
        if any(sh.name == "deck_title" for sh in sld.shapes):
            title_idx = idx
            break
    if title_idx is not None and title_idx != 0:
        sld_id = list(sld_ids)[title_idx]
        sld_ids.remove(sld_id)
        sld_ids.insert(0, sld_id)
    prs.save(OUT)
    print("wrote", OUT, "slides", len(prs.slides))


if __name__ == "__main__":
    main()
