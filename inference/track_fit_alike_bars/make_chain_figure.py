"""Draw the BARS v2.3.2 reconstruction chain and where the port can differ.

    python inference/track_fit_alike_bars/make_chain_figure.py

Writes `figures/reconstruction_chain.png`, which `doc/track_fit_alike_bars.md`
embeds.  The figure is meant to be read without context: every box says what the
BARS task does in plain words, and every box carries a status chip saying whether
the port reproduces it and what that was checked against.

Box heights are computed from the text they hold (point sizes, one line = size x
line spacing), so editing a description cannot make it overlap the next box.

Status colours are the reserved status steps of the project's visual reference
(the dataviz palette: good `#0ca30c`, warning `#fab219`, critical `#d03b3b`), used
with a word as well as a colour, as the palette requires.  The palette's own
validator could not be run here (no JavaScript runtime on this machine); the steps
are used unchanged from the published reference rather than re-derived.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_name("figures") / "reconstruction_chain.png"

SURFACE = "#fcfcfb"
PANEL = "#f4f4f0"
INK = "#0b0b0b"
INK_2 = "#52514e"
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"
#: the hit sets a quantity can be computed from, so the same marker in a box and in
#: the summary table means the same set.  Categorical slots 1-3 of the palette, the
#: three it documents as validating all pairs.
SET_COLOUR = {"scanfit": "#2a78d6", "fit": "#eb6834", "all": "#1baf7a",
              "init": "#8a8a84", "recoll": "#8a8a84"}

STATUS = {  # status -> (icon, wording, colour)
    "same": ("✓", "identical, checked", GOOD),
    "fragile": ("~", "reproduced, but fragile", WARN),
    "differs": ("✗", "can differ", BAD),
}

FIG_W_PT = 12.6 * 72
MARGIN_PT = 26
GAP_PT = 15            # between boxes
TITLE_PT, BODY_PT, NOTE_PT = 11.5, 9.5, 8.5
LINE = 1.45            # line spacing


def height_pt(stage) -> float:
    """Box height: padding + title + body + note, measured in points."""
    lines = len(stage["body"].split("\n")) if stage["body"] else 0
    h = 15 + TITLE_PT * 1.2 + lines * BODY_PT * LINE
    if stage.get("produced"):
        h += NOTE_PT * LINE + 6
    if stage.get("note"):
        h += 6 + len(stage["note"].split("\n")) * NOTE_PT * LINE
    return h + 12


class Sheet:
    """A one-page vertical flow: place boxes by their measured height.

    Distances are in points from the top edge of the page; `y` converts one to a
    matplotlib coordinate.  `cursor` is the top edge of the next box.
    """

    def __init__(self, total_pt: float):
        self.total = total_pt
        self.fig = plt.figure(figsize=(FIG_W_PT / 72, total_pt / 72), dpi=150)
        self.fig.patch.set_facecolor(SURFACE)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, FIG_W_PT)
        self.ax.set_ylim(0, total_pt)
        self.ax.axis("off")
        self.cursor = 0.0

    def y(self, pt_from_top: float) -> float:
        return self.total - pt_from_top

    def text(self, x, pt_from_top, s, size, **kw):
        self.ax.text(x, self.y(pt_from_top), s, ha="left", va="top",
                     fontsize=size, **kw)

    def stage(self, x, w, stage, panel=False):
        h = height_pt(stage)
        top = self.cursor
        if panel:
            self.ax.add_patch(plt.Rectangle((x - 12, self.y(top + h)), w + 24, h,
                                            facecolor=PANEL, edgecolor="none",
                                            zorder=0))
        self.ax.add_patch(FancyBboxPatch(
            (x, self.y(top + h)), w, h, boxstyle="round,pad=0,rounding_size=5",
            linewidth=1.1, edgecolor=INK_2, facecolor=SURFACE, zorder=2))
        icon, wording, colour = STATUS[stage["status"]]
        self.ax.add_patch(plt.Rectangle((x, self.y(top + h)), 3.2, h,
                                        facecolor=colour, edgecolor="none", zorder=3))
        self.text(x + 16, top + 13, stage["title"], TITLE_PT, color=INK,
                  fontweight="bold", zorder=4)
        self.ax.text(x + w - 14, self.y(top + 14), f"{icon} {wording}", ha="right",
                     va="top", fontsize=NOTE_PT, color=INK_2, zorder=4)
        y = top + 15 + TITLE_PT * 1.2
        if stage.get("produced"):
            marker = stage["produced"]
            mx = x + 16
            self.ax.add_patch(plt.Rectangle((mx, self.y(y + 3) - 7), 7, 7,
                                            facecolor=SET_COLOUR[marker[0]],
                                            edgecolor="none", zorder=4))
            self.ax.text(mx + 12, self.y(y + 3) - 4, marker[1], ha="left",
                         va="center", fontsize=NOTE_PT, color=INK_2, zorder=4)
            y += NOTE_PT * LINE + 4
        if stage["body"]:
            self.text(x + 16, y + 5, stage["body"], BODY_PT, color=INK_2,
                      linespacing=LINE, zorder=4)
            y += len(stage["body"].split("\n")) * BODY_PT * LINE
        if stage.get("note"):
            self.text(x + 16, y + 7, stage["note"], NOTE_PT, color=INK_2,
                      style="italic", linespacing=LINE, zorder=4)
        self.cursor = top + h

    def table(self, x, w, rows):
        """A five-column mapping: marker, count, what the set is, what uses it."""
        col = (x + 18, x + 158, x + 440)
        self.cursor += 4
        for key, name, what, used_for in rows:
            y = self.cursor
            self.ax.add_patch(plt.Rectangle((x + 2, self.y(y + 11)), 8, 8,
                                            facecolor=SET_COLOUR[key],
                                            edgecolor="none", zorder=4))
            self.text(col[0], y, name, 9, color=INK, fontweight="bold", zorder=4)
            self.text(col[1], y, what, 8.5, color=INK_2, zorder=4)
            self.text(col[2], y, used_for, 8.5, color=INK_2, zorder=4)
            self.ax.plot([x, x + w], [self.y(y + 17)] * 2, color="#e2e2dc",
                         linewidth=0.8, zorder=1)
            self.cursor += 17
        self.cursor += 2

    def arrow(self, x):
        self.ax.add_patch(FancyArrowPatch(
            (x, self.y(self.cursor + 2)), (x, self.y(self.cursor + GAP_PT - 2)),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=INK_2,
            zorder=1))
        self.cursor += GAP_PT

    def save(self, path):
        self.fig.savefig(path, facecolor=SURFACE)
        plt.close(self.fig)


EVENT = [
    dict(title="BChannelMaskFilter", status="differs",
         body="Drops the pulses of channels that BARS' channel mask switches off.\n"
              "Data only — the MC files are reconstructed with every channel on.",
         note="The mask is not in the HDF5 files, so the port must be handed it (masks.py).\n"
              "Without it the scanfit mask disagrees with BARS on 6.6% of events; with it, on 0%."),
    dict(title="BCausalityClique", status="same",
         body="Pairs two pulses when |Δt − d/c| < 30 ns, with an extra 137 ns\n"
              "condition for distant OMs.  Finds the maximal cliques of ≥ 8 pulses\n"
              "and keeps the union of those no more than 2 smaller than the largest.",
         note="Ported with BARS' own pruning and vertex order; checked against networkx and\n"
              "against a line-by-line transliteration of its C++."),
    dict(title="BLocalTriggerSelect", status="same",
         body="A pulse above 3 p.e. with a neighbouring pulse above 1 p.e. within 100 ns.",
         note="Monitoring only — nothing downstream in this chain reads its mask."),
    dict(title="BGlobalTriggerSelect", status="same",
         body="Needs at least 2 strings and 8 channels among the selected pulses.",
         note="Otherwise the event is dropped here: this is where most of the raw\n"
              "trigger stream ends (94% of a random sample)."),
    dict(title="BInitializeCausScanGrid", status="same",
         body="Lays a 6° × 6° grid over the whole sky: 1891 candidate directions,\n"
              "the count BARS' float32 step accumulation produces (not 1800).",
         note="Reproduced down to that count."),
    dict(title="BScanFitClique  →  ScanfitMask", status="same",
         produced=("scanfit", "produces ScanfitMask — the set the fit starts from"),
         body="Per direction: pulses that are causal under a 30 ns window within 66 m\n"
              "form cliques of ≥ 7; each candidate clique is fitted in (x0, y0) by an\n"
              "M-estimator on the pulse times.  The best direction's clique becomes\n"
              "ScanfitMask — the hit set every later step starts from.",
         note="The GSL simplex is re-implemented and agrees with libgsl bit for bit.  The 180 s\n"
              "wall-clock limit on the scan is not reproduced — Python has no such timeout."),
    dict(title="BOfflineTriggerCut", status="same",
         body="Needs at least 2 strings and 8 channels in the scanfit mask.",
         note="Otherwise the event is dropped.  What survives is ~99% of the reco sample."),
]

FIT = [
    dict(title="1.  Start from the scanfit hits", status="same",
         body="Trial 0 takes every pulse of the ScanfitMask on an enabled channel."),
    dict(title="2.  Fit 15 candidate tracks", status="same",
         body="For the best scanfit directions: take that direction's pulses, start from\n"
              "its direction, and fit five parameters — x0, y0, \u03b8, \u03c6 and the time —\n"
              "to the pulse times (the plain \u03c7\u00b2, quality type 2)."),
    dict(title="3.  Keep the candidate with the smallest \u03c7\u00b2/(N \u2212 5)", status="same",
         body=""),
    dict(title="4.  RefitExclusive — remove one pulse", status="fragile",
         body="Try removing each pulse in turn, refit the five parameters, and keep the\n"
              "single removal that most improves Q/(N \u2212 5).  It is a best-of-N search, not a\n"
              "walk: exactly one pulse goes, and if none improves, the list is left alone.",
         note="A discrete choice among near-ties: a 10\u207b\u2075 change in the fit flips which pulse\n"
              "goes, so the final set inherits every difference in the fit.  (BARS' loop never\n"
              "even tries its last pulse, and skips the step altogether below 8 pulses.)"),
    dict(title="5.  RecollectHits — add pulses back", status="fragile",
         body="Walks every pulse of the event and adds those with q > 0.03 p.e. that lie\n"
              "within \u221210 \u2026 +10 ns of the track and closer than 100 m to it, at most one\n"
              "per channel.  The added pulses are used in the fit.",
         note="A threshold on the fitted track, so it inherits the same differences.  It is\n"
              "called a second time after the final fit, with a wider +40 ns window: that one\n"
              "only feeds the energy estimate and is counted separately, as fNHitsRecoll."),
    dict(title="6.  The fit that is reported", status="differs",
         produced=("fit", "produces the final pulse set — the one fNHits counts"),
         body="Fit the same five parameters again, now on the \u03c7\u00b2 plus a\n"
              "charge-weighted distance term (quality type 4).",
         note="ROOT's Minuit2 against iminuit: on 8-16-pulse events the \u03c7\u00b2 surface is flat,\n"
              "and the two land in different local minima on ~13% of events, by up to tens of\n"
              "degrees.  A tighter convergence tolerance moves the port away from BARS, not\n"
              "towards it, so tolerance is not the cause.  Where they differ, the port's \u03c7\u00b2\n"
              "is the lower one and, on single-muon events, its direction is the closer to\n"
              "the MC truth."),
]

#: The counts BARS writes, and the hit set each quantity is really computed from --
#: they are not the same set, which is the point of this table.
COUNTS = [
    ("scanfit", "ScanfitMask", "the clique the scanfit chose (no count of its own)",
     "starts the fit; q_acc_20 is summed over it"),
    ("fit", "fNHits", "pulses of the final fit",
     "the track and its errors, timeChi2, zDist, nTriplets, pHit - all the cuts"),
    ("all", "no count", "every pulse of the event",
     "q_tot_20, the paired-pulse test, t_res_30_med, log_phit_30_min, ev_center_z"),
    ("init", "fNHitsInit", "best candidate before the removal",
     "a diagnostic; only TimeChi2Init and FuncValueInit are read"),
    ("recoll", "fNHitsRecoll", "after the second, wider collection",
     "the energy estimate only; it is not the fit"),
]


def main() -> None:
    header = 104                 # title, subtitle, legend row, then the panel label
    body = (sum(height_pt(s) for s in EVENT) + GAP_PT * (len(EVENT) - 1)
            + 30 + sum(height_pt(s) for s in FIT) + GAP_PT * (len(FIT) - 1)
            + 48 + 17 * len(COUNTS) + 18 + 34)
    sheet = Sheet(header + body + 22)

    x = MARGIN_PT
    w = FIG_W_PT - 2 * MARGIN_PT
    sheet.text(x, 20, "How BARS v2.3.2 reconstructs a muon track, and where this port can differ",
               15, color=INK, fontweight="bold")
    sheet.text(x, 44, "One cluster's event.  Every box is a BARS task; the chip on its top right "
                      "says whether the port reproduces it, and what that was checked against.",
               9.5, color=INK_2)
    for i, (key, (icon, wording, colour)) in enumerate(STATUS.items()):
        lx = x + i * 250
        sheet.ax.add_patch(plt.Rectangle((lx, sheet.y(80)), 8, 8, facecolor=colour,
                                         edgecolor="none"))
        sheet.ax.text(lx + 13, sheet.y(79), f"{icon} {wording}", ha="left",
                      va="center", fontsize=8.5, color=INK_2)
    sheet.cursor = 100

    sheet.text(x, sheet.cursor + 6, "EVENT SELECTION — which pulses survive, and when the event "
                                    "is given up on", 10.5, color=INK, fontweight="bold")
    sheet.cursor += 26
    for i, stage in enumerate(EVENT):
        sheet.stage(x, w, stage)
        if i < len(EVENT) - 1:
            sheet.arrow(x + 32)

    sheet.cursor += 12
    sheet.text(x, sheet.cursor + 6, "THE FIT — BRecoMuonProducer, and the three steps that go "
                                    "their own way", 10.5, color=INK, fontweight="bold")
    sheet.cursor += 26
    for i, stage in enumerate(FIT):
        sheet.stage(x, w, stage, panel=True)
        if i < len(FIT) - 1:
            sheet.arrow(x + 32)

    sheet.cursor += 16
    sheet.text(x, sheet.cursor + 6, "THREE COUNTS, FIVE HIT SETS — what each quantity is "
                                    "really computed from", 10.5, color=INK,
               fontweight="bold")
    sheet.cursor += 26
    sheet.table(x, w, COUNTS)

    sheet.cursor += 12
    sheet.text(x, sheet.cursor, "Output: BRecoMuon — \u03b8_rec, \u03c6_rec, \u03b8_err, "
                                "covMatrixStatus, nHits, nStrings, timeChi2, scfMaxTheta, "
                                "nTriplets, zDist, evCenterZ, pHit, q_acc_20 \u2026\n"
                                "nCalls and covMatrixStatus count calls and states of a different "
                                "minimiser, so they do not transfer.",
               8.5, color=INK_2, linespacing=LINE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(f"written {OUT}  ({int(FIG_W_PT)}x{int(sheet.total)} pt)")


if __name__ == "__main__":
    main()