"""Conceptual figure: what alignment means in biology.

    python make_alignment_figure.py <outdir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import make_figures as M
from make_figures import AXIS, DNA, INK, INK2, MUTED, PROT, RNA, SURFACE

M.title = lambda *a, **k: None
M.caption = lambda *a, **k: None
GREY = "#8a8880"
PANEL = "#f2f1ed"


def box(ax, x, y, w, h, text, fc="white", ec=AXIS, lw=1.0, fs=7.2, colour=INK, bold=False, r=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=colour,
                fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)


def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=0.9, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=c, lw=lw,
                                 shrinkA=1, shrinkB=1, zorder=1, connectionstyle=f"arc3,rad={rad}"))


def title(ax, text):
    ax.set_title(text, fontsize=9.5, fontweight="bold", loc="left", pad=8, color=INK)


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 5.0), gridspec_kw={"width_ratios": [0.85, 1.35, 1.0]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.03, wspace=0.05)
    for ax in axes:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- a: the machine-learning picture
    ax = axes[0]
    title(ax, "a   Alignment in machine learning")
    box(ax, 30, 80, 40, 11, "one scene", fc=PANEL, ec=INK, fs=7.6, bold=True)
    box(ax, 6, 56, 38, 11, "view 1\n(an image)", fs=7.0)
    box(ax, 56, 56, 38, 11, "view 2\n(a caption)", fs=7.0)
    arrow(ax, 42, 80, 25, 67); arrow(ax, 58, 80, 75, 67)
    box(ax, 6, 34, 38, 11, "encoder 1", fs=7.0)
    box(ax, 56, 34, 38, 11, "encoder 2", fs=7.0)
    arrow(ax, 25, 56, 25, 45); arrow(ax, 75, 56, 75, 45)
    box(ax, 22, 10, 56, 13, "same point in a\nshared space", fc=PANEL, ec=INK, fs=7.4, bold=True)
    arrow(ax, 25, 34, 40, 23); arrow(ax, 75, 34, 60, 23)
    ax.text(50, 2.5, "Two views of one thing share almost everything,\nso alignment is a training target.",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic", linespacing=1.4)

    # ---- b: the central dogma
    ax = axes[1]
    title(ax, "b   The central dogma is not two views of one thing")
    # DNA
    box(ax, 4, 74, 92, 13, "", fc="white", ec=DNA, lw=1.3)
    ax.text(6, 84.5, "genomic DNA", fontsize=7.2, color=DNA, fontweight="bold", ha="left", va="center")
    for x, w, t, fc in ((7, 15, "promoter", "white"), (24, 9, "exon", "#dbe7f7"), (35, 13, "intron", "white"),
                        (50, 9, "exon", "#dbe7f7"), (61, 12, "intron", "white"), (75, 9, "exon", "#dbe7f7"),
                        (86, 8, "...", "white")):
        ax.add_patch(Rectangle((x, 75.5), w, 6.5, fc=fc, ec=DNA, lw=0.7, zorder=3))
        ax.text(x + w / 2, 78.7, t, ha="center", va="center", fontsize=5.8, color=DNA, zorder=4)
    # transcript
    box(ax, 18, 47, 64, 13, "", fc="white", ec=RNA, lw=1.3)
    ax.text(20, 57.5, "transcript", fontsize=7.2, color=RNA, fontweight="bold", ha="left", va="center")
    for x, w, t, fc in ((21, 12, "5' UTR", "white"), (34, 30, "coding sequence", "#dff3ea"), (65, 15, "3' UTR", "white")):
        ax.add_patch(Rectangle((x, 48.5), w, 6.5, fc=fc, ec=RNA, lw=0.7, zorder=3))
        ax.text(x + w / 2, 51.7, t, ha="center", va="center", fontsize=6.0, color=RNA, zorder=4,
                fontweight="bold" if "coding" in t else "normal")
    # protein
    box(ax, 34, 20, 30, 13, "", fc="white", ec=PROT, lw=1.3)
    ax.text(36, 30.5, "protein", fontsize=7.2, color=PROT, fontweight="bold", ha="left", va="center")
    ax.add_patch(Rectangle((37, 21.5), 24, 6.5, fc="#fde4d8", ec=PROT, lw=0.7, zorder=3))
    ax.text(49, 24.7, "amino acids", ha="center", va="center", fontsize=6.0, color=PROT, zorder=4, fontweight="bold")
    # arrows and annotations
    arrow(ax, 49, 74, 49, 60, c=INK2, lw=1.0)
    ax.text(51.5, 67, "transcription and splicing", fontsize=6.6, color=INK2, ha="left", va="center")
    ax.text(51.5, 63.2, "drops introns and the genomic context", fontsize=6.2, color=MUTED, ha="left", va="center", style="italic")
    arrow(ax, 49, 47, 49, 33, c=INK2, lw=1.0)
    ax.text(51.5, 41.5, "translation: 61 sense codons to 20 amino acids", fontsize=6.6, color=INK2, ha="left", va="center")
    ax.text(51.5, 37.5, "drops the UTRs and the synonymous codon choice", fontsize=6.2, color=MUTED, ha="left", va="center", style="italic")
    # shared core bracket
    ax.add_patch(FancyArrowPatch((34, 46), (34, 34), arrowstyle="-", color=GREY, lw=0.8, ls=(0, (2, 2))))
    ax.add_patch(FancyArrowPatch((64, 46), (64, 34), arrowstyle="-", color=GREY, lw=0.8, ls=(0, (2, 2))))
    ax.text(5, 40, "shared by\nconstruction:\none codon is\none amino acid", fontsize=6.4, color=INK2, ha="left", va="center", linespacing=1.3)
    ax.add_patch(FancyArrowPatch((22, 40), (33, 40), arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.7))
    # unique information labels
    ax.text(5, 93.5, "unique to DNA: promoter, introns, exon junctions", fontsize=6.3, color=DNA, ha="left", va="center")
    ax.text(84, 53.5, "unique to RNA:\nUTRs, codon\nchoice", fontsize=6.3, color=RNA, ha="left", va="center", linespacing=1.3)
    ax.text(70, 26.5, "unique to protein:\nfold, function", fontsize=6.3, color=PROT, ha="left", va="center", linespacing=1.3)
    ax.text(50, 6, "The map runs one way and loses information at each step, so each molecule holds\n"
                   "information the next one cannot have. Only the coding sequence and its translation are shared.",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic", linespacing=1.4)

    # ---- c: what follows for alignment
    ax = axes[2]
    title(ax, "c   What follows for alignment")
    rows = [
        (86, "Given by the genetic code", "Positional alignment exists only on coding sequence,\n"
              "where every encoder's tokens sit on one codon grid.\n"
              "Stage 1: the grid resamples every stream; on GTEx\nno grid exists.", INK),
        (58, "Measured on the shared part, and partial", "Representational alignment is agreement that survives\n"
              "the composition control.\n"
              "Stage 1: CaLM still finds its protein partner 58% to 69%\n"
              "of the time; most DNA agreement is letter counts.", INK),
        (30, "Not forced on the unique parts", "An alignment loss would pull encoders toward what they\n"
              "share and erase what each one adds.\n"
              "Stage 1: translation erases GC3 (protein loses 0.26 to\n"
              "0.30); on GTEx, DNA adds signal with Recall@1 at most 0.016.", INK),
    ]
    for y, head, body, c in rows:
        box(ax, 4, y - 22, 92, 26, "", fc=PANEL, ec=AXIS, lw=0.9)
        ax.text(7, y + 0.5, head, fontsize=7.6, color=INK, fontweight="bold", ha="left", va="center")
        ax.text(7, y - 10.5, body, fontsize=6.4, color=INK2, ha="left", va="center", linespacing=1.35)
    ax.text(50, 2.5, "Alignment in biology: agreement on what the central dogma shares,\nbeyond composition, with nothing imposed on what it does not.",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic", linespacing=1.4)

    fig.savefig(os.path.join(outdir, "alignment_in_biology.pdf"))
    fig.savefig(os.path.join(outdir, "alignment_in_biology.png"), dpi=200)
    print("wrote alignment_in_biology")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
