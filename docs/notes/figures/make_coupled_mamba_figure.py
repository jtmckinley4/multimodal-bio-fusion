"""The project's first design as built: Coupled Mamba and its two controls, beside concatenation.

Four fusion heads read the same codon-grid streams and feed the same TextCNN head. Left to right,
each adds one thing to the one before, so each neighboring pair isolates one property:
concatenation -> bottleneck (width), bottleneck -> uncoupled (recurrence along the codons),
uncoupled -> coupled (coupling of the modalities' states). The architecture is
Code/shared_code/fusion/coupled_mamba.py; nothing here is a result.

    python make_coupled_mamba_figure.py <output_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, SURFACE, FILL = "#c3c2b7", "#fcfcfb", "#e8e6df"
plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 7.5, "pdf.fonttype": 42})
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, ec=AXIS, colour=INK, fs=6.8, fc="white", bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc,
                                ec=ec, lw=1.0, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=colour,
            fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)


def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=0.9, ms=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=c,
                                 lw=lw, shrinkA=1.5, shrinkB=1.5, zorder=1))


STREAMS = [("DNA", DNA, 84), ("RNA", RNA, 76), ("protein", PROT, 68)]
ROWS = {"DNA": 56, "RNA": 46, "protein": 36}
STEPS = [16, 50, 84]


def inputs(ax, label):
    for name, c, y in STREAMS:
        ax.text(9, y, name, ha="left", va="center", fontsize=6.8, color=c, fontweight="bold")
        box(ax, 34, y - 3.2, 58, 6.4, label, ec=c, colour=c, fs=6.3)


def chains(ax, coupled):
    w, h = 10.0, 6.4
    for name, c, _ in STREAMS:
        y = ROWS[name]
        for i, x in enumerate(STEPS):
            box(ax, x - w / 2, y - h / 2, w, h, "h", ec=c, colour=c, fs=6.4)
            if i < len(STEPS) - 1 and not coupled:
                arrow(ax, x + w / 2, y, STEPS[i + 1] - w / 2, y, c=c, lw=1.0)
    if coupled:
        mw, mh = 13.0, 7.4
        for i in range(len(STEPS) - 1):
            xm = (STEPS[i] + STEPS[i + 1]) / 2
            box(ax, xm - mw / 2, 46 - mh / 2, mw, mh, "mean", ec=INK2, colour=INK, fs=5.9, fc=FILL)
            for name, c, _ in STREAMS:
                y = ROWS[name]
                arrow(ax, STEPS[i] + w / 2, y, xm - mw / 2, 46 + (y - 46) * 0.3, c=c, lw=0.8, ms=6)
                arrow(ax, xm + mw / 2, 46 + (y - 46) * 0.3, STEPS[i + 1] - w / 2, y, c=c, lw=0.8, ms=6)
    for i, x in enumerate(STEPS):
        ax.text(x, 29.5, ["codon t-1", "codon t", "codon t+1"][i], ha="center", va="center",
                fontsize=5.6, color=MUTED)


fig, axes = plt.subplots(1, 4, figsize=(7.6, 4.15))
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.17, wspace=0.04)
for ax in axes:
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
titles = [("Concatenation", "BioLangFusion head, Phase A"),
          ("mamba_bottleneck", "control"),
          ("uncoupled_mamba", "ablation"),
          ("coupled_mamba", "the design")]
for ax, (t, sub) in zip(axes, titles):
    ax.text(50, 99, t, ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK,
            family="monospace" if "_" in t else "sans-serif")
    ax.text(50, 92.5, sub, ha="center", va="top", fontsize=6.8, color=MUTED, style="italic")

# a: concatenation
ax = axes[0]
for name, c, y in STREAMS:
    ax.text(9, y, name, ha="left", va="center", fontsize=6.8, color=c, fontweight="bold")
    box(ax, 34, y - 3.2, 58, 6.4, "project DNA to 600" if name == "DNA" else "as encoded",
        ec=c, colour=c, fs=6.3)
box(ax, 10, 36, 80, 18, "concatenate the three\nstreams at each codon;\neach codon on its own", fs=6.6)
ax.text(50, 29.5, "no state from codon to codon", ha="center", va="center", fontsize=5.8, color=MUTED)

# b: bottleneck
ax = axes[1]
inputs(ax, "project to 128")
box(ax, 10, 36, 80, 18, "no Mamba layers", ec=MUTED, colour=MUTED, fs=6.8, ls=(0, (3, 2)))
ax.text(50, 29.5, "each codon on its own", ha="center", va="center", fontsize=5.8, color=MUTED)

# c: uncoupled
ax = axes[2]
inputs(ax, "project to 128")
chains(ax, coupled=False)

# d: coupled
ax = axes[3]
inputs(ax, "project to 128")
chains(ax, coupled=True)

for ax in axes:
    arrow(ax, 50, 63.5, 50, 60.5 if ax is not axes[0] and ax is not axes[1] else 55, c=MUTED)
for i, ax in enumerate(axes):
    text = "project to 600" if i == 0 else "normalize, concatenate\nat each codon, project to 600"
    box(ax, 12, 12.5, 76, 11, text, fs=6.2)
    arrow(ax, 50, 26.5, 50, 23.8)

eqs = ["", "",
       "h$^m_t$ = $\\bar{A}^m_t$ h$^m_{t-1}$ + $\\bar{B}^m_t$ x$^m_t$\neach modality keeps its own state",
       "h$^m_t$ = $\\bar{A}^m_t$ H$_{t-1}$ + $\\bar{B}^m_t$ x$^m_t$\nH$_{t-1}$ = mean over m of h$^m_{t-1}$"]
for ax, e in zip(axes, eqs):
    if e:
        ax.text(50, 3.0, e, ha="center", va="center", fontsize=6.0, color=INK2, linespacing=1.45)

# the ladder: what each neighboring pair isolates
labels = ["width: 128 per modality", "recurrence along the codons", "coupling of the states"]
for i, lab in enumerate(labels):
    x0 = axes[i].get_position().x0 + axes[i].get_position().width * 0.5
    x1 = axes[i + 1].get_position().x0 + axes[i + 1].get_position().width * 0.5
    y = 0.075
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=9, color=INK2, lw=1.0))
    fig.text((x0 + x1) / 2, y - 0.03, lab, ha="center", va="top", fontsize=6.8, color=INK,
             fontweight="bold")
fig.text(0.5, 0.995, "Same codon grid, same TextCNN head, same recipe and splits: only the fusion head changes",
         ha="center", va="top", fontsize=7.2, color=INK2, style="italic")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"coupled_mamba.{ext}"), dpi=220)
plt.close(fig)
print("coupled_mamba")
