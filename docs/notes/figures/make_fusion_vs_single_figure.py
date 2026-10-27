"""Does fusion beat the best single encoder? One figure for slide 5.

(a) The two scores behind each row: the best single encoder (open marker) and the best
fusion head (filled marker), on the published test split. (b) The paired-bootstrap
difference with its 95% interval; filled markers are resolved differences.

Values are the deck's and the Overleaf's (run records 2026-10-18 to 2026-10-22). E. coli
accuracy is shown as a fraction so that it shares the axis with Spearman.

    python make_fusion_vs_single_figure.py <output_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PUB = "#b3b1a8"
WIN = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 7.5, "legend.labelcolor": INK2, "pdf.fonttype": 42,
})
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
SLIDE = os.environ.get("SLIDE") == "1"   # larger type, no caption, for the deck
F = 1.35 if SLIDE else 1.0
if SLIDE:
    plt.rcParams.update({"font.size": 8 * F, "axes.titlesize": 9 * F, "xtick.labelsize": 7.5 * F, "ytick.labelsize": 7.5 * F, "legend.fontsize": 7.5 * F})

# label, single name, single score, fusion name, fusion score, delta, lo, hi, resolved, metric note
ROWS = [
    ("Fungal expression",                 "NT",      0.807, "cross-attention", 0.823,  0.016, -0.014, 0.048, False, ""),
    ("CoV-Vac",                           "RNA-FM",  0.838, "cross-attention", 0.836, -0.002, -0.026, 0.020, False, ""),
    ("mRNA stability, published split",   "RNA-FM",  0.552, "concatenation",   0.558,  0.007, -0.006, 0.019, False, ""),
    ("mRNA stability, unseen sequences",  "RNA-FM",  0.319, "concatenation",   0.359,  0.040, -0.003, 0.083, False, ""),
    ("E. coli, the paper's encoders",     "ESM-2",   0.517, "concatenation",   0.587,  0.069,  0.042, 0.096, True,  "accuracy"),
    ("E. coli, mRNA-FM in the RNA slot",  "mRNA-FM", 0.587, "concatenation",   0.590,  0.003, -0.029, 0.037, False, "accuracy"),
    ("GTEx, three different molecules",   "RNA",     0.452, "DNA + RNA + protein", 0.499, 0.047, 0.027, 0.069, True, ""),
]
n = len(ROWS)
y = np.arange(n)[::-1]

fig, (ax, bx) = plt.subplots(1, 2, figsize=((9.0, 3.9) if SLIDE else (7.6, 3.5)), gridspec_kw={"width_ratios": [1.45, 1.1], "wspace": 0.2})

# ---------------------------------------------------------------- (a) the two scores
ax.set_title("a   The two scores behind each comparison")
for yi, (lab, sn, sv, fn, fv, d, lo, hi, res, met) in zip(y, ROWS):
    ax.plot([sv, fv], [yi, yi], color=AXIS, lw=1.6, zorder=1)
    ax.scatter([sv], [yi], s=44, facecolor=SURFACE, edgecolor=INK, lw=1.3, zorder=3)
    ax.scatter([fv], [yi], s=44, color=WIN if res else INK, zorder=3)
    left, right = (sv, fv) if sv <= fv else (fv, sv)
    stext = f"{sn} {sv:.3f}" if not met else f"{sn} {sv * 100:.1f}%"
    ftext = f"{fn} {fv:.3f}" if not met else f"{fn} {fv * 100:.1f}%"
    if sv <= fv:
        ax.text(sv - 0.03, yi, stext, ha="right", va="center", fontsize=6.6 * F, color=INK2)
        ax.text(fv + 0.03, yi, ftext, ha="left", va="center", fontsize=6.6 * F, color=INK)
    else:
        ax.text(fv - 0.03, yi, ftext, ha="right", va="center", fontsize=6.6 * F, color=INK)
        ax.text(sv + 0.03, yi, stext, ha="left", va="center", fontsize=6.6 * F, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.5 * F)
ax.set_xlim(0.0, 1.12)
ax.set_xticks([0.3, 0.5, 0.7, 0.9])
ax.set_xlabel("test score (Spearman; E. coli as an accuracy fraction)", fontsize=7.5 * F)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="x"); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
# group separator between row 6 and 7
ax.axhline(0.5, color=AXIS, lw=0.8, ls=(0, (3, 2)))
ax.text(0.02, 6.62, "all three encoders read one coding sequence", fontsize=6.3 * F, color=MUTED, va="bottom")
ax.text(0.02, 0.5 - 0.95, "three different molecules, no codon grid", fontsize=6.3 * F, color=MUTED, va="bottom")
ax.set_ylim(-1.1, n + 0.15)
# legend, figure level
h1 = ax.scatter([], [], s=44, facecolor=SURFACE, edgecolor=INK, lw=1.3, label="best single encoder")
h2 = ax.scatter([], [], s=44, color=INK, label="best fusion head")
h3 = ax.scatter([], [], s=44, color=WIN, label="best fusion head, gain resolved")
fig.legend(handles=[h1, h2, h3], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.6, fontsize=6.8 * F)
# arrow: E. coli phase A -> phase B, drawn on the score panel
ya, yb = y[4], y[5]
ax.add_patch(FancyArrowPatch((0.517, ya - 0.28), (0.587, yb + 0.28), arrowstyle="-|>", mutation_scale=8,
                             color=MUTED, lw=0.9, connectionstyle="arc3,rad=-0.3", zorder=4))
ax.text(0.02, (ya + yb) / 2, ("swap RNA-FM\nfor mRNA-FM:\nsingle +8 pts" if SLIDE else "swap RNA-FM for\nmRNA-FM: the single\nencoder gains 8 points"), ha="left", va="center", fontsize=6.0 * F, color=MUTED, linespacing=1.2)

# ---------------------------------------------------------------- (b) the difference
bx.set_title("b   Fusion minus best single encoder, 95% interval")
bx.axvline(0, color=INK2, lw=0.9)
for yi, (lab, sn, sv, fn, fv, d, lo, hi, res, met) in zip(y, ROWS):
    bx.plot([lo, hi], [yi, yi], color=WIN if res else INK, lw=1.4, zorder=2)
    bx.scatter([d], [yi], s=44, facecolor=(WIN if res else SURFACE), edgecolor=(WIN if res else INK), lw=1.3, zorder=3)
    verdict = "fusion wins" if res else "tie"
    if lab.startswith("E. coli, mRNA-FM"):
        verdict = "tie: the gain is gone"
    if lab.endswith("unseen sequences"):
        verdict = "tie, leans fusion"
    bx.text(0.125, yi, verdict, ha="left", va="center", fontsize=6.8 * F, color=(WIN if res else INK2),
            fontweight="bold" if res else "normal")
    dtext = f"{d:+.3f}" if not met else f"{d * 100:+.1f} pts"
    bx.text(lo - 0.006, yi, dtext, ha="right", va="center", fontsize=6.4 * F, color=INK2)
bx.set_yticks(y); bx.set_yticklabels([])
bx.tick_params(axis="y", length=0)
bx.set_xlim(-0.09, 0.27)
bx.set_xticks([-0.05, 0, 0.05, 0.1]); bx.set_xticklabels(["-0.05", "0", "0.05", "0.10"])
bx.set_xlabel("fusion minus single (interval excludes 0: resolved)", fontsize=7.5 * F)
for side in ("top", "right", "left"):
    bx.spines[side].set_visible(False)
bx.grid(axis="x"); bx.set_axisbelow(True)
bx.axhline(0.5, color=AXIS, lw=0.8, ls=(0, (3, 2)))
bx.set_ylim(-1.1, n + 0.15)

if not SLIDE:
    fig.text(0.01, -0.03, "Best single encoder and best head chosen on validation, scored on the published test split; paired bootstrap, "
             "4,000 draws, seeds averaged inside each draw, repeated sequences resampled together. GTEx: IsoFormer's aggregator, one seed.",
             fontsize=6.3 * F, color=MUTED, va="top")
name = "fusion_vs_single_slide" if SLIDE else "fusion_vs_single"
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
print(name)
