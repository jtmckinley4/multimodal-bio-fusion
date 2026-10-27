"""The three BioLangFusion fusion heads, drawn as mechanisms.

Same style module as docs/notes/figures/make_r34_figures.py. The gated head's example
weights are the E. coli, seed 0, lambda 0.01 test-set means from
Code/stage4/results/phaseA/ecoli_proteins/.

    python make_fusion_heads_figure.py <output_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FUSED = "#0b0b0b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 6, "pdf.fonttype": 42,
})

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

W, H, GAP = 0.40, 0.34, 0.07          # one codon box
MODS = [("DNA", DNA), ("RNA", RNA), ("protein", PROT)]


def box(ax, x, y, w, h, fc, ec=None, lw=0.8, alpha=1.0, z=3, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or fc, lw=lw, alpha=alpha, zorder=z, ls=ls))


def rbox(ax, x, y, w, h, text, fc=SURFACE, ec=AXIS, fs=7, bold=False, z=3, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, lw=0.8, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", zorder=z + 1)


def stream(ax, x0, y0, color, n=6, alpha=1.0, ec=None, label=None, lw=0.8):
    xs = []
    for i in range(n):
        x = x0 + i * (W + GAP)
        box(ax, x, y0, W, H, color, ec=ec, alpha=alpha, lw=lw)
        xs.append(x + W / 2)
    if label:
        ax.text(x0 - 0.12, y0 + H / 2, label, ha="right", va="center", fontsize=7.5, color=INK2)
    return xs


def arrow(ax, p, q, color=INK2, lw=0.9, rad=0.0, z=4, alpha=1.0, style="-|>", ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, alpha=alpha,
                                 shrinkA=1, shrinkB=1))


def note(ax, x, y, text, fs=7, color=INK2, ha="left", va="top", bold=False, w=None):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, wrap=True,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def three_streams(ax, x0, ytop, label_codons=True, alphas=(1, 1, 1), ec=None):
    rows = {}
    for k, (name, col) in enumerate(MODS):
        y = ytop - k * (H + 0.16)
        rows[name] = (y, stream(ax, x0, y, col, alpha=alphas[k], label=name, ec=ec))
    if label_codons:
        for i, xc in enumerate(rows["DNA"][1]):
            ax.text(xc, ytop + H + 0.05, f"{i + 1}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.text(x0, ytop + H + 0.26, "codon position", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    return rows


fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.3), gridspec_kw={"hspace": 0.3, "height_ratios": [2.6, 3.1, 3.6]})
for ax, (lo, hi) in zip(axes, [(1.0, 3.75), (0.5, 3.75), (0.1, 3.95)]):
    ax.set_xlim(-0.9, 13.4)
    ax.set_ylim(lo, hi)
    ax.axis("off")

# ====================================================================== (a)
ax = axes[0]
ax.set_title("a   Codon-aware concatenation: keep everything, position by position")
rows = three_streams(ax, 0.3, 2.95)
# highlight codon 3
xc = rows["DNA"][1][2]
ax.add_patch(Rectangle((xc - W / 2 - 0.07, rows["protein"][0] - 0.08), W + 0.14, 3 * H + 2 * 0.16 + 0.16,
                       facecolor="none", edgecolor=INK, lw=0.9, ls=(0, (3, 2)), zorder=5))
note(ax, 0.3, 1.58, "the same operation at every codon", fs=6.5, color=MUTED)
# stacked vector
sx, sy = 4.35, 1.45
heights = [0.72, 0.77, 0.38]   # 600, 640, 320, drawn to scale-ish
labels = ["DNA, projected\n512 to 600", "RNA, 640", "protein, 320"]
y = sy + sum(heights)
for (name, col), hh, lab in zip(MODS, heights, labels):
    y -= hh
    box(ax, sx, y, 0.55, hh, col)
    ax.text(sx + 0.62, y + hh / 2, lab, ha="left", va="center", fontsize=6.5, color=INK2)
arrow(ax, (xc + W / 2 + 0.12, rows["RNA"][0] + H / 2), (sx - 0.05, sy + sum(heights) / 2), lw=1.0)
ax.text(sx + 0.62, sy - 0.02, "glued end to end:\n1,560 numbers", ha="left", va="top", fontsize=6.5, color=INK2)
# linear
rbox(ax, 6.55, 2.05, 1.45, 0.55, "linear layer\n1,560 to 600", fs=6.8)
arrow(ax, (sx + 0.6 + 1.05, sy + sum(heights) / 2), (6.5, 2.32), lw=1.0)
# fused vector
box(ax, 8.55, 2.0, 0.42, 0.65, FUSED)
ax.text(8.76, 1.9, "fused vector,\ncodon 3", ha="center", va="top", fontsize=6.5, color=INK2)
arrow(ax, (8.05, 2.32), (8.5, 2.32), lw=1.0)
# fused stream
fx = stream(ax, 9.7, 2.95, FUSED, label=None)
ax.text(9.7, 2.95 + H + 0.08, "fused stream: one vector per codon", ha="left", va="bottom", fontsize=6.5, color=INK2)
arrow(ax, (8.76, 2.7), (9.6, 3.0), rad=-0.25, lw=0.9)
rbox(ax, 9.7, 1.9, 2.3, 0.6, "fixed TextCNN head,\nthen prediction", fs=6.8)
arrow(ax, (fx[2], 2.9), (fx[2], 2.55), lw=0.9)
note(ax, 9.7, 1.72, "Nothing is discarded. The TextCNN decides\nwhich features of which modality to use.", fs=6.5, color=INK2)

# ====================================================================== (b)
ax = axes[1]
ax.set_title("b   Gated MIL attention: one weight per modality for the whole sequence")
rows = three_streams(ax, 0.3, 2.95)
alphas_ex = [0.11, 0.00, 0.89]
# mean over codons -> summary boxes
for k, (name, col) in enumerate(MODS):
    y0 = rows[name][0]
    box(ax, 4.35, y0, 0.42, H, col)
    arrow(ax, (rows[name][1][-1] + W / 2 + 0.05, y0 + H / 2), (4.3, y0 + H / 2), lw=0.9)
    rbox(ax, 5.15, y0 - 0.02, 1.05, H + 0.04, "gate", fs=6.8)
    arrow(ax, (4.8, y0 + H / 2), (5.1, y0 + H / 2), lw=0.9)
    arrow(ax, (6.22, y0 + H / 2), (6.75, y0 + H / 2), lw=0.9)
ax.text(4.56, 2.95 + H + 0.08, "mean over codons", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.text(5.67, rows["protein"][0] - 0.1, "gate: tanh(Vh) × σ(Uh),\nthen one score", ha="center", va="top", fontsize=6.5, color=INK2)
# softmax bracket and weights
ax.add_patch(Rectangle((6.8, rows["protein"][0] - 0.08), 0.28, 3 * H + 2 * 0.16 + 0.16, facecolor=GRID, edgecolor="none", zorder=2))
ax.text(6.94, rows["RNA"][0] + H / 2, "softmax", rotation=90, ha="center", va="center", fontsize=6.5, color=INK2, zorder=3)
for k, (name, col) in enumerate(MODS):
    y0 = rows[name][0]
    a = alphas_ex[k]
    box(ax, 7.25, y0 + 0.05, max(a, 0.012) * 1.1, H - 0.1, col)
ax.text(7.25, 2.95 + H + 0.08, "weights, add to 1\n(E. coli, seed 0)", ha="left", va="bottom", fontsize=6.5, color=INK2)
# weighted sum on the right
rx = 9.7
for k, (name, col) in enumerate(MODS):
    y0 = rows[name][0]
    stream(ax, rx, y0, col, alpha=max(alphas_ex[k], 0.07))
    ax.text(rx - 0.08, y0 + H / 2, f"{alphas_ex[k]:.2f} ×", ha="right", va="center", fontsize=6.8, color=INK)
ax.text(rx + 1.2, 2.95 + H + 0.08, "weighted sum, codon by codon", ha="center", va="bottom", fontsize=6.5, color=INK2)
fy = rows["protein"][0] - 0.62
stream(ax, rx, fy, FUSED)
ax.text(rx - 0.1, fy + H / 2, "=", ha="right", va="center", fontsize=8, color=INK)
ax.text(rx, fy - 0.08, "fused stream, then the same TextCNN", ha="left", va="top", fontsize=6.5, color=INK2)
note(ax, 0.3, 1.42, "Entropy term in the loss: λ × Σ α log α (λ = 0.01), which\n"
                    "rewards sharp weights. The decision is three numbers per\n"
                    "sequence: the head can choose a modality but cannot pick\n"
                    "different features from different modalities at different\n"
                    "codons. Here the RNA stream is multiplied by 0.00: dropped.", fs=6.5, color=INK2)

# ====================================================================== (c)
ax = axes[2]
ax.set_title("c   Cross-modal attention: every position can read every modality at every position")
# queries: three streams, small, top-left
rows = three_streams(ax, 0.3, 3.3, label_codons=False)
ax.text(0.3, 3.3 + H + 0.08, "the three streams, projected to width 600, are the queries", ha="left", va="bottom", fontsize=6.5, color=INK2)
# context row
cy = 1.0
cx = []
for k, (name, col) in enumerate(MODS):
    xs = stream(ax, 0.3 + k * 6 * (W + GAP) + k * 0.25, cy, col)
    cx.extend(xs)
    ax.text(xs[0] - W / 2, cy - 0.08, f"{name} 1 to 6", ha="left", va="top", fontsize=6.5, color=INK2)
ax.text(0.3, cy - 0.38, "context: the three streams laid end to end along the position axis, 18 positions (the keys and values).\n"
        "DNA codon 3 as the query attends over all 18 positions; line width = attention weight (4 heads).\n"
        "The RNA and protein streams do the same with their own attention; the three updated streams are then merged.",
        ha="left", va="top", fontsize=6.5, color=INK2, linespacing=1.3)
# attention arrows from DNA codon 3 (query) into the context
qx = rows["DNA"][1][2]
qy = rows["DNA"][0]
ax.add_patch(Rectangle((qx - W / 2 - 0.06, qy - 0.06), W + 0.12, H + 0.12, facecolor="none", edgecolor=INK, lw=0.9, ls=(0, (3, 2)), zorder=5))
targets = [(1, 0.6), (2, 1.2), (8, 2.2), (14, 2.4), (16, 0.8)]
for idx, lw in targets:
    arrow(ax, (qx, qy - 0.02), (cx[idx], cy + H + 0.02), lw=lw, rad=-0.12 if cx[idx] > qx else 0.1, color=INK, alpha=0.75, ms=6)
# updated streams and merge, right side
ux = 10.1
for k, (name, col) in enumerate(MODS):
    y0 = rows[name][0]
    stream(ax, ux, y0, col, n=3, ec=INK, lw=0.9)
ax.text(ux, 3.3 + H + 0.08, "updated streams (residual, layer norm)", ha="left", va="bottom", fontsize=6.5, color=INK2)
arrow(ax, (8.7, rows["RNA"][0] + H / 2), (ux - 0.08, rows["RNA"][0] + H / 2), lw=0.9)
ax.text(9.4, rows["RNA"][0] + H / 2 + 0.08, "after\nattention", ha="center", va="bottom", fontsize=6.3, color=INK2)
rbox(ax, ux, 1.58, 3.05, 0.62, "concatenate along features\n+ linear, averaged with the\nthree streams, layer norm", fs=6.1)
arrow(ax, (ux + 0.7, rows["protein"][0] - 0.04), (ux + 0.7, 2.25), lw=0.9)
fx = stream(ax, ux, 1.0, FUSED, n=5)
ax.text(ux, 0.92, "fused stream, then the same TextCNN", ha="left", va="top", fontsize=6.3, color=INK2)
arrow(ax, (ux + 0.7, 1.55), (ux + 0.7, 1.38), lw=0.9)

fig.text(0.02, 0.012, "All three heads start from the same input (three streams on the codon grid, one vector per codon per modality) "
         "and end in the same fixed TextCNN (convolutions over 3, 4 and 5 codons, a max over positions, one linear layer).\n"
         "Expressiveness: gated attention (three numbers per sequence) < concatenation (every feature, position by position) "
         "< cross-attention (every feature, any position to any position).", fontsize=6.6, color=INK2, va="bottom")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fusion_heads.{ext}"), dpi=220, bbox_inches="tight")
print("fusion_heads")
