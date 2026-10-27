"""Slide-shaped version of the three BioLangFusion heads: three panels side by side, each a
vertical flow from the three streams to the fused stream. Same style and example weights as
make_fusion_heads_figure.py (E. coli, seed 0: 0.11 / 0.00 / 0.89).

    python make_fusion_heads_slide.py <output_dir>
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
# The deck's canvas is a shade warmer than the manuscript's page; set SLIDE_BG to match it
# so the figure does not read as a pasted-in rectangle.
SURFACE = os.environ.get("SLIDE_BG", SURFACE)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 8, "pdf.fonttype": 42,
})
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

W, H, GAP = 0.78, 0.6, 0.1
N = 4
MODS = [("DNA", DNA), ("RNA", RNA), ("protein", PROT)]
ALPHA = [0.11, 0.00, 0.89]
FS = 8.2


def box(ax, x, y, w, h, fc, ec=None, lw=0.8, alpha=1.0, z=3, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or fc, lw=lw, alpha=alpha, zorder=z, ls=ls))


def rbox(ax, x, y, w, h, text, fs=FS, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor=SURFACE, edgecolor=AXIS, lw=0.9, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)


def stream(ax, x0, y0, color, alpha=1.0, ec=None, label=None, n=N, lw=0.8):
    xs = []
    for i in range(n):
        x = x0 + i * (W + GAP)
        box(ax, x, y0, W, H, color, ec=ec, alpha=alpha, lw=lw)
        xs.append(x + W / 2)
    if label:
        ax.text(x0 - 0.15, y0 + H / 2, label, ha="right", va="center", fontsize=FS, color=INK2)
    return xs


def arrow(ax, p, q, lw=1.0, rad=0.0, color=INK2, alpha=1.0, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5, alpha=alpha, shrinkA=1, shrinkB=1))


def note(ax, x, y, text, fs=FS - 0.6, color=INK2, ha="left", va="top"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.25)


def three_streams(ax, x0, ytop, step=0.85, alphas=(1, 1, 1), ec=None):
    rows = {}
    for k, (name, col) in enumerate(MODS):
        y = ytop - k * step
        rows[name] = (y, stream(ax, x0, y, col, alpha=alphas[k], label=name, ec=ec))
    return rows


fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0), gridspec_kw={"wspace": 0.05})
for ax in axes:
    ax.set_xlim(-1.2, 9.4)
    ax.set_ylim(-0.3, 9.6)
    ax.axis("off")

# ====================================================================== (a) concatenation
ax = axes[0]
ax.set_title("a   Concatenation: keep everything")
rows = three_streams(ax, 0.3, 8.3)
ax.text(0.3, 8.3 + H + 0.12, "three streams on the codon grid, one vector per codon", fontsize=FS - 0.6, color=MUTED, va="bottom")
# highlight codon 2
xc = rows["DNA"][1][1]
ax.add_patch(Rectangle((xc - W / 2 - 0.1, rows["protein"][0] - 0.1), W + 0.2, 2 * 0.85 + H + 0.2,
                       facecolor="none", edgecolor=INK, lw=1.0, ls=(0, (3, 2)), zorder=6))
# stacked vector for codon 2
sx, sy = 0.8, 2.3
heights = [1.0, 1.07, 0.53]
y = sy + sum(heights)
for (name, col), hh, lab in zip(MODS, heights, ["DNA (600)", "RNA (640)", "protein (320)"]):
    y -= hh
    box(ax, sx, y, 0.7, hh, col)
    ax.text(sx - 0.12, y + hh / 2, lab, ha="right", va="center", fontsize=FS - 0.8, color=INK2)
arrow(ax, (xc, rows["protein"][0] - 0.12), (sx + 0.35, sy + sum(heights) + 0.05), lw=1.1)
note(ax, 1.9, 5.95, "the three vectors at codon 2, glued\nend to end: 1,560 numbers (DNA first\nprojected from 512 to 600)")
# linear layer, fused vector, fused stream
rbox(ax, 2.2, 3.35, 3.1, 1.05, "linear layer\n1,560 to 600", fs=FS - 0.4)
arrow(ax, (sx + 0.75, 3.88), (2.12, 3.88), lw=1.1)
box(ax, 6.3, 3.25, 0.5, 1.25, FUSED)
ax.text(6.95, 3.88, "fused vector,\ncodon 2", ha="left", va="center", fontsize=FS - 0.8, color=INK2)
arrow(ax, (5.38, 3.88), (6.25, 3.88), lw=1.1)
fx = stream(ax, 2.1, 1.45, FUSED, label="fused")
arrow(ax, (6.55, 3.2), (fx[-1] + 0.1, 2.12), rad=-0.3, lw=1.0)
ax.text(5.75, 1.75, "the same at every codon", ha="left", va="center", fontsize=FS - 0.9, color=INK2)
note(ax, 0.3, 0.95, "Nothing is discarded. The TextCNN that follows\ndecides which features of which modality to use.", fs=FS - 0.6)

# ====================================================================== (b) gated attention
ax = axes[1]
ax.set_title("b   Gated attention: three weights")
rows = three_streams(ax, 0.3, 8.3)
ax.text(0.3, 8.3 + H + 0.12, "same streams", fontsize=FS - 0.6, color=MUTED, va="bottom")
# per-encoder means -> gates -> softmax -> weights
for k, (name, col) in enumerate(MODS):
    y0 = rows[name][0]
    arrow(ax, (rows[name][1][-1] + W / 2 + 0.05, y0 + H / 2), (4.15, y0 + H / 2), lw=1.0)
    box(ax, 4.2, y0, 0.5, H, col)
    rbox(ax, 4.95, y0 - 0.02, 0.95, H + 0.04, "gate", fs=FS - 0.6)
    arrow(ax, (4.72, y0 + H / 2), (4.9, y0 + H / 2), lw=0.9)
    arrow(ax, (5.92, y0 + H / 2), (6.3, y0 + H / 2), lw=0.9)
ax.text(4.45, 8.3 + H + 0.12, "mean over\ncodons", ha="center", va="bottom", fontsize=FS - 0.8, color=INK2)
ax.add_patch(Rectangle((6.35, rows["protein"][0] - 0.1), 0.36, 2 * 0.85 + H + 0.2, facecolor=GRID, edgecolor="none", zorder=2))
ax.text(6.53, rows["RNA"][0] + H / 2, "softmax", rotation=90, ha="center", va="center", fontsize=FS - 1, color=INK2, zorder=3)
for k, (name, col) in enumerate(MODS):
    y0 = rows[name][0]
    ax.text(7.0, y0 + H / 2, f"{ALPHA[k]:.2f}", ha="left", va="center", fontsize=FS + 1, color=INK, fontweight="bold")
ax.text(7.0, 8.3 + H + 0.12, "weights", ha="left", va="bottom", fontsize=FS - 0.8, color=INK2)
# weighted sum of the full streams
note(ax, 0.3, 5.75, "the three weights multiply the full streams, codon by codon:", fs=FS - 0.6)
wy = 4.55
for k, (name, col) in enumerate(MODS):
    y0 = wy - k * 0.75
    stream(ax, 1.25, y0, col, alpha=max(ALPHA[k], 0.08))
    ax.text(1.1, y0 + H / 2, f"{ALPHA[k]:.2f} ×", ha="right", va="center", fontsize=FS, color=INK)
    if k < 2:
        ax.text(1.25 + N * (W + GAP) + 0.05, y0 - 0.1, "+", ha="left", va="center", fontsize=11, color=INK)
fy = wy - 3 * 0.75 - 0.25
stream(ax, 1.25, fy, FUSED)
ax.text(1.1, fy + H / 2, "=", ha="right", va="center", fontsize=11, color=INK)
ax.text(1.25 + N * (W + GAP) + 0.1, fy + H / 2, "fused", ha="left", va="center", fontsize=FS, color=INK2)
note(ax, 0.3, 1.0, "Three numbers per sequence, decided from averages and\napplied at every codon; the entropy term pushes them toward\none-hot. With RNA at 0.00 the whole RNA stream is dropped.", fs=FS - 0.6)

# ====================================================================== (c) cross-modal attention
ax = axes[2]
ax.set_title("c   Cross-modal attention: read anything anywhere")
rows = three_streams(ax, 0.3, 8.3)
ax.text(0.3, 8.3 + H + 0.12, "same streams, projected to width 600: the queries", fontsize=FS - 0.6, color=MUTED, va="bottom")
# context row: 12 boxes
cy = 4.6
cw, cg = 0.5, 0.07
cx = []
x = 0.3
for k, (name, col) in enumerate(MODS):
    for i in range(N):
        box(ax, x, cy, cw, H, col)
        cx.append(x + cw / 2)
        x += cw + cg
    x += 0.15
ax.text(0.3, cy - 0.12, "context: the three streams laid end to end, 12 positions (keys and values)", ha="left", va="top", fontsize=FS - 0.8, color=INK2)
# query codon 2 of DNA attends into the context
qx, qy = rows["DNA"][1][1], rows["DNA"][0]
ax.add_patch(Rectangle((qx - W / 2 - 0.08, qy - 0.08), W + 0.16, H + 0.16, facecolor="none", edgecolor=INK, lw=1.0, ls=(0, (3, 2)), zorder=6))
for idx, lw in [(0, 0.7), (1, 1.6), (5, 2.4), (8, 2.8), (9, 0.9)]:
    arrow(ax, (qx, qy - 0.05), (cx[idx], cy + H + 0.03), lw=lw, color=INK, alpha=0.7, rad=-0.1 if cx[idx] > qx else 0.1, ms=7)
note(ax, 6.1, 7.75, "DNA codon 2 (the query)\nattends over all 12\npositions; line width\n= attention weight.\nRNA and protein\ndo the same.", fs=FS - 0.8)
# updated streams -> merge -> fused
rbox(ax, 0.3, 2.35, 7.0, 1.15, "three updated streams (residual, layer norm);\nconcatenate along features + linear,\naverage with the three, layer norm", fs=FS - 0.8)
arrow(ax, (3.7, cy - 0.55), (3.7, 3.55), lw=1.0)
fx = stream(ax, 2.6, 1.05, FUSED, label="fused")
arrow(ax, (3.7, 2.3), (3.7, 1.8), lw=1.0)
note(ax, 0.3, 0.8, "Every position can combine features from any modality at any\ncodon: the most expressive head, and the only one that never\nlost to the best single encoder in Phase A.", fs=FS - 0.6)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fusion_heads_slide.{ext}"), dpi=220, bbox_inches="tight")
print("fusion_heads_slide")
