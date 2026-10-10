"""Paper-layout Stage 1 figures for the Overleaf manuscript.

Two things happen here.

1. The six figures from make_figures.py are re-rendered without their embedded
   title and caption (the LaTeX caption carries that text) and with larger
   type, so they stay legible when scaled to the text width of a letter page.
2. Additional figures are built from the same 66-pair tables (``*_geom.csv``,
   ``*_gain.csv``, ``*_share.csv``) and from values recorded in the manuscript,
   all of them saved outputs of the October 6 notebook runs.

    cd docs/notes/figures
    python make_paper_figures.py paper

Writes one PDF (for Overleaf) and one PNG per figure into the output folder.
Set STAGE1_TABLES to point at the csv files if they live elsewhere.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import figdata as D
import make_figures as M
from make_figures import (AFTER, AXIS, DNA, GRID, INK, INK2, MODC, MUTED, PROT, RNA,
                          SURFACE, tidy)

# The LaTeX caption carries the title and the reading note.
M.title = lambda *a, **k: None
M.caption = lambda *a, **k: None

SEQ = LinearSegmentedColormap.from_list(
    "ink", ["#fbfaf7", "#f3e3d2", "#e8b48c", "#d9743f", "#b53f16", "#6e2109"])
DIV = LinearSegmentedColormap.from_list(
    "gain", ["#2a5c9e", "#7fa6d2", "#fbfaf7", "#e9a070", "#b53f16"])


def scale_text(fig, factor):
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontsize(t.get_fontsize() * factor)


def resize(fig, w, h):
    fig.set_size_inches(w, h)


# ------------------------------------------------- the six, paper layout
def paper_fig1():
    fig = M.fig1()
    scale_text(fig, 1.12)
    return fig


def paper_fig2(stab, gtex):
    fig = M.fig2(stab, gtex)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.17, wspace=0.42)
    resize(fig, 11.0, 4.3)
    scale_text(fig, 1.22)
    for ax, t in zip(fig.axes, ("a   Composition share of each embedding",
                                "b   Stability: less composition, better probe",
                                "c   GTEx: no such relation")):
        ax.set_title(t, fontsize=9.5)
    fig.axes[1].set_xlabel("Composition share of the embedding (%)")
    fig.axes[2].set_xlabel("Composition share (%)")
    return fig


def paper_fig3(stab):
    fig = M.fig3(stab)
    fig.subplots_adjust(left=0.275, right=0.985, top=0.935, bottom=0.11, wspace=0.06)
    resize(fig, 10.6, 5.4)
    scale_text(fig, 1.22)
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() == "survive the control":
            t.set_color(INK)
    return fig


def paper_fig4():
    fig = M.fig4()
    fig.subplots_adjust(left=0.19, right=0.975, top=0.92, bottom=0.12, wspace=0.30)
    resize(fig, 10.4, 4.6)
    scale_text(fig, 1.2)
    fig.axes[0].set_title("a   Agreement peaks early and decays with depth", fontsize=9.5)
    fig.axes[1].set_title("b   mRNA-FM's isolation arises in its last blocks", fontsize=9.5)
    return fig


def paper_fig5(stab, gtex):
    fig = M.fig5(stab, gtex)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.26, wspace=0.26)
    for leg in fig.legends:
        leg.set_bbox_to_anchor((0.5, 0.0))
        leg.set_loc("lower center")
    resize(fig, 10.6, 5.3)
    scale_text(fig, 1.22)
    fig.axes[0].set_title("a   mRNA stability: same coding sequence to every encoder", fontsize=9.5)
    fig.axes[1].set_title("b   GTEx: a different molecule per modality", fontsize=9.5)
    return fig


def paper_fig6():
    fig = M.fig6()
    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.13, wspace=0.06)
    resize(fig, 10.6, 5.1)
    scale_text(fig, 1.22)
    fig.axes[0].set_xticks([0, 0.05, 0.10, 0.15])
    return fig


# ---------------------------------------------------------------- helpers
def matrix(frame, col):
    """Symmetric 12 x 12 matrix of one column of a 66-pair table."""
    m = np.full((12, 12), np.nan)
    idx = {k: i for i, k in enumerate(D.KEYS)}
    for pair, v in frame[col].items():
        a, b = pair.split("|")
        m[idx[a], idx[b]] = v
        m[idx[b], idx[a]] = v
    return m


def heat(ax, m, cmap, norm, fmt="{:.2f}", fs=6.4, dot=None, light_above=0.55):
    im = ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
    n = m.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(m[i, j]):
                continue
            v = m[i, j]
            shade = norm(v)
            colour = "white" if shade > light_above else INK
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color=colour)
            if dot is not None and dot[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec=INK, lw=1.3, zorder=3))
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fc="#e8e6df", ec="none", zorder=2))
    ax.set_xticks(range(n), [D.SHORT[k] for k in D.KEYS], rotation=55, ha="right",
                  rotation_mode="anchor", fontsize=7.2)
    ax.set_yticks(range(n), [D.SHORT[k] for k in D.KEYS], fontsize=7.2)
    for lab, k in zip(ax.get_xticklabels(), D.KEYS):
        lab.set_color(MODC[D.MOD[k]])
    for lab, k in zip(ax.get_yticklabels(), D.KEYS):
        lab.set_color(MODC[D.MOD[k]])
    for cut in (3.5, 7.5):
        ax.axhline(cut, color=INK, lw=1.0)
        ax.axvline(cut, color=INK, lw=1.0)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    return im


def panel_title(ax, text):
    ax.set_title(text, fontsize=9.5, fontweight="bold", loc="left", pad=8, color=INK)


# ------------------------------------------- agreement matrices (per data)
def fig_agreement(data, dataset, chance_r1):
    p = data["pairs"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.4))
    fig.subplots_adjust(left=0.085, right=0.915, top=0.955, bottom=0.07, wspace=0.30, hspace=0.40)
    norm1 = matplotlib.colors.Normalize(0, 1)
    specs = [
        (axes[0, 0], "CKA", f"a   Linear CKA, before the composition control"),
        (axes[0, 1], "CKA_after", f"b   Linear CKA, after the composition control"),
        (axes[1, 0], "R1", f"c   CCA retrieval Recall@1, before the control  (chance {chance_r1:.3f})"),
        (axes[1, 1], "R1_after", f"d   CCA retrieval Recall@1, after the control"),
    ]
    for ax, col, head in specs:
        im = heat(ax, matrix(p, col), SEQ, norm1)
        panel_title(ax, head)
    cax = fig.add_axes([0.935, 0.30, 0.013, 0.42])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label("0 = unrelated,  1 = identical", fontsize=7.5, color=INK2)
    return fig


# --------------------------------------------- concatenation gain matrices
def fig_gains(stab, gtex):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    fig.subplots_adjust(left=0.085, right=0.95, top=0.90, bottom=0.14, wspace=0.30)
    for ax, data, head, lim in (
            (axes[0], stab, "a   mRNA stability: mean gain in $R^2$", 0.045),
            (axes[1], gtex, "b   GTEx expression: mean gain in $R^2$", 0.045)):
        p = data["pairs"]
        m = matrix(p, "gain")
        n10 = matrix(p, "n_pos") == 10
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = heat(ax, m, DIV, norm, fmt="{:+.3f}", fs=5.6, dot=n10, light_above=0.80)
        panel_title(ax, head)
    cax = fig.add_axes([0.962, 0.22, 0.012, 0.56])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.04, -0.02, 0, 0.02, 0.04])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label("gain over the better single encoder", fontsize=7.5, color=INK2)
    handles = [Line2D([], [], marker="s", ls="", mfc="none", mec=INK, markersize=9, mew=1.3,
                      label="boxed: gains under all ten fold assignments")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=8)
    return fig


# ---------------------------------------------------- Track A controls
GC = {  # GC R2, GC3 R2, Nussinov R2, Nussinov fold SD   (Table: Track A)
    "nt500m": (0.985, 0.968, 0.038, 0.084), "ntv2_100m": (0.991, 0.986, 0.060, 0.083),
    "dnabert2": (0.990, 0.948, 0.046, 0.148), "hyenadna": (0.997, 0.896, 0.316, 0.159),
    "rnafm": (0.951, 0.917, 0.142, 0.084), "rinalmo": (0.954, 0.949, 0.093, 0.153),
    "mrnafm": (0.951, 0.946, 0.098, 0.045), "calm": (0.984, 0.982, 0.070, 0.024),
    "esm2_8m": (0.679, 0.379, 0.130, 0.071), "esm2_35m": (0.686, 0.387, 0.088, 0.052),
    "esm2_150m": (0.708, 0.436, 0.098, 0.103), "protbert": (0.700, 0.440, 0.077, 0.093),
}


def fig_controls():
    order = list(reversed(D.KEYS))
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.14, wspace=0.10)

    ax = axes[0]
    for yi, k in zip(y, order):
        gc, gc3 = GC[k][0], GC[k][1]
        c = MODC[D.MOD[k]]
        ax.plot([gc3, gc], [yi, yi], color=c, lw=2.6, alpha=0.35, zorder=1, solid_capstyle="round")
        ax.scatter([gc], [yi], s=46, facecolor=SURFACE, edgecolor=c, lw=1.4, zorder=3)
        ax.scatter([gc3], [yi], s=46, color=c, zorder=4, edgecolor=SURFACE, lw=0.8)
        drop = gc - gc3
        if drop >= 0.05:
            ax.text(gc3 - 0.015, yi, f"drop {drop:.2f}", ha="right", va="center",
                    fontsize=7, color=INK2)
    ax.set_yticks(y, [D.SHORT[k] for k in order])
    ax.set_xlim(0.0, 1.03)
    ax.set_xlabel("Held-out $R^2$ (pinned folds)")
    panel_title(ax, "a   GC content (open) and GC3 (filled): what translation erases")
    handles = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, markersize=6, label="GC content"),
               Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, markersize=6, label="GC3 (third codon position)")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5)
    tidy(ax)

    ax = axes[1]
    for yi, k in zip(y, order):
        v, sd = GC[k][2], GC[k][3]
        c = MODC[D.MOD[k]]
        ax.plot([max(v - sd, -0.12), v + sd], [yi, yi], color=c, lw=1.3, alpha=0.4, zorder=2)
        ax.scatter([v], [yi], s=46, color=c, zorder=4, edgecolor=SURFACE, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y, [""] * len(order))
    ax.set_xlim(-0.12, 0.5)
    ax.set_xlabel("Held-out $R^2$ (mean ± fold SD),  n = 200")
    panel_title(ax, "b   Nussinov pairing fraction (first 200 nt)")
    tidy(ax)
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=6, label=m) for m, c in MODC.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    return fig


# ----------------------------------------------- published-split comparison
BLF_SINGLE = {  # Spearman, all test rows; Spearman, unseen sequences
    "nt500m": (0.275, 0.280), "ntv2_100m": (0.275, 0.290), "dnabert2": (0.297, 0.270),
    "hyenadna": (0.286, 0.224), "rnafm": (0.292, 0.317), "rinalmo": (0.322, 0.330),
    "mrnafm": (0.328, 0.326), "calm": (0.377, 0.391), "esm2_8m": (0.329, 0.299),
    "esm2_35m": (0.329, 0.312), "esm2_150m": (0.371, 0.373), "protbert": (0.396, 0.433),
}
BLF_COMBO = [("mRNA-FM + ESM-2 8M", 0.336, 0.314), ("mRNA-FM + ESM-2 35M", 0.337, 0.320),
             ("mRNA-FM + ESM-2 150M", 0.377, 0.368), ("mRNA-FM + ProtBERT", 0.392, 0.417),
             ("NT v2 + RNA-FM + ESM-2 8M", 0.364, 0.351), ("All twelve encoders", 0.401, 0.402)]
BLF_PUBLISHED = [("Nucleotide Transformer", 0.530), ("RNA-FM", 0.553), ("ESM-2 8M", 0.539),
                 ("Concatenation", 0.539), ("Cross-attention", 0.550), ("MIL + entropy", 0.563)]

ISO_ROWS = [  # label, frozen-probe R2 (lo, hi), IsoFormer published R2
    ("DNA alone", (0.034, 0.054), 0.13),
    ("RNA alone", (0.150, 0.204), 0.36),
    ("Protein alone", (0.113, 0.200), 0.20),
    ("DNA + protein", (0.206, 0.206), 0.28),
    ("DNA + RNA", (0.163, 0.235), 0.39),
    ("DNA + RNA + protein", (0.216, 0.249), 0.43),
    ("All twelve encoders", (0.269, 0.269), None),
]
ISO_POINT = {"DNA alone": 0.054, "Protein alone": 0.179}   # IsoFormer's own encoders: NT v2, ESM-2 150M


def fig_published():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.165, right=0.985, top=0.91, bottom=0.21, wspace=0.55)

    ax = axes[0]
    rows = [(D.SHORT[k], BLF_SINGLE[k][0], BLF_SINGLE[k][1], MODC[D.MOD[k]]) for k in D.KEYS]
    rows += [(lab, a, u, INK2) for lab, a, u in BLF_COMBO]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, a, u, c) in zip(y, rows):
        ax.hlines(yi, 0, a, color="#e6e4dd", lw=1.4, zorder=1)
        ax.scatter([a], [yi], s=44, color=c, zorder=4, edgecolor=SURFACE, lw=0.8)
        ax.scatter([u], [yi], s=30, facecolor="none", edgecolor=c, lw=1.1, zorder=3)
    ax.axhline(5.5, color=AXIS, lw=0.8, ls=(0, (3, 2)))
    for lab, v in BLF_PUBLISHED:
        ax.axvline(v, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(0.518, len(rows) - 0.4, "BioLangFusion, published\n(0.530 to 0.563)", fontsize=7,
            color=INK2, ha="right", va="top")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Spearman correlation on the published test split")
    panel_title(ax, "a   mRNA stability: frozen probes against BioLangFusion")
    handles = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, markersize=6, label="all 981 test rows"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, markersize=6, label="295 unseen sequences only")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5)
    tidy(ax)

    ax = axes[1]
    y = np.arange(len(ISO_ROWS))[::-1]
    for yi, (lab, (lo, hi), pub) in zip(y, ISO_ROWS):
        ax.plot([lo, hi], [yi, yi], color=INK2, lw=3.2, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter([(lo + hi) / 2 if lo != hi else lo], [yi], s=0, zorder=1)
        if lab in ISO_POINT:
            ax.scatter([ISO_POINT[lab]], [yi], s=44, color=INK2, zorder=4, edgecolor=SURFACE, lw=0.8)
        elif lo == hi:
            ax.scatter([lo], [yi], s=44, color=INK2, zorder=4, edgecolor=SURFACE, lw=0.8)
        if pub is not None:
            ax.scatter([pub], [yi], s=60, marker="D", color=PROT, zorder=5, edgecolor=SURFACE, lw=0.8)
            ax.annotate("", xy=(pub - 0.008, yi), xytext=(hi + 0.008, yi),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=8))
    ax.set_yticks(y, [r[0] for r in ISO_ROWS])
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("$R^2$ on the test transcripts, mean over 30 tissues")
    panel_title(ax, "b   GTEx: frozen probes against IsoFormer")
    handles = [Line2D([], [], color=INK2, lw=3.2, alpha=0.35, label="frozen probe, range over encoders in the role"),
               Line2D([], [], marker="o", ls="", color=INK2, markersize=6, label="frozen probe, IsoFormer's own encoder"),
               Line2D([], [], marker="D", ls="", color=PROT, markersize=6, label="IsoFormer, published (Table 2)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.2, ncols=1)
    tidy(ax)
    return fig


# ---------------------------------------------------------- variance bands
BANDS = {  # variance in top 10, composition share top, rest, stability R2 top, rest, whole
    "nt500m": (0.747, 0.742, 0.252, 0.039, -0.010, 0.032),
    "ntv2_100m": (0.891, 0.858, 0.356, 0.057, -0.010, 0.080),
    "dnabert2": (0.715, 0.764, 0.224, 0.039, -0.020, 0.054),
    "hyenadna": (0.963, 0.950, 0.236, 0.024, -0.009, 0.026),
    "rnafm": (0.858, 0.745, 0.265, 0.038, -0.008, 0.052),
    "rinalmo": (0.796, 0.720, 0.364, 0.047, -0.008, 0.078),
    "mrnafm": (0.606, 0.369, 0.182, 0.061, -0.009, 0.127),
    "calm": (0.548, 0.647, 0.259, 0.066, -0.008, 0.127),
    "esm2_8m": (0.625, 0.575, 0.319, 0.108, -0.010, 0.127),
    "esm2_35m": (0.648, 0.534, 0.285, 0.107, -0.009, 0.114),
    "esm2_150m": (0.698, 0.500, 0.292, 0.095, -0.008, 0.144),
    "protbert": (0.818, 0.453, 0.246, 0.086, -0.009, 0.143),
}


def fig_bands():
    order = list(reversed(D.KEYS))
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.20, wspace=0.14)

    ax = axes[0]
    vals = [BANDS[k][0] * 100 for k in order]
    ax.barh(y, vals, color=[MODC[D.MOD[k]] for k in order], height=0.62, alpha=0.9)
    ax.barh(y, [100 - v for v in vals], left=vals, color="#e8e6df", height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v - 1.5, yi, f"{v:.0f}%", ha="right", va="center", fontsize=7, color="white", fontweight="bold")
    ax.set_yticks(y, [D.SHORT[k] for k in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Variance held by the top ten components (%)")
    panel_title(ax, "a   Variance in the top ten components")
    tidy(ax)

    ax = axes[1]
    for yi, k in zip(y, order):
        top, rest = BANDS[k][1] * 100, BANDS[k][2] * 100
        c = MODC[D.MOD[k]]
        ax.plot([rest, top], [yi, yi], color=c, lw=2.6, alpha=0.35, zorder=1, solid_capstyle="round")
        ax.scatter([top], [yi], s=44, color=c, zorder=4, edgecolor=SURFACE, lw=0.8)
        ax.scatter([rest], [yi], s=44, facecolor=SURFACE, edgecolor=c, lw=1.4, zorder=3)
    ax.set_yticks(y, [""] * len(order))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Variance explained by composition (%)")
    ax.text(0.98, 0.02, "filled: top ten components\nopen: remaining components", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=INK2, linespacing=1.4)
    panel_title(ax, "b   Composition share: top ten vs rest")
    tidy(ax)

    ax = axes[2]
    for yi, k in zip(y, order):
        top, rest, whole = BANDS[k][3], BANDS[k][4], BANDS[k][5]
        c = MODC[D.MOD[k]]
        ax.plot([rest, whole], [yi, yi], color="#e6e4dd", lw=1.4, zorder=1)
        ax.scatter([rest], [yi], s=34, marker="x", color=MUTED, zorder=3, lw=1.2)
        ax.scatter([top], [yi], s=44, color=c, zorder=4, edgecolor=SURFACE, lw=0.8)
        ax.scatter([whole], [yi], s=44, facecolor=SURFACE, edgecolor=c, lw=1.4, zorder=5)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y, [""] * len(order))
    ax.set_xlim(-0.04, 0.17)
    ax.set_xlabel("Stability probe $R^2$ (pinned folds)")
    panel_title(ax, "c   Where the stability signal sits")
    handles = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, markersize=6, label="top ten components only"),
               Line2D([], [], marker="x", ls="", color=MUTED, markersize=6, mew=1.2, label="remaining components only"),
               Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, markersize=6, label="whole embedding")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), fontsize=7.5, ncols=3)
    tidy(ax)
    return fig


# ----------------------------------------------------------- attention tests
ATTN = {  # encoder: {motif: (fraction of sequences with p < 0.05, mean gap)}
    "NT 500M": {"All pooled": (0.07, -0.0009), "ARE pentamer": (0.05, -0.0027), "DRACH": (0.08, -0.0009)},
    "NT v2 100M": {"All pooled": (0.15, -0.0042), "ARE pentamer": (0.34, 0.0110), "DRACH": (0.14, -0.0051)},
    "RNA-FM": {"All pooled": (0.09, -0.0015), "ARE pentamer": (0.12, -0.0006), "DRACH": (0.08, -0.0016)},
    "RiNALMo": {"All pooled": (0.04, -0.0084), "ARE pentamer": (0.24, 0.0400), "DRACH": (0.02, -0.0099)},
    "mRNA-FM": {"All pooled": (0.93, 0.0014), "ARE pentamer": (0.10, -0.0011), "DRACH": (0.92, 0.0015)},
    "CaLM": {"All pooled": (0.15, -0.0072), "ARE pentamer": (0.37, 0.0298), "DRACH": (0.14, -0.0084)},
}
NULLS = [  # label, observed gap, position-null excess, p, codon-and-frame excess, p
    ("NT v2 100M, ARE pentamer", 0.0110, 0.0149, 0.014, 0.0061, 0.200),
    ("RiNALMo, ARE pentamer", 0.0400, 0.0424, 0.002, -0.0008, 0.585),
    ("mRNA-FM, all pooled", 0.0014, 0.0014, 0.002, 0.0001, 0.072),
    ("mRNA-FM, DRACH", 0.0015, 0.0014, 0.002, 0.0001, 0.082),
    ("CaLM, ARE pentamer", 0.0298, 0.0302, 0.002, 0.0140, 0.002),
]


def fig_attention():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.26, wspace=0.55)

    ax = axes[0]
    encs = list(ATTN)
    motifs = ["All pooled", "ARE pentamer", "DRACH"]
    y = np.arange(len(encs))[::-1]
    for j, mo in enumerate(motifs):
        for yi, e in zip(y, encs):
            frac, gap = ATTN[e][mo]
            colour = PROT if gap > 0 else "#9a9891"
            ax.scatter([j], [yi], s=frac * 900 + 12, color=colour, alpha=0.85, edgecolor=SURFACE, lw=0.8, zorder=3)
            ax.text(j, yi, f"{frac:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if frac > 0.3 else INK, zorder=4)
    ax.set_xticks(range(3), motifs)
    ax.set_yticks(y, encs)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.7, len(encs) - 0.3)
    ax.tick_params(length=0)
    panel_title(ax, "a   Share of 100 sequences with nominal p < 0.05")
    for side in ax.spines.values():
        side.set_visible(False)

    ax = axes[1]
    y = np.arange(len(NULLS))[::-1]
    for yi, (lab, obs, pos, ppos, cod, pcod) in zip(y, NULLS):
        ax.hlines(yi, min(obs, pos, cod, 0), max(obs, pos, cod), color="#e6e4dd", lw=1.4, zorder=1)
        ax.scatter([obs], [yi], s=46, facecolor=SURFACE, edgecolor=INK2, lw=1.3, zorder=4)
        ax.scatter([pos], [yi], s=44, marker="s", color=DNA, zorder=3, edgecolor=SURFACE, lw=0.8)
        ax.scatter([cod], [yi], s=52, marker="D", color=PROT if pcod < 0.05 else "#9a9891",
                   zorder=5, edgecolor=SURFACE, lw=0.8)
        ax.text(0.0635, yi, f"{ppos:.3f}      {pcod:.3f}", ha="right", va="center",
                fontsize=6.9, color=INK2)
    ax.text(0.0635, len(NULLS) - 0.25, "p, position   p, codon", ha="right", va="center",
            fontsize=6.6, color=MUTED, style="italic")
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y, [n[0] for n in NULLS])
    ax.set_xlim(-0.004, 0.064)
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
    ax.set_ylim(-0.7, len(NULLS) + 0.1)
    ax.set_xlabel("Attention gap, motif positions minus elsewhere (0 to 1 scale)")
    panel_title(ax, "b   Does position or codon context explain the gap?")
    tidy(ax)

    handles = [Line2D([], [], marker="o", ls="", color=PROT, markersize=7, label="motif positions get more attention"),
               Line2D([], [], marker="o", ls="", color="#9a9891", markersize=7, label="motif positions get less attention"),
               Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, markersize=6, label="observed gap"),
               Line2D([], [], marker="s", ls="", color=DNA, markersize=6, label="excess over the position null"),
               Line2D([], [], marker="D", ls="", color=PROT, markersize=6, label="excess over the codon-and-frame null (filled if p < 0.05)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=7.3, ncols=2)
    return fig


# ----------------------------------------------------- alignment vs error
def fig_align_error(stab, gtex):
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.26)
    rng = np.random.default_rng(3)
    kinds = [("within modality", MUTED, "o"), ("DNA with RNA or protein", DNA, "s"),
             ("nucleotide RNA with protein", RNA, "^"), ("codon RNA with protein", PROT, "D")]
    for base, data, n_test, label in ((0, stab, 291, "mRNA stability  (291 test sequences)"),
                                      (1, gtex, 319, "GTEx expression  (319 test transcripts)")):
        p = data["pairs"]
        nominal = 1.96 / np.sqrt(n_test - 2)
        bonf = 3.35 / np.sqrt(n_test - 2)
        ax.axvspan(base - 0.42, base + 0.42, ymin=0, ymax=1, color="#f2f1ed", zorder=0)
        for kind, colour, marker in kinds:
            sub = p[p["kind"] == kind]
            x = base + rng.uniform(-0.3, 0.3, len(sub))
            ax.scatter(x, sub["align_err_rho"], s=30, marker=marker, color=colour,
                       edgecolor=SURFACE, lw=0.7, zorder=3)
        for lvl, ls, txt in ((nominal, (0, (3, 2)), "nominal p = 0.05"), (bonf, "-", "Bonferroni, 66 tests")):
            ax.plot([base - 0.42, base + 0.42], [lvl, lvl], color=MUTED, lw=0.9, ls=ls, zorder=1)
            ax.plot([base - 0.42, base + 0.42], [-lvl, -lvl], color=MUTED, lw=0.9, ls=ls, zorder=1)
            ax.text(base - 0.41, lvl + 0.006, txt, fontsize=6.6, color=MUTED, va="bottom", ha="left")
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xticks([0, 1], ["mRNA stability  (291 test sequences)", "GTEx expression  (319 test transcripts)"])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(-0.26, 0.26)
    ax.set_ylabel("Spearman $\\rho$: per-sequence alignment vs prediction error")
    ax.tick_params(axis="x", length=0)
    panel_title(ax, "Per-sequence alignment does not track prediction error, for any of the 66 pairs")
    handles = [Line2D([], [], marker=m, ls="", color=c, markersize=6, label=k) for k, c, m in kinds]
    fig.legend(handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0.01), fontsize=7.5)
    tidy(ax, grid="y")
    return fig


# ------------------------------------------------------- codon grid schematic
def fig_codon_grid():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.03, wspace=0.10)

    # ---- panel a: same coding sequence, five token streams, one codon grid
    ax = axes[0]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    panel_title(ax, "a   mRNA stability: every encoder reads one coding sequence")
    seq = "AUGGCUAAGCCGUUAGAC"    # 18 nt = 6 codons shown
    ncod = 6
    x0, cw = 25, 8.8              # left edge, width of one codon
    rows = [
        ("Codon grid\n(fusion step t)", "grid", INK),
        ("ESM-2, ProtBERT\n1 token = amino acid", "aa", PROT),
        ("mRNA-FM, CaLM\n1 token = codon", "codon", RNA),
        ("RNA-FM, RiNALMo,\nHyenaDNA\n1 token = nucleotide", "nt", RNA),
        ("NT 500M, NT v2\n1 token = 6-mer", "sixmer", DNA),
        ("DNABERT-2\nbyte-pair tokens", "bpe", DNA),
    ]
    ys = [86, 71, 58, 45, 32, 19]
    h = 8.5
    aas = ["M", "A", "K", "P", "L", "D"]
    for (label, kind, colour), y in zip(rows, ys):
        ax.text(x0 - 2.0, y + h / 2, label, ha="right", va="center", fontsize=7.4,
                color=INK if kind == "grid" else INK2, linespacing=1.3,
                fontweight="bold" if kind == "grid" else "normal")
        if kind == "grid":
            for c in range(ncod):
                ax.add_patch(FancyBboxPatch((x0 + c * cw, y), cw - 0.6, h, boxstyle="round,pad=0,rounding_size=0.8",
                                            fc="#e8e6df", ec=INK, lw=1.0))
                ax.text(x0 + c * cw + (cw - 0.6) / 2, y + h / 2, f"t = {c + 1}", ha="center", va="center",
                        fontsize=7.2, color=INK, fontweight="bold")
        elif kind == "aa":
            for c in range(ncod):
                ax.add_patch(FancyBboxPatch((x0 + c * cw, y), cw - 0.6, h, boxstyle="round,pad=0,rounding_size=0.8",
                                            fc="white", ec=colour, lw=1.1))
                ax.text(x0 + c * cw + (cw - 0.6) / 2, y + h / 2, aas[c], ha="center", va="center",
                        fontsize=8, color=colour, fontweight="bold")
        elif kind == "codon":
            for c in range(ncod):
                ax.add_patch(FancyBboxPatch((x0 + c * cw, y), cw - 0.6, h, boxstyle="round,pad=0,rounding_size=0.8",
                                            fc="white", ec=colour, lw=1.1))
                ax.text(x0 + c * cw + (cw - 0.6) / 2, y + h / 2, seq[3 * c:3 * c + 3], ha="center", va="center",
                        fontsize=7.4, color=colour, fontweight="bold")
        elif kind == "nt":
            w = cw / 3
            for i in range(ncod * 3):
                ax.add_patch(FancyBboxPatch((x0 + i * w, y), w - 0.45, h, boxstyle="round,pad=0,rounding_size=0.6",
                                            fc="white", ec=colour, lw=0.9))
                ax.text(x0 + i * w + (w - 0.45) / 2, y + h / 2, seq[i], ha="center", va="center",
                        fontsize=7, color=colour)
        elif kind == "sixmer":
            w = cw * 2
            for i in range(ncod // 2):
                ax.add_patch(FancyBboxPatch((x0 + i * w, y), w - 0.6, h, boxstyle="round,pad=0,rounding_size=0.8",
                                            fc="white", ec=colour, lw=1.1))
                ax.text(x0 + i * w + (w - 0.6) / 2, y + h / 2, seq[6 * i:6 * i + 6], ha="center", va="center",
                        fontsize=7.4, color=colour, fontweight="bold")
        elif kind == "bpe":
            cuts = [0, 4, 7, 12, 18]   # illustrative byte-pair boundaries in nucleotides
            w = cw / 3
            for a, b in zip(cuts[:-1], cuts[1:]):
                ax.add_patch(FancyBboxPatch((x0 + a * w, y), (b - a) * w - 0.5, h,
                                            boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=colour, lw=1.1))
                ax.text(x0 + (a + b) / 2 * w, y + h / 2, seq[a:b], ha="center", va="center",
                        fontsize=7.2, color=colour, fontweight="bold")
        ax.text(x0 + ncod * cw + 1.0, y + h / 2, "...", ha="left", va="center", fontsize=9, color=MUTED)

    notes = [("already one\nper codon", 71), ("already one\nper codon", 58),
             ("average the three\nvectors of each codon", 45),
             ("split each token\ninto its two codons", 32),
             ("assign each token to\nthe codons it covers", 19)]
    for txt, y in notes:
        ax.text(x0 + ncod * cw + 5.0, y + h / 2, txt, ha="left", va="center", fontsize=6.9,
                color=MUTED, style="italic", linespacing=1.3)
    for y in ys[1:]:
        ax.add_patch(FancyArrowPatch((x0 - 0.8, y + h + 1.0), (x0 - 0.8, ys[0] - 0.8),
                                     arrowstyle="-", color="#d6d4cc", lw=0.6, zorder=0))
    ax.add_patch(FancyArrowPatch((x0 + ncod * cw / 2, ys[1] + h + 0.8), (x0 + ncod * cw / 2, ys[0] - 0.8),
                                 arrowstyle="-|>", mutation_scale=10, color=INK, lw=1.0))
    ax.text(x0 + ncod * cw / 2 + 1.5, (ys[1] + h + ys[0]) / 2, "resample every stream to the codon grid",
            ha="left", va="center", fontsize=7.0, color=INK2)
    ax.text(2, 6, "Each codon fixes one amino acid, so the grid is given by the genetic code: no model has to learn it.\n"
                  "Encoders that truncate at 1,022 nt cover only the start of the grid.",
            ha="left", va="center", fontsize=6.9, color=MUTED, linespacing=1.4)

    # ---- panel b: GTEx, three different molecules, no shared grid
    ax = axes[1]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    panel_title(ax, "b   GTEx: each modality is a different molecule")
    # DNA window
    ax.add_patch(FancyBboxPatch((8, 74), 84, 9, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=DNA, lw=1.3))
    ax.add_patch(Rectangle((50, 74), 0.9, 9, fc=DNA, ec="none"))
    ax.text(29, 78.5, "3,000 nt upstream", ha="center", va="center", fontsize=7.2, color=DNA)
    ax.text(71, 78.5, "2,999 nt downstream", ha="center", va="center", fontsize=7.2, color=DNA)
    ax.text(50.5, 85.5, "transcription start", ha="center", va="bottom", fontsize=6.6, color=MUTED)
    ax.text(8, 70.5, "DNA window, 6,000 nt   (NT 500M, NT v2, DNABERT-2, HyenaDNA)", ha="left", va="top", fontsize=7.2, color=INK2)
    # transcript
    ax.add_patch(FancyBboxPatch((50.5, 52), 10, 9, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=RNA, lw=1.3))
    ax.add_patch(FancyBboxPatch((60.5, 52), 25, 9, boxstyle="round,pad=0,rounding_size=1", fc="#dff3ea", ec=RNA, lw=1.3))
    ax.add_patch(FancyBboxPatch((85.5, 52), 13, 9, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=RNA, lw=1.3))
    ax.text(55.5, 56.5, "5' UTR", ha="center", va="center", fontsize=6.8, color=RNA)
    ax.text(73, 56.5, "coding sequence", ha="center", va="center", fontsize=7, color=RNA, fontweight="bold")
    ax.text(92, 56.5, "3' UTR", ha="center", va="center", fontsize=6.8, color=RNA)
    ax.text(8, 48.5, "transcript, median 1,738 nt   (RNA-FM and RiNALMo read it;\nmRNA-FM and CaLM read only the coding sequence)",
            ha="left", va="top", fontsize=7.2, color=INK2, linespacing=1.3)
    ax.add_patch(FancyArrowPatch((50.5, 74), (50.5, 61.5), arrowstyle="-", color="#d6d4cc", lw=0.8, ls=(0, (3, 2))))
    # protein
    ax.add_patch(FancyBboxPatch((60.5, 30), 25, 9, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=PROT, lw=1.3))
    ax.text(73, 34.5, "protein, median 251 aa", ha="center", va="center", fontsize=7, color=PROT, fontweight="bold")
    ax.text(8, 26.5, "protein   (ESM-2 8M, 35M, 150M, ProtBERT)", ha="left", va="top", fontsize=7.2, color=INK2)
    ax.add_patch(FancyArrowPatch((73, 52), (73, 39.5), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=0.8))
    ax.text(74.5, 41.5, "translation", ha="left", va="center", fontsize=6.6, color=MUTED)
    ax.text(8, 8, "No position-by-position correspondence links the window to the transcript\n"
                  "or the protein, so there is no shared grid: fusion on GTEx uses cross-attention\n"
                  "or pooled vectors.",
            ha="left", va="center", fontsize=6.9, color=MUTED, linespacing=1.4)
    return fig


# --------------------------------------------- fusion architecture schematic (Stage 3 designs)
def fig_architectures():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.06)

    def box(ax, x, y, w, h, text, fc="white", ec=AXIS, lw=1.0, fs=7.2, colour=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=colour,
                fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)

    def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=0.9, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=c, lw=lw,
                                     shrinkA=1, shrinkB=1, zorder=1))

    for ax in axes:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- a: concatenation + MLP
    ax = axes[0]
    panel_title(ax, "a   Concatenation + MLP  (reference baseline B4)")
    for x, lab, c in ((10, "RNA encoder\n(frozen)", RNA), (55, "protein encoder\n(frozen)", PROT)):
        box(ax, x, 74, 35, 13, lab, ec=c, colour=c, bold=True)
        box(ax, x, 56, 35, 10, "mean-pooled vector", fs=6.8, colour=INK2)
        arrow(ax, x + 17.5, 74, x + 17.5, 66)
    box(ax, 20, 36, 60, 10, "concatenate  [ z_RNA ; z_protein ]", fs=7.0)
    arrow(ax, 27.5, 56, 40, 46); arrow(ax, 72.5, 56, 60, 46)
    box(ax, 30, 18, 40, 10, "2-layer MLP head", fs=7.0)
    arrow(ax, 50, 36, 50, 28)
    box(ax, 35, 3, 30, 9, "prediction", fc="#e8e6df", ec=INK, fs=7.0, bold=True)
    arrow(ax, 50, 18, 50, 12)
    ax.text(50, 93, "no token interaction; sets the bar every fusion model must clear",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")

    # ---- b: Coupled Mamba on the codon grid
    ax = axes[1]
    panel_title(ax, "b   Coupled Mamba on the codon grid  (priority 1)")
    ax.text(50, 93, "one recurrent chain per modality, coupled at every codon step t",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
    chains = [("DNA", DNA, 78), ("RNA", RNA, 58), ("protein", PROT, 38)]
    steps = [18, 40, 62, 84]
    for name, c, y in chains:
        ax.text(3, y, name, ha="left", va="center", fontsize=7.4, color=c, fontweight="bold")
        for i, x in enumerate(steps):
            box(ax, x - 6, y - 4.5, 12, 9, f"h$_{{{i + 1}}}$", ec=c, colour=c, fs=7.0)
            if i < len(steps) - 1:
                arrow(ax, x + 6, y, steps[i + 1] - 6, y, c=c, lw=1.1)
    for i in range(len(steps) - 1):
        xa, xb = steps[i] + 6, steps[i + 1] - 6
        for (_, ca, ya) in chains:
            for (_, cb, yb) in chains:
                if ya != yb:
                    arrow(ax, xa, ya, xb, yb, c="#c9c7bf", lw=0.6, style="-|>")
    for x in steps:
        ax.text(x, 24, f"codon {steps.index(x) + 1}", ha="center", va="center", fontsize=6.4, color=MUTED)
    ax.text(50, 14, "h$^m_t$ = S$_m$ (sum over modalities of h$_{t-1}$) + B$_m$ x$^m_t$",
            ha="center", va="center", fontsize=7.2, color=INK2)
    ax.text(50, 6, "linear in sequence length; no alignment loss;\nneeds the shared codon grid",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic", linespacing=1.4)

    # ---- c: cross-attention
    ax = axes[2]
    panel_title(ax, "c   Cross-attention, IsoFormer style  (priority 2)")
    ax.text(50, 93, "token streams keep their own lengths; one stream queries the others",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic")
    box(ax, 1, 70, 31, 12, "DNA window\ntokens (keys, values)", ec=DNA, colour=DNA, fs=6.8)
    box(ax, 34.5, 70, 31, 12, "transcript\ntokens (queries)", ec=RNA, colour=RNA, fs=6.8, bold=True)
    box(ax, 68, 70, 31, 12, "protein\ntokens (keys, values)", ec=PROT, colour=PROT, fs=6.8)
    box(ax, 22, 46, 56, 11, "multi-head cross-attention\nsoftmax(Q K$^T$ / sqrt(d)) V", fs=6.8)
    arrow(ax, 16.5, 70, 36, 57); arrow(ax, 50, 70, 50, 57); arrow(ax, 83.5, 70, 64, 57)
    box(ax, 30, 27, 40, 10, "pool + prediction head\n(30 tissues)", fs=6.8)
    arrow(ax, 50, 46, 50, 37)
    box(ax, 35, 10, 30, 9, "expression", fc="#e8e6df", ec=INK, fs=7.0, bold=True)
    arrow(ax, 50, 27, 50, 19)
    ax.text(50, 2.5, "quadratic in length; no shared grid needed;\nalso the benchmark competitor on stability",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic", linespacing=1.4)
    return fig


# --------------------------------------------- BioLangFusion reproduction pipeline (add-on baseline)
def fig_blf_pipeline():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.0), gridspec_kw={"width_ratios": [1.45, 1.0, 0.75]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)

    def box(ax, x, y, w, h, text, fc="white", ec=AXIS, lw=1.0, fs=7.2, colour=INK, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec,
                                    lw=lw, ls=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=colour,
                fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)

    def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=0.9, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=c, lw=lw,
                                     shrinkA=1, shrinkB=1, zorder=1))

    for ax in axes:
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax = axes[0]
    panel_title(ax, "a   BioLangFusion, reproduced then encoders swapped")
    cols = ((2, "DNA", DNA, "NT v2 100M\n6-mer tokens", "swap: NT 500M,\nDNABERT-2, HyenaDNA", "TConv k=2, s=2\n(2 codons per token)"),
            (35, "mRNA", RNA, "RNA-FM\nnucleotide tokens", "swap: CaLM,\nmRNA-FM, RiNALMo", "AvgPool k=3, s=3\n(3 tokens per codon)"),
            (68, "protein", PROT, "ESM-2 8M\namino-acid tokens", "swap: ESM-2 150M,\nProtBERT", "identity\n(1 residue per codon)"))
    for x, name, c, orig, swap, resamp in cols:
        box(ax, x, 80, 30, 10, orig, ec=c, colour=c, bold=True, fs=6.6)
        box(ax, x, 68, 30, 8.5, swap, ec=c, colour=c, fs=6.0, ls=(0, (3, 2)))
        box(ax, x, 52, 30, 10, resamp, fs=6.2, colour=INK2)
        arrow(ax, x + 15, 80, x + 15, 62)
    ax.text(50, 47, "codon grid, t = 1 ... L/3  (positional alignment: a token-location step)",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")
    heads = ((2, "codon-aware\nconcatenation"), (35, "gated MIL attention\n+ entropy regularizer"), (68, "cross-modal\nmulti-head attention"))
    for x, lab in heads:
        box(ax, x, 30, 30, 11, lab, fs=6.6)
        arrow(ax, x + 15, 52, x + 15, 41)
    box(ax, 25, 15, 50, 9, "TextCNN head (fixed)  +  task layer", fs=6.8)
    for x, _ in heads:
        arrow(ax, x + 15, 30, 50, 24)
    box(ax, 35, 2, 30, 8.5, "prediction", fc="#e8e6df", ec=INK, fs=7.0, bold=True)
    arrow(ax, 50, 15, 50, 10.5)
    ax.text(50, 94, "Fungal, E. coli, mRNA stability, Ab1, CoV-Vac; published splits, recipe, metrics",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")

    ax = axes[1]
    panel_title(ax, "b   No grid (GTEx, UTRs)")
    ax.text(50, 94, "pooled or region-aware fusion; UTRs handled apart from the CDS",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")
    box(ax, 2, 76, 29, 12, "DNA window\n(frozen)", ec=DNA, colour=DNA, bold=True, fs=6.6)
    box(ax, 35.5, 76, 29, 12, "transcript\n(frozen)", ec=RNA, colour=RNA, bold=True, fs=6.6)
    box(ax, 69, 76, 29, 12, "protein\n(frozen)", ec=PROT, colour=PROT, bold=True, fs=6.6)
    box(ax, 2, 58, 29, 10, "pooled vector", fs=6.4, colour=INK2)
    box(ax, 35.5, 58, 29, 10, "region pools:\n5' UTR, CDS, 3' UTR", fs=6.0, colour=INK2)
    box(ax, 69, 58, 29, 10, "pooled vector", fs=6.4, colour=INK2)
    for x in (16.5, 50, 83.5):
        arrow(ax, x, 76, x, 68)
    box(ax, 15, 38, 70, 11, "concatenate, or gate the regions,\nthen MLP head (30 tissues on GTEx)", fs=6.6)
    arrow(ax, 16.5, 58, 35, 49); arrow(ax, 50, 58, 50, 49); arrow(ax, 83.5, 58, 65, 49)
    box(ax, 35, 22, 30, 8.5, "prediction", fc="#e8e6df", ec=INK, fs=7.0, bold=True)
    arrow(ax, 50, 38, 50, 30.5)
    box(ax, 8, 4, 84, 11, "always beside it: length and composition baseline,\nbest single encoder, same-modality ensemble", fs=6.3, colour=INK2, ls=(0, (3, 2)))

    ax = axes[2]
    panel_title(ax, "c   Compared with")
    box(ax, 6, 70, 88, 20, "LucaOne (add-on baseline)\none model, shared 39-token vocabulary\nfor nucleic acid and protein;\nembeds each input separately,\n1,280-token limit", fs=6.3, ec=INK, colour=INK)
    ax.text(50, 60, "jointly pretrained alternative to\nfusing separately pretrained encoders",
            ha="center", va="center", fontsize=6.3, color=MUTED, style="italic", linespacing=1.3)
    box(ax, 6, 30, 88, 20, "the Stage 3 designs (Figure 17)\nCoupled Mamba on the codon grid\n(priority 1, Gantt 20 to 21);\nIsoFormer-style cross-attention\n(priority 2, Gantt 25)", fs=6.3, ec=INK, colour=INK)
    ax.text(50, 20, "the reproduced heads are the matched\nbaseline these are scored against",
            ha="center", va="center", fontsize=6.3, color=MUTED, style="italic", linespacing=1.3)
    return fig


# --------------------------------------------- project logic: the chain of stages
def fig_logic():
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def card(x, y, w, h, head, ask, body, forces, ec=INK, dash=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc="white", ec=ec, lw=1.0,
                                    ls=(0, (3, 2)) if dash else "-", zorder=2))
        ax.text(x + 1.6, y + h - 2.4, head, ha="left", va="top", fontsize=7.6, color=INK, fontweight="bold", zorder=3)
        ax.text(x + 1.6, y + h - 8.2, ask, ha="left", va="top", fontsize=6.4, color=INK2, style="italic", linespacing=1.3, zorder=3)
        ax.text(x + 1.6, y + h - 17.5, body, ha="left", va="top", fontsize=6.2, color=INK2, linespacing=1.32, zorder=3)
        ax.text(x + 1.6, y + 2.2, forces, ha="left", va="bottom", fontsize=6.2, color=INK, fontweight="bold", linespacing=1.3, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1,
                                     shrinkA=1, shrinkB=1, zorder=1))

    ax.text(50, 98, "Goal: decide whether, when, and how to fuse frozen DNA, RNA, and protein encoders for a biological task, and leave a method that predicts it before training.\n"
                    "Fusion has to beat strong single-encoder and simple-fusion baselines; different representations are not by themselves independent information.",
            ha="center", va="top", fontsize=6.8, color=INK, linespacing=1.4)

    top = 10; h = 76; w = 23.2; gap = 1.9
    xs = [1.0 + i * (w + gap) for i in range(4)]
    card(xs[0], top, w, h, "Stage 1  Representation analysis, done",
         "What does each frozen encoder carry,\nand how do their spaces relate?",
         "12 encoders, 2 datasets, 3 tracks.\n1. Protein and codon-level encoders carry the\n   stability signal; nucleotide encoders carry\n   GC3, which translation erases.\n2. Most raw agreement is composition; only\n   within-family and codon-with-protein pairs\n   stay aligned after the control.\n3. Alignment never predicts error; gains\n   appear where alignment is low.\n4. Four length features match all twelve\n   encoders on GTEx.\nRedundant families found: ESM-2; RNA-FM\nwith RiNALMo.",
         "Forces: a biology-specific definition of\nalignment; a baseline hierarchy; a pruned\nset of pairs to fuse first.")
    card(xs[1], top, w, h, "Stage 2  Define alignment (decided)",
         "What should alignment mean here,\nand should the fusion models enforce it?",
         "Alignment = agreement on what the central\ndogma shares, after composition is removed.\nPositional alignment (the codon grid) only\non coding sequence; a token-location step.\nNo alignment loss by default.\nAlignment is measured and reported beside\nevery result; fusion benefit is a separate\nheld-out gain, Δfusion.\nLiterature: alignment theory (redundancy vs\nuniqueness); MSAlign (alignment as retrieval);\nsequence-structure alignment (SA-PLM,\nOmniGenome) is a different sense of the word.",
         "Forces: architectures that let modalities\ninteract without forcing similarity.")
    card(xs[2], top, w, h, "Stage 3  Fusion design",
         "How do we combine the encoders,\ngiven the Stage 2 decision?",
         "Floor: concatenation + MLP.\nPriority 1: Coupled Mamba on the codon grid\n(coding sequence; interaction through the\nrecurrence, nothing forces similarity).\nPriority 2: IsoFormer-style cross-attention\n(GTEx; learns correspondence, no grid).\nAdd-on baselines: BioLangFusion reproduced\nexactly (the published system on identical\nsplits; its attention head is the simple\nfusion baseline of Fusion or Confusion);\nLucaOne (joint pretraining instead of fusion).",
         "Forces: every design is scored against\nthe reproduced heads on identical splits.")
    card(xs[3], top, w, h, "Stage 4  Experiments, benchmarking",
         "Does fusion help, for which task, and\ncould Stage 1 have told us in advance?",
         "In order: baselines and controls (mRFP);\nBioLangFusion reproduction; Coupled Mamba\nvs the reproduced heads; ablations (encoder\nswaps, alignment loss, grid); the task set,\neach dataset isolating one question; the\nmethod test: do Stage 1 metrics predict\nΔfusion?\n\"Which RNA with which DNA for which task\" =\nencoder swaps under a fixed architecture\nacross that task set. COMET adds multi-\nmolecule tasks with published combinations;\nmRNABench adds homology-aware splits.",
         "Delivers: a tested decision method, or\nevidence that frozen geometry does not\ntransfer.")
    for i in range(3):
        arrow(xs[i] + w, top + h / 2, xs[i + 1], top + h / 2)
    ax.text(50, 3.5, "Evaluation rules from Fusion or Confusion on every comparison: identical heads, tuned baselines, grouped cross-validation, mean ± SD over seeds, a statistical comparison.",
            ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")
    return fig


# --------------------------------------------- research plan: priorities and gates
def fig_plan():
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def card(x, y, w, h, head, body, ec=INK, fc="white", ls="-", headc=INK, fs=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec=ec, lw=1.0,
                                    ls=ls, zorder=2))
        ax.text(x + 1.6, y + h - 2.4, head, ha="left", va="top", fontsize=7.4, color=headc, fontweight="bold", zorder=3)
        ax.text(x + 1.6, y + h - 10.5, body, ha="left", va="top", fontsize=fs, color=INK2, linespacing=1.32, zorder=3)

    def arrow(x1, y1, x2, y2, c=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=c, lw=1.0,
                                     shrinkA=1, shrinkB=1, zorder=1))

    top = 58; h = 40; w = 22.4; gap = 2.0
    xs = [1.0 + i * (w + gap) for i in range(4)]
    card(xs[0], top, w, h, "Add-on 1  Reproduce BioLangFusion",
         "Phase A. Its encoders (NT v2 100M,\nRNA-FM, ESM-2 8M), codon grid, three\nheads, TextCNN; published splits and\nrecipe. Five benchmarks: Fungal,\nE. coli, mRNA stability, Ab1, CoV-Vac.\nGate: match the published scores or\nrecord each deviation.\nSlots into Tier 2, Gantt 17.")
    card(xs[1], top, w, h, "Add-on 2  Change encoders only",
         "Phase B. Same splits, grid, heads, head\nand evaluation; one encoder swapped at\na time (CaLM, mRNA-FM, ProtBERT, other\nDNA encoders), then triples.\nReport: published, reproduced, best\nsingle, each head, each swap, gain.\nStage 1 says which swaps should gain.\nSlots into Tier 3, Gantt 24 (now planned).")
    card(xs[2], top, w, h, "Add-on 3  LucaOne, jointly pretrained",
         "One model for nucleic acid and protein,\nfrozen; same folds and head on stability\nand GTEx. Verify input limits (1,280\ntokens) and sequence types first.\nCompare against the best single encoder\nand the best fusion; report cost and\ntruncation.\nSlots into Tier 2, beside Gantt 16 and 17.")
    card(xs[3], top, w, h, "Add-on 4  Full-transcript tasks",
         "Half-life (Saluki human, mouse; LBKWK)\nand translation efficiency (TE human,\nmouse; MRL Sugimoto). Gates: UTRs and\nCDS coordinates present, labels distinct,\ngrouped non-leaking splits. Compare\nCDS-only, CDS + 3' UTR, full transcript,\n+ protein; gain over the best full-\ntranscript baseline. Tier 4, Gantt 38.")
    for i in range(3):
        arrow(xs[i] + w, top + h / 2, xs[i + 1], top + h / 2)

    bot = 16; hb = 37
    card(1.0, bot, 30.5, hb, "Add-on 5, parallel  mRFP control",
         "Cached embeddings and labels in hand. Protein\nis identical by construction, so protein encoders\nare an invariant negative control; nucleotide\nencoders against GC3, codon-use and structure\nproxies. Tests representation specificity, not\nfusion. Tier 1, Gantt 18.", ec=AXIS)
    card(33.5, bot, 30.5, hb, "Add-on 6, later  Structure features",
         "Only on a task where structure plausibly acts.\nProtein: sequence vs + ESMFold vs + AlphaFold2\nfeatures. RNA: sequence vs + predicted pairing\n(RNAfold thermodynamic vs EternaFold, MXfold2).\nReport gains over sequence-only and over simple\nstructure features. Conditional, Gantt 38.", ec=AXIS)
    card(66.0, bot, 33.0, hb, "Add-on 7, exploratory  Multi-teacher distillation",
         "Distill the DNA, RNA and protein teachers into one\ncompact student per modality with a relational loss\n(match each teacher's cosine-similarity matrix through\nits own projection), then fuse the students. A new\nhypothesis; not implemented before the core work.\nStage 1 bears on it: much teacher similarity is\ncomposition.", ec=AXIS, ls=(0, (3, 2)))
    ax.text(50, 6.5, "Add-ons to the Stage 3 and 4 design and to the Gantt chart (as of October 3; dates subject to change). Coupled Mamba stays the priority architecture (Gantt 20 to 21).\nEvery result reports Δfusion = Perf(fused) minus the best single encoder, confidence intervals, and the baselines B0 to B5.",
            ha="center", va="center", fontsize=6.8, color=MUTED, style="italic", linespacing=1.4)
    return fig


# ------------------------------------------------------------------ main
def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    stab, gtex = D.load()
    figs = [
        ("fig1_project_schematic", paper_fig1()),
        ("fig2_composition_share", paper_fig2(stab, gtex)),
        ("fig3_composition_control", paper_fig3(stab)),
        ("fig4_layerwise_agreement", paper_fig4()),
        ("fig5_gain_vs_alignment", paper_fig5(stab, gtex)),
        ("fig6_probe_results", paper_fig6()),
        ("agreement_matrices_stability", fig_agreement(stab, "stability", 0.0034)),
        ("agreement_matrices_gtex", fig_agreement(gtex, "GTEx", 0.0031)),
        ("gain_matrices", fig_gains(stab, gtex)),
        ("track_a_controls", fig_controls()),
        ("published_comparison", fig_published()),
        ("variance_bands", fig_bands()),
        ("attention_tests", fig_attention()),
        ("alignment_vs_error", fig_align_error(stab, gtex)),
        ("codon_grid", fig_codon_grid()),
        ("fusion_architectures", fig_architectures()),
        ("blf_pipeline", fig_blf_pipeline()),
        ("research_plan", fig_plan()),
        ("project_logic", fig_logic()),
    ]
    for name, fig in figs:
        fig.savefig(os.path.join(outdir, f"{name}.pdf"))
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=200)
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
