"""Stage 3 and Stage 4 round-1 figures for the Overleaf manuscripts.

Every value is copied from the result tables in the manuscript (which were
checked against the saved analyses/*.json files), so the figures and the
tables cannot disagree.

    python make_r34_figures.py <output_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

# ------------------------------------------------------------------ style
# identical to docs/notes/figures/make_figures.py
DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
MODC = {"DNA": DNA, "RNA": RNA, "protein": PROT}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FUSED = "#0b0b0b"
PUB = "#b3b1a8"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 7.5, "legend.labelcolor": INK2, "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
})

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)


def tidy(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid == "both":
        ax.grid(axis="both", linestyle="-")
    elif grid:
        ax.grid(axis=grid, linestyle="-")
        ax.grid(axis="y" if grid == "x" else "x", visible=False)
    else:
        ax.grid(visible=False)


def panel(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(name)


def dumbbell(ax, labels, a, b, color_a=INK, color_b=PUB, size=34, lw=1.4):
    y = np.arange(len(labels))[::-1]
    for yi, xa, xb in zip(y, a, b):
        if xb is not None and not np.isnan(xb):
            ax.plot([xb, xa], [yi, yi], color=AXIS, lw=lw, zorder=1)
            ax.scatter([xb], [yi], s=size, facecolor=SURFACE, edgecolor=color_b, lw=1.4, zorder=2)
        ax.scatter([xa], [yi], s=size, color=color_a, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    return y


def forest(ax, rows, resolved_fill=True, markers=None):
    """rows: list of (label, est, lo, hi, kind)."""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, est, lo, hi, kind) in zip(y, rows):
        res = lo > 0 or hi < 0
        m = (markers or {}).get(kind, "o")
        ax.plot([lo, hi], [yi, yi], color=INK if res else MUTED, lw=1.3, zorder=2)
        ax.scatter([est], [yi], s=30, marker=m, facecolor=INK if res else SURFACE,
                   edgecolor=INK if res else MUTED, lw=1.2, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    return y


# ============================================================ figure A
# Phase A reproduction, ours against published (Table tab:blf)
def fig_reproduction():
    models = ["Nucleotide Transformer alone", "RNA-FM alone", "RNA-FM on the codon grid",
              "ESM-2 8M alone", "Concatenation", "Gated MIL attention", "Cross-attention"]
    ours = {
        "Fungal expression":   [0.807, 0.722, 0.794, 0.706, 0.791, 0.801, 0.823],
        "CoV-Vac":             [0.779, 0.838, 0.823, 0.815, 0.799, 0.783, 0.836],
        "E. coli (accuracy, %)": [47.8, 50.7, 49.3, 51.7, 58.7, 54.5, 53.5],
        "mRNA stability":      [0.539, 0.552, 0.547, 0.531, 0.558, 0.551, 0.553],
    }
    pub = {
        "Fungal expression":   [0.804, 0.767, np.nan, 0.695, 0.805, 0.824, 0.812],
        "CoV-Vac":             [0.780, 0.841, np.nan, 0.806, 0.831, 0.864, 0.828],
        "E. coli (accuracy, %)": [41.3, 52.9, np.nan, 49.0, 50.3, 52.1, 53.9],
        "mRNA stability":      [0.530, 0.553, np.nan, 0.539, 0.539, 0.563, 0.550],
    }
    lims = {"Fungal expression": (0.68, 0.84), "CoV-Vac": (0.76, 0.875),
            "E. coli (accuracy, %)": (38, 62), "mRNA stability": (0.52, 0.575)}
    ticks = {"Fungal expression": [0.70, 0.75, 0.80], "CoV-Vac": [0.78, 0.80, 0.82, 0.84, 0.86],
             "E. coli (accuracy, %)": [40, 45, 50, 55, 60], "mRNA stability": [0.53, 0.54, 0.55, 0.56, 0.57]}
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.7), sharey=True,
                             gridspec_kw={"wspace": 0.14, "left": 0.17, "right": 0.99, "top": 0.86, "bottom": 0.24})
    for k, (ax, (name, o)) in enumerate(zip(axes, ours.items())):
        y = dumbbell(ax, models, o, pub[name])
        ax.set_xlim(*lims[name])
        ax.set_xticks(ticks[name])
        tidy(ax, "x")
        panel(ax, "abcd"[k], name)
        # shade the fusion heads
        ax.axhspan(-0.5, 2.5, color=GRID, alpha=0.45, zorder=0, lw=0)
        if k == 0:
            ax.text(lims[name][0] + 0.003, -0.42, "fusion heads", fontsize=6.4, color=MUTED, va="bottom")
    axes[0].set_ylim(-0.6, len(models) - 0.4)
    for k in (0, 1, 3):
        axes[k].set_xlabel("Spearman, published test split", labelpad=6)
    axes[2].set_xlabel("accuracy, %, published test split", labelpad=6)
    handles = [Line2D([], [], marker="o", color=INK, ls="", ms=5.5, label="ours (seed means where three seeds exist)"),
               Line2D([], [], marker="o", mfc=SURFACE, mec=PUB, mew=1.4, color=PUB, ls="", ms=5.5, label="published, BioLangFusion Tables 1 and 4")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.58, 0.0), handletextpad=0.3, columnspacing=2.0)
    save(fig, "r34_reproduction")


# ============================================================ figure B
# Fusion against the best single encoder (Table tab:boot, plus the two extras)
def fig_fusion_forest():
    mk = {"concat": "o", "gated": "s", "xattn": "^"}
    spear = [
        ("Fungal: concatenation", -0.015, -0.049, 0.017, "concat"),
        ("Fungal: gated MIL", -0.005, -0.041, 0.031, "gated"),
        ("Fungal: cross-attention", 0.016, -0.014, 0.048, "xattn"),
        ("CoV-Vac: concatenation", -0.038, -0.066, -0.012, "concat"),
        ("CoV-Vac: gated MIL", -0.055, -0.093, -0.019, "gated"),
        ("CoV-Vac: cross-attention", -0.002, -0.026, 0.020, "xattn"),
        ("Stability, published split: concatenation", 0.007, -0.006, 0.019, "concat"),
        ("Stability, published split: gated MIL", -0.001, -0.014, 0.012, "gated"),
        ("Stability, published split: cross-attention", 0.001, -0.012, 0.015, "xattn"),
        ("Stability, unseen sequences: concatenation", 0.040, -0.003, 0.083, "concat"),
        ("GTEx pilot: IsoFormer's aggregator, all three minus RNA", 0.047, 0.027, 0.069, "xattn"),
    ]
    ecoli = [
        ("Concatenation", 6.9, 4.2, 9.6, "concat"),
        ("Gated MIL", 2.8, 0.9, 4.8, "gated"),
        ("Cross-attention", 1.8, -0.9, 4.5, "xattn"),
    ]
    fig = plt.figure(figsize=(11.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.55, left=0.30, right=0.98, top=0.87, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    forest(ax, spear, markers=mk)
    ax.set_xlim(-0.11, 0.11)
    tidy(ax, "x")
    panel(ax, "a", "Spearman benchmarks: head minus the best single encoder")
    ax.set_xlabel("difference in Spearman, 95% paired bootstrap interval")
    # group separators
    for yy in (7.5, 4.5, 1.5, 0.5):
        ax.axhline(yy, color=AXIS, lw=0.6, zorder=0)
    sub = gs[1].subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.1)
    ax2 = fig.add_subplot(sub[0])
    forest(ax2, ecoli, markers=mk)
    ax2.set_xlim(-3, 12)
    ax2.set_ylim(-0.7, 2.7)
    tidy(ax2, "x")
    panel(ax2, "b", "E. coli: head minus ESM-2 alone")
    ax2.set_xlabel("difference in accuracy, points")
    ax2.text(11.8, 1.5, "a better RNA encoder\nalone matches it", fontsize=6.4, color=MUTED, ha="right", va="center", style="italic")
    handles = [Line2D([], [], marker="o", color=INK, ls="", ms=5, label="resolved (interval excludes zero)"),
               Line2D([], [], marker="o", mfc=SURFACE, mec=MUTED, color=MUTED, ls="", ms=5, label="not resolved"),
               Line2D([], [], marker="o", mfc=SURFACE, mec=INK2, color=INK2, ls="", ms=5, label="concatenation"),
               Line2D([], [], marker="s", mfc=SURFACE, mec=INK2, color=INK2, ls="", ms=5, label="gated MIL"),
               Line2D([], [], marker="^", mfc=SURFACE, mec=INK2, color=INK2, ls="", ms=5, label="cross-attention")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.36), ncol=5, handletextpad=0.3, columnspacing=1.3)
    save(fig, "r34_fusion_forest")


# ============================================================ figure C
# Gated head weights (Table tab:alpha) and the stability leak (Table tab:leak)
def fig_gated_leak():
    fig = plt.figure(figsize=(11.6, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.75, left=0.12, right=0.98, top=0.86, bottom=0.2)
    # (a) weights at entropy 0.01, seeds averaged where there are three
    ax = fig.add_subplot(gs[0])
    bench = ["Fungal", "CoV-Vac", "E. coli", "Stability"]
    w = np.array([
        [0.20, 0.20, 0.60],
        [np.mean([0.83, 0.92, 0.94]), np.mean([0.01, 0.03, 0.00]), np.mean([0.16, 0.05, 0.06])],
        [np.mean([0.11, 0.18, 0.10]), 0.00, np.mean([0.89, 0.82, 0.90])],
        [0.28, 0.00, 0.71],
    ])
    w = w / w.sum(axis=1, keepdims=True)
    best = ["NT", "RNA-FM", "ESM-2", "RNA-FM"]
    y = np.arange(len(bench))[::-1]
    left = np.zeros(len(bench))
    for j, (mod, col) in enumerate(zip(["DNA", "RNA", "protein"], [DNA, RNA, PROT])):
        ax.barh(y, w[:, j], left=left, color=col, height=0.62, label=mod, lw=0)
        for yi, l, v in zip(y, left, w[:, j]):
            if v > 0.12:
                ax.text(l + v / 2, yi, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
        left += w[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b}\nbest single: {s}" for b, s in zip(bench, best)], fontsize=7.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("modality weight, entropy weight 0.01 (seeds averaged)")
    tidy(ax, "x")
    panel(ax, "a", "The gated head keeps one modality")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.36), ncol=3, handlelength=1.0, columnspacing=1.2)
    # (b) stability leak
    ax2 = fig.add_subplot(gs[1])
    models = ["Nucleotide Transformer alone", "RNA-FM alone", "RNA-FM on the codon grid", "ESM-2 8M alone",
              "Concatenation", "Gated MIL, 0.01", "Gated MIL without entropy", "Cross-attention"]
    full = [0.539, 0.552, 0.547, 0.531, 0.558, 0.551, 0.558, 0.553]
    clean = [0.266, 0.319, 0.340, 0.314, 0.359, 0.353, 0.359, 0.347]
    yy = np.arange(len(models))[::-1]
    for yi, a, b in zip(yy, full, clean):
        ax2.plot([b, a], [yi, yi], color=AXIS, lw=1.4, zorder=1)
        ax2.scatter([a], [yi], s=34, color=INK, zorder=3)
        ax2.scatter([b], [yi], s=34, facecolor=SURFACE, edgecolor=INK, lw=1.3, zorder=3)
        ax2.text((a + b) / 2, yi + 0.12, f"{b - a:+.2f}", ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(models)
    ax2.axhspan(-0.5, 3.5, color=GRID, alpha=0.45, zorder=0, lw=0)
    ax2.text(0.225, -0.42, "fusion heads", fontsize=6.4, color=MUTED, va="bottom")
    ax2.set_ylim(-0.6, len(models) - 0.3)
    ax2.set_xlim(0.22, 0.6)
    ax2.set_xlabel("Spearman, mRNA stability, seed 0")
    tidy(ax2, "x")
    panel(ax2, "b", "The published stability split is mostly memorization")
    handles = [Line2D([], [], marker="o", color=INK, ls="", ms=5.5, label="published test split (3,579 rows)"),
               Line2D([], [], marker="o", mfc=SURFACE, mec=INK, color=INK, ls="", ms=5.5, label="rows unseen in training (1,135)")]
    ax2.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.36), ncol=2, handletextpad=0.3, columnspacing=1.2)
    save(fig, "r34_gated_leak")


# ============================================================ figure D
# mRFP control (Table tab:mrfp)
def fig_mrfp():
    rows = [("mRNA-FM", "RNA", 0.466, 0.261, 0.182), ("NT 500M human-ref", "DNA", 0.291, 0.130, 0.076),
            ("RiNALMo", "RNA", 0.284, 0.098, 0.049), ("CaLM", "RNA", 0.304, 0.093, 0.022),
            ("HyenaDNA", "DNA", 0.247, 0.077, 0.036), ("NT v2 100M", "DNA", 0.264, 0.061, 0.029),
            ("RNA-FM", "RNA", 0.223, 0.054, 0.021), ("DNABERT-2", "DNA", 0.206, 0.044, 0.026),
            ("Protein encoders (4)", "protein", -0.005, 0.0, 0.0), ("640 random columns", "none", -0.007, -0.001, -0.001)]
    labels = [r[0] for r in rows]
    cols = [MODC.get(r[1], MUTED) for r in rows]
    y = np.arange(len(rows))[::-1]
    fig = plt.figure(figsize=(11.6, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.5, left=0.15, right=0.98, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    ax.barh(y, [r[2] for r in rows], color=cols, height=0.62, lw=0)
    for v, lab, col, ha in ((0.190, "Track A controls\nalone: 0.190", MUTED, "right"), (0.306, "all controls\nalone: 0.306", INK2, "left")):
        ax.axvline(v, color=col, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.text(v + (0.006 if ha == "left" else -0.006), -0.75, lab, fontsize=6.4, color=col, va="top", ha=ha, linespacing=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1.9, len(rows) - 0.4)
    ax.set_xlim(-0.03, 0.5)
    ax.set_xlabel("held-out R$^2$ of the probe alone, mean of ten fold assignments")
    tidy(ax, "x")
    panel(ax, "a", "What each encoder reads from 1,000 synonymous recodings")
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    ax2.barh(y, [r[4] for r in rows], color=cols, height=0.62, lw=0)
    for yi, r in zip(y, rows):
        if r[4] > 0:
            ax2.text(r[4] + 0.004, yi, "10 / 10" if r[0] != "CaLM" else "10 / 9", fontsize=6.4, color=MUTED, va="center")
    ax2.axvline(0, color=INK2, lw=0.8)
    ax2.text(0.004, 1.0, "exactly 0.000: the control holds", fontsize=6.4, color=PROT, va="center")
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlim(-0.02, 0.22)
    ax2.set_xlabel("held-out gain above GC, GC3, codon frequencies, pairing and folding")
    tidy(ax2, "x")
    panel(ax2, "b", "Gain over every control, and fold assignments above zero")
    handles = [Rectangle((0, 0), 1, 1, color=c, lw=0) for c in (DNA, RNA, PROT)]
    ax2.legend(handles, ["DNA", "RNA", "protein"], loc="lower right", handlelength=1.0)
    save(fig, "r34_mrfp")


# ============================================================ figure E
# Phase B (Tables tab:swaps, tab:swapboot)
def fig_phaseB():
    enc = ["RNA-FM, the paper's own", "mRNA-FM", "CaLM"]
    cov_alone, cov_cat = [0.833, 0.812, 0.844], [0.801, 0.759, 0.817]
    ec_alone, ec_cat = [50.7, 58.7, 57.6], [60.7, 59.0, 57.7]
    boot_cov = [
        ("mRNA-FM alone minus RNA-FM alone", -0.021, -0.056, 0.013, "s"),
        ("CaLM alone minus RNA-FM alone", 0.011, -0.017, 0.041, "s"),
        ("Concatenation, mRNA-FM minus original", -0.043, -0.074, -0.011, "c"),
        ("Concatenation, mRNA-FM minus its own best single", -0.054, -0.090, -0.019, "c"),
        ("Gated head, mRNA-FM minus its own best single", -0.050, -0.085, -0.018, "c"),
        ("Cross-attention, mRNA-FM minus its own best single", 0.009, -0.025, 0.043, "c"),
        ("Concatenation, CaLM minus its own best single", -0.027, -0.053, -0.002, "c"),
    ]
    boot_ec = [
        ("", 8.0, 4.0, 11.8, "s"), ("", 6.9, 2.8, 10.8, "s"), ("", -1.7, -4.8, 1.5, "c"),
        ("", 0.3, -2.9, 3.7, "c"), ("", -5.6, -9.7, -1.5, "c"), ("", -3.9, -7.9, 0.1, "c"),
        ("", 0.1, -2.5, 2.8, "c"),
    ]
    fig = plt.figure(figsize=(11.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55], width_ratios=[1, 1], hspace=0.75, wspace=0.18,
                          left=0.33, right=0.98, top=0.92, bottom=0.1)
    # top row: scores
    for k, (alone, cat, lim, xl, ttl) in enumerate([
            (cov_alone, cov_cat, (0.74, 0.86), "Spearman", "CoV-Vac, seed 0"),
            (ec_alone, ec_cat, (48, 63), "accuracy, %", "E. coli, seed 0")]):
        ax = fig.add_subplot(gs[0, k])
        y = np.arange(3)[::-1]
        for yi, a, c in zip(y, alone, cat):
            ax.annotate("", xy=(c, yi), xytext=(a, yi),
                        arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.4, shrinkA=4, shrinkB=3))
            ax.scatter([a], [yi], s=38, color=RNA, zorder=3)
            ax.scatter([c], [yi], s=38, facecolor=SURFACE, edgecolor=INK, lw=1.4, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(enc if k == 0 else ["", "", ""])
        ax.set_xlim(*lim)
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel(xl)
        tidy(ax, "x")
        panel(ax, "ab"[k], ttl)
        if k == 1:
            ax.text(58.7, 1.42, "alone: +8.0 over RNA-FM, resolved", fontsize=6.6, color=INK2, ha="center", va="bottom")
            ax.text(59.0, 0.58, "concatenation: +0.3 over it, not resolved", fontsize=6.6, color=INK2, ha="center", va="top")
    h = [Line2D([], [], marker="o", color=RNA, ls="", ms=5.5, label="encoder in the RNA slot, read alone through the same head"),
         Line2D([], [], marker="o", mfc=SURFACE, mec=INK, color=INK, ls="", ms=5.5, label="codon-aware concatenation built on it")]
    fig.axes[0].legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=2, handletextpad=0.3, columnspacing=1.5)
    # bottom row: paired bootstrap
    axc = fig.add_subplot(gs[1, 0])
    forest(axc, boot_cov, markers={"s": "o", "c": "o"})
    axc.set_xlim(-0.1, 0.06)
    axc.set_ylim(-0.6, 6.6)
    tidy(axc, "x")
    panel(axc, "c", "CoV-Vac: paired differences, 95% interval")
    axc.set_xlabel("difference in Spearman")
    axe = fig.add_subplot(gs[1, 1])
    forest(axe, boot_ec, markers={"s": "o", "c": "o"})
    axe.set_ylim(-0.6, 6.6)
    axe.set_xlim(-11, 13)
    tidy(axe, "x")
    panel(axe, "d", "E. coli: paired differences, 95% interval")
    axe.set_xlabel("difference in accuracy, points")
    for a in (axc, axe):
        a.axhline(4.5, color=AXIS, lw=0.6, zorder=0)
        a.axhline(0.5, color=AXIS, lw=0.6, zorder=0)
    axc.text(-0.098, 5.0, "the swapped encoder alone", fontsize=6.4, color=MUTED, va="bottom")
    axc.text(-0.098, 1.0, "fusion heads built on the swap", fontsize=6.4, color=MUTED, va="bottom")
    h2 = [Line2D([], [], marker="o", color=INK, ls="", ms=5, label="resolved"),
          Line2D([], [], marker="o", mfc=SURFACE, mec=MUTED, color=MUTED, ls="", ms=5, label="not resolved")]
    axe.legend(handles=h2, loc="lower right", handletextpad=0.3)
    save(fig, "r34_phaseB")


# ============================================================ figure F
# LucaOne on stability (Table tab:lucaone)
def fig_lucaone():
    rows = [("Composition (GC, GC3, codon usage, length)", 0.283, MUTED),
            ("Nucleotide Transformer v2 100M", 0.312, DNA), ("RNA-FM", 0.308, RNA), ("ESM-2 8M", 0.309, PROT),
            ("BioLangFusion's three encoders concatenated", 0.390, FUSED),
            ("LucaOne, gene stream", 0.443, "#6b6a65"), ("LucaOne, protein stream", 0.401, "#6b6a65"),
            ("LucaOne, both streams", 0.483, FUSED)]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(11.6, 3.3), gridspec_kw={"left": 0.3, "right": 0.98, "top": 0.86, "bottom": 0.2})
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, lw=0)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.005, yi, f"{r[1]:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 0.56)
    ax.set_xlabel("probe Spearman, mRNA stability, published test split, one ridge probe for every row")
    tidy(ax, "x")
    panel(ax, "", "One jointly pretrained model against three separately pretrained encoders fused")
    # brackets
    def bracket(x0, x1, yy, text):
        ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        ax.plot([x0, x0], [yy - 0.12, yy + 0.12], color=INK2, lw=0.8)
        ax.plot([x1, x1], [yy - 0.12, yy + 0.12], color=INK2, lw=0.8)
        ax.text(x0 - 0.006, yy, text, ha="right", va="center", fontsize=6.8, color=INK2)
    bracket(0.390, 0.483, -0.85, "both streams minus the three fused: +0.092 [+0.066, +0.120], resolved")
    bracket(0.312, 0.483, -1.55, "minus the best single encoder: +0.170 [+0.137, +0.204], resolved")
    ax.set_ylim(-2.1, len(rows) - 0.4)
    save(fig, "r34_lucaone")


# ============================================================ figure G
# Half-life and translation efficiency (Table tab:halflife)
def fig_halflife():
    rows = [("Composition of the coding sequence", 0.487, 0.683, MUTED, "comp"),
            ("Composition of the whole transcript", 0.542, 0.702, MUTED, "comp"),
            ("5' UTR, RNA encoder", 0.178, 0.433, RNA, "enc"),
            ("Coding sequence, RNA encoder", 0.473, 0.630, RNA, "enc"),
            ("Full transcript, RNA encoder", 0.512, 0.610, RNA, "enc"),
            ("Full transcript, DNA encoder", 0.514, 0.660, DNA, "enc"),
            ("Protein of the coding sequence", 0.552, 0.682, PROT, "enc"),
            ("Full transcript + protein", 0.606, 0.712, FUSED, "fused"),
            ("Three encoders fused", 0.602, 0.713, FUSED, "fused")]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.1, "left": 0.24, "right": 0.98, "top": 0.87, "bottom": 0.27})
    for k, (ax, col, ttl, lim) in enumerate(zip(axes, (1, 2), ("Half-life, human", "Translation efficiency, human"),
                                                ((0.1, 0.7), (0.38, 0.78)))):
        floor_cds, floor_all = rows[0][col], rows[1][col]
        ax.axvline(floor_all, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.text(floor_all, -0.95, "whole-transcript composition floor", fontsize=6.4, color=INK2, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
        for yi, r in zip(y, rows):
            v = r[col]
            if r[4] == "comp":
                ax.scatter([v], [yi], s=34, marker="D", facecolor=SURFACE, edgecolor=MUTED, lw=1.3, zorder=3)
            else:
                ax.plot([lim[0], v], [yi, yi], color=GRID, lw=3.2, zorder=1, solid_capstyle="butt")
                ax.scatter([v], [yi], s=38, color=r[3], zorder=3)
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=6.6, color=INK2)
        ax.axhspan(-0.5, 1.5, color=GRID, alpha=0.45, zorder=0, lw=0)
        ax.set_xlim(*lim)
        tidy(ax, "x")
        panel(ax, "ab"[k], ttl)
        ax.set_xlabel("mean test Pearson r (encoders: 9 splits; counted: 3)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_ylim(-1.4, len(rows) - 0.4)
    h = [Line2D([], [], marker="o", color=c, ls="", ms=5.5, label=l) for c, l in ((DNA, "DNA encoder"), (RNA, "RNA encoder"), (PROT, "protein encoder"), (FUSED, "fused, concatenated before the probe"))]
    h.append(Line2D([], [], marker="D", mfc=SURFACE, mec=MUTED, color=MUTED, ls="", ms=5, label="counted features, no encoder"))
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.6, 0.0), handletextpad=0.3, columnspacing=1.6)
    save(fig, "r34_halflife")


# ============================================================ figure H
# IsoFormer on GTEx (Table tab:iso)
def fig_isoformer():
    sets = ["DNA", "RNA", "Protein", "DNA + protein", "DNA + RNA", "DNA + RNA + protein"]
    ours = [0.199, 0.452, 0.160, 0.439, 0.507, 0.499]
    iso = [0.43, 0.61, 0.46, 0.52, 0.64, 0.65]
    fig, ax = plt.subplots(figsize=(11.6, 3.0), gridspec_kw={"left": 0.2, "right": 0.98, "top": 0.84, "bottom": 0.22})
    y = dumbbell(ax, sets, ours, iso)
    for yi, a, b in zip(y, ours, iso):
        ax.text(a - 0.012, yi, f"{a:.3f}", ha="right", va="center", fontsize=6.6, color=INK2)
        ax.text(b + 0.012, yi, f"{b:.2f}", ha="left", va="center", fontsize=6.6, color=MUTED)
    ax.axhspan(-0.5, 2.5, color=GRID, alpha=0.45, zorder=0, lw=0)
    ax.set_xlim(0.1, 0.72)
    ax.set_ylim(-0.6, len(sets) - 0.4)
    ax.set_xlabel("test Spearman averaged over thirty tissues, GTEx pilot, one seed")
    tidy(ax, "x")
    panel(ax, "", "IsoFormer's modality ablation: frozen encoders here, fine-tuned in the paper")
    ax.text(0.107, 0.5, "all three minus RNA alone: +0.047 [+0.027, +0.069], resolved\nDNA with RNA minus RNA alone: +0.055 [+0.032, +0.078], resolved",
            fontsize=6.6, color=INK2, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=AXIS, lw=0.6))
    h = [Line2D([], [], marker="o", color=INK, ls="", ms=5.5, label="ours, encoders frozen"),
         Line2D([], [], marker="o", mfc=SURFACE, mec=PUB, mew=1.4, color=PUB, ls="", ms=5.5, label="IsoFormer, encoders fine-tuned")]
    ax.legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 1.02), handletextpad=0.3)
    save(fig, "r34_isoformer")


# ============================================================ figure I
# Stage 3: the DNA upsampling, Equation 1 against the appendix (impulse test)
def fig_convolution():
    T = 5                       # 6-mer tokens
    ncod = 2 * T                # codon grid positions
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.1), gridspec_kw={"wspace": 0.12, "left": 0.03, "right": 0.99, "top": 0.84, "bottom": 0.06})
    specs = [("a", "Equation 1: kernel 2, stride 2", lambda t: [2 * t, 2 * t + 1], 2 * T),
             ("b", "Appendix: kernel 3, stride 2, padding 2", lambda t: [2 * t - 2, 2 * t - 1, 2 * t], 2 * T - 3)]
    for ax, (letter, ttl, where, outlen) in zip(axes, specs):
        ax.set_xlim(-0.6, ncod + 0.4)
        ax.set_ylim(-0.4, 4.9)
        ax.axis("off")
        panel(ax, letter, ttl)
        # codon grid (protein frame) on the bottom
        for c in range(ncod):
            ax.add_patch(FancyBboxPatch((c, 0.2), 0.92, 0.7, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=AXIS, lw=0.9))
            ax.text(c + 0.46, 0.55, f"codon {c}", ha="center", va="center", fontsize=6.0, color=INK2)
        ax.text(-0.5, 0.55, "protein\nframe", ha="right", va="center", fontsize=6.6, color=PROT, fontweight="bold")
        # where the 6-mers really sit
        for t in range(T):
            ax.add_patch(Rectangle((2 * t, 4.05), 1.92, 0.55, fc=SURFACE, ec=DNA, lw=1.0, ls=(0, (2, 1.5))))
            ax.text(2 * t + 0.96, 4.32, f"6-mer {t}", ha="center", va="center", fontsize=6.4, color=DNA)
        ax.text(-0.5, 4.32, "6-mer token\nreally covers", ha="right", va="center", fontsize=6.4, color=DNA)
        # where the transposed convolution places it: one thin row per token, so overlaps stay visible
        base, step, hgt = 1.15, 0.42, 0.34
        for t in range(T):
            pos = [p for p in where(t) if 0 <= p < outlen]
            yy = base + step * t
            if not pos:
                continue
            x0, x1 = min(pos), max(pos) + 1
            ok = pos == [2 * t, 2 * t + 1]
            ax.add_patch(Rectangle((x0 + 0.04, yy), x1 - x0 - 0.08, hgt, fc=(DNA if ok else "#f3c9b9"), ec=(DNA if ok else "#c0392b"), lw=0.8))
            ax.text((x0 + x1) / 2, yy + hgt / 2, f"token {t}", ha="center", va="center", fontsize=6.0,
                    color="white" if ok else INK, fontweight="bold")
            # a faint tie to where the 6-mer really sits
            ax.plot([2 * t + 0.96, (x0 + x1) / 2], [4.05, yy + hgt], color=AXIS, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.text(-0.5, base + step * 2 + hgt / 2, "where the\nconvolution\nplaces it", ha="right", va="center", fontsize=6.4, color=INK2)
        if outlen < ncod:
            ax.add_patch(Rectangle((outlen, base - 0.1), ncod - outlen, step * (T - 1) + hgt + 0.2, fc=SURFACE, ec=MUTED, lw=0.8, ls=(0, (2, 1.5)), hatch="///"))
            ax.text(outlen + (ncod - outlen) / 2, base + step * 2 + hgt / 2, "not\nwritten", ha="center", va="center", fontsize=6.2, color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
            ax.text(0.0, base + step * (T - 1) + hgt + 0.12, "each token lands two codons early, overlapping its neighbors; the output is three positions short",
                    ha="left", va="bottom", fontsize=6.6, color="#c0392b", style="italic")
        else:
            ax.text(0.0, base + step * (T - 1) + hgt + 0.12, "each token lands on exactly the two codons its 6-mer covers",
                    ha="left", va="bottom", fontsize=6.6, color=INK2, style="italic")
    save(fig, "s3_convolution")


if __name__ == "__main__":
    fig_reproduction()
    fig_fusion_forest()
    fig_gated_leak()
    fig_mrfp()
    fig_phaseB()
    fig_lucaone()
    fig_halflife()
    fig_isoformer()
    fig_convolution()
