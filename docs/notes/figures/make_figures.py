"""Build the Stage 1 figures.

Writes one PDF and one PNG per figure plus a combined all_figures.pdf.
Values come from figdata.py; see its docstring for provenance.

    python make_figures.py [output_dir]
"""
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr

import figdata as D

# ------------------------------------------------------------------ style
DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
MODC = {"DNA": DNA, "RNA": RNA, "protein": PROT}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ACCENT, AFTER = "#2a78d6", "#d94f1f"

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


def tidy(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid == "both":
        ax.grid(axis="both", linestyle="-")
    elif grid:
        ax.grid(axis=grid, linestyle="-")
        ax.grid(axis="y" if grid == "x" else "x", visible=False)


def caption(fig, text, y=0.012, width=198):
    fig.text(0.007, y, textwrap.fill(text, width), fontsize=6.6, color=MUTED,
             va="bottom", ha="left", linespacing=1.45)


def title(fig, text, y=0.982):
    fig.suptitle(text, x=0.007, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=y)


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ------------------------------------------------------------- figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(15, 100); ax.axis("off")
    ax.set_position([0.004, 0.004, 0.992, 0.952])

    def box(x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.9, r=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(x1, y1, x2, y2, c=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=c, lw=0.9, shrinkA=2, shrinkB=2, zorder=1))

    for x, t in ((10.0, "Data"), (39, "12 frozen encoders"), (83.5, "Stage 1 analyses")):
        ax.text(x, 97.6, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)

    ds = [(73.5, "mRNA stability", "CodonBERT benchmark,  n = 981",
           "SAME input: one coding sequence read\nas DNA, as RNA, and translated",
           "target: stability     compare: BioLangFusion"),
          (50.5, "GTEx expression", "IsoFormer pilot,  n = 999 + 996",
           "DISTINCT inputs: 6 kb DNA window,\ntranscript, protein",
           "target: 30 tissues     compare: IsoFormer")]
    for y, head, sub, body, foot in ds:
        box(0.6, y, 19.4, 18.5, fc="#f2f1ed", ec=AXIS)
        ax.text(10.3, y + 16.0, head, ha="center", fontsize=9.2, fontweight="bold", color=INK)
        ax.text(10.3, y + 13.3, sub, ha="center", fontsize=6.9, color=MUTED)
        ax.text(10.3, y + 8.3, body, ha="center", va="center", fontsize=7.3, color=INK2)
        ax.text(10.3, y + 2.4, foot, ha="center", fontsize=6.5, color=MUTED, style="italic")

    cols = [("DNA", DNA, D.DNA, 22.5), ("RNA", RNA, D.NUC_RNA + D.CODON_RNA, 37.5),
            ("protein", PROT, D.PROTEIN, 52.5)]
    cw, ch, gap = 13.4, 7.3, 0.9
    for name, colour, keys, x in cols:
        box(x, 87.0, cw, 5.4, fc=colour, ec=colour, r=1.0)
        ax.text(x + cw / 2, 89.7, name, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color="white")
        for i, k in enumerate(keys):
            y = 87.0 - (i + 1) * (ch + gap)
            box(x, y, cw, ch, fc="white", ec=colour, lw=1.0)
            ax.text(x + 0.9, y + ch - 2.2, D.SHORT[k], ha="left", fontsize=8,
                    fontweight="bold", color=INK)
            ax.text(x + 0.9, y + ch - 4.5, f"1 token = {D.TOKEN[k]}", ha="left",
                    fontsize=6.9, color=colour, fontweight="bold")
            ax.text(x + 0.9, y + 1.1, f"{D.WIDTH[k]}-d embedding", ha="left",
                    fontsize=6.4, color=MUTED)
    ax.text(46.2, 50.0, "frozen  ·  final layer  ·  mean-pooled to one vector per sequence",
            ha="center", fontsize=7.2, color=MUTED, style="italic")

    tracks = [(79.0, "Track A: Decodability", "ridge probes for stability, expression,\n"
               "GC / GC3, structure; concatenation gains", ACCENT),
              (66.0, "Track B: Geometry", "CKA, mutual k-NN, CCA retrieval, RSA,\n"
               "before and after the composition control", RNA),
              (53.0, "Track C: Diagnostics", "layer-wise CKA, attention against motifs,\n"
               "per-sequence alignment vs error", PROT)]
    for y, head, body, colour in tracks:
        box(67.5, y, 31.9, 11.4, fc="white", ec=colour, lw=1.1)
        ax.add_patch(FancyBboxPatch((67.5, y), 0.9, 11.4,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=colour, ec=colour, zorder=3))
        ax.text(69.6, y + 8.4, head, ha="left", fontsize=8.4, fontweight="bold", color=INK)
        ax.text(69.6, y + 4.0, body, ha="left", va="center", fontsize=7.1, color=INK2)

    arrow(20.2, 70.0, 22.1, 70.0); arrow(20.2, 59.0, 22.1, 62.5)
    arrow(66.2, 70.0, 67.1, 73.0); arrow(66.2, 66.0, 67.1, 62.0)

    stages = [(0.6, "Stage 2  ·  define alignment",
               "Aligned = agreement on held-out sequences\n"
               "AFTER the composition control. Reported per\n"
               "pair, kept separate from complementarity.", "#4a3aa7"),
              (33.6, "Stage 3  ·  fusion architecture",
               "Token-level fusion. Coupled Mamba on the\n"
               "codon grid where biology supplies one; cross-\n"
               "attention on GTEx. No alignment loss by default.", ACCENT),
              (66.6, "Stage 4  ·  benchmarking + method test",
               "Must beat length, composition, concatenation\n"
               "and the best single encoder. Then: do Stage 1\n"
               "measures predict which pairs gain?", RNA)]
    for x, head, body, colour in stages:
        box(x, 19.0, 32.8, 19.5, fc="#f2f1ed", ec=colour, lw=1.1)
        ax.text(x + 1.6, 34.6, head, ha="left", fontsize=8.6, fontweight="bold", color=colour)
        ax.text(x + 1.6, 26.5, body, ha="left", va="center", fontsize=7.1, color=INK2)
    arrow(33.4, 28.5, 33.8, 28.5); arrow(66.4, 28.5, 66.8, 28.5)
    arrow(83.0, 52.7, 83.0, 38.8)
    ax.text(83.8, 45.5, "evidence", fontsize=6.8, color=MUTED, rotation=90,
            ha="left", va="center", style="italic")

    title(fig, "Choosing and combining biological foundation models: Stage 1 design", y=0.985)
    return fig


# ------------------------------------------------------------- figure 2
LAB2 = {  # per-encoder label offsets, (dx, dy, ha) in points
    "b": {"hyenadna": (0, 9, "center"), "mrnafm": (-8, 0, "right"),
          "protbert": (9, 2, "left"), "calm": (-9, -1, "right"),
          "ntv2_100m": (8, -1, "left"), "dnabert2": (-8, -2, "right")},
    "c": {"mrnafm": (0, 9, "center"), "protbert": (8, 1, "left"),
          "calm": (-8, 0, "right"), "hyenadna": (7, -2, "left"),
          "ntv2_100m": (0, -13, "center"), "dnabert2": (-8, 1, "right")},
}


def fig2(stab, gtex):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.185, wspace=0.42)
    order = stab["share"].sort_values().index.tolist()
    y = np.arange(len(order))

    ax = axes[0]
    ax.barh(y + 0.19, [stab["share"][k] * 100 for k in order], height=0.36,
            color=ACCENT, label="stability (87 features)")
    ax.barh(y - 0.19, [gtex["share"][k] * 100 for k in order], height=0.36,
            color=PROT, label="GTEx (92 features)")
    ax.set_yticks(y, [D.SHORT[k] for k in order])
    ax.set_xlabel("Variance explained by composition (%)")
    ax.set_title("a   How much of each embedding is composition")
    ax.axvline(9, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(10.5, len(order) - 0.35, "chance ≈ 9%", fontsize=6.8, color=MUTED, va="center")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.03))
    tidy(ax)

    for ax, key, share, probe, ylab, head in (
            (axes[1], "b", stab["share"], D.STABILITY, "Stability probe $R^2$",
             "b   Stability: less composition, better probe"),
            (axes[2], "c", gtex["share"], D.GTEX, "Expression probe $R^2$",
             "c   GTEx: no such relation")):
        xs = np.array([share[k] * 100 for k in D.KEYS])
        ys = np.array([probe[k][2] for k in D.KEYS])
        ax.scatter(xs, ys, s=46, color=[MODC[D.MOD[k]] for k in D.KEYS], zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        rho, p = spearmanr(xs, ys)
        ax.text(0.96, 0.96, f"Spearman $\\rho$ = {rho:+.2f}\n{pfmt(p)},  n = 12",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
                color=INK if p < 0.05 else MUTED, linespacing=1.5)
        ax.set_xlabel("Composition share of the embedding (%)")
        ax.set_ylabel(ylab)
        ax.set_title(head)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(ys) * 1.38)
        tidy(ax, grid="both")
        for k, (dx, dy, ha) in LAB2[key].items():
            i = D.KEYS.index(k)
            ax.annotate(D.SHORT[k], (xs[i], ys[i]), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center", fontsize=6.5, color=INK2)

    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=6, label=m)
               for m, c in MODC.items()]
    axes[1].legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, -0.02))

    caption(fig, "Composition = GC, GC3, log length, 64 codon and 20 amino-acid frequencies (87 on stability; "
                 "92 on GTEx, which adds the DNA window and the untranslated regions). Each panel uses its own "
                 "dataset's features: plotting GTEx $R^2$ against the stability shares instead gives "
                 "$\\rho$ = -0.63, p = 0.028, but compares two different feature sets. $R^2$ is the mean over "
                 "ten fold assignments.")
    title(fig, "What sequence composition alone explains", y=0.975)
    return fig


# ------------------------------------------------------------- figure 3
def fig3(stab):
    pairs = stab["pairs"]
    rows = []
    for name, keys in D.GROUPS.items():
        s = pairs.loc[keys]
        rows.append((name, len(keys), s["CKA"].mean(), s["CKA_after"].mean(),
                     s["CKA_after"].min(), s["CKA_after"].max(), s["R1"].mean(),
                     s["R1_after"].mean(), s["R1_after"].min(), s["R1_after"].max()))
    rows.sort(key=lambda r: r[3])
    labels = [f"{r[0]}   (n = {r[1]})" for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    fig.subplots_adjust(left=0.215, right=0.985, top=0.855, bottom=0.145, wspace=0.06)
    for ax, idx, head, xlab in (
            (axes[0], (2, 3, 4, 5), "a   Linear CKA", "Agreement (group mean)"),
            (axes[1], (6, 7, 8, 9), "b   CCA retrieval, Recall@1", "Retrieval (group mean)")):
        before = np.array([r[idx[0]] for r in rows])
        after = np.array([r[idx[1]] for r in rows])
        lo = np.array([r[idx[2]] for r in rows])
        hi = np.array([r[idx[3]] for r in rows])
        ax.axhspan(len(rows) - 3.5, len(rows) - 0.35, color=ACCENT, alpha=0.05, zorder=0)
        ax.hlines(y, after, before, color="#d6d4cc", lw=3.6, zorder=1)
        ax.hlines(y, lo, hi, color="#8f2f18", lw=1.6, zorder=2)
        ax.scatter(before, y, s=42, facecolor=SURFACE, edgecolor=MUTED, lw=1.2,
                   zorder=3, label="before the control")
        ax.scatter(after, y, s=46, color=AFTER, zorder=4, edgecolor=SURFACE, lw=1.0,
                   label="after the control")
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(rows))
        ax.set_xlim(-0.02, 1.0)
        ax.set_xlabel(xlab)
        ax.set_title(head)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        tidy(ax)
    axes[1].axvline(0.0034, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    axes[1].text(0.02, -0.5, "chance 0.003", fontsize=6.6, color=MUTED, va="center")
    handles = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, markersize=6,
                      label="before the control"),
               Line2D([], [], marker="o", ls="", color=AFTER, markersize=6,
                      label="after the control"),
               Line2D([], [], color="#8f2f18", lw=1.6, label="range across the group's pairs")]
    axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, -0.02))
    axes[1].text(0.99, len(rows) - 0.6, "survive the control", ha="right", fontsize=7.2,
                 color="#1c5cab", fontweight="bold")

    caption(fig, "Composition control: each embedding is replaced by its residual after ordinary least squares "
                 "on the 87 composition features. All 981 stability sequences, all 66 encoder pairs, grouped.")
    title(fig, "Most cross-encoder agreement is sequence composition")
    return fig


# ------------------------------------------------------------- figure 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.175, right=0.975, top=0.845, bottom=0.185, wspace=0.30)

    ax = axes[0]
    palette = {"ESM-2 with ESM-2": MUTED, "RNA-FM with RiNALMo": MUTED,
               "CaLM with ESM-2": RNA, "NT 500M with RNA-FM": DNA,
               "mRNA-FM with ESM-2": PROT}
    nudge = {"ESM-2 with ESM-2": 0.045, "RNA-FM with RiNALMo": -0.052}
    for label, early, where, final, ref in D.LAYERWISE:
        c = palette[label]
        style = (dict(color=c, lw=1.3, ls=(0, (4, 2)), zorder=2) if ref
                 else dict(color=c, lw=1.9, zorder=3))
        ax.plot([0, 1], [np.mean(early), np.mean(final)], **style)
        for x, (lo, hi) in ((0, early), (1, final)):
            if hi - lo > 0.004:
                ax.plot([x, x], [lo, hi], color=c, lw=3.4, solid_capstyle="butt",
                        alpha=0.45, zorder=2)
            ax.scatter([x], [np.mean((lo, hi))], s=40, color=c, zorder=4,
                       edgecolor=SURFACE, lw=1.0)
        ly = np.mean(early) + nudge.get(label, 0.0)
        ax.annotate(label, (0, ly), textcoords="offset points", xytext=(-9, 3.5),
                    ha="right", va="center", fontsize=7.2,
                    color=MUTED if ref else INK2, fontweight="normal" if ref else "bold")
        ax.annotate(f"peak at {where}", (0, ly), textcoords="offset points", xytext=(-9, -4.5),
                    ha="right", va="center", fontsize=6.3, color=MUTED)
        ax.annotate(f"{np.mean(final):.2f}", (1, np.mean(final)), textcoords="offset points",
                    xytext=(9, 0), ha="left", va="center", fontsize=7.4, color=c,
                    fontweight="bold")
    ax.set_xlim(-0.95, 1.22); ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 1], ["peak over all layer pairs", "final layer vs final layer"])
    ax.set_ylabel("Linear CKA")
    ax.set_title("a   Agreement peaks early and decays with depth")
    ax.tick_params(axis="x", length=0)
    tidy(ax, grid="y")
    ax.text(0.5, 0.025, "dashed grey: same-family reference pairs", fontsize=6.6,
            color=MUTED, ha="center", transform=ax.transAxes)

    ax = axes[1]
    ax.bar(0, D.MRNAFM_MID[1] - D.MRNAFM_MID[0], bottom=D.MRNAFM_MID[0], width=0.5,
           color=PROT, alpha=0.85)
    ax.bar(1, D.MRNAFM_FINAL_CAP, width=0.5, color=PROT, alpha=0.20,
           edgecolor=PROT, lw=1.2, hatch="///")
    ax.text(0, D.MRNAFM_MID[1] + 0.035, f"{D.MRNAFM_MID[0]:.2f} to {D.MRNAFM_MID[1]:.2f}",
            ha="center", fontsize=7.6, color=INK, fontweight="bold")
    ax.text(1, D.MRNAFM_FINAL_CAP + 0.035, f"at most {D.MRNAFM_FINAL_CAP:.2f}",
            ha="center", fontsize=7.6, color=INK, fontweight="bold")
    ax.text(1, D.MRNAFM_FINAL_CAP + 0.105, f"({D.MRNAFM_FINAL_PROTBERT:.2f} with ProtBERT)",
            ha="center", fontsize=6.6, color=MUTED)
    ax.set_xticks([0, 1], ["states 2 to 5", "final state"])
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.08)
    ax.set_ylabel("CKA with every other encoder")
    ax.set_title("b   mRNA-FM's isolation is built in its last blocks")
    ax.tick_params(axis="x", length=0)
    tidy(ax, grid="y")

    caption(fig, f"Layer-wise CKA on a {D.LAYERWISE_N}-sequence subsample of the stability data, every hidden state "
                 "mean-pooled. Linear CKA is inflated at small n: these final-layer values exceed the 981-sequence "
                 "values behind Figure 3 for 63 of the 66 pairs, by up to 0.079, so read this figure within itself "
                 "rather than against Figure 3. Shaded bars span the ESM-2 models where a row covers more than one pair.")
    title(fig, "Where cross-encoder agreement lives inside the networks")
    return fig


# ------------------------------------------------------------- figure 5
def fig5(stab, gtex):
    kinds = [("within modality", MUTED, "o"),
             ("DNA with RNA or protein", DNA, "s"),
             ("nucleotide RNA with protein", RNA, "^"),
             ("codon RNA with protein", PROT, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.275, wspace=0.26)
    for ax, data, head in (
            (axes[0], stab, "a   mRNA stability: every encoder reads one coding sequence"),
            (axes[1], gtex, "b   GTEx: each encoder reads a different molecule")):
        p = data["pairs"]
        for kind, colour, marker in kinds:
            sub = p[p["kind"] == kind]
            for consistent, face in ((True, colour), (False, "none")):
                s = sub[(sub["n_pos"] == 10) == consistent]
                if len(s):
                    ax.scatter(s["CKA_after"], s["gain"], s=34, marker=marker,
                               facecolor=face, edgecolor=colour, linewidth=1.1, zorder=3)
        ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
        ra, pa = spearmanr(p["CKA_after"], p["gain"])
        cross = p[p["kind"] != "within modality"]
        rx, px = spearmanr(cross["CKA_after"], cross["gain"])
        ax.text(0.975, 0.965,
                f"all 66 pairs:   $\\rho$ = {ra:+.2f},  {pfmt(pa)}\n"
                f"cross-modal only (n = 48):   $\\rho$ = {rx:+.2f},  {pfmt(px)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
                color=INK2, linespacing=1.6)
        ax.set_xlabel("Alignment: linear CKA after the composition control")
        ax.set_ylabel("Concatenation gain in $R^2$\n(over the better single encoder)")
        ax.set_title(head)
        tidy(ax, grid="both")
    axes[0].set_xlim(-0.03, 0.70)
    axes[1].set_xlim(-0.04, 0.98)

    handles = [Line2D([], [], marker=m, ls="", mfc=c, mec=c, markersize=6, label=k)
               for k, c, m in kinds]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, markersize=6,
                       label="gains under all 10 fold assignments"),
                Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, markersize=6,
                       label="gains under fewer than 10")]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.098),
               columnspacing=2.2, handletextpad=0.6)

    caption(fig, "Gain is the mean over ten further fold assignments. On GTEx the pooled correlation is carried "
                 "largely by the within-modality pairs ($\\rho$ = -0.85, n = 18), where two DNA encoders read the "
                 "same window and are redundant by construction; the cross-modal half is the result that bears on "
                 "the Stage 2 decision. No trend line is drawn, because Spearman is a rank statistic.")
    title(fig, "Does alignment predict where combining encoders helps?")
    return fig


# ------------------------------------------------------------- figure 6
def fig6():
    order = sorted(D.KEYS, key=lambda k: D.STABILITY[k][0])
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    fig.subplots_adjust(left=0.125, right=0.985, top=0.845, bottom=0.165, wspace=0.06)

    for ax, probe, base, head, xmax in (
            (axes[0], D.STABILITY, D.STAB_LENGTH, "a   mRNA stability  (n = 981)", 0.20),
            (axes[1], D.GTEX, D.GTEX_LENGTH, "b   GTEx expression  (n = 999)", 0.325)):
        vals = np.array([probe[k][0] for k in order])
        sds = np.array([probe[k][1] for k in order])
        cols = [MODC[D.MOD[k]] for k in order]
        ax.hlines(y, 0, vals, color="#e6e4dd", lw=1.4, zorder=1)
        for yi, v, sd, c in zip(y, vals, sds, cols):
            ax.plot([max(v - sd, 0), v + sd], [yi, yi], color=c, lw=1.3, alpha=0.4, zorder=2)
        ax.scatter(vals, y, s=48, color=cols, zorder=4, edgecolor=SURFACE, lw=1.0)
        ax.axvline(base[0], color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
        right = base[0] > xmax / 2
        ax.text(base[0] + (-0.005 if right else 0.004), -0.62,
                f"length baseline  {base[0]:.3f}", fontsize=6.9, color=INK2,
                va="center", ha="right" if right else "left")
        ax.set_yticks(y, [D.SHORT[k] for k in order] if ax is axes[0] else [""] * len(order))
        ax.set_xlabel("Held-out $R^2$   (mean ± fold SD)")
        ax.set_title(head)
        ax.set_ylim(-1.15, len(order) - 0.35)
        ax.set_xlim(0, xmax)
        tidy(ax)

    axes[0].annotate("sequence-only ceiling\nset by label noise: $R^2$ ≈ 0.59",
                     xy=(0.192, 7.2), xytext=(0.104, 4.1), fontsize=7, color=INK2,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                     connectionstyle="arc3,rad=-0.2"))
    axes[1].text(D.GTEX_LENGTH[0] - 0.005, -0.95,
                 f"all 12 encoders concatenated  {D.GTEX_COMPOSITION[0]:.3f}",
                 fontsize=6.9, color=MUTED, va="center", ha="right")
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=6, label=m)
               for m, c in MODC.items()]
    axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 0.02))

    caption(fig, "Ridge probes on frozen mean-pooled embeddings; five folds that keep identical sequences "
                 "(stability) or transcripts of one gene (GTEx) together. On stability, repeated sequences carry "
                 "labels differing by a median SD of 0.49, which caps any sequence-only model near $R^2$ = 0.59, so "
                 "0.14 is about a quarter of what is reachable. Note the two panels use different x ranges.")
    title(fig, "What a linear probe can read from each frozen encoder")
    return fig


# ------------------------------------------------------------------ main
def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    stab, gtex = D.load()
    figs = [("fig1_project_schematic", fig1()),
            ("fig2_composition_share", fig2(stab, gtex)),
            ("fig3_composition_control", fig3(stab)),
            ("fig4_layerwise_agreement", fig4()),
            ("fig5_gain_vs_alignment", fig5(stab, gtex)),
            ("fig6_probe_results", fig6())]
    for name, fig in figs:
        fig.savefig(os.path.join(outdir, f"{name}.pdf"))
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=220)
        print("wrote", name)
    with PdfPages(os.path.join(outdir, "all_figures.pdf")) as pdf:
        for _, fig in figs:
            pdf.savefig(fig)
    print("wrote all_figures.pdf")
    for _, fig in figs:
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
