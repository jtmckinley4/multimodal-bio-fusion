"""Slide-size figures for the December 1 progress deck.

The deck shows the manuscript's figures, drawn at the size a slide needs: the same saved
results, the same marks and the same colors as make_newton_figures.py,
make_three_seed_figures.py and make_coupled_mamba_figure.py, with type at 8.5 to 10 point at
true slide size, a white ground to sit on the deck's white cards, panel titles short enough
not to collide, and no figure title where the slide title already says it.

Every plotted result is read from a saved file under Code/stage4/results; nothing is typed in.
The one exception is rna_encoder_corpora, whose parameter counts and pretraining corpora come
from each model's card and paper, as recorded in Code/shared_code/encoders.py and
docs/methods/sequences-to-predictions.md.

  newton_ernie_swaps_slide        RNAErnie and ERNIE-RNA against RiNALMo on E. coli, seed by seed
  rna_encoder_corpora_slide       what each RNA encoder was pretrained on, and each swap read alone
  newton_grouped_stability_slide  grouped mRNA stability: each head and the encoder swap
  newton_gate_study_slide         what the gated head keeps, and what each design change buys
  newton_anchor_attention_slide   anchor-preserving attention arms against the best single encoder
  newton_anchor_correction_slide  correcting the best single encoder, beside the two shams
  halflife_pilot_slide            the five region-aware half-life fits
  seeds_operation_slide           cross-attention and the gated head minus concatenation
  seeds_fusion_counts_slide       ahead / unresolved / behind for every head and benchmark
  coupled_mamba_slide             the design and its two controls, named by what each adds
  partner_panel_rpi2241_slide     the six partner arms on rpi2241's three splits

    python make_dec1_deck_figures.py <repository root> <output dir>
"""
import glob
import json
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
CROSS, GATED = "#7a4fb5", "#c98a12"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
PAIR, SINGLE = "#3d3c39", "#c9c7c0"
WIN, LOSS, TIE = "#cfe1f5", "#f7d6c6", "#f1f0ec"
FILL = "#e8e6df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 8.8,
    "axes.titlesize": 9.8, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.8, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
})

ROOT, OUT = sys.argv[1], sys.argv[2]
R = os.path.join(ROOT, "Code", "stage4", "results")
os.makedirs(OUT, exist_ok=True)
W = 6.65          # inches: the width of the figure area on the slide
DPI = 300


def load(*parts):
    path = os.path.join(R, *parts)
    if any(ch in path for ch in "*?"):
        found = sorted(glob.glob(path))
        assert len(found) == 1, (path, found)
        path = found[0]
    with open(path) as handle:
        return json.load(handle)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}_slide.png"), dpi=DPI)
    plt.close(fig)
    print(name + "_slide")


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "\u2212")


def resolved(lo, hi):
    return lo > 0 or hi < 0


def mark(ax, y, d, lo, hi, color, size=6.5, lw=2.0):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=2)
    ax.plot([d], [y], "o", ms=size, mfc=color if resolved(lo, hi) else SURFACE,
            mec=color, mew=1.5, zorder=3)


def zero(ax):
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rows_axis(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)


def filled_open(color=INK2):
    return [Line2D([], [], marker="o", ls="", ms=6.5, mfc=color, mec=color, label="interval excludes zero"),
            Line2D([], [], marker="o", ls="", ms=6.5, mfc=SURFACE, mec=color, mew=1.5,
                   label="interval crosses zero")]


def bundles():
    return [Line2D([], [], marker="o", ls="", ms=6.5, color=INK2, label="BioLangFusion's encoders"),
            Line2D([], [], marker="o", ls="", ms=6.5, color=RNA, label="mRNA-FM in the RNA slot")]


# ------------------------------------------------------------------ the Newton three-seed reports
paired = {}
for path in glob.glob(os.path.join(R, "newton_h100_seeds012", "local_analyses", "paired-report-v1", "*.json")):
    d = json.load(open(path))
    b = d["identity"]["benchmark"]
    for r in d["analysis"]["seed_summary"]:
        assert r["status"] == "complete", (b, r["encoder"], r["comparison"])
        paired[(b, r["encoder"], r["comparison"])] = r
BENCH = [("fungal_expression", "Fungal", "Spearman difference"),
         ("cov_vac_degradation", "CoV-Vac", "Spearman difference"),
         ("ecoli_proteins", "E. coli", "accuracy points"),
         ("mrna_stability", "Stability", "Spearman difference")]
SWAPS = [("nt500m", "NT 500M", DNA), ("hyenadna", "HyenaDNA", DNA), ("rinalmo", "RiNALMo", RNA),
         ("mrnafm", "mRNA-FM", RNA), ("calm", "CaLM", RNA), ("esm2_35m", "ESM-2 35M", PROT),
         ("esm2_150m", "ESM-2 150M", PROT), ("protbert", "ProtBERT", PROT)]


def verdict(r):
    return "ahead" if r["ci_low"] > 0 else "behind" if r["ci_high"] < 0 else "tie"


# ------------------------------------------------------------------ 1. ERNIE swaps
er = load("newton_ecoli_ernie", "local_analyses", "ecoli_ernie_three_seed.json")
names = {"rinalmo": "RiNALMo\n(reference)", "rnaernie": "RNAErnie", "ernierna": "ERNIE-RNA"}
colors = {"rinalmo": INK2, "rnaernie": RNA, "ernierna": DNA}
fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
fig.subplots_adjust(left=0.095, right=0.99, top=0.9, bottom=0.27, wspace=0.07)
for ax, head in zip(axes, ("MIL gated", "Concatenation")):
    rows = [r for r in er["rows"] if r["head"] == head]
    ref = next(r for r in rows if r["rna_encoder"] == "rinalmo")["mean_accuracy_percent"]
    ax.plot([0.36, len(rows) - 0.25], [ref, ref], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for x, r in enumerate(rows):
        vals = [s["test_accuracy"] * 100 for s in r["seeds"]]
        c = colors[r["rna_encoder"]]
        ax.plot([x - 0.14, x, x + 0.14], vals, "o", ms=5, color=c, alpha=0.75, zorder=3)
        ax.plot([x - 0.3, x + 0.3], [r["mean_accuracy_percent"]] * 2, color=c, lw=3, zorder=4)
        ax.text(x + 0.34, r["mean_accuracy_percent"], f"{r['mean_accuracy_percent']:.1f}", fontsize=9,
                color=INK, va="center", fontweight="bold", zorder=5,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([names[r["rna_encoder"]] for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.2)
    ax.set_title("Gated head" if head == "MIL gated" else "Concatenation")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("E. coli accuracy (%)")
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=MUTED, label="one training seed"),
                    Line2D([], [], color=MUTED, lw=3, label="three-seed mean"),
                    Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 3)), label="RiNALMo's mean")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.54, -0.015), columnspacing=2.2)
save(fig, "newton_ernie_swaps")

# ------------------------------------------------------------------ 2. what each RNA encoder was pretrained on
# name, parameters in millions, label, coding-pretrained; from each model's card and paper
ENCODERS = [("NucleicBERT", 400, "~400M  ·  MARS, ~30M", False),
            ("mRNA-FM", 240, "~240M  ·  RefSeq coding, ~45M", True),
            ("RiNALMo mega", 148, "148M  ·  RNAcentral and others, 36M", False),
            ("RNA-FM", 100, "100M  ·  RNAcentral, 23.7M", False),
            ("CaLM", 86, "86M  ·  ENA coding DNA, 9.9M", True),
            ("ERNIE-RNA", 86, "86M  ·  RNAcentral, 20.4M", False),
            ("RNAErnie", 86, "~86M  ·  RNAcentral", False)]
fig = plt.figure(figsize=(W, 2.3))
ax = fig.add_axes([0.168, 0.305, 0.3, 0.585])
for y, (name, params, label, coding) in enumerate(ENCODERS):
    ax.barh(y, params, height=0.64, color=RNA if coding else SINGLE, zorder=2)
    inside = params > 300
    ax.text(params - 8 if inside else params + 8, y, label, fontsize=7.9, color=INK if inside else INK2,
            va="center", ha="right" if inside else "left", zorder=5)
rows_axis(ax, [e[0] for e in ENCODERS])
ax.set_xlim(0, 430)
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_xlabel("parameters (millions)")
ax.set_title("(a) Size and pretraining corpus")
ax.grid(axis="x")
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.legend(handles=[Patch(color=RNA, label="pretrained on coding sequences"),
                    Patch(color=SINGLE, label="non-coding RNA only")],
           loc="lower left", bbox_to_anchor=(0.03, -0.02), ncol=2, fontsize=7.9, columnspacing=1.2,
           handlelength=1.2, handletextpad=0.5)
alone = [("rinalmo", "RiNALMo", False), ("mrnafm", "mRNA-FM", True), ("calm", "CaLM", True)]
for k, (bench, title, scale_label) in enumerate((("fungal_expression", "Fungal", "Spearman"),
                                                 ("ecoli_proteins", "E. coli", "accuracy points"))):
    ax = fig.add_axes([0.722 + k * 0.142, 0.305, 0.125, 0.585])
    for y, (key, label, coding) in enumerate(alone):
        r = paired[(bench, key, "replacement single vs original single")]
        mark(ax, y, r["difference"], r["ci_low"], r["ci_high"], RNA if coding else MUTED, size=6)
    rows_axis(ax, [a[1] for a in alone])
    if k:
        ax.set_yticklabels([])
    zero(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.set_xlim(0, None)
    ax.set_xlabel(scale_label, fontsize=8)
    ax.set_title(("(b) " if k == 0 else "") + title)
fig.text(0.845, 0.03, "read alone, minus RNA-FM alone", fontsize=7.9, color=INK2, ha="center")
save(fig, "rna_encoder_corpora")

# ------------------------------------------------------------------ 3. grouped stability
g = load("grouped_stability_v1", "reports", "mrna_stability__*.json")
gs = {r["comparison"]: r for r in g["analysis"]["seed_summary"]}
spec = [("mRNA-FM alone minus RNA-FM alone", "mRNA-FM single vs RNA-FM single", RNA),
        (None, "BioLangFusion's encoders: head minus ESM-2 alone", None),
        ("concatenation", "concat vs selected single (original)", INK2),
        ("gated head", "mil vs selected single (original)", GATED),
        ("cross-attention", "cross vs selected single (original)", CROSS),
        (None, "mRNA-FM in the RNA slot: head minus mRNA-FM alone", None),
        ("concatenation", "concat vs selected single (mrnafm)", INK2),
        ("gated head", "mil vs selected single (mrnafm)", GATED),
        ("cross-attention", "cross vs selected single (mrnafm)", CROSS)]
fig, ax = plt.subplots(figsize=(W, 2.55))
fig.subplots_adjust(left=0.37, right=0.985, top=0.975, bottom=0.285)
labels = []
for y, (label, key, color) in enumerate(spec):
    if label is None:
        labels.append("")
        ax.text(-0.118, y, key, fontsize=8.8, fontweight="bold", color=INK, va="center", ha="left",
                zorder=5, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
        continue
    r = gs[key]
    mark(ax, y, r["difference"], r["ci_low"], r["ci_high"], color)
    ax.text(r["ci_high"] + 0.004, y, signed(r["difference"]), fontsize=8.5, color=INK, va="center")
    labels.append(label)
rows_axis(ax, labels)
zero(ax)
ax.set_xlim(-0.12, 0.125)
ax.set_xlabel("Spearman difference, 3,496 test rows in 1,687 sequence groups")
fig.legend(handles=filled_open(), loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.675, -0.015),
           columnspacing=1.6, handletextpad=0.3)
save(fig, "newton_grouped_stability")

# ------------------------------------------------------------------ 4. gate study
dist = load("gate_diagnostics_v1", "reports", "gate-distributions-v3.json")["summary"]
cmp_ = load("gate_diagnostics_v1", "reports", "three-seed-comparison-v4.json")
gates = {}
for r in dist:
    if (r["quantity"] == "gates" and r["benchmark"] == "cov_vac_degradation"
            and r["bundle"] == "original" and r["model"] in ("mil_entropy", "sequence_sigmoid")):
        gates.setdefault((r["model"], r["modality"]), []).append(r["median"])
intervals = {(p["comparison"], p["benchmark"], p["bundle"]): p for p in cmp_["paired_intervals"]}
contrasts = [("Sigmoid minus softmax\n(removes competition)", "sequence_sigmoid minus mil_no_entropy"),
             ("Without minus with\nthe entropy term", "mil_no_entropy minus mil_entropy"),
             ("Per codon minus\nper sequence", "position_sigmoid minus joint_sequence_sigmoid"),
             ("Per-codon gate minus\nsame-size ungated control", "position_sigmoid minus position_residual_control")]
fig = plt.figure(figsize=(W, 2.75))
panel = {0: [0.075, 0.3, 0.215, 0.6], 1: [0.575, 0.3, 0.19, 0.6], 2: [0.795, 0.3, 0.19, 0.6]}
ax = fig.add_axes(panel[0])
mods = [("dna", "DNA", DNA), ("rna", "RNA-FM", RNA), ("prot", "protein", PROT)]
for i, (m, name, color) in enumerate(mods):
    for j, (model, shade) in enumerate((("mil_entropy", 0.35), ("sequence_sigmoid", 1.0))):
        vals = gates[(model, m)]
        x = i + (j - 0.5) * 0.4
        ax.bar(x, np.mean(vals), width=0.36, color=color, alpha=shade, edgecolor=color, lw=0.8)
        ax.plot([x] * len(vals), vals, "o", ms=3, color=INK, zorder=3)
ax.set_xticks(range(3))
ax.set_xticklabels([m[1] for m in mods])
ax.set_ylim(0, 1.05)
ax.set_ylabel("gate weight")
ax.set_title("(a) Gate weights, CoV-Vac")
ax.tick_params(axis="x", length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y")
ax.set_axisbelow(True)
fig.legend(handles=[Patch(facecolor=INK2, alpha=0.35, label="softmax, as published"),
                    Patch(facecolor=INK2, label="sigmoid"),
                    Line2D([], [], marker="o", ls="", ms=3, color=INK, label="one seed")],
           loc="lower left", bbox_to_anchor=(0.0, -0.01), ncol=1, fontsize=7.9, labelspacing=0.25, handlelength=1.2)
for k, (bench, title, unit, scale) in enumerate(
        (("cov_vac_degradation", "(b) CoV-Vac", "Spearman difference", 1),
         ("ecoli_proteins", "(c) E. coli", "accuracy points", 100))):
    ax = fig.add_axes(panel[k + 1])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    for y, (label, key) in enumerate(contrasts):
        for off, bundle, color in ((-0.16, "original", INK2), (0.16, "mrnafm", RNA)):
            p = intervals[(key, bench, bundle)]
            mark(ax, y + off, p["difference"] * scale, p["ci_low"] * scale, p["ci_high"] * scale, color, 6)
    rows_axis(ax, [c[0] for c in contrasts])
    ax.tick_params(axis="y", labelsize=8.4)
    if k == 1:
        ax.set_yticklabels([])
    zero(ax)
    ax.set_xlabel(unit)
    ax.set_title(title)
fig.legend(handles=bundles() + filled_open(), loc="lower right", ncol=2, fontsize=7.9,
           bbox_to_anchor=(0.995, -0.01), columnspacing=1.2, labelspacing=0.25, handletextpad=0.2)
save(fig, "newton_gate_study")

# ------------------------------------------------------------------ 5. anchor attention
att = load("anchor_cross_attention_v1", "reports", "three-seed-comparison-v3.json")
ai = {(p["comparison"], p["benchmark"], p["bundle"]): p for p in att["paired_intervals"]}
arms = [("projected anchor alone", "anchor_only"), ("same-size mixer, no attention", "anchor_concat_control"),
        ("attention, no skip", "attention_ordinary"), ("attention with skip", "attention_skip"),
        ("skip with sequence gate", "attention_sequence_sigmoid"), ("skip with position gate", "attention_position_sigmoid")]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
fig.subplots_adjust(left=0.3, right=0.985, top=0.89, bottom=0.27, wspace=0.09)
for ax, (bench, title, unit, scale) in zip(axes, (
        ("cov_vac_degradation", "(a) CoV-Vac", "Spearman difference", 1),
        ("ecoli_proteins", "(b) E. coli", "accuracy points", 100))):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    for y, (label, key) in enumerate(arms):
        for off, bundle, color in ((-0.16, "original", INK2), (0.16, "mrnafm", RNA)):
            p = ai[(f"{key} minus selected_single", bench, bundle)]
            mark(ax, y + off, p["difference"] * scale, p["ci_low"] * scale, p["ci_high"] * scale, color, 6)
    rows_axis(ax, [a[0] for a in arms])
    zero(ax)
    ax.set_xlabel(unit)
    ax.set_title(title)
fig.text(0.285, 0.985, "each arm minus the\nbest single encoder", fontsize=8.4, color=INK, fontweight="bold",
         ha="right", va="top", linespacing=1.25)
fig.legend(handles=bundles() + filled_open(), loc="lower center", ncol=4, fontsize=7.9,
           bbox_to_anchor=(0.5, -0.015), columnspacing=0.9, handletextpad=0.15)
save(fig, "newton_anchor_attention")

# ------------------------------------------------------------------ 6. anchor correction
corr = {}
for path in glob.glob(os.path.join(R, "anchor_correction", "*", "analyses", "seed-summary-*.json")):
    d = json.load(open(path))
    corr[d["identity"]["benchmark"]] = d
configs = [("CoV-Vac", "cov_vac_degradation", "original", "dna+prot", "RNA-FM corrected by DNA + protein"),
           ("CoV-Vac", "cov_vac_degradation", "mrnafm", "dna+rna", "ESM-2 corrected by DNA + RNA"),
           ("Stability, grouped", "mrna_stability", "original", "rna", "ESM-2 corrected by RNA"),
           ("Stability, grouped", "mrna_stability", "mrnafm", "dna+prot", "mRNA-FM corrected by DNA + protein"),
           ("Fungal", "fungal_expression", "original", "rna+prot", "NT corrected by RNA + protein"),
           ("Fungal", "fungal_expression", "mrnafm", "dna+prot", "mRNA-FM corrected by DNA + protein")]
fig, ax = plt.subplots(figsize=(W, 2.75))
fig.subplots_adjust(left=0.385, right=0.985, top=0.975, bottom=0.275)
for y, (bench_label, bench, bundle, streams, how) in enumerate(configs):
    d = corr[bench]
    means = {r["model"]: r["mean"] for r in d["analysis"]["summary"] if r["bundle"] == bundle}
    anchor = means["anchor"]
    rows = {r["comparison"]: r for r in d["analysis"]["paired"][bundle]["seed_summary"]}
    p = rows[f"{streams}/textcnn: correction vs anchor"]
    for sham, marker in (("row_shuffle_v1", "s"), ("random", "^")):
        ax.plot([means[f"{streams}/{sham}/textcnn"] - anchor], [y], marker, ms=5.5, mfc=SURFACE,
                mec=MUTED, mew=1.1, zorder=2)
    mark(ax, y, p["difference"], p["ci_low"], p["ci_high"], INK2)
    ax.text(max(p["ci_high"], 0.0) + 0.002, y, signed(p["difference"]), fontsize=8.5, color=INK, va="center")
    ax.text(-0.02, y - 0.17, bench_label, transform=ax.get_yaxis_transform(), fontsize=8.8, color=INK,
            fontweight="bold", ha="right", va="center")
    ax.text(-0.02, y + 0.2, how, transform=ax.get_yaxis_transform(), fontsize=8, color=INK2,
            ha="right", va="center")
rows_axis(ax, [""] * len(configs))
zero(ax)
ax.set_xlim(-0.0125, 0.068)
ax.set_xlabel("Spearman difference from the anchor (the best single encoder)")
fig.legend(handles=[Line2D([], [], marker="o", ls="-", lw=2, ms=6.5, color=INK2, label="real corrector, 95% interval"),
                    Line2D([], [], marker="s", ls="", ms=5.5, mfc=SURFACE, mec=MUTED, label="sham: shuffled rows"),
                    Line2D([], [], marker="^", ls="", ms=5.5, mfc=SURFACE, mec=MUTED, label="sham: random features")],
           loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.015), columnspacing=1.4)
save(fig, "newton_anchor_correction")

# ------------------------------------------------------------------ 7. the half-life pilot
pilot = [("Native tokens", "RNA alone", False, ("region_aware_v1", "rnafm__native__rna_only__true")),
         ("Native tokens", "with protein: coding-region attention", True,
          ("region_aware_native_sliced", "rnafm__native__attention_cds__true")),
         ("Codon-pooled", "RNA alone", False, ("region_aware_v1", "rnafm__codon_pool__rna_only__true")),
         ("Codon-pooled", "with protein: position gate", True, ("region_aware_v1", "rnafm__codon_pool__position_gate__true")),
         ("Codon-pooled", "with protein: coding-region attention", True,
          ("region_aware_v1", "rnafm__codon_pool__attention_cds__true"))]
fig, ax = plt.subplots(figsize=(W, 2.8))
fig.subplots_adjust(left=0.5, right=0.975, top=0.975, bottom=0.245)
ypos = [0, 1, 2.45, 3.45, 4.45]
for y, (group, label, both, (study, fit)) in zip(ypos, pilot):
    d = load(study, "rnahl-human", "fits", fit, "seed0.json")
    colour = PAIR if both else MUTED
    ax.axhline(y, color=GRID, lw=0.8, zorder=1)
    ax.plot([d["test_score"]], [y], "o", ms=9, color=colour, zorder=3)
    ax.plot([d["val_score"]], [y], "o", ms=6.5, mfc=SURFACE, mec=INK2, mew=1.3, zorder=3)
    ax.text(d["test_score"], y - 0.36, f"{d['test_score']:.3f}", fontsize=8.8, fontweight="bold", color=INK,
            va="center", ha="center", zorder=5)
    ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), fontsize=8.6, color=INK2, ha="right", va="center")
for y, group in ((0, "Native tokens"), (2.45, "Codon-pooled")):
    ax.text(-0.985, y, group, transform=ax.get_yaxis_transform(), fontsize=8.8, fontweight="bold", color=INK,
            ha="left", va="center")
ax.set_yticks([])
ax.set_ylim(5.0, -0.75)
ax.set_xlim(0.5, 0.655)
ax.set_xlabel("Spearman, human mRNA half-life, seed 0")
ax.grid(axis="x")
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=MUTED, label="RNA alone, test"),
                    Line2D([], [], marker="o", ls="", ms=9, color=PAIR, label="RNA with protein, test"),
                    Line2D([], [], marker="o", ls="", ms=6.5, mfc=SURFACE, mec=INK2, mew=1.3, label="validation")],
           loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.015), columnspacing=1.6)
save(fig, "halflife_pilot")

# ------------------------------------------------------------------ 8. the operation question
CONTRASTS = [("cross vs concat", "cross-attention minus concatenation", CROSS, -0.18),
             ("mil vs concat", "gated head minus concatenation", GATED, 0.18)]
fig, axes = plt.subplots(1, 4, figsize=(W, 2.6), sharey=True)
fig.subplots_adjust(left=0.142, right=0.99, top=0.905, bottom=0.27, wspace=0.11)
order = SWAPS[::-1]
for ax, (b, name, unit) in zip(axes, BENCH):
    for yi, (k, _, _) in enumerate(order):
        for comp, _, colour, dy in CONTRASTS:
            r = paired[(b, k, comp)]
            ax.plot([r["ci_low"], r["ci_high"]], [yi + dy, yi + dy], color=colour, lw=1.5,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r["difference"]], [yi + dy], "o", ms=4.6, zorder=3, mew=1.2, mec=colour,
                    mfc=SURFACE if verdict(r) == "tie" else colour)
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    for yb in (5.5, 2.5):
        ax.axhline(yb, color=AXIS, lw=0.6, ls=(0, (3, 2)))
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(range(len(order)))
    ax.set_title(name)
    ax.set_xlabel(unit, fontsize=8.2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=False))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticklabels([lab for _, lab, _ in order], fontsize=8.6)
handles = [Line2D([], [], color=c, marker="o", ms=5, mfc=c, mec=c, lw=1.5, label=lab) for _, lab, c, _ in CONTRASTS]
handles.append(Line2D([], [], color=MUTED, marker="o", ms=5, mfc=SURFACE, mec=MUTED, lw=0, label="open: unresolved"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3, fontsize=8, columnspacing=1.3)
save(fig, "seeds_operation")

# ------------------------------------------------------------------ 9. ahead / unresolved / behind
HEADS = [("concat", "concatenation"), ("mil", "gated head"), ("cross", "cross-attention")]
fig, axes = plt.subplots(1, 4, figsize=(W, 2.7), sharey=True)
fig.subplots_adjust(left=0.185, right=0.985, top=0.81, bottom=0.29, wspace=0.1)
total = {"ahead": 0, "tie": 0, "behind": 0}
for ax, (b, name, _) in zip(axes, BENCH):
    sub = {"ahead": 0, "tie": 0, "behind": 0}
    for y, (h, _) in enumerate(HEADS):
        counts = {"ahead": 0, "tie": 0, "behind": 0}
        for k, _, _ in SWAPS:
            counts[verdict(paired[(b, k, f"{h} vs validation-selected single")])] += 1
        left = 0
        for key, colour in (("ahead", WIN), ("tie", TIE), ("behind", LOSS)):
            n = counts[key]
            sub[key] += n
            total[key] += n
            if n:
                ax.barh(y, n, left=left, height=0.66, color=colour, edgecolor=SURFACE, lw=1.2, zorder=2)
                ax.text(left + n / 2, y, str(n), fontsize=9, color=INK, ha="center", va="center",
                        fontweight="normal" if key == "tie" else "bold", zorder=3)
            left += n
    ax.set_xlim(0, 8)
    ax.set_xticks([0, 4, 8])
    ax.set_ylim(2.55, -0.55)
    ax.set_yticks(range(3))
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{name}\n{sub['ahead']} / {sub['tie']} / {sub['behind']}", fontsize=9.4, linespacing=1.35)
    ax.set_xlabel("of 8 encoder swaps", fontsize=8.2, labelpad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
axes[0].set_yticklabels([h[1] for h in HEADS])
fig.legend(handles=[Patch(facecolor=WIN, edgecolor=AXIS, label=f"fusion ahead of the best single: {total['ahead']}"),
                    Patch(facecolor=TIE, edgecolor=AXIS, label=f"unresolved: {total['tie']}"),
                    Patch(facecolor=LOSS, edgecolor=AXIS, label=f"fusion behind: {total['behind']}")],
           loc="lower center", ncol=3, fontsize=8.2, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
save(fig, "seeds_fusion_counts")

# ------------------------------------------------------------------ 10. Coupled Mamba, named by what each design adds


def box(ax, x, y, w, h, text, ec=AXIS, colour=INK, fs=6.8, fc="white", bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc,
                                ec=ec, lw=1.0, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=colour,
            fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)


def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=0.9, ms=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=c,
                                 lw=lw, shrinkA=1.5, shrinkB=1.5, zorder=1))


STREAMS = [("DNA", DNA, 84), ("RNA", RNA, 76), ("protein", PROT, 68)]
CHAIN_ROWS = {"DNA": 56, "RNA": 46, "protein": 36}
STEPS = [16, 50, 84]


def inputs(ax, label):
    for name, c, y in STREAMS:
        ax.text(7, y, name, ha="left", va="center", fontsize=7.6, color=c, fontweight="bold")
        box(ax, 34, y - 3.2, 58, 6.4, label, ec=c, colour=c, fs=7.1)


def chains(ax, coupled):
    w, h = 10.0, 6.4
    for name, c, _ in STREAMS:
        y = CHAIN_ROWS[name]
        for i, x in enumerate(STEPS):
            box(ax, x - w / 2, y - h / 2, w, h, "h", ec=c, colour=c, fs=7.2)
            if i < len(STEPS) - 1 and not coupled:
                arrow(ax, x + w / 2, y, STEPS[i + 1] - w / 2, y, c=c, lw=1.0)
    if coupled:
        mw, mh = 14.0, 7.4
        for i in range(len(STEPS) - 1):
            xm = (STEPS[i] + STEPS[i + 1]) / 2
            box(ax, xm - mw / 2, 46 - mh / 2, mw, mh, "mean", ec=INK2, colour=INK, fs=6.0, fc=FILL)
            for name, c, _ in STREAMS:
                y = CHAIN_ROWS[name]
                arrow(ax, STEPS[i] + w / 2, y, xm - mw / 2, 46 + (y - 46) * 0.3, c=c, lw=0.8, ms=6)
                arrow(ax, xm + mw / 2, 46 + (y - 46) * 0.3, STEPS[i + 1] - w / 2, y, c=c, lw=0.8, ms=6)
    for i, x in enumerate(STEPS):
        ax.text(x, 29.5, ["codon t-1", "codon t", "codon t+1"][i], ha="center", va="center",
                fontsize=6.4, color=MUTED)


fig, axes = plt.subplots(1, 4, figsize=(7.45, 4.0))
fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.175, wspace=0.04)
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
titles = [("Concatenation", "BioLangFusion head, Phase A"),
          ("No recurrence", "control (mamba_bottleneck)"),
          ("Separate recurrence", "ablation (uncoupled_mamba)"),
          ("Shared recurrence", "the design (coupled_mamba)")]
for ax, (t, sub) in zip(axes, titles):
    ax.text(50, 100, t, ha="center", va="top", fontsize=9.8, fontweight="bold", color=INK)
    ax.text(50, 93.6, sub, ha="center", va="top", fontsize=7.6, color=INK2, style="italic")
ax = axes[0]
for name, c, y in STREAMS:
    ax.text(7, y, name, ha="left", va="center", fontsize=7.6, color=c, fontweight="bold")
    box(ax, 34, y - 3.2, 58, 6.4, "project DNA to 600" if name == "DNA" else "as encoded", ec=c, colour=c, fs=7.1)
box(ax, 8, 36, 84, 18, "concatenate the three\nstreams at each codon;\neach codon on its own", fs=7.4)
ax.text(50, 29.5, "no state from codon to codon", ha="center", va="center", fontsize=6.6, color=MUTED)
ax = axes[1]
inputs(ax, "project to 128")
box(ax, 8, 36, 84, 18, "no Mamba layers", ec=MUTED, colour=MUTED, fs=7.6, ls=(0, (3, 2)))
ax.text(50, 29.5, "each codon on its own", ha="center", va="center", fontsize=6.6, color=MUTED)
ax = axes[2]
inputs(ax, "project to 128")
chains(ax, coupled=False)
ax = axes[3]
inputs(ax, "project to 128")
chains(ax, coupled=True)
for ax in axes:
    arrow(ax, 50, 63.5, 50, 60.5 if ax is not axes[0] and ax is not axes[1] else 55, c=MUTED)
for i, ax in enumerate(axes):
    text = "project to 600" if i == 0 else "normalize, concatenate at\neach codon, project to 600"
    box(ax, 6, 12.5, 88, 11, text, fs=7.0)
    arrow(ax, 50, 26.5, 50, 23.8)
eqs = ["", "",
       "h$^m_t$ = $\\bar{A}^m_t$ h$^m_{t-1}$ + $\\bar{B}^m_t$ x$^m_t$\neach modality keeps its own state",
       "h$^m_t$ = $\\bar{A}^m_t$ H$_{t-1}$ + $\\bar{B}^m_t$ x$^m_t$\nH$_{t-1}$ = mean over m of h$^m_{t-1}$"]
for ax, e in zip(axes, eqs):
    if e:
        ax.text(50, 3.0, e, ha="center", va="center", fontsize=6.8, color=INK2, linespacing=1.45)
for i, lab in enumerate(["adds the narrowing\n(width 128 per modality)", "adds the recurrence\nalong the codons",
                         "adds the coupling\nof the states"]):
    x0 = axes[i].get_position().x0 + axes[i].get_position().width * 0.5
    x1 = axes[i + 1].get_position().x0 + axes[i + 1].get_position().width * 0.5
    y = 0.082
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=9, color=INK2, lw=1.0))
    fig.text((x0 + x1) / 2, y - 0.018, lab, ha="center", va="top", fontsize=7.6, color=INK,
             fontweight="bold", linespacing=1.25)
save(fig, "coupled_mamba")

# ------------------------------------------------------------------ 11. partner panel
arms = [("cross_attention", "cross-attention", True), ("bilinear", "bilinear", True),
        ("concat", "concatenation", True), ("interleaved_mamba", "interleaved Mamba", True),
        ("left_only", "RNA alone", False), ("right_only", "protein alone", False)]
splits = [("held_out_both", "(a) Both held out"), ("held_out_left", "(b) RNAs held out"),
          ("held_out_right", "(c) Proteins held out")]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.55), sharey=True)
fig.subplots_adjust(left=0.19, right=0.985, top=0.895, bottom=0.29, wspace=0.1)
for ax, (split, title) in zip(axes, splits):
    degree = None
    for y, (key, label, pair) in enumerate(arms):
        d = load("partner_tasks", "rpi2241", split, f"{key}__seed0.json")
        auc = d["result"]["test"]["auroc"]
        degree = d["controls"]["degree"]["auroc"]
        ax.barh(y, auc - 0.4, left=0.4, height=0.66, color=PAIR if pair else SINGLE, zorder=2)
        ax.text(auc - 0.008, y, f"{auc:.3f}", fontsize=8.4, color="white" if pair else INK, va="center",
                ha="right", fontweight="bold", zorder=5)
    ax.axvline(degree, color=PROT, lw=1.8, zorder=1.5)
    ax.text(0.985, 0.955, f"degree baseline {degree:.3f}", transform=ax.transAxes, fontsize=7.8, color=PROT,
            ha="right", va="top", fontweight="bold",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0}, zorder=6)
    rows_axis(ax, [a[1] for a in arms])
    ax.set_ylim(len(arms) - 0.5, -1.3)
    ax.set_xlim(0.4, 0.95)
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.set_xlabel("test AUROC")
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
fig.legend(handles=[Patch(color=PAIR, label="reads both molecules"), Patch(color=SINGLE, label="reads one molecule"),
                    Line2D([], [], color=PROT, lw=1.8, label="degree baseline, reads no sequence")],
           loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.015), columnspacing=1.4)
save(fig, "partner_panel_rpi2241")
