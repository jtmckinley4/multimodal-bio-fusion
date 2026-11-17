"""Phase B, the fusion-method question: does a fusion operation beat concatenation of the same inputs?

For each of the 32 swapped configurations (eight swaps on four benchmarks), the gated head minus
codon-wise concatenation and cross-attention minus codon-wise concatenation, each fitted on the
identical inputs, seed 0, with the 95% sequence-grouped paired-bootstrap interval (2,000 draws).

Source: the saved reports of tier2_encoder_swaps.ipynb,
Code/stage4/results/phaseB/analyses/paired-report-v1/<benchmark>__<hash>.json, rows of kind
"method". No value is typed by hand.

    python make_operation_figure.py <report_dir> <output_dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CROSS, GATED = "#0b0b0b", "#9a978d"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7.5, "legend.labelcolor": INK2, "pdf.fonttype": 42,
})

REPORTS = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BENCH = [("fungal_expression", "Fungal", "Spearman difference"),
         ("cov_vac_degradation", "CoV-Vac", "Spearman difference"),
         ("ecoli_proteins", "E. coli", "accuracy difference (points)"),
         ("mrna_stability", "Stability", "Spearman difference")]
SWAPS = [("nt500m", "NT 500M", DNA), ("hyenadna", "HyenaDNA", DNA), ("rinalmo", "RiNALMo", RNA),
         ("mrnafm", "mRNA-FM", RNA), ("calm", "CaLM", RNA), ("esm2_35m", "ESM-2 35M", PROT),
         ("esm2_150m", "ESM-2 150M", PROT), ("protbert", "ProtBERT", PROT)]
CONTRASTS = [("cross vs concat", "cross-attention minus concatenation", CROSS, -0.17),
             ("mil vs concat", "gated head minus concatenation", GATED, 0.17)]

method = {}
for f in glob.glob(os.path.join(REPORTS, "*.json")):
    b = os.path.basename(f).split("__")[0]
    for r in json.load(open(f))["analysis"]["per_seed"]:
        if r["kind"] == "method" and r["seed"] == 0:
            method[(b, r["encoder"], r["comparison"])] = r

fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.55), gridspec_kw={"wspace": 0.10})
fig.subplots_adjust(left=0.115, right=0.995, top=0.76, bottom=0.215)
order = SWAPS[::-1]
for ax, (b, name, unit) in zip(axes, BENCH):
    counts = {c: {"ahead": 0, "tie": 0, "behind": 0} for c, *_ in CONTRASTS}
    for yi, (k, _, _) in enumerate(order):
        for comp, _, colour, dy in CONTRASTS:
            r = method[(b, k, comp)]
            res = r["ci_low"] > 0 or r["ci_high"] < 0
            counts[comp]["ahead" if r["ci_low"] > 0 else "behind" if r["ci_high"] < 0 else "tie"] += 1
            y = yi + dy
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=colour, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.scatter([r["difference"]], [y], s=22, zorder=3, color=colour if res else SURFACE,
                       edgecolor=colour, linewidth=1.0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    for yb in (5.5, 2.5):
        ax.axhline(yb, color=AXIS, lw=0.6, ls=(0, (3, 2)))
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(range(len(order)))
    ax.set_title(name, pad=27)
    ax.set_xlabel(unit, fontsize=7.0)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    c, g = counts["cross vs concat"], counts["mil vs concat"]
    ax.text(0.0, 1.02, f"cross: {c['ahead']} ahead, {c['tie']} tie, {c['behind']} behind\n"
                        f"gated: {g['ahead']} ahead, {g['tie']} tie, {g['behind']} behind",
            transform=ax.transAxes, fontsize=6.5, color=INK2, va="bottom", linespacing=1.35)
axes[0].set_yticklabels([lab for _, lab, _ in order], fontsize=7.4)
for tick, (_, _, colour) in zip(axes[0].get_yticklabels(), order):
    tick.set_color(colour)
for ax in axes[1:]:
    ax.set_yticklabels([])
handles = [Line2D([], [], color=c, marker="o", ms=5, mfc=c, mec=c, lw=1.1, label=lab)
           for _, lab, c, _ in CONTRASTS]
handles.append(Line2D([], [], color=MUTED, marker="o", ms=5, mfc=SURFACE, mec=MUTED, lw=0,
                      label="open marker: interval crosses zero"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=3, fontsize=7.2,
           handlelength=1.6, columnspacing=1.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"phaseB_operation.{ext}"), dpi=220)
plt.close(fig)
print("phaseB_operation")
