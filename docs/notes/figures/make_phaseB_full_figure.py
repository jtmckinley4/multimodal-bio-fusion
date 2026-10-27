"""Phase B, all eight encoder swaps: one figure for the manuscripts and one panel for the deck.

(a) Each swapped encoder read alone, against the original encoder it replaces read alone.
(b) Each fusion head minus the best single encoder validation chose, for every swap and
    benchmark, with the cell shaded when the 95% paired-bootstrap interval excludes zero.

Inputs are Chase's tier2_encoder_swaps.ipynb outputs (score table and paired intervals, seed 0,
2,000 draws, identical sequences resampled together), extracted to phaseB_readout.json, plus the
original encoders' seed-0 single scores from Phase A and Stage 3.

    python make_phaseB_full_figure.py <phaseB_readout.json> <output_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DNA, RNA, PROT = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", os.environ.get("SLIDE_BG", "#fcfcfb")
WIN, LOSS, TIE, NONE = "#cfe1f5", "#f7d6c6", "#f1f0ec", SURFACE
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

R = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

BENCH = [("fungal_expression", "Fungal"), ("cov_vac_degradation", "CoV-Vac"),
         ("ecoli_proteins", "E. coli (pts)"), ("mrna_stability", "Stability")]
SWAPS = [  # key, label, slot, replaces
    ("nt500m", "NT 500M", "dna", "NT v2 100M"), ("hyenadna", "HyenaDNA", "dna", "NT v2 100M"),
    ("rinalmo", "RiNALMo", "rna", "RNA-FM"), ("mrnafm", "mRNA-FM", "rna", "RNA-FM"), ("calm", "CaLM", "rna", "RNA-FM"),
    ("esm2_35m", "ESM-2 35M", "prot", "ESM-2 8M"), ("esm2_150m", "ESM-2 150M", "prot", "ESM-2 8M"),
    ("protbert", "ProtBERT", "prot", "ESM-2 8M"),
]
SLOT_COLOR = {"dna": DNA, "rna": RNA, "prot": PROT}
# The original encoders read alone, seed 0 (Phase A equation runs; fungal from Stage 3 step 3).
ORIGINAL = {
    "fungal_expression": {"dna": 0.8125, "rna": 0.7219, "prot": 0.6975},
    "cov_vac_degradation": {"dna": 0.7790, "rna": 0.8332, "prot": 0.8155},
    "ecoli_proteins": {"dna": 47.75, "rna": 50.70, "prot": 50.14},
    "mrna_stability": {"dna": 0.5391, "rna": 0.5517, "prot": 0.5311},
}
HEADS = [("concat", "concat"), ("mil", "gated"), ("cross", "cross")]
# Swaps where validation chose a single encoder that scores clearly below another eligible single
# encoder on test. Fungal with ProtBERT: validation chose ProtBERT alone (0.761) while NT v2 100M
# alone scores 0.813, so these differences are measured against a weak reference.
WEAK_REFERENCE = {("fungal_expression", "protbert")}


def fmt(b, v):
    return f"{v:+.1f}" if b == "ecoli_proteins" else f"{v:+.3f}"


def panel_alone(axes, big=1.0):
    order = SWAPS[::-1]
    for ax, (b, name) in zip(axes, BENCH):
        tab = R["table"][b]
        ys = []
        for yi, (k, lab, slot, rep) in enumerate(order):
            if k not in tab or tab[k]["alone"] is None:
                ax.text(0.5, yi, "not run", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=6.6 * big, color=MUTED)
                continue
            new, old = tab[k]["alone"], ORIGINAL[b][slot]
            ax.plot([old, new], [yi, yi], color=AXIS, lw=1.5 * big, zorder=1)
            ax.scatter([old], [yi], s=30 * big, facecolor=SURFACE, edgecolor=INK, lw=1.1, zorder=3)
            ax.scatter([new], [yi], s=34 * big, color=SLOT_COLOR[slot], zorder=4)
            ys.append(yi)
        ax.set_title(name, fontsize=8.6 * big)
        ax.set_yticks(range(len(order)))
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.grid(axis="x"); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for yb in (2.5, 5.5):
            ax.axhline(yb, color=AXIS, lw=0.6, ls=(0, (3, 2)))
    axes[0].set_yticklabels([f"{lab}  (for {rep})" for _, lab, _, rep in order], fontsize=7.4 * big)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_xlim(0.66, 0.88); axes[1].set_xlim(0.70, 0.86); axes[2].set_xlim(45, 61); axes[3].set_xlim(0.52, 0.575)
    axes[0].set_xticks([0.70, 0.78, 0.86]); axes[1].set_xticks([0.72, 0.78, 0.84]); axes[2].set_xticks([46, 52, 58]); axes[3].set_xticks([0.53, 0.55, 0.57])


def panel_grid(ax, big=1.0, title=True):
    rows = [("original", "BioLangFusion's own", None)] + [(k, lab, slot) for k, lab, slot, _ in SWAPS]
    ncol = len(BENCH) * 3
    cw, rh = 1.0, 1.0
    for ri, (k, lab, slot) in enumerate(rows):
        y = len(rows) - 1 - ri
        for bi, (b, _) in enumerate(BENCH):
            for hi, (h, _) in enumerate(HEADS):
                x = bi * 3 + hi + bi * 0.35
                tab = R["table"][b].get(k)
                if tab is None:
                    fill, txt, w = NONE, "", "normal"
                elif k == "original":
                    fill, txt, w = SURFACE, fmt(b, tab[f"{h}_d"]), "normal"
                else:
                    iv = R["intervals"][b][k][h]["best"]
                    res, d = iv["resolved"], iv["d"]
                    fill = (WIN if d > 0 else LOSS) if res else TIE
                    txt, w = fmt(b, d) + ("†" if (b, k) in WEAK_REFERENCE else ""), ("bold" if res else "normal")
                ax.add_patch(Rectangle((x, y), cw, rh, facecolor=fill, edgecolor=SURFACE, lw=1.6))
                if k == "original" and tab is not None:
                    ax.add_patch(Rectangle((x + 0.04, y + 0.06), cw - 0.08, rh - 0.12, facecolor="none", edgecolor=AXIS, lw=0.7, ls=(0, (2, 2))))
                if txt:
                    ax.text(x + cw / 2, y + rh / 2, txt, ha="center", va="center", fontsize=6.9 * big, color=INK if tab is not None else MUTED, fontweight=w)
                elif tab is None and hi == 1:
                    ax.text(x + cw / 2, y + rh / 2, "not run", ha="center", va="center", fontsize=6.3 * big, color=MUTED)
        ax.text(-0.28, y + rh / 2, lab, ha="right", va="center", fontsize=7.4 * big, color=INK2 if slot else INK, fontweight="normal" if slot else "bold")
        if slot:
            ax.add_patch(Rectangle((-0.17, y + 0.18), 0.08, rh - 0.36, facecolor=SLOT_COLOR[slot], edgecolor="none"))
    top = len(rows)
    for bi, (b, name) in enumerate(BENCH):
        x0 = bi * 3 + bi * 0.35
        ax.text(x0 + 1.5, top + 0.95, name, ha="center", va="bottom", fontsize=8.0 * big, color=INK, fontweight="bold")
        for hi, (_, hl) in enumerate(HEADS):
            ax.text(x0 + hi + 0.5, top + 0.12, hl, ha="center", va="bottom", fontsize=6.9 * big, color=INK2)
    ax.axhline(len(rows) - 1, xmin=0, xmax=1, color=INK2, lw=0.6)
    ax.set_xlim(-0.2, ncol + 3 * 0.35 + 0.05)
    ax.set_ylim(-0.1, top + 1.7)
    ax.axis("off")
    if title:
        ax.set_title("b   Fusion minus the best single encoder, one seed, 95% paired interval", loc="left")


def legend_grid(fig, y, big=1.0):
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=WIN, label="fusion ahead, resolved"), Patch(facecolor=LOSS, label="fusion behind, resolved"),
          Patch(facecolor=TIE, label="tie: interval crosses zero"),
          Patch(facecolor=SURFACE, edgecolor=AXIS, ls="--", label="original encoders, seed 0, for reference")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4, fontsize=7.2 * big, handlelength=1.4, columnspacing=1.4)


# --------------------------------------------------------------- manuscript figure: (a) over (b)
fig = plt.figure(figsize=(7.6, 7.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.36, wspace=0.12, left=0.19, right=0.99, top=0.935, bottom=0.07)
axa = [fig.add_subplot(gs[0, i]) for i in range(4)]
panel_alone(axa)
fig.text(0.01, 0.995, "a   Each swapped encoder read alone (filled) against the encoder it replaces (open)", fontsize=9, fontweight="bold", color=INK, va="top")
axb = fig.add_subplot(gs[1, :])
panel_grid(axb, title=False)
fig.text(0.01, 0.525, "b   Fusion minus the best single encoder validation chose, one seed, 95% paired interval", fontsize=9, fontweight="bold", color=INK, va="top")
legend_grid(fig, 0.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"r34_phaseB_full.{ext}"), dpi=220, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- deck panel: (b) alone, larger type
big = 1.3
fig, ax = plt.subplots(figsize=(11.6, 4.4))
fig.subplots_adjust(left=0.13, right=0.995, top=0.98, bottom=0.12)
panel_grid(ax, big=big, title=False)
legend_grid(fig, -0.02, big=big)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"r34_phaseB_grid_slide.{ext}"), dpi=220, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- deck panel: (a) alone, larger type
big = 1.25
fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.5), gridspec_kw={"wspace": 0.10, "left": 0.175, "right": 0.995, "top": 0.88, "bottom": 0.16})
panel_alone(axes, big=big)
for ax in axes:
    ax.tick_params(axis="x", labelsize=8.0 * big)
axes[0].set_yticklabels([f"{lab}  (for {rep})" for _, lab, _, rep in SWAPS[::-1]], fontsize=9.0)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker="o", mfc=SURFACE, mec=INK, mew=1.1, color=AXIS, ls="-", ms=6, label="the encoder it replaces"),
      Line2D([], [], marker="o", color=DNA, ls="", ms=6.5, label="swapped in: DNA slot"),
      Line2D([], [], marker="o", color=RNA, ls="", ms=6.5, label="RNA slot"),
      Line2D([], [], marker="o", color=PROT, ls="", ms=6.5, label="protein slot")]
fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.57, -0.04), ncol=4, fontsize=8.6, handletextpad=0.4, columnspacing=1.8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"r34_phaseB_alone_slide.{ext}"), dpi=220, bbox_inches="tight")
plt.close(fig)
print("r34_phaseB_full, r34_phaseB_grid_slide, r34_phaseB_alone_slide")
