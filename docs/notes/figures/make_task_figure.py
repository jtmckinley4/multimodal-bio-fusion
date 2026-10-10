"""Task coverage matrix for the Stage 4 benchmark plan.

Rows are candidate datasets, columns are the information each one makes
available. Values are drawn from the cited papers' own descriptions.

    python make_task_figure.py <outdir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import make_figures as M
from make_figures import AXIS, DNA, INK, INK2, MUTED, PROT, RNA, SURFACE

M.title = lambda *a, **k: None
M.caption = lambda *a, **k: None

COLS = ["genomic\nDNA", "5' UTR", "CDS", "3' UTR", "protein", "protein held\nconstant",
        "multiple\nspecies", "tissues or\ncell types", "in hand"]
COLC = [DNA, RNA, RNA, RNA, PROT, INK2, INK2, INK2, INK2]

# 1 = available, 0 = not, 0.5 = partial (see notes in the manuscript)
# The last field is the item's priority in the research plan (blank: original Stage 4 set).
ROWS = [
    ("mRNA stability (CodonBERT / Diez 2022)",  [0, 0, 1, 0, 1, 0, 1, 0, 1],   "1"),
    ("mRFP synonymous variants (CodonBERT)",     [0, 0, 1, 0, 1, 1, 0, 0, 1],   "5, parallel"),
    ("Fungal expression (Wint 2022)",            [0, 0, 1, 0, 1, 0, 1, 0, 0],   "1"),
    ("E. coli expression (Ding 2022)",           [0, 0, 1, 0, 1, 0, 0, 0, 0],   "1"),
    ("Ab1 antibody mRNA expression (Yazdani)",   [0, 0, 1, 0, 1, 0, 0, 0, 0],   "1, added"),
    ("CoV-Vac mRNA degradation (Leppek 2022)",   [0, 0, 0.5, 0, 0.5, 0, 0, 0, 0], "1, added"),
    ("GTEx isoform expression (IsoFormer)",      [1, 1, 1, 1, 1, 0, 0, 1, 0.5], "2"),
    ("mRNA half-life, Saluki (mRNABench)",       [0.5, 1, 1, 1, 1, 0, 1, 1, 0], "3"),
    ("LBKWK synthetic mRNA: half-life and MRL",  [0, 1, 1, 1, 1, 0, 0, 0, 0],   "3, 4, added"),
    ("Translation efficiency (mRNABench)",       [0, 1, 1, 1, 1, 0, 1, 1, 0],   "4"),
    ("Mean ribosome load, Sugimoto (mRNABench)", [0, 1, 1, 1, 1, 0, 0, 0, 0],   "4, added"),
    ("COMET multi-molecule: RPI, EPI, CRISPR off-target", [1, 0, 0.5, 0, 1, 0, 0, 0, 0], "listed"),
    ("Protein-to-mRNA ratio (Eraslan)",          [0, 1, 1, 1, 1, 0, 0, 1, 0],   ""),
    ("CaLM suite: abundance, Tm, solubility",    [0, 0, 1, 0, 1, 0, 1, 0, 0],   ""),
    ("COMET cross-molecule CDS/protein: Flu, EC, Beta-Lac", [0, 0, 1, 0, 1, 0, 0, 0, 0], "listed"),
    ("mRNABench: GO, localization, RBP, UTRs",   [0, 1, 1, 1, 1, 0, 1, 0.5, 0], ""),
    ("5' UTR ribosome load (MRL MPRA)",          [0, 1, 0, 0, 0, 1, 0, 0, 0],   ""),
    ("3' UTR polyadenylation (APA)",             [0, 0, 0, 1, 0, 1, 0, 0, 0],   ""),
    ("Enhancer perturbation (CDT)",              [1, 0, 0, 0, 1, 0, 0, 0, 0],   ""),
]


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    n, m = len(ROWS), len(COLS)
    fig, ax = plt.subplots(figsize=(10.4, 7.9))
    fig.subplots_adjust(left=0.31, right=0.985, top=0.88, bottom=0.05)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (label, vals, _) in enumerate(ROWS):
        for j, v in enumerate(vals):
            if v == 1:
                ax.add_patch(Rectangle((j - 0.42, i - 0.38), 0.84, 0.76, fc=COLC[j], ec="none", alpha=0.9))
            elif v == 0.5:
                ax.add_patch(Rectangle((j - 0.42, i - 0.38), 0.84, 0.76, fc="white", ec=COLC[j], lw=1.3, hatch="////"))
            else:
                ax.add_patch(Rectangle((j - 0.42, i - 0.38), 0.84, 0.76, fc="#f1f0eb", ec="none"))
    for cut in (5.5, 11.5, 15.5, 17.5):
        ax.axhline(cut, color=AXIS, lw=0.8)
    ax.axvline(4.5, color=AXIS, lw=0.8)
    ax.set_yticks(range(n), [r[0] for r in ROWS], fontsize=7.6)
    ax.set_xticks(range(m), COLS, fontsize=7.3)
    ax.xaxis.tick_top()
    for lab, c in zip(ax.get_xticklabels(), COLC):
        lab.set_color(c)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(-0.5, -1.55, "what the dataset provides", fontsize=8, color=MUTED, ha="left", va="center", style="italic")
    ax.text(m + 0.1, -1.55, "research plan\npriority", fontsize=8, color=MUTED, ha="left", va="center", style="italic")
    for i, (_, _, pr) in enumerate(ROWS):
        if pr:
            ax.text(m + 0.1, i, pr, fontsize=7.6, color=INK, ha="left", va="center", fontweight="bold")
    fig.text(0.31, 0.012, "filled: available    hatched: partly (see text)    grey: not available    listed: from the October paper list",
             fontsize=7.4, color=MUTED, ha="left", va="bottom")
    for y, txt in ((2.5, "same coding\nsequence"), (8.5, "distinct\nmolecules"), (13.5, "codon vs\namino acid"),
                   (16.5, "single region,\nno protein"), (18.0, "regulatory\nDNA")):
        ax.text(m + 2.1, y, txt, fontsize=6.8, color=MUTED, ha="left", va="center", style="italic")
    ax.set_xlim(-0.5, m + 3.9)
    fig.savefig(os.path.join(outdir, "task_coverage.pdf"))
    fig.savefig(os.path.join(outdir, "task_coverage.png"), dpi=200)
    print("wrote task_coverage")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
