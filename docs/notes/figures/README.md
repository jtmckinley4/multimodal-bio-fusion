# Stage 1 figures

Six figures covering the Stage 1 results, as PDF (vector, for Overleaf) and PNG
(220 dpi, for slides). `all_figures.pdf` holds all six in order.

| File | What it shows |
| --- | --- |
| `fig1_project_schematic` | The pipeline: both datasets and their contrasting input designs, the twelve encoders with the token unit and embedding width of each, the three tracks, and Stages 2 to 4. |
| `fig2_composition_share` | How much of each embedding the composition features explain, and whether that predicts probe performance. It does on stability, not on GTEx. |
| `fig3_composition_control` | Cross-encoder agreement before and after the composition control, by pair group, in both CKA and CCA retrieval. |
| `fig4_layerwise_agreement` | Where agreement sits inside the networks: it peaks in early or middle layers and decays, and mRNA-FM's isolation is built in its last blocks. |
| `fig5_gain_vs_alignment` | Concatenation gain against alignment after the control, on both datasets, split by pair type. |
| `fig6_probe_results` | What a linear probe reads from each frozen encoder on both tasks, against the length baseline and the label-noise ceiling. |

## Reading notes

- **Figure 2** uses each dataset's own composition features: 87 on stability, 92
  on GTEx, which adds the DNA window and the untranslated regions. Plotting GTEx
  expression against the *stability* shares instead gives a significant-looking
  rho of -0.63, but it compares two different feature sets. With GTEx's own
  features there is no relationship (rho -0.20, p = 0.53), because the DNA
  encoders there read a fixed 6 kb window, which compresses the shares into a
  33 to 53% band.
- **Figure 4** comes from a 194-sequence subsample. Linear CKA is inflated at
  small n, and these final-layer values exceed the 981-sequence values behind
  Figure 3 for 63 of the 66 pairs, so the two figures are not directly
  comparable. A true layer-by-layer heatmap needs the 66 matrices in `layer_cka`
  from a notebook run.
- **Figure 5** reports the pooled correlation and the cross-modal-only
  correlation. On GTEx the pooled -0.64 is carried largely by the
  within-modality pairs (rho -0.85, n = 18), where two DNA encoders read the
  same window and are redundant by construction. The cross-modal half
  (rho -0.42, p = 0.003) is the number that bears on the Stage 2 decision. No
  trend line is drawn, since Spearman is a rank statistic.

## Rebuilding

```bash
cd docs/notes/figures
python make_figures.py out
```

Needs matplotlib, numpy, pandas and scipy. `figdata.py` holds the per-encoder
values, all of them saved outputs of `Code/Stage1_stability.ipynb` and
`Code/Stage1_gtex.ipynb` from the October 6 runs in
[run records](../../run-records.md#recorded-runs). The `*_geom.csv`,
`*_gain.csv` and `*_share.csv` tables carry the full 66-pair results, which the
saved notebook outputs truncate; they were recomputed read-only from the cached
embeddings in `Code/stage1_embeddings/` and reproduce every value the notebooks
print.

## Paper figures

`paper/` holds the figures the Overleaf manuscript includes, built by
`make_paper_figures.py` from the same values and tables as the six above:

```bash
cd docs/notes/figures
python make_paper_figures.py paper
```

The six figures above are re-rendered there without their embedded title and
caption (the LaTeX caption carries that text) and with larger type, so they
stay legible at the text width of a letter page. Fourteen further figures come from
the 66-pair tables and from values recorded in the manuscript:

| File | What it shows |
| --- | --- |
| `agreement_matrices_stability` | Linear CKA and CCA Recall@1 for every pair, before and after the composition control, on the stability data. |
| `agreement_matrices_gtex` | The same on GTEx. |
| `gain_matrices` | Mean concatenation gain for every pair on both datasets; boxed cells gain under all ten fold assignments. |
| `track_a_controls` | GC against GC3 per encoder (the drop in the protein encoders), and the Nussinov pairing-fraction probe. |
| `published_comparison` | Frozen probes on the published splits against BioLangFusion's and IsoFormer's reported values. |
| `variance_bands` | Top ten principal components against the rest: variance held, composition share, and where the stability signal sits. |
| `attention_tests` | Share of sequences with nominal p < 0.05 per encoder and motif, and the position and codon-and-frame nulls. |
| `alignment_vs_error` | Spearman rho between per-sequence alignment and prediction error for all 66 pairs on both datasets. |
| `codon_grid` | Schematic of positional alignment: the codon grid on the stability data and its absence on GTEx. |
| `fusion_architectures` | Schematic of the three Stage 3 designs: concatenation + MLP, Coupled Mamba on the codon grid (priority 1), cross-attention (priority 2). |
| `blf_pipeline` | The research plan's add-on baseline: the BioLangFusion architecture (codon grid, three heads, TextCNN) reproduced then run with swapped encoders; pooled or region-aware fusion where no grid exists; LucaOne and the Stage 3 designs it is compared with. |
| `project_logic` | The chain of stages: the question each stage asks, what it found or will test, and what that forces next, with the research plan's add-ons and the October papers placed in the chain. |
| `research_plan` | The research plan's seven add-ons (BioLangFusion reproduction, encoder swaps, LucaOne, full-transcript tasks, the mRFP control, structure, distillation) and where each slots into the tiers and the Gantt chart. |
| `task_coverage` | Candidate Stage 4 datasets grouped by the question they answer, the regions, molecules, and design properties each provides, and the research plan's priority for each, including COMET's multi-molecule and cross-molecule tasks (built by `make_task_figure.py`). |
| `alignment_in_biology` | Conceptual figure for Stage 2: alignment in machine learning, the central dogma's one-way information loss, and what follows for alignment (built by `make_alignment_figure.py`). |
