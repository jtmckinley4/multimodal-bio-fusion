# Saved Stage 1 figures

These are deliberately shared figure bundles from captured completed results. Open the PNG or SVG files directly; no kernel restart or Python execution is needed to view them. The JSON snapshots preserve the numeric inputs needed for the figure cells, not an entire Jupyter session. Embedding weights and large embedding caches remain local.

| Bundle | Contents | Recorded result state |
| --- | --- | --- |
| [Stability](stability/) | Eight panels: raw agreement, codon-pair comparison, PCA bands, layerwise CKA, probes, composition controls, concatenation gains, and whole-embedding composition versus stability. | Local completed run captured October 7, 2026. Some controls and PCA values differ from earlier saved results; their cause remains unresolved. |
| [GTEx](gtex/) | Four panels: encoder probes, length/composition/group baselines, raw versus controlled agreement, and concatenation gains. | Local completed run captured October 7, 2026; 999 training-half transcripts and 996 separate test-half transcripts. Each panel labels which evaluation it uses. |

## Regenerate without rerunning models

From the repository root, using the project Python environment:

```shell
python Code/export_figures.py stability
python Code/export_figures.py gtex
```

The [exporter](../../Code/export_figures.py) executes only the tagged plotting cells in the selected notebook, in a separate Python namespace. It loads `results.json` from the selected bundle and writes SVG, 300-dpi PNG, CSV and `manifest.json` back there. It never connects to a kernel or executes the research sections. NumPy, pandas and Matplotlib are needed. Use `--output <folder>` to keep a new render separate, or `--source <results.json>` to explicitly select another compatible snapshot. Missing snapshots stop; the exporter does not compute replacements.

To display the recorded figures inside a notebook, run its Figure export settings cell and Figure exports section. Those cells read the same saved snapshot, so they also work without rerunning the analysis. They do not automatically adopt newer kernel variables. Both notebooks' original scientific outputs and execution counts are preserved.

## Read the evidence with the image

Each numbered image has a CSV containing the values plotted. `manifest.json` records source cells, the snapshot checksum, settings and interpretation limits; the standalone exporter also records plot-source and artifact checksums. `pair_metrics_all.csv` and `assignment_gains_all.csv` retain full pair tables instead of truncated displays. Some baseline and whole-composition values exist only as rounded retained tables; the figures and CSVs identify that precision. Retain each image, its data, manifest and result snapshot together.

The whole-embedding composition plot asks how much total embedding variance the fitted composition features explain. The PCA plot asks that question within selected component bands. Its fractions have different denominators. Neither panel directly measures causation. Pair-marker borders and centers identify the first and second modalities; each complete marker is one pair, not two independent model observations. Crowded all-pair plots may contain overlapping observations.

The [earlier-versus-local stability comparison](stability/earlier-saved-vs-local.csv) preserves observed disagreements. Matching an exported CSV proves faithful plotting, not reproduction of another machine's computation. The [run records](../run-records.md) distinguish these captures, earlier runs and outstanding provenance gaps. Replacing a snapshot is a deliberate result-record update, not a side effect of rendering or saving a notebook.

These files are inside the repository and are not ignored. They appear on GitHub after a contributor commits and pushes them; the export command performs neither operation.
