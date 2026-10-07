# Run records

Use this record to connect a shared result to the code, inputs, settings, and environment that produced it. Team members can run on different computers; hardware belongs to an individual execution. The [README](../README.md#setup) describes how to prepare and run the project.

## Record a run or rerun

Append an entry when producing results for comparison, a meeting, or the manuscript, including relevant failed or interrupted attempts. Give each execution a distinct ID, such as its UTC start time plus an experiment label. Identify which sections actually ran; a partial rerun does not refresh every saved notebook output. Link the earlier entry when comparing reruns, and preserve earlier observations when the interpretation changes.

Use the template below and link existing configuration or environment exports instead of copying long inventories. Record unknown details as `unknown`; do not substitute today's checkout or computer for an older result's producer. Use repository-relative evidence links, with a retained version or checksum for files that can be overwritten. Follow the [generated-file policy](../README.md#generated-files) when preserving outputs.

```markdown
### <run ID> — <purpose>

- Execution: start/end with timezone; runner; completed, partial, failed, or interrupted.
- Scope: notebook and sections/cells executed; earlier run compared and intended change.
- Code: producing Git commit; uncommitted changes and a retained patch/snapshot if present.
- Inputs: dataset version/path and hash; selected rows/order, filtering and split identity.
- Configuration: encoder/checkpoint revisions actually resolved; tokenization/pooling and analysis settings; sampling, model and fold seeds, including unset values.
- Environment: OS, Python and relevant package versions or an environment export; selected backend and hardware model; applicable precision, fallback and deterministic settings.
- Reuse: which encoder/stage outputs were computed or reused; cache/artifact identity and known producing run/backend. Keep the consuming environment separate.
- Evidence: versioned output links/checksums, relevant metrics and warnings/errors.
- Interpretation: comparison with the earlier run, unresolved differences and next check.
```

This is a manually maintained record that a researcher or an assistant can update during authorized work. The notebooks do not automatically populate it. Matching seeds or a device name alone does not establish identical results across platforms or releases; see [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## What the current cache records

The current [`embed_with_cache` format](methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse) records a cache/calculation version, checkpoint, requested revision, token limit, effective-input fingerprint, row count, and the selected device when a matrix is computed. A cache hit does not rewrite that device. A new session's selected backend therefore cannot identify where reused embeddings were produced; fallback may also execute some operations on a different backend. This format does not record code/package versions, execution time, seeds, or label/split identity. Older artifacts may lack even these cache fields.

## Legacy observations without complete run identities

These entries identify existing evidence, not reconstructed executions. They were located on October 5, 2026; that is not their execution date.

- **Retained Stage 1 outputs:** The [October 6 runs](#2026-10-06-stage1-full-runs-complete-runs-of-both-stage-1-notebooks) identify the earlier saved outputs. The user saved completed local outputs on October 7 before figure cells were reapplied; the October 7 capture entry below identifies that separate result state.
- **HyenaDNA backend comparison:** The [stability loading notes](../Code/Stage1_stability.ipynb#Interpreting-the-loading-messages), cell `9bdd4bf8`, preserve the reported CPU/MPS comparison. Its original execution date, hardware models, and complete producing environment are not established here. Keep that observation attached to its evidence rather than treating it as a requirement or a general backend guarantee.
- **Import compatibility report:** The README at commit `81317de433938dc41ad97987350559ae27f647b9` reported `multimolecule` 0.2.1 importing with `transformers` 5.14.1 and 5.15.1, and failing with 5.16 or later. This historical report motivated the current pin; its original logs, execution date, and full environment are not linked. It does not establish compatibility for every future release or diagnose another contributor's import failure.

## Recorded runs

### 2026-10-06 stage1-full-runs: complete runs of both Stage 1 notebooks

- Execution: 2026-10-06, Julian McKinley's Mac. `Code/Stage1_stability.ipynb` started about 14:57 and `Code/Stage1_gtex.ipynb` about 17:20 local time (EDT), going by the earliest logged warning in each; each completed top to bottom in one kernel session (execution counts 1 to 66 and 1 to 29).
- Scope: every cell of both notebooks. The stability dataset-audit cell was afterwards replaced by the audit-ceiling recomputation below.
- Code: producing commit unknown; the saved notebooks are committed in `a038413`.
- Inputs: `datasets/mRNA_Stability.csv` (hash in the audit-ceiling entry) and `datasets/GTEx_pilot.csv` (hash unknown); samples and splits as set in each notebook's settings cells.
- Configuration: `SEED = 42`, `PROBE_SEED = 42`, ten further fold assignments; encoder checkpoints and requested revisions as listed in each notebook's encoder table (resolved revisions of the six `latest` checkpoints unknown).
- Environment: Python 3.12.7 (Anaconda `base` kernel); PyTorch MPS backend on an Apple GPU; package versions unknown.
- Reuse: the main stability and GTEx embeddings were loaded from the `Code/stage1_embeddings/` caches; the layer-wise states, attention maps, synonymous-recoding control embeddings, and published-split embeddings were computed in the run, as their progress output shows.
- Evidence: the saved outputs at `a038413`, which truncate the 66-pair tables. On 2026-10-06 a read-only recomputation of those tables from the same cached embeddings (Linux, CPU, scikit-learn 1.9.1) matched every pair value the saved outputs show, for geometry and concatenation gains on both datasets, and every group range the overview reports. Against the Overleaf per-pair tables it differed only in GTEx Recall@1 for RNA-FM with ESM-2 35M (0.097 against 0.100, one held-out query) and one mutual k-NN value at a rounding boundary.
- Interpretation: these are the outputs the overview, README, notes, slides, and literature review report.

### 2026-10-06T21:20Z audit-ceiling: label-variance ceiling in the stability dataset audit

- Execution: 2026-10-06, about 21:20 UTC; Claude (assistant) in a Linux cloud workspace; completed.
- Scope: only the dataset-audit cell of [`Code/Stage1_stability.ipynb`](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) (cell `7f9bf305`, `datasets.audit("mrna_stability")`), recomputed outside Jupyter and saved into that cell; its saved execution count is unchanged. No other notebook cell was run. The intended change adds the label variance, the pooled label variance within a sequence, and the sequence-only ceiling on explained label variance to the audit.
- Code: commit `0b96d44` plus an uncommitted change to `Code/shared_code/datasets.py` (`audit`) and its expected values in `tests/test_validation.py`.
- Inputs: `datasets/mRNA_Stability.csv`, SHA-256 `d922e7d4b07516949d316856373031136b38e9b92a06e81bb4afbafd0c00d608`; all 65,356 rows.
- Configuration: no encoders, sampling, or folds; sequences stripped and uppercased as in `audit`.
- Environment: Python 3.13.16, pandas 3.0.5, NumPy 2.5.3, Linux x86_64; CPU only.
- Reuse: none.
- Evidence: label variance 1.009355; pooled label variance within a sequence 0.418495; ceiling 0.585383; the earlier audit rows are unchanged (median label SD within a repeated sequence 0.487024). `tests.test_validation` and `tests.test_dataset_paths` pass with the updated audit.
- Interpretation: a model that sees only the sequence can explain at most about 59% of label variance in the full file, assuming repeated sequences show the label noise of every sequence.

### 2026-10-07 stability-figure-export: captured kernel results, plotting only

- Execution: captured at 2026-10-07T12:54:39.213341+00:00; assistant read existing result objects from the running local stability kernel and executed only the new plot cells in a separate Python process. No kernel restart, encoder loading, download, experiment, or research-function rerun. New notebook cells have no saved execution counts or outputs in the notebook UI.
- Code and input identity: checkout `662b89fee8f0c7efb09fabf5194a77b7f278e8cd` with pre-existing notebook changes; pre-edit notebook SHA-256 `7090891364cf16918198274efce3b7beb03128b5749fadc118a159606381cbd1`. The retained input-history text matches 64 of 66 notebook code cells, including every figure-producing analysis cell. Imports and the encoder smoke check differ. History text does not identify the producer of reused embedding caches.
- Scope and settings: 981 retained rows (964 distinct sequences), 690 CCA training and 291 test rows; 194 layer-sample rows. Seeds 42 for sampling and pinned probes; ten further assignments use seeds 0–9; PCA top block has ten components. Captured numeric evidence includes all 66 pairs, 660 assignment gains, all layer matrices, and complete probe/PCA tables. Sequence-order SHA-256 `f31e012c56d792d9d71ceefe65614ee115ea77b9eb1ceb6c3ce429e65df7657f`.
- Environment: captured kernel reports NumPy 2.3.5, pandas 2.3.3 and scikit-learn 1.7.2. Its original calculation time and embedding-producing environment remain unknown. The local export manifest records plotting-package versions and output hashes; these are separate from the earlier saved Mac-run outputs above.
- Evidence: seven SVG/PNG figures plus CSVs, `captured-results.json`, `kernel-history-check.json`, `saved_vs_captured_differences.csv`, and `manifest.json` are retained in the requesting contributor's `notebook-figures-v3` export bundle outside Git. All 137 original notebook cells, outputs, counts and metadata were preserved. The default portable export location and override are documented in the [generated-file inventory](../README.md#generated-files).
- Unresolved comparison: controlled CKA/retrieval and parts of the PCA results differ from saved outputs. The captured control values agree with retained `Out[42]` and `Out[46]`; PCA values agree with `Out[44]`. No later overwrite was found, and loaded residualization/PCA function bytecode, constants and names match the current helper file. The numerical cause remains unresolved. All twelve whole-embedding composition fractions differ; stability probe scores agree within saved rounding. Do not transfer earlier composition correlations or mix saved and captured controlled values. GTEx was not inspected or changed.
- Interpretation limits: residualization and PCA-band fits use all rows before later evaluation; assignment ranges are not confidence intervals. These exports retain a result state and do not validate a scientific conclusion or declare a research stage complete.


### 2026-10-07 shared-figure-snapshots: local completed stability and GTEx results

- Execution: the user completed and saved both notebooks in VS Code, then authorized figure exports. Existing stability objects were captured at 2026-10-07T14:15:23.831469+00:00; GTEx objects at 2026-10-07T14:12:12.927934+00:00. Original scientific calculation start/end times are not reconstructed. Only result serialization, plotting and bounded composition-design numerical diagnostics were performed by this task; no encoder inference, probe fit, whole-notebook rerun or kernel restart.
- Preserved notebook evidence: all 137 user-saved stability cells and all 71 user-saved GTEx cells, including outputs, execution counts and metadata, were retained when adding figure sections. The new plotting cells remain unexecuted in the saved notebook UI. GTEx's 29 submitted code cells match retained input history; stability matches 64 of 66, with the import and loading-check source mismatches already recorded above.
- Result bundles: [stability](figures/stability/) and [GTEx](figures/gtex/) contain `results.json`, CSVs, SVG/PNG and `manifest.json`. The [export command](figures/README.md#regenerate-without-rerunning-models) regenerates these captured results without restoring a kernel or rerunning experiments. The snapshot hash, plotting-cell identity and artifact hashes identify what was rendered; they do not prove another run used identical scientific inputs.
- GTEx scope: 999 training-half transcripts from 951 genes; 996 separate test-half transcripts from 558 genes; zero genes shared between halves; 30 tissue labels. Retrieval uses 680 fit and 319 test rows within the training half. Those 319 rows are not the published 996-row test half. Baseline and group panels compare the same pinned-fold columns; retained length/composition baselines have only three-decimal precision. Full pair geometry and 660 assignment gains are preserved.
- Reuse and environment: all 72 local encoder caches inspected across six folders had the current cache version, matching requested checkpoint/revision/token-limit metadata, and `device=cuda`; this header check did not reconstruct their input fingerprints or prove their original execution environments. Captured local Python is 3.13.9, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2. [Snapshot metadata](figures/gtex/results.json) records remaining reported versions. NumPy/scikit-learn controls use CPU numerical libraries even when encoder inference used a GPU.
- Earlier-run disagreement: [earlier saved versus local stability values](figures/stability/earlier-saved-vs-local.csv) differ beyond display rounding for controls and some PCA quantities. For NT 500M human-ref / RNA-FM, controlled CKA is 0.101 earlier versus 0.196161 locally; controlled Recall@1 is 0.107 versus 0.257732. The local capture agrees with retained kernel outputs; export did not introduce the difference. `analysis.py` and `sequences.py` are identical between the commit containing the October 6 outputs (`a0384134112ced29d89ed5500a4994e62900c068`) and current files. That commit does not prove which code was loaded to produce the older outputs.
- Bounded diagnosis: the [composition-design diagnostic](figures/composition-design-diagnostics.json) checked existing local feature matrices, not embedding or prediction recomputation. Standardized design ranks are 84/88 for stability and 68/93 for GTEx, unchanged across tested relative cutoffs from 1e-15 through 1e-6. There is a large gap at the current default cutoff, weakening a threshold-flip explanation for these local matrices. The older design and embedding arrays are unavailable for direct comparison; hardware, dependency and input differences remain unisolated. No scientific solver, rank threshold or precision was changed.
- Interpretation and next discriminant: compare the older run's exact input order, composition arrays and embedding arrays before changing methods or attributing the difference to hardware. Record actual resolved model revisions, dependency versions, BLAS/LAPACK backend, dtypes, solver settings and convergence per pair on future runs. Pinning seeds alone does not guarantee cross-platform equality; see [numerical reproducibility sources](sources.md#numerical-reproducibility-across-computers). The current figure snapshots solve result retention and plot regeneration; they do not claim the cross-machine discrepancy is resolved.
