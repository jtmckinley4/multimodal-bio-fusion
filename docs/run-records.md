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

- **Retained Stage 1 outputs:** The separated notebooks no longer carry outputs from the combined notebook; their saved outputs come from the [October 6 runs](#2026-10-06-stage1-full-runs-complete-runs-of-both-stage-1-notebooks).
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
