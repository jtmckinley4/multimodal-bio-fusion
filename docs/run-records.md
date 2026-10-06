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

- **Retained Stage 1 outputs:** The [overview](stage1-overview.md#summary) identifies outputs inherited from the combined notebook. Exact producing runs, code states, and full environments remain unassigned. Do not group all saved cells into one execution based on their current location.
- **HyenaDNA backend comparison:** The [stability loading notes](../Code/Stage1_stability.ipynb#Interpreting-the-loading-messages), cell `9bdd4bf8`, preserve the reported CPU/MPS comparison. Its original execution date, hardware models, and complete producing environment are not established here. Keep that observation attached to its evidence rather than treating it as a requirement or a general backend guarantee.
- **Import compatibility report:** The README at commit `81317de433938dc41ad97987350559ae27f647b9` reported `multimolecule` 0.2.1 importing with `transformers` 5.14.1 and 5.15.1, and failing with 5.16 or later. This historical report motivated the current pin; its original logs, execution date, and full environment are not linked. It does not establish compatibility for every future release or diagnose another contributor's import failure.
