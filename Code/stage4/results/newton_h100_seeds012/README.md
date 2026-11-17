# Newton encoder swaps: seeds 0, 1 and 2

This result set contains **504 completed H100 fits** across four benchmarks and three training seeds. The four completed three-seed reports compare fusion with selected single encoders and same-input concatenation. Read the [findings](../../README.md#newton-three-seed-findings) or open [encoder swaps](../../encoder_swaps.ipynb) in its default saved-results mode. These results do not include the separate grouped-stability or new Mina experiments.

## Where the evidence lives

| Location | Role |
| --- | --- |
| [outputs/phaseA/](outputs/phaseA/) | Fresh original-encoder fits. |
| [outputs/phaseB/](outputs/phaseB/) | Fresh replacement fits and retained producer reports. |
| [outputs/logs/](outputs/logs/) and [outputs/notebooks/](outputs/notebooks/) | Launch/environment records, logs and executed notebooks. |
| [local_analyses/paired-report-v1/](local_analyses/paired-report-v1/) | Separately calculated three-seed reports from the saved Newton predictions. |
| [provenance/source-snapshot.json](provenance/source-snapshot.json) | Unchanged original preparation manifest: recorded paths, hashes and selection membership. |
| [provenance/execution-source/](provenance/execution-source/) | Archived notebook, helpers and launcher/support files, retaining their original relative paths. These are producer evidence, not maintained entry points. |

The manifest and launch records retain their original paths and bytes. Selection/report readers resolve the recorded result paths to retained files. For execution-source entries, the recorded relative path is now beneath `provenance/execution-source/`. Some execution sources changed after initial preparation, so launch-record hashes identify the actual runtime source; the initial snapshot alone does not identify every later version.

## Reuse canonical inputs and historical selections

The four input CSVs remain under [datasets/](../../../../datasets/). The 256 duplicate historical selection JSONs resolve to their existing repository result files. Only paths listed in the original snapshot, with matching recorded byte hashes, are eligible; other historical files cannot silently enter selection. Historical seed-0 validation choices remain separate from the fresh Newton scores. The [deduplication record](../../../../docs/run-records.md#2026-11-17-newton-evidence-deduplication-and-git-byte-preservation) documents the verification.

Use an ordinary Git checkout when verifying these bytes. The repository's [attributes](../../../../.gitattributes) preserve declared CRLF/LF forms; `git archive` exports index bytes without applying checkout line endings. An archive export must restore the declared forms before byte verification. Do not normalize hashes or rewrite producer manifests to make a mismatch pass.

This compact evidence set contains no embedding caches and is not a training workspace. The original sibling `ARCC Execution/` and future [preparation contract](../../../../execution/arcc/encoder_swaps/README.md) remain unchanged. The [runbook](../../RUNBOOK.md#select-a-returned-result-set) owns result selection and separately enabled analysis; reading this package does not authorize new calculation or fitting.
