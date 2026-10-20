# Stage 3: build the fusion models

Each notebook here builds one model, or the data it needs, and checks that it works. The model code itself lives in [shared_code/fusion/](../shared_code/fusion/) so that the Stage 4 experiments import exactly what was checked here. Experiments with results belong in [stage4/](../stage4/README.md).

Two rounds, following the research plan (Overleaf Section 11):

**Round 1, now: the pipeline and the research plan's baselines**

| Notebook | What it builds and how it is checked |
| --- | --- |
| [01_blf_datasets.ipynb](01_blf_datasets.ipynb) **done** | Audits four supplied BioLangFusion benchmarks and their split labels; Ab1 is not in the inspected CodonBERT release. A 999-nucleotide row filter matches all four published "used" counts, but does not establish the authors' preprocessing or row identity. Records four Phase A deviations, including about 68% exact train/test sequence overlap among the retained stability test rows. |
| [02_token_embeddings.ipynb](02_token_embeddings.ipynb) **done** | Per-token embeddings for BioLangFusion's encoders (NT v2 100M, RNA-FM, ESM-2 8M) into a resumable shard cache under `embeddings/`, about 34 GB in float16 across the four benchmarks. Ran on the Mac 2026-10-16 in 53 minutes: 34.2 GB, 405 shards, every row verified ([run record](../../docs/run-records.md#2026-10-16-stage3-token-embeddings-token-level-caches-for-the-four-biolangfusion-benchmarks)). Implements `grid_protein` with one position per triplet and stops represented as `X`; this records an input choice without resolving its biological-frame or author-policy qualification. |
| [03_blf_build_check.ipynb](03_blf_build_check.ipynb) **completed outputs; comparison review pending** | Codon grid, three fusion heads, and the shared TextCNN head checked on fungal expression. The local walkthrough retains 15 Windows fits: nine appendix and six equation configurations, including the entropy-weight sweep, at seed 0 and 100 channels. Earlier and incoming fungal records are also retained, with different provenance; they must not be pooled by directory glob. The paper's Table 1 and Table 4 remain the comparison targets. Its reported best single and gated scores are 0.804 and 0.824. See the [original build report](../../docs/run-records.md#2026-10-17-stage3-blf-build-check-biolangfusion-reproduced-on-fungal-expression) and [preserved result sets](../../docs/run-records.md#2026-10-19-stage3-integration-preservation-of-completed-results-no-research-execution). The notebook's comparison directory and convolution interpretation still need review. |
| [04_lucaone_isoformer_check.ipynb](04_lucaone_isoformer_check.ipynb) **contributor outputs retained; training integration pending** | LucaOne input-limit measurements and a six-way IsoFormer-style modality ablation on a 2,000-transcript GTEx pilot. The reported 1,280-token LucaOne limit excludes the unchanged 6,000-nt DNA windows and admits about 40% of transcripts; alternative input policies would require their own protocol. The IsoFormer experiment uses frozen encoders and a pilot dataset, so it is not an exact end-to-end reproduction. Its custom GTEx loaders need execution identity and sampler recovery before the merged checkpointed trainer can run them. The notebook contains active extraction, GPU-probe, and training cells; do not use Run All merely to inspect its saved outputs. |

**Round 2, January: the project's designs**

| Notebook (planned) | What it builds |
| --- | --- |
| `05_coupled_mamba_check.ipynb` | Coupled Mamba as a fourth head in the same pipeline |
| `06_cross_attention_check.ipynb` | IsoFormer-style cross-attention with the encoders Stage 1 points to |

Token-level embeddings are large; keep them out of Git. Training results are small JSON
files and are tracked; the checkpoints beside them are not.

**Why fungal expression is the build check.** It provides a regression task with published values for the three single encoders and three fusion heads, and the saved audit detects no exact sequence crossing its retained split boundaries. This makes it useful for inspecting the build, without establishing independent biological samples or eliminating project preprocessing choices. The [target and evaluation guide](../../docs/methods/biological-targets-and-controls.md#fungal-and-e-coli-benchmark-targets) explains the RNA-seq source context, unresolved local label transformation, length filter and input-policy qualifications.

The other benchmarks test different aspects of the pipeline. In the saved stability audit, about 68% of retained test rows reuse a training sequence, often with a different label; this limits an unseen-sequence interpretation but does not quantify memorization or make every matching score evidence of leakage. CoV-Vac's 81-nucleotide fragments require a protein-input policy whose biological correspondence remains unresolved. E. coli tests the classification path against supplied class IDs, with its assay meaning and missing semantic label mapping kept distinct from the loss-function check.
