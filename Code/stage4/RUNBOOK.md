# Stage 4 runbook: who runs what, in what order

Use this runbook to coordinate Stage 4 execution: identify the work already saved, the inputs needed for the next run, and who is responsible for it. Created October 20; the status and resource tables below were checked against locally saved results and notebook outputs on **October 25, 2026**. They are a dated snapshot, not a live view of either machine.

## The split

The original assignment put the CPU gate and composition probes with Julian, and the large extractions and training sweeps with Chase. The October 22 run record subsequently documents Chase running the human half-life/TE and LucaOne analyses locally. The table distinguishes recorded runners from assignments for remaining work; it does not assign the unresolved mouse work to either contributor.

Coordinate before taking over an assigned run. Separate result folders reduce write conflicts, but notebooks, shared code and run records can still be edited by both contributors. Check the current files before starting or resuming work.

## The order

Task IDs refer to the `Gantt` and `Details` sheets in the [research plan](../../docs/Research_Project_Plan_Gantt.xlsx). A task can appear in more than one row because a completed gate, a completed human analysis and an unverified mouse analysis need different next actions. The workbook owns research priorities and scheduled windows; the dependencies here identify files or checks required to run the stated work. For example, LucaOne does not technically require the Phase B sweep to finish, even though the plan places it later.

**Complete** means that the stated scope has retained outputs consistent with its run record. It does not mean the whole Gantt task is finished or its scientific conclusions have been accepted. **No local completion evidence** does not establish that work was never attempted on another machine. Dates below come from run records or saved execution output, not file modification times. All rows were checked on October 25.

| Task ID | Work and scope | Notebook / section | Owner or recorded runner | Status / progress | Recorded run | Depends on | Output / evidence | Next action / blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | Phase A: the saved sweep on the available benchmarks, alongside the earlier Stage 3 fungal baseline | [Phase A](tier2_blf_phaseA.ipynb) | Julian — recorded runner | Complete for the saved 48-fit sweep; the five-benchmark task remains incomplete | Oct 19–20 | Prepared Stage 3 benchmark inputs, original token caches and fusion implementation | [48 fit records](results/phaseA/); [run record](../../docs/run-records.md#2026-10-20-stage4-blf-phasea-biolangfusion-phase-a-on-cov-vac-e-coli-and-mrna-stability-research-plan-item-1) | Reuse the completed cohort. Ab1 availability still blocks the fifth benchmark; this row does not establish a full multi-seed comparison on every dataset. |
| 18 | mRFP synonymous control: 12 encoder representations and composition/folding controls on the 1,000-row sample | [mRFP control](tier1_mrfp_control.ipynb) | Julian — recorded runner | Complete for the saved pilot control | Oct 19 | Stage 1 mRFP pooled embeddings with matching rows; sequence controls and ViennaRNA | [Three result CSVs](results/tier1_mrfp/); [run record](../../docs/run-records.md#2026-10-19-stage4-mrfp-control-the-mrfp-synonymous-control-research-plan-item-5) | Reuse these results for interpretation. No additional execution is required to recover this completed control. |
| 24 | Phase B: the configured 117-fit, seed-0 encoder-swap sweep | [Encoder swaps](tier2_encoder_swaps.ipynb) | Chase — assigned | Partial: **87/117** canonical fit records; 30 remain | Oct 25 partial snapshot; full sweep end not recorded | Prepared benchmark inputs; 39 identified original/swap token caches; compatible configuration and checkpoints | [Fit records](results/phaseB/); [partial-run record](../../docs/run-records.md#2026-10-25-encoder-swap-partial-results-and-gtex-reuse-compatibility) | On resumption, run the compatibility preflight and complete the missing fits before the reading cells. Saved output ends in an interrupted stability fit; this inventory does not establish live kernel status. |
| 42 | Half-life/translation-efficiency (TE) feasibility gate and composition baselines | [Half-life and TE](tier3_halflife_te.ipynb), Parts 1–2 | Chase — Oct 22 notebook run; initially assigned to Julian | Complete: six gate rows and 25 composition rows across five eligible datasets | Documented Oct 22 | Six downloaded dataset tables; no encoder extraction required | [Gate](results/halflife_te/gate.csv); [composition results](results/halflife_te/composition_baselines.csv); [run record](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) | Reuse the saved checks. The 203-row synthetic dataset is excluded from composition fitting by the 1,000-row minimum. The record does not identify the original producer of every reused artifact. |
| 41 | LucaOne on mRNA stability using the published split | [LucaOne / IsoFormer](tier2_lucaone_isoformer.ipynb), Parts 1–3 | Chase — recorded runner | Complete for stability: eight probe-score rows and three bootstrap contrasts | Oct 22 | Prepared stability records and splits; original Stage 3 stability token caches; two LucaOne pooled matrices | [Probe scores](results/lucaone/mrna_stability_probe_scores.csv); [contrasts](results/lucaone/mrna_stability_comparisons.csv); [run record](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) | Preserve and interpret the completed stability comparison. GTEx remains excluded under the selected 1,280-token input rule; it is not a completed LucaOne result. |
| 44 | IsoFormer-style GTEx repeat seeds 1 and 2: six variants per seed | [LucaOne / IsoFormer](tier2_lucaone_isoformer.ipynb), Part 4 | Chase — assigned, subject to coordination | No local completion evidence for the 12 repeat fits | No repeat-run date recorded | GTEx pilot records and three Stage 3 token caches; six budgeted seed-0 results for comparison | [Seed-0 record](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders); [reuse repair](../../docs/run-records.md#2026-10-25-encoder-swap-partial-results-and-gtex-reuse-compatibility). No local `results/gtex_isoformer/` folder; saved run switch is off. | Coordinate any repeat run. Before reporting the combined table, resolve the summary reader that groups distinct modality configurations under the same historical `IsoFormer` label. The reuse repair does not fix that separate reporting issue. |
| 42 | Human half-life and TE encoder probes | [Half-life and TE](tier3_halflife_te.ipynb), Part 3 | Chase — recorded runner | Complete for two human datasets: 14 representation-score rows and four fusion-gain rows | Oct 22; extraction finished 08:40:56 EDT | Completed gate, retained human records and ten pooled embedding matrices | [Probe scores](results/halflife_te/encoder_probes.csv); [fusion gains](results/halflife_te/fusion_gains.csv); [run record](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) | Interpret within the recorded limits: gene grouping, three composition assignments versus nine probe assignments, and aggregate CSVs without paired uncertainty. Completion does not include mouse replication. |
| 42 | Mouse half-life and TE encoder replication | [Half-life and TE](tier3_halflife_te.ipynb), Part 3 | Assignment to confirm | Completion unverified: mouse gate/composition results exist, but no local mouse embeddings or encoder-probe rows were found | No verified encoder-run date | Eligible mouse records, gate checks and identified mouse embeddings | [Gate](results/halflife_te/gate.csv) and [composition results](results/halflife_te/composition_baselines.csv) only; the [Oct 22 record](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) excludes mouse encoder replication | Locate any artifacts behind `Details` task 42's statement that the mouse sets are embedded, and reconcile that statement before duplicating extraction. The Gantt is unchanged. |

### Resources

Disk figures below are **existing local file sizes**, in decimal MB/GB, measured October 25. They are not additional free-space requirements or download totals. Shared input caches can serve several rows and should not be added repeatedly. A recorded device capacity is not a measured memory peak; unrecorded values remain unknown.

| Task / work | Execution machine | Runtime and basis | Disk and basis | RAM / GPU memory |
| --- | --- | --- | --- | --- |
| 17 — Phase A saved sweep | Julian's Mac; MPS with CPU fallback | **26.85 h observed** for the saved sweep. Earlier Stage 3 fungal work is outside this timer. | 48 result JSONs occupy **2.88 MB**; embeddings and weights are separate. Original token caches are shared with Phase B. | Peak RAM / GPU memory not recorded. |
| 18 — mRFP control | Julian's Mac; CPU | **About 15 min reported** in the run record; no retained confirming timer. No new encoder extraction. | **34.46 MB** of existing pooled inputs; **0.744 MB** of result CSVs. | Peak RAM unknown; no GPU extraction required. |
| 24 — Phase B | Chase's desktop; saved device is RTX 2060 SUPER / CUDA | Latest saved extraction pass: **30.7 min observed**, reusing 36 caches and extracting three. Full sweep runtime and remaining duration are unknown. | **52.29 GB** of replacement caches plus **34.15 GB** of shared original caches; **0.879 GB** of completed-fit weights, **2.73 MB** of fit JSONs and a **46.1 MB** checkpoint. Model downloads excluded. | GPU capacity **8 GB**; peak VRAM and RAM unrecorded. |
| 42 — gate and composition | CPU; documented Oct 22 local notebook run | Saved download/load timers sum to **253 s**; full gate/composition elapsed time unrecorded. | Six local parquet inputs occupy **97.04 MB**. | Peak RAM unknown; no encoder GPU work. |
| 41 — LucaOne stability | Chase's desktop; CUDA | Two saved extraction timers total **91.9 min** (63.1 + 28.8). This excludes surrounding preparation and probe work. | Two pooled matrices and their manifest occupy **490.07 MB**. The earlier **7 GB download** figure was an estimate, not a measured local total. | GPU capacity **8 GB**; peak VRAM and RAM unrecorded. |
| 44 — GTEx repeats | Assigned to Chase's CUDA desktop; no repeat execution verified | Repeat runtime **unmeasured**. The earlier one-hour CUDA allowance is an estimate. Stage 3's roughly 3.5 h Mac/MPS run included extraction and seed-0 fits, so it is not a repeat-run timing. | Three existing GTEx token caches occupy **3.735 GB** across 27 files. Additional repeat outputs are not measured. | Proposed device capacity **8 GB**; repeat RAM / VRAM peaks unknown. |
| 42 — human encoder probes | Chase's desktop; CUDA extraction | Ten saved extraction timers total **188.4 min**; full analysis elapsed time unrecorded. | Ten pooled caches occupy **255.59 MB**; dataset tables are counted in the gate row. | Peak VRAM and RAM unrecorded. |
| 42 — mouse encoder replication | Not verified | No verified local runtime. | No local mouse embedding total to report; downloaded tables are included in the gate row. | Not measured. |

## Phase B, step by step (Chase)

```bash
cd ~/.../Mina/Code/stage4          # wherever the repo lives on your machine
git pull
jupyter lab tier2_encoder_swaps.ipynb
```

1. Reload the saved notebook and restart its kernel to import the updated helpers. Run from the top. Paths resolve from the repository's `Code/` folder, so VS Code can start the kernel at the repository root or inside `Code/stage4/`.
2. Read the disk projection, which includes already cached swaps. The **cache preflight** then checks all required original and swapped streams without loading models: recorded input identities, extraction settings, shard headers, widths and ordered token counts. Historical records in the local ignored `Code/stage3/embeddings/recovery-records.json` also verify recorded shard SHA256 values. Missing or incompatible evidence stops before extraction; preserve the affected files.
3. Input identity is strict throughout this notebook. The 23 unidentified legacy caches and the 56 dependent fits and their weights were removed from active folders on October 25 at Chase's request. At that reset, 12 verified original caches, four verified swap caches and 16 verified fits were preserved, leaving 23 caches to extract and 101 fits remaining under the unchanged 117-fit plan. Those counts describe the [reset snapshot](../../docs/run-records.md#2026-10-25-encoder-swap-unverified-cache-reset). No legacy opt-in is enabled; unidentified or incompatible files must not be treated as reusable.

   **October 25 follow-up inventory.** The four benchmark folders now contain 87 canonical fit JSONs: 16 retained results and 71 additional results, leaving 30 of the configured 117 fits without a completed record. This is a dated file inventory; it does not establish completion of the sweep or independent numerical validation. The [partial-results and GTEx compatibility record](../../docs/run-records.md#2026-10-25-encoder-swap-partial-results-and-gtex-reuse-compatibility) distinguishes the saved progress, interrupted stability fit, and reuse repair.

4. The **extraction** cell opens complete caches without loading encoders and computes only missing shards of identified extractions. New inputs are bound before the first shard is written; token counts and finite values are checked before new shards are published. An interruption retains completed chunks. On Windows, shard and metadata publication retry sharing/lock errors up to ten times, with at most 11.1 seconds of waiting. A persistent lock still stops the cell; completed shards remain reusable, and an unpublished `.partial` shard is recomputed on resumption. Restart the kernel before the next attempt after updating the helpers. All caches remain local and ignored.
5. The **sweep** runs cheap heads first and cross-attention last. It reuses a completed JSON only when the benchmark, variant and full training configuration match, preserving saved weights. New checkpoints save optimizer, scheduler, random generators and batch order at epoch boundaries; an interruption can repeat up to ten epochs. Older checkpoints warn about any missing recovery state. Training and extraction are separate: restarting a kernel does not discard disk caches or completed fits.
6. Run the reading cells after the sweep. They need no pretrained encoder. Record the observed outcome in [run records](../../docs/run-records.md); the [October 25 recovery entry](../../docs/run-records.md#2026-10-25-encoder-swap-cache-and-checkpoint-recovery) distinguishes restored files and readiness checks from research fits.

mRNA stability is last and is limited to three swaps by default (`STABILITY_SWAPS`), for the
reason the notebook gives: mRNA-FM is what the mRFP control predicts should win the RNA slot,
CaLM is its control, and ESM-2 150M tests whether the protein slot was too small. Widen that
list only if the disk projection says the rest fit.

Single-modality fits load, pad and transfer only their selected token stream. The codon-grid single-modality variants also place only that stream on the grid, while retaining the original layer construction for initialization and checkpoint compatibility. Multimodal fits still use all three streams, and the initial cache validation is unchanged. This reduces unnecessary batch preparation without changing the experiment settings; runtime improvement has not been measured. An already-running kernel keeps its imported helpers: let that run continue and pick up updated helpers at a later kernel restart.

**What to watch.** If a fit runs out of GPU memory, preserve its files and inspect the failure. Changing `BATCH_SIZE` changes the training recipe and will now fail compatibility checks against completed results under the same tag. A different recipe needs a distinct result destination and its own run record; it must not overwrite the existing comparison.

## The gate and composition baselines

```bash
cd ~/Desktop/Masters/CapstoneCourse/Mina/Code/stage4
jupyter lab tier3_halflife_te.ipynb
```

Parts 1 and 2 already have retained results; opening the notebook does not require repeating their computation. Part 1 checks six parquet datasets for untranslated regions, recoverable coding frames and gene groups for splitting. Part 2 fits predictors from sequence-composition features to provide reference scores for the encoder probes. Neither step requires encoder extraction or a GPU. The measured local input sizes and available timers are in the resource table.

Use [gate.csv](results/halflife_te/gate.csv) and [composition_baselines.csv](results/halflife_te/composition_baselines.csv) to review eligibility and reference scores with the advisor. Keep the distinction between these completed checks and the human or mouse encoder work in Part 3.

## LucaOne (Chase)

The completed stability results are listed above. For a separately authorized extraction, `RUN_LUCAONE = True` enables the expensive path, whose timing check uses eight sequences to project extraction time. The saved diagnostic reports 1.57 billion parameters for the selected model, loaded in float16 on CUDA. Weight size alone does not establish total GPU memory use; the retained outputs do not report a measured peak.

It needs `transformers` 5.15.1, which the project already pins, and nothing else: the
checkpoint ships its own model and tokenizer code, pinned to a commit in
`shared_code/lucaone.py`. The model card's instruction to `pip install lucaone` is not needed.

## Rules that keep us out of each other's way

1. **One owner per notebook per night.** If you want to run someone else's, message first.
2. **Never edit a result file by hand.** Every number in a table comes from a JSON or CSV a
   run wrote.
3. **A run record per run**, appended to `docs/run-records.md` without rewriting another contributor's entry. Coordinate concurrent edits; appending reduces overlap but does not eliminate merge conflicts.
4. **Push when a notebook finishes**, not at the end of the week. A sweep that is still
   running is fine to leave unpushed; a finished one is not.
5. **Embeddings and checkpoints stay out of Git.** They are ignored already; do not force-add
   them. The result JSON beside a checkpoint is the record.
6. **If you change shared code, say so in the message.** A change to `shared_code/fusion/train.py` can affect later execution of notebooks that import it. Already-saved results remain unchanged; retain their implementation provenance rather than assuming they were produced by the current code.

## Remaining dependencies and deferred work

* **Ab1**, BioLangFusion's fifth benchmark, is not in the public CodonBERT release. Dataset access remains a dependency to resolve with the advisor; the available-benchmark work does not complete the five-benchmark comparison.
* **Mouse half-life and TE encoder replication** remains unverified locally. The human probes are complete. Reconcile the mouse artifact evidence described in the table before assigning another extraction.
* **DNABERT-2 as a Phase B swap.** Its byte-pair tokens cover a variable number of
  nucleotides, so the codon grid needs an adapter designed once and held fixed. It is written
  up in the Overleaf as an open item rather than quietly skipped.
* **Coupled Mamba and the project's own cross-attention design** remain governed by their entries in the research plan. This execution-status update does not change their scheduling or authorize new comparisons.
