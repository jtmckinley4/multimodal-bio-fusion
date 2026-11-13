# Stage 4: comparative and explanatory experiments

Stage 4 asks when combining representations improves biological prediction and what accounts for that improvement. [Stage 1](../stage1/README.md) characterizes representations and controls; [Stage 3](../stage3/README.md) establishes reference methods and baselines, including their validation comparisons and repeats. Here we use those references to investigate encoder choices, sequence context and fusion behavior.

The retained prediction experiments use **frozen encoders**: pretrained weights stay fixed while a predictor, and sometimes fusion layers, learns from their outputs. A baseline is the complete prediction setup used as a reference, including its trained head and evaluation. Comparative results and proposed explanatory interventions belong to the same investigation, but an available implementation, an executed result and an unrun proposal remain different evidence.

## How the experiments connect

Three questions organize the investigation:

- **Representation usefulness:** what can a predictor recover from one encoder?
- **Predictive complementarity:** does combining representations improve on the relevant strong single-encoder predictor?
- **Fusion-method value:** does an interaction such as attention improve on a simpler combination of the same inputs?

A stronger replacement can improve a fused model without establishing complementarity. Likewise, successful concatenation does not establish a benefit from attention or gating. A controlled comparison that finds no gain is still useful evidence; each baseline tests a particular claim.

| Investigation | Question and current role |
| --- | --- |
| [Encoder swaps](encoder_swaps.ipynb) | Does replacing one encoder improve the representation, the combination, or the value of a fusion method? Existing comparisons; repeat coverage varies. |
| [Half-life and translation efficiency](halflife_translation_efficiency.ipynb) | Does extra RNA context help, and does protein add value once that context is available? Existing pooled comparisons and separately enabled matched follow-ups. |
| [Auxiliary correction, gates and attention](#proposed-tests-of-fusion-explanations) | Which controlled change could explain or improve the contribution of another representation? Proposed experiments, with no implemented notebooks or completed runs. |
| [Transcript-region correspondence](#does-respecting-transcript-regions-improve-prediction) | Does mapping protein features only to annotated coding regions help beyond simpler controls? Proposed token-level experiments. |
| [Molecular-partner tasks](#conditional-outlook-transfer-to-other-molecular-relationships) | Do selected explanations transfer to RNA–protein or protein–ligand pairs? Conditional dataset and evaluation work. |

Encoders reading a common coding sequence can expose different predictive patterns without adding an independently measured molecule. Genomic flanks and untranslated regions can instead supply sequence context absent from a translated protein. Neither guarantees a fusion gain. Cross-notebook differences can also change datasets, targets, predictors or evaluation, so they cannot isolate one biological cause.

The BLF benchmark and appendix comparisons, frozen GTEx reference, GTEx audit and adapted Coupled Mamba now belong to [Stage 3](../stage3/README.md). The mRFP control and LucaOne ridge comparison belong to [Stage 1](../stage1/README.md). Their original result directories and dated run identities are retained.

## Read completed results from the top

Saved notebook outputs can be read without running cells. For a reading-only Run All of the retained experiment notebooks, set `READ_SAVED_RESULTS = True` and leave separate follow-up switches off. This reads available results and refreshes displays; missing or incompatible evidence is reported rather than computed. The [reading instructions](RUNBOOK.md#read-saved-results-without-starting-work) also cover the relocated reference and representation notebooks.

Reading historical records does not recover unknown producer settings or establish that current code reproduces them. For computation, use the [runbook](RUNBOOK.md): encoder swaps have a missing-work continuation route, but setting another notebook's reading flag to false does not guarantee that only missing work will run. Cache recovery, checkpoint identity and repair history remain in the [repair record](RUNBOOK.md#concrete-repair-list).

## The current experiments

### Phase B: changing one encoder at a time

The [swap notebook](encoder_swaps.ipynb) replaces one encoder while retaining the other two and the surrounding BLF comparison recipe. It distinguishes replacement fusion versus original fusion, fusion versus the validation-selected single, and attention/gating versus same-input concatenation. The [comparison section](encoder_swaps.ipynb#Report-representation,-complementarity-and-method-differences) presents these separately.

Swaps inherit the gated-fusion penalty selected for the original encoders. They therefore test substitutions under that recipe, not the best performance each replacement could achieve after individual tuning. Coverage tables identify available evidence by seed; missing repeats must not become an incomplete three-seed average. [Replacement settings and continuation](RUNBOOK.md#complete-the-encoder-swaps-one-seed-at-a-time).

#### RNA-FM and mRNA-FM: connecting pretraining to the fusion question

The [worked encoder comparison](encoder_swaps.ipynb#RNA-FM-and-mRNA-FM-as-a-worked-encoder-comparison) examines pretraining, supported capabilities, our results and competing explanations. RNA-FM's noncoding-RNA/nucleotide-token training differs from mRNA-FM's coding-sequence/codon-token training. Corpus, tokenization, capacity and predictor input dimensions change together, so the swap cannot isolate a pretraining mechanism.

On fungal expression, the recorded concatenation score rises from about **0.793 to 0.853 Spearman**, while **mRNA-FM alone scores 0.852**. This supports a better RNA encoder; the [paired comparison](encoder_swaps.ipynb#3.-What-the-saved-RNA-FM-and-mRNA-FM-results-show) leaves additional fusion benefit unresolved. The recorded single-encoder ordering favors mRNA-FM on fungal expression, E. coli and stability, but reverses on CoV-Vac. RNA abundance and chemical degradation are different targets, so this is not a universal encoder ranking.

Together with mRFP, these observations motivate asking what another representation adds beyond a strong task-specific predictor. They do not establish which biological features caused the scores, that other streams contain no useful information, or that one training seed establishes reliability.

### Full-transcript half-life and translation efficiency

The [half-life/TE notebook](halflife_translation_efficiency.ipynb) introduces transcript regions absent from the earlier coding-sequence inputs. The 5′ untranslated region can affect translation initiation, while the 3′ untranslated region can contain sites relevant to RNA persistence. Both are absent from the translated protein. This motivates two questions: does extra RNA context improve prediction, and does adding protein help once that context is available?

These are different datasets and target constructions from the earlier stability benchmark. Human half-life is a **consensus stability score**, not hours. Human translation efficiency is an **RNA-abundance-adjusted ribosome-occupancy score**, not direct protein output or a raw ribosome/RNA ratio. The [target guide](../../docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explains their processing and examples.

The retained human encoder comparisons use pooled frozen embeddings and ridge, not reconstructed Saluki, RiboNN or BLF heads. Long transcripts are encoded in separate windows; pooling their outputs does not model cross-window interactions. The notebook retains the exact region conditions, suitability checks and partition procedure. Inspection of additional tables does not establish completed encoder comparisons on them.

#### Human findings and the limits of their scope

The [historical human results](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) report mean Pearson improvements of about **0.053 for half-life** and **0.030 for TE** when joining full-transcript and protein representations, relative to a single selected by mean test performance. These are descriptive correlation gains, not percentage increases in lifetime or protein output. The reported standard errors are not paired confidence intervals for the gains.

The repeated seeds vary held-out partitions; they are not neural-initialization repeats. Gene-grouped splits do not establish independence between related gene families, and human evidence does not establish mouse replication. The [matched follow-up](halflife_translation_efficiency.ipynb#Part-4:-test-complementarity-with-matched-evaluation) uses separately identified comparisons rather than silently replacing these historical scores.

### Focused comparisons using completed inputs

Three follow-ups make the comparison questions explicit: [swaps](encoder_swaps.ipynb) compare saved predictions against original, single-encoder and simple-fusion references; [LucaOne](../stage1/lucaone_representation_comparison.ipynb) compares the combined vector with each constituent; [half-life/TE](halflife_translation_efficiency.ipynb) compares RNA, protein, concatenation and composition on common actual partitions.

The latter two use validation-selected ridge penalties and retained test predictions, separately from the historical internal-`RidgeCV` evaluations. Their paired group-resampled intervals condition on the fitted models; they do not repair split overlap or establish family-level generalization. The notebooks report available evidence, while the [focused run instructions](RUNBOOK.md#run-focused-reports-and-cpu-probes) own switches, reuse rules and destinations.

## Reliability and the next comparison

Selected gains need independent training seeds, with training variation distinguished from paired test-group bootstrap intervals. A separate stability evaluation groups identical sequences before train/validation/test assignment, considering near-identical groups where feasible. It needs new predictor fits and a separate experiment identity; compatible frozen caches may be reused. Preserve supplied-split results: an unseen test subset or grouped bootstrap does not remove training overlap.

Unfinished reference repeats remain with their [Stage 3 methods](../stage3/README.md); swap repeats and matched half-life/TE comparisons remain with their experiments here. A new question need not wait for every notebook to finish, but it needs the evidence and evaluation relevant to that question. Each new experiment includes matched baselines, training seeds and uncertainty analysis. The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx) retain tasks 17/24/28 for reference comparisons, swaps and reliability, and 42 for transcript-region work.

## Proposed tests of fusion explanations

[Mina's revised protocol](../../docs/Biological_Fusion_Plan.docx) supplies the proposed comparisons below. They develop four questions from that wider plan and retain conditional partner-task extensions. **These are proposals, not completed experiments:** their notebooks have not been implemented or run. The starting point is frozen encoders with trainable prediction and fusion components.

### Can auxiliary features correct the anchor's errors?

The **anchor** is the strongest eligible single-encoder predictor selected using validation, including its trained head. The comparison asks whether auxiliary representations improve on that complete predictor under matched evaluation, making useful information accessible after the strong individual predictor has been fitted.

Fit auxiliary representations to the anchor's regression errors and add the predicted correction to its output; classification instead uses additive logit corrections. Obtain training errors from **group-aware out-of-fold predictions**, so each example's group was excluded from the head that predicted it. Validation selects the correction; final test data stay outside selection.

Compare auxiliaries separately and jointly, with shuffled-input and parameter-matched random-feature controls. The [mRFP analysis](../stage1/mrfp_representation_control.ipynb) offers a related precedent starting from calculated controls, not an encoder anchor. Success supports accessible complementarity; failure need not mean that no useful information exists.

[Liang et al.](../../docs/sources.md#liang-et-al-quantifying-and-modeling-multimodal-interactions) provide prior work on task-related redundancy, unique information and synergy, including model selection. Our proposed correction test measures predictive improvement; it does not implement their information decomposition. Connecting representation relationships to fusion choices therefore needs a specific biological contribution beyond that existing idea.

### Do gates preserve useful contributions?

If an auxiliary representation helps under a controlled comparison, a gate should not be assumed to retain that contribution. The existing BLF gate assigns one weight per modality per sequence. Compare its softmax competition with and without entropy regularization, independent sigmoid gates, and mean-plus-max summaries. Record per-example weights alongside prediction: a small weight alone does not establish harmful suppression or biological importance. Control feature-scale changes when comparing gate normalizations.

The contribution may also vary along a sequence. Compare sequence-wide gates with position-wise weights and a comparable-capacity positional model without gates. Keep encoders, alignment and the downstream predictor fixed. The [token-cache route](../stage3/blf_token_embeddings.ipynb) supplies a starting point; equal widths do not establish correspondence, and codon pooling changes information. CoV-Vac's translated fragment has no established biological reading frame. Gate patterns can motivate perturbations, not establish local biological mechanisms.

### Should attention retrieve auxiliary information while retaining the anchor?

Let anchor positions query auxiliary streams, potentially of different lengths, then add gated updates through a skip connection. Compare anchor-only, concatenation, ordinary attention, anchor-skip attention and sequence- versus position-gated updates with comparable capacity and training budgets. Match the skip when isolating attention's contribution.

BLF's joint-context attention is a different computation. A skip retains a representation, not a guaranteed prediction score. After the correction, gate and attention comparisons, an optional experiment could compare raw auxiliary embeddings with their reconstruction residuals and both together. Those feature residuals differ from target errors; they are not necessarily uniquely biological information.

### Does respecting transcript regions improve prediction?

Start with human half-life, then translation efficiency, building on the [region checks and pooled comparisons](halflife_translation_efficiency.ipynb). Keep their [different target meanings](../../docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explicit. Protein residues correspond to coding-sequence (CDS) nucleotides, with no corresponding translated residue in the untranslated regions (UTRs).

Retain nucleotide-resolution RNA and map protein features to annotated CDS positions:

| Comparison | Question |
| --- | --- |
| RNA alone → add protein, zero-filled outside CDS | Does the additional representation help? |
| Add a CDS/UTR availability mask | Does region information help beyond zero filling? |
| Add position-wise gates | Does adaptive weighting help beyond the mask? |
| Add CDS-restricted gated attention with an RNA skip | Does this design help beyond mixing, with the skip matched to isolate attention? |

Compare native-resolution with codon-pooled RNA and true with appropriately permuted masks. Supplying a mask as a feature does not enforce exclusion: multiply the protein-update gate by the CDS mask to block direct updates outside CDS. This tests correspondence, not whether proteins can regulate UTRs. These are new token-level experiments; existing pooled probes cannot recover positions lost through averaging.

### Conditional outlook: transfer to other molecular relationships

#### RNA–protein binding

Binding pairs an RNA with a separate protein partner rather than its translation. Compare single-partner and simple pair baselines with selected attention/gate variants, including partner reassignment controls. Establish label evidence and artificial-negative construction, then distinguish RNA-disjoint, protein-disjoint and dual-disjoint evaluation. Candidate datasets remain conditional, not a committed suite.

#### Protein–ligand affinity

Protein–small-molecule pairs provide another transfer test. Select an assay endpoint, single-partner/simple-pair baselines and cold-protein, cold-ligand or cold-both splits before testing fusion. Affinity and inhibition measurements are not interchangeable. Both partner-task extensions require new preparation and representations; their purpose is to test the explanation's transfer.

### Organize the proposed notebooks around questions

These filenames are proposals, not runnable notebooks. The protocol numbers below locate Mina's original steps; they are not a new execution order or evidence that the experiments have run.

| Project home or proposed notebook | Mina's protocol steps | Gantt task |
| --- | --- | --- |
| Existing investigations and separate stability evaluation | 1: reliability and grouped evaluation | 24, 28, 41, 42 |
| `anchor_correction.ipynb` | 2: auxiliary target-error correction | 45 |
| `gate_diagnostics.ipynb` | 3–4: sequence and position gates | 46 |
| `anchor_cross_attention.ipynb` | 5: skip, attention and gated updates | 25 |
| `region_aware_fusion.ipynb` | 6: CDS/UTR correspondence | 42 |
| Conditional partner-task extensions; dataset selection first | 7–8: RNA–protein binding and protein–ligand affinity | 47, 48 |

[Gantt and Details](../../docs/Research_Project_Plan_Gantt.xlsx) owns dependencies and schedule qualifications. Task 42 spans two scopes: the existing pooled comparisons remain in Stage 4, and the proposed positional intervention is a new experiment within the same investigation.

### Keep published references and proposed methods distinct

The [IsoFormer audit](../stage3/isoformer_gtex_audit.ipynb) has saved evidence; the [shared model/input foundations](../shared_code/fusion/isoformer_reference.py) have synthetic checks. Pretrained integration and research training remain unfinished. [Full experimental reproduction is deferred](RUNBOOK.md#isoformer-reproduction-specification) from the main investigation route; the private preparation notebook and specification are not supplied with this checkout. Faithful reference operations can support a controlled comparison without reconstructing the paper's whole experiment; generic anchor-query attention should not be labeled an IsoFormer reproduction.

The [Coupled Mamba reference](../stage3/coupled_mamba_reference.ipynb) supplies an adapted implementation and controls. Its corresponding DNA/RNA/protein codon streams do not by themselves establish a pipeline for separately paired molecules. Further Mamba work is conditional on a measured long-sequence runtime or memory bottleneck (tasks 20/21), not a prerequisite for gate/attention tests. The reference methods remain in Stage 3; using one on a new partner task requires a separate input and evaluation design.

### Decide what the evidence would justify

If true masks help but local gates do not, region information has stronger support than adaptive weighting. If concatenation performs similarly to attention, extra complexity has not established its value. These are possible outcomes, not findings.

Select candidate gains, inputs, grouping rules, controls and tuning budgets before fitting. Existing test results informed these hypotheses; stronger prospective transfer claims need an independent final evaluation. Reusing caches saves extraction, but out-of-fold fits and long token inputs still cost computation. Measure a representative workload before expanding the experiment matrix.

### Source and decision provenance

The [unchanged advisor protocol](../../docs/Biological_Fusion_Plan.docx), supplied by Chase as Mina's notes, retains the full equations, controls and dataset candidates; its [source entry](../../docs/sources.md#minas-revised-biological-fusion-protocol) records reading scope. This guide connects those proposals to existing work without authorizing execution.

The November 10 inspection found canonical seed-0 swaps complete but no canonical replacement seed-1/2 results. It also found no saved matched LucaOne/half-life follow-up or released-reference IsoFormer or Coupled Mamba prediction results. [Gantt/Details tasks 20/24/42/44](../../docs/Research_Project_Plan_Gantt.xlsx) retain the relevant coverage and limitations; these are dated observations, not live progress. Mina's unnamed candidate gains still need an explicit shortlist.


## Results and supporting files

Small result JSON/CSV files under [results/](results/) retain evidence; the [run records](../../docs/run-records.md#recorded-runs) identify producing settings and limitations. The notebooks provide the route from those files to interpreted tables. A stored score, an available prediction array and a verified repeat are different kinds of evidence.

Large data, embeddings and weights follow the [repository cache policy](../../README.md#results-caches-and-intermediate-files). Compatible frozen caches can be reused, but existence alone does not establish compatibility. Use the [runbook](RUNBOOK.md) for execution and recovery; its dated assignments are not a live report of what is running.

Shared [fusion utilities](../shared_code/fusion/), [LucaOne helpers](../shared_code/lucaone.py), [full-transcript helpers](../shared_code/mrnabench.py) and [pooled-cache handling](../shared_code/matrix_cache.py) support the notebooks. The [test guide](../../tests/README.md) documents their checks. Experimental choices and interpretation remain with the notebooks; reusable background remains in [the method guides](../../docs/README.md#understand-the-methods).
