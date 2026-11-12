# Stage 5: controlled tests of fusion hypotheses

Stage 4 compares encoders and fusion methods; its retained results suggest that a stronger individual encoder can explain an apparent fusion improvement, while some combinations add predictive value. Stage 5 tests specific explanations through controlled changes: can auxiliary features correct a strong predictor, does a gate suppress useful contributions, and would anchor-preserving attention or biological correspondence help?

[Mina's revised protocol](../../docs/Biological_Fusion_Plan.docx) supplies the proposed comparisons. This stage develops four questions from that wider plan; existing evaluation work remains in Stage 4 and partner-task extensions remain conditional. **These are proposals, not completed experiments:** this folder contains only the plan, and its proposed notebooks have not been implemented or run. The starting point is frozen encoders with trainable prediction and fusion components.

## What Stage 4 supplies and still needs

The [Stage 4 guide](../stage4/README.md) and [swap report](../stage4/tier2_encoder_swaps.ipynb) separate better encoder choice, fusion beyond a single predictor, and attention or gating beyond simpler fusion. Its unfinished work includes selected [Phase A](../stage4/tier2_blf_phaseA.ipynb) and swap repeats, plus the matched [LucaOne](../stage4/tier2_lucaone_isoformer.ipynb) and [half-life/TE](../stage4/tier3_halflife_te.ipynb) comparisons. Unchanged repeats retain their Stage 4 identities and [continuation route](../stage4/RUNBOOK.md#complete-the-encoder-swaps-one-seed-at-a-time).

Stage 4's reliability work includes at least three independent training seeds for selected gains, separating seed variation from paired test-group bootstrap intervals. A separate stability evaluation groups identical sequences before train/validation/test assignment, considering near-identical groups where feasible. It needs new predictor fits and a separate experiment identity; compatible frozen caches may be reused. Preserve supplied-split results: an unseen test subset or grouped bootstrap does not remove training overlap.

Stage 4 need not be globally complete before preparing a bounded Stage 5 comparison. Select the evidence and evaluation needed for that question. Inspecting an existing gate belongs to Stage 4 interpretation; changing its behavior to test an explanation belongs here. Each new Stage 5 experiment includes its own matched baselines, training seeds and uncertainty analysis.

## Can auxiliary features correct the anchor's errors?

The **anchor** is the strongest eligible single-encoder predictor selected using validation, including its trained head. The comparison asks whether auxiliary representations improve on that complete predictor under matched evaluation, making useful information accessible after the strong individual predictor has been fitted.

Fit auxiliary representations to the anchor's regression errors and add the predicted correction to its output; classification instead uses additive logit corrections. Obtain training errors from **group-aware out-of-fold predictions**, so each example's group was excluded from the head that predicted it. Validation selects the correction; final test data stay outside selection.

Compare auxiliaries separately and jointly, with shuffled-input and parameter-matched random-feature controls. The [mRFP analysis](../stage4/tier1_mrfp_control.ipynb) offers a related precedent starting from calculated controls, not an encoder anchor. Success supports accessible complementarity; failure need not mean that no useful information exists.

[Liang et al.](../../docs/sources.md#liang-et-al-quantifying-and-modeling-multimodal-interactions) provide prior work on task-related redundancy, unique information and synergy, including model selection. Our proposed correction test measures predictive improvement; it does not implement their information decomposition. Connecting representation relationships to fusion choices therefore needs a specific biological contribution beyond that existing idea.

## Do gates preserve useful contributions?

If an auxiliary representation helps under a controlled comparison, a gate should not be assumed to retain that contribution. The existing BLF gate assigns one weight per modality per sequence. Compare its softmax competition with and without entropy regularization, independent sigmoid gates, and mean-plus-max summaries. Record per-example weights alongside prediction: a small weight alone does not establish harmful suppression or biological importance. Control feature-scale changes when comparing gate normalizations.

The contribution may also vary along a sequence. Compare sequence-wide gates with position-wise weights and a comparable-capacity positional model without gates. Keep encoders, alignment and the downstream predictor fixed. The [token-cache route](../stage3/02_token_embeddings.ipynb) supplies a starting point; equal widths do not establish correspondence, and codon pooling changes information. CoV-Vac's translated fragment has no established biological reading frame. Gate patterns can motivate perturbations, not establish local biological mechanisms.

## Should attention retrieve auxiliary information while retaining the anchor?

Let anchor positions query auxiliary streams, potentially of different lengths, then add gated updates through a skip connection. Compare anchor-only, concatenation, ordinary attention, anchor-skip attention and sequence- versus position-gated updates with comparable capacity and training budgets. Match the skip when isolating attention's contribution.

BLF's joint-context attention is a different computation. A skip retains a representation, not a guaranteed prediction score. After the correction, gate and attention comparisons, an optional experiment could compare raw auxiliary embeddings with their reconstruction residuals and both together. Those feature residuals differ from target errors; they are not necessarily uniquely biological information.

## Does respecting transcript regions improve prediction?

Start with human half-life, then translation efficiency, building on the [region checks and pooled comparisons](../stage4/tier3_halflife_te.ipynb). Keep their [different target meanings](../../docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explicit. Protein residues correspond to coding-sequence (CDS) nucleotides, with no corresponding translated residue in the untranslated regions (UTRs).

Retain nucleotide-resolution RNA and map protein features to annotated CDS positions:

| Comparison | Question |
| --- | --- |
| RNA alone → add protein, zero-filled outside CDS | Does the additional representation help? |
| Add a CDS/UTR availability mask | Does region information help beyond zero filling? |
| Add position-wise gates | Does adaptive weighting help beyond the mask? |
| Add CDS-restricted gated attention with an RNA skip | Does this design help beyond mixing, with the skip matched to isolate attention? |

Compare native-resolution with codon-pooled RNA and true with appropriately permuted masks. Supplying a mask as a feature does not enforce exclusion: multiply the protein-update gate by the CDS mask to block direct updates outside CDS. This tests correspondence, not whether proteins can regulate UTRs. These are new token-level experiments; existing pooled probes cannot recover positions lost through averaging.

## Conditional outlook: transfer to other molecular relationships

### RNA–protein binding

Binding pairs an RNA with a separate protein partner rather than its translation. Compare single-partner and simple pair baselines with selected attention/gate variants, including partner reassignment controls. Establish label evidence and artificial-negative construction, then distinguish RNA-disjoint, protein-disjoint and dual-disjoint evaluation. Candidate datasets remain conditional, not a committed suite.

### Protein–ligand affinity

Protein–small-molecule pairs provide another transfer test. Select an assay endpoint, single-partner/simple-pair baselines and cold-protein, cold-ligand or cold-both splits before testing fusion. Affinity and inhibition measurements are not interchangeable. Both partner-task extensions require new preparation and representations; their purpose is to test the explanation's transfer.

## Organize the proposed notebooks around questions

These filenames are proposals, not runnable notebooks. The protocol numbers below locate Mina's original steps; they are not a new execution order or a claim that every step belongs to Stage 5.

| Project home or proposed notebook | Mina's protocol steps | Gantt task |
| --- | --- | --- |
| Existing Stage 4 comparisons and separate stability evaluation | 1: reliability and grouped evaluation | 24, 28, 41, 42 |
| `01_anchor_correction.ipynb` | 2: auxiliary target-error correction | 45 |
| `02_gate_diagnostics.ipynb` | 3–4: sequence and position gates | 46 |
| `03_anchor_cross_attention.ipynb` | 5: skip, attention and gated updates | 25 |
| `04_region_aware_fusion.ipynb` | 6: CDS/UTR correspondence | 42 |
| Conditional partner-task extensions; dataset selection first | 7–8: RNA–protein binding and protein–ligand affinity | 47, 48 |

[Gantt and Details](../../docs/Research_Project_Plan_Gantt.xlsx) owns dependencies and schedule qualifications. Task 42 spans two scopes: the existing pooled comparisons remain in Stage 4, while the proposed positional intervention belongs here.

## Keep published references and proposed methods distinct

The [IsoFormer audit](../stage4/tier2_isoformer_gtex_audit.ipynb) has saved evidence; the [shared model/input foundations](../shared_code/fusion/isoformer_reference.py) have synthetic checks. Pretrained integration and research training remain unfinished. [Full experimental reproduction is deferred](../stage4/RUNBOOK.md#isoformer-reproduction-specification) from the main investigation route; the private preparation notebook and specification are not supplied with this checkout. Faithful reference operations can support a controlled comparison without reconstructing the paper's whole experiment; generic anchor-query attention should not be labeled an IsoFormer reproduction.

The [Coupled Mamba notebook](../stage4/tier4_designs.ipynb) supplies an implementation and controls, with fidelity and biological adaptation still distinct questions. Further Mamba work is conditional on a measured long-sequence runtime or memory bottleneck (tasks 20/21), not a prerequisite for gate/attention tests. Neither unfinished reference work nor cluster preparation automatically becomes a Stage 5 experiment.

## Decide what the evidence would justify

If true masks help but local gates do not, region information has stronger support than adaptive weighting. If concatenation performs similarly to attention, extra complexity has not established its value. These are possible outcomes, not findings.

Select candidate gains, inputs, grouping rules, controls and tuning budgets before fitting. Existing test results informed these hypotheses; stronger prospective transfer claims need an independent final evaluation. Reusing caches saves extraction, but out-of-fold fits and long token inputs still cost computation. Measure a representative workload before expanding the experiment matrix.

## Source and decision provenance

The [unchanged advisor protocol](../../docs/Biological_Fusion_Plan.docx), supplied by Chase as Mina's notes, retains the full equations, controls and dataset candidates; its [source entry](../../docs/sources.md#minas-revised-biological-fusion-protocol) records reading scope. This guide connects those proposals to existing work without authorizing execution.

The November 10 inspection found canonical seed-0 swaps complete but no canonical replacement seed-1/2 results. It also found no saved matched LucaOne/half-life follow-up or released-reference IsoFormer or Coupled Mamba prediction results. [Gantt/Details tasks 20/24/42/44](../../docs/Research_Project_Plan_Gantt.xlsx) retain the relevant coverage and limitations; these are dated observations, not live progress. Mina's unnamed candidate gains still need an explicit shortlist.
