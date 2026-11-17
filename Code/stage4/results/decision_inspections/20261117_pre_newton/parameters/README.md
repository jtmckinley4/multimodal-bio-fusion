# Gate and attention parameter inspection

The CPU previews confirm that the retained rank-150 nongated control has substantially more parameters than the corrected joint linear gate. Choosing the nearest parameter count instead would constrain its affine mixer to rank 2. Neither is automatically the right control: the next decision concerns the computational constraint we want to test, not simply matching two numbers. No control, training panel or maintained notebook was changed by this inspection.

## Read the observed counts

The [gate preview](gate_diagnostics_preview.ipynb) and [attention preview](anchor_cross_attention_preview.ipynb) retain the executed setup, exact notebook preview code and displayed tables. The [machine-readable tables](tables.json) contain the same displays. The first table in each set is the declared panel; the second contains model counts. Gate diagnostics has a third table of exploratory control structures.

These are **fusion-module** counts for the original NT/RNA-FM/ESM-2 combination; the common TextCNN and codon-grid modules are excluded from this table.

| Operation | Fusion parameters | What the comparison exposes |
| --- | ---: | --- |
| Mean-summary softmax MIL | 1,245,701 | Entropy removal leaves the count unchanged |
| Mean-summary independent sigmoid | 1,245,700 | One fewer learned temperature parameter; normalization and temperature both change |
| Mean-plus-max softmax | 1,605,701 | The wider scorer adds 360,000 parameters |
| Joint position sigmoid | 890,403 | The affine joint scorer is much smaller than the retained nonlinear sequence scorer |
| Retained nongated rank-150 mixer | 1,245,750 | 355,347 more parameters than the joint position gate |
| Ordinary or anchor-skip attention | 3,769,800 each | Adding the skip introduces no learned parameters |
| Sequence- or position-gated attention | 3,773,402 each | Each adds 3,602 parameters; gate location does not change their count |

Replacing RNA-FM with mRNA-FM adds 384,000 projection parameters to every three-stream fusion in this table. The saved preview enumerates both combinations and, for attention, each possible anchor without choosing one from data.

## Review the nongated control tradeoff

The joint position gate contains 5,403 gate-specific parameters. Keeping the common projections and prediction head fixed gives these alternatives:

| Nongated mixer | Mixer parameters | Difference from joint gates | Maximum affine rank |
| --- | ---: | ---: | ---: |
| Current two-map bottleneck 150 | 360,750 | +355,347 | 150 |
| Nearest-count two-map bottleneck 2 | 5,402 | -1 | 2 |
| Direct affine map | 1,080,600 | +1,075,197 | 600 |

The rank bound describes the linear map after the common projections. Input-dependent sigmoid gating is nonlinear, so it does not have the same fixed affine-rank interpretation. The rank-2 option nearly matches count while imposing a severe structural restriction. Retain these as alternatives to discuss with Mina; do not describe the current rank-150 comparison as parameter-matched, and do not automatically add all alternatives as fits.

The attention sequence/position comparison holds the scorer formula and parameter count fixed. Ordinary versus skip attention also has identical counts. Those contrasts address narrower changes than the A4 sequence/position comparison, which changes input context and scorer structure as well as gate location. None of these count comparisons demonstrates which method predicts better.

## Interpretation limits and execution evidence

The previews count every registered parameter with `requires_grad=True`. In the anchor-only RNA/protein controls, the total also includes the 524,800-parameter DNA transposed convolution constructed by the common codon-grid wrapper, although those selected-modality paths never use it. The fusion and prediction counts are unaffected. These totals should not be presented as measured active gradient counts or used to infer runtime or memory demand.

Only CPU setup, configuration and parameter-preview cells ran: 14 gate cases and 30 attention cases. No dataset, embedding cache, encoder checkpoint, forward pass, training, bootstrap or job submission was involved. The one-output preview is for regression; the existing three-class readout adds 602 parameters. No accuracy or biological conclusion follows from this inspection.

The [execution record](execution.json) identifies command, CPU producer, settings, start/end and consumed source hashes; [source_snapshot.zip](source_snapshot.zip) preserves those source bytes, and [the runner](run_parameter_previews.py) makes the selected-cell execution reviewable. The [verification record](verification.json) confirms unchanged source hashes and valid notebook outputs. It also corrects a recording-label error: `model_preview_rows` in the execution record counted the first panel table (7/5); the actual model-construction tables contain 14/30 rows. The original record and observed tables are preserved.

The source import cells currently reference `shared_code` before their later path setup. The inspection records an explicit checkout-path bootstrap before executing those unchanged cells. This preview does not establish that running the maintained notebooks in an unprepared kernel would succeed from the top.
