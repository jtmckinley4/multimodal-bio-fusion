# Notebook and shared-code checks

These automated software checks help detect changes in notebook setup and shared calculations. They use controlled examples to check code behavior; the experiment notebooks contain the biological analyses and their results. [Wilson et al.'s scientific-computing guidance](../docs/sources.md#wilson-et-al-best-practices-for-scientific-computing) explains the role of controlled checks and small changes.

Run these checks from the repository root using the Python environment prepared for the [notebooks](../README.md#notebook-dependencies):

```shell
python -B -m unittest discover -s tests -v
```

The tests use Python's built-in `unittest` runner and the existing notebook dependencies. They require no research dataset files, downloaded model weights, or additional test framework.

## What the checks cover

- [Setup checks](test_notebook_setup.py) execute only the `## Setup` sections of the [stability](../Code/stage1/Stage1_stability.ipynb) and [GTEx](../Code/stage1/Stage1_gtex.ipynb) notebooks, each in its own namespace. They check local configuration, encoder selections, GTEx pairs, and an explicit PyTorch seed call. Known project computation, data-loading, model-loading, and network entry points are blocked during setup execution.
- The setup suite also compiles every code cell and scans for global names missing from that notebook's preceding code. This is a conservative static check: it does not prove that every branch initializes a variable, and function-body references are checked where defined rather than where called.
- [Dataset-path checks](test_dataset_paths.py) verify that default inputs resolve to the repository's `datasets/` folder when called from the repository root, `Code/`, or another working directory. Tiny fixtures also check that explicit absolute and relative `data_dir` overrides load the supplied files. These checks do not read the research CSVs.
- [Source-display checks](test_source_displays.py) verify full listings and compact links for functions, decorated functions, and classes, plus rejection of unsupported compact sources. They check that saved compact links still identify the named definitions, including the first decorator line, rather than merely pointing to a line that exists.

- [Concatenation checks](test_concatenation.py) compare the shared pairwise calculation with the original loops on synthetic scalar and 30-output targets. They check feature and table order, grouped probe calls, gains against the supplied single-encoder means, unchanged inputs, and invalid-input errors.

- [Repeated-assignment checks](test_assignment_gains.py) compare per-assignment gains with the original loops on synthetic scalar and 30-output targets. They verify seed order, reuse of single-encoder scores, selection of the better single encoder within each assignment, unchanged inputs, and invalid-input errors.

- [Probe construction](test_probe_factory.py) compares scalar/multioutput predictions and grouped/ungrouped scores with the prior estimator definition, including classification and independent fitted state. [Seed routing](test_probe_seeds.py) checks the retained default, nondefault single/pair correspondence, unchanged repeated assignments, explicit notebook settings, and mirroring of an asymmetric callback.
- [Scientific-option validation](test_validation.py) rejects unsupported permutation nulls and multi-target scalar audits before work; hand-worked examples and retained seeded answers check supported behavior.
- [Model cleanup](test_embedding_resources.py) uses substitute inference to check normal release, inference errors, interruption, and simultaneous inference/cleanup failure. It does not measure real GPU-memory release.
- [Embedding-cache checks](test_embedding_cache.py) use tiny local NPZ files and substitute models to check effective-input identity, legacy/corrupt errors, matrix shape, deterministic archive closure, and interrupted publication. Completed fixture caches survive failed replacement; no team caches are used or migrated.
- [Fusion checks](test_fusion_model.py) cover model shapes, CPU training fixtures, seeded construction, interrupted checkpoint continuation, result publication, and older result formats. They also check paired-bootstrap inputs and reject requests that would replace existing results to obtain missing predictions.
- [LucaOne checks](test_lucaone.py) cover the context arithmetic Stage 3 measured (one token per character, the 957-nucleotide cap on the joint input, what each mode admits), pooling that drops the framing tokens, windowed pooling past the context, and a cache that is reused only for the same checkpoint, stream type and rows. A stub tokenizer and a four-wide stub model stand in for the 1.8B-parameter checkpoint; nothing is downloaded.
- [mRNABench checks](test_mrnabench.py) cover the transcript regions recovered from the codon-start track, including rejection of a track that does not describe a reading frame; the feasibility gate's verdicts on synthetic datasets with and without untranslated regions; grouped splits that keep a gene on one side and the homology map when it has been downloaded; the ridge probe; in-frame codon counting and the nesting of the region feature sets; and chunked embedding of a transcript longer than an encoder's context. No dataset is downloaded.
- [IsoFormer checks](test_isoformer.py) cover the aggregator, expression-panel scoring, budgeted batches, and small noncheckpointed fits with synthetic inputs. They do not validate the research notebook's custom training/checkpoint path; that path is explicitly rejected until input identity and sampler recovery are supported.
- [GTEx coordinates](test_gtex_coordinates.py) checks strand-aware endpoints, requested length, TSS position, unchanged boundaries, and prior-expression equivalence using invented rows. [GTEx input/output](test_gtex_io.py) checks downloads, schema, response validation, bounded retry decisions, and CSV publication with local fixtures and mocked requests.

The tests do not run the research bodies or update notebooks. Setup uses separate namespaces in one Python process, not separate Jupyter kernels. Passing these checks does not reproduce scientific results or verify how a notebook viewer renders or follows links. The blocked entry points guard against accidental execution; they are not a sandbox for untrusted code.

## Forward-only codon-grid checks

From the repository root, select only the codon-grid and TextCNN forward checks:

```shell
python -B -m unittest discover -s tests -p test_fusion_model.py -k CodonGridTests -k TextCNNTests -v
```

These checks use small CPU tensors. Hand-derived impulses verify Equation 1's actual kernel 2, stride 2, padding 0 and positional values before and after length fitting for complete 6-mers. RNA fixtures check triplet means, dropped incomplete triplets, masks, and nonzero batch padding; per-row invariance covers both convolution settings. TextCNN checks cover padded inputs and short sequences. No optimizer, backward pass, research data, cache opening, checkpoint access, or pretrained model is needed; codon-grid fixtures also block training and artifact entry points.

This command does not run the training tests elsewhere in `test_fusion_model.py`. It does not validate biological reading frames or the placement of NT remainder tokens, compare trained convolution variants, or implement the pending standalone codon-pooled RNA baseline. Julian's reported standalone baseline needs source inspection, integration, and its own acceptance checks.


## After editing a notebook or shared code

Keep each notebook's imports and settings inside its bounded `## Setup` section, followed by the next main section. The notebooks' saved reading mode uses `SHOW_IMPLEMENTATION = False`. If code moves and a source-link check fails, use an isolated, guarded Setup namespace and only the affected `show_source` calls. Save only their changed implementation-display content, preserving execution counts, scientific outputs, unrelated metadata, and existing device/setup output. Displaying source does not run the referenced research functions; the setup guard in `test_notebook_setup.py` provides the bounded environment.

Setup and source-display checks use current settings and definitions rather than fixed cell IDs, model counts, or historical notebook backups. Numerical preservation checks deliberately retain prior constructions and the agreed default seed as references; they do not independently validate the scientific method. If a configuration or contract intentionally changes, update the affected checks with that decision.
