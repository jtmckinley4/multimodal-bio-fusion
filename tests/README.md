# Notebook checks

These automated software checks help detect changes in notebook setup and shared calculations. They use controlled examples to check code behavior; the experiment notebooks contain the biological analyses and their results.

Run these checks from the repository root using the Python environment prepared for the [notebooks](../README.md#notebook-dependencies):

```shell
python -B -m unittest discover -s tests -v
```

The tests use Python's built-in `unittest` runner and the existing notebook dependencies. They require no research dataset files, downloaded model weights, or additional test framework.

## What the checks cover

- [Setup checks](test_notebook_setup.py) execute only the `## Setup` sections of the [stability](../Code/Stage1_stability.ipynb) and [GTEx](../Code/Stage1_gtex.ipynb) notebooks, each in its own namespace. They check local configuration, encoder selections, GTEx pairs, and an explicit PyTorch seed call. Known project computation, data-loading, model-loading, and network entry points are blocked during setup execution.
- The setup suite also compiles every code cell and scans for global names missing from that notebook's preceding code. This is a conservative static check: it does not prove that every branch initializes a variable, and function-body references are checked where defined rather than where called.
- [Dataset-path checks](test_dataset_paths.py) verify that default inputs resolve to the repository's `datasets/` folder when called from the repository root, `Code/`, or another working directory. Tiny fixtures also check that explicit absolute and relative `data_dir` overrides load the supplied files. These checks do not read the research CSVs.
- [Source-display checks](test_source_displays.py) verify full listings and compact links for functions, decorated functions, and classes, plus rejection of unsupported compact sources. They check that saved compact links still identify the named definitions, including the first decorator line, rather than merely pointing to a line that exists.

- [Concatenation checks](test_concatenation.py) compare the shared pairwise calculation with the original loops on synthetic scalar and 30-output targets. They check feature and table order, grouped probe calls, gains against the supplied single-encoder means, unchanged inputs, and invalid-input errors.

- [Repeated-assignment checks](test_assignment_gains.py) compare per-assignment gains with the original loops on synthetic scalar and 30-output targets. They verify seed order, reuse of single-encoder scores, selection of the better single encoder within each assignment, unchanged inputs, and invalid-input errors.

The tests do not run the research bodies or update notebooks. Setup uses separate namespaces in one Python process, not separate Jupyter kernels. Passing these checks does not reproduce scientific results or verify how a notebook viewer renders or follows links. The blocked entry points guard against accidental execution; they are not a sandbox for untrusted code.

## After editing a notebook or shared code

Keep each notebook's imports and settings inside its bounded `## Setup` section, followed by the next main section. The notebooks' saved reading mode uses `SHOW_IMPLEMENTATION = False`. If code moves and a source-link check fails, run Setup and only the affected `show_source` cells in the relevant notebook, then save and rerun these checks. Displaying source does not run the referenced research functions.

The tests use current settings and definitions rather than fixed cell IDs, model counts, or historical notebook backups. If the display format or setup structure intentionally changes, update the affected checks alongside that change.
