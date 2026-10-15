# Stage 1: representation analysis

Twelve frozen encoders compared on two datasets. Both notebooks are complete; their saved outputs are the Stage 1 results.

| File | Contents |
| --- | --- |
| [Stage1_stability.ipynb](Stage1_stability.ipynb) | mRNA stability, where every encoder reads the same coding sequence, plus the mRFP synonymous-recoding control |
| [Stage1_gtex.ipynb](Stage1_gtex.ipynb) | The GTEx pilot, where the DNA window, transcript, and protein are different molecules |
| [build_gtex_pilot.py](build_gtex_pilot.py) | Rebuilds `datasets/GTEx_pilot.csv`; run from the repository root with `python Code/stage1/build_gtex_pilot.py` |
| `embeddings/` | Saved embedding caches, one folder per dataset sample; local and ignored by Git |
| `embeddings_legacy/` | Older caches from before the October 6 runs; local and ignored by Git |

Run a notebook with this folder, `Code/stage1/`, as the working directory (the default in VS Code and Jupyter). The Imports cell finds `Code/shared_code/` from here, and the embedding paths are relative to this folder. See [Setup](../../README.md#setup).
