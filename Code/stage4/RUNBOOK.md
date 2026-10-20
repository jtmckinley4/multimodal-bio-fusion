# Stage 4 runbook: who runs what, in what order

Written 2026-10-20, for the October 22 meeting and the nights before it. The point of this
file is that two people and two machines never do the same work twice or write to the same
place at once. If you are about to start something, find it here first.

## The split

| | Julian (MacBook, Apple GPU) | Chase (desktop, RTX 2060 SUPER) |
| --- | --- | --- |
| Owns | the feasibility gate and everything that is probes rather than training | every training sweep and every large embedding extraction |
| Why | these are minutes of CPU and need no CUDA | an 8 GB CUDA card is the only GPU in the project, and float16 on it is what makes LucaOne fit at all |
| Writes to | `results/halflife_te/`, `.data-cache/` | `results/phaseB/`, `results/lucaone/`, `results/gtex_isoformer/`, `../stage3/embeddings/` |

Nobody runs a notebook that is not theirs in the table below without saying so first. The
results folders are disjoint on purpose: two people can push on the same day without a
merge conflict in anything but `docs/run-records.md`.

## The order

Numbers are the research plan's priorities. Everything in **bold** is new code pushed on
2026-10-20 and has never been run.

| # | What | Notebook | Owner | Cost | Depends on |
| --- | --- | --- | --- | --- | --- |
| 1 | Phase A, four benchmarks | `tier2_blf_phaseA.ipynb` | done 2026-10-20 | 26.9 h | |
| 5 | mRFP synonymous control | `tier1_mrfp_control.ipynb` | done 2026-10-19 | 15 min | |
| 1 | **Phase B, encoder swaps** | `tier2_encoder_swaps.ipynb` | **Chase** | about 53 GB of cache, then the sweep | Phase A's results, which are pushed |
| 3, 4 | **Half-life and TE gate, and the composition floor** | `tier3_halflife_te.ipynb`, Parts 1 and 2 | **Julian** | 150 MB download, about 30 min, no GPU | nothing |
| 2 | **LucaOne on mRNA stability** | `tier2_lucaone_isoformer.ipynb`, Parts 1 to 3 | **Chase** | 7 GB download, then the projection the notebook prints | the Stage 3 stability token caches, which exist |
| 44 | **IsoFormer seeds 1 and 2** | `tier2_lucaone_isoformer.ipynb`, Part 4 | **Chase**, if a GPU hour is spare | about 1 h on CUDA | the GTEx caches, which exist |
| 3, 4 | **Half-life and TE encoder probes** | `tier3_halflife_te.ipynb`, Part 3 | after Thursday | a GPU night | the gate passing |

## Phase B, step by step (Chase)

```bash
cd ~/.../Mina/Code/stage4          # wherever the repo lives on your machine
git pull
jupyter lab tier2_encoder_swaps.ipynb
```

1. Run every cell down to the **disk projection**. It prints what the token caches will cost,
   per benchmark and encoder, before anything is written. Stop if that number does not fit.
2. The **token-count check** loads each swapped encoder once and verifies that its stream has
   the token count the codon grid expects. A `MISMATCH` here means stop: a wrong token count
   silently misaligns the grid, which is exactly the bug Stage 3 found in the paper.
3. The **extraction** is resumable shard by shard. If it is interrupted, rerun the same cell
   and it continues. It writes to `../stage3/embeddings/<benchmark>/<encoder>/`, which Git
   ignores.
4. The **sweep** runs the cheap heads for every swap first and cross-attention last, so an
   interrupted night still leaves every swap measured with two heads out of three. It is
   resumable: a run already on disk is read rather than repeated.
5. Then the reading cells, which need no GPU.

mRNA stability is last and is limited to three swaps by default (`STABILITY_SWAPS`), for the
reason the notebook gives: mRNA-FM is what the mRFP control predicts should win the RNA slot,
CaLM is its control, and ESM-2 150M tests whether the protein slot was too small. Widen that
list only if the disk projection says the rest fit.

**What to watch.** The 2060 SUPER has 8 GB. If a run runs out of memory, lower `BATCH_SIZE`
in the settings cell rather than changing anything about the model, and say so in the run
record: batch size is part of the training recipe being reproduced.

## The gate (Julian)

```bash
cd ~/Desktop/Masters/CapstoneCourse/Mina/Code/stage4
jupyter lab tier3_halflife_te.ipynb
```

Parts 1 and 2 only. Part 1 downloads six parquet files, about 150 MB in all, into
`.data-cache/`, and prints a verdict per dataset: whether the transcripts really carry both
untranslated regions, whether the coding frame is recoverable, how many gene groups there are
to split on. Part 2 is the composition floor, which is the number any encoder has to beat.
Neither needs a GPU and the whole thing is about half an hour, most of it downloading.

Push `results/halflife_te/gate.csv` and `composition_baselines.csv` when it finishes, and add
a run record. Those two files are what goes in front of Mina for Priorities 3 and 4.

## LucaOne (Chase)

The notebook will not start the expensive part until `RUN_LUCAONE = True`, and before that
flag it times eight sequences and prints the projected hours for the whole benchmark. Read
that number first. The model is 1.8B parameters; on CUDA it loads in float16, which is about
3.6 GB of the 8 GB card.

It needs `transformers` 5.15.1, which the project already pins, and nothing else: the
checkpoint ships its own model and tokenizer code, pinned to a commit in
`shared_code/lucaone.py`. The model card's instruction to `pip install lucaone` is not needed.

## Rules that keep us out of each other's way

1. **One owner per notebook per night.** If you want to run someone else's, message first.
2. **Never edit a result file by hand.** Every number in a table comes from a JSON or CSV a
   run wrote.
3. **A run record per run**, appended to the end of `docs/run-records.md`, never edited into
   the middle of someone else's section. That file is the only one we are both likely to
   touch; appending keeps the merges trivial.
4. **Push when a notebook finishes**, not at the end of the week. A sweep that is still
   running is fine to leave unpushed; a finished one is not.
5. **Embeddings and checkpoints stay out of Git.** They are ignored already; do not force-add
   them. The result JSON beside a checkpoint is the record.
6. **If you change shared code, say so in the message.** Both Stage 4 notebooks import
   `shared_code/fusion/train.py`; a change there changes results that have already been run.

## What is deliberately not being run before Thursday

* **Ab1**, BioLangFusion's fifth benchmark. It is Yazdani-Jahromi et al. 2024, from UCF, and
  is not in the public CodonBERT release. Asking Mina whether it can be obtained is on the
  meeting agenda; until then Phase A is four benchmarks out of five and says so.
* **The half-life and TE encoder probes** (Part 3). The gate comes first, by the plan's own
  rule, and the embeddings are a GPU night that Phase B has a better claim on.
* **DNABERT-2 as a Phase B swap.** Its byte-pair tokens cover a variable number of
  nucleotides, so the codon grid needs an adapter designed once and held fixed. It is written
  up in the Overleaf as an open item rather than quietly skipped.
* **Coupled Mamba and the project's own cross-attention design.** January, against every
  baseline at once, as the plan's ordering has it.
