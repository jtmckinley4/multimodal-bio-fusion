# From sequences to embeddings

This guide follows the opening pipeline in [Stage1_stability.ipynb](../../Code/Stage1_stability.ipynb): prepare matched biological inputs, load a frozen encoder, and turn its token representations into one vector per sequence. The notebook keeps the actual settings, checks, saved results, and local interpretations. Use [Setup](../../README.md#setup) for installation and loading-message guidance.

The main stability analysis starts from one coding sequence per row. DNA and RNA inputs differ in their letter convention; the protein input is translated and loses synonymous codon distinctions. A common sequence origin therefore does not imply identical information in every input. The [GTEx experiment](../../Code/Stage1_gtex.ipynb#Distinct-inputs:-GTEx-transcript-expression) uses a separate path: genomic DNA, full transcripts for single-nucleotide RNA encoders, coding sequences for codon RNA encoders, and supplied proteins. `Dataset.inputs_for` prepares those inputs, which are passed to embedding with `prepared=True`; the stability path performs coding-sequence conversion inside the embedding helper.

Read in pipeline order, or jump to [sequence checks](#sequence-checks-and-translation), [sampling](#sampling-and-retained-rows), [token limits](#tokens-and-input-limits), [loading](#loading-an-encoder), [pooling](#producing-a-sequence-embedding), or [cache reuse](#embedding-matrices-and-cache-reuse). The implementations are in [sequences.py](../../Code/mbf/sequences.py), [datasets.py](../../Code/mbf/datasets.py), [encoders.py](../../Code/mbf/encoders.py), and [embeddings.py](../../Code/mbf/embeddings.py).

## Sequence checks and translation

These helpers in `mbf.sequences` prepare the sequence forms that the encoders read: the coding sequence in DNA letters for DNA encoders, the same sequence in RNA letters for RNA encoders, and an amino acid sequence for protein encoders. `datasets.is_retained` combines the length and start check with a minimum translated length of five amino acids, and `encoders.encoder_input` selects the form for each encoder from its `alphabet` in the registry.

### Checking the sequence length and start

`is_in_frame_and_starts_correctly(seq)` converts the string to uppercase, removes surrounding whitespace, and returns `True` only when its length is divisible by three and it begins with `ATG` or `AUG`. These are the DNA-letter and RNA-letter forms of the start codon accepted by this notebook.

Codons contain three nucleotides, so the length check is:

$$
L \bmod 3 = 0.
$$

In this notation:

- $L$ is the normalized string length, counted in nucleotide characters.
- $\bmod$ gives the remainder after integer division.
- The divisor 3 is the number of nucleotides per codon; a remainder of 0 means the length can be divided into complete groups of three.

**Connection to the code.** This equation is the Boolean test `len(seq) % 3 == 0` inside `is_in_frame_and_starts_correctly`; `%` is Python's remainder operator. The function combines that test with `seq.startswith(("ATG", "AUG"))` using `and`, so both must hold.

A 12-nucleotide sequence contains four complete codons and passes the length check; a 13-nucleotide sequence has one nucleotide left over and fails it. Passing the length check alone is insufficient: the starting letters must also match.

This is a preliminary filter. It does not validate the nucleotide alphabet, check for a terminal stop codon, or establish that the input is a complete biological coding sequence.

### Translating the sequence

`translate_cds(seq)` returns an amino acid string using Biopython's default standard genetic code. CDS stands for coding sequence. With `to_stop=True`, translation ends at the first in-frame stop codon, and the stop symbol is omitted. If there is no stop codon, all complete codons are translated. The call does not enable Biopython's separate `cds=True` validation. See the [Biopython translation documentation](https://biopython.org/docs/latest/api/Bio.Seq.html#Bio.Seq.Seq.translate).

Before translation, `len(seq) % 3` gives the number of trailing characters outside a complete codon. Subtracting that remainder identifies where to end the slice: an input of length 13 would be trimmed to length 12. In the current callers, surrounding whitespace has already been removed and the preceding length check requires a multiple of three, so this trimming normally removes nothing. It remains part of the helper's behavior when called independently. The protein input ends at the first in-frame stop, while the nucleotide inputs retain the supplied sequence after it. Invalid symbols can cause translation to raise an error; these filters do not guarantee that every malformed sequence becomes a counted rejection.

### Converting between RNA and DNA letters

`rna_to_dna(seq)` returns an uppercase string with every `U` replaced by `T`; for example, `AUGGCU` becomes `ATGGCT`. `dna_to_rna(seq)` performs the reverse replacement, so `ATGGCT` becomes `AUGGCU`. Both change only the alphabet used to represent the nucleotide sequence. Neither translates the sequence into a protein or computes a complementary strand.

The stability analysis uses the DNA-letter form as input to DNA encoders and in Track C's motif comparisons, and the RNA-letter form as input to RNA encoders. The CSV sequences are written in RNA letters, so `dna_to_rna` leaves them unchanged; it makes the RNA encoders' input explicit rather than relying on the file's alphabet. The RNA-FM and RiNALMo tokenizers would also convert any T to U. Supporting both `ATG` and `AUG` in the initial check accommodates either input convention; it does not establish that both conventions occur in the CSV data.

## Sampling and retained rows

The stability loading cell prepares its common sample as follows. `datasets.load_dataset`:

1. Loads the dataset's file and drops rows missing a sequence.
2. Samples `N_ROWS` rows without replacement using `SEED`. The current setting requests 1,000 rows before the filters. With the same input rows, row order, and software environment, the seed makes this sample repeatable. See the [pandas sampling documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html).
3. Removes surrounding whitespace, converts each sequence to uppercase, and keeps the rows that pass `is_retained`, in sample order. It counts the rows each filter removed; a row that fails the length/start check is not translated or counted again.

`kept_seqs` holds the retained sequences and `labels` their stability values in the same order. `kept_seqs` supplies the coding-sequence inputs and groups for the stability probes, so identical strings stay together in the outer folds. Internal model selection has the separate limitation explained in the [linear-probing guide](linear-probing.md#current-evaluation-limitation). `dataset.rows` keeps the original file index and columns of the retained rows. This sampling step does not use the CSV's `Split` column; it does not establish separate training, validation, or test sets.

The sampling step does not remove duplicate sequences. The notebook's [dataset audit](../../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) retains the full-file findings separately from this preparation method.

## Tokens and input limits

The encoder input limits are measured in tokens and recorded in the registry; the notebook's settings table lists them. The structural-proxy crop is measured in nucleotides. A token covers a different number of letters for each encoder: most Nucleotide Transformer tokens cover six nucleotides, DNABERT-2's byte-pair tokens cover a variable number, HyenaDNA, RNA-FM, and RiNALMo tokens cover one nucleotide, mRNA-FM and CaLM tokens cover one codon, and ESM-2 and ProtBERT tokens cover one amino acid. The limits of RNA-FM and RiNALMo, 1,024 tokens, include two special tokens, so each sees at most 1,022 nucleotide positions from a sequence. CaLM's limit of 1,026 tokens covers 1,024 codons, or 3,072 nucleotides, and HyenaDNA's limit of one million tokens is far above any sequence here.

## Loading an encoder

`encoders.load_encoder(encoder, device)` loads the tokenizer and pretrained encoder for one registry entry at its recorded revision, moves the model to `DEVICE`, and returns the tokenizer and model together. Checkpoints that load with the Transformers auto classes use them directly. Nucleotide Transformer v2 and DNABERT-2 ship their own model code, written for an earlier Transformers version; `load_encoder` builds those models from their configuration, loads the checkpoint weights, and raises an error if any weight other than the unused pooler fails to match. ProtBERT's checkpoint does not name its model type, so `load_encoder` loads it with the BERT tokenizer, with letter case preserved, and the BERT model class directly; its protein input has residues separated by spaces. For ProtBERT, `protein_input` also replaces U, Z, O, and B with X. Importing the package does not load an encoder. Each analysis loads an encoder when it needs one and releases it afterwards with `encoders.free_memory`, so only one model occupies memory at a time.

`model.eval()` switches layers with distinct training and evaluation behavior, such as dropout, into evaluation mode. It does not disable gradient tracking or set the model's parameters to `requires_grad=False`. The separate `@torch.no_grad()` decorator on `embed` disables gradient tracking during embedding extraction, avoiding the memory cost of recording operations for a backward pass. The encoders remain unchanged because this workflow does not perform training updates. See the PyTorch documentation for [evaluation mode](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval) and [gradient tracking](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html).

## Producing a sequence embedding

`embeddings.embed(seq, tokenizer, model, max_len, device)` tokenizes one sequence, moves the resulting input tensors to the device, and runs the encoder. `truncation=True` limits the tokenized input to `max_len`. Token counts follow the model's tokenizer and need not equal the number of nucleotides or amino acids in the original sequence. This function processes one sequence at a time and does not request padding. See the Transformers documentation for [padding and truncation](https://huggingface.co/docs/transformers/main/en/pad_truncation).

The encoder returns a final hidden-state vector for each token. The `last_hidden_state` array has shape `(1, T, d)`: one sequence, `T` token positions, and `d` embedding features per position. This follows the Transformers [model-output convention](https://huggingface.co/docs/transformers/main_classes/output).

Mean pooling averages the token vectors to produce one vector for the sequence:

$$
\mathbf{z}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{h}_{t}.
$$

In this notation:

- $T$ is the number of token positions returned after tokenization and truncation, including any special tokens.
- $t$ identifies one of those positions, from 1 through $T$ in the equation; tensor positions are indexed from 0 through `T - 1`.
- $\mathbf{h}_{t}$, read as "bold h sub t," is the final hidden-state vector at position $t$. Bold symbols represent vectors.
- $d$ is the number of features in each token vector and in the pooled sequence vector.
- $\mathbf{z}$ is the resulting sequence embedding.
- $\sum$ means to add the vectors across token positions. Dividing by $T$ takes their average, feature by feature.

For example, token vectors $(1, 3)$ and $(5, 7)$ would average to the sequence vector $(3, 5)$. This is an illustrative calculation, not a model output.

**Connection to the code.** The equation describes `token_embeddings.mean(dim=1)` inside `embeddings.embed`; it is a direct average of the encoder output, not an additional fitted model. The surrounding operations prepare and return that vector.

| Code operation | Input shape → output shape | Connection to the mathematics |
|---|---|---|
| `out[0]` | Model output → `(1, T, d)` | Supplies the vectors $\mathbf h_t$, the `last_hidden_state`, and retains the one-sequence batch axis. |
| `token_embeddings.mean(dim=1)` | `(1, T, d)` → `(1, d)` | Compute $\mathbf z$ by averaging over the token axis. |
| `.squeeze()` | `(1, d)` → `(d,)` | Remove the singleton batch axis for these encoder widths. |
| `.float().cpu().numpy()` | `(d,)` → `(d,)` | Return the vector as a 32-bit NumPy array on the CPU. |

`out[0]` selects the first model output, which is `last_hidden_state` for every encoder used here; DNABERT-2's model code returns a tuple rather than a named output, so the code indexes it by position. Sequences with different token counts therefore produce vectors of the same size for a given encoder. The pooled output no longer retains separate token positions.

### Current pooling choice

The mean includes every returned token position, including special tokens added by the tokenizer. The pooling operation does not use a separate mask to exclude positions. This documents the current representation definition; excluding special tokens or changing the weighting would change the embeddings and require a separate methodological decision.

## Embedding matrices and cache reuse

`embed_with_cache` produces one matrix per encoder, with one row per supplied sequence and one column per pooled feature. In the stability analysis, the supplied strings are `kept_seqs`, and the helper converts each coding sequence through `encoder_input` before embedding. In GTEx, the caller supplies each encoder's prepared strings with `prepared=True`. Keep input, label, and group row order aligned throughout; the cache does not align labels or groups for the caller.

The cache file is named for `encoder.key` in the selected output directory. Before reusing it, the helper compares four recorded values:

| Recorded field | What is compared |
| --- | --- |
| `checkpoint` | The encoder's checkpoint identifier. |
| `revision` | The registry's revision string, or an empty string when no revision is pinned. |
| `max_len` | The encoder's token limit. |
| `fingerprint` | The first 16 hexadecimal characters of a SHA-1 hash of the newline-joined supplied sequences, in order. |

A mismatch causes that encoder's matrix to be recomputed. The fingerprint covers the strings passed to the helper: coding sequences on the stability path, prepared encoder inputs on the GTEx path. It does not record the conversion implementation or the `prepared` flag itself.

**Limits of reuse.** Pooling code, conversion code, package versions, and tokenizer behavior are not separately recorded. `device` is saved but is not part of the reuse comparison. An empty revision string does not identify the resolved checkpoint commit, so a change to an unpinned remote checkpoint is not detected by this metadata comparison. A cache match therefore establishes only that the recorded fields agree. A representation-method change can require deliberate recomputation even when those fields still match.

The [Generated files policy](../../README.md#generated-files) identifies output locations and their Git treatment. The [notebook embedding section](../../Code/Stage1_stability.ipynb#Embedding-the-sample-with-every-encoder) keeps the matrix-shape outputs associated with the producing run. A width disagreement with its loading check is an inconsistency to investigate; width alone does not identify its cause.
