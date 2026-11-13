# From sequences to predictions

This guide follows a sequence through input preparation, frozen encoder representations, and a fitted prediction model. It then explains the calculated targets and controls used to interpret what those representations make accessible. Start with [Data and biological targets](data-and-targets.md) for what a dataset row and its measured or supplied target mean; the methods below do not establish that measurement's biological validity.

Read in three phases:

1. [From sequences to embeddings](#from-sequences-to-embeddings): prepare matched inputs, handle token limits, pool representations, and understand cache reuse.
2. [Linear probing](#linear-probing): fit a simple predictor, separate training from evaluation, and interpret its score and limitations.
3. [Calculated targets and controls](#calculated-targets-and-controls): distinguish a length baseline, composition and folding targets, and a synonymous-recoding pipeline check.

The detailed procedures primarily describe Stage 1's pooled-embedding analyses. Stage 1's [mRFP control](../../Code/stage1/mrfp_representation_control.ipynb) reuses those pooled representations, while the [BioLangFusion references](../../Code/stage3/README.md) and [encoder-swap experiments](../../Code/stage4/encoder_swaps.ipynb) use a separate token-level extraction and fusion path. A pooled vector has no separate token positions; a token cache retains content-token vectors for the downstream fusion model. Their translation, special-token handling, cache format, and evaluation choices must not be treated as interchangeable. For a complementary question, [Comparing and interpreting representations](../../Code/stage1/representation-analysis.md) explains comparisons of representation spaces and further diagnostics.

## From sequences to embeddings

The opening pipeline in [Stage1_stability.ipynb](../../Code/stage1/Stage1_stability.ipynb) prepares matched biological inputs, loads a frozen encoder, and turns its token representations into one vector per sequence. The notebook keeps the actual settings, checks, saved results, and local interpretations. Use [Setup](../../README.md#setup) for installation and loading-message guidance.

Along the [gene-expression pathway](data-and-targets.md#connecting-the-targets-to-gene-expression), the supplied sequence can describe different biological regions. Genomic DNA may include parts of a gene and surrounding regulatory sequence. In a typical intron-containing eukaryotic gene, RNA processing removes introns and joins the retained segments, called exons. For a protein-coding transcript, the mature RNA contains a coding sequence (CDS) and untranslated regions (UTRs) before and after it; exons need not be entirely protein-coding. The 5′ and 3′ UTRs are transcribed sequence outside the main protein-coding region and can help control how the RNA is used. A protein sequence omits those regions and the synonymous codon distinctions within the CDS. [Molecular Biology of the Cell, 7th ed., ch. 6, RNA processing and splicing; ch. 7, p. 456](../sources.md#molecular-biology-of-the-cell-7th-edition).

The main stability analysis starts from one coding sequence per row. DNA and RNA inputs differ in their letter convention; the protein input is translated and loses synonymous codon distinctions. A common sequence origin therefore does not imply identical information in every input. The [GTEx experiment](../../Code/stage1/Stage1_gtex.ipynb#Distinct-inputs:-GTEx-transcript-expression) uses a separate path: genomic DNA, full transcripts for single-nucleotide RNA encoders, coding sequences for codon RNA encoders, and supplied proteins. `Dataset.inputs_for` prepares those inputs, which are passed to embedding with `prepared=True`; the stability path performs coding-sequence conversion inside the embedding helper.

GTEx genomic windows are prepared separately by `transcript_regions` in [the pilot builder](../../Code/stage1/build_gtex_pilot.py). Source transcript coordinates are one-based and inclusive; the helper calculates a zero-based transcription start site (TSS), then requests one-based inclusive Ensembl regions on the transcript's strand. In the current 6,000-base window, the TSS is at zero-based index 3,000 on either strand. For example, a positive-strand TSS at source position 101 with a six-base window requests positions 98–103; a negative-strand TSS at 200 requests 198–203 in reverse-complement orientation. The extraction preserves the previous formulas, including nonpositive endpoints near chromosome boundaries; it does not clamp or redefine those windows. [Coordinate checks](../../tests/test_gtex_coordinates.py) use small known answers without requesting sequences.

The implementations are in [sequences.py](../../Code/shared_code/sequences.py), [datasets.py](../../Code/shared_code/datasets.py), [encoders.py](../../Code/shared_code/encoders.py), and [embeddings.py](../../Code/shared_code/embeddings.py).

### Sequence checks and translation

These helpers in `shared_code.sequences` prepare the sequence forms that the encoders read: the coding sequence in DNA letters for DNA encoders, the same sequence in RNA letters for RNA encoders, and an amino acid sequence for protein encoders. `datasets.is_retained` combines the length and start check with a minimum translated length of five amino acids, and `encoders.encoder_input` selects the form for each encoder from its `alphabet` in the registry.

#### Checking the sequence length and start

`is_in_frame_and_starts_correctly(seq)` converts the string to uppercase, removes surrounding whitespace, and returns `True` only when its length is divisible by three and it begins with `ATG` or `AUG`. These are the DNA-letter and RNA-letter forms of the start codon accepted by this notebook.

Codons contain three nucleotides, so the length check is:

$$
L \bmod 3 = 0.
$$

In this notation:

- $L$ is the normalized string length, counted in nucleotide characters.
- $\bmod$ gives the remainder after integer division.
- The divisor 3 is the number of nucleotides per codon; a remainder of 0 means the length can be divided into complete groups of three.

**Connection to the code.** `is_in_frame_and_starts_correctly` combines `len(seq) % 3 == 0` with `seq.startswith(("ATG", "AUG"))` using `and`: the remainder (`%`) must be zero and the start must match.

A 12-nucleotide sequence contains four complete codons and passes the length check; a 13-nucleotide sequence has one nucleotide left over and fails it. Passing the length check alone is insufficient: the starting letters must also match.

This is a preliminary filter. It does not validate the nucleotide alphabet, check for a terminal stop codon, or establish that the input is a complete biological coding sequence.

#### Translating the sequence

`translate_cds(seq)` returns an amino acid string using Biopython's default standard genetic code. CDS stands for coding sequence. With `to_stop=True`, translation ends at the first in-frame stop codon, and the stop symbol is omitted. If there is no stop codon, all complete codons are translated. The call does not enable Biopython's separate `cds=True` validation. See the [Biopython translation documentation](https://biopython.org/docs/latest/api/Bio.Seq.html#Bio.Seq.Seq.translate).

For the biological codon-to-amino-acid mapping, use [NCBI's Standard Code, table 1](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG1). Its tables use T in place of RNA's U and list alternative genetic codes separately. The helper's default is an implementation choice; it does not establish the appropriate code for every organism or sequence in a dataset.

The helper removes `len(seq) % 3` trailing characters before translation. The current callers have already removed surrounding whitespace and required a multiple-of-three length, so this trimming normally removes nothing. The protein input ends at the first in-frame stop, while the nucleotide inputs retain the supplied sequence after it. Invalid symbols can cause translation to raise an error; these filters do not guarantee that every malformed sequence becomes a counted rejection.

This stop-at-first-stop behavior belongs to the Stage 1 pooled-input path. The BioLangFusion token path uses a different translation helper that retains translated positions and replaces stop symbols with `X`, then preserves content-token vectors for fusion. Its [token embedding implementation](../../Code/shared_code/fusion/token_embeddings.py) and experiment notebooks own that procedure; the pooled-input and cache descriptions here do not specify the token-grid experiment.

#### Converting between RNA and DNA letters

`rna_to_dna(seq)` returns an uppercase string with every `U` replaced by `T`; for example, `AUGGCU` becomes `ATGGCT`. `dna_to_rna(seq)` performs the reverse replacement, so `ATGGCT` becomes `AUGGCU`. Both change only the alphabet used to represent the nucleotide sequence. Neither translates the sequence into a protein or computes a complementary strand.

The letter substitution is a representation step. Biological DNA and RNA also differ in their sugar chemistry, and producing a mature RNA can involve transcription and processing. Changing `T` to `U` in a supplied coding string does not simulate those cellular processes or reconstruct a complete transcript. [Essential Cell Biology, 6th ed., ch. 7, “From DNA to RNA”](../sources.md#essential-cell-biology-6th-edition).

The stability analysis uses the DNA-letter form as input to DNA encoders and in Track C's motif comparisons, and the RNA-letter form as input to RNA encoders. The CSV sequences are written in RNA letters, so `dna_to_rna` leaves them unchanged; it makes the RNA encoders' input explicit rather than relying on the file's alphabet. The RNA-FM and RiNALMo tokenizers would also convert any T to U. Supporting both `ATG` and `AUG` in the initial check accommodates either input convention; it does not establish that both conventions occur in the CSV data.

### Sampling and retained rows

The stability loading cell prepares its common sample as follows. `datasets.load_dataset`:

1. Loads the dataset's file and drops rows missing a sequence.
2. Samples `N_ROWS` rows without replacement using `SEED`. The current setting requests 1,000 rows before the filters. With the same input rows, row order, and software environment, the seed makes this sample repeatable. See the [pandas sampling documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html).
3. Removes surrounding whitespace, converts each sequence to uppercase, and keeps the rows that pass `is_retained`, in sample order. It counts the rows each filter removed; a row that fails the length/start check is not translated or counted again.

`kept_seqs` holds the retained sequences and `labels` their stability values in the same order. `kept_seqs` supplies the coding-sequence inputs and groups for the stability probes, so identical strings stay together in the outer folds. Internal model selection has the separate limitation explained in the [linear-probing guide](#current-evaluation-limitation). `dataset.rows` keeps the original file index and columns of the retained rows. This sampling step does not use the CSV's `Split` column; it does not establish separate training, validation, or test sets.

The sampling step does not remove duplicate sequences. The notebook's [dataset audit](../../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) retains the full-file findings separately from this preparation method.

### Tokens and input limits

The encoder input limits are measured in tokens and recorded in the registry; the notebook's settings table lists them. The structural-proxy crop is measured in nucleotides. A token covers a different number of letters for each encoder: most Nucleotide Transformer tokens cover six nucleotides, DNABERT-2's byte-pair tokens cover a variable number, HyenaDNA, RNA-FM, and RiNALMo tokens cover one nucleotide, mRNA-FM and CaLM tokens cover one codon, and ESM-2 and ProtBERT tokens cover one amino acid. The limits of RNA-FM and RiNALMo, 1,024 tokens, include two special tokens, so each sees at most 1,022 nucleotide positions from a sequence. CaLM's limit of 1,026 tokens covers 1,024 codons, or 3,072 nucleotides, and HyenaDNA's limit of one million tokens is far above any sequence here.

### Loading an encoder

`encoders.load_encoder(encoder, device)` loads the tokenizer and pretrained encoder for one registry entry at its recorded revision, moves the model to `DEVICE`, and returns the tokenizer and model together. Importing the package does not load an encoder.

Checkpoints that load with the Transformers auto classes use them directly. Nucleotide Transformer v2 and DNABERT-2 ship their own model code, written for an earlier Transformers version; `load_encoder` builds those models from their configuration, loads the checkpoint weights, and raises an error if any weight other than the unused pooler fails to match. ProtBERT's checkpoint does not name its model type, so `load_encoder` uses the BERT tokenizer with letter case preserved and the BERT model class directly. Its `protein_input` separates residues with spaces and replaces U, Z, O, and B with X.

The [pooled-extraction loop](../../Code/shared_code/embeddings.py) loads one encoder at a time, deletes its model/tokenizer references after extraction, then calls `encoders.free_memory` for garbage collection and unused backend allocations. This is the loop's cleanup behavior, not a guarantee that no other model or live tensor occupies memory.

`model.eval()` switches layers with distinct training and evaluation behavior, such as dropout, into evaluation mode. It does not disable gradient tracking or set the model's parameters to `requires_grad=False`. The separate `@torch.no_grad()` decorator on `embed` disables gradient tracking during embedding extraction, avoiding the memory cost of recording operations for a backward pass. The encoders remain unchanged because this workflow does not perform training updates. See the PyTorch documentation for [evaluation mode](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval) and [gradient tracking](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html).

### Producing a sequence embedding

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

#### Current pooling choice

The mean includes every returned token position, including special tokens added by the tokenizer. The pooling operation does not use a separate mask to exclude positions. This documents the current representation definition; excluding special tokens or changing the weighting would change the embeddings and require a separate methodological decision.

### Embedding matrices and cache reuse

`embed_with_cache` produces one matrix per encoder, with one row per supplied sequence and one column per pooled feature. In the stability analysis, the supplied strings are `kept_seqs`, and the helper converts each coding sequence through `encoder_input` before embedding. In GTEx, the caller supplies each encoder's prepared strings with `prepared=True`. Keep input, label, and group row order aligned throughout; the cache does not align labels or groups for the caller.

The cache file is named for `encoder.key` in the selected output directory. Version 1 identifies the effective strings actually passed to the encoder, after any coding-sequence conversion:

| Recorded field | What is checked |
| --- | --- |
| `cache_version` | Version 1 of the archive and current embedding calculation; change it when the schema or pooling calculation changes. |
| `checkpoint` | The encoder's checkpoint identifier. |
| `revision` | The requested registry revision, or an empty string when no revision is pinned. |
| `max_len` | The encoder's token limit. |
| `fingerprint` | Full SHA-256 of the ordered effective strings serialized as a JSON list (UTF-8, no ASCII escaping, compact separators). |
| `n_sequences` | The number of effective strings, checked against stored matrix rows. |

The helper validates scalar metadata and a real numeric, two-dimensional matrix with positive width before considering reuse. The archive closes on a hit, mismatch, or error. Equivalent effective strings can reuse a matrix even when one caller supplies prepared strings and another requests conversion; the `prepared` flag itself is not an extra identity field.

| Cache state | Outcome of an explicit `embed_with_cache` call |
| --- | --- |
| Absent | Compute and publish a checked matrix. |
| Valid version 1, matching identity | Reuse; return `(matrix, True)`. |
| Valid version 1, changed input/checkpoint/revision/token limit | Compute a replacement; return `(matrix, False)`. |
| Legacy, incompatible version, missing metadata, or damaged matrix/archive | Raise a recovery error before loading a model; preserve the existing file. |

**Recovery.** Preserve an old or damaged file and its known producing evidence. For a deliberately authorized new computation, select a fresh cache directory in the notebook setup or archive the specific old file before retrying. Do not relabel an old archive as version 1: its effective inputs and producing code have not been established by that edit.

New matrices are written to a unique temporary sibling, reopened and checked, then published with `os.replace`. Failure before replacement preserves the last completed cache. A cleanup failure retains the primary error and, where supported by Python, adds the temporary path as an exception note. This prevents partial publication within the helper; it is not a guarantee against filesystem failure or concurrent writers to the same destination.

**Limits of reuse.** The version is a maintained calculation marker, not an automatic source-code hash. Package versions and tokenizer behavior are not separately checked. `device` is saved when computing and neither compared nor rewritten on a hit; it records the selected creation-time backend, not proof that every operation ran there. [Run records](../run-records.md) distinguish producing conditions from later reuse. An empty revision does not identify the resolved remote commit, so an unpinned checkpoint change is not detected. A matching cache is not evidence that labels, groups, biological identities, or scientific interpretations are correct.

The [Results and cache policy](../../README.md#results-caches-and-intermediate-files) identifies output locations and their Git treatment. The [notebook embedding section](../../Code/stage1/Stage1_stability.ipynb#Embedding-the-sample-with-every-encoder) keeps the matrix-shape outputs associated with the producing run. A width disagreement with its loading check is an inconsistency to investigate; width alone does not identify its cause.


#### Restarting without repeating inference

Completed caches are written per encoder. A matching `embed_with_cache` call returns its matrix before loading the tokenizer or model. Progress messages such as `hyenadna: 200/2937` report processed sequences, not a partial disk checkpoint; an interruption before that encoder finishes loses its unfinished progress. Previously completed encoder archives remain reusable. A missing cache or valid identity mismatch starts inference, so this helper is not a cache-only restore command.

A kernel restart also discards analysis objects. The current layerwise embedding and attention sections are not cached, and ordinary reruns of probes, PCA/CCA, permutations and diagnostics recompute them. The stability loading check intentionally loads models even when embedding caches exist. Avoid using Run All simply to recover figures. The [saved-result exporter](../figures/README.md) recreates the recorded graphics directly from JSON snapshots, with no kernel connection, model load or probe fit. It restores figures, not the entire notebook session or an interrupted encoder run.

## Linear probing

With a feature matrix available, the next question is what a simple fitted model can predict from it. This section explains the standardization, ridge regression, and evaluation used in the [stability notebook](../../Code/stage1/Stage1_stability.ipynb#Linear-probing) and the [GTEx expression probes](../../Code/stage1/Stage1_gtex.ipynb#Expression-prediction-with-linear-probes). Start with the respective notebook for its biological target, experiment settings, and saved results; use this section for the derivations and worked example. The maintained implementation is in [analysis.py](../../Code/shared_code/analysis.py) and the group assignment is in [splits.py](../../Code/shared_code/splits.py).

A linear probe tests whether a target can be predicted from an embedding with a simple fitted model. The main inputs are embeddings produced by the frozen encoders; the probe functions also accept simpler feature matrices, such as sequence length. Fitting a probe does not update the encoders.

For the scalar-target stability analyses described here, `analysis.probe_scores(X, y, groups=None, continuous=True, seed=42)` receives an embedding matrix `X` with shape `(n, p)` and a target array `y` with shape `(n,)`. There are `n` example rows and `p` features per row, and the targets must follow the same row order. `continuous` selects regression or classification, `groups`, when supplied, gives one group identifier per row, and `seed` selects the fold assignment; the default is the pinned assignment. It returns the five held-out fold scores and does not return a fitted model. `analysis.probe_assignment_means` repeats the evaluation over further fold assignments, and `analysis.probe_table` reports both for several embeddings and targets.

### Standardization and model choice

The private `analysis._ridge_pipeline()` constructs a fresh scaler and ridge model for each caller. Both `probe_scores` and `fit_predict` use this same definition; their cross-validation and published-split evaluation remain separate. The notebook implementation displays link to the constructor as well as the callers.

**1. Estimate feature means and scales on the outer training rows.**

`StandardScaler` is inside `Pipeline([("scale", StandardScaler()), ("model", model)])`. Each outer fold fits its own scaler on that fold's training rows. For a finite feature, its training mean is:

$$
\mu_j = \frac{1}{m}\sum_{i\in\mathcal T}x_{ij}.
$$

In this notation:

- $X$ is the complete input matrix with shape $(n,p)$; $x_{ij}$ is its scalar entry in row $i$, feature column $j$.
- $i$ identifies an example, and $j$ identifies a feature from 1 through $p$. These equations count from 1; Python counts from 0.
- $\mathcal T$ is the set of row indices used for the current outer training fold; $i\in\mathcal T$ means that row $i$ belongs to that set.
- $m$ is the number of rows in $\mathcal T$.
- $\mu_j$, read as "mu sub j," is the mean of feature $j$ on those training rows.
- $\sum_{i\in\mathcal T}$ adds the indicated feature values over training rows. Dividing by $m$ takes their average.

For a feature with nonzero variance, the scale is:

$$
s_j = \sqrt{\frac{1}{m}\sum_{i\in\mathcal T}(x_{ij}-\mu_j)^2}.
$$

In this notation:

- $s_j$ is feature $j$'s training standard deviation. The subscript identifies a feature, not a fold.
- $x_{ij}-\mu_j$ is the feature value's deviation from its training mean; the exponent $2$ squares that deviation.
- $\sqrt{\phantom{x}}$ takes the square root after averaging the squared deviations.
- $m$, $\mathcal T$, $i$, $j$ and $\mu_j$ retain the meanings above. Dividing by $m$, rather than $m-1$, matches `StandardScaler`'s `ddof=0` convention.

**2. Apply those same training statistics to training and held-out rows.**

$$
z_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
$$

In this notation:

- $z_{ij}$ is the standardized feature value for row $i$, column $j$; the standardized matrix has the same shape as its input subset.
- $x_{ij}$ may now come from a training row or a held-out row. In both cases, $\mu_j$ and $s_j$ come only from the outer training set.
- The division expresses the deviation in units of that feature's training standard deviation.

For example, training feature values `(2, 4)` give a mean of 3 and a scale of 1. A held-out value of 6 is then standardized to `(6 - 3) / 1 = 3`, using those same training statistics. These are illustrative values.

For a constant training feature, the scaler uses a scale factor of one instead of dividing by zero. These equations explain library operations performed by `StandardScaler`; the implementation does not use separate loops for them. See [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

**3. Choose the model for the target.**

- A continuous target, such as a stability measurement, uses `RidgeCV(alphas=np.logspace(-3, 5, 20))` and is evaluated with $R^2$, defined below.
- A categorical target uses `LogisticRegression(max_iter=2000)` and is evaluated with accuracy: the fraction of held-out labels predicted correctly. The equations below describe the continuous-target branch; they are not the logistic-regression objective.

For a continuous target, a candidate ridge model predicts a weighted sum of standardized features plus an intercept:

$$
\hat y_i=\beta_0+\sum_{j=1}^{p}\beta_j z_{ij}.
$$

In this notation:

- $\hat y_i$, read as "y hat sub i," is the predicted target for example $i$; the hat distinguishes a prediction from the observed target $y_i$.
- $\beta_0$, read as "beta zero," is a scalar intercept.
- $\beta_j$ is the fitted coefficient multiplying standardized feature $j$.
- $p$ is the number of features, and $\sum_{j=1}^{p}$ adds their weighted contributions. The product $\beta_j z_{ij}$ is ordinary scalar multiplication.

For a fixed penalty strength, ridge regression balances training error against coefficient size:

$$
L_\alpha(\beta_0,\boldsymbol\beta)
=\sum_{i\in\mathcal T}(y_i-\hat y_i)^2
+\alpha\sum_{j=1}^{p}\beta_j^2.
$$

In this notation:

- $L_\alpha$ names the scalar objective at penalty strength $\alpha$; it is explanatory notation, not a function defined in the implementation.
- $\boldsymbol\beta=(\beta_1,\ldots,\beta_p)$ is the coefficient vector with shape $(p,)$. Bold type denotes a vector; the ellipsis means the intervening coefficients.
- $y_i-\hat y_i$ is the prediction error for training row $i$; the first sum adds its square over $\mathcal T$.
- $\alpha$, read as "alpha," is a positive scalar controlling the penalty. The second sum adds the squared feature coefficients and multiplies their total by $\alpha$.
- The intercept $\beta_0$ is not included in the penalty. Standardizing the features makes their scales comparable when penalizing their coefficients.

The fitted coefficients for that fixed penalty minimize this objective:

$$
(\beta_0^*,\boldsymbol\beta^*)
=\underset{\beta_0,\boldsymbol\beta}{\operatorname{argmin}}
L_\alpha(\beta_0,\boldsymbol\beta).
$$

In this notation:

- $\operatorname{argmin}$ means "choose the parameter values that make the objective smallest." It returns the intercept and coefficients, not the minimum objective value.
- The quantities below $\operatorname{argmin}$ identify the parameters being varied; $\alpha$ is held fixed in this equation.
- The superscript $*$ labels the selected parameters. It does not mean multiplication.

These are the [ridge regression objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and its mathematical minimization, performed inside scikit-learn. They are not a literal `argmin` call in `probe_scores`. `RidgeCV` also chooses the penalty from the code's 20 candidates, evenly spaced on a logarithmic scale from $10^{-3}$ to $10^5$. ISLR calls this tuning parameter $\lambda$ ("lambda"); scikit-learn calls it `alpha`. ISLR, Sections 6.2.1 and 6.2.3, discuss scaling and selecting the penalty.

### Evaluation and interpretation

[DOME's recommendations for biological machine-learning validation](../sources.md#dome-biological-machine-learning-validation), especially Box 1 and Table 1, provide questions to ask about data independence, preprocessing, model selection, and evaluation. Use them to examine the procedure below and its [remaining limitation](#current-evaluation-limitation); retaining the guidance does not establish that this analysis satisfies all of its recommendations.

**1. Hold out complete sequence groups for the outer evaluation.**

The stability probes in the notebook supply `groups=kept_seqs`, or `groups=structure_groups` for the matching structure subsample. `splits.grouped_folds` assigns each sequence group to exactly one of five held-out folds from a hash of the group's string:

$$
f(g)=H(\mathrm{seed}\,{:}\,g) \bmod 5.
$$

In this notation:

- $g$ is one group identifier, here a normalized sequence string, and $f(g)$ is its held-out fold, an integer from 0 through 4.
- $\mathrm{seed}\,{:}\,g$ is the text formed by the seed, a colon, and $g$.
- $H$ is the SHA-256 hash of that text, read as a nonnegative integer. See [Python's hashlib](https://docs.python.org/3/library/hashlib.html).
- $\bmod 5$ keeps the remainder after division by five.

**Interpretation.** Identical retained sequence strings receive the same fold, so they stay together across the outer train/test boundary. With the same normalized group strings, seed, fold count, and hash procedure, a sequence keeps its fold across machines and in any sample that contains it. This fixes partition membership, not numerical equality of embeddings or scores across environments. The [hash assignment](../../Code/shared_code/splits.py) does not balance group or row counts; a fold with no rows raises an error. The earlier description of sizes varying only slightly around one fifth overstated this guarantee. This clarification does not change assignments or recompute scores.

**Seed roles.** Both notebooks explicitly pass `PROBE_SEED = 42` to single-encoder tables, concatenated-pair comparisons, and baseline probes. The public helpers retain `seed=42` as their default. Sampling, CCA, and PyTorch initialization use `SEED` separately; further probe assignments retain seeds 0 through `N_FOLD_ASSIGNMENTS - 1`. Changing the pinned seed requires corresponding single and pair results from that same partition.

The custom hash rule avoids depending on the ordering of size-tied groups in the unshuffled [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) implementation. That branch uses NumPy's default sort, which does not guarantee stable tie ordering; see [NumPy's sort documentation](https://numpy.org/doc/stable/reference/generated/numpy.sort.html). This rationale concerns that implementation branch, not every GroupKFold configuration.

For each outer fold, `cross_val_score` fits the pipeline on the other four folds and evaluates its predictions on the held-out fold. Within that outer training subset, `RidgeCV` selects an alpha using its default efficient leave-one-out procedure, with negative mean squared error as its selection score. This internal model selection differs from the outer $R^2$ evaluation. See [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html).

The fallback `groups=None` branch uses `cross_val_score(..., cv=5)`. For the target types handled here, that means unshuffled `KFold` for regression or unshuffled `StratifiedKFold` for classification. This fallback does not enforce sequence groups. See [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html).

**2. Compare held-out regression errors with a mean-target reference.**

First find the held-out fold's observed target mean:

$$
\bar y_{\mathcal F}=\frac{1}{h}\sum_{i\in\mathcal F}y_i.
$$

In this notation:

- $\mathcal F$ is the set of held-out row indices for the current outer fold, and $h$ is its number of rows.
- $y_i$ is the observed target for held-out row $i$.
- $\bar y_{\mathcal F}$, read as "y bar for fold F," is the mean of those observed targets. The bar denotes an average.
- $\sum_{i\in\mathcal F}$ adds over held-out rows; division by $h$ gives the average.

For a fold whose targets vary, its score is:

$$
R^2=1-\frac{\sum_{i\in\mathcal F}(y_i-\hat y_i)^2}
{\sum_{i\in\mathcal F}(y_i-\bar y_{\mathcal F})^2}.
$$

In this notation:

- $R^2$ names the coefficient of determination; the superscript is part of its conventional name, not a direction to square an already calculated score.
- $\hat y_i$ is the prediction made without fitting on fold $\mathcal F$.
- The numerator adds squared prediction errors; the denominator adds squared errors for the reference that predicts the held-out mean $\bar y_{\mathcal F}$.
- Subtracting their ratio from one gives a score of one for perfect prediction and zero when the two error totals match.

The fold mean is used to define the score, not to train the model. A negative score means the model's errors exceed that reference error. Held-out $R^2$ can therefore be negative even though ISLR's introductory discussion considers a fitted least-squares model with an intercept. Constant targets require separate handling; see [scikit-learn's scoring documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

**3. Summarize the five outer scores.**

`scores.mean()` gives their arithmetic mean. `scores.std()` uses `ddof=0`, so it divides the sum of squared deviations by five before taking a square root. The reported mean and fold standard deviation describe variation across these folds. It is not a confidence interval or a standard error, and the mean gives each fold equal weight even if their row counts differ.

**4. Repeat the evaluation with further fold assignments.**

One assignment is a single partition of the groups, and on a sample of this size the five-fold mean can change noticeably with which sequences share a fold. `analysis.probe_assignment_means` therefore repeats steps 1 to 3 with seeds 0 through `N_FOLD_ASSIGNMENTS - 1`, and `probe_table` reports the mean and standard deviation (`ddof=0`) of the resulting five-fold means. The pinned result is the one reported and compared across analyses and stages; the summary shows how much it depends on the partition. Every assignment reuses the same rows, so the summary is not a confidence interval or a standard error.

A probe assesses information accessible to this model under this evaluation procedure. A weak score alone does not establish that the embedding contains no useful information.

### Current evaluation limitation

The current pipeline prevents outer held-out rows from setting the scaler statistics, and grouping prevents identical sequence strings from crossing the outer train/test boundary. See scikit-learn's [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

The internal alpha-selection procedure has a narrower boundary: the scaler is fit once on the whole outer training subset before `RidgeCV` performs its internal leave-one-out calculation, and this code does not supply sequence groups to that internal calculation. This follows from the helper's ordering and the documented [Pipeline fitting sequence](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit). Thus preprocessing is not refit for each internal held-out row, and internal model selection is not grouped by sequence. Outer held-out rows remain excluded from fitting; the pipeline should not be described as refitting preprocessing and enforcing groups at every level of cross-validation. The numerical effect of changing the internal procedure has not been measured here.

Grouping addresses exact duplicate strings, not independence among related sequences. The probes do not deduplicate rows. The [notebook's evaluation limitation](../../Code/stage1/Stage1_stability.ipynb#Current-evaluation-limitation) retains the dataset counts and unresolved provenance questions associated with its saved analysis.

### Statistical and course references

- James, Witten, Hastie, and Tibshirani, *An Introduction to Statistical Learning with Applications in R*, second edition: Section 3.1.3, p. 70 ($R^2$); Sections 5.1.2-5.1.3, pp. 200 and 203-204 (leave-one-out and k-fold cross-validation); Section 6.2.1, pp. 237-239 (ridge regression and scaling); Section 6.2.3, p. 250 (tuning). The original explanation records consultation of the local June 2023 corrected PDF. The [authors' book site](https://www.statlearning.com/) provides the book.
- UCF Machine Learning, *Week 04_Validation*, slides 5, 7, and 9: separating fitting, model selection, and evaluation, and rotating the held-out fold.
- UCF Machine Learning, *Data Preprocessing for Machine Learning*, slide 5: the role of feature scaling.
- UCF Capstone general lecture, *Overfitting and Regularization* (Yousefi, 2020), slides 19-20: the ridge penalty and selecting its strength by cross-validation.

The course decks provide explanatory context; linked library documentation explains the API behavior, and the [notebook](../../Code/stage1/Stage1_stability.ipynb#Linear-probing) and [probe implementation](../../Code/shared_code/analysis.py) specify the arguments used. Library documentation alone does not identify the environment or producing run of a saved result.

## Calculated targets and controls

The same fitting procedure can answer different questions depending on what supplies the features and what supplies the target. A dataset label such as stability is distinct from a sequence property calculated directly from its string. The sections below retain the Stage 1 calculations and their notebook links.

| Analysis | Input to the calculation or predictor | Target or reported quantity | Question it answers |
| --- | --- | --- | --- |
| Sequence-length baseline | One length feature per sequence | Supplied stability label, predicted with ridge regression | How much can length alone predict? |
| GC and GC3 prediction | One encoder's pooled embedding per sequence | Calculated GC or GC3 fraction, predicted with ridge regression | Is nucleotide composition linearly recoverable from the embedding? |
| Nussinov pairing-fraction probe | Pooled embeddings for selected rows | Calculated pairing fraction of a sequence prefix, predicted with ridge regression | Is this simplified structural proxy linearly recoverable? |
| Synonymous-recoding check | Each encoder's embedding matrix for the mRFP variants | Mean distance from that matrix's centroid; no fitted predictor or target label | Does the translation and embedding pipeline behave as expected when protein sequence is constant? |

A calculated quantity can change roles between analyses. GC and GC3 are prediction **targets** in the Stage 1 probes below. In the [dedicated mRFP notebook](../../Code/stage1/mrfp_representation_control.ipynb), selected composition and folding summaries are **input features** used to predict expression and assess what embeddings add beyond those controls. The Stage 1 recoding-spread calculation does not fit expression or evaluate predictive accuracy. Its shared dataset does not make it the same experiment as the separate expression analysis.

### Sequence-length baseline

See the [notebook calculation](../../Code/stage1/Stage1_stability.ipynb#Sequence-length-baseline).

This baseline predicts stability from one feature: the full input-sequence length in nucleotides. Comparing it with the encoder probes asks how much predictive value is available from this simple sequence property alone. The inputs are the retained sequences and their stability `labels`.

`dataset.lengths` measures each retained sequence after the normalization and filters described in [Sampling and retained rows](#sampling-and-retained-rows). For each retained row, the feature is:

$$
X_{i1} = |s_i|.
$$

**In this notation:**

- $n$ is the number of retained examples.
- $i$ identifies a retained example, from 1 through $n$ in the equation; Python indexes the corresponding rows from 0 through `n - 1`.
- $s_i$ is that example's normalized nucleotide string.
- $|s_i|$ means the number of characters in that string, measured here as nucleotide-sequence length. These bars indicate string length, not a vector norm.
- $X$ is the length-feature matrix, stored as `seq_lengths`, with shape `(n, 1)`.
- $X_{i1}$ is the value in row $i$ of its only feature column. The subscript 1 identifies that column; Python calls it column 0.

For example, `ATG | GCC | AAA | GCT | TTT` has five codons and a length feature of 15. The spaces and separators only show the codons; they are not part of the input string. Translation determines whether a row is retained but does not shorten the sequence being measured. This is the full nucleotide length, rather than translated protein length, token count, or the truncated length seen by an encoder.

`Dataset.lengths` uses Python's `len` for each retained sequence, and NumPy's `.reshape(-1, 1)` makes the lengths a single-column feature matrix. Its rows follow `kept_seqs`, so they correspond to `labels` by construction.

The notebook call reuses `probe_table`: it fits ridge regression and reports the mean and standard deviation of five outer-fold $R^2$ scores on the pinned folds, then the summary over further fold assignments. Passing `groups=kept_seqs` groups exact repeated sequence strings within the same outer fold. See [Linear probing](#linear-probing) for the model and evaluation procedure, and the [current evaluation limitations](#current-evaluation-limitation) for interpretation. The length calculation is deterministic and needs no additional seed.

Implementation: [Dataset.lengths](../../Code/shared_code/datasets.py).

### GC-content prediction

See the [notebook calculation](../../Code/stage1/Stage1_stability.ipynb#GC-content-prediction).

These probes ask how well each encoder's embedding predicts the GC fraction of its source nucleotide sequence. The inputs are the matched embedding matrices and the corresponding strings in `kept_seqs`. The target is calculated directly from each string; it is not the dataset's stability label.

For one nonempty sequence, add the G and C counts and divide by its full length:

$$
\mathrm{GC}(s) = \frac{N_G(s) + N_C(s)}{L(s)}.
$$

**In this notation:**

- $s$ is one normalized, uppercase nucleotide string from `kept_seqs`.
- $n$ is the number of retained examples; the resulting target array `seq_gc` has shape `(n,)`.
- $N_G(s)$ counts the G characters in that string; $N_C(s)$ counts the C characters. The subscripts identify the base being counted, not positions in the sequence.
- $L(s)$ is the number of characters in the full string, measured here as nucleotide-sequence length.
- $\mathrm{GC}(s)$ is the resulting GC fraction: one scalar between zero and one.
- The parentheses mean that the count, length, or fraction is calculated for sequence $s$. Adding the two counts gives the number of G-or-C positions; dividing by the length gives their fraction of all positions.

For example, `GCAU` contains two G-or-C bases among four nucleotides, giving $2/4 = 0.5$, or 50% GC. This illustrates `gc_content`; the four-base string would not pass the dataset's retention filters.

**Connection to the code.** `sequences.gc_content` implements the equation using Python string `.count()` and `len()`. It expects a nonempty uppercase string and does not independently validate the nucleotide alphabet. All characters contribute to the denominator, while only uppercase G and C contribute to the numerator. The calculation uses the full retained sequence, including any sequence after the first translated stop; it is not cropped to an encoder's input limit.

`kept_seqs` also supplies the notebook's GC3, structure, and attention analyses, and every embedding matrix follows its order.

The list comprehension applies `gc_content` once per retained string, and NumPy stores those values as `seq_gc` with shape `(n,)`, where `n` is the number of retained examples. `probe_table` fits one probe per encoder in `EMBEDDINGS` with this same target array, and `groups=kept_seqs` keeps exact repeated sequence strings within the same outer fold. See [Linear probing](#linear-probing) for the ridge model, cross-validation, and ISLR references. Computing GC content is deterministic and needs no additional seed. The [current evaluation limitations](#current-evaluation-limitation) still apply.

Implementation: [sequence helpers](../../Code/shared_code/sequences.py).

### GC3 prediction

See the [notebook calculation](../../Code/stage1/Stage1_stability.ipynb#GC3-prediction).

This probe asks whether the embeddings make third-position nucleotide composition linearly recoverable. Its inputs are the matched embedding matrices and the corresponding strings in `kept_seqs`. GC3 is the fraction of complete codons whose third base is G or C. It is related to synonymous codon usage, but it is not a complete measure of codon bias, codon optimality, or translation efficiency. There is no universal 50% GC3 reference against which bias can be declared.

RNA codons `GCU` and `GCC` both specify alanine in the standard code. Replacing the first with the second changes the third base from U to C while preserving that amino acid. This redundancy is called genetic-code degeneracy; it explains how third-position composition can vary without necessarily changing the protein sequence. Cells decode codons through transfer RNAs (tRNAs), which use three-base recognition sequences called anticodons to pair with mRNA codons and carry the corresponding amino acids. Some tRNAs recognize more than one codon through flexible pairing at the third position of the codon. GC3 counts bases rather than measuring that decoding process, so a higher GC3 value alone does not establish faster translation. [Essential Cell Biology, 6th ed., ch. 7, “From RNA to Protein”](../sources.md#essential-cell-biology-6th-edition).

Group the sequence into codons starting at its first base, count the G and C bases at third positions, and divide by the number of complete codons:

$$
\mathrm{GC3}(s)
= \frac{N_{G,3}(s) + N_{C,3}(s)}{N_{\mathrm{codons}}(s)}.
$$

**In this notation:**

- $s$ is one normalized, uppercase nucleotide string from `kept_seqs`.
- $n$ is the number of retained examples; the resulting target array `seq_gc3` has shape `(n,)`.
- $N_{G,3}(s)$ counts complete codons in $s$ whose third base is G; $N_{C,3}(s)$ counts those whose third base is C.
- The G or C in each subscript identifies the base; the 3 identifies its position within a codon. It is not an exponent.
- $N_{\mathrm{codons}}(s)$ is the number of complete groups of three bases, starting from the sequence's first base. An incomplete trailing codon is excluded.
- $\mathrm{GC3}(s)$ is the resulting fraction: one scalar between zero and one, defined here for a sequence with at least one complete codon.
- The parentheses identify the sequence being measured. The numerator adds the two third-position counts; the denominator counts all complete codons, each of which contributes one third base.

For example, `AUG | GCC | AAA` has third bases `G, C, A`, so GC3 is $2/3$, approximately 0.667. Separators only show codon boundaries. This is an example of the helper calculation; the translated sequence is too short to pass the dataset's retention filters.

**Connection to the code.** `sequences.gc3_content` selects the third bases using Python slicing, then calls `sequences.gc_content`. It does not call a separate codon-bias library. The table names the selected string `third_positions` for explanation; the helper itself directly returns `gc_content(seq[2::3])`.

| Code operation | Result | Connection to the calculation |
|---|---|---|
| `third_positions = seq[2::3]` | One string containing third-position bases | Python indices 2, 5, 8, and so on correspond to nucleotide positions 3, 6, 9. |
| `gc_content(third_positions)` | One scalar fraction | Count G and C in that string and divide by its length, which is the number of complete codons. |
| `np.array([gc3_content(s) for s in kept_seqs])` | Target array `seq_gc3`, shape `(n,)` | Calculate one value for each of the `n` retained examples, preserving their order. |

An incomplete trailing codon has no third base to select under [Python's slicing rules](https://docs.python.org/3/builtins/stdtypes.html#common-sequence-operations). The helper expects an uppercase sequence with at least one complete codon and adds no normalization or alphabet validation.

The calculation uses the full retained input sequence, including complete codons after any translated stop or beyond the encoder's input limit. `kept_seqs` supplies the same retained order as in the [GC-content calculation](#gc-content-prediction); the earlier filters do not establish experimentally verified complete coding sequences. The probes reuse the existing [linear-probe procedure](#linear-probing), with `groups=kept_seqs` keeping exact repeated sequence strings within the same outer fold. The GC3 calculation is deterministic and needs no additional seed.

GC3 should also be distinguished from other codon-usage measures. The Codon Adaptation Index uses a reference set, whereas measures such as [Wright's effective number of codons (1990)](https://pubmed.ncbi.nlm.nih.gov/2110097/) can be calculated from a sequence alone. The notebook calculation uses GC3 only.

Finally, NumPy's `np.corrcoef(seq_gc, seq_gc3)` calculates the [Pearson correlation](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html) between the two target arrays across retained examples. It returns a `(2, 2)` correlation matrix; `[0, 1]` selects the entry comparing overall GC with GC3, using Python's zero-based indexing. This describes their linear association, not a probe $R^2$ or representation-alignment score. Strong association indicates overlap between the composition targets; it does not establish whether GC3 adds predictive value beyond overall GC. Likewise, the notebook table's difference between GC3 and GC probe $R^2$ values compares separate target predictions; it does not fit a model testing incremental value conditional on overall GC. The [current evaluation limitations](#current-evaluation-limitation) still apply to the probes.

Implementation: [sequence helpers](../../Code/shared_code/sequences.py).

### Secondary-structure proxy: Nussinov pairing fraction

See the [notebook calculation](../../Code/stage1/Stage1_stability.ipynb#Secondary-structure-proxy:-Nussinov-pairing-fraction).

An RNA strand can fold back so that complementary regions pair with one another. Adjacent paired bases form stems, while unpaired stretches form loops or bulges. The pattern of these interactions is its secondary structure; the RNA can also adopt a more complex three-dimensional arrangement. Base stacking and other interactions help stabilize these structures, so counting permitted pairs captures only part of folding. [Lehninger Principles of Biochemistry, 8th ed., §8.2, “Many RNAs Have More Complex Three-Dimensional Structures”](../sources.md#lehninger-principles-of-biochemistry-8th-edition).

A folding free-energy difference describes how favorable one RNA conformation is relative to another under specified conditions. It accounts for molecular interactions and entropy, which is related to the number of molecular arrangements accessible to the RNA and its surroundings. Lower free energy favors a state at equilibrium; it does not by itself identify the structure present in a living cell. **Structural stability** in this sense differs from **mRNA stability**, which concerns persistence before degradation. RNA structure can affect access by cellular molecules, but a favorable folding energy does not by itself determine an RNA's lifetime. The Nussinov score below measures neither quantity directly: it optimizes a permitted pair count. [Lehninger Principles of Biochemistry, 8th ed., §13.1](../sources.md#lehninger-principles-of-biochemistry-8th-edition); [Molecular Biology of the Cell, 7th ed., ch. 7, pp. 459–460](../sources.md#molecular-biology-of-the-cell-7th-edition).

This probe predicts a simplified folding score: the largest fraction of nucleotides that can participate in permitted, noncrossing pairs within a sequence prefix. Its targets come from cropped strings in `kept_seqs`; its features are the already computed DNA, RNA, or protein embeddings for the same selected rows. The Nussinov-style calculation maximizes a pair count. It does not estimate thermodynamic free energy or measure an experimental structure. See [Nussinov and Jacobson (1980)](https://pubmed.ncbi.nlm.nih.gov/6161375/).

**Pairing rules.** `sequences.can_pair` accepts A-U, G-C, and G-U in either order, plus A-T and G-T for DNA-letter inputs. Each position can participate in at most one pair. Nested pairs are allowed; crossing pairs are excluded. Inputs are already uppercase in this pipeline; the helpers do not independently normalize or validate the sequence.

A permitted pair must leave enough positions between its two bases:

$$
k - i - 1 \geq m.
$$

**In this notation:**

- $n$ is the length of the sequence prefix passed to `nussinov_pairing_fraction`.
- $i$ and $k$ identify positions in that prefix, with $k$ after $i$. These equations use Python's zero-based positions, from 0 through $n-1$.
- $k-i-1$ counts positions strictly between the two proposed partners.
- $m$ is `min_loop`, expected to be a nonnegative integer and defaulting to 3.
- $\geq$ means "greater than or equal to": the pair must enclose at least $m$ positions. Those positions need not all be unpaired; nested pairs may occupy the enclosed interval.

**Dynamic program.** `sequences.nussinov_pairing_fraction` implements the calculation directly with loops and a table of solved intervals. It does not call a folding library. Shorter intervals are solved before longer ones so their counts are available when needed.

Let $D(i,j)$ be the largest permitted pair count in the inclusive interval from position $i$ through $j$. The first choice is to leave $i$ unpaired. The other choices pair $i$ with an eligible position $k$, then combine the best counts inside that pair and after it. When at least one eligible partner exists, the comparison is:

$$
D(i,j)
= \max\left\{
D(i+1,j),
\max_{k \in \mathcal{K}(i,j)}
\left[1 + D(i+1,k-1) + D(k+1,j)\right]
\right\}.
$$

**In this notation:**

- $j$ is the final position of the interval, also using zero-based indexing.
- $D(i,j)$ is one nonnegative integer, stored as `max_pairs[i][j]`. The full `max_pairs` table is a Python list of lists with `n` rows and `n` columns.
- $\mathcal{K}(i,j)$ is the set of eligible partners for position $i$: positions $k$ no later than $j$ that satisfy the minimum separation and for which `can_pair(seq[i], seq[k])` is true.
- $k \in \mathcal{K}(i,j)$ means that $k$ belongs to that set.
- $\max$ selects the largest count among the alternatives. The inner maximum considers each eligible partner; the outer maximum compares the best such pairing with leaving $i$ unpaired.
- $D(i+1,j)$ is the best count after leaving position $i$ unpaired.
- Inside the brackets, 1 counts the new pair between $i$ and $k$, $D(i+1,k-1)$ counts the best pairs inside it, and $D(k+1,j)$ counts the best pairs after it.
- Parentheses specify interval endpoints; brackets and braces group the quantities being added or compared. They are not additional arrays.

If there is no eligible partner, only the unpaired choice applies. Empty intervals and intervals too short for an eligible pair contribute zero. Combining separate inside and remaining intervals prevents crossings and reuse of a position. The routine finds the optimal count but does not reconstruct a list of pairs.

The recurrence is carried out by these operations:

| Code operation | Role in the recurrence |
|---|---|
| The `span` loop and `j = i + span` | Visit intervals in increasing endpoint distance, so shorter intervals are already solved. |
| `best_pair_count = max_pairs[i + 1][j]` | Start with the option that leaves position $i$ unpaired. |
| The `k` loop and `can_pair(seq[i], seq[k])` | Consider eligible partners with the required separation and base-pair rule. |
| `inside_pairs + 1 + remaining_pairs` | Add the best inner count, the new pair, and the best remaining count. The conditional expressions supply zero for empty intervals. |
| The comparison and `max_pairs[i][j] = best_pair_count` | Retain the largest candidate and store the answer for the interval. |

**Pairing fraction.** For a nonempty input, the entry covering its entire prefix gives the maximum pair count:

$$
P_{\max} = D(0,n-1).
$$

**In this notation:**

- $P_{\max}$ is the largest permitted number of pairs in the supplied prefix; the subscript "max" labels the optimal count.
- $D(0,n-1)$ covers all prefix positions, from the first at 0 to the last at $n-1$.

Each pair uses two nucleotides. The returned fraction is therefore:

$$
f_{\mathrm{paired}} = \frac{2P_{\max}}{n}.
$$

**In this notation:**

- $f_{\mathrm{paired}}$ is the paired-nucleotide fraction, a scalar between zero and one. The subscript labels what the fraction measures.
- $2P_{\max}$ is the number of paired nucleotide positions; division by $n$ compares it with all positions in the supplied prefix.

For example, `AAAAU` with `min_loop=3` permits positions 0 and 4 to pair. They enclose $4-0-1=3$ positions, meeting the separation rule. The best count is one pair, so the returned fraction is $2/5=0.4$. This illustrates the folding helper, not a sequence that passes the dataset's retention filters. Inputs too short to contain an eligible pair, including an empty input, return zero before the division.

**Sampling and probe inputs.** The calculation takes $O(n^3)$ time and $O(n^2)$ memory. The $O$, read as “big O,” describes an upper bound on how resource use grows with prefix length $n$, rather than an exact time or byte count. Here the loops give cubic worst-case work and the stored table grows quadratically. The current settings sample 200 retained row indices without replacement and score up to the first 200 nucleotides of each selected sequence. Distinct row indices need not contain distinct sequences. The [sampling call](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html) requires at least the requested number of retained rows.

`default_rng(SEED)` uses the existing seed. Repeating the selection assumes unchanged retained rows, row order, settings, and compatible software behavior. Sorting the selected indices restores retained input order. The same indices select rows from every embedding matrix, keeping them aligned with the one-dimensional `pairing_scores` target array. `structure_groups` contains the selected full sequence strings and keeps exact repeated strings within the same outer probe fold.

Only the target calculation uses the cropped prefix. The notebook cell reuses existing embeddings for those rows; it does not re-encode the prefixes. A supplied prefix is not necessarily a verified 5′ UTR or the full transcript. The probes use the existing [linear-probe procedure](#linear-probing) and its [evaluation limitations](#current-evaluation-limitation).

**Minimum-separation rule.** The candidate-pair loop enforces the minimum separation for every pair, as the recurrence above requires.

Implementation: [sequence helpers](../../Code/shared_code/sequences.py).

### Pipeline control: synonymous recoding

See the [notebook calculation](../../Code/stage1/Stage1_stability.ipynb#Pipeline-control:-synonymous-recoding).

The `mRFP_Expression` control asks whether the pipeline behaves as expected when different nucleotide sequences encode the same protein. A synonymous recoding changes codons while retaining the amino-acid sequence. Identical translated strings should therefore give every protein encoder the same input, while distinct nucleotide strings can give the DNA and RNA encoders different inputs, subject to their input limits.

Preserving the amino-acid sequence does not preserve every possible RNA interaction: changed nucleotides may participate in different base pairs. RNA-level regulation can therefore matter even when the encoded protein sequence is unchanged. Here, the spread calculation checks the representation consequence of those input differences; it does not identify the mechanism behind the variants' expression measurements. The [mRFP target explanation](data-and-targets.md#mrfp-fluorescence-to-the-value-column) follows the separate measurement question. [Essential Cell Biology, 6th ed., ch. 8, “Post-Transcriptional Controls”](../sources.md#essential-cell-biology-6th-edition).

The code loads existing sequences from `CONTROL_DATASET`; it does not generate synonymous recodings. It uses `datasets.load_dataset`, which removes rows with missing sequences, samples `N_ROWS` rows with `SEED`, and applies the sequence filters. It then counts distinct translated protein strings among the retained rows, embeds the rows with every encoder, and examines each encoder's matrix separately to measure how spread out its rows are.

**1. Find the centroid: the average embedding.**

Treat each embedding as a point whose coordinates are its numerical features. The centroid is the average of those points, calculated one feature at a time:

$$
\bar{\mathbf z}
= \frac{1}{n}\sum_{i=1}^{n}\mathbf z_i.
$$

**In this notation:**

- $Z$ denotes the control embedding matrix of the encoder being examined, `CONTROL_EMBEDDINGS[key]`. Its shape is `(n, d)`.
- $n$ is the number of retained control rows in that matrix.
- $d$ is the number of embedding features in each row; the encoders have different feature counts.
- $i$ identifies an example row, from 1 through $n$ in the equation. Python indexes those same rows from 0 through `n - 1`.
- $\mathbf z_i$, read as "bold z sub i," is the embedding vector in row $i$. Bold symbols here represent vectors with $d$ coordinates.
- $\bar{\mathbf z}$, read as "z bar," is the centroid vector. The bar denotes the average across rows.
- $\sum_{i=1}^{n}$ means to add the row vectors coordinate by coordinate. Dividing by $n$ takes the average for each feature.

**2. Measure each embedding's distance from the centroid.**

Subtract the centroid from a row, then measure the length of that difference vector:

$$
r_i = \left\|\mathbf z_i-\bar{\mathbf z}\right\|_2.
$$

**In this notation:**

- $r_i$ is a single nonnegative number: the distance of row $i$ from the centroid.
- $\mathbf z_i-\bar{\mathbf z}$ subtracts the centroid's value from each corresponding feature in that row.
- $\|\cdot\|_2$ denotes Euclidean length: square each coordinate of the vector inside the bars, add the squares, and take the square root. The subscript 2 identifies this choice of norm; it does not mean the embedding has two features.

An embedding at the centroid has distance zero. A larger distance means that embedding lies farther from the average within this representation space.

**3. Average the distances to obtain one spread value.**

$$
\operatorname{spread}(Z)
= \frac{1}{n}\sum_{i=1}^{n}r_i.
$$

**In this notation:**

- $\operatorname{spread}(Z)$ names the mean distance for matrix $Z$. It is one scalar, not another embedding vector or a variance.
- $n$, $i$ and the summation have the same meanings as above; this time the quantities being averaged are the scalar distances $r_i$.

**Connection to the code.**

These equations describe the NumPy calculation in `analysis.centroid_spread`; there is no fitted model.

For one encoder's matrix `Z`, the calculation performs the following operations in order.

| Code operation | Input shape → output shape | Connection to the mathematics |
|---|---|---|
| `Z.mean(axis=0)` | `(n, d)` → `(d,)` | Average down the example rows to obtain the centroid $\bar{\mathbf z}$. |
| `Z - Z.mean(axis=0)` | `(n, d)` → `(n, d)` | Subtract that same centroid from every row. |
| `np.linalg.norm(..., axis=1)` | `(n, d)` → `(n,)` | Combine the feature differences within each row into its distance $r_i$. |
| The final `.mean()` | `(n,)` → scalar | Average the distances to obtain $\operatorname{spread}(Z)$. |

The `...` denotes the centered matrix from the preceding row. With `axis=1`, `np.linalg.norm` calculates [Euclidean length](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) across each row's features.

**Small worked example.**

Suppose there are two embeddings, $(1,2)$ and $(5,2)$, so $n=2$ and $d=2$.

1. Their centroid is $(3,2)$: each coordinate is the average of that feature's two values.
2. Subtracting the centroid gives $(-2,0)$ and $(2,0)$. Each difference vector has Euclidean length 2.
3. The spread is therefore $(2+2)/2=2$.

These are illustrative numbers, not encoder outputs.

**What this control can establish.**

Identical vectors have zero spread in exact arithmetic. Near-zero protein spread together with one distinct translated protein is consistent with the expected behavior for synonymous recodings; nonzero DNA and RNA spread shows variation among the nucleotide representations. Compare each matrix with its own centroid. Because the spaces have different dimensions and scales, their raw spread values do not form a normalized comparison of information content. This control checks this particular translation/embedding relationship, not every assumption or analysis elsewhere in the notebook.

Implementation: [centroid_spread](../../Code/shared_code/analysis.py).
