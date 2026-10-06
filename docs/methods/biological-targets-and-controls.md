# Biological targets and controls

These methods answer different questions. The sequence-length baseline predicts the dataset's stability label. GC, GC3, and Nussinov pairing fraction are targets calculated from nucleotide strings, used to ask what the embeddings make recoverable. The synonymous-recoding control checks a translation/embedding relationship through descriptive spread, without fitting a probe.

This guide retains the definitions, mathematical development, worked examples, and implementation connections. The [stability notebook](../../Code/Stage1_stability.ipynb) owns the calls, saved outputs, and result interpretation; each section links directly back to its calculation.

| Reader question | Method |
| --- | --- |
| How much does sequence length alone predict stability? | [Sequence-length baseline](#sequence-length-baseline) |
| Can embeddings predict overall nucleotide composition? | [GC-content prediction](#gc-content-prediction) |
| Can they predict composition at third codon positions? | [GC3 prediction](#gc3-prediction) |
| Can they predict a simplified maximum-pairing score? | [Nussinov pairing fraction](#secondary-structure-proxy-nussinov-pairing-fraction) |
| Do synonymous inputs behave as expected after translation and embedding? | [Synonymous-recoding control](#pipeline-control-synonymous-recoding) |

The first four methods reuse the [linear-probe procedure](linear-probing.md), including its [evaluation limitations](linear-probing.md#current-evaluation-limitation). Sequence retention and encoder inputs are explained in [From sequences to embeddings](inputs-and-embeddings.md).

## Sequence-length baseline

Return to the [notebook calculation](../../Code/Stage1_stability.ipynb#Sequence-length-baseline).

This baseline predicts stability from one feature: the full input-sequence length in nucleotides. Comparing it with the encoder probes asks how much predictive value is available from this simple sequence property alone. The inputs are the retained sequences and their stability `labels`.

`dataset.lengths` measures each retained sequence after the normalization and filters described in [Sampling and retained rows](inputs-and-embeddings.md#sampling-and-retained-rows). For each retained row, the feature is:

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

The notebook call reuses `probe_table`: it fits ridge regression and reports the mean and standard deviation of five outer-fold $R^2$ scores on the pinned folds, then the summary over further fold assignments. Passing `groups=kept_seqs` groups exact repeated sequence strings within the same outer fold. See [Linear probing](linear-probing.md) for the model and evaluation procedure, and the [current evaluation limitations](linear-probing.md#current-evaluation-limitation) for interpretation. The length calculation is deterministic and needs no additional seed.

Implementation: [Dataset.lengths](../../Code/shared_code/datasets.py).

## GC-content prediction

Return to the [notebook calculation](../../Code/Stage1_stability.ipynb#GC-content-prediction).

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

The list comprehension applies `gc_content` once per retained string, and NumPy stores those values as `seq_gc` with shape `(n,)`, where `n` is the number of retained examples. `probe_table` fits one probe per encoder in `EMBEDDINGS` with this same target array, and `groups=kept_seqs` keeps exact repeated sequence strings within the same outer fold. See [Linear probing](linear-probing.md) for the ridge model, cross-validation, and ISLR references. Computing GC content is deterministic and needs no additional seed. The [current evaluation limitations](linear-probing.md#current-evaluation-limitation) still apply.

Implementation: [sequence helpers](../../Code/shared_code/sequences.py).

## GC3 prediction

Return to the [notebook calculation](../../Code/Stage1_stability.ipynb#GC3-prediction).

This probe asks whether the embeddings make third-position nucleotide composition linearly recoverable. Its inputs are the matched embedding matrices and the corresponding strings in `kept_seqs`. GC3 is the fraction of complete codons whose third base is G or C. It is related to synonymous codon usage, but it is not a complete measure of codon bias, codon optimality, or translation efficiency. There is no universal 50% GC3 reference against which bias can be declared.

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

The slice's first 2 is a zero-based starting index; the 3 is the step between selected indices. Its omitted stop means to continue to the end of the string. An incomplete trailing codon has no third base to select. See [Python's slicing rules](https://docs.python.org/3/builtins/stdtypes.html#common-sequence-operations). The helper expects an uppercase sequence with at least one complete codon and adds no normalization or alphabet validation.

The calculation uses the full retained input sequence, including complete codons after any translated stop or beyond the encoder's input limit. `kept_seqs` supplies the same retained order as in the [GC-content calculation](#gc-content-prediction); the earlier filters do not establish experimentally verified complete coding sequences. The probes reuse the existing [linear-probe procedure](linear-probing.md), with `groups=kept_seqs` keeping exact repeated sequence strings within the same outer fold. The GC3 calculation is deterministic and needs no additional seed.

GC3 should also be distinguished from other codon-usage measures. The Codon Adaptation Index uses a reference set, whereas measures such as [Wright's effective number of codons (1990)](https://pubmed.ncbi.nlm.nih.gov/2110097/) can be calculated from a sequence alone. The notebook calculation uses GC3 only.

Finally, NumPy's `np.corrcoef(seq_gc, seq_gc3)` calculates the [Pearson correlation](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html) between the two target arrays across retained examples. It returns a `(2, 2)` correlation matrix; `[0, 1]` selects the entry comparing overall GC with GC3, using Python's zero-based indexing. This describes their linear association, not a probe $R^2$ or representation-alignment score. Strong association indicates overlap between the composition targets; it does not establish whether GC3 adds predictive value beyond overall GC. Likewise, the notebook table's difference between GC3 and GC probe $R^2$ values compares separate target predictions; it does not fit a model testing incremental value conditional on overall GC. The [current evaluation limitations](linear-probing.md#current-evaluation-limitation) still apply to the probes.

Implementation: [sequence helpers](../../Code/shared_code/sequences.py).

## Secondary-structure proxy: Nussinov pairing fraction

Return to the [notebook calculation](../../Code/Stage1_stability.ipynb#Secondary-structure-proxy:-Nussinov-pairing-fraction).

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

Only the target calculation uses the cropped prefix. The notebook cell reuses existing embeddings for those rows; it does not re-encode the prefixes. A supplied prefix is not necessarily a verified 5′ UTR or the full transcript. The probes use the existing [linear-probe procedure](linear-probing.md) and its [evaluation limitations](linear-probing.md#current-evaluation-limitation).

**Minimum-separation rule.** The candidate-pair loop enforces the minimum separation for every pair, as the recurrence above requires.

Implementation: [sequence helpers](../../Code/shared_code/sequences.py).

## Pipeline control: synonymous recoding

Return to the [notebook calculation](../../Code/Stage1_stability.ipynb#Pipeline-control:-synonymous-recoding).

The `mRFP_Expression` control asks whether the pipeline behaves as expected when different nucleotide sequences encode the same protein. A synonymous recoding changes codons while retaining the amino-acid sequence. Identical translated strings should therefore give every protein encoder the same input, while distinct nucleotide strings can give the DNA and RNA encoders different inputs, subject to their input limits.

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

These equations describe the NumPy calculation in `analysis.centroid_spread`. There is no fitted model. The mathematical names for the centroid and individual distances explain intermediate results that the function calculates in one expression.

For one encoder's matrix `Z`, the calculation performs the following operations in order.

| Code operation | Input shape → output shape | Connection to the mathematics |
|---|---|---|
| `Z.mean(axis=0)` | `(n, d)` → `(d,)` | Average down the example rows to obtain the centroid $\bar{\mathbf z}$. |
| `Z - Z.mean(axis=0)` | `(n, d)` → `(n, d)` | Subtract that same centroid from every row. |
| `np.linalg.norm(..., axis=1)` | `(n, d)` → `(n,)` | Combine the feature differences within each row into its distance $r_i$. |
| The final `.mean()` | `(n,)` → scalar | Average the distances to obtain $\operatorname{spread}(Z)$. |

The `...` in the table stands for the centered matrix from the preceding row. NumPy applies the centroid subtraction to every row automatically. With `axis=1` and the default norm setting, `np.linalg.norm` calculates Euclidean length across each row's features; see [NumPy's norm definition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).

**Small worked example.**

Suppose there are two embeddings, $(1,2)$ and $(5,2)$, so $n=2$ and $d=2$.

1. Their centroid is $(3,2)$: each coordinate is the average of that feature's two values.
2. Subtracting the centroid gives $(-2,0)$ and $(2,0)$. Each difference vector has Euclidean length 2.
3. The spread is therefore $(2+2)/2=2$.

These are illustrative numbers, not encoder outputs.

**What this control can establish.**

Identical vectors have zero spread in exact arithmetic. Near-zero protein spread together with one distinct translated protein is consistent with the expected behavior for synonymous recodings; nonzero DNA and RNA spread shows variation among the nucleotide representations. Compare each matrix with its own centroid. Because the spaces have different dimensions and scales, their raw spread values do not form a normalized comparison of information content. This control checks this particular translation/embedding relationship, not every assumption or analysis elsewhere in the notebook.

Implementation: [centroid_spread](../../Code/shared_code/analysis.py).
