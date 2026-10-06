# Attention and candidate motifs

This guide develops the attention extraction, token-to-nucleotide mapping, candidate-pattern scan, and comparisons used in [Stage1_stability.ipynb](../../Code/Stage1_stability.ipynb#Track-C:-Attention-and-exploratory-diagnostics). The notebook keeps the execution settings, code calls, saved outputs, and result interpretations; this page retains the mathematical walkthroughs, worked examples, implementation details, and sources. Attention is a model behavior, and a sequence match is a candidate annotation: neither is a direct assay of regulatory function.

The maintained implementation is in [embeddings.py](../../Code/shared_code/embeddings.py), [analysis.py](../../Code/shared_code/analysis.py), [sequences.py](../../Code/shared_code/sequences.py), and the [encoder registry and loader](../../Code/shared_code/encoders.py). The [input and embedding guide](inputs-and-embeddings.md) supplies the earlier biological-input and tokenization context.

| Step | Walkthrough |
| --- | --- |
| 1. Extract attention | [Extracting self-attention from the frozen nucleotide encoders](#extracting-self-attention-from-the-frozen-nucleotide-encoders) |
| 2. Summarize and map scores | [Summarizing attention and mapping tokens to nucleotides](#summarizing-attention-and-mapping-tokens-to-nucleotides) |
| 3. Scan candidate patterns | [Scanning candidate RNA stability-associated patterns](#scanning-candidate-rna-stability-associated-patterns) |
| 4. Compare matched and other positions | [Comparing attention at matched and other positions](#comparing-attention-at-matched-and-other-positions) |
| 5. Run and summarize sequences | [Running the attention-motif analysis](#running-the-attention-motif-analysis) |
| 6. Control position and codons | [Controlling the attention test for position and codons](#controlling-the-attention-test-for-position-and-codons) |

## Extracting self-attention from the frozen nucleotide encoders

Return to [Extracting self-attention from the frozen nucleotide encoders](../../Code/Stage1_stability.ipynb#Extracting-self-attention-from-the-frozen-nucleotide-encoders) in the notebook.

The notebook's [representation comparisons](../../Code/Stage1_stability.ipynb#Track-B:-Cross-modal-geometry) use extracted representations, including those from intermediate layers. Here, `embeddings.attention_maps` requests `output_attentions=True` for one nucleotide sequence from one encoder at a time. These are **self-attention** weights: query and key positions belong to the same sequence and the same encoder, rather than to separate modalities.

Each layer returns an array with shape `(1, H, T, T)`. The leading 1 is the one-sequence batch size. Removing that axis leaves `(H, T, T)`: heads, query positions and key positions.

In this notation:

- $H$ is the number of attention heads in this layer. A head is one of the parallel attention computations.
- $T$ is the number of processed input tokens, including special tokens.
- $h$ identifies a head, while $q$ and $k$ identify a query and a key token position. The equations below count heads from 1 through $H$ and positions from 1 through $T$; NumPy indexes their axes from zero.
- $A^{(h)}_{qk}$ is one scalar weight assigned by query position $q$ to key position $k$ in head $h$. The parenthesized superscript selects a head; it is not a power. The two subscripts select a row and column of that head's attention matrix.

**Connection to the code.** Inside `attention_maps`, the model output's `attentions` holds the per-layer arrays. Indexing `[0]` removes the batch axis, and `.float().cpu().numpy()` returns that layer's weights as a NumPy array. `layers` selects which layers to convert; the current analysis converts only the final layer to NumPy, reducing the returned arrays. The model call first requests attention outputs, so this selection does not establish that attention is computed or stored only for the final layer. The returned list preserves the requested layer order, and the token strings come from the same model call. These weights participate in a weighted combination of value vectors; they are not a complete decomposition of the final representation or a direct measure of biological importance. See the attention construction in [Vaswani et al. (2017), Section 3.2](https://arxiv.org/html/1706.03762v7#S3.SS2) and the [ESM output definitions](https://huggingface.co/docs/transformers/en/model_doc/esm#transformers.EsmModel). RNA-FM, RiNALMo, mRNA-FM, and CaLM, loaded through `multimolecule`, return attention arrays with the same layout.

`encoders.load_encoder(encoder, device, eager_attention=True)` loads an encoder with `attn_implementation="eager"` to obtain the explicit attention weights used below; encoders built from their own model code use that code's attention implementation. This is a choice for attention inspection in this pipeline; support for returning weights depends on the model and attention backend. The [Transformers attention-backend documentation](https://huggingface.co/docs/transformers/en/attention_interface) explains how that argument selects the implementation. The embeddings computed earlier are unaffected.

Each loaded model is placed in evaluation mode, and inference runs inside `torch.no_grad()`. No parameter update is performed. Tokenization uses `truncation=True` with the same token limit as embedding extraction, the encoder's `max_len` in the registry. The returned token list describes this model call's processed input and may cover less than the full sequence; its coverage must be checked rather than assumed to match earlier extraction.


Loading an encoder with eager attention can print the same kind of weight report described in [Interpreting the loading messages](../../Code/Stage1_stability.ipynb#Interpreting-the-loading-messages). Pooler parameters reported as newly initialized do not affect this analysis, which reads `outputs.attentions` and does not use a pooled output. The report should be interpreted in relation to the outputs consumed here, rather than treated as a general confirmation that every model component has pretrained weights.

## Summarizing attention and mapping tokens to nucleotides

Return to [Summarizing attention and mapping tokens to nucleotides](../../Code/Stage1_stability.ipynb#Summarizing-attention-and-mapping-tokens-to-nucleotides) in the notebook.

`analysis.aggregate_attention` turns one layer's `(H, T, T)` array into one score per token. With `mode="incoming"`, the score summarizes how much attention a key position receives from all query positions after averaging heads. `embeddings.nucleotide_attention` requests the original model's final layer with `layers=[-1]`, then calls `aggregate_attention(..., layer=0)` on the returned one-element list. The index 0 therefore selects that already extracted final layer, not the model's first layer. This is the current analysis choice, not evidence that the final layer is the most informative one.

**1. Average the heads.**

$$
\bar A_{qk}=\frac{1}{H}\sum_{h=1}^{H}A^{(h)}_{qk}.
$$

In this notation:

- $A^{(h)}_{qk}$ is the selected layer's weight from query token $q$ to key token $k$ in head $h$, as defined above.
- $H$ is the number of heads; $h$ runs from 1 through $H$ in the equation.
- $q$ and $k$ each run from 1 through $T$, where $T$ is the processed token count, including special tokens.
- $\bar A$ is a `(T, T)` matrix. The bar means an average across heads at each fixed query/key pair; it does not average token positions.
- $\sum$ adds the $H$ weights for that pair, and division by $H$ gives their mean.

**2. Sum incoming weight for each key token.**

$$
s_k=\sum_{q=1}^{T}\bar A_{qk}.
$$

In this notation:

- $s_k$ is the incoming score for key token $k$ before rescaling.
- The sum varies the query index $q$ while holding $k$ fixed: it adds one column of $\bar A$.
- The full score vector has $T$ entries. It is a summary of attention weights, not an embedding vector of biological features.

**3. Rescale scores within this sequence.**

$$
\widetilde s_k=
\frac{s_k-\min_j s_j}{\max_j s_j-\min_j s_j+\varepsilon}.
$$

In this notation:

- $\widetilde s_k$ is the returned score; the tilde marks the rescaled quantity.
- $j$ ranges over all $T$ token positions. $\min_j s_j$ and $\max_j s_j$ are the smallest and largest incoming scores in this sequence.
- $\varepsilon=10^{-8}$ is the small constant `1e-8` in the code. It prevents division by zero when every score is equal; in that case all returned scores are zero.

This subtraction and division retain the ordering of unequal scores while changing their scale. For nonconstant scores, the maximum is below 1 in exact arithmetic because of the added constant; it approaches 1 when the score range is large relative to that constant. Scores are rescaled separately for each sequence, so equal displayed scores from different sequences are not calibrated to the same amount of incoming attention. Special-token positions participate in all three steps.

**Connection to the code.** These are direct array operations inside `aggregate_attention`; none of the three equations fits a model.

| Code operation | Input shape → output shape | Connection to the mathematics |
|---|---|---|
| `attentions[layer].mean(axis=0)` | `(H, T, T)` → `(T, T)` | Compute the head average $\bar A$. |
| `layer_att.sum(axis=0)` | `(T, T)` → `(T,)` | Sum query rows to obtain the incoming scores $s_k$. |
| `(score - score.min()) / (score.max() - score.min() + 1e-8)` | `(T,)` → `(T,)` | Return the rescaled scores $\widetilde s_k$. |

The axes have different meanings at each step: after averaging removes the head axis, axis 0 refers to query positions. Any `mode` value other than `"incoming"` selects `sum(axis=1)` instead, producing outgoing row sums; the calls in this notebook use `"incoming"`.

For an illustrative head-averaged matrix with rows `(0.8, 0.2)` and `(0.4, 0.6)`, the incoming scores are `(1.2, 0.8)`. Rescaling gives approximately `(1, 0)`. This example describes the calculation, not an observed model output or a claim that the second token has no biological role.

**Map token scores to nucleotide positions.**

The motif scan uses nucleotide positions rather than token positions. The [Nucleotide Transformer model card](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref#preprocessing) describes six-nucleotide tokens where possible and individual-nucleotide tokens otherwise. RNA-FM and RiNALMo use one token per nucleotide, so each of their token scores maps to exactly one position. mRNA-FM's and CaLM's tokens cover one codon each, so each of their scores maps to three positions. `analysis.expand_token_attention_to_nucleotides`:

1. Pairs each token string with its returned score.
2. Skips strings beginning with `<` or `[`, the helper's rule for special tokens.
3. Repeats each remaining score once per character in that token and appends the copies in token order.

For example, if `ACGTAC` is the first retained DNA-encoder token and has score 0.25, expanded positions 0 through 5 all receive 0.25. This copies one token score onto six positions; it does not estimate six independent nucleotide scores. For RNA-FM, a token `A` with score 0.25 fills only position 0. No further normalization occurs after removing special tokens.

The mapping assumes the retained token strings spell the covered nucleotides in order. It is not a general mapping for arbitrary tokenizers or unknown-token behavior, and it cannot restore positions removed by truncation. The token limits of RNA-FM and RiNALMo cover at most 1,022 nucleotides, so their expanded arrays are shorter than the sequence for most inputs; pattern matches beyond that point are clipped by the comparison below. The example execution prints sequence and expanded-array lengths as a diagnostic; it does not enforce equality.

## Scanning candidate RNA stability-associated patterns

Return to [Scanning candidate RNA stability-associated patterns](../../Code/Stage1_stability.ipynb#Scanning-candidate-RNA-stability-associated-patterns) in the notebook.

The scan creates **candidate sequence-pattern annotations** for the attention comparison. It searches the input sequence for three RNA-relevant patterns; it does not measure binding, modification, or a change in stability.

| Pattern | What the scan identifies | Interpretation limit |
|---|---|---|
| AUUUA | ARE-associated pentamer | A match alone does not establish an active AU-rich element or altered stability. |
| UUAUUUAUU | ARE-associated nonamer | Contains the pentamer; these are related patterns rather than independent mechanistic classes. |
| DRACH | Consensus associated with candidate m6A sites | D = A/G/U, R = A/G, H = A/C/U. The central A is a potential methylation site; a match does not establish methylation. |

The ARE patterns have experimental precedent in [Zubiaga et al. (1995)](https://pmc.ncbi.nlm.nih.gov/articles/PMC230450/). Single-nucleotide mapping shows that only a subset of DRACH candidates are methylated; see [Linder et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487409/). Motif occurrence, molecular modification or binding, and a functional effect on stability are distinct claims.

`sequences.iupac_to_regex` converts each IUPAC symbol to its permitted letters in the DNA alphabet. For example, DRACH becomes `[AGT][AG]AC[ACT]`, with U represented as T to match the converted sequence. `sequences.find_iupac_matches` uses a regex lookahead to include overlapping occurrences. It returns zero-based, end-exclusive intervals: `start` is covered and `end` is the first position after the match. `sequences.scan_sequence_for_motifs` adds the pattern name to each span. Subsequent masks cover the entire matched span, including all five DRACH positions, rather than only its central A.

These annotations depend only on sequence and require no external database call. The [CodonBERT source repository](https://github.com/Sanofi-Public/CodonBERT) links the stability task to [iCodon](https://www.nature.com/articles/s41598-022-15526-7), which studies coding-sequence codon composition. The exact region and label mapping for the local CSV still need verification; its columns do not establish UTR coverage. Presence or absence of these patterns therefore needs interpretation in the context of the sequence region.

## Comparing attention at matched and other positions

Return to [Comparing attention at matched and other positions](../../Code/Stage1_stability.ipynb#Comparing-attention-at-matched-and-other-positions) in the notebook.

`analysis.attention_motif_overlap` compares expanded attention scores at candidate-pattern positions with scores at the other covered positions in the same sequence. It forms the union of matched spans for the pooled comparison, counting each position once, then repeats the comparison separately for each pattern name. Match ends are clipped to the attention array's length. For a per-pattern comparison, the other-position group can include positions matching a different pattern.

**1. Average the scores at matched positions.**

$$
\bar\alpha_M=\frac{1}{n_M}\sum_{p\in M}\alpha_p.
$$

In this notation:

- $L$ is the length of `attention_score`, the expanded array. Its Python positions run from 0 through `L - 1`; this need not cover the full original sequence after truncation.
- $p$ is one nucleotide-array position, rather than an example row or a p-value.
- $\alpha_p$ is the expanded attention score at position $p$. The Greek letter alpha names the value; it is not a biological measurement or a fitted coefficient.
- $M$ is the set of positions inside at least one of the spans being compared; $n_M=|M|$ is its size. The bars count distinct positions, so overlaps do not increase the count twice.
- $p\in M$ means that $p$ belongs to that set. The summation adds the scores at those positions, and division by $n_M$ takes their mean.
- $\bar\alpha_M$ is the resulting scalar mean. The bar denotes an average, and subscript $M$ identifies the group.

**2. Average the scores at the other positions.**

$$
\bar\alpha_O=\frac{1}{n_O}\sum_{p\in O}\alpha_p.
$$

In this notation:

- $O$ is the complement of $M$ within the expanded array: every covered position that is not in $M$.
- $n_O=|O|$ counts those other positions; together, $n_M+n_O=L$.
- $\bar\alpha_O$ is their scalar mean. The summation and bar have the same meanings as in step 1, applied to the other group.

Both means require a nonempty group. The nested `_test` helper returns `None` if either group is empty. That marks an unavailable comparison, not a zero effect; an unavailable pooled comparison makes the outer function return `None` too.

**3. Distinguish the mean gap from the rank test.**

The two returned means also define a descriptive difference:

$$
\Delta=\bar\alpha_M-\bar\alpha_O.
$$

In this notation:

- $\Delta$, read as “delta,” is the matched-minus-other mean gap. Positive values mean the matched group has a larger average score; negative values mean it has a smaller one.
- The two mean symbols refer to steps 1 and 2. Their difference is in the same rescaled score units, rather than a probability.

For example, matched scores `(0.2, 0.4)` have mean 0.3, while other scores `(0.1, 0.1)` have mean 0.1. Their gap is 0.2. This illustrative calculation does not determine the rank test's p-value.

The one-sided Mann-Whitney U call uses `alternative="greater"` to examine a tendency toward larger scores in the matched group. It compares ranks/distributions, rather than directly testing whether $\Delta$ is positive. The standard null is equality of the two underlying distributions, with an alternative of stochastically greater matched scores; see [SciPy's Mann-Whitney U documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).

**Connection to the code.** The means are calculated now; the subtraction is performed later in the multi-sequence summary.

| Code operation | Result | Connection to the calculation |
|---|---|---|
| `_test`: select `at` and `off` from `attention_score` | Arrays with shapes `(n_M,)` and `(n_O,)` | Gather the scores in $M$ and its complement $O$. |
| `at.mean()` and `off.mean()` | Two scalars stored in the returned dictionary | Compute $\bar\alpha_M$ and $\bar\alpha_O$. |
| `mannwhitneyu(at, off, alternative="greater")` | A U statistic and a nominal p-value | SciPy performs the rank test; the dictionary retains its p-value. |
| Later, in `analysis.attention_summary` | One difference per usable sequence, followed by an average | Subtract the dictionary's two mean fields to compute each $\Delta$. |

There is no returned `Delta` field or separate gap function in `attention_motif_overlap`. The equation names a quantity formed from its outputs by `attention_summary` in [Running the attention-motif analysis](#running-the-attention-motif-analysis).

Treat the p-values as **nominal**: they are the test's reported values without calibration for this positional setting. Expanding token scores repeats values; neighboring positions are dependent; patterns can overlap; and many sequence/pattern comparisons are performed without multiplicity correction. Sequence composition and tokenization can also affect both masks and scores. A small p-value or positive mean gap identifies a candidate association for follow-up, not evidence that the encoder learned functional RNA regulation. Calibrated null models and experimentally supported annotations require a separate methodological decision.

## Running the attention-motif analysis

Return to [Running the attention-motif analysis](../../Code/Stage1_stability.ipynb#Running-the-attention-motif-analysis) in the notebook.

The first execution follows one retained sequence through the complete pipeline for each nucleotide encoder in `ATTENTION_ENCODERS`: convert its alphabet for that encoder, extract attention, summarize the final layer, expand token scores, scan candidate patterns, compare groups, and plot the result. `embeddings.nucleotide_attention` performs the first four steps. The printed sequence and attention-array lengths help inspect coverage, but they are not an assertion that the coordinates agree for every input.

The second execution repeats the attention extraction for the **first `ATTENTION_N_ROWS` retained sequences**, currently 100, separately for each encoder. `embeddings.attention_profiles` returns each sequence's expanded attention array, kept in `attention_profiles` for the permutation tests below, and `analysis.attention_motif_tests` runs the pattern comparison on each array and keeps rows with an available pooled comparison. The per-sequence comparison dictionaries are stored in `attention_results`. `analysis.attention_summary` then calculates two summaries per encoder and pattern and returns the DataFrame displayed by the notebook. The equations below describe this second execution, the code cell after the single-example result.

**1. Count nominal threshold crossings.**

$$
f_{0.05}=\frac{1}{m}\sum_{i=1}^{m}\mathbf{1}\{p_i<0.05\}.
$$

In this notation:

- $m$ is the number of sequences with a usable comparison, and must be greater than zero for this calculation.
- $i$ indexes those usable sequences from 1 through $m$. It does not index nucleotide positions within one sequence.
- $p_i$ is the nominal Mann-Whitney p-value returned for sequence $i$.
- $\mathbf{1}\{p_i<0.05\}$ is an indicator: 1 if that p-value is below 0.05, otherwise 0. The braces state the condition being counted.
- $\sum$ adds these zeros and ones; dividing by $m$ gives the fraction $f_{0.05}$. Its subscript identifies the chosen threshold, not a sequence position.

**2. Average the per-sequence mean gaps.**

$$
\overline{\Delta}=\frac{1}{m}\sum_{i=1}^{m}\Delta_i.
$$

In this notation:

- $\Delta_i$ is the matched-minus-other mean attention gap for sequence $i$, as defined in the comparison section. It is calculated from that sequence's two returned mean fields.
- $\overline{\Delta}$ is the mean of those gaps; the bar marks the average across sequences.
- $m$, $i$ and the summation refer to the same usable sequence set as in step 1. Each sequence has equal weight, irrespective of its number of nucleotide positions or matches.

For example, if three usable sequences have p-values `(0.01, 0.20, 0.04)` and gaps `(0.02, -0.01, 0.05)`, the threshold fraction is $2/3$ and the mean gap is $0.06/3=0.02$. These are illustrative values. The fraction is not itself a p-value, and the average gap is not a test pooling all nucleotide positions across sequences.

**Connection to the code.** These are direct summaries of the dictionaries collected in `attention_results`.

| Code operation | Result | Connection to the mathematics |
|---|---|---|
| `attention_motif_tests` appends a result when it is not `None` | A list of usable comparison dictionaries per encoder | Determines the included sequence set and $m$ for that encoder. |
| `np.mean([g["p_value"] < 0.05 for g in group])` | The `fraction with nominal p < 0.05` column | NumPy counts `True` as 1 and `False` as 0, implementing $f_{0.05}$. |
| Subtract the two mean fields inside the `mean gap` list comprehension | One gap per dictionary | Computes each $\Delta_i$. |
| `np.mean(...)` around that list | The `mean gap` column | Computes $\overline{\Delta}$. |

Each pattern row repeats these operations using only dictionaries with an available comparison for that pattern. Its usable count can differ from the pooled count or another pattern's count. A pattern with no usable comparison appears with zero sequences and empty summary columns.

The table's label “fraction with nominal p < 0.05” refers only to crossing the unadjusted 0.05 threshold. They do not establish calibrated statistical significance under the dependence and multiple-comparison limitations described above. The single example illustrates the calculation; the 100-sequence summary describes the selected sample under the same procedure.

## Controlling the attention test for position and codons

Return to [Controlling the attention test for position and codons](../../Code/Stage1_stability.ipynb#Controlling-the-attention-test-for-position-and-codons) in the notebook.

The per-sequence rank test compares matched positions with every other position. Two properties of the attention arrays can make that comparison favor matches without any link to the patterns themselves: attention can depend on position, for example by concentrating near the start of a sequence, and codon-level encoders give all three nucleotides of a codon one shared score, while a pattern such as DRACH overlaps only some codons. `analysis.motif_permutation_test` shuffles matched labels within groups that preserve the number of matched positions in each 100-nucleotide bin or each identical fixed-triplet-and-offset group. These preserve selected counts, not arbitrary positional or contextual token effects.

**1. Average the matched-minus-other gap over sequences.**

$$
G = \frac{1}{m}\sum_{i=1}^{m}\Delta_i ,
$$

In this notation:

- $\Delta_i$ is sequence $i$'s matched-minus-other mean attention gap, as defined in [Comparing attention at matched and other positions](#comparing-attention-at-matched-and-other-positions), for one pattern or for all patterns pooled.
- $m$ counts the sequences with at least one matched and one other position in the encoder's covered range.
- $G$ is the observed test statistic, using the same per-sequence gap formula as the notebook summary's `mean gap` column. Its usable sequence set is prepared independently for the requested motif and need not equal the summary's set: `attention_motif_tests` first discards rows with unavailable pooled comparisons, whereas `motif_permutation_test` directly checks each requested comparison.

**2. Shuffle which positions count as matched, within strata.**

For each of `ATTENTION_NULL_PERMUTATIONS` rounds, every sequence's matched labels are permuted among positions that share a stratum, keeping the number of matched positions in each stratum, and $G$ is recomputed. Two kinds of strata give two nulls:

| Null | Positions that exchange labels | What the null keeps |
|---|---|---|
| Position | Positions in the same 100-nucleotide bin | The count of matched positions in each bin; arbitrary within-bin positional trends are not retained |
| Codon and frame | Positions with the same codon and the same place within it | The count of matched positions for each identical fixed three-nucleotide block and offset within it; arbitrary contextual token effects are not retained |

**3. Compare the observed statistic with the shuffled ones.**

$$
p = \frac{1 + \#\{r : G^{(r)} \ge G\}}{R + 1}.
$$

In this notation:

- $R$ is the number of shuffles and $G^{(r)}$ the statistic after shuffle $r$.
- $\#\{\cdot\}$ counts the shuffles whose statistic reaches the observed one; adding 1 to the count and to $R$ keeps $p$ above zero, so the smallest attainable value is $1/(R+1)$, about 0.002 with 500 shuffles.

The table reports $G$, the mean of the shuffled statistics, their difference (the excess gap), and $p$, for each encoder, pattern, and null. A small p-value identifies a departure from the particular null that was implemented. The position null preserves bin-level occupancy rather than exact positions, and the codon null preserves the stated codon/frame grouping; neither establishes a regulatory mechanism. Shuffling matched labels within strata assumes those labels are exchangeable there under the chosen null, and shuffled masks need not form intact motif occurrences. Patterns with no usable sequence, such as a pattern that never occurs, are omitted. The tests reuse the same 100 sequences and are not corrected for the number of encoders, patterns, and nulls.

## Reusing the method on untranslated regions

The notebook's [GTEx attention analysis](../../Code/Stage1_gtex.ipynb#Attention-at-candidate-patterns-in-3'-untranslated-regions) applies the same summary, pattern scan, comparison, and permutation functions to selected 3' untranslated regions using the single-nucleotide RNA encoders. The notebook owns its selection settings and saved results.

The `strata="codon"` option still groups positions by the same nonoverlapping three-nucleotide block and offset within that block, beginning at the region's first nucleotide. Because these are untranslated regions, those blocks are sequence-context groups rather than biological codons or reading frames. The position null still uses 100-nucleotide bins; all coverage and permutation assumptions above continue to apply.
