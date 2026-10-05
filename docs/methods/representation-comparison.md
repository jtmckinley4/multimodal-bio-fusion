# Comparing representations

Use this guide for the equations, notation, worked examples, and implementation details behind the [stability representation comparisons](../../Code/Stage1_stability.ipynb#Track-B:-Cross-modal-geometry). GTEx reuses these methods in [Track B](../../Code/Stage1_gtex.ipynb#GTEx-Track-B:-Cross-modal-geometry) and its [alignment and similarity diagnostics](../../Code/Stage1_gtex.ipynb#Alignment,-prediction-error,-and-representational-similarity-on-GTEx). The notebooks keep their experiment settings, outputs, and interpretations. Section return links lead to the stability walkthrough unless explicitly labeled GTEx; references to notebook cells and tables mean those linked cells, unless an explanatory table appears here.

All comparisons require corresponding rows to identify the same examples. Two encoders may have different feature widths. Similarity, fitted correspondence, and predictive benefit answer different questions; a high score on one does not establish the others. The [linear-probing guide](linear-probing.md) explains target prediction, and [From sequences to embeddings](inputs-and-embeddings.md) explains the inputs and pooling.

| Question | Walkthrough | Fitting boundary |
| --- | --- | --- |
| Do the complete representations organize examples similarly? | [Linear CKA](#final-layer-representation-similarity-linear-cka) | Descriptive comparison of the supplied rows; no predictor or cross-modal projection. |
| Where in the encoders does that agreement appear? | [Layerwise CKA](#layerwise-representation-similarity-linear-cka) | Repeats CKA over returned states on a shared subsample. |
| Are the same examples nearby in each space? | [Neighborhood overlap](#neighborhood-overlap-across-representations) | Neighbor searches on the supplied rows. |
| Can a learned mapping recover a paired example? | [CCA and retrieval](#canonical-correlation-and-cross-modal-retrieval) | PCA/CCA fit on training rows, followed by retrieval among test rows. |
| How much agreement remains after a linear composition fit? | [Composition control](#composition-control) | Residualization fits all supplied rows before subsequent comparisons. |
| Is the agreement in dominant or remaining variance directions? | [PCA bands](#high--and-low-variance-directions) | PCA fits all supplied rows before the block comparisons and probes. |
| Are examples with greater paired similarity easier to predict? | [Alignment and prediction error](#sample-level-alignment-and-prediction-error) | Row-wise regression folds within the earlier CCA test subset. |
| Do the representations rank example-pair distances similarly? | [RSA](#representational-similarity-analysis) | Descriptive distances; significance depends on the permutation assumptions. |

## Final-layer representation similarity: linear CKA

Return to [Final-layer representation similarity: linear CKA](../../Code/Stage1_stability.ipynb#Final-layer-representation-similarity:-linear-CKA) in the notebook.

Linear centered kernel alignment (CKA) compares the final-layer mean-pooled representations of the same retained examples, one pair of encoders at a time, for every pair in `PAIRS`. Corresponding rows refer to the same example, while the feature counts can differ.

Each calculation uses two separate embedding matrices, not their concatenation. It asks whether relationships between examples are similar across the two representations. The helper centers each feature across rows, computes dot products within and between the representations, and normalizes the comparison. The equation and notation are explained in [Linear centered kernel alignment](#linear-centered-kernel-alignment) below.

Unlike the predictive probes, this calculation does not fit a prediction model or use stability labels. The probes ask which target properties a simple model can recover; CKA asks how similarly two representations organize the same examples.

In the derived stability setting, pairs within one modality provide a reference for pairs across modalities. The nucleotide encoders start from the same coding sequence in DNA or RNA letters; protein encoders receive its translation. Architecture, pretraining, tokenization, model size, and input truncation can all affect this comparison. GTEx has distinct inputs, including full transcripts for single-nucleotide RNA encoders and CDS inputs for codon encoders.

### Linear centered kernel alignment

`analysis.linear_cka(X, Y)` compares representations of the same examples. `X` has shape `(n, p)` and `Y` has shape `(n, q)`: their feature counts may differ, but corresponding rows must identify the same example. Capital `Y` here is a second embedding matrix, distinct from the lowercase target `y` used by the [linear probe](linear-probing.md).

**1. Center each feature across examples.**

For feature $j$ of the first representation, calculate its mean:

$$
\bar x_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij}.
$$

In this notation:

- $n$ is the shared number of example rows; $p$ and $q$ are the numbers of features in $X$ and $Y$, respectively.
- $x_{ij}$ is the scalar entry of $X$ in example row $i$, feature column $j$.
- $\bar x_j$ is the mean of that feature across all $n$ rows. The bar denotes an average.
- $\sum_{i=1}^{n}$ adds the feature values across rows; division by $n$ takes their average.

Subtract that mean from each value in the feature:

$$
(X_c)_{ij}=x_{ij}-\bar x_j.
$$

In this notation:

- $X_c$ is the centered matrix, still of shape $(n,p)$; the subscript $c$ labels centering.
- $(X_c)_{ij}$ selects its entry at row $i$, column $j$.
- $x_{ij}$ and $\bar x_j$ have the meanings above: each row loses the same mean for feature $j$.

The code applies the same two operations independently to `Y` to obtain $Y_c$, with shape $(n,q)$. It subtracts each representation's own column means. Unlike the probe's standardization, CKA centering does not divide by feature standard deviations.

**2. Measure feature dot-product matrices with the Frobenius norm.**

The Frobenius norm combines all entries of a matrix into a scalar size:

$$
\|A\|_F=\sqrt{\sum_{a=1}^{r}\sum_{b=1}^{s}A_{ab}^{\,2}}.
$$

In this notation:

- $A$ is any real-valued matrix of shape $(r,s)$, and $A_{ab}$ is its entry in row $a$, column $b$.
- $r$ and $s$ are its row and column counts; $a$ and $b$ index those entries.
- The two summations add over every row and column. The exponent $2$ squares each entry, and the square root converts the total to a norm.
- $\|\cdot\|_F$ names the Frobenius norm. The subscript $F$ identifies the norm; it does not denote an evaluation fold here.

For example, the matrix with rows $(1,2)$ and $(2,4)$ has norm $\sqrt{1^2+2^2+2^2+4^2}=5$. Its squared Frobenius norm is $25$. These are illustrative values, not encoder results. NumPy implements this matrix norm with `np.linalg.norm(..., ord="fro")`; see [NumPy's norm definition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).

**3. Normalize the cross-representation quantity.**

$$
\operatorname{CKA}(X,Y)
=\frac{\|Y_c^\top X_c\|_F^2}
{\|X_c^\top X_c\|_F\,\|Y_c^\top Y_c\|_F}.
$$

In this notation:

- $\operatorname{CKA}(X,Y)$ is the scalar linear CKA value returned by `linear_cka` for the two input matrices.
- $X_c$ and $Y_c$ are the centered matrices defined above.
- $\top$, written `.T` in NumPy, means transpose: exchange rows and columns.
- Juxtaposing matrices, as in $Y_c^\top X_c$, means matrix multiplication, written `@` in NumPy. Each output entry sums products along the shared dimension.
- $Y_c^\top X_c$ has shape $(q,p)$ and contains dot products between centered features in the two representations.
- $X_c^\top X_c$ has shape $(p,p)$, and $Y_c^\top Y_c$ has shape $(q,q)$. They contain feature dot products within each representation.
- $\|\cdot\|_F$ has the definition above. The numerator squares that norm; the denominator multiplies two unsquared scalar norms.

**Connection to the code.**

This is a direct NumPy calculation in `linear_cka`; no model is fitted. The table follows its operations in their actual order.

| Code operation | Input shape → output shape | Connection to the mathematics |
|---|---|---|
| `X.mean(axis=0, keepdims=True)` | `(n, p)` → `(1, p)` | Calculate each feature mean $\bar x_j$, retaining a row dimension. |
| `X - X.mean(axis=0, keepdims=True)` | `(n, p)` → `(n, p)` | Subtract those means from every example to form $X_c$. The `Y` assignment similarly forms $Y_c$. |
| `Y_centered.T @ X_centered` | `(q, n)` and `(n, p)` → `(q, p)` | Form the cross-representation feature dot products. |
| `np.linalg.norm(..., ord="fro") ** 2` | `(q, p)` → scalar | Square their Frobenius norm to obtain `numerator`. |
| `X_centered.T @ X_centered`, then its Frobenius norm | `(p, n)` and `(n, p)` → `(p, p)` → scalar | Obtain `norm_x`; the corresponding `Y` operation gives `norm_y` through a `(q, q)` matrix. |
| `numerator / (norm_x * norm_y)` | Three scalars → scalar | Return the normalized CKA value. |

The `...` stands for the matrix product in the preceding row. NumPy broadcasts the `(1, p)` or `(1, q)` mean row during subtraction, applying it to every example row.

This feature-based calculation is equivalent to comparing the centered example-similarity matrices $X_cX_c^\top$ and $Y_cY_c^\top$, each of shape $(n,n)$. An entry in either matrix is the dot product between two centered example embeddings. The common normalization factor from the Hilbert-Schmidt independence criterion (HSIC) cancels in the CKA ratio, so the code names the unnormalized top term `numerator`.

For nonzero centered representations, linear CKA lies between zero and one in exact arithmetic. It is unchanged by orthogonal transformations, such as rotations and reflections, or by nonzero uniform scaling of either representation. Arbitrary changes of basis or separate scaling of individual features can change it. See [Kornblith et al. (2019), Section 3 and Table 1](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf).

If either centered representation is entirely zero, the denominator is zero and CKA is undefined; this helper has no special handling for that case. High CKA describes representational similarity. It does not by itself establish good target prediction or a benefit from combining the modalities.

## Layerwise representation similarity: linear CKA

Return to [Layerwise representation similarity: linear CKA](../../Code/Stage1_stability.ipynb#Layerwise-representation-similarity:-linear-CKA) in the notebook.

The final-layer comparison gives one summary of representation similarity. This analysis asks whether that similarity changes with encoder depth by comparing every returned hidden state of one encoder with every returned hidden state of another, for every pair of encoders. A hidden state is the model's numerical representation at a particular point in its computation; it still contains one vector per token before pooling.

**1. Build one embedding matrix per returned state.**

The code first samples up to `LAYERWISE_N_ROWS = 200` rows, with `SEED`, from the pre-filter sample used for the main stability analysis, then applies the existing sequence filters with `datasets.retain`. Thus, 200 is a limit before filtering, not a guaranteed number of retained rows; the saved output records the retained count. For each retained row, `embeddings.embed_all_layers` requests all hidden states in one forward pass per encoder and averages each state over the token axis. DNABERT-2's model code does not return all hidden states on request, so its states are captured with forward hooks on its embedding module and each transformer layer. Pooling still includes the returned special-token positions and uses the configured token limits.

Index 0 denotes the initial embedding output; later indices denote returned encoder-layer states. For transformer encoders, these are successive transformer-block outputs; HyenaDNA is not a transformer. An encoder with $L$ transformer blocks therefore returns $L+1$ states; Nucleotide Transformer 500M, for example, has 24 blocks and returns 25. The notebook prints the count for every encoder. Index 0 is therefore not an additional transformer block. See the [Transformers hidden-state output convention](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutput).

`embeddings.embed_layers` stacks the pooled vectors from corresponding retained rows into one matrix per hidden-state index, and `layer_matrices` maps each encoder key to its list of matrices. Every matrix for one encoder has shape `(n, d)`, where $n$ is the number of rows retained in this sampled analysis and $d$ is that encoder's feature width. The same row always denotes the same example across all matrices.

**2. Compare each state of one encoder with each state of another.**

For one encoder pair, such as a DNA encoder and a protein encoder:

$$
C_{ij} = \operatorname{CKA}\!\left(X^{(i)}, Y^{(j)}\right).
$$

In this notation:

- $n$ is the number of examples retained for this layerwise sample. It can differ from the main dataset's retained count.
- $X^{(i)}$ is the first encoder's embedding matrix at hidden-state index $i$, stored as `layer_matrices[a][i]` for the first encoder `a` of the pair. $Y^{(j)}$ is the second encoder's matrix at state $j$, stored as `layer_matrices[b][j]`.
- The parenthesized superscripts identify states; they are not powers.
- $i$ runs over the first encoder's states and $j$ over the second encoder's states. These indices match the Python code directly.
- $\operatorname{CKA}$ names the scalar similarity calculated by `analysis.linear_cka`. The [linear CKA derivation](#linear-centered-kernel-alignment) explains its centering and normalization.
- $C$ is the pair's full comparison matrix, stored as `layer_cka[(a, b)]`, with one row per state of the first encoder and one column per state of the second; $C_{ij}$ is its entry at row $i$ and column $j$.

For example, for a pair of transformer encoders in the DNA-protein matrix, $C_{2,3}$ compares the pooled DNA vectors after DNA transformer block 2 with the pooled protein vectors after protein transformer block 3, across the same retained examples. It is one score for two whole matrices, not a comparison between example rows 2 and 3.

**Connection to the code.** For each pair in `PAIRS`, `analysis.layerwise_cka` calls `linear_cka` once for each state pair and stores the result. `analysis.layerwise_summary` reports each matrix's largest entry and its final-state pair, `matrix[-1, -1]`; Python index `-1` selects the last state on each axis. With twelve encoders there are 66 matrices, so the table summarizes all of them and heatmaps are drawn for `LAYERWISE_PLOT_PAIRS`. These are descriptive comparisons of representations, not fitted cross-modal projections or evidence that a particular layer pair improves prediction.

**Check the role of the second encoder's embedding output.** A maximum at state 0 of an encoder occurs at the initial embedding output before the later encoder layers. For each pair, `layerwise_summary` also repeats the maximum search after excluding state 0 of the second encoder in the pair.

The slice `matrix[:, 1:]` keeps every state of the first encoder, including its state 0, and only states 1 onward of the second. Adding 1 to the returned column index maps the sliced matrix back to the original state numbering. This asks whether another high-scoring pair remains after removing that embedding output; it does not exclude the first encoder's initial embedding state.

## Neighborhood overlap across representations

Return to [Neighborhood overlap across representations](../../Code/Stage1_stability.ipynb#Neighborhood-overlap-across-representations) in the notebook.

CKA summarizes similarity across representation matrices. Neighborhood overlap asks a more local question: for a particular retained row, are the examples nearest to it in one representation also nearest to it in another? The notebook computes it for every pair of encoders. Neighbor identities refer to the same paired rows in both matrices, not to equal feature coordinates.

**1. Count shared neighbors for each row.**

$$
o_i = \frac{\left|N_X(i)\cap N_Y(i)\right|}{k}.
$$

In this notation:

- $X$ and $Y$ are the two embedding matrices of one encoder pair, such as `EMBEDDINGS["nt500m"]` with shape `(n, p)` and `EMBEDDINGS["esm2_35m"]` with shape `(n, q)`. The matrices contain $n$ matched examples; $p$ and $q$ are their respective feature counts.
- $i$ identifies the query example, from 1 through $n$ in the equation. Python indexes those same rows from 0 through `n - 1`.
- $k$ is the number of neighbor indices retained per query in each space; the notebook call uses $k=10$. Requesting one extra neighbor requires $1\le k\le n-1$.
- $N_X(i)$ and $N_Y(i)$ are the sets of retained neighbor-row indices for query $i$ in the spaces of $X$ and $Y$, respectively. Parentheses specify which query's set is being selected.
- $\cap$ means set intersection: keep only indices appearing in both sets.
- The vertical bars $|\cdot|$ count the indices in that intersection. They denote a set's size, not a vector norm.
- $o_i$ is the shared-neighbor fraction for row $i$, a scalar between 0 and 1. Dividing the shared count by $k$ puts queries on the same scale.

For example, if 3 of a row's 10 retained neighbors are shared, its fraction is $o_i=3/10=0.3$.

**2. Average the row fractions.**

$$
\operatorname{overlap@}k
=\frac{1}{n}\sum_{i=1}^{n}o_i.
$$

In this notation:

- $\operatorname{overlap@}k$ names the final mean overlap at the chosen neighbor count $k$. The `@` reads “at”; it is part of the metric's name.
- $\sum_{i=1}^{n}$ adds the scalar fractions $o_i$ over all $n$ query rows; dividing by $n$ takes their mean.

**Connection to the code.** `analysis.mutual_knn_alignment` calculates each $o_i$ in its `overlaps` list and returns their mean with `np.mean(overlaps)`. Despite the helper's name, this is overlap between two neighbor sets, not a check that neighbor relationships are reciprocal.

The helper uses the default Euclidean distance of [`NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). It requests $k+1$ neighbors and removes the first returned index with `[i, 1:]`, intending to exclude the query itself. With duplicate vectors or distance ties, that first index need not be the query's own index; explicit self-exclusion remains an evaluation issue. The sets in the equations describe what the code retains under this rule, rather than guaranteeing that all retained neighbors are other rows.

**The chance reference uses an idealized selection rule.** For two independent, uniformly selected sets of $k$ nonself neighbors among $n-1$ candidates, the expected overlap fraction is $k/(n-1)$. Here, $n-1$ excludes the query row. Each of the $k$ neighbors in one set has probability $k/(n-1)$ of appearing in the other. The printed chance level uses $k/(n-1)$. This reference is not a simulated random-embedding result or a significance test, and disagreement with CKA would not by itself invalidate either metric.

## Canonical correlation and cross-modal retrieval

Return to [Canonical correlation and cross-modal retrieval](../../Code/Stage1_stability.ipynb#Canonical-correlation-and-cross-modal-retrieval) in the notebook.

The previous metrics compare the representations without learning a cross-modal projection. Canonical correlation analysis (CCA) instead fits linear combinations of features that vary together across matched examples of two representations. The notebook fits a separate CCA for every pair of encoders, querying with the encoder that comes first in `ENCODER_KEYS` and searching the second; the matrices show each pair's value in both positions. The explanation below uses the first encoder's representation as $X$ and the second's as $Y$. It can reveal correspondence in selected directions even when a whole-matrix comparison is modest.

**1. Fit the transformations on training rows.**

`splits.grouped_holdout_split` assigns each retained sequence to the training or test side from the SHA-256 hash of `SEED` and the sequence string, using the same grouped-hash idea as the probe folds. This CCA call explicitly receives the notebook's `SEED`; `probe_table` uses its helper's default seed, as explained in the [linear-probing guide](linear-probing.md#evaluation-and-interpretation). About 30% of rows become test rows, and every copy of an identical sequence falls on the same side. The same indices select every encoder's matrix, so every pair uses the same training and test rows. Write $n_{\mathrm{train}}$ for the number of training pairs and $n_{\mathrm{test}}$ for the number of test pairs; $p$ and $q$ are the embedding widths of the pair's first and second encoders.

Each encoder's representation receives a separate principal component analysis (PCA), fitted only on its training matrix, requesting 50 components, capped at `min(50, n_train - 1, p, q)`. The 50-component shapes below describe the saved configuration where this cap does not reduce the request. PCA supplies a smaller feature space using directions of large training-set variation. The implementation centers features without standardizing their variances before PCA. See [the PCA definition and implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

The code then fits `CCA(n_components=10)` on the two reduced training matrices, using CCA's default input scaling. It applies the fitted PCA and CCA transformations to test rows without refitting them. The shape changes make the roles of 50 and 10 explicit:

| Code operation | Input shape → output shape | Meaning |
|---|---|---|
| `pca_a.transform(X_a[test_idx])` | `(n_test, p)` → `(n_test, 50)` | Express the first encoder's test rows using the 50 PCA directions learned from its training rows. |
| `pca_b.transform(X_b[test_idx])` | `(n_test, q)` → `(n_test, 50)` | Express the matched rows of the second encoder using its separately fitted PCA. |
| `cca.transform(...)` on the two reduced test matrices | Two `(n_test, 50)` matrices → two `(n_test, 10)` matrices | Produce canonical coordinates for both encoders using the fitted CCA transformations. |

In the table, `n_test` means $n_{\mathrm{test}}$. The training PCA outputs similarly have shape `(n_train, 50)` before CCA is fitted.

**2. Understand one canonical pair.**

CCA starts by seeking one weighted combination of the first modality's reduced features and one of the second modality's reduced features. For the first pair, write the two projected vectors as

$$
\mathbf u = \widetilde X_{\mathrm{train}}\mathbf a.
$$

The corresponding projection of the second modality is:

$$
\mathbf v = \widetilde Y_{\mathrm{train}}\mathbf b.
$$

In this notation:

- $n_{\mathrm{train}}$ and $n_{\mathrm{test}}$ count the paired training and test examples. The subscripts identify their partitions.
- $p$ and $q$ are the respective original embedding widths of the first and second encoders, before PCA reduces each to 50 features.
- $\widetilde X_{\mathrm{train}}$ and $\widetilde Y_{\mathrm{train}}$ are the first and second encoders' training matrices after PCA reduction and CCA's internal centering and scaling. Each has shape `(n_train, 50)`.
- The tilde, read as “X tilde” or “Y tilde,” marks these processed matrices. It does not mean an approximation here. The subscript “train” identifies their training rows.
- $\mathbf a$ and $\mathbf b$ are coefficient vectors with 50 entries each: one coefficient for each PCA feature in the respective representation. Bold lowercase symbols here represent vectors.
- Matrix-vector multiplication, such as $\widetilde X_{\mathrm{train}}\mathbf a$, multiplies each row's 50 feature values by the corresponding 50 coefficients and adds them. It produces one number per training row.
- $\mathbf u$ and $\mathbf v$ are therefore vectors of length $n_{\mathrm{train}}$, containing one canonical coordinate per training example in each representation. They are not the 50 coefficients themselves.

For a small illustration with only two features, a processed row `[2, 3]` and coefficients `[0.5, -1]` give the single coordinate $2(0.5)+3(-1)=-2$. The actual calculation combines 50 features per row. These illustrative values are not fitted notebook results.

The first-pair objective is

$$
(\mathbf a^\star,\mathbf b^\star)
=
\underset{\mathbf a,\mathbf b}{\operatorname{arg\,max}}\;
\operatorname{corr}\!\left(\widetilde X_{\mathrm{train}}\mathbf a,\,
                          \widetilde Y_{\mathrm{train}}\mathbf b\right).
$$

In this notation:

- $\operatorname{corr}$ means Pearson correlation across paired training rows: it measures how the two projected vectors vary together. Both projected vectors must have nonzero variance for this correlation to be defined.
- $\operatorname{arg\,max}$ means “find coefficient vectors that make the correlation as large as possible.” It returns the coefficient choices, rather than the maximum correlation value.
- The $\mathbf a,\mathbf b$ below the operator identify the quantities being chosen; the processed training matrices remain fixed.
- The stars in $\mathbf a^\star$ and $\mathbf b^\star$ label optimizing coefficient vectors. They do not mean multiplication or another power. Each selected vector still has 50 entries.

This equation describes the first-pair mathematical objective behind the library method. The notebook delegates fitting to `CCA.fit`; it does not implement this optimization itself. The library learns subsequent pairs after removing previously captured variation and constructs the transformations used by `CCA.transform`. Requesting 10 components therefore yields 10 coordinates per example in each representation, with each coordinate formed from the 50 input features. A canonical coordinate is a learned numerical combination, not an identified biological mechanism. See [CCA](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html) and the [cross-decomposition guide](https://scikit-learn.org/stable/modules/cross_decomposition.html#canonical-correlation-analysis).

`analysis.cca_retrieval` measures test correlations by applying `np.corrcoef` to corresponding columns of the two test coordinate matrices, pairing the same test rows. It returns these test coordinates as `cca_results[(a, b)]["test_coords"]` for the [sample-level alignment diagnostic](#sample-level-alignment-and-prediction-error). Test correlations need not decrease with component index: the component order was learned from the training rows.

**3. Retrieve the matched row of the second encoder for each test row of the first.**

Each test vector of the first encoder queries the test vectors of the second by Euclidean distance in the resulting 10-dimensional coordinates. Its designated correct match is the row of the second encoder with the same test-set index. The code measures

$$
\operatorname{Recall@}k
=
\frac{1}{n_{\mathrm{test}}}
\sum_{i=1}^{n_{\mathrm{test}}}
\mathbf{1}\!\left\{i\in R_k(i)\right\}.
$$

In this notation:

- $k$ is the number of retrieved candidates considered per query. The notebook reports 1, 5 and 10; the formula assumes $1\le k\le n_{\mathrm{test}}$.
- $i$ identifies a paired test row, from 1 through $n_{\mathrm{test}}$ in the equation. Python uses indices 0 through `n_test - 1`. These are positions within the test arrays, not the original dataset's row numbers.
- $R_k(i)$ is the set of the first $k$ indices of the second encoder returned for query $i$; the subscript $k$ records the chosen candidate count.
- $\in$ means “belongs to,” also read as “is an element of” or “is a member of.” These are equivalent expressions for set membership. Thus, $i\in R_k(i)$ asks whether the designated row index belongs to the retrieved candidate set.
- $\mathbf{1}\{\cdot\}$ is an indicator: it returns the scalar 1 if the statement in braces is true and 0 otherwise. This bold 1 is not an embedding vector.
- $\sum_{i=1}^{n_{\mathrm{test}}}$ adds the successful-match indicators, and division by $n_{\mathrm{test}}$ gives the fraction of successful queries.
- $\operatorname{Recall@}k$ names that fraction, read as “recall at k.” There is one designated correct match per query here.

**Plain-English interpretation.** Take each test entry of the first encoder in turn and retrieve the $k$ nearest test entries of the second encoder using their coordinates from the fitted CCA transformations. Check whether the entry from the same test row appears among those candidates. Count one success if it appears and zero otherwise, then divide the total successes by the number of test queries. The result is the fraction of queries whose designated partner appears within the first $k$ positions. A different row containing an identical sequence does not count as that designated partner.

For example, if query 4 returns protein indices `[8, 4, 2]`, it fails at $k=1$ but succeeds at $k=3$. In the code, `i in neighbors[i, :k]` performs the membership check, and the mean over queries gives the reported fraction. Tied distances at the cutoff inherit the library's returned ordering.

**The chance reference is a ranking calculation.** A uniformly random ranking has expected success $k/n_{\mathrm{test}}$ for $k\le n_{\mathrm{test}}$: the one designated match has $k$ successful positions among $n_{\mathrm{test}}$ possible positions. The notebook prints the number of test rows and the top-1 chance $1/n_{\mathrm{test}}$. The actual test count and corresponding reference stay with the notebook output. This is a ranking reference, not a $p$-value.

Training the transformations only on training rows avoids fitting those transformations to test rows. The split keeps identical sequences together, but not biologically related ones, such as homologous genes. Neither CCA fitting nor retrieval uses the stability labels.

## Composition control

Return to [Composition control](../../Code/Stage1_stability.ipynb#Composition-control) in the notebook.

The [layer-wise CKA](../../Code/Stage1_stability.ipynb#Layerwise-representation-similarity:-linear-CKA) and [GC-content](../../Code/Stage1_stability.ipynb#GC-content-prediction) results suggest a hypothesis: the representations may agree largely because each tracks sequence composition. This analysis tests that hypothesis by removing composition from every embedding and repeating two of the preceding notebook comparisons.

**1. Describe each sequence by its composition.**

$$
\mathbf f(s) = \big(\mathrm{GC}(s),\ \mathrm{GC3}(s),\ \log L(s),\ c_1(s),\ldots,c_{64}(s),\ a_1(s),\ldots,a_{20}(s)\big).
$$

In this notation:

- $s$ is one retained coding sequence, and $L(s)$ is its length in nucleotides.
- $\mathrm{GC}(s)$ and $\mathrm{GC3}(s)$ are the GC fractions over the whole sequence and over third codon positions, as defined in [GC-content prediction](../../Code/Stage1_stability.ipynb#GC-content-prediction) and [GC3 prediction](../../Code/Stage1_stability.ipynb#GC3-prediction).
- $\log$ is the natural logarithm.
- $c_k(s)$ is the frequency of codon $k$, for the 64 possible codons in a fixed order, among the sequence's complete codons. The 64 frequencies sum to 1 when every counted codon contains only A, C, G, and T; the preliminary sequence filter does not itself establish that alphabet condition.
- $a_m(s)$ is the frequency of amino acid $m$, for the 20 standard amino acids, in the translated protein.
- $\mathbf f(s)$ is the resulting vector of 87 features. Stacking one row per retained sequence gives the matrix `composition` with shape `(n, 87)`.

These hand-computed features describe nucleotide, codon, and translated amino-acid composition; they contain no learned embedding.

**2. Remove the part of each embedding that composition predicts linearly.**

$$
X^{\perp} = X - F_s\,\hat B.
$$

In this notation:

- $X$ is one encoder's $n \times d$ embedding matrix, such as `EMBEDDINGS["nt500m"]`.
- $F_s$ is the $n \times 88$ matrix of standardized composition features plus a column of ones for the intercept.
- $\hat B$ is the least-squares coefficient matrix, with one column per embedding coordinate.
- $X^{\perp}$ is the residual embedding, with the same shape as $X$. Its columns are orthogonal to the fitted design columns on these rows, up to numerical precision.

In the helper, the mathematical $F_s$ is named `design`; the code variable `F_s` holds the 87 standardized features before the intercept is added.

The share of embedding variance that composition explains is

$$
R^2_{\mathrm{comp}}(X) = 1 - \frac{\lVert X^{\perp}\rVert_F^2}{\lVert X - \bar X\rVert_F^2},
$$

where $\lVert\cdot\rVert_F^2$ sums the squares of every entry and $\bar X$ repeats the column means in every row. This is an in-sample fit. The notebook prints the dimension-count ratio $88/n$ as a chance reference, but it is a heuristic rather than a calibrated null: the composition features are dependent, so the effective design rank need not equal the number of columns. The helper does not estimate a null distribution.

**3. Repeat the comparisons on the residuals.**

For every pair of encoders, the cell reports linear CKA and CCA Recall@1 before and after residualization. Recall@1 uses the same PCA-CCA pipeline and the same `train_idx` and `test_idx` rows as [Canonical correlation and cross-modal retrieval](#canonical-correlation-and-cross-modal-retrieval), so its "before" value should match the Recall@1 printed in the [earlier notebook retrieval output](../../Code/Stage1_stability.ipynb#Canonical-correlation-and-cross-modal-retrieval).

**Interpretation.** Residualization removes the fitted linear component associated with the specified features on these rows. A decrease shows that the comparison is sensitive to removing that component. Nonlinear composition effects and other shared influences can remain, so residual agreement alone does not identify a biological mechanism.

**Connection to the code.** `sequences.composition_features` computes one row of $\mathbf f(s)$, counting non-overlapping codons from the first nucleotide to match `translate_cds`. `analysis.residualize` fits the least-squares regression on all retained rows and returns $X^{\perp}$, and `analysis.composition_share` computes $R^2_{\mathrm{comp}}$. The regression uses composition only, never stability labels. It is linear, so nonlinear functions of composition can remain in the residuals.

**Fitting boundary.** Residualization is fitted on all retained rows before the train/test indices select inputs to PCA and CCA. PCA/CCA are still fitted on their training subset, but the preceding residualization includes their test rows. No stability labels enter this composition fit.

## High- and low-variance directions

Return to [High- and low-variance directions](../../Code/Stage1_stability.ipynb#High--and-low-variance-directions) in the notebook.

An embedding's variance is usually concentrated in a few directions. This analysis asks whether agreement between encoders, composition, and stability signal sit in those dominant directions or in the many low-variance ones. The protein encoders are the main case: they agree strongly with each other, even after the composition control, and the question is which directions carry that agreement.

**1. Split each embedding into two blocks of principal components.**

`analysis.pca_bands` fits a principal component analysis (PCA) to one encoder's full embedding matrix, without labels, and returns each row's coordinates on the first `PCA_TOP_K` components and on all remaining components:

$$
X \;\longmapsto\; \big(Z_{\mathrm{top}},\; Z_{\mathrm{rest}}\big).
$$

In this notation:

- $X$ is one encoder's $n \times d$ embedding matrix, such as `EMBEDDINGS["esm2_35m"]`.
- $Z_{\mathrm{top}}$ holds each row's coordinates on the `PCA_TOP_K = 10` directions of largest variance, with shape `(n, 10)`.
- $Z_{\mathrm{rest}}$ holds the coordinates on every remaining direction, up to $\min(n-1, d) - 10$ columns.
- The arrow means "is split into"; the two blocks are uncorrelated and together retain the centered sample's variance, up to numerical precision. This refers to the observed sample, not every possible unseen input.

**2. Measure each block separately.**

The first table reports, for each encoder, the share of total variance in the top block, the composition share $R^2_{\mathrm{comp}}$ of each block as defined in the [composition control](#composition-control), and the mean stability $R^2$ of a probe on each block with the pinned folds, next to the whole embedding's value. The heatmaps then repeat linear CKA between every pair of encoders within each block: the top block of one encoder against the top block of the other, and likewise for the remaining blocks.

The PCA is fit on all rows without the stability labels, so held-out rows help choose the directions but never the target. PCA is therefore not refitted within each probe fold. The composition shares remain in-sample fits, and the predictor-count ratio discussed in the [composition control](#composition-control) is a heuristic rather than a calibrated null for either block.

## Sample-level alignment and prediction error

Return to [Sample-level alignment and prediction error](../../Code/Stage1_stability.ipynb#Sample-level-alignment-and-prediction-error) in the notebook.

This diagnostic asks whether examples whose representations in two modalities are more similar also tend to be easier to predict. It is run once for every pair of encoders. It is an **exploratory sample-level adaptation** inspired by [Tjandrasuwita et al. (ICML 2025), Sections 5-6](https://proceedings.mlr.press/v267/tjandrasuwita25a.html). The paper studies relationships across models/configurations and a contrastive-training intervention; this code compares examples from one fixed encoder pair at a time. It does not reproduce those experiments.

For each pair `(a, b)`, the call in [Run and interpret the alignment diagnostics](../../Code/Stage1_stability.ipynb#Run-and-interpret-the-alignment-diagnostics) supplies the two coordinate matrices stored in `cca_results[(a, b)]["test_coords"]` and `labels[test_idx]`: matched CCA coordinates and stability labels from the earlier CCA test subset. Corresponding rows must identify the same examples. The encoder weights and the earlier fitted PCA/CCA transformations remain fixed here; the new models fitted in this diagnostic are ordinary linear regressions.

**1. Compare the directions of each example's paired CCA vectors.**

For nonzero vectors, cosine similarity is:

$$
c_i = \frac{\mathbf u_i^\top\mathbf v_i}
{\|\mathbf u_i\|_2\,\|\mathbf v_i\|_2}.
$$

In this notation:

- $n$ is the number of paired examples supplied to this diagnostic; each input coordinate matrix has shape `(n, d)`.
- $i$ identifies one example, from 1 through $n$ in the equations. Python indexes those rows from 0 through `n - 1`.
- $d$ is the number of CCA coordinates per vector, currently 10.
- $\mathbf u_i$ and $\mathbf v_i$ are the CCA vectors of the pair's first and second encoders for example $i$. Bold symbols denote vectors; each has $d$ coordinates. The mathematics treats them as column vectors even though the NumPy matrices store examples in rows.
- $\top$ means transpose: it turns the mathematical column $\mathbf u_i$ into a row so that $\mathbf u_i^\top\mathbf v_i$ is a dot product. This multiplies corresponding coordinates and sums those products, giving one scalar.
- $\|\mathbf u_i\|_2$ and $\|\mathbf v_i\|_2$ are Euclidean lengths: square the coordinates, add them, and take the square root. The subscript 2 names the norm, not the number of coordinates.
- $c_i$ is one cosine score, between $-1$ and $1$ for nonzero vectors. A larger value means their directions are more similar in the fitted CCA coordinates. It is a chosen per-example alignment score, not a biological measurement.

The notebook delegates this calculation to [scikit-learn's `cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html). Each `i:i+1` slice retains shape `(1, d)`; the resulting `(1, 1)` similarity matrix contains one score, selected by `[0, 0]`. The displayed ratio is undefined for a zero vector, whereas the library returns zero for a comparison involving a zero vector; that value does not define its direction.

**2. Combine the coordinates and obtain an error from a model that did not train on that row.**

Concatenation appends the second encoder's coordinates after the first encoder's coordinates:

$$
\mathbf z_i = [\mathbf u_i;\mathbf v_i].
$$

In this notation:

- $\mathbf z_i$ is the combined feature vector for example $i$, with $2d$ coordinates, currently 20.
- The brackets and semicolon mean to append the two vectors' coordinates in the stated order. They do not average or multiply the vectors.
- The NumPy array `fused` stores these combined vectors as rows, with shape `(n, 2d)`.

The code divides these $n$ examples into five shuffled folds using `SEED`. For each fold, scikit-learn's `LinearRegression().fit(...)` fits an intercept and coefficients on the other folds by ordinary least squares. Its `.predict(...)` call predicts the held-out fold. The notebook then calculates each row's absolute error:

$$
e_i = \left|y_i - \hat f_{-F(i)}(\mathbf z_i)\right|.
$$

In this notation:

- $y_i$ is the observed stability label for example $i$, a scalar.
- $F(i)$ identifies the set of rows in the held-out regression fold containing example $i$.
- $\hat f_{-F(i)}$ is the regression predictor fitted using all the other folds. The hat marks a fitted predictor; the subscript $-F(i)$ means that this fold was excluded from fitting, not that a negative function was calculated.
- $\hat f_{-F(i)}(\mathbf z_i)$ is the scalar prediction obtained by giving that predictor the combined vector for row $i$.
- $e_i$ is the absolute prediction error. The single vertical bars mean absolute value: predictions equally far above and below the observed label have equal error. They are different from the double bars used for vector norms above.

The mathematical name $\hat f_{-F(i)}$ describes each fitted `reg` object; there is no Python function with that name. This diagnostic fits its regression folds within the earlier CCA test subset. Those rows were held out when PCA and CCA were fitted, and each is held out again from its own regression fit. No PCA or CCA refitting occurs inside the regression loop.

**3. Relate alignment ranks to error ranks.**

$$
\rho_s = \operatorname{corr}\!\left(
\operatorname{rank}(\mathbf c),\operatorname{rank}(\mathbf e)
\right).
$$

In this notation:

- $\mathbf c$ collects the $n$ scores $c_i$ in row order, stored as `local_alignment` with shape `(n,)`.
- $\mathbf e$ collects the matching $n$ errors $e_i$, stored as `errors` with shape `(n,)`.
- $\operatorname{rank}$ replaces values by their positions when sorted from smallest to largest, retaining the original row order. Ties receive their average rank; for example, values `(4, 4, 9)` have ranks `(1.5, 1.5, 3)`.
- $\operatorname{corr}$ denotes Pearson correlation between the two rank vectors. Applying it to ranks produces Spearman correlation.
- $\rho_s$, read as "rho sub s," is the resulting scalar Spearman coefficient; the subscript identifies the rank-correlation statistic.

SciPy's [`spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) performs the ranking and correlation and returns a coefficient and nominal p-value. The notebook does not separately create rank arrays. If all alignment scores or all errors are equal, their rank correlation is undefined; SciPy returns `NaN` with a constant-input warning. A negative coefficient means higher alignment tends to accompany smaller errors; a near-zero coefficient indicates little rank association in this sample. `analysis.alignment_error_association` returns this coefficient and its nominal p-value; it does not fit or return a regression slope between alignment and error.

**Small worked illustration.** Suppose three examples have cosine scores `(0.2, 0.5, 0.9)` and held-out absolute errors `(0.8, 0.4, 0.1)`. Their ranks are `(1, 2, 3)` and `(3, 2, 1)`, so the Spearman coefficient is $-1$: higher alignment accompanies lower error in this illustrative ordering. These are invented scores, not a three-row execution of the five-fold procedure.

The earlier stability CCA split groups identical sequence strings; the GTEx CCA split groups transcripts by gene. These later regression folds instead use ungrouped `KFold` inside the earlier CCA test subset, so they do not keep those groups together. Shared fitted transformations, overlapping training folds, and repeated sequences limit a simple independent-observation interpretation of the p-value. No result here estimates the effect of training with an alignment loss or establishes modality uniqueness; those questions require separate controlled comparisons.

## Representational similarity analysis

Return to [Representational similarity analysis](../../Code/Stage1_stability.ipynb#Representational-similarity-analysis) in the notebook.

Representational similarity analysis (RSA) asks whether pairs of examples that are dissimilar within one representation also tend to be dissimilar within the other. For every pair of encoders, the call in [Run and interpret the alignment diagnostics](../../Code/Stage1_stability.ipynb#Run-and-interpret-the-alignment-diagnostics) supplies the two original final-layer matrices from `EMBEDDINGS`, rather than the CCA projections. Corresponding rows must identify the same examples in the same order; the feature counts can differ. This diagnostic fits no predictor or projection.

**1. Center each example across its own features.**

For one embedding of the pair's first modality, first find its mean feature value:

$$
\bar x_i = \frac{1}{p}\sum_{k=1}^{p}x_{ik}.
$$

In this notation:

- $X$ is the first encoder's embedding matrix with shape `(n, p)`; $Y$ is the second encoder's embedding matrix with shape `(n, q)`. For Nucleotide Transformer 500M and ESM-2 35M, for example, these are `EMBEDDINGS["nt500m"]` and `EMBEDDINGS["esm2_35m"]`.
- $n$ is the number of matched examples, while $p$ and $q$ are the two encoders' feature counts. The latter may differ.
- $i$ identifies an example row, from 1 through $n$ in the equations. Python indexes those same rows from 0 through `n - 1`.
- $k$ identifies a feature of $X$, from 1 through $p$; $x_{ik}$ is the scalar value of feature $k$ in example $i$. Python feature indices begin at 0.
- $\bar x_i$, read as "x bar sub i," is the mean of the feature values within row $i$, a scalar. The bar marks that average.
- $\sum_{k=1}^{p}$ adds this row's $p$ feature values, and dividing by $p$ takes their mean.

Subtract that mean from every coordinate of the row:

$$
\widetilde{\mathbf x}_i
= \mathbf x_i - \bar x_i\mathbf 1_p.
$$

In this notation:

- $\mathbf x_i$ is the full embedding vector of $X$ for example $i$, with $p$ coordinates. Bold symbols denote vectors.
- $\mathbf 1_p$ is a vector of $p$ ones. Multiplying it by the scalar $\bar x_i$ gives a vector with that mean in every coordinate.
- $\widetilde{\mathbf x}_i$, read as "x tilde sub i," is the centered row vector; the tilde distinguishes it from the original embedding. Its length is still $p$ coordinates.
- The subtraction acts coordinate by coordinate. It centers each row across its features, unlike CKA's centering of each feature across example rows.

The same construction applies to a row of $Y$ using its own $q$ coordinates and its own mean. These are explanatory intermediate quantities for the correlation-distance calculation inside SciPy; the notebook does not create variables named `x_bar` or `x_tilde`.

**2. Calculate the correlation distance between two rows of the same representation.**

$$
d_X(i,j)
= 1 - \frac{\widetilde{\mathbf x}_i^\top\widetilde{\mathbf x}_j}
{\|\widetilde{\mathbf x}_i\|_2\,\|\widetilde{\mathbf x}_j\|_2}.
$$

In this notation:

- $i$ and $j$ identify two different examples in $X$; each vector has the same $p$ features within this representation.
- $d_X(i,j)$ is their scalar correlation distance. The subscript $X$ names the representation being compared, and the parentheses identify the pair of rows.
- $\top$ means transpose. Treating the mathematical vectors as columns, it turns the first vector into a row so their product is a dot product: multiply corresponding coordinates and sum the products.
- $\|\cdot\|_2$ is Euclidean length: square the vector's coordinates, sum those squares, and take the square root. The subscript 2 identifies the norm.
- The fraction is the Pearson correlation of the two original feature profiles, calculated by normalizing the centered dot product. Subtracting it from 1 changes similarity into distance.

Correlation 1 gives distance 0, correlation 0 gives distance 1, and correlation $-1$ gives distance 2. A row whose features are all identical becomes a zero vector after centering, making this distance undefined. The helper adds no special handling for that case. See [SciPy's definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html). Applying the same calculation within $Y$ gives $d_Y(i,j)$, using its $q$ features; no individual feature of $X$ is matched to an individual feature of $Y$.

**3. Collect matching pairs and compare their distance rankings.**

Each representation contributes one distance for every unordered pair of distinct examples. The number of pairs is:

$$
M = \frac{n(n-1)}{2}.
$$

In this notation:

- $M$ is the number of distances in each comparison vector.
- There are $n$ choices for the first example and $n-1$ other examples. Dividing by 2 removes the reversed duplicate of every pair.
- Keeping only $i<j$ selects each distinct pair once and excludes comparisons of a row with itself. For $n=3$, the order is $(1,2)$, $(1,3)$, $(2,3)$; Python labels those pairs `(0, 1)`, `(0, 2)`, `(1, 2)`.

SciPy's [`pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) returns these values as a one-dimensional condensed distance vector, rather than a square distance matrix. Because the input rows correspond, the two distance vectors use matching pair order. For a 981-row input, each vector contains 480,690 pair distances; those distances are not 480,690 independent examples.

The final statistic compares the ranks of the two distance vectors:

$$
\rho_{\mathrm{RSA}}
= \operatorname{corr}\!\left(
\operatorname{rank}(\mathbf d_X),
\operatorname{rank}(\mathbf d_Y)
\right).
$$

In this notation:

- $\mathbf d_X$ and $\mathbf d_Y$ are the distance vectors of $X$ and $Y$, each with $M$ scalar entries in matching pair order. `dist_a` and `dist_b` in the table below are illustrative aliases; the actual helper passes both `pdist` results directly into `spearmanr`.
- $\operatorname{rank}$ assigns positions from smallest to largest distance, retaining the original pair order and assigning tied distances their average rank.
- $\operatorname{corr}$ denotes Pearson correlation of those rank vectors, which is Spearman correlation.
- $\rho_{\mathrm{RSA}}$, read as "rho sub RSA," is the resulting scalar coefficient. The subscript labels this RSA statistic.

If either distance vector is constant, the rank correlation is undefined and SciPy returns `NaN` with a constant-input warning. This is distinct from a single constant feature profile making its pair distances undefined. A positive coefficient indicates agreement in which pairs are more or less dissimilar. It describes geometry; it does not identify a particular shared subspace or biological mechanism.

**Connection to the code.** `analysis.compute_rsa` is a wrapper around three library calls. The equations explain those library calculations; the code contains no handwritten centering, distance, or ranking implementation.

| Code operation | Input shape → output shape | Connection to the mathematics |
|---|---|---|
| `pdist(embeddings_a, metric="correlation")` | `(n, p)` → `(M,)` | Calculate the first encoder's correlation distances for each distinct pair, producing $\mathbf d_X$. |
| `pdist(embeddings_b, metric="correlation")` | `(n, q)` → `(M,)` | Calculate the second encoder's distances for the same pairs, producing $\mathbf d_Y$. |
| `spearmanr(dist_a, dist_b)` (illustrative aliases) | Two `(M,)` arrays → two scalar results | Rank each distance vector and return their Spearman coefficient and nominal p-value. |

**Small worked illustration.** Rows `(1, 2, 3)` and `(2, 4, 6)` of $X$ have means 2 and 4. Their centered profiles are `(-1, 0, 1)` and `(-2, 0, 2)`: identical directions with different magnitudes, giving correlation 1 and distance 0. Adding a third row `(3, 2, 1)` gives the opposite centered direction, so the pair distances in $X$ are `(0, 2, 2)`. If the corresponding distances in $Y$ are `(0.1, 0.9, 0.9)`, both distance vectors have ranks `(1, 2.5, 2.5)` and RSA is 1. This illustrates agreement in distance rankings without equality of distance values; these are invented numbers, not encoder outputs.

Pairwise distances share examples, so their entries are statistically dependent. The default Spearman p-value is therefore a nominal summary, not a calibrated significance test for this geometry comparison.

**Mantel permutation procedure.** Shuffling complete row identities preserves dependence among distances sharing a row, unlike shuffling individual distances. The code reorders the second distance structure and counts how often its correlation reaches the observed value:

$$
p_{\mathrm{Mantel}} = \frac{1 + \#\{\pi : r_\pi \ge r_{\mathrm{obs}}\}}{P + 1}.
$$

In this notation:

- $r_{\mathrm{obs}}$ is the observed RSA Spearman correlation.
- $\pi$ is one random reordering of the $n$ sequences, applied to the rows and columns of $Y$'s distance matrix, and $r_\pi$ is the RSA correlation after that reordering.
- $P$ is the number of reorderings, 499 here, and $\#\{\cdot\}$ counts the reorderings in the set.
- Adding 1 to the numerator and denominator counts the observed arrangement as one possible arrangement, so the smallest attainable value is $1/500 = 0.002$.

**Interpretation.** The calculation is one-sided: only shuffled correlations at least as large as the observed one are counted. Its significance interpretation requires row identities to be exchangeable under the chosen null. Preserving distance-matrix structure does not establish that assumption. This implementation shuffles individual rows, not duplicate-sequence or gene groups, so repeated or biologically related samples require additional justification.

**Connection to the code.** `analysis.mantel_p_value` ranks both distance vectors once, then reorders the rows and columns of the second rank matrix for each permutation. Spearman correlation is the Pearson correlation of these ranks, so the observed value equals the one returned by `compute_rsa`. RSA also reuses embeddings examined by the earlier metrics, so it is not independent confirmation of their conclusions.
