# Comparing and interpreting representations

Once sequences have been encoded, we can ask two related questions: do the encoders organize the same examples similarly, and which sequence positions receive attention inside a nucleotide encoder? The first concerns representation geometry across examples; the second concerns model behavior within a sequence. Neither representation similarity nor an attention pattern alone establishes predictive benefit or a biological mechanism.

This guide explains the methods behind the [stability notebook's representation comparisons](Stage1_stability.ipynb#Track-B:-Cross-modal-geometry) and exploratory attention analysis. GTEx reuses the comparisons in [Track B](Stage1_gtex.ipynb#GTEx-Track-B:-Cross-modal-geometry) and its [alignment and similarity diagnostics](Stage1_gtex.ipynb#Alignment,-prediction-error,-and-representational-similarity-on-GTEx). The notebooks retain experiment settings, code calls, saved outputs and result interpretations. Return links below lead to the stability walkthrough unless labeled GTEx.

Read [Data and biological targets](../../docs/methods/data-and-targets.md) for what the observations and labels mean, and [From sequences to predictions](../../docs/methods/sequences-to-predictions.md) for sequence preparation, pooling and target prediction. Here, [comparing representations](#comparing-representations) asks how the encoded examples relate across feature spaces, while [attention and candidate motifs](#attention-and-candidate-motifs) asks how model attention relates to sequence-pattern annotations.

## Comparing representations

All comparisons require corresponding rows to identify the same examples. Two encoders may have different feature widths. Similarity, fitted correspondence and predictive benefit answer different questions; a high score on one does not establish the others. The fitting boundary in each method matters when interpreting its result.

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

### Final-layer representation similarity: linear CKA

Return to [Final-layer representation similarity: linear CKA](Stage1_stability.ipynb#Final-layer-representation-similarity:-linear-CKA) in the notebook.

Linear centered kernel alignment (CKA) compares the final-layer mean-pooled representations of the same retained examples, one pair of encoders at a time, for every pair in `PAIRS`. Corresponding rows refer to the same example, while the feature counts can differ.

Each calculation uses two separate embedding matrices, not their concatenation. The helper centers each feature across rows, computes dot products within and between representations, and normalizes the comparison as derived in [Linear centered kernel alignment](#linear-centered-kernel-alignment).

CKA uses no stability labels and fits no prediction model. It measures representational similarity; the probes ask which target properties a fitted model can recover.

In the derived stability setting, pairs within one modality provide a reference for pairs across modalities. The nucleotide encoders start from the same coding sequence in DNA or RNA letters; protein encoders receive its translation. Architecture, pretraining, tokenization, model size, and input truncation can all affect this comparison. GTEx has distinct inputs, including full transcripts for single-nucleotide RNA encoders and CDS inputs for codon encoders.

#### Linear centered kernel alignment

`analysis.linear_cka(X, Y)` compares representations of the same examples. `X` has shape `(n, p)` and `Y` has shape `(n, q)`: their feature counts may differ, but corresponding rows must identify the same example. Capital `Y` here is a second embedding matrix, distinct from the lowercase target `y` used by the [linear probe](../../docs/methods/sequences-to-predictions.md#linear-probing).

**1. Center each feature across examples.**

For feature $j$ of the first representation, calculate its mean:

$$
\bar x_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij}.
$$

Here, $X$ and $Y$ have $n$ matched example rows and $p$ and $q$ features, respectively. The scalar $x_{ij}$ is feature $j$ of example $i$, and $\bar x_j$ is that feature's mean across rows.

Subtract that mean from each value in the feature:

$$
(X_c)_{ij}=x_{ij}-\bar x_j.
$$

The centered matrix $X_c$ retains shape $(n,p)$: every row loses the same mean for feature $j$.

The code applies the same two operations independently to `Y` to obtain $Y_c$, with shape $(n,q)$. It subtracts each representation's own column means. Unlike the probe's standardization, CKA centering does not divide by feature standard deviations.

**2. Measure feature dot-product matrices with the Frobenius norm.**

The Frobenius norm combines all entries of a matrix into a scalar size:

$$
\|A\|_F=\sqrt{\sum_{a=1}^{r}\sum_{b=1}^{s}A_{ab}^{\,2}}.
$$

For a real matrix $A$ of shape $(r,s)$, $A_{ab}$ is its entry in row $a$, column $b$. The Frobenius norm $\|\cdot\|_F$ takes the square root of the sum of squared entries across the entire matrix.

For example, the matrix with rows $(1,2)$ and $(2,4)$ has norm $\sqrt{1^2+2^2+2^2+4^2}=5$. Its squared Frobenius norm is $25$. These are illustrative values, not encoder results. NumPy implements this matrix norm with `np.linalg.norm(..., ord="fro")`; see [NumPy's norm definition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).

**3. Normalize the cross-representation quantity.**

$$
\operatorname{CKA}(X,Y)
=\frac{\|Y_c^\top X_c\|_F^2}
{\|X_c^\top X_c\|_F\,\|Y_c^\top Y_c\|_F}.
$$

The terms correspond to these matrix operations:

- Transpose $\top$ is `.T` in NumPy; adjacent matrices are multiplied with `@`.
- $Y_c^\top X_c$ has shape $(q,p)$ and contains cross-representation dot products between centered features.
- $X_c^\top X_c$ and $Y_c^\top Y_c$ have shapes $(p,p)$ and $(q,q)$ and contain within-representation feature dot products.
- `linear_cka` returns one scalar: the squared Frobenius norm of the cross-representation matrix divided by the product of the two unsquared within-representation norms.

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

The `...` stands for the matrix product in the preceding row; the subtraction applies each mean row to every example.

This feature-based calculation is equivalent to comparing the centered example-similarity matrices $X_cX_c^\top$ and $Y_cY_c^\top$, each of shape $(n,n)$. An entry in either matrix is the dot product between two centered example embeddings. The common normalization factor from the Hilbert-Schmidt independence criterion (HSIC) cancels in the CKA ratio, so the code names the unnormalized top term `numerator`.

For nonzero centered representations, linear CKA lies between zero and one in exact arithmetic. It is unchanged by orthogonal transformations, such as rotations and reflections, or by nonzero uniform scaling of either representation. Arbitrary changes of basis or separate scaling of individual features can change it. See [Kornblith et al. (2019), Section 3 and Table 1](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf).

If either centered representation is entirely zero, the denominator is zero and CKA is undefined; this helper has no special handling for that case. High CKA describes representational similarity. It does not by itself establish good target prediction or a benefit from combining the modalities.

### Layerwise representation similarity: linear CKA

Return to [Layerwise representation similarity: linear CKA](Stage1_stability.ipynb#Layerwise-representation-similarity:-linear-CKA) in the notebook.

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

**Connection to the code.** For each pair in `PAIRS`, `analysis.layerwise_cka` calls `linear_cka` for every state pair. `analysis.layerwise_summary` reports the matrix's largest entry and its final-state pair, `matrix[-1, -1]`. For twelve encoders, this produces 66 matrices; the table summarizes all pairs and heatmaps show `LAYERWISE_PLOT_PAIRS`. These descriptive comparisons do not fit cross-modal projections or establish that a particular layer pair improves prediction.

**Check the role of the second encoder's embedding output.** A maximum at state 0 of an encoder occurs at the initial embedding output before the later encoder layers. For each pair, `layerwise_summary` also repeats the maximum search after excluding state 0 of the second encoder in the pair.

The slice `matrix[:, 1:]` keeps every state of the first encoder, including its state 0, and only states 1 onward of the second. Adding 1 to the returned column index maps the sliced matrix back to the original state numbering. This asks whether another high-scoring pair remains after removing that embedding output; it does not exclude the first encoder's initial embedding state.

### Neighborhood overlap across representations

Return to [Neighborhood overlap across representations](Stage1_stability.ipynb#Neighborhood-overlap-across-representations) in the notebook.

CKA summarizes similarity across representation matrices. Neighborhood overlap asks a more local question: for a particular retained row, are the examples nearest to it in one representation also nearest to it in another? The notebook computes it for every pair of encoders. Neighbor identities refer to the same paired rows in both matrices, not to equal feature coordinates.

**1. Count shared neighbors for each row.**

$$
o_i = \frac{\left|N_X(i)\cap N_Y(i)\right|}{k}.
$$

The comparison uses:

- $X$ and $Y$, the two embedding matrices of one encoder pair, such as `EMBEDDINGS["nt500m"]` with shape `(n, p)` and `EMBEDDINGS["esm2_35m"]` with shape `(n, q)`. They contain $n$ matched examples and can have different feature counts.
- $k$ retained neighbors per query row $i$, with $k=10$ in the notebook. Requesting one extra neighbor requires $1\le k\le n-1$.
- $N_X(i)$ and $N_Y(i)$, the retained neighbor-row index sets in each space. Their intersection contains the shared indices, and $|\cdot|$ counts them.
- $o_i$, the shared count divided by $k$: one fraction between zero and one per query.

For example, if 3 of a row's 10 retained neighbors are shared, its fraction is $o_i=3/10=0.3$.

**2. Average the row fractions.**

$$
\operatorname{overlap@}k
=\frac{1}{n}\sum_{i=1}^{n}o_i.
$$

$\operatorname{overlap@}k$ is the mean of the $n$ query-level fractions $o_i$; it gives every query equal weight.

**Connection to the code.** `analysis.mutual_knn_alignment` calculates each $o_i$ in its `overlaps` list and returns their mean with `np.mean(overlaps)`. Despite the helper's name, this is overlap between two neighbor sets, not a check that neighbor relationships are reciprocal.

The helper uses the default Euclidean distance of [`NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). It requests $k+1$ neighbors and removes the first returned index with `[i, 1:]`, intending to exclude the query itself. With duplicate vectors or distance ties, that first index need not be the query's own index; explicit self-exclusion remains an evaluation issue. The sets in the equations describe what the code retains under this rule, rather than guaranteeing that all retained neighbors are other rows.

**The chance reference uses an idealized selection rule.** For two independent, uniformly selected sets of $k$ nonself neighbors among $n-1$ candidates, the expected overlap fraction is $k/(n-1)$, the printed reference. Each neighbor in one set has that probability of appearing in the other. This is neither a simulated random-embedding result nor a significance test; disagreement with CKA would not by itself invalidate either metric.

### Canonical correlation and cross-modal retrieval

Return to [Canonical correlation and cross-modal retrieval](Stage1_stability.ipynb#Canonical-correlation-and-cross-modal-retrieval) in the notebook.

The previous metrics compare the representations without learning a cross-modal projection. Canonical correlation analysis (CCA) instead fits linear combinations of features that vary together across matched examples of two representations. The notebook fits a separate CCA for every pair of encoders, querying with the encoder that comes first in `ENCODER_KEYS` and searching the second; the matrices show each pair's value in both positions. The explanation below uses the first encoder's representation as $X$ and the second's as $Y$. It can reveal correspondence in selected directions even when a whole-matrix comparison is modest.

**1. Fit the transformations on training rows.**

`splits.grouped_holdout_split` assigns each retained sequence to training or testing from the SHA-256 hash of `SEED` and the sequence string. The [implementation](../shared_code/splits.py) uses a threshold targeting a 30% test share; it does not balance row counts, and every copy of an identical sequence stays on the same side. The same indices select every encoder's matrix, giving every pair the same training and test rows.

This CCA call receives `SEED`; `probe_table` separately receives `PROBE_SEED` for its pinned folds, as explained in the [linear-probing guide](../../docs/methods/sequences-to-predictions.md#evaluation-and-interpretation). Write $n_{\mathrm{train}}$ and $n_{\mathrm{test}}$ for the numbers of training and test pairs, and $p$ and $q$ for the pair's embedding widths.

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

The projected quantities have different roles:

- $\widetilde X_{\mathrm{train}}$ and $\widetilde Y_{\mathrm{train}}$ contain paired training rows after PCA reduction and CCA's internal centering and scaling. Each has shape `(n_train, 50)`; the tilde marks this preprocessing.
- $\mathbf a$ and $\mathbf b$ are coefficient vectors with 50 entries, one per PCA feature in the respective representation.
- $\mathbf u$ and $\mathbf v$ each contain $n_{\mathrm{train}}$ coordinates, one per training example. Multiplying a processed matrix by its coefficient vector produces these coordinates; they are distinct from the 50 coefficients.

For a small illustration with only two features, a processed row `[2, 3]` and coefficients `[0.5, -1]` give the single coordinate $2(0.5)+3(-1)=-2$. The actual calculation combines 50 features per row. These illustrative values are not fitted notebook results.

The first-pair objective is

$$
(\mathbf a^\star,\mathbf b^\star)
=
\underset{\mathbf a,\mathbf b}{\operatorname{arg\,max}}\;
\operatorname{corr}\!\left(\widetilde X_{\mathrm{train}}\mathbf a,\,
                          \widetilde Y_{\mathrm{train}}\mathbf b\right).
$$

Here, $\operatorname{corr}$ is Pearson correlation across paired training rows; both projected vectors need nonzero variance. The $\operatorname{arg\,max}$ selects coefficient vectors $\mathbf a^\star,\mathbf b^\star$ that maximize it while keeping the processed training matrices fixed. It returns those vectors, not the correlation value.

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

For retrieval:

- $k$ is the candidate count, reported at 1, 5 and 10; the formula assumes $1\le k\le n_{\mathrm{test}}$.
- $i$ identifies a paired row within the test arrays, not its original dataset row number.
- $R_k(i)$ contains the first $k$ retrieved indices of the second encoder. The indicator $\mathbf{1}\{i\in R_k(i)\}$ is 1 when the designated match is present and 0 otherwise.
- $\operatorname{Recall@}k$ averages those indicators over test queries. There is one designated correct match per query.

**Plain-English interpretation.** Recall@k is the fraction of test queries whose designated partner appears among the first $k$ retrieved rows in the fitted CCA coordinates. A different row containing an identical sequence does not count as that partner.

For example, if query 4 returns protein indices `[8, 4, 2]`, it fails at $k=1$ but succeeds at $k=3$. In the code, `i in neighbors[i, :k]` performs the membership check, and the mean over queries gives the reported fraction. Tied distances at the cutoff inherit the library's returned ordering.

**The chance reference is a ranking calculation.** A uniformly random ranking has expected success $k/n_{\mathrm{test}}$ for $k\le n_{\mathrm{test}}$: the one designated match has $k$ successful positions among $n_{\mathrm{test}}$ possible positions. The notebook prints the number of test rows and the top-1 chance $1/n_{\mathrm{test}}$. The actual test count and corresponding reference stay with the notebook output. This is a ranking reference, not a $p$-value.

Training the transformations only on training rows avoids fitting those transformations to test rows. The split keeps identical sequences together, but not biologically related ones, such as homologous genes. Neither CCA fitting nor retrieval uses the stability labels.

### Composition control

Return to [Composition control](Stage1_stability.ipynb#Composition-control) in the notebook.

The [layer-wise CKA](Stage1_stability.ipynb#Layerwise-representation-similarity:-linear-CKA) and [GC-content](Stage1_stability.ipynb#GC-content-prediction) results suggest a hypothesis: the representations may agree largely because each tracks sequence composition. This analysis subtracts the component predicted linearly by specified composition features from every embedding, then repeats two of the preceding notebook comparisons.

**1. Describe each sequence by its composition.**

$$
\mathbf f(s) = \big(\mathrm{GC}(s),\ \mathrm{GC3}(s),\ \log L(s),\ c_1(s),\ldots,c_{64}(s),\ a_1(s),\ldots,a_{20}(s)\big).
$$

In this notation:

- $s$ is one retained coding sequence, and $L(s)$ is its length in nucleotides.
- $\mathrm{GC}(s)$ and $\mathrm{GC3}(s)$ are the GC fractions over the whole sequence and over third codon positions, as defined in [GC-content prediction](Stage1_stability.ipynb#GC-content-prediction) and [GC3 prediction](Stage1_stability.ipynb#GC3-prediction).
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

For every pair of encoders, the cell reports linear CKA and CCA Recall@1 before and after residualization. Recall@1 uses the same PCA-CCA pipeline and the same `train_idx` and `test_idx` rows as [Canonical correlation and cross-modal retrieval](#canonical-correlation-and-cross-modal-retrieval), so its "before" value should match the Recall@1 printed in the [earlier notebook retrieval output](Stage1_stability.ipynb#Canonical-correlation-and-cross-modal-retrieval).

**Interpretation.** Residualization removes the fitted linear component associated with the specified features on these rows. A decrease shows that the comparison is sensitive to removing that component. Nonlinear composition effects and other shared influences can remain, so residual agreement alone does not identify a biological mechanism.

**Connection to the code.** `sequences.composition_features` computes one row of $\mathbf f(s)$, counting non-overlapping codons from the first nucleotide to match `translate_cds`. `analysis.residualize` returns the all-row least-squares residual $X^{\perp}$, and `analysis.composition_share` computes $R^2_{\mathrm{comp}}$.

**Fitting boundary.** Residualization is fitted on all retained rows before the train/test indices select inputs to PCA and CCA. PCA/CCA are still fitted on their training subset, but the preceding residualization includes their test rows. No stability labels enter this composition fit.

### High- and low-variance directions

Return to [High- and low-variance directions](Stage1_stability.ipynb#High--and-low-variance-directions) in the notebook.

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

### Sample-level alignment and prediction error

Return to [Sample-level alignment and prediction error](Stage1_stability.ipynb#Sample-level-alignment-and-prediction-error) in the notebook.

This diagnostic asks whether examples whose representations in two modalities are more similar also tend to be easier to predict. It is run once for every pair of encoders. It is an **exploratory sample-level adaptation** inspired by [Tjandrasuwita et al. (ICML 2025), Sections 5-6](https://proceedings.mlr.press/v267/tjandrasuwita25a.html). The paper studies relationships across models/configurations and a contrastive-training intervention; this code compares examples from one fixed encoder pair at a time. It does not reproduce those experiments.

For each pair `(a, b)`, the call in [Run and interpret the alignment diagnostics](Stage1_stability.ipynb#Run-and-interpret-the-alignment-diagnostics) supplies the two coordinate matrices stored in `cca_results[(a, b)]["test_coords"]` and `labels[test_idx]`: matched CCA coordinates and stability labels from the earlier CCA test subset. Corresponding rows must identify the same examples. The encoder weights and the earlier fitted PCA/CCA transformations remain fixed here; the new models fitted in this diagnostic are ordinary linear regressions.

**1. Compare the directions of each example's paired CCA vectors.**

For nonzero vectors, cosine similarity is:

$$
c_i = \frac{\mathbf u_i^\top\mathbf v_i}
{\|\mathbf u_i\|_2\,\|\mathbf v_i\|_2}.
$$

For this diagnostic:

- $\mathbf u_i$ and $\mathbf v_i$ are paired CCA vectors for example $i$, with $d=10$ coordinates. Their input matrices have shape `(n, d)` for $n$ paired examples.
- The equation treats these as column vectors, while NumPy stores examples in rows. Their transposed product is a scalar dot product.
- $\|\mathbf u_i\|_2$ and $\|\mathbf v_i\|_2$ are Euclidean lengths.
- $c_i$ is one cosine score in $[-1,1]$ for nonzero vectors. Larger values mean more similar directions in the fitted CCA coordinates, not a biological measurement.

`analysis.alignment_error_association` calls [scikit-learn's `cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) for each paired row. The displayed ratio is undefined for a zero vector, whereas the library returns zero for a comparison involving one; that value does not define its direction.

**2. Combine the coordinates and obtain an error from a model that did not train on that row.**

Concatenation appends the second encoder's coordinates after the first encoder's coordinates:

$$
\mathbf z_i = [\mathbf u_i;\mathbf v_i].
$$

$\mathbf z_i$ appends the two vectors in order, giving $2d=20$ coordinates for example $i$. The array `fused` stores these combined vectors as rows with shape `(n, 2d)`.

The code divides these $n$ examples into five shuffled folds using `SEED`. For each fold, scikit-learn's `LinearRegression().fit(...)` fits an intercept and coefficients on the other folds by ordinary least squares. Its `.predict(...)` call predicts the held-out fold. The notebook then calculates each row's absolute error:

$$
e_i = \left|y_i - \hat f_{-F(i)}(\mathbf z_i)\right|.
$$

The prediction and error are defined by:

- $y_i$, the observed scalar stability label.
- $F(i)$, the held-out regression fold containing example $i$, and $\hat f_{-F(i)}$, the predictor fitted on the other folds. Giving it $\mathbf z_i$ produces that row's scalar prediction.
- $e_i$, the absolute difference between prediction and label: equally distant overpredictions and underpredictions have the same error.

These regression folds lie within the earlier CCA test subset. Those rows were held out when PCA and CCA were fitted, and each is held out again from its own regression fit. No PCA or CCA refitting occurs inside the regression loop.

**3. Relate alignment ranks to error ranks.**

$$
\rho_s = \operatorname{corr}\!\left(
\operatorname{rank}(\mathbf c),\operatorname{rank}(\mathbf e)
\right).
$$

The rank comparison uses:

- $\mathbf c$ and $\mathbf e$, the matching row-ordered cosine scores and errors, stored as `local_alignment` and `errors`, each with shape `(n,)`.
- $\operatorname{rank}$, which replaces values by their sorted positions while retaining row order. Ties receive their average rank; for example, values `(4, 4, 9)` have ranks `(1.5, 1.5, 3)`.
- $\rho_s$, the scalar Spearman coefficient obtained by Pearson correlation of those rank vectors.

SciPy's [`spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) performs the ranking and correlation and returns a coefficient and nominal p-value. The notebook does not separately create rank arrays. If all alignment scores or all errors are equal, their rank correlation is undefined; SciPy returns `NaN` with a constant-input warning. A negative coefficient means higher alignment tends to accompany smaller errors; a near-zero coefficient indicates little rank association in this sample. `analysis.alignment_error_association` returns this coefficient and its nominal p-value; it does not fit or return a regression slope between alignment and error.

**Small worked illustration.** Suppose three examples have cosine scores `(0.2, 0.5, 0.9)` and held-out absolute errors `(0.8, 0.4, 0.1)`. Their ranks are `(1, 2, 3)` and `(3, 2, 1)`, so the Spearman coefficient is $-1$: higher alignment accompanies lower error in this illustrative ordering. These are invented scores, not a three-row execution of the five-fold procedure.

The earlier stability CCA split groups identical sequence strings; the GTEx CCA split groups transcripts by gene. These later regression folds instead use ungrouped `KFold` inside the earlier CCA test subset, so they do not keep those groups together. Shared fitted transformations, overlapping training folds, and repeated sequences limit a simple independent-observation interpretation of the p-value. No result here estimates the effect of training with an alignment loss or establishes modality uniqueness; those questions require separate controlled comparisons.

### Representational similarity analysis

Return to [Representational similarity analysis](Stage1_stability.ipynb#Representational-similarity-analysis) in the notebook.

Representational similarity analysis (RSA) asks whether pairs of examples that are dissimilar within one representation also tend to be dissimilar within the other. For every pair of encoders, the call in [Run and interpret the alignment diagnostics](Stage1_stability.ipynb#Run-and-interpret-the-alignment-diagnostics) supplies the two original final-layer matrices from `EMBEDDINGS`, rather than the CCA projections. Corresponding rows must identify the same examples in the same order; the feature counts can differ. This diagnostic fits no predictor or projection.

**1. Center each example across its own features.**

For one embedding of the pair's first modality, first find its mean feature value:

$$
\bar x_i = \frac{1}{p}\sum_{k=1}^{p}x_{ik}.
$$

For this calculation:

- $X$ and $Y$ have shapes `(n, p)` and `(n, q)` for $n$ matched examples and their respective feature counts. For Nucleotide Transformer 500M and ESM-2 35M, these are `EMBEDDINGS["nt500m"]` and `EMBEDDINGS["esm2_35m"]`.
- $x_{ik}$ is feature $k$ of example row $i$ in $X$.
- $\bar x_i$ is the mean across that row's $p$ features, not a feature mean across examples.

Subtract that mean from every coordinate of the row:

$$
\widetilde{\mathbf x}_i
= \mathbf x_i - \bar x_i\mathbf 1_p.
$$

$\mathbf x_i$ is row $i$'s embedding vector, and $\mathbf 1_p$ is a vector of $p$ ones. Subtracting $\bar x_i\mathbf 1_p$ gives the centered vector $\widetilde{\mathbf x}_i$, still with $p$ coordinates. This centers each row across features, unlike CKA's feature centering across examples.

The same construction applies within $Y$, using each row's own $q$ coordinates and mean. SciPy performs this centering inside its correlation-distance calculation.

**2. Calculate the correlation distance between two rows of the same representation.**

$$
d_X(i,j)
= 1 - \frac{\widetilde{\mathbf x}_i^\top\widetilde{\mathbf x}_j}
{\|\widetilde{\mathbf x}_i\|_2\,\|\widetilde{\mathbf x}_j\|_2}.
$$

Here, $d_X(i,j)$ is the scalar distance between distinct examples $i$ and $j$, each with $p$ features. The ratio is their centered dot product divided by their Euclidean lengths: Pearson correlation of the original feature profiles. Subtracting it from 1 changes similarity into distance.

Correlation 1 gives distance 0, correlation 0 gives distance 1, and correlation $-1$ gives distance 2. A row whose features are all identical becomes a zero vector after centering, making this distance undefined. The helper adds no special handling for that case. See [SciPy's definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html). Applying the same calculation within $Y$ gives $d_Y(i,j)$, using its $q$ features; no individual feature of $X$ is matched to an individual feature of $Y$.

**3. Collect matching pairs and compare their distance rankings.**

Each representation contributes one distance for every unordered pair of distinct examples. The number of pairs is:

$$
M = \frac{n(n-1)}{2}.
$$

$M$ counts the distances in each comparison vector. The factor of two removes reversed pairs; keeping $i<j$ selects each pair once and excludes self-comparisons. For $n=3$, the order is $(1,2)$, $(1,3)$, $(2,3)$; Python labels those pairs `(0, 1)`, `(0, 2)`, `(1, 2)`.

SciPy's [`pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) returns these values as a one-dimensional condensed distance vector, rather than a square distance matrix. Because the input rows correspond, the two distance vectors use matching pair order. For a 981-row input, each vector contains 480,690 pair distances; those distances are not 480,690 independent examples.

The final statistic compares the ranks of the two distance vectors:

$$
\rho_{\mathrm{RSA}}
= \operatorname{corr}\!\left(
\operatorname{rank}(\mathbf d_X),
\operatorname{rank}(\mathbf d_Y)
\right).
$$

The comparison uses:

- $\mathbf d_X$ and $\mathbf d_Y$, each containing $M$ distances in matching pair order. `dist_a` and `dist_b` are illustrative table aliases; the helper passes both `pdist` results directly to `spearmanr`.
- $\operatorname{rank}$, which orders distances from smallest to largest, assigns ties their average rank, and retains pair order.
- $\rho_{\mathrm{RSA}}$, the scalar Spearman coefficient: Pearson correlation of those rank vectors.

If either distance vector is constant, the rank correlation is undefined and SciPy returns `NaN` with a constant-input warning. This is distinct from a single constant feature profile making its pair distances undefined. A positive coefficient indicates agreement in which pairs are more or less dissimilar. It describes geometry; it does not identify a particular shared subspace or biological mechanism.

**Connection to the code.** `analysis.compute_rsa` delegates the distance and rank calculations to SciPy:

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

## Attention and candidate motifs

Representation geometry compares examples; the following analysis returns to positions within each sequence. It develops the attention extraction, token-to-nucleotide mapping, candidate-pattern scan and comparisons used in the [stability notebook's Track C](Stage1_stability.ipynb#Track-C:-Attention-and-exploratory-diagnostics). Attention is a model behavior, and a sequence match is a candidate annotation: neither is a direct assay of regulatory function.

The attention here is **self-attention inside a frozen nucleotide encoder**, relating tokens within one input sequence. Stage 4 fusion cross-attention has a separate role: it combines information across RNA and protein representations. An attention-pattern result from this Stage 1 analysis therefore does not evaluate a Stage 4 fusion mechanism.

The maintained implementation is in [embeddings.py](../shared_code/embeddings.py), [analysis.py](../shared_code/analysis.py), [sequences.py](../shared_code/sequences.py), and the [encoder registry and loader](../shared_code/encoders.py). The [sequence preparation and embedding explanation](../../docs/methods/sequences-to-predictions.md) supplies the earlier biological-input and tokenization context.

| Step | Walkthrough |
| --- | --- |
| 1. Extract attention | [Extracting self-attention from the frozen nucleotide encoders](#extracting-self-attention-from-the-frozen-nucleotide-encoders) |
| 2. Summarize and map scores | [Summarizing attention and mapping tokens to nucleotides](#summarizing-attention-and-mapping-tokens-to-nucleotides) |
| 3. Scan candidate patterns | [Scanning candidate RNA stability-associated patterns](#scanning-candidate-rna-stability-associated-patterns) |
| 4. Compare matched and other positions | [Comparing attention at matched and other positions](#comparing-attention-at-matched-and-other-positions) |
| 5. Run and summarize sequences | [Running the attention-motif analysis](#running-the-attention-motif-analysis) |
| 6. Control position and codons | [Controlling the attention test for position and codons](#controlling-the-attention-test-for-position-and-codons) |

### Extracting self-attention from the frozen nucleotide encoders

Return to [Extracting self-attention from the frozen nucleotide encoders](Stage1_stability.ipynb#Extracting-self-attention-from-the-frozen-nucleotide-encoders) in the notebook.

`embeddings.attention_maps` requests `output_attentions=True` for one nucleotide sequence and encoder. Each query and key therefore belongs to that same input, rather than to separate modalities. This follows the [representation comparisons](Stage1_stability.ipynb#Track-B:-Cross-modal-geometry), which use extracted embeddings instead.

Each layer returns an array with shape `(1, H, T, T)`. The leading 1 is the one-sequence batch size. Removing that axis leaves `(H, T, T)`: heads, query positions and key positions.

The array axes are:

- $H$ attention heads, indexed by $h$; each head is a parallel attention computation.
- $T$ processed input tokens, including special tokens, with query index $q$ and key index $k$.
- $A^{(h)}_{qk}$, the scalar weight assigned by query $q$ to key $k$ in head $h$. Within a head's matrix, queries occupy rows and keys occupy columns.

**Connection to the code.** Inside `attention_maps`, `attentions` holds the per-layer arrays. Indexing `[0]` removes the batch axis, and `.float().cpu().numpy()` returns that layer's weights as a NumPy array. `layers` selects which arrays to convert; the current analysis retains only the final layer. The model call first requests attention outputs, so this selection does not establish that attention is computed or stored only for that layer. The returned list preserves the requested layer order, and its token strings come from the same call.

These weights contribute to a weighted combination of value vectors; they do not fully decompose the final representation or directly measure biological importance. See [Vaswani et al. (2017), Section 3.2](https://arxiv.org/html/1706.03762v7#S3.SS2) and the [ESM output definitions](https://huggingface.co/docs/transformers/en/model_doc/esm#transformers.EsmModel). RNA-FM, RiNALMo, mRNA-FM, and CaLM, loaded through `multimolecule`, return attention arrays with the same layout.

`encoders.load_encoder(encoder, device, eager_attention=True)` loads an encoder with `attn_implementation="eager"` to obtain the explicit attention weights used below; encoders built from their own model code use that code's attention implementation. This is a choice for attention inspection in this pipeline; support for returning weights depends on the model and attention backend. The [Transformers attention-backend documentation](https://huggingface.co/docs/transformers/en/attention_interface) explains how that argument selects the implementation. The embeddings computed earlier are unaffected.

Each loaded model is placed in evaluation mode, and inference runs inside `torch.no_grad()`. No parameter update is performed. Tokenization uses `truncation=True` with the same token limit as embedding extraction, the encoder's `max_len` in the registry. The returned token list describes this model call's processed input and may cover less than the full sequence; its coverage must be checked rather than assumed to match earlier extraction.


Loading an encoder with eager attention can print the same kind of weight report described in [Interpreting the loading messages](Stage1_stability.ipynb#Interpreting-the-loading-messages). Pooler parameters reported as newly initialized do not affect this analysis, which reads `outputs.attentions` and does not use a pooled output. The report should be interpreted in relation to the outputs consumed here, rather than treated as a general confirmation that every model component has pretrained weights.

### Summarizing attention and mapping tokens to nucleotides

Return to [Summarizing attention and mapping tokens to nucleotides](Stage1_stability.ipynb#Summarizing-attention-and-mapping-tokens-to-nucleotides) in the notebook.

`analysis.aggregate_attention` turns one layer's `(H, T, T)` array into one score per token. With `mode="incoming"`, the score summarizes how much attention a key position receives from all query positions after averaging heads. `embeddings.nucleotide_attention` requests the original model's final layer with `layers=[-1]`, then calls `aggregate_attention(..., layer=0)` on the returned one-element list. The index 0 therefore selects that already extracted final layer, not the model's first layer. This is the current analysis choice, not evidence that the final layer is the most informative one.

**1. Average the heads.**

$$
\bar A_{qk}=\frac{1}{H}\sum_{h=1}^{H}A^{(h)}_{qk}.
$$

$\bar A$ has shape `(T, T)`: each query/key entry is averaged over the $H$ heads. Query positions remain rows and key positions remain columns; this step does not average token positions.

**2. Sum incoming weight for each key token.**

$$
s_k=\sum_{q=1}^{T}\bar A_{qk}.
$$

$s_k$ adds one column of $\bar A$: all queries' weights for the fixed key $k$. The resulting $T$-entry vector summarizes incoming attention before rescaling; it is not a sequence embedding.

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

### Scanning candidate RNA stability-associated patterns

Return to [Scanning candidate RNA stability-associated patterns](Stage1_stability.ipynb#Scanning-candidate-RNA-stability-associated-patterns) in the notebook.

A sequence pattern can matter biologically when cellular molecules recognize it and change what happens to the RNA. In eukaryotic mRNAs, proteins bound to particular 3′-UTR sequences can influence poly-A-tail shortening, removal of the 5′ cap, or RNA cleavage and degradation. Recognition can involve the RNA’s folded shape as well as its letters; a bound protein can also prevent a degrading enzyme from accessing a cleavage site. The sequence region, RNA structure and molecular partners therefore help determine what a pattern can do. [Molecular Biology of the Cell, 7th ed., ch. 7, pp. 459–460](../../docs/sources.md#molecular-biology-of-the-cell-7th-edition)

These interactions occur in a cellular setting. The relevant proteins must be present and active, and regulatory responses can differ between cell types or conditions. Identifying the same letters in two sequences does not establish the same molecular interaction or outcome in both. A sequence-only input does not identify all of these cellular conditions when the dataset omits them. [Molecular Biology of the Cell, 7th ed., ch. 7, p. 401](../../docs/sources.md#molecular-biology-of-the-cell-7th-edition)

The scan creates **candidate sequence-pattern annotations** for the attention comparison. It searches the input sequence for three RNA-relevant patterns; it does not measure binding, modification, or a change in stability.

| Pattern | What the scan identifies | Interpretation limit |
|---|---|---|
| AUUUA | ARE-associated pentamer | A match alone does not establish an active AU-rich element or altered stability. |
| UUAUUUAUU | ARE-associated nonamer | Contains the pentamer; these are related patterns rather than independent mechanistic classes. |
| DRACH | Consensus associated with candidate m6A sites | D = A/G/U, R = A/G, H = A/C/U. The central A is a potential methylation site; a match does not establish methylation. |

The ARE patterns have experimental precedent in [Zubiaga et al. (1995)](https://pmc.ncbi.nlm.nih.gov/articles/PMC230450/). Single-nucleotide mapping shows that only a subset of DRACH candidates are methylated; see [Linder et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487409/). Motif occurrence, molecular modification or binding, and a functional effect on stability are distinct claims.

This biology connects the scan to a sequence of questions: are the pattern’s letters present, are they available for molecular recognition, does binding or modification occur, and does that interaction change RNA persistence? The present analysis observes the sequence match and compares model attention at those positions. It does not measure the intervening molecular events. The textbook examples explain why those distinctions matter; the studies cited above support the specific candidate patterns, and neither source establishes their activity in our retained coding-sequence inputs.

`sequences.iupac_to_regex` converts each IUPAC symbol to its permitted letters in the DNA alphabet. For example, DRACH becomes `[AGT][AG]AC[ACT]`, with U represented as T to match the converted sequence. `sequences.find_iupac_matches` uses a regex lookahead to include overlapping occurrences. It returns zero-based, end-exclusive intervals: `start` is covered and `end` is the first position after the match. `sequences.scan_sequence_for_motifs` adds the pattern name to each span. Subsequent masks cover the entire matched span, including all five DRACH positions, rather than only its central A.

These annotations depend only on sequence and require no external database call. The [CodonBERT source repository](https://github.com/Sanofi-Public/CodonBERT) links the stability task to [iCodon](https://www.nature.com/articles/s41598-022-15526-7), which studies coding-sequence codon composition. The [October 28 target trace](../../docs/run-records.md#2026-10-28-dataset-target-provenance) resolves the earlier uncertainty about the local label mapping: the released sequence/value observations match iCodon's within-profile standardized stability scores, as explained in the [target guide](../../docs/methods/data-and-targets.md#mrna-stability-standardized-decay-profiles). The released file omits the source profile identifiers, and this label verification does not establish UTR coverage. Presence or absence of these patterns therefore still needs interpretation in the context of the sequence region.

### Comparing attention at matched and other positions

Return to [Comparing attention at matched and other positions](Stage1_stability.ipynb#Comparing-attention-at-matched-and-other-positions) in the notebook.

`analysis.attention_motif_overlap` compares expanded attention scores at candidate-pattern positions with scores at the other covered positions in the same sequence. It forms the union of matched spans for the pooled comparison, counting each position once, then repeats the comparison separately for each pattern name. Match ends are clipped to the attention array's length. For a per-pattern comparison, the other-position group can include positions matching a different pattern.

**1. Average the scores at matched positions.**

$$
\bar\alpha_M=\frac{1}{n_M}\sum_{p\in M}\alpha_p.
$$

The matched-position mean uses:

- `attention_score`, an expanded array of length $L$ that may cover only part of the original sequence after truncation.
- $\alpha_p$, the score at nucleotide-array position $p$, not an example row or a p-value. This score is not a biological measurement or a fitted coefficient.
- $M$, the union of positions in the matched spans, with $n_M=|M|$ distinct positions; overlaps are counted once.
- $\bar\alpha_M$, the mean score across those positions.

**2. Average the scores at the other positions.**

$$
\bar\alpha_O=\frac{1}{n_O}\sum_{p\in O}\alpha_p.
$$

$O$ contains every covered position outside $M$, with $n_O=|O|$ and $n_M+n_O=L$. Its mean score is $\bar\alpha_O$.

Both means require a nonempty group. The nested `_test` helper returns `None` if either group is empty. That marks an unavailable comparison, not a zero effect; an unavailable pooled comparison makes the outer function return `None` too.

**3. Distinguish the mean gap from the rank test.**

The two returned means also define a descriptive difference:

$$
\Delta=\bar\alpha_M-\bar\alpha_O.
$$

$\Delta$ is the matched-minus-other mean gap: positive when matched positions have the larger average score, negative when they have the smaller one. Its units are the rescaled attention-score units, not probability.

For example, matched scores `(0.2, 0.4)` have mean 0.3, while other scores `(0.1, 0.1)` have mean 0.1. Their gap is 0.2. This illustrative calculation does not determine the rank test's p-value.

The one-sided Mann-Whitney U call uses `alternative="greater"` to examine a tendency toward larger scores in the matched group. It compares ranks/distributions, rather than directly testing whether $\Delta$ is positive. The standard null is equality of the two underlying distributions, with an alternative of stochastically greater matched scores; see [SciPy's Mann-Whitney U documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).

**Connection to the code.** The means are calculated now; the subtraction is performed later in the multi-sequence summary.

| Code operation | Result | Connection to the calculation |
|---|---|---|
| `_test`: select `at` and `off` from `attention_score` | Arrays with shapes `(n_M,)` and `(n_O,)` | Gather the scores in $M$ and its complement $O$. |
| `at.mean()` and `off.mean()` | Two scalars stored in the returned dictionary | Compute $\bar\alpha_M$ and $\bar\alpha_O$. |
| `mannwhitneyu(at, off, alternative="greater")` | A U statistic and a nominal p-value | SciPy performs the rank test; the dictionary retains its p-value. |
| Later, in `analysis.attention_summary` | One difference per usable sequence, followed by an average | Subtract the dictionary's two mean fields to compute each $\Delta$. |

`attention_motif_overlap` returns the means, not a `Delta` field. [The multi-sequence summary](#running-the-attention-motif-analysis) forms the gap from those outputs.

Treat the p-values as **nominal**: they are the test's reported values without calibration for this positional setting. Expanding token scores repeats values; neighboring positions are dependent; patterns can overlap; and many sequence/pattern comparisons are performed without multiplicity correction. Sequence composition and tokenization can also affect both masks and scores. A small p-value or positive mean gap identifies a candidate association for follow-up, not evidence that the encoder learned functional RNA regulation. Calibrated null models and experimentally supported annotations require a separate methodological decision.

### Running the attention-motif analysis

Return to [Running the attention-motif analysis](Stage1_stability.ipynb#Running-the-attention-motif-analysis) in the notebook.

The first execution follows one retained sequence through the complete pipeline for each nucleotide encoder in `ATTENTION_ENCODERS`: convert its alphabet for that encoder, extract attention, summarize the final layer, expand token scores, scan candidate patterns, compare groups, and plot the result. `embeddings.nucleotide_attention` performs the first four steps. The printed sequence and attention-array lengths help inspect coverage, but they are not an assertion that the coordinates agree for every input.

The second execution uses the **first `ATTENTION_N_ROWS` retained sequences**, currently 100, separately for each encoder. `embeddings.attention_profiles` returns their expanded arrays in `attention_profiles` for the later permutation tests. `analysis.attention_motif_tests` compares each array and keeps rows with an available pooled comparison in `attention_results`.

`analysis.attention_summary` calculates two summaries per encoder and pattern for the notebook's displayed DataFrame. The following equations describe that summary, after the single-example walkthrough.

**1. Count nominal threshold crossings.**

$$
f_{0.05}=\frac{1}{m}\sum_{i=1}^{m}\mathbf{1}\{p_i<0.05\}.
$$

The threshold summary uses:

- $m>0$ usable sequences, indexed by $i$ rather than nucleotide position.
- $p_i$, sequence $i$'s nominal Mann-Whitney p-value.
- $\mathbf{1}\{p_i<0.05\}$, equal to 1 below the threshold and 0 otherwise. Averaging these indicators gives the fraction $f_{0.05}$.

**2. Average the per-sequence mean gaps.**

$$
\overline{\Delta}=\frac{1}{m}\sum_{i=1}^{m}\Delta_i.
$$

$\Delta_i$ is sequence $i$'s gap calculated from its two returned means. $\overline{\Delta}$ averages these gaps over the same $m$ usable sequences as the threshold summary. Each sequence has equal weight, irrespective of its number of nucleotide positions or matches.

For example, if three usable sequences have p-values `(0.01, 0.20, 0.04)` and gaps `(0.02, -0.01, 0.05)`, the threshold fraction is $2/3$ and the mean gap is $0.06/3=0.02$. These are illustrative values. The fraction is not itself a p-value, and the average gap is not a test pooling all nucleotide positions across sequences.

**Connection to the code.** These are direct summaries of the dictionaries collected in `attention_results`.

| Code operation | Result | Connection to the mathematics |
|---|---|---|
| `attention_motif_tests` appends a result when it is not `None` | A list of usable comparison dictionaries per encoder | Determines the included sequence set and $m$ for that encoder. |
| `np.mean([g["p_value"] < 0.05 for g in group])` | The `fraction with nominal p < 0.05` column | NumPy counts `True` as 1 and `False` as 0, implementing $f_{0.05}$. |
| Subtract the two mean fields inside the `mean gap` list comprehension | One gap per dictionary | Computes each $\Delta_i$. |
| `np.mean(...)` around that list | The `mean gap` column | Computes $\overline{\Delta}$. |

Each pattern row repeats these operations using only dictionaries with an available comparison for that pattern. Its usable count can differ from the pooled count or another pattern's count. A pattern with no usable comparison appears with zero sequences and empty summary columns.

The “fraction with nominal p < 0.05” counts unadjusted threshold crossings, subject to the [dependence and multiple-comparison limits](#comparing-attention-at-matched-and-other-positions). It is not calibrated significance. The single example illustrates the calculation; the multi-sequence summary describes the selected sample under the same procedure.

### Controlling the attention test for position and codons

Return to [Controlling the attention test for position and codons](Stage1_stability.ipynb#Controlling-the-attention-test-for-position-and-codons) in the notebook.

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

The table reports $G$, the mean shuffled statistic, their difference (the excess gap), and $p$ for each encoder, pattern and null. A small p-value identifies a departure from the implemented null, not a regulatory mechanism. Each null preserves only the counts listed above; shuffling assumes matched labels are exchangeable within those strata, and shuffled masks need not form intact motif occurrences. Patterns with no usable sequence, such as a pattern that never occurs, are omitted. The tests reuse the same selected sequences and are not corrected for the number of encoders, patterns and nulls.

### Reusing the method on untranslated regions

The notebook's [GTEx attention analysis](Stage1_gtex.ipynb#Attention-at-candidate-patterns-in-3'-untranslated-regions) applies the same summary, pattern scan, comparison, and permutation functions to selected 3' untranslated regions using the single-nucleotide RNA encoders. The notebook owns its selection settings and saved results.

The `strata="codon"` option still groups positions by the same nonoverlapping three-nucleotide block and offset within that block, beginning at the region's first nucleotide. Because these are untranslated regions, those blocks are sequence-context groups rather than biological codons or reading frames. The position null still uses 100-nucleotide bins; all coverage and permutation assumptions above continue to apply.
