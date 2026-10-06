# Linear probing

This guide explains the standardization, ridge regression, and evaluation used in the [stability notebook](../../Code/Stage1_stability.ipynb#Linear-probing) and the [GTEx expression probes](../../Code/Stage1_gtex.ipynb#Expression-prediction-with-linear-probes). Start with the respective notebook for its biological target, experiment settings, and saved results; use this page for the derivations and worked example. The maintained implementation is in [analysis.py](../../Code/mbf/analysis.py) and the group assignment is in [splits.py](../../Code/mbf/splits.py).

A linear probe tests whether a target can be predicted from an embedding with a simple fitted model. The main inputs are embeddings produced by the frozen encoders; the probe functions also accept simpler feature matrices, such as sequence length. Fitting a probe does not update the encoders.

For the scalar-target stability analyses described here, `analysis.probe_scores(X, y, groups=None, continuous=True, seed=42)` receives an embedding matrix `X` with shape `(n, p)` and a target array `y` with shape `(n,)`. There are `n` example rows and `p` features per row, and the targets must follow the same row order. `continuous` selects regression or classification, `groups`, when supplied, gives one group identifier per row, and `seed` selects the fold assignment; the default is the pinned assignment. It returns the five held-out fold scores and does not return a fitted model. `analysis.probe_assignment_means` repeats the evaluation over further fold assignments, and `analysis.probe_table` reports both for several embeddings and targets.

## Standardization and model choice

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

## Evaluation and interpretation

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

**Interpretation.** Identical retained sequence strings receive the same fold, so they stay together across the outer train/test boundary. With the same normalized group strings, seed, fold count, and hash procedure, the fold assignment is independent of the machine and library version; a sequence keeps its fold in any sample that contains it. This fixes partition membership, not numerical equality of embeddings or scores across environments. Fold sizes vary slightly around one fifth of the groups, and a fold with no rows raises an error. Both notebooks explicitly pass `PROBE_SEED = 42` to single-encoder tables, concatenated-pair comparisons, and baseline probes. The public helpers retain `seed=42` as their default. Sampling, CCA, and PyTorch initialization use `SEED` separately; further probe assignments retain seeds 0 through `N_FOLD_ASSIGNMENTS - 1`. Changing the pinned seed requires corresponding single and pair results from that same partition; historical saved values were not rerun by this refactor. scikit-learn's [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) is not used because it orders groups of equal size with NumPy's default sort, which is not stable, so its folds can differ between library versions and machines; see [NumPy's sort documentation](https://numpy.org/doc/stable/reference/generated/numpy.sort.html).

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

## Current evaluation limitation

The current pipeline prevents outer held-out rows from setting the scaler statistics, and grouping prevents identical sequence strings from crossing the outer train/test boundary. See scikit-learn's [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

The internal alpha-selection procedure has a narrower boundary: the scaler is fit once on the whole outer training subset before `RidgeCV` performs its internal leave-one-out calculation, and this code does not supply sequence groups to that internal calculation. This follows from the helper's ordering and the documented [Pipeline fitting sequence](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit). Thus preprocessing is not refit for each internal held-out row, and internal model selection is not grouped by sequence. Outer held-out rows remain excluded from fitting; the pipeline should not be described as refitting preprocessing and enforcing groups at every level of cross-validation. The numerical effect of changing the internal procedure has not been measured here.

Grouping addresses exact duplicate strings, not independence among related sequences. The probes do not deduplicate rows. The [notebook's evaluation limitation](../../Code/Stage1_stability.ipynb#Current-evaluation-limitation) retains the dataset counts and unresolved provenance questions associated with its saved analysis.

## Statistical and course references

- James, Witten, Hastie, and Tibshirani, *An Introduction to Statistical Learning with Applications in R*, second edition: Section 3.1.3, p. 70 ($R^2$); Sections 5.1.2-5.1.3, pp. 200 and 203-204 (leave-one-out and k-fold cross-validation); Section 6.2.1, pp. 237-239 (ridge regression and scaling); Section 6.2.3, p. 250 (tuning). The original explanation records consultation of the local June 2023 corrected PDF. The [authors' book site](https://www.statlearning.com/) provides the book.
- UCF Machine Learning, *Week 04_Validation*, slides 5, 7, and 9: separating fitting, model selection, and evaluation, and rotating the held-out fold.
- UCF Machine Learning, *Data Preprocessing for Machine Learning*, slide 5: the role of feature scaling.
- UCF Capstone general lecture, *Overfitting and Regularization* (Yousefi, 2020), slides 19-20: the ridge penalty and selecting its strength by cross-validation.

The course decks provide explanatory context; linked library documentation explains the API behavior, and the [notebook](../../Code/Stage1_stability.ipynb#Linear-probing) and [probe implementation](../../Code/mbf/analysis.py) specify the arguments used. Library documentation alone does not identify the environment or producing run of a saved result.