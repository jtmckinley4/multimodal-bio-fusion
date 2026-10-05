"""Representation analyses shared by the notebooks.

Linear probes, representation similarity (CKA, mutual k-NN, CCA retrieval, RSA), the
composition control, principal-component bands, the attention-motif test and its
permutation nulls, and summary tables and figures.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sequences import scan_sequence_for_motifs
from .splits import grouped_folds


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_scores(X, y, groups=None, continuous=True, seed=42):
    """Return the five held-out fold scores of a standardized linear probe.

    Continuous targets use RidgeCV over 20 log-spaced penalties and R^2. A target with
    several columns is fit with one penalty for all columns and scored by R^2 averaged
    over the columns. With `groups` (the retained sequence strings, or another group
    label per row), grouped_folds with this seed keeps each group in one fold; the
    default seed is the pinned assignment.
    """
    if continuous:
        model, scoring = RidgeCV(alphas=np.logspace(-3, 5, 20)), "r2"
    else:
        model, scoring = LogisticRegression(max_iter=2000), "accuracy"
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    if groups is not None:
        return cross_val_score(pipeline, X, y, cv=grouped_folds(groups, seed=seed), scoring=scoring)
    return cross_val_score(pipeline, X, y, cv=5, scoring=scoring)


def probe_assignment_means(X, y, groups, n_assignments=10, continuous=True):
    """Return the five-fold mean probe score for each of n_assignments fold assignments.

    The assignments use seeds 0 through n_assignments - 1. They reuse the same rows, so
    the spread of these means shows how much a result depends on the partition; it is
    not a standard error.
    """
    return np.array([
        probe_scores(X, y, groups=groups, continuous=continuous, seed=s).mean()
        for s in range(n_assignments)
    ])


def probe_table(embeddings, targets, groups, n_assignments=10):
    """Return probe R^2 for every (embedding, target) combination.

    embeddings: {name: matrix}; targets: {target name: vector, or matrix with one
    column per output}. For each target, the table gives the mean and fold SD on the
    pinned folds, then the mean and SD of the five-fold means over n_assignments
    further fold assignments (omitted when 0).
    """
    rows = []
    for name, X in embeddings.items():
        row = {"encoder": name}
        for target, y in targets.items():
            scores = probe_scores(X, y, groups=groups)
            row[f"{target} R2"] = scores.mean()
            row[f"{target} SD"] = scores.std()
            if n_assignments:
                means = probe_assignment_means(X, y, groups, n_assignments)
                row[f"{target} R2, {n_assignments} assignments"] = means.mean()
                row[f"{target} SD, {n_assignments} assignments"] = means.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("encoder")


def _embedding_matrices(embeddings, keys):
    """Validate the selected feature matrices without changing their row order."""
    keys = list(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select at least one encoder, with no duplicate keys.")
    matrices = [np.asarray(embeddings[key]) for key in keys]
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("Each embedding matrix must be two-dimensional.")
    if len({matrix.shape[0] for matrix in matrices}) != 1:
        raise ValueError("Embedding matrices must have the same number of rows.")
    return matrices


def concatenate_embeddings(embeddings, keys):
    """Join the selected encoders' feature columns in the given key order.

    embeddings maps encoder keys to two-dimensional arrays. Select at least one
    unique key; unequal feature widths are allowed. Missing keys raise KeyError;
    duplicate/empty selections or mismatched dimensions/row counts raise ValueError.
    The caller must ensure that rows represent the same examples in the same order:
    matching shapes cannot establish this biological correspondence.
    """
    return np.hstack(_embedding_matrices(embeddings, keys))


def pairwise_concatenation_scores(embeddings, keys, y, single_r2, groups=None):
    """Return (pair R^2, gain over the better single encoder) on pinned folds.

    keys selects at least two unique encoders and orders both DataFrame axes.
    Matrices are symmetric with NaN diagonals. y is a vector or a multioutput
    matrix; groups, when provided, has one label per row. All selected matrices,
    targets and groups must contain the same examples in the same row order.

    single_r2 maps each selected key to its already-computed, finite scalar mean
    R^2 on the same targets, rows, groups and pinned folds. This correspondence is
    the caller's responsibility; scores are not recomputed here. A gain is the
    pair's five-fold mean minus max(single_r2[a], single_r2[b]).

    probe_scores owns scaling, ridge fitting, multioutput scoring and the pinned
    default seed. No notebook seed is substituted and no rows are filtered here.
    Missing embedding/baseline keys raise KeyError; invalid shapes, selections or
    baselines raise ValueError before any probes run.
    """
    keys = list(keys)
    matrices = _embedding_matrices(embeddings, keys)
    if len(keys) < 2:
        raise ValueError("Pairwise concatenation requires at least two encoders.")
    n_rows = matrices[0].shape[0]
    if np.ndim(y) not in (1, 2) or np.shape(y)[0] != n_rows:
        raise ValueError("Targets must be a vector or matrix with one row per example.")
    if groups is not None and (np.ndim(groups) != 1 or len(groups) != n_rows):
        raise ValueError("Groups must have one label per example.")
    baselines = [single_r2[key] for key in keys]
    try:
        valid_baselines = (all(np.ndim(score) == 0 for score in baselines)
                           and np.isrealobj(baselines) and np.isfinite(baselines).all())
    except (TypeError, ValueError):
        valid_baselines = False
    if not valid_baselines:
        raise ValueError("Single-encoder baselines must be finite scalar mean R^2 scores.")

    scores = pairwise_matrix(
        keys,
        lambda a, b: probe_scores(
            concatenate_embeddings(embeddings, [a, b]), y, groups=groups
        ).mean(),
    )
    gains = pairwise_matrix(
        keys, lambda a, b: scores.loc[a, b] - max(single_r2[a], single_r2[b])
    )
    return scores, gains


def pairwise_assignment_gains(embeddings, keys, y, groups, n_assignments=10):
    """Return each encoder pair's gain array over further grouped fold assignments.

    keys selects at least two distinct encoders. Each (a, b) key follows their
    combinations order; its array has one gain per assignment, using seeds 0
    through n_assignments - 1 via probe_assignment_means. For each assignment,
    subtract the better single encoder's five-fold mean from the concatenation's
    five-fold mean. The better single encoder can differ between assignments.
    Each selected single encoder and each pair is evaluated once per assignment.

    Embeddings, y (a vector or multioutput matrix), and groups must have matching
    rows. Callers must ensure the same examples and row order, which shapes alone
    cannot establish. Groups are required: ungrouped probe_scores does not change
    its folds with the seed. n_assignments must be a positive integer, not bool.
    Missing keys raise KeyError; invalid selections, shapes, groups or assignment
    counts raise ValueError before probing. Inputs are neither filtered nor mutated.

    Notebook callers own encoder labels, modality descriptions, pinned gains and
    table summaries. Assignment gains reuse the same examples; their spread is
    sensitivity to partitioning, not a standard error.
    """
    keys = list(keys)
    matrices = _embedding_matrices(embeddings, keys)
    if len(keys) < 2:
        raise ValueError("Pairwise gains require at least two encoders.")
    n_rows = matrices[0].shape[0]
    if np.ndim(y) not in (1, 2) or np.shape(y)[0] != n_rows:
        raise ValueError("Targets must be a vector or matrix with one row per example.")
    if groups is None or np.ndim(groups) != 1 or len(groups) != n_rows:
        raise ValueError("Grouped fold assignments require one group label per example.")
    if (isinstance(n_assignments, (bool, np.bool_))
            or not isinstance(n_assignments, (int, np.integer)) or n_assignments < 1):
        raise ValueError("n_assignments must be a positive integer.")

    single_means = {
        key: probe_assignment_means(embeddings[key], y, groups, n_assignments)
        for key in keys
    }
    gains = {}
    for a, b in itertools.combinations(keys, 2):
        combined = probe_assignment_means(
            concatenate_embeddings(embeddings, [a, b]), y, groups, n_assignments
        )
        gains[a, b] = combined - np.maximum(single_means[a], single_means[b])
    return gains


def fit_predict(X_train, y_train, X_test):
    """Fit the standardized ridge probe on training rows and return its test predictions.

    This is the probe of probe_scores, fit once on a given training set, for evaluations
    on a fixed published split.
    """
    pipeline = Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 5, 20)))])
    return pipeline.fit(X_train, y_train).predict(X_test)


# ---------------------------------------------------------------------------
# Representation similarity
# ---------------------------------------------------------------------------

def linear_cka(X, Y):
    """Return linear CKA for representations with matching example rows."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(Y_centered.T @ X_centered, ord="fro") ** 2
    norm_x = np.linalg.norm(X_centered.T @ X_centered, ord="fro")
    norm_y = np.linalg.norm(Y_centered.T @ Y_centered, ord="fro")
    return numerator / (norm_x * norm_y)


def mutual_knn_alignment(X, Y, k=10):
    """Average neighbor-set overlap after discarding each first returned index."""
    n = X.shape[0]
    neighbors_x = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    neighbors_y = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)
    overlaps = [len(set(neighbors_x[i, 1:]) & set(neighbors_y[i, 1:])) / k for i in range(n)]
    return np.mean(overlaps)


def cca_retrieval(X_a, X_b, train_idx, test_idx, seed, n_pca=50, n_cca=10, ks=(1, 5, 10)):
    """Fit PCA then CCA on training rows; return held-out correlations and retrieval.

    The procedure uses PCA with 50 components per space, CCA with 10 components,
    correlations of each component pair on the test rows, and Recall@k for querying the
    second space with the first in the CCA space.
    """
    n_pca = min(n_pca, len(train_idx) - 1, X_a.shape[1], X_b.shape[1])
    pca_a = PCA(n_components=n_pca, random_state=seed).fit(X_a[train_idx])
    pca_b = PCA(n_components=n_pca, random_state=seed).fit(X_b[train_idx])
    cca = CCA(n_components=n_cca).fit(pca_a.transform(X_a[train_idx]), pca_b.transform(X_b[train_idx]))
    a_c, b_c = cca.transform(pca_a.transform(X_a[test_idx]), pca_b.transform(X_b[test_idx]))
    corrs = np.array([np.corrcoef(a_c[:, i], b_c[:, i])[0, 1] for i in range(n_cca)])
    n_test = len(test_idx)
    neighbors = NearestNeighbors(n_neighbors=min(max(ks), n_test)).fit(b_c).kneighbors(
        a_c, return_distance=False
    )
    recall = {k: np.mean([i in neighbors[i, :k] for i in range(n_test)]) for k in ks}
    return {"correlations": corrs, "recall": recall, "test_coords": (a_c, b_c)}


def alignment_error_association(cca_a, cca_b, labels, n_folds=5, seed=42):
    """Correlate sample-level CCA cosine similarity with out-of-fold error.

    Inputs are matched CCA-projected rows of two modalities and one label per row.
    This exploratory rank association adapts the paper's motivation; it is
    not its across-model alignment-performance slope or an alignment-loss test.
    """
    # Compare paired rows in the existing CCA coordinate system.
    local_alignment = np.array([
        cosine_similarity(cca_a[i:i+1], cca_b[i:i+1])[0, 0]
        for i in range(len(labels))
    ])

    # Fit ordinary linear regression and collect held-out absolute errors.
    fused = np.concatenate([cca_a, cca_b], axis=1)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    errors = np.zeros(len(labels))
    for train_idx, test_idx in kf.split(fused):
        reg = LinearRegression().fit(fused[train_idx], labels[train_idx])
        errors[test_idx] = np.abs(reg.predict(fused[test_idx]) - labels[test_idx])

    rho, p_value = spearmanr(local_alignment, errors)
    return {"spearman": rho, "p_value": p_value}


def layerwise_cka(layers_a, layers_b):
    """Return the matrix of linear CKA between every hidden state of two encoders.

    Row i is state i of the first encoder and column j state j of the second; state 0
    is the embedding output before the first transformer block.
    """
    return np.array([[linear_cka(A, B) for B in layers_b] for A in layers_a])


def layerwise_summary(matrix):
    """Return the maximum CKA and its states, the maximum without the second encoder's
    state 0, and the final-state CKA for one layer-wise matrix."""
    i, j = np.unravel_index(matrix.argmax(), matrix.shape)
    i1, j1 = np.unravel_index(matrix[:, 1:].argmax(), matrix[:, 1:].shape)
    return {
        "maximum": matrix.max(), "at states": f"{i}, {j}",
        "maximum without second state 0": matrix[:, 1:].max(), "at states (without)": f"{i1}, {j1 + 1}",
        "final states": matrix[-1, -1],
    }


def compute_rsa(embeddings_a, embeddings_b):
    """Compare matched pairwise correlation distances using Spearman correlation."""
    return spearmanr(pdist(embeddings_a, metric="correlation"), pdist(embeddings_b, metric="correlation"))


def mantel_p_value(embeddings_a, embeddings_b, n_permutations=499, seed=42):
    """Permutation p-value for the RSA correlation, reordering the sequences of embeddings_b."""
    rank_a = rankdata(pdist(embeddings_a, metric="correlation"))
    rank_b = squareform(rankdata(pdist(embeddings_b, metric="correlation")))
    z_a = (rank_a - rank_a.mean()) / rank_a.std()

    def correlation_with_a(rank_matrix):
        r = squareform(rank_matrix, checks=False)
        return np.mean(z_a * (r - r.mean()) / r.std())

    observed = correlation_with_a(rank_b)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        order = rng.permutation(len(rank_b))
        if correlation_with_a(rank_b[np.ix_(order, order)]) >= observed:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)


def pairwise_matrix(names, function):
    """Return a symmetric DataFrame of function(a, b) over every pair of names.

    The diagonal is left empty because each metric compares two different encoders.
    """
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        value = function(a, b)
        matrix.loc[a, b] = matrix.loc[b, a] = value
    return matrix


# ---------------------------------------------------------------------------
# Composition control
# ---------------------------------------------------------------------------

def residualize(X, F):
    """Remove the least-squares fit on standardized F, with an intercept, from every column of X."""
    F_s = StandardScaler().fit_transform(F)
    design = np.column_stack([np.ones(len(F_s)), F_s])
    coef, *_ = np.linalg.lstsq(design, X, rcond=None)
    return X - design @ coef


def centroid_spread(Z):
    """Mean Euclidean distance of the rows of Z from their centroid."""
    return np.linalg.norm(Z - Z.mean(axis=0), axis=1).mean()


def composition_share(X, X_residual):
    """Share of an embedding's total variance explained by the composition fit (in sample)."""
    total = ((X - X.mean(axis=0)) ** 2).sum()
    return 1 - (X_residual ** 2).sum() / total


# ---------------------------------------------------------------------------
# High- and low-variance directions
# ---------------------------------------------------------------------------

def pca_bands(X, k=10):
    """Split an embedding into its top-k principal components and all remaining ones.

    Returns (top scores, remaining scores, share of total variance in the top k). PCA is
    fit on every row without labels. The two blocks are orthogonal and together keep all
    of the embedding's variance, so a comparison computed on each block shows whether a
    property lives in the few dominant directions or in the many low-variance ones.
    """
    n_components = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full").fit(X)
    scores = pca.transform(X)
    return scores[:, :k], scores[:, k:], pca.explained_variance_ratio_[:k].sum()


# ---------------------------------------------------------------------------
# Attention at candidate motifs
# ---------------------------------------------------------------------------

def aggregate_attention(attentions, layer=-1, mode="incoming"):
    """Summarize one layer and min-max rescale its token-position scores.

    Incoming scores sum over queries after averaging heads. Other mode values
    use row sums. Special-token positions remain included at this step.
    """
    layer_att = attentions[layer].mean(axis=0)  # (query_tokens, key_tokens)
    score = layer_att.sum(axis=0) if mode == "incoming" else layer_att.sum(axis=1)
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score


def expand_token_attention_to_nucleotides(attn_score, tokens):
    """Repeat scores by token-string length after excluding special tokens.

    Assumes retained token strings spell the covered nucleotides in order.
    The expansion cannot restore sequence positions removed by truncation.
    """
    expanded = []
    for score, tok in zip(attn_score, tokens):
        if tok.startswith("<") or tok.startswith("["):
            continue
        expanded.extend([score] * len(tok))
    return np.array(expanded)


def attention_motif_overlap(attention_score, motif_hits):
    """Return descriptive means and nominal one-sided rank-test p-values.

    Compare the union of covered positions with its complement, then repeat
    by pattern. Return None when the pooled comparison has an empty group.
    """
    def _test(positions):
        """Compare one position set with all other positions in the array."""
        non_positions = set(range(len(attention_score))) - positions
        if not positions or not non_positions:
            return None
        at = attention_score[list(positions)]
        off = attention_score[list(non_positions)]
        stat, pval = mannwhitneyu(at, off, alternative="greater")
        return {
            "mean_attention_at_motifs": at.mean(),
            "mean_attention_elsewhere": off.mean(),
            "n_motif_positions": len(positions),
            "p_value": pval,
        }

    all_positions = set()
    by_name = {}
    for start, end, name in motif_hits:
        positions = set(range(start, min(end, len(attention_score))))
        all_positions.update(positions)
        by_name.setdefault(name, set()).update(positions)

    result = _test(all_positions)
    if result is None:
        return None
    result["by_motif"] = {
        name: _test(positions) for name, positions in by_name.items()
    }
    return result


def attention_motif_tests(profiles, seqs):
    """Run attention_motif_overlap on each sequence's attention profile.

    profiles holds one nucleotide-level attention array per sequence, in the order of
    seqs. Returns the usable results (sequences with an available pooled comparison).
    """
    results = [attention_motif_overlap(a, scan_sequence_for_motifs(s)) for a, s in zip(profiles, seqs)]
    return [r for r in results if r is not None]


def _strata(seq, length, kind, bin_size):
    """Label each covered position by its position bin or by its codon and reading frame."""
    positions = np.arange(length)
    if kind == "position":
        return positions // bin_size
    keys = [seq[3 * (p // 3): 3 * (p // 3) + 3] + str(p % 3) for p in positions]
    return np.unique(keys, return_inverse=True)[1]


def motif_permutation_test(profiles, seqs, motif=None, strata="position", n_permutations=500,
                           bin_size=100, seed=42):
    """Test whether attention at candidate matches exceeds a null that keeps position or codons.

    For each sequence, the gap is the mean attention at positions covered by a match
    (one pattern, or every pattern when motif is None) minus the mean elsewhere. The null
    shuffles which positions count as matched, but only within strata: 100-nucleotide
    position bins (strata="position"), or positions sharing a codon and reading frame
    (strata="codon"). A position effect or a codon-composition effect therefore appears in
    the null as well as in the observed gap. The test statistic is the gap averaged over
    sequences, and p = (1 + null averages at least as large) / (n_permutations + 1).
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for attention, seq in zip(profiles, seqs):
        length = len(attention)
        mask = np.zeros(length, dtype=bool)
        for start, end, name in scan_sequence_for_motifs(seq):
            if motif is None or name == motif:
                mask[start:min(end, length)] = True
        n_at = int(mask.sum())
        if n_at == 0 or n_at == length:
            continue
        labels = _strata(seq, length, strata, bin_size)
        order = np.argsort(labels, kind="stable")
        prepared.append((np.asarray(attention, dtype=float), labels, mask[order], n_at, mask))
    if not prepared:
        return None

    def mean_gap(masks):
        gaps = []
        for (attention, _, _, n_at, _), m in zip(prepared, masks):
            at_sum = attention[m].sum()
            gaps.append(at_sum / n_at - (attention.sum() - at_sum) / (len(attention) - n_at))
        return np.mean(gaps)

    observed = mean_gap([p[4] for p in prepared])
    null = np.empty(n_permutations)
    for r in range(n_permutations):
        shuffled = []
        for attention, labels, sorted_mask, _, _ in prepared:
            order = np.lexsort((rng.random(len(attention)), labels))
            m = np.empty(len(attention), dtype=bool)
            m[order] = sorted_mask
            shuffled.append(m)
        null[r] = mean_gap(shuffled)
    return {
        "sequences": len(prepared),
        "observed gap": observed,
        "null gap": null.mean(),
        "excess gap": observed - null.mean(),
        "permutation p": (1 + (null >= observed).sum()) / (n_permutations + 1),
    }


def attention_summary(results_by_encoder, labels, motifs):
    """Return one row per encoder and pattern: usable sequences, nominal p < 0.05 share, mean gap.

    results_by_encoder maps an encoder key to its list of attention_motif_overlap results.
    The pooled row uses every result; each pattern row uses only the results with an
    available comparison for that pattern.
    """
    rows = []
    for key, results in results_by_encoder.items():
        groups = {"All patterns pooled": results}
        for motif in motifs:
            groups[motif] = [r["by_motif"][motif] for r in results if r["by_motif"].get(motif)]
        for motif, group in groups.items():
            row = {"encoder": labels[key], "pattern": motif, "sequences with hits": len(group)}
            if group:
                row["fraction with nominal p < 0.05"] = np.mean([g["p_value"] < 0.05 for g in group])
                row["mean gap"] = np.mean(
                    [g["mean_attention_at_motifs"] - g["mean_attention_elsewhere"] for g in group]
                )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["encoder", "pattern"])


MOTIF_COLORS = {
    "ARE_pentamer": "#97BC62",
    "ARE_nonamer": "#2C5F2D",
    "m6A_DRACH": "#6B6B63",
}


def plot_attention_with_motifs(attention_score, motif_hits, title=""):
    """Plot expanded attention scores and candidate sequence-pattern spans."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(attention_score, color="black", linewidth=1)
    ax.fill_between(
        range(len(attention_score)), attention_score, color="gray", alpha=0.15
    )
    seen = set()
    for start, end, name in motif_hits:
        color = MOTIF_COLORS.get(name, "gray")
        ax.axvspan(
            start, end, color=color, alpha=0.35,
            label=name if name not in seen else None,
        )
        seen.add(name)
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Attention score (normalized)")
    ax.set_title(title)
    if seen:
        ax.legend(loc="upper right", fontsize=8, ncol=len(seen))
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

def pair_table(matrices, pairs, labels, modality):
    """Return one row per encoder pair with each metric from a dict of pairwise matrices."""
    rows = []
    for a, b in pairs:
        same = modality[a] == modality[b]
        row = {
            "pair": f"{labels[a]} / {labels[b]}",
            "comparison": f"within {modality[a]}" if same else "across modalities",
        }
        row.update({name: m.loc[a, b] for name, m in matrices.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def plot_matrices(matrices, labels, diverging=(), fmt="{:.2f}"):
    """Draw encoder-by-encoder heatmaps side by side, annotated with their values.

    Titles listed in `diverging` use a symmetric red-blue scale centered on zero.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(matrices), figsize=(6.2 * len(matrices), 5.6), squeeze=False)
    for ax, (title, m) in zip(axes[0], matrices.items()):
        values = m.values.astype(float)
        if title in diverging:
            bound = np.nanmax(np.abs(values)) or 1.0
            im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
            dark = lambda v: abs(v) > 0.6 * bound
        else:
            top = np.nanmax(values) or 1.0
            im = ax.imshow(values, cmap="Greens", vmin=0, vmax=top)
            dark = lambda v: v > 0.6 * top
        names = [labels.get(k, k) for k in m.index]
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isnan(values[i, j]):
                    ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if dark(values[i, j]) else "black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


MODALITY_COLORS = {"DNA": "#2C5F2D", "RNA": "#6B6B63", "protein": "#97BC62"}


def plot_layerwise(matrices, labels):
    """Draw layer-wise CKA heatmaps for several encoder pairs on one shared color scale."""
    import matplotlib.pyplot as plt

    vmax = max(m.max() for m in matrices.values())
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5.5), squeeze=False)
    for ax, ((a, b), m) in zip(axes[0], matrices.items()):
        im = ax.imshow(m, cmap="Greens", aspect="auto", origin="lower", vmin=0, vmax=vmax)
        ax.set_xlabel(f"{labels[b]} state (0 = embedding output)")
        ax.set_ylabel(f"{labels[a]} state (0 = embedding output)")
        ax.set_title(f"Layer-wise linear CKA, {labels[a]} vs {labels[b]}", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label="CKA")
    plt.show()


def plot_umap(embeddings, labels, modality, seed=42, n_cols=4):
    """Fit and draw a separate two-dimensional UMAP layout for each embedding matrix."""
    import matplotlib.pyplot as plt
    import umap

    keys = list(embeddings)
    n_rows = -(-len(keys) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        coords = umap.UMAP(random_state=seed).fit_transform(embeddings[key])
        ax.scatter(coords[:, 0], coords[:, 1], s=8, color=MODALITY_COLORS.get(modality[key], "gray"))
        ax.set_title(labels[key], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
