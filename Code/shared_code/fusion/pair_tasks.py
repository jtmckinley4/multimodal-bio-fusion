"""Pairs of separate molecules: splits, controls and binary scoring.

Everything the project has fitted so far reads one coding sequence through three encoders,
placed on a shared codon grid. The plan's Parts B and C change the object of study: an RNA and
a protein that bind, a protein and a small molecule with a measured affinity, a DNA region and
the factor that occupies it. Two sequences of different alphabets and unrelated lengths, with
no position-by-position correspondence, and a label that belongs to the pair rather than to
either partner.

That changes the evaluation before it changes the model, which is what this module holds.

**A pair table, not a row per sequence.** Each row names a left and a right entity and carries
one label, and each entity appears in many rows. Sequences live beside the table, one per
entity, so an encoder reads each entity once.

**Splits are over entities, not pairs.** A random split of pairs leaves the same protein in
training and in test, so a model can score well by learning which proteins bind often. The
plan asks for left-disjoint, right-disjoint and dual-disjoint evaluation; ``disjoint_split``
builds them over *groups* of entities, so related sequences cannot be separated by the split
either. Dual-disjoint discards every pair that crosses two partitions, and the count discarded
is returned rather than hidden.

**Negatives carry their provenance.** A measured non-binding pair and a pair invented by
sampling two entities are different evidence. Every row records which it is, and
``label_provenance_counts`` reports the mixture, so a result can never be read as if artificial
negatives were measured.

**Affinity tables are the same object with a different label.** The plan's Step 8 measures a
continuous endpoint rather than a yes or no, so ``kind="affinity"`` keeps the pair table, the
entity-disjoint splits and the grouped intervals, and changes only what a label may hold and
which metrics are reported. Partner reassignment has no meaning there, because an invented
pair has no affinity to compare with, and it is refused rather than scored.

**Two controls belong with the task, not with the model.** ``partner_reassignment`` rebuilds the
evaluation set with partners permuted within the split, which is the plan's partner-reassignment
control: a model that has learned the pair rather than the partner's popularity should lose its
score. ``degree_baseline`` predicts a pair from how often each entity binds in training alone,
without reading a sequence at all; a fusion model that cannot beat it has not used its inputs.

Scoring is AUROC and average precision, with ``score_rows`` written to the signature
``train.seed_averaged_bootstrap`` expects, so grouped paired intervals use the same resampling
rule as every other comparison in the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIDES = ("left", "right")
SPLITS = ("train", "val", "test")
# How a row's label was obtained. Only "measured_positive" and "measured_negative" are evidence
# about binding; the rest are constructed and are reported separately wherever they are used.
PROVENANCE = ("measured_positive", "measured_negative", "artificial_sampled",
              "artificial_reassigned")
HELD_OUT = ("left", "right", "both")
KINDS = ("binding", "affinity")
# The kinds of string a side can hold. Each one names the alphabet an encoder must read:
# a protein side is amino acids, a ligand side is SMILES.
MODALITIES = ("dna", "rna", "protein", "ligand")


@dataclass
class PairFrame:
    """One pair task: the labelled pairs, each side's sequences, and each side's groups.

    ``pairs`` holds ``left``, ``right``, ``label`` and ``provenance`` columns in a pandas frame.
    ``left_sequences`` and ``right_sequences`` map an entity name to the string an encoder
    reads. ``left_groups`` and ``right_groups`` map an entity name to the group it may not be
    separated from (a protein family, a sequence-identity cluster); an entity with no group
    entry is its own group, which is weaker and is counted by ``grouping_coverage``.
    """

    name: str
    pairs: object
    left_sequences: dict
    right_sequences: dict
    left_groups: dict = field(default_factory=dict)
    right_groups: dict = field(default_factory=dict)
    # "binding" holds 0/1 labels; "affinity" holds one measured endpoint per pair, which the
    # table must already have put on a single scale (the plan forbids pooling Kd, Ki and IC50
    # without explicit handling, and this class cannot check that for you).
    kind: str = "binding"
    # What each side's strings actually are, which decides which encoder may read them. The
    # defaults describe the RNA-and-protein tables this class was written for; a DNA table
    # declares "dna" and a ligand table declares "ligand", and nothing downstream has to
    # guess from the table's name.
    left_modality: str = "rna"
    right_modality: str = "protein"

    def __post_init__(self):
        columns = {"left", "right", "label", "provenance"}
        missing = columns - set(self.pairs.columns)
        if missing:
            raise ValueError(f"{self.name}: pair table needs columns {sorted(missing)}")
        unknown = set(self.pairs["provenance"]) - set(PROVENANCE)
        if unknown:
            raise ValueError(f"{self.name}: unknown label provenance {sorted(unknown)}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}")
        for side in SIDES:
            modality = self.modality(side)
            if modality not in MODALITIES:
                raise ValueError(f"{self.name}: {side} modality {modality!r} is not one of "
                                 f"{MODALITIES}")
        labels = np.asarray(self.pairs["label"])
        if self.kind == "binding" and not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{self.name}: labels must be 0 or 1 for a binding task")
        if self.kind == "affinity":
            if not np.isfinite(labels.astype(float)).all():
                raise ValueError(f"{self.name}: affinity labels must all be finite")
            if "endpoint" not in self.pairs.columns:
                raise ValueError(f"{self.name}: an affinity table needs an 'endpoint' column "
                                 "naming the assay each label came from")
            if self.pairs["endpoint"].nunique() > 1:
                raise ValueError(f"{self.name}: this table mixes "
                                 f"{sorted(set(self.pairs['endpoint']))}; put one endpoint in a "
                                 "table, or record the conversion that made them comparable")
        for side, sequences in (("left", self.left_sequences), ("right", self.right_sequences)):
            absent = set(self.pairs[side]) - set(sequences)
            if absent:
                raise ValueError(f"{self.name}: {len(absent)} {side} entities have no sequence")
        if self.pairs.duplicated(["left", "right"]).any():
            raise ValueError(f"{self.name}: the same pair appears more than once")

    def modality(self, side):
        """What one side's strings are: "dna", "rna", "protein" or "ligand"."""
        return self.left_modality if side == "left" else self.right_modality

    def sequences(self, side):
        """One side's entity name to string map."""
        return self.left_sequences if side == "left" else self.right_sequences

    def group_of(self, side, entity):
        groups = self.left_groups if side == "left" else self.right_groups
        return groups.get(entity, f"{side}:{entity}")

    def groups(self, side):
        """The group label of every row, for grouped resampling of this side."""
        return np.array([self.group_of(side, entity) for entity in self.pairs[side]])

    def grouping_coverage(self):
        """How many entities carry a real group; the rest stand alone, which is weaker."""
        return {side: {"entities": len(sequences),
                       "grouped": sum(1 for e in sequences
                                      if e in (self.left_groups if side == "left"
                                               else self.right_groups))}
                for side, sequences in (("left", self.left_sequences),
                                        ("right", self.right_sequences))}

    def label_provenance_counts(self, rows=None):
        table = self.pairs if rows is None else self.pairs.iloc[np.asarray(rows)]
        return {name: int(count) for name, count in
                table["provenance"].value_counts().sort_index().items()}


def _partition_groups(groups, ratios, rng):
    """Assign whole groups to train, val and test by size, largest first.

    Entity groups are uneven, so drawing them at random leaves partition sizes far from the
    ratios. Placing the largest remaining group in whichever partition is furthest below its
    share keeps the sizes close while the assignment still depends on the seed through the
    shuffle of equal-sized groups.
    """
    names, sizes = np.unique(groups, return_counts=True)
    order = rng.permutation(len(names))
    names, sizes = names[order], sizes[order]
    order = np.argsort(-sizes, kind="stable")
    wanted = np.asarray(ratios, dtype=float) * sizes.sum()
    placed = np.zeros(len(SPLITS))
    assignment = {}
    for index in order:
        choice = int(np.argmax(wanted - placed))
        assignment[names[index]] = SPLITS[choice]
        placed[choice] += sizes[index]
    return assignment


def disjoint_split(frame, *, held_out="left", ratios=(0.7, 0.15, 0.15), seed=42):
    """Split pairs so that held-out entity groups never appear in training.

    ``held_out`` is "left", "right" or "both". The split is over groups of entities: with
    "left", every pair follows its left entity's group; with "both", a pair is kept only when
    both of its entities land in the same partition, and the pairs that cross partitions are
    discarded. Returns ``{"indices": {split: row indices}, "entities": {...}, "dropped": n}``.

    The dropped count matters: dual-disjoint splits discard most of a dense pair table, and a
    test set that small is a different experiment from the left-disjoint one, not a stricter
    version of the same one.
    """
    if held_out not in HELD_OUT:
        raise ValueError(f"held_out must be one of {HELD_OUT}")
    if len(ratios) != len(SPLITS) or not np.isclose(sum(ratios), 1.0) or min(ratios) <= 0:
        raise ValueError("ratios must be three positive fractions summing to one")
    rng = np.random.default_rng(seed)
    sides = ("left", "right") if held_out == "both" else (held_out,)
    # Each side is partitioned independently, so "both" is not a relabeling of one side.
    assignment = {side: _partition_groups(
        np.array([frame.group_of(side, e) for e in
                  (frame.left_sequences if side == "left" else frame.right_sequences)]),
        ratios, rng) for side in sides}
    rows = {name: [] for name in SPLITS}
    dropped = 0
    for position, pair in enumerate(frame.pairs.itertuples(index=False)):
        entities = {side: getattr(pair, side) for side in sides}
        chosen = {assignment[side][frame.group_of(side, entity)]
                  for side, entity in entities.items()}
        if len(chosen) == 1:
            rows[chosen.pop()].append(position)
        else:
            dropped += 1
    indices = {name: np.asarray(values, dtype=np.int64) for name, values in rows.items()}
    for name, values in indices.items():
        if len(values) == 0:
            raise ValueError(f"{frame.name}: the {name} partition is empty under this split")
    return {"held_out": held_out, "indices": indices, "dropped": int(dropped),
            "entities": {side: {name: sorted(e for e, s in assignment[side].items() if s == name)
                                for name in SPLITS} for side in sides},
            "ratios": tuple(ratios), "seed": int(seed),
            "provenance": {name: frame.label_provenance_counts(values)
                           for name, values in indices.items()}}


def partner_reassignment(frame, rows, *, seed=0, side="right"):
    """The plan's control: keep each left entity, give it a different partner.

    Only positive pairs are reassigned, and only to partners the entity is not recorded as
    binding, so a reassigned row is a pair with no evidence behind it. The result is scored
    exactly like the real evaluation set; a model that reads the pair should score lower on it,
    while a model that has learned which entities are popular should not notice the change.

    A row whose entity is already recorded against every available partner cannot be
    reassigned and is dropped. On a table dense enough that no row can be reassigned, this
    control does not exist, which is an error rather than an empty set to score.
    """
    if frame.kind != "binding":
        raise ValueError(f"{frame.name}: partner reassignment needs binding labels; an invented "
                         "pair has no measured affinity to compare with")
    rng = np.random.default_rng(seed)
    table = frame.pairs.iloc[np.asarray(rows)].reset_index(drop=True)
    keep = table["label"].to_numpy() == 1
    known = set(zip(frame.pairs["left"], frame.pairs["right"]))
    partners = sorted(set(table[side]))
    if len(partners) < 2:
        raise ValueError(f"{frame.name}: reassignment needs at least two {side} entities here")
    other = "left" if side == "right" else "right"
    reassigned, usable = table.copy(), np.zeros(len(table), dtype=bool)
    built = set()  # a reassigned pair is built once, so the control holds no repeats
    for position in np.flatnonzero(keep):
        anchor = table.at[position, other]
        choices = [p for p in partners
                   if (anchor, p) not in known and (p, anchor) not in known
                   and (anchor, p) not in built]
        if not choices:
            continue  # every available partner is already recorded or already used here
        chosen = choices[rng.integers(len(choices))]
        built.add((anchor, chosen))
        reassigned.at[position, side] = chosen
        reassigned.at[position, "provenance"] = "artificial_reassigned"
        usable[position] = True
    if not usable.any():
        raise ValueError(f"{frame.name}: no positive pair here has an unrecorded partner, so "
                         "partner reassignment cannot be built for this evaluation set")
    return PairFrame(f"{frame.name}:partner-reassigned",
                     reassigned.loc[usable].reset_index(drop=True),
                     frame.left_sequences, frame.right_sequences,
                     frame.left_groups, frame.right_groups, frame.kind,
                     frame.left_modality, frame.right_modality)


def degree_baseline(frame, train_rows, evaluate_rows, *, prior=1.0):
    """Score a pair from its entities' training labels alone, reading no sequence.

    For entity e let p(e) be the share of its training pairs that are positive, smoothed
    towards the training positive rate. The pair's score is the product of the two shares.
    For an affinity table the same idea reads as each entity's mean measured affinity, and the
    pair's prediction is the mean of the two, since a product of affinities means nothing.

    This is the control for "the label is predictable from the partners alone"; it is not a
    model of binding or affinity, and is never a baseline a method should be praised for
    beating by a small margin.
    """
    table = frame.pairs
    train = table.iloc[np.asarray(train_rows)]
    base = float(train["label"].mean())
    rates = {}
    for side in ("left", "right"):
        totals = train.groupby(side)["label"].agg(["sum", "count"])
        rates[side] = {entity: (row["sum"] + prior * base) / (row["count"] + prior)
                       for entity, row in totals.iterrows()}
    evaluate = table.iloc[np.asarray(evaluate_rows)]
    join = (lambda a, b: a * b) if frame.kind == "binding" else (lambda a, b: (a + b) / 2)
    return np.array([join(rates["left"].get(pair.left, base), rates["right"].get(pair.right, base))
                     for pair in evaluate.itertuples(index=False)], dtype=np.float64)


def auroc(truth, score):
    """Rank-based AUROC, ties shared. Undefined with one class present, which returns NaN."""
    truth = np.asarray(truth)
    score = np.asarray(score, dtype=np.float64)
    positives, negatives = int((truth == 1).sum()), int((truth == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_score = score[order]
    start = 0
    for index in range(1, len(score) + 1):
        if index == len(score) or sorted_score[index] != sorted_score[start]:
            ranks[order[start:index]] = 0.5 * (start + index + 1)
            start = index
    return float((ranks[truth == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def average_precision(truth, score):
    """Average precision: the step-wise area under the precision-recall curve.

    Rows with equal scores are grouped, so a model that gives every pair the same score cannot
    be flattered by the order rows happen to sit in.
    """
    truth = np.asarray(truth)
    score = np.asarray(score, dtype=np.float64)
    positives = int((truth == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    labels, scores = truth[order], score[order]
    total, previous_recall, seen, hits = 0.0, 0.0, 0, 0
    for index in range(len(labels)):
        seen += 1
        hits += int(labels[index] == 1)
        if index + 1 < len(labels) and scores[index + 1] == scores[index]:
            continue  # a tied block is one decision point, taken at its end
        recall = hits / positives
        total += (recall - previous_recall) * (hits / seen)
        previous_recall = recall
    return float(total)


def rmse(truth, prediction):
    """Root mean squared error of an affinity prediction, in the label's own units."""
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def concordance_index(truth, prediction):
    """Share of comparable pairs of rows ranked in the right order, ties counting a half.

    The standard measure for affinity benchmarks: it asks whether the model orders two
    measurements correctly, which is the question a screening decision actually poses. Rows
    with equal measured affinity are not comparable and are excluded.
    """
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    difference = truth[:, None] - truth[None, :]
    comparable = difference > 0
    if not comparable.any():
        return float("nan")
    predicted = prediction[:, None] - prediction[None, :]
    correct = (predicted[comparable] > 0).sum() + 0.5 * (predicted[comparable] == 0).sum()
    return float(correct / comparable.sum())


def spearman(truth, prediction):
    """Rank correlation, the project's regression score, read for affinity too."""
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(truth), np.asarray(prediction)).statistic)


def score_rows(task, truth, prediction):
    """A pair metric in ``train.score_rows``' signature.

    Pass as ``score_fn`` to ``train.seed_averaged_bootstrap`` so that pair comparisons use the
    project's grouped paired-bootstrap rule unchanged; ``task`` selects the metric.
    """
    if task == "auroc":
        return auroc(truth, prediction)
    if task == "auprc":
        return average_precision(truth, prediction)
    if task == "spearman":
        return spearman(truth, prediction)
    if task == "concordance":
        return concordance_index(truth, prediction)
    if task == "neg_rmse":
        # Negated so that every pair metric is higher-is-better, which selection assumes.
        return -rmse(truth, prediction)
    raise ValueError(f"unsupported pair task metric: {task!r}")


def affinity_scores(truth, prediction):
    """The three reported affinity numbers, with the spread the error is read against."""
    truth = np.asarray(truth, dtype=np.float64)
    return {"rmse": rmse(truth, prediction), "spearman": spearman(truth, prediction),
            "concordance": concordance_index(truth, prediction), "rows": int(len(truth)),
            "label_sd": float(np.std(truth))}


def scores_for(kind, truth, prediction):
    """The reported numbers for a pair task of either kind."""
    if kind == "binding":
        return binding_scores(truth, prediction)
    if kind == "affinity":
        return affinity_scores(truth, prediction)
    raise ValueError(f"unsupported pair task kind: {kind!r}")


def binding_scores(truth, prediction):
    """Both reported metrics, with the positive rate the average precision is read against."""
    truth = np.asarray(truth)
    return {"auroc": auroc(truth, prediction), "auprc": average_precision(truth, prediction),
            "positives": int((truth == 1).sum()), "rows": int(len(truth)),
            "positive_rate": float((truth == 1).mean()) if len(truth) else float("nan")}
