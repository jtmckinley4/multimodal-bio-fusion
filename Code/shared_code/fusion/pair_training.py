"""Fit and evaluate a pair scorer, with the controls the comparison needs beside it.

One fit is one arm of the comparison: a scorer from ``pair_models``, a split from
``pair_tasks``, frozen embeddings read once per entity rather than once per pair, and the same
recipe for every arm. What comes back is the saved test predictions, so any two arms can be
compared afterwards by the project's grouped paired bootstrap, plus the controls that say
whether a score means what it appears to mean.

**Embeddings are per entity.** A dense pair table holds each protein in hundreds of rows; its
tokens are encoded once and indexed by name. ``PairEmbeddings`` is the small interface a token
cache has to satisfy, and ``DictEmbeddings`` is the in-memory version used for checks.

**Selection never touches test data.** Early stopping and the epoch choice read validation
only, and the test set is scored once, at the end, with the selected weights.

**Class imbalance is handled where it is visible.** Binding tables are mostly negative, so the
loss can carry a positive-class weight computed from the training rows alone; it is recorded
with the result rather than tuned per arm.

**An affinity table runs through the same loop.** ``PairFrame.kind`` decides the loss (binary
cross-entropy or squared error) and which metrics are reported; the arms, the splits and the
intervals are unchanged, which is the point of keeping the two tasks in one place. Selection
still reads validation only, with a metric that is higher-is-better either way.

**Three controls run with every evaluation**, because each answers a question a single AUROC
cannot: the degree baseline (is the label predictable from how often each entity binds?), the
partner-reassignment set (does the score survive replacing the partner?), and the single-partner
arms fitted in the same loop (does the pair matter at all?). The first two are computed here;
the third is simply another arm of the same comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..encoders import free_memory  # noqa: F401  (re-exported: pair_study frees between arms)
from . import pair_tasks
from .pair_models import build

SIDES = ("left", "right")


@dataclass
class PairConfig:
    """The recipe every arm shares. Changing one of these changes the comparison."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5
    max_tokens: int = 1024          # per side; longer entities are truncated, and it is recorded
    positive_weight: str = "balanced"   # "balanced" or "none"; binding tables only
    selection_metric: str = None        # validation metric; the kind's default if unset
    seed: int = 0
    # How many pairs are put through the model at once. This is a memory setting, not a
    # training setting: a batch is still one optimizer step, and the gradient it takes is the
    # gradient of the whole batch, accumulated over slices of this size. Cross-attention
    # between two 1,024-token streams holds an attention matrix per head per direction, which
    # at batch 32 is several gigabytes before the backward pass; slicing the batch divides
    # that peak without changing the step. None puts the whole batch through at once.
    microbatch: int = None
    # Whether a batch is drawn from rows of similar length rather than at random. A batch is
    # padded to its longest member, so when a table holds a few entities at the token cap and
    # many short ones, a random batch is nearly always padded to the cap and the model spends
    # almost all of its arithmetic, and its memory, on padding. Grouping by length removes
    # that waste. It changes which rows share a batch, so it changes the fit, and it is
    # recorded.
    bucket_by_length: bool = False

    DEFAULT_SELECTION = {"binding": "auprc", "affinity": "spearman"}

    def selection_for(self, kind):
        """The validation metric, higher-is-better, for this task kind."""
        return self.selection_metric or self.DEFAULT_SELECTION[kind]

    def slices(self, size):
        """Index ranges that cover a batch of ``size`` pairs, one per forward pass."""
        if self.microbatch is not None and self.microbatch < 1:
            raise ValueError("microbatch must be a positive integer or None")
        step = size if self.microbatch is None else self.microbatch
        return [(start, min(start + step, size)) for start in range(0, size, step)]

    def as_record(self):
        """What identifies this configuration.

        ``microbatch`` is left out when it is unset, because no slicing happens then and the
        record would otherwise differ from every fit saved before the setting existed. When it
        is set it is recorded, since accumulating a gradient in slices sums the same terms in a
        different order and the last digits of a result can move.
        """
        record = {field: getattr(self, field) for field in self.__dataclass_fields__}
        if record.get("microbatch") is None:
            record.pop("microbatch", None)
        if not record.get("bucket_by_length"):
            record.pop("bucket_by_length", None)
        return record


class PairEmbeddings:
    """Token embeddings for one side, by entity name: ``tokens(name) -> (length, width)``."""

    width: int

    def tokens(self, name):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class DictEmbeddings(PairEmbeddings):
    """In-memory embeddings, for checks and for tables small enough to hold at once."""

    vectors: dict
    width: int = field(init=False)

    def __post_init__(self):
        widths = {int(np.asarray(v).shape[-1]) for v in self.vectors.values()}
        if len(widths) != 1:
            raise ValueError(f"embeddings must share one width; found {sorted(widths)}")
        self.width = widths.pop()

    def tokens(self, name):
        return torch.as_tensor(np.asarray(self.vectors[name]), dtype=torch.float32)


def _padded(rows, limit):
    """Pad a list of (length, width) tensors into one batch, truncating past ``limit``."""
    rows = [row[:limit] for row in rows]
    length = max(row.shape[0] for row in rows)
    stacked = rows[0].new_zeros(len(rows), length, rows[0].shape[1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for index, row in enumerate(rows):
        stacked[index, :row.shape[0]] = row
        mask[index, :row.shape[0]] = True
    return stacked, mask


class PairBatches:
    """Batches of pairs as padded token streams, in a fixed order or shuffled by seed."""

    def __init__(self, frame, rows, embeddings, config, *, shuffle=False, seed=0):
        self.frame, self.rows = frame, np.asarray(rows)
        self.embeddings, self.config = embeddings, config
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __len__(self):
        return math.ceil(len(self.rows) / self.config.batch_size)

    def _lengths(self):
        """Each row's padded width: the longer of its two sides, which is what it costs."""
        table = self.frame.pairs
        cache = {}
        widths = []
        for row in self.rows:
            pair = table.iloc[row]
            total = 0
            for side in SIDES:
                name = pair[side]
                key = (side, name)
                if key not in cache:
                    cache[key] = min(len(self.embeddings[side].tokens(name)),
                                     self.config.max_tokens)
                total = max(total, cache[key])
            widths.append(total)
        return np.asarray(widths)

    def _batches(self):
        """Row indices per batch.

        Drawn at random, a batch of this table is padded to its longest member, and with a
        quarter of the entities at the token cap almost every batch inherits that width even
        though the median entity is an order of magnitude shorter. Grouping rows of similar
        length means a batch is padded to what it actually holds. The order of the batches is
        still shuffled every epoch, so the optimizer does not see a length schedule.
        """
        order = self.rows
        if self.config.bucket_by_length:
            order = self.rows[np.argsort(self._lengths(), kind="stable")]
        elif self.shuffle:
            order = np.random.default_rng((self.seed, self.epoch)).permutation(order)
        size = self.config.batch_size
        batches = [order[start:start + size] for start in range(0, len(order), size)]
        if self.config.bucket_by_length and self.shuffle:
            rng = np.random.default_rng((self.seed, self.epoch))
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        table = self.frame.pairs
        batches = self._batches()
        if self.shuffle:
            self.epoch += 1          # after drawing, so epoch 0 draws as it always did
        for picked in batches:
            batch = table.iloc[picked]
            streams = {}
            for side in SIDES:
                streams[side] = _padded([self.embeddings[side].tokens(name)
                                         for name in batch[side]], self.config.max_tokens)
            labels = torch.as_tensor(batch["label"].to_numpy().copy(), dtype=torch.float32)
            yield (streams["left"][0], streams["right"][0],
                   streams["left"][1], streams["right"][1], labels, picked)


def _positive_weight(frame, rows, setting):
    if setting == "none" or frame.kind != "binding":
        return None
    labels = frame.pairs.iloc[np.asarray(rows)]["label"].to_numpy()
    positives = int((labels == 1).sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("the training rows hold a single class")
    return torch.tensor((len(labels) - positives) / positives, dtype=torch.float32)


def predict(model, batches, device="cpu"):
    """Scores for every row of ``batches``, in the order its rows were given."""
    model.eval()
    scores = np.zeros(len(batches.rows), dtype=np.float64)
    position = {row: index for index, row in enumerate(batches.rows)}
    with torch.no_grad():
        for left, right, left_mask, right_mask, _, picked in batches:
            # Scoring holds no graph, but a padded attention matrix is large on its own, so the
            # batch is sliced here too. Each row's score depends only on its own pair, so this
            # changes nothing but the peak memory.
            for start, stop in batches.config.slices(len(picked)):
                logit, _ = model(left[start:stop].to(device), right[start:stop].to(device),
                                 left_mask[start:stop].to(device),
                                 right_mask[start:stop].to(device))
                for row, value in zip(picked[start:stop], logit.detach().cpu().numpy()):
                    scores[position[row]] = float(value)
    return scores


def fit(arm, frame, split, embeddings, config, *, device="cpu", model_kwargs=None, log=print):
    """Fit one arm on the training rows, select on validation, score the test rows once.

    Returns the arm's record: the selected epoch, the validation and test metrics, the saved
    test scores and the rows they belong to, and the recipe that produced them.
    """
    torch.manual_seed(config.seed)
    widths = {side: embeddings[side].width for side in SIDES}
    model = build(arm, widths, **(model_kwargs or {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    weight = _positive_weight(frame, split["indices"]["train"], config.positive_weight)
    loss_fn = (nn.BCEWithLogitsLoss(pos_weight=None if weight is None else weight.to(device))
               if frame.kind == "binding" else nn.MSELoss())
    selection = config.selection_for(frame.kind)
    batches = {name: PairBatches(frame, split["indices"][name], embeddings, config,
                                 shuffle=(name == "train"), seed=config.seed)
               for name in pair_tasks.SPLITS}
    dtype = np.int64 if frame.kind == "binding" else np.float64
    truth = {name: frame.pairs.iloc[split["indices"][name]]["label"].to_numpy(dtype=dtype)
             for name in pair_tasks.SPLITS}

    # The selected weights are kept on the host. On a unified-memory GPU a state dict copied
    # on the device is held in the same pool the next batch needs, and a fit keeps one of
    # these for every epoch that improves.
    def _snapshot():
        return {name: tensor.detach().to("cpu", copy=True)
                for name, tensor in model.state_dict().items()}

    best = {"score": -np.inf, "epoch": 0, "state": _snapshot()}
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for left, right, left_mask, right_mask, labels, _ in batches["train"]:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            # One optimizer step per batch, whatever the slicing. Each slice's mean loss is
            # weighted by its share of the batch, so the accumulated gradient is the gradient
            # of the batch's own mean loss rather than of a smaller batch.
            for start, stop in config.slices(len(labels)):
                share = (stop - start) / len(labels)
                logit, _ = model(left[start:stop].to(device), right[start:stop].to(device),
                                 left_mask[start:stop].to(device),
                                 right_mask[start:stop].to(device))
                loss = loss_fn(logit, labels[start:stop].to(device))
                (loss * share).backward()
                batch_loss += loss.item() * (stop - start)
            optimizer.step()
            total += batch_loss
        validation = pair_tasks.score_rows(selection, truth["val"],
                                           predict(model, batches["val"], device))
        history.append({"epoch": epoch, "train_loss": total / len(batches["train"].rows),
                        f"val_{selection}": validation})
        if log is not None:
            log(f"{arm} epoch {epoch}: loss {history[-1]['train_loss']:.4f}, "
                f"val {selection} {validation:.4f}")
        if validation > best["score"]:
            best = {"score": validation, "epoch": epoch, "state": _snapshot()}
        elif epoch - best["epoch"] >= config.early_stopping_patience:
            break
    model.load_state_dict(best["state"])
    # Test data is read once, with the weights validation selected, and never before.
    test_scores = predict(model, batches["test"], device)
    return {"arm": arm, "kind": frame.kind, "selection_metric": selection,
            "model_kwargs": dict(model_kwargs or {}), "config": config.as_record(),
            "widths": widths, "best_epoch": best["epoch"], "history": history,
            "val": pair_tasks.scores_for(frame.kind, truth["val"],
                                         predict(model, batches["val"], device)),
            "test": pair_tasks.scores_for(frame.kind, truth["test"], test_scores),
            "test_rows": split["indices"]["test"].tolist(),
            "test_truth": truth["test"].tolist(), "test_score": test_scores.tolist(),
            "positive_weight": None if weight is None else float(weight),
            "split": {"held_out": split["held_out"], "seed": split["seed"],
                      "dropped": split["dropped"], "provenance": split["provenance"]},
            "model": model}


def control_scores(result, frame, split, embeddings, config, *, device="cpu", seed=0):
    """The two task-level controls for a fitted arm, scored like its own test set.

    ``degree`` reads no sequence at all. ``partner_reassigned`` keeps each positive pair's
    anchor and gives it a partner nobody measured; a scorer that reads the pair should fall
    towards chance there, while one that has learned which entities are popular should not.
    """
    truth = np.asarray(result["test_truth"])
    degree = pair_tasks.degree_baseline(frame, split["indices"]["train"],
                                        split["indices"]["test"])
    controls = {"degree": {**pair_tasks.scores_for(frame.kind, truth, degree),
                           "reads_sequences": False}}
    if frame.kind != "binding":
        # An invented pair has no measured affinity, so there is nothing to score against.
        return controls
    control = pair_tasks.partner_reassignment(frame, split["indices"]["test"], seed=seed)
    reassigned = PairBatches(control, np.arange(len(control.pairs)), embeddings, config)
    scores = predict(result["model"], reassigned, device)
    controls["partner_reassigned"] = {
        "rows": int(len(control.pairs)),
        "mean_score_real": float(np.mean(np.asarray(result["test_score"])[truth == 1])),
        "mean_score_reassigned": float(np.mean(scores))}
    return controls


def comparison_rows(results, frame, split, *, metric=None, groups="right", draws=2000,
                    seed=0, reference=None):
    """Paired intervals between arms on the same test rows, grouped by one side's entities.

    Uses ``train.seed_averaged_bootstrap`` with the pair metric, so the resampling rule is the
    project's: whole entity groups move together, and the comparison stays paired. ``reference``
    names the arm every other arm is compared with; by default each fusion arm is compared with
    the better single-partner baseline it has to beat.
    """
    from .train import seed_averaged_bootstrap

    metric = metric or {"binding": "auroc", "affinity": "spearman"}[frame.kind]
    names = [result["arm"] for result in results]
    rows = results[0]["test_rows"]
    if any(result["test_rows"] != rows for result in results):
        raise ValueError("arms were scored on different test rows")
    truth = np.asarray(results[0]["test_truth"])
    predictions = {result["arm"]: [np.asarray(result["test_score"])] for result in results}
    if reference is None:
        singles = [name for name in ("left_only", "right_only") if name in predictions]
        if not singles:
            raise ValueError("no single-partner baseline among these arms; name a reference")
        reference = max(singles, key=lambda name: pair_tasks.score_rows(
            metric, truth, predictions[name][0]))
    comparisons = [(f"{name} vs {reference}", name, reference)
                   for name in names if name != reference]
    group_labels = frame.groups(groups)[np.asarray(rows)]
    return {"reference": reference, "metric": metric, "grouped_by": groups,
            "rows": seed_averaged_bootstrap(metric, truth, predictions, comparisons,
                                            groups=group_labels, draws=draws, seed=seed,
                                            score_fn=pair_tasks.score_rows)}
