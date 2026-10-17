"""One training and scoring loop, shared by every Stage 3 and Stage 4 fusion model.

The settings are BioLangFusion's (Appendix A.4, List 3): Adam at 3e-5 with weight decay
1e-5, batch size 32, up to 500 epochs, ReduceLROnPlateau with patience 5, early stopping
with patience 20, mean squared error for regression and cross-entropy for classification,
plus the entropy penalty when the gated head is in use. Regression is scored by Spearman
correlation and classification by accuracy.

Two things the paper does not state are fixed here and reported with every run.

**What the schedule and the early stop watch.** The paper names both mechanisms and their
patience but not the quantity they monitor. Both watch the validation score, the same
quantity the paper reports, so the epoch that stops training is the epoch that is scored.

**Which epoch is scored.** The weights with the best validation score are restored before
the test split is touched, so the test score belongs to a model chosen without it.

A validation score can also be undefined: Spearman correlation is NaN whenever the
predictions or the labels do not vary, which the first epochs at this learning rate can
produce. An undefined score counts as no improvement, and the first epoch's weights are
kept whatever happens, so a run whose score never becomes defined ends with a model rather
than an error.

Results are written to disk per run, so a long sweep continues rather than restarts, and a
run in progress checkpoints every few epochs, so an interruption costs those epochs rather
than the run. Cross-attention over a 999-position joint context is the slow variant and the
one this matters for.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from .blf import build, loader
from .heads import MODALITIES


@dataclass
class TrainConfig:
    """BioLangFusion's optimisation settings; the defaults are the paper's."""

    lr: float = 3e-5
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 500
    early_stopping_patience: int = 20
    plateau_patience: int = 5
    entropy_lambda: float = 0.0
    channels: int = 100
    tconv: str = "appendix"
    seed: int = 0


def loss_function(task):
    return nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()


def score(task, truth, prediction):
    """Spearman correlation for regression, accuracy for classification."""
    if task == "regression":
        return float(spearmanr(truth, prediction).statistic)
    return float((truth == prediction.argmax(axis=1)).mean())


def predict(model, batches, device, task):
    """Return the labels, the predictions, and the mean modality weights over a split."""
    model.eval()
    truth, predicted, weights = [], [], []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            output, aux = model(batch)
            truth.append(batch.labels.cpu().numpy())
            predicted.append(output.squeeze(-1).cpu().numpy() if task == "regression"
                             else output.cpu().numpy())
            if "alpha" in aux:
                weights.append(aux["alpha"].cpu().numpy())
    alpha = np.concatenate(weights).mean(axis=0) if weights else None
    return np.concatenate(truth), np.concatenate(predicted), alpha


@dataclass
class Result:
    """What one training run produced."""

    variant: str
    benchmark: str
    config: dict
    epochs: int
    best_epoch: int
    val_score: float | None
    test_score: float
    seconds: float
    history: list = field(default_factory=list)
    alpha: dict | None = None

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @staticmethod
    def load(path):
        return Result(**json.loads(Path(path).read_text()))


def _save_checkpoint(path, epoch, model, optimiser, best, history, elapsed):
    """Write the run's state atomically, so an interrupted write cannot be resumed from."""
    temporary = path.with_name(path.name + ".partial")
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimiser": optimiser.state_dict(), "best": best, "history": history,
                "elapsed": elapsed}, temporary)
    os.replace(temporary, path)


def fit(model, streams, config, device, log=print, checkpoint=None, checkpoint_every=10):
    """Train to the early stop, restore the best epoch, and score the test split.

    With a checkpoint path, the run is written there every checkpoint_every epochs and
    resumed from it if it already exists; the file is removed once the run finishes.
    """
    torch.manual_seed(config.seed)
    task = streams.task
    splits = {name: loader(streams, name, batch_size=config.batch_size, seed=config.seed)
              for name in ("train", "val", "test")}
    criterion = loss_function(task)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=config.plateau_patience)

    model.to(device)
    best = {"score": -float("inf"), "epoch": -1, "state": None}
    history, started, first_epoch, before = [], time.perf_counter(), 0, 0.0
    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        best, history = saved["best"], saved["history"]
        first_epoch, before = saved["epoch"] + 1, saved.get("elapsed", 0.0)
        if log:
            log(f"    resumed at epoch {first_epoch}, best {best['score']:.4f} @ {best['epoch']}")
    for epoch in range(first_epoch, config.max_epochs):
        model.train()
        running = 0.0
        for batch in splits["train"]:
            batch = batch.to(device)
            optimiser.zero_grad()
            output, aux = model(batch)
            target = batch.labels if task == "classification" else batch.labels.view(-1, 1)
            loss = criterion(output, target)
            if config.entropy_lambda and "alpha" in aux:
                alpha = aux["alpha"].clamp_min(1e-12)
                loss = loss + config.entropy_lambda * (-(alpha * alpha.log()).sum(1).mean())
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(batch.codons)

        truth, predicted, _ = predict(model, splits["val"], device, task)
        val = score(task, truth, predicted)
        # An undefined score is not an improvement, but the first epoch's weights are kept
        # regardless so that there is always a model to restore.
        comparable = val if np.isfinite(val) else -float("inf")
        schedule.step(comparable)
        history.append({"epoch": epoch, "train_loss": running / len(splits["train"].dataset),
                        "val_score": val if np.isfinite(val) else None,
                        "lr": optimiser.param_groups[0]["lr"]})
        if comparable > best["score"] or best["state"] is None:
            best = {"score": comparable, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= config.early_stopping_patience:
            break
        if log and (epoch % 10 == 0 or epoch == config.max_epochs - 1):
            log(f"    epoch {epoch:3d}  train loss {history[-1]['train_loss']:.4f}  "
                f"val {val:.4f}  best {best['score']:.4f} @ {best['epoch']}")
        if checkpoint and epoch % checkpoint_every == 0:
            _save_checkpoint(checkpoint, epoch, model, optimiser, best, history,
                             before + time.perf_counter() - started)

    model.load_state_dict(best["state"])
    truth, predicted, alpha = predict(model, splits["test"], device, task)
    result = Result(
        variant=getattr(model, "fusion_name", type(model).__name__),
        benchmark=streams.key, config=asdict(config), epochs=len(history),
        best_epoch=best["epoch"],
        val_score=best["score"] if np.isfinite(best["score"]) else None,
        test_score=score(task, truth, predicted),
        seconds=before + time.perf_counter() - started,
        history=history,
        alpha=None if alpha is None else dict(zip(MODALITIES, [float(a) for a in alpha])))
    if checkpoint and checkpoint.exists():
        checkpoint.unlink()
    return result


def run(streams, variant, config, device, results_dir, log=print, **model_kwargs):
    """Train one variant, or return the run already on disk under the same settings."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{streams.key}__{variant.replace(':', '-')}__lambda{config.entropy_lambda}"
           f"__ch{config.channels}__{config.tconv}__seed{config.seed}")
    path = results_dir / f"{tag}.json"
    if path.exists():
        done = Result.load(path)
        if log:
            log(f"  {tag}: already run, test {done.test_score:.4f}")
        return done
    if log:
        log(f"  {tag}: training")
    model = build(streams, variant, channels=config.channels, tconv=config.tconv,
                  **model_kwargs)
    result = fit(model, streams, config, device, log=log,
                 checkpoint=results_dir / f"{tag}.checkpoint")
    result.save(path)
    if log:
        val = "undefined" if result.val_score is None else f"{result.val_score:.4f}"
        log(f"  {tag}: test {result.test_score:.4f} "
            f"(val {val} at epoch {result.best_epoch}, "
            f"{result.seconds / 60:.1f} min)")
    return result
