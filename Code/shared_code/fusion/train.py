"""One training and scoring loop, shared by every Stage 3 and Stage 4 fusion model.

The settings are BioLangFusion's (Appendix A.4, List 3): Adam at 3e-5 with weight decay
1e-5, batch size 32, up to 500 epochs, ReduceLROnPlateau with patience 5, early stopping
with patience 20, mean squared error for regression and cross-entropy for classification,
plus the entropy penalty when the gated head is in use. Regression is scored by Spearman
correlation and classification by accuracy.

Two things the paper does not state are fixed here and reported with every run.

**What the schedule and the early stop watch.** The paper names both mechanisms and their
patience but not the quantity they monitor. Both watch the validation score, the same
quantity the paper reports. Testing uses the best validation epoch, which can precede
the epoch at which training stops.

**Which epoch is scored.** The weights with the best validation score are restored before
the test split is touched, so the test score belongs to a model chosen without it.

A validation score can also be undefined: Spearman correlation is NaN whenever the
predictions or the labels do not vary, which the first epochs at this learning rate can
produce. An undefined score counts as no improvement, and the first epoch's weights are
kept whatever happens, so a run whose score never becomes defined ends with a model rather
than an error.

Results and checkpoints are reused only when their recorded computation identity matches.
Checkpoint state includes the scheduler and random generators, allowing an epoch-boundary
continuation in the same environment. This does not guarantee identical floating-point
results across devices or eliminate nondeterministic accelerator kernels. Legacy results
remain readable as historical evidence and are never silently overwritten.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import spearmanr
from torch import nn

from .blf import build, loader
from .heads import MODALITIES
from .token_embeddings import file_fingerprint


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
    """Mean squared error for one or many continuous outputs, cross-entropy for classes."""
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


def score(task, truth, prediction):
    """Spearman for regression, accuracy for classification, mean Spearman per output for
    an expression panel. The panel is averaged because IsoFormer reports one number for
    "Spearman correlation across tissues" over its thirty tissues."""
    if task == "regression":
        return float(spearmanr(truth, prediction).statistic)
    if task == "expression":
        return float(np.mean([spearmanr(truth[:, i], prediction[:, i]).statistic
                              for i in range(truth.shape[1])]))
    return float((truth == prediction.argmax(axis=1)).mean())


def r_squared(truth, prediction):
    """Mean coefficient of determination over the columns of an expression panel."""
    residual = ((truth - prediction) ** 2).sum(axis=0)
    total = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1 - residual / np.where(total == 0, np.nan, total)))


def score_rows(task, truth, prediction):
    """The same score from stored per-row predictions, which are already class ids."""
    if task == "classification":
        return float((truth == prediction).mean())
    return score(task, truth, prediction)


def paired_bootstrap(first, second, task, draws=2000, seed=0):
    """Resample the test rows and ask how often one run scores above another.

    Two models scored on the same rows are a paired comparison, so whether their separately
    computed confidence intervals overlap says little: the question is the distribution of
    the difference, and that is what this resamples. Returns the observed difference, a 95%
    interval for it, and the share of resamples the first run wins.

    Both runs need stored labels and predictions in the same row order. Matching labels
    are a necessary check, not proof of row identity: the caller must verify the test-row
    correspondence and evaluation protocol before using this comparison.
    """
    if any(values is None for result in (first, second)
           for values in (result.test_truth, result.test_prediction)):
        raise ValueError("Both runs need stored labels and per-row predictions; preserve "
                         "score-only results and explicitly plan any additional fits.")
    if first.benchmark != second.benchmark:
        raise ValueError("The two runs name different benchmarks.")
    if first.test_truth != second.test_truth:
        raise ValueError("The stored test labels differ; check the test rows and their order.")
    truth = np.asarray(first.test_truth)
    a, b = np.asarray(first.test_prediction), np.asarray(second.test_prediction)
    if len(truth) == 0 or a.shape != truth.shape or b.shape != truth.shape:
        raise ValueError("Stored predictions must match the nonempty test-label array.")
    if not all(np.isfinite(values).all() for values in (truth, a, b)):
        raise ValueError("Stored test labels and predictions must be finite.")
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer.")
    difference = score_rows(task, truth, a) - score_rows(task, truth, b)
    if not np.isfinite(difference):
        raise ValueError("The observed paired score difference is undefined.")
    rng = np.random.default_rng(seed)
    spread = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(truth), len(truth))
        spread[i] = (score_rows(task, truth[pick], a[pick])
                     - score_rows(task, truth[pick], b[pick]))
        if not np.isfinite(spread[i]):
            raise ValueError(f"Bootstrap draw {i} has an undefined score; no interval was produced.")
    low, high = np.percentile(spread, [2.5, 97.5])
    return {"difference": difference,
            "ci_low": float(low), "ci_high": float(high),
            "first_wins": float((spread > 0).mean())}


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
            # A classification output keeps its class axis for argmax; an expression panel
            # keeps its tissue axis; only single-output regression is squeezed.
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
    identity: dict | None = None
    test_truth: list | None = None
    test_prediction: list | None = None
    extra_fields: dict = field(default_factory=dict, repr=False)

    def save(self, path):
        # Keep unfamiliar fields from another writer at their original JSON keys.
        values = asdict(self)
        payload = values.pop("extra_fields")
        payload.update(values)
        path = Path(path)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)

    @staticmethod
    def load(path):
        # Older records may lack identity, predictions, or both. Loading them does not
        # establish reuse eligibility; run() checks that separately before any fitting.
        payload = json.loads(Path(path).read_text())
        known = {item.name for item in fields(Result)} - {"extra_fields"}
        return Result(**{key: value for key, value in payload.items() if key in known},
                      extra_fields={key: value for key, value in payload.items()
                                    if key not in known})


def run_identity(streams, variant, config, device, model_kwargs):
    """The requested computation, its exact inputs, and its current execution environment."""
    if not callable(getattr(streams, "identity", None)):
        raise ValueError("These streams do not provide a run identity. Their input and "
                         "split identity must be supported before checkpointed training.")
    package = Path(__file__).resolve().parent
    source_names = ("train.py", "blf.py", "codon_grid.py", "heads.py", "textcnn.py",
                    "token_embeddings.py", "../datasets.py", "../sequences.py", "../encoders.py")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
        "device": str(device), "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if str(device).startswith("cuda"):
        environment["cuda"] = torch.version.cuda
        environment["device_name"] = torch.cuda.get_device_name(device)
        environment["device_index"] = (torch.cuda.current_device() if str(device) == "cuda"
                                       else torch.device(device).index)
        environment["cuda_device_count"] = torch.cuda.device_count()
    identity = {
        "version": 1, "variant": variant, "config": asdict(config),
        "model_kwargs": model_kwargs, "widths": streams.widths, "n_outputs": streams.n_outputs,
        "inputs": streams.identity(), "environment": environment,
        "code_sha256": {name: file_fingerprint(package / name) for name in source_names},
    }
    # Require JSON-compatible settings and normalise tuples so a saved identity compares
    # equal after reloading. Unsupported values must fail before a training run starts.
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def _random_state(device, train_generator):
    """Capture randomness used by dropout, batch order, and the active accelerator."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "loader": train_generator.get_state()}
    if str(device).startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif str(device).startswith("mps"):
        if not hasattr(torch.mps, "get_rng_state") or not hasattr(torch.mps, "set_rng_state"):
            raise RuntimeError("This PyTorch version cannot checkpoint MPS randomness.")
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_random_state(state, train_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["loader"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _save_checkpoint(path, epoch, model, optimiser, schedule, train_generator,
                     device, best, history, elapsed, identity, complete=False):
    """Publish an epoch-boundary snapshot; a failed write leaves the prior one intact."""
    temporary = path.with_name(path.name + ".partial")
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimiser": optimiser.state_dict(), "schedule": schedule.state_dict(),
                "random": _random_state(device, train_generator), "best": best,
                "history": history, "elapsed": elapsed, "identity": identity,
                "complete": complete}, temporary)
    os.replace(temporary, path)


def fit(model, streams, config, device, log=print, checkpoint=None, checkpoint_every=10,
        identity=None, remove_checkpoint=True, loaders=None):
    """Train to the early stop, restore the best epoch, and score the test split.

    With a checkpoint path, the run is written there every checkpoint_every epochs and
    resumed only when its identity matches. run() retains the final checkpoint until the
    result is published. Direct callers own model initialisation; run() seeds before build.
    Custom loaders support direct, noncheckpointed fits only: their split identity and
    sampler recovery are not yet part of the checkpoint contract.
    """
    if config.max_epochs < 1 or checkpoint_every < 1:
        raise ValueError("max_epochs and checkpoint_every must be positive.")
    if checkpoint and loaders is not None:
        raise ValueError("Custom loaders do not yet support checkpoint recovery or split "
                         "identity. Use them only for an explicit noncheckpointed fit.")
    if checkpoint and identity is None:
        raise ValueError("Checkpointed fit requires an explicit run identity; use run().")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    task = streams.task
    splits = (loaders if loaders is not None else {
        name: loader(streams, name, batch_size=config.batch_size, seed=config.seed)
        for name in ("train", "val", "test")})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("loaders must provide exactly the train, val, and test splits.")
    train_generator = splits["train"].generator if loaders is None else None
    if checkpoint:
        _random_state(device, train_generator)  # Fail before training on unsupported devices.
    criterion = loss_function(task)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=config.plateau_patience)

    best = {"score": -float("inf"), "epoch": -1, "state": None}
    history, started, first_epoch, before = [], time.perf_counter(), 0, 0.0
    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint and checkpoint.exists():
        # These are trusted local checkpoints, not downloaded model files. CPU loading
        # keeps RNG byte tensors on CPU; optimiser.load_state_dict handles its parameters.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity or "schedule" not in saved or "random" not in saved:
            raise ValueError(f"{checkpoint} is legacy or incompatible; use a new results directory.")
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        schedule.load_state_dict(saved["schedule"])
        best, history = saved["best"], saved["history"]
        first_epoch, before = saved["epoch"] + 1, saved.get("elapsed", 0.0)
        if saved.get("complete"):
            first_epoch = config.max_epochs
        _restore_random_state(saved["random"], train_generator)
        if log:
            log(f"    resumed at epoch {first_epoch}, best {best['score']:.4f} @ {best['epoch']}")
    for epoch in range(first_epoch, config.max_epochs):
        model.train()
        running = 0.0
        for batch in splits["train"]:
            batch = batch.to(device)
            optimiser.zero_grad()
            output, aux = model(batch)
            target = (batch.labels if task in ("classification", "expression")
                      else batch.labels.view(-1, 1))
            loss = criterion(output, target)
            if config.entropy_lambda and "alpha" in aux:
                alpha = aux["alpha"].clamp_min(1e-12)
                loss = loss + config.entropy_lambda * (-(alpha * alpha.log()).sum(1).mean())
            loss.backward()
            optimiser.step()
            # len(labels), not len(codons): the same loop trains the codon-grid models
            # and the expression panel, and only one of those batches has codons.
            running += float(loss.detach()) * len(batch.labels)

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
            _save_checkpoint(checkpoint, epoch, model, optimiser, schedule, train_generator,
                             device, best, history, before + time.perf_counter() - started,
                             identity)

    if checkpoint:
        _save_checkpoint(checkpoint, history[-1]["epoch"], model, optimiser, schedule,
                         train_generator, device, best, history,
                         before + time.perf_counter() - started, identity, complete=True)
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
        identity=identity,
        alpha=None if alpha is None else dict(zip(MODALITIES, [float(a) for a in alpha])),
        # tolist handles a column of values and a panel of them alike; a classification
        # run stores the predicted class rather than its logits.
        test_truth=np.asarray(truth).tolist(),
        test_prediction=([int(v) for v in predicted.argmax(axis=1)]
                         if task == "classification" else np.asarray(predicted).tolist()))
    if remove_checkpoint and checkpoint and checkpoint.exists():
        checkpoint.unlink()
    return result


def run(streams, variant, config, device, results_dir, log=print,
        require_predictions=False, builder=None, loaders=None, tag=None, **model_kwargs):
    """Train one variant, reusing only an exactly matching recorded computation.

    Filenames are readable labels, not proof of identity. A legacy or mismatched file is
    preserved and rejected; select a new results directory to start a separate experiment.
    require_predictions rejects score-only records; it never authorizes replacing them.
    Custom builders and loaders are reserved until their code, inputs, splits, and sampler
    state can participate in the same identity and checkpoint checks.
    """
    if builder is not None or loaders is not None:
        raise ValueError("Custom builders/loaders are not yet supported by run(): their "
                         "identity and checkpoint recovery need validation before training.")
    results_dir = Path(results_dir)
    tag = tag or (f"{streams.key}__{variant.replace(':', '-')}__lambda{config.entropy_lambda}"
                  f"__ch{config.channels}__{config.tconv}__seed{config.seed}")
    path = results_dir / f"{tag}.json"
    done = Result.load(path) if path.exists() else None
    if done is not None and require_predictions and (
            done.test_truth is None or done.test_prediction is None):
        raise ValueError(f"{path} has no complete per-row predictions; preserve it and "
                         "explicitly plan any additional fit in a new results directory.")
    identity = run_identity(streams, variant, config, device, model_kwargs)
    if done is not None:
        if done.identity != identity:
            raise ValueError(f"{path} is legacy or incompatible with this computation; "
                             "preserve it and use a new results directory.")
        if log:
            log(f"  {tag}: already run, test {done.test_score:.4f}")
        return done
    results_dir.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"  {tag}: training")
    # Initial weights are part of the run. Seeding only inside fit() is too late.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = build(streams, variant, channels=config.channels, tconv=config.tconv,
                  **model_kwargs)
    checkpoint = results_dir / f"{tag}.checkpoint"
    result = fit(model, streams, config, device, log=log, checkpoint=checkpoint,
                 identity=identity, remove_checkpoint=False)
    result.save(path)
    if checkpoint.exists():
        checkpoint.unlink()
    if log:
        val = "undefined" if result.val_score is None else f"{result.val_score:.4f}"
        log(f"  {tag}: test {result.test_score:.4f} "
            f"(val {val} at epoch {result.best_epoch}, "
            f"{result.seconds / 60:.1f} min)")
    return result
