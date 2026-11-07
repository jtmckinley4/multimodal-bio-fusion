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
one this matters for. New runs also retain the best-validation weights beside the result
JSON; existing results are reused without retraining to obtain missing weights.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from .blf import build, loader
from .heads import MODALITIES


@dataclass
class TrainConfig:
    """BioLangFusion's optimization settings; the defaults are the paper's."""

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

    Both runs need per-row predictions, which results written before 2026-10-18 do not have.
    """
    if first.test_prediction is None or second.test_prediction is None:
        raise ValueError("both runs need saved per-row predictions; preserve the completed "
                         "results and plan any prediction recovery separately")
    if first.test_truth != second.test_truth:
        raise ValueError("the two runs were not scored on the same rows")
    truth = np.asarray(first.test_truth)
    a, b = np.asarray(first.test_prediction), np.asarray(second.test_prediction)
    rng = np.random.default_rng(seed)
    spread = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(truth), len(truth))
        spread[i] = (score_rows(task, truth[pick], a[pick])
                     - score_rows(task, truth[pick], b[pick]))
    low, high = np.percentile(spread, [2.5, 97.5])
    return {"difference": score_rows(task, truth, a) - score_rows(task, truth, b),
            "ci_low": float(low), "ci_high": float(high),
            "first_wins": float((spread > 0).mean())}


def seed_averaged_bootstrap(task, truth, predictions, comparisons, rows=None, groups=None,
                            draws=4000, seed=0):
    """Paired differences between models on one test split, seeds averaged inside each draw.

    predictions maps a model name to a list of per-seed prediction arrays, every one scored
    on the rows of truth in the same order; comparisons is a list of (label, first, second)
    model names. Each draw resamples the test rows once and scores every model on that same
    resample, averaging its seeds inside the draw, so the interval measures the split rather
    than initialization and the comparison stays paired.

    rows restricts everything to a subset of positions, such as the test sequences never
    seen in training. groups gives a label per row, for example the sequence itself; whole
    groups are then resampled, so repeated measurements of one sequence move together rather
    than counting as independent evidence. Without groups each row is its own group, and with
    one seed per model this is exactly paired_bootstrap.
    """
    truth = np.asarray(truth)
    rows = np.arange(len(truth)) if rows is None else np.asarray(rows)
    needed = {name for _, first, second in comparisons for name in (first, second)}
    seeds_of = {}
    for name in needed:
        runs = [np.asarray(p) for p in predictions[name]]
        if not runs:
            raise ValueError(f"{name}: no predictions")
        for p in runs:
            if len(p) != len(truth):
                raise ValueError(f"{name}: {len(p)} predictions for {len(truth)} rows")
        seeds_of[name] = runs
    if groups is None:
        members = [rows[i:i + 1] for i in range(len(rows))]
    else:
        _, inverse = np.unique(np.asarray(groups)[rows], return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        members = np.split(rows[order], np.cumsum(np.bincount(inverse))[:-1])

    def averaged(name, pick):
        return float(np.mean([score_rows(task, truth[pick], p[pick]) for p in seeds_of[name]]))

    observed = {name: averaged(name, rows) for name in needed}
    rng = np.random.default_rng(seed)
    drawn = {name: np.empty(draws) for name in needed}
    for i in range(draws):
        pick = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for name in needed:
            drawn[name][i] = averaged(name, pick)
    results = []
    for label, first, second in comparisons:
        spread = drawn[first] - drawn[second]
        low, high = np.percentile(spread, [2.5, 97.5])
        results.append({"comparison": label, "difference": observed[first] - observed[second],
                        "ci_low": float(low), "ci_high": float(high),
                        "first_wins": float((spread > 0).mean()),
                        "rows": len(rows), "groups": len(members)})
    return results


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
    test_truth: list | None = None
    test_prediction: list | None = None
    # Keys another writer added (for example a run identity); kept so that reading and
    # rewriting a file never drops them, and never interpreted here.
    extra_fields: dict = field(default_factory=dict, repr=False)

    def save(self, path, *, overwrite=True):
        """Publish a complete record; training uses exclusive creation, not replacement."""
        values = asdict(self)
        payload = dict(values.pop("extra_fields"))
        payload.update(values)
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".partial",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(payload, indent=2) + "\n")
            if overwrite:
                os.replace(temporary, path)
            else:
                # Like weight publication, a same-directory link fails if a
                # concurrent writer has already published the destination.
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def load(path):
        payload = json.loads(Path(path).read_text())
        known = {f.name for f in fields(Result)} - {"extra_fields"}
        return Result(**{k: v for k, v in payload.items() if k in known},
                      extra_fields={k: v for k, v in payload.items() if k not in known})


def _run_identity(config, variant, benchmark):
    return {"config": asdict(config), "variant": variant, "benchmark": benchmark}


def _check_identity(saved, requested, source):
    mismatched = [name for name, value in requested.items() if saved.get(name) != value]
    if mismatched:
        raise ValueError(f"{source}: saved {', '.join(mismatched)} do not match the requested "
                         "run; preserve this file and use a different result tag or directory")


def checkpoint_preflight(path, requested_identity):
    """Inspect a trusted local checkpoint on CPU before expensive preparation.

    This checks recorded identity and recovery structure, without constructing a
    model, restoring RNG state, fitting, or writing. It does not establish missing
    producer/input provenance or identical recovery across devices. fit() still
    performs its own checks and loads state into the actual model and loaders.
    """
    path = Path(path)
    if isinstance(requested_identity, dict) and "schema" in requested_identity:
        from .run_identity import validate_identity
        requested_identity = validate_identity(requested_identity)
    saved = _read_checkpoint(path)
    _validate_checkpoint_payload(saved, requested_identity, path)


def _read_checkpoint(path):
    """Read trusted local recovery state on CPU without modifying process RNGs."""
    try:
        # Training checkpoints contain Python/NumPy RNG tuples: trusted local files only.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"{path}: checkpoint could not be read; preserve this file: {error}") from error


def _validate_adam_state(state, config, source, parameter_groups=None):
    """Check saved Adam slots; actual parameter dimensions are known only after build."""
    def require(condition, reason):
        if not condition:
            raise ValueError(f"{source}: checkpoint preflight failed: {reason}; preserve this file")

    groups = state["param_groups"]
    ids = [index for group in groups for index in group["params"]]
    require(len(ids) == len(set(ids)), "duplicate optimiser parameter IDs")
    require(bool(state["state"]), "post-epoch Adam moment state is empty")
    settings = {"weight_decay": config.get("weight_decay"), "betas": (0.9, 0.999),
                "eps": 1e-8, "amsgrad": False}
    for group in groups:
        require(all(group.get(key) == value for key, value in settings.items()),
                "Adam settings differ from the requested training configuration")
        require(all(not group.get(key, False) for key in
                    ("maximize", "capturable", "differentiable", "decoupled_weight_decay")),
                "unsupported Adam update settings")
    parameters = {}
    if parameter_groups is not None:
        require(len(groups) == len(parameter_groups), "optimiser parameter group count differs")
        for saved_group, group in zip(groups, parameter_groups):
            require(len(saved_group["params"]) == len(group["params"]),
                    "optimiser parameter group size differs")
            parameters.update(zip(saved_group["params"], group["params"]))
    for index, slot in state["state"].items():
        require(type(index) is int and index in ids and isinstance(slot, dict),
                "invalid Adam parameter slot")
        require({"step", "exp_avg", "exp_avg_sq"} <= slot.keys(), "incomplete Adam moment state")
        first, second, step = slot["exp_avg"], slot["exp_avg_sq"], slot["step"]
        require(isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
                and first.shape == second.shape and first.dtype == second.dtype,
                "Adam moment shapes or dtypes differ")
        if isinstance(step, torch.Tensor):
            require(step.numel() == 1, "invalid Adam step counter")
            step = step.item()
        require(isinstance(step, (int, float)) and not isinstance(step, bool)
                and np.isfinite(step) and step > 0 and step == int(step), "invalid Adam step counter")
        if index in parameters:
            parameter = parameters[index]
            require(first.shape == parameter.shape and first.dtype == parameter.dtype,
                    "Adam moments differ from the constructed model parameter")


def _validate_checkpoint_payload(saved, requested_identity, path):
    """Check the same recovery contract before preparation and again before restoration."""

    def require(condition, reason):
        if not condition:
            raise ValueError(f"{path}: checkpoint preflight failed: {reason}; preserve this file")

    def integer(value):
        return type(value) is int and value >= 0

    def byte_state(value):
        return (isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                and value.ndim == 1 and value.numel() > 0)

    try:
        required = {"identity", "scheduler", "rng", "loaders", "model", "optimiser",
                    "best", "history", "epoch", "elapsed"}
        require(isinstance(saved, dict), "payload must be a mapping")
        require(required <= saved.keys(), "missing fields: " + ", ".join(sorted(required - saved.keys())))
        require(isinstance(requested_identity, dict)
                and {"config", "benchmark", "variant"} <= requested_identity.keys(),
                "requested identity must include config, benchmark and variant")
        require(isinstance(requested_identity["config"], dict), "requested config must be a mapping")
        try:
            recorded = json.dumps(saved["identity"], sort_keys=True, allow_nan=False)
            requested = json.dumps(requested_identity, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid checkpoint/requested identity; preserve this file") from error
        require(recorded == requested, "recorded identity does not exactly match the requested run")
        for name in ("model", "optimiser", "scheduler", "rng", "loaders", "best"):
            require(isinstance(saved[name], dict), f"{name} must be a mapping")

        epoch, history, best = saved["epoch"], saved["history"], saved["best"]
        limit = requested_identity["config"].get("max_epochs")
        require(integer(epoch) and type(limit) is int and 0 <= epoch < limit,
                "epoch must be within the requested max_epochs")
        require(isinstance(history, list) and len(history) == epoch + 1,
                "history length must equal epoch + 1")
        require(all(isinstance(row, dict) and type(row.get("epoch")) is int
                    and row["epoch"] == index for index, row in enumerate(history)),
                "history must contain consecutive recorded epochs")
        elapsed = saved["elapsed"]
        require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                and np.isfinite(elapsed) and elapsed >= 0, "elapsed must be finite and nonnegative")
        require(integer(best.get("epoch")) and best["epoch"] <= epoch, "invalid best epoch")
        score = best.get("score")
        require(isinstance(score, (int, float)) and not isinstance(score, bool)
                and (np.isfinite(score) or score == -float("inf")), "invalid best score")
        state = best.get("state")
        require(isinstance(state, dict) and bool(state) and bool(saved["model"]),
                "current and best model states must be nonempty mappings")
        require(state.keys() == saved["model"].keys(), "current and best model keys differ")
        require(all(isinstance(name, str) and isinstance(value, torch.Tensor)
                    and isinstance(state[name], torch.Tensor)
                    and value.shape == state[name].shape and value.dtype == state[name].dtype
                    for name, value in saved["model"].items()), "invalid current/best model tensors")
        optimiser = saved["optimiser"]
        require(isinstance(optimiser.get("state"), dict)
                and isinstance(optimiser.get("param_groups"), list)
                and bool(optimiser["param_groups"]), "invalid optimiser state/parameter groups")
        require(all(isinstance(group, dict) and isinstance(group.get("params"), list)
                    and all(integer(value) for value in group["params"])
                    for group in optimiser["param_groups"]), "invalid optimiser parameter IDs")
        require(bool(saved["scheduler"]), "scheduler state is empty")

        rng = saved["rng"]
        require(byte_state(rng.get("torch")) and isinstance(rng.get("numpy"), tuple)
                and len(rng["numpy"]) == 5 and isinstance(rng.get("python"), tuple)
                and len(rng["python"]) == 3, "incomplete RNG state")
        require(rng.get("device_type") in ("cpu", "cuda", "mps"), "invalid RNG device type")
        if rng["device_type"] == "cuda":
            require(isinstance(rng.get("cuda"), list) and bool(rng["cuda"])
                    and all(byte_state(value) for value in rng["cuda"]), "missing CUDA RNG state")
        elif rng["device_type"] == "mps":
            require(byte_state(rng.get("mps")), "missing MPS RNG state")
        require(set(saved["loaders"]) == {"train", "val", "test"}, "missing or unexpected loader splits")
        for split, components in saved["loaders"].items():
            require(isinstance(components, dict), f"{split} loader components must be a mapping")
            for name, component in components.items():
                require(name in ("loader", "sampler", "batch_sampler", "batch_sampler.sampler")
                        and isinstance(component, dict) and isinstance(component.get("type"), str)
                        and bool(component["type"]), f"invalid {split} loader component")
                if "generator" in component:
                    require(byte_state(component["generator"]), f"invalid {split} shuffle generator")
                if "epoch" in component:
                    require(integer(component["epoch"]), f"invalid {split} sampler epoch")
        if "schema" in requested_identity:
            from .run_identity import validate_identity
            identity = validate_identity(requested_identity)
            _validate_adam_state(optimiser, identity["config"], path)
            schedule = saved["scheduler"]
            schedule_fields = {"factor", "min_lrs", "patience", "cooldown", "eps", "last_epoch",
                               "_last_lr", "mode_worse", "mode", "threshold", "threshold_mode",
                               "best", "cooldown_counter", "num_bad_epochs"}
            require(schedule_fields <= schedule.keys(), "incomplete scheduler recovery state")
            defaults = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters
            require("plateau_patience" in identity["config"],
                    "requested configuration lacks scheduler patience")
            expected_settings = {
                "mode": "max", "patience": identity["config"]["plateau_patience"],
                **{key: defaults[key].default for key in
                   ("factor", "cooldown", "eps", "threshold", "threshold_mode")},
            }
            require(all(schedule[key] == value for key, value in expected_settings.items()),
                    "scheduler settings differ from the requested training configuration")
            groups = optimiser["param_groups"]
            require(schedule["last_epoch"] == epoch + 1
                    and integer(schedule["cooldown_counter"])
                    and integer(schedule["num_bad_epochs"]), "invalid scheduler epoch/counters")
            require(schedule["min_lrs"] == [defaults["min_lr"].default] * len(groups)
                    and ("default_min_lr" not in schedule
                         or schedule["default_min_lr"] == defaults["min_lr"].default),
                    "scheduler minimum learning rates differ")
            require(schedule["_last_lr"] == [group.get("lr") for group in groups]
                    and all(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0
                            for value in schedule["_last_lr"]), "invalid scheduler learning rates")
            require(schedule["mode_worse"] == -float("inf")
                    and isinstance(schedule["best"], (int, float))
                    and (np.isfinite(schedule["best"]) or schedule["best"] == -float("inf")),
                    "invalid scheduler best score")
            # Private generators validate serialized CPU states without touching global RNGs.
            # Device-specific RNG payloads retain structural checks until actual restoration.
            try:
                torch.Generator(device="cpu").set_state(rng["torch"].cpu())
                np.random.RandomState().set_state(rng["numpy"])
                random.Random().setstate(rng["python"])
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"{path}: checkpoint preflight failed: invalid RNG recovery "
                                 "state; preserve this file") from error
            contract = identity["loader"]["recovery"]
            for split, expected in contract.items():
                actual = saved["loaders"][split]
                require(actual.keys() == expected.keys(),
                        f"{split} recovery components differ from the requested loader")
                for name, component in expected.items():
                    require(actual[name]["type"] == component["type"],
                            f"{split} {name} recovery type differs")
                    require(set(actual[name]) - {"type"} == set(component["fields"]),
                            f"{split} {name} recovery fields differ")
                    if "generator" in actual[name]:
                        try:
                            torch.Generator(device="cpu").set_state(actual[name]["generator"].cpu())
                        except (TypeError, ValueError, RuntimeError) as error:
                            raise ValueError(f"{path}: checkpoint preflight failed: invalid "
                                             f"{split} {name} generator state; preserve this file") from error
    finally:
        del saved


def _check_result_identity(result, requested, path, builder, model_kwargs):
    """Recognize the six historical GTEx modality labels without rewriting results.

    Those records say ``IsoFormer``; their budget300M/seed0 filenames identify the
    modalities. This narrow alias is based on the known notebook caller, not proof
    of missing historical code or input identity. Checkpoint checks stay strict.
    """
    saved = {"config": result.config, "variant": result.variant,
             "benchmark": result.benchmark}
    recorded = result.extra_fields.get("identity")
    if recorded is not None and not isinstance(recorded, dict):
        raise ValueError(f"{path}: completed result has a malformed computation identity; "
                         "preserve this file and inspect its provenance")
    if isinstance(recorded, dict) and "schema" in recorded and "schema" not in requested:
        raise ValueError(f"{path}: versioned completed result requires its full requested "
                         "computation identity; preserve this file or read it with Result.load")
    if "schema" in requested:
        from .run_identity import validate_identity
        if not isinstance(recorded, dict) or "schema" not in recorded:
            raise ValueError(f"{path}: completed result has no versioned computation identity; "
                             "preserve it as historical evidence and use a separate destination")
        recorded = validate_identity(recorded, config=result.config,
                                     variant=result.variant, benchmark=result.benchmark)
        if recorded != requested:
            raise ValueError(f"{path}: saved computation identity does not match the requested "
                             "run; preserve this file and use a different result tag or directory")
        return False
    legacy_alias = False
    if result.variant == "IsoFormer" and result.benchmark == "gtex_pilot":
        from . import isoformer

        variants = ("dna", "rna", "prot", "dna+prot", "dna+rna", "dna+rna+prot")
        variant = requested["variant"]
        legacy_settings = {"n_heads": 8, "n_layers": 1, "dropout": 0.1}
        defaults = inspect.signature(isoformer.IsoFormer).parameters
        effective_settings = {name: model_kwargs.get(name, defaults[name].default)
                              for name in legacy_settings}
        legacy_alias = (
            builder is isoformer.builder
            and variant in variants
            and requested["config"]["seed"] == 0
            and path.stem == f"gtex_pilot__{variant}__heads8__budget300M__seed0"
            and model_kwargs.keys() <= legacy_settings.keys()
            and effective_settings == legacy_settings)
        if legacy_alias:
            saved["variant"] = variant  # Normalize only the comparison, not the record.
    _check_identity(saved, requested, path)
    if legacy_alias:
        warnings.warn(
            f"{path}: interpreting legacy GTEx variant 'IsoFormer' as "
            f"'{requested['variant']}' from its historical filename and caller; "
            "the stored record is unchanged and historical producer identity remains limited",
            RuntimeWarning, stacklevel=2)
    return legacy_alias


def _rng_state(device):
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate(), "device_type": torch.device(device).type}
    if state["device_type"] == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif state["device_type"] == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    device_type = torch.device(device).type
    if state.get("device_type", "cpu") != device_type:
        warnings.warn("checkpoint device differs from the requested device; resumed numerical "
                      "and random trajectories may differ", RuntimeWarning, stacklevel=2)
    if device_type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    elif device_type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def _loader_components(batches):
    sampler = getattr(batches, "sampler", None)
    batch_sampler = getattr(batches, "batch_sampler", None)
    return {"loader": batches, "sampler": sampler, "batch_sampler": batch_sampler,
            "batch_sampler.sampler": getattr(batch_sampler, "sampler", None)}


def _loader_states(splits):
    """Keep shuffle generators, epoch samplers, and custom state_dict protocols."""
    states = {}
    for split, batches in splits.items():
        states[split] = {}
        for name, component in _loader_components(batches).items():
            state = {}
            generator = getattr(component, "generator", None)
            if isinstance(generator, torch.Generator):
                state["generator"] = generator.get_state()
            if (callable(getattr(component, "state_dict", None))
                    and callable(getattr(component, "load_state_dict", None))):
                state["state_dict"] = component.state_dict()
            # IsoFormer's budgeted batch sampler advances this counter each epoch.
            if isinstance(getattr(component, "epoch", None), int):
                state["epoch"] = component.epoch
            if state:
                state["type"] = f"{type(component).__module__}.{type(component).__qualname__}"
                states[split][name] = state
    return states


def _restore_loaders(states, splits):
    for split, saved_components in states.items():
        components = _loader_components(splits[split])
        for name, state in saved_components.items():
            component = components.get(name)
            kind = f"{type(component).__module__}.{type(component).__qualname__}"
            if kind != state["type"]:
                raise ValueError(f"checkpoint {split} {name} differs from the requested loader")
            if "state_dict" in state:
                restore = getattr(component, "load_state_dict", None)
                if not callable(restore):
                    raise ValueError(f"checkpoint {split} {name} requires load_state_dict()")
                restore(state["state_dict"])
            if "generator" in state:
                generator = getattr(component, "generator", None)
                if not isinstance(generator, torch.Generator):
                    raise ValueError(f"checkpoint {split} {name} requires a shuffle generator")
                generator.set_state(state["generator"].cpu())
            if "epoch" in state:
                component.epoch = state["epoch"]


def _save_checkpoint(path, epoch, model, optimiser, best, history, elapsed, *,
                     scheduler=None, splits=None, identity=None, device="cpu", producer=None):
    """Write the run's state atomically, so an interrupted write cannot be resumed from."""
    path = Path(path)
    payload = {"epoch": epoch, "model": model.state_dict(),
               "optimiser": optimiser.state_dict(), "best": best, "history": history,
               "elapsed": elapsed}
    # Keep the old positional helper usable by callers that build legacy checkpoints.
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if splits is not None:
        payload["loaders"] = _loader_states(splits)
    if identity is not None:
        payload["identity"] = identity
    if producer is not None:
        payload["producer"] = producer
    payload["rng"] = _rng_state(device)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_weights(path, model, result):
    """Keep the best state restored by fit(), without replacing existing weights.

    Rebuild the matching model before loading state_dict. This is an inference artifact,
    not an optimizer checkpoint; the result JSON links its filename, checksum and size.
    """
    payload = {"format": "fusion-state-dict-v1", "benchmark": result.benchmark,
               "variant": result.variant, "config": result.config,
               "best_epoch": result.best_epoch,
               "state_dict": {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}}
    for key in ("identity", "producer"):
        if key in result.extra_fields:
            payload[key] = result.extra_fields[key]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        # A same-directory hard link publishes the complete file and fails if path exists.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"version": 1, "format": payload["format"], "file": path.name,
            "sha256": checksum, "size_bytes": path.stat().st_size}


def fit(model, streams, config, device, log=print, checkpoint=None, checkpoint_every=10,
        loaders=None, keep_checkpoint=False, run_identity=None, producer=None):
    """Train to the early stop, restore the best epoch, and score the test split.

    With a checkpoint path, the run is written there every checkpoint_every epochs and
    resumed from it if it already exists. run() keeps it until weights and JSON are saved;
    direct callers remove it on success unless keep_checkpoint=True. New checkpoints retain
    scheduler, RNG and loader shuffle state at epoch boundaries. Custom stateful loaders or
    samplers can expose state_dict/load_state_dict; persistent worker RNG is not recoverable.
    Direct legacy callers without a versioned identity retain warned best-effort recovery.
    Declared versioned runs require the complete recovery contract before restoration.
    """
    identity = run_identity or _run_identity(
        config, getattr(model, "fusion_name", type(model).__name__), streams.key)
    strict = "schema" in identity
    if strict:
        from .run_identity import validate_identity, validate_loaders
        identity = validate_identity(identity, config=asdict(config), benchmark=streams.key)
    checkpoint = Path(checkpoint) if checkpoint else None
    saved = _read_checkpoint(checkpoint) if checkpoint and checkpoint.exists() else None
    if saved is not None:
        # run() always supplies an identity, so its preflight is repeated on the
        # payload actually restored. Only unspecified direct callers retain legacy recovery.
        if run_identity is not None:
            _validate_checkpoint_payload(saved, identity, checkpoint)
        elif "identity" in saved:
            if isinstance(saved["identity"], dict) and "schema" in saved["identity"]:
                raise ValueError(f"{checkpoint}: a versioned checkpoint requires its full "
                                 "requested computation identity; preserve this file")
            _check_identity(saved["identity"], identity, checkpoint)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    task = streams.task
    splits = loaders or {
        name: loader(streams, name, batch_size=config.batch_size, seed=config.seed,
                     modalities=getattr(model, "required_modalities", None))
        for name in ("train", "val", "test")}
    if strict:
        validate_loaders(identity, splits)
    criterion = loss_function(task)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=config.plateau_patience)

    model.to(device)
    if strict and saved is not None:
        _validate_adam_state(saved["optimiser"], identity["config"], checkpoint,
                             parameter_groups=optimiser.param_groups)
    best = {"score": -float("inf"), "epoch": -1, "state": None}
    history, started, first_epoch, before = [], time.perf_counter(), 0, 0.0
    if checkpoint and any(getattr(batches, "persistent_workers", False)
                          for batches in splits.values()):
        warnings.warn("persistent DataLoader worker RNG is not checkpointed; resumed "
                      "trajectories may differ", RuntimeWarning, stacklevel=2)
    if saved is not None:
        missing = [key for key in ("identity", "scheduler", "rng", "loaders") if key not in saved]
        if missing:
            warnings.warn(f"{checkpoint}: legacy checkpoint lacks {', '.join(missing)}; "
                          "resuming with best-effort recovery, so the training trajectory "
                          "may differ from uninterrupted training", RuntimeWarning, stacklevel=2)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        if "scheduler" in saved:
            schedule.load_state_dict(saved["scheduler"])
        if "loaders" in saved:
            _restore_loaders(saved["loaders"], splits)
        if "rng" in saved:
            _restore_rng(saved["rng"], device)
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
            _save_checkpoint(checkpoint, epoch, model, optimiser, best, history,
                             before + time.perf_counter() - started, scheduler=schedule,
                             splits=splits, identity=identity, device=device, producer=producer)

    model.load_state_dict(best["state"])
    truth, predicted, alpha = predict(model, splits["test"], device, task)
    result = Result(
        variant=identity["variant"],
        benchmark=streams.key, config=asdict(config), epochs=len(history),
        best_epoch=best["epoch"],
        val_score=best["score"] if np.isfinite(best["score"]) else None,
        test_score=score(task, truth, predicted),
        seconds=before + time.perf_counter() - started,
        history=history,
        alpha=None if alpha is None else dict(zip(MODALITIES, [float(a) for a in alpha])),
        # tolist handles a column of values and a panel of them alike; a classification
        # run stores the predicted class rather than its logits.
        test_truth=np.asarray(truth).tolist(),
        test_prediction=([int(v) for v in predicted.argmax(axis=1)]
                         if task == "classification" else np.asarray(predicted).tolist()))
    result.extra_fields["identity"] = identity
    if producer is not None:
        result.extra_fields["producer"] = producer
    if checkpoint and checkpoint.exists() and not keep_checkpoint:
        checkpoint.unlink()
    return result


def run(streams, variant, config, device, results_dir, log=print,
        require_predictions=False, builder=None, loaders=None, tag=None,
        run_identity=None, producer=None, **model_kwargs):
    """Train one variant, or return the run already on disk under the same settings.

    require_predictions refuses a stored run without per-row predictions; it never
    replaces completed evidence to supply an analysis with missing inputs.
    New fits save their best weights; missing historical weights never trigger retraining.
    An explicit versioned identity enables strict computation matching. Producer metadata
    records execution context separately and does not decide whether a result matches.
    """
    results_dir = Path(results_dir)
    tag = tag or (f"{streams.key}__{variant.replace(':', '-')}__lambda{config.entropy_lambda}"
                  f"__ch{config.channels}__{config.tconv}__seed{config.seed}")
    path = results_dir / f"{tag}.json"
    if run_identity is None:
        identity = _run_identity(config, variant, streams.key)
    else:
        from .run_identity import validate_identity, validate_model_request
        identity = validate_identity(run_identity, config=asdict(config),
                                     variant=variant, benchmark=streams.key)
        validate_model_request(identity, builder=builder, model_kwargs=model_kwargs)
    if producer is not None:
        if not isinstance(producer, dict):
            raise ValueError("producer must be a mapping")
        producer = json.loads(json.dumps(producer, sort_keys=True, allow_nan=False))
    if path.exists():
        done = Result.load(path)
        legacy_alias = _check_result_identity(done, identity, path, builder, model_kwargs)
        if require_predictions and (done.test_prediction is None or done.test_truth is None):
            kind = "legacy GTEx result" if legacy_alias else "completed result"
            raise ValueError(f"{path}: {kind} has no per-row predictions or targets; "
                             "preserve it and plan any recovery in a separate destination")
        if log:
            log(f"  {tag}: already run, test {done.test_score:.4f}")
        return done
    weights_path = results_dir / f"{tag}.weights.pt"
    checkpoint = results_dir / f"{tag}.checkpoint"
    if weights_path.exists():
        raise FileExistsError(f"{weights_path}: weights already exist without a reusable result; "
                              "preserve them and inspect the interrupted publication before retrying")
    if checkpoint.exists():
        checkpoint_preflight(checkpoint, identity)
    results_dir.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"  {tag}: training")
    # Seed before building, so the seed fixes the initial weights as well as the batch order
    # and dropout. Before 2026-10-19 fit() seeded only after the model existed, so a run's
    # initial weights were not determined by its seed.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model = (builder(streams, variant, config, **model_kwargs) if builder
             else build(streams, variant, channels=config.channels, tconv=config.tconv,
                        **model_kwargs))
    result = fit(model, streams, config, device, log=log, loaders=loaders,
                 checkpoint=checkpoint, keep_checkpoint=True, run_identity=identity,
                 producer=producer)
    result.extra_fields["identity"] = identity
    if producer is not None:
        result.extra_fields["producer"] = producer
    result.extra_fields["model_artifact"] = _save_weights(weights_path, model, result)
    result.save(path, overwrite=False)  # Never replace a concurrently completed result.
    checkpoint.unlink(missing_ok=True)
    if log:
        val = "undefined" if result.val_score is None else f"{result.val_score:.4f}"
        log(f"  {tag}: test {result.test_score:.4f} "
            f"(val {val} at epoch {result.best_epoch}, "
            f"{result.seconds / 60:.1f} min)")
    return result
