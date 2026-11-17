"""Explicit fitting and diagnostic entry points for controlled BLF-token experiments.

The notebook supplies the scientific panel, settings, verified token-cache records
and result destination. This module neither selects an experiment nor extracts an
embedding. Existing BLF fits remain untouched. Diagnostics are a separate explicit
best-weight inference call; absent diagnostics never trigger fitting.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
from torch import nn

from . import blf, controlled_models, intervention_ops, run_identity, train
from .codon_grid import CodonGrid, mask_of
from .heads import GatedMILFusion, TEMPERATURE_RANGE
from .textcnn import TextCNN


CONTRACT = "controlled-blf-interventions-v1"
DIAGNOSTIC_SCHEMA = "controlled-fusion-diagnostics-v2"
NORM_FIELDS = (
    "feature_norms", "contribution_norms", "anchor_norm",
    "raw_update_norms", "update_norms", "fused_norm",
)
IMPLEMENTATION_FILES = {
    "controlled_experiment.py", "controlled_models.py", "intervention_ops.py",
    "codon_grid.py", "heads.py", "textcnn.py", "train.py",
}
SEQUENCE_VARIANTS = {
    "sequence_softmax": ("mean", "softmax"),
    "sequence_sigmoid": ("mean", "sigmoid"),
    "sequence_meanmax_softmax": ("mean_max", "softmax"),
    "sequence_meanmax_sigmoid": ("mean_max", "sigmoid"),
}
ATTENTION_VARIANTS = {
    "attention_ordinary": "ordinary", "attention_skip": "skip",
    "attention_sequence_sigmoid": "sequence_sigmoid",
    "attention_position_sigmoid": "position_sigmoid",
}
JOINT_GATE_VARIANTS = ("joint_sequence_sigmoid", "position_sigmoid")
VARIANTS = (*SEQUENCE_VARIANTS, *JOINT_GATE_VARIANTS, "position_control",
            "position_residual_control", "anchor_only", "anchor_concat_control", *ATTENTION_VARIANTS)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_settings(variant, config, settings):
    """Require declared settings; legacy scorer width is inactive for joint linear gates."""
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported controlled variant: {variant!r}.")
    if config.tconv != "equation" or config.channels != 100:
        raise ValueError("This panel requires the equation codon grid and 100-channel TextCNN.")
    if (not np.isfinite(config.entropy_lambda) or config.entropy_lambda < 0
            or (variant != "sequence_softmax" and config.entropy_lambda != 0)):
        raise ValueError("Only sequence_softmax accepts the explicitly declared entropy penalty.")
    required = {"projection_width", "projection_activation", "projection_norm"}
    if variant == "anchor_only":
        required.add("anchor")
    elif variant == "anchor_concat_control":
        required.update(("anchor", "control_hidden_width"))
    elif variant in ("position_control", "position_residual_control"):
        required.add("control_bottleneck")
        if variant == "position_residual_control":
            required.update(("control_scale", "control_initialization"))
    else:
        required.add("scorer_width")
    if variant in SEQUENCE_VARIANTS:
        required.update(("shared_scorer", "temperature_range"))
    if variant in ATTENTION_VARIANTS:
        required.update(("anchor", "n_heads", "dropout", "update_norm"))
    if not isinstance(settings, dict) or set(settings) != required:
        supplied = set(settings) if isinstance(settings, dict) else set()
        raise ValueError(f"Settings differ: missing {sorted(required - supplied)}, "
                         f"unused {sorted(supplied - required)}.")
    made = _json(settings)
    for name in required & {"projection_width", "scorer_width", "control_bottleneck", "control_hidden_width", "n_heads"}:
        if type(made[name]) is not int or made[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if "shared_scorer" in made and type(made["shared_scorer"]) is not bool:
        raise ValueError("shared_scorer must be an explicit boolean.")
    if variant == "anchor_concat_control":
        expected = controlled_models.attention_control_budget(made["projection_width"], len(blf.MODALITIES))
        if made["control_hidden_width"] != expected["hidden_width"]:
            raise ValueError("Control hidden width must be derived from the joint attention budget.")
    if variant == "position_residual_control" and (
            made["control_scale"] != .5 or isinstance(made["control_scale"], bool)
            or made["control_initialization"] != "zero_output_random_input"):
        raise ValueError("The residual control requires scale 0.5 and zero_output_random_input initialization.")
    if variant == "sequence_softmax" and (
            made["projection_activation"] != "tanh" or made["projection_norm"] != "none"
            or made["temperature_range"] != list(TEMPERATURE_RANGE)):
        raise ValueError("sequence_softmax retains BLF's tanh projection, no projection norm, "
                         "and released temperature range; use a separately declared new variant.")
    return made


def gate_formula(variant):
    """Record the dependency changed by A4/A5, separately from inherited settings."""
    if variant == "position_sigmoid":
        return {"input": "all_projected_modalities", "scorer": "affine_sigmoid", "scorer_width_active": False}
    if variant == "joint_sequence_sigmoid":
        return {"input": "masked_mean_all_projected_modalities", "scorer": "affine_sigmoid",
                "scorer_width_active": False, "broadcast": "all_valid_positions"}
    if variant in ATTENTION_VARIANTS:
        return {"input": "anchor_all_auxiliary_updates", "scorer": "affine_sigmoid" if variant.endswith("sigmoid") else "unit_update",
                "scorer_width_active": False}
    return {"input": "per_modality_summary", "scorer": "gated_tanh_sigmoid", "scorer_width_active": True} if variant in SEQUENCE_VARIANTS else None


def residual_control_formula(settings):
    return {"version": "full_width_half_sum_affine_update_v1",
            "fixed_path": "scaled_sum_of_all_projected_modalities",
            "scale": settings["control_scale"], "update_rank": settings["control_bottleneck"],
            "update_initialization": settings["control_initialization"]}


def attention_control_formula(settings):
    return {"version": "anchor_skip_joint_local_gelu_v1", "input": "all_projected_modalities",
            "skip": "projected_anchor", "activation": "gelu",
            "budget": controlled_models.attention_control_budget(settings["projection_width"], len(blf.MODALITIES)),
            "initialization": "ordinary_linear", "gates": False, "attention": False}


def _fusion(widths, variant, settings):
    common = {name: settings[name] for name in
              ("projection_width", "projection_activation", "projection_norm")}
    if variant == "anchor_only":
        return controlled_models.ProjectedAnchor(widths, **common, anchor=settings["anchor"])
    if variant == "anchor_concat_control":
        return controlled_models.ResidualConcatFusion(
            widths, **common, anchor=settings["anchor"], hidden_width=settings["control_hidden_width"])
    if variant == "sequence_softmax":
        return GatedMILFusion(widths, out_width=settings["projection_width"],
                              attention_width=settings["scorer_width"],
                              shared_projection=settings["shared_scorer"])
    if variant in SEQUENCE_VARIANTS:
        summary, normalization = SEQUENCE_VARIANTS[variant]
        return controlled_models.SequenceGateFusion(
            widths, **common, scorer_width=settings["scorer_width"], summary=summary,
            normalization=normalization, shared_scorer=settings["shared_scorer"],
            temperature_range=settings["temperature_range"])
    if variant == "position_sigmoid":
        return controlled_models.PositionGateFusion(widths, **common, scorer_width=settings["scorer_width"])
    if variant == "joint_sequence_sigmoid":
        return controlled_models.JointSequenceGateFusion(widths, **common, scorer_width=settings["scorer_width"])
    if variant == "position_control":
        return controlled_models.PositionalControl(widths, **common, bottleneck_width=settings["control_bottleneck"])
    if variant == "position_residual_control":
        return controlled_models.ResidualPositionalControl(
            widths, **common, bottleneck_width=settings["control_bottleneck"],
            fixed_scale=settings["control_scale"], update_initialization=settings["control_initialization"])
    return controlled_models.AnchorAttentionFusion(
        widths, **common, mode=ATTENTION_VARIANTS[variant], **{
            name: settings[name] for name in ("anchor", "scorer_width", "n_heads", "dropout", "update_norm")})


class ControlledBLF(nn.Module):
    """Retained equation-grid preprocessing and TextCNN around one explicit new head."""

    required_modalities = blf.MODALITIES

    def __init__(self, streams, variant, config, settings):
        super().__init__()
        self.variant = variant
        self.required_modalities = (settings["anchor"],) if variant == "anchor_only" else blf.MODALITIES
        self.grid = CodonGrid(streams.widths["dna"], tconv=config.tconv, resolutions=streams.resolutions)
        self.fusion = _fusion(streams.widths, variant, settings)
        self.head = TextCNN(self.fusion.out_width, streams.n_outputs, channels=config.channels)

    def features(self, batch):
        if self.variant == "anchor_only":
            modality = self.required_modalities[0]
            target = int(batch.codons.max())
            mask = mask_of(batch.codons, target)
            placed = self.grid.place(modality, batch.streams[modality], target, batch.tokens[modality])
            grid = {modality: placed.masked_fill(~mask.unsqueeze(-1), 0)}
        else:
            grid, mask = self.grid(batch.streams["dna"], batch.streams["rna"], batch.streams["prot"],
                                   batch.codons, batch.tokens["dna"])
        if self.variant in ATTENTION_VARIANTS:
            fused, aux = self.fusion(grid, {m: mask for m in blf.MODALITIES})
        else:
            fused, aux = self.fusion(grid, mask)
        if self.variant == "sequence_softmax":
            aux = {**aux, "gates": aux["alpha"], "gate_kind": "sequence_softmax",
                   "gate_modalities": blf.MODALITIES}
            if getattr(self.fusion, "collect_norms", False):
                aux.update(_mil_norms(self.fusion, grid, mask, fused, aux["alpha"]))
        else:
            # The experiment contract exposes legacy entropy only for its baseline.
            aux = {name: value for name, value in aux.items() if name != "alpha"}
        return fused, mask, aux

    def forward(self, batch):
        fused, mask, aux = self.features(batch)
        return self.head(fused, mask), aux


def build_model(streams, variant, config, *, settings):
    settings = validate_settings(variant, config, settings)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("This controlled model contract requires float32 parameters.")
    return ControlledBLF(streams, variant, config, settings)


def make_identity(streams, variant, config, settings, cache_records):
    """Bind actual ordered rows, targets, splits, cache manifests and constructed settings."""
    settings = validate_settings(variant, config, settings)
    if set(streams.encoders) != set(blf.MODALITIES) or set(cache_records) != set(blf.MODALITIES):
        raise ValueError("This BLF-token panel requires DNA, RNA and protein streams.")
    streams.check()
    labels = streams.labels
    if not np.isfinite(labels).all() or not np.isin(streams.rows[streams.split_column], ("train", "val", "test")).all():
        raise ValueError("Every row needs a finite target and an explicit train/val/test assignment.")
    if any(not len(streams.split_rows(split)) for split in ("train", "val", "test")):
        raise ValueError("All three partitions must contain rows.")
    for modality, cache in streams.caches.items():
        if cache_records[modality].get("width") != cache.width:
            raise ValueError(f"{modality}: declared cache width differs from the opened cache.")
    base = run_identity.blf_identity(
        rows=streams.rows, benchmark=streams.key, variant="mil", config=config,
        encoders=streams.encoders, caches=cache_records, label_column=streams.label_column,
        split_column=streams.split_column, task=streams.task)
    # Construction validates every factory option; the surrounding RNG state is preserved.
    with torch.random.fork_rng(devices=[]):
        model = build_model(streams, variant, config, settings=settings)
        counts = {name: controlled_models.parameter_count(component) for name, component in
                  (("total", model), ("grid", model.grid), ("fusion", model.fusion), ("prediction", model.head))}
    from . import codon_grid, heads, textcnn
    sources = {Path(module.__file__).name: _sha(module.__file__) for module in
               (controlled_models, intervention_ops, codon_grid, heads, textcnn, train)}
    sources[Path(__file__).name] = _sha(__file__)
    model_record = {"contract": CONTRACT, "builder": f"{__name__}.build_model", "settings": settings,
                    "gate_formula": gate_formula(variant),
                    "widths": streams.widths, "n_outputs": streams.n_outputs, "parameter_counts": counts,
                    "modalities": list(model.required_modalities),
                    "implementation_sha256": sources, "textcnn": {"kernels": [3, 4, 5], "dropout": .2},
                    "numerics": {"parameters": "float32", "inputs": "float32", "amp": False}}
    if variant == "position_residual_control":
        model_record["control_formula"] = residual_control_formula(settings)
    if variant == "anchor_concat_control":
        model_record["control_formula"] = attention_control_formula(settings)
    loader = {**base["loader"], "kind": "custom", "implementation": "shared_code.fusion.blf.loader"}
    if variant == "anchor_only":
        loader["modalities"] = [settings["anchor"]]
    return run_identity.make_identity(benchmark=streams.key, variant=variant, config=config,
                                      inputs=base["inputs"], model=model_record, loader=loader)


def _loaders(streams, config, identity):
    loaders = {name: blf.loader(streams, name, batch_size=config.batch_size, seed=config.seed,
                               modalities=identity["loader"]["modalities"])
               for name in ("train", "val", "test")}
    # The generic trainer permits custom loaders; this entry point specifically requires
    # the existing BLF sampler, collator, row order, labels and RNG recovery contract.
    checked = {**identity, "loader": {**identity["loader"], "kind": "blf"}}
    run_identity.validate_loaders(checked, loaders)
    return loaders


def _tag(tag):
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("Use a plain result tag, without directories or special path characters.")
    return tag


def read_fit(path, *, expected_identity):
    """Read one complete compatible fit; absent results never start computation."""
    result = train.Result.load(path)
    identity = run_identity.validate_identity(result.extra_fields.get("identity"), config=result.config,
                                              benchmark=result.benchmark, variant=result.variant)
    if identity != expected_identity or identity["model"].get("contract") != CONTRACT:
        raise ValueError("Completed controlled fit has a different computation identity; preserve it.")
    if (type(result.epochs) is not int or result.epochs < 1 or type(result.best_epoch) is not int
            or not 0 <= result.best_epoch < result.epochs):
        raise ValueError("Completed fit has invalid epoch evidence.")
    if result.test_prediction is None or result.test_truth is None:
        raise ValueError("Completed fit lacks ordered predictions; it will not be replaced.")
    truth, prediction = np.asarray(result.test_truth), np.asarray(result.test_prediction)
    count = identity["inputs"]["splits"]["test"]["shape"][0]
    if (truth.shape != (count,) or prediction.shape != (count,)
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("Completed fit prediction dimensions or values are invalid.")
    dtype = np.int64 if identity["inputs"]["targets"]["task"] == "classification" else np.float32
    if run_identity.array_record(truth, dtype) != identity["inputs"]["split_targets"]["test"]:
        raise ValueError("Completed fit targets differ from the identified test rows.")
    return result


def read_panel_fit(path, *, benchmark, variant, config, encoders, settings):
    """Check a saved fit against the requested panel without caches or model construction.

    Recorded producer-source hashes are retained. They are not required to equal
    today's reader code; a comparison must separately require one consistent
    recorded implementation across its new fits.
    """
    settings = validate_settings(variant, config, settings)
    saved = train.Result.load(path)
    if (saved.variant != variant or saved.benchmark != benchmark
            or saved.config != asdict(config)):
        raise ValueError("Saved record differs from the declared panel variant, benchmark or configuration.")
    identity = run_identity.validate_identity(saved.extra_fields.get("identity"), config=config,
                                              variant=variant, benchmark=benchmark)
    inputs, model = identity["inputs"], identity["model"]
    modalities = [settings["anchor"]] if variant == "anchor_only" else list(blf.MODALITIES)
    if inputs.get("encoders") != encoders or set(encoders) != set(blf.MODALITIES):
        raise ValueError("Saved record belongs to a different declared encoder bundle.")
    if (model.get("contract") != CONTRACT or model.get("builder") != f"{__name__}.build_model"
            or model.get("settings") != settings or model.get("modalities") != modalities
            or ((variant in JOINT_GATE_VARIANTS or variant in ATTENTION_VARIANTS)
                and model.get("gate_formula") != gate_formula(variant))
            or (variant == "position_residual_control"
                and model.get("control_formula") != residual_control_formula(settings))
            or (variant == "anchor_concat_control"
                and model.get("control_formula") != attention_control_formula(settings))
            or model.get("textcnn") != {"kernels": [3, 4, 5], "dropout": .2}
            or model.get("numerics") != {"parameters": "float32", "inputs": "float32", "amp": False}):
        raise ValueError("Saved architecture, modalities or computation contract differs from the declared panel.")
    caches = inputs.get("caches", {})
    if (set(caches) != set(encoders)
            or any(caches[m].get("encoder") != encoders[m] for m in encoders)
            or model.get("widths") != {m: caches[m].get("width") for m in encoders}):
        raise ValueError("Saved model widths or cache declarations differ from the encoder bundle.")
    expected_loader = {**run_identity.blf_loader_spec(config, modalities), "kind": "custom",
                       "implementation": "shared_code.fusion.blf.loader"}
    if identity["loader"] != expected_loader:
        raise ValueError("Saved loader modalities, seed, batch size or recovery contract differ.")
    sources = model.get("implementation_sha256", {})
    if (not isinstance(sources, dict) or set(sources) != IMPLEMENTATION_FILES
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in sources.values())):
        raise ValueError("Saved implementation-source contract is missing or malformed.")
    return read_fit(path, expected_identity=identity)


def paired_intervals(records, *, rows, split_column, task, groups, seeds,
                     comparisons, draws, bootstrap_seed):
    """Group-paired test intervals, averaging fixed training seeds inside each draw.

    This explicit CPU analysis reads supplied predictions only. It validates the
    actual ordered cohort and score calculation before resampling exact-sequence
    groups. Intervals describe test-group sampling conditional on these fits;
    they do not replace the separately reported training-seed standard deviation.
    """
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or type(draws) is not int or draws < 2
            or type(bootstrap_seed) is not int or bootstrap_seed < 0):
        raise ValueError("Declare distinct seeds, at least two draws and a nonnegative bootstrap seed.")
    groups = np.asarray(groups)
    if groups.shape != (len(rows),):
        raise ValueError("Supply one exact-sequence group for every ordered input row.")
    labels = rows["Value"].to_numpy(dtype=np.int64 if task == "classification" else np.float32)
    splits = {name: np.flatnonzero(rows[split_column].to_numpy() == name)
              for name in ("train", "val", "test")}
    expected = run_identity._inputs(rows, ("sequence", "length"), labels, {}, {},
                                    splits, ("Value",), task)
    truth = labels[splits["test"]]
    if not len(truth) or not np.isfinite(truth).all():
        raise ValueError("The comparison needs finite, nonempty test targets.")
    predictions = {}
    needed = {name for _, first, second in comparisons for name in (first, second)}
    if not needed:
        raise ValueError("Declare at least one paired comparison.")
    for name in needed:
        runs = records.get(name, {})
        if set(runs) != set(seeds):
            raise ValueError(f"{name}: all declared seeds are required; no partial averaging.")
        predictions[name] = []
        for seed in seeds:
            result = runs[seed]
            if result.test_prediction is None or result.test_truth is None:
                raise ValueError(f"{name}: saved ordered predictions and targets are required.")
            inputs = result.extra_fields["identity"]["inputs"]
            if any(inputs[key] != expected[key] for key in ("rows", "targets", "splits", "split_targets")):
                raise ValueError(f"{name}: current cohort identity differs from its saved fit.")
            predicted = np.asarray(result.test_prediction)
            if (result.config.get("seed") != seed or predicted.shape != truth.shape
                    or not np.isfinite(predicted).all()
                    or not np.array_equal(np.asarray(result.test_truth), truth)):
                raise ValueError(f"{name}: predictions, targets or seed differ from the declared rows.")
            measured = train.score_rows(task, truth, predicted)
            if not np.isfinite(measured) or not np.isclose(measured, result.test_score, rtol=1e-7, atol=1e-8):
                raise ValueError(f"{name}: saved score differs from its predictions or is undefined.")
            predictions[name].append(predicted)
    intervals = train.seed_averaged_bootstrap(
        task, truth, predictions, comparisons, groups=groups[splits["test"]],
        draws=draws, seed=bootstrap_seed,
    )
    if any(not np.isfinite(row[key]) for row in intervals
           for key in ("difference", "ci_low", "ci_high", "first_wins")):
        raise ValueError("A bootstrap score was undefined; preserve fits and review the interval protocol.")
    return intervals


def run_fit(streams, variant, config, settings, cache_records, output_dir, device, producer, tag):
    """Fit or strictly reuse one requested variant; never overwrite completed evidence."""
    identity = make_identity(streams, variant, config, settings, cache_records)
    path = Path(output_dir) / f"{_tag(tag)}.json"
    if path.exists():
        return read_fit(path, expected_identity=identity)
    producer = _json(producer)
    if not isinstance(producer, dict) or not producer:
        raise ValueError("New fits require explicit producer metadata.")
    train.run(streams, variant, config, device, output_dir, require_predictions=True,
              builder=build_model, loaders=_loaders(streams, config, identity), tag=tag,
              run_identity=identity, producer=producer, settings=identity["model"]["settings"])
    return read_fit(path, expected_identity=identity)


def _diagnostic_paths(stem):
    stem = Path(stem)
    _tag(stem.name)
    return stem.with_name(stem.name + ".json"), stem.with_name(stem.name + ".npz")


def _mil_norms(fusion, grid, mask, fused, alpha):
    """Observe the retained BLF operation without changing its forward computation."""
    projected = torch.stack([fusion.project[m](grid[m]) for m in blf.MODALITIES], dim=2)
    features = projected.detach().norm(dim=-1).masked_fill(~mask.unsqueeze(-1), 0)
    return {"feature_norms": features, "contribution_norms": features * alpha.detach()[:, None],
            "fused_norm": fused.detach().norm(dim=-1), "temperature": fusion.temperature().detach()}


def read_diagnostics(output_stem, *, expected_identity=None):
    """Validate and read saved arrays without constructing a model or fitting."""
    metadata_path, arrays_path = _diagnostic_paths(output_stem)
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record.get("schema") not in ("controlled-fusion-diagnostics-v1", DIAGNOSTIC_SCHEMA):
        raise ValueError("Unsupported diagnostic record.")
    if expected_identity is not None and record.get("identity") != expected_identity:
        raise ValueError("Existing diagnostics describe different evidence; preserve them.")
    artifact = record["arrays"]
    if (artifact.get("file") != arrays_path.name or artifact.get("sha256") != _sha(arrays_path)
            or artifact.get("bytes") != arrays_path.stat().st_size):
        raise ValueError("Diagnostic array identity differs from its saved manifest.")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    rows = record["rows"]
    if (type(rows) is not int or rows < 1 or arrays["row_ids"].shape != (rows,)
            or arrays["row_ids"].dtype.kind not in "US"
            or len(set(arrays["row_ids"].tolist())) != rows):
        raise ValueError("Diagnostic row identities are incomplete or duplicated.")
    if (arrays["truth"].shape != (rows,) or arrays["prediction_outputs"].shape[0] != rows
            or arrays["query_lengths"].shape != (rows,)
            or arrays["query_lengths"].dtype.kind not in "iu" or (arrays["query_lengths"] < 1).any()):
        raise ValueError("Diagnostic prediction or query-length rows are invalid.")
    for prefix in ("gates", "update_weights"):
        if prefix not in arrays:
            continue
        lengths, values = arrays[prefix + "_lengths"], arrays[prefix]
        if (lengths.shape != (rows,) or lengths.dtype.kind not in "iu" or (lengths < 1).any()
                or values.shape != (int(lengths.sum()), len(record["gate_modalities"]))
                or ((values < 0) | (values > 1)).any()):
            raise ValueError(f"Invalid per-example {prefix} dimensions or values.")
        position_key = prefix + "_positions"
        if position_key in arrays:
            positions = arrays[position_key]
            if (positions.shape != (int(lengths.sum()),) or positions.dtype.kind not in "iu"
                    or not np.array_equal(lengths, arrays["query_lengths"])
                    or not np.array_equal(positions, np.concatenate([np.arange(n) for n in lengths]))):
                raise ValueError(f"Invalid codon-grid positions for {prefix}.")
        elif not np.array_equal(lengths, np.ones(rows, dtype=np.int64)):
            raise ValueError(f"Sequence {prefix} must contain one gate vector per row.")
    if record["schema"] == DIAGNOSTIC_SCHEMA:
        columns = record.get("norm_columns", {})
        if not columns or set(columns) - set(NORM_FIELDS) or "fused_norm" not in columns:
            raise ValueError("Version-2 diagnostics require declared contribution-norm fields.")
        lengths = arrays.get("norm_lengths")
        if not np.array_equal(lengths, arrays["query_lengths"]):
            raise ValueError("Norm lengths differ from real query positions.")
        positions = arrays.get("norm_positions")
        if not np.array_equal(positions, np.concatenate([np.arange(n) for n in lengths])):
            raise ValueError("Norm positions differ from the recorded codon grid.")
        for name, labels in columns.items():
            values = arrays.get(name)
            if (not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
                    or values is None or values.shape != (int(lengths.sum()), len(labels)) or (values < 0).any()):
                raise ValueError(f"Invalid nonnegative norm array: {name}.")
        temperature = record.get("temperature")
        if temperature is not None and (not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("Saved softmax temperature must be finite and positive.")
    for name, values in arrays.items():
        if values.dtype.kind in "fc" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite saved diagnostic array: {name}.")
    return record, arrays


def _packed_gates(batches, prefix):
    if not batches:
        return {}
    records = intervention_ops.collect_gate_records(batches)
    positional = records[0].positions is not None
    arrays = {prefix: np.concatenate([r.weights.numpy() if positional else r.weights.numpy()[None]
                                      for r in records]),
              prefix + "_lengths": np.asarray([len(r.weights) if positional else 1 for r in records], dtype=np.int64)}
    if positional:
        arrays[prefix + "_positions"] = np.concatenate([r.positions.numpy() for r in records])
    return arrays


def summarize_diagnostics(record, arrays):
    """Explicit CPU analysis of saved diagnostics, weighting transcripts equally.

    Within-row variation preserves the positional question instead of averaging it
    away. Norm ratios use each transcript's mean real-position norms; a zero
    denominator remains unavailable. These are descriptions, not importance tests.
    """
    summaries, variation, norm_summary, ratios, weight_mass = [], [], [], [], []

    def row_statistics(values, lengths):
        ends = np.cumsum(lengths)
        pieces = [values[start:end] for start, end in zip(np.r_[0, ends[:-1]], ends)]
        return (np.stack([part.mean(axis=0) for part in pieces]),
                np.stack([part.std(axis=0) for part in pieces]),
                np.stack([np.ptp(part, axis=0) for part in pieces]))

    def quartiles(values):
        low, middle, high = np.quantile(values, [.25, .5, .75])
        return {"q25": float(low), "median": float(middle), "q75": float(high)}

    for quantity in ("gates", "update_weights"):
        if quantity not in arrays:
            continue
        means, deviations, spans = row_statistics(arrays[quantity], arrays[quantity + "_lengths"])
        weight_mass.append({"quantity": quantity, "transcripts": len(means),
                            **quartiles(means.sum(axis=1))})
        for index, modality in enumerate(record["gate_modalities"]):
            base = {"quantity": quantity, "modality": modality, "transcripts": len(means)}
            summaries.append({**base, **quartiles(means[:, index])})
            if quantity + "_positions" in arrays:
                variation.append({**base, "within_row_sd": float(np.median(deviations[:, index])),
                                  "within_row_range": float(np.median(spans[:, index]))})
    norm_means = {}
    for quantity, labels in record.get("norm_columns", {}).items():
        means, _, _ = row_statistics(arrays[quantity], arrays["norm_lengths"])
        norm_means[quantity] = means
        for index, modality in enumerate(labels):
            norm_summary.append({"quantity": quantity, "modality": modality,
                                 "transcripts": len(means), **quartiles(means[:, index])})
    if "anchor_norm" in norm_means:
        denominator = norm_means["anchor_norm"][:, 0]
        for quantity in ("raw_update_norms", "update_norms"):
            if quantity not in norm_means:
                continue
            valid = denominator > 0
            for index, modality in enumerate(record["norm_columns"][quantity]):
                values = norm_means[quantity][valid, index] / denominator[valid]
                ratios.append({"quantity": quantity + " / anchor_norm", "modality": modality,
                               "transcripts": len(denominator), "undefined_rows": int((~valid).sum()),
                               **(quartiles(values) if len(values) else {"q25": None, "median": None, "q75": None})})
    return {"summary": summaries, "positional_variation": variation, "norm_summary": norm_summary,
            "update_anchor_ratios": ratios, "weight_mass": weight_mass,
            "temperature": record.get("temperature"),
            "norm_status": "recorded" if norm_means else "unavailable in legacy diagnostic record"}


def collect_diagnostics(streams, variant, config, settings, cache_records, *, result_path,
                        output_stem, device, producer, reference_result=None):
    """Explicit test inference from the checksum-verified best weights; never fit.

    Row IDs are benchmark plus ordered cache-row position. Position arrays retain
    codon-grid query indices, not genomic coordinates. Gate arrays remain per row
    or position; no test interpretation statistics or split averages are computed.
    """
    result_path = Path(result_path)
    if reference_result is None:
        identity = make_identity(streams, variant, config, settings, cache_records)
        result = read_fit(result_path, expected_identity=identity)
    else:
        # The reference adapter has checked current inputs. Keep the original BLF
        # identity and variant; never relabel it as a fit of our new wrapper.
        source = reference_result.extra_fields.get("reference_source", {})
        if (reference_result.variant != "mil" or not source.get("current_inputs_verified")
                or Path(source.get("path", "")).resolve() != result_path.resolve()
                or source.get("sha256") != _sha(result_path)):
            raise ValueError("MIL diagnostics require a current-input-verified reference source.")
        result = train.Result.load(result_path)
        identity = run_identity.blf_identity(
            rows=streams.rows, benchmark=streams.key, variant="mil", config=config,
            encoders=streams.encoders, caches=cache_records, task=streams.task,
            split_column=streams.split_column)
        if result.extra_fields.get("identity") != identity:
            raise ValueError("Reference MIL identity differs from the actual current inputs.")
        checked = validate_settings("sequence_softmax", config, settings)
        expected = identity["model"]["kwargs"]
        if (checked["projection_width"] != expected["out_width"]
                or checked["scorer_width"] != expected["attention_width"]
                or checked["shared_scorer"] != expected["shared_projection"]):
            raise ValueError("Requested diagnostic settings differ from the retained MIL.")
    artifact = result.extra_fields.get("model_artifact", {})
    filename = artifact.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Completed result has no safe best-weight artifact name.")
    weights_path = result_path.parent / filename
    if artifact.get("sha256") != _sha(weights_path) or artifact.get("size_bytes") != weights_path.stat().st_size:
        raise ValueError("Best weights are missing or differ from the completed result; no fit will start.")
    diagnostic_identity = {"fit_identity": identity, "fit_sha256": _sha(result_path),
                           "weights_sha256": artifact["sha256"], "split": "test",
                           "diagnostic_schema": DIAGNOSTIC_SCHEMA,
                           "observer_sha256": _sha(__file__)}
    metadata_path, arrays_path = _diagnostic_paths(output_stem)
    if metadata_path.exists() and arrays_path.exists():
        return (*read_diagnostics(output_stem, expected_identity=diagnostic_identity), True)
    if metadata_path.exists() or arrays_path.exists():
        raise FileExistsError("Partial diagnostic publication exists; preserve it before retrying.")
    producer = _json(producer)
    if not isinstance(producer, dict) or not producer:
        raise ValueError("Diagnostic inference requires its own producer metadata.")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if (payload.get("format") != "fusion-state-dict-v1" or payload.get("identity") != identity
            or payload.get("best_epoch") != result.best_epoch or payload.get("config") != result.config
            or payload.get("variant") != result.variant or payload.get("benchmark") != streams.key):
        raise ValueError("Best-weight payload does not match the completed fit.")
    with torch.random.fork_rng(devices=[]):
        model = (build_model(streams, variant, config, settings=settings) if reference_result is None
                 else blf.build(streams, "mil", **identity["model"]["kwargs"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    model.fusion.collect_norms = True  # Never enabled in the training path.
    batches = _loaders(streams, config, identity)["test"]
    row_ids = [f"{streams.key}:{int(index)}" for index in batches.dataset.indices]
    gates, updates, truth, predictions, lengths = [], [], [], [], []
    seen, modalities, gate_kind = 0, (), None
    norms, norm_columns, observed_temperatures = {}, {}, []
    with torch.inference_mode():
        for batch in batches:
            batch = batch.to(device)
            if reference_result is None:
                fused, mask, aux = model.features(batch)
            else:
                grid, mask = model.grid(batch.streams["dna"], batch.streams["rna"], batch.streams["prot"],
                                        batch.codons, batch.tokens["dna"])
                fused, aux = model.fusion(grid, mask)
                aux = {**aux, "gates": aux["alpha"], "gate_kind": "sequence_softmax",
                       "gate_modalities": blf.MODALITIES}
                aux.update(_mil_norms(model.fusion, grid, mask, fused, aux["alpha"]))
            predicted = model.head(fused, mask)
            ids = row_ids[seen:seen + len(batch.labels)]
            seen += len(ids)
            truth.append(batch.labels.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
            lengths.append(batch.codons.cpu().numpy())
            modalities = tuple(aux.get("gate_modalities", ()))
            gate_kind = aux.get("gate_kind")
            if "gates" in aux:
                gates.append((ids, aux["gates"].detach().cpu(),
                              mask.cpu() if aux["gates"].ndim == 3 else None))
            if "update_weights" in aux:
                updates.append((ids, aux["update_weights"].detach().cpu(), mask.cpu()))
            for name in NORM_FIELDS:
                if name not in aux:
                    continue
                values = aux[name].detach()
                if values.ndim == 2:
                    values = values.unsqueeze(-1)
                labels = ([settings["anchor"]] if name == "anchor_norm" else ["fused"] if name == "fused_norm"
                          else list(blf.MODALITIES) if name == "feature_norms" else list(modalities))
                if values.shape != (*mask.shape, len(labels)):
                    raise ValueError(f"Norm field has unexpected shape: {name}.")
                norm_columns[name] = labels
                norms.setdefault(name, []).append(values[mask].cpu().numpy())
            if "temperature" in aux:
                observed_temperatures.append(float(aux["temperature"]))
    arrays = {"row_ids": np.asarray(row_ids), "truth": np.concatenate(truth),
              "query_lengths": np.concatenate(lengths),
              "prediction_outputs": np.concatenate(predictions),
              **_packed_gates(gates, "gates"), **_packed_gates(updates, "update_weights")}
    if "fused_norm" not in norms or any(sum(len(a) for a in chunks) != int(arrays["query_lengths"].sum())
                                      for chunks in norms.values()):
        raise ValueError("Diagnostic norms do not cover every real query position.")
    arrays.update({name: np.concatenate(chunks) for name, chunks in norms.items()})
    arrays.update(norm_lengths=arrays["query_lengths"].copy(),
                  norm_positions=np.concatenate([np.arange(n) for n in arrays["query_lengths"]]))
    if observed_temperatures and any(value != observed_temperatures[0] for value in observed_temperatures):
        raise ValueError("Softmax temperature changed within diagnostic inference.")
    if not np.array_equal(arrays["truth"], np.asarray(result.test_truth, dtype=arrays["truth"].dtype)):
        raise ValueError("Diagnostic targets/order differ from the saved result.")
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fc"):
        raise ValueError("Nonfinite inference outputs; diagnostics were not published.")
    record = {"schema": DIAGNOSTIC_SCHEMA, "identity": diagnostic_identity,
              "rows": seen, "gate_kind": gate_kind, "gate_modalities": list(modalities),
              "norm_columns": norm_columns,
              "temperature": observed_temperatures[0] if observed_temperatures else None,
              "parameter_counts": {name: controlled_models.parameter_count(component) for name, component in
                                   (("total", model), ("grid", model.grid), ("fusion", model.fusion),
                                    ("prediction", model.head))},
              "producer": producer,
              "created_utc": datetime.now(timezone.utc).isoformat()}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=arrays_path.parent, suffix=".partial", delete=False) as handle:
            npz_temporary = Path(handle.name)
            temporary.append(npz_temporary)
            np.savez_compressed(handle, **arrays)
        record["arrays"] = {"file": arrays_path.name, "sha256": _sha(npz_temporary),
                             "bytes": npz_temporary.stat().st_size}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=metadata_path.parent,
                                          suffix=".partial", delete=False) as handle:
            json_temporary = Path(handle.name)
            temporary.append(json_temporary)
            json.dump(record, handle, indent=2, allow_nan=False)
        os.link(npz_temporary, arrays_path)
        os.link(json_temporary, metadata_path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return record, arrays, False
