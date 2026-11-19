"""Explicit grouped OOF and fixed-anchor regression correction execution.

Callers own benchmarks, anchor selection, fold settings, tuning grids and run budgets.
This module accepts already-open frozen token caches. Nothing executes at import time;
the run functions fit only when called and preserve incompatible/completed artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import anchor_correction as evidence
from . import blf, run_identity, train
from .codon_grid import CodonGrid, mask_of
from .textcnn import TextCNN
from .token_embeddings import inputs_for_rows
from ..encoders import ENCODERS


CONTRACT = "fixed-anchor-regression-correction-v1"
ROW_SHUFFLE = "row_shuffle_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _seed(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} requires an explicit nonnegative integer.")
    return int(value)


def _row_positions(values, count, name):
    positions = evidence._positions(values, name)
    if any(position >= count for position in positions):
        raise ValueError(f"{name} contains a position outside the row table.")
    return np.asarray(positions, dtype=np.int64)


def _splits(split_positions, count):
    if set(split_positions) != {"train", "val", "test"}:
        raise ValueError("Supply explicit nonempty train, val and test positions.")
    splits = {name: _row_positions(values, count, name) for name, values in split_positions.items()}
    seen = set()
    for name, positions in splits.items():
        if seen & set(positions):
            raise ValueError(f"{name} overlaps another split by row position.")
        seen.update(positions)
    return splits


def fold_plan(train_positions, groups, *, n_splits, seed, stop_fraction, inner_seed):
    """Hash whole groups into caller-requested folds and inner stopping subsets.

    Hash thresholds give approximate inner proportions. Empty roles fail explicitly;
    no fold count, fraction or seed is searched to repair an inconvenient assignment.
    """
    groups = evidence._labels(groups, "groups")
    positions = _row_positions(train_positions, len(groups), "training positions")
    seed, inner_seed = _seed(seed, "fold seed"), _seed(inner_seed, "inner seed")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an explicit integer of at least two.")
    if isinstance(stop_fraction, bool) or not isinstance(stop_fraction, (int, float)) or not 0 < stop_fraction < 1:
        raise ValueError("stop_fraction must lie strictly between zero and one.")
    assignment = {group: int(_digest([seed, "oof", group]), 16) % int(n_splits)
                  for group in {groups[index] for index in positions}}
    plans = []
    for fold in range(int(n_splits)):
        predict = tuple(int(index) for index in positions if assignment[groups[index]] == fold)
        remaining = [int(index) for index in positions if assignment[groups[index]] != fold]
        stopping_groups = {
            groups[index] for index in remaining
            if int(_digest([inner_seed, "stop", fold, groups[index]]), 16) % 1_000_000
            < stop_fraction * 1_000_000
        }
        stop = tuple(index for index in remaining if groups[index] in stopping_groups)
        fit = tuple(index for index in remaining if groups[index] not in stopping_groups)
        if not stop:
            raise ValueError(f"Fold {fold} has no inner stopping rows; review the declared protocol.")
        plans.append(evidence.FoldRows(fit, stop, predict))
    return tuple(plans)


def group_permutation(groups, positions, *, strata_by_group, seed):
    """Historical whole-group control; retain one donor for joint inputs.

    Each stratum is a random single cycle, with no unchanged group. Every receiver
    observation uses its donor group's first supplied row. This requires identical
    auxiliary inputs within a declared group. Donor lengths travel with features;
    receiver targets and fixed-anchor predictions do not move.
    """
    groups = evidence._labels(groups, "groups")
    positions = _row_positions(positions, len(groups), "permutation positions")
    seed = _seed(seed, "permutation seed")
    members = {}
    for index in positions:
        members.setdefault(groups[index], []).append(int(index))
    if set(strata_by_group) != set(members):
        raise ValueError("Provide exactly one explicit stratum for each participating group.")
    strata = {}
    for group, stratum in strata_by_group.items():
        strata.setdefault(_json(stratum), []).append(group)
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(groups), dtype=np.int64)
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=_json)
        if len(ordered) < 2:
            raise ValueError("A permutation stratum has fewer than two groups; no silent identity shuffle.")
        cycle = [ordered[index] for index in rng.permutation(len(ordered))]
        for receiver, donor in zip(cycle, cycle[1:] + cycle[:1]):
            mapping[members[receiver]] = min(members[donor])
    return mapping


def row_permutation(groups, positions, *, strata_by_group, seed):
    """Use every donor row once, avoiding its receiver's exact-sequence group.

    Each explicit stratum must have no group larger than half its rows. Randomized
    contiguous group blocks rotated by the largest block guarantee this constraint
    without an N-by-N matching matrix. This is a seeded construction, not uniform
    sampling over all valid permutations. Repeated receivers can get different
    donors; joint auxiliary streams must use this one mapping together.
    """
    groups = evidence._labels(groups, "groups")
    positions = _row_positions(positions, len(groups), "permutation positions")
    seed = _seed(seed, "permutation seed")
    members = {}
    for index in positions:
        members.setdefault(groups[index], []).append(int(index))
    if set(strata_by_group) != set(members):
        raise ValueError("Provide exactly one explicit stratum for each participating group.")
    strata = {}
    for group, stratum in strata_by_group.items():
        strata.setdefault(_json(stratum), []).append(group)
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(groups), dtype=np.int64)
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=_json)
        largest = max(len(members[group]) for group in ordered)
        count = sum(len(members[group]) for group in ordered)
        if 2 * largest > count:
            raise ValueError(
                f"Row shuffle is infeasible in stratum {stratum}: largest group has "
                f"{largest}/{count} rows; no rows were dropped or identity donors substituted."
            )
        blocks = [rng.permutation(sorted(members[ordered[index]]))
                  for index in rng.permutation(len(ordered))]
        receivers = np.concatenate(blocks)
        mapping[receivers] = np.roll(receivers, -largest)
    return mapping


def training_random_statistics(streams, train_positions, modalities):
    """Feature-wise token mean/scale from permitted training rows only.

    Repeated observations retain their training multiplicity. Constant dimensions
    remain constant rather than receiving an invented variance floor.
    """
    positions = _row_positions(train_positions, len(streams.rows), "random-control training rows")
    result = {}
    for modality in modalities:
        count, mean, centered = 0, None, None
        for index in positions:
            values = np.asarray(streams.caches[modality][int(index)], dtype=np.float64)
            if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
                raise ValueError("Random-control statistics require finite nonempty token matrices.")
            row_mean = values.mean(axis=0)
            row_centered = ((values - row_mean) ** 2).sum(axis=0)
            if mean is None:
                count, mean, centered = len(values), row_mean, row_centered
                continue
            delta = row_mean - mean
            combined = count + len(values)
            centered += row_centered + delta ** 2 * count * len(values) / combined
            mean += delta * len(values) / combined
            count = combined
        result[modality] = {
            "mean": mean.tolist(), "scale": np.sqrt(np.maximum(centered / count, 0)).tolist(),
            "tokens": count, "training_positions": positions.tolist(),
        }
    return result


def _source_hashes():
    files = [Path(__file__), Path(evidence.__file__), Path(blf.__file__), Path(train.__file__),
             Path(run_identity.__file__), Path(__file__).with_name("codon_grid.py"),
             Path(__file__).with_name("textcnn.py")]
    return {path.name: _sha(path) for path in files}


def _offset_sources(input_contract):
    sources = input_contract.get("offset_sources", {})
    anchor = sources.get("final_anchor_sha256")
    folds = sources.get("oof_artifact_sha256")
    hashes = [anchor, *(folds if isinstance(folds, list) else [])]
    if (not isinstance(folds, list) or not folds or any(
        not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef")
        for value in hashes
    )):
        raise ValueError("Correction identity requires final-anchor and OOF-artifact SHA-256 bindings.")


def _control_record(control):
    """Keep reproducible mappings without duplicating full sequence strings per fit."""
    recorded = json.loads(_json(control))
    if recorded["kind"] in ("shuffle", ROW_SHUFFLE):
        recorded["strata"] = {
            split: {_digest(group): stratum for group, stratum in strata.items()}
            for split, strata in recorded["strata"].items()
        }
    return recorded


def _publish(path, record):
    """Publish a complete new record; never replace an existing result."""
    path = Path(path)
    encoded = (_json(record) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".correction-", suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FitArtifact:
    path: Path
    record: dict

    @property
    def sha256(self):
        return _sha(self.path)

    def prediction(self, split):
        values = self.record["predictions"][split]
        return evidence.Predictions("regression", tuple(values["row_ids"]), values["values"])


def _read_artifact(path, identity, targets, row_ids, splits, *, ridge_inputs=None):
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("identity") != identity or record.get("contract") != CONTRACT:
        raise ValueError(f"{path}: incompatible completed correction evidence; preserve it.")
    neural = "schema" in identity
    if record.get("kind") != ("neural" if neural else "ridge"):
        raise ValueError("Completed evidence has a different predictor family.")
    artifacts = record.get("artifacts", {})
    if neural and (not isinstance(artifacts, dict) or set(artifacts) != {"fit", "weights"}):
        raise ValueError("Completed neural evidence requires its fit and weight artifacts.")
    if set(record.get("predictions", {})) != {"val", "test"}:
        raise ValueError("Completed evidence is missing validation/test predictions.")
    for split in ("val", "test"):
        entry = record["predictions"][split]
        expected_ids = [row_ids[index] for index in splits[split]]
        if (entry.get("positions") != splits[split].tolist() or entry.get("row_ids") != expected_ids
                or entry.get("truth") != targets[splits[split]].tolist()):
            raise ValueError("Completed prediction rows or targets differ from the requested inputs.")
        evidence.Predictions("regression", tuple(expected_ids), entry.get("values"))
    for name, descriptor in artifacts.items():
        filename = descriptor["file"]
        if Path(filename).name != filename or _sha(path.parent / filename) != descriptor["sha256"]:
            raise ValueError(f"Completed {name} artifact is missing or has changed; preserve the record.")
    if neural:
        fit = json.loads((path.parent / artifacts["fit"]["file"]).read_text(encoding="utf-8"))
        epochs, best_epoch = fit.get("epochs"), fit.get("best_epoch")
        if (fit.get("identity") != identity or type(epochs) is not int or epochs <= 0
                or type(best_epoch) is not int or not 0 <= best_epoch < epochs
                or fit.get("test_prediction") != record["predictions"]["test"]["values"]
                or fit.get("test_truth") != record["predictions"]["test"]["truth"]
                or fit.get("model_artifact", {}).get("sha256") != artifacts["weights"]["sha256"]):
            raise ValueError("Completed neural summary differs from its fitted model evidence.")
    elif ridge_inputs is not None:
        features, offsets = ridge_inputs
        fitted = record.get("fitted", {})
        mean, scale, coefficient = (np.asarray(fitted.get(key), dtype=float) for key in (
            "scaler_mean", "scaler_scale", "coefficient"))
        intercept = fitted.get("intercept")
        width = features.shape[1]
        if (any(value.shape != (width,) or not np.isfinite(value).all() for value in (mean, scale, coefficient))
                or (scale <= 0).any() or type(intercept) not in (int, float) or not np.isfinite(intercept)):
            raise ValueError("Completed ridge coefficients/scaler are malformed.")
        candidates = record.get("candidates", [])
        if ([item.get("alpha") for item in candidates] != identity["alphas"]
                or any(type(item.get("validation_mse")) not in (int, float)
                       or not np.isfinite(item["validation_mse"]) for item in candidates)):
            raise ValueError("Completed ridge candidate evidence differs from its declared alpha grid.")
        winner = min(candidates, key=lambda item: item["validation_mse"])
        if (record.get("selected_alpha") != winner["alpha"]
                or record.get("validation_mse") != winner["validation_mse"]):
            raise ValueError("Completed ridge selection does not follow validation MSE.")
        for split in ("val", "test"):
            positions = splits[split]
            predicted = offsets[positions] + ((features[positions] - mean) / scale) @ coefficient + intercept
            if not np.allclose(predicted, record["predictions"][split]["values"], rtol=1e-12, atol=1e-12):
                raise ValueError("Completed ridge predictions differ from its fitted coefficients and anchor.")
    return FitArtifact(path, record)


def _prediction_record(values, positions, targets, row_ids):
    record = {
        "positions": positions.tolist(), "row_ids": [row_ids[index] for index in positions],
        "truth": targets[positions].tolist(), "values": np.asarray(values, dtype=float).tolist(),
    }
    evidence.Predictions("regression", tuple(record["row_ids"]), record["values"])
    return record


@dataclass
class CorrectionBatch(blf.GridBatch):
    offsets: object

    def to(self, device):
        base = super().to(device)
        return CorrectionBatch(base.streams, base.tokens, base.codons, base.labels, self.offsets.to(device))


def _collate(rows):
    base = blf.collate([row[:3] for row in rows])
    return CorrectionBatch(base.streams, base.tokens, base.codons, base.labels,
                           torch.as_tensor([row[3] for row in rows], dtype=torch.float32))


class CorrectionDataset(Dataset):
    def __init__(self, streams, positions, modalities, offsets, groups, control):
        self.streams, self.indices, self.modalities = streams, positions.copy(), tuple(modalities)
        self.labels = np.asarray(streams.labels[self.indices], dtype=np.float32)
        self.offsets, self.groups, self.control = offsets, tuple(groups), control
        if control.get("kind") not in ("real", "shuffle", ROW_SHUFFLE, "random"):
            raise ValueError("An unknown input control cannot fall back to real features.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        receiver = int(self.indices[index])
        donor = (self.control["row_map"][receiver]
                 if self.control["kind"] in ("shuffle", ROW_SHUFFLE) else receiver)
        features = {}
        for modality in self.modalities:
            values = np.array(self.streams.caches[modality][donor], copy=True)
            if self.control["kind"] == "random":
                stats = self.control["statistics"][modality]
                seed = int(_digest([self.control["seed"], self.groups[receiver], modality]), 16)
                rng = np.random.default_rng(seed)
                values = (rng.standard_normal(values.shape) * np.asarray(stats["scale"])
                          + np.asarray(stats["mean"])).astype(values.dtype)
            if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
                raise ValueError("Auxiliary inputs must remain finite nonempty token matrices.")
            features[modality] = torch.from_numpy(values)
        codons = int(self.streams.rows["length"].iloc[donor]) // 3
        return features, codons, self.labels[index], float(self.offsets[receiver])


class CorrectionTextCNN(nn.Module):
    """Native single-stream or codon-concatenated auxiliaries plus a fixed offset."""

    def __init__(self, widths, resolutions, modalities, layout, config):
        super().__init__()
        self.modalities, self.layout = tuple(modalities), layout
        self.required_modalities = self.modalities
        self.grid = (CodonGrid(widths["dna"], tconv=config.tconv, resolutions=resolutions)
                     if layout == "codon" else None)
        self.head = TextCNN(sum(widths[name] for name in modalities), 1, channels=config.channels)

    def forward(self, batch):
        if self.layout == "native":
            name = self.modalities[0]
            features = batch.streams[name]
            mask = mask_of(batch.tokens[name], features.shape[1])
        else:
            length = int(batch.codons.max())
            mask = mask_of(batch.codons, length)
            features = torch.cat([
                self.grid.place(name, batch.streams[name], length, batch.tokens[name])
                for name in self.modalities
            ], dim=-1) * mask.unsqueeze(-1)
        return self.head(features, mask) + batch.offsets.view(-1, 1), {}


def _builder(streams, variant, config, *, modalities, layout):
    return CorrectionTextCNN(streams.widths, streams.resolutions, modalities, layout, config)


def _control(control, streams, splits, groups, modalities):
    control = json.loads(_json(control))
    kind = control.get("kind")
    count = len(streams.rows)
    if kind == "real" and set(control) == {"kind"}:
        return control
    if kind in ("shuffle", ROW_SHUFFLE) and set(control) == {"kind", "row_map", "seed", "strata"}:
        _seed(control["seed"], "shuffle seed")
        mapping = np.asarray(control["row_map"])
        if mapping.shape != (count,) or mapping.dtype.kind not in "iu" or (mapping < 0).any() or (mapping >= count).any():
            raise ValueError("Shuffle control needs one valid donor position per source row.")
        if set(control["strata"]) != set(splits):
            raise ValueError("Shuffle strata must be declared for train, val and test separately.")
        for split, positions in splits.items():
            if not set(mapping[positions]) <= set(positions):
                raise ValueError("A shuffled auxiliary crosses its permitted partition.")
            if kind == ROW_SHUFFLE and sorted(mapping[positions]) != sorted(positions):
                raise ValueError("Row shuffle must use every donor row exactly once per partition.")
            strata = control["strata"][split]
            if set(strata) != {str(groups[index]) for index in positions}:
                raise ValueError("Shuffle strata do not describe the split's exact groups.")
            donors = {}
            for receiver in positions:
                group, donor = groups[receiver], groups[mapping[receiver]]
                if strata[str(group)] != strata[str(donor)]:
                    raise ValueError("A shuffled auxiliary crosses its declared permutation stratum.")
                if group == donor:
                    raise ValueError("Shuffle cannot use a donor from the receiver's exact-sequence group.")
                if kind == "shuffle" and group in donors and donors[group] != donor:
                    raise ValueError("Shuffle must move each whole group to one different donor group.")
                donors[group] = donor
            if kind == "shuffle" and len(set(donors.values())) != len(donors):
                raise ValueError("Shuffle donor groups must form a group-level permutation.")
        return control
    if kind == "random" and set(control) == {"kind", "seed", "statistics"}:
        _seed(control["seed"], "random-feature seed")
        if set(control["statistics"]) != set(modalities):
            raise ValueError("Random statistics must match the requested auxiliary modalities.")
        for modality, stats in control["statistics"].items():
            mean, scale = np.asarray(stats["mean"]), np.asarray(stats["scale"])
            if (mean.shape != (streams.widths[modality],) or scale.shape != mean.shape
                    or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale < 0).any()
                    or stats.get("training_positions") != splits["train"].tolist()):
                raise ValueError("Random statistics differ from the permitted training rows or feature widths.")
        return control
    raise ValueError("Control must explicitly declare real, row_shuffle_v1, historical shuffle, or random features.")


def run_neural(streams, *, split_positions, row_ids, groups, modalities, layout, offsets,
               config, output_dir, cache_records, input_contract, device, producer, role,
               control, log=print):
    """Fit/reuse a real TextCNN adapter through the existing checkpointed trainer.

    ``role`` is anchor or correction. Anchor offsets must be zero and use real
    inputs. Corrections use OOF offsets on training rows and final-anchor offsets
    elsewhere. Custom loader/model identities are checked here before train.run.
    """
    count = len(streams.rows)
    row_ids = evidence._labels(row_ids, "row_ids", unique=True)
    groups = evidence._labels(groups, "groups")
    splits = _splits(split_positions, count)
    modalities = tuple(modalities)
    if (streams.task != "regression" or len(row_ids) != count or len(groups) != count
            or not modalities or len(set(modalities)) != len(modalities)
            or not set(modalities) <= set(streams.caches)
            or layout not in ("native", "codon") or (layout == "native" and len(modalities) != 1)):
        raise ValueError("Declare regression rows, valid modalities and native-single or codon layout.")
    if role not in ("anchor", "correction") or config.entropy_lambda != 0:
        raise ValueError("Explicit anchor/correction roles use no gate entropy penalty.")
    if role == "correction":
        _offset_sources(input_contract)
    offsets = np.asarray(offsets, dtype=np.float32)
    active = np.concatenate(list(splits.values()))
    if offsets.shape != (count,) or not np.isfinite(offsets[active]).all():
        raise ValueError("Every admitted row requires a finite fixed-anchor offset.")
    control = _control(control, streams, splits, groups, modalities)
    if role == "anchor" and (np.any(offsets[active] != 0) or control["kind"] != "real"):
        raise ValueError("An anchor fits original targets with zero offsets and real inputs.")
    targets = np.asarray(streams.labels, dtype=np.float32)
    if not np.isfinite(targets[active]).all():
        raise ValueError("Training/evaluation targets must be finite.")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("The neural correction contract requires float32 parameters.")
    encoders = {name: streams.encoders[name] for name in modalities}
    records = {name: cache_records[name] for name in modalities}
    texts = {name: inputs_for_rows(ENCODERS[key], streams.rows["sequence"].tolist())
             for name, key in encoders.items()}
    run_identity._verify_cache_inputs(encoders, records, texts)
    for name in modalities:
        if records[name]["width"] != streams.widths[name] or len(streams.caches[name]) != count:
            raise ValueError("Opened cache shape differs from its supplied verified descriptor.")
    datasets = {name: CorrectionDataset(streams, positions, modalities, offsets, groups, control)
                for name, positions in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=name == "train",
                         generator=torch.Generator().manual_seed(config.seed) if name == "train" else None,
                         collate_fn=_collate, num_workers=0, drop_last=False)
        for name, dataset in datasets.items()
    }
    inputs = run_identity._inputs(
        streams.rows, ("sequence", "length"), targets, encoders, records, splits,
        (streams.label_column,), "regression",
    )
    inputs.update({"row_ids": list(row_ids), "groups": [_digest(group) for group in groups],
                   "group_identity": "sha256(JSON canonical exact-sequence group)", "role": role,
                   "offsets": {name: run_identity.array_record(offsets[positions])
                               for name, positions in splits.items()},
                   "protocol": input_contract, "control": _control_record(control)})
    identity = run_identity.make_identity(
        benchmark=streams.key, variant=f"{role}:{layout}:{'+'.join(modalities)}", config=asdict(config),
        inputs=inputs, model={"contract": CONTRACT, "builder": "correction_experiment._builder",
                              "modalities": list(modalities), "layout": layout,
                              "widths": streams.widths, "resolutions": streams.resolutions,
                              "source_sha256": _source_hashes(), "numerics": "float32-no-amp",
                              "textcnn": {"kernels": [3, 4, 5], "dropout": 0.2}},
        loader={"kind": "custom", "batch_size": config.batch_size, "num_workers": 0,
                "recovery": {name: run_identity._component_contract(loader) for name, loader in loaders.items()}},
    )
    # JSON normalization also removes NumPy scalar ambiguity before exact reuse checks.
    identity = json.loads(_json(identity))
    output_dir = Path(output_dir)
    summary = output_dir / "predictions.json"
    reused = _read_artifact(summary, identity, targets, row_ids, splits)
    if reused is not None:
        return reused
    result = train.run(
        streams, identity["variant"], config, device, output_dir, log=log,
        require_predictions=True, builder=_builder, loaders=loaders, tag="fit",
        run_identity=identity, producer=producer, modalities=modalities, layout=layout,
    )
    result_path = output_dir / "fit.json"
    weight_info = result.extra_fields.get("model_artifact", {})
    filename = weight_info.get("file", "")
    if Path(filename).name != filename or not filename:
        raise ValueError("Completed fit lacks a safe weight-artifact reference; do not retrain it.")
    weights = output_dir / filename
    if _sha(weights) != weight_info.get("sha256"):
        raise ValueError("Completed anchor/correction weights differ from their saved checksum.")
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    if payload.get("identity") != identity or payload.get("config") != asdict(config):
        raise ValueError("Saved weights do not match the requested scientific identity.")
    model = _builder(streams, identity["variant"], config, modalities=modalities, layout=layout).to(device)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
        validation_truth, validation_prediction, _ = train.predict(model, loaders["val"], device, "regression")
    finally:
        model.to("cpu")
        del model
    if not np.array_equal(validation_truth, targets[splits["val"]]):
        raise ValueError("Validation inference changed target ordering.")
    if not np.array_equal(np.asarray(result.test_truth, dtype=np.float32), targets[splits["test"]]):
        raise ValueError("Completed fit test targets differ from the declared rows.")
    record = {
        "contract": CONTRACT, "kind": "neural", "identity": identity,
        "producer": result.extra_fields.get("producer"), "prediction_producer": producer,
        "artifacts": {"fit": {"file": result_path.name, "sha256": _sha(result_path)},
                      "weights": {"file": weights.name, "sha256": _sha(weights)}},
        "predictions": {
            "val": _prediction_record(validation_prediction, splits["val"], targets, row_ids),
            "test": _prediction_record(result.test_prediction, splits["test"], targets, row_ids),
        },
    }
    _publish(summary, record)
    return FitArtifact(summary, record)


def reference_artifacts(record, reference_root):
    """Resolve unchanged Newton evidence under an explicit portable import root."""
    root = Path(reference_root).resolve()
    if record.get("kind") != "reference_anchor" or set(record.get("artifacts", {})) != {"fit", "weights"}:
        raise ValueError("A reference anchor requires its original fit and best weights.")
    paths = {}
    for name, item in record["artifacts"].items():
        path = (root / item["reference_relative"]).resolve()
        if not path.is_relative_to(root) or _sha(path) != item["sha256"]:
            raise ValueError("Newton anchor evidence changed or escaped its declared root.")
        paths[name] = path
    fit = json.loads(paths["fit"].read_text(encoding="utf-8"))
    test = record["predictions"]["test"]
    if (fit.get("identity") != record["identity"] or fit.get("test_truth") != test["truth"]
            or fit.get("test_prediction") != test["values"]
            or fit.get("model_artifact", {}).get("sha256") != record["artifacts"]["weights"]["sha256"]):
        raise ValueError("Reference predictions differ from their unchanged Newton source.")
    return paths


def reuse_newton_anchor(streams, *, reference_root, bundle, modality, layout, config,
                       row_ids, groups, split_positions, cache_records, output_dir,
                       input_contract, device, producer):
    """Infer validation predictions from matching Newton best weights; never fit.

    The old fit identity, score and test predictions remain unchanged. The new
    summary separately records current inference, ordered groups and A2 policy.
    Missing weights are a hold, not permission to silently retrain a baseline.
    """
    from . import study_references

    if layout not in ("native", "codon") or modality not in streams.encoders or streams.task != "regression":
        raise ValueError("Reference reuse requires an explicit regression single-encoder layout.")
    variant = f"{'unimodal' if layout == 'native' else 'unimodal-grid'}:{modality}"
    result = study_references.read_reference(
        reference_root, streams.key, bundle, variant, config.seed, config,
        streams=streams, cache_records=cache_records,
    )
    if result is None:
        raise FileNotFoundError("Compatible Newton anchor is missing; no automatic replacement fit.")
    return _reference_inference(
        streams, result, reference_root=reference_root, reference_location="newton",
        modality=modality, layout=layout, config=config, row_ids=row_ids, groups=groups,
        split_positions=split_positions, output_dir=output_dir, input_contract=input_contract,
        device=device, producer=producer,
    )


def reuse_grouped_anchor(streams, *, reference_root, candidate, config, row_ids, groups,
                         split_positions, cache_records, output_dir, input_contract, device, producer):
    """Reuse a completed grouped-panel single after matching its full current identity."""
    from . import grouped_stability

    record = grouped_stability.read_split(Path(reference_root) / "split.json", split_seed=42)
    specs = {spec.key: spec for spec in grouped_stability.fit_panel() if spec.variant.startswith("unimodal")}
    if candidate not in specs or streams.key != "mrna_stability":
        raise ValueError("Choose a declared grouped-stability single predictor.")
    spec = specs[candidate]
    path = grouped_stability.fit_path(reference_root, spec, config.seed)
    data = json.loads(path.read_text(encoding="utf-8"))  # Missing baseline is a hold, never a fit.
    grouped_stability.validate_result(data, record, spec, config)
    current = grouped_stability.make_identity(
        streams.rows, record, spec, config,
        {streams.encoders[name]: cache for name, cache in cache_records.items()},
    )
    if data["identity"] != current:
        raise ValueError("Completed grouped anchor differs from the currently opened model/input identity.")
    result = train.Result.load(path)
    result.extra_fields["reference_source"] = {
        "path": str(path.resolve()), "sha256": _sha(path), "current_inputs_verified": True}
    return _reference_inference(
        streams, result, reference_root=reference_root, reference_location="grouped_stability",
        modality=spec.modalities[0], layout="codon" if "-grid:" in spec.variant else "native",
        config=config, row_ids=row_ids, groups=groups, split_positions=split_positions,
        output_dir=output_dir, input_contract=input_contract, device=device, producer=producer,
    )


def _reference_inference(streams, result, *, reference_root, reference_location, modality, layout,
                         config, row_ids, groups, split_positions, output_dir, input_contract,
                         device, producer):
    row_ids = evidence._labels(row_ids, "row_ids", unique=True)
    groups = evidence._labels(groups, "groups")
    count = len(streams.rows)
    if len(row_ids) != count or len(groups) != count or torch.get_default_dtype() != torch.float32:
        raise ValueError("Reference row/group axes and float32 inference must match current streams.")
    splits = _splits(split_positions, count)
    variant = f"{'unimodal' if layout == 'native' else 'unimodal-grid'}:{modality}"
    source = result.extra_fields["reference_source"]
    if not source.get("current_inputs_verified"):
        raise ValueError("Newton anchor requires verification against currently opened inputs.")
    fit_path = Path(source["path"]).resolve()
    root = Path(reference_root).resolve()
    if not fit_path.is_relative_to(root) or _sha(fit_path) != source["sha256"]:
        raise ValueError("Newton fit evidence differs from the verified source.")
    original = json.loads(fit_path.read_text(encoding="utf-8"))
    identity = original["identity"]
    for name, positions in splits.items():
        if identity["inputs"]["splits"][name] != run_identity.array_record(positions, np.int64):
            raise ValueError("Newton anchor split differs from the correction experiment.")
    weight_info = original.get("model_artifact", {})
    filename = weight_info.get("file", "")
    if not filename or Path(filename).name != filename:
        raise ValueError("Newton anchor has no safe best-weight reference; preserve the fit.")
    weights = (fit_path.parent / filename).resolve()
    if not weights.is_relative_to(root):
        raise ValueError("Newton weight artifact escaped the explicit reference root.")
    if not weights.is_file():
        raise FileNotFoundError(f"Newton best weights are needed for validation inference: {weights}")
    if _sha(weights) != weight_info.get("sha256") or weights.stat().st_size != weight_info.get("size_bytes"):
        raise ValueError("Newton best weights differ from their recorded checksum/size.")
    artifacts = {name: {"reference_relative": path.relative_to(root).as_posix(), "sha256": _sha(path)}
                 for name, path in (("fit", fit_path), ("weights", weights))}
    inference_identity = {"config": asdict(config), "inputs": {
        "role": "anchor", "protocol": input_contract, "row_ids": list(row_ids),
        "groups": [_digest(group) for group in groups], "splits": identity["inputs"]["splits"],
        "group_identity": "sha256(JSON canonical exact-sequence group)"},
        "source_sha256": _source_hashes(), "artifacts": artifacts,
        "reference_location": reference_location,
        "operation": "BLF best-weight validation inference; original test predictions retained"}
    inference_identity = json.loads(_json(inference_identity))
    summary = Path(output_dir) / "predictions.json"
    targets = np.asarray(streams.labels, dtype=np.float32)
    if summary.exists():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        if saved.get("contract") != CONTRACT or saved.get("inference_identity") != inference_identity:
            raise ValueError("Incompatible completed reference-anchor inference; preserve it.")
        reference_artifacts(saved, root)
        for name in ("val", "test"):
            entry = saved["predictions"][name]
            if entry != _prediction_record(entry["values"], splits[name], targets, row_ids):
                raise ValueError("Saved Newton inference row order/targets differ.")
        return FitArtifact(summary, saved)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    if (payload.get("format") != "fusion-state-dict-v1" or payload.get("identity") != identity
            or payload.get("config") != asdict(config) or payload.get("variant") != variant
            or payload.get("benchmark") != streams.key or payload.get("best_epoch") != original["best_epoch"]):
        raise ValueError("Newton weight payload differs from its fitted model identity.")
    model = blf.build(streams, variant, channels=config.channels, tconv=config.tconv).to(device)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
        loader = blf.loader(streams, "val", batch_size=config.batch_size, shuffle=False,
                            seed=config.seed, modalities=(modality,))
        truth, values, _ = train.predict(model, loader, device, "regression")
    finally:
        model.to("cpu")
        del model
    if not np.array_equal(truth, targets[splits["val"]]):
        raise ValueError("Newton validation inference changed target order.")
    record = {"contract": CONTRACT, "kind": "reference_anchor", "reference_location": reference_location,
              "identity": identity,
              "inference_identity": inference_identity, "producer": original.get("producer"),
              "prediction_producer": producer, "artifacts": artifacts, "predictions": {
                  "val": _prediction_record(values, splits["val"], targets, row_ids),
                  "test": _prediction_record(original["test_prediction"], splits["test"], targets, row_ids)}}
    reference_artifacts(record, root)
    _publish(summary, record)
    return FitArtifact(summary, record)


def run_anchor_oof(streams, *, plans, row_ids, groups, train_positions, validation_positions,
                   test_positions, output_dir, **neural_options):
    """Train/reuse each fold with its own stopping rows, then assemble honest OOF values."""
    plans = tuple(plans)
    context = dict(row_ids=tuple(row_ids), groups=tuple(groups), train_positions=tuple(train_positions),
                   validation_positions=tuple(validation_positions), test_positions=tuple(test_positions))
    evidence.validate_plan(plans, **context)
    artifacts, predictions = [], []
    for index, plan in enumerate(plans):
        options = dict(neural_options)
        options["input_contract"] = {
            **options["input_contract"], "oof_fold": index,
            "fold_plan": [asdict(item) for item in plans], "outer_partitions": {
                name: list(context[name]) for name in ("train_positions", "validation_positions", "test_positions")},
        }
        artifact = run_neural(
            streams, split_positions={"train": plan.fit, "val": plan.stop, "test": plan.predict},
            row_ids=context["row_ids"], groups=context["groups"],
            offsets=np.zeros(len(streams.rows), dtype=np.float32), role="anchor", control={"kind": "real"},
            output_dir=Path(output_dir) / f"fold{index}", **options,
        )
        artifacts.append(artifact)
        predictions.append(artifact.prediction("test"))
    return evidence.assemble_oof(plans, predictions, task="regression", **context), tuple(artifacts)


def correction_offsets(oof, final_anchor, *, row_ids, split_positions):
    """Use the same final anchor evidence for correction validation/test and baseline."""
    row_ids = evidence._labels(row_ids, "row_ids", unique=True)
    splits = _splits(split_positions, len(row_ids))
    expected_train = tuple(row_ids[index] for index in splits["train"])
    if oof.task != "regression" or oof.row_ids != expected_train:
        raise ValueError("OOF predictions must cover the declared training rows in order.")
    identity = final_anchor.record.get("inference_identity", final_anchor.record["identity"])
    if identity["inputs"].get("role") != "anchor":
        raise ValueError("Corrections require a declared final anchor artifact.")
    for name, positions in splits.items():
        if identity["inputs"]["splits"][name] != run_identity.array_record(positions, np.int64):
            raise ValueError("The final anchor was fitted/evaluated on different rows; no train+val refit substitution.")
    offsets = np.full(len(row_ids), np.nan, dtype=np.float32)
    offsets[splits["train"]] = oof.values
    for name in ("val", "test"):
        prediction = final_anchor.prediction(name)
        if prediction.row_ids != tuple(row_ids[index] for index in splits[name]):
            raise ValueError("Final anchor prediction row IDs differ.")
        offsets[splits[name]] = prediction.values
    return offsets


def pooled_features(streams, positions, modalities, *, groups, control, split_positions):
    """Mean-pool each requested native stream; joint ridge features concatenate means."""
    positions = _row_positions(positions, len(streams.rows), "feature positions")
    splits = _splits(split_positions, len(streams.rows))
    control = _control(control, streams, splits, groups, modalities)
    dataset = CorrectionDataset(streams, positions, modalities, np.zeros(len(streams.rows)), groups, control)
    features = []
    for index in range(len(dataset)):
        streams_row = dataset[index][0]
        features.append(np.concatenate([streams_row[name].numpy().astype(np.float64).mean(axis=0)
                                        for name in modalities]))
    result = np.asarray(features)
    if not np.isfinite(result).all():
        raise ValueError("Pooled correction features contain nonfinite values.")
    return result


def inspect_shuffle(streams, *, groups, split_positions, control, sample_positions):
    """Inspect the declared donor mapping and only explicitly sampled token arrays.

    Metadata covers all declared rows. Feature equality is a bounded sample, not
    a population estimate. Identical protein arrays can survive nucleotide shuffling.
    """
    splits = _splits(split_positions, len(streams.rows))
    modalities = tuple(streams.caches)
    control = _control(control, streams, splits, groups, modalities)
    if control["kind"] not in ("shuffle", ROW_SHUFFLE):
        raise ValueError("This inspection requires an explicit shuffled-input control.")
    sample = _row_positions(sample_positions, len(streams.rows), "sample_positions")
    active = set(np.concatenate(list(splits.values())))
    if not set(sample) <= active:
        raise ValueError("Sampled positions must belong to a declared partition.")
    mapping = np.asarray(control["row_map"], dtype=int)
    lengths = streams.rows["length"].to_numpy()
    metadata, features = [], []
    for split, indices in splits.items():
        members = {}
        for index in indices:
            members[groups[index]] = members.get(groups[index], 0) + 1
        donor_counts = {group: 0 for group in members}
        for index in indices:
            donor_counts[groups[mapping[index]]] += 1
        metadata.append({
            "partition": split, "rows": len(indices), "groups": len(members),
            "length_changed_rows": int(np.sum(lengths[indices] != lengths[mapping[indices]])),
            "donor_exposure_total_variation": float(sum(
                abs(donor_counts[group] - members[group]) for group in members
            ) / (2 * len(indices))),
        })
        for receiver in sample[np.isin(sample, indices)]:
            donor = int(mapping[receiver])
            for modality in modalities:
                real = np.asarray(streams.caches[modality][int(receiver)])
                moved = np.asarray(streams.caches[modality][donor])
                if any(value.ndim != 2 or not len(value) or not np.isfinite(value).all()
                       for value in (real, moved)):
                    raise ValueError("Sampled auxiliary arrays must be finite nonempty token matrices.")
                features.append({
                    "partition": split, "position": int(receiver), "donor_position": donor,
                    "modality": modality, "real_tokens": len(real), "donor_tokens": len(moved),
                    "identical_features": bool(np.array_equal(real, moved)),
                    "real_pooled_norm": float(np.linalg.norm(real.astype(float).mean(axis=0))),
                    "donor_pooled_norm": float(np.linalg.norm(moved.astype(float).mean(axis=0))),
                    "real_sha256": run_identity.array_record(real)["sha256"],
                    "donor_sha256": run_identity.array_record(moved)["sha256"],
                })
    return {"metadata": metadata, "sample": features}


def run_ridge(features, targets, offsets, *, row_ids, split_positions, alphas, output_dir,
              input_contract, producer):
    """Train-only scaler/ridge residual fits; choose alpha by corrected validation MSE."""
    _offset_sources(input_contract)
    row_ids = evidence._labels(row_ids, "row_ids", unique=True)
    splits = _splits(split_positions, len(row_ids))
    features, targets, offsets = (np.asarray(value) for value in (features, targets, offsets))
    if features.ndim != 2 or features.shape[0] != len(row_ids) or not features.shape[1]:
        raise ValueError("Ridge features require an explicit nonempty row-by-feature matrix.")
    if targets.shape != (len(row_ids),) or offsets.shape != targets.shape:
        raise ValueError("Targets and fixed-anchor predictions must match the feature row order.")
    active = np.concatenate(list(splits.values()))
    for name, values in (("features", features), ("targets", targets), ("offsets", offsets)):
        evidence._finite_values(values[active], name)
    features, targets, offsets = (np.array(value, dtype=np.float64, copy=True)
                                 for value in (features, targets, offsets))
    alphas = tuple(alphas)
    if not alphas or any(isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                         or not np.isfinite(alpha) or alpha <= 0 for alpha in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("Provide an ordered, unique, finite positive ridge-alpha grid.")
    identity = {
        "contract": CONTRACT, "family": "ridge", "alphas": list(alphas),
        "selection": "corrected-validation-MSE; first declared alpha breaks ties; no refit",
        "active_positions": active.tolist(), "features": run_identity.array_record(features[active]),
        "targets": run_identity.array_record(targets[active]),
        "offsets": run_identity.array_record(offsets[active]), "row_ids": list(row_ids),
        "splits": {name: positions.tolist() for name, positions in splits.items()},
        "protocol": input_contract, "source_sha256": _source_hashes(),
    }
    identity = json.loads(_json(identity))
    path = Path(output_dir) / "predictions.json"
    reused = _read_artifact(path, identity, targets, row_ids, splits, ridge_inputs=(features, offsets))
    if reused is not None:
        return reused
    fitting, validation, testing = (splits[name] for name in ("train", "val", "test"))
    candidates = []
    best = None
    for alpha in alphas:
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipeline.fit(features[fitting], (targets - offsets)[fitting])
        prediction = offsets[validation] + pipeline.predict(features[validation])
        mse = float(np.mean((targets[validation] - prediction) ** 2))
        if not np.isfinite(mse):
            raise ValueError("Ridge validation loss is nonfinite; no completed record was published.")
        candidates.append({"alpha": alpha, "validation_mse": mse})
        if best is None or mse < best[0]:
            best = mse, alpha, pipeline, prediction
    mse, alpha, pipeline, validation_prediction = best
    scaler, ridge = pipeline.steps[0][1], pipeline.steps[1][1]
    record = {
        "contract": CONTRACT, "kind": "ridge", "identity": identity, "producer": producer,
        "selected_alpha": alpha, "validation_mse": mse, "candidates": candidates,
        "fitted": {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                   "coefficient": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)},
        "predictions": {
            "val": _prediction_record(validation_prediction, validation, targets, row_ids),
            "test": _prediction_record(offsets[testing] + pipeline.predict(features[testing]), testing, targets, row_ids),
        },
    }
    _publish(path, record)
    return FitArtifact(path, record)
