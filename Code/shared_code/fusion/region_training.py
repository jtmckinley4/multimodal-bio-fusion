"""A6 preparation, explicit model panel and strict reusable neural fits.

The notebook owns execution switches and settings. This adapter uses existing
token caches and the shared trainer; it never acquires a dataset or an encoder.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from .. import mrnabench
from . import controlled_models, region_experiment as inputs, run_identity, train
from .textcnn import TextCNN

CONTRACT = "full-transcript-region-fusion-v1"
SPLITS = ("train", "val", "test")
CORE = (("rna_only", "true"), ("concat", "true"), ("concat_mask", "true"),
        ("concat_mask", "permuted"), ("position_gate", "true"),
        ("position_gate", "permuted"), ("attention_cds", "true"),
        ("attention_cds", "permuted"))
OPTIONAL = (("attention_uniform", "true"), ("position_control", "true"),
            ("position_control", "permuted"))
DIAGNOSTIC_DEFINITION = {
    "version": 2,
    "population": "test rows in saved fit order; true biological regions",
    "positions": "native bases or pooled-bin midpoint membership",
    "reduction": "mean over real positions within each transcript and region",
    "norms": "L2 over the feature axis before regional averaging; unsupported updates zero",
}


@dataclass(frozen=True)
class FitSpec:
    rna_encoder: str
    grid: str
    method: str
    mask: str

    @property
    def key(self):
        return f"{self.rna_encoder}__{self.grid}__{self.method}__{self.mask}"


def fit_panel(*, include_optional=False):
    arms = CORE + OPTIONAL if include_optional else CORE
    return tuple(FitSpec(encoder, grid, method, mask)
                 for encoder in ("rnafm", "rinalmo") for grid in ("native", "codon_pool")
                 for method, mask in arms)


def _interval(cohort, recipient, donor):
    source, target = cohort.layouts[donor], cohort.layouts[recipient]
    # Integer arithmetic makes endpoint rounding reproducible across platforms.
    start = source.cds_start * target.sequence_length // source.sequence_length
    end = (source.cds_end * target.sequence_length + source.sequence_length - 1) // source.sequence_length
    return [start, end]


def prepare_study(cohort, *, split_seed, mask_seed):
    """Split recorded homology/gene groups and assign within-partition mask donors."""
    inputs._check_cohort(cohort)
    if cohort.identity["grouping"] != "homology" or cohort.identity["unmapped_gene_policy"] != "gene":
        raise ValueError("A6 requires the homology map and explicit gene grouping for unmapped rows")
    if split_seed != 42 or type(split_seed) is not int or type(mask_seed) is not int or mask_seed < 0:
        raise ValueError("Use agreed split seed 42 and an explicit nonnegative mask seed")
    split = mrnabench.grouped_split(cohort.groups, seed=split_seed, ratios=(.70, .15, .15))
    positions = {name: indices.tolist() for name, indices in zip(SPLITS, split)}
    donors = list(range(len(cohort.frame)))
    rng = np.random.default_rng(mask_seed)
    for name in SPLITS:
        members = np.asarray(positions[name])
        if len(members) < 2:
            raise ValueError(f"{name} needs at least two rows for a within-partition mask control")
        order = rng.permutation(members)
        for recipient, donor in zip(order, np.roll(order, 1)):
            donors[int(recipient)] = int(donor)
    record = {"schema": CONTRACT, "cohort": cohort.identity, "split_seed": split_seed,
              "ratios": [.70, .15, .15], "positions": positions, "mask_seed": mask_seed,
              "mask_rule": "within-partition seeded cyclic donor permutation; normalized CDS endpoints; floor-start ceil-end; pooled-bin midpoint",
              "donor_positions": donors,
              "permuted_intervals": [_interval(cohort, row, donor) for row, donor in enumerate(donors)]}
    record["sha256"] = inputs._digest(record)
    return validate_preparation(cohort, record)


def validate_preparation(cohort, record):
    inputs._check_cohort(cohort)
    content = {key: value for key, value in record.items() if key != "sha256"}
    if (record.get("schema") != CONTRACT or record.get("cohort") != cohort.identity
            or record.get("sha256") != inputs._digest(content) or record.get("split_seed") != 42
            or record.get("ratios") != [.70, .15, .15] or set(record.get("positions", {})) != set(SPLITS)):
        raise ValueError("Preparation identity or agreed protocol differs")
    seen, groups, sequences = set(), {}, {}
    donors = record["donor_positions"]
    if len(donors) != len(cohort.frame) or len(record["permuted_intervals"]) != len(donors):
        raise ValueError("Donor mapping must cover the complete cohort")
    for name in SPLITS:
        indices = record["positions"][name]
        if (len(indices) < 2 or any(type(i) is not int or not 0 <= i < len(cohort.frame) for i in indices)
                or len(set(indices)) != len(indices) or seen & set(indices)):
            raise ValueError("Invalid, repeated, overlapping or too-small split positions")
        seen.update(indices)
        groups[name] = set(cohort.groups[indices])
        sequences[name] = set(cohort.frame.iloc[indices]["sequence"])
        selected = [donors[i] for i in indices]
        if set(selected) != set(indices) or len(set(selected)) != len(selected):
            raise ValueError("Mask donors must form a permutation within their own partition")
        for row, donor in zip(indices, selected):
            if row == donor or record["permuted_intervals"][row] != _interval(cohort, row, donor):
                raise ValueError("Donor identity or normalized interval mapping differs")
    if seen != set(range(len(cohort.frame))):
        raise ValueError("Split positions do not cover every eligible transcript")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b] or sequences[a] & sequences[b]:
            raise ValueError(f"Recorded group or exact-sequence overlap in {a}/{b}; inspect the source mapping")
    return record


def publish_preparation(path, record):
    """Publish once; do not replace existing cohort, split or donor evidence."""
    inputs._publish_once(Path(path), record)


def read_preparation(path, cohort=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (record.get("schema") != CONTRACT or record.get("sha256") != inputs._digest(
            {key: value for key, value in record.items() if key != "sha256"})):
        raise ValueError("Saved preparation schema or content hash is invalid")
    return validate_preparation(cohort, record) if cohort is not None else record


def _settings(settings):
    required = {"projection_width", "scorer_width", "n_heads", "dropout", "projection_activation",
                "projection_norm", "update_norm", "control_bottleneck"}
    if set(settings) != required:
        raise ValueError(f"Declare exactly the A6 architecture settings: {sorted(required)}")
    if settings["projection_activation"] != "tanh" or settings["projection_norm"] != "none":
        raise ValueError("A6 retains tanh projections without projection normalization")
    if settings["update_norm"] not in ("none", "layer"):
        raise ValueError("Choose the attention update normalization explicitly")
    # scorer_width is retained in saved configurations; joint affine gates do not use it.
    for key in ("projection_width", "scorer_width", "n_heads", "control_bottleneck"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["projection_width"] % settings["n_heads"] or not 0 <= settings["dropout"] < 1:
        raise ValueError("Invalid attention width, heads or dropout")
    return dict(settings)


class RegionModel(nn.Module):
    """A–E feature operations with one shared TextCNN output width.

    D applies joint RNA/protein affine sigmoid gates before C's concatenation,
    explicit mask and mixer. Retaining that readout is our controlled adaptation;
    it is not the literal weighted sum in Mina A4. E uses the A5 RNA skip path.
    """

    def __init__(self, widths, method, settings, channels):
        super().__init__()
        settings = _settings(settings)
        if method not in {name for name, _ in CORE + OPTIONAL}:
            raise ValueError("Unknown A6 method")
        self.method, self.width = method, settings["projection_width"]
        self.collect_norms = False
        self.head = TextCNN(self.width, 1, channels=channels)
        if method in ("attention_cds", "attention_uniform"):
            # Legacy key attention_uniform means ungated learned attention:
            # update coefficients are one; token-attention weights still learn.
            self.attention = controlled_models.AnchorAttentionFusion(
                widths, anchor="rna", mode="position_sigmoid" if method == "attention_cds" else "skip",
                **{key: settings[key] for key in ("projection_width", "scorer_width", "n_heads",
                                                "dropout", "projection_activation", "projection_norm", "update_norm")})
            return
        names = ("rna",) if method == "rna_only" else ("rna", "prot")
        self.project = nn.ModuleDict({name: nn.Sequential(nn.Linear(widths[name], self.width), nn.Tanh())
                                      for name in names})
        if method == "rna_only":
            return
        self.with_mask = method != "concat"
        joined_width = 2 * self.width + int(self.with_mask)
        if method == "position_control":
            rank = settings["control_bottleneck"]
            self.mix = nn.Sequential(nn.Linear(joined_width, rank), nn.Linear(rank, self.width))
        else:
            self.mix = nn.Linear(joined_width, self.width)
        if method == "position_gate":
            self.gate = nn.ModuleDict({m: nn.Linear(2 * self.width, 1) for m in names})

    def features(self, batch):
        real = batch.masks["rna"]
        if self.method in ("attention_cds", "attention_uniform"):
            allowed = ({"prot": batch.region & batch.masks["prot"]}
                       if self.method == "attention_cds" else None)
            self.attention.collect_norms = self.collect_norms
            features, aux = self.attention(batch.streams, batch.masks, availability=allowed)
            return features.masked_fill(~real.unsqueeze(-1), 0), real, aux
        rna = self.project["rna"](batch.streams["rna"]).masked_fill(~real.unsqueeze(-1), 0)
        if self.method == "rna_only":
            aux = {"fused_norm": rna.detach().norm(dim=-1)} if self.collect_norms else {}
            return rna, real, aux
        protein = self.project["prot"](batch.streams["prot"])
        protein = protein.masked_fill(~batch.masks["prot"].unsqueeze(-1), 0)
        aux = {}
        if self.method == "position_gate":
            projected = {"rna": rna, "prot": protein}
            joint = torch.cat((rna, protein), dim=-1)
            gates = {name: torch.sigmoid(self.gate[name](joint)).squeeze(-1) for name in projected}
            gates["rna"] = gates["rna"].masked_fill(~real, 0)
            gates["prot"] = gates["prot"].masked_fill(~(batch.region & batch.masks["prot"]), 0)
            rna, protein = rna * gates["rna"].unsqueeze(-1), protein * gates["prot"].unsqueeze(-1)
            aux = {"gates": torch.stack([gates["rna"], gates["prot"]], -1),
                   "gate_kind": "region_position_sigmoid", "gate_modalities": ("rna", "prot")}
            if self.collect_norms:
                norms = torch.stack([value.detach().norm(dim=-1) for value in projected.values()], -1)
                aux.update(feature_norms=norms, contribution_norms=norms * aux["gates"].detach())
        pieces = [rna, protein]
        if self.with_mask:
            pieces.append(batch.region.to(rna.dtype).unsqueeze(-1))
        features = self.mix(torch.cat(pieces, dim=-1)).masked_fill(~real.unsqueeze(-1), 0)
        if self.collect_norms:
            aux["fused_norm"] = features.detach().norm(dim=-1)
        return features, real, aux

    def forward(self, batch):
        features, mask, aux = self.features(batch)
        return self.head(features, mask), aux


@dataclass
class RegionStreams:
    cohort: inputs.RegionCohort
    caches: inputs.RegionCaches
    preparation: dict
    spec: FitSpec

    @property
    def key(self):
        return self.cohort.dataset_key

    task = "regression"

    @property
    def widths(self):
        keys = {"rna": self.spec.rna_encoder}
        if self.spec.method != "rna_only":
            keys["prot"] = "esm2_8m"
        return {name: self.caches.caches[key].width for name, key in keys.items()}


def regression_collate(rows):
    batch = inputs.collate_region_rows(rows)
    if batch.labels.shape[1:] != (1,):
        raise ValueError("A6 requires one scalar target per transcript")
    batch.labels = batch.labels[:, 0]
    return batch


def make_loaders(streams, config):
    """Validate a direct call, then build loaders without rehashing each partition."""
    validate_preparation(streams.cohort, streams.preparation)
    return _make_loaders(streams, config)


def _make_loaders(streams, config):
    """Internal path after this operation has validated the shared cohort once."""
    result = {}
    for name in SPLITS:
        dataset = inputs.RegionDataset._from_validated_cohort(
            streams.cohort, streams.caches, rna_encoder=streams.spec.rna_encoder, grid=streams.spec.grid,
            indices=streams.preparation["positions"][name], include_protein=streams.spec.method != "rna_only",
            region_intervals=(streams.preparation["permuted_intervals"] if streams.spec.mask == "permuted" else None))
        result[name] = DataLoader(dataset, batch_size=config.batch_size, shuffle=name == "train",
                                  generator=torch.Generator().manual_seed(config.seed), num_workers=0,
                                  collate_fn=regression_collate)
    return result


def build_model(streams, variant, config, *, settings):
    _validate_model_request(streams, variant, config)
    return RegionModel(streams.widths, variant, settings, config.channels)


def _validate_model_request(streams, variant, config):
    if (variant != streams.spec.method or config.entropy_lambda != 0
            or type(config.channels) is not int or config.channels != 100):
        raise ValueError("A6 requires its declared method, no entropy and the common 100-channel head")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("A6 uses float32 predictor parameters")


def _parameter_counts(streams, config, settings):
    """Count declared layers without allocating a model merely to reuse a result.

    Tests compare this arithmetic against every actual A6 model. These counts
    remain descriptive metadata; they do not establish matched capacity.
    """
    _validate_model_request(streams, streams.spec.method, config)
    width, channels, method = settings["projection_width"], config.channels, streams.spec.method
    prediction = sum(channels * (width * kernel + 1) for kernel in (3, 4, 5)) + channels * 3 + 1
    fusion = sum((value + 1) * width for value in streams.widths.values())
    if method in ("attention_cds", "attention_uniform"):
        fusion += 4 * width * width + 4 * width  # Q/K/V and output projections, with biases.
        fusion += 2 * width if settings["update_norm"] == "layer" else 0
        fusion += 2 * width + 1 if method == "attention_cds" else 0
    elif method != "rna_only":
        joined = 2 * width + int(method != "concat")
        if method == "position_control":
            rank = settings["control_bottleneck"]
            fusion += (joined + 1) * rank + (rank + 1) * width
        else:
            fusion += (joined + 1) * width
        if method == "position_gate":
            fusion += 2 * (2 * width + 1)
    return {"total": prediction + fusion, "prediction": prediction}


def _loader_record(config):
    """The existing fixed DataLoader contract, checked against constructed loaders."""
    if (type(config.batch_size) is not int or config.batch_size < 1
            or type(config.seed) is not int or config.seed < 0):
        raise ValueError("Loader batch size and seed require positive/nonnegative integers")
    def generator_contract(cls):
        return {"type": f"{cls.__module__}.{cls.__qualname__}", "fields": ["generator"]}

    recovery = {name: {"loader": generator_contract(DataLoader)} for name in SPLITS}
    for name in ("sampler", "batch_sampler.sampler"):
        recovery["train"][name] = generator_contract(RandomSampler)
    return {"kind": "custom", "batch_size": config.batch_size, "seed": config.seed,
            "num_workers": 0, "implementation": f"{__name__}.make_loaders", "recovery": recovery}


def make_identity(streams, config, settings, loaders, cache_records):
    """Validate a standalone request, including its actual loader contracts."""
    validate_preparation(streams.cohort, streams.preparation)
    identity = _make_identity(streams, config, settings, cache_records)
    validate_loaders(streams, config, loaders)
    run_identity.validate_loaders(identity, loaders)
    return identity


def _make_identity(streams, config, settings, cache_records):
    """Build identity after this operation's cohort/preparation boundary check."""
    if streams.spec not in fit_panel(include_optional=True):
        raise ValueError("Fit is outside the declared A6 panel")
    settings = _settings(settings)
    required = {streams.spec.rna_encoder} | ({"esm2_8m"} if streams.spec.method != "rna_only" else set())
    if set(cache_records) != required:
        raise ValueError("Cache evidence must identify exactly the streams this fit uses")
    for key, record in cache_records.items():
        if (key not in streams.caches.caches or key not in streams.caches.plans
                or streams.caches.plans[key].identity["cohort_sha256"] != streams.cohort.identity["sha256"]
                or record["input_sha256"] != streams.caches.plans[key].identity["input_sha256"]
                or record["width"] != streams.caches.caches[key].width):
            raise ValueError("Cache evidence differs from the planned ordered windows")
    labels = streams.cohort.labels[:, 0].astype(np.float32)
    positions = streams.preparation["positions"]
    inputs_record = {"cohort_sha256": streams.cohort.identity["sha256"],
                     "preparation_sha256": streams.preparation["sha256"],
                     "spec": asdict(streams.spec), "encoders": cache_records,
                     "chunks": {key: streams.caches.plans[key].identity for key in required},
                     "targets": {"task": "regression", **run_identity.array_record(labels)},
                     "splits": {name: run_identity.array_record(positions[name], np.int64) for name in SPLITS},
                     "split_targets": {name: run_identity.array_record(labels[positions[name]]) for name in SPLITS}}
    counts = _parameter_counts(streams, config, settings)
    from . import region_inputs, textcnn
    sources = {Path(module.__file__).name: tokens_sha(module.__file__) for module in
               (inputs, region_inputs, controlled_models, train, textcnn)}
    sources[Path(__file__).name] = tokens_sha(__file__)
    model_record = {"contract": CONTRACT, "builder": f"{__name__}.build_model", "settings": settings,
                    "widths": streams.widths, "parameter_counts": counts, "implementation_sha256": sources,
                    "numerics": {"parameters": "float32", "inputs": "float32", "amp": False},
                    "textcnn": {"kernels": [3, 4, 5], "channels": 100, "dropout": .2}}
    if streams.spec.method in ("position_gate", "attention_cds"):
        model_record["gate"] = {
            "scorer": "affine_sigmoid", "scorer_width_active": False,
            "input": ("joint_projected_rna_protein" if streams.spec.method == "position_gate"
                      else "anchor_and_all_auxiliary_updates"),
            "readout": ("gated_concat_mask_mixer" if streams.spec.method == "position_gate"
                        else "anchor_plus_gated_updates"),
        }
    identity = run_identity.make_identity(benchmark=streams.key, variant=streams.spec.method, config=config,
                                          inputs=inputs_record, model=model_record, loader=_loader_record(config))
    return identity


def tokens_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_loaders(streams, config, loaders):
    if set(loaders) != set(SPLITS):
        raise ValueError("All three loaders are required")
    for name, loader in loaders.items():
        expected = streams.preparation["positions"][name]
        dataset = loader.dataset
        if (type(dataset) is not inputs.RegionDataset or dataset.cohort is not streams.cohort
                or dataset.caches is not streams.caches or dataset.indices.tolist() != expected
                or dataset.rna_encoder != streams.spec.rna_encoder or dataset.grid != streams.spec.grid
                or dataset.include_protein != (streams.spec.method != "rna_only")
                or loader.batch_size != config.batch_size or loader.num_workers != 0
                or loader.collate_fn is not regression_collate or loader.drop_last):
            raise ValueError(f"{name}: loader differs from declared rows, inputs or batch settings")
        desired = streams.preparation["permuted_intervals"] if streams.spec.mask == "permuted" else None
        if dataset.region_intervals != desired:
            raise ValueError("Loader region-mask policy differs")
        expected_sampler = "RandomSampler" if name == "train" else "SequentialSampler"
        if type(loader.sampler).__name__ != expected_sampler or loader.generator.initial_seed() != config.seed:
            raise ValueError("Loader ordering or random seed differs")


def read_fit(path, *, expected_identity=None, preparation=None):
    result = train.Result.load(path)
    identity = run_identity.validate_identity(result.extra_fields.get("identity"), config=result.config,
                                              benchmark=result.benchmark, variant=result.variant)
    if identity["model"].get("contract") != CONTRACT or (expected_identity is not None and identity != expected_identity):
        raise ValueError("Saved fit belongs to a different region experiment")
    if any(type(identity["loader"].get(key)) is not int
           or identity["loader"][key] != result.config.get(key) for key in ("seed", "batch_size")):
        raise ValueError("Saved loader seed or batch size differs from the fit configuration")
    if preparation is not None and identity["inputs"]["preparation_sha256"] != preparation["sha256"]:
        raise ValueError("Saved fit uses different cohort/splits/mask donors")
    if preparation is not None:
        if identity["inputs"]["cohort_sha256"] != preparation["cohort"]["sha256"]:
            raise ValueError("Saved fit cohort differs from prepared input evidence")
        for name in SPLITS:
            if identity["inputs"]["splits"][name] != run_identity.array_record(preparation["positions"][name], np.int64):
                raise ValueError("Saved fit row assignment differs from the prepared partition")
    count = identity["inputs"]["splits"]["test"]["shape"][0]
    truth, prediction = np.asarray(result.test_truth), np.asarray(result.test_prediction)
    if (truth.shape != (count,) or prediction.shape != (count,)
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()
            or type(result.epochs) is not int or result.epochs < 1
            or type(result.best_epoch) is not int or not 0 <= result.best_epoch < result.epochs
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   for value in (result.val_score, result.test_score))):
        raise ValueError("Saved fit lacks complete valid predictions or epoch evidence")
    if run_identity.array_record(truth, np.float32) != identity["inputs"]["split_targets"]["test"]:
        raise ValueError("Saved test targets differ from identified rows")
    return result


def run_fit(streams, config, settings, output_dir, device, *, producer, log=print):
    """Reuse exact completed results or train once with checkpoint/RNG recovery."""
    validate_preparation(streams.cohort, streams.preparation)
    records = {key: run_identity.cache_record(streams.caches.caches[key].cache_dir)
               for key in ({streams.spec.rna_encoder} | ({"esm2_8m"} if streams.spec.method != "rna_only" else set()))}
    identity = _make_identity(streams, config, settings, records)
    destination = Path(output_dir) / streams.spec.key
    path = destination / f"seed{config.seed}.json"
    if path.exists():
        return read_fit(path, expected_identity=identity)
    if not isinstance(producer, dict) or not producer:
        raise ValueError("New fits require explicit execution producer metadata")
    loaders = _make_loaders(streams, config)
    validate_loaders(streams, config, loaders)
    run_identity.validate_loaders(identity, loaders)
    train.run(streams, streams.spec.method, config, device, destination, log=log,
              require_predictions=True, builder=build_model, loaders=loaders, tag=f"seed{config.seed}",
              run_identity=identity, producer=producer, settings=settings)
    return read_fit(path, expected_identity=identity)


def saved_inventory(root, *, dataset_key, seeds=(0, 1, 2), include_optional=False):
    """Read saved JSON records only; missing work is reported, never computed."""
    root = Path(root) / dataset_key
    path = root / "preparation.json"
    preparation = read_preparation(path) if path.is_file() else None
    rows = []
    for spec in fit_panel(include_optional=include_optional):
        for seed in seeds:
            path = root / "fits" / spec.key / f"seed{seed}.json"
            row = {**asdict(spec), "seed": seed, "path": str(path), "status": "missing"}
            if path.exists():
                try:
                    if preparation is None:
                        raise ValueError("Saved preparation evidence is missing")
                    result = read_fit(path, preparation=preparation)
                    if result.config["seed"] != seed or result.extra_fields["identity"]["inputs"]["spec"] != asdict(spec):
                        raise ValueError("Result path and identified fit differ")
                    row.update(status="complete", validation=result.val_score, test=result.test_score,
                               epochs=result.epochs, seconds=result.seconds)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(status="incompatible", issue=str(error))
            rows.append(row)
    return rows


def read_diagnostics(fit_path):
    """Read v2, or legacy v1 if absent, using only diagnostic and source-fit JSON.

    V2 checks its content digest and binding to the unchanged fit. V1 never had a
    content digest: its structural/source checks are reported separately, and no
    norm fields are inferred. This function never loads weights/caches or infers.
    A corrupt v2 is an error, not permission to fall back to an older record.
    """
    fit_path = Path(fit_path)
    path = fit_path.with_name(f"{fit_path.stem}.diagnostics-v2.json")
    version = 2
    if not path.is_file():
        path, version = fit_path.with_name(f"{fit_path.stem}.diagnostics.json"), 1
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("schema") != f"region-diagnostics.v{version}":
        raise ValueError("Saved diagnostic schema differs from its versioned destination")
    if version == 2:
        actual = inputs._digest({key: value for key, value in saved.items() if key != "sha256"})
        if saved.get("sha256") != actual:
            raise ValueError("Saved diagnostic content hash is invalid")
    result = read_fit(fit_path)
    context = saved.get("context", {})
    artifact = result.extra_fields.get("model_artifact", {})
    if (not artifact.get("sha256") or context.get("fit_sha256") != tokens_sha(fit_path)
            or context.get("identity") != result.extra_fields["identity"]
            or context.get("weights_sha256") != artifact["sha256"]):
        raise ValueError("Saved diagnostics are not bound to the unchanged source fit")
    if version == 2:
        sources = context.get("implementation_sha256", {})
        if (context.get("diagnostic_definition") != DIAGNOSTIC_DEFINITION
                or not isinstance(sources, dict) or not sources
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in "0123456789abcdef" for c in value) for value in sources.values())):
            raise ValueError("Saved diagnostic definition or implementation identity is invalid")
    rows = saved.get("rows")
    if (not isinstance(rows, list) or len(rows) != len(result.test_truth)
            or not isinstance(saved.get("producer"), dict) or not saved["producer"]):
        raise ValueError("Saved diagnostics lack complete rows or producer evidence")
    required = {"fused_norm"}
    if result.variant == "position_gate":
        required |= {"gates", "feature_norms", "contribution_norms"}
    elif result.variant in ("attention_cds", "attention_uniform"):
        required |= {"anchor_norm", "raw_update_norms", "update_norms", "update_weights"}
        if result.variant == "attention_cds":
            required.add("gates")
    row_ids = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("row_id"), str) or not row["row_id"]:
            raise ValueError("Saved diagnostic row identity is invalid")
        row_ids.append(row["row_id"])
        regions = row.get("regions", {})
        if (type(row.get("positions")) is not int or row["positions"] < 1
                or set(regions) != {"utr5", "cds", "utr3"}):
            raise ValueError("Saved diagnostic region coverage is invalid")
        for values in regions.values():
            count = values.get("positions")
            if type(count) is not int or count < 0:
                raise ValueError("Saved diagnostic region position count is invalid")
            if version == 2 and not required.issubset(values):
                raise ValueError("Saved diagnostic norm fields are incomplete")
            for name, value in values.items():
                if name == "positions":
                    continue
                if count == 0 and value is None:
                    continue
                if count == 0 or value is None or not np.isfinite(np.asarray(value, dtype=float)).all():
                    raise ValueError("Saved diagnostic values are missing or nonfinite")
        if sum(value["positions"] for value in regions.values()) != row["positions"]:
            raise ValueError("Saved diagnostic region counts do not cover the row")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Saved diagnostic row identities are duplicated")
    return {"path": str(path), "record": saved, "version": version,
            "norm_status": "present" if version == 2 else "legacy_unavailable",
            "integrity": ("record_sha256_and_fit_binding" if version == 2
                          else "legacy_structure_and_fit_binding")}


def collect_diagnostics(streams, config, settings, output_dir, device, *, producer):
    """Summarize gates by true biological region from verified best weights only.

    Position means are per transcript; they are descriptive, not causal importance.
    The permuted control never changes the biological regions used for this report.
    """
    validate_preparation(streams.cohort, streams.preparation)
    records = {key: run_identity.cache_record(streams.caches.caches[key].cache_dir)
               for key in ({streams.spec.rna_encoder} | ({"esm2_8m"} if streams.spec.method != "rna_only" else set()))}
    identity = _make_identity(streams, config, settings, records)
    fit_path = Path(output_dir) / streams.spec.key / f"seed{config.seed}.json"
    result = read_fit(fit_path, expected_identity=identity)
    artifact = result.extra_fields.get("model_artifact", {})
    filename = artifact.get("file", "")
    if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Completed fit does not identify safe best-weight evidence")
    weights = fit_path.parent / filename
    if tokens_sha(weights) != artifact.get("sha256") or weights.stat().st_size != artifact.get("size_bytes"):
        raise ValueError("Completed weights changed; diagnostics never retrain")
    context = {"fit_sha256": tokens_sha(fit_path), "weights_sha256": artifact["sha256"],
               "identity": identity, "prediction_tolerance": {"rtol": 1e-5, "atol": 1e-6},
               "diagnostic_definition": DIAGNOSTIC_DEFINITION,
               "implementation_sha256": {Path(module.__file__).name: tokens_sha(module.__file__)
                                         for module in (controlled_models, inputs)},
               }
    context["implementation_sha256"][Path(__file__).name] = tokens_sha(__file__)
    output = fit_path.parent / f"seed{config.seed}.diagnostics-v2.json"
    if output.exists():
        saved = read_diagnostics(fit_path)["record"]
        if saved.get("context") != context:
            raise ValueError("Existing diagnostics belong to different evidence")
        return saved
    if not producer:
        raise ValueError("Diagnostic inference requires execution producer evidence")
    loaders = _make_loaders(streams, config)
    validate_loaders(streams, config, loaders)
    run_identity.validate_loaders(identity, loaders)
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if (payload.get("format") != "fusion-state-dict-v1" or payload.get("identity") != identity
            or payload.get("best_epoch") != result.best_epoch or payload.get("config") != result.config):
        raise ValueError("Best-weight payload differs from the saved fit")
    with torch.random.fork_rng(devices=[]):
        model = build_model(streams, streams.spec.method, config, settings=settings)
    error = None
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        model.collect_norms = True
        summaries, prediction, truth = _diagnostic_rows(model, loaders["test"], streams.cohort, device)
    except BaseException as caught:
        error = caught
        raise
    finally:
        try:
            model.cpu()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Diagnostic model cleanup also failed: {cleanup_error}")
        finally:
            del model
    if (not np.array_equal(np.asarray(truth, np.float32), np.asarray(result.test_truth, np.float32))
            or not np.allclose(prediction, result.test_prediction, rtol=1e-5, atol=1e-6)):
        raise ValueError("Best-weight inference does not reproduce saved test order/predictions within tolerance")
    record = {"schema": "region-diagnostics.v2", "context": context, "producer": producer,
              "summary": "per-row mean gates and feature/update L2 norms in each true annotated region",
              "rows": summaries}
    record["sha256"] = inputs._digest(record)
    inputs._publish_once(output, record)
    return record


def _diagnostic_rows(model, loader, cohort, device):
    """Collect the same test-row quantities while keeping inference separate from I/O."""
    row_lookup = {row_id: index for index, row_id in enumerate(cohort.row_ids)}
    summaries, prediction, truth = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            features, valid, aux = model.features(batch)
            prediction.extend(model.head(features, valid).squeeze(-1).cpu().tolist())
            truth.extend(batch.labels.cpu().tolist())
            for batch_index, row_id in enumerate(batch.row_ids):
                layout = cohort.layouts[row_lookup[row_id]]
                spans = batch.positions[batch_index]
                centers = np.array([(left + right - 1) // 2 for left, right in spans])
                regions = {"utr5": centers < layout.cds_start,
                           "cds": (centers >= layout.cds_start) & (centers < layout.cds_end),
                           "utr3": centers >= layout.cds_end}
                row = {"row_id": row_id, "positions": len(spans), "regions": {}}
                for name, select in regions.items():
                    values = {"positions": int(select.sum())}
                    for diagnostic in ("gates", "update_weights", "feature_norms", "contribution_norms",
                                       "anchor_norm", "raw_update_norms", "update_norms", "fused_norm"):
                        if diagnostic in aux:
                            matrix = aux[diagnostic][batch_index, :len(spans)].cpu().numpy()
                            values[diagnostic] = matrix[select].mean(axis=0).tolist() if select.any() else None
                    row["regions"][name] = values
                summaries.append(row)
    return summaries, prediction, truth


def comparisons(panel):
    """Prespecified contrasts; pair seeds and rows rather than pool predictions."""
    by = {(spec.rna_encoder, spec.grid, spec.method, spec.mask): spec.key for spec in panel}
    contrasts = []
    for encoder in ("rnafm", "rinalmo"):
        for grid in ("native", "codon_pool"):
            lookup = lambda method, mask="true": by[(encoder, grid, method, mask)]
            for label, first, second in (("B-A", "concat", "rna_only"), ("C-B", "concat_mask", "concat"),
                                         ("D-C", "position_gate", "concat_mask"), ("E-D", "attention_cds", "position_gate")):
                contrasts.append((f"{encoder}/{grid}/{label}", lookup(first), lookup(second)))
            for method in ("concat_mask", "position_gate", "attention_cds"):
                contrasts.append((f"{encoder}/{grid}/{method}/true-permuted", lookup(method), lookup(method, "permuted")))
        for method, mask in CORE:
            contrasts.append((f"{encoder}/{method}/{mask}/native-pooled",
                              by[(encoder, "native", method, mask)], by[(encoder, "codon_pool", method, mask)]))
    return contrasts


def saved_comparisons(root, *, dataset_key, draws, bootstrap_seed, producer, run_analysis=False):
    """Read or explicitly compute the complete three-seed core report from JSON fits."""
    directory = Path(root) / dataset_key
    preparation = read_preparation(directory / "preparation.json")
    panel = fit_panel()
    fits, evidence = {}, {}
    reference_budget = reference_settings = reference_implementation = None
    seen_encoders = {}
    for spec in panel:
        fits[spec.key] = []
        for seed in (0, 1, 2):
            path = directory / "fits" / spec.key / f"seed{seed}.json"
            result = read_fit(path, preparation=preparation)
            if result.config["seed"] != seed or result.extra_fields["identity"]["inputs"]["spec"] != asdict(spec):
                raise ValueError("Result name and requested comparison cell differ")
            identity = result.extra_fields["identity"]
            budget = {key: value for key, value in result.config.items() if key != "seed"}
            settings = identity["model"]["settings"]
            implementation = identity["model"].get("implementation_sha256")
            if not isinstance(implementation, dict) or not implementation:
                raise ValueError("Comparison cells require recorded implementation hashes")
            if reference_budget is None:
                reference_budget, reference_settings = budget, settings
                reference_implementation = implementation
            if budget != reference_budget or settings != reference_settings:
                raise ValueError("Comparison cells have different optimizer or architecture settings")
            if implementation != reference_implementation:
                raise ValueError("Comparison cells were produced by different implementation revisions")
            for key, cache in identity["inputs"]["encoders"].items():
                if key in seen_encoders and seen_encoders[key] != cache:
                    raise ValueError("Comparison cells use different cached representations for the same encoder")
                seen_encoders[key] = cache
            fits[spec.key].append(result)
            evidence[str(path.relative_to(directory))] = tokens_sha(path)
    requested = comparisons(panel)
    # Keep older complete-map report descriptions intact when reading that evidence.
    resampling = ("recorded homology groups and unmapped-gene groups"
                  if preparation["cohort"]["unmapped_gene_policy"] == "gene" else "homology groups")
    context = {"preparation_sha256": preparation["sha256"], "sources": evidence,
               "implementation_sha256": reference_implementation,
               "seeds": [0, 1, 2], "draws": draws, "bootstrap_seed": bootstrap_seed,
               "contrasts": [list(c) for c in requested], "metric": "Spearman",
               "resampling": f"{resampling}; average per-seed scores in each paired draw"}
    path = directory / "reports" / "three_seed_core.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("context") != context:
            raise ValueError("Existing report differs; preserve it and select a separate study destination")
        return saved
    if not run_analysis:
        return None
    if type(draws) is not int or draws < 1 or type(bootstrap_seed) is not int or not producer:
        raise ValueError("Explicit positive draw count, seed and producer evidence are required")
    first = next(iter(fits.values()))[0]
    truth = np.asarray(first.test_truth)
    # Check score/prediction agreement only when calculating a new report.
    for fit_key, results in fits.items():
        for result in results:
            if not np.array_equal(np.asarray(result.test_truth), truth):
                raise ValueError("Comparison fits have different ordered test targets")
            measured = train.score_rows("regression", truth, np.asarray(result.test_prediction))
            if not np.isfinite(measured) or not np.isclose(measured, result.test_score, rtol=1e-7, atol=1e-8):
                raise ValueError(
                    f"{fit_key} seed {result.config['seed']}: "
                    "saved score differs from its predictions or is undefined."
                )
    groups = np.asarray(preparation["cohort"]["groups"])[preparation["positions"]["test"]]
    predictions = {key: [np.asarray(result.test_prediction) for result in results] for key, results in fits.items()}
    intervals = train.seed_averaged_bootstrap("regression", truth, predictions, requested,
                                             groups=groups, draws=draws, seed=bootstrap_seed)
    # Undefined group draws are not silently dropped or converted to evidence.
    if any(not np.isfinite(row[key]) for row in intervals for key in ("difference", "ci_low", "ci_high")):
        raise ValueError("Bootstrap contains undefined statistics; no report was published")
    summary = []
    for spec in panel:
        values = [result.test_score for result in fits[spec.key]]
        summary.append({**asdict(spec), "seed_scores": values, "mean": float(np.mean(values)),
                        "seed_sd": float(np.std(values, ddof=1)),
                        "parameters": fits[spec.key][0].extra_fields["identity"]["model"]["parameter_counts"]})
    record = {"schema": "region-three-seed-report.v1", "context": context, "producer": producer,
              "summary": summary, "contrasts": intervals}
    inputs._publish_once(path, record)
    return record
