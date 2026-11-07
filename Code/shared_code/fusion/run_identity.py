"""Scientific identity for new fusion fits, separate from their execution provenance.

These records bind a fit to ordered inputs, targets, splits and cache contents. They
do not turn historical results into verified new runs or make hardware interchangeable.
Construction reads metadata and array headers; it never materializes token arrays.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import warnings
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = "fusion.run.v1"
SPLITS = ("train", "val", "test")
MODALITIES = ("dna", "rna", "prot")


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _config(config):
    return _json(asdict(config) if is_dataclass(config) else config)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def array_record(values, dtype=None):
    """Fingerprint actual ordered numeric values, including shape and dtype."""
    array = np.asarray(values, dtype=dtype)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError("Identity arrays must contain finite numeric values.")
    record = {"shape": list(array.shape), "dtype": array.dtype.str}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return {**record, "sha256": digest.hexdigest()}


def _rows_record(rows, columns):
    # Explicit columns avoid binding the experiment to an unused display/index column.
    values = rows[list(columns)].astype(str).to_numpy().tolist()
    return {"columns": list(columns), "count": len(values), "sha256": _digest(values)}


def cache_record(cache_dir, *, verification=None):
    """Read a complete cache manifest and its width without opening token arrays.

    Without ``verification``, SHA values are recorded evidence, not a fresh content
    check. Pass the extraction session's CacheVerification to verify once per file
    stat signature. Unknown historical revisions and input provenance stay unknown.
    """
    directory = Path(cache_dir)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return _cache_record(directory, meta, verification)


def _token_header(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("tokens.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:
                raise ValueError(f"Unsupported token-array header version: {version}")
            shape, _, dtype = reader(stream)
    if len(shape) != 2 or shape[1] < 1:
        raise ValueError(f"{path}: invalid token-array dimensions.")
    return shape, dtype


def _cache_record(directory, meta, verification):
    """Validate metadata supplied in memory; never update an extraction manifest."""
    if not isinstance(meta, dict):
        raise ValueError(f"{directory}: cache metadata must be an object.")
    count = meta.get("n_shards")
    if not isinstance(count, int) or count < 1:
        raise ValueError(f"{directory}: a complete, nonempty shard manifest is required.")
    entries = meta.get("shards", [])
    if isinstance(entries, dict):
        shas = dict(entries)
    elif isinstance(entries, list) and all(
            isinstance(entry, dict) and {"name", "sha256"} <= entry.keys() for entry in entries):
        shas = {entry["name"]: entry["sha256"] for entry in entries}
    else:
        raise ValueError(f"{directory}: malformed shard checksum records.")
    names = [f"shard_{index:05d}.npz" for index in range(count)]
    if set(shas) != set(names):
        raise ValueError(f"{directory}: missing or unexpected shard checksums.")
    for name in names:
        expected = shas[name]
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"{directory / name}: invalid SHA-256 record.")
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
        if verification is not None and verification.sha256(directory / name) != expected:
            raise ValueError(f"{directory / name}: contents differ from the cache manifest.")
    shape, dtype = _token_header(directory / names[0])
    keys = ("shard_version", "encoder", "checkpoint", "revision", "max_len",
            "dtype", "n_sequences", "chunk_size", "n_shards", "input_sha256")
    return {**{key: meta.get(key) for key in keys}, "width": int(shape[1]),
            "array_dtype": dtype.str,
            "shards": [{"name": name, "sha256": shas[name]} for name in names]}


def make_identity(*, benchmark, variant, config, inputs, model, loader):
    return validate_identity({"schema": SCHEMA, "benchmark": benchmark,
                              "variant": variant, "config": _config(config),
                              "inputs": inputs, "model": model, "loader": loader})


def validate_identity(identity, *, config=None, variant=None, benchmark=None):
    """Validate a strict new identity, without interpreting a legacy minimal record."""
    made = _json(identity)
    if not isinstance(made, dict) or made.get("schema") != SCHEMA:
        raise ValueError(f"Expected identity schema {SCHEMA!r}.")
    for field in ("config", "inputs", "model", "loader"):
        if not isinstance(made.get(field), dict):
            raise ValueError(f"Identity {field} must be an object.")
    for field in ("variant", "benchmark"):
        if not isinstance(made.get(field), str) or not made[field]:
            raise ValueError(f"Identity {field} must be a nonempty string.")
    if not isinstance(made["inputs"].get("encoders"), dict):
        raise ValueError("Identity inputs.encoders must be an object.")
    if not made["model"].get("contract"):
        raise ValueError("Identity model requires an explicit computation contract.")
    recovery = made["loader"].get("recovery")
    if made["loader"].get("kind") not in ("blf", "isoformer", "custom"):
        raise ValueError("Unknown loader identity kind.")
    if not isinstance(recovery, dict) or set(recovery) != set(SPLITS):
        raise ValueError("Loader recovery must declare train, val and test components.")
    for components in recovery.values():
        if not isinstance(components, dict):
            raise ValueError("Loader recovery components must be objects.")
        for component in components.values():
            if not isinstance(component, dict):
                raise ValueError("Loader recovery component must be an object.")
            fields = component.get("fields")
            if (not isinstance(component.get("type"), str) or not isinstance(fields, list)
                    or not fields or fields != sorted(set(fields))
                    or not set(fields) <= {"generator", "epoch", "state_dict"}):
                raise ValueError("Invalid loader recovery component contract.")
    for name, expected in (("config", _config(config) if config is not None else None),
                           ("variant", variant), ("benchmark", benchmark)):
        if expected is not None and made[name] != expected:
            raise ValueError(f"Identity {name} differs from the requested run.")
    return made


def blf_loader_spec(config, modalities):
    config = _config(config)
    generator = {"type": "torch.utils.data.dataloader.DataLoader", "fields": ["generator"]}
    sampler = {"type": "torch.utils.data.sampler.RandomSampler", "fields": ["generator"]}
    return {"kind": "blf", "batch_size": config["batch_size"], "seed": config["seed"],
            "modalities": list(modalities), "num_workers": 0, "drop_last": False,
            "recovery": {"train": {"loader": generator, "sampler": sampler,
                                    "batch_sampler.sampler": sampler}, "val": {}, "test": {}}}


def isoformer_loader_spec(*, batch_size=16, val_fraction=0.05, seed=0,
                         budget=300_000_000, n_heads=8):
    component = {"type": "shared_code.fusion.isoformer.BudgetedBatches", "fields": ["epoch"]}
    return {"kind": "isoformer", "batch_size": batch_size, "val_fraction": val_fraction,
            "seed": seed, "budget": budget, "n_heads": n_heads,
            "num_workers": 0, "drop_last": False,
            "recovery": {split: {"batch_sampler": component} for split in SPLITS}}


def _inputs(rows, columns, labels, encoders, caches, splits, target_columns, task):
    if set(splits) != set(SPLITS):
        raise ValueError("Provide actual train, val and test row indices.")
    return {"encoders": dict(encoders), "rows": _rows_record(rows, columns),
            "targets": {**array_record(labels), "columns": list(target_columns),
                        "task": task, "transform": "as_supplied"},
            "splits": {name: array_record(indices, np.int64) for name, indices in splits.items()},
            "split_targets": {name: array_record(labels[np.asarray(indices, dtype=np.int64)])
                              for name, indices in splits.items()},
            "caches": caches}


def _verify_cache_inputs(encoders, caches, texts):
    from .token_embeddings import input_fingerprint
    if set(encoders) != set(caches) or set(encoders) != set(texts):
        raise ValueError("Encoder, cache and prepared-input modalities must match.")
    for modality, inputs in texts.items():
        cache = caches[modality]
        expected = input_fingerprint(inputs)
        if cache.get("input_sha256") != expected:
            raise ValueError(f"{modality}: cache ordered-input provenance is missing or differs; "
                             "preserve the cache and recover its recorded provenance before a new fit.")
        if cache.get("encoder") != encoders[modality] or cache.get("n_sequences") != len(inputs):
            raise ValueError(f"{modality}: cache encoder or row count differs from the prepared inputs.")


def blf_identity(*, rows, benchmark, variant, config, encoders, caches,
                 model_kwargs=None, label_column="Value", split_column="Split",
                 task="regression"):
    """Identity from prepared benchmark rows and metadata-only token-cache records."""
    config = _config(config)
    from ..encoders import ENCODERS
    from .token_embeddings import inputs_for_rows
    texts = {m: inputs_for_rows(ENCODERS[key], rows["sequence"].tolist())
             for m, key in encoders.items()}
    _verify_cache_inputs(encoders, caches, texts)
    kwargs = dict(model_kwargs or {})
    modalities = (variant.split(":", 1)[1],) if variant.startswith("unimodal") else MODALITIES
    defaults = {"channels": config["channels"]}
    if not variant.startswith("unimodal:"):
        defaults["tconv"] = config["tconv"]
    if variant in ("concat", "mil", "cross"):
        defaults["out_width"] = 600
        defaults.update({"concat": {"dna_width": 600},
                         "mil": {"attention_width": 100, "shared_projection": False},
                         "cross": {"n_heads": 4, "dropout": 0.1}}[variant])
    if set(kwargs) - set(defaults):
        raise ValueError(f"Unsupported BLF model settings: {sorted(set(kwargs) - set(defaults))}")
    defaults.update(kwargs)
    labels = rows[label_column].to_numpy(dtype=np.int64 if task == "classification" else np.float32)
    splits = {name: np.flatnonzero(rows[split_column].to_numpy() == name) for name in SPLITS}
    inputs = _inputs(rows, ("sequence", "length"), labels, encoders, caches, splits,
                     (label_column,), task)
    model = {"contract": "blf-frozen-codon-grid-v1", "builder": "shared_code.fusion.blf.build",
             "kwargs": defaults, "widths": {m: c["width"] for m, c in caches.items()},
             "modalities": list(modalities),
             "n_outputs": int(labels.max()) + 1 if task == "classification" else 1,
             "textcnn": {"kernel_sizes": [3, 4, 5], "dropout": 0.2},
             "numerics": {"inputs": "float32", "parameters": "float32", "amp": False}}
    return make_identity(benchmark=benchmark, variant=variant, config=config, inputs=inputs,
                         model=model, loader=blf_loader_spec(config, modalities))


def _gtex_inputs(rows, encoders, crop):
    # Match the Stage 3 pilot's extraction: all DNA, RNA's last 12 kb, first 1,022 residues.
    # Derive texts from these actual rows even when a caller supplies different crop limits.
    from ..encoders import ENCODERS, encoder_input, protein_input
    crop = {"dna": None, "rna": 12000, "prot": 1022} if crop is None else dict(crop)
    if set(crop) != set(MODALITIES) or any(
            value is not None and (type(value) is not int or value < 1) for value in crop.values()):
        raise ValueError("GTEx crop must declare each modality's positive limit or None.")
    input_texts = {}
    for modality, column in {"dna": "DNA", "rna": "RNA", "prot": "Protein"}.items():
        raw = rows[column].astype(str).str.strip().str.upper()
        limit = crop[modality]
        kept = raw if limit is None else raw.str[-limit:] if modality == "rna" else raw.str[:limit]
        encoder = ENCODERS[encoders[modality]]
        prepare = protein_input if modality == "prot" else encoder_input
        input_texts[modality] = [prepare(encoder, value) for value in kept]
    return crop, input_texts


def isoformer_identity(*, rows, benchmark, variant, config, encoders, caches, tissues,
                       split_indices, loader_settings, model_kwargs=None, crop=None):
    crop, input_texts = _gtex_inputs(rows, encoders, crop)
    _verify_cache_inputs(encoders, caches, input_texts)
    defaults = {"n_layers": 1, "n_heads": 8, "dropout": 0.1}
    kwargs = dict(model_kwargs or {})
    if set(kwargs) - set(defaults):
        raise ValueError(f"Unsupported IsoFormer model settings: {sorted(set(kwargs) - set(defaults))}")
    defaults.update(kwargs)
    inputs = _inputs(rows, ("DNA", "RNA", "Protein"),
                     rows[list(tissues)].to_numpy(dtype=np.float32), encoders, caches,
                     split_indices, tissues, "expression")
    inputs["preprocessing"] = {"crop": crop, "case": "strip_upper", "rna_crop": "keep_3prime"}
    model = {"contract": "isoformer-frozen-pilot-v1", "builder": "shared_code.fusion.isoformer.builder",
             "kwargs": defaults, "widths": {m: c["width"] for m, c in caches.items()},
             "modalities": variant.split("+"), "n_outputs": len(tissues),
             "numerics": {"inputs": "float32", "parameters": "float32", "amp": False}}
    return make_identity(benchmark=benchmark, variant=variant, config=config, inputs=inputs,
                         model=model, loader=isoformer_loader_spec(**loader_settings))


def _component_contract(loader):
    sampler, batch = getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)
    components = {"loader": loader, "sampler": sampler, "batch_sampler": batch,
                  "batch_sampler.sampler": getattr(batch, "sampler", None)}
    made = {}
    for name, component in components.items():
        if component is None:
            continue
        fields = []
        generator = getattr(component, "generator", None)
        if generator is not None and callable(getattr(generator, "get_state", None)):
            fields.append("generator")
        if isinstance(getattr(component, "epoch", None), int):
            fields.append("epoch")
        if (callable(getattr(component, "state_dict", None))
                and callable(getattr(component, "load_state_dict", None))):
            fields.append("state_dict")
        if fields:
            kind = type(component)
            made[name] = {"type": f"{kind.__module__}.{kind.__qualname__}", "fields": sorted(fields)}
    return made


def validate_loaders(identity, splits):
    """Check actual loader contracts and row membership before iteration or restore."""
    identity = validate_identity(identity)
    spec = identity["loader"]
    if set(splits) != set(SPLITS):
        raise ValueError("Actual loaders must provide train, val and test.")
    for name, loader in splits.items():
        if _component_contract(loader) != spec["recovery"][name]:
            raise ValueError(f"{name} loader recovery components differ from the run identity.")
        if spec["kind"] == "custom":
            continue
        expected_module = f"shared_code.fusion.{spec['kind']}"
        expected_dataset = "GridDataset" if spec["kind"] == "blf" else "ExpressionDataset"
        dataset_type = type(loader.dataset)
        collate = loader.collate_fn
        if (dataset_type.__module__ != expected_module
                or dataset_type.__qualname__ != expected_dataset
                or collate.__module__ != expected_module or collate.__qualname__ != "collate"):
            raise ValueError(f"{name} dataset/collation implementation differs from the run identity.")
        for key in ("num_workers", "drop_last"):
            if getattr(loader, key) != spec[key]:
                raise ValueError(f"{name} loader {key} differs from the run identity.")
        if array_record(loader.dataset.indices, np.int64) != identity["inputs"]["splits"][name]:
            raise ValueError(f"{name} loader row order differs from the run identity.")
        if array_record(loader.dataset.labels) != identity["inputs"]["split_targets"][name]:
            raise ValueError(f"{name} loader targets differ from the run identity.")
        if spec["kind"] == "blf":
            if (loader.batch_size != spec["batch_size"]
                    or list(loader.dataset.modalities) != spec["modalities"]):
                raise ValueError(f"{name} loader batch size/modalities differ from the run identity.")
            expected_sampler = "RandomSampler" if name == "train" else "SequentialSampler"
            if (type(loader.sampler).__module__ != "torch.utils.data.sampler"
                    or type(loader.sampler).__qualname__ != expected_sampler):
                raise ValueError(f"{name} sampling order differs from the run identity.")
            if name == "train" and loader.generator.initial_seed() != spec["seed"]:
                raise ValueError("Training loader seed differs from the run identity.")
        else:
            sampler = loader.batch_sampler
            for key in ("budget", "max_size", "n_heads", "seed", "shuffle"):
                expected = {"max_size": spec["batch_size"], "shuffle": name == "train"}.get(
                    key, spec.get(key))
                if getattr(sampler, key, None) != expected:
                    raise ValueError(f"{name} loader {key} differs from the run identity.")
    return identity


def validate_model_request(identity, builder=None, model_kwargs=None):
    """Reject a caller's changed builder/settings before constructing a model."""
    identity = validate_identity(identity)
    kind = identity["loader"]["kind"]
    if kind == "custom":
        return identity
    import torch
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("The built-in model identity requires the float32 default parameter dtype.")
    expected_builder = identity["model"]["builder"]
    actual_builder = (f"{builder.__module__}.{builder.__qualname__}" if builder is not None
                      else "shared_code.fusion.blf.build")
    if actual_builder != expected_builder:
        raise ValueError("Requested model builder differs from the run identity.")
    kwargs = dict(model_kwargs or {})
    if kind == "blf":
        variant, config = identity["variant"], identity["config"]
        defaults = {"channels": config["channels"]}
        if not variant.startswith("unimodal:"):
            defaults["tconv"] = config["tconv"]
        if variant in ("concat", "mil", "cross"):
            defaults["out_width"] = 600
            defaults.update({"concat": {"dna_width": 600},
                             "mil": {"attention_width": 100, "shared_projection": False},
                             "cross": {"n_heads": 4, "dropout": 0.1}}[variant])
    else:
        defaults = {"n_layers": 1, "n_heads": 8, "dropout": 0.1}
    if set(kwargs) - set(defaults):
        raise ValueError("Requested model settings are outside the recorded model contract.")
    defaults.update(kwargs)
    if defaults != identity["model"]["kwargs"]:
        raise ValueError("Requested effective model settings differ from the run identity.")
    return identity


def gtex_recovered_cache_records(streams, record_path, *, crop=None, verification=None,
                                 modalities=None):
    """Admit only the explicitly reviewed GTEx pilot reconstruction, without rewriting it.

    The record binds today's exact bytes to a qualified historical reconstruction.
    These checks do not recreate an extraction-time manifest or resolve unknown producers.
    """
    from .isoformer import STREAM_ENCODER
    from .token_embeddings import CacheVerification, input_fingerprint

    def require(condition, reason):
        if not condition:
            raise ValueError(f"GTEx recovery: {reason}; preserve the cache and recovery record.")

    selected = set(MODALITIES if modalities is None else modalities)
    require(bool(selected) and selected <= set(MODALITIES), "invalid recovery modality selection")
    raw_record = Path(record_path).read_bytes()
    record = json.loads(raw_record)
    require(isinstance(record, dict), "record must be an object")
    require(record.get("schema") == "gtex-pilot-recovery.v1"
            and record.get("status") == "historically_reconstructed", "unsupported record contract")
    require(isinstance(record.get("record_id"), str) and bool(record["record_id"]), "missing record ID")
    require(streams.key == record.get("benchmark") == "gtex_pilot", "only gtex_pilot is supported")
    require(record.get("encoders") == STREAM_ENCODER, "encoder selection differs")
    require(set(streams.caches) == set(MODALITIES), "all three pilot caches are required")
    crop, texts = _gtex_inputs(streams.rows, STREAM_ENCODER, crop)
    require(record.get("crop") == crop, "crop settings differ")
    require(record.get("rows") == _rows_record(streams.rows, ("DNA", "RNA", "Protein", "split")),
            "ordered rows or supplied split membership differ")
    require(record.get("tissues") == list(streams.tissues), "tissue order differs")
    require(record.get("targets") == array_record(
                streams.rows[list(streams.tissues)].to_numpy(dtype=np.float32)),
            "ordered targets differ")
    recorded_caches = record.get("caches")
    require(isinstance(recorded_caches, dict) and set(recorded_caches) == set(MODALITIES),
            "cache records must name exactly the three pilot modalities")
    verification = CacheVerification() if verification is None else verification
    provenance = {"status": "historically_reconstructed", "record_id": record["record_id"],
                  "record_sha256": hashlib.sha256(raw_record).hexdigest()}
    caches = {}
    for modality, encoder in STREAM_ENCODER.items():
        if modality not in selected:
            continue
        saved = recorded_caches[modality]
        require(isinstance(saved, dict), f"{modality} cache record must be an object")
        directory = Path(streams.caches[modality].cache_dir)
        name = f"{modality}_{encoder}"
        require(directory.name == saved.get("directory") == name
                and directory.parent.name == "gtex_pilot", f"{modality} cache directory differs")
        metadata_bytes = (directory / "meta.json").read_bytes()
        require(hashlib.sha256(metadata_bytes).hexdigest() == saved.get("metadata_sha256"),
                f"{modality} metadata bytes differ")
        meta = json.loads(metadata_bytes)
        require(isinstance(meta, dict), f"{modality} metadata must be an object")
        require("input_sha256" not in meta and "shards" not in meta,
                f"{modality} has modern provenance fields and must use strict validation")
        input_sha = input_fingerprint(texts[modality])
        require(saved.get("reconstructed_input_sha256") == input_sha,
                f"{modality} reconstructed inputs differ")
        shards = saved.get("shards")
        require(isinstance(shards, list), f"{modality} shard records are missing")
        require(len(shards) == meta.get("n_shards"), f"{modality} shard count differs")
        # Only this in-memory copy gains the reconstructed association and measured hashes.
        cache = _cache_record(directory, {**meta, "input_sha256": input_sha, "shards": shards},
                              verification)
        require({path.name for path in directory.glob("shard_*.npz")}
                == {shard["name"] for shard in shards}, f"{modality} shard inventory differs")
        total = 0
        for shard in shards:
            shape, dtype = _token_header(directory / shard["name"])
            require(shape[1] == saved.get("width") and dtype.name == meta.get("dtype"),
                    f"{modality} shard shape or dtype differs")
            total += shape[0]
        require(total == saved.get("total_tokens"), f"{modality} total token count differs")
        caches[modality] = {**cache, "input_provenance": provenance}
    _verify_cache_inputs({m: STREAM_ENCODER[m] for m in selected}, caches,
                         {m: texts[m] for m in selected})
    warnings.warn("Using the reviewed GTEx historical reconstruction for legacy caches; their "
                  "current bytes are verified, while extraction-time input proof and producer "
                  "details remain unknown.",
                  UserWarning, stacklevel=2)
    return caches


def identity_for_streams(streams, variant, config, *, loaders=None, model_kwargs=None,
                         verification=None, crop=None, gtex_recovery=None):
    """Adapter for already opened streams; metadata-only planners use the builders above."""
    if gtex_recovery is not None:
        caches, historical = {}, []
        for name, cache in streams.caches.items():
            directory = Path(cache.cache_dir)
            metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"{directory}: cache metadata must be an object.")
            # Present-but-invalid provenance is an error, never a reason to use recovery.
            if "input_sha256" in metadata or "shards" in metadata:
                caches[name] = _cache_record(directory, metadata, verification)
            else:
                historical.append(name)
        if historical:
            from .isoformer import STREAM_ENCODER
            if streams.key != "gtex_pilot" or not hasattr(streams, "tissues"):
                raise ValueError("Historical cache recovery is restricted to the GTEx pilot.")
            # Refuse invalid fresh inputs before consulting the separate historical record.
            _, texts = _gtex_inputs(streams.rows, STREAM_ENCODER, crop)
            _verify_cache_inputs({m: STREAM_ENCODER[m] for m in caches}, caches,
                                 {m: texts[m] for m in caches})
            caches.update(gtex_recovered_cache_records(
                streams, gtex_recovery, crop=crop, verification=verification,
                modalities=historical))
    else:
        caches = {name: cache_record(cache.cache_dir, verification=verification)
                  for name, cache in streams.caches.items()}
    if hasattr(streams, "tissues"):
        if loaders is None:
            raise ValueError("IsoFormer identity needs the actual prepared split loaders.")
        from .isoformer import STREAM_ENCODER
        sampler = loaders["train"].batch_sampler
        # The actual ordered indices, rather than this descriptive ratio, bind validation.
        n_train = len(loaders["train"].dataset) + len(loaders["val"].dataset)
        settings = {"batch_size": sampler.max_size, "budget": sampler.budget,
                    "n_heads": sampler.n_heads, "seed": sampler.seed,
                    "val_fraction": len(loaders["val"].dataset) / n_train}
        return isoformer_identity(rows=streams.rows, benchmark=streams.key, variant=variant,
                                  config=config, encoders=STREAM_ENCODER, caches=caches,
                                  tissues=streams.tissues,
                                  split_indices={k: v.dataset.indices for k, v in loaders.items()},
                                  loader_settings=settings, model_kwargs=model_kwargs,
                                  crop=crop)
    return blf_identity(rows=streams.rows, benchmark=streams.key, variant=variant,
                        config=config, encoders=streams.encoders, caches=caches,
                        task=streams.task, label_column=streams.label_column,
                        split_column=streams.split_column, model_kwargs=model_kwargs)


def execution_records(streams, variant, config, results_dir, *, tag=None, loaders=None,
                      model_kwargs=None, verification=None, crop=None,
                      repository=None, device=None, gtex_recovery=None):
    """Opt production callers into strict new fits while preserving legacy completed reads.

    Legacy results keep the trainer's existing limited checks and are never relabeled.
    A legacy checkpoint is different: it must satisfy strict preflight before any new fit.
    """
    tag = tag or (f"{streams.key}__{variant.replace(':', '-')}__lambda{config.entropy_lambda}"
                  f"__ch{config.channels}__{config.tconv}__seed{config.seed}")
    path = Path(results_dir) / f"{tag}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}: completed result must be an object; preserve this file.")
        saved_identity = record.get("identity")
        if saved_identity is None:
            saved_identity = {}
        if not isinstance(saved_identity, dict):
            raise ValueError(f"{path.name}: malformed saved identity; preserve this result.")
        if "schema" not in saved_identity:
            warnings.warn(f"{path.name}: historical completed result; full input identity is "
                          "unknown. Reusing under the existing legacy checks without relabeling.",
                          UserWarning, stacklevel=2)
            return {"run_identity": None, "producer": None}
        validate_identity(saved_identity)
    return {"run_identity": identity_for_streams(
                streams, variant, config, loaders=loaders, model_kwargs=model_kwargs,
                verification=verification, crop=crop, gtex_recovery=gtex_recovery),
            "producer": producer_record(device=device, repository=repository)}


def producer_record(*, device=None, repository=None, packages=None):
    """Observed execution context; never part of scientific equality checks."""
    versions = {}
    for package in ("numpy", "torch") if packages is None else packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = {"commit": None, "dirty": None}
    if repository is not None:
        try:
            git["commit"] = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
            git["dirty"] = bool(subprocess.check_output(
                ["git", "--no-optional-locks", "-C", str(repository), "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    sources = {}
    if repository is not None:
        root = Path(repository)
        for name in ("train", "blf", "isoformer", "heads", "codon_grid", "textcnn", "run_identity"):
            path = root / "Code" / "shared_code" / "fusion" / f"{name}.py"
            if path.is_file():
                sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"host": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "device": str(device) if device is not None else None, "packages": versions,
            "git": git, "source_sha256": sources}
