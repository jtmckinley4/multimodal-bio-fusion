"""Qualified recovery for BioLangFusion token caches written before shard checksums existed.

A new fit records the identity of everything it reads, and for a token cache that means the
ordered-input fingerprint and a SHA-256 for every shard file (``run_identity._cache_record``).
The benchmark caches extracted in October predate those fields: their ``meta.json`` carries the
encoder, checkpoint, revision, token limit, dtype, row count and shard count, and nothing that
binds the stored vectors to a set of inputs. Strict validation therefore refuses them, which is
what stops a Coupled Mamba fit on a machine holding only those caches.

This module is the same qualified route the GTEx pilot already uses
(``run_identity.gtex_recovered_cache_records``), written for the codon-grid benchmarks: an
inspection measures today's bytes and writes a separate record, and a later run may read that
record instead of an extraction manifest. The association it creates is explicitly a
reconstruction.

**What the inspection establishes.** The cache's shard inventory is complete and its files hash
to the recorded values; the metadata's encoder and row count match the prepared inputs; every
row holds the token count its coding-sequence length implies, through ``BenchmarkStreams.check``;
and the ordered-input fingerprint recomputed from the benchmark CSV is recorded as the
reconstructed input hash. Optionally, a completed Newton result for the same benchmark and
encoder can be supplied: its identity carries a verified ``input_sha256`` written at extraction
time on another machine, so agreement with the recomputed fingerprint is independent
corroboration that both caches describe the same ordered inputs under the same encoder settings.

**What it cannot establish.** Nothing here proves which program produced the stored vectors, or
that the declared checkpoint and revision are the ones that ran. A matching input fingerprint
says the inputs can be reconstructed, not that the extraction used them. Where that matters, the
honest route is re-extraction with the current writer, which records provenance as it goes.
Recovered caches are labeled ``historically_reconstructed`` wherever they are used, and a run
that reads one carries that label into its saved identity.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

from . import run_identity
from .blf import MODALITIES

RECOVERY_SCHEMA = "blf-cache-recovery.v1"
RECOVERY_STATUS = "historically_reconstructed"
# Extraction settings that must agree between a recovered cache and any corroborating record.
CORROBORATED_SETTINGS = ("encoder", "checkpoint", "revision", "max_len", "dtype", "n_sequences")


def _require(condition, reason):
    if not condition:
        raise ValueError(f"BLF cache recovery: {reason}; preserve the cache and its record.")


def _benchmark_streams(streams):
    """Reject panels this recovery was not written for before any file is read."""
    _require(not hasattr(streams, "tissues"),
             "the GTEx pilot has its own recovery route (run_identity.gtex_recovered_cache_records)")
    _require(set(streams.caches) == set(MODALITIES) and set(streams.encoders) == set(MODALITIES),
             "all three codon-grid caches are required")


def _ordered_inputs(streams):
    """The exact strings each encoder reads, in row order, rebuilt from the benchmark frame."""
    from ..encoders import ENCODERS
    from .token_embeddings import inputs_for_rows

    sequences = streams.rows["sequence"].tolist()
    return {modality: inputs_for_rows(ENCODERS[encoder], sequences)
            for modality, encoder in streams.encoders.items()}


def _legacy_metadata(directory):
    """Read a cache manifest and confirm it is the pre-checksum format this route accepts."""
    metadata_bytes = (directory / "meta.json").read_bytes()
    meta = json.loads(metadata_bytes)
    _require(isinstance(meta, dict), f"{directory}: cache metadata must be an object")
    _require("input_sha256" not in meta and "shards" not in meta,
             f"{directory} carries extraction provenance and must use strict validation")
    return metadata_bytes, meta


def _shard_names(directory, meta):
    count = meta.get("n_shards")
    _require(isinstance(count, int) and count >= 1, f"{directory}: shard count is missing")
    names = [f"shard_{index:05d}.npz" for index in range(count)]
    present = {path.name for path in directory.glob("shard_*.npz")}
    _require(present == set(names), f"{directory}: shard inventory differs from the manifest")
    return names


def _corroborating_caches(paths):
    """Collect verified extraction records from completed results, keyed by encoder.

    A completed fit's saved identity holds one record per modality under
    ``inputs.caches``; those written by the current extractor carry ``input_sha256``.
    Only those are usable here, and a disagreement between two records for the same
    encoder is reported rather than resolved.
    """
    found = {}
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        _require(isinstance(record, dict), f"{path}: corroborating result must be an object")
        caches = (record.get("identity") or {}).get("inputs", {}).get("caches")
        _require(isinstance(caches, dict) and caches,
                 f"{path}: no cache records in this result's identity")
        for cache in caches.values():
            encoder, fingerprint = cache.get("encoder"), cache.get("input_sha256")
            if not isinstance(encoder, str) or not isinstance(fingerprint, str):
                continue
            settings = {key: cache.get(key) for key in CORROBORATED_SETTINGS}
            settings["input_sha256"] = fingerprint
            previous = found.get(encoder)
            _require(previous is None or previous["settings"] == settings,
                     f"corroborating results disagree about {encoder}")
            found[encoder] = {"settings": settings, "source": Path(path).name}
    return found


def inspect_caches(streams, *, verification=None, corroborating_results=(), note=None):
    """Measure today's cache bytes and build the recovery record; write nothing yet.

    ``corroborating_results`` are completed result files whose saved identity carries
    extraction-time cache provenance, such as the returned Newton fits. When one covers a
    modality's encoder, its recorded input fingerprint and extraction settings must agree with
    this cache's; the agreement is recorded with the result it came from.
    """
    from .token_embeddings import CacheVerification, input_fingerprint

    _benchmark_streams(streams)
    # Binds every cache row to the benchmark frame: one row per sequence, with the token
    # count that row's coding-sequence length implies under its encoder's tokenization.
    streams.check()
    verification = CacheVerification() if verification is None else verification
    texts = _ordered_inputs(streams)
    corroboration = _corroborating_caches(corroborating_results)

    caches, corroborated = {}, {}
    for modality, encoder in streams.encoders.items():
        directory = Path(streams.caches[modality].cache_dir)
        metadata_bytes, meta = _legacy_metadata(directory)
        _require(meta.get("encoder") == encoder,
                 f"{modality}: cache metadata names {meta.get('encoder')!r}, not {encoder!r}")
        _require(meta.get("n_sequences") == len(texts[modality]),
                 f"{modality}: cache row count differs from the benchmark frame")
        fingerprint = input_fingerprint(texts[modality])
        names = _shard_names(directory, meta)
        shards, total, width = [], 0, None
        for name in names:
            shape, dtype = run_identity._token_header(directory / name)
            _require(dtype.name == meta.get("dtype"), f"{modality}/{name}: dtype differs")
            _require(width in (None, shape[1]), f"{modality}/{name}: token width differs")
            width, total = shape[1], total + shape[0]
            shards.append({"name": name, "sha256": verification.sha256(directory / name)})
        record = {"directory": directory.name, "benchmark": streams.key, "encoder": encoder,
                  "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                  "reconstructed_input_sha256": fingerprint,
                  "n_sequences": meta.get("n_sequences"), "width": int(width),
                  "total_tokens": int(total), "shards": shards}
        match = corroboration.get(encoder)
        if match is not None:
            settings = {key: meta.get(key) for key in CORROBORATED_SETTINGS}
            settings["input_sha256"] = fingerprint
            _require(match["settings"] == settings,
                     f"{modality}: {match['source']} records different extraction settings or "
                     "ordered inputs for this encoder")
            record["corroboration"] = {"source": match["source"],
                                       "agrees_on": [*CORROBORATED_SETTINGS, "input_sha256"]}
            corroborated[modality] = match["source"]
        caches[modality] = record

    measured = hashlib.sha256(json.dumps(caches, sort_keys=True).encode()).hexdigest()
    return {"schema": RECOVERY_SCHEMA, "status": RECOVERY_STATUS,
            "record_id": f"{streams.key}-{measured[:16]}",
            "benchmark": streams.key, "encoders": dict(streams.encoders),
            "rows": run_identity._rows_record(streams.rows, ("sequence", "length")),
            "corroborated_modalities": sorted(corroborated),
            "note": note or "Reconstructed association for pre-checksum token caches; "
                            "extraction-time producer evidence remains unknown.",
            "caches": caches}


def write_recovery_record(streams, path, *, verification=None, corroborating_results=(),
                          note=None):
    """Inspect the caches and save the record beside the results it will qualify."""
    record = inspect_caches(streams, verification=verification, note=note,
                            corroborating_results=corroborating_results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"{path} already exists; review it before replacing it")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def recovered_cache_records(streams, record_path, *, verification=None, modalities=None):
    """Admit a reviewed recovery record for legacy caches, without rewriting any manifest.

    The record binds today's exact bytes to a reconstructed input association. These checks do
    not recreate an extraction-time manifest or identify the producer.
    """
    from .token_embeddings import CacheVerification, input_fingerprint

    _benchmark_streams(streams)
    selected = set(MODALITIES if modalities is None else modalities)
    _require(bool(selected) and selected <= set(MODALITIES), "invalid recovery modality selection")
    raw_record = Path(record_path).read_bytes()
    record = json.loads(raw_record)
    _require(isinstance(record, dict), "record must be an object")
    _require(record.get("schema") == RECOVERY_SCHEMA and record.get("status") == RECOVERY_STATUS,
             "unsupported record contract")
    _require(isinstance(record.get("record_id"), str) and bool(record["record_id"]),
             "missing record ID")
    _require(streams.key == record.get("benchmark"), "record was written for another benchmark")
    _require(record.get("encoders") == dict(streams.encoders), "encoder selection differs")
    _require(record.get("rows") == run_identity._rows_record(streams.rows, ("sequence", "length")),
             "ordered rows differ from the record")
    recorded = record.get("caches")
    _require(isinstance(recorded, dict) and set(recorded) == set(MODALITIES),
             "cache records must name exactly the three modalities")
    streams.check()

    verification = CacheVerification() if verification is None else verification
    texts = _ordered_inputs(streams)
    provenance = {"status": RECOVERY_STATUS, "record_id": record["record_id"],
                  "record_sha256": hashlib.sha256(raw_record).hexdigest()}
    caches = {}
    for modality in sorted(selected):
        saved = recorded[modality]
        _require(isinstance(saved, dict), f"{modality} cache record must be an object")
        directory = Path(streams.caches[modality].cache_dir)
        _require(directory.name == saved.get("directory"), f"{modality} cache directory differs")
        metadata_bytes, meta = _legacy_metadata(directory)
        _require(hashlib.sha256(metadata_bytes).hexdigest() == saved.get("metadata_sha256"),
                 f"{modality} metadata bytes differ")
        fingerprint = input_fingerprint(texts[modality])
        _require(saved.get("reconstructed_input_sha256") == fingerprint,
                 f"{modality} reconstructed inputs differ")
        shards = saved.get("shards")
        _require(isinstance(shards, list) and len(shards) == meta.get("n_shards"),
                 f"{modality} shard records are missing or incomplete")
        # Only this in-memory copy gains the reconstructed association and measured hashes.
        cache = run_identity._cache_record(
            directory, {**meta, "input_sha256": fingerprint, "shards": shards}, verification)
        _require({path.name for path in directory.glob("shard_*.npz")}
                 == {shard["name"] for shard in shards}, f"{modality} shard inventory differs")
        total = 0
        for shard in shards:
            shape, dtype = run_identity._token_header(directory / shard["name"])
            _require(shape[1] == saved.get("width") and dtype.name == meta.get("dtype"),
                     f"{modality} shard shape or dtype differs")
            total += shape[0]
        _require(total == saved.get("total_tokens"), f"{modality} total token count differs")
        entry = dict(provenance)
        if isinstance(saved.get("corroboration"), dict):
            entry["corroboration"] = saved["corroboration"]
        caches[modality] = {**cache, "input_provenance": entry}
    run_identity._verify_cache_inputs({m: streams.encoders[m] for m in caches}, caches,
                                      {m: texts[m] for m in caches})
    warnings.warn("Using a reviewed historical reconstruction for legacy BioLangFusion caches; "
                  "their current bytes are verified, while extraction-time input proof and "
                  "producer details remain unknown.", UserWarning, stacklevel=2)
    return caches


def cache_records(streams, *, verification=None, recovery=None):
    """Cache records for a benchmark, with the qualified route only where it is needed.

    Caches carrying extraction provenance always validate strictly; present-but-invalid
    provenance is an error, never a reason to fall back to recovery.
    """
    if recovery is None:
        return {modality: run_identity.cache_record(cache.cache_dir, verification=verification)
                for modality, cache in streams.caches.items()}
    _benchmark_streams(streams)
    caches, historical = {}, []
    for modality, cache in streams.caches.items():
        directory = Path(cache.cache_dir)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        _require(isinstance(meta, dict), f"{directory}: cache metadata must be an object")
        if "input_sha256" in meta or "shards" in meta:
            caches[modality] = run_identity._cache_record(directory, meta, verification)
        else:
            historical.append(modality)
    if historical:
        # Refuse invalid fresh inputs before consulting the separate historical record.
        texts = _ordered_inputs(streams)
        run_identity._verify_cache_inputs({m: streams.encoders[m] for m in caches}, caches,
                                          {m: texts[m] for m in caches})
        caches.update(recovered_cache_records(streams, recovery, verification=verification,
                                              modalities=historical))
    return caches


def identity_for_streams(streams, variant, config, *, model_kwargs=None, verification=None,
                         recovery=None):
    """``run_identity.identity_for_streams`` for codon-grid streams with qualified caches."""
    if recovery is None:
        return run_identity.identity_for_streams(streams, variant, config,
                                                 model_kwargs=model_kwargs,
                                                 verification=verification)
    caches = cache_records(streams, verification=verification, recovery=recovery)
    return run_identity.blf_identity(
        rows=streams.rows, benchmark=streams.key, variant=variant, config=config,
        encoders=streams.encoders, caches=caches, task=streams.task,
        label_column=streams.label_column, split_column=streams.split_column,
        model_kwargs=model_kwargs)


def execution_records(streams, variant, config, results_dir, *, tag=None, model_kwargs=None,
                      verification=None, repository=None, device=None, recovery=None):
    """``run_identity.execution_records`` with the qualified cache route for new fits.

    Completed results keep the trainer's existing checks and are never relabeled; only a new
    fit consults the recovery record, and its identity then carries the reconstruction label.
    """
    if recovery is None:
        return run_identity.execution_records(streams, variant, config, results_dir, tag=tag,
                                              model_kwargs=model_kwargs,
                                              verification=verification,
                                              repository=repository, device=device)
    tag = tag or (f"{streams.key}__{variant.replace(':', '-')}__lambda{config.entropy_lambda}"
                  f"__ch{config.channels}__{config.tconv}__seed{config.seed}")
    path = Path(results_dir) / f"{tag}.json"
    if path.exists():
        saved = run_identity.execution_records(streams, variant, config, results_dir, tag=tag,
                                               model_kwargs=model_kwargs,
                                               verification=verification,
                                               repository=repository, device=device)
        if saved["run_identity"] is None:
            return saved
    return {"run_identity": identity_for_streams(streams, variant, config,
                                                 model_kwargs=model_kwargs,
                                                 verification=verification, recovery=recovery),
            "producer": run_identity.producer_record(device=device, repository=repository)}
