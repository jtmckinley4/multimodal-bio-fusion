"""Existing benchmark inputs for the controlled fusion studies.

This is the shared data boundary, not an experiment scheduler. It opens only
verified existing caches, preserves CSV order and requires the separate grouped
stability split. The notebooks own model panels and execution switches.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np

from .. import datasets
from . import blf, grouped_stability, run_identity, token_embeddings


def benchmark_rows(benchmark, data_root, *, stability_split=None):
    """Load the retained cohort; never construct or repair a research partition."""
    if benchmark not in datasets.BLF_BENCHMARKS:
        raise ValueError("Choose one of the existing BLF benchmarks.")
    source = Path(data_root) / datasets.DATASETS[benchmark].path
    rows = datasets.blf_benchmark_rows(benchmark, data_dir=data_root)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    split_column = "Split"
    split_identity = {"policy": "supplied", "source_sha256": source_sha}
    if benchmark == "mrna_stability":
        if stability_split is None:
            raise ValueError("The controlled stability study requires its saved grouped split.")
        record = grouped_stability.read_split(stability_split, split_seed=42)
        if (record["source_sha256"] != source_sha
                or record["source_indices"] != rows.index.tolist()):
            raise ValueError("Grouped stability source or retained row order differs.")
        row_values = rows[["sequence", "length"]].astype(str).to_numpy().tolist()
        if (record["rows"]["sha256"] != grouped_stability._digest(row_values)
                or record["targets"]["sha256"] != run_identity.array_record(
                    rows["Value"].to_numpy(dtype=np.float32))["sha256"]):
            raise ValueError("Grouped stability rows or targets differ from the saved allocation.")
        split_column = grouped_stability.SPLIT_COLUMN
        rows[split_column] = record["partitions"]
        split_identity = {"policy": record["protocol"], "source_sha256": source_sha,
                          "split_fingerprint": record["fingerprint"]}
    groups = np.asarray([
        grouped_stability.canonical_sequence(value) for value in rows["sequence"]
    ])
    positions = {name: np.flatnonzero(rows[split_column].to_numpy() == name)
                 for name in ("train", "val", "test")}
    if any(not len(indices) for indices in positions.values()):
        raise ValueError("Every outer partition must contain observations.")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(groups[positions[first]]) & set(groups[positions[second]]):
            raise ValueError(f"Exact sequence groups overlap {first}/{second}; review this cohort.")
    return rows, split_column, groups, positions, split_identity


@contextmanager
def open_streams(benchmark, encoder_mapping, data_root, cache_root, *,
                 stability_split=None, verification=None):
    """Open a compatible three-stream cohort and release every owned cache."""
    rows, split_column, groups, positions, split_identity = benchmark_rows(
        benchmark, data_root, stability_split=stability_split,
    )
    verification = verification or token_embeddings.CacheVerification()
    caches, records = {}, {}
    streams = blf.BenchmarkStreams(
        benchmark, rows, caches, datasets.DATASETS[benchmark].task,
        split_column=split_column, encoders=dict(encoder_mapping),
    )
    primary_error = None
    try:
        if set(encoder_mapping) != set(blf.MODALITIES):
            raise ValueError("Name exactly the DNA, RNA and protein encoders.")
        for modality, encoder in encoder_mapping.items():
            path = Path(cache_root) / benchmark / encoder
            records[modality] = run_identity.cache_record(path, verification=verification)
            caches[modality] = token_embeddings.summarize(path)
        streams.check()
        # Constructing the identity checks ordered encoder inputs against the cache
        # manifests before any predictor is built. It does not fit a model.
        from .train import TrainConfig
        run_identity.blf_identity(
            rows=rows, benchmark=benchmark, variant="concat", config=TrainConfig(),
            encoders=encoder_mapping, caches=records, task=streams.task,
            split_column=split_column,
        )
        yield streams, records, groups, positions, split_identity
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            streams.close()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"Study cache cleanup also failed: {cleanup_error}")
