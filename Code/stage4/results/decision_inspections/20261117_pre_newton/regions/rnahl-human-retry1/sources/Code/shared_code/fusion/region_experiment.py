"""Inputs and cache-backed batches for A6's full-transcript RNA/protein study.

Chunking retains every supplied position but does not give an encoder context
across chunk boundaries. Dataset acquisition, grouping policy, window sizes and
execution mode remain explicit caller choices. Nothing runs at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .. import mrnabench
from ..encoders import ENCODERS
from . import region_inputs, token_embeddings as tokens
from .run_identity import array_record

DATASET_KEYS = ("rnahl-human", "translation-efficiency-human")
ENCODER_KEYS = ("rnafm", "rinalmo", "esm2_8m")
SCHEMA = "region-inputs.v1"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported frame value for identity: {type(value).__name__}")


def _digest(value):
    data = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _frame_hash(frame):
    return _digest({"columns": list(frame.columns), "index": frame.index.tolist(),
                    "rows": frame.to_numpy(dtype=object).tolist()})


def load_local_frame(dataset_key, cache_dir, *, reader=None):
    """Read an already downloaded mRNABench parquet; never invoke its downloader.

    Return the unmodified table and source-file evidence. Preparation below owns
    letter-case/alphabet normalization and records each rejected row separately.
    """
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"A6 currently supports {DATASET_KEYS}")
    path = Path(cache_dir) / f"{dataset_key}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Local dataset required: {path}. No download was attempted.")
    before = tokens.file_fingerprint(path)
    frame = (reader or pd.read_parquet)(path)
    if tokens.file_fingerprint(path) != before:
        raise ValueError("Dataset changed while it was being read")
    source = {"path": str(path.resolve()), "sha256": before, "bytes": path.stat().st_size,
              "catalog_source": mrnabench.DATASETS[dataset_key].source}
    return frame, source


def load_local_homology_map(cache_dir, species="human"):
    """Read the existing named map and its hash; missing or ambiguous maps fail."""
    path = Path(cache_dir) / "homology_maps" / f"{species}_homology_map.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Explicit homology grouping requires {path}")
    before = tokens.file_fingerprint(path)
    table = pd.read_csv(path)
    if tokens.file_fingerprint(path) != before:
        raise ValueError("Homology map changed while it was being read")
    if not {"gene_name", "gene_group"}.issubset(table.columns):
        raise ValueError("Homology map needs gene_name and gene_group columns")
    if table[["gene_name", "gene_group"]].isna().any().any():
        raise ValueError("Homology map contains missing names or groups")
    if table["gene_name"].astype(str).duplicated().any():
        raise ValueError("Homology map contains repeated gene names")
    mapping = dict(zip(table["gene_name"].astype(str), table["gene_group"].astype(str)))
    return mapping, {"path": str(path.resolve()), "sha256": before}


@dataclass
class RegionCohort:
    dataset_key: str
    frame: pd.DataFrame
    layouts: tuple
    row_ids: tuple
    groups: np.ndarray
    exclusions: tuple
    identity: dict
    source: dict

    @property
    def labels(self):
        targets = list(mrnabench.DATASETS[self.dataset_key].targets)
        return self.frame[targets].to_numpy(dtype=np.float64)


def prepare_cohort(frame, dataset_key, *, grouping, homology_map=None,
                   unmapped_gene_policy="error", supplied_protein_column=None, source=None):
    """Apply one common, recorded eligibility filter before any model comparison.

    CDS validation uses the standard genetic code and masks a terminal stop;
    internal stops, ambiguous transcript bases, frame/translation mismatches, absent genes
    and non-finite targets are excluded with reasons. Gene grouping is explicitly
    weaker than homology grouping. Missing homology assignments raise unless the
    caller explicitly selects ``unmapped_gene_policy='gene'``. That policy retains
    mapped families and groups only unmapped rows by their recorded gene identifier;
    it neither normalizes aliases nor infers that an unmapped gene has no homologs.
    """
    if dataset_key not in DATASET_KEYS:
        raise ValueError(f"A6 currently supports {DATASET_KEYS}")
    if grouping not in ("gene", "homology"):
        raise ValueError("Choose grouping='gene' or grouping='homology' explicitly")
    if unmapped_gene_policy not in ("error", "gene"):
        raise ValueError("unmapped_gene_policy must be 'error' or 'gene'")
    if grouping == "homology" and not homology_map:
        raise ValueError("Homology grouping requires an explicitly supplied nonempty map")
    if grouping == "gene" and homology_map is not None:
        raise ValueError("A supplied homology map is unused under grouping='gene'")
    targets = list(mrnabench.DATASETS[dataset_key].targets)
    required = {"sequence", "cds", "gene", *targets}
    if supplied_protein_column is not None:
        required.add(supplied_protein_column)
    if frame.columns.duplicated().any() or not required.issubset(frame.columns):
        raise ValueError(f"Unique columns required, including {sorted(required)}")

    retained, layouts, row_ids, genes, exclusions, normalized = [], [], [], [], [], []
    for position, (_, row) in enumerate(frame.iterrows()):
        row_id = f"{dataset_key}:{position}"
        reasons = []
        sequence = row["sequence"]
        if isinstance(sequence, str) and set(sequence.upper().replace("U", "T")) - set("ACGT"):
            reasons.append("transcript_input: ambiguous or non-nucleotide characters")
        try:
            supplied = row[supplied_protein_column] if supplied_protein_column else None
            if supplied_protein_column and not isinstance(supplied, str):
                raise ValueError("supplied protein is missing")
            layout = region_inputs.coding_layout(row["sequence"], row["cds"],
                                                 terminal_stop="mask", supplied_protein=supplied)
        except (ValueError, TypeError) as error:
            reasons.append(f"coding_input: {error}")
        try:
            if not np.isfinite(np.asarray(row[targets], dtype=np.float64)).all():
                raise ValueError("target is not finite")
        except (ValueError, TypeError) as error:
            reasons.append(f"target: {error}")
        gene = row["gene"]
        if not isinstance(gene, str) or not gene.strip():
            reasons.append("gene: missing or non-string identifier")
        if reasons:
            exclusions.append({"row_id": row_id, "source_position": position, "reasons": reasons})
            continue
        retained.append(position)
        layouts.append(layout)
        row_ids.append(row_id)
        genes.append(gene)
        normalized.append(row["sequence"].upper().replace("U", "T"))
    if not retained:
        raise ValueError("No eligible transcripts remain after the common input filter")

    groups = []
    for gene in genes:
        if grouping == "gene":
            groups.append(f"gene:{gene}")
            continue
        group = homology_map.get(gene)
        if group is None or pd.isna(group):
            if unmapped_gene_policy == "error":
                raise ValueError(f"No homology assignment for {gene!r}; no fallback was selected")
            groups.append(f"gene:{gene}")
        else:
            groups.append(f"homology:{group}")
    selected = frame.iloc[retained].copy().reset_index(drop=True)
    selected["sequence"] = normalized
    source = dict(source or {"kind": "in_memory"})
    identity = {
        # The file's location is execution evidence, not a reason to reject the
        # same verified dataset after transferring it between desktop and Newton.
        "schema": SCHEMA, "dataset": dataset_key,
        "source": {key: value for key, value in source.items() if key != "path"},
        "source_frame_sha256": _frame_hash(frame), "frame_sha256": _frame_hash(selected),
        "row_ids": row_ids, "groups": groups, "grouping": grouping,
        "homology_map_sha256": _digest(homology_map) if homology_map is not None else None,
        "unmapped_gene_policy": unmapped_gene_policy, "terminal_stop": "mask",
        "transcript_alphabet": "ACGT_only_after_uppercase_U_to_T",
        "supplied_protein_column": supplied_protein_column,
        "targets": {"columns": targets, **array_record(selected[targets], dtype=np.float64)},
        "exclusions": exclusions,
    }
    identity["sha256"] = _digest(identity)
    return RegionCohort(dataset_key, selected, tuple(layouts), tuple(row_ids),
                        np.asarray(groups), tuple(exclusions), identity, source)


def _check_cohort(cohort):
    identity = {key: value for key, value in cohort.identity.items() if key != "sha256"}
    if (_digest(identity) != cohort.identity["sha256"]
            or _frame_hash(cohort.frame) != cohort.identity["frame_sha256"]
            or list(cohort.row_ids) != cohort.identity["row_ids"]
            or cohort.groups.tolist() != cohort.identity["groups"]):
        raise ValueError("Prepared cohort changed; prepare a new input identity")


@dataclass(frozen=True)
class InputChunk:
    row: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ChunkPlan:
    encoder_key: str
    chunks: tuple
    row_chunks: tuple
    identity: dict

    @property
    def texts(self):
        return [chunk.text for chunk in self.chunks]


def plan_chunks(cohort, *, window_lengths):
    """Plan complete non-overlapping windows, in bases for RNA and residues for ESM.

    Sizes must be supplied for every requested encoder. Two structural-token
    positions are reserved conservatively; extraction additionally checks actual
    token counts. This is chunked-context inference, not full-context inference.
    """
    _check_cohort(cohort)
    if not window_lengths or set(window_lengths) - set(ENCODER_KEYS):
        raise ValueError(f"Window lengths must name a nonempty subset of {ENCODER_KEYS}")
    plans = {}
    for key, size in window_lengths.items():
        encoder = ENCODERS[key]
        if (type(size) is not int or size < 1 or size > encoder.max_len - 2):
            raise ValueError(f"{key}: window length must be 1..{encoder.max_len - 2}")
        chunks, row_chunks = [], []
        for row, (sequence, layout) in enumerate(zip(cohort.frame["sequence"], cohort.layouts)):
            text = layout.protein if key == "esm2_8m" else sequence.replace("T", "U")
            indices = []
            for start in range(0, len(text), size):
                end = min(start + size, len(text))
                indices.append(len(chunks))
                chunks.append(InputChunk(row, start, end, text[start:end]))
            row_chunks.append(tuple(indices))
        identity = {"schema": SCHEMA, "cohort_sha256": cohort.identity["sha256"],
                    "dataset": cohort.dataset_key, "encoder": key,
                    "checkpoint": encoder.checkpoint, "revision": encoder.revision,
                    "window_length": size, "structural_token_reserve": 2,
                    "coordinate_unit": "residue" if key == "esm2_8m" else "nucleotide",
                    "coordinates": [[chunk.row, chunk.start, chunk.end] for chunk in chunks],
                    "input_sha256": tokens.input_fingerprint([chunk.text for chunk in chunks]),
                    "context": "independent_nonoverlapping_chunks"}
        identity["sha256"] = _digest(identity)
        plans[key] = ChunkPlan(key, tuple(chunks), tuple(row_chunks), identity)
    return plans


def _cache_paths(cache_root, plan):
    directory = Path(cache_root) / plan.identity["dataset"] / plan.encoder_key
    return directory, directory / "region-inputs.json"


def _cache_identity(plan, chunk_size, dtype):
    if type(chunk_size) is not int or chunk_size < 1 or np.dtype(dtype).kind != "f":
        raise ValueError("Positive shard rows and floating-point cache dtype required")
    return {"plan": plan.identity, "shard_rows": chunk_size, "dtype": np.dtype(dtype).name}


def cache_inventory(plans, cache_root, *, chunk_size=tokens.DEFAULT_CHUNK, dtype=np.float16):
    """Inspect metadata only; a recorded complete cache still requires verification.

    This does not open token shards, load weights, download data or create files.
    ``open_token_caches`` performs the input, shard-header and checksum checks.
    """
    rows = []
    for key, plan in plans.items():
        directory, identity_path = _cache_paths(cache_root, plan)
        issue, recorded = "", {}
        try:
            saved = json.loads(identity_path.read_text(encoding="utf-8"))
            if saved != _cache_identity(plan, chunk_size, dtype):
                raise ValueError("region input identity differs")
            recorded = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if recorded.get("input_sha256") != plan.identity["input_sha256"]:
                raise ValueError("ordered input identity differs")
            count = (len(plan.chunks) + chunk_size - 1) // chunk_size
            missing = [i for i in range(count) if not (directory / f"shard_{i:05d}.npz").is_file()]
            if missing:
                raise ValueError(f"{len(missing)} shards missing")
        except (OSError, ValueError, TypeError) as error:
            issue = str(error)
        rows.append({"encoder": key, "chunks": len(plan.chunks), "cache_dir": str(directory),
                     "status": "not_ready" if issue else "recorded_complete_unverified",
                     "issues": issue, "verified": False, "metadata": recorded})
    return rows


def _publish_once(path, record):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Existing input record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination atomically;
                # unlike hard links it also works in redirected AppContainer temp.
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"Another writer published a different input record: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class RegionCaches:
    plans: dict
    caches: dict

    def close(self):
        first = None
        for cache in self.caches.values():
            try:
                cache.close()
            except BaseException as error:
                first = first or error
        if first is not None:
            raise first

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Region cache cleanup also failed: {cleanup_error}")

    def row_features(self, encoder_key, row, length):
        plan, cache = self.plans[encoder_key], self.caches[encoder_key]
        chunks = [(plan.chunks[i].start, plan.chunks[i].end, cache[i])
                  for i in plan.row_chunks[row]]
        return region_inputs.stitch_chunks(chunks, sequence_length=length)


def open_token_caches(plans, cache_root, *, mode, device=None,
                      chunk_size=tokens.DEFAULT_CHUNK, dtype=np.float16,
                      extractor=None, inspector=None, reader=None):
    """Verify/reuse caches, or explicitly extract missing windows using the shared writer.

    ``mode='read'`` never calls extraction or writes identity records. ``'extract'``
    can load encoder weights and must only be called in an authorized execution.
    Existing mismatched/legacy directories are preserved and refused. The returned
    owner must be closed, preferably with a ``with`` block.
    """
    if mode not in ("read", "extract"):
        raise ValueError("mode must explicitly be 'read' or 'extract'")
    if mode == "extract" and device is None:
        raise ValueError("Extraction requires an explicit device")
    if type(chunk_size) is not int or chunk_size < 1 or np.dtype(dtype).kind != "f":
        raise ValueError("Positive shard rows and floating-point cache dtype required")
    inspector = inspector or tokens.inspect_cache
    reader = reader or tokens.summarize
    extractor = extractor or tokens.extract
    requests = []
    # Refuse every incompatible input before creating any new extraction directory.
    for key, plan in plans.items():
        if key != plan.encoder_key:
            raise ValueError("Plan key and encoder disagree")
        directory, identity_path = _cache_paths(cache_root, plan)
        identity = _cache_identity(plan, chunk_size, dtype)
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError(f"Existing region input identity differs: {directory}")
        elif mode == "read" or (directory.exists() and any(directory.iterdir())):
            raise ValueError(f"Region input record is missing: {identity_path}")
        readiness = inspector(ENCODERS[key], plan.texts, directory,
                              chunk_size=chunk_size, dtype=dtype).require_ready()
        if mode == "read" and readiness.status != "complete":
            raise ValueError(f"Cache is incomplete; reading never extracts: {readiness.summary}")
        requests.append((key, plan, directory, identity_path, identity))

    opened = RegionCaches(dict(plans), {})
    try:
        for key, plan, directory, identity_path, identity in requests:
            if mode == "extract":
                _publish_once(identity_path, identity)
                cache, _ = extractor(ENCODERS[key], plan.texts, directory, device,
                                     chunk_size=chunk_size, dtype=dtype)
            else:
                cache = reader(directory)
            opened.caches[key] = cache
            expected = [chunk.end - chunk.start for chunk in plan.chunks]
            if len(cache) != len(expected) or not np.array_equal(cache.lengths, expected):
                raise ValueError(f"{key}: cached tokens do not cover their declared chunk coordinates")
        return opened
    except BaseException as error:
        try:
            opened.close()
        except BaseException as cleanup_error:
            error.add_note(f"Region cache cleanup also failed: {cleanup_error}")
        raise


def codon_pool_spans(layout):
    """CDS-frame codons and independent UTR triplets, with partial UTR edge bins.

    No bin crosses a UTR/CDS boundary. Each bin uses the mean of its actual bases;
    UTR triplets are computational pooling units, not biological codons.
    """
    spans = []
    for start, end in ((0, layout.cds_start), (layout.cds_start, layout.cds_end),
                       (layout.cds_end, layout.sequence_length)):
        spans.extend((left, min(left + 3, end)) for left in range(start, end, 3))
    return tuple(spans)


def mask_change_summary(layout, interval, grid):
    """Count changes on the actual model grid, including availability restriction.

    A different donor can supply the same effective mask. Protein availability
    stays fixed: a permuted mask can remove updates, not invent residues in UTRs.
    """
    if grid not in ("native", "codon_pool"):
        raise ValueError("Choose the native or codon_pool grid")
    start, end = interval
    if not 0 <= start < end <= layout.sequence_length:
        raise ValueError("Control interval must be inside the recipient transcript")
    spans = (tuple((i, i + 1) for i in range(layout.sequence_length))
             if grid == "native" else codon_pool_spans(layout))
    centers = np.array([(left + right - 1) // 2 for left, right in spans])
    truth = (centers >= layout.cds_start) & (centers < layout.cds_end)
    permuted = (centers >= start) & (centers < end)
    available = np.array([left >= layout.cds_start and right <= layout.residue_end
                          for left, right in spans])
    original_updates, permuted_updates = truth & available, permuted & available
    union = int((truth | permuted).sum())
    before = int(original_updates.sum())
    return {"grid": grid, "positions": len(spans), "true_cds_positions": int(truth.sum()),
            "permuted_cds_positions": int(permuted.sum()),
            "mask_overlap": int((truth & permuted).sum()) / union if union else None,
            "changed_mask_positions": int((truth != permuted).sum()),
            "protein_available_positions": int(available.sum()),
            "changed_update_positions": int((original_updates != permuted_updates).sum()),
            "retained_update_fraction": int(permuted_updates.sum()) / before if before else None,
            "effective_mask_unchanged": bool(np.array_equal(original_updates, permuted_updates))}


def input_audit(frame, cohort, plans, preparation, *, widths, dtype):
    """Inspect the agreed inputs/control without opening caches or constructing models.

    Targets are deliberately omitted from the audit output. The report describes
    cohort and input geometry, not model selection or an alternative split search.
    Payload estimates count stored token features only, not peak RAM/VRAM.
    """
    from collections import Counter
    from .region_training import validate_preparation

    _check_cohort(cohort)
    validate_preparation(cohort, preparation)
    if _frame_hash(frame) != cohort.identity["source_frame_sha256"]:
        raise ValueError("Audit source differs from the prepared cohort")
    if (not plans or set(plans) != set(widths) or np.dtype(dtype).kind != "f"
            or any(type(width) is not int or width < 1 for width in widths.values())):
        raise ValueError("Each planned encoder needs an explicit positive width and floating dtype")
    for key, plan in plans.items():
        if (plan.identity["cohort_sha256"] != cohort.identity["sha256"]
                or plan.identity["sha256"] != _digest({k: v for k, v in plan.identity.items() if k != "sha256"})):
            raise ValueError("Audit window plan differs from its recorded identity")
        row_chunks = [[] for _ in cohort.row_ids]
        for index, chunk in enumerate(plan.chunks):
            if not 0 <= chunk.row < len(row_chunks) or len(chunk.text) != chunk.end - chunk.start:
                raise ValueError("Audit window contents differ from their coordinates")
            row_chunks[chunk.row].append(index)
        if (key != plan.encoder_key or key != plan.identity["encoder"]
                or [[c.row, c.start, c.end] for c in plan.chunks] != plan.identity["coordinates"]
                or tokens.input_fingerprint(plan.texts) != plan.identity["input_sha256"]
                or tuple(map(tuple, row_chunks)) != plan.row_chunks):
            raise ValueError("Audit window contents differ from their recorded identity")
    partition = {i: name for name, rows in preparation["positions"].items() for i in rows}
    kept = {int(row_id.rsplit(":", 1)[1]): i for i, row_id in enumerate(cohort.row_ids)}
    excluded = {row["source_position"]: row["reasons"] for row in cohort.exclusions}
    rows, masks = [], []
    for position, (_, row) in enumerate(frame.iterrows()):
        index = kept.get(position)
        entry = {"source_position": position, "status": "retained" if index is not None else "excluded",
                 "length": len(row["sequence"]) if isinstance(row["sequence"], str) else None,
                 "reasons": excluded.get(position, [])}
        try:
            layout = (cohort.layouts[index] if index is not None else
                      region_inputs.coding_layout(row["sequence"], row["cds"], terminal_stop="mask"))
        except (TypeError, ValueError):
            layout = None  # Invalid annotations cannot provide trustworthy region lengths.
        if layout is not None:
            entry.update(utr5=layout.cds_start, cds=layout.cds_end - layout.cds_start,
                         utr3=layout.sequence_length - layout.cds_end, protein_length=len(layout.protein))
        if index is not None:
            entry.update(row_id=cohort.row_ids[index], group=str(cohort.groups[index]),
                         group_kind="mapped" if cohort.groups[index].startswith("homology:") else "unmapped",
                         partition=partition[index])
            for key, plan in plans.items():
                entry[f"{key}_windows"] = len(plan.row_chunks[index])
                if key != "esm2_8m":
                    boundaries = [plan.chunks[i].start for i in plan.row_chunks[index][1:]]
                    entry[f"{key}_boundaries_in_cds"] = sum(layout.cds_start < b < layout.cds_end for b in boundaries)
            entry["pooled_positions"] = len(codon_pool_spans(layout))
            for grid in ("native", "codon_pool"):
                masks.append({"row_id": cohort.row_ids[index], "partition": partition[index],
                              **mask_change_summary(layout, preparation["permuted_intervals"][index], grid)})
        rows.append(entry)
    groups = []
    partitions = []
    for name, positions in preparation["positions"].items():
        counts = Counter(str(cohort.groups[i]) for i in positions)
        groups.extend({"partition": name, "group": group, "rows": count,
                       "group_kind": "mapped" if group.startswith("homology:") else "unmapped"}
                      for group, count in sorted(counts.items()))
        partitions.append({"partition": name, "rows": len(positions), "groups": len(counts),
                           "largest_group_fraction": max(counts.values()) / len(positions)})
    windows = []
    for key, plan in plans.items():
        token_count = sum(len(chunk.text) for chunk in plan.chunks)
        windows.append({"encoder": key, "window_length": plan.identity["window_length"],
                        "coordinate_unit": plan.identity["coordinate_unit"], "chunks": len(plan.chunks),
                        "multiwindow_rows": sum(len(indices) > 1 for indices in plan.row_chunks),
                        "tokens": token_count, "width": widths[key],
                        "payload_bytes": token_count * widths[key] * np.dtype(dtype).itemsize})
    # Retain interpretable source/policy evidence even when no training preparation
    # has been published. Full sequences, targets and duplicate coordinates stay out.
    cohort_fields = ("dataset", "source", "source_frame_sha256", "homology_map_sha256",
                     "grouping", "unmapped_gene_policy", "terminal_stop", "transcript_alphabet",
                     "supplied_protein_column")
    context = {"schema": "region-input-audit.v1", "cohort_sha256": cohort.identity["sha256"],
               "cohort_policy": {key: cohort.identity[key] for key in cohort_fields},
               "preparation_policy": {key: preparation[key] for key in
                                      ("split_seed", "ratios", "mask_seed", "mask_rule")},
               "preparation_sha256": preparation["sha256"], "widths": widths, "dtype": np.dtype(dtype).name,
               "plans": {key: {field: value for field, value in plan.identity.items() if field != "coordinates"}
                         for key, plan in plans.items()},
               "implementation_sha256": {"region_inputs.py": tokens.file_fingerprint(region_inputs.__file__),
                                         "region_experiment.py": tokens.file_fingerprint(__file__)}}
    return {"context": context, "rows": rows, "groups": groups, "partitions": partitions,
            "windows": windows, "masks": masks, "raw_rows": len(frame), "retained_rows": len(cohort.frame)}


def read_input_audit(path):
    """Read retained inspection evidence without loading the underlying dataset."""
    path = Path(path)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("context", {}).get("schema") != "region-input-audit.v1"
            or record.get("sha256") != _digest({k: v for k, v in record.items() if k != "sha256"})):
        raise ValueError("Input audit is malformed or changed")
    return record


def publish_input_audit(path, audit, *, producer):
    """Reuse identical evidence or publish once; never replace a different audit."""
    if not producer:
        raise ValueError("Input audit publication needs observed producer evidence")
    old = read_input_audit(path)
    if old is not None:
        if {k: v for k, v in old.items() if k not in ("producer", "sha256")} != audit:
            raise ValueError("Existing input audit differs; preserve it and review a separate destination")
        return old
    record = {**audit, "producer": producer}
    record["sha256"] = _digest(record)
    _publish_once(Path(path), record)
    return record


def validate_mask_donors(cohort, donors):
    """Validate an explicitly supplied, same-length transcript-mask permutation.

    This does not choose donors. The study must separately agree the admissible
    grouping/partition rule and record it; no length rescaling or label use occurs.
    """
    values = np.asarray(donors)
    if (values.shape != (len(cohort.frame),) or values.dtype.kind not in "iu"
            or sorted(values.tolist()) != list(range(len(values)))):
        raise ValueError("Mask donors must be an explicit permutation of cohort row positions")
    for row, donor in enumerate(values):
        if row == donor:
            raise ValueError("Permuted-mask controls require a different donor transcript")
        if cohort.layouts[row].sequence_length != cohort.layouts[donor].sequence_length:
            raise ValueError("Mask donors must have equal transcript length; rescaling is not defined")
    return values.astype(np.int64, copy=True)


class RegionDataset(Dataset):
    """Read one transcript's cached windows at a time; retain separate availability masks."""

    def __init__(self, cohort, caches, *, rna_encoder, grid, indices=None,
                 mask_donors=None, include_protein=True, region_intervals=None):
        _check_cohort(cohort)
        if rna_encoder not in ("rnafm", "rinalmo") or grid not in ("native", "codon_pool"):
            raise ValueError("Choose RNA-FM/RiNALMo and grid='native' or 'codon_pool'")
        for key in ((rna_encoder, "esm2_8m") if include_protein else (rna_encoder,)):
            if key not in caches.caches or caches.plans[key].identity["cohort_sha256"] != cohort.identity["sha256"]:
                raise ValueError("Caches must belong to this exact ordered cohort")
        self.cohort, self.caches, self.rna_encoder, self.grid = cohort, caches, rna_encoder, grid
        self.include_protein = include_protein
        self.labels = cohort.labels
        self.indices = np.arange(len(cohort.frame)) if indices is None else np.asarray(indices)
        if (self.indices.ndim != 1 or self.indices.dtype.kind not in "iu"
                or len(np.unique(self.indices)) != len(self.indices)
                or np.any(self.indices < 0) or np.any(self.indices >= len(cohort.frame))):
            raise ValueError("Dataset indices must be unique valid cohort positions")
        self.mask_donors = None if mask_donors is None else validate_mask_donors(cohort, mask_donors)
        if region_intervals is not None and mask_donors is not None:
            raise ValueError("Supply explicit region intervals or donor rows, not both")
        self.region_intervals = region_intervals
        if region_intervals is not None:
            if len(region_intervals) != len(cohort.frame):
                raise ValueError("One explicit region interval is required per cohort row")
            for layout, interval in zip(cohort.layouts, region_intervals):
                if (len(interval) != 2 or any(type(value) is not int for value in interval)
                        or not 0 <= interval[0] < interval[1] <= layout.sequence_length):
                    raise ValueError("Region intervals must be nonempty in-bounds nucleotide spans")
        if self.mask_donors is not None and grid != "native":
            raise ValueError("Permuted masks on the pooled grid need an explicit projection policy")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])
        layout = self.cohort.layouts[row]
        rna = self.caches.row_features(self.rna_encoder, row, layout.sequence_length)
        streams = {"rna": rna}
        masks = {"rna": np.ones(len(rna), dtype=bool)}
        if self.include_protein:
            residues = self.caches.row_features("esm2_8m", row, len(layout.protein))
            protein, available = region_inputs.protein_to_nucleotides(layout, residues)
            streams["prot"], masks["prot"] = protein, available
        region = layout.cds_mask
        spans = tuple((i, i + 1) for i in range(layout.sequence_length))
        if self.grid == "codon_pool":
            spans = codon_pool_spans(layout)
            # Pool spans never straddle CDS or terminal-stop/protein boundaries.
            streams = {key: np.stack([values[start:end].mean(axis=0, dtype=np.float32)
                                      for start, end in spans]) for key, values in streams.items()}
            masks = {key: np.array([values[start:end].all() for start, end in spans])
                     for key, values in masks.items()}
            region = np.array([region[start:end].all() for start, end in spans])
        if self.mask_donors is not None:
            region = self.cohort.layouts[self.mask_donors[row]].cds_mask
        if self.region_intervals is not None:
            start, end = self.region_intervals[row]
            # A pooled bin uses its midpoint's membership. This keeps the
            # shuffled mask binary and contiguous without moving protein values.
            region = np.array([start <= (left + right - 1) // 2 < end for left, right in spans])
        return {"streams": streams, "masks": masks,
                "region": region, "label": self.labels[row],
                "row_id": self.cohort.row_ids[row], "positions": spans}


@dataclass
class RegionBatch:
    streams: dict
    masks: dict
    region: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    row_ids: tuple
    positions: tuple

    def to(self, device):
        return RegionBatch({key: value.to(device) for key, value in self.streams.items()},
                           {key: value.to(device) for key, value in self.masks.items()},
                           self.region.to(device), self.lengths.to(device), self.labels.to(device),
                           self.row_ids, self.positions)


def collate_region_rows(rows):
    """Pad only after native/pooled alignment; padding is unavailable in every mask."""
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(row["region"]) for row in rows], dtype=torch.long)
    longest = int(lengths.max())
    streams, masks = {}, {}
    modalities = tuple(rows[0]["streams"])
    if "rna" not in modalities or set(modalities) - {"rna", "prot"}:
        raise ValueError("Region batches require RNA and optional protein streams")
    if any(set(row["streams"]) != set(modalities) or set(row["masks"]) != set(modalities)
           for row in rows):
        raise ValueError("Every row in a batch must supply the same streams and masks")
    for modality in modalities:
        width = rows[0]["streams"][modality].shape[1]
        streams[modality] = torch.zeros(len(rows), longest, width, dtype=torch.float32)
        masks[modality] = torch.zeros(len(rows), longest, dtype=torch.bool)
        for index, row in enumerate(rows):
            length = int(lengths[index])
            features, valid = row["streams"][modality], row["masks"][modality]
            if features.shape != (length, width) or np.asarray(valid).shape != (length,):
                raise ValueError("Row features and masks must share their declared grid")
            if not np.isfinite(features).all():
                raise ValueError("Region features must be finite")
            streams[modality][index, :length] = torch.as_tensor(features, dtype=torch.float32)
            masks[modality][index, :length] = torch.as_tensor(valid, dtype=torch.bool)
    region = torch.zeros(len(rows), longest, dtype=torch.bool)
    for index, row in enumerate(rows):
        region[index, :lengths[index]] = torch.as_tensor(row["region"], dtype=torch.bool)
    labels = torch.tensor(np.stack([row["label"] for row in rows]), dtype=torch.float32)
    if not torch.isfinite(labels).all():
        raise ValueError("Region labels must be finite")
    return RegionBatch(streams, masks, region, lengths, labels,
                       tuple(row["row_id"] for row in rows), tuple(row["positions"] for row in rows))


def make_loader(dataset, *, batch_size, shuffle, seed, num_workers=0):
    """Build a deterministic loader; the notebook chooses splits and batch ordering."""
    if type(batch_size) is not int or batch_size < 1 or num_workers != 0:
        raise ValueError("Positive batch_size and num_workers=0 required for owned mapped caches")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, collate_fn=collate_region_rows)
