"""Fixed validation selection shared by correction and anchor-attention studies."""

import hashlib
import json
import math
from pathlib import Path

from .correction_experiment import _publish


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _roots(path, evidence_root, evidence_roots):
    if evidence_roots is not None:
        if evidence_root is not None or not isinstance(evidence_roots, dict) or not evidence_roots:
            raise ValueError("Provide one evidence root or a nonempty named-root mapping.")
        if any(not isinstance(key, str) or not key.isidentifier() for key in evidence_roots):
            raise ValueError("Evidence-root names must be simple identifiers.")
        return "named", {key: Path(value).resolve() for key, value in evidence_roots.items()}
    location = "external" if evidence_root is not None else "selection"
    return location, {location: Path(evidence_root or path.parent).resolve()}


def select_anchor(candidates, *, path, context, evidence_root=None, evidence_roots=None):
    """Choose a declared seed-zero candidate from saved validation scores only.

    Candidate order breaks ties. Paths are recorded relative to the selection
    file, or to an explicitly supplied evidence root. External completed results
    stay in place; their paths and bytes are bound without duplicating them.
    """
    path = Path(path)
    location, roots = _roots(path, evidence_root, evidence_roots)
    candidates = list(candidates)
    if not candidates or len({item["key"] for item in candidates}) != len(candidates):
        raise ValueError("Declare a nonempty panel with unique candidate keys.")
    records = []
    for item in candidates:
        source = Path(item["path"]).resolve()
        matches = [key for key, value in roots.items() if source.is_relative_to(value)]
        if not matches:
            raise ValueError("Anchor evidence must be inside its declared evidence root.")
        # Most-specific roots avoid an accidental dependence on shared ancestors
        # whose directory names differ between Newton and a returned package.
        root_name = max(matches, key=lambda key: len(roots[key].parts))
        root = roots[root_name]
        result = json.loads(source.read_text(encoding="utf-8"))
        score = result.get("val_score")
        seed = result.get("config", {}).get("seed")
        if (type(seed) is not int or seed != 0 or type(score) not in (int, float)
                or not math.isfinite(score)):
            raise ValueError("Every anchor candidate needs a finite seed-zero validation score.")
        if item["modality"] not in ("dna", "rna", "prot") or item["layout"] not in ("native", "codon"):
            raise ValueError("Anchor candidate has an unknown modality or input layout.")
        records.append({"key": item["key"], "modality": item["modality"],
                        "layout": item["layout"], "validation_score": score,
                        "root": root_name, "path": source.relative_to(root).as_posix(),
                        "sha256": file_sha256(source)})
    selected = max(records, key=lambda item: item["validation_score"])
    record = {"schema": "fusion.anchor-selection.v1", "selection_seed": 0,
              "evidence_location": location,
              "policy": "maximum-validation-score; declared order breaks ties",
              "context": context, "candidates": records, "selected": selected["key"]}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("An incompatible anchor selection already exists; preserve it.")
    else:
        try:
            _publish(path, record)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != record:
                raise ValueError("Another writer published a different anchor selection.")
    return read_anchor(path, context=context, evidence_root=evidence_root, evidence_roots=evidence_roots)


def read_anchor(path, *, context, evidence_root=None, evidence_roots=None):
    """Verify immutable selection evidence without loading weights or rescoring."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    location, roots = _roots(path, evidence_root, evidence_roots)
    if (record.get("schema") != "fusion.anchor-selection.v1"
            or record.get("selection_seed") != 0 or record.get("context") != context
            or record.get("evidence_location", "selection") != location):
        raise ValueError("Saved anchor selection belongs to a different experiment.")
    candidates = record.get("candidates", [])
    if not candidates or len({item["key"] for item in candidates}) != len(candidates):
        raise ValueError("Saved anchor candidate panel is invalid.")
    for item in candidates:
        name = item.get("root", location)
        if name not in roots:
            raise ValueError("Saved selection requires an explicit named evidence root.")
        root = roots[name]
        source = (root / item["path"]).resolve()
        if not source.is_relative_to(root) or file_sha256(source) != item["sha256"]:
            raise ValueError("Saved anchor candidate source changed or escaped its result root.")
        result = json.loads(source.read_text(encoding="utf-8"))
        seed = result.get("config", {}).get("seed")
        if (type(seed) is not int or seed != 0
                or result.get("val_score") != item["validation_score"]):
            raise ValueError("Saved anchor selection does not match its validation evidence.")
        score = item["validation_score"]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Saved anchor validation score is invalid.")
    selected = max(candidates, key=lambda item: item["validation_score"])
    if selected["key"] != record.get("selected"):
        raise ValueError("Saved anchor winner does not follow the declared validation rule.")
    return {**selected, "selection_sha256": file_sha256(path)}
