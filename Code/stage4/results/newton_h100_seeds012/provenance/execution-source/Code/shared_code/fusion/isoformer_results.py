"""Read the frozen GTEx pilot cohort without loading models or changing results.

Selection checks the saved training configuration and the caller's filename recipe.
The historical JSONs do not record attention-budget, validation-split, input or code
identity. A selected file therefore remains a limited-provenance observation, not a
fully verified reproduction. In particular, a filename is not proof of its contents.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


VARIANTS = ("dna", "rna", "prot", "dna+prot", "dna+rna", "dna+rna+prot")
_TAG = re.compile(
    r"gtex_pilot__(?P<variant>[^_]+)__heads(?P<heads>\d+)"
    r"__budget(?P<budget>\d+)M__seed(?P<seed>\d+)"
)
_LEGACY_CONFIG = {
    "lr": 3e-5, "weight_decay": 1e-5, "batch_size": 16, "max_epochs": 500,
    "early_stopping_patience": 20, "plateau_patience": 5,
    "entropy_lambda": 0.0, "channels": 100, "tconv": "appendix",
}


@dataclass
class Cohort:
    """Selected observations, explicit exclusions, and expected-seed coverage."""

    records: list
    issues: list
    coverage: list
    duplicate_keys: list


def _prediction_shape(payload, expected_shape):
    """Require finite rectangular panels without computing scores or importing ML."""
    shapes = []
    for name in ("test_truth", "test_prediction"):
        values = payload.get(name)
        if (not isinstance(values, list) or not values
                or not isinstance(values[0], list) or not values[0]):
            raise ValueError(f"{name} must be a nonempty two-dimensional panel")
        width = len(values[0])
        for row in values:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"{name} has inconsistent panel widths")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in row):
                raise ValueError(f"{name} contains a nonfinite or nonnumeric value")
        shapes.append((len(values), width))
    if shapes[0] != shapes[1]:
        raise ValueError("saved target and prediction panel shapes differ")
    if expected_shape is not None and shapes[0] != expected_shape:
        raise ValueError(f"saved panel shape {shapes[0]} differs from expected {expected_shape}")
    return shapes[0]


def _check_recorded_identity(payload, *, label, n_heads, budget, val_fraction, shape):
    """Validate a present versioned identity; never reinterpret a malformed one as legacy."""
    if "identity" not in payload:
        return
    # Import only for records that claim an identity. Ordinary legacy score reading
    # retains this module's standard-library-only import path.
    from . import run_identity

    identity = run_identity.validate_identity(
        payload["identity"], config=payload["config"],
        variant=payload["variant"], benchmark="gtex_pilot",
    )
    loader = identity["loader"]
    if loader["kind"] != "isoformer":
        raise ValueError("recorded loader is not the IsoFormer pilot loader")
    requested = {
        "n_heads": n_heads, "budget": budget, "val_fraction": val_fraction,
        "batch_size": payload["config"]["batch_size"], "seed": payload["config"]["seed"],
    }
    for key, value in requested.items():
        if key in loader and loader[key] != value:
            raise ValueError(f"recorded loader {key} differs from the requested cohort")
    model = identity["model"]
    kwargs = model.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("recorded model kwargs must be an object")
    if "n_heads" in kwargs and kwargs["n_heads"] != n_heads:
        raise ValueError("recorded model n_heads differs from the requested cohort")
    if "modalities" in model and model["modalities"] != label.split("+"):
        raise ValueError("recorded model modalities differ from the requested combination")
    if shape is not None:
        if "n_outputs" in model and model["n_outputs"] != shape[1]:
            raise ValueError("recorded model output width differs from the saved panel")
    # Recorded split targets use dataset-index order. BudgetedBatches emits rows
    # in attention-cost order, so that fingerprint cannot validate the saved
    # prediction order. The cohort check compares the emitted target panels only.


def _read_record(path, *, reference, config, variants, seeds, n_heads, budget,
                 val_fraction, require_predictions, expected_shape):
    """Return a reader row or raise a concrete exclusion reason."""
    match = _TAG.fullmatch(path.stem)
    if match is None:
        raise ValueError("filename does not identify the selected budgeted GTEx cohort")
    label = match["variant"]
    seed = int(match["seed"])
    if label not in variants or seed not in seeds:
        raise ValueError("modality combination or seed is outside the requested cohort")
    if int(match["heads"]) != n_heads or int(match["budget"]) * 1_000_000 != budget:
        raise ValueError("filename attention heads or budget differ from the requested cohort")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("result is not a JSON object")
    if payload.get("benchmark") != "gtex_pilot":
        raise ValueError("saved benchmark is not gtex_pilot")
    saved_config = payload.get("config")
    expected_config = dict(config, seed=seed)
    if saved_config != expected_config:
        raise ValueError("saved training configuration differs or has missing/unknown fields")
    if type(saved_config["seed"]) is not int:
        raise ValueError("saved seed must be an integer, not a Boolean or floating-point value")

    stored_variant = payload.get("variant")
    legacy = stored_variant == "IsoFormer"
    if legacy:
        # Only these six seed-0 paths have a documented historical caller.
        if not (reference and seed == 0 and n_heads == 8 and budget == 300_000_000
                and val_fraction == 0.05 and config == _LEGACY_CONFIG
                and path.stem == f"gtex_pilot__{label}__heads8__budget300M__seed0"):
            raise ValueError("generic IsoFormer label has no supported historical mapping here")
    elif stored_variant != label:
        raise ValueError("saved modality combination disagrees with its filename")

    score = payload.get("test_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("test score is missing or nonfinite")
    if not -1 <= score <= 1:
        raise ValueError("test Spearman is outside [-1, 1]")
    epochs = payload.get("epochs")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("record does not report a positive completed epoch count")
    shape = _prediction_shape(payload, expected_shape) if require_predictions else None
    _check_recorded_identity(
        payload, label=label, n_heads=n_heads, budget=budget,
        val_fraction=val_fraction, shape=shape,
    )
    row = {
        "variant": label,
        "seed": seed,
        "Spearman": score,
        "path": str(path),
        "identity": "historical label mapped" if legacy else "stored modality label",
    }
    if require_predictions:
        row["result"] = payload
    return row


def read_cohort(reference_dir, repeats_dir=None, *, train_config, variants=VARIANTS,
                expected_seeds=(0, 1, 2), n_heads=8, budget=300_000_000,
                val_fraction=0.05, require_predictions=False, expected_shape=None):
    """Inventory compatible saved records; never fit, write, or drop duplicate keys.

    ``train_config`` is the notebook's complete TrainConfig dictionary. Its seed is
    replaced by each requested seed for comparison. Legacy label recovery applies
    only to the reference directory's exact known six heads8/budget300M/seed0 tags.
    Unbudgeted siblings and unknown labels are reported as exclusions. Duplicate
    modality/seed keys are removed from selection and block the notebook summary.

    ``repeats_dir=None`` reads only the reference directory. The default returns
    score rows without requiring predictions. With ``require_predictions=True``,
    each accepted row also contains its validated JSON under ``result``; target
    and prediction panels must be finite, nonempty and identically shaped, with
    optional ``expected_shape=(rows, targets)``. All selected target panels must
    agree in row and target-column order. Conflicting panels block the entire
    selected prediction cohort because this reader has no independent target
    source from which to choose the correct order. Matching emitted target panels
    does not recover evaluation row identity; dataset-order fingerprints cannot
    establish the budgeted loader's emission order. No scores are recomputed.

    Historical budget/heads come from the tag and validation fraction is absent
    from those JSONs. Present versioned identities must validate, and any recorded
    request settings must agree. Neither route independently verifies input/cache
    contents or reconstructs the historical producing implementation.
    """
    variants, seeds = tuple(variants), tuple(expected_seeds)
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("variants must be distinct supported GTEx modality combinations")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("expected_seeds must be distinct nonnegative integers")
    if type(n_heads) is not int or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")
    if type(budget) is not int or budget <= 0 or budget % 1_000_000:
        raise ValueError("budget must be a positive whole number of million attention cells")
    if not isinstance(val_fraction, (int, float)) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between zero and one")
    if type(require_predictions) is not bool:
        raise ValueError("require_predictions must be a Boolean")
    if expected_shape is not None:
        if not require_predictions:
            raise ValueError("expected_shape requires require_predictions=True")
        if (not isinstance(expected_shape, (tuple, list)) or len(expected_shape) != 2
                or any(type(size) is not int or size < 1 for size in expected_shape)):
            raise ValueError("expected_shape must contain two positive integer dimensions")
        expected_shape = tuple(expected_shape)
    config = dict(train_config)
    config.pop("seed", None)
    folders = (Path(reference_dir),) if repeats_dir is None else (Path(reference_dir), Path(repeats_dir))
    if len(folders) == 2 and folders[0].resolve() == folders[1].resolve():
        raise ValueError("reference and repeats must be different directories")

    by_key = defaultdict(list)
    issues = []
    for index, folder in enumerate(folders):
        if not folder.is_dir():
            issues.append({"file": str(folder), "status": "missing directory",
                           "reason": "no saved results read from this destination"})
            continue
        for path in sorted(folder.glob("*.json")):
            try:
                row = _read_record(
                    path, reference=index == 0, config=config, variants=variants,
                    seeds=seeds, n_heads=n_heads, budget=budget, val_fraction=val_fraction,
                    require_predictions=require_predictions, expected_shape=expected_shape,
                )
            except (OSError, UnicodeError, ValueError, TypeError) as error:
                issues.append({"file": str(path), "status": "excluded", "reason": str(error)})
                continue
            by_key[(row["variant"], row["seed"])].append(row)

    duplicates = sorted(key for key, rows in by_key.items() if len(rows) > 1)
    records = []
    for key, rows in by_key.items():
        if key in duplicates:
            for row in rows:
                issues.append({"file": row["path"], "status": "duplicate; blocked",
                               "reason": f"multiple selected files for {key[0]}, seed {key[1]}"})
        else:
            records.append(rows[0])
    order = {variant: index for index, variant in enumerate(variants)}
    records.sort(key=lambda row: (order[row["variant"]], row["seed"]))
    if require_predictions and records:
        targets = records[0]["result"]["test_truth"]
        if any(row["result"]["test_truth"] != targets for row in records[1:]):
            for row in records:
                issues.append({
                    "file": row["path"], "status": "excluded",
                    "reason": "selected target panels disagree in values, shape or order; "
                              "the entire prediction cohort is withheld without an independent target reference",
                })
            records = []
    if records:
        issues.append({
            "file": "selected cohort", "status": "provenance limit",
            "reason": "Stored training settings match. Historical heads/budget come from filenames; "
                      "available versioned settings are checked against the request. "
                      f"Validation fraction {val_fraction} is checked when recorded. Actual input/cache "
                      "contents and historical producing code/model settings are not established here.",
        })

    coverage = []
    for variant in variants:
        available = [row["seed"] for row in records if row["variant"] == variant]
        missing = [seed for seed in seeds if seed not in available]
        blocked = any(key[0] == variant for key in duplicates)
        coverage.append({
            "variant": variant, "available seeds": tuple(available),
            "missing seeds": tuple(missing), "count": len(available),
            "expected count": len(seeds),
            "status": "duplicate; blocked" if blocked else
                      "complete" if not missing else "partial" if available else "missing",
        })
    return Cohort(records, issues, coverage, duplicates)
