"""Read BioLangFusion Phase A/B results without mixing training seeds or recipes.

These checks read saved JSONs only. Legacy files can support score comparisons
without establishing their input/cache or producer identity; diagnostics retain
that distinction. No model, cache, training, or numerical library is imported.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path


HEADS = ("concat", "mil", "cross")
MODALITIES = ("dna", "rna", "prot")
VARIANTS = {*HEADS, *(f"unimodal:{m}" for m in MODALITIES),
            *(f"unimodal-grid:{m}" for m in MODALITIES)}


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def prediction_issue(record, truth=None, *, task=None, n_classes=None):
    """Check pairing against loader-precision targets, not original CSV precision.

    Equal targets alone do not prove sequence identity. The caller retains the
    bootstrap's original targets and supplies its loader-cast copy for this check.
    """
    if n_classes is not None and (type(n_classes) is not int or n_classes < 1):
        raise ValueError("n_classes must be a positive integer when provided")
    saved = record.get("test_truth")
    predicted = record.get("test_prediction")
    if not isinstance(saved, list) or not isinstance(predicted, list):
        return "saved targets or predictions absent"
    if not saved or len(saved) != len(predicted):
        return "saved targets and predictions have different or empty row counts"

    def finite_array(values):
        return (bool(values) and all(
            finite_array(v) if isinstance(v, list) else _finite(v) for v in values
        ))

    if not finite_array(saved) or not finite_array(predicted):
        return "saved targets or predictions contain invalid values"
    if task in ("regression", "classification"):
        if any(isinstance(row, list) for row in (*saved, *predicted)):
            return f"{task} results require one saved target and prediction per row"
        if task == "classification" and any(
            type(value) is not int or value < 0 for value in (*saved, *predicted)
        ):
            return "classification results require nonnegative integer class IDs, not logits"
        if task == "classification" and n_classes is not None and any(
            value >= n_classes for value in (*saved, *predicted)
        ):
            return f"classification target or prediction is outside the {n_classes}-class range"
    elif task == "expression":
        if not all(isinstance(row, list) for row in (*saved, *predicted)):
            return "expression results require a target and prediction panel per row"
        width = len(saved[0])
        if any(len(row) != width for row in (*saved, *predicted)):
            return "expression target and prediction panel widths differ"
    elif task is not None:
        return f"unsupported prediction task: {task}"
    # Without a declared task, only generic rectangular-array structure is checked.
    if any(isinstance(row, list) != isinstance(predicted[0], list) for row in predicted):
        return "saved prediction rows have inconsistent shapes"
    if isinstance(predicted[0], list):
        if any(len(row) != len(predicted[0]) for row in predicted):
            return "saved prediction rows have inconsistent widths"
    if truth is not None and saved != truth:
        return "saved targets differ from the current test targets/order"
    return None


@dataclass
class PairedSelection:
    """Compatible record groups sharing the same actual seed identities."""

    records: dict
    seeds: tuple
    coverage: list
    issues: list


def validation_choices(records, *, singles, positive_lambdas):
    """Choose the seed-0 single and positive MIL penalty only after their grids exist.

    Records must already be uniquely selected at seed 0. Ties retain input order,
    matching the notebook's historical validation selection; test scores are unused.
    """
    if any(entry["data"]["config"]["seed"] != 0 for entry in records):
        raise ValueError("validation choices require only seed-0 records")
    candidates = [entry["data"] for entry in records]
    single_runs = [r for r in candidates if r["variant"] in singles]
    gated_runs = [r for r in candidates if r["variant"] == "mil"
                  and r["config"]["entropy_lambda"] in positive_lambdas]
    have_singles = {r["variant"] for r in single_runs}
    have_lambdas = {r["config"]["entropy_lambda"] for r in gated_runs}
    best = (max(single_runs, key=lambda r: r["val_score"])["variant"]
            if singles and set(singles) <= have_singles else None)
    penalty = (max(gated_runs, key=lambda r: r["val_score"])["config"]["entropy_lambda"]
               if positive_lambdas and set(positive_lambdas) <= have_lambdas else None)
    return best, penalty


def paired_records(records, *, variants, seeds, mil_lambda, truth, task, n_classes=None):
    """Select paired arrays from already recipe-filtered entries, without scoring.

    ``truth`` must use the training loader's label precision. Every requested model
    receives the same ordered intersection of available seed IDs; equal file counts
    are insufficient. Coverage retains omissions and malformed/misordered arrays.
    This validates target correspondence, not unrecorded input or producer identity.
    """
    variants, seeds = tuple(variants), tuple(seeds)
    if n_classes is not None and (type(n_classes) is not int or n_classes < 1):
        raise ValueError("n_classes must be a positive integer when provided")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("paired variants must be nonempty and distinct")
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("paired seeds must be distinct nonnegative integers")
    by_key, issues = {}, []
    for entry in records:
        data = entry["data"]
        name, seed = data["variant"], data["config"]["seed"]
        if name not in variants or seed not in seeds:
            continue
        if name == "mil" and data["config"]["entropy_lambda"] != mil_lambda:
            continue
        key = name, seed
        if key in by_key:
            raise ValueError(f"Ambiguous paired result: {name}, seed {seed}")
        # Reserve the identity even if its arrays fail; a second file is still a duplicate.
        by_key[key] = None
        issue = prediction_issue(data, truth, task=task, n_classes=n_classes)
        if issue:
            issues.append({"variant": name, "seed": seed,
                           "source": str(entry["path"]), "reason": issue})
        else:
            by_key[key] = entry
    available = {
        name: tuple(seed for seed in seeds if by_key.get((name, seed)) is not None)
        for name in variants
    }
    common = tuple(seed for seed in seeds if all(seed in available[name] for name in variants))
    coverage = [{"variant": name, "available seeds": available[name],
                 "missing seeds": tuple(seed for seed in seeds if seed not in available[name]),
                 "common paired seeds": common} for name in variants]
    selected = {name: [by_key[name, seed] for seed in common] for name in variants}
    return PairedSelection(selected, common, coverage, issues)


@dataclass
class Selection:
    """Eligible records plus exclusions; duplicate identities are never overwritten."""

    records: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)

    def require_unique(self):
        if self.duplicates:
            raise ValueError("Ambiguous saved results: " + "; ".join(self.duplicates))
        return self.records

    def positive_lambda(self):
        """Retain validation selection among original positive MIL penalties."""
        candidates = [r for r in self.require_unique()
                      if r["data"]["variant"] == "mil"
                      and r["data"]["config"]["entropy_lambda"] > 0]
        if not candidates:
            raise ValueError("No compatible original positive-penalty MIL result")
        return max(candidates, key=lambda r: r["data"]["val_score"])["data"]["config"]["entropy_lambda"]

    def coverage(self, variants, *, mil_lambda=None):
        """Report score/prediction availability without calculating any result."""
        names = {r["data"]["variant"] for r in self.records
                 if r["data"]["variant"] != "mil"
                 or (r["data"]["config"]["entropy_lambda"] > 0
                     and (mil_lambda is None or r["data"]["config"]["entropy_lambda"] == mil_lambda))}
        paired = {r["data"]["variant"] for r in self.records
                  if prediction_issue(r["data"]) is None
                  and (r["data"]["variant"] != "mil"
                       or (r["data"]["config"]["entropy_lambda"] > 0
                           and (mil_lambda is None or r["data"]["config"]["entropy_lambda"] == mil_lambda)))}
        return {
            "available": ", ".join(v for v in variants if v in names),
            "missing": ", ".join(v for v in variants if v not in names),
            "without paired arrays": ", ".join(v for v in variants if v in names - paired),
            "ambiguous": "; ".join(self.duplicates),
            "excluded files": len(self.issues),
            "legacy identity": sum(r["legacy"] for r in self.records),
        }


def read_results(folders, *, benchmark, config, encoders, native_fallback=False,
                 mil_lambda=None, mil_lambdas=None, variants=VARIANTS):
    """Select one seed and full TrainConfig from direct result-folder children.

    With mil_lambda=None the original MIL validation grid remains eligible;
    mil_lambdas can restrict it to a declared candidate grid. Native
    fallback is only for original baselines that bypass convolution: prefer the
    declared tconv, otherwise accept a unique appendix/equation record. A duplicate
    (variant, penalty) is withheld, even if its scores happen to agree.
    """
    report = Selection()
    grouped = {}
    paths = sorted({p.resolve() for folder in folders for p in Path(folder).glob("*.json")})
    for path in paths:
        variant = None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("result is not an object")
            variant = data.get("variant")
            actual = data.get("config")
            if variant not in variants or data.get("benchmark") != benchmark:
                raise ValueError("different benchmark or unsupported variant")
            if not isinstance(actual, dict):
                raise ValueError("missing configuration")
            penalty = actual.get("entropy_lambda")
            if not _finite(penalty) or penalty < 0:
                raise ValueError("invalid entropy penalty")
            if variant == "mil" and mil_lambdas is not None and penalty not in mil_lambdas:
                raise ValueError("entropy penalty is outside the declared candidate grid")
            expected = dict(config)
            expected["entropy_lambda"] = (
                penalty if variant == "mil" and mil_lambda is None
                else mil_lambda if variant == "mil" else 0.0
            )
            native = native_fallback and variant.startswith("unimodal:")
            if native and actual.get("tconv") in ("appendix", "equation"):
                expected["tconv"] = actual["tconv"]
            different = sorted(k for k in actual.keys() | expected.keys()
                               if k not in actual or k not in expected or actual[k] != expected[k])
            if different or type(actual.get("seed")) is not int:
                raise ValueError("configuration differs: " + ", ".join(different or ["seed"]))
            if (type(data.get("epochs")) is not int or data["epochs"] < 1
                    or not _finite(data.get("val_score")) or not _finite(data.get("test_score"))):
                raise ValueError("missing or invalid completed-fit scores/epochs")
            # The notebook converts selected records with train.Result.load; do not
            # advertise a record that lacks that consumer's required fit metadata.
            if (type(data.get("best_epoch")) is not int
                    or not 0 <= data["best_epoch"] <= data["epochs"]
                    or not _finite(data.get("seconds")) or data["seconds"] < 0):
                raise ValueError("missing or invalid best_epoch/seconds")

            identity = data.get("identity")
            recorded_encoders = None
            if identity is not None:
                if not isinstance(identity, dict):
                    raise ValueError("invalid recorded identity")
                for key in ("benchmark", "config", "variant"):
                    if key in identity and identity[key] != data[key]:
                        raise ValueError("recorded identity contradicts " + key)
                inputs = identity.get("inputs", {})
                if not isinstance(inputs, dict):
                    raise ValueError("invalid input identity")
                recorded_encoders = inputs.get("encoders")
                if recorded_encoders is not None and recorded_encoders != encoders:
                    raise ValueError("recorded encoder mapping differs")
            entry = {"path": path, "data": data, "legacy": recorded_encoders is None}
            grouped.setdefault((variant, penalty), []).append(entry)
        except (OSError, ValueError, TypeError) as exc:
            report.issues.append({"source": str(path), "variant": variant, "reason": str(exc)})

    for (variant, penalty), entries in grouped.items():
        if native_fallback and variant.startswith("unimodal:"):
            preferred = [r for r in entries if r["data"]["config"]["tconv"] == config["tconv"]]
            if preferred:
                for entry in entries:
                    if entry not in preferred:
                        report.issues.append({"source": str(entry["path"]), "variant": variant,
                                              "reason": "native baseline with requested tconv preferred"})
                entries = preferred
        if len(entries) != 1:
            detail = f"{variant} (lambda {penalty}): " + ", ".join(str(r["path"]) for r in entries)
            report.duplicates.append(detail)
        else:
            report.records.append(entries[0])
    return report


def swap_variants(modality, *, grid):
    """Expected fit names for one replacement; grid baselines are conditional."""
    if modality not in MODALITIES:
        raise ValueError(f"Unsupported modality: {modality}")
    return (f"unimodal:{modality}",
            *((f"unimodal-grid:{modality}",) if grid else ()), *HEADS)
