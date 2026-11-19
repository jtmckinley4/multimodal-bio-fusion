"""Paired reporting of completed encoder swaps; never fits or opens token caches.

Selection is fixed by seed-0 validation. A multi-seed mean requires every requested
seed, and averages scores rather than predictions. Legacy input identity remains
qualified even when saved targets agree with the current benchmark ordering.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.stats import spearmanr

from . import run_identity, swap_results


SCHEMA = "swap.paired-report.v1"
HEADS = swap_results.HEADS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _saved_source_path(source, *, field, source_roots):
    """Map one recorded source into its declared evidence tree, never a fallback tree."""
    if not isinstance(source, str) or not source:
        raise ValueError("Saved report source paths must be nonempty strings")
    if source_roots is None:
        path = Path(source)
        if not path.is_absolute():
            raise ValueError(f"Saved source needs an absolute path or explicit root mapping: {source}")
        return path
    marker = "outputs" if field == "sources" else "historical_results/Code"
    root = Path(source_roots[marker]).resolve()
    parts = PurePosixPath(source.replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError(f"Saved source contains parent traversal: {source}")
    # Resolve the recorded tree marker first: an old returned historical tree
    # can itself sit beneath canonical Code/, even after its copies are removed.
    anchor = tuple(marker.split("/"))
    matches = [i for i in range(len(parts) - len(anchor) + 1)
               if parts[i:i + len(anchor)] == anchor]
    if not matches:
        local = Path(source)
        if local.is_absolute() and local.resolve().is_relative_to(root):
            return local.resolve()
    if len(matches) != 1:
        raise ValueError(f"Saved source has no unique {marker} root: {source}")
    suffix = parts[matches[0] + len(anchor):]
    path = root.joinpath(*suffix).resolve()
    if not suffix or not path.is_relative_to(root):
        raise ValueError(f"Saved source escapes its selected {marker} root: {source}")
    return path


def _validate_saved_report(saved, path, *, benchmark, seeds, source_roots, selection_snapshot=None):
    """Validate a producer report and referenced bytes without recomputing its analysis."""
    identity, analysis = saved.get("identity"), saved.get("analysis")
    if not isinstance(identity, dict) or identity.get("schema") != SCHEMA:
        raise ValueError("Saved report has an unsupported or missing identity schema")
    if (identity.get("benchmark") != benchmark or identity.get("seeds") != list(seeds)
            or any(type(seed) is not int for seed in identity.get("seeds", []))):
        raise ValueError("Saved report benchmark or requested seeds differ")
    if type(identity.get("selection_seed")) is not int or identity["selection_seed"] != 0:
        raise ValueError("Saved report selection seed must remain 0")
    if path.name != f"{benchmark}__{_digest(identity)[:20]}.json":
        raise ValueError("Saved report filename does not match its identity hash")
    if (not isinstance(analysis, dict)
            or any(not isinstance(analysis.get(key), list)
                   for key in ("per_seed", "seed_summary", "issues"))
            or not isinstance(analysis.get("units"), str) or not analysis["units"]
            or saved.get("analysis_sha256") != _digest(analysis)):
        raise ValueError("Saved report analysis is incomplete or its hash differs")
    for row in [*analysis["per_seed"], *analysis["seed_summary"]]:
        if (not isinstance(row, dict)
                or any(not isinstance(row.get(key), str)
                       for key in ("encoder", "kind", "comparison", "first"))
                or "second" not in row
                or (row["second"] is not None and not isinstance(row["second"], str))):
            raise ValueError("Saved report comparison table has missing or invalid fields")
    if any(not isinstance(row, dict) or type(row.get("seed")) is not int
           or row["seed"] not in seeds for row in analysis["per_seed"]):
        raise ValueError("Saved report per-seed analysis differs from its declared seeds")
    for row in analysis["seed_summary"]:
        if not isinstance(row, dict):
            raise ValueError("Saved report seed summary is not a record")
        available, missing = row.get("available_seeds"), row.get("missing_seeds")
        if (not isinstance(available, list) or not isinstance(missing, list)
                or any(type(seed) is not int for seed in [*available, *missing])
                or available != [seed for seed in seeds if seed in available]
                or missing != [seed for seed in seeds if seed not in available]
                or row.get("status") != ("complete" if not missing else "incomplete")):
            raise ValueError("Saved report seed summary contradicts its declared seeds")
    if identity.get("selection_policy") == "fixed-historical-seed0-validation" and not identity.get("selection_sources"):
        raise ValueError("Saved report is missing its historical selection sources")
    if (selection_snapshot is not None
            and identity.get("selection_policy") != "fixed-historical-seed0-validation"):
        raise ValueError("Newton snapshot reuse requires the fixed historical selection policy.")
    # These are producer-code fingerprints, not the current consumer's helpers.
    # Validate their format but never compare or replace them with current hashes.
    fingerprints = identity.get("source_sha256")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("Saved report is missing producer source fingerprints")
    source_fields = {key: identity.get(key, {}) for key in ("sources", "selection_sources")}
    if "sources" not in identity or any(not isinstance(value, dict) for value in source_fields.values()):
        raise ValueError("Saved report source hashes must be objects")
    allowed_selection = (None if selection_snapshot is None
                         else selection_snapshot.file_hashes())
    for field, values in {"source_sha256": fingerprints, **source_fields}.items():
        for source, digest in values.items():
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError(f"Saved report has an invalid {field} hash: {source}")
            if field == "source_sha256":
                continue
            roots = source_roots
            if field == "selection_sources" and selection_snapshot is not None:
                roots = {**source_roots, "historical_results/Code": selection_snapshot.code_dir}
            actual = _saved_source_path(source, field=field, source_roots=roots)
            if (field == "selection_sources" and allowed_selection is not None
                    and allowed_selection.get(actual) != digest):
                raise ValueError(f"Saved report selection source is not in the original Newton snapshot: {source}")
            if not actual.is_file():
                raise ValueError(f"Saved report source unavailable: {actual}")
            if hashlib.sha256(actual.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Saved report source hash differs: {actual}")


def read_existing_report(*, benchmark, report_dirs, seeds=(0, 1, 2),
                         selected_report=None, expected=None, source_roots=None,
                         selection_snapshot=None):
    """Read one eligible saved report; never load benchmark rows, score, or write.

    ``expected`` matches supplied identity fields exactly (for example config,
    draws, bootstrap_seed and selection_policy). No match returns ``(None, None)``;
    multiple matches require ``selected_report``, an explicit path or filename.
    Corrupt selected evidence raises ValueError and is preserved. Selection never
    depends on directory order, file modification time, or current helper hashes.

    Imported reports use ``source_roots`` with exactly ``outputs`` and
    ``historical_results/Code`` mapped to the selected local evidence roots. Their
    recorded paths and producer metadata are returned unchanged; only hash checks
    use the bounded local mapping. No missing source falls back to another cohort.
    selection_snapshot maps this completed Newton batch's historical paths to
    exact canonical files; it never changes the report identity or analysis hash.
    """
    if not isinstance(benchmark, str) or not benchmark or any(c in benchmark for c in "/\\"):
        raise ValueError("Benchmark must be one nonempty path component")
    seeds = tuple(seeds)
    if (not seeds or 0 not in seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Report seeds must be distinct nonnegative integers including 0")
    expected = {} if expected is None else dict(expected)
    required = {"schema": SCHEMA, "benchmark": benchmark, "seeds": list(seeds)}
    if any(key in expected and expected[key] != value for key, value in required.items()):
        raise ValueError("Expected identity contradicts the requested schema, benchmark or seeds")
    if source_roots is not None:
        if set(source_roots) != {"outputs", "historical_results/Code"}:
            raise ValueError("Source mapping requires outputs and historical_results/Code roots")
        roots = [Path(value).resolve() for value in source_roots.values()]
        if roots[0].is_relative_to(roots[1]) or roots[1].is_relative_to(roots[0]):
            raise ValueError("Saved result and historical selection roots must be separate")
    if selection_snapshot is not None and source_roots is None:
        raise ValueError("Newton selection snapshot requires explicit returned source roots.")
    directories = tuple(dict.fromkeys(Path(folder).resolve() for folder in report_dirs))
    if selected_report is None:
        paths = sorted({path.resolve() for directory in directories
                        for path in directory.glob(f"{benchmark}__*.json")})
    else:
        selected = Path(selected_report)
        if selected.is_absolute():
            paths = [selected.resolve()]
        elif selected.parent == Path("."):
            paths = [directory / selected for directory in directories
                     if (directory / selected).exists()]
        else:
            raise ValueError("Select an absolute report path or a filename in the report directories")
        if any(path.parent not in directories for path in paths):
            raise ValueError("Selected report is outside the declared report directories")
        paths = [path for path in paths if path.is_file()]
    eligible = []
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or not isinstance(saved.get("identity"), dict):
            raise ValueError(f"Saved report identity is missing or malformed: {path}")
        identity = saved["identity"]
        if identity.get("schema") != SCHEMA:
            raise ValueError(f"Saved report has an unsupported identity schema: {path}")
        if any(identity.get(key) != value for key, value in {**required, **expected}.items()):
            if selected_report is not None:
                raise ValueError(f"Selected report differs from the requested identity: {path}")
            continue
        eligible.append((saved, path))
    if len(eligible) > 1:
        raise ValueError("Multiple eligible saved reports; select an explicit report file: "
                         + ", ".join(str(path) for _, path in eligible))
    if not eligible:
        return None, None
    saved, path = eligible[0]
    _validate_saved_report(saved, path, benchmark=benchmark, seeds=seeds,
                           source_roots=source_roots, selection_snapshot=selection_snapshot)
    return saved, path


def benchmark_context(rows, *, task, label_column, split_column):
    """Bind current rows, loader-precision labels and supplied partitions, without caches."""
    labels = rows[label_column].to_numpy(dtype=np.int64 if task == "classification" else np.float32)
    splits = {name: np.flatnonzero(rows[split_column].to_numpy() == name)
              for name in ("train", "val", "test")}
    return {
        "rows": run_identity._rows_record(rows, ("sequence", "length")),
        "targets": {**run_identity.array_record(labels), "columns": [label_column],
                    "task": task, "transform": "as_supplied"},
        "splits": {k: run_identity.array_record(v, np.int64) for k, v in splits.items()},
        "split_targets": {k: run_identity.array_record(labels[v]) for k, v in splits.items()},
    }


def collect(*, benchmark, config, original_folders, original_encoders, swaps,
            swap_root, seeds=(0, 1, 2), positive_lambdas=(0.01, 0.5, 1.0),
            selection_original_folders=None, selection_swap_root=None, selection_snapshot=None):
    """Read metadata/predictions and declare comparisons, without scoring them.

    ``swaps`` contains (modality, encoder key, needs grid control) tuples. All paths
    go through the existing recipe/identity reader; ambiguous results stop reporting.
    Separate selection roots freeze historical choices, including at new seed 0;
    only fresh result roots supply predictions and scores for comparisons.
    """
    seeds = tuple(seeds)
    if (not seeds or 0 not in seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError("Report seeds must be distinct nonnegative integers including 0")
    original_folders, swaps = tuple(original_folders), tuple(swaps)
    selection_roots = swap_results.separate_selection_roots(
        original_folders=original_folders, swap_root=swap_root,
        selection_original_folders=selection_original_folders,
        selection_swap_root=selection_swap_root)
    if selection_snapshot is not None and selection_roots is None:
        raise ValueError("A frozen selection snapshot requires separate selection roots.")
    historical = None
    if selection_roots is not None:
        historical = swap_results.read_validation_selection(
            benchmark=benchmark, config=config, original_folders=selection_roots[0],
            original_encoders=original_encoders, swaps=swaps, swap_root=selection_roots[1],
            positive_lambdas=positive_lambdas, selection_snapshot=selection_snapshot)
    originals, replacements, issues, sources = {}, {}, [], {}

    def retained(report):
        entries = report.require_unique()
        issues.extend(report.issues)
        for entry in entries:
            if historical is not None:
                data = entry["data"]
                identity = data.get("identity")
                if not isinstance(identity, dict) or "schema" not in identity:
                    raise ValueError(f"Fresh comparison result lacks full identity: {entry['path']}")
                run_identity.validate_identity(identity, config=data["config"],
                                               benchmark=benchmark, variant=data["variant"])
            path = entry["path"]
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return entries

    for seed in seeds:
        originals[seed] = retained(swap_results.read_results(
            original_folders, benchmark=benchmark, config={**config, "seed": seed},
            encoders=original_encoders, native_fallback=True,
            mil_lambdas=(historical["mil_lambda"],) if historical else positive_lambdas))
    if historical is not None:
        penalty = historical["mil_lambda"]
    else:
        penalties = [e for e in originals[0] if e["data"]["variant"] == "mil"]
        if {e["data"]["config"]["entropy_lambda"] for e in penalties} != set(positive_lambdas):
            raise ValueError("Complete original seed-0 positive MIL grid required")
        penalty = max(penalties, key=lambda e: e["data"]["val_score"])["data"]["config"]["entropy_lambda"]

    def variants(entries):
        return {e["data"]["variant"]: e for e in entries
                if e["data"]["variant"] != "mil"
                or e["data"]["config"]["entropy_lambda"] == penalty}

    models, choices, contrasts = {}, {}, []
    for seed, entries in originals.items():
        for name, entry in variants(entries).items():
            models.setdefault(f"original/{name}", {})[seed] = entry

    for modality, encoder, grid in swaps:
        for seed in seeds:
            entries = retained(swap_results.read_results(
                [Path(swap_root) / encoder], benchmark=benchmark,
                config={**config, "seed": seed},
                encoders={**original_encoders, modality: encoder}, mil_lambda=penalty,
                variants=swap_results.swap_variants(modality, grid=grid)))
            combined = {n: e for n, e in variants(originals[seed]).items()
                        if n not in HEADS and not n.endswith(f":{modality}")}
            combined.update(variants(entries))
            replacements[encoder, seed] = combined
            for name, entry in combined.items():
                models.setdefault(f"{encoder}/{name}", {})[seed] = entry
        if historical is not None:
            best = historical["best_singles"][encoder]
        else:
            selection = replacements[encoder, 0]
            required = {f"unimodal:{m}" for m in original_encoders}
            if grid:
                required.add(f"unimodal-grid:{modality}")
            singles = [n for n in selection if n.startswith("unimodal")]
            best = None
            if required <= selection.keys():
                best, _ = swap_results.validation_choices(
                    list(selection.values()), singles=singles, positive_lambdas=())
        choices[encoder] = best

        def contrast(kind, name, first, second):
            contrasts.append({"encoder": encoder, "kind": kind, "comparison": name,
                              "first": first, "second": second})

        native = f"unimodal:{modality}"
        contrast("representation", "replacement single vs original single",
                 f"{encoder}/{native}", f"original/{native}")
        for head in HEADS:
            contrast("replacement", f"{head} vs original {head}",
                     f"{encoder}/{head}", f"original/{head}")
            contrast("complementarity", f"{head} vs validation-selected single",
                     f"{encoder}/{head}", f"{encoder}/{best}" if best else None)
        for head in ("mil", "cross"):
            contrast("method", f"{head} vs concat",
                     f"{encoder}/{head}", f"{encoder}/concat")
    return {"benchmark": benchmark, "config": dict(config), "seeds": seeds,
            "models": models, "comparisons": contrasts, "selection_seed": 0,
            "best_singles": choices, "mil_lambda": penalty, "sources": sources,
            "selection_sources": historical["sources"] if historical else {},
            "selection_issues": historical["issues"] if historical else [],
            "issues": issues}


def _score(task, truth, prediction):
    if task == "classification":
        return float(np.mean(truth == prediction))
    if np.unique(truth).size < 2 or np.unique(prediction).size < 2:
        return float("nan")
    return float(spearmanr(truth, prediction).statistic)


def calculate(bundle, *, truth, groups, task, context, draws=2000, bootstrap_seed=0,
              n_classes=None):
    """Compute seed-specific and complete requested-seed group-bootstrap contrasts.

    The resampling interval conditions on the retained fitted seeds. It does not
    resample training randomness, repair split overlap or correct multiplicity.
    """
    if task not in ("classification", "regression"):
        raise ValueError("Swap reports support classification or regression")
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    truth, groups = np.asarray(truth), np.asarray(groups)
    if truth.ndim != 1 or groups.ndim != 1 or len(truth) != len(groups) or len(truth) < 2:
        raise ValueError("truth and groups need the same nonempty one-dimensional row order")
    if not np.isfinite(truth).all() or any(
            x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x))
            for x in groups.tolist()):
        raise ValueError("truth and group identities must be present and finite")
    if task == "classification" and (np.any(truth < 0) or np.any(truth != np.floor(truth))):
        raise ValueError("Classification targets must be nonnegative integer class IDs")
    # The trainer scored loader-precision targets. Casting only for correspondence
    # checks can leave a different Spearman ranking when CSV values round to ties.
    truth = truth.astype(np.int64 if task == "classification" else np.float32)
    if not np.isfinite(truth).all():
        raise ValueError("Targets are not finite at the training loader's precision")
    loader_truth = truth.tolist()
    arrays, scores, problems = {}, {}, []
    needed = {c[k] for c in bundle["comparisons"] for k in ("first", "second") if c[k]}
    for name in sorted(needed):
        for seed in bundle["seeds"]:
            entry = bundle["models"].get(name, {}).get(seed)
            problem = "completed fit absent" if entry is None else swap_results.prediction_issue(
                entry["data"], loader_truth, task=task, n_classes=n_classes)
            if entry is not None and problem is None:
                recorded = (entry["data"].get("identity") or {}).get("inputs", {})
                for field in ("rows", "targets", "splits", "split_targets"):
                    if field in recorded and recorded[field] != context[field]:
                        problem = f"recorded {field} differ from current benchmark"
                        break
            if problem:
                problems.append({"model": name, "seed": seed, "reason": problem})
                continue
            values = np.asarray(entry["data"]["test_prediction"])
            value = _score(task, truth, values)
            if not np.isfinite(value):
                problems.append({"model": name, "seed": seed, "reason": "undefined score"})
                continue
            arrays[name, seed], scores[name, seed] = values, value

    # Draw the same sequence groups for every model and seed. Score each model only
    # once per draw even when it participates in several contrasts.
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
    rng = np.random.default_rng(bootstrap_seed)
    sampled = {key: np.full(draws, np.nan) for key in arrays}
    for index in range(draws if len(members) > 1 else 0):
        pick = np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
        for key, prediction in arrays.items():
            sampled[key][index] = _score(task, truth[pick], prediction[pick])

    summaries, per_seed = [], []
    scale = 100.0 if task == "classification" else 1.0
    for contrast in bundle["comparisons"]:
        first, second = contrast["first"], contrast["second"]
        available = tuple(s for s in bundle["seeds"] if (first, s) in scores and (second, s) in scores)
        common = {**contrast, "available_seeds": list(available),
                  "missing_seeds": [s for s in bundle["seeds"] if s not in available]}

        def statistics(seeds):
            differences = [scores[first, s] - scores[second, s] for s in seeds]
            spread = np.mean([sampled[first, s] - sampled[second, s] for s in seeds], axis=0)
            finite = spread[np.isfinite(spread)]
            # Degenerate rank-correlation draws are visible. Do not silently claim
            # a full requested bootstrap when some group draws have no defined score.
            low, high = np.percentile(finite, (2.5, 97.5)) if len(finite) == draws else (None, None)
            return {
                "first_score": float(np.mean([scores[first, s] for s in seeds])) * scale,
                "second_score": float(np.mean([scores[second, s] for s in seeds])) * scale,
                "difference": float(np.mean(differences)) * scale,
                "seed_sd": float(np.std(differences, ddof=1)) * scale if len(seeds) > 1 else None,
                "ci_low": float(low) * scale if low is not None else None,
                "ci_high": float(high) * scale if high is not None else None,
                "first_wins": float(np.mean(finite > 0)) if len(finite) == draws else None,
                "valid_draws": int(len(finite)), "requested_draws": draws,
                "interval_issue": ("fewer than two sequence groups" if len(members) < 2
                                   else "undefined resampled score" if len(finite) != draws else ""),
            }

        for seed in available:
            per_seed.append({**contrast, "seed": seed, **statistics((seed,))})
        if available == tuple(bundle["seeds"]):
            summaries.append({**common, "status": "complete", **statistics(available)})
        else:
            summaries.append({**common, "status": "incomplete", "first_score": None,
                              "second_score": None, "difference": None, "seed_sd": None,
                              "ci_low": None, "ci_high": None, "first_wins": None})
    return {"per_seed": per_seed, "seed_summary": summaries, "issues": problems,
            "rows": len(truth), "groups": len(members),
            "units": "accuracy percentage points" if task == "classification" else "Spearman",
            "interval_scope": "test-sequence groups conditional on fitted seeds; unadjusted",
            "row_provenance": "Current targets/order and recorded identities checked where present; "
                              "matching labels do not establish unrecorded historical input identity."}


def report(bundle, *, truth, groups, task, context, output_dir, draws=2000,
           bootstrap_seed=0, n_classes=None, repository=None):
    """Reuse an identical analysis or publish a new immutable content-identified JSON."""
    identity = {"schema": SCHEMA, "benchmark": bundle["benchmark"],
                "config": bundle["config"], "seeds": list(bundle["seeds"]),
                "selection_seed": 0, "best_singles": bundle["best_singles"],
                "mil_lambda": bundle["mil_lambda"], "comparisons": bundle["comparisons"],
                "sources": bundle["sources"], "context": context,
                "truth": run_identity.array_record(truth),
                "groups_sha256": _digest(np.asarray(groups).tolist()), "task": task,
                "scoring_target_precision": "int64" if task == "classification" else "float32",
                "n_classes": n_classes, "draws": draws, "bootstrap_seed": bootstrap_seed,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(swap_results.__file__),
                                            Path(run_identity.__file__))}}
    if bundle.get("selection_sources"):
        identity["selection_policy"] = "fixed-historical-seed0-validation"
        identity["selection_sources"] = bundle["selection_sources"]
    # Refuse a source changed since collection; a digest cannot describe arrays read earlier.
    for source, sha in {**bundle["sources"], **bundle.get("selection_sources", {})}.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Saved result changed during reporting: {source}")
    path = Path(output_dir) / f"{bundle['benchmark']}__{_digest(identity)[:20]}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved.get("identity") != identity or not isinstance(saved.get("analysis"), dict)
                or saved.get("analysis_sha256") != _digest(saved["analysis"])):
            raise ValueError(f"Existing analysis is incomplete or incompatible; preserved: {path}")
        return saved, path, False
    analysis = calculate(bundle, truth=truth, groups=groups, task=task, context=context,
                         draws=draws, bootstrap_seed=bootstrap_seed, n_classes=n_classes)
    saved = {"identity": identity, "analysis": analysis, "analysis_sha256": _digest(analysis),
             "reader_issues": bundle["issues"],
             "selection_reader_issues": bundle.get("selection_issues", []),
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "producer": run_identity.producer_record(device="cpu", repository=repository,
                                                       packages=("numpy", "scipy"))}
    encoded = json.dumps(saved, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing publication, including interrupted files.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    return saved, path, True
