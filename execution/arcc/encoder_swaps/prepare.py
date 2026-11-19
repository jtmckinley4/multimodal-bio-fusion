"""Inventory a new ARCC execution snapshot; copy only with --destination.

The maintained repository supplies scientific source and runtime files. An
explicit selection workspace supplies the hash-bound original CSVs and historical
selection records. Frozen arrays, weights and recovery checkpoints are never
inventoried or copied by this command. It does not execute or submit research.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import subprocess


SCHEMA = "biological-fusion.arcc-package"
SCHEMA_VERSION = 2
NOTEBOOK = "Code/stage4/encoder_swaps.ipynb"
# The notebook's final context lookup reads this completed capture even in training mode.
HISTORICAL_CONTEXT = ("docs/figures/stability/results.json",)
REQUIRED_SHARED = (
    "Code/shared_code/__init__.py", "Code/shared_code/fusion/swap_plan.py",
    "Code/shared_code/fusion/result_sets.py",
)
DATASETS = (
    "datasets/CoV_Vaccine_Degradation.csv",
    "datasets/E.Coli_proteins.csv",
    "datasets/Fungal_expression.csv",
    "datasets/mRNA_Stability.csv",
)
HISTORICAL_ROOTS = (
    "historical_results/Code/stage3/results/fungal_expression",
    "historical_results/Code/stage4/results/phaseA",
    "historical_results/Code/stage4/results/phaseB",
)
RUNTIME_FILES = (
    "README.md", "batches.tsv", "newton_preflight.sbatch.example",
    "newton_report.sbatch.example", "newton_swaps.sbatch.example", "prepare.py",
    "requirements-notebooks.txt", "run_encoder_swaps.py", "newton_ernie.sbatch.example",
    "ecoli_ernie_batches.tsv",
)
RUNTIME_TESTS = (
    "test_arcc_swap_selection.py", "test_arcc_execution.py",
    "test_arcc_batch_startup.py", "test_arcc_live_output.py", "test_arcc_ernie_extension.py",
)
SNAPSHOT = "source-snapshot.json"
SELECTION_EVIDENCE = "evidence/selection-source-snapshot.json"


def _relative(value):
    """Require a portable, normalized file path rather than repairing one."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid relative inventory path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"Invalid relative inventory path: {value!r}")
    return path


def _is_link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _reject_links(path):
    for component in (path, *path.parents):
        if _is_link(component):
            raise ValueError(f"Symlinks and junctions are not package sources/destinations: {component}")


def _root(path):
    path = Path(path).expanduser().absolute()
    _reject_links(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Expected an existing directory: {resolved}")
    return resolved


def _source_file(root, relative):
    path = root.joinpath(*_relative(relative).parts)
    _reject_links(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing or escaping package source: {relative}")
    return resolved


def file_identity(path):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def git_state(source_root):
    """Read source identity without modifying the index or contacting a remote."""
    def git(*arguments):
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source_root), *arguments],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout.rstrip("\n")

    if Path(git("rev-parse", "--show-toplevel")).resolve() != source_root:
        raise ValueError("--source-root must be the maintained Git repository root.")
    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("The source repository did not supply a valid commit identity.")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"source_commit": commit, "source_git_status": status, "source_dirty": bool(status)}


def _overlap(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def _entry(root, source, destination, origin, role):
    _relative(destination)
    digest, size = file_identity(_source_file(root, source))
    return {
        "source": source, "destination": destination, "source_origin": origin,
        "role": role, "source_sha256": digest, "initial_copy_sha256": digest,
        "bytes": size,
    }


def _selection_entries(workspace):
    snapshot_path = _source_file(workspace, SNAPSHOT)
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") not in (1, 2):
        raise ValueError("Selection evidence requires a version 1 or 2 source-snapshot.json.")
    if not isinstance(snapshot.get("files"), list):
        raise ValueError("Selection snapshot has no file inventory.")
    if not isinstance(snapshot.get("source_commit"), str) or not snapshot["source_commit"]:
        raise ValueError("Selection snapshot must identify its original source commit.")

    entries, destinations, datasets_found, roots_found = [], set(), set(), set()
    for recorded in snapshot["files"]:
        if not isinstance(recorded, dict):
            raise ValueError("Malformed selection snapshot inventory entry.")
        relative = recorded.get("destination")
        _relative(relative)
        identity = relative.casefold()
        if identity in destinations:
            raise ValueError(f"Duplicate selection snapshot destination: {relative}")
        destinations.add(identity)
        historical = next((root for root in HISTORICAL_ROOTS if relative.startswith(root + "/")), None)
        if relative.startswith("datasets/") and relative not in DATASETS:
            raise ValueError(f"Unexpected dataset in the selected source snapshot: {relative}")
        if relative.startswith("historical_results/") and (historical is None or not relative.endswith(".json")):
            raise ValueError(f"Unsupported historical selection input: {relative}")
        if relative not in DATASETS and historical is None:
            continue
        expected = recorded.get("initial_copy_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Missing initial-copy hash for preserved input: {relative}")
        item = _entry(workspace, relative, relative, "historical_selection_workspace",
                      "dataset" if relative in DATASETS else "historical_selection")
        if item["source_sha256"] != expected:
            raise ValueError(f"Preserved input changed: {relative}; inspect the original evidence.")
        item["selection_original_source"] = recorded.get("source")
        item["selection_original_source_sha256"] = recorded.get("source_sha256")
        entries.append(item)
        if relative in DATASETS:
            datasets_found.add(relative)
        else:
            roots_found.add(historical)
    if datasets_found != set(DATASETS):
        raise ValueError("Selection snapshot must contain all four named CSV inputs.")
    if roots_found != set(HISTORICAL_ROOTS):
        raise ValueError("Selection snapshot must retain fungal, Phase A and Phase B selection evidence.")
    evidence = _entry(workspace, SNAPSHOT, SELECTION_EVIDENCE,
                      "historical_selection_workspace", "selection_manifest")
    if evidence["source_sha256"] != hashlib.sha256(snapshot_bytes).hexdigest():
        raise ValueError("Selection source snapshot changed during inventory.")
    entries.append(evidence)
    identity = {
        "workspace": str(workspace), "snapshot_sha256": evidence["source_sha256"],
        "snapshot_schema_version": snapshot["schema_version"],
        "source_commit": snapshot["source_commit"],
        "source_root": snapshot.get("source_root"),
        "captured_utc": snapshot.get("captured_utc"),
        "preserved_manifest": SELECTION_EVIDENCE,
        "meaning": "Original CSV and fixed-selection bytes, not fresh result coverage or a new selection.",
    }
    return entries, identity


def build_plan(source_root, selection_workspace, cache_root=None, *, experiment="full"):
    """Return a deterministic inventory without creating any directory or artifact."""
    source_root, selection_workspace = _root(source_root), _root(selection_workspace)
    if experiment not in {"full", "ecoli_ernie"}:
        raise ValueError("Choose the full panel or the focused ecoli_ernie extension.")
    if _overlap(source_root, selection_workspace):
        raise ValueError("The maintained repository and selection workspace must be separate.")
    state = git_state(source_root)
    files = [_entry(source_root, NOTEBOOK, NOTEBOOK, "maintained_repository", "scientific_notebook")]
    for relative in HISTORICAL_CONTEXT:
        files.append(_entry(source_root, relative, relative, "maintained_repository", "historical_context"))
    shared = _root(source_root / "Code/shared_code")
    if not shared.is_relative_to(source_root):
        raise ValueError("Shared scientific source must be inside the maintained repository.")
    python_files = []
    for path in shared.rglob("*"):
        if _is_link(path):
            raise ValueError(f"Linked shared-code source is not supported: {path}")
        if path.is_file() and path.suffix == ".py":
            python_files.append(path.relative_to(source_root).as_posix())
    if not set(REQUIRED_SHARED) <= set(python_files):
        raise ValueError(f"Missing required shared execution source: {sorted(set(REQUIRED_SHARED) - set(python_files))}")
    for relative in sorted(python_files):
        files.append(_entry(source_root, relative, relative, "maintained_repository", "scientific_python"))
    for filename in RUNTIME_FILES:
        files.append(_entry(source_root, "execution/arcc/encoder_swaps/" + filename, filename,
                            "maintained_repository", "execution_runtime"))
    for filename in RUNTIME_TESTS:
        relative = "tests/" + filename
        files.append(_entry(source_root, relative, relative, "maintained_repository", "synthetic_test"))
    selection_files, selection_identity = _selection_entries(selection_workspace)
    if experiment == "ecoli_ernie":
        # Keep the original manifest, copying only this task's data and original
        # MIL tuning evidence. New RNA encoders have no historical single fits.
        selection_files = [item for item in selection_files if (
            item["destination"] in {"datasets/E.Coli_proteins.csv", SELECTION_EVIDENCE}
            or item["destination"].startswith("historical_results/Code/stage4/results/phaseA/ecoli_proteins/")
        )]
        if not any(item["role"] == "historical_selection" for item in selection_files):
            raise ValueError("The extension requires preserved E. coli Phase A MIL selection records.")
    files.extend(selection_files)
    files.sort(key=lambda item: item["destination"])
    if len({item["destination"].casefold() for item in files}) != len(files):
        raise ValueError("Package destination paths collide.")
    declared_cache = None if cache_root is None else str(Path(cache_root).expanduser().resolve())
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "experiment": experiment,
        "purpose": "New isolated execution snapshot; no transfer, launch or research execution",
        "source_host": platform.node(), "source_root": str(source_root), **state,
        "historical_selection": selection_identity,
        "cache_reference": {
            "path": declared_cache, "copied": False,
            "verification": "Explicit runtime preflight must verify required frozen caches at --cache-root.",
        },
        "files": files, "total_bytes": sum(item["bytes"] for item in files),
        "directories": (["outputs/logs", "outputs/ecoli_ernie/logs", "outputs/ecoli_ernie/notebooks"]
                        if experiment == "ecoli_ernie" else ["outputs/logs", "outputs/notebooks"]),
        "declared_workload": ({"benchmark": "ecoli_proteins", "encoders": ["rnaernie", "ernierna"],
                               "heads": ["mil", "concat"], "fits_per_seed": 4,
                               "default_training_seed": 0, "available_training_seeds": [0, 1, 2],
                               "new_rna_extractions": 2, "single_baseline_selection": "not available"}
                              if experiment == "ecoli_ernie" else None),
        "excluded": ["embedding payloads and cache recovery metadata", "model weights",
                     "recovery checkpoints", "fresh outputs", ".git"],
        "research_executed": False, "newton_validated": False,
    }


def _destination(plan, destination):
    source = _root(plan["source_root"])
    selection = _root(plan["historical_selection"]["workspace"])
    candidate = Path(destination).expanduser().absolute()
    _reject_links(candidate)
    candidate = candidate.resolve()
    if candidate.exists():
        raise FileExistsError(f"Destination already exists; preserve it: {candidate}")
    if candidate.parent != source.parent:
        raise ValueError("Destination must be a new sibling directory of the maintained repository.")
    protected = [source, selection]
    cache = plan["cache_reference"]["path"]
    if cache is not None:
        protected.append(Path(cache).resolve())
    if any(_overlap(candidate, root) for root in protected):
        raise ValueError("Destination overlaps a source, selection workspace or declared cache root.")
    return candidate


def _verify_entry(plan, item):
    roots = {
        "maintained_repository": _root(plan["source_root"]),
        "historical_selection_workspace": _root(plan["historical_selection"]["workspace"]),
    }
    if item.get("source_origin") not in roots:
        raise ValueError("Unknown file source origin in package inventory.")
    _relative(item["destination"])
    source = _source_file(roots[item["source_origin"]], item["source"])
    if file_identity(source) != (item["source_sha256"], item["bytes"]):
        raise ValueError(f"Source changed after planning: {item['source']}")
    if item["initial_copy_sha256"] != item["source_sha256"]:
        raise ValueError("A package cannot rewrite source bytes during copying.")
    return source


def _copy_file(source, destination):
    """Exclusive publication prevents overwriting an existing file in a race."""
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)


def write_package(plan, destination):
    """Create a new sibling package; failed partial work remains without a manifest."""
    if plan.get("schema") != SCHEMA or plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported package plan schema.")
    destination = _destination(plan, destination)
    files = plan["files"]
    if len({item["destination"].casefold() for item in files}) != len(files):
        raise ValueError("Package destination paths collide.")
    if SNAPSHOT.casefold() in {item["destination"].casefold() for item in files}:
        raise ValueError("The completion manifest cannot be supplied as a copied input.")
    current_state = git_state(_root(plan["source_root"]))
    if any(plan[key] != value for key, value in current_state.items()):
        raise ValueError("Source commit or working-tree status changed after planning; review a new inventory.")
    # Check every byte identity before the first destination write.
    for item in files:
        _verify_entry(plan, item)
    for relative in plan["directories"]:
        _relative(relative)
    destination.mkdir(exist_ok=False)
    for item in files:
        source = _verify_entry(plan, item)
        target = destination.joinpath(*_relative(item["destination"]).parts)
        _reject_links(target)
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f"Escaping package destination: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(source, target)
        if file_identity(target) != (item["initial_copy_sha256"], item["bytes"]):
            raise ValueError(f"Copied bytes failed verification: {target}")
    # Recheck sources after all copies; a partial package never receives completion evidence.
    for item in files:
        _verify_entry(plan, item)
    # Slurm opens its log path before the launcher starts; create only empty folders.
    for relative in plan["directories"]:
        directory = destination.joinpath(*_relative(relative).parts)
        _reject_links(directory)
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {**plan, "captured_utc": datetime.now(timezone.utc).isoformat(),
                "package_status": "prepared; not transferred or executed"}
    _reject_links(destination)
    with (destination / SNAPSHOT).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--selection-workspace", type=Path, required=True,
                        help="Explicit original selection workspace with source-snapshot.json")
    parser.add_argument("--cache-root", type=Path,
                        help="Declared external cache reference only; never scanned or copied")
    parser.add_argument("--destination", type=Path,
                        help="Create this NEW sibling directory; omit for a read-only JSON inventory")
    parser.add_argument("--experiment", choices=("full", "ecoli_ernie"), default="full")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.source_root, args.selection_workspace, args.cache_root,
                          experiment=args.experiment)
        result = plan if args.destination is None else write_package(plan, args.destination)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Package preparation stopped: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
