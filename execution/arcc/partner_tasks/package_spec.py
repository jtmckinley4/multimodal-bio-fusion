"""Prepare partner-task source only; stage the declared tables and caches separately."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import shutil


def _load_support(filename, name):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the source updater's portable paths, Windows junction checks and stream hashes.
support = _load_support("package_update.py", "partner_package_files")
git_state = _load_support("encoder_swaps/package_spec.py", "partner_source_identity").git_state

SCHEMA = "partner-tasks.package.v1"
RECIPE = "execution/arcc/partner_tasks/"
RUNTIME = (
    "README.md", "run_partner_tasks.py", "newton_partner_tasks.sbatch",
    "batches.tsv", "input-identity.json",
)
SHARED = (
    "Code/shared_code/__init__.py",
    "Code/shared_code/encoders.py",
    "Code/shared_code/sequences.py",
    "Code/shared_code/fusion/__init__.py",
    "Code/shared_code/fusion/coupled_mamba.py",
    "Code/shared_code/fusion/entity_embeddings.py",
    "Code/shared_code/fusion/heads.py",
    "Code/shared_code/fusion/isoformer_reference.py",
    "Code/shared_code/fusion/pair_models.py",
    "Code/shared_code/fusion/pair_sources.py",
    "Code/shared_code/fusion/pair_study.py",
    "Code/shared_code/fusion/pair_tasks.py",
    "Code/shared_code/fusion/pair_training.py",
    "Code/shared_code/fusion/run_identity.py",
    "Code/shared_code/fusion/token_embeddings.py",
)
INPUTS = {
    "table": "inputs/rpi2241",
    "entity_caches": {
        "left": "inputs/pair_entities/rpi2241/left__rna__rinalmo",
        "right": "inputs/pair_entities/rpi2241/right__protein__esm2_150m",
    },
    "identity": "input-identity.json",
}
DIRECTORIES = ("outputs/logs", "outputs/rpi2241-newton")
SNAPSHOT = "source-snapshot.json"


def _sources():
    pairs = {name: name for name in SHARED}
    pairs.update({RECIPE + name: name for name in RUNTIME})
    pairs["execution/arcc/requirements.txt"] = "requirements.txt"
    return pairs


def _root(path):
    root = support._no_links(Path(path).expanduser().absolute()).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected an existing source directory: {root}")
    return root


def _source_file(root, name):
    path = support._no_links(root / support._relative_path(name)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping package source: {name}")
    return path


def inventory(source_root):
    """Hash only the recipe's source allowlist, without opening scientific inputs."""
    root = _root(source_root)
    state = git_state(root)
    files = []
    for source, destination in _sources().items():
        support._relative_path(destination)
        digest, size = support._file_identity(_source_file(root, source))
        files.append({
            "source": source, "destination": destination,
            "source_origin": "maintained_repository",
            "role": "scientific_python" if source in SHARED else "execution_runtime",
            "source_sha256": digest, "initial_copy_sha256": digest, "bytes": size,
        })
    return {
        "schema": SCHEMA, "experiment": "partner_tasks", "source_root": str(root),
        "source_host": platform.node(), **state,
        "files": sorted(files, key=lambda item: item["destination"]),
        "total_bytes": sum(item["bytes"] for item in files),
        "dataset": "rpi2241", "held_out": ["left", "right", "both"],
        "training_seeds": [0, 1, 2], "fits_per_batch": 6,
        "inputs": json.loads(json.dumps(INPUTS)), "directories": list(DIRECTORIES),
        "copied_data": False, "copied_caches": False,
        "excluded": ["notebooks", "datasets", "embedding payloads", "model weights",
                     "existing fits", "tests", ".git"],
        "research_executed": False, "newton_validated": False,
    }


def write_package(plan, destination):
    """Create an exclusive sibling folder, publishing its snapshot only after verification."""
    files = support._recorded_files(plan, "partner_tasks")
    if ({item["source"]: item["destination"] for item in plan["files"]} != _sources()
            or len(plan["files"]) != len(_sources())
            or any(item.get("source_origin") != "maintained_repository" for item in plan["files"])
            or plan.get("directories") != list(DIRECTORIES)
            or plan.get("inputs") != INPUTS):
        raise ValueError("The partner-task package must match its source-only recipe.")
    root = _root(plan["source_root"])
    target = support._no_links(Path(destination).expanduser().absolute()).resolve()
    if target.exists():
        raise FileExistsError(f"Preserve the existing destination: {target}")
    if target.parent != root.parent or target == root:
        raise ValueError("Destination must be a new sibling of the repository.")
    current = git_state(root)
    if any(plan.get(key) != value for key, value in current.items()):
        raise ValueError("Source state changed; inspect a fresh preparation inventory.")
    sources = support._verified_sources(plan, {"maintained_repository": root}, files)

    target.mkdir(exist_ok=False)
    for name, source in sources.items():
        copied = support._no_links(target / name)
        copied.parent.mkdir(parents=True, exist_ok=True)
        support._verify_file(source, files[name])
        with source.open("rb") as reader, copied.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        support._verify_file(copied, files[name])
    for name, source in sources.items():
        support._verify_file(source, files[name])
    for directory in DIRECTORIES:
        support._no_links(target / directory).mkdir(parents=True, exist_ok=True)
    record = {**plan, "captured_utc": datetime.now(timezone.utc).isoformat(),
              "package_status": "prepared; not transferred or executed"}
    with (target / SNAPSHOT).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    return record
