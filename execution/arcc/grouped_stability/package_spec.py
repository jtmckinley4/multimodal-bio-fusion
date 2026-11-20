"""Files and provenance for the grouped-stability Newton package."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform


def _load_support():
    here = Path(__file__).resolve().parent
    path = here / "arcc_package_support.py"
    if not path.is_file():
        path = here.parent / "encoder_swaps/package_spec.py"
    spec = importlib.util.spec_from_file_location("arcc_package_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = _load_support()
SCHEMA = "biological-fusion.grouped-stability-package"
VERSION = 1
NOTEBOOK = "Code/stage4/grouped_stability.ipynb"
DATASET = "datasets/mRNA_Stability.csv"
RECIPE = "execution/arcc/grouped_stability/"
RUNTIME = (
    "README.md", "package_spec.py", "run_grouped_stability.py", "batches.tsv",
    "newton_grouped_stability.sbatch.example", "grouped_stability_preflight.sbatch",
)
SUPPORT = {
    "execution/arcc/encoder_swaps/package_spec.py": "arcc_package_support.py",
    "execution/arcc/encoder_swaps/run_encoder_swaps.py": "arcc_execution_support.py",
    "execution/arcc/requirements.txt": "requirements.txt",
}


def build_plan(source_root, cache_root=None):
    """Hash explicitly selected source files; do not inspect embedding payloads."""
    root = support._root(source_root)
    state = support.git_state(root)
    files = []

    def add(source, destination=None, role="scientific_source"):
        files.append(support._entry(root, source, destination or source,
                                    "maintained_repository", role))

    add(NOTEBOOK, role="scientific_notebook")
    add(DATASET, role="dataset")
    shared = support._root(root / "Code/shared_code")
    for path in sorted(shared.rglob("*")):
        if support._is_link(path):
            raise ValueError(f"Linked shared source is not supported: {path}")
        if path.is_file() and path.suffix == ".py":
            add(path.relative_to(root).as_posix())
    for name in RUNTIME:
        add(RECIPE + name, name, "execution_recipe")
    for source, target in SUPPORT.items():
        add(source, target, "reused_execution_support")
    for name in ("test_arcc_grouped_stability.py", "test_grouped_stability.py"):
        add("tests/" + name, role="synthetic_test")
    required = {"Code/shared_code/fusion/grouped_stability.py",
                "Code/shared_code/fusion/run_identity.py"}
    names = [entry["destination"] for entry in files]
    if not required <= set(names):
        raise ValueError("The grouped-stability scientific helpers are incomplete.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Package file destinations collide.")
    return {
        "schema": SCHEMA, "schema_version": VERSION, "source_root": str(root),
        "source_host": platform.node(), **state,
        "experiment": "grouped_stability", "training_seeds": [0, 1, 2],
        "fits_per_seed": 12,
        "cache_reference": {"path": None if cache_root is None else
                            str(Path(cache_root).expanduser().resolve()), "copied": False},
        "files": sorted(files, key=lambda item: item["destination"]),
        "total_bytes": sum(item["bytes"] for item in files),
        "directories": ["outputs/logs", "outputs/notebooks"],
        "excluded": ["existing results", "embedding payloads", "model weights",
                     "recovery checkpoints", "historical selection inputs", ".git"],
        "research_executed": False, "newton_validated": False,
    }


def write_package(plan, destination):
    """Publish an exclusive new sibling package, leaving any failed partial copy."""
    if plan.get("schema") != SCHEMA or plan.get("schema_version") != VERSION:
        raise ValueError("Unsupported grouped-stability package schema.")
    root = support._root(plan["source_root"])
    target = Path(destination).expanduser().absolute()
    support._reject_links(target)
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Preserve the existing destination: {target}")
    if target.parent != root.parent or support._overlap(target, root):
        raise ValueError("Destination must be a new sibling of the repository.")
    cache = plan["cache_reference"]["path"]
    if cache and support._overlap(target, Path(cache).resolve()):
        raise ValueError("The new workspace overlaps the declared cache root.")
    current = support.git_state(root)
    if any(plan[key] != value for key, value in current.items()):
        raise ValueError("Source state changed; inspect a fresh preparation inventory.")
    names = [item["destination"] for item in plan["files"]]
    if len(set(name.casefold() for name in names)) != len(names):
        raise ValueError("Package file destinations collide.")
    if "source-snapshot.json" in names:
        raise ValueError("The completion manifest is not a copied input.")

    def checked_source(item):
        support._relative(item["destination"])
        path = support._source_file(root, item["source"])
        expected = (item["source_sha256"], item["bytes"])
        if (support.file_identity(path) != expected or
                item["initial_copy_sha256"] != item["source_sha256"]):
            raise ValueError(f"Source changed: {item['source']}")
        return path

    for item in plan["files"]:
        checked_source(item)
    for relative in plan["directories"]:
        support._relative(relative)
    target.mkdir(exist_ok=False)
    for item in plan["files"]:
        source = checked_source(item)
        copied = target / item["destination"]
        support._reject_links(copied)
        if not copied.resolve().is_relative_to(target):
            raise ValueError("Copied path escapes the new workspace.")
        copied.parent.mkdir(parents=True, exist_ok=True)
        support._copy_file(source, copied)
        if support.file_identity(copied) != (item["source_sha256"], item["bytes"]):
            raise ValueError(f"Copy verification failed: {copied}")
    for item in plan["files"]:
        checked_source(item)
    for relative in plan["directories"]:
        (target / relative).mkdir(parents=True, exist_ok=True)
    record = {**plan, "captured_utc": datetime.now(timezone.utc).isoformat(),
              "package_status": "prepared; not transferred or executed"}
    with (target / "source-snapshot.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    return record
