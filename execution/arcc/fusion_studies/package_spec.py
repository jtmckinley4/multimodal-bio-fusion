"""Inventory or copy a new source-only A2-A6 workspace; no datasets or caches."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

NOTEBOOKS = ("anchor_correction", "gate_diagnostics", "anchor_cross_attention", "region_aware_fusion")
SCHEMA = "fusion-studies.package.v1"
TESTS = ("test_arcc_fusion_studies.py", "test_arcc_fusion_inputs.py",
         "test_arcc_fusion_package.py", "test_controlled_models.py",
         "test_controlled_experiment.py", "test_region_training.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    root = Path(root).resolve()
    pairs = {f"Code/stage4/{name}.ipynb": f"Code/stage4/{name}.ipynb" for name in NOTEBOOKS}
    for path in (root / "Code/shared_code").rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        pairs[relative] = relative
    recipe = root / "execution/arcc/fusion_studies"
    for path in recipe.iterdir():
        if path.is_file() and path.suffix in {".py", ".sbatch", ".example", ".json", ".tsv"}:
            pairs[path.relative_to(root).as_posix()] = path.name
    pairs["execution/arcc/fusion_studies/README.md"] = "README.md"
    pairs["execution/arcc/requirements.txt"] = "requirements.txt"
    pairs["execution/arcc/encoder_swaps/run_encoder_swaps.py"] = "arcc_execution_support.py"
    for name in TESTS:
        pairs[f"tests/{name}"] = f"tests/{name}"
    destinations = [name.casefold() for name in pairs.values()]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Package destinations collide.")
    files = []
    for source, destination in sorted(pairs.items()):
        path = root / source
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or linked source: {source}")
        files.append({"source": source, "destination": destination,
                      "sha256": digest(path), "bytes": path.stat().st_size})
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True)
    return {"schema": SCHEMA, "files": files, "source_root": str(root),
            "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "total_bytes": sum(item["bytes"] for item in files),
            "copied_data": False, "copied_caches": False, "research_executed": False}


def write_package(plan, destination):
    """Write the inventoried sources to a new sibling of the checkout."""
    root, destination = Path(plan["source_root"]), Path(destination).absolute()
    if destination.exists() or destination.parent.resolve() != root.parent:
        raise ValueError("Choose a new sibling directory; existing destinations are preserved.")
    destination.mkdir(exist_ok=False)
    for item in plan["files"]:
        source, target = root / item["source"], destination / item["destination"]
        if digest(source) != item["sha256"]:
            raise ValueError("Source changed during preparation; retain this incomplete package.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
        if digest(target) != item["sha256"]:
            raise ValueError(f"Copy differs: {target}")
    for item in plan["files"]:
        if digest(root / item["source"]) != item["sha256"]:
            raise ValueError("Source changed before publication; no complete snapshot published.")
    for name in ("logs", "notebooks"):
        (destination / "outputs" / name).mkdir(parents=True, exist_ok=True)
    plan["prepared_utc"] = datetime.now(timezone.utc).isoformat()
    with (destination / "source-snapshot.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    return plan
