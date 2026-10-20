"""Export this cohort's saved-result analysis without fitting or saving notebook outputs.

From the repository root:
python -B Code/stage4/results/blf_phaseA/ecoli_proteins/records/export_analysis.py --cache-root PATH_TO_EXISTING_EMBEDDINGS --output NEW_ANALYSIS_DIRECTORY

The output directory must be new. The retained notebook supplies the calculations;
its hash and the explicitly overridden settings are recorded with the exports.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
root = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "Code/shared_code/datasets.py").is_file())
if Path.cwd().resolve() != root:
    raise RuntimeError("Run this export from the repository root.")
output = args.output.resolve()
if output.exists():
    raise FileExistsError("Preserve the earlier export; choose a new analysis directory.")
output.mkdir(parents=True)
figures = output / "figures"
figures.mkdir()
notebook_path = root / "Code/stage4/tier2_blf_phaseA.ipynb"
notebook_bytes = notebook_path.read_bytes()
notebook = json.loads(notebook_bytes)
namespace = {}


def forbid_training(*args, **kwargs):
    raise AssertionError("Analysis must not open token caches, construct models, or train.")


for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if cell["id"] == "settings":
        declaration = 'BENCHMARK = "fungal_expression"'
        if source.count(declaration) != 1:
            raise RuntimeError("Review the notebook settings before repeating this export.")
        source = source.replace(declaration, 'BENCHMARK = "ecoli_proteins"', 1)
    exec(compile(source, notebook_path.name + ":" + cell["id"], "exec"), namespace)
    if cell["id"] == "imports":
        namespace["display"] = lambda value: print(value.to_string() if hasattr(value, "to_string") else value)
        namespace["plt"].show = lambda: None
        namespace["train"].run = forbid_training
        namespace["train"].fit = forbid_training
        namespace["train"].build = forbid_training
        namespace["blf"].build = forbid_training
        namespace["blf"].BenchmarkStreams.open = forbid_training
    if cell["id"] == "settings":
        namespace["CACHE_ROOT"] = args.cache_root.resolve()
        print("Export override: CACHE_ROOT =", namespace["CACHE_ROOT"])
    if "plt" in namespace:
        for number in namespace["plt"].get_fignums():
            figure = namespace["plt"].figure(number)
            for suffix in ("png", "svg"):
                figure.savefig(figures / (cell["id"] + "." + suffix), dpi=180)
        namespace["plt"].close("all")

assert namespace["COMPARISON_READY"] and namespace["UNCERTAINTY_READY"]
assert not namespace["RUN_TRAINING"]
assert len(namespace["results"]) == 27 and len(namespace["selected"]) == 21
assert len(namespace["uncertainty"]) == 12
assert notebook_path.read_bytes() == notebook_bytes, "Notebook changed during export."

# Check diagnostics in every fitted candidate, including those validation did not select.
for result in namespace["results_by_path"].values():
    for epoch in result.history:
        for name in ("train_loss", "lr"):
            value = epoch.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Nonfinite or missing training diagnostic: " + name)

tables = {
    "coverage": namespace["coverage"],
    "validation_choices": namespace["choices"],
    "selected_scores": namespace["selected"],
    "seed_summary": namespace["summary"].reset_index(),
    "paired_differences": namespace["paired"],
    "paired_intervals": namespace["uncertainty"],
}
for name, table in tables.items():
    table.to_csv(output / (name + ".csv"), index=False)

manifest = namespace["analysis_manifest"]
manifest["export"] = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "script": Path(__file__).resolve().relative_to(root).as_posix(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "explicit_overrides": {"BENCHMARK": "ecoli_proteins", "CACHE_ROOT": str(args.cache_root.resolve())},
    "training_calls_blocked": True,
    "notebook_outputs_modified": False,
    "all_27_training_histories_finite": True,
    "table_units": "Accuracy and differences are proportions; notebook figures use percent and percentage points.",
    "selected_epoch_units": "best_epoch in selected_scores.csv is one-based; raw result epochs are zero-based.",
    "python": sys.version,
}
manifest["artifacts"] = [
    {"path": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(output.rglob("*")) if path.is_file()
]
(output / "analysis-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print("Exported:", output)
print("No models trained and no notebook outputs changed.")
