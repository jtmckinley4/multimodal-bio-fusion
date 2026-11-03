"""Export the retained appendix cohort only after explicit analysis-only preflight.

Run from the repository root with --cache-root and a new --output directory.
Importing this module does not parse arguments, create files or execute cells.
The exporter is a guarded adapter for the maintained appendix structure, not a
sandbox for arbitrary notebook code. A changed cell layout requires review.
"""

import argparse
import ast
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


NOTEBOOK_RELATIVE = Path("Code/stage4/tier2_blf_phaseA_appendix.ipynb")
EXPECTED_CODE_CELLS = (
    "imports", "24235bba", "settings", "benchmark-contract", "1e4967f0",
    "comparison-plan", "saved-records", "saved-export", "rows", "headers", "check-protocol", "371a2a48",
    "check-scores", "check-predictions", "c3cd94b2", "5f5685c1", "collect",
    "coverage", "execution-preview", "select", "tables", "score-figure",
    "delta-figure", "57853f8c", "uncertainty", "uncertainty-figure", "analysis-manifest",
)


def forbid_research(*args, **kwargs):
    raise RuntimeError("Analysis export must not load models/caches, fit, or rewrite research results.")


def prepared_cells(notebook):
    """Check the complete layout/settings before compiling any executable cell."""
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    ids = tuple(cell.get("id") for cell in cells)
    if ids != EXPECTED_CODE_CELLS:
        raise ValueError("Appendix code-cell structure changed; review this exporter before execution.")
    prepared = []
    for cell in cells:
        source = "".join(cell["source"])
        tree = ast.parse(source, filename="appendix:" + cell["id"])
        # Dynamic code/loading could bypass the fixed guarded-module bindings.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name in {"exec", "eval", "compile", "__import__", "reload"}:
                    raise ValueError("Dynamic code/loading is outside the reviewed appendix export.")
        if cell["id"] == "settings":
            assignments = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in {
                        "BENCHMARK", "RUN_TRAINING", "READ_SAVED_RESULTS"
                    }:
                        assignments.setdefault(target.id, []).append(node)
            if any(len(assignments.get(name, [])) != 1 for name in (
                "BENCHMARK", "RUN_TRAINING", "READ_SAVED_RESULTS"
            )):
                raise ValueError("Expected explicit benchmark, read mode and disabled-training declarations.")
            benchmark = assignments["BENCHMARK"][0]
            training = assignments["RUN_TRAINING"][0]
            reading = assignments["READ_SAVED_RESULTS"][0]
            if (not isinstance(training.value, ast.Constant) or training.value.value is not False
                    or not isinstance(reading.value, ast.Constant) or reading.value.value is not True
                    or not isinstance(benchmark.value, ast.Constant)
                    or benchmark.value.value not in {"fungal_expression", "ecoli_proteins"}):
                raise ValueError("Appendix defaults must enable reading and explicitly disable training.")
            # Modify only the in-memory execution copy, before result paths are derived.
            benchmark.value = ast.Constant("ecoli_proteins")
            # Explicit CLI export requests the historical analysis, not the default saved-file view.
            reading.value = ast.Constant(False)
            ast.fix_missing_locations(tree)
        prepared.append((cell["id"], compile(tree, "appendix:" + cell["id"], "exec")))
    return prepared


@contextmanager
def analysis_guards(runtime):
    """Install all known research entry-point guards before the first cell executes."""
    targets = (
        (runtime.train, ("run", "fit", "build")),
        (runtime.train.Result, ("save",)),
        (runtime.blf, ("build", "loader")),
        (runtime.blf.BenchmarkStreams, ("open",)),
        (runtime.tokens, ("extract", "summarize", "open_shard")),
        (runtime.encoders, ("load_encoder",)),
    )
    with ExitStack() as stack:
        # A missing expected interface is a preflight failure, not a skipped guard.
        for owner, names in targets:
            for name in names:
                stack.enter_context(patch.object(owner, name, forbid_research))
        yield


def evaluate_notebook(notebook, runtime, cache_root, after_cell=None):
    """Execute a validated analysis layout; callers must supply the guarded runtime."""
    cells = prepared_cells(notebook)
    namespace = {"__name__": "__analysis__", "train": runtime.train, "blf": runtime.blf,
                 "tokens": runtime.tokens, "encoders": runtime.encoders}
    with analysis_guards(runtime):
        for cell_id, code in cells:
            exec(code, namespace)
            if cell_id == "imports":
                namespace["display"] = lambda value: print(
                    value.to_string() if hasattr(value, "to_string") else value)
                namespace["plt"].show = lambda: None
            elif cell_id == "settings":
                namespace["CACHE_ROOT"] = cache_root
            if after_cell is not None:
                after_cell(cell_id, namespace)
    return namespace


def load_runtime(root):
    """Import the analysis dependencies without constructing a model or opening a cache."""
    code_dir = root / "Code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import matplotlib
    matplotlib.use("Agg")
    # Preserve the shared notebook environment's numerical-library import order.
    import numpy
    import pandas
    from shared_code import encoders
    from shared_code.fusion import blf, train, token_embeddings
    for module in (encoders, blf, train, token_embeddings):
        if not Path(module.__file__).resolve().is_relative_to(code_dir / "shared_code"):
            raise RuntimeError("Analysis dependencies came from another checkout.")
    return SimpleNamespace(train=train, blf=blf, tokens=token_embeddings, encoders=encoders)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "Code/shared_code/datasets.py").is_file())
    if Path.cwd().resolve() != root:
        raise RuntimeError("Run this export from the repository root.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve the earlier export; choose a new analysis directory.")
    notebook_path = root / NOTEBOOK_RELATIVE
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    prepared_cells(notebook)  # Fail closed before imports, files, or any notebook execution.
    runtime = load_runtime(root)
    # Check that every guard can be installed before creating an export destination.
    with analysis_guards(runtime):
        pass
    output.mkdir(parents=True)
    figures = output / "figures"
    figures.mkdir()

    def save_figures(cell_id, namespace):
        if "plt" not in namespace:
            return
        for number in namespace["plt"].get_fignums():
            figure = namespace["plt"].figure(number)
            for suffix in ("png", "svg"):
                figure.savefig(figures / (cell_id + "." + suffix), dpi=180)
        namespace["plt"].close("all")

    namespace = evaluate_notebook(notebook, runtime, args.cache_root.resolve(), save_figures)
    if not namespace["COMPARISON_READY"] or not namespace["UNCERTAINTY_READY"]:
        raise RuntimeError("The historical comparison/uncertainty is incomplete; no tables exported.")
    if namespace["RUN_TRAINING"]:
        raise RuntimeError("Training must remain disabled.")
    if (len(namespace["results"]) != 27 or len(namespace["selected"]) != 21
            or len(namespace["uncertainty"]) != 12):
        raise RuntimeError("Selected historical cohort differs from the declared 27/21/12 export.")
    if notebook_path.read_bytes() != notebook_bytes:
        raise RuntimeError("Notebook changed during export.")

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
        "explicit_overrides": {"BENCHMARK": "ecoli_proteins", "CACHE_ROOT": str(args.cache_root.resolve()),
                               "READ_SAVED_RESULTS": False},
        "training_calls_blocked": True,
        "guards_installed_before_first_cell": True,
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


if __name__ == "__main__":
    main()
