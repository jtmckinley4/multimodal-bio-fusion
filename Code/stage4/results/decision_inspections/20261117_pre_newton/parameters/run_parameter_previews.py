"""One-off, CPU-only execution of the two notebooks' declared parameter previews.

Run from the checkout with python -B. This saved runner executes the listed cells,
not the research notebooks. It refuses to replace an earlier inspection artifact.
"""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = next(parent for parent in OUT.parents if (parent / "Code/shared_code").is_dir())
if list(OUT.glob("*.json")) or list(OUT.glob("*.ipynb")) or (OUT / "source_snapshot.zip").exists():
    raise FileExistsError("This inspection destination already has evidence; preserve it.")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save_json(name, payload):
    with (OUT / name).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


start = datetime.now(timezone.utc).isoformat()
selected = ("imports", "paths", "settings", "execution-guards", "panel",
            "configuration-helpers", "parameter-preview")
source_paths = [ROOT / "Code/stage4" / f"{name}.ipynb"
                for name in ("gate_diagnostics", "anchor_cross_attention")]
snapshots = {str(path.relative_to(ROOT)).replace("\\", "/"): path.read_bytes()
             for path in [*source_paths, *sorted((ROOT / "Code/shared_code").rglob("*.py"))]}
record = {
    "run_id": start + "_gate_attention_parameter_previews", "start_utc": start,
    "purpose": "Inspect A.3-A.5 model parameter budgets before Newton execution; Gantt/Details tasks 46 and 25.",
    "authorization": "User authorized inexpensive inspections and parameter previews, preserving existing results; no training or jobs.",
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    "source_notebooks": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p.read_bytes()) for p in source_paths},
    "selected_code_cells": list(selected), "overrides": {"RUN_PARAMETER_PREVIEW": True},
    "bootstrap": "Expose checkout Code on sys.path before the unchanged imports cell; select CPU/default float32 and one torch thread. Capture display tables without changing calculations.",
    "not_executed": ["datasets", "embedding caches", "model checkpoint loading", "forward inference",
                     "training", "saved-prediction analysis", "bootstrap", "network", "job submission"],
    "command": [sys.executable, "-B", str(Path(__file__).resolve())],
    "environment_overrides": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "MPLBACKEND")},
    "initialization_seed": "unset; only parameter counts inspected; no forward pass or fitting",
    "runner_sha256": sha(Path(__file__).read_bytes()), "executions": [],
}
save_json("intent.json", record)
failure = None
try:
    import nbformat
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    assert torch.get_default_dtype() == torch.float32
    record["producer"] = {"host": platform.node(), "platform": platform.platform(),
                          "processor": platform.processor(), "python": sys.version,
                          "python_executable": sys.executable, "device": "cpu",
                          "torch_threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
                          "packages": {"torch": torch.__version__, "numpy": np.__version__,
                                       "pandas": pd.__version__, "nbformat": nbformat.__version__}}
    sys.path.insert(0, str(ROOT / "Code"))
    os.chdir(ROOT)
    all_tables = {}
    for source_path in source_paths:
        name = source_path.stem
        original = nbformat.read(source_path, as_version=4)
        original_cells = {cell.id: cell for cell in original.cells}
        source_relative = Path(os.path.relpath(source_path, OUT)).as_posix()
        preview = nbformat.v4.new_notebook(metadata={"source_notebook": source_relative,
            "source_sha256": record["source_notebooks"][str(source_path.relative_to(ROOT)).replace("\\", "/")],
            "execution_scope": "Selected setup/parameter-preview cells only; no research analysis."})
        preview.cells.append(nbformat.v4.new_markdown_cell(
            f"# CPU parameter inspection: {name.replace('_', ' ')}\n\n"
            f"Executed only the setup and parameter-preview cells from the [maintained notebook]({source_relative}). "
            "This measures trainable parameter counts, not predictive performance, runtime or peak memory. "
            "Settings below are retained; only the preview switch is enabled in this copy. "
            "No dataset, cache, checkpoint or prediction was consumed. See [execution.json](execution.json) for provenance."))
        bootstrap = nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nimport torch\n"
            "root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'Code/shared_code').is_dir())\n"
            "sys.path.insert(0, str(root / 'Code'))\ntorch.set_num_threads(1)\ntorch.set_default_device('cpu')\n")
        namespace, tables, executed = {}, [], []
        ordered = [bootstrap]
        for identity in selected:
            cell = deepcopy(original_cells[identity])
            cell.outputs, cell.execution_count = [], None
            cell.metadata["source_cell_id"] = identity
            ordered.append(cell)
            if identity == "settings":
                ordered.append(nbformat.v4.new_code_cell(
                    "RUN_PARAMETER_PREVIEW = True\n"
                    "assert not any((RUN_TRAINING, RUN_ANALYSIS, RUN_DIAGNOSTICS))\n"))
        began = datetime.now(timezone.utc).isoformat()
        for count, cell in enumerate(ordered, 1):
            stream, err, outputs = io.StringIO(), io.StringIO(), []
            def capture_display(value):
                if isinstance(value, pd.DataFrame):
                    tables.append(json.loads(value.to_json(orient="records")))
                    outputs.append(nbformat.v4.new_output("display_data", data={
                        "text/plain": value.to_string(index=False), "text/html": value.to_html(index=False)}))
                else:
                    outputs.append(nbformat.v4.new_output("display_data", data={"text/plain": repr(value)}))
            namespace["display"] = capture_display
            try:
                with redirect_stdout(stream), redirect_stderr(err):
                    exec(compile(cell.source, f"{source_relative}:{cell.id}", "exec"), namespace)
            except BaseException as exc:
                outputs.append(nbformat.v4.new_output("error", ename=type(exc).__name__,
                    evalue=str(exc), traceback=traceback.format_exc().splitlines()))
                raise
            finally:
                if stream.getvalue(): outputs.insert(0, nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
                if err.getvalue(): outputs.append(nbformat.v4.new_output("stream", name="stderr", text=err.getvalue()))
                cell.outputs, cell.execution_count = outputs, count
                preview.cells.append(cell)
                executed.append(cell.metadata.get("source_cell_id", "inspection override/bootstrap"))
                with (OUT / f"{name}_preview.ipynb").open("w", encoding="utf-8") as handle:
                    nbformat.write(preview, handle)
        all_tables[name] = tables
        record["executions"].append({"notebook": name, "start_utc": began,
            "end_utc": datetime.now(timezone.utc).isoformat(), "cells_executed": executed,
            "tables": len(tables), "model_preview_rows": len(tables[0]),
            "settings": {key: namespace[key] for key in ("PROJECTION_WIDTH", "SCORER_WIDTH", "PROJECTION_ACTIVATION",
                "PROJECTION_NORM", "CONTROL_BOTTLENECK", "N_HEADS", "ATTENTION_DROPOUT", "UPDATE_NORM")},
            "train_config": namespace["asdict"](namespace["BASE_CONFIG"])})
    save_json("tables.json", all_tables)
    consumed = set(record["source_notebooks"])
    for module in tuple(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if location:
            path = Path(location).resolve()
            if path.is_relative_to(ROOT / "Code/shared_code") and path.suffix == ".py":
                consumed.add(path.relative_to(ROOT).as_posix())
    record["source_sha256"] = {name: sha(snapshots[name]) for name in sorted(consumed)}
    for name in consumed:
        if (ROOT / name).read_bytes() != snapshots[name]:
            raise RuntimeError(f"Consumed source changed during preview: {name}")
    with zipfile.ZipFile(OUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(consumed): archive.writestr(name, snapshots[name])
    record["status"] = "completed"
except BaseException as exc:
    failure = exc
    record.update(status="failed", error=traceback.format_exc())
finally:
    record["end_utc"] = datetime.now(timezone.utc).isoformat()
    record["artifacts"] = {p.name: {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
                           for p in OUT.iterdir() if p.is_file()}
    save_json("execution.json", record)
print(json.dumps({"status": record["status"], "executions": record["executions"]}, indent=2))
if failure is not None:
    raise failure
