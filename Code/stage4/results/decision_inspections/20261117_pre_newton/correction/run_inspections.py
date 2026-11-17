"""Run only the existing A2 input/cache inspections; preserve the source notebook."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import nbformat
from nbclient import NotebookClient


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Code/shared_code").is_dir())
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "Code/stage4/anchor_correction.ipynb"
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(ROOT / "Code"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(benchmark, operation):
    stem = f"{benchmark}-{operation}"
    record_path = OUTPUT / f"{stem}.json"
    if record_path.exists():
        print(f"Preserving existing execution record: {record_path.name}", flush=True)
        return
    record = {
        "operation": operation, "benchmark": benchmark,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "source_notebook": str(SOURCE.relative_to(ROOT)), "source_notebook_sha256": sha(SOURCE),
        "runner_sha256": sha(Path(__file__)), "python": sys.version, "executable": sys.executable,
        "host": platform.node(), "platform": platform.platform(),
        "scope": "Authorized CPU fold/input and bounded-cache inspection; no fitting, inference, extraction or resampling.",
        "source_hashes": {str(p.relative_to(ROOT)): sha(p)
                          for p in sorted((ROOT / "Code/shared_code").rglob("*.py"))},
    }
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    notebook = nbformat.read(SOURCE, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    settings_index = next(i for i, c in enumerate(notebook.cells) if c.id == "settings")
    overrides = f'''# This executed copy enables only the user-authorized inspection.
BENCHMARK = {benchmark!r}
READ_SAVED_RESULTS = True
RUN_INPUT_AUDIT = {operation == 'folds'}
RUN_CACHE_AUDIT = {operation == 'controls'}
RUN_TRAINING = RUN_ANALYSIS = RUN_DIAGNOSTICS = False
DEVICE = "cpu"
def _outside_inspection(*args, **kwargs):
    raise RuntimeError("Training, extraction, model inference and bootstrap are outside this inspection.")
train.run = _outside_inspection
train.seed_averaged_bootstrap = _outside_inspection
token_embeddings.extract = _outside_inspection
swap_reporting.calculate = _outside_inspection
print("Authorized inspection:", BENCHMARK, {operation!r}, "on CPU")
'''
    notebook.cells.insert(settings_index + 1, nbformat.v4.new_code_cell(overrides, id="authorized-inspection"))
    print(f"Starting {stem}", flush=True)
    try:
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       allow_errors=False, resources={"metadata": {"path": str(ROOT)}}).execute()
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(f"{stem} failed: {type(error).__name__}: {str(error)[-1300:]}", flush=True)
    finally:
        executed = OUTPUT / f"{stem}.ipynb"
        nbformat.write(notebook, executed)
        record.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                      executed_notebook=str(executed.relative_to(ROOT)),
                      executed_notebook_sha256=sha(executed))
        assert sha(SOURCE) == record["source_notebook_sha256"], "Maintained source changed during inspection."
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{stem}: {record['status']}", flush=True)


if __name__ == "__main__":
    # Stability is attempted once at its input boundary. A missing grouped split
    # is recorded; it never falls back to the overlapping supplied partitions.
    for benchmark in ("cov_vac_degradation", "fungal_expression", "mrna_stability"):
        inspect(benchmark, "folds")
        if benchmark != "mrna_stability" or (ROOT / "Code/stage4/results/grouped_stability_v1/split.json").is_file():
            inspect(benchmark, "controls")
