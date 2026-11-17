"""One-off, bounded A6 input audit; executes unchanged source cells in saved copies."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import nbformat
from nbclient import NotebookClient


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "Code/shared_code").is_dir())
EVIDENCE = Path(__file__).resolve().parent
SOURCE = ROOT / "Code/stage4/region_aware_fusion.ipynb"
DATASET = sys.argv[1]
if DATASET not in ("rnahl-human", "translation-efficiency-human"):
    raise SystemExit("Choose one authorized human dataset.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


destination = EVIDENCE / DATASET
destination.mkdir(parents=True, exist_ok=False)
manifest_path = destination / "execution.json"
notebook_path = destination / "executed-input-audit.ipynb"
audit_path = ROOT / "Code/stage4/results/region_aware_v1" / DATASET / "input-audit-v1.json"
preparation_path = audit_path.with_name("preparation.json")
sources = [SOURCE, Path(__file__), *[ROOT / path for path in (
    "Code/shared_code/fusion/region_experiment.py", "Code/shared_code/fusion/region_inputs.py",
    "Code/shared_code/fusion/region_training.py", "Code/shared_code/fusion/run_identity.py",
    "Code/shared_code/mrnabench.py", "Code/shared_code/encoders.py",
)]]
source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
for path in sources:
    target = destination / "sources" / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())

book = nbformat.read(SOURCE, as_version=4)
for cell in book.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
stop = next(i for i, cell in enumerate(book.cells) if cell.get("id") == "region-audit-mask-display")
for cell in book.cells[stop + 1:]:
    if cell.cell_type == "code":
        cell.metadata.setdefault("tags", []).append("inspection-not-executed")
settings_index = next(i for i, cell in enumerate(book.cells) if cell.get("id") == "region-05")
override = f'''# Authorized inspection-only overrides; source notebook cells are otherwise unchanged.
DATASET_KEY = {DATASET!r}
READ_SAVED_RESULTS = True
RUN_INPUT_AUDIT = True
RUN_PREPARATION = RUN_EXTRACTION = RUN_TRAINING = RUN_DIAGNOSTICS = RUN_ANALYSIS = False
DEVICE = torch.device("cpu")
torch.set_num_threads(1)
def outside_inspection(*args, **kwargs):
    raise RuntimeError("This execution permits input inspection only.")
region.open_token_caches = outside_inspection
study.publish_preparation = outside_inspection
study.run_fit = study.collect_diagnostics = study.build_model = outside_inspection
train.run = outside_inspection
%matplotlib inline
'''
book.cells.insert(settings_index + 1, nbformat.v4.new_code_cell(override, id="authorized-input-inspection"))
book.metadata["inspection"] = {
    "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
    "scope": "Unchanged setup/input-audit/table/plot cells through region-audit-mask-display only",
    "dataset": DATASET, "source_outputs_updated": False,
}
record = {
    "operation": "A6 authorized CPU-only input inspection", "dataset": DATASET,
    "status": "running", "started_utc": utc(), "source_sha256": source_hashes,
    "source_notebook_unchanged": None, "notebook": str(notebook_path.relative_to(ROOT)),
    "settings": {"READ_SAVED_RESULTS": True, "RUN_INPUT_AUDIT": True,
                 "RUN_PREPARATION": False, "RUN_EXTRACTION": False, "RUN_TRAINING": False,
                 "RUN_DIAGNOSTICS": False, "RUN_ANALYSIS": False, "device": "cpu"},
    "environment": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "hostname": platform.node(),
                    "packages": {name: version(name) for name in
                                 ("numpy", "pandas", "torch", "pyarrow", "nbformat", "nbclient", "ipykernel")},
                    "threads": {key: os.environ.get(key) for key in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
    "prior_audit_sha256": sha(audit_path) if audit_path.exists() else None,
    "prior_preparation_sha256": sha(preparation_path) if preparation_path.exists() else None,
}
manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
error = None
try:
    NotebookClient(book, timeout=900, kernel_name="python3", record_timing=True,
                   skip_cells_with_tag="inspection-not-executed",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    record["status"] = "completed"
except BaseException as caught:
    error = caught
    record["status"] = "failed"
    record["error"] = traceback.format_exc()
finally:
    nbformat.write(book, notebook_path)
    record["ended_utc"] = utc()
    record["source_notebook_unchanged"] = sha(SOURCE) == source_hashes[str(SOURCE.relative_to(ROOT))]
    record["changed_sources"] = [name for name, digest in source_hashes.items() if sha(ROOT / name) != digest]
    record["notebook_sha256"] = sha(notebook_path)
    record["audit_sha256"] = sha(audit_path) if audit_path.exists() else None
    record["preparation_sha256"] = sha(preparation_path) if preparation_path.exists() else None
    record["executed_source_cells"] = [c.get("id") for c in book.cells
                                       if c.cell_type == "code" and c.execution_count is not None]
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in
                      ("dataset", "status", "started_utc", "ended_utc", "changed_sources", "audit_sha256",
                       "source_notebook_unchanged", "preparation_sha256")}, indent=2))
if error is not None:
    raise error
