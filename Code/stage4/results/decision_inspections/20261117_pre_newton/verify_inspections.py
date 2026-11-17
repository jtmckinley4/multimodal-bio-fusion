"""Verify preserved files and recorded inspection notebooks, without analyses."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "Code/shared_code").is_dir())
EVIDENCE = Path(__file__).resolve().parent
BASELINE = ROOT / ".data-cache/decision-inspections-preservation.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
changed_files = [name for name, expected in baseline["files"].items()
                 if not (ROOT / name).is_file() or sha(ROOT / name) != expected]
changed_caches = []
for name, expected in baseline["caches"].items():
    path = ROOT / name
    if not path.is_file():
        changed_caches.append(name)
        continue
    stat = path.stat()
    if [stat.st_size, stat.st_mtime_ns] != expected:
        changed_caches.append(name)

notebooks = []
for path in sorted(EVIDENCE.rglob("*.ipynb")):
    if "sources" in path.relative_to(EVIDENCE).parts:
        continue
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    errors = [dict(cell=cell.id, error=output.get("ename"), message=output.get("evalue"))
              for cell in book.cells if cell.cell_type == "code"
              for output in cell.outputs if output.output_type == "error"]
    notebooks.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                      "executed_cells": sum(cell.cell_type == "code" and
                                            cell.execution_count is not None for cell in book.cells),
                      "recorded_errors": errors})

reports = sorted((ROOT / "Code/stage4/results/anchor_correction").glob(
    "*/analyses/decision-support-*.json"))
reports += sorted((ROOT / "Code/stage4/results/region_aware_v1").glob("*/input-audit-v1.json"))
record = {
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "File preservation and saved notebook schema; no research recomputation",
    "baseline_sha256": sha(BASELINE), "verification_source_sha256": sha(Path(__file__)),
    "protected_content_hashes": len(baseline["files"]), "changed_files": changed_files,
    "cache_size_and_mtime_checks": len(baseline["caches"]), "changed_caches": changed_caches,
    "cache_limit": "Existing cache payloads were not rehashed; size and modification time were compared.",
    "notebooks": notebooks,
    "reports": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)} for path in reports],
}
with (EVIDENCE / "verification.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps({key: record[key] for key in
                  ("protected_content_hashes", "changed_files", "cache_size_and_mtime_checks", "changed_caches")},
                 indent=2))
print(f"Validated {len(notebooks)} saved notebook copies; inventoried {len(reports)} reports.")
if changed_files or changed_caches:
    raise SystemExit("Preservation differences need review.")
