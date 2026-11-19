"""Resolve encoder-swap evidence locations without creating or importing files.

Scoring evidence and validation-selection evidence have different roles. Newton
scores stay in the returned batch; the original snapshot pins historical choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath


BENCHMARKS = (
    "cov_vac_degradation", "fungal_expression", "ecoli_proteins", "mrna_stability",
)
NEWTON_SNAPSHOT_SHA256 = "6bc161b8ea78eda8bf8c316ce27fca60bb1b0975b1bf01d8ec241aa82863c96f"


@dataclass(frozen=True)
class NewtonSelectionSnapshot:
    """Resolve this completed batch's fixed selection files, not a current result pool."""

    path: Path
    code_dir: Path
    expected_sha256: str

    def file_hashes(self):
        """Verify the original manifest and every listed canonical selection file."""
        payload = self.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.expected_sha256:
            raise ValueError("Newton selection snapshot differs from the original recorded bytes.")
        snapshot = json.loads(payload)
        if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("files"), list):
            raise ValueError("Newton selection requires the original version-1 source snapshot.")
        code_dir = self.code_dir.resolve()
        files, names = {}, set()
        for entry in snapshot["files"]:
            if not isinstance(entry, dict):
                raise ValueError("Newton source snapshot contains a malformed file entry.")
            destination = entry.get("destination")
            if not isinstance(destination, str):
                raise ValueError("Newton source snapshot contains an invalid destination.")
            if not destination.startswith("historical_results/"):
                continue
            relative = PurePosixPath(destination)
            source = destination.removeprefix("historical_results/")
            if (relative.is_absolute() or ".." in relative.parts or "\\" in destination
                    or ":" in destination or relative.as_posix() != destination
                    or not source.startswith(("Code/stage3/results/fungal_expression/",
                                              "Code/stage4/results/phaseA/",
                                              "Code/stage4/results/phaseB/"))
                    or relative.suffix != ".json" or entry.get("source") != source):
                raise ValueError(f"Invalid Newton selection path: {destination}")
            digest = entry.get("initial_copy_sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or set(digest) - set("0123456789abcdef")
                    or entry.get("source_sha256") != digest):
                raise ValueError(f"Invalid Newton selection hash: {destination}")
            if destination.casefold() in names:
                raise ValueError(f"Duplicate Newton selection path: {destination}")
            names.add(destination.casefold())
            path = (code_dir / source.removeprefix("Code/")).resolve()
            if (not path.is_relative_to(code_dir) or path in files
                    or path.relative_to(code_dir).as_posix() != source.removeprefix("Code/")):
                raise ValueError(f"Newton selection path escapes or aliases its canonical root: {destination}")
            if not path.is_file():
                raise ValueError(f"Recorded Newton selection file is missing: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Recorded Newton selection file changed: {path}")
            files[path] = digest
        if not files:
            raise ValueError("The Newton snapshot contains no historical selection files.")
        return files


def historical_originals(code_dir):
    """Retain the existing fungal reference route and the other Phase A roots."""
    code_dir = Path(code_dir)
    folders = {key: (code_dir / "stage4/results/phaseA" / key,) for key in BENCHMARKS}
    folders["fungal_expression"] = (
        code_dir / "stage3/results/fungal_expression", *folders["fungal_expression"],
    )
    return folders


@dataclass(frozen=True)
class SwapResultSet:
    name: str
    root: Path
    originals: dict
    swaps: Path
    selection_originals: dict
    selection_swaps: Path
    saved_report_dirs: tuple
    analysis_dir: Path
    fixed_historical_selection: bool
    imported: bool
    selection_snapshot: NewtonSelectionSnapshot | None = None

    def require_writable_fits(self):
        """Imported fits are immutable; local reports have a separate destination."""
        if self.imported:
            raise ValueError(
                "Imported Newton results are read-only. Use an explicit execution "
                "workspace for new fits; do not fill gaps in the imported cohort."
            )


def resolve_swap_results(code_dir, name="historical", *, workspace=None):
    """Select historical evidence or a returned workspace with its original tree.

    ``workspace`` is the returned package root, containing ``outputs/`` and its
    frozen snapshot under ``provenance/``. Selection records resolve to canonical
    Code paths only through that snapshot's allowlist and hashes. Existence is not inferred here:
    missing evidence must remain visible to the readers. This function does no IO
    beyond path resolution and never creates directories or substitutes results.
    """
    code_dir = Path(code_dir).resolve()
    results = code_dir / "stage4/results"
    if name == "historical":
        if workspace is not None:
            raise ValueError("Historical results use their maintained paths, not a workspace override.")
        originals = historical_originals(code_dir)
        swaps = results / "phaseB"
        reports = swaps / "analyses/paired-report-v1"
        return SwapResultSet(name, results, originals, swaps, originals, swaps,
                             (reports,), reports, False, False)

    if name != "newton_h100_seeds012":
        raise ValueError(f"Unknown result set: {name!r}")
    root = (Path(workspace) if workspace is not None else results / name).resolve()
    protected = (results / "phaseA", results / "phaseB",
                 code_dir / "stage3/results/fungal_expression")
    for old in protected:
        old = old.resolve()
        if root == old or root in old.parents or old in root.parents:
            raise ValueError(f"Imported workspace overlaps historical evidence: {old}")
    originals = {key: (root / "outputs/phaseA" / key,) for key in BENCHMARKS}
    swaps = root / "outputs/phaseB"
    selection = code_dir
    producer_reports = swaps / "analyses/paired-report-v1"
    local_reports = root / "local_analyses/paired-report-v1"
    return SwapResultSet(
        name, root, originals, swaps, historical_originals(selection),
        selection / "stage4/results/phaseB", (local_reports, producer_reports),
        local_reports, True, True,
        NewtonSelectionSnapshot(root / "provenance/source-snapshot.json", code_dir,
                                NEWTON_SNAPSHOT_SHA256),
    )
