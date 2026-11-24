"""
Compare an existing local upload folder with the files an experiment needs now.

prepare.py calls update_package for both an update preview and --apply. The
folder's source-snapshot.json lists the files the preparation tool owns and the
hash of each file's contents. Only those files and the proposed additions
participate in an update. Results and embedding caches stay in place.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
from tempfile import NamedTemporaryFile


SCHEMAS = {
    "encoder_swaps": ("biological-fusion.arcc-package", 2),
    "grouped_stability": ("biological-fusion.grouped-stability-package", 1),
    "fusion_studies": ("fusion-studies.package.v1", None),
    "partner_tasks": ("partner-tasks.package.v1", None),
}
SNAPSHOT = "source-snapshot.json"
HISTORY = ".arcc-history"


# -------- Preview or update an existing upload folder --------


def update_package(recipe, plan, destination, source_roots, *, apply=False):
    """
    List files to add, replace or remove. Write the changes when apply is true.
    """
    root = _no_links(Path(destination).expanduser().absolute()).resolve(strict=True)
    roots = {
        name: _no_links(Path(path).expanduser().absolute()).resolve(strict=True)
        for name, path in source_roots.items()
    }
    repo = roots["maintained_repository"]
    if (
        root.parent != repo.parent
        or not root.is_dir()
        or any(root.is_relative_to(path) or path.is_relative_to(root) for path in roots.values())
    ):
        raise ValueError("Update requires a separate prepared sibling folder outside source roots.")

    snapshot = _no_links(root / SNAPSHOT)
    history = _no_links(root / HISTORY)
    new_files = _recorded_files(plan, recipe)
    snapshot_bytes = snapshot.read_bytes()
    previous_plan = json.loads(snapshot_bytes)
    old_files = _recorded_files(previous_plan, recipe)
    if previous_plan.get("experiment") != plan.get("experiment"):
        raise ValueError("The destination belongs to a different experiment profile.")
    previous_cache = _cache_path(previous_plan)
    _check_cache_paths(root, previous_cache, _cache_path(plan))

    with _update_lock(history) if apply else nullcontext():
        if snapshot.read_bytes() != snapshot_bytes:
            raise ValueError("The package changed during planning. Preview the update again.")
        previous_hash = hashlib.sha256(snapshot_bytes).hexdigest()
        pending = _no_links(history / "pending-snapshot.json")
        if pending.exists():
            _recover_pending_update(
                root, recipe, old_files, pending, snapshot_bytes, previous_hash, apply
            )

        # Verify the existing files and their proposed replacements. A local edit
        # or an unrelated file at a proposed destination must be resolved first.
        names = new_files.keys() | old_files.keys()
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Case-only path renames require a new package.")
        for name in names:
            _verify_file(root / name, old_files.get(name))
        sources = _verified_sources(plan, roots, new_files)

        added = sorted(new_files.keys() - old_files.keys())
        changed = sorted(
            name for name in new_files.keys() & old_files.keys()
            if new_files[name] != old_files[name]
        )
        removed = sorted(old_files.keys() - new_files.keys())
        result = {
            "added": added,
            "changed": changed,
            "removed": removed,
            "unchanged": len(new_files) - len(added) - len(changed),
            "applied": False,
        }
        if not apply or not (added or changed or removed):
            return result

        # Save the previous snapshot and the files that would be lost. The pending
        # snapshot identifies the expected file contents if an update is interrupted.
        _archive_previous_files(
            root, history, previous_hash, snapshot_bytes, old_files, changed + removed
        )
        stamp = datetime.now(timezone.utc).isoformat()
        record = {**plan, "previous_snapshot_sha256": previous_hash}
        if previous_cache is not None:
            record["cache_reference"] = {
                **previous_plan["cache_reference"],
                **plan.get("cache_reference", {}),
                "path": previous_cache,
            }
        if recipe == "fusion_studies":
            record["prepared_utc"] = stamp
        else:
            record.update(captured_utc=stamp, package_status="prepared; not transferred or executed")
        payload = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode()
        _install(pending, lambda temporary: temporary.write_bytes(payload))

        # Recheck each file at the moment of replacement, then verify the entire
        # proposed inventory before making the new snapshot current.
        try:
            for name in added + changed:
                _verify_file(root / name, old_files.get(name))
                _copy(sources[name], root / name, new_files[name])
            for name in removed:
                _verify_file(root / name, old_files[name])
                (root / name).unlink()
            for name in new_files:
                _verify_file(root / name, new_files[name])
                _verify_file(sources[name], new_files[name])
            _verify_file(snapshot, (previous_hash, len(snapshot_bytes)))
            os.replace(pending, snapshot)
        except (OSError, ValueError):
            _restore(root, old_files, new_files)
            pending.unlink()
            raise
        return {**result, "applied": True}


# -------- Check package paths and recorded file contents --------


def _no_links(path):
    # Python 3.11 on Windows lacks Path.is_junction().
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"Linked path in package update: {part}")
    return path


def _relative_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Invalid package path: {name!r}")
    if any(ord(char) < 32 or char in '<>"|?*' for char in name):
        raise ValueError(f"Invalid package path: {name!r}")

    # Manifests travel between Windows and Newton. Use the Windows filename
    # rules on either system, including Python 3.11's older reserved-name check.
    is_reserved = getattr(ntpath, "isreserved", None)
    for part in name.split("/"):
        if part in ("", ".", "..") or part.endswith((" ", ".")):
            raise ValueError(f"Invalid package path: {name!r}")
        reserved = is_reserved(part) if is_reserved else PureWindowsPath(part).is_reserved()
        if reserved:
            raise ValueError(f"Invalid package path: {name!r}")
    return PurePosixPath(name)


def _is_managed_path(name):
    # Earlier snapshots may include retired source files. Only locations in the
    # upload layout can be replaced or removed, even if a snapshot is damaged.
    path = _relative_path(name)
    root, *rest = path.parts
    if not rest:
        return (
            name.casefold() != SNAPSHOT and not name.startswith(".")
            and path.suffix in {".py", ".sbatch", ".example", ".md", ".txt", ".tsv", ".json"}
        )
    if root == "Code":
        return (
            (rest[0] == "shared_code" and path.suffix == ".py")
            or (len(rest) == 2 and rest[0] == "stage4" and path.suffix == ".ipynb")
        )
    return (
        (root == "datasets" and path.suffix == ".csv")
        or (root == "tests" and path.suffix == ".py")
        or (root == "historical_results" and path.suffix == ".json")
        or name in {"evidence/selection-source-snapshot.json", "docs/figures/stability/results.json"}
    )


def _recorded_files(manifest, recipe):
    if (
        not isinstance(manifest, dict)
        or recipe not in SCHEMAS
        or (manifest.get("schema"), manifest.get("schema_version")) != SCHEMAS[recipe]
    ):
        raise ValueError(f"Update requires an existing {recipe} package of the supported schema.")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("Missing or empty package inventory.")
    files = {}
    casefolded_names = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Expected a file entry in the package manifest.")
        _relative_path(entry.get("source"))
        name = entry.get("destination")
        sha = entry.get("sha256" if recipe == "fusion_studies" else "initial_copy_sha256")
        size = entry.get("bytes")
        if (
            not _is_managed_path(name)
            or name.casefold() in casefolded_names
            or not isinstance(sha, str)
            or not re.fullmatch(r"[a-f0-9]{64}", sha)
            or (recipe != "fusion_studies" and entry.get("source_sha256") != sha)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(f"Invalid managed file in package manifest: {name!r}")
        files[name] = (sha, size)
        casefolded_names.add(name.casefold())
    return files


def _cache_path(manifest):
    reference = manifest.get("cache_reference", {})
    if not isinstance(reference, dict):
        raise ValueError("Invalid cache reference in the package manifest.")
    path = reference.get("path")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("Invalid cache path in the package manifest.")
    return path


def _file_identity(path):
    _no_links(path)
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Expected a regular package file: {path}")
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest(), size


def _verify_file(path, identity):
    """
    Check the (SHA-256 hash, byte count) pair. None means the path must be unused.
    """
    if _file_identity(path) != identity:
        raise ValueError(f"Package bytes changed or file collision at {path}")


def _verified_sources(plan, roots, new_files):
    sources = {}
    for entry in plan["files"]:
        origin = entry.get("source_origin", "maintained_repository")
        if not isinstance(origin, str) or origin not in roots:
            raise ValueError(f"Unknown source origin in package manifest: {origin!r}")
        name = entry["destination"]
        source = roots[origin] / _relative_path(entry["source"])
        _verify_file(source, new_files[name])
        sources[name] = source
    return sources


def _check_cache_paths(root, previous_cache, requested_cache):
    for cache in (previous_cache, requested_cache):
        if cache is None:
            continue
        cache_dir = Path(cache).resolve()
        if root.is_relative_to(cache_dir) or cache_dir.is_relative_to(root):
            raise ValueError("The prepared folder overlaps a declared cache root.")
    if requested_cache is not None and (
        previous_cache is None or Path(requested_cache).resolve() != Path(previous_cache).resolve()
    ):
        raise ValueError(
            "A source update retains its recorded cache reference. Omit --cache-root "
            "or specify the original path. Select external caches when launching on Newton."
        )


# -------- Replace files and keep the previous versions --------


def _install(path, writer):
    """
    Write to a temporary file and replace the destination only after writing succeeds.
    """
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".arcc-tmp-", delete=False) as temp:
        scratch = Path(temp.name)
    try:
        writer(scratch)
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def _copy(source, target, identity):
    _no_links(source)

    def write(scratch):
        shutil.copyfile(source, scratch)
        _verify_file(scratch, identity)

    _install(target, write)


def _archive_previous_files(root, history, previous_hash, snapshot_bytes, old_files, affected):
    archived = _no_links(history / "snapshots" / (previous_hash + ".json"))
    if not archived.exists():
        _install(archived, lambda temporary: temporary.write_bytes(snapshot_bytes))
    _verify_file(archived, (previous_hash, len(snapshot_bytes)))
    for name in affected:
        identity = old_files[name]
        backup = _no_links(history / "objects" / identity[0])
        if not backup.exists():
            _copy(root / name, backup, identity)
        _verify_file(backup, identity)


@contextmanager
def _update_lock(history):
    path = _no_links(history / "update.lock")
    history.mkdir(exist_ok=True)
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Another process is updating this workspace.") from error
        yield


# -------- Recover an interrupted update --------


def _recover_pending_update(root, recipe, old_files, pending, snapshot_bytes, previous_hash, apply):
    interrupted_bytes = pending.read_bytes()
    if interrupted_bytes != snapshot_bytes:
        if not apply:
            raise ValueError("Interrupted package update. Re-run with --apply to restore and continue.")
        interrupted = json.loads(interrupted_bytes)
        if interrupted.get("previous_snapshot_sha256") != previous_hash:
            raise ValueError("Interrupted update belongs to another snapshot.")
        _restore(root, old_files, _recorded_files(interrupted, recipe))
    if apply:
        pending.unlink()


def _restore(root, old, interrupted):
    """
    Restore the previous contents when each affected file matches a known version.
    """
    changed = [
        name for name in old.keys() | interrupted.keys()
        if old.get(name) != interrupted.get(name)
    ]
    for name in changed:
        actual = _file_identity(root / name)
        if actual not in (old.get(name), interrupted.get(name)):
            raise ValueError(f"Unrecognized edits after interrupted update: {name}")
        if name in old:
            _verify_file(root / HISTORY / "objects" / old[name][0], old[name])
    for name in changed:
        if _file_identity(root / name) == old.get(name):
            continue
        if name in old:
            _copy(root / HISTORY / "objects" / old[name][0], root / name, old[name])
        else:
            (root / name).unlink()
