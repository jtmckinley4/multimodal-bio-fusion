"""Exact-key reuse for mRFP CSVs, without certifying unidentified legacy rows.

The notebook owns calculation and publication. These pure checks keep that path
testable without executing notebook cells or fitting the scientific models.
"""

import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import re


IDENTITY_COLUMN = "calculation_sha256"


@dataclass
class SavedTable:
    """Readable historical rows and reasons for withholding other rows.

    A syntactically valid recorded identity is not a comparison with today's
    inputs. Reading this report never certifies historical calculation settings.
    """

    rows: dict
    issues: list


def read_saved(path, *, keys, numeric):
    """Read a CSV without fitting, writing, or requiring its original inputs.

    Legacy rows are usable for historical display. Invalid rows and every copy
    of a duplicated key are withheld; a bad file does not block other tables.
    """
    rows, issues, seen = {}, [], set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = set(keys) | set(numeric)
            if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
                return SavedTable({}, [f"{path.name}: duplicate column headers; table withheld"])
            if not required.issubset(reader.fieldnames or ()):
                return SavedTable({}, [f"{path.name}: missing required columns {sorted(required)}"])
            for line, raw in enumerate(reader, start=2):
                row = dict(raw)
                try:
                    if any(not isinstance(row.get(name), str) or not row[name] for name in keys):
                        raise ValueError("missing key")
                    if "seed" in keys:
                        if re.fullmatch(r"0|[1-9][0-9]*", row["seed"]) is None:
                            raise ValueError("seed must be a nonnegative integer")
                        row["seed"] = int(row["seed"])
                    key = tuple(row[name] for name in keys)
                except ValueError as error:
                    issues.append(f"{path.name}:{line}: {error}; row withheld")
                    continue
                # Count keys even when another field is invalid: never choose a
                # seemingly valid copy of an ambiguous repeated observation.
                if key in seen:
                    rows.pop(key, None)
                    issues.append(f"{path.name}:{line}: duplicate {key}; all copies withheld")
                    continue
                seen.add(key)
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("row field count differs from header")
                    for name in numeric:
                        if name != "seed":
                            row[name] = float(row[name])
                        if not math.isfinite(row[name]):
                            raise ValueError(f"nonfinite {name}")
                    identity = row.get(IDENTITY_COLUMN)
                    if identity not in (None, "") and re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                        raise ValueError("malformed calculation identity")
                except (ValueError, TypeError, OverflowError) as error:
                    issues.append(f"{path.name}:{line}: {error}; {key} withheld")
                    continue
                rows[key] = row
    except (OSError, UnicodeError, csv.Error) as error:
        return SavedTable({}, [f"{path.name}: unavailable ({error})"])
    return SavedTable(rows, issues)


def complete_groups(table, groups):
    """Select only complete requested assignment groups, with explicit coverage.

    Group labels describe models or control/model pairs. Extra saved keys remain
    historical evidence but cannot substitute for a missing requested seed.
    """
    selected, coverage = [], []
    for label, requested in groups.items():
        requested = tuple(requested)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError(f"{label}: expected keys must be nonempty and distinct")
        missing = [key for key in requested if key not in table.rows]
        present = [table.rows[key] for key in requested if key in table.rows]
        unknown = sum(not row.get(IDENTITY_COLUMN) for row in present)
        coverage.append({
            "comparison": label,
            "available assignments": len(present),
            "expected assignments": len(requested),
            "missing keys": missing,
            "status": "complete historical rows" if not missing else "incomplete; summary skipped",
            "identity": (
                "unknown for all saved rows" if unknown == len(present) and present
                else f"unknown for {unknown} saved rows" if unknown
                else "recorded; not revalidated" if present
                else "no usable rows"
            ),
        })
        if not missing:
            selected.extend(present)
    return selected, coverage


def fingerprint(settings, *, arrays=(), functions=(), packages=()):
    """Identify declared settings, ordered numeric inputs and calculation source.

    Arrays follow NumPy's shape/dtype/tobytes interface; no numerical calculation
    or model import occurs here. Source lookup failures remain errors rather than
    silently omitting a requested part of the identity.
    """
    digest = hashlib.sha256()
    description = {
        "settings": settings,
        "functions": [inspect.getsource(function) for function in functions],
        "packages": {name: version(name) for name in packages},
    }
    digest.update(json.dumps(description, sort_keys=True, allow_nan=False).encode("utf-8"))
    for array in arrays:
        if array.dtype.hasobject:
            raise ValueError("Cache identity requires numeric arrays, not object storage")
        digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def checked_rows(rows, *, keys, expected, numeric, source):
    """Index distinct saved keys; refuse unknown or conflicting calculation identity.

    An existing table must identify every row, including rows outside the current
    request. Current requested keys must match their expected identity. Other
    identified rows may remain stored but are not returned as current results by
    the notebook. A missing key is a missing calculation, not a matching row count.
    """
    indexed = {}
    for row in rows:
        identity = row.get(IDENTITY_COLUMN)
        if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError(
                f"{source}: legacy or unidentified cache; settings cannot be verified. "
                "Preserve this CSV. Historical inspection remains available with "
                "pd.read_csv(path); use a separate destination for an authorized new run."
            )
        if any(key not in row or (key != "seed" and
                                 (not isinstance(row[key], str) or not row[key]))
               for key in keys):
            raise ValueError(f"{source}: missing cache key {keys}")
        key = tuple(row[name] for name in keys)
        if key in indexed:
            raise ValueError(f"{source}: duplicate cache key {key}; preserve and inspect the CSV")
        if any(name not in row or isinstance(row[name], (bool, str))
               or not isinstance(row[name], (int, float)) or not math.isfinite(row[name])
               for name in numeric):
            raise ValueError(f"{source}: missing or nonfinite numeric fields for {key}")
        if "seed" in keys and (type(row["seed"]) is not int or row["seed"] < 0):
            raise ValueError(f"{source}: invalid seed for {key}")
        if key in expected and identity != expected[key]:
            raise ValueError(
                f"{source}: input/settings identity differs for {key}; "
                "preserve this CSV and use a separate destination for a changed experiment"
            )
        indexed[key] = dict(row)
    return indexed


def upsert(rows, additions, *, keys, expected, numeric, source):
    """Replace each exact key once, retaining other verified rows without duplicates."""
    existing = checked_rows(rows, keys=keys, expected=expected, numeric=numeric, source=source)
    incoming = checked_rows(additions, keys=keys, expected=expected, numeric=numeric, source=source)
    if set(incoming) - set(expected):
        raise ValueError(f"{source}: new rows include keys outside the current request")
    existing.update(incoming)
    return list(existing.values())
