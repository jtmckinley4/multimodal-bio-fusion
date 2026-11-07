"""Read historical CSV summaries without running or repairing their producers.

These checks establish table structure and declared coverage, not historical input,
checkpoint, split-seed, or software identity that the original CSV did not record.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def read_historical_csv(
    path, *, required, key, numeric=(), integers=(), expected_keys=None,
    expected_values=None, bounds=None, missing_numeric=None, report=print,
):
    """Return validated rows, or report why this independent table is unavailable.

    ``key`` and ``expected_keys`` use tuples of column values. Numeric columns are
    converted to finite floats (or integers); other fields retain their saved text.
    ``bounds`` gives inclusive numeric limits, with None for an unbounded endpoint.
    An optional ``missing_numeric(raw_row)`` policy names numeric columns where
    blank/NaN means unavailable for that row. All other values remain strictly
    validated; infinity is never accepted. Unavailable numbers remain NaN, not zero.
    No directory, result, provenance field, or replacement calculation is created.
    """
    path = Path(path)
    required, key = tuple(required), tuple(key)
    numeric, integers = set(numeric), set(integers)
    bounds = bounds or {}
    expected_values = expected_values or {}
    checked = set(key) | numeric | integers | set(bounds) | set(expected_values)
    if not checked.issubset(required):
        raise ValueError("Every checked column must be declared in required.")
    if not set(bounds).issubset(numeric | integers):
        raise ValueError("Bounds require numeric or integer columns.")

    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            if headers is None or len(headers) != len(set(headers)):
                raise ValueError("missing or duplicate column headers")
            missing = set(required) - set(headers)
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError("empty summary")

        seen = set()
        for number, row in enumerate(rows, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"row {number} has an unexpected number of fields")
            identity = tuple(row[column] for column in key)
            if any(not value.strip() for value in identity):
                raise ValueError(f"row {number} has an empty identity field")
            if identity in seen:
                raise ValueError(f"duplicate entry: {identity}")
            seen.add(identity)
            missing = set(missing_numeric(row)) if missing_numeric is not None else set()
            if not missing.issubset(numeric | integers):
                raise ValueError("Missing-number policy must name only declared numeric columns.")
            for column in numeric | integers:
                if column in missing and row[column].strip().lower() in ("", "nan"):
                    row[column] = math.nan
                    continue
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(f"row {number}, {column}: nonfinite number")
                if column in integers:
                    if not value.is_integer():
                        raise ValueError(f"row {number}, {column}: expected an integer")
                    value = int(value)
                row[column] = value
            for column, (lower, upper) in bounds.items():
                if math.isnan(row[column]):  # Only the explicit row policy can admit this.
                    continue
                if ((lower is not None and row[column] < lower)
                        or (upper is not None and row[column] > upper)):
                    raise ValueError(f"row {number}, {column}: outside {lower} to {upper}")
            for column, expected in expected_values.items():
                if row[column] != expected:
                    raise ValueError(f"row {number}, {column}: expected {expected!r}")

        if expected_keys is not None:
            expected = {tuple(entry) for entry in expected_keys}
            missing, unexpected = expected - seen, seen - expected
            if missing or unexpected:
                raise ValueError(
                    f"entry coverage mismatch: missing {sorted(missing)}, "
                    f"unexpected {sorted(unexpected)}"
                )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        report(f"Saved table unavailable: {path}: {error}. Dependent display skipped.")
        return None

    report(
        f"Historical saved summary: {path.name}. Structure and entry coverage checked; "
        "historical input, actual seed IDs and producer identity are not fully recorded."
    )
    return rows


def gate_missing_diagnostics(row):
    """Name diagnostics left uncomputed by a known, explained transcript-gate failure.

    Called on raw CSV text. Row/gene counts remain required even for blocked input.
    A missing value never changes the saved verdict or makes a dataset eligible.
    """
    if not row.get("caveats", "").strip():
        return ()
    if row.get("verdict") == "invalid input: full-transcript comparison blocked":
        return (
            "rows with a coding region", "rows with a 5' region", "rows with a 3' region",
            "coding sequence in frame", "median transcript nt", "longest transcript nt",
            "median cds nt", "repeated sequences",
        )
    if row.get("verdict") == "no usable coding regions: not a full-transcript test":
        # These fractions have an empty denominator; the zero usable-row fraction
        # and sequence-level summaries were still computed and must remain numeric.
        try:
            if float(row.get("rows with a coding region", "")) == 0:
                return ("rows with a 5' region", "rows with a 3' region",
                        "coding sequence in frame")
        except ValueError:
            pass  # The ordinary numeric checks will report a malformed fraction.
    return ()
