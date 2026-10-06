"""Build datasets/GTEx_pilot.csv, the input of Stage1_gtex.ipynb.

Run from the repository root: python Code/build_gtex_pilot.py
Input-cache and output paths are anchored to this script, not the working directory.

1. Download IsoFormer's GTEx table (about 645 MB) unless it is already in DOWNLOAD_DIR:
   huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression,
   file transcript_expression/GTEx_final.csv.
2. Keep protein-coding transcripts outside the mitochondrial genome whose coding
   sequence is in frame (length a multiple of three, starting with ATG).
3. Sample N_PER_SPLIT transcripts from IsoFormer's train and test splits with SEED.
4. Add each transcript's GRCh38 DNA: WINDOW // 2 bases upstream of the transcription
   start site, then the start site and the bases downstream, read on the transcript's
   strand from the Ensembl REST service, so the transcript's first base is at index
   WINDOW // 2 of its window.

The output keeps transcript and gene identifiers, the split, coordinates, the DNA
window, the transcript and its parts, the protein, and the 30 tissue labels.
"""

import json
import math
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from shared_code.datasets import GTEX_TISSUES

SOURCE_URL = (
    "https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression/"
    "resolve/main/transcript_expression/GTEx_final.csv"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = REPO_ROOT / ".data-cache"
OUTPUT = REPO_ROOT / "datasets" / "GTEx_pilot.csv"
WINDOW = 6000
SEED = 42
N_PER_SPLIT = {"train": 1000, "test": 1000}
ENSEMBL_URL = "https://rest.ensembl.org/sequence/region/human"


@contextmanager
def _temporary_sibling(destination):
    """Own a temporary sibling, preserving any original failure during cleanup."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    failure = None
    try:
        yield temporary
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if failure is None:
                raise
            if hasattr(failure, "add_note"):
                failure.add_note(f"Could not remove temporary file {temporary}: {cleanup_error}")


def _read_source_table(path):
    """Read the required source fields and the original ordered tissue block."""
    table = pd.read_csv(path, low_memory=False)
    required = {"transcript_id_gtex", "gene_id_gtex", "split", "chr", "strand", "start", "end",
                "RNA", "5UTR", "CDS", "3UTR", "Protein"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"GTEx source table {path} is missing required columns: {missing}")
    if tuple(table.columns[9:39]) != GTEX_TISSUES:
        raise ValueError(f"GTEx source table {path} has unexpected tissue names or order in columns 9:39")
    if table.empty:
        raise ValueError(f"GTEx source table {path} contains no rows")
    return table


def _close_http_error(error):
    """Close a failed response while keeping its request failure as the cause."""
    try:
        error.close()
    except OSError as cleanup_error:
        if hasattr(error, "add_note"):
            error.add_note(f"Could not close failed HTTP response: {cleanup_error}")


def source_table():
    """Return a schema-checked source table; publish a first download after validation.

    Content-Length, when supplied, checks the received byte count. This transport
    check and the schema do not establish biological completeness or old cache provenance.
    An invalid existing cache raises rather than triggering an automatic redownload.
    """
    path = Path(DOWNLOAD_DIR) / "GTEx_final.csv"
    if path.exists():
        return _read_source_table(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL}")
    with _temporary_sibling(path) as temporary:
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
                declared = response.headers.get("Content-Length")
                try:
                    expected = int(declared) if declared is not None else None
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid source Content-Length: {declared!r}") from error
                if expected is not None and expected < 0:
                    raise ValueError(f"Invalid source Content-Length: {declared!r}")
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
        except urllib.error.HTTPError as error:
            _close_http_error(error)
            raise
        if expected is not None and received != expected:
            raise ValueError(f"Incomplete GTEx source download: received {received} bytes, expected {expected}")
        table = _read_source_table(temporary)
        os.replace(temporary, path)
    return table


def _validated_sequences(payload, regions):
    """Validate the whole response before exposing any sequence from this batch."""
    if not isinstance(payload, list):
        raise ValueError("Ensembl response must be a list of query/seq entries")
    expected = set(regions)
    sequences = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Ensembl response entries must contain query and seq fields")
        query, sequence = item.get("query"), item.get("seq")
        if not isinstance(query, str) or query not in expected:
            raise ValueError(f"Unexpected Ensembl response query: {query!r}")
        if query in sequences:
            raise ValueError(f"Duplicate Ensembl response query: {query!r}")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"Invalid Ensembl sequence for query: {query!r}")
        sequences[query] = sequence.upper()
    missing = expected - sequences.keys()
    if missing:
        raise ValueError(f"Ensembl response is missing requested regions: {sorted(missing)}")
    return sequences


def _fetch_region_batch(regions):
    """Fetch one batch, retrying only known transient failures up to five attempts."""
    request = urllib.request.Request(
        ENSEMBL_URL, data=json.dumps({"regions": regions}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError):
                retryable = error.code in {429, 500, 502, 503, 504}
                _close_http_error(error)
            else:
                reason = error.reason if isinstance(error, urllib.error.URLError) else error
                retryable = isinstance(reason, (TimeoutError, ConnectionError)) or (
                    isinstance(reason, socket.gaierror) and reason.errno == socket.EAI_AGAIN
                )
            if not retryable:
                raise
            if attempt == 4:
                raise RuntimeError(
                    f"Ensembl request failed after five attempts (batch size {len(regions)}, "
                    f"first region {regions[0]!r})"
                ) from error
            delay = 5 * (attempt + 1)
            retry_after = (error.headers.get("Retry-After")
                           if isinstance(error, urllib.error.HTTPError) and error.headers else None)
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    raise RuntimeError(f"Invalid Ensembl Retry-After: {retry_after!r}") from error
                if not math.isfinite(delay) or delay < 0:
                    raise RuntimeError(f"Invalid Ensembl Retry-After: {retry_after!r}") from error
                # Do not wait without a bound or retry earlier than the service asks.
                if delay > 120:
                    raise RuntimeError(f"Ensembl Retry-After {delay:g}s exceeds the 120s retry cap") from error
            print(f"Retrying Ensembl request in {delay:g}s after {error}", file=sys.stderr)
            time.sleep(delay)
        else:
            return _validated_sequences(payload, regions)


def fetch_regions(regions):
    """Return validated sequences, requesting each distinct region once in batches of 50.

    Repeated transcript windows share a request; callers retain their original row
    order when looking up the returned sequences. Every response batch is validated
    before merging, and a successful batch retains the original 0.3 second pacing.
    """
    regions = list(dict.fromkeys(regions))
    sequences = {}
    for i in range(0, len(regions), 50):
        sequences.update(_fetch_region_batch(regions[i:i + 50]))
        time.sleep(0.3)
    return sequences


def transcript_regions(rows, window=WINDOW):
    """Return zero-based TSS values and strand-oriented Ensembl region strings.

    ``rows`` supplies one-based inclusive ``start``/``end`` coordinates, ``chr``,
    and ``strand``. The TSS is start - 1 on '+' and end - 1 otherwise, matching
    the original builder. The returned Series keeps the input index; region
    strings keep row order and use one-based inclusive endpoints.

    A positive integer ``window`` requests that many bases. Ensembl's -1 strand
    reads the interval in reverse-complement orientation, so the TSS is at
    zero-based index window // 2 on either strand, including odd window sizes.
    This calculation does not filter or alter rows, validate coordinates, or
    clamp chromosome boundaries: nonpositive requested endpoints are preserved.
    """
    plus = rows["strand"] == "+"
    tss = (rows["start"] - 1).where(plus, rows["end"] - 1)
    lo = (tss - window // 2).where(plus, tss - (window - window // 2) + 1)  # zero-based, inclusive
    hi = (tss + (window - window // 2)).where(plus, tss + window // 2 + 1)  # zero-based, exclusive
    regions = [
        f"{c.replace('chr', '')}:{a + 1}..{b}:{1 if s == '+' else -1}"
        for c, a, b, s in zip(rows["chr"], lo, hi, rows["strand"])
    ]
    return tss, regions


def _write_pilot(pilot, destination):
    """Publish the selected pilot CSV after a complete write and a schema/row-count check."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_sibling(destination) as temporary:
        pilot.to_csv(temporary, index=False)
        written = pd.read_csv(temporary, low_memory=False)
        if list(written.columns) != list(pilot.columns) or len(written) != len(pilot):
            raise ValueError("Written GTEx pilot CSV has unexpected columns or row count")
        os.replace(temporary, destination)


def main():
    df = source_table()
    tissues = list(df.columns[9:39])
    coding = df[df["Protein"].notna() & df["CDS"].notna() & df["RNA"].notna() & (df["chr"] != "chrMT")]
    cds = coding["CDS"].astype(str).str.upper()
    coding = coding[(cds.str.len() % 3 == 0) & cds.str.startswith("ATG")]
    pilot = pd.concat(
        [coding[coding["split"] == split].sample(n=n, random_state=SEED) for split, n in N_PER_SPLIT.items()]
    )

    tss, regions = transcript_regions(pilot, window=WINDOW)
    sequences = fetch_regions(regions)
    pilot = pilot.assign(TSS=tss, DNA=[sequences[r] for r in regions])

    columns = ["transcript_id_gtex", "gene_id_gtex", "split", "chr", "strand", "TSS",
               "DNA", "RNA", "5UTR", "CDS", "3UTR", "Protein"] + tissues
    output = pilot[columns].rename(columns={"transcript_id_gtex": "transcript_id", "gene_id_gtex": "gene_id"})
    _write_pilot(output, OUTPUT)
    print(f"Wrote {OUTPUT}: {len(pilot)} transcripts")


if __name__ == "__main__":
    main()
