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
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

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


def source_table():
    """Return IsoFormer's GTEx table, downloading it on first use."""
    path = os.path.join(DOWNLOAD_DIR, "GTEx_final.csv")
    if not os.path.exists(path):
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, path)
    return pd.read_csv(path, low_memory=False)


def fetch_regions(regions):
    """Return {region: sequence} from the Ensembl REST service, 50 regions per request."""
    sequences = {}
    for i in range(0, len(regions), 50):
        request = urllib.request.Request(
            ENSEMBL_URL,
            data=json.dumps({"regions": regions[i:i + 50]}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    for item in json.load(response):
                        sequences[item["query"]] = item["seq"].upper()
                break
            except Exception as error:  # retry transient network or rate-limit errors
                print(f"Retrying after {error}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("Ensembl request failed five times")
        time.sleep(0.3)
    return sequences


def main():
    df = source_table()
    tissues = list(df.columns[9:39])
    coding = df[df["Protein"].notna() & df["CDS"].notna() & df["RNA"].notna() & (df["chr"] != "chrMT")]
    cds = coding["CDS"].astype(str).str.upper()
    coding = coding[(cds.str.len() % 3 == 0) & cds.str.startswith("ATG")]
    pilot = pd.concat(
        [coding[coding["split"] == split].sample(n=n, random_state=SEED) for split, n in N_PER_SPLIT.items()]
    )

    # Zero-based transcription start site, as in the IsoFormer loader
    plus = pilot["strand"] == "+"
    tss = (pilot["start"] - 1).where(plus, pilot["end"] - 1)
    lo = (tss - WINDOW // 2).where(plus, tss - (WINDOW - WINDOW // 2) + 1)  # zero-based, inclusive
    hi = (tss + (WINDOW - WINDOW // 2)).where(plus, tss + WINDOW // 2 + 1)  # zero-based, exclusive
    regions = [
        f"{c.replace('chr', '')}:{a + 1}..{b}:{1 if s == '+' else -1}"
        for c, a, b, s in zip(pilot["chr"], lo, hi, pilot["strand"])
    ]
    sequences = fetch_regions(regions)
    pilot = pilot.assign(TSS=tss, DNA=[sequences[r] for r in regions])

    columns = ["transcript_id_gtex", "gene_id_gtex", "split", "chr", "strand", "TSS",
               "DNA", "RNA", "5UTR", "CDS", "3UTR", "Protein"] + tissues
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pilot[columns].rename(columns={"transcript_id_gtex": "transcript_id", "gene_id_gtex": "gene_id"}).to_csv(
        OUTPUT, index=False
    )
    print(f"Wrote {OUTPUT}: {len(pilot)} transcripts")


if __name__ == "__main__":
    main()
