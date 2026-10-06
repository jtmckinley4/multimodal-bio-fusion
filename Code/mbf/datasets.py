"""Registry of labeled sequence datasets, and the inputs each encoder reads from them.

Every entry names its file, sequence and target columns, task, published split column,
and setting. In the "derived" setting (Setting A in the README), each row holds one
coding sequence: DNA encoders read it in DNA letters, RNA encoders read it in RNA
letters, and protein encoders read its translation. In the "distinct" setting (Setting B),
GTEx rows carry separate genomic DNA, transcript, coding sequence, and protein columns,
with expression targets for 30 tissues and gene identifiers for grouped evaluation.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .encoders import encoder_input, protein_input
from .sequences import dna_to_rna, rna_to_dna
from .sequences import is_in_frame_and_starts_correctly, translate_cds


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"


@dataclass(frozen=True)
class DatasetSpec:
    """A labeled sequence dataset and where its encoder inputs come from.

    path is a filename within the selected data directory, which defaults to the
    repository's datasets folder. split_column names the column holding the published
    train, validation, and test assignment, when the file has one.

    In the "derived" setting every encoder input comes from sequence_column, one coding
    sequence. In the "distinct" setting, input_columns names separate columns: "dna"
    (genomic DNA, read by DNA encoders), "transcript" (the full transcript, read by RNA
    encoders with one token per nucleotide), "cds" (the coding sequence, read by codon
    encoders and used for the sequence filters), and "protein". label_columns, when given,
    holds several targets, such as expression in each tissue; group_column names the
    column whose values must stay within one fold, such as the gene.
    """

    key: str
    label: str
    path: str
    sequence_column: str
    label_column: str
    task: str
    setting: str
    split_column: str | None = None
    source: str = ""
    label_columns: tuple = ()
    group_column: str | None = None
    input_columns: dict | None = None


_CODONBERT = "CodonBERT fine-tuning benchmark, github.com/Sanofi-Public/CodonBERT"
_ISOFORMER = (
    "IsoFormer GTEx transcript expression, huggingface.co/datasets/InstaDeepAI/"
    "multi_omics_transcript_expression; DNA windows from Ensembl GRCh38"
)

# The 30 tissues of IsoFormer's GTEx task, in the file's column order
GTEX_TISSUES = (
    "Adipose Tissue", "Adrenal Gland", "Bladder", "Blood", "Blood Vessel", "Brain", "Breast",
    "Cervix Uteri", "Colon", "Esophagus", "Fallopian Tube", "Heart", "Kidney", "Liver", "Lung",
    "Muscle", "Nerve", "Ovary", "Pancreas", "Pituitary", "Prostate", "Salivary Gland", "Skin",
    "Small Intestine", "Spleen", "Stomach", "Testis", "Thyroid", "Uterus", "Vagina",
)

DATASETS = {
    d.key: d
    for d in [
        DatasetSpec("mrna_stability", "mRNA stability", "mRNA_Stability.csv", "Sequence", "Value",
                    task="regression", setting="derived", split_column="Split", source=_CODONBERT),
        DatasetSpec("mrfp_expression", "mRFP expression (synonymous-recoding control)",
                    "mRFP_Expression.csv", "Sequence", "Value",
                    task="regression", setting="derived", split_column="Split", source=_CODONBERT),
        DatasetSpec("gtex_pilot", "GTEx transcript expression, 2,000-transcript pilot", "GTEx_pilot.csv",
                    "CDS", "", task="regression", setting="distinct", split_column="split",
                    source=_ISOFORMER, label_columns=GTEX_TISSUES, group_column="gene_id",
                    input_columns={"dna": "DNA", "transcript": "RNA", "cds": "CDS", "protein": "Protein",
                                   "utr5": "5UTR", "utr3": "3UTR"}),
    ]
}


def is_retained(seq):
    """Apply the Stage 1 filters to one uppercase, stripped coding sequence.

    A sequence is kept when its length is a multiple of three, it starts with ATG or
    AUG, and it translates to at least five amino acids.
    """
    return is_in_frame_and_starts_correctly(seq) and len(translate_cds(seq)) >= 5


@dataclass
class Dataset:
    """The retained rows of one dataset, in input order.

    rows keeps the original file index and columns, so published splits and other
    columns stay attached to each retained sequence. n_sampled counts the rows before
    filtering, and dropped counts the rows each filter removed.
    """

    spec: DatasetSpec
    rows: pd.DataFrame
    sequences: list[str]
    labels: np.ndarray
    n_sampled: int = 0
    dropped: dict = field(default_factory=dict)

    @property
    def groups(self):
        """Cross-validation groups: the group column when the dataset has one, otherwise
        the sequences, so that identical sequences share a group."""
        if self.spec.group_column is not None:
            return self.rows[self.spec.group_column].astype(str).tolist()
        return self.sequences

    def column(self, name):
        """Return one of the distinct-setting input columns as uppercase strings, "" when missing."""
        values = self.rows[self.spec.input_columns[name]].fillna("")
        return [str(v).strip().upper() for v in values]

    @property
    def lengths(self):
        """Sequence lengths as a one-column matrix, for the length-only baseline probe."""
        return np.array([len(s) for s in self.sequences]).reshape(-1, 1)

    def inputs_for(self, encoder):
        """Return the string each retained row contributes to one encoder.

        Derived setting: the coding sequence in the encoder's alphabet, or its translation.
        Distinct setting: the genomic DNA for a DNA encoder, the coding sequence for a codon
        encoder, the full transcript for any other RNA encoder, and the protein for a
        protein encoder.
        """
        if self.spec.setting == "derived":
            return [encoder_input(encoder, s) for s in self.sequences]
        if encoder.modality == "DNA":
            return [rna_to_dna(s) for s in self.column("dna")]
        if encoder.modality == "RNA":
            return [dna_to_rna(s) for s in self.column("cds" if encoder.codon else "transcript")]
        return [protein_input(encoder, s) for s in self.column("protein")]


def sample_rows(key, n_rows=None, seed=42, data_dir=DEFAULT_DATA_DIR):
    """Return a registered dataset's rows with a sequence, sampled before any filtering.

    With n_rows, n_rows rows are drawn without replacement using seed; otherwise every
    row is returned. data_dir defaults to the repository's datasets folder, regardless
    of the working directory. An explicit relative data_dir uses the working directory.
    """
    spec = DATASETS[key]
    df = pd.read_csv(os.path.join(data_dir, spec.path)).dropna(subset=[spec.sequence_column])
    return df if n_rows is None else df.sample(n=n_rows, random_state=seed)


def retain(spec, df):
    """Apply the Stage 1 filters to sampled rows, keeping their order.

    The filters act on sequence_column, the coding sequence in both settings.
    """
    sequences = [str(value).strip().upper() for value in df[spec.sequence_column].fillna("")]
    in_frame = np.array([is_in_frame_and_starts_correctly(s) for s in sequences], dtype=bool)
    keep = np.array([is_retained(s) for s in sequences], dtype=bool)
    rows = df[keep]
    return Dataset(
        spec=spec,
        rows=rows,
        sequences=[s for s, kept in zip(sequences, keep) if kept],
        labels=rows[list(spec.label_columns) if spec.label_columns else spec.label_column].to_numpy(),
        n_sampled=len(df),
        dropped={
            "failed the length or start check": int((~in_frame).sum()),
            "translated to fewer than five amino acids": int((in_frame & ~keep).sum()),
        },
    )


def load_dataset(key, n_rows=None, seed=42, data_dir=DEFAULT_DATA_DIR):
    """Load a registered dataset, sample it, and keep the rows that pass the Stage 1 filters."""
    return retain(DATASETS[key], sample_rows(key, n_rows, seed, data_dir))


def official_split_sample(key, n_train, n_test, seed=42, data_dir=DEFAULT_DATA_DIR):
    """Sample rows from a dataset's published train and test splits and apply the filters.

    Returns (train, test) Datasets. Each split is sampled before filtering, with n_train
    or n_test rows drawn without replacement using seed, so the retained counts can be
    slightly smaller. Use these for comparisons with published results on the same split.
    """
    spec = DATASETS[key]
    if spec.split_column is None:
        raise ValueError(f"{key} has no published split column")
    df = sample_rows(key, data_dir=data_dir)
    parts = []
    for name, n in (("train", n_train), ("test", n_test)):
        rows = df[df[spec.split_column] == name]
        parts.append(retain(spec, rows.sample(n=min(n, len(rows)), random_state=seed)))
    return tuple(parts)


def audit(key, data_dir=DEFAULT_DATA_DIR, length_limit=1000):
    """Summarize a scalar-label dataset's whole file for comparisons with published results.

    Reports rows and distinct sequences, rows per published split and the distinct
    sequences shared by each pair of splits, repeated sequences and how many carry
    differing labels, the median label SD within a repeated sequence, and the share of
    rows no longer than length_limit nucleotides. Reject datasets without one scalar
    label column, including multi-target GTEx, before reading the file.
    """
    spec = DATASETS[key]
    if spec.label_columns or not spec.label_column:
        raise ValueError("audit requires one scalar label column; multi-target datasets are unsupported.")
    df = pd.read_csv(os.path.join(data_dir, spec.path))
    seqs = df[spec.sequence_column].astype(str).str.strip().str.upper()
    summary = {"Rows": len(df), "Distinct sequences": seqs.nunique()}
    if spec.split_column is not None:
        splits = df[spec.split_column]
        summary["Rows per split"] = splits.value_counts().to_dict()
        sets = {name: set(seqs[splits == name]) for name in splits.unique()}
        for a, b in itertools.combinations(sorted(sets), 2):
            summary[f"Distinct sequences in both {a} and {b}"] = len(sets[a] & sets[b])
    by_sequence = df.assign(seq=seqs).groupby("seq")[spec.label_column].agg(["count", "nunique", "std"])
    repeated = by_sequence[by_sequence["count"] > 1]
    summary["Repeated sequences"] = len(repeated)
    summary["Repeated sequences with differing labels"] = int((repeated["nunique"] > 1).sum())
    summary["Median label SD within a repeated sequence"] = repeated["std"].median()
    summary[f"Share of rows with at most {length_limit:,} nucleotides"] = (seqs.str.len() <= length_limit).mean()
    return pd.Series(summary, name=spec.label, dtype=object)
