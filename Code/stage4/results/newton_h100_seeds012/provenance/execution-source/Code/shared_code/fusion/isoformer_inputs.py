"""In-memory GTEx preparation for the separate IsoFormer reference workflow.

These functions never read CSVs, open a genome, download assets, normalize targets,
or truncate proteins. They implement only the input contracts that can be checked
without resolving the original trainer. The notebook owns the selected settings.

Source: InstaDeepAI/multi_omics_transcript_expression, revision
330ab1795abca683e692c7ee138a2e89c0c882f1, ``generate_examples`` and ``pad_sequence``:
https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression/blob/
330ab1795abca683e692c7ee138a2e89c0c882f1/multi_omics_transcript_expression.py

The released window convention is deliberately preserved, including its strict
right-boundary exclusion and optional filter's shifted anchor. It differs from the
older Stage 1 pilot's strand-centered convention. Naming it ``released_window``
does not establish which historical dependency/configuration produced the paper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


SOURCE_COLUMNS = (
    "transcript_id_gtex", "gene_id_gtex", "chr", "start", "end", "strand",
    "RNA", "Protein", "split",
)


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (Integral, Real))
            or not np.isfinite(value) or value < 1 or int(value) != value):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sequence(value, name):
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a nonempty sequence without whitespace")
    return value


def protein_present(value):
    """Distinguish an absent field from a supplied, unmodified protein string.

    Missing fields do not establish a noncoding annotation. Nonempty strings are
    preserved, including unusual residue symbols; whitespace or other nonmissing
    values are malformed input, not another representation of absence.
    """
    if isinstance(value, str):
        if value == "":
            return False
        _sequence(value, "Protein")
        return True
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    raise ValueError("Protein must be a supplied sequence or a missing field")


@dataclass(frozen=True)
class GTExCohort:
    """Validated rows and the explicitly selected order of transcript targets.

    ``rows.index`` retains the caller's source-row identity. Coordinates are the
    release's one-based ``start``/``end`` values, not already shifted coordinates.
    ``labels`` returns supplied values in tissue order, without a log or scaling.
    """

    rows: pd.DataFrame
    tissues: tuple[str, ...]

    @property
    def labels(self):
        return self.rows.loc[:, list(self.tissues)].to_numpy(dtype=np.float64, copy=True)


def validate_cohort(rows, tissues):
    """Validate a supplied table without filtering, reordering or changing it.

    The caller supplies tissue order rather than inferring it from the last columns.
    Raw train/test/eval names are retained as supplied; accepting the raw eval label
    does not decide how those rows enter an experiment. ``validation_partitions``
    refuses them until that policy is applied explicitly. Missing protein fields are
    permitted without inventing a biological annotation. DNA eligibility, including
    the release's ``chrMT`` exclusion, is not selected by this schema validator.
    """
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        raise ValueError("rows must be a nonempty DataFrame")
    if not rows.columns.is_unique:
        raise ValueError("duplicate column names would make the input ambiguous")
    if (not rows.index.is_unique or rows.index.nlevels != 1
            or rows.index.hasnans):
        raise ValueError("source row index must be unique, nonmissing and one-dimensional")
    if isinstance(tissues, str):
        raise ValueError("tissues must be an ordered collection, not one string")
    tissues = tuple(tissues)
    if (not tissues or any(not isinstance(t, str) or not t.strip() for t in tissues)
            or len(set(tissues)) != len(tissues)
            or set(tissues).intersection(SOURCE_COLUMNS)):
        raise ValueError("tissues must contain unique, nonempty target column names")
    missing = set(SOURCE_COLUMNS + tissues).difference(rows.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    for name in ("transcript_id_gtex", "gene_id_gtex", "chr"):
        if not all(isinstance(value, str) and value.strip() for value in rows[name]):
            raise ValueError(f"{name} must contain nonempty string identifiers")
    if rows["transcript_id_gtex"].duplicated().any():
        raise ValueError("transcript_id_gtex must be unique across the supplied cohort")
    if not rows["strand"].isin(("+", "-")).all():
        raise ValueError("strand must be '+' or '-'")
    for start, end in zip(rows["start"], rows["end"]):
        if _positive_integer(start, "start") > _positive_integer(end, "end"):
            raise ValueError("start must not exceed end")
    for value in rows["RNA"]:
        _sequence(value, "RNA")
    for value in rows["Protein"]:
        protein_present(value)
    if not rows["split"].isin(("train", "test", "eval")).all():
        raise ValueError("split must be raw 'train', 'test' or 'eval'; "
                         "validation must come from training")
    if not {"train", "test"}.issubset(set(rows["split"])):
        raise ValueError("the supplied cohort must contain both train and test rows")
    for tissue in tissues:
        dtype = rows[tissue].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError(f"target {tissue!r} must be real numeric data")
    labels = rows.loc[:, list(tissues)].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(labels).all():
        raise ValueError("all tissue targets must be finite")
    return GTExCohort(rows.copy(deep=True), tissues)


@dataclass(frozen=True)
class ReleasedWindow:
    """A zero-based, half-open slice and its release-defined eligibility."""

    anchor: int
    low: int
    high: int
    strand: str
    chromosome_length: int
    exclusion: str | None

    @property
    def eligible(self):
        return self.exclusion is None


def released_window(*, start, end, strand, sequence_length, chromosome_length,
                    extra_filter_length):
    """Plan the pinned loader's slice, requiring an explicit optional filter.

    ``extra_filter_length=None`` disables the optional release filter. When enabled,
    its bounds are computed around the already shifted ``low``, as in the source;
    this is not a corrected symmetric exclusion around the transcript anchor.
    The right boundary rejects ``high == chromosome_length`` even though a normal
    Python half-open slice would allow it. No boundary is padded or clamped.
    """
    start = _positive_integer(start, "start")
    end = _positive_integer(end, "end")
    length = _positive_integer(sequence_length, "sequence_length")
    chromosome_length = _positive_integer(chromosome_length, "chromosome_length")
    if start > end:
        raise ValueError("start must not exceed end")
    if strand not in ("+", "-"):
        raise ValueError("strand must be '+' or '-'")
    anchor = (start if strand == "+" else end) - 1
    low = anchor - length // 2
    high = anchor + length // 2 + length % 2
    exclusion = None
    if extra_filter_length is not None:
        filter_length = _positive_integer(extra_filter_length, "extra_filter_length")
        filter_low = low - filter_length // 2
        filter_high = low + filter_length // 2 + filter_length % 2
        if filter_low < 0 or filter_high >= chromosome_length:
            exclusion = "released_extra_filter_boundary"
    if exclusion is None and (low < 0 or high >= chromosome_length):
        exclusion = "released_window_boundary"
    return ReleasedWindow(anchor, low, high, strand, chromosome_length, exclusion)


def extract_released_window(chromosome, window):
    """Extract from a supplied chromosome string; no FASTA or file is opened.

    Reverse complementation precedes the release's A/C/G/T-or-N standardization.
    A rejected window returns None; it cannot silently become a shorter sequence.
    """
    # Validate only the requested slice; scanning an entire chromosome per transcript
    # would repeat genome-scale work for every otherwise small window extraction.
    if not isinstance(chromosome, str) or not chromosome:
        raise ValueError("chromosome must be a nonempty sequence string")
    if not isinstance(window, ReleasedWindow):
        raise TypeError("window must be a ReleasedWindow")
    if len(chromosome) != window.chromosome_length:
        raise ValueError("chromosome length does not match the window plan")
    if not window.eligible:
        return None
    if (window.strand not in ("+", "-") or window.low < 0
            or window.high >= len(chromosome) or window.low >= window.high):
        raise ValueError("invalid eligible window bounds or strand")
    sequence = _sequence(chromosome[window.low:window.high], "DNA window").upper()
    if window.strand == "-":
        # IUPAC complements remain ambiguous until standardization; U complements A.
        complement = str.maketrans("ACGTRYSWKMBDHVNU", "TGCAYRSWMKVHDBNA")
        sequence = sequence.translate(complement)[::-1]
    return "".join(base if base in "ACGT" else "N" for base in sequence)


def crop_rna_3prime(sequence, *, max_nucleotides):
    """Retain the requested RNA 3-prime suffix, preserving its supplied alphabet.

    This is a nucleotide crop, not a tokenizer-capacity check. It does not impose a
    second token limit or change special-token accounting.
    """
    sequence = _sequence(sequence, "RNA")
    maximum = _positive_integer(max_nucleotides, "max_nucleotides")
    return sequence[-maximum:]


def apply_sequence_policies(cohort, *, rna_max_nucleotides, protein_max_characters):
    """Apply explicit RNA-suffix and supplied-protein exclusion rules in memory.

    Return ``(prepared_cohort, report)``. The report covers every source row, including
    excluded rows; its RNA lengths describe the suffix rule before protein exclusion.
    The retained cohort preserves row order, indices, targets, partition labels and
    supplied protein fields. Only retained RNA strings can change. Missing proteins
    are retained, whereas supplied overlength proteins exclude their entire row from
    the common comparison cohort rather than becoming an absent modality.

    The notebook supplies the agreed 12,288-base and 1,200-character limits. Counting
    protein source characters precedes tokenization and includes unusual symbols.
    Neither DNA-boundary eligibility nor raw-eval admission is selected here, so the
    output is sequence-prepared data, not the final training population.
    """
    if not isinstance(cohort, GTExCohort):
        raise TypeError("cohort must come from validate_cohort")
    cohort = validate_cohort(cohort.rows, cohort.tissues)
    rna_maximum = _positive_integer(rna_max_nucleotides, "rna_max_nucleotides")
    protein_maximum = _positive_integer(protein_max_characters, "protein_max_characters")
    rows = cohort.rows
    present = rows["Protein"].map(protein_present)
    protein_lengths = pd.Series(
        [len(value) if supplied else pd.NA
         for value, supplied in zip(rows["Protein"], present)],
        index=rows.index, dtype="Int64",
    )
    too_long = protein_lengths.gt(protein_maximum).fillna(False).astype(bool)
    rna_lengths = rows["RNA"].str.len()
    retained_lengths = rna_lengths.clip(upper=rna_maximum)
    report = rows.loc[:, ["transcript_id_gtex", "split"]].copy()
    report["protein_present"] = present
    report["protein_source_characters"] = protein_lengths
    report["rna_source_nucleotides"] = rna_lengths
    report["rna_retained_nucleotides"] = retained_lengths
    report["rna_removed_5prime_nucleotides"] = rna_lengths - retained_lengths
    report["sequence_eligible"] = ~too_long
    report["sequence_exclusion"] = np.where(
        too_long, "supplied_protein_exceeds_character_limit", None,
    )

    prepared = rows.loc[~too_long].copy(deep=True)
    if not {"train", "test"}.issubset(set(prepared["split"])):
        raise ValueError("sequence policies must retain both train and test rows; "
                         "the supplied cohort cannot support that comparison")
    prepared["RNA"] = prepared["RNA"].map(
        lambda sequence: crop_rna_3prime(sequence, max_nucleotides=rna_maximum)
    )
    return GTExCohort(prepared, cohort.tissues), report


def apply_reference_eligibility(cohort, chromosome_sizes, *, sequence_length,
                                extra_filter_length, raw_eval_policy):
    """Apply the selected released DNA boundaries and leave raw eval unused.

    Call after the common RNA/protein sequence policies. Return a new cohort and
    an in-memory report covering every supplied row, without reading a genome,
    changing sequences/targets or assigning validation. Exact ``chrMT`` exclusion
    follows the loader; chromosome names are never aliased or repaired here.

    Both lengths must be explicit. The selected reference retains the release's
    shifted extra filter and strict right-boundary comparison. Its first boundary
    reason checks the extra filter before the main window; separate flags expose
    overlapping failures rather than implying the historical audit's main-first
    reason counts are identical. Raw eval and mitochondrial rows need no window
    assessment and have missing window fields. Neither rejection damages the data:
    these are the declared admission rules, not recovered training membership.
    """
    if not isinstance(cohort, GTExCohort):
        raise TypeError("cohort must come from validate_cohort")
    cohort = validate_cohort(cohort.rows, cohort.tissues)
    length = _positive_integer(sequence_length, "sequence_length")
    filter_length = _positive_integer(extra_filter_length, "extra_filter_length")
    if filter_length < length:
        raise ValueError("extra_filter_length must be at least sequence_length, "
                         "as required by the released loader")
    if raw_eval_policy != "exclude":
        raise ValueError("raw_eval_policy must explicitly be 'exclude'; no reassignment "
                         "of raw eval is implemented")
    if not isinstance(chromosome_sizes, Mapping):
        raise TypeError("chromosome_sizes must map exact chromosome names to lengths")
    sizes = {}
    for name, value in chromosome_sizes.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("chromosome_sizes keys must be exact nonempty chromosome names")
        sizes[name] = _positive_integer(value, f"chromosome_sizes[{name!r}]")

    rows = cohort.rows
    records = []
    for _, row in rows.iterrows():
        raw_eval = row["split"] == "eval"
        mitochondrial = row["chr"] == "chrMT"
        record = {
            "transcript_id_gtex": row["transcript_id_gtex"],
            "split": row["split"],
            "raw_eval_excluded": raw_eval,
            "mitochondrial_excluded": mitochondrial,
            "dna_window_checked": False,
            "dna_anchor": pd.NA,
            "dna_low": pd.NA,
            "dna_high": pd.NA,
            "chromosome_length": pd.NA,
            "main_window_excluded": pd.NA,
            "extra_filter_excluded": pd.NA,
            "reference_exclusion": None,
        }
        if raw_eval:
            record["reference_exclusion"] = "raw_eval_unused"
        elif mitochondrial:
            record["reference_exclusion"] = "chrMT"
        else:
            if row["chr"] not in sizes:
                raise ValueError(f"Missing chromosome length for {row['chr']!r} "
                                 f"at transcript {row['transcript_id_gtex']!r}; "
                                 "the candidate row was not silently excluded")
            window = released_window(
                start=row["start"], end=row["end"], strand=row["strand"],
                sequence_length=length, chromosome_length=sizes[row["chr"]],
                extra_filter_length=filter_length,
            )
            record.update(
                dna_window_checked=True, dna_anchor=window.anchor,
                dna_low=window.low, dna_high=window.high,
                chromosome_length=window.chromosome_length,
                main_window_excluded=(window.low < 0
                                      or window.high >= window.chromosome_length),
                extra_filter_excluded=(window.exclusion == "released_extra_filter_boundary"),
                reference_exclusion=window.exclusion,
            )
        record["reference_eligible"] = record["reference_exclusion"] is None
        records.append(record)

    report = pd.DataFrame(records, index=rows.index)
    for name in ("dna_anchor", "dna_low", "dna_high", "chromosome_length"):
        report[name] = report[name].astype("Int64")
    for name in ("main_window_excluded", "extra_filter_excluded"):
        report[name] = report[name].astype("boolean")
    retained = rows.loc[report["reference_eligible"]].copy(deep=True)
    if not {"train", "test"}.issubset(set(retained["split"])):
        raise ValueError("reference eligibility must retain both train and test rows; "
                         "the supplied cohort cannot support that comparison")
    return GTExCohort(retained, cohort.tissues), report


def validation_partitions(cohort, *, sampling_unit, seed, validation_fraction=0.05):
    """Select validation from supplied training rows and return source-index arrays.

    The paper specifies 5% of training and changes validation membership across
    repeats. Exact membership, sampler and sampling unit were not recovered.
    This implementation uses NumPy permutation and rounds the selected unit count
    upward. Choosing transcript or gene units is explicit reconstruction behavior;
    gene units can give a validation row fraction different from the unit fraction.
    No normalization is fitted and the supplied test membership is untouched.
    """
    if not isinstance(cohort, GTExCohort):
        raise TypeError("cohort must come from validate_cohort")
    # A DataFrame remains mutable even inside a frozen dataclass. Recheck the boundary.
    cohort = validate_cohort(cohort.rows, cohort.tissues)
    if (cohort.rows["split"] == "eval").any():
        raise ValueError("raw eval rows remain: apply an agreed admission policy "
                         "explicitly before constructing validation")
    if sampling_unit not in ("transcript", "gene"):
        raise ValueError("sampling_unit must explicitly be 'transcript' or 'gene'")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(validation_fraction, (bool, np.bool_))
            or not isinstance(validation_fraction, Real)
            or not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be strictly between zero and one")
    training = cohort.rows.loc[cohort.rows["split"] == "train"]
    unit_column = "gene_id_gtex" if sampling_unit == "gene" else "transcript_id_gtex"
    units = training[unit_column].drop_duplicates().to_numpy()
    count = int(np.ceil(len(units) * validation_fraction))
    if count >= len(units):
        raise ValueError("not enough training units for nonempty train and validation sets")
    chosen = np.random.default_rng(seed).permutation(units)[:count]
    in_validation = training[unit_column].isin(chosen)
    return {
        "train": training.index[~in_validation].to_numpy(copy=True),
        "val": training.index[in_validation].to_numpy(copy=True),
        "test": cohort.rows.index[cohort.rows["split"] == "test"].to_numpy(copy=True),
    }


def gene_overlap(cohort, partitions):
    """Report gene intersections for declared partitions without changing them."""
    if set(partitions) != {"train", "val", "test"}:
        raise ValueError("partitions must contain train, val and test source-row indices")
    flattened = [key for part in partitions.values() for key in part]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(cohort.rows.index):
        raise ValueError("partitions must cover every source row exactly once")
    genes = {name: set(cohort.rows.loc[indices, "gene_id_gtex"])
             for name, indices in partitions.items()}
    return {f"{left}/{right}": tuple(sorted(genes[left] & genes[right]))
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}
