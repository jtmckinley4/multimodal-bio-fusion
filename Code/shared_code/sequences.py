"""Sequence checks, translation, composition features, and motif scanning.

Functions here act on single sequence strings and need neither PyTorch nor an encoder.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from Bio.Seq import Seq


def is_in_frame_and_starts_correctly(seq):
    """Check for a length divisible by three and an ATG or AUG start."""
    seq = seq.upper().strip()
    return len(seq) % 3 == 0 and seq.startswith(("ATG", "AUG"))


def translate_cds(seq):
    """Translate complete codons using the standard code, up to the first stop."""
    trimmed_seq = seq[: len(seq) - (len(seq) % 3)]
    return str(Seq(trimmed_seq).translate(to_stop=True))


def rna_to_dna(seq):
    """Return an uppercase sequence with U replaced by T."""
    return seq.upper().replace("U", "T")


def dna_to_rna(seq):
    """Return an uppercase sequence with T replaced by U."""
    return seq.upper().replace("T", "U")


def gc_content(seq):
    """Return the G-or-C fraction of a nonempty uppercase sequence."""
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc3_content(seq):
    """Return the GC fraction at third positions of complete codons."""
    return gc_content(seq[2::3])


# ---------------------------------------------------------------------------
# Composition features
# ---------------------------------------------------------------------------

CODONS = ["".join(t) for t in itertools.product("ACGT", repeat=3)]


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def composition_features(seq):
    """Return GC, GC3, log length, 64 codon frequencies, and 20 amino-acid frequencies."""
    dna = rna_to_dna(seq)
    protein = translate_cds(seq)
    codons = [dna[i:i + 3] for i in range(0, len(dna) - 2, 3)]
    codon_counts = pd.Series(codons).value_counts()
    codon_freq = [codon_counts.get(c, 0) / len(codons) for c in CODONS]
    aa_freq = [protein.count(a) / max(len(protein), 1) for a in AMINO_ACIDS]
    return [gc_content(dna), gc3_content(dna), np.log(len(dna))] + codon_freq + aa_freq


def transcript_composition_features(dna, transcript, cds, utr5, utr3):
    """Return composition_features of the coding sequence plus five whole-transcript features.

    The additions are the GC fractions of the transcript and of the genomic DNA window,
    the log transcript length, and log(1 + length) of each untranslated region, so that
    the composition control also covers what lies outside the coding sequence.
    """
    return composition_features(cds) + [
        gc_content(rna_to_dna(transcript)), gc_content(rna_to_dna(dna)), np.log(len(transcript)),
        np.log1p(len(utr5)), np.log1p(len(utr3)),
    ]


# ---------------------------------------------------------------------------
# Candidate stability motifs
# ---------------------------------------------------------------------------

STABILITY_MOTIFS = {"ARE_pentamer": "AUUUA", "ARE_nonamer": "UUAUUUAUU", "m6A_DRACH": "DRACH"}


_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T", "R": "[AG]", "Y": "[CT]", "S": "[GC]",
    "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]", "H": "[ACT]",
    "V": "[ACG]", "N": "[ACGT]",
}


def iupac_to_regex(pattern):
    """Convert an IUPAC consensus pattern to a regex on the DNA alphabet."""
    return "".join(_IUPAC[b] for b in pattern.upper())


def find_iupac_matches(sequence, iupac_pattern):
    """Return all matching zero-based, end-exclusive spans, including overlaps."""
    seq = sequence.upper().replace("U", "T")
    regex = iupac_to_regex(iupac_pattern)
    length = len(iupac_pattern)
    return [
        (m.start(), m.start() + length)
        for m in re.finditer(f"(?={regex})", seq)
    ]


def scan_sequence_for_motifs(sequence, motifs=STABILITY_MOTIFS):
    """Return (start, end, motif_name) for each consensus match, including overlaps.

    Spans are zero-based and end-exclusive; a match is a candidate annotation, not a
    validated regulatory site.
    """
    return [
        (start, end, name)
        for name, pattern in motifs.items()
        for start, end in find_iupac_matches(sequence, pattern)
    ]


# ---------------------------------------------------------------------------
# Secondary-structure proxy
# ---------------------------------------------------------------------------

def can_pair(a, b):
    """Return whether two uppercase bases form an allowed RNA or T-equivalent pair."""
    pair = {a, b}
    return pair in ({"A", "U"}, {"A", "T"}, {"G", "C"}, {"G", "U"}, {"G", "T"})


def nussinov_pairing_fraction(seq, min_loop=3):
    """Return the maximum paired-nucleotide fraction under noncrossing pairing.

    Expect an uppercase sequence and a nonnegative integer min_loop.
    Each pair must enclose at least min_loop sequence positions.
    """
    n = len(seq)
    if n < min_loop + 2:
        return 0.0

    # Each entry covers the inclusive interval from i through j.
    max_pairs = [[0] * n for _ in range(n)]
    for span in range(min_loop + 1, n):
        for i in range(0, n - span):
            j = i + span

            # First consider leaving position i unpaired.
            best_pair_count = max_pairs[i + 1][j]

            # Then consider eligible partners, enforcing the minimum separation.
            for k in range(i + min_loop + 1, j + 1):
                if can_pair(seq[i], seq[k]):
                    inside_pairs = max_pairs[i + 1][k - 1] if k - 1 >= i + 1 else 0
                    remaining_pairs = max_pairs[k + 1][j] if k + 1 <= j else 0
                    candidate_pairs = inside_pairs + 1 + remaining_pairs
                    if candidate_pairs > best_pair_count:
                        best_pair_count = candidate_pairs

            max_pairs[i][j] = best_pair_count

    return (2 * max_pairs[0][n - 1]) / n
