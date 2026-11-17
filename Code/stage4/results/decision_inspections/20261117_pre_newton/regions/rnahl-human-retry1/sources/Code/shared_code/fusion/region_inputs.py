"""Pure input mechanics for the proposed nucleotide-resolution region experiment.

Coordinates are zero-based, half-open positions in the supplied transcript, which
must already be in transcriptional orientation. These helpers do not choose an
encoder, context window, cohort exclusion rule or terminal-stop policy. They do
not load data, extract embeddings or publish caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mrnabench import cds_bounds
from ..sequences import translate_cds


@dataclass(frozen=True)
class CodingLayout:
    """The annotated CDS and the smaller span with actual translated residues."""

    sequence_length: int
    cds_start: int
    cds_end: int
    residue_end: int
    protein: str
    has_terminal_stop: bool

    @property
    def cds_mask(self):
        """Annotated CDS bases, including a terminal stop when the track marks it."""
        mask = np.zeros(self.sequence_length, dtype=bool)
        mask[self.cds_start:self.cds_end] = True
        return mask

    @property
    def protein_mask(self):
        """Bases with a corresponding amino acid, independent of feature values."""
        mask = np.zeros(self.sequence_length, dtype=bool)
        mask[self.cds_start:self.residue_end] = True
        return mask


def coding_layout(sequence, cds_track, *, terminal_stop, supplied_protein=None):
    """Validate the codon-start track and translate its complete coding region.

    ``terminal_stop`` must be explicit: ``"reject"`` refuses a marked terminal
    stop; ``"mask"`` retains its three transcript positions but excludes them
    from the protein-available span. It never invents a residue for a stop codon.
    A supplied protein must match the resulting amino-acid string, without a
    stop marker. The standard genetic code matches the existing translator;
    alternative codes, recoding and partial-CDS policies are not inferred here.
    """
    if terminal_stop not in ("reject", "mask"):
        raise ValueError("terminal_stop must explicitly be 'reject' or 'mask'")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a nonempty transcript string")
    # Changing letter case/alphabet preserves coordinates; trimming does not.
    sequence = sequence.upper().replace("U", "T")
    if set(sequence) - set("ACGTRYSWKMBDHVN"):
        raise ValueError("sequence contains whitespace or non-nucleotide characters")

    track = np.asarray(cds_track)
    if track.ndim != 1 or len(track) != len(sequence):
        raise ValueError("cds_track must have one value per transcript nucleotide")
    if track.dtype.kind not in "biuf" or not np.isin(track, (0, 1)).all():
        raise ValueError("cds_track must contain only binary codon-start marks")
    start, end = cds_bounds(track)
    if end > len(sequence):
        raise ValueError("the last marked codon extends beyond the transcript")
    cds = sequence[start:end]
    if len(cds) % 3:
        raise ValueError("the annotated CDS does not contain complete codons")
    if set(cds) - set("ACGT"):
        raise ValueError("the annotated CDS contains ambiguous nucleotides")

    stop_positions = [offset for offset in range(0, len(cds), 3)
                      if cds[offset:offset + 3] in ("TAA", "TAG", "TGA")]
    if any(offset != len(cds) - 3 for offset in stop_positions):
        raise ValueError("the annotated CDS contains an internal stop codon")
    has_terminal_stop = bool(stop_positions)
    if has_terminal_stop and terminal_stop == "reject":
        raise ValueError("a terminal stop is present but terminal_stop='reject'")

    # The legacy translator stops at '*'; validation above prevents silent loss
    # of internal coding sequence and ensures the complete frame is translated.
    protein = translate_cds(cds)
    if not protein:
        raise ValueError("the annotated CDS does not encode any amino-acid residues")
    if supplied_protein is not None:
        if not isinstance(supplied_protein, str) or supplied_protein.upper() != protein:
            raise ValueError("supplied protein does not match the annotated CDS translation")
    residue_end = end - (3 if has_terminal_stop else 0)
    if 3 * len(protein) != residue_end - start:
        raise ValueError("translation does not provide exactly one residue per coding triplet")
    return CodingLayout(len(sequence), start, end, residue_end, protein, has_terminal_stop)


def _feature_matrix(values, name):
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a position-by-feature matrix with positive width")
    if matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite floating-point features")
    return matrix


def protein_to_nucleotides(layout, embeddings):
    """Repeat each residue vector at its three coding bases; zero-fill elsewhere.

    Embeddings contain one row per translated residue, with no special or padding
    tokens. The returned Boolean mask indicates availability even if an actual
    residue's vector is all zeros. It is not inferred from nonzero feature values.
    """
    if not isinstance(layout, CodingLayout):
        raise TypeError("layout must be the CodingLayout returned by coding_layout")
    matrix = _feature_matrix(embeddings, "protein embeddings")
    if len(matrix) != len(layout.protein):
        raise ValueError("protein embeddings must have exactly one row per translated residue")
    if layout.residue_end - layout.cds_start != 3 * len(matrix):
        raise ValueError("protein layout and translated residue span disagree")
    aligned = np.zeros((layout.sequence_length, matrix.shape[1]), dtype=matrix.dtype)
    aligned[layout.cds_start:layout.residue_end] = np.repeat(matrix, 3, axis=0)
    return aligned, layout.protein_mask


def stitch_chunks(chunks, *, sequence_length):
    """Join explicit ``(start, end, matrix)`` chunks into one complete position grid.

    Chunks must already be ordered, adjacent and non-overlapping, cover [0, length),
    and contain one feature row per declared position. Coordinates can refer to
    nucleotides or residues, but the caller must use one unit consistently. This
    validates supplied coverage; it chooses no windows, overlap blending or crop.
    """
    if (isinstance(sequence_length, (bool, np.bool_))
            or not isinstance(sequence_length, (int, np.integer))
            or sequence_length <= 0):
        raise ValueError("sequence_length must be a positive integer")
    cursor, matrices = 0, []
    width = dtype = None
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, (tuple, list)) or len(chunk) != 3:
            raise ValueError("each chunk must explicitly supply (start, end, matrix)")
        start, end, values = chunk
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) for value in (start, end)):
            raise ValueError("chunk start and end must be integer positions")
        if start != cursor or end <= start or end > sequence_length:
            raise ValueError(f"chunk {index} is unordered, overlapping, gapped or out of bounds")
        matrix = _feature_matrix(values, f"chunk {index}")
        if len(matrix) != end - start:
            raise ValueError(f"chunk {index} feature rows do not match its coordinate span")
        if width is None:
            width, dtype = matrix.shape[1], matrix.dtype
        elif matrix.shape[1] != width or matrix.dtype != dtype:
            raise ValueError("chunks must have the same feature width and dtype")
        matrices.append(matrix)
        cursor = end
    if cursor != sequence_length:
        raise ValueError("chunks do not cover the entire declared sequence")
    return np.concatenate(matrices, axis=0)
