"""Synthetic position/translation checks; no dataset, encoder or cache execution."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import region_inputs as regions


def transcript(utr5="CC", cds="ATGAAATAG", utr3="GGG"):
    sequence = utr5 + cds + utr3
    track = np.zeros(len(sequence), dtype=np.int8)
    track[len(utr5):len(utr5) + len(cds):3] = 1
    return sequence, track


class CodingLayoutTests(unittest.TestCase):
    def test_terminal_stop_retains_cds_but_has_no_protein_residue(self):
        sequence, track = transcript()
        result = regions.coding_layout(sequence, track, terminal_stop="mask",
                                       supplied_protein="MK")
        self.assertEqual((result.cds_start, result.cds_end, result.residue_end), (2, 11, 8))
        self.assertEqual(result.protein, "MK")
        self.assertTrue(result.has_terminal_stop)
        np.testing.assert_array_equal(result.cds_mask, [False] * 2 + [True] * 9 + [False] * 3)
        np.testing.assert_array_equal(result.protein_mask,
                                      [False] * 2 + [True] * 6 + [False] * 6)

    def test_terminal_stop_policy_is_required_and_invalid_policies_fail(self):
        sequence, track = transcript()
        with self.assertRaises(TypeError):
            regions.coding_layout(sequence, track)
        for policy in (None, "trim", "", True):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                regions.coding_layout(sequence, track, terminal_stop=policy)
        with self.assertRaisesRegex(ValueError, "terminal stop"):
            regions.coding_layout(sequence, track, terminal_stop="reject")

    def test_stop_free_cds_and_rna_alphabet_share_the_same_positions(self):
        sequence, track = transcript(cds="AUGAAA")
        result = regions.coding_layout(sequence.lower(), track, terminal_stop="reject")
        self.assertFalse(result.has_terminal_stop)
        self.assertEqual(result.protein, "MK")
        np.testing.assert_array_equal(result.cds_mask, result.protein_mask)
        self.assertEqual(result.sequence_length, len(sequence))

    def test_internal_stop_is_not_silently_truncated(self):
        for cds in ("ATGTAAAAA", "ATGTAGTAA"):
            sequence, track = transcript(cds=cds)
            with self.subTest(cds=cds), self.assertRaisesRegex(ValueError, "internal stop"):
                regions.coding_layout(sequence, track, terminal_stop="mask")

    def test_ambiguous_cds_is_rejected_without_rejecting_ambiguous_utrs(self):
        sequence, track = transcript(utr5="NN", cds="ATGAAA")
        self.assertEqual(regions.coding_layout(sequence, track, terminal_stop="reject").protein,
                         "MK")
        for cds in ("ATGNNN", "ATGAAR"):
            sequence, track = transcript(cds=cds)
            with self.subTest(cds=cds), self.assertRaisesRegex(ValueError, "ambiguous"):
                regions.coding_layout(sequence, track, terminal_stop="mask")

    def test_supplied_protein_must_match_without_invented_stop_residue(self):
        sequence, track = transcript()
        for protein in ("MA", "MK*", "MKX", " MK", 12):
            with self.subTest(protein=protein), self.assertRaisesRegex(ValueError, "supplied"):
                regions.coding_layout(sequence, track, terminal_stop="mask",
                                      supplied_protein=protein)

    def test_invalid_track_dimensions_values_and_missing_cds_fail(self):
        sequence, track = transcript()
        bad_tracks = [track[:-1], track[:, None], np.zeros_like(track),
                      track.astype(float) + 0.5, np.full(len(track), np.nan),
                      np.array(["0"] * len(track))]
        for bad in bad_tracks:
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                regions.coding_layout(sequence, bad, terminal_stop="mask")

    def test_irregular_codon_marks_and_final_incomplete_triplet_fail(self):
        with self.assertRaisesRegex(ValueError, "frame"):
            regions.coding_layout("ATGAAA", [1, 0, 1, 0, 0, 0], terminal_stop="reject")
        # The existing regions() reader clips bounds; this positional path must not.
        with self.assertRaisesRegex(ValueError, "beyond"):
            regions.coding_layout("ATGAA", [1, 0, 0, 1, 0], terminal_stop="reject")

    def test_whitespace_invalid_letters_and_empty_protein_fail_without_cropping(self):
        for sequence in ("", " ATG", "ATG ", "AT-G", "AT?G"):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                regions.coding_layout(sequence, [1] * len(sequence), terminal_stop="mask")
        with self.assertRaisesRegex(ValueError, "does not encode"):
            regions.coding_layout("TAA", [1, 0, 0], terminal_stop="mask")


class ProteinMappingTests(unittest.TestCase):
    def setUp(self):
        sequence, track = transcript()
        self.layout = regions.coding_layout(sequence, track, terminal_stop="mask")

    def test_each_residue_repeats_three_times_and_utr_and_stop_are_zero(self):
        features = np.array([[2., 5.], [7., 11.]], dtype=np.float32)
        before = features.copy()
        aligned, available = regions.protein_to_nucleotides(self.layout, features)
        expected = np.zeros((14, 2), dtype=np.float32)
        expected[2:5] = [2., 5.]
        expected[5:8] = [7., 11.]
        np.testing.assert_array_equal(aligned, expected)
        np.testing.assert_array_equal(available, self.layout.protein_mask)
        np.testing.assert_array_equal(features, before)
        self.assertEqual(aligned.dtype, features.dtype)

    def test_zero_embedding_is_still_available_and_masks_do_not_alias(self):
        aligned, available = regions.protein_to_nucleotides(
            self.layout, np.zeros((2, 3), dtype=np.float16))
        self.assertFalse(aligned.any())
        self.assertEqual(available.sum(), 6)
        available[:] = False
        self.assertEqual(self.layout.protein_mask.sum(), 6)

    def test_extra_special_tokens_nonfinite_values_and_bad_shapes_fail(self):
        cases = [np.zeros((3, 2)), np.zeros((1, 2)), np.zeros((2,)),
                 np.zeros((2, 0)), np.full((2, 2), np.nan),
                 np.zeros((2, 2), dtype=np.int32)]
        for features in cases:
            with self.subTest(shape=features.shape), self.assertRaises(ValueError):
                regions.protein_to_nucleotides(self.layout, features)


class ChunkStitchingTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        self.b = np.array([[5., 6.], [7., 8.], [9., 10.]], dtype=np.float32)

    def test_exact_ordered_coverage_restores_the_known_position_matrix(self):
        result = regions.stitch_chunks([(0, 2, self.a), (2, 5, self.b)], sequence_length=5)
        np.testing.assert_array_equal(result,
                                      np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]))
        self.assertEqual(result.dtype, np.float32)
        result[0] = 0
        np.testing.assert_array_equal(self.a[0], [1, 2])

    def test_reordering_gaps_overlap_and_missing_final_chunk_fail(self):
        cases = [[(2, 5, self.b), (0, 2, self.a)],
                 [(0, 2, self.a), (3, 6, self.b)],
                 [(0, 2, self.a), (1, 4, self.b)],
                 [(0, 2, self.a)], [], [(1, 3, self.a)]]
        for chunks in cases:
            with self.subTest(chunks=len(chunks)), self.assertRaises(ValueError):
                regions.stitch_chunks(chunks, sequence_length=5)

    def test_coordinate_span_width_dtype_and_numeric_values_must_agree(self):
        cases = [[(0, 5, self.a)],
                 [(0, 2, self.a), (2, 5, np.zeros((3, 1), dtype=np.float32))],
                 [(0, 2, self.a), (2, 5, self.b.astype(np.float64))],
                 [(0, 2, self.a), (2, 5, np.full((3, 2), np.inf, dtype=np.float32))],
                 [(False, 2, self.a), (2, 5, self.b)],
                 [(0., 2, self.a), (2, 5, self.b)]]
        for chunks in cases:
            with self.subTest(chunks=len(chunks)), self.assertRaises(ValueError):
                regions.stitch_chunks(chunks, sequence_length=5)
        for length in (0, -1, True, 5.0):
            with self.subTest(length=length), self.assertRaises(ValueError):
                regions.stitch_chunks([(0, 2, self.a)], sequence_length=length)


if __name__ == "__main__":
    unittest.main()
