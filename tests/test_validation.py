"""Check supported scalar audits and motif nulls, and reject unsupported options."""

from dataclasses import replace
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

# Preserve the notebook environment's NumPy-before-PyTorch import order on Windows.
import numpy as np
import pandas as pd
from shared_code import analysis, datasets


class MotifValidationTests(unittest.TestCase):
    def test_invalid_strata_fails_before_processing_including_empty_inputs(self):
        for profiles, seqs in (([], []), ([np.ones(9)], ["ACGACGACG"])):
            with self.subTest(sequences=len(seqs)):
                with patch.object(analysis, "scan_sequence_for_motifs") as scan:
                    with self.assertRaisesRegex(ValueError, "strata.*position.*codon"):
                        analysis.motif_permutation_test(profiles, seqs, strata="positon")
                scan.assert_not_called()

    def test_supported_empty_inputs_still_return_none(self):
        for kind in ("position", "codon"):
            with self.subTest(strata=kind):
                self.assertIsNone(analysis.motif_permutation_test([], [], strata=kind))

    def test_supported_seeded_results_and_inputs_are_preserved(self):
        attention = np.array([9., 2., 3., 9., 2., 3., 9., 2., 3.])
        original = attention.copy()
        # Recorded from the prior implementation on this synthetic fixture.
        expected = {"position": (0.20625, 4.66875, 8 / 21), "codon": (4.875, 0., 1.)}
        for kind, (null_gap, excess_gap, p_value) in expected.items():
            with self.subTest(strata=kind):
                with patch.object(analysis, "scan_sequence_for_motifs", return_value=[(0, 1, "candidate")]):
                    result = analysis.motif_permutation_test(
                        [attention], ["ACGACGACG"], strata=kind,
                        n_permutations=20, bin_size=3, seed=7,
                    )
                self.assertEqual(result["sequences"], 1)
                self.assertAlmostEqual(result["observed gap"], 4.875)
                self.assertAlmostEqual(result["null gap"], null_gap)
                self.assertAlmostEqual(result["excess gap"], excess_gap)
                self.assertAlmostEqual(result["permutation p"], p_value)
                np.testing.assert_array_equal(attention, original)

    def test_position_null_matches_all_nine_within_bin_assignments(self):
        attention = np.array([1., 3., 5., 10., 12., 14.])
        # Each three-position bin has one matched position. Enumerate all 3 x 3 choices.
        draws = []
        for first, second in itertools.product(range(3), range(3, 6)):
            draw = np.full(6, 0.1)
            draw[[first, second]] = 0.9
            draws.append(draw)
        rng = Mock()
        rng.random.side_effect = draws
        with patch.object(analysis.np.random, "default_rng", return_value=rng):
            with patch.object(analysis, "scan_sequence_for_motifs", return_value=[
                (2, 3, "candidate"), (5, 6, "candidate"),
            ]):
                result = analysis.motif_permutation_test(
                    [attention], ["ACGACG"], strata="position", bin_size=3, n_permutations=9,
                )
        # Observed means are 9.5 and 6.5. The nine null gaps are
        # -3, -1.5, 0, -1.5, 0, 1.5, 0, 1.5, 3; only one reaches the observation.
        self.assertAlmostEqual(result["observed gap"], 3.)
        self.assertAlmostEqual(result["null gap"], 0.)
        self.assertAlmostEqual(result["excess gap"], 3.)
        self.assertAlmostEqual(result["permutation p"], 2 / 10)

    def test_codon_null_preserves_gap_when_each_offset_has_one_score(self):
        with patch.object(analysis, "scan_sequence_for_motifs", return_value=[(0, 1, "candidate")]):
            result = analysis.motif_permutation_test(
                [np.array([9., 2., 3.] * 3)], ["ACGACGACG"], strata="codon", n_permutations=7,
            )
        # The matched score is always 9; the other eight sum to 33, for a gap of 39/8.
        self.assertAlmostEqual(result["observed gap"], 39 / 8)
        self.assertAlmostEqual(result["null gap"], 39 / 8)
        self.assertEqual(result["excess gap"], 0.)
        self.assertEqual(result["permutation p"], 1.)


class AuditValidationTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            "Sequence": [" acg ", "ACG", "TTTT", "TTTT", "CC", "GGGGG"],
            "Value": [1., 3., 2., 2., 9., 4.],
            "Split": ["train", "test", "train", "validation", "train", "test"],
        })

    def test_scalar_audits_preserve_full_file_summary_and_input(self):
        for key in ("mrna_stability", "mrfp_expression"):
            with self.subTest(dataset=key):
                fixture = self.fixture()
                original = fixture.copy(deep=True)
                with patch.object(datasets.pd, "read_csv", return_value=fixture) as read:
                    actual = datasets.audit(key, length_limit=3)
                read.assert_called_once()
                expected = pd.Series({
                    "Rows": 6,
                    "Distinct sequences": 4,
                    "Rows per split": {"train": 3, "test": 2, "validation": 1},
                    "Distinct sequences in both test and train": 1,
                    "Distinct sequences in both test and validation": 0,
                    "Distinct sequences in both train and validation": 1,
                    "Repeated sequences": 2,
                    "Repeated sequences with differing labels": 1,
                    # Repeated groups have sample SDs sqrt(2) and 0.
                    "Median label SD within a repeated sequence": np.sqrt(2) / 2,
                    "Share of rows with at most 3 nucleotides": 0.5,
                }, name=datasets.DATASETS[key].label, dtype=object)
                pd.testing.assert_series_equal(actual, expected)
                pd.testing.assert_frame_equal(fixture, original)

    def test_gtex_audit_is_rejected_before_reading_data(self):
        with patch.object(datasets.pd, "read_csv") as read:
            with self.assertRaisesRegex(ValueError, "audit.*scalar.*label"):
                datasets.audit("gtex_pilot")
        read.assert_not_called()

    def test_audit_without_a_scalar_label_is_rejected_before_reading_data(self):
        spec = replace(datasets.DATASETS["mrna_stability"], label_column="")
        with patch.dict(datasets.DATASETS, {"missing_label": spec}):
            with patch.object(datasets.pd, "read_csv") as read:
                with self.assertRaisesRegex(ValueError, "audit.*scalar.*label"):
                    datasets.audit("missing_label")
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
