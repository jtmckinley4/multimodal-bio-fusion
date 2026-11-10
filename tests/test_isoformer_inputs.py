"""Synthetic input checks; no research tables, genomes, models or network access."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

with patch("urllib.request.urlopen", side_effect=AssertionError("No downloads")), \
        patch.object(pd, "read_csv", side_effect=AssertionError("No research data reads")):
    from shared_code.fusion import isoformer_inputs as inputs


TISSUES = ("second_tissue", "first_tissue")


def rows():
    n = 12
    return pd.DataFrame({
        "transcript_id_gtex": [f"tx-{i}" for i in range(n)],
        "gene_id_gtex": [f"gene-{i // 2}" for i in range(n)],
        "chr": ["chr1"] * n,
        "start": [101] * n,
        "end": [200] * n,
        "strand": ["+", "-"] * (n // 2),
        "RNA": ["ACGTN"] * n,
        "Protein": ["MKT"] * n,
        "split": ["train"] * 10 + ["test"] * 2,
        "first_tissue": np.arange(n, dtype=float),
        "second_tissue": np.arange(n, dtype=float) + 100,
        "unrelated": ["retained"] * n,
    }, index=[f"source-{i}" for i in range(n)])


class CohortTests(unittest.TestCase):
    def test_explicit_tissue_order_original_values_and_row_identity_are_retained(self):
        original = rows()
        before = original.copy(deep=True)
        cohort = inputs.validate_cohort(original, TISSUES)
        pd.testing.assert_frame_equal(original, before)
        pd.testing.assert_frame_equal(cohort.rows, before)
        np.testing.assert_array_equal(cohort.labels[0], [100.0, 0.0])
        self.assertEqual(cohort.tissues, TISSUES)
        original.loc["source-0", "second_tissue"] = 999
        self.assertEqual(cohort.labels[0, 0], 100)
        self.assertEqual(cohort.rows["unrelated"].unique().tolist(), ["retained"])

    def test_duplicate_identity_and_missing_or_duplicate_columns_fail(self):
        duplicate_id = rows()
        duplicate_id.loc["source-1", "transcript_id_gtex"] = "tx-0"
        duplicate_index = rows()
        duplicate_index.index = [0] * len(duplicate_index)
        duplicate_column = pd.concat([rows(), rows()[["RNA"]]], axis=1)
        for frame in (duplicate_id, duplicate_index, duplicate_column, rows().drop(columns="RNA")):
            with self.subTest(columns=list(frame.columns)), self.assertRaises(ValueError):
                inputs.validate_cohort(frame, TISSUES)

    def test_bad_coordinate_strand_and_sequence_fields_fail(self):
        cases = [("start", 0), ("end", 1.5), ("start", 201), ("start", True),
                 ("strand", "?"), ("RNA", ""), ("RNA", "A C"), ("Protein", "M K"),
                 ("gene_id_gtex", " "), ("chr", None)]
        for column, value in cases:
            frame = rows()
            frame[column] = frame[column].astype(object)
            frame.loc["source-0", column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                inputs.validate_cohort(frame, TISSUES)

    def test_missing_proteins_are_retained_without_rewriting_the_source_fields(self):
        frame = rows()
        for index, value in zip(frame.index, [None, pd.NA, np.nan, ""]):
            frame.at[index, "Protein"] = value
        before = frame.copy(deep=True)
        cohort = inputs.validate_cohort(frame, TISSUES)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(cohort.rows, before)
        self.assertEqual(cohort.rows["Protein"].map(inputs.protein_present).tolist(),
                         [False] * 4 + [True] * 8)
        self.assertNotIn("noncoding", cohort.rows.columns)

    def test_malformed_nonempty_protein_fields_are_not_treated_as_missing(self):
        for value in [" ", "M\nK", "MK T", 0, False, np.inf, [], {}]:
            frame = rows()
            frame.at["source-0", "Protein"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                inputs.validate_cohort(frame, TISSUES)

    def test_known_raw_eval_label_is_retained_without_becoming_validation(self):
        frame = rows()
        frame.at["source-0", "split"] = "eval"
        cohort = inputs.validate_cohort(frame, TISSUES)
        pd.testing.assert_frame_equal(cohort.rows, frame)

    def test_invalid_tissue_order_and_nonfinite_or_nonnumeric_targets_fail(self):
        for tissues in [(), ("first_tissue", "first_tissue"), "first_tissue", ("split",)]:
            with self.subTest(tissues=tissues), self.assertRaises(ValueError):
                inputs.validate_cohort(rows(), tissues)
        for target in [np.nan, np.inf, "1.2", True, 1 + 2j]:
            frame = rows()
            frame["first_tissue"] = target
            with self.subTest(target=target), self.assertRaises(ValueError):
                inputs.validate_cohort(frame, TISSUES)

    def test_validation_alias_and_absent_test_partition_are_refused(self):
        frame = rows()
        frame.loc["source-0", "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "validation must come from training"):
            inputs.validate_cohort(frame, TISSUES)
        with self.assertRaisesRegex(ValueError, "both train and test"):
            inputs.validate_cohort(rows().assign(split="train"), TISSUES)


class WindowTests(unittest.TestCase):
    def window(self, **changes):
        args = dict(start=11, end=20, strand="+", sequence_length=6,
                    chromosome_length=40, extra_filter_length=None)
        return inputs.released_window(**(args | changes))

    def test_known_even_and_odd_bounds_and_negative_strand_anchor(self):
        plus, minus = self.window(), self.window(strand="-")
        self.assertEqual((plus.anchor, plus.low, plus.high), (10, 7, 13))
        self.assertEqual((minus.anchor, minus.low, minus.high), (19, 16, 22))
        odd = self.window(sequence_length=5, strand="-")
        self.assertEqual((odd.low, odd.high), (17, 22))
        # The release reverses the same anchored slice. Its even reverse window
        # places the anchor at offset 2, unlike the pilot's offset-3 convention.
        self.assertEqual(minus.high - 1 - minus.anchor, 2)

    def test_extraction_reverse_complement_and_standardization(self):
        chromosome = "ACGT" * 10
        self.assertEqual(inputs.extract_released_window(chromosome, self.window()), "TACGTA")
        self.assertEqual(inputs.extract_released_window(chromosome, self.window(strand="-")), "GTACGT")
        chromosome = "AAAAAAAtRyU?c" + "A" * 27
        self.assertEqual(inputs.extract_released_window(chromosome, self.window()), "TNNNNC")
        reverse = self.window(start=1, end=11, strand="-")
        self.assertEqual(inputs.extract_released_window(chromosome, reverse), "GNANNA")

    def test_boundary_exclusion_is_not_clamped_or_silently_corrected(self):
        negative = self.window(start=1)
        exactly_at_end = self.window(start=38, end=38)
        self.assertEqual(negative.exclusion, "released_window_boundary")
        self.assertEqual(exactly_at_end.high, 40)
        self.assertFalse(exactly_at_end.eligible)
        self.assertIsNone(inputs.extract_released_window("A" * 40, exactly_at_end))
        self.assertTrue(self.window(start=37, end=37).eligible)

    def test_extraction_only_scans_the_selected_window(self):
        class Chromosome(str):
            def __iter__(self):
                raise AssertionError("Do not scan the whole chromosome per transcript")

        chromosome = Chromosome("ACGT" * 10)
        self.assertEqual(inputs.extract_released_window(chromosome, self.window()), "TACGTA")
        invalid_window = Chromosome("A" * 7 + "AC GTN" + "A" * 27)
        with self.assertRaisesRegex(ValueError, "DNA window"):
            inputs.extract_released_window(invalid_window, self.window())

    def test_optional_filter_preserves_shifted_source_behavior(self):
        # Window [2:8] fits, and an 8-base filter around anchor 5 would fit [1:9].
        # The released helper instead filters around low=2, giving [-2:6].
        ordinary = self.window(start=6, end=6)
        filtered = self.window(start=6, end=6, extra_filter_length=8)
        self.assertTrue(ordinary.eligible)
        self.assertEqual(filtered.exclusion, "released_extra_filter_boundary")
        with self.assertRaises(TypeError):
            inputs.released_window(start=6, end=6, strand="+", sequence_length=6,
                                   chromosome_length=40)

    def test_invalid_lengths_bounds_and_wrong_chromosome_are_refused(self):
        for changes in [dict(sequence_length=0), dict(sequence_length=1.5), dict(start=True),
                        dict(strand="?"), dict(start=21), dict(extra_filter_length=0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.window(**changes)
        with self.assertRaisesRegex(ValueError, "length does not match"):
            inputs.extract_released_window("A" * 39, self.window())

    def test_rna_suffix_crop_keeps_the_three_prime_end_without_changing_alphabet(self):
        self.assertEqual(inputs.crop_rna_3prime("AAACGU", max_nucleotides=4), "ACGU")
        self.assertEqual(inputs.crop_rna_3prime("ACGU", max_nucleotides=10), "ACGU")
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                inputs.crop_rna_3prime("ACGU", max_nucleotides=value)


class SequencePolicyTests(unittest.TestCase):
    def apply(self, frame, **changes):
        choices = {"rna_max_nucleotides": 12_288, "protein_max_characters": 1_200}
        return inputs.apply_sequence_policies(
            inputs.validate_cohort(frame, TISSUES), **(choices | changes),
        )

    def test_exact_limits_and_suffix_direction_with_preserved_order_and_targets(self):
        frame = rows().iloc[[2, 0, 1, 4, 3, 5, 6, 7, 8, 9, 11, 10]].copy()
        frame.at["source-0", "RNA"] = "A" + "C" * 12_288
        frame.at["source-1", "RNA"] = "G" * 12_288
        frame.at["source-2", "Protein"] = "M" * 1_200
        frame.at["source-3", "Protein"] = "K" * 1_201
        before = frame.copy(deep=True)
        prepared, report = self.apply(frame)

        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(prepared.rows.index.tolist(),
                         [index for index in frame.index if index != "source-3"])
        self.assertEqual(report.index.tolist(), frame.index.tolist())
        expected = frame.drop(index="source-3").copy()
        expected.at["source-0", "RNA"] = "C" * 12_288
        pd.testing.assert_frame_equal(prepared.rows, expected)
        self.assertEqual(prepared.tissues, TISSUES)
        np.testing.assert_array_equal(prepared.labels, expected.loc[:, list(TISSUES)])
        self.assertEqual(report.at["source-0", "rna_removed_5prime_nucleotides"], 1)
        self.assertEqual(report.at["source-1", "rna_removed_5prime_nucleotides"], 0)
        self.assertTrue(report.at["source-2", "sequence_eligible"])
        self.assertTrue(report.at["source-3", "protein_present"])
        self.assertEqual(report.at["source-3", "protein_source_characters"], 1_201)
        self.assertFalse(report.at["source-3", "sequence_eligible"])
        self.assertEqual(report.at["source-3", "sequence_exclusion"],
                         "supplied_protein_exceeds_character_limit")

    def test_absent_proteins_remain_eligible_and_distinct_from_excluded_proteins(self):
        frame = rows()
        for index, value in zip(frame.index, [None, pd.NA, np.nan, ""]):
            frame.at[index, "Protein"] = value
        prepared, report = self.apply(frame)
        pd.testing.assert_frame_equal(prepared.rows, frame)
        self.assertTrue(report["sequence_eligible"].all())
        self.assertTrue(report.loc[frame.index[:4], "protein_source_characters"].isna().all())
        self.assertEqual(report["protein_present"].tolist(), [False] * 4 + [True] * 8)
        self.assertTrue(report["sequence_exclusion"].isna().all())
        # An all-absent group still supplies RNA, targets and the original partitions.
        all_absent = rows().assign(Protein=None)
        prepared, report = self.apply(all_absent)
        pd.testing.assert_frame_equal(prepared.rows, all_absent)
        self.assertFalse(report["protein_present"].any())
        self.assertTrue(report["sequence_eligible"].all())

    def test_unusual_symbols_are_preserved_and_counted_before_tokenization(self):
        frame = rows()
        frame.at["source-0", "RNA"] = "acguN"
        frame.at["source-0", "Protein"] = "M" * 1_199 + "*"
        frame.at["source-1", "Protein"] = "M" * 1_200 + "*"
        prepared, report = self.apply(frame)
        self.assertEqual(prepared.rows.at["source-0", "Protein"], "M" * 1_199 + "*")
        self.assertEqual(prepared.rows.at["source-0", "RNA"], "acguN")
        self.assertEqual(report.at["source-0", "protein_source_characters"], 1_200)
        self.assertEqual(report.at["source-1", "protein_source_characters"], 1_201)
        self.assertNotIn("source-1", prepared.rows.index)

    def test_sequence_rules_do_not_choose_dna_boundaries_or_raw_eval_admission(self):
        frame = rows()
        frame.at["source-0", "split"] = "eval"
        frame.at["source-0", "chr"] = "chrMT"
        frame.at["source-0", "start"] = 1
        prepared, report = self.apply(frame)
        pd.testing.assert_frame_equal(prepared.rows, frame)
        self.assertTrue(report.at["source-0", "sequence_eligible"])
        self.assertEqual(report.at["source-0", "split"], "eval")
        with self.assertRaisesRegex(ValueError, "raw eval rows remain"):
            inputs.validation_partitions(prepared, sampling_unit="transcript", seed=0)

    def test_missing_or_invalid_limits_and_loss_of_required_partition_are_refused(self):
        cohort = inputs.validate_cohort(rows(), TISSUES)
        with self.assertRaises(TypeError):
            inputs.apply_sequence_policies(cohort)
        for key in ("rna_max_nucleotides", "protein_max_characters"):
            for value in (None, 0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.apply(rows(), **{key: value})
        frame = rows()
        frame.loc[frame["split"] == "test", "Protein"] = "M" * 1_201
        before = frame.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "retain both train and test"):
            self.apply(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_mutated_cohort_is_revalidated_and_returned_data_are_independent(self):
        cohort = inputs.validate_cohort(rows(), TISSUES)
        prepared, report = inputs.apply_sequence_policies(
            cohort, rna_max_nucleotides=12_288, protein_max_characters=1_200,
        )
        prepared.rows.at["source-0", "first_tissue"] = -1
        report.at["source-0", "split"] = "changed-report"
        self.assertEqual(cohort.rows.at["source-0", "first_tissue"], 0)
        self.assertEqual(cohort.rows.at["source-0", "split"], "train")
        cohort.rows.at["source-0", "Protein"] = "M K"
        with self.assertRaises(ValueError):
            inputs.apply_sequence_policies(
                cohort, rna_max_nucleotides=12_288, protein_max_characters=1_200,
            )


class ReferenceEligibilityTests(unittest.TestCase):
    def frame(self):
        return rows().assign(start=11, end=20)

    def apply(self, frame, sizes=None, **changes):
        choices = dict(sequence_length=6, extra_filter_length=6,
                       raw_eval_policy="exclude")
        return inputs.apply_reference_eligibility(
            inputs.validate_cohort(frame, TISSUES),
            {"chr1": 40} if sizes is None else sizes, **(choices | changes),
        )

    def test_boundary_arithmetic_and_shifted_filter_are_not_recentered(self):
        frame = self.frame()
        frame.loc["source-0", ["start", "end", "strand"]] = [38, 38, "+"]
        frame.loc["source-1", ["start", "end", "strand"]] = [37, 37, "+"]
        frame.loc["source-2", ["start", "end", "strand"]] = [6, 6, "+"]
        frame.loc["source-3", ["start", "end", "strand"]] = [7, 7, "+"]
        frame.loc["source-4", ["start", "end", "strand"]] = [1, 1, "+"]
        frame.loc["source-5", ["start", "end", "strand"]] = [1, 11, "-"]
        frame.loc["source-6", ["start", "end", "strand"]] = [11, 40, "+"]
        retained, report = self.apply(frame)

        # [34:40] touches the right boundary and is rejected; [33:39] is retained.
        self.assertEqual(report.loc["source-0", "dna_high"], 40)
        self.assertEqual(report.loc["source-0", "reference_exclusion"],
                         "released_window_boundary")
        self.assertTrue(report.loc["source-1", "reference_eligible"])
        # Main window [2:8] fits, but the released shifted filter [-1:5] does not.
        self.assertEqual(report.loc["source-2", ["dna_low", "dna_high"]].tolist(), [2, 8])
        self.assertFalse(report.loc["source-2", "main_window_excluded"])
        self.assertTrue(report.loc["source-2", "extra_filter_excluded"])
        self.assertEqual(report.loc["source-2", "reference_exclusion"],
                         "released_extra_filter_boundary")
        self.assertTrue(report.loc["source-3", "reference_eligible"])
        # Both left-boundary checks fail; flags preserve overlap, reason follows code order.
        self.assertTrue(report.loc["source-4", "main_window_excluded"])
        self.assertTrue(report.loc["source-4", "extra_filter_excluded"])
        self.assertEqual(report.loc["source-4", "reference_exclusion"],
                         "released_extra_filter_boundary")
        # Negative strand anchors at end - 1, positive strand at start - 1.
        for index in ("source-5", "source-6"):
            self.assertEqual(report.loc[index, ["dna_anchor", "dna_low", "dna_high"]].tolist(),
                             [10, 7, 13])
            self.assertTrue(report.loc[index, "reference_eligible"])
        self.assertEqual(retained.rows.index.tolist(),
                         [i for i in frame.index if i not in ("source-0", "source-2", "source-4")])

    def test_raw_eval_and_literal_chrMT_exclusions_need_no_genome_lookup(self):
        frame = self.frame()
        frame.at["source-0", "chr"] = "chrMT"
        frame.loc["source-1", ["split", "chr"]] = ["eval", "not-in-sizes"]
        frame.loc["source-2", ["split", "chr"]] = ["eval", "chrMT"]
        frame.at["source-3", "Protein"] = None
        before = frame.copy(deep=True)
        with patch.object(pd, "read_csv", side_effect=AssertionError("No data read")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("No download")):
            retained, report = self.apply(frame)
        self.assertEqual(report.loc["source-0", "reference_exclusion"], "chrMT")
        self.assertEqual(report.loc["source-1", "reference_exclusion"], "raw_eval_unused")
        self.assertTrue(report.loc["source-2", "raw_eval_excluded"])
        self.assertTrue(report.loc["source-2", "mitochondrial_excluded"])
        self.assertEqual(report.loc["source-2", "reference_exclusion"], "raw_eval_unused")
        self.assertFalse(report.loc[frame.index[:3], "dna_window_checked"].any())
        self.assertTrue(report.loc[frame.index[:3], "dna_low"].isna().all())
        self.assertTrue(report.loc[frame.index[:3], "main_window_excluded"].isna().all())
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(retained.rows, frame.iloc[3:])
        self.assertEqual(report.index.tolist(), frame.index.tolist())
        self.assertEqual(retained.tissues, TISSUES)
        np.testing.assert_array_equal(retained.labels, frame.iloc[3:].loc[:, list(TISSUES)])
        self.assertIsNone(retained.rows.at["source-3", "Protein"])

    def test_missing_chromosome_or_implicit_alias_is_not_silently_dropped(self):
        for chromosome in ("unknown", "chrM", "1"):
            frame = self.frame()
            frame.at["source-0", "chr"] = chromosome
            with self.subTest(chromosome=chromosome), self.assertRaisesRegex(
                    ValueError, "Missing chromosome length.*not silently excluded"):
                self.apply(frame)
        # chrM is not the release's literal chrMT exclusion when explicitly supplied.
        frame = self.frame()
        frame.at["source-0", "chr"] = "chrM"
        retained, report = self.apply(frame, {"chr1": 40, "chrM": 40})
        self.assertIn("source-0", retained.rows.index)
        self.assertFalse(report.at["source-0", "mitochondrial_excluded"])

    def test_lengths_sizes_and_eval_policy_are_explicit(self):
        cohort = inputs.validate_cohort(self.frame(), TISSUES)
        with self.assertRaises(TypeError):
            inputs.apply_reference_eligibility(cohort, {"chr1": 40})
        for changes in (dict(sequence_length=None), dict(sequence_length=True),
                        dict(sequence_length=0), dict(extra_filter_length=None),
                        dict(extra_filter_length=True), dict(extra_filter_length=5),
                        dict(raw_eval_policy=None), dict(raw_eval_policy="train"),
                        dict(raw_eval_policy="validation")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.apply(self.frame(), **changes)
        for sizes in ({"chr1": True}, {"chr1": 0}, {"chr1": 1.5},
                      {"chr1": np.inf}, {" chr1": 40}, {None: 40}, {}):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                self.apply(self.frame(), sizes)
        with self.assertRaises(TypeError):
            self.apply(self.frame(), [("chr1", 40)])

    def test_declared_reference_lengths_use_the_same_strict_windows(self):
        frame = self.frame().assign(start=196_609, end=196_610)
        retained, report = self.apply(frame, {"chr1": 1_000_000},
                                      sequence_length=196_608, extra_filter_length=196_608)
        self.assertEqual(report.loc["source-0", ["dna_low", "dna_high"]].tolist(),
                         [98_304, 294_912])
        self.assertEqual(report.loc["source-1", ["dna_low", "dna_high"]].tolist(),
                         [98_305, 294_913])
        pd.testing.assert_frame_equal(retained.rows, frame)

    def test_sequence_then_reference_policies_preserve_a_common_comparison_cohort(self):
        frame = self.frame()
        frame.at["source-0", "Protein"] = "M" * 1_201
        frame.at["source-1", "Protein"] = None
        frame.at["source-1", "RNA"] = "A" + "C" * 12_288
        frame.at["source-2", "split"] = "eval"
        frame.at["source-3", "chr"] = "chrMT"
        prepared, sequence_report = inputs.apply_sequence_policies(
            inputs.validate_cohort(frame, TISSUES),
            rna_max_nucleotides=12_288, protein_max_characters=1_200,
        )
        eligible, report = inputs.apply_reference_eligibility(
            prepared, {"chr1": 40}, sequence_length=6,
            extra_filter_length=6, raw_eval_policy="exclude",
        )
        self.assertNotIn("source-0", report.index)
        self.assertFalse(sequence_report.loc["source-0", "sequence_eligible"])
        expected = frame.drop(index=["source-0", "source-2", "source-3"]).copy()
        expected.at["source-1", "RNA"] = "C" * 12_288
        pd.testing.assert_frame_equal(eligible.rows, expected)
        partitions = inputs.validation_partitions(
            eligible, sampling_unit="transcript", seed=3, validation_fraction=0.2,
        )
        self.assertEqual(partitions["test"].tolist(), ["source-10", "source-11"])
        self.assertEqual(set(partitions["train"]) | set(partitions["val"]),
                         set(expected.index[expected["split"] == "train"]))
        self.assertNotIn("source-2", {i for part in partitions.values() for i in part})

    def test_loss_of_partition_mutation_and_independent_outputs(self):
        frame = self.frame()
        frame.loc[frame["split"] == "test", "chr"] = "chrMT"
        with self.assertRaisesRegex(ValueError, "retain both train and test"):
            self.apply(frame)
        cohort = inputs.validate_cohort(self.frame(), TISSUES)
        choices = dict(sequence_length=6, extra_filter_length=6, raw_eval_policy="exclude")
        eligible, report = inputs.apply_reference_eligibility(cohort, {"chr1": 40}, **choices)
        eligible.rows.at["source-0", "first_tissue"] = 999
        report.at["source-0", "split"] = "changed-report"
        self.assertEqual(cohort.rows.at["source-0", "first_tissue"], 0)
        self.assertEqual(cohort.rows.at["source-0", "split"], "train")
        cohort.rows.at["source-0", "first_tissue"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            inputs.apply_reference_eligibility(cohort, {"chr1": 40}, **choices)


class PartitionTests(unittest.TestCase):
    def test_partitioning_is_repeatable_and_changes_validation_without_touching_test(self):
        cohort = inputs.validate_cohort(rows(), TISSUES)
        first = inputs.validation_partitions(cohort, sampling_unit="transcript", seed=1,
                                             validation_fraction=0.3)
        again = inputs.validation_partitions(cohort, sampling_unit="transcript", seed=1,
                                             validation_fraction=0.3)
        other = inputs.validation_partitions(cohort, sampling_unit="transcript", seed=2,
                                             validation_fraction=0.3)
        for name in first:
            np.testing.assert_array_equal(first[name], again[name])
        self.assertNotEqual(first["val"].tolist(), other["val"].tolist())
        self.assertEqual(first["test"].tolist(), ["source-10", "source-11"])
        self.assertEqual(other["test"].tolist(), first["test"].tolist())
        combined = [key for part in first.values() for key in part]
        self.assertEqual(set(combined), set(rows().index))
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(len(first["val"]), 3)

    def test_gene_sampling_keeps_training_gene_members_together(self):
        cohort = inputs.validate_cohort(rows(), TISSUES)
        parts = inputs.validation_partitions(cohort, sampling_unit="gene", seed=3,
                                             validation_fraction=0.2)
        self.assertEqual(len(parts["val"]), 2)
        self.assertEqual(inputs.gene_overlap(cohort, parts)["train/val"], ())

    def test_existing_test_gene_overlap_is_reported_not_removed(self):
        frame = rows()
        frame.loc["source-10", "gene_id_gtex"] = "gene-0"
        cohort = inputs.validate_cohort(frame, TISSUES)
        parts = inputs.validation_partitions(cohort, sampling_unit="gene", seed=0,
                                             validation_fraction=0.2)
        overlap = inputs.gene_overlap(cohort, parts)
        self.assertIn("gene-0", overlap["train/test"] + overlap["val/test"])
        self.assertEqual(parts["test"].tolist(), ["source-10", "source-11"])

    def test_invalid_or_implicit_sampling_choices_fail(self):
        cohort = inputs.validate_cohort(rows(), TISSUES)
        for kwargs in [dict(sampling_unit=None, seed=0), dict(sampling_unit="row", seed=0),
                       dict(sampling_unit="gene", seed=-1), dict(sampling_unit="gene", seed=True),
                       dict(sampling_unit="gene", seed=0, validation_fraction=0),
                       dict(sampling_unit="gene", seed=0, validation_fraction=1),
                       dict(sampling_unit="gene", seed=0, validation_fraction=0.99)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inputs.validation_partitions(cohort, **kwargs)
        with self.assertRaises(TypeError):
            inputs.validation_partitions(cohort, seed=0)

    def test_mutated_cohort_and_overlapping_membership_are_rejected(self):
        cohort = inputs.validate_cohort(rows(), TISSUES)
        parts = inputs.validation_partitions(cohort, sampling_unit="transcript", seed=0)
        parts["val"] = parts["test"]
        with self.assertRaisesRegex(ValueError, "every source row exactly once"):
            inputs.gene_overlap(cohort, parts)
        cohort.rows.loc["source-0", "first_tissue"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            inputs.validation_partitions(cohort, sampling_unit="gene", seed=0)


if __name__ == "__main__":
    unittest.main()
