"""Tiny CSV/tokenizer fixtures; no GTEx files, downloaded assets, or models."""

import csv
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import isoformer_cohort as cohort
from shared_code.fusion import isoformer_data_checks as gtex_checks


class RNATokenizer:
    """Independent scanner emitting actual invented tokens, rather than a formula."""

    def __init__(self):
        self.inputs = []

    def __call__(self, text, **kwargs):
        self.inputs.append(text)
        tokens = ["CLS"]
        index = 0
        while index < len(text):
            block = text[index:index + 6]
            if len(block) == 6 and all(c in "ACGT" for c in block):
                tokens.append(block)
                index += 6
            else:
                tokens.append(text[index])
                index += 1
        return {"input_ids": tokens}


class ProteinTokenizer:
    def get_vocab(self):
        return {token: index for index, token in enumerate(["CLS", "EOS", "M", "K", "T"])}

    def __call__(self, text, **kwargs):
        # Deliberately different handling of an unsupported symbol proves fallback.
        tokens = ["CLS"]
        for character in text:
            tokens.extend(["unknown", "unknown"] if character == "*" else [character])
        return {"input_ids": tokens + ["EOS"]}


class CohortAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tiny.csv"
        self.rna = RNATokenizer()
        self.protein = ProteinTokenizer()

    def write(self, rows, fieldnames=None):
        fieldnames = fieldnames or ["transcript_id_gtex", "split", "RNA", "Protein"]
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return self.path.read_bytes()

    def audit(self, **kwargs):
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            return cohort.audit_csv(self.path, self.rna, self.protein, **kwargs)

    def test_counts_limits_suffixes_raw_splits_and_source_are_preserved(self):
        original = self.write([
            dict(transcript_id_gtex="tx1", split="train", RNA="A" * 12288, Protein="M" * 2047),
            dict(transcript_id_gtex="tx2", split="eval", RNA="AAAAANAAAAAA", Protein="MKT"),
            dict(transcript_id_gtex="tx2", split="test", RNA="N", Protein=""),
        ])
        report = self.audit(expected_sha256=hashlib.sha256(original).hexdigest(),
                            expected_bytes=len(original))
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["unique_transcript_ids"], 2)
        self.assertEqual(report["duplicate_transcript_ids"], 1)
        self.assertEqual(report["splits"], {"train": 1, "eval": 1, "test": 1})
        counts = report["scopes"]["all"]["counts"]
        self.assertEqual(counts["RNA_length_gt_12282"], 1)
        self.assertEqual(counts["RNA_length_gt_12288"], 0)
        self.assertEqual(counts["RNA_tokens_gt_2048"], 1)
        self.assertEqual(counts["RNA_after_3prime_12282_tokens_gt_2048"], 0)
        self.assertEqual(counts["RNA_after_3prime_12288_tokens_gt_2048"], 1)
        self.assertEqual(counts["Protein_length_gt_2047"], 0)
        self.assertEqual(counts["Protein_tokens_gt_2048"], 1)
        self.assertEqual(report["scopes"]["all"]["stats"]["RNA_tokens"],
                         {"n": 3, "min": 2, "max": 2049, "sum": 2059})
        self.assertEqual(report["scopes"]["split:eval"]["stats"]["RNA_tokens"]["sum"], 8)
        self.assertEqual(report["missing_sequences"], {"RNA": 0, "Protein": 1})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_missing_fields_are_not_cleaned_or_tokenized_as_empty_sequences(self):
        markers = ["", "nan", "none", "null"]
        rows = [dict(transcript_id_gtex=f"tx{i}", split="train", RNA=value, Protein=value)
                for i, value in enumerate(markers)]
        rows.append(dict(transcript_id_gtex="literal", split="test", RNA=" nan", Protein="M*"))
        original = self.write(rows)
        report = self.audit()
        self.assertEqual(report["missing_sequences"], {"RNA": 4, "Protein": 4})
        stats = report["scopes"]["all"]["stats"]
        self.assertEqual(stats["RNA_length"], {"n": 1, "min": 4, "max": 4, "sum": 4})
        self.assertEqual(stats["RNA_tokens"]["sum"], 5)
        self.assertEqual(stats["Protein_tokens"]["sum"], 5)
        self.assertEqual(report["scopes"]["all"]["counts"]["Protein_tokenizer_fallback"], 1)
        self.assertEqual(report["fallback_rows"], {"RNA": 1, "Protein": 1})
        self.assertEqual(report["alphabets"]["Protein"], {"*": 1, "M": 1})
        self.assertEqual(report["alphabets"]["RNA"][" "], 1)
        self.assertIn(" nan", self.rna.inputs)
        self.assertEqual(self.path.read_bytes(), original)

    def test_uppercase_missing_marker_spelling_remains_a_sequence(self):
        original = self.write([
            dict(transcript_id_gtex="tx1", split="train", RNA="NAN", Protein="NAN"),
            dict(transcript_id_gtex="tx2", split="train", RNA="A", Protein="NONE"),
            dict(transcript_id_gtex="tx3", split="test", RNA="A", Protein="NULL"),
        ])
        report = self.audit()
        self.assertEqual(report["missing_sequences"], {"RNA": 0, "Protein": 0})
        self.assertEqual(report["scopes"]["all"]["stats"]["Protein_length"],
                         {"n": 3, "min": 3, "max": 4, "sum": 11})
        self.assertEqual(report["scopes"]["split:train"]["stats"]["RNA_tokens"]["sum"], 6)
        self.assertEqual(cohort.rna_fast_token_count("NAN"), 4)
        self.assertEqual(self.path.read_bytes(), original)
        for spelling in ("NAN", "NONE", "NULL", "NaN", "None", " nan"):
            with self.subTest(spelling=spelling):
                self.assertFalse(cohort.is_missing_sequence(spelling))

    def test_periodic_actual_tokenizer_checks_detect_unusual_wrong_fast_count(self):
        self.write([dict(transcript_id_gtex=f"tx{i}", split="train", RNA="ACGTAC", Protein="M")
                    for i in range(194)])
        report = self.audit()
        self.assertEqual(report["token_count_validation"]["actual_row_checks"], 3)
        self.assertEqual(report["token_count_validation"]["actual_sequence_checks"], 15)

        class WrongOnActualRow(RNATokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if text == "ACGTAC":
                    result["input_ids"].append("wrong")
                return result

        self.rna = WrongOnActualRow()
        with self.assertRaisesRegex(ValueError, "count mismatch at CSV row 1"):
            self.audit()

    def test_tokenizer_contract_failure_is_explicit(self):
        class ExtraEOS(RNATokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result["input_ids"].append("EOS")
                return result

        with self.assertRaisesRegex(ValueError, "RNA tokenizer count contract failed"):
            cohort.validate_tokenizers(ExtraEOS(), self.protein)

    def test_long_csv_field_is_measured_without_leaving_global_parser_changes(self):
        self.write([dict(transcript_id_gtex="long", split="eval", RNA="A" * 140000, Protein="M")])
        old_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(1024)
            report = self.audit()
            self.assertEqual(report["scopes"]["all"]["stats"]["RNA_length"]["max"], 140000)
            self.assertEqual(csv.field_size_limit(), 1024)
        finally:
            csv.field_size_limit(old_limit)

    def test_bad_source_identity_refuses_before_tokenization(self):
        original = self.write([dict(transcript_id_gtex="tx", split="train", RNA="A", Protein="M")])
        for kwargs in ({"expected_sha256": "0" * 64}, {"expected_bytes": len(original) + 1}):
            with patch.object(cohort, "validate_tokenizers", side_effect=AssertionError("Too early")):
                with self.assertRaises(ValueError):
                    self.audit(**kwargs)
        self.assertEqual(self.path.read_bytes(), original)

    def test_boundary_scenario_retains_raw_cohort_and_respects_negative_strand(self):
        coordinates = [
            ("chr1", 300001, 300010, "+"),  # retained
            ("chr1", 1, 300001, "-"),       # anchor is end, not start
            ("chrMT", 300001, 300010, "+"),
            ("chr1", 98304, 98310, "+"),    # main left edge negative
            ("chr1", 98305, 98310, "+"),    # main fits; extra filter does not
            ("chr1", 901697, 901700, "+"),  # main end exactly chromosome size
            ("chr1", 196609, 196610, "+"),  # extra left edge exactly zero
        ]
        rows = [dict(transcript_id_gtex=f"tx{i}", split="train", RNA="A", Protein="M",
                     chr=chromosome, start=start, end=end, strand=strand)
                for i, (chromosome, start, end, strand) in enumerate(coordinates)]
        original = self.write(rows, fieldnames=list(rows[0]))
        report = self.audit(chromosome_sizes={"chr1": 1_000_000})
        self.assertEqual(report["row_count"], 7)
        self.assertEqual(report["example_boundary_policy"]["exclusions"],
                         {"chrMT": 1, "main_boundary": 2, "extra_shifted_boundary": 1})
        self.assertEqual(report["example_boundary_policy"]["retained_rows"], 3)
        self.assertEqual(report["example_boundary_exclusions"],
                         {"chrMT": 1, "main_window": 2, "additional_shifted_filter": 1})
        self.assertEqual(report["scopes"]["example_boundary_policy"]["rows"], 3)
        self.assertEqual(report["scopes"]["example_boundary_policy:train"]["rows"], 3)
        self.assertEqual(report["scopes"]["split:train"]["rows"], 7)
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "No chromosome size"):
            self.audit(chromosome_sizes={})

    def test_empty_file_bad_headers_and_malformed_row_fail(self):
        for contents in ("", "RNA,Protein,split\nA,M,train\n",
                         "transcript_id_gtex,split,RNA,Protein,RNA\nt,train,A,M,A\n",
                         "transcript_id_gtex,split,RNA,Protein\nt,train,A\n"):
            self.path.write_text(contents, encoding="utf-8")
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                self.audit()


class ExtendedDataChecksTests(unittest.TestCase):
    """Hand-counted annotations, identities, targets and policy scenarios."""

    setUp = CohortAuditTests.setUp
    write = CohortAuditTests.write
    audit = CohortAuditTests.audit

    def row(self, transcript="t1", gene="g1", split="train", rna="AAAAAA",
            protein="MKT", annotation="protein_coding", protein_id="p1",
            cds="ATGAAAACA", **changes):
        row = {
            "transcript_id_gtex": transcript, "gene_id_gtex": gene, "split": split,
            "RNA": rna, "Protein": protein, "RNA_type_gff": annotation,
            "transcript_id_ensembl_protein": protein_id, "CDS": cds,
            "chr": "chr1", "strand": "+", "start": 300001, "end": 300010,
        }
        row.update({tissue: str(index + 1) for index, tissue in enumerate(cohort.GTEX_TISSUES)})
        row.update(changes)
        return row

    def measure(self, rows, fieldnames=None, **kwargs):
        self.write(rows, fieldnames=fieldnames or list(rows[0]))
        return self.audit(include_data_checks=True, **kwargs)

    def complete(self, report):
        report.update({
            "started_utc": "2026-11-09T10:00:00+00:00",
            "finished_utc": "2026-11-09T10:00:01+00:00",
            "dataset_revision": "a" * 40, "model_revision": "b" * 40,
            "vocabulary_sha256": {"rna": "c" * 64, "protein": "d" * 64},
            "chromosome_sizes_sha256": "e" * 64,
            "transformers": "fixture-version", "python": "fixture-python", "host": "fixture-host",
        })
        report["audit_identity"] = {
            key: report[key] for key in (
                "source_sha256", "source_bytes", "dataset_revision", "model_revision",
                "vocabulary_sha256", "chromosome_sizes_sha256", "transformers",
            )
        }
        report["audit_identity"].update(
            method_version=cohort.METHOD_VERSION, data_checks_version=cohort.DATA_CHECKS_VERSION,
        )
        return report

    def test_annotations_report_field_availability_without_selecting_coding_rows(self):
        rows = [
            self.row(),
            self.row("t2", "g2", protein="", protein_id="p2", cds="ATG"),
            self.row("t3", "g3", annotation="lncRNA", protein="", protein_id="", cds=""),
            self.row("t4", "g4", annotation="retained_intron", protein="M", protein_id="", cds=""),
            self.row("t5", "g5", annotation="", protein="", protein_id="", cds=""),
        ]
        original = self.write(rows, fieldnames=list(rows[0]))
        report = self.audit(include_data_checks=True)
        checks = report["data_checks"]
        groups = checks["annotations"]["by_type"]
        self.assertEqual(checks["row_count"], 5)
        self.assertEqual(report["row_count"], 5)
        self.assertEqual(set(groups), {"protein_coding", "lncRNA", "retained_intron", ""})
        coding = groups["protein_coding"]
        self.assertEqual((coding["rows"], coding["protein_present"], coding["protein_missing"]), (2, 1, 1))
        self.assertEqual(coding["missing_protein_with_id"], 1)
        self.assertEqual(coding["missing_protein_with_cds"], 1)
        self.assertEqual(groups["lncRNA"]["protein_missing"], 1)
        self.assertEqual(groups["lncRNA"]["protein_id_missing"], 1)
        self.assertEqual(groups["retained_intron"]["protein_present"], 1)
        self.assertEqual(groups["retained_intron"]["cds_missing"], 1)
        self.assertEqual(sum(group["rows"] for group in groups.values()), report["row_count"])
        self.assertEqual(self.path.read_bytes(), original)

    def test_conflicting_mappings_are_reported_without_treating_gene_isoforms_as_conflicts(self):
        rows = [
            self.row("t1", "g1", rna="AAAAAA", protein="MKT", protein_id="p1"),
            self.row("t1", "g2", split="test", rna="CCCCCC", protein="M", protein_id="p1",
                     chr="chr2", strand="-"),
            self.row("t2", "g1", split="eval", rna="GGGGGG", protein="", protein_id="",
                     chr="chr2", strand="-"),
            self.row("", "", rna="", protein="", protein_id="", cds=""),
            self.row("", "", split="test", rna="nan", protein="null", protein_id="", cds=""),
        ]
        checks = self.measure(rows)["data_checks"]["mappings"]
        self.assertEqual(checks["missing_gene_ids"], 2)
        self.assertEqual(checks["missing_transcript_ids"], 2)
        for key in (
            "transcripts_with_multiple_gene_ids", "transcripts_with_multiple_rna_sequences",
            "transcripts_with_multiple_protein_sequences", "genes_with_multiple_chromosomes",
            "genes_with_multiple_strands", "protein_ids_with_multiple_sequences",
        ):
            with self.subTest(key=key):
                self.assertEqual(checks[key], 1)
        self.assertEqual(checks["examples"]["transcripts_with_multiple_gene_ids"], ["t1"])
        self.assertEqual(checks["examples"]["protein_ids_with_multiple_sequences"], ["p1"])

        consistent_isoforms = self.measure([
            self.row("t1", "g1", rna="AAAAAA", protein="M", protein_id="p1"),
            self.row("t2", "g1", rna="CCCCCC", protein="K", protein_id="p2"),
        ])["data_checks"]["mappings"]
        self.assertEqual(consistent_isoforms["transcripts_with_multiple_gene_ids"], 0)
        self.assertEqual(consistent_isoforms["genes_with_multiple_chromosomes"], 0)
        self.assertEqual(consistent_isoforms["genes_with_multiple_strands"], 0)

    def test_exact_split_overlap_counts_exclude_missing_values_and_keep_row_multiplicity(self):
        rows = [
            self.row("t1", "g1", split="train", rna="A", protein="M"),
            self.row("t2", "g2", split="test", rna="A", protein="M"),
            self.row("t3", "g1", split="eval", rna="C", protein="K"),
            self.row("t1", "g1", split="test", rna="C", protein="M"),
            self.row("", "", split="train", rna="", protein="", protein_id="", cds=""),
            self.row("", "", split="test", rna="nan", protein="null", protein_id="", cds=""),
        ]
        report = self.measure(rows)
        overlap = report["data_checks"]["overlaps"]
        for label in ("gene_id", "transcript_id", "RNA", "Protein"):
            self.assertEqual(overlap[label]["missing_rows"], 2)
        self.assertEqual(overlap["gene_id"]["unique_values"], 2)
        self.assertEqual(overlap["transcript_id"]["unique_values"], 3)
        self.assertEqual(overlap["RNA"]["unique_values"], 2)
        self.assertEqual(overlap["Protein"]["unique_values"], 2)
        for label in ("gene_id", "transcript_id", "RNA", "Protein"):
            pair = next(p for p in overlap[label]["pairs"] if {p["split_a"], p["split_b"]} == {"train", "test"})
            self.assertEqual(pair["shared_unique_values"], 1)
            row_counts = {pair["split_a"]: pair["rows_a"], pair["split_b"]: pair["rows_b"]}
            self.assertEqual(row_counts, {"train": 1, "test": 2 if label == "Protein" else 1})
        self.assertEqual(report["row_count"], 6)
        self.assertEqual(report["splits"], {"train": 2, "test": 3, "eval": 1})

    def test_target_diagnostics_keep_declared_order_and_distinguish_invalid_categories(self):
        tissues = list(cohort.GTEX_TISSUES)
        self.assertEqual(len(tissues), 30)
        self.assertEqual(len(set(tissues)), 30)
        invalid = ("", " ", "NaN", "inf", "oops")
        varying = ("1", "1", "-1", "0", "1")
        rows = [self.row(f"t{i}", f"g{i}", **{
            tissues[0]: invalid[i], tissues[1]: varying[i], tissues[2]: "2",
        }) for i in range(5)]
        for row in rows:
            del row[tissues[3]]
        metadata = [key for key in rows[0] if key not in tissues]
        observed_order = [name for name in reversed(tissues) if name != tissues[3]]
        report = self.measure(rows, fieldnames=metadata + observed_order)
        targets = report["data_checks"]["targets"]
        self.assertEqual(targets["expected_order"], tissues)
        self.assertEqual(targets["observed_order"], observed_order)
        self.assertEqual(targets["missing_columns"], [tissues[3]])
        self.assertFalse(targets["order_matches"])
        self.assertFalse(targets["columns"][tissues[3]]["present"])
        bad = targets["columns"][tissues[0]]
        self.assertEqual((bad["rows"], bad["missing"], bad["nonfinite"], bad["nonnumeric"], bad["finite"]),
                         (5, 2, 2, 1, 0))
        self.assertIsNone(bad["min"])
        self.assertIsNone(bad["variance"])
        values = targets["columns"][tissues[1]]
        self.assertEqual((values["finite"], values["negative"], values["zero"]), (5, 1, 1))
        self.assertAlmostEqual(values["mean"], 0.4)
        self.assertAlmostEqual(values["variance"], 0.64)
        constant = targets["columns"][tissues[2]]
        self.assertEqual((constant["finite"], constant["mean"], constant["variance"]), (5, 2, 0))
        self.assertEqual(report["row_count"], 5)  # Invalid targets are diagnosed, not removed.

    def test_policy_groups_keep_overlapping_reasons_and_protein_availability_denominators(self):
        rows = [
            self.row("long", "g1", rna="A" * 12288, protein="M" * 2049),
            self.row("missing", "g2", protein="", protein_id="", cds=""),
            self.row("boundary", "g3", protein="M", start=1, end=10),
            self.row("medium", "g4", protein="M" * 1201),
        ]
        checks = self.measure(rows, chromosome_sizes={"chr1": 1_000_000})["data_checks"]
        groups = checks["policy_impacts"]["groups"]
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual((group["split"], group["annotation"], group["rows"]), ("train", "protein_coding", 4))
        self.assertEqual((group["rna_present"], group["protein_present"], group["rna_and_protein_present"]), (4, 3, 3))
        self.assertEqual((group["dna_eligible"], group["all_three_present_after_boundary"]), (3, 2))
        reasons = group["reasons"]
        self.assertEqual(reasons["protein_missing"], 1)
        self.assertEqual(reasons["protein_length_gt_1200"], 2)
        self.assertEqual(reasons["protein_length_gt_2048"], 1)
        self.assertEqual(reasons["rna_tokens_gt_2048"], 1)
        self.assertEqual(reasons["dna_boundary_excluded"], 1)
        self.assertEqual(group["overlapping_reason_pairs"]["protein_length_gt_1200 & protein_length_gt_2048"], 1)
        self.assertEqual(group["overlapping_reason_pairs"]["protein_id_missing & protein_missing"], 1)
        self.assertEqual(group["example_ids"]["protein_length_gt_2048"], ["long"])
        self.assertEqual(group["example_ids"]["protein_missing"], ["missing"])
        representatives = checks["representative_rows"]
        self.assertEqual(representatives["longest_RNA"]["transcript_id"], "long")
        self.assertEqual(representatives["longest_Protein"]["Protein_length"], 2049)
        self.assertTrue(all("RNA" not in row and "Protein" not in row
                            for row in representatives["first_rows"]))

    def test_distributions_use_inclusive_upper_edges_and_nonmissing_denominators(self):
        rna_lengths = (200, 201, 500, 501, 12000, 12001, 0)
        protein_lengths = (100, 101, 300, 301, 1200, 1201, 0)
        rows = [self.row(f"t{i}", f"g{i}", rna="A" * n, protein="M" * m)
                for i, (n, m) in enumerate(zip(rna_lengths, protein_lengths))]
        distributions = self.measure(rows)["data_checks"]["length_distributions"]
        for modality in ("RNA", "Protein"):
            distribution = distributions[modality]
            self.assertEqual((distribution["present"], distribution["missing"]), (6, 1))
            self.assertEqual(sum(distribution["counts"]), 6)
            self.assertEqual(distribution["counts"][:3], [1, 2, 1])
            self.assertEqual(distribution["counts"][5:7], [1, 1])
            self.assertIsNone(distribution["upper_bounds"][-1])

    def test_examples_are_bounded_and_required_annotation_fields_are_not_invented(self):
        rows = [self.row(f"t{i}", f"g{i}", protein="", protein_id="", cds="") for i in range(7)]
        checks = self.measure(rows)["data_checks"]
        self.assertEqual(len(checks["representative_rows"]["first_rows"]), 5)
        self.assertEqual(len(checks["representative_rows"]["missing_protein_examples"]), 5)
        group = checks["policy_impacts"]["groups"][0]
        self.assertEqual(group["reasons"]["protein_missing"], 7)
        self.assertEqual(len(group["example_ids"]["protein_missing"]), 5)
        for row in rows:
            del row["RNA_type_gff"]
        self.write(rows, fieldnames=list(rows[0]))
        with self.assertRaises(ValueError):
            self.audit(include_data_checks=True)

    def test_extended_report_roundtrip_and_inconsistent_sections_refuse_publication(self):
        report = self.complete(self.measure([
            self.row(), self.row("t2", "g2", split="test", protein="", protein_id="", cds=""),
        ]))
        self.assertTrue(report["data_checks"]["targets"]["order_matches"])
        destination = self.path.parent / "extended.json"
        self.assertTrue(cohort.save_audit(destination, report))
        self.assertEqual(cohort.read_audit(destination, {
            "data_checks_version": cohort.DATA_CHECKS_VERSION,
        }), report)
        original = destination.read_bytes()
        changes = []
        for mutation in ("missing_section", "missing_identity_version", "wrong_version",
                         "annotation_rows", "target_rows", "distribution"):
            altered = json.loads(json.dumps(report))
            if mutation == "missing_section":
                del altered["data_checks"]
            elif mutation == "missing_identity_version":
                del altered["audit_identity"]["data_checks_version"]
            elif mutation == "wrong_version":
                altered["data_checks"]["version"] = "another-version"
            elif mutation == "annotation_rows":
                altered["data_checks"]["annotations"]["by_type"]["protein_coding"]["rows"] += 1
            elif mutation == "target_rows":
                altered["data_checks"]["targets"]["columns"][cohort.GTEX_TISSUES[0]]["finite"] += 1
            elif mutation == "distribution":
                altered["data_checks"]["length_distributions"]["Protein"]["counts"][0] += 1
            changes.append((mutation, altered))
        for mutation, altered in changes:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cohort.save_audit(self.path.parent / f"bad-{mutation}.json", altered)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(sorted(path.name for path in self.path.parent.glob("*.json")), ["extended.json"])


class AuditStorageTests(unittest.TestCase):
    """Completed invented reports must survive reuse, conflicts, and failed writes."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.path = self.folder / "audit.json"
        source = self.folder / "tiny.csv"
        source.write_text("transcript_id_gtex,split,RNA,Protein\n"
                          "t1,train,ACGTAC,M\nt2,test,N,\n", encoding="utf-8")
        self.report = cohort.audit_csv(source, RNATokenizer(), ProteinTokenizer())
        self.report.update({
            "started_utc": "2026-11-09T08:00:00+00:00",
            "finished_utc": "2026-11-09T08:00:01+00:00",
            "dataset_revision": "a" * 40, "model_revision": "b" * 40,
            "vocabulary_sha256": {"rna": "c" * 64, "protein": "d" * 64},
            "chromosome_sizes_sha256": "e" * 64,
            "transformers": "fixture-version", "python": "fixture-python",
            "host": "fixture-host", "source_url": "https://example.invalid/source",
            "implementation_sha256": "f" * 64,
        })
        self.report["audit_identity"] = {
            key: self.report[key] for key in (
                "source_sha256", "source_bytes", "dataset_revision", "model_revision",
                "vocabulary_sha256", "chromosome_sizes_sha256", "transformers",
            )
        }
        self.report["audit_identity"]["method_version"] = cohort.METHOD_VERSION

    def copy_report(self):
        return json.loads(json.dumps(self.report))

    def test_publish_and_subset_read_then_reuse_preserves_original_bytes_and_dates(self):
        original = self.copy_report()
        self.assertTrue(cohort.save_audit(self.path, self.report))
        saved = self.path.read_bytes()
        modified_time = self.path.stat().st_mtime_ns
        self.assertEqual(cohort.read_audit(self.path, {"method_version": cohort.METHOD_VERSION}), original)
        later = self.copy_report()
        later.update(started_utc="2026-11-10T08:00:00+00:00", finished_utc="2026-11-10T08:00:01+00:00",
                     python="other-python", host="other-host", source_url="other-source",
                     implementation_sha256="0" * 64)
        self.assertFalse(cohort.save_audit(self.path, later))
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(self.path.stat().st_mtime_ns, modified_time)
        self.assertEqual(self.report, original)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["audit.json", "tiny.csv"])

    def test_identity_and_unknown_measurement_changes_cannot_replace_existing_report(self):
        cohort.save_audit(self.path, self.report)
        original = self.path.read_bytes()
        changes = []
        version = self.copy_report()
        version["transformers"] = version["audit_identity"]["transformers"] = "another-version"
        changes.append(version)
        method = self.copy_report()
        method["audit_identity"]["method_version"] = "another-method"
        changes.append(method)
        unknown = self.copy_report()
        unknown["new_measurement"] = {"count": 7}
        changes.append(unknown)
        for changed in changes:
            with self.subTest(changed=changed["audit_identity"]):
                with self.assertRaisesRegex(FileExistsError, "RESULT_PATH"):
                    cohort.save_audit(self.path, changed)
                self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "requested identity"):
            cohort.read_audit(self.path, {"method_version": "wrong"})

    def test_incomplete_counts_and_metadata_are_rejected_before_publication(self):
        changes = []
        bad_total = self.copy_report()
        bad_total["splits"]["train"] = 100
        changes.append(bad_total)
        bad_missing = self.copy_report()
        bad_missing["missing_sequences"]["Protein"] = 0
        changes.append(bad_missing)
        bad_suffix = self.copy_report()
        del bad_suffix["scopes"]["all"]["stats"]["RNA_after_3prime_12000_tokens"]
        changes.append(bad_suffix)
        bad_identity = self.copy_report()
        bad_identity["audit_identity"]["source_sha256"] = "0" * 64
        changes.append(bad_identity)
        bad_dates = self.copy_report()
        bad_dates["finished_utc"] = "2026-11-08T08:00:00+00:00"
        changes.append(bad_dates)
        unfinished = self.copy_report()
        del unfinished["finished_utc"]
        changes.append(unfinished)
        for changed in changes:
            with self.assertRaises(ValueError):
                cohort.save_audit(self.path, changed)
            self.assertFalse(self.path.exists())
        self.assertEqual([p.name for p in self.folder.iterdir()], ["tiny.csv"])

    def test_legacy_read_requires_opt_in_and_cannot_be_rewritten_as_modern(self):
        legacy = self.copy_report()
        del legacy["audit_identity"]
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "allow_legacy"):
            cohort.read_audit(self.path)
        expected = {"source_sha256": legacy["source_sha256"], "model_revision": legacy["model_revision"]}
        self.assertEqual(cohort.read_audit(self.path, expected, allow_legacy=True), legacy)
        with self.assertRaises(FileExistsError):
            cohort.save_audit(self.path, self.report)
        with self.assertRaises(ValueError):
            cohort.save_audit(self.path, legacy)
        self.assertEqual(self.path.read_bytes(), original)

    def test_failed_publication_cleans_temporary_file_and_does_not_create_result(self):
        with patch.object(cohort.os, "link", side_effect=OSError("hard links unavailable")):
            with self.assertRaisesRegex(OSError, "hard links unavailable"):
                cohort.save_audit(self.path, self.report)
        self.assertFalse(self.path.exists())
        self.assertEqual([p.name for p in self.folder.iterdir()], ["tiny.csv"])

    def test_destination_race_keeps_competing_result(self):
        competing = self.copy_report()
        competing["other_measurement"] = 1
        competing_bytes = json.dumps(competing).encode("utf-8")

        def race(source, destination):
            destination.write_bytes(competing_bytes)
            raise FileExistsError("competing writer won")

        with patch.object(cohort.os, "link", side_effect=race):
            with self.assertRaisesRegex(FileExistsError, "RESULT_PATH"):
                cohort.save_audit(self.path, self.report)
        self.assertEqual(self.path.read_bytes(), competing_bytes)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["audit.json", "tiny.csv"])


class NotebookAuditPathTests(unittest.TestCase):
    """Exercise isolated workflow cells with invented inputs and no network."""

    def setUp(self):
        AuditStorageTests.setUp(self)  # Reuse the completed synthetic-report fixture.
        notebook = (Path(__file__).resolve().parents[1] / "Code" / "stage3"
                    / "isoformer_gtex_audit.ipynb")
        self.cells = {
            cell["id"]: "".join(cell["source"])
            for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"
        }
        identity = {key: value for key, value in self.report["audit_identity"].items()
                    if key != "transformers"}
        self.namespace = {
            "Path": Path, "hashlib": hashlib, "importlib": importlib,
            "urllib": urllib, "json": json,
            "tempfile": tempfile, "datetime": datetime, "timezone": timezone, "sys": sys,
            "platform": SimpleNamespace(node=lambda: "fixture-host"),
            "cohort": cohort, "gtex_checks": gtex_checks, "SOURCE_IDENTITY": identity,
            "PREVIOUS_IDENTITY": dict(identity),
            "LEGACY_IDENTITY": {key: value for key, value in identity.items()
                                if key != "method_version"},
            "AUDIT_MODE": "read_results", "RESULT_PATH": self.path,
            "PREVIOUS_RECORD": self.folder / "previous.json",
            "HISTORICAL_RECORD": self.folder / "historical.json",
            "CSV_PATH": self.folder / "tiny.csv",
            "CSV_URL": "https://example.invalid/source",
            "CSV_SHA256": self.report["source_sha256"],
            "CSV_BYTES": self.report["source_bytes"],
            "CHROMOSOME_URL": "https://example.invalid/chromosomes",
            "CHROMOSOME_SHA256": self.report["chromosome_sizes_sha256"],
            "DATASET_REVISION": self.report["dataset_revision"],
            "MODEL_REVISION": self.report["model_revision"],
            "VOCAB_SHA256": self.report["vocabulary_sha256"],
        }

    def define(self, cell_id):
        exec(compile(self.cells[cell_id], cell_id, "exec"), self.namespace)

    def select(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            with redirect_stdout(io.StringIO()):
                self.define("select-audit")

    def test_setup_refreshes_helper_already_loaded_without_storage_api(self):
        version = cohort.METHOD_VERSION
        # Simulate the older helper retained by an already-open notebook kernel.
        with patch.object(cohort, "METHOD_VERSION", None), \
                patch.object(cohort, "read_audit", None), \
                patch.object(cohort, "save_audit", None), \
                patch.object(gtex_checks, "DATA_CHECKS_VERSION", None):
            del cohort.METHOD_VERSION, cohort.read_audit, cohort.save_audit
            del gtex_checks.DATA_CHECKS_VERSION
            with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
                self.define("project-paths")
                self.define("source-identity")
            self.assertEqual(self.namespace["SOURCE_IDENTITY"]["method_version"], version)
            self.assertEqual(self.namespace["SOURCE_IDENTITY"]["data_checks_version"],
                             cohort.DATA_CHECKS_VERSION)
            self.assertEqual(cohort.DATA_CHECKS_VERSION, gtex_checks.DATA_CHECKS_VERSION)
            self.assertTrue(gtex_checks.DATA_CHECKS_VERSION)
            self.assertTrue(callable(self.namespace["cohort"].read_audit))
            self.assertTrue(callable(self.namespace["cohort"].save_audit))

    def test_verified_download_checks_identity_without_hf_cache(self):
        self.define("verified-download")
        content = b"invented source"
        destination = self.folder / "source.txt"
        digest = hashlib.sha256(content).hexdigest()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(content)):
            actual = self.namespace["download_verified"](
                "https://example.invalid/source", destination, digest, len(content)
            )
        self.assertEqual(actual.read_bytes(), content)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(content)):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.namespace["download_verified"](
                    "https://example.invalid/source", destination, "wrong-sha"
                )

    def test_read_results_prefers_named_report_without_network_or_measurement(self):
        cohort.save_audit(self.path, self.report)
        historical = dict(self.report, host="historical-host")
        del historical["audit_identity"]
        self.namespace["HISTORICAL_RECORD"].write_text(json.dumps(historical), encoding="utf-8")
        fresh = Mock(side_effect=AssertionError("No fresh audit"))
        self.namespace["repeat_audit"] = fresh
        original = self.path.read_bytes()
        self.select()
        fresh.assert_not_called()
        self.assertEqual(self.namespace["selected_path"], self.path)
        self.assertEqual(self.namespace["audit"], self.report)
        self.assertEqual(self.path.read_bytes(), original)

    def test_previous_notebook_result_precedes_historical_without_starting_new_checks(self):
        previous_path = self.namespace["PREVIOUS_RECORD"]
        cohort.save_audit(previous_path, self.report)
        original = previous_path.read_bytes()
        historical = dict(self.report)
        del historical["audit_identity"]
        self.namespace["HISTORICAL_RECORD"].write_text(json.dumps(historical), encoding="utf-8")
        self.namespace["repeat_audit"] = Mock(side_effect=AssertionError("No new audit"))
        self.select()
        self.assertEqual(self.namespace["selected_path"], previous_path)
        self.assertEqual(self.namespace["audit"], self.report)
        self.assertNotIn("data_checks", self.namespace["audit"])
        self.namespace["repeat_audit"].assert_not_called()
        self.assertEqual(previous_path.read_bytes(), original)
        self.assertFalse(self.path.exists())

    def test_invalid_previous_record_does_not_fall_back_to_valid_historical_record(self):
        previous_path = self.namespace["PREVIOUS_RECORD"]
        previous_path.write_text("{}", encoding="utf-8")
        historical = dict(self.report)
        del historical["audit_identity"]
        self.namespace["HISTORICAL_RECORD"].write_text(json.dumps(historical), encoding="utf-8")
        self.namespace["repeat_audit"] = Mock(side_effect=AssertionError("No audit"))
        with self.assertRaises(ValueError):
            self.select()
        self.assertEqual(previous_path.read_text(encoding="utf-8"), "{}")
        self.assertFalse(self.path.exists())
        self.namespace["repeat_audit"].assert_not_called()

    def test_historical_fallback_is_explicit_and_creates_no_new_result(self):
        historical = dict(self.report)
        del historical["audit_identity"]
        path = self.namespace["HISTORICAL_RECORD"]
        path.write_text(json.dumps(historical), encoding="utf-8")
        original = path.read_bytes()
        self.select()
        self.assertEqual(self.namespace["evidence_kind"], "Historical reference")
        self.assertFalse(self.path.exists())
        self.assertEqual(path.read_bytes(), original)

    def test_legacy_record_skips_unmeasured_extended_displays_without_computation(self):
        self.namespace.update(audit=self.report, repeat_audit=Mock(
            side_effect=AssertionError("No audit from a display cell")))
        forbidden_display = Mock(side_effect=AssertionError("No table or plot for absent checks"))
        self.namespace.update(display=forbidden_display, pd=forbidden_display,
                              plt=forbidden_display, np=forbidden_display)
        display_cells = (
            "data-check-status", "row-examples", "target-table", "target-range-plot",
            "annotation-table", "annotation-plot", "mapping-checks", "overlap-table",
            "overlap-plot", "length-distribution-plot", "longest-input-examples",
            "policy-table", "policy-split-rates", "policy-by-split-plot", "policy-overlap-examples",
        )
        output = io.StringIO()
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            with redirect_stdout(output):
                for cell_id in display_cells:
                    self.define(cell_id)
        self.assertIsNone(self.namespace["data_checks"])
        self.assertIn("unmeasured, not zero", output.getvalue())
        self.namespace["repeat_audit"].assert_not_called()
        forbidden_display.assert_not_called()
        self.assertFalse(list(self.folder.glob("*.json")))

    def test_missing_or_invalid_results_never_trigger_an_audit(self):
        self.namespace["repeat_audit"] = Mock(side_effect=AssertionError("No audit"))
        with self.assertRaisesRegex(FileNotFoundError, "run_audit"):
            self.select()
        historical = dict(self.report)
        del historical["audit_identity"]
        self.namespace["HISTORICAL_RECORD"].write_text(json.dumps(historical), encoding="utf-8")
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.select()
        self.namespace["repeat_audit"].assert_not_called()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{}")

    def test_invalid_mode_fails_before_reading_or_downloading(self):
        self.namespace["AUDIT_MODE"] = "typo"
        with self.assertRaisesRegex(ValueError, "AUDIT_MODE"):
            self.select()

    def test_run_audit_saves_once_and_identical_repeat_preserves_file(self):
        self.namespace.update(AUDIT_MODE="run_audit", repeat_audit=Mock(return_value=self.report))
        self.select()
        original = self.path.read_bytes()
        original_time = self.path.stat().st_mtime_ns
        later = dict(self.report, started_utc="2026-11-10T08:00:00+00:00",
                     finished_utc="2026-11-10T08:00:01+00:00")
        self.namespace["repeat_audit"].return_value = later
        self.select()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.path.stat().st_mtime_ns, original_time)
        self.assertEqual(self.namespace["audit"]["started_utc"], self.report["started_utc"])
        self.assertEqual(sorted(p.name for p in self.folder.glob("*.json")), ["audit.json"])

    def test_failed_or_different_run_preserves_existing_result(self):
        cohort.save_audit(self.path, self.report)
        original = self.path.read_bytes()
        self.namespace.update(AUDIT_MODE="run_audit",
                              repeat_audit=Mock(side_effect=ValueError("audit failed")))
        with self.assertRaisesRegex(ValueError, "audit failed"):
            self.select()
        self.namespace["repeat_audit"] = Mock(return_value=dict(self.report, extra_measurement=1))
        with self.assertRaises(FileExistsError):
            self.select()
        self.assertEqual(self.path.read_bytes(), original)

    def test_historical_output_destination_is_refused_before_measurement(self):
        for name in ("HISTORICAL_RECORD", "PREVIOUS_RECORD"):
            with self.subTest(destination=name):
                self.namespace.update(AUDIT_MODE="run_audit", RESULT_PATH=self.namespace[name],
                                      repeat_audit=Mock(side_effect=AssertionError("No audit")))
                with self.assertRaises(ValueError):
                    self.select()
                self.namespace["repeat_audit"].assert_not_called()

    def test_local_input_preserved_and_temporary_inputs_removed_on_failure(self):
        self.define("fresh-audit")
        downloads = []

        def fake_download(url, path, digest, expected_bytes=None):
            path.write_text("chr1\t1000000\n", encoding="utf-8")
            downloads.append(path)
            return path

        measure = Mock(return_value={"row_count": 3})
        self.namespace.update({
            "download_verified": fake_download,
            "local_tokenizers": lambda directory: ("rna", "protein", "fixture-version"),
            "cohort": SimpleNamespace(audit_csv=measure, __file__=cohort.__file__),
        })
        local = self.folder / "tiny.csv"
        original = local.read_bytes()
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            result = self.namespace["repeat_audit"](local)
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["audit_identity"]["transformers"], "fixture-version")
            self.assertEqual(measure.call_args.args[0], local)
            self.assertTrue(measure.call_args.kwargs["include_data_checks"])
            self.assertEqual(len(downloads), 1)  # Only chromosome text; vocab is mocked.
            self.assertTrue(all(not path.exists() for path in downloads))
            self.assertEqual(local.read_bytes(), original)
            measure.side_effect = ValueError("invented audit failure")
            with self.assertRaisesRegex(ValueError, "invented audit failure"):
                self.namespace["repeat_audit"](self.folder / "missing.csv")
            self.assertTrue(all(not path.exists() for path in downloads))
            self.assertEqual(local.read_bytes(), original)

    def test_directory_csv_path_is_refused(self):
        self.define("fresh-audit")
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            with self.assertRaisesRegex(ValueError, "file"):
                self.namespace["repeat_audit"](self.folder)


if __name__ == "__main__":
    unittest.main()
