"""Check notebook-local readiness and isolated branches, without research I/O."""

import ast
from contextlib import redirect_stdout
from dataclasses import dataclass, fields
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "Code"
NOTEBOOK = CODE / "stage4" / "tier2_isoformer_reproduction.ipynb"
sys.path.insert(0, str(CODE))
from shared_code.fusion import isoformer_inputs


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    }


def readiness_namespace(code):
    """Load only local settings/status cells; no project imports or data access."""
    namespace = {
        "__name__": __name__, "CODE_DIR": CODE,
        "dataclass": dataclass, "fields": fields,
    }
    for name in ("settings", "readiness-details", "readiness-guard"):
        exec(compile(code[name], name, "exec"), namespace)
    return namespace


class ReproductionReadinessTests(unittest.TestCase):
    def setUp(self):
        _, self.code = notebook_code()
        self.namespace = readiness_namespace(self.code)

    def assert_training_refused(self, message):
        self.namespace["RUN_TRAINING"] = True
        with self.assertRaisesRegex(RuntimeError, message):
            exec(compile(self.code["readiness-guard"], "training guard", "exec"),
                 self.namespace)

    def test_agreed_policies_are_explicit_and_other_choices_stay_unset(self):
        choices = self.namespace["choices"]
        rows = self.namespace["decision_rows"](choices)
        self.assertEqual({row["setting"] for row in rows},
                         {field.name for field in fields(choices)})
        agreed = {
            "pool_window": (765, 770),
            "dna_input_nucleotides": 196_608,
            "dna_extra_filter_nucleotides": 196_608,
            "protein_mask_policy": "released_unmasked",
            "missing_protein_policy": "bypass_projected_update",
            "rna_max_nucleotides": 12_288,
            "protein_max_characters": 1_200,
            "protein_crop": "exclude",
            "raw_eval_policy": "exclude",
            "cohort_boundary_policy": "released_example",
        }
        for row in rows:
            if row["setting"] in agreed:
                self.assertEqual(row["value"], agreed[row["setting"]])
                self.assertEqual(row["status"], "declared, not validated")
            else:
                self.assertIsNone(row["value"])
                self.assertEqual(row["status"], "unresolved")
        self.assert_training_refused("target_normalization")

    def test_filling_choices_cannot_claim_an_implemented_trainer(self):
        choices_type = self.namespace["ReproductionChoices"]
        choices = choices_type(**{
            field.name: "a declared choice" for field in fields(choices_type)
        })
        self.assertTrue(all(row["status"] == "declared, not validated"
                            for row in self.namespace["decision_rows"](choices)))
        self.namespace["choices"] = choices
        self.assert_training_refused("trainer, checkpoint recovery")

    def test_no_reintroduced_stopping_or_normalization_recipe(self):
        # These withdrawn defaults must not become an executable training recipe.
        tree = ast.parse(self.code["paper-settings"])
        paper_settings = next(
            ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PAPER_SETTINGS"
                    for target in node.targets)
        )
        self.assertFalse({"max_epochs", "patience", "target_normalization"}
                         .intersection(paper_settings))
        self.assertIsNone(self.namespace["choices"].stopping_rule)
        self.assertIsNone(self.namespace["choices"].target_normalization)


class NotebookBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.code = notebook_code()

    def test_notebook_code_compiles(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_enabled_training_refuses_before_optional_data_read(self):
        code = dict(self.code)
        code["settings"] = code["settings"].replace(
            "RUN_TRAINING = False", "RUN_TRAINING = True")
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")):
            with self.assertRaisesRegex(RuntimeError, "training is unavailable"):
                readiness_namespace(code)
        # Source ordering makes the guard visible before the data-reading branch.
        ids = list(self.code)
        self.assertLess(ids.index("settings"), ids.index("readiness-details"))
        self.assertLess(ids.index("readiness-details"), ids.index("readiness-guard"))
        self.assertLess(ids.index("readiness-guard"), ids.index("hardware"))
        self.assertLess(ids.index("readiness-guard"), ids.index("cohort-inspection"))

    def test_default_cohort_branch_does_not_read_or_validate_data(self):
        pd = Mock()
        pd.read_csv.side_effect = AssertionError("must not read a research CSV")
        inputs = Mock()
        inputs.validate_cohort.side_effect = AssertionError("must not inspect a cohort")
        namespace = readiness_namespace(self.code)
        namespace.update(pd=pd, inputs=inputs)
        with redirect_stdout(io.StringIO()):
            exec(compile(self.code["cohort-inspection"], "isolated cohort branch", "exec"), namespace)
        self.assertIsNone(namespace["cohort"])
        pd.read_csv.assert_not_called()
        inputs.validate_cohort.assert_not_called()

    def test_default_sequence_branch_does_not_prepare_data(self):
        inputs = Mock()
        inputs.apply_sequence_policies.side_effect = AssertionError("unexpected data preparation")
        namespace = readiness_namespace(self.code)
        namespace.update(inputs=inputs, cohort=None)
        exec(compile(self.code["sequence-policy-preparation"], "sequence branch", "exec"),
             namespace)
        self.assertIsNone(namespace["prepared_cohort"])
        self.assertIsNone(namespace["sequence_policy_report"])
        inputs.apply_sequence_policies.assert_not_called()

    def test_sequence_branch_uses_agreed_settings_and_preserves_raw_eval(self):
        # Invented rows exercise notebook wiring, not a project dataset or CSV reader.
        rows = pd.DataFrame({
            "transcript_id_gtex": ["t1", "t2", "t3", "t4"],
            "gene_id_gtex": ["g1", "g2", "g3", "g4"],
            "chr": ["chr1"] * 4, "start": [100] * 4,
            "end": [200] * 4, "strand": ["+"] * 4,
            "RNA": ["G" + "A" * 12_288, "ACGT", "GATT", "TT"],
            "Protein": [None, "M" * 1_200, "M" * 1_201, ""],
            "split": ["train", "test", "train", "eval"],
            "tissue": [0.1, 0.2, 0.3, 0.4],
        }, index=[11, 22, 33, 44])
        original = rows.copy(deep=True)
        namespace = readiness_namespace(self.code)
        namespace.update(
            inputs=isoformer_inputs,
            cohort=isoformer_inputs.validate_cohort(rows, ("tissue",)),
            display=Mock(),
        )
        with redirect_stdout(io.StringIO()):
            exec(compile(self.code["sequence-policy-preparation"], "sequence branch", "exec"),
                 namespace)
        prepared = namespace["prepared_cohort"].rows
        self.assertEqual(prepared.index.tolist(), [11, 22, 44])
        self.assertEqual(prepared.loc[11, "RNA"], "A" * 12_288)
        self.assertEqual(prepared.loc[44, "split"], "eval")
        pd.testing.assert_series_equal(prepared["tissue"], rows.loc[[11, 22, 44], "tissue"])
        pd.testing.assert_frame_equal(rows, original)
        report = namespace["sequence_policy_report"]
        self.assertTrue(report.loc[33, "protein_present"])
        self.assertFalse(report.loc[33, "sequence_eligible"])

    def test_eligibility_without_cohort_or_metadata_does_not_read(self):
        for prepared, sizes_path in ((None, "must-not-open"), (object(), None)):
            namespace = readiness_namespace(self.code)
            reader = Mock(side_effect=AssertionError("must not read metadata"))
            inputs = Mock()
            inputs.apply_reference_eligibility.side_effect = AssertionError("must not filter")
            namespace.update(
                prepared_cohort=prepared, LOCAL_CHROMOSOME_SIZES=sizes_path,
                read_verified_chromosome_sizes=reader, inputs=inputs,
            )
            with self.subTest(prepared=prepared), redirect_stdout(io.StringIO()):
                exec(compile(self.code["cohort-eligibility"], "eligibility branch", "exec"),
                     namespace)
            self.assertIsNone(namespace["eligible_cohort"])
            self.assertIsNone(namespace["eligibility_report"])
            reader.assert_not_called()
            inputs.apply_reference_eligibility.assert_not_called()

    def test_chromosome_size_reader_checks_identity_duplicates_and_lengths(self):
        namespace = {"Path": Path, "hashlib": hashlib}
        exec(compile(self.code["chromosome-size-reader"], "metadata reader", "exec"), namespace)
        read = namespace["read_verified_chromosome_sizes"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.chrom.sizes"
            valid = b"chr1\t1000\nchr2\t2000\n"
            path.write_bytes(valid)
            digest = hashlib.sha256(valid).hexdigest()
            self.assertEqual(read(path, digest), {"chr1": 1000, "chr2": 2000})
            with self.assertRaisesRegex(ValueError, "declared hg38"):
                read(path, "0" * 64)
            self.assertEqual(path.read_bytes(), valid)
            for payload in (b"chr1 10\nchr1 20\n", b"chr1 0\n", b"chr1 -1\n",
                            b"chr1 1.5\n", b"chr1 True\n", b"chr1 2 extra\n", b""):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    read(path, hashlib.sha256(payload).hexdigest())
                self.assertEqual(path.read_bytes(), payload)

    def test_eligibility_branch_preserves_membership_and_applies_declared_policies(self):
        rows = pd.DataFrame({
            "transcript_id_gtex": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "gene_id_gtex": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "chr": ["chr1", "chr1", "chr1", "ignored-eval-chrom", "chrMT", "chr1"],
            "start": [500_001, 400_000, 150_001, 1, 1, 901_697],
            "end": [500_100, 500_001, 150_100, 2, 2, 901_700],
            "strand": ["+", "-", "+", "+", "+", "+"],
            "RNA": ["ACGT"] * 6, "Protein": [None, "MK", None, None, "MK", "MK"],
            "split": ["train", "test", "train", "eval", "train", "train"],
            "tissue": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }, index=[11, 22, 33, 44, 55, 66])
        original = rows.copy(deep=True)
        namespace = readiness_namespace(self.code)
        namespace.update(Path=Path, hashlib=hashlib, inputs=isoformer_inputs, display=Mock(),
                         prepared_cohort=isoformer_inputs.validate_cohort(rows, ("tissue",)))
        exec(compile(self.code["chromosome-size-reader"], "metadata reader", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.chrom.sizes"
            payload = b"chr1\t1000000\n"
            path.write_bytes(payload)
            namespace.update(LOCAL_CHROMOSOME_SIZES=path,
                             CHROMOSOME_SIZES_SHA256=hashlib.sha256(payload).hexdigest())
            with redirect_stdout(io.StringIO()):
                exec(compile(self.code["cohort-eligibility"], "eligibility branch", "exec"), namespace)
            self.assertEqual(path.read_bytes(), payload)
        pd.testing.assert_frame_equal(namespace["eligible_cohort"].rows, original.loc[[11, 22]])
        pd.testing.assert_frame_equal(rows, original)
        report = namespace["eligibility_report"]
        self.assertEqual(report.index.tolist(), original.index.tolist())
        self.assertEqual(report.loc[44, "reference_exclusion"], "raw_eval_unused")
        self.assertTrue(report.loc[55, "mitochondrial_excluded"])
        self.assertTrue(report.loc[33, "extra_filter_excluded"])
        self.assertFalse(report.loc[33, "main_window_excluded"])
        self.assertTrue(report.loc[66, "main_window_excluded"])
        self.assertFalse(report.loc[44, "dna_window_checked"])

    def test_no_download_training_or_result_writer_in_notebook(self):
        # Refuse hidden producer calls in this preparation notebook, including a
        # future accidental copy of pilot code. This is an AST check, not a sandbox.
        forbidden = {"from_pretrained", "load_dataset", "load_encoder", "extract",
                     "fit", "backward", "step", "save", "savez", "to_csv",
                     "write_text", "write_bytes", "mkdir"}
        for name, source in self.code.items():
            tree = ast.parse(source)
            called = {node.func.attr for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            self.assertFalse(called & forbidden, (name, called & forbidden))


if __name__ == "__main__":
    unittest.main()
