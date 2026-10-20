"""Check historical-result inspection with invented identities and metadata headers."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Code/stage4/tier2_blf_phaseA.ipynb"
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion.train import Result, TrainConfig, paired_bootstrap
from stage4 import run_blf_phaseA as phasea_runner


class PhaseAResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = {cell["id"]: cell for cell in notebook["cells"]
                     if cell["cell_type"] == "code"}

    def setUp(self):
        self.stored = {}
        self.values = {
            "REPO_ROOT": ROOT, "CODE_DIR": ROOT / "Code", "Path": Path,
            "json": json, "asdict": asdict, "replace": replace, "np": np, "pd": pd,
            "display": lambda *args: None,
            # The reader cannot reach a model, a cache opener, or fitting through this stub.
            "train": SimpleNamespace(TrainConfig=TrainConfig, Result=SimpleNamespace(load=self.load),
                                     paired_bootstrap=paired_bootstrap),
            "blf": SimpleNamespace(ENCODER_OF_MODALITY={"dna": "DNA", "rna": "RNA", "prot": "Protein"}),
            "ENCODERS": {"DNA": object(), "RNA": object(), "Protein": object()},
            "inputs_for_rows": lambda encoder, sequences: sequences,
            "input_fingerprint": lambda values: "c" * 64,
            "file_fingerprint": lambda path: "d" * 64,
            "rows": pd.DataFrame({"sequence": ["AAACCC", "GGGTTT", "CCCAAA"]}),
            "row_columns": ["sequence", "length", "label", "split"], "rows_sha256": "a" * 64,
            "phasea_runner": SimpleNamespace(
                result_path=phasea_runner.result_path,
                validate_prediction_arrays=phasea_runner.validate_prediction_arrays,
            ),
            "expected_test_truth": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        }
        for cell_id in ("settings", "benchmark-contract", "comparison-plan"):
            self.execute(cell_id)
        self.read_headers("missing")
        self.execute("check-protocol")
        self.execute("check-scores")
        self.execute("check-predictions")

    def execute(self, cell_id, source=None):
        with redirect_stdout(io.StringIO()) as output, ExitStack() as guards:
            for name in ("open", "mkdir", "write_text", "write_bytes"):
                guards.enter_context(patch.object(
                    Path, name, side_effect=AssertionError("Unexpected file access")))
            guards.enter_context(patch("builtins.open", side_effect=AssertionError("Unexpected file access")))
            cell = self.cells[cell_id]
            exec(compile(source or "".join(cell["source"]), f"{NOTEBOOK.name}:{cell_id}", "exec"), self.values)
        return output.getvalue()

    def classification(self):
        source = "".join(self.cells["settings"]["source"]).replace(
            'BENCHMARK = "fungal_expression"', 'BENCHMARK = "ecoli_proteins"', 1)
        self.execute("settings", source)
        self.execute("benchmark-contract")
        self.execute("comparison-plan")
        self.values["expected_test_truth"] = np.array([0, 1, 2], dtype=np.int64)

    def read_headers(self, mode):
        def read_text(path, *args, **kwargs):
            if mode == "missing":
                raise FileNotFoundError(path)
            if path.name != "meta.json":
                raise AssertionError(f"Reader tried to open a shard: {path}")
            encoder = path.parent.name
            return json.dumps({"encoder": encoder, "n_sequences": 3, "n_shards": 1,
                               "input_sha256": ("e" if mode == "mismatched" else "c") * 64})

        with patch.object(Path, "read_text", read_text), patch.object(
                Path, "is_file", lambda path: mode != "missing"):
            return self.execute("headers")

    def record(self, variant="concat", weight=0.0, seed=0):
        config = asdict(replace(self.values["BASE_CONFIG"], seed=seed, entropy_lambda=weight))
        environment = {
            "python": "fixture", "platform": "fixture", "torch": "fixture", "numpy": "fixture",
            "scipy": "fixture", "device": "cpu", "threads": 1, "interop_threads": 1,
            "float32_matmul_precision": "highest", "deterministic_algorithms": False,
            "cudnn_deterministic": False, "cudnn_benchmark": False, "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False, "cublas_workspace_config": None,
        }
        identity = {
            "version": 1, "variant": variant, "config": deepcopy(config),
            "model_kwargs": {}, "widths": dict(self.values["EXPECTED_WIDTHS"]),
            "n_outputs": self.values["EXPECTED_OUTPUTS"],
            "code_sha256": {name: "b" * 64 for name in self.values["source_names"]},
            "environment": environment,
            "inputs": {
                "benchmark": self.values["BENCHMARK"], "task": self.values["TASK"],
                "row_columns": list(self.values["row_columns"]), "rows_sha256": "a" * 64,
                "input_identity_status": "recorded_inputs",
                "cache_files": {modality: {"meta.json": "d" * 64, "shard_00000.npz": "f" * 64}
                                for modality in ("dna", "rna", "prot")},
            },
        }
        return Result(variant, self.values["BENCHMARK"], config, 2, 1, 0.5, 0.4, 1.0,
                      history=[{"epoch": 0, "val_score": 0.3}, {"epoch": 1, "val_score": 0.5}],
                      identity=identity)

    def load(self, path):
        return self.stored[Path(path)]

    def collect(self, records):
        folder = self.values["RESULT_DIRECTORY"]
        self.stored = {folder / f"fixture_{index}.json": result for index, result in enumerate(records)}
        # The explicit reference is a protocol-valid concat run, independently of scan order.
        reference_path = next((path for path, result in self.stored.items()
                               if result.variant == "concat" and result.config["seed"] == 0), None)
        if reference_path is not None:
            self.stored[self.values["REFERENCE_RESULT_PATH"]] = self.stored.pop(reference_path)
        with patch.object(Path, "glob", lambda folder, pattern: list(self.stored)):
            self.execute("collect")
        return self.execute("coverage")

    def test_historical_producer_is_accepted_without_hashing_the_reader(self):
        result = self.record()
        # Deliberately different current-reader hashes must not become producer evidence.
        self.values["source_hashes"] = {name: "0" * 64 for name in self.values["source_names"]}
        self.assertEqual(self.values["check_protocol"](result), (0, "concat", 0.0))
        self.values["check_scores"](result)
        self.assertFalse(self.values["RUN_TRAINING"])
        self.assertFalse(self.values["CACHE_HEADERS_VERIFIED"])

    def test_protocol_rejects_changed_rows_config_and_malformed_identities(self):
        def change_row(result):
            result.identity["inputs"]["rows_sha256"] = "0" * 64

        def change_config(result):
            result.config["batch_size"] = 64
            result.identity["config"]["batch_size"] = 64

        def missing_source(result):
            result.identity["code_sha256"].pop("train.py")

        def bad_cache_hash(result):
            result.identity["inputs"]["cache_files"]["dna"]["shard_00000.npz"] = "not-a-hash"

        for change in (change_row, change_config, missing_source, bad_cache_hash):
            with self.subTest(change=change.__name__):
                result = self.record()
                change(result)
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    self.values["check_protocol"](result)

    def test_baseline_candidates_accept_only_the_three_native_encoder_variants(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot"):
            with self.subTest(variant=variant):
                self.assertEqual(self.values["check_protocol"](self.record(variant)),
                                 (0, variant, 0.0))
        # Sharing the prefix does not make a differently pooled encoder an eligible baseline.
        extra = self.record("unimodal:rna_codon")
        with self.assertRaises(ValueError):
            self.values["check_protocol"](extra)
        self.collect([self.record(), extra])
        self.assertEqual(len(self.values["results"]), 1)
        self.assertEqual(len(self.values["excluded"]), 1)

    def test_scores_require_consistent_completed_history_and_validation_selection(self):
        for field, bad_value in (("epochs", 3), ("best_epoch", 0), ("test_score", float("nan"))):
            with self.subTest(field=field):
                result = self.record()
                setattr(result, field, bad_value)
                with self.assertRaises(ValueError):
                    self.values["check_scores"](result)

    def test_reference_cohort_rejects_different_inputs_producer_or_environment(self):
        reference = self.record()
        other = self.record("cross")
        self.values["check_reference_cohort"](other, reference.identity)
        for field in ("inputs", "code_sha256", "environment"):
            with self.subTest(field=field):
                changed = deepcopy(other)
                changed.identity[field]["fixture_difference"] = "different"
                with self.assertRaises(ValueError):
                    self.values["check_reference_cohort"](changed, reference.identity)

    def test_headers_distinguish_pending_verified_and_mismatched_inputs(self):
        self.assertFalse(self.values["CACHE_HEADERS_VERIFIED"])
        self.read_headers("matching")
        self.assertTrue(self.values["CACHE_HEADERS_VERIFIED"])
        self.values["check_protocol"](self.record())
        changed = self.record()
        changed.identity["inputs"]["cache_files"]["dna"]["meta.json"] = "0" * 64
        with self.assertRaises(ValueError):
            self.values["check_protocol"](changed)
        self.read_headers("mismatched")
        self.assertFalse(self.values["CACHE_HEADERS_VERIFIED"])

    def test_partial_grid_and_invalid_records_do_not_produce_a_comparison(self):
        records = [self.record(variant, weight) for _, variant, weight in self.values["CANDIDATES"]]
        invalid = self.record("cross", seed=1)
        invalid.val_score = 2.0
        other_producer = self.record("concat", seed=1)
        other_producer.identity["code_sha256"]["train.py"] = "0" * 64
        self.collect([invalid, other_producer, *records])
        self.assertEqual(len(self.values["results"]), 9)
        self.assertEqual(len(self.values["excluded"]), 2)
        self.assertEqual(self.values["coverage"]["matching_files"].eq(1).sum(), 9)
        self.assertFalse(self.values["COMPARISON_READY"])

    def test_complete_grid_needs_verified_headers_and_exactly_one_record_per_candidate(self):
        records = [self.record(variant, weight, seed) for seed in self.values["SEEDS"]
                   for _, variant, weight in self.values["CANDIDATES"]]
        self.collect(records)
        self.assertFalse(self.values["COMPARISON_READY"])
        self.read_headers("matching")
        self.collect(records)
        self.assertTrue(self.values["COMPARISON_READY"])
        self.collect([*records, deepcopy(records[0])])
        self.assertFalse(self.values["COMPARISON_READY"])
        self.assertIn("duplicate", self.values["coverage"]["status"].values)

    def test_invalid_reference_blocks_fallback_and_clears_previous_comparisons(self):
        reference = self.record()
        reference.val_score = 2.0
        self.collect([self.record("cross"), reference])
        self.assertTrue(self.values["results"].empty)
        self.assertEqual(self.values["prediction_availability"]["matching_records"].sum(), 1)
        self.assertFalse(self.values["COMPARISON_READY"])
        for table in ("selected", "choices", "paired"):
            self.values[table] = pd.DataFrame({"old": [1]})
        self.execute("select")
        self.assertTrue(all(self.values[table].empty for table in ("selected", "choices", "paired")))

    def test_prediction_inventory_counts_presence_without_claiming_valid_predictions(self):
        records = [self.record(variant) for variant in
                   ("concat", "cross", "unimodal:dna", "unimodal:rna", "unimodal:prot")]
        records[1].test_truth = [1.0]
        records[2].test_prediction = [0.4]
        records[3].test_truth, records[3].test_prediction = [], []
        # Empty or differently sized fields are still present. This inventory deliberately
        # makes no claim that they are suitable for paired bootstrap or row-level analysis.
        records[4].test_truth, records[4].test_prediction = [1.0, 2.0], [0.4]
        self.collect(records)
        availability = {
            (truth, prediction): count for truth, prediction, count
            in self.values["prediction_availability"].itertuples(index=False, name=None)
        }
        self.assertEqual(availability, {(False, False): 1, (True, False): 1,
                                        (False, True): 1, (True, True): 2})
        self.assertEqual(len(self.values["checked_results"]), 5)
        self.assertEqual(len(self.values["results"]), 3)
        self.assertEqual(len(self.values["excluded"]), 2)
        self.assertFalse(self.values["COMPARISON_READY"])

    def test_classification_protocol_uses_three_outputs_and_fraction_scores(self):
        self.classification()
        self.assertEqual(self.values["EXPECTED_SPLITS"], {"train": 3010, "val": 728, "test": 712})
        self.assertEqual(self.values["DISPLAY_SCALE"], 100.0)
        self.assertEqual(self.values["EXPECTED_OUTPUTS"], 3)
        result = self.record()
        self.values["check_protocol"](result)
        self.values["check_scores"](result)
        for score in (-0.1, 52.949):
            result.test_score = score
            with self.assertRaises(ValueError):
                self.values["check_scores"](result)

    def test_empty_classification_grid_reports_local_absence_without_a_reference(self):
        self.classification()
        self.collect([])
        with patch.object(Path, "is_file", return_value=False):
            self.execute("execution-preview")
        preview = self.values["execution_preview"]
        self.assertEqual(len(preview), len(self.values["plan"]))
        self.assertTrue(preview["status"].eq("not_found_locally").all())
        self.assertTrue(self.values["coverage"]["status"].eq("not_represented").all())
        self.assertFalse(self.values["COMPARISON_READY"])
        self.assertFalse(self.values["RUN_TRAINING"])
        self.assertIn("stage4", str(self.values["RESULT_DIRECTORY"]))

    def test_existing_record_waits_for_reference_and_score_only_is_preserved(self):
        self.classification()
        saved = self.record("unimodal:dna")
        saved.test_truth = [0, 1, 2]
        saved.test_prediction = [0, 1, 0]
        saved.test_score = 2 / 3
        self.collect([saved])
        with patch.object(Path, "is_file", return_value=False):
            self.execute("execution-preview")
        status = self.values["execution_preview"].set_index(["seed", "variant", "entropy_lambda"])["status"]
        self.assertEqual(status.loc[(0, "unimodal:dna", 0.0)], "reference_pending")
        self.assertFalse(self.values["COMPARISON_READY"])
        self.collect([self.record()])
        with patch.object(Path, "is_file", return_value=False):
            self.execute("execution-preview")
        status = self.values["execution_preview"].set_index(["seed", "variant", "entropy_lambda"])["status"]
        self.assertEqual(status.loc[(0, "concat", 0.0)], "predictions_absent")

    def test_local_inventory_distinguishes_invalid_predictions_and_incompatible_records(self):
        self.classification()
        records = [self.record(variant) for variant in
                   ("concat", "cross", "unimodal:dna", "unimodal:prot")]
        for result in records[1:]:
            result.test_truth = [0, 1, 2]
            result.test_prediction = [0, 1, 2]
            result.test_score = 1.0
        records[1].test_score = 0.5  # Present predictions contradict the saved score.
        records[2].identity["code_sha256"]["train.py"] = "0" * 64
        original = deepcopy(records)
        self.collect(records)
        with patch.object(Path, "is_file", return_value=False):
            self.execute("execution-preview")
        status = self.values["execution_preview"].set_index(
            ["seed", "variant", "entropy_lambda"])["status"]
        self.assertEqual(status.loc[(0, "concat", 0.0)], "predictions_absent")
        self.assertEqual(status.loc[(0, "cross", 0.0)], "predictions_invalid")
        self.assertEqual(status.loc[(0, "unimodal:dna", 0.0)], "incompatible_with_comparison")
        self.assertEqual(status.loc[(0, "unimodal:prot", 0.0)], "recorded_evidence_available")
        coverage = self.values["coverage"].set_index(["seed", "variant", "entropy_lambda"])
        # An existing but incompatible file leaves the comparison unrepresented, not absent locally.
        self.assertEqual(coverage.loc[(0, "unimodal:dna", 0.0), "status"], "not_represented")
        self.assertEqual(coverage.loc[(0, "unimodal:dna", 0.0), "matching_files"], 0)
        self.assertEqual(records, original)
        self.assertFalse(self.values["COMPARISON_READY"])
        self.assertFalse(self.values["RUN_TRAINING"])

    def test_prediction_validation_uses_training_dtype_and_rejects_bad_arrays(self):
        result = self.record()
        self.values["expected_test_truth"] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        result.test_truth = self.values["expected_test_truth"].astype(np.float32).tolist()
        result.test_prediction = [0.2, 0.4, 0.1]
        result.test_score = -0.5
        self.values["check_predictions"](result)
        self.classification()
        result = self.record()
        result.test_truth, result.test_prediction, result.test_score = [0, 1, 2], [0, 1, 0], 2 / 3
        self.values["check_predictions"](result)
        for values in ([0.5, 1, 2], [0, 1, 3], [0, 1], [0, 1, float("nan")]):
            changed = deepcopy(result)
            changed.test_prediction = values
            with self.assertRaises(ValueError):
                self.values["check_predictions"](changed)
        changed = deepcopy(result)
        changed.test_truth = [2, 1, 0]
        with self.assertRaises(ValueError):
            self.values["check_predictions"](changed)

    def test_inconsistent_present_predictions_block_an_otherwise_complete_grid(self):
        self.classification()
        records = [self.record(variant, weight, seed) for seed in self.values["SEEDS"]
                   for _, variant, weight in self.values["CANDIDATES"]]
        invalid = next(record for record in records if record.variant == "cross")
        invalid.test_truth, invalid.test_prediction = [0, 1, 2], [0, 1, 2]
        invalid.test_score = 0.5  # The supplied predictions instead have accuracy 1.
        self.read_headers("matching")
        self.collect(records)
        self.assertEqual(len(self.values["results"]), len(records) - 1)
        self.assertFalse(self.values["COMPARISON_READY"])
        self.assertIn("Invalid saved prediction arrays", self.values["excluded"][0]["reason"])

    def test_paired_interval_figure_uses_percentage_points_and_requires_all_intervals(self):
        self.classification()
        axis = MagicMock()
        plot = SimpleNamespace(subplots=MagicMock(return_value=(object(), axis)), show=MagicMock())
        methods = ["Concat", "MIL", "Cross", "MIL without entropy"]
        self.values.update({
            "plt": plot, "UNCERTAINTY_READY": True,
            "offsets": np.linspace(-0.15, 0.15, 3),
            "seed_colors": ("#0072B2", "#D55E00", "#009E73"),
            "uncertainty": pd.DataFrame([
                {"seed": seed, "method": method, "baseline": baseline,
                 "difference": 0.02, "ci_low": -0.01, "ci_high": 0.06}
                for seed, baseline in enumerate(("Protein", "RNA", "RNA")) for method in methods
            ]),
        })
        self.execute("uncertainty-figure")
        self.assertEqual(len(axis.hlines.call_args_list), 3)
        for call in axis.hlines.call_args_list:
            self.assertEqual(len(call.args[0]), len(methods))
            np.testing.assert_allclose(call.args[1], -1.0)
            np.testing.assert_allclose(call.args[2], 6.0)
        for call in axis.scatter.call_args_list:
            np.testing.assert_allclose(call.args[0], 2.0)
        self.assertEqual([call.kwargs["label"] for call in axis.scatter.call_args_list],
                         ["Seed 0 vs Protein", "Seed 1 vs RNA", "Seed 2 vs RNA"])
        self.assertEqual(axis.set.call_args.kwargs["yticklabels"], methods)
        self.assertIn("percentage points", axis.set.call_args.kwargs["xlabel"])
        self.assertEqual(axis.axvline.call_args.args, (0,))
        plot.subplots.reset_mock()
        plot.show.reset_mock()
        self.values["UNCERTAINTY_READY"] = False
        self.execute("uncertainty-figure")
        plot.subplots.assert_not_called()
        plot.show.assert_not_called()

    def test_classification_selection_bootstrap_and_manifest_keep_units_and_lineage(self):
        self.classification()
        records = [self.record(variant, weight, seed) for seed in self.values["SEEDS"]
                   for _, variant, weight in self.values["CANDIDATES"]]
        for record in records:
            record.test_truth = [0, 1, 2]
            record.test_prediction = [0, 1, 2] if record.variant == "cross" else [0, 0, 2]
            record.test_score = 1.0 if record.variant == "cross" else 2 / 3
        self.read_headers("matching")
        self.collect(records)
        self.assertTrue(self.values["COMPARISON_READY"])
        self.execute("select")
        choices = self.values["choices"]
        self.assertTrue(choices["single encoder"].eq("DNA").all())
        self.assertTrue(choices["MIL lambda"].eq(0.01).all())
        self.values["BOOTSTRAP_DRAWS"] = 40
        self.execute("tables")
        self.assertAlmostEqual(self.values["summary_display"].loc["Cross", "mean"], 100.0)
        self.assertAlmostEqual(self.values["selected"].query("method == 'Cross'")["test_score"].iloc[0], 1.0)
        self.execute("uncertainty")
        self.assertTrue(self.values["UNCERTAINTY_READY"])
        self.assertEqual(len(self.values["uncertainty"]), len(self.values["SEEDS"]) * 4)
        cross = self.values["uncertainty"].query("method == 'Cross'")
        np.testing.assert_allclose(cross["difference"], 1 / 3)
        self.execute("analysis-manifest")
        manifest = self.values["analysis_manifest"]
        self.assertEqual(len(manifest["candidate_files"]), len(records))
        self.assertEqual(len(manifest["selected_files"]), 21)
        relative_paths = [manifest["reference_path"], *self.values["results"]["path"],
                          *self.values["selected"]["path"]]
        relative_paths.extend(item["path"] for item in manifest["candidate_files"])
        relative_paths.extend(item["path"] for item in manifest["selected_files"])
        self.execute("execution-preview")
        relative_paths.extend(self.values["execution_preview"]["expected_path"])
        self.assertTrue(all("\\" not in path and path.startswith("Code/") for path in relative_paths))
        self.assertTrue(all(item["sha256"] == "d" * 64 for item in manifest["candidate_files"]))
        self.assertEqual(set(manifest["reader_helper_sha256"]), {
            "Code/shared_code/fusion/train.py", "Code/stage4/run_blf_phaseA.py"})
        self.assertTrue(all(value == "d" * 64 for value in manifest["reader_helper_sha256"].values()))
        self.assertEqual(manifest["stored_score_units"], "accuracy proportion")
        self.assertEqual(manifest["bootstrap"]["draws"], 40)
        json.dumps(manifest, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
