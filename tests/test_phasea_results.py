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
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Code/stage4/tier2_blf_phaseA.ipynb"
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion.train import Result, TrainConfig


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
            "train": SimpleNamespace(TrainConfig=TrainConfig, Result=SimpleNamespace(load=self.load)),
            "blf": SimpleNamespace(ENCODER_OF_MODALITY={"dna": "DNA", "rna": "RNA", "prot": "Protein"}),
            "ENCODERS": {"DNA": object(), "RNA": object(), "Protein": object()},
            "inputs_for_rows": lambda encoder, sequences: sequences,
            "input_fingerprint": lambda values: "c" * 64,
            "file_fingerprint": lambda path: "d" * 64,
            "rows": pd.DataFrame({"sequence": ["AAACCC", "GGGTTT", "CCCAAA"]}),
            "row_columns": ["sequence", "length", "label", "split"], "rows_sha256": "a" * 64,
        }
        self.execute("settings")
        self.read_headers("missing")
        self.execute("check-protocol")
        self.execute("check-scores")

    def execute(self, cell_id):
        with redirect_stdout(io.StringIO()) as output, ExitStack() as guards:
            for name in ("open", "mkdir", "write_text", "write_bytes"):
                guards.enter_context(patch.object(
                    Path, name, side_effect=AssertionError("Unexpected file access")))
            guards.enter_context(patch("builtins.open", side_effect=AssertionError("Unexpected file access")))
            cell = self.cells[cell_id]
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell_id}", "exec"), self.values)
        return output.getvalue()

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
            "model_kwargs": {}, "widths": dict(self.values["EXPECTED_WIDTHS"]), "n_outputs": 1,
            "code_sha256": {name: "b" * 64 for name in self.values["source_names"]},
            "environment": environment,
            "inputs": {
                "benchmark": "fungal_expression", "task": "regression",
                "row_columns": list(self.values["row_columns"]), "rows_sha256": "a" * 64,
                "input_identity_status": "recorded_inputs",
                "cache_files": {modality: {"meta.json": "d" * 64, "shard_00000.npz": "f" * 64}
                                for modality in ("dna", "rna", "prot")},
            },
        }
        return Result(variant, "fungal_expression", config, 2, 1, 0.5, 0.4, 1.0,
                      history=[{"epoch": 0, "val_score": 0.3}, {"epoch": 1, "val_score": 0.5}],
                      identity=identity)

    def load(self, path):
        return self.stored[Path(path)]

    def collect(self, records):
        folder = self.values["RESULT_DIRECTORIES"][0]
        self.stored = {folder / f"fixture_{index}.json": result for index, result in enumerate(records)}
        # The explicit reference is a protocol-valid concat run, independently of scan order.
        reference_path = next(path for path, result in self.stored.items()
                              if result.variant == "concat" and result.config["seed"] == 0)
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
        self.assertTrue(self.values["prediction_availability"].empty)
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
        self.assertEqual(len(self.values["results"]), 5)
        self.assertFalse(self.values["COMPARISON_READY"])


if __name__ == "__main__":
    unittest.main()
