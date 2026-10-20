"""Exercise Phase A planning and a tiny CPU classification fit, never research inputs."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from stage4 import run_blf_phaseA as runner
from shared_code.fusion import blf, train


def fixture_streams():
    rows = pd.DataFrame({"Value": [0, 1, 2] * 3,
                         "Split": ["train"] * 3 + ["val"] * 3 + ["test"] * 3})
    streams = SimpleNamespace(key="ecoli_proteins", task="classification", rows=rows,
                              labels=rows["Value"].to_numpy(), widths=dict(runner.EXPECTED_WIDTHS),
                              n_outputs=3, split_column="Split", close=mock.Mock())
    streams.split_rows = lambda name: np.flatnonzero(streams.rows["Split"].to_numpy() == name)
    return streams


def fixture_identity(streams, variant, config, device, kwargs):
    return {"version": 1, "variant": variant, "config": asdict(config),
            "inputs": {"rows_sha256": "fixture ordered rows"}, "environment": {"device": device},
            "code_sha256": {"train.py": "fixture source"}, "model_kwargs": kwargs}


class PhaseAExecutionTests(unittest.TestCase):
    def setUp(self):
        self.streams = fixture_streams()
        self.config = train.TrainConfig()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        identity = mock.patch.object(train, "run_identity", side_effect=fixture_identity)
        identity.start()
        self.addCleanup(identity.stop)
        counts = mock.patch.object(runner, "EXPECTED_SPLITS", {"ecoli_proteins": {"train": 3, "val": 3, "test": 3}})
        counts.start()
        self.addCleanup(counts.stop)

    def result(self, variant="concat"):
        return train.Result(
            variant, self.streams.key, asdict(self.config), 1, 0, 2 / 3, 2 / 3, 1.0,
            history=[{"epoch": 0, "val_score": 2 / 3}],
            identity=fixture_identity(self.streams, variant, self.config, "cpu", {}),
            test_truth=[0, 1, 2], test_prediction=[0, 1, 1],
        )

    def test_grid_is_27_unique_fixed_recipe_runs(self):
        runs = runner.planned_runs(self.config)
        paths = [runner.result_path(self.root, "ecoli_proteins", variant, config)
                 for variant, config in runs]
        self.assertEqual(len(runs), 27)
        self.assertEqual(len(set(paths)), 27)
        self.assertEqual({config.seed for _, config in runs}, {0, 1, 2})
        self.assertTrue(all(config.channels == 100 and config.tconv == "appendix" for _, config in runs))
        for seeds in ([], [0, 0], [-1], [True]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                runner.planned_runs(self.config, seeds)

    def test_preflight_rejects_invalid_classes_splits_and_widths(self):
        runner.validate_streams(self.streams)
        for labels in ([0, 1, 2.5] * 3, [-1, 0, 1] * 3, [0, 1, np.nan] * 3, [0, 1, 1] * 3):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                streams = fixture_streams()
                streams.labels = np.array(labels)
                runner.validate_streams(streams)
        self.streams.rows["Split"] = "train"
        with self.assertRaisesRegex(ValueError, "splits"):
            runner.validate_streams(self.streams)
        streams = fixture_streams()
        streams.widths["dna"] = 1
        with self.assertRaisesRegex(ValueError, "widths"):
            runner.validate_streams(streams)
        streams = fixture_streams()
        streams.rows.loc[0, "Split"] = "test"
        with self.assertRaisesRegex(ValueError, "counts"):
            runner.validate_streams(streams)

    def test_predictions_require_order_shape_class_ids_and_correct_score(self):
        runner.validate_predictions(self.result(), self.streams)
        for field, value in (("test_truth", None), ("test_prediction", []),
                             ("test_truth", [2, 1, 0]), ("test_prediction", [0, 1, 2.5]),
                             ("test_prediction", [0, 1, 3]), ("test_score", 1.0)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                result = self.result()
                setattr(result, field, value)
                runner.validate_predictions(result, self.streams)
        targets = np.array([0.123456789, 0.987654321, 0.333333333], dtype=np.float64)
        result = self.result()
        result.test_truth = targets.astype(np.float32).tolist()
        result.test_prediction = list(result.test_truth)
        result.test_score = 1.0
        runner.validate_prediction_arrays(result, "regression", targets)

    def test_default_preview_never_constructs_or_fits_and_closes_streams(self):
        destination = self.root / "not-created"
        with mock.patch.object(blf.BenchmarkStreams, "open", return_value=self.streams), \
                mock.patch.object(train, "build", side_effect=AssertionError("model construction")) as building, \
                mock.patch.object(train, "fit", side_effect=AssertionError("fitting")) as fitting, \
                mock.patch.object(train, "run", side_effect=AssertionError("execution")) as running, \
                redirect_stdout(io.StringIO()) as output:
            status = runner.main(["--benchmark", "ecoli_proteins", "--cache-root", str(self.root),
                                  "--results-dir", str(destination), "--device", "cpu"])
        self.assertEqual(status, 0)
        self.assertIn("PREVIEW ONLY", output.getvalue())
        self.assertIn("27 candidates", output.getvalue())
        self.assertFalse(destination.exists())
        self.streams.close.assert_called_once_with()
        for operation in (building, fitting, running):
            operation.assert_not_called()

    def test_exact_completed_result_is_reused_without_building_or_fitting(self):
        path = runner.result_path(self.root, self.streams.key, "concat", self.config)
        self.result().save(path)
        before = path.read_bytes()
        prepared = runner.preview_runs(self.streams, [("concat", self.config)], "cpu", self.root)
        self.assertEqual(prepared[0]["status"], "reuse")
        with mock.patch.object(train, "build", side_effect=AssertionError("model construction")), \
                mock.patch.object(train, "fit", side_effect=AssertionError("fitting")), \
                redirect_stdout(io.StringIO()):
            runner.execute_runs(self.streams, prepared, "cpu", self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_any_blocked_artifact_prevents_all_fits_and_stays_unchanged(self):
        path = runner.result_path(self.root, self.streams.key, "concat", self.config)
        for change in ("identity", "test_prediction"):
            with self.subTest(change=change):
                result = self.result()
                setattr(result, change, None)
                result.save(path)
                before = path.read_bytes()
                prepared = runner.preview_runs(self.streams, [("cross", self.config), ("concat", self.config)],
                                               "cpu", self.root)
                self.assertEqual([item["status"] for item in prepared], ["fit_required", "blocked_preserve"])
                with mock.patch.object(train, "run") as running, self.assertRaisesRegex(RuntimeError, "blocked"):
                    runner.execute_runs(self.streams, prepared, "cpu", self.root)
                running.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_checkpoint_preview_requires_matching_identity_and_recovery_fields(self):
        path = runner.result_path(self.root, self.streams.key, "concat", self.config).with_suffix(".checkpoint")
        saved = {"epoch": 0, "model": {}, "optimiser": {}, "schedule": {}, "best": {},
                 "history": [{"epoch": 0}], "identity": self.result().identity,
                 "random": dict.fromkeys(("python", "numpy", "torch", "loader"))}
        torch.save(saved, path)
        prepared = runner.preview_runs(self.streams, [("concat", self.config)], "cpu", self.root)
        self.assertEqual(prepared[0]["status"], "resume")
        for field, value in (("identity", {}), ("random", {}), ("epoch", 1)):
            with self.subTest(field=field):
                invalid = deepcopy(saved)
                invalid[field] = value
                torch.save(invalid, path)
                before = path.read_bytes()
                prepared = runner.preview_runs(self.streams, [("concat", self.config)], "cpu", self.root)
                self.assertEqual(prepared[0]["status"], "blocked_preserve")
                self.assertEqual(path.read_bytes(), before)

    def test_explicit_plan_output_never_overwrites_a_different_record(self):
        path = self.root / "plan.json"
        record = {"fixture": "first preview"}
        runner.write_plan(path, record)
        before = path.read_bytes()
        runner.write_plan(path, record)
        with self.assertRaises(FileExistsError):
            runner.write_plan(path, {"fixture": "later preview"})
        self.assertEqual(path.read_bytes(), before)

    def test_changed_computation_after_preview_stops_before_execution(self):
        prepared = runner.preview_runs(self.streams, [("concat", self.config)], "cpu", self.root)
        with mock.patch.object(train, "run_identity", return_value={"changed": "source"}), \
                mock.patch.object(train, "run") as running, self.assertRaisesRegex(RuntimeError, "changed"):
            runner.execute_runs(self.streams, prepared, "cpu", self.root)
        running.assert_not_called()


class ClassificationFitTests(unittest.TestCase):
    def test_one_epoch_cpu_run_stores_class_ids_and_reuses_the_completed_record(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        rows = pd.DataFrame({"sequence": ["A" * 18] * 12, "length": [18] * 12,
                             "Value": [0, 1, 2] * 4,
                             "Split": ["train"] * 6 + ["val"] * 3 + ["test"] * 3})

        class ArrayCache:
            def __init__(self, tokens, width):
                self.width = width
                self.values = rng.normal(size=(12, tokens, width)).astype(np.float16)

            def __getitem__(self, index):
                return self.values[index]

        caches = {"dna": ArrayCache(3, 3), "rna": ArrayCache(18, 4), "prot": ArrayCache(6, 2)}
        streams = blf.BenchmarkStreams("fixture", rows, caches, "classification")
        config = train.TrainConfig(max_epochs=1, batch_size=3, channels=2)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(train, "run_identity", side_effect=fixture_identity):
            result = train.run(streams, "concat", config, "cpu", directory, log=None,
                               require_predictions=True, out_width=6, dna_width=6)
            path = runner.result_path(directory, streams.key, "concat", config)
            before = path.read_bytes()
            self.assertFalse(path.with_suffix(".checkpoint").exists())
            with mock.patch.object(train, "build", side_effect=AssertionError("second model")), \
                    mock.patch.object(train, "fit", side_effect=AssertionError("second fit")):
                reused = train.run(streams, "concat", config, "cpu", directory, log=None,
                                   require_predictions=True, out_width=6, dna_width=6)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(reused.test_prediction, result.test_prediction)
        self.assertEqual(streams.n_outputs, 3)
        self.assertEqual(result.test_truth, [0, 1, 2])
        self.assertTrue(all(type(value) is int and value in (0, 1, 2) for value in result.test_prediction))
        self.assertEqual(len(result.test_prediction), 3)
        runner.validate_predictions(result, streams)
        runner.validate_completion(result, config, streams.task)
        self.assertGreaterEqual(result.test_score, 0.0)
        self.assertLessEqual(result.test_score, 1.0)


if __name__ == "__main__":
    unittest.main()
