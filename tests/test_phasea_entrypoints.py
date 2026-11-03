"""Synthetic checks for historical inspection/export; no real notebook is executed."""

import argparse
import builtins
from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "Code/stage4/run_blf_phaseA.py"
EXPORTER = ROOT / "Code/stage4/results/blf_phaseA/ecoli_proteins/records/export_analysis.py"


@dataclass
class FixtureConfig:
    seed: int = 0
    entropy_lambda: float = 0.0
    channels: int = 100
    tconv: str = "appendix"
    max_epochs: int = 500


class FixtureResult(SimpleNamespace):
    @staticmethod
    def load(path):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        record["extra_fields"] = {"identity": record.pop("identity")} if "identity" in record else {}
        return FixtureResult(**record)

    def save(self, path):
        raise AssertionError("No result publication expected")


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_runner():
    """Replace heavy dependencies at the import boundary, not research data or calls."""
    original_import = builtins.__import__
    train = SimpleNamespace(TrainConfig=FixtureConfig, Result=FixtureResult)
    blf = SimpleNamespace(BenchmarkStreams=SimpleNamespace(open=Mock(side_effect=AssertionError)))
    numerical = SimpleNamespace(isfinite=math.isfinite,
                                argmax=lambda values: max(range(len(values)), key=values.__getitem__))

    def guarded_import(name, *args, **kwargs):
        if name == "numpy":
            return numerical
        if name in {"pandas", "torch"}:
            return SimpleNamespace()
        if name == "shared_code.fusion":
            return SimpleNamespace(train=train, blf=blf)
        if name == "shared_code.fusion.token_embeddings":
            return SimpleNamespace(file_fingerprint=lambda path: "a" * 64)
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        return import_file("phasea_runner_fixture", RUNNER)


class PhaseAInspectionTests(unittest.TestCase):
    def setUp(self):
        self.module = import_runner()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def result(self, variant="unimodal:dna", config=None, **identity_changes):
        config = config or FixtureConfig()
        identity = dict(version=1, variant=variant, config=asdict(config),
                        model_kwargs={}, widths=self.module.EXPECTED_WIDTHS, n_outputs=3,
                        inputs={"rows_sha256": "b" * 64}, environment={"python": "fixture"},
                        code_sha256={"train.py": "c" * 64})
        identity.update(identity_changes)
        return dict(benchmark="ecoli_proteins", variant=variant, config=asdict(config),
                    identity=identity, epochs=1, best_epoch=0, val_score=0.5, test_score=0.4,
                    history=[{"epoch": 0, "val_score": 0.5}])

    def save(self, record):
        config = FixtureConfig(**record["config"])
        path = self.module.result_path(self.folder, "ecoli_proteins", record["variant"], config)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_identity_reads_extra_fields_and_never_fabricates_missing_fields(self):
        identity = {"version": 1}
        record = SimpleNamespace(extra_fields={"identity": identity})
        self.assertIs(self.module.recorded_identity(record), identity)
        for missing in (SimpleNamespace(), SimpleNamespace(extra_fields={}),
                        SimpleNamespace(extra_fields={"identity": None})):
            with self.assertRaisesRegex(ValueError, "Missing or invalid"):
                self.module.recorded_identity(missing)

    def test_execute_refuses_before_cache_model_or_output_access(self):
        with patch.object(self.module, "preview_runs", side_effect=AssertionError), \
                patch.object(self.module, "write_plan", side_effect=AssertionError):
            with self.assertRaisesRegex(RuntimeError, "execution is disabled"):
                self.module.main(["--benchmark", "ecoli_proteins", "--cache-root", "absent",
                                  "--results-dir", "absent", "--execute"])
        self.module.blf.BenchmarkStreams.open.assert_not_called()

    def test_current_trainer_minimal_identity_is_not_promoted_to_full_identity(self):
        record = self.result()
        record["identity"] = {key: record["identity"][key] for key in ("config", "variant")}
        self.save(record)
        report = self.module.preview_runs("ecoli_proteins", [("unimodal:dna", FixtureConfig())], self.folder)
        self.assertEqual(report[0]["status"], "blocked_preserve")
        self.assertIn("Incomplete", report[0]["detail"])

    def test_metadata_preview_preserves_bytes_and_distinguishes_missing(self):
        path = self.save(self.result())
        before = path.read_bytes()
        runs = [("unimodal:dna", FixtureConfig()), ("concat", FixtureConfig())]
        report = self.module.preview_runs("ecoli_proteins", runs, self.folder)
        self.assertEqual([r["status"] for r in report], ["recorded_evidence_available", "not_found_locally"])
        self.assertEqual(path.read_bytes(), before)
        self.assertIn("not independently verified", report[0]["detail"])

    def test_changed_producer_cohort_is_excluded(self):
        self.save(self.result())
        self.save(self.result("concat", environment={"python": "different producer"}))
        report = self.module.preview_runs("ecoli_proteins",
                                         [("unimodal:dna", FixtureConfig()), ("concat", FixtureConfig())],
                                         self.folder)
        self.assertEqual(report[1]["status"], "blocked_preserve")

    def test_checkpoint_presence_never_claims_resumability_or_loads_pickle(self):
        self.save(self.result())
        path = self.module.result_path(self.folder, "ecoli_proteins", "concat", FixtureConfig())
        checkpoint = path.with_suffix(".checkpoint")
        checkpoint.write_bytes(b"fixture: not a pickle")
        report = self.module.preview_runs("ecoli_proteins",
                                         [("unimodal:dna", FixtureConfig()), ("concat", FixtureConfig())],
                                         self.folder)
        self.assertEqual(report[1]["status"], "checkpoint_requires_review")
        self.assertEqual(checkpoint.read_bytes(), b"fixture: not a pickle")


class PhaseAExporterTests(unittest.TestCase):
    def setUp(self):
        # Import must not parse CLI arguments, create an output folder or import plotting/models.
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "pandas", "matplotlib", "shared_code"}:
                raise AssertionError("Exporter imported runtime dependencies at import time")
            return original_import(name, *args, **kwargs)

        with patch("argparse.ArgumentParser.parse_args", side_effect=AssertionError), \
                patch("pathlib.Path.mkdir", side_effect=AssertionError), \
                patch("builtins.__import__", side_effect=guard):
            self.module = import_file("phasea_exporter_fixture", EXPORTER)
        self.called = []

        def forbidden_original(*args, **kwargs):
            self.called.append("research")
            raise AssertionError("Original research function should never be reached")

        result_type = type("Result", (), {"save": forbidden_original})
        self.runtime = SimpleNamespace(
            train=SimpleNamespace(run=forbidden_original, fit=forbidden_original,
                                  build=forbidden_original, Result=result_type,
                                  plot=SimpleNamespace(show=lambda: None)),
            blf=SimpleNamespace(build=forbidden_original, loader=forbidden_original,
                                BenchmarkStreams=SimpleNamespace(open=forbidden_original)),
            tokens=SimpleNamespace(extract=forbidden_original, summarize=forbidden_original,
                                   open_shard=forbidden_original),
            encoders=SimpleNamespace(load_encoder=forbidden_original),
        )
        self.book = {"cells": [dict(cell_type="code", id=name, source=["pass\n"])
                               for name in self.module.EXPECTED_CODE_CELLS]}
        self.book["cells"][0]["source"] = ["plt = train.plot\n"]
        self.book["cells"][2]["source"] = [
            'BENCHMARK = "fungal_expression"\nRUN_TRAINING = False\nREAD_SAVED_RESULTS = True\nCACHE_ROOT = None\n'
        ]

    def evaluate(self, book=None):
        return self.module.evaluate_notebook(book or self.book, self.runtime, Path("fixture-cache"))

    def test_export_target_is_appendix(self):
        self.assertEqual(self.module.NOTEBOOK_RELATIVE.name, "tier2_blf_phaseA_appendix.ipynb")

    def test_synthetic_cells_receive_only_in_memory_settings_override(self):
        before = deepcopy(self.book)
        values = self.evaluate()
        self.assertEqual(values["BENCHMARK"], "ecoli_proteins")
        self.assertEqual(values["CACHE_ROOT"], Path("fixture-cache"))
        self.assertFalse(values["RUN_TRAINING"])
        self.assertFalse(values["READ_SAVED_RESULTS"])
        self.assertEqual(self.book, before)

    def test_first_cell_training_and_cache_calls_are_already_blocked(self):
        for call in ("train.run()", "train.fit()", "blf.BenchmarkStreams.open()",
                     "tokens.extract()", "encoders.load_encoder()", "train.Result().save(None)"):
            with self.subTest(call=call):
                self.book["cells"][0]["source"] = [call]
                with self.assertRaisesRegex(RuntimeError, "must not load"):
                    self.evaluate()
                self.assertEqual(self.called, [])

    def test_changed_layout_fails_before_first_cell(self):
        self.book["cells"][0]["source"] = ["raise AssertionError('executed')"]
        self.book["cells"].pop()
        with self.assertRaisesRegex(ValueError, "structure changed"):
            self.evaluate()

    def test_enabled_or_ambiguous_training_setting_fails_before_first_cell(self):
        self.book["cells"][0]["source"] = ["raise AssertionError('executed')"]
        for setting in ("RUN_TRAINING = True", "RUN_TRAINING = False\nRUN_TRAINING = False"):
            self.book["cells"][2]["source"] = ['BENCHMARK = "ecoli_proteins"\n' + setting]
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_missing_guard_interface_fails_before_first_cell(self):
        del self.runtime.tokens.extract
        self.book["cells"][0]["source"] = ["raise AssertionError('executed')"]
        with self.assertRaises(AttributeError):
            self.evaluate()

    def test_dynamic_reload_is_not_in_reviewed_execution_contract(self):
        self.book["cells"][0]["source"] = ["reload(train)"]
        with self.assertRaisesRegex(ValueError, "Dynamic code"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
