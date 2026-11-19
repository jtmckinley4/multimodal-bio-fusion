"""Focused execution guards with invented files and mocked notebook kernels."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "execution/arcc/encoder_swaps"
if not RECIPE.is_dir():
    RECIPE = ROOT


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runtime = module("ernie_runtime_test", RECIPE / "run_encoder_swaps.py")
prepare = module("ernie_prepare_test", RECIPE / "prepare.py")


class ErnieExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.root = self.parent / "extension"
        self.root.mkdir()
        self.reference = self.parent / "completed"
        (self.reference / "outputs/phaseB/ecoli_proteins/rinalmo").mkdir(parents=True)
        self.cache = self.reference / "Code/stage3/embeddings"
        self.output = self.root / "outputs/ecoli_ernie"
        self.new_cache = self.root / "extension_embeddings"
        self.notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("EXPERIMENT = 'full'", id="swaps-settings",
                                      execution_count=4, outputs=[nbformat.v4.new_output(
                                          "stream", name="stdout", text="historical fixture")]),
        ])
        self.put(runtime.NOTEBOOK, nbformat.writes(self.notebook))
        for name in ("run_encoder_swaps.py", "Code/shared_code/fusion/swap_plan.py",
                     "Code/shared_code/fusion/result_sets.py"):
            self.put(name, "# fixture only\n")
        self.put("datasets/E.Coli_proteins.csv", "Sequence,Value\nAAA,1\n")
        self.put("historical_results/Code/stage4/results/phaseA/ecoli_proteins/mil.json", "{}")
        self.manifest = {
            "schema_version": 2, "experiment": "ecoli_ernie", "source_commit": "fixture",
            "files": [{"destination": path.relative_to(self.root).as_posix(),
                       "initial_copy_sha256": runtime.sha256(path)}
                      for path in self.root.rglob("*") if path.is_file()],
        }
        self.put("source-snapshot.json", json.dumps(self.manifest))
        self.config = {
            "workspace": str(self.root), "output_root": str(self.output),
            "cache_root": str(self.cache), "new_cache_root": str(self.new_cache),
            "reference_root": str(self.reference), "experiment": "ecoli_ernie",
            "mode": "extract", "purpose": "newton", "device": "cuda",
            "benchmarks": ["ecoli_proteins"], "seed": 0, "expected_gpu": "H100",
        }

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_extension_paths_cannot_write_into_completed_cache_or_result_tree(self):
        _, files = runtime.verify_preserved_inputs(self.root)
        runtime.validate_extension(self.root, self.output, self.cache, self.new_cache, files)
        for new_cache in (self.cache, self.cache / "new", self.output, self.root,
                          self.root / "Code/new", self.parent / "external"):
            with self.subTest(path=new_cache), self.assertRaises(ValueError):
                runtime.validate_extension(self.root, self.output, self.cache, new_cache, files)
        for output in (self.root / "outputs", self.root / "pilot_outputs"):
            with self.assertRaisesRegex(ValueError, "own output"):
                runtime.validate_extension(self.root, output, self.cache, self.new_cache, files)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.new_cache.exists())

    def test_profile_mismatch_is_refused_before_any_kernel_or_writes(self):
        with self.assertRaisesRegex(ValueError, "experiment differs"):
            runtime.NotebookExecution({**self.config, "experiment": "full"})
        with self.assertRaisesRegex(ValueError, "another benchmark"):
            runtime.NotebookExecution({**self.config, "benchmarks": ["fungal_expression"]})
        self.assertFalse(self.output.exists())

    def test_extraction_requires_a_gpu_allocation_before_opening_inputs(self):
        execution = runtime.NotebookExecution(self.config)
        fake_torch = Mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "Slurm"):
            execution.observe_kernel(fake_torch)
        fake_torch.cuda.is_available.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_extraction_and_training_inject_distinct_guards_without_mutating_notebook(self):
        original = deepcopy(self.notebook)
        for mode in ("extract", "train"):
            with self.subTest(mode=mode):
                fake = ModuleType("run_encoder_swaps")
                execution = Mock(device="cuda", output=self.output)
                execution.result_paths.return_value = type("Routes", (), {
                    "swaps": self.output / "phaseB", "originals": {},
                    "analysis_dir": self.output / "analyses"})()
                fake.NotebookExecution = Mock(return_value=execution)
                prepared = runtime.prepare_notebook(self.notebook, {**self.config, "mode": mode})
                injected = next(c.source for c in prepared.cells if c.id == "execution-settings")
                namespace = {"torch": object()}
                with patch.dict(sys.modules, {"run_encoder_swaps": fake}), \
                        patch.object(sys, "path", list(sys.path)), redirect_stdout(StringIO()):
                    exec(compile(injected, "fixture-settings", "exec"), namespace)
                self.assertFalse(namespace["READ_SAVED_RESULTS"])
                self.assertEqual(namespace["RUN_TRAINING"], mode == "train")
                self.assertEqual(namespace["CACHE_REUSE_ONLY"], mode == "train")
                self.assertFalse(namespace["RUN_SAVED_COMPARISONS"])
                self.assertEqual(namespace["SWAPS"], [("rna", "rnaernie"), ("rna", "ernierna")])
                self.assertEqual(namespace["NEW_CACHE_ROOT"], self.new_cache)
                self.assertEqual(namespace["REFERENCE_SWAP_ROOT"], self.reference / "outputs/phaseB")
        self.assertEqual(self.notebook, original)

    def test_extract_cli_is_offline_and_does_not_schedule_training_seed(self):
        with patch.object(runtime, "WORKSPACE", self.root), \
                patch.object(runtime, "LiveOutputPreprocessor") as processor, \
                patch.object(runtime, "acquire_locks") as locks, \
                patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), redirect_stdout(StringIO()):
            def inspect(notebook, resources):
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                self.assertEqual(resources["metadata"]["path"], str(self.root))
            processor.return_value.preprocess.side_effect = inspect
            runtime.main(["--mode", "extract", "--kernel", "fixture", "--expected-gpu", "H100",
                          "--reference-root", str(self.reference), "--cache-root", str(self.cache)])
        processor.assert_called_once()
        self.assertEqual(locks.call_args_list[0].args[-1], (0, 1, 2))
        self.assertEqual(locks.call_args_list[1].args[1], self.new_cache / ".locks")
        records = list((self.output / "logs").glob("*.json"))
        self.assertEqual(len(records), 1)
        saved = json.loads(records[0].read_text())
        self.assertEqual(saved["configuration"]["mode"], "extract")
        self.assertEqual(saved["configuration"]["experiment"], "ecoli_ernie")
        self.assertFalse((self.output / "phaseB").exists())

    def test_extension_defaults_to_only_seed_zero_and_keeps_output_separate(self):
        with patch.object(runtime, "WORKSPACE", self.root), \
                patch.object(runtime, "LiveOutputPreprocessor"), \
                patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), redirect_stdout(StringIO()):
            runtime.main(["--mode", "train", "--kernel", "fixture", "--expected-gpu", "H100",
                          "--reference-root", str(self.reference), "--cache-root", str(self.cache)])
        saved = json.loads(next((self.output / "logs").glob("*.json")).read_text())
        self.assertEqual(saved["configuration"]["seed"], 0)
        self.assertEqual(saved["configuration"]["benchmarks"], ["ecoli_proteins"])
        self.assertEqual(saved["configuration"]["output_root"], str(self.output))

    def test_unsupported_report_other_benchmark_and_extraction_seed_fail_early(self):
        options = [
            ["--mode", "report"],
            ["--mode", "train", "--benchmark", "fungal_expression"],
            ["--mode", "extract", "--seed", "2"],
            ["--mode", "train", "--experiment", "full"],
        ]
        for arguments in options:
            with self.subTest(arguments=arguments), patch.object(runtime, "WORKSPACE", self.root), \
                    patch.object(runtime, "LiveOutputPreprocessor") as processor, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                runtime.main(arguments)
            processor.assert_not_called()
        self.assertFalse(self.output.exists())


class ErniePackageTests(unittest.TestCase):
    def test_focused_inventory_copies_only_ecoli_data_and_original_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, selection = root / "repo", root / "old"
            source.mkdir()
            selection.mkdir()
            for path in prepare.REQUIRED_SHARED:
                target = source / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# synthetic\n")
            def entry(owner, name, destination, origin, role):
                return {"source": name, "destination": destination, "source_origin": origin,
                        "role": role, "source_sha256": "a" * 64,
                        "initial_copy_sha256": "a" * 64, "bytes": 1}
            selected = [entry(selection, name, name, "historical_selection_workspace", role)
                        for name, role in (
                            ("datasets/E.Coli_proteins.csv", "dataset"),
                            ("datasets/Fungal_expression.csv", "dataset"),
                            ("historical_results/Code/stage4/results/phaseA/ecoli_proteins/mil.json", "historical_selection"),
                            ("historical_results/Code/stage4/results/phaseB/ecoli_proteins/rinalmo/mil.json", "historical_selection"),
                            (prepare.SELECTION_EVIDENCE, "selection_manifest"))]
            with patch.object(prepare, "git_state", return_value={"source_commit": "fixture"}), \
                    patch.object(prepare, "_entry", side_effect=entry), \
                    patch.object(prepare, "_selection_entries", return_value=(selected, {})):
                plan = prepare.build_plan(source, selection, experiment="ecoli_ernie")
            names = {item["destination"] for item in plan["files"]}
            self.assertIn("datasets/E.Coli_proteins.csv", names)
            self.assertNotIn("datasets/Fungal_expression.csv", names)
            self.assertNotIn(selected[3]["destination"], names)
            self.assertIn(prepare.SELECTION_EVIDENCE, names)
            self.assertEqual(plan["declared_workload"]["fits_per_seed"], 4)
            self.assertEqual(plan["declared_workload"]["default_training_seed"], 0)
            self.assertFalse(plan["research_executed"])
            self.assertFalse(plan["cache_reference"]["copied"])


if __name__ == "__main__":
    unittest.main()
