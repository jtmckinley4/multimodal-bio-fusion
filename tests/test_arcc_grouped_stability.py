"""Invented package and mocked-kernel checks; no research inputs or fits are used."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nbformat


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "execution/arcc/grouped_stability"
if not RECIPE.is_dir():
    RECIPE = ROOT  # The preparation preserves flat runtime entrypoints.


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, RECIPE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("grouped_prepare", "prepare.py")
runtime = load("grouped_runtime", "run_grouped_stability.py")


class PackageCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "repo"
        self.source.mkdir()
        self.cache = self.base / "existing-cache"
        self.cache.mkdir()
        self.state = {"source_commit": "a" * 40, "source_dirty": False, "source_git_status": ""}
        self.put(prepare.NOTEBOOK, nbformat.writes(self.notebook()))
        self.put(prepare.DATASET, "Sequence,Value,Dataset,Split\nACG,1,fixture,train\n")
        for name in ("__init__.py", "fusion/grouped_stability.py", "fusion/run_identity.py"):
            self.put("Code/shared_code/" + name, "# synthetic package source\n")
        for name in prepare.RUNTIME:
            self.put(prepare.RECIPE + name, (RECIPE / name).read_text(encoding="utf-8"))
        for source, target in prepare.SUPPORT.items():
            # Reused utility copies must have their real bytes for packaged imports.
            path = ROOT / source
            if not path.is_file():
                path = ROOT / target
            self.put(source, path.read_text(encoding="utf-8"))
        for name in ("test_arcc_grouped_stability.py", "test_grouped_stability.py"):
            self.put("tests/" + name, "# synthetic fixture\n")
        # Presence of excluded payloads must not cause them to enter the package.
        self.put("Code/stage3/embeddings/mrna_stability/shard.npz", "DO NOT COPY")
        self.put("Code/stage4/results/old.json", '{"preserve": true}')
        self.git = patch.object(prepare.support, "git_state", return_value=self.state)
        self.git.start()
        self.addCleanup(self.git.stop)

    def put(self, name, content):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    @staticmethod
    def notebook():
        setup = nbformat.v4.new_code_cell(
            "SPLIT_SEED = 42\nTRAINING_SEEDS = (0, 1, 2)\nSELECTION_SEED = 0",
            metadata={"tags": ["grouped-stability-settings"]}, id="execution-settings",
        )
        setup.execution_count = 7
        setup.outputs = [nbformat.v4.new_output("stream", name="stdout", text="retained original")]
        return nbformat.v4.new_notebook(cells=[setup, nbformat.v4.new_code_cell("scientific_body()")])

    def package(self):
        plan = prepare.build_plan(self.source, self.cache)
        target = self.base / "new-package"
        prepare.write_package(plan, target)
        return target

    def config(self, target):
        return {"workspace": str(target.resolve()), "output_root": str((target / "outputs").resolve()),
                "cache_root": str(self.cache.resolve()), "seed": 0, "expected_gpu": "H100",
                "launch_record": str((target / "outputs/logs/fixture.json").resolve())}

    def test_inventory_excludes_existing_results_caches_and_selection(self):
        before = set(self.base.rglob("*"))
        plan = prepare.build_plan(self.source, self.cache)
        names = {item["destination"] for item in plan["files"]}
        self.assertEqual(before, set(self.base.rglob("*")))
        self.assertEqual({name for name in names if name.startswith("datasets/")}, {prepare.DATASET})
        self.assertFalse(any("embeddings/" in name or "results/" in name for name in names))
        self.assertFalse(any(name.startswith("historical_results/") for name in names))
        self.assertEqual(plan["training_seeds"], [0, 1, 2])
        self.assertEqual(plan["fits_per_seed"], 12)

    def test_package_verifies_and_never_replaces_existing_destination(self):
        target = self.package()
        snapshot, files = runtime.verify_package(target)
        self.assertEqual(snapshot["schema"], runtime.SCHEMA)
        self.assertIn(prepare.NOTEBOOK, files)
        before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            prepare.write_package(prepare.build_plan(self.source), target)
        self.assertEqual(before, {p.relative_to(target): p.read_bytes()
                                 for p in target.rglob("*") if p.is_file()})

    def test_flat_package_imports_its_hash_bound_support(self):
        target = self.package()
        spec = importlib.util.spec_from_file_location("packaged_grouped_runtime",
                                                      target / "run_grouped_stability.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(Path(module.support.__file__).resolve(),
                         (target / "arcc_execution_support.py").resolve())
        _, files = module.verify_package()
        self.assertIn(prepare.NOTEBOOK, files)

    def test_preparation_refuses_changed_source_and_cache_overlap(self):
        plan = prepare.build_plan(self.source)
        self.put(prepare.DATASET, "changed after planning\n")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            prepare.write_package(plan, self.base / "changed-package")
        self.assertFalse((self.base / "changed-package").exists())
        target = self.base / "new-package"
        plan = prepare.build_plan(self.source, target / "cache")
        with self.assertRaisesRegex(ValueError, "cache root"):
            prepare.write_package(plan, target)

    def test_verify_refuses_tampered_missing_and_wrong_experiment(self):
        target = self.package()
        dataset = target / prepare.DATASET
        saved = dataset.read_bytes()
        dataset.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared file changed"):
            runtime.verify_package(target)
        dataset.unlink()
        with self.assertRaises(FileNotFoundError):
            runtime.verify_package(target)
        dataset.write_bytes(saved)
        manifest = target / "source-snapshot.json"
        payload = json.loads(manifest.read_text())
        payload["schema"] = "encoder-swaps"
        manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "grouped-stability package"):
            runtime.verify_package(target)

    def test_injection_preserves_source_and_only_enables_training(self):
        original = self.notebook()
        before = copy.deepcopy(original)
        prepared = runtime.prepare_notebook(original, self.config(self.base))
        self.assertEqual(original, before)
        self.assertEqual(prepared.cells[0].source, original.cells[0].source)
        self.assertEqual(prepared.cells[2].source, original.cells[1].source)
        self.assertEqual(prepared.cells[0].outputs, [])
        self.assertEqual(prepared.cells[1].id, "arcc-grouped-execution")
        self.assertEqual(len({cell.id for cell in prepared.cells}), len(prepared.cells))
        injection = prepared.cells[1].source
        for text in ("READ_SAVED_RESULTS = False", "RUN_TRAINING = True",
                     "RUN_COMPARISONS = False", "SPLIT_SEED != 42", "SELECTION_SEED != 0",
                     "RESULT_VALIDATOR = EXECUTION.require_producer", " / 'study'"):
            self.assertIn(text, injection)
        self.assertNotIn("MAX_EPOCHS =", injection)
        self.assertNotIn("BATCH_SIZE =", injection)
        original.cells[0].metadata["tags"] = []
        with self.assertRaisesRegex(ValueError, "settings tag"):
            runtime.prepare_notebook(original, self.config(self.base))

    def test_actual_notebook_injection_keeps_scientific_cells_unchanged(self):
        notebook = nbformat.read(ROOT / runtime.NOTEBOOK, as_version=4)
        original = copy.deepcopy(notebook)
        prepared = runtime.prepare_notebook(notebook, self.config(self.base))
        nbformat.validate(prepared)
        self.assertEqual(notebook, original)
        retained = [cell for cell in prepared.cells if cell.id != "arcc-grouped-execution"]
        self.assertEqual([cell.source for cell in retained], [cell.source for cell in original.cells])
        index = next(i for i, cell in enumerate(prepared.cells)
                     if cell.id == "arcc-grouped-execution")
        self.assertIn("grouped-stability-settings", prepared.cells[index - 1].metadata["tags"])
        self.assertEqual(prepared.cells[index + 1].id, "mode-guard")

    def test_injected_boundary_checks_panel_then_sets_callbacks_without_fitting(self):
        config = self.config(self.base)
        prepared = runtime.prepare_notebook(self.notebook(), config)
        source = prepared.cells[1].source
        instance = Mock()
        constructor = Mock(return_value=instance)
        module = SimpleNamespace(NotebookExecution=constructor)
        namespace = {"SPLIT_SEED": 42, "TRAINING_SEEDS": (0, 1, 2), "SELECTION_SEED": 0,
                     "PANEL": [SimpleNamespace(key=f"fit{i}") for i in range(12)],
                     "sys": SimpleNamespace(path=[]), "Path": Path, "torch": object()}
        with patch.dict(sys.modules, {"run_grouped_stability": module}), \
                redirect_stdout(io.StringIO()):
            exec(source, namespace)
        constructor.assert_called_once_with(config)
        instance.observe_kernel.assert_called_once_with(namespace["torch"])
        instance.producer.assert_not_called()
        self.assertTrue(namespace["RUN_TRAINING"])
        self.assertFalse(namespace["RUN_COMPARISONS"])
        self.assertFalse(namespace["READ_SAVED_RESULTS"])
        self.assertEqual(namespace["RESULTS_ROOT"], Path(config["output_root"]) / "study")
        self.assertIs(namespace["FIT_PRODUCER"], instance.producer)
        self.assertIs(namespace["RESULT_VALIDATOR"], instance.require_producer)
        self.assertIs(namespace["CHECKPOINT_VALIDATOR"], instance.checkpoint_preflight)
        for changes in ({"SPLIT_SEED": 7}, {"TRAINING_SEEDS": (0, 1)},
                        {"PANEL": namespace["PANEL"][:11]}):
            constructor.reset_mock()
            with patch.dict(sys.modules, {"run_grouped_stability": module}):
                with self.assertRaises(ValueError):
                    exec(source, {**namespace, **changes})
            constructor.assert_not_called()

    def test_verify_only_never_starts_kernel_or_writes_evidence(self):
        target = self.package()
        before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
        with patch.object(runtime, "WORKSPACE", target), redirect_stdout(io.StringIO()), \
                patch.object(runtime.support, "LiveOutputPreprocessor") as processor:
            runtime.main(["--verify-only"])
        processor.assert_not_called()
        self.assertEqual(before, {p.relative_to(target): p.read_bytes()
                                 for p in target.rglob("*") if p.is_file()})

    def test_training_refuses_login_shell_and_unsafe_output_root(self):
        target = self.package()
        arguments = ["--seed", "0", "--kernel", "fixture", "--cache-root", str(self.cache),
                     "--expected-gpu", "H100"]
        with patch.object(runtime, "WORKSPACE", target), patch.dict(os.environ, {}, clear=True), \
                patch.object(runtime.support, "LiveOutputPreprocessor") as processor, \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runtime.main(arguments)
        processor.assert_not_called()
        with patch.object(runtime, "WORKSPACE", target), patch.dict(os.environ, {"SLURM_JOB_ID": "1"}):
            with self.assertRaises(ValueError):
                runtime.main(arguments + ["--output-root", str(target / "datasets")])

    def test_mocked_kernel_failure_preserves_source_and_records_failure(self):
        target = self.package()
        source = target / runtime.NOTEBOOK
        before = source.read_bytes()
        with patch.object(runtime, "WORKSPACE", target), patch.dict(os.environ, {"SLURM_JOB_ID": "1"}), \
                patch.object(runtime.support, "LiveOutputPreprocessor") as processor:
            processor.return_value.preprocess.side_effect = RuntimeError("synthetic kernel failure")
            with self.assertRaisesRegex(RuntimeError, "synthetic kernel failure"):
                runtime.main(["--seed", "1", "--kernel", "fixture", "--cache-root", str(self.cache),
                              "--expected-gpu", "H100"])
        self.assertEqual(source.read_bytes(), before)
        launches = list((target / "outputs/logs").glob("*.json"))
        self.assertEqual(len(launches), 1)
        self.assertEqual(json.loads(launches[0].read_text())["status"], "failed_or_interrupted")
        self.assertEqual(len(list((target / "outputs/notebooks").glob("*.ipynb"))), 1)
        self.assertFalse((target / "outputs/study").exists())

    def test_kernel_gpu_observation_and_reuse_bindings(self):
        target = self.package()
        config = self.config(target)
        execution = runtime.NotebookExecution(config)
        environment = {"device": "cuda", "gpu": "NVIDIA H100 fixture"}
        launch = {"status": "completed", "configuration": config,
                  "snapshot_sha256": execution.snapshot_sha,
                  "execution_source_sha256": execution.files, "kernel_environment": environment}
        Path(config["launch_record"]).write_text(json.dumps(launch))
        record = {"config": {"seed": 0}, "producer": {"grouped_stability_execution": {
            "source_snapshot_sha256": execution.snapshot_sha, "output_root": config["output_root"],
            "training_seed": 0, "launch_record": "fixture.json", "kernel_environment": environment}}}
        execution.require_producer(record)
        with self.assertRaisesRegex(ValueError, "cohort"):
            altered = copy.deepcopy(record)
            altered["producer"]["grouped_stability_execution"]["training_seed"] = 1
            execution.require_producer(altered)
        with self.assertRaisesRegex(ValueError, "kernel evidence"):
            altered = copy.deepcopy(record)
            altered["producer"]["grouped_stability_execution"]["kernel_environment"]["gpu"] = "different"
            execution.require_producer(altered)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "1"}):
            fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                execution.observe_kernel(fake)
            fake.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "V100")
            with self.assertRaisesRegex(RuntimeError, "differs from requested"):
                execution.observe_kernel(fake)

    def test_observed_environment_is_persisted_without_a_fit(self):
        target = self.package()
        config = self.config(target)
        execution = runtime.NotebookExecution(config)
        launch = {"status": "running", "configuration": config}
        record_path = Path(config["launch_record"])
        record_path.write_text(json.dumps(launch))
        fake = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True,
                                 get_device_name=lambda _: "NVIDIA H100 fixture"),
            version=SimpleNamespace(cuda="fixture CUDA"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 1)),
        )
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture-job"}), \
                patch("importlib.metadata.version", return_value="fixture-version"), \
                redirect_stdout(io.StringIO()):
            execution.observe_kernel(fake)
        observed = json.loads(record_path.read_text())["kernel_environment"]
        self.assertEqual(observed["gpu"], "NVIDIA H100 fixture")
        self.assertEqual(observed["device"], "cuda")
        self.assertEqual(observed["cuda"], "fixture CUDA")
        self.assertEqual(observed, execution.environment)
        self.assertFalse((target / "outputs/study").exists())

    def test_shell_templates_bootstrap_conda_before_module_and_never_install(self):
        for name in ("newton_grouped_stability.sbatch.example", "newton_preflight.sbatch.example"):
            text = (RECIPE / name).read_text(encoding="utf-8")
            self.assertLess(text.index('source "$ARCC_BOOTSTRAP_CONDA_BASE'), text.index("\nmodule load"))
            self.assertIn("run_grouped_stability.py --verify-only", text)
            self.assertNotIn("pip install", text)
            self.assertFalse(any(line.startswith("sbatch ") for line in text.splitlines()))
        batches = (RECIPE / "batches.tsv").read_text().strip().splitlines()[1:]
        self.assertEqual(batches, ["0\t0\t12", "1\t1\t12", "2\t2\t12"])


if __name__ == "__main__":
    unittest.main()
