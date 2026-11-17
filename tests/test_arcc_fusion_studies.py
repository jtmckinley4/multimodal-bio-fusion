"""Offline execution-boundary checks; no notebook kernel or research input."""

import copy
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "execution/arcc/fusion_studies"


def module(name):
    spec = importlib.util.spec_from_file_location("study_" + name, RECIPE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare, runner = module("prepare"), module("run_fusion_study")


class ExecutionTests(unittest.TestCase):
    def configuration(self, root):
        return dict(workspace=str(root), output_root=str(root / "outputs"),
            cache_root=str(root.parent / "existing-caches"), data_root=str(root.parent / "datasets"),
            reference_root=str(root.parent / "existing-results"),
            grouped_split=str(root.parent / "grouped/split.json"),
            region_data_root=str(root.parent / "mrnabench"),
            region_cache_root=str(root.parent / "region-caches"),
            study="correction", benchmark="fungal_expression", seed=0, action="train",
            expected_gpu="H100", snapshot_sha256="1" * 64,
            launch_record=str(root / "outputs/logs/launch.json"))

    def test_override_is_inserted_after_settings_and_preserves_source(self):
        notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("READ_SAVED_RESULTS = True",
                metadata={"tags": ["fusion-study-settings"]}, execution_count=7,
                outputs=[nbformat.v4.new_output("stream", text="retained")]),
            nbformat.v4.new_code_cell("consumer()")])
        original = copy.deepcopy(notebook)
        result = runner.prepare_notebook(notebook, self.configuration(Path("/test")))
        self.assertEqual(notebook, original)
        self.assertEqual(result.cells[2].source, "consumer()")
        self.assertIn("RUN_ANALYSIS = False", result.cells[1].source)
        self.assertIn("CONFIG = replace(BASE_CONFIG, seed=SEED)", result.cells[1].source)
        self.assertEqual(result.cells[0].outputs, [])
        self.assertIsNone(result.cells[0].execution_count)

    def test_ambiguous_settings_refused(self):
        notebook = nbformat.v4.new_notebook(cells=[])
        with self.assertRaisesRegex(ValueError, "one fusion-study-settings"):
            runner.prepare_notebook(notebook, {})

    def test_pre_kernel_preparation_failure_records_failure_without_writing_notebook(self):
        for failure_at in ("read", "prepare"):
            with self.subTest(failure_at=failure_at), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "source-snapshot.json").write_text("{}")
                config = self.configuration(root)
                failure = ValueError(f"Synthetic {failure_at} failure")
                argv = ["--study", "gates", "--benchmark", "cov_vac_degradation",
                        "--seed", "0", "--kernel", "synthetic", "--expected-gpu", "H100"]
                for key in ("data_root", "cache_root", "reference_root", "output_root"):
                    argv.extend(("--" + key.replace("_", "-"), config[key]))
                with patch.object(runner, "WORKSPACE", root), \
                     patch.object(runner, "verify_package", return_value=({}, {})), \
                     patch.object(runner.support, "acquire_locks"), \
                     patch.dict(runner.os.environ, {"SLURM_JOB_ID": "synthetic"}), \
                     patch.object(nbformat, "read", side_effect=failure if failure_at == "read" else None,
                                  return_value=nbformat.v4.new_notebook()), \
                     patch.object(runner, "prepare_notebook", side_effect=failure) as prepare_call, \
                     patch.object(nbformat, "write") as notebook_write, \
                     patch.object(runner.support, "LiveOutputPreprocessor") as kernel:
                    with self.assertRaises(ValueError) as caught:
                        runner.main(argv)
                self.assertIs(caught.exception, failure)
                kernel.assert_not_called()
                notebook_write.assert_not_called()
                self.assertEqual(prepare_call.call_count, 0 if failure_at == "read" else 1)
                records = list((root / "outputs/logs").glob("*.json"))
                self.assertEqual(len(records), 1)
                record = json.loads(records[0].read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "failed_or_interrupted")
                self.assertEqual(record["error"], f"ValueError: Synthetic {failure_at} failure")
                self.assertIn("ended_utc", record)
                self.assertFalse(list((root / "outputs/notebooks").iterdir()))

    def test_outputs_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.configuration(root)
            runner.validate_configuration(config, {})
            config["reference_root"] = config["output_root"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner.validate_configuration(config, {})

    def test_expansion_and_duplicate_extraction_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.configuration(Path(folder))
            for edits in ({"benchmark": "ecoli_proteins"}, {"seed": 3},
                          {"action": "extract"}, {"expected_gpu": ""},
                          {"study": "regions", "benchmark": "rnahl-human", "action": "extract", "seed": 2}):
                with self.subTest(edits=edits), self.assertRaises(ValueError):
                    runner.validate_configuration({**config, **edits}, {})

    def test_gpu_failure_happens_before_producer_publication(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with patch.dict(runner.os.environ, {"SLURM_JOB_ID": "fake"}):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                runner.observe_kernel(torch, self.configuration(Path("/test")))

    def test_source_inventory_excludes_payloads_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            root.mkdir()
            for name in prepare.NOTEBOOKS:
                path = root / f"Code/stage4/{name}.ipynb"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            for name in ("Code/shared_code/fusion/train.py", "Code/stage3/embeddings/shard.npz",
                         "datasets/large.csv", "Code/stage4/results/old.json",
                         "execution/arcc/encoder_swaps/run_encoder_swaps.py",
                         "execution/arcc/fusion_studies/run_fusion_study.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            plan = prepare.inventory(root)
            names = [item["destination"] for item in plan["files"]]
            self.assertFalse(any("datasets" in p or "embeddings" in p or "results" in p for p in names))
            destination = root.parent / "package"
            prepare.prepare(root, destination)
            runner.verify_package(destination)
            with self.assertRaisesRegex(ValueError, "existing destinations"):
                prepare.prepare(root, destination)
            (destination / "Code/shared_code/fusion/train.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed prepared source"):
                runner.verify_package(destination)

    def test_actual_settings_injection_binds_one_job_without_loading_inputs(self):
        @dataclass
        class FakeConfig:
            lr: float = 3e-5
            weight_decay: float = 1e-5
            batch_size: int = 32
            max_epochs: int = 500
            early_stopping_patience: int = 20
            plateau_patience: int = 5
            entropy_lambda: float = 0.
            channels: int = 100
            tconv: str = "equation"
            seed: int = 0

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                                    device=lambda value: value)
        for study, filename in runner.STUDIES.items():
            notebook = nbformat.read(ROOT / "Code/stage4" / f"{filename}.ipynb", as_version=4)
            actions = ("train",) if study == "correction" else ("train", "diagnostics")
            if study == "regions":
                actions += ("prepare", "extract")
            for action in actions:
                with self.subTest(study=study, action=action):
                    config = self.configuration(Path("/synthetic"))
                    config.update(study=study, benchmark=runner.PANELS[study][0], action=action,
                                  seed=0 if action in ("prepare", "extract") else 2,
                                  fit_keys=["rnafm__native__rna_only__true"] if study == "regions" and action in ("train", "diagnostics") else None)
                    prepared = runner.prepare_notebook(notebook, config)
                    index = next(i for i, cell in enumerate(prepared.cells)
                                 if "fusion-study-settings" in cell.metadata.get("tags", []))
                    self.assertEqual(prepared.cells[index + 1].id, "newton-study-execution")
                    producer = {"synthetic": True, "action": action}
                    observer = Mock(return_value=producer)
                    namespace = {"Path": Path, "replace": replace, "train": SimpleNamespace(TrainConfig=FakeConfig),
                                 "torch": fake_torch, "sys": sys, "PROJECT_DIR": Path("/repo"),
                                 "REPO_DIR": Path("/repo"), "CODE_DIR": Path("/repo/Code")}
                    # Only declarative settings and the injected boundary are evaluated,
                    # against fake modules. No scientific or input cell is executed.
                    with patch.dict(sys.modules, {"run_fusion_study": SimpleNamespace(observe_kernel=observer)}), \
                         patch.object(sys, "path", list(sys.path)), redirect_stdout(io.StringIO()):
                        exec(prepared.cells[index].source, namespace)
                        exec(prepared.cells[index + 1].source, namespace)
                    observer.assert_called_once_with(fake_torch, config)
                    self.assertIs(namespace["PRODUCER"], producer)
                    self.assertEqual(namespace["CONFIG"].seed, config["seed"])
                    self.assertEqual(namespace["BASE_CONFIG"].seed, 0)
                    self.assertEqual(namespace["BENCHMARK"], config["benchmark"])
                    self.assertEqual(namespace["DATASET_KEY"], config["benchmark"])
                    self.assertEqual(namespace["RESULTS_ROOT"], Path(config["output_root"]) / study)
                    self.assertEqual(namespace["DEVICE"], "cpu" if action == "prepare" else "cuda")
                    self.assertEqual(namespace["READ_SAVED_RESULTS"], action == "diagnostics")
                    for flag, enabled in (("RUN_TRAINING", "train"), ("RUN_EXTRACTION", "extract"),
                                          ("RUN_PREPARATION", "prepare"), ("RUN_DIAGNOSTICS", "diagnostics")):
                        self.assertEqual(namespace[flag], action == enabled)
                    self.assertFalse(namespace["RUN_ANALYSIS"])
                    if config["fit_keys"]:
                        self.assertEqual(namespace["FIT_KEYS"], config["fit_keys"])

    def test_later_top_level_assignments_do_not_restore_local_paths_or_modes(self):
        protected = {"DEVICE", "PRODUCER", "BENCHMARK", "DATASET_KEY", "SEED", "DATA_ROOT", "CACHE_ROOT",
                     "REFERENCE_ROOT", "GROUPED_SPLIT_PATH", "DATA_CACHE_ROOT", "TOKEN_CACHE_ROOT", "RESULTS_ROOT",
                     "READ_SAVED_RESULTS", "RUN_TRAINING", "RUN_EXTRACTION", "RUN_PREPARATION", "RUN_DIAGNOSTICS", "RUN_ANALYSIS"}
        for filename in runner.STUDIES.values():
            notebook = nbformat.read(ROOT / "Code/stage4" / f"{filename}.ipynb", as_version=4)
            index = next(i for i, cell in enumerate(notebook.cells)
                         if "fusion-study-settings" in cell.metadata.get("tags", []))
            for cell in notebook.cells[index + 1:]:
                if cell.cell_type != "code":
                    continue
                # Local names inside helper functions do not override notebook settings.
                assigned = {target.id for node in ast.parse(cell.source).body if isinstance(node, ast.Assign)
                            for target in node.targets if isinstance(target, ast.Name)}
                self.assertFalse(assigned & protected, (filename, assigned & protected))

    def test_seed_scoped_diagnostics_do_not_expand_to_training_seed_panel(self):
        for filename in ("gate_diagnostics", "anchor_cross_attention"):
            notebook = nbformat.read(ROOT / "Code/stage4" / f"{filename}.ipynb", as_version=4)
            cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"
                     and "if RUN_DIAGNOSTICS:" in cell.source and "collect_context_diagnostics(" in cell.source]
            self.assertEqual(len(cells), 1)
            parsed = ast.parse(cells[0])
            calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "collect_context_diagnostics"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.unparse(calls[0].args[2]), "SEED")
            self.assertFalse(any(isinstance(node, ast.For) and ast.unparse(node.iter) == "TRAINING_SEEDS"
                                 for node in ast.walk(parsed)))

    def test_prepare_uses_slurm_cpu_and_wrong_gpu_never_publishes(self):
        config = self.configuration(Path("/synthetic"))
        fake_torch = Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_name.return_value = "NVIDIA V100"
        with patch.dict(runner.os.environ, {"SLURM_JOB_ID": "synthetic"}):
            with patch.object(runner.support, "save_record") as publish:
                with self.assertRaisesRegex(RuntimeError, "Unexpected GPU"):
                    runner.observe_kernel(fake_torch, config)
                publish.assert_not_called()
        config.update(study="regions", benchmark="rnahl-human", action="prepare", expected_gpu="")
        fake_torch.cuda.is_available.return_value = False
        fake_torch.version.cuda = None
        fake_identity = SimpleNamespace(producer_record=Mock(return_value={"host": "synthetic"}))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launch.json"
            path.write_text('{"status": "running"}')
            config["launch_record"] = str(path)
            with patch.dict(runner.os.environ, {"SLURM_JOB_ID": "synthetic"}), \
                 patch.dict(sys.modules, {
                     "shared_code": SimpleNamespace(fusion=SimpleNamespace(run_identity=fake_identity)),
                     "shared_code.fusion": SimpleNamespace(run_identity=fake_identity),
                     "shared_code.fusion.run_identity": fake_identity,
                 }), redirect_stdout(io.StringIO()):
                producer = runner.observe_kernel(fake_torch, config)
            self.assertIsNone(producer["fusion_study_execution"]["gpu"])
            self.assertEqual(producer["fusion_study_execution"]["action"], "prepare")
            fake_identity.producer_record.assert_called_once_with(device="cpu", repository=config["workspace"])

    def test_a6_fit_subset_and_phase_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.configuration(Path(temporary))
            config.update(study="regions", benchmark="rnahl-human", fit_keys=["rnafm__native__rna_only__true"])
            runner.validate_configuration(config, {})
            for changes in ({"action": "extract"}, {"action": "prepare"}, {"fit_keys": []},
                            {"fit_keys": ["same", "same"]}, {"study": "gates", "benchmark": "ecoli_proteins"}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    runner.validate_configuration({**config, **changes}, {})

    def test_a6_writable_cache_must_not_overlap_sources_or_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.configuration(root)
            config.update(study="regions", benchmark="rnahl-human", action="extract")
            protected = [root / "Code", Path(config["region_data_root"]), Path(config["reference_root"]),
                         Path(config["grouped_split"]).parent, Path(config["data_root"])]
            for location in protected:
                with self.subTest(location=location), self.assertRaises(ValueError):
                    runner.validate_configuration({**config, "region_cache_root": str(location)}, {})


if __name__ == "__main__":
    unittest.main()
