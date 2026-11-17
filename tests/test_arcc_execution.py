"""Check prepared-workspace execution with invented files and mocked kernels.

No kernel, research dataset, cache array, checkpoint deserialization, fit or Slurm
job is started. Lock fixtures use the host OS on temporary files in tests/.
"""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np  # Preserve NumPy-before-PyTorch on Windows.
import pandas as pd
import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import run_identity, train

RUNTIME_PATH = ROOT / "execution/arcc/encoder_swaps/run_encoder_swaps.py"
if not RUNTIME_PATH.is_file():
    RUNTIME_PATH = ROOT / "run_encoder_swaps.py"
SPEC = importlib.util.spec_from_file_location("arcc_execution_under_test", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class ARCCExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".arcc-execution-", dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "outputs"
        self.cache = self.root / "external_cache"
        self.launch = self.output / "logs/fixture-launch.json"
        self.selection = (
            "historical_results/Code/stage4/results/phaseA/fungal_expression/first.json",
            "historical_results/Code/stage4/results/phaseB/fungal_expression/fixture/second.json",
        )
        self.notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell("Invented notebook", id="fixture-title"),
            nbformat.v4.new_code_cell(
                "SCIENCE_SETTING = 32\nREAD_SAVED_RESULTS = True", id="swaps-settings",
                execution_count=7, outputs=[nbformat.v4.new_output(
                    "stream", name="stdout", text="retained historical fixture\n")],
                metadata={"execution": {"iopub.status.busy": "old"}, "keep": "yes"}),
            nbformat.v4.new_code_cell("raise AssertionError('kernel must never execute')",
                                      id="fixture-producer"),
        ], metadata={"custom": "retained notebook metadata"})
        self.put(runtime.NOTEBOOK, nbformat.writes(self.notebook))
        for name in ("run_encoder_swaps.py", "Code/shared_code/fusion/swap_plan.py",
                     "Code/shared_code/fusion/result_sets.py"):
            self.put(name, "# Invented packaged source; never imported.\n")
        self.put("datasets/fixture.csv", "Sequence,Value\nAAA,1\n")
        for name in self.selection:
            self.put(name, '{"synthetic_selection": true}\n')
        self.manifest = {
            "schema_version": 2, "source_commit": "invented-source-commit",
            "files": [{"destination": path.relative_to(self.root).as_posix(),
                       "initial_copy_sha256": runtime.sha256(path)}
                      for path in sorted(self.root.rglob("*")) if path.is_file()],
        }
        self.save_manifest()
        self.config = {
            "workspace": str(self.root), "output_root": str(self.output),
            "cache_root": str(self.cache), "mode": "train", "purpose": "newton",
            "device": "cuda", "benchmarks": ["fungal_expression"], "seed": 0,
            "expected_gpu": "H100", "launch_record": str(self.launch),
        }
        self.environment = {
            "device": "cuda", "gpu": "NVIDIA H100 fixture", "host": "fixture-host",
            "python": "fixture-python", "packages": {"torch": "fixture"},
        }

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def save_manifest(self):
        self.put("source-snapshot.json", json.dumps(self.manifest))

    def refresh_inventory_hash(self, relative):
        for entry in self.manifest["files"]:
            if entry["destination"] == relative:
                entry["initial_copy_sha256"] = runtime.sha256(self.root / relative)
        self.save_manifest()

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def execution(self, **changes):
        return runtime.NotebookExecution({**self.config, **changes})

    def write_launch(self, *, status="completed", **fields):
        record = {
            "status": status, "configuration": deepcopy(self.config),
            "snapshot_sha256": runtime.sha256(self.root / "source-snapshot.json"),
            "kernel_environment": deepcopy(self.environment), **fields,
        }
        self.put(self.launch.relative_to(self.root), json.dumps(record))
        return record

    def producer_record(self, execution=None):
        execution = execution or self.execution()
        execution.environment = deepcopy(self.environment)
        self.write_launch()
        plan = SimpleNamespace(selected_singles={"fixture": "unimodal:dna"},
                               selection_sources={str(self.root / p): runtime.sha256(self.root / p)
                                                  for p in self.selection})
        with patch.object(run_identity, "producer_record", return_value={"synthetic": True}):
            producer = execution.producer(plan)
        return execution, {
            "benchmark": "fungal_expression", "config": {"seed": 0}, "producer": producer,
        }

    def fake_torch(self, *, available=True, gpu="NVIDIA H100 fixture"):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=available),
                                 get_device_name=Mock(return_value=gpu)),
            version=SimpleNamespace(cuda="fixture-cuda"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=Mock(return_value=1))),
        )

    def test_verify_only_checks_all_files_without_kernel_or_output_creation(self):
        before = self.snapshot()
        with patch.object(runtime, "WORKSPACE", self.root), \
                patch.object(runtime, "LiveOutputPreprocessor") as processor, \
                patch.object(runtime, "save_record") as save, \
                patch.object(runtime, "acquire_locks") as locks, redirect_stdout(StringIO()):
            runtime.main(["--verify-only"])
        processor.assert_not_called()
        save.assert_not_called()
        locks.assert_not_called()
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.output.exists())

    def test_manifest_rejects_old_schema_empty_missing_duplicate_and_tampered_entries(self):
        original = deepcopy(self.manifest)
        changes = [
            lambda d: d.update(schema_version=1),
            lambda d: d.update(files=[]),
            lambda d: d["files"].append(deepcopy(d["files"][0])),
            lambda d: d["files"].append({"destination": "missing.txt", "initial_copy_sha256": "a" * 64}),
            lambda d: d["files"][0].update(initial_copy_sha256="b" * 64),
            lambda d: d.update(files=[e for e in d["files"] if e["destination"] != runtime.NOTEBOOK]),
            lambda d: d.update(files=[e for e in d["files"] if not e["destination"].startswith("datasets/")]),
            lambda d: d.update(files=[e for e in d["files"] if not e["destination"].startswith("historical_results/")]),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                self.manifest = deepcopy(original)
                change(self.manifest)
                self.save_manifest()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    runtime.verify_preserved_inputs(self.root)
                self.assertFalse(self.output.exists())

    def test_manifest_rejects_nonportable_or_escaping_source_paths(self):
        for name in ("../outside", "/absolute", "C:/absolute", "Code\\outside", "", "."):
            with self.subTest(path=name), self.assertRaises(ValueError):
                runtime.package_path(self.root, name)
        self.manifest["files"][0]["destination"] = "../outside"
        self.save_manifest()
        with self.assertRaises(ValueError):
            runtime.verify_preserved_inputs(self.root)

    def test_tampered_source_is_preserved_and_refused(self):
        path = self.root / runtime.NOTEBOOK
        path.write_text("changed fixture source", encoding="utf-8")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "changed"):
            runtime.verify_preserved_inputs(self.root)
        self.assertEqual(before, self.snapshot())

    def test_destinations_keep_newton_pilots_inputs_and_caches_separate(self):
        _, files = runtime.verify_preserved_inputs(self.root)
        runtime.validate_destinations(self.root, self.output, self.cache, files, "newton")
        runtime.validate_destinations(self.root, self.root / "pilot_outputs", self.cache, files, "local_pilot")
        for purpose, output in (
            ("newton", self.root), ("newton", self.root.parent / "outside"),
            ("newton", self.cache), ("newton", self.cache / "nested"),
            ("newton", self.root / "Code/generated"), ("newton", self.root / "datasets/generated"),
            ("newton", self.root / "historical_results/generated"), ("newton", self.root / "evidence/generated"),
            ("newton", self.root / "pilot_outputs"), ("local_pilot", self.output),
            ("local_pilot", self.output / "nested"),
        ):
            with self.subTest(purpose=purpose, output=output), self.assertRaises(ValueError):
                runtime.validate_destinations(self.root, output, self.cache, files, purpose)
        self.assertFalse(self.output.exists())

    def test_result_routes_never_use_historical_evidence_as_fresh_results(self):
        routes = self.execution().result_paths()
        self.assertTrue(routes.fixed_historical_selection)
        self.assertEqual(routes.swaps, self.output / "phaseB")
        self.assertEqual(routes.originals["fungal_expression"], [self.output / "phaseA/fungal_expression"])
        self.assertEqual(routes.selection_swaps, self.root / "historical_results/Code/stage4/results/phaseB")
        self.assertFalse(self.output.exists())

    def test_kernel_observation_rejects_cuda_fallback_wrong_gpu_and_unallocated_newton(self):
        for torch_fixture in (self.fake_torch(available=False), self.fake_torch(gpu="other GPU")):
            with self.subTest(torch=torch_fixture), patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}):
                with self.assertRaises(RuntimeError):
                    self.execution().observe_kernel(torch_fixture)
                self.assertFalse(self.launch.exists())
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "Slurm"):
            self.execution().observe_kernel(self.fake_torch())
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.execution(device="cpu").observe_kernel(self.fake_torch())

    def test_actual_kernel_environment_is_recorded_and_report_mode_stays_on_cpu(self):
        for mode, device in (("train", "cuda"), ("report", "cpu")):
            with self.subTest(mode=mode):
                self.write_launch(status="running")
                fake = self.fake_torch()
                execution = self.execution(mode=mode, device=device)
                with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), \
                        patch("importlib.metadata.version", return_value="fixture"), redirect_stdout(StringIO()):
                    execution.observe_kernel(fake)
                saved = json.loads(self.launch.read_text(encoding="utf-8"))
                self.assertEqual(saved["kernel_environment"]["device"], device)
                self.assertEqual(saved["kernel_environment"], execution.environment)
                if mode == "report":
                    fake.cuda.get_device_name.assert_not_called()
                    fake.cuda.is_available.assert_not_called()

    def test_settings_injection_changes_only_an_executed_clone(self):
        original = deepcopy(self.notebook)
        prepared = runtime.prepare_notebook(self.notebook, self.config)
        self.assertEqual(self.notebook, original)
        self.assertEqual(prepared.metadata, original.metadata)
        self.assertEqual([c.id for c in prepared.cells],
                         ["fixture-title", "swaps-settings", "execution-settings", "fixture-producer"])
        by_id = {cell.id: cell for cell in prepared.cells}
        for source in original.cells:
            self.assertEqual(by_id[source.id].source, source.source)
            if source.cell_type == "code":
                self.assertEqual(by_id[source.id].outputs, [])
                self.assertIsNone(by_id[source.id].execution_count)
                self.assertNotIn("execution", by_id[source.id].metadata)
        self.assertEqual(by_id["swaps-settings"].metadata["keep"], "yes")
        injected = by_id["execution-settings"].source
        compile(injected, "synthetic-settings", "exec")
        self.assertIn("CACHE_REUSE_ONLY = True", injected)
        self.assertIn("CHECKPOINT_VALIDATOR = EXECUTION.checkpoint_preflight", injected)
        self.assertIn("FIT_PRODUCER =", injected)

    def test_injection_requires_one_settings_cell(self):
        missing = nbformat.v4.new_notebook(cells=[])
        duplicate = deepcopy(self.notebook)
        duplicate.cells.append(deepcopy(duplicate.cells[1]))
        for notebook in (missing, duplicate):
            with self.subTest(cells=len(notebook.cells)), self.assertRaises(ValueError):
                runtime.prepare_notebook(notebook, self.config)

    def test_injected_settings_reach_actual_notebook_plan_in_reading_order(self):
        # Use the actual packaged/maintained cell ordering. Only settings and helper
        # definitions execute; the planner and runtime are stubs, with no data access.
        path = ROOT / runtime.NOTEBOOK
        source_bytes = path.read_bytes()
        notebook = nbformat.reads(source_bytes.decode("utf-8"), as_version=4)
        original = deepcopy(notebook)
        selected_ids = {
            "1a72c669", "swaps-locations", "swaps-settings", "execution-settings",
            "swaps-selection", "swaps-plan-defs", "swaps-execution-plan",
        }
        for mode in ("train", "report"):
            with self.subTest(mode=mode):
                config = {**self.config, "mode": mode, "seed": 2,
                          "device": "cuda" if mode == "train" else "cpu"}
                routes = SimpleNamespace(
                    root=self.root, name="newton", swaps=self.output / "phaseB",
                    originals={"fungal_expression": [self.output / "phaseA/fungal_expression"]},
                    analysis_dir=self.output / "phaseB/analyses", fixed_historical_selection=True,
                    selection_originals={"fungal_expression": [
                        self.root / "historical_results/Code/stage3/results/fungal_expression"]},
                    selection_swaps=self.root / "historical_results/Code/stage4/results/phaseB",
                    selection_snapshot=None,
                )
                events = []

                class StubExecution:
                    def __init__(self, supplied):
                        self.device = supplied["device"]
                        self.output = Path(supplied["output_root"])
                        events.append("initialize")

                    def observe_kernel(self, supplied_torch):
                        events.append("observe_kernel")

                    def check_results(self):
                        events.append("check_results")

                    def result_paths(self):
                        events.append("result_paths")
                        return routes

                    def checkpoint_preflight(self, *args):
                        raise AssertionError("No checkpoint should be opened by this fixture")

                    def producer(self, plan):
                        return {"fixture_producer": True}

                fake_module = ModuleType("run_encoder_swaps")
                fake_module.NotebookExecution = StubExecution
                planned = SimpleNamespace(rows=lambda: [], issues=[], runs=[], require_ready=Mock())
                planner = Mock(return_value=planned)
                fake_torch = SimpleNamespace(
                    __version__="fixture", cuda=SimpleNamespace(
                        is_available=lambda: True, get_device_name=lambda: "H100 fixture"),
                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                )
                environment = {
                    "np": np, "pd": pd, "torch": fake_torch, "CODE_DIR": self.root / "Code",
                    "Path": Path, "display": Mock(), "train": train,
                    "tokens": SimpleNamespace(DEFAULT_CHUNK=256, grid_resolution=lambda key: "nucleotide"),
                    "blf": SimpleNamespace(ENCODER_OF_MODALITY={
                        "dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"}),
                    "swap_plan": SimpleNamespace(plan_benchmark=planner),
                }
                prepared = runtime.prepare_notebook(notebook, config)
                visited = []
                with patch.dict(sys.modules, {"run_encoder_swaps": fake_module}), \
                        patch.object(sys, "path", list(sys.path)), redirect_stdout(StringIO()):
                    for cell in prepared.cells:
                        if cell.cell_type == "code" and cell.id in selected_ids:
                            visited.append(cell.id)
                            exec(compile(cell.source, f"synthetic-flow:{cell.id}", "exec"), environment)
                self.assertEqual(set(visited), selected_ids)
                self.assertEqual(events, ["initialize", "observe_kernel", "check_results", "result_paths"])
                self.assertEqual(environment["READ_SAVED_RESULTS"], mode != "train")
                self.assertEqual(environment["RUN_SAVED_COMPARISONS"], mode == "report")
                self.assertIs(environment["CACHE_REUSE_ONLY"], True)
                self.assertIs(environment["RUN_BOOTSTRAP"], False)
                self.assertEqual(environment["SEED"], 2)
                self.assertEqual(environment["SELECTION_SEED"], 0)
                self.assertEqual(environment["BENCHMARKS"], ("fungal_expression",))
                arguments = planner.call_args.kwargs
                self.assertEqual(arguments["original_folders"], routes.originals["fungal_expression"])
                self.assertEqual(arguments["selection_original_folders"],
                                 routes.selection_originals["fungal_expression"])
                self.assertEqual(arguments["swap_root"], routes.swaps / "fungal_expression")
                self.assertEqual(arguments["checkpoint_validator"],
                                 environment["EXECUTION"].checkpoint_preflight if mode == "train" else None)
                self.assertEqual(arguments["identity_factory"],
                                 environment["requested_identity"] if mode == "train" else None)
                self.assertEqual(environment["FIT_PRODUCER"]("fungal_expression"),
                                 {"fixture_producer": True})
                self.assertEqual(notebook, original)
        self.assertEqual(path.read_bytes(), source_bytes)

    def test_os_locks_refuse_duplicate_writers_and_release_after_failure(self):
        directory = self.root / "locks"
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with ExitStack() as first:
                runtime.acquire_locks(first, directory, ("fixture",), (0,))
                with ExitStack() as second, self.assertRaisesRegex(RuntimeError, "Another writer"):
                    runtime.acquire_locks(second, directory, ("fixture",), (0,))
                # Independent seeds remain available while seed 0 is locked.
                with ExitStack() as other:
                    runtime.acquire_locks(other, directory, ("fixture",), (1,))
                raise RuntimeError("synthetic failure")
        with ExitStack() as later:
            runtime.acquire_locks(later, directory, ("fixture",), (0, 1))

    def test_partial_lock_acquisition_releases_earlier_locks_after_collision(self):
        directory = self.root / "locks"
        with ExitStack() as occupied:
            runtime.acquire_locks(occupied, directory, ("fixture",), (1,))
            with self.assertRaisesRegex(RuntimeError, "Another writer"):
                with ExitStack() as partial:
                    runtime.acquire_locks(partial, directory, ("fixture",), (0, 1))
            with ExitStack() as recovered:
                runtime.acquire_locks(recovered, directory, ("fixture",), (0,))

    def test_valid_producer_binds_selection_snapshot_and_observed_launch(self):
        execution, record = self.producer_record()
        before = self.snapshot()
        execution.require_producer(record)
        context = record["producer"]["arcc_execution"]
        self.assertEqual(context["selection_seed"], 0)
        self.assertEqual(context["selection_source_sha256"],
                         {p: runtime.sha256(self.root / p) for p in self.selection})
        self.assertEqual(context["kernel_environment"], self.environment)
        self.assertEqual(before, self.snapshot())

    def test_malformed_missing_and_wrong_cohort_producers_are_refused(self):
        execution, valid = self.producer_record()
        malformed = (None, [], {}, {"producer": None}, {"producer": []},
                     {"producer": {"arcc_execution": None}})
        for record in malformed:
            with self.subTest(record=record), self.assertRaises(ValueError):
                execution.require_producer(record)
        for field, value in (("source_snapshot_sha256", "wrong"), ("purpose", "local_pilot"),
                             ("output_root", "elsewhere"), ("selection_seed", 1),
                             ("launch_record", "../fixture-launch.json"),
                             ("launch_record", "missing-launch.json"),
                             ("kernel_environment", {"device": "cpu"}),
                             ("selection_source_sha256", {})):
            record = deepcopy(valid)
            record["producer"]["arcc_execution"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                execution.require_producer(record)

    def test_producer_selection_hashes_must_match_preserved_manifest_and_actual_files(self):
        execution, valid = self.producer_record()
        wrong_hash = deepcopy(valid)
        wrong_hash["producer"]["arcc_execution"]["selection_source_sha256"][self.selection[0]] = "0" * 64
        outside_selection = deepcopy(valid)
        outside_selection["producer"]["arcc_execution"]["selection_source_sha256"] = {
            "datasets/fixture.csv": runtime.sha256(self.root / "datasets/fixture.csv")}
        unknown = self.put("historical_results/Code/unlisted.json", "{}")
        unlisted = deepcopy(valid)
        unlisted["producer"]["arcc_execution"]["selection_source_sha256"] = {
            unknown.relative_to(self.root).as_posix(): runtime.sha256(unknown)}
        for record in (wrong_hash, outside_selection, unlisted):
            with self.subTest(record=record), self.assertRaises(ValueError):
                execution.require_producer(record)
        (self.root / self.selection[0]).write_text("changed after preflight", encoding="utf-8")
        with self.assertRaises(ValueError):
            execution.require_producer(valid)

    def test_producer_launch_context_and_fit_identity_must_match(self):
        execution, valid = self.producer_record()
        original_launch = json.loads(self.launch.read_text(encoding="utf-8"))
        for field, value in (("status", "unrecognized"), ("snapshot_sha256", "other-snapshot"),
                             ("kernel_environment", {"device": "cpu"})):
            changed = {**original_launch, field: value}
            self.launch.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                execution.require_producer(valid)
        self.launch.write_text(json.dumps(original_launch), encoding="utf-8")
        for change in ({"benchmark": "ecoli_proteins"}, {"config": {"seed": 1}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                execution.require_producer({**valid, **change})
        # Completed fits survive interruption of a later fit in the same job.
        self.write_launch(status="failed_or_interrupted")
        execution.require_producer(valid)

    def test_checkpoint_cohort_validation_precedes_full_recovery_preflight(self):
        execution, valid = self.producer_record()
        path = self.put("outputs/fixture.checkpoint", "invented bytes; never deserialize")
        requested = {"synthetic_identity": True}
        with patch.object(torch, "load", return_value=valid) as load, \
                patch.object(train, "checkpoint_preflight") as preflight:
            execution.checkpoint_preflight(path, requested)
        load.assert_called_once_with(path, map_location="cpu", weights_only=False)
        preflight.assert_called_once_with(path, requested)
        for payload in ({}, {"producer": {"arcc_execution": {"purpose": "wrong"}}}):
            with patch.object(torch, "load", return_value=payload), \
                    patch.object(train, "checkpoint_preflight") as preflight:
                with self.assertRaises(ValueError):
                    execution.checkpoint_preflight(path, requested)
                preflight.assert_not_called()

    def test_checkpoint_read_failure_never_reaches_full_recovery_preflight(self):
        execution = self.execution()
        failure = FileNotFoundError("missing invented checkpoint")
        with patch.object(torch, "load", side_effect=failure), \
                patch.object(train, "checkpoint_preflight") as preflight:
            with self.assertRaises(FileNotFoundError) as caught:
                execution.checkpoint_preflight(self.output / "missing.checkpoint", {})
        self.assertIs(caught.exception, failure)
        preflight.assert_not_called()

    def main_arguments(self):
        return ["--mode", "report", "--benchmark", "fungal_expression", "--kernel", "invented-kernel"]

    def test_launcher_success_and_interruption_keep_sources_and_record_terminal_status(self):
        for failure in (None, RuntimeError("synthetic kernel failure"), KeyboardInterrupt("synthetic stop")):
            with self.subTest(failure=failure):
                before = {p: b for p, b in self.snapshot().items() if not p.startswith("outputs/")}
                previous_logs = set((self.output / "logs").glob("*.json"))
                def execute(notebook, resources):
                    self.assertEqual(resources["metadata"]["path"], str(self.root))
                    notebook.cells[-1].execution_count = 1
                    notebook.cells[-1].outputs = [nbformat.v4.new_output(
                        "stream", name="stdout", text="synthetic execution only\n")]
                    current = (set((self.output / "logs").glob("*.json")) - previous_logs).pop()
                    record = json.loads(current.read_text(encoding="utf-8"))
                    record["kernel_environment"] = {"device": "cpu", "host": "fixture"}
                    current.write_text(json.dumps(record), encoding="utf-8")
                    if failure:
                        raise failure
                    return notebook, resources
                processor = SimpleNamespace(preprocess=Mock(side_effect=execute))
                original_env = {k: os.environ.get(k) for k in (
                    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")}
                with patch.object(runtime, "WORKSPACE", self.root), \
                        patch.object(runtime, "LiveOutputPreprocessor", return_value=processor), \
                        redirect_stdout(StringIO()):
                    if failure:
                        with self.assertRaises(type(failure)) as caught:
                            runtime.main(self.main_arguments())
                        self.assertIs(caught.exception, failure)
                    else:
                        runtime.main(self.main_arguments())
                logs = set((self.output / "logs").glob("*.json")) - previous_logs
                self.assertEqual(len(logs), 1)
                record = json.loads(logs.pop().read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "failed_or_interrupted" if failure else "completed")
                self.assertIn("ended_utc", record)
                self.assertEqual(record["kernel_environment"], {"device": "cpu", "host": "fixture"})
                self.assertTrue((self.root / record["executed_notebook"]).is_file())
                self.assertEqual(before, {p: b for p, b in self.snapshot().items() if not p.startswith("outputs/")})
                self.assertEqual(original_env, {k: os.environ.get(k) for k in original_env})

    def test_executor_initialization_failure_never_leaves_running_record_or_lock(self):
        failure = RuntimeError("synthetic executor initialization failure")
        with patch.object(runtime, "WORKSPACE", self.root), \
                patch.object(runtime, "LiveOutputPreprocessor", side_effect=failure), \
                redirect_stdout(StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                runtime.main(self.main_arguments())
        self.assertIs(caught.exception, failure)
        logs = list((self.output / "logs").glob("*.json"))
        self.assertTrue(logs)
        self.assertTrue(all(json.loads(p.read_text())["status"] == "failed_or_interrupted" for p in logs))
        with ExitStack() as stack:
            runtime.acquire_locks(stack, self.output / ".locks", ("fungal_expression",), (0, 1, 2))

    def test_invalid_notebook_preparation_does_not_leave_running_records(self):
        path = self.root / runtime.NOTEBOOK
        path.write_text(nbformat.writes(nbformat.v4.new_notebook(cells=[])), encoding="utf-8")
        self.refresh_inventory_hash(runtime.NOTEBOOK)
        with patch.object(runtime, "WORKSPACE", self.root), \
                patch.object(runtime, "LiveOutputPreprocessor") as processor:
            with self.assertRaises(ValueError):
                runtime.main(self.main_arguments())
        processor.assert_not_called()
        self.assertTrue(all(json.loads(p.read_text())["status"] != "running"
                            for p in (self.output / "logs").glob("*.json")))

    def test_evidence_save_failure_does_not_hide_primary_execution_error(self):
        failure = RuntimeError("synthetic primary execution error")
        processor = SimpleNamespace(preprocess=Mock(side_effect=failure))
        with patch.object(runtime, "WORKSPACE", self.root), \
                patch.object(runtime, "LiveOutputPreprocessor", return_value=processor), \
                patch.object(nbformat, "write", side_effect=OSError("synthetic evidence write failure")), \
                redirect_stdout(StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                runtime.main(self.main_arguments())
        self.assertIs(caught.exception, failure)
        self.assertTrue(any("Saving execution evidence also failed" in note
                            for note in getattr(failure, "__notes__", [])))
        logs = list((self.output / "logs").glob("*.json"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(json.loads(logs[0].read_text())["status"], "failed_or_interrupted")


if __name__ == "__main__":
    unittest.main()
