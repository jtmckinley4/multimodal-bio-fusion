"""Source and guarded-fixture checks; neither research notebook is executed."""

import ast
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import controlled_experiment as experiment, train


NAMES = ("gate_diagnostics.ipynb", "anchor_cross_attention.ipynb")


def notebook(name):
    document = json.loads((ROOT / "Code/stage4" / name).read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in document["cells"]
             if cell["cell_type"] == "code"}
    return document, cells


class ControlledNotebookTests(unittest.TestCase):
    def fixture(self, name, root):
        _, cells = notebook(name)
        forbidden = Mock(side_effect=AssertionError("No research action in a reading fixture"))
        namespace = dict(CODE_DIR=root / "Code", Path=Path, train=train, replace=replace,
                         asdict=asdict, np=np, pd=pd, torch=torch, json=json, hashlib=hashlib,
                         print=Mock(), display=Mock(), experiment=experiment,
                         study_references=SimpleNamespace(read_reference=Mock(return_value=None),
                             collect_reference_diagnostics=forbidden),
                         study_inputs=SimpleNamespace(open_streams=forbidden),
                         correction_experiment=SimpleNamespace(_publish=forbidden),
                         study_selection=SimpleNamespace(select_anchor=forbidden, read_anchor=forbidden),
                         token_embeddings=SimpleNamespace(CacheVerification=Mock()))
        exec(cells["settings"], namespace)
        return cells, namespace, forbidden

    def test_structure_outputs_and_single_injection_boundary(self):
        for name in NAMES:
            with self.subTest(name=name):
                document, cells = notebook(name)
                self.assertEqual(len({cell["id"] for cell in document["cells"]}), len(document["cells"]))
                tagged = [cell for cell in document["cells"]
                          if "fusion-study-settings" in cell["metadata"].get("tags", [])]
                self.assertEqual(len(tagged), 1)
                self.assertEqual(tagged[0]["id"], "settings")
                assigned = {target.id for node in ast.walk(ast.parse(cells["settings"]))
                            if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
                self.assertTrue({"DATA_ROOT", "CACHE_ROOT", "REFERENCE_ROOT", "GROUPED_SPLIT_PATH",
                                 "RESULTS_ROOT", "READ_SAVED_RESULTS", "PRODUCER"} <= assigned)
                for cell in document["cells"]:
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), f"{name}:{cell['id']}", "exec")
                        self.assertEqual(cell["outputs"], [])
                        self.assertIsNone(cell["execution_count"])

    def test_default_reading_does_not_open_inputs_fit_infer_or_analyze(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, forbidden = self.fixture(name, Path(folder))
                with patch.object(torch.cuda, "is_available", side_effect=AssertionError("No GPU check while reading")):
                    exec(cells["execution-guards"], namespace)
                for identity in ("panel", "configuration-helpers", "parameter-preview", "reference-readers", "selection-helper",
                                 "saved-fit-readers", "training-helper", "diagnostic-helper",
                                 "report-source-helpers", "comparison-helper", "mechanism-contrast-helper", "interval-helper", "gate-summary-helper",
                                 "coverage", "training", "diagnostics", "analysis", "saved-report-plot",
                                 "gate-analysis", "saved-gate-plot", "saved-diagnostic-detail"):
                    exec(cells[identity], namespace)
                forbidden.assert_not_called()
                self.assertFalse(list(Path(folder).rglob("*")))

    def test_injected_paths_and_seed_survive_until_helpers(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                cells, namespace, _ = self.fixture(name, root)
                paths = {key: root / key for key in
                         ("DATA_ROOT", "CACHE_ROOT", "REFERENCE_ROOT", "GROUPED_SPLIT_PATH", "RESULTS_ROOT")}
                namespace.update(paths, SEED=2, BENCHMARK="cov_vac_degradation", DEVICE="cuda",
                                 READ_SAVED_RESULTS=False, RUN_TRAINING=True, PRODUCER={"synthetic": True})
                with patch.object(torch.cuda, "is_available", return_value=True):
                    exec(cells["execution-guards"], namespace)
                exec(cells["configuration-helpers"], namespace)
                self.assertEqual(namespace["CONFIG"].seed, 2)
                for key, expected in paths.items():
                    self.assertEqual(namespace[key], expected)

    def test_training_requires_explicit_cuda_and_diagnostic_producer(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                namespace.update(READ_SAVED_RESULTS=False, RUN_TRAINING=True,
                                 BENCHMARK="cov_vac_degradation", PRODUCER={"synthetic": True})
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    exec(cells["execution-guards"], namespace)
                namespace.update(READ_SAVED_RESULTS=True, RUN_TRAINING=False,
                                 RUN_DIAGNOSTICS=True, PRODUCER=None)
                with self.assertRaisesRegex(ValueError, "producer"):
                    exec(cells["execution-guards"], namespace)

    def test_training_dispatch_reuses_mil_and_keeps_contexts_explicit(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                for identity in ("panel", "configuration-helpers", "training-helper"):
                    exec(cells[identity], namespace)
                closed = []
                @contextmanager
                def opened(*args, **kwargs):
                    try:
                        yield object(), {}, [], {}, {"fixture": True}
                    finally:
                        closed.append(True)
                namespace["study_inputs"] = SimpleNamespace(open_streams=opened)
                namespace["require_reference"] = Mock(return_value=object())
                namespace["selected_anchor"] = Mock(return_value={"key": "unimodal:rna", "modality": "rna"})
                run_fit = Mock()
                namespace["experiment"] = SimpleNamespace(run_fit=run_fit,
                    SEQUENCE_VARIANTS=experiment.SEQUENCE_VARIANTS, ATTENTION_VARIANTS=experiment.ATTENTION_VARIANTS)
                namespace["run_context"]("cov_vac_degradation", "mrnafm")
                self.assertEqual(run_fit.call_count, 6 if name.startswith("gate") else 5)
                self.assertEqual(closed, [True])
                for call in run_fit.call_args_list:
                    variant, config, configured = call.args[1:4]
                    self.assertEqual(config.seed, 0)
                    self.assertEqual(config.entropy_lambda, 0)
                    self.assertEqual(config.tconv, "equation")
                    if variant == "position_residual_control":
                        self.assertEqual(configured["control_bottleneck"], 2)
                        self.assertEqual(configured["control_scale"], .5)
                        self.assertEqual(configured["control_initialization"], "zero_output_random_input")
                    if variant == "anchor_only" or variant.startswith("attention_"):
                        self.assertEqual(configured["anchor"], "rna")
                namespace["require_reference"].side_effect = FileNotFoundError("Missing baseline")
                run_fit.reset_mock()
                with self.assertRaises(FileNotFoundError):
                    namespace["run_context"]("cov_vac_degradation", "mrnafm")
                run_fit.assert_not_called()

    def test_selection_reads_seed_zero_even_for_a_different_training_seed(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                for identity in ("reference-readers", "selection-helper"):
                    exec(cells[identity], namespace)
                calls = []
                def candidate(benchmark, bundle, model, seed, **kwargs):
                    calls.append((model, seed))
                    return SimpleNamespace(extra_fields={
                        "reference_source": {"path": str(Path(folder) / model.replace(":", "-"))},
                        "identity": {"inputs": {key: {} for key in ("rows", "targets", "splits", "split_targets")}}})
                namespace["require_reference"] = candidate
                selected = Mock(return_value={"modality": "rna"})
                namespace["study_selection"] = SimpleNamespace(select_anchor=selected)
                namespace["SEED"] = 2
                namespace["selected_anchor"]("cov_vac_degradation", "original", write=True)
                self.assertEqual([seed for _, seed in calls], [0, 0, 0, 0])
                self.assertEqual(selected.call_args.kwargs["evidence_root"], namespace["REFERENCE_ROOT"])

    def test_diagnostics_skip_nongated_controls_without_starting_fits(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                for identity in ("panel", "configuration-helpers", "diagnostic-helper"):
                    exec(cells[identity], namespace)
                @contextmanager
                def opened(*args, **kwargs):
                    yield object(), {}, [], {}, {}
                new_inference, reference_inference = Mock(), Mock()
                namespace.update(study_inputs=SimpleNamespace(open_streams=opened),
                    selected_anchor=lambda *a, **k: {"modality": "rna"},
                    experiment=SimpleNamespace(collect_diagnostics=new_inference,
                        SEQUENCE_VARIANTS=experiment.SEQUENCE_VARIANTS,
                        ATTENTION_VARIANTS=experiment.ATTENTION_VARIANTS),
                    study_references=SimpleNamespace(collect_reference_diagnostics=reference_inference))
                namespace["collect_context_diagnostics"]("cov_vac_degradation", "original", 2)
                self.assertEqual(new_inference.call_count, 5 if name.startswith("gate") else 4)
                self.assertEqual(reference_inference.call_count, 1 if name.startswith("gate") else 0)
                self.assertTrue(all(call.args[2].seed == 2 for call in new_inference.call_args_list))

    def test_saved_reader_supplies_declared_bundle_architecture_and_selected_anchor(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                for identity in ("panel", "configuration-helpers", "saved-fit-readers"):
                    exec(cells[identity], namespace)
                variant = "sequence_sigmoid" if name.startswith("gate") else "attention_skip"
                entry = next(entry for entry in namespace["PANEL"] if entry["variant"] == variant)
                path = namespace["fit_path"]("cov_vac_degradation", "mrnafm", entry["key"], 2)
                path.parent.mkdir(parents=True)
                path.write_text("synthetic dispatch fixture")
                reader = Mock(return_value=object())
                namespace["experiment"] = SimpleNamespace(read_panel_fit=reader,
                    SEQUENCE_VARIANTS=experiment.SEQUENCE_VARIANTS,
                    ATTENTION_VARIANTS=experiment.ATTENTION_VARIANTS)
                namespace["selected_anchor"] = Mock(return_value={"modality": "rna"})
                namespace["saved_new_fit"]("cov_vac_degradation", "mrnafm", entry, 2)
                requested = reader.call_args.kwargs
                self.assertEqual(requested["config"].seed, 2)
                self.assertEqual(requested["encoders"], namespace["BUNDLES"]["mrnafm"])
                self.assertEqual(requested["settings"]["projection_width"], 600)
                if variant.startswith("attention_"):
                    self.assertEqual(requested["settings"]["anchor"], "rna")
                    namespace["selected_anchor"].return_value = None
                    with self.assertRaisesRegex(FileNotFoundError, "validation anchor"):
                        namespace["saved_new_fit"]("cov_vac_degradation", "mrnafm", entry, 2)

    def test_score_comparison_requires_complete_ordered_targets(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                exec(cells["comparison-helper"], namespace)
                signature = {key: {"fixed": True} for key in ("rows", "targets", "splits", "split_targets")}
                def result(score, truth=(0., 1.)):
                    return SimpleNamespace(test_truth=list(truth), test_score=score, extra_fields={
                        "identity": {"inputs": signature, "model": {
                            "parameter_counts": {"total": 100, "fusion": 50},
                            "implementation_sha256": {"synthetic.py": "a" * 64}}},
                        "reference_source": {"path": str(Path(folder) / "baseline.json"), "sha256": "synthetic"}})
                namespace.update(ACTIVE_BENCHMARKS=("cov_vac_degradation",), BUNDLES={"original": {}},
                    PANEL=({"key": "candidate"},), selected_anchor=lambda *a: {"key": "unimodal:rna"},
                    require_reference=lambda *a: result(.5 if a[2] == "concat" else .6),
                    source_record=lambda p: (str(p), "synthetic"),
                    fit_source=lambda b,c,e,s: (result(.7), Path(folder) / f"seed{s}.json"))
                rows, sources = namespace["comparison_rows"]()
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertAlmostEqual(row["minus concat"], .2)
                    self.assertAlmostEqual(row["minus selected single"], .1)
                def different_source(b,c,e,s):
                    changed = result(.7)
                    if s == 2:
                        changed.extra_fields["identity"]["model"]["implementation_sha256"] = {"synthetic.py": "b" * 64}
                    return changed, Path(folder) / f"seed{s}.json"
                namespace["fit_source"] = different_source
                with self.assertRaisesRegex(ValueError, "different recorded implementation"):
                    namespace["comparison_rows"]()
                namespace["fit_source"] = lambda *a: (result(.7, truth=(1., 0.)), Path(folder) / "fit.json")
                with self.assertRaisesRegex(ValueError, "ordered test targets"):
                    namespace["comparison_rows"]()
                namespace["fit_source"] = lambda *a: (None, Path(folder) / "missing.json")
                with self.assertRaises(FileNotFoundError):
                    namespace["comparison_rows"]()

    def test_gate_summary_gives_each_transcript_equal_weight(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                cells, namespace, _ = self.fixture(name, root)
                exec(cells["gate-summary-helper"], namespace)
                fit = root / "fit.json"
                fit.write_text("synthetic retained fit")
                def stem(b,c,key,seed):
                    return root / f"diagnostic{seed}"
                for seed in (0,1,2):
                    Path(str(stem(None,None,None,seed)) + ".json").write_text("{}")
                metadata = {"identity": {"fit_sha256": hashlib.sha256(fit.read_bytes()).hexdigest()},
                            "gate_modalities": ["rna"]}
                arrays = {"gates": np.array([[.1],[.9],[.9],[.9]]), "gates_lengths": np.array([1,3])}
                namespace.update(ACTIVE_BENCHMARKS=("cov_vac_degradation",), BUNDLES={"original": {}},
                    PANEL=({"key": "position", "variant": "position_sigmoid"},),
                    diagnostic_stem=stem, fit_source=lambda *a: (object(), fit),
                    selected_anchor=lambda *a: {"modality": "rna"},
                    source_record=lambda p: (str(p), "synthetic"),
                    experiment=SimpleNamespace(SEQUENCE_VARIANTS=experiment.SEQUENCE_VARIANTS,
                                               ATTENTION_VARIANTS=experiment.ATTENTION_VARIANTS,
                                               summarize_diagnostics=experiment.summarize_diagnostics,
                                               __file__=experiment.__file__,
                                               read_diagnostics=lambda *a: (metadata, arrays)))
                report, missing = namespace["summarize_saved_gates"]()
                self.assertEqual(missing, [])
                self.assertEqual(len(report["summary"]), 3)
                for row in report["summary"]:
                    self.assertAlmostEqual(row["median"], .5)
                    self.assertEqual(row["transcripts"], 2)

    def test_direct_mechanism_contrasts_and_versioned_destinations(self):
        expected = {
            NAMES[0]: {("sequence_sigmoid", "mil_no_entropy"), ("meanmax_softmax", "mil_no_entropy"),
                       ("meanmax_sigmoid", "sequence_sigmoid"), ("position_sigmoid", "sequence_sigmoid")},
            NAMES[1]: {("attention_skip", "attention_ordinary"),
                       ("attention_sequence_sigmoid", "attention_skip"),
                       ("attention_position_sigmoid", "attention_sequence_sigmoid"),
                       ("attention_position_sigmoid", "attention_skip")},
        }
        for name in NAMES:
            _, cells = notebook(name)
            namespace = {}
            exec(cells["mechanism-contrast-helper"], namespace)
            self.assertEqual({(a, b) for _, a, b in namespace["mechanism_contrasts"]()}, expected[name])
            self.assertIn("comparisons.extend(mechanism_contrasts())", cells["interval-helper"])
            self.assertIn('"diagnostics-v2"', cells["configuration-helpers"])
            self.assertIn("three-seed-comparison-v2.json", cells["analysis"])
            if name == NAMES[0]:
                self.assertIn('REPORT_PATH = RESULTS_ROOT / "reports" / "three-seed-comparison-v3.json"', cells["analysis"])
                self.assertIn("position_sigmoid minus position_residual_control", cells["interval-helper"])
            self.assertIn("gate-distributions-v2.json", cells["gate-analysis"])

    def test_legacy_saved_reports_read_without_new_analysis_or_writes(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, forbidden = self.fixture(name, Path(folder))
                destination = namespace["RESULTS_ROOT"] / "reports"
                destination.mkdir(parents=True)
                legacy = {"sources": {}, "summary": []}
                for filename in ("three-seed-comparison.json", "gate-distributions.json"):
                    (destination / filename).write_text(json.dumps(legacy))
                namespace["verify_report_sources"] = Mock()
                originals = {p.name: p.read_bytes() for p in destination.iterdir()}
                exec(cells["analysis"], namespace)
                exec(cells["gate-analysis"], namespace)
                exec(cells["saved-diagnostic-detail"], namespace)
                self.assertEqual(namespace["report"], legacy)
                self.assertEqual(namespace["gate_report"], legacy)
                self.assertEqual(originals, {p.name: p.read_bytes() for p in destination.iterdir()})
                forbidden.assert_not_called()

    def test_identical_versioned_report_reuses_json_without_republishing(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                cells, namespace, _ = self.fixture(name, root)
                for identity in ("mechanism-contrast-helper", "report-source-helpers"):
                    exec(cells[identity], namespace)
                source = root / "Code" / "stage4" / name
                source.parent.mkdir(parents=True)
                source.write_text("invented notebook source identity")
                evidence = namespace["RESULTS_ROOT"] / "synthetic-source.txt"
                evidence.parent.mkdir(parents=True)
                evidence.write_text("invented prediction evidence")
                source_map = {"study/synthetic-source.txt": hashlib.sha256(evidence.read_bytes()).hexdigest()}
                rows = [{"benchmark": "invented", "bundle": "test", "model": "candidate", "seed": seed,
                         "trainable parameters": 10, "fusion parameters": 5, "score": .5,
                         "minus concat": .1, "minus selected single": .05} for seed in (0, 1, 2)]
                def publish(path, payload):
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(payload))
                publication = Mock(side_effect=publish)
                namespace.update(RUN_ANALYSIS=True, comparison_rows=lambda: (rows, source_map),
                    comparison_intervals=lambda: ([], {}),
                    correction_experiment=SimpleNamespace(_publish=publication),
                    study_inputs=SimpleNamespace(__file__=str(source)))
                exec(cells["analysis"], namespace)
                original = namespace["REPORT_PATH"].read_bytes()
                exec(cells["analysis"], namespace)
                self.assertEqual(publication.call_count, 1)
                self.assertEqual(namespace["REPORT_PATH"].read_bytes(), original)
                namespace["RUN_ANALYSIS"] = False
                exec(cells["analysis"], namespace)
                self.assertEqual(namespace["report"]["schema"],
                    "controlled-fusion-three-seed-v3" if name.startswith("gate") else "controlled-fusion-three-seed-v2")
                # A changed source must still fail through the real reader, for every schema.
                evidence.write_text("changed fixture evidence")
                with self.assertRaisesRegex(ValueError, "source is missing, changed"):
                    exec(cells["analysis"], namespace)

    def test_report_versions_validate_bootstrap_and_keep_gate_only_records_readable(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                cells, namespace, _ = self.fixture(name, Path(folder))
                exec(cells["report-source-helpers"], namespace)
                expected = {"draws": namespace["BOOTSTRAP_DRAWS"], "seed": namespace["BOOTSTRAP_SEED"],
                            "sampling_unit": "exact nucleotide sequence group",
                            "seeds_averaged_within_draw": [0, 1, 2]}
                for version in (1, 2, 3):
                    report = {"schema": f"controlled-fusion-three-seed-v{version}", "seeds": [0, 1, 2],
                              "sources": {}, "bootstrap": expected.copy()}
                    namespace["verify_report_sources"](report)
                    report["bootstrap"]["sampling_unit"] = "individual row"
                    with self.assertRaisesRegex(ValueError, "interval settings"):
                        namespace["verify_report_sources"](report)
                for version in (1, 2):
                    namespace["verify_report_sources"]({"schema": f"controlled-gate-distributions-v{version}",
                                                         "seeds": [0, 1, 2], "sources": {}})
                with self.assertRaisesRegex(ValueError, "contract"):
                    namespace["verify_report_sources"]({"schema": "unknown-v99", "seeds": [0, 1, 2]})

    def test_gate_notebook_reads_previous_v2_panel_as_history_through_real_validator(self):
        with tempfile.TemporaryDirectory() as folder:
            cells, namespace, _ = self.fixture(NAMES[0], Path(folder))
            exec(cells["report-source-helpers"], namespace)
            historical = {"schema": "controlled-fusion-three-seed-v2", "seeds": [0, 1, 2],
                          "sources": {}, "bootstrap": {"draws": namespace["BOOTSTRAP_DRAWS"],
                          "seed": namespace["BOOTSTRAP_SEED"], "sampling_unit": "exact nucleotide sequence group",
                          "seeds_averaged_within_draw": [0, 1, 2]}}
            path = namespace["RESULTS_ROOT"] / "reports/three-seed-comparison-v2.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(historical))
            exec(cells["analysis"], namespace)
            self.assertEqual(namespace["report"], historical)
            self.assertTrue(any("Historical report" in str(call) for call in namespace["print"].call_args_list))
            self.assertFalse(namespace["REPORT_PATH"].exists())

    def test_parameter_preview_requires_explicit_flag_and_does_not_retune_control(self):
        with tempfile.TemporaryDirectory() as folder:
            cells, namespace, forbidden = self.fixture(NAMES[0], Path(folder))
            self.assertFalse(namespace["RUN_PARAMETER_PREVIEW"])
            exec(cells["parameter-preview"], namespace)
            forbidden.assert_not_called()
            # Invented constructors exercise only metadata arithmetic and RNG preservation.
            class FakeModel:
                def __init__(self):
                    self.fusion = torch.nn.Linear(2, 1)
                    self.head = torch.nn.Linear(1, 1)
                def parameters(self):
                    return (*self.fusion.parameters(), *self.head.parameters())
            exec(cells["panel"], namespace)
            exec(cells["configuration-helpers"], namespace)
            namespace.update(RUN_PARAMETER_PREVIEW=True,
                token_embeddings=SimpleNamespace(grid_resolution=lambda key: "codon"))
            before = torch.get_rng_state().clone()
            with patch.object(experiment, "build_model", side_effect=lambda *a, **k: FakeModel()):
                exec(cells["parameter-preview"], namespace)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            self.assertEqual(namespace["CONTROL_BOTTLENECK"], 2)
            self.assertEqual(namespace["LEGACY_CONTROL_BOTTLENECK"], 150)
            alternatives = namespace["structural_options"]
            self.assertEqual(alternatives[0]["maximum affine-update rank"], 2)
            self.assertEqual(alternatives[0]["fixed-path width"], 600)
            self.assertEqual(alternatives[0]["difference from joint gate"], -1)
            self.assertEqual(alternatives[1]["fixed-path width"], 0)
            self.assertEqual(alternatives[1]["difference from joint gate"], 355347)

    def test_residual_control_replaces_one_arm_without_expanding_the_panel(self):
        with tempfile.TemporaryDirectory() as folder:
            cells, namespace, _ = self.fixture(NAMES[0], Path(folder))
            exec(cells["panel"], namespace)
            entries = namespace["PANEL"]
            self.assertEqual(len(entries), 7)
            self.assertEqual(sum(not entry.get("reference") for entry in entries), 6)
            keys = {entry["key"] for entry in entries}
            self.assertIn("position_residual_control", keys)
            self.assertNotIn("position_control", keys)
            exec(cells["configuration-helpers"], namespace)
            new_path = namespace["fit_path"]("synthetic", "original", "position_residual_control", 0)
            old_path = namespace["fit_path"]("synthetic", "original", "position_control", 0)
            self.assertNotEqual(new_path, old_path)
            namespace["CONTROL_SCALE"] = 1.
            with self.assertRaisesRegex(ValueError, "residual control"):
                exec(cells["execution-guards"], namespace)

    def test_imports_work_from_checkout_root_without_preexisting_project_path(self):
        # Fresh isolated process imports setup only; it cannot inherit this test's Code path.
        program = """
import json
from pathlib import Path
import sys
root = Path.cwd()
for name in ('gate_diagnostics', 'anchor_cross_attention'):
    code = str(root / 'Code')
    sys.path[:] = [path for path in sys.path if path != code]
    assert code not in sys.path
    cells = {cell['id']: ''.join(cell['source']) for cell in json.loads(
        (root / 'Code/stage4' / (name + '.ipynb')).read_text(encoding='utf-8'))['cells']}
    assert 'from shared_code' not in cells['imports']
    namespace = {}
    exec(cells['imports'], namespace)
    exec(cells['paths'], namespace)
    assert namespace['CODE_DIR'] == root / 'Code'
    assert namespace['datasets'].__name__ == 'shared_code.datasets'
"""
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", program], cwd=ROOT,
                                   env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
