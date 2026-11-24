"""Isolated correction-notebook control flow with invented inputs and blocked producers."""

from contextlib import ExitStack, contextmanager, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
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
from shared_code.fusion import correction_experiment as correction
from shared_code.fusion import study_inputs, study_selection, swap_reporting, token_embeddings, train

NOTEBOOK = ROOT / "Code/stage4/anchor_correction.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources():
    return {cell["id"]: "".join(cell["source"])
            for cell in notebook()["cells"] if cell["cell_type"] == "code"}


def definitions(root):
    scope = {"__name__": "synthetic_notebook", "display": Mock()}
    selected = {"imports", "project-paths", "settings", "panels", "execution-guard",
                "coverage-function", "path-and-selection-helpers", "anchor-fit-helper",
                "auxiliary-helpers", "correction-fit-helper", "run-study-function",
                "saved-artifact-reader", "report-panel-helpers", "report-function", "saved-report-reader",
                "diagnostic-report-helpers", "fold-review-function", "control-review-function",
                "saved-diagnostics-function"}
    with redirect_stdout(StringIO()):
        for identity, source in sources().items():
            if identity in selected:
                exec(compile(source, f"{NOTEBOOK.name}:{identity}", "exec"), scope)
                if identity == "settings":
                    scope["RESULTS_ROOT"] = Path(root)
    return scope


class NotebookStructureTests(unittest.TestCase):
    def test_sources_compile_saved_outputs_are_empty_and_settings_are_tagged(self):
        tagged = []
        for cell in notebook()["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
            if "fusion-study-settings" in cell["metadata"].get("tags", []):
                tagged.append(cell["id"])
        self.assertEqual(tagged, ["settings"])

    def test_relative_file_links_exist_and_equations_have_normal_escapes(self):
        for cell in notebook()["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            source = "".join(cell["source"])
            self.assertNotIn("\\\\mathrm", source)
            for link in re.findall(r"\]\(([^)]+)\)", source):
                if not link.startswith(("https:", "http:", "#")):
                    self.assertTrue((NOTEBOOK.parent / link.split("#")[0]).is_file(), link)

    def test_saved_defaults_do_not_read_datasets_caches_fit_score_or_publish(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as blocked:
            for module, names in (
                (study_inputs, ("open_streams", "benchmark_rows")),
                (token_embeddings, ("summarize", "extract")),
                (correction, ("run_neural", "run_ridge", "run_anchor_oof", "training_random_statistics",
                              "fold_plan", "row_permutation", "inspect_shuffle", "_publish")),
                (train, ("run", "fit", "score", "predict", "seed_averaged_bootstrap")),
                (swap_reporting, ("calculate",)),
            ):
                for name in names:
                    blocked.enter_context(patch.object(module, name, side_effect=AssertionError(name)))
            blocked.enter_context(patch.object(pd, "read_csv", side_effect=AssertionError("CSV")))
            scope = {}
            with redirect_stdout(StringIO()):
                for identity, source in sources().items():
                    exec(compile(source, f"guarded:{identity}", "exec"), scope)
                    if identity == "settings":
                        scope["RESULTS_ROOT"] = Path(directory)
            self.assertTrue(scope["READ_SAVED_RESULTS"])
            self.assertFalse(scope["RUN_TRAINING"])
            self.assertFalse(scope["RUN_ANALYSIS"])
            self.assertFalse(scope["RUN_DIAGNOSTICS"])
            self.assertFalse(scope["RUN_INPUT_AUDIT"])
            self.assertFalse(scope["RUN_CACHE_AUDIT"])
            self.assertEqual(list(Path(directory).iterdir()), [])


class DecisionSupportTests(unittest.TestCase):
    def test_cache_audit_display_labels_historical_shuffle_without_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope["inspect_cached_controls"] = Mock(side_effect=AssertionError("must not inspect caches"))
            records = [(Path("historical.json"), {"identity": {"inputs": {"seed": 0}},
                                                  "analysis": {"bundles": []}}),
                       (Path("row-permutation.json"), {"identity": {"inputs": {"control_kind": "row_shuffle_v1"}},
                                                       "analysis": {"bundles": []}})]
            scope["read_decision_reports"] = Mock(return_value=records)
            before = json.dumps([record for _, record in records], sort_keys=True)
            output = StringIO()
            with redirect_stdout(output):
                exec(sources()["control-review-display"], scope)
            self.assertIn("Historical whole-group shuffle; this audit does not verify", output.getvalue())
            self.assertIn("Frequency-preserving row shuffle v1", output.getvalue())
            self.assertEqual(json.dumps([record for _, record in records], sort_keys=True), before)

    def test_input_audit_uses_rows_only_and_does_not_change_assignment_or_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            groups = np.array([f"g{index // 2}" for index in range(240)] + ["val", "test"])
            rows = pd.DataFrame({"Value": np.arange(242.), "length": [30] * 242})
            original = rows.copy(deep=True)
            positions = {"train": np.arange(240), "val": np.array([240]), "test": np.array([241])}
            scope["save_decision_report"] = Mock(return_value="saved")
            with patch.object(study_inputs, "benchmark_rows", return_value=(rows, "Split", groups, positions, {"source": "invented"})), \
                 patch.object(study_inputs, "open_streams", side_effect=AssertionError("no cache")), \
                 patch.object(train, "run", side_effect=AssertionError("no fit")):
                self.assertEqual(scope["inspect_fold_inputs"](), "saved")
            _, analysis, identity = scope["save_decision_report"].call_args.args
            self.assertEqual(len(analysis["folds"]), 15)
            self.assertEqual(identity["settings"]["n_folds"], 5)
            self.assertEqual(identity["settings"]["stop_fraction"], .2)
            pd.testing.assert_frame_equal(rows, original)

    def test_diagnostic_publication_versions_inputs_reuses_exact_report_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)

            def publish(path, record):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf8") as handle:
                    json.dump(record, handle)

            with patch.object(correction, "_publish", side_effect=publish) as writer, \
                 patch.object(scope["run_identity"], "producer_record", return_value={"fixture": True}):
                first = scope["save_decision_report"]("folds", {"folds": []}, {"source": "a"})
                before = first.read_bytes()
                self.assertEqual(first, scope["save_decision_report"]("folds", {"folds": []}, {"source": "a"}))
                second = scope["save_decision_report"]("folds", {"folds": []}, {"source": "b"})
                self.assertNotEqual(first, second)
                with self.assertRaisesRegex(ValueError, "same identity"):
                    scope["save_decision_report"]("folds", {"folds": ["changed"]}, {"source": "a"})
                self.assertEqual(writer.call_count, 2)
            self.assertEqual(before, first.read_bytes())
            self.assertEqual(len(scope["read_decision_reports"]("folds")), 2)
            damaged = json.loads(first.read_text(encoding="utf8"))
            damaged["analysis"]["folds"] = ["corrupt"]
            first.write_text(json.dumps(damaged), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "changed"):
                scope["read_decision_reports"]("folds")

    def test_saved_diagnostics_use_oof_and_validation_not_final_test_for_error_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            root = scope["benchmark_root"](scope["BENCHMARK"])

            def save_neural(model, values, *, protocol=None):
                folder = root / model
                folder.mkdir(parents=True)
                identity = {"inputs": {"protocol": protocol or {}}}
                fit = {"identity": identity, "best_epoch": 0, "epochs": 1, "val_score": .2,
                       "history": [{"epoch": 0, "train_loss": 2., "val_score": .2}]}
                fit_path = folder / "fit.json"
                fit_path.write_text(json.dumps(fit), encoding="utf8")
                entry = {"row_ids": [1, 2], "values": values, "truth": [2., 4.]}
                record = {"contract": scope["STUDY"], "kind": "neural", "identity": identity,
                          "artifacts": {"fit": {"file": "fit.json", "sha256": study_selection.file_sha256(fit_path)}},
                          "predictions": {"val": entry, "test": {**entry, "values": [100., 100.]}}}
                if model.startswith("oof/"):
                    record["predictions"]["test"] = entry
                path = folder / "predictions.json"
                path.write_text(json.dumps(record), encoding="utf8")
                return path

            anchor = save_neural("anchors/ntv2_native/seed0", [1., 3.])
            save_neural("oof/ntv2_native/seed0/fold0", [0., 2.])
            save_neural("corrections/original/seed0/rna/real/textcnn", [2., 4.], protocol={
                "offset_sources": {"final_anchor_sha256": study_selection.file_sha256(anchor)}})
            scope["save_decision_report"] = Mock(return_value="summary")
            with patch.object(train, "predict", side_effect=AssertionError("no model inference")), \
                 patch.object(study_inputs, "open_streams", side_effect=AssertionError("no cache")), \
                 patch.object(train, "score", side_effect=AssertionError("no test scoring")):
                self.assertEqual(scope["inspect_saved_correction_evidence"](), "summary")
            _, analysis, _ = scope["save_decision_report"].call_args.args
            self.assertEqual(analysis["issues"], [])
            residuals = {row["model"]: row for row in analysis["residuals"]}
            self.assertEqual(residuals["anchors/ntv2_native/seed0"]["rmse"], 1.)
            self.assertEqual(residuals["oof/ntv2_native/seed0/fold0"]["rmse"], 2.)
            self.assertEqual(analysis["validation_updates"][0]["correction_mean"], 1.)
            self.assertEqual(analysis["validation_updates"][0]["correction_sd"], 0.)


class NotebookTrainingTests(unittest.TestCase):
    def test_training_guard_refuses_cpu_missing_producer_and_protected_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope.update(READ_SAVED_RESULTS=False, RUN_TRAINING=True)
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                scope["require_training_permission"]()
            scope["DEVICE"] = "cuda"
            with patch.object(torch.cuda, "is_available", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "producer"):
                    scope["require_training_permission"]()
                scope.update(PRODUCER={"synthetic": True}, RESULTS_ROOT=scope["CACHE_ROOT"])
                with self.assertRaisesRegex(ValueError, "overlap"):
                    scope["require_training_permission"]()

    def test_later_seed_requires_selection_before_opening_any_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope.update(SEED=1, require_training_permission=Mock())
            with patch.object(study_inputs, "open_streams", side_effect=AssertionError("must not load")):
                with self.assertRaises(FileNotFoundError):
                    scope["run_study"]()

    def test_anchor_fits_use_runtime_seed_and_only_requested_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope["USE_NEWTON_ANCHORS"] = False
            streams = SimpleNamespace(rows=pd.DataFrame(index=range(9)))
            opened = {"original": (streams, {}, np.array(list("abcdefghi")),
                                    {"train": (0, 1, 2), "val": (3, 4, 5), "test": (6, 7, 8)}, {"split": "invented"})}
            with patch.object(correction, "run_neural", return_value="artifact") as fitted:
                self.assertEqual(scope["fit_anchor"](opened, "fungal_expression", "ntv2_native", 2), "artifact")
            kwargs = fitted.call_args.kwargs
            self.assertEqual(kwargs["config"].seed, 2)
            self.assertEqual(kwargs["modalities"], ("dna",))
            self.assertEqual(kwargs["role"], "anchor")
            self.assertFalse(np.any(kwargs["offsets"]))

    def test_both_families_all_controls_share_final_anchor_and_oof_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope["SEED"] = 2
            streams = SimpleNamespace(rows=pd.DataFrame(index=range(9)), labels=np.arange(9.))
            positions = {"train": np.array([0, 1, 2]), "val": np.array([3, 4, 5]), "test": np.array([6, 7, 8])}
            opened = {"original": (streams, {}, np.array(list("abcdefghi")), positions, {"split": "invented"})}
            final = SimpleNamespace(sha256="a" * 64)
            folds = [SimpleNamespace(sha256="b" * 64)]
            selected = {"modality": "rna", "selection_sha256": "s" * 64}
            offsets = np.arange(9.) + .25
            with patch.object(correction, "correction_offsets", return_value=offsets), \
                 patch.object(correction, "training_random_statistics", return_value={}), \
                 patch.object(correction, "pooled_features", return_value=np.ones((9, 2))), \
                 patch.object(correction, "run_ridge") as ridge, \
                 patch.object(correction, "run_neural") as neural:
                scope["fit_corrections"](opened, "fungal_expression", "original", selected, final, object(), folds)
            self.assertEqual(ridge.call_count, 9)
            self.assertEqual(neural.call_count, 9)
            for call in neural.call_args_list:
                kwargs = call.kwargs
                self.assertEqual(kwargs["config"].seed, 2)
                self.assertIs(kwargs["offsets"], offsets)
                self.assertEqual(kwargs["input_contract"]["offset_sources"], {
                    "final_anchor_sha256": "a" * 64, "oof_artifact_sha256": ["b" * 64]})
                self.assertEqual(kwargs["layout"], "native" if len(kwargs["modalities"]) == 1 else "codon")
            self.assertEqual({call.kwargs["control"]["kind"] for call in neural.call_args_list},
                             {"real", "row_shuffle_v1", "random"})
            for call in neural.call_args_list:
                control = call.kwargs["control"]
                if control["kind"] == "row_shuffle_v1":
                    self.assertIn("row_shuffle_v1", call.kwargs["output_dir"].parts)
                    self.assertNotIn("shuffle", call.kwargs["output_dir"].parts)
                    for members in positions.values():
                        self.assertEqual(sorted(control["row_map"][index] for index in members), list(members))

    def test_reference_route_preserves_missing_weight_hold_and_keeps_new_cases_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            streams = SimpleNamespace(rows=pd.DataFrame(index=range(9)))
            opened = {"original": (streams, {}, np.array(list("abcdefghi")),
                                    {"train": (0, 1, 2), "val": (3, 4, 5), "test": (6, 7, 8)}, {})}
            with patch.object(correction, "reuse_newton_anchor", side_effect=FileNotFoundError("missing weights")) as reuse, \
                 patch.object(correction, "reuse_grouped_anchor", return_value="grouped") as grouped, \
                 patch.object(correction, "run_neural", return_value="new") as fitted:
                with self.assertRaisesRegex(FileNotFoundError, "missing weights"):
                    scope["fit_anchor"](opened, "fungal_expression", "rnafm_native", 2)
                fitted.assert_not_called()
                self.assertEqual(reuse.call_args.kwargs["config"].seed, 2)
                self.assertEqual(scope["fit_anchor"](opened, "fungal_expression", "ntv2_grid", 0), "new")
                self.assertEqual(scope["fit_anchor"](opened, "mrna_stability", "ntv2_native", 1), "grouped")
                self.assertEqual(grouped.call_args.kwargs["config"].seed, 1)
                self.assertEqual(fitted.call_count, 1)

    def test_shared_selected_anchor_is_fitted_and_cross_fitted_once_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            streams = SimpleNamespace(rows=pd.DataFrame(index=range(9)))
            positions = {"train": (0, 1, 2), "val": (3, 4, 5), "test": (6, 7, 8)}
            selections = {bundle: {"key": "ntv2_native", "modality": "dna", "selection_sha256": bundle}
                          for bundle in scope["BUNDLES"]}

            @contextmanager
            def opened(*args, **kwargs):
                yield streams, {}, np.array(list("abcdefghi")), positions, {"split": "invented"}

            scope.update(require_training_permission=Mock(), prepare_selections=Mock(return_value=selections),
                         fit_anchor=Mock(return_value="final"), fit_corrections=Mock())
            with patch.object(study_inputs, "open_streams", side_effect=opened), \
                 patch.object(correction, "fold_plan", return_value="plans"), \
                 patch.object(correction, "run_anchor_oof", return_value=("oof", "folds")) as oof:
                scope["run_study"]()
            self.assertEqual(scope["fit_anchor"].call_count, 1)
            self.assertEqual(oof.call_count, 1)
            self.assertEqual(scope["fit_corrections"].call_count, 2)


class NotebookReportingTests(unittest.TestCase):
    def test_correction_report_pairs_exact_rows_and_float32_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            selected = {"key": "ntv2_native", "modality": "dna", "selection_sha256": "s" * 64}
            scope.update(
                TRAINING_SEEDS=(0,), CONTROL_KINDS=("real",), CORRECTION_FAMILIES=("ridge",),
                auxiliary_sets=Mock(return_value=("rna",)), read_selection=Mock(return_value=selected),
            )
            rows = [6, 7, 8]
            ridge_truth = (0.1, 0.2, 0.3)
            anchor_truth = np.asarray(ridge_truth, dtype=np.float32).tolist()
            self.assertNotEqual(list(ridge_truth), anchor_truth)
            entry = {"positions": rows, "row_ids": rows, "truth": anchor_truth, "values": [1., 2., 3.]}
            anchor = {
                "kind": "neural", "predictions": {"test": entry},
                "identity": {"config": {"seed": 0}, "inputs": {
                    "protocol": {"candidate": selected["key"]}, "groups": list("abcdefghi"),
                }},
            }
            protocol = {
                "training_seed": 0, "bundle": "original", "selection_sha256": selected["selection_sha256"],
                "offset_sources": {"final_anchor_sha256": "a" * 64}, "control": {"kind": "real"},
            }
            ridge_entry = {**entry, "truth": list(ridge_truth)}
            ridge = {"kind": "ridge", "identity": {"protocol": protocol}, "predictions": {"test": ridge_entry}}
            scope["read_prediction_record"] = Mock(side_effect=lambda path: anchor if "anchors" in path.parts else ridge)

            with (
                patch.object(study_selection, "file_sha256", return_value="a" * 64),
                patch.object(train, "score", return_value=0.5) as score,
            ):
                _, _, _, panel = scope["collect_report_panel"]("fungal_expression", "original")
                payload = panel["models"]["rna/real/ridge"][0]["data"]
                self.assertEqual(payload["test_truth"], anchor_truth)
                self.assertEqual(score.call_args.args[1].tolist(), anchor_truth)
                self.assertEqual(ridge_entry["truth"], list(ridge_truth))

                for change in ({"truth": [0.1, 0.25, 0.3]}, {"row_ids": [7, 6, 8]}):
                    with self.subTest(change=change):
                        ridge["predictions"]["test"] = {**ridge_entry, **change}
                        with self.assertRaisesRegex(ValueError, "test observations differ"):
                            scope["collect_report_panel"]("fungal_expression", "original")

                ridge["predictions"]["test"] = ridge_entry
                protocol["offset_sources"]["final_anchor_sha256"] = "b" * 64
                with self.assertRaisesRegex(ValueError, "different anchor or validation selection"):
                    scope["collect_report_panel"]("fungal_expression", "original")

    def test_complete_three_seed_report_has_paired_anchors_and_no_training(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope["BOOTSTRAP_DRAWS"] = 20
            selected = {"key": "ntv2_native", "modality": "dna", "selection_sha256": "s" * 64}
            scope["read_selection"] = Mock(return_value=selected)

            def saved(path):
                parts = Path(path).parts
                seed = next(int(part[4:]) for part in parts if part.startswith("seed"))
                values = [0., 1., 2.] if seed != 1 else [0., 2., 1.]
                predictions = {"test": {"positions": [6, 7, 8], "row_ids": [6, 7, 8],
                                        "truth": [0., 1., 2.], "values": values}}
                if "anchors" in parts:
                    identity = {"config": {"seed": seed}, "inputs": {
                        "protocol": {"candidate": "ntv2_native"}, "groups": list("abcdefghi")}}
                    return {"identity": identity, "kind": "neural", "predictions": predictions}
                bundle = parts[parts.index("corrections") + 1]
                protocol = {"training_seed": seed, "bundle": bundle, "selection_sha256": "s" * 64,
                            "offset_sources": {"final_anchor_sha256": "a" * 64},
                            "control": {"kind": parts[-3]}}
                kind = "ridge" if "ridge" in parts else "neural"
                identity = {"protocol": protocol} if kind == "ridge" else {"inputs": {"protocol": protocol}}
                return {"kind": kind, "identity": identity, "predictions": predictions}

            scope["read_prediction_record"] = Mock(side_effect=saved)
            with patch.object(study_selection, "file_sha256", return_value="a" * 64), \
                 patch.object(correction, "_publish") as published, \
                 patch.object(train, "run", side_effect=AssertionError("report must not fit")):
                scores, summary, destination = scope["calculate_three_seed_report"]("fungal_expression")
            self.assertEqual(len(scores), 2 * 19 * 3)
            self.assertTrue((summary["seeds"] == 3).all())
            self.assertEqual(published.call_count, 1)
            produced = published.call_args.args[1]
            self.assertEqual(set(produced["analysis"]["paired"]), {"original", "mrnafm"})
            self.assertEqual(len(produced["analysis"]["paired"]["original"]["seed_summary"]), 18)
            self.assertEqual(produced["identity"]["groups"], {
                "original": correction._digest(list("ghi")), "mrnafm": correction._digest(list("ghi"))})
            self.assertTrue(str(destination).startswith(directory))
            self.assertTrue(np.allclose(summary["mean"], (1. + .5 + 1.) / 3))
            destination.parent.mkdir(parents=True)
            destination.write_text(json.dumps(produced), encoding="utf8")
            with patch.object(study_selection, "file_sha256", return_value="a" * 64), \
                 patch.object(swap_reporting, "calculate", side_effect=AssertionError("reuse must not bootstrap")), \
                 patch.object(correction, "_publish", side_effect=AssertionError("reuse must not publish")):
                repeated_scores, repeated_summary, repeated_path = scope["calculate_three_seed_report"]("fungal_expression")
            pd.testing.assert_frame_equal(scores, repeated_scores)
            pd.testing.assert_frame_equal(summary, repeated_summary)
            self.assertEqual(destination, repeated_path)

            def historical_control(path):
                record = saved(path)
                if "row_shuffle_v1" in Path(path).parts:
                    identity = record["identity"]
                    source = identity["protocol"] if record["kind"] == "ridge" else identity["inputs"]["protocol"]
                    source["control"]["kind"] = "shuffle"
                return record

            scope["read_prediction_record"] = Mock(side_effect=historical_control)
            before = destination.read_bytes()
            with patch.object(study_selection, "file_sha256", return_value="a" * 64), \
                 patch.object(correction, "_publish", side_effect=AssertionError("must preserve old report")):
                with self.assertRaisesRegex(ValueError, "control differs"):
                    scope["calculate_three_seed_report"]("fungal_expression")
            self.assertEqual(destination.read_bytes(), before)

            def changed_group(path):
                record = saved(path)
                if "anchors" in Path(path).parts and "seed1" in Path(path).parts:
                    record["identity"]["inputs"]["groups"][-1] = "different-group"
                return record

            scope["read_prediction_record"] = Mock(side_effect=changed_group)
            with patch.object(study_selection, "file_sha256", return_value="a" * 64), \
                 patch.object(correction, "_publish", side_effect=AssertionError("incompatible report must not publish")):
                with self.assertRaisesRegex(ValueError, "groups differ"):
                    scope["calculate_three_seed_report"]("fungal_expression")

    def test_missing_evidence_does_not_publish_partial_seed_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = definitions(directory)
            scope["read_selection"] = Mock(return_value={"key": "ntv2_native", "modality": "dna", "selection_sha256": "s"})
            with patch.object(study_selection, "file_sha256", return_value="a" * 64), \
                 patch.object(correction, "_publish", side_effect=AssertionError("must not publish")):
                with self.assertRaises(FileNotFoundError):
                    scope["calculate_three_seed_report"]("fungal_expression")


if __name__ == "__main__":
    unittest.main()
