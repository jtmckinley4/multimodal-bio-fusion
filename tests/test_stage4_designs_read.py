"""Mamba notebook guards with invented records and blocked research entry points."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import run_identity, swap_results
from test_swap_results import ENCODERS, RECIPE


ORIGINAL_ENCODERS = {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"}


def invented_result(variant, *, val_score=0.5, test_score=0.4, tconv="equation"):
    config = {**RECIPE, "tconv": tconv}
    labels = np.array([0, 1, 1, 0, 1, 1], dtype=np.float32)
    indices = {"train": [0, 1], "val": [2, 3], "test": [4, 5]}
    inputs = {
        "encoders": ENCODERS,
        "rows": {"columns": ["sequence", "length"], "count": 6, "sha256": "a" * 64},
        "targets": {**run_identity.array_record(labels), "columns": ["Value"],
                    "task": "regression", "transform": "as_supplied"},
        "splits": {key: run_identity.array_record(value, np.int64)
                   for key, value in indices.items()},
        "split_targets": {key: run_identity.array_record(labels[value])
                          for key, value in indices.items()},
    }
    identity = run_identity.make_identity(
        benchmark="fixture", variant=variant, config=config, inputs=inputs,
        model={"contract": "synthetic"},
        loader=run_identity.blf_loader_spec(config, ("dna", "rna", "prot")),
    )
    return {
        "benchmark": "fixture", "variant": variant, "config": config,
        "epochs": 4, "best_epoch": 3, "seconds": 1.5,
        "val_score": val_score, "test_score": test_score,
        "test_truth": labels[indices["test"]].tolist(),
        "test_prediction": [0.8, 1.2], "identity": identity,
    }


class DesignReadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.references = self.folder / "references"
        self.designs = self.folder / "designs" / "fixture"
        self.references.mkdir()
        self.designs.mkdir(parents=True)
        notebook = json.loads((ROOT / "Code/stage4/tier4_designs.ipynb").read_text(
            encoding="utf-8"))
        self.cells = {cell["id"]: "".join(cell["source"])
                      for cell in notebook["cells"] if cell["cell_type"] == "code"}
        forbidden = Mock(side_effect=AssertionError("Research entry point reached"))
        self.bootstrap = Mock(return_value={"difference": 0.1, "ci_low": 0.01, "ci_high": 0.2})
        self.env = {
            "Path": Path, "SimpleNamespace": SimpleNamespace, "np": np, "pd": pd,
            "CODE_DIR": self.folder / "Code", "display": Mock(),
            "run_identity": run_identity, "swap_results": swap_results,
            "train": SimpleNamespace(
                TrainConfig=lambda **settings: SimpleNamespace(**{**RECIPE, **settings}),
                run=forbidden, fit=forbidden, paired_bootstrap=self.bootstrap,
            ),
            "blf": SimpleNamespace(
                ENCODER_OF_MODALITY=ENCODERS,
                BenchmarkStreams=SimpleNamespace(open=forbidden),
            ),
            "time": SimpleNamespace(time=lambda: 0),
        }
        self.execute("36da38f2")
        self.env.update(BENCHMARKS=("fixture",), REFERENCE_DIRS={"fixture": (self.references,)},
                        RESULTS_ROOT=self.designs.parent)
        for cell_id in ("design-reference-evidence", "9cf640cc", "design-single-selection",
                        "design-pairing-check", "8a27ac49"):
            self.execute(cell_id)

    def execute(self, cell_id):
        with redirect_stdout(StringIO()):
            exec(compile(self.cells[cell_id], f"synthetic:{cell_id}", "exec"), self.env)

    def save(self, name, record, *, design=False):
        path = (self.designs if design else self.references) / name
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def save_singles(self):
        return {
            variant: self.save(f"single-{index}.json", invented_result(variant))
            for index, variant in enumerate(self.env["SINGLE_VARIANTS"])
        }

    def original_encoder_result(self, variant, *, legacy=False, **settings):
        """Use the declared original mapping while keeping every value invented."""
        self.env["blf"].ENCODER_OF_MODALITY = ORIGINAL_ENCODERS
        record = invented_result(variant, **settings)
        record["identity"]["inputs"]["encoders"] = dict(ORIGINAL_ENCODERS)
        if legacy:
            record.pop("identity")
        return record

    def approve_legacy(self, paths):
        """Bind temporary reference files to an invented later-audit context."""
        exemplar = self.original_encoder_result("concat")
        context = deepcopy(exemplar["identity"]["inputs"])
        context.pop("encoders")
        references, sources = [], {}
        for path in paths:
            data = json.loads(path.read_bytes())
            relative = path.relative_to(self.folder).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            references.append({"path": relative, "sha256": digest,
                               "benchmark": data["benchmark"], "variant": data["variant"]})
            sources["D:\\historical\\repository\\" + relative.replace("/", "\\")] = digest
        audit = self.folder / "audit.json"
        audit.write_text(json.dumps({"identity": {
            "schema": "swap.paired-report.v1", "benchmark": "fixture",
            "config": RECIPE, "context": context, "sources": sources,
        }}), encoding="utf-8")
        manifest = self.env["LEGACY_REFERENCE_RECORD"]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "schema": "blf.reference-reuse.v1", "scope": "tier4_designs.references",
            "encoders": ORIGINAL_ENCODERS, "references": references,
            "audits": {"fixture": {"path": audit.relative_to(self.folder).as_posix(),
                                    "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}},
            "qualification": "Synthetic accepted comparison evidence; producer unknown.",
        }), encoding="utf-8")

    def save_legacy_singles(self, *, missing_predictions=False):
        paths = []
        for index, variant in enumerate(self.env["SINGLE_VARIANTS"]):
            native_fallback = variant in ("unimodal:rna", "unimodal:prot")
            record = self.original_encoder_result(
                variant, legacy=True, tconv="appendix" if native_fallback else "equation",
            )
            if missing_predictions and native_fallback:
                record.pop("test_truth")
                record.pop("test_prediction")
            paths.append(self.save(f"legacy-{index}.json", record))
        self.approve_legacy(paths)
        return paths

    def inventory(self):
        before = {path: path.read_bytes() for path in self.folder.rglob("*.json")}
        self.execute("design-result-inventory")
        self.execute("design-score-display")
        self.assertEqual(before, {path: path.read_bytes() for path in self.folder.rglob("*.json")})
        return self.env["runs"]["fixture"]

    def test_selection_uses_validation_and_declared_tie_order(self):
        self.save_singles()
        found = self.inventory()
        self.assertEqual(found["best single name"], "unimodal:dna")
        self.assertEqual(self.env["coverage_rows"][0]["input provenance"], "recorded inputs")
        record = invented_result("unimodal:rna", val_score=0.8, test_score=-0.9)
        self.save("single-1.json", record)
        found = self.inventory()
        self.assertEqual(found["best single name"], "unimodal:rna")

    def test_empty_results_report_missing_without_bootstrapping(self):
        found = self.inventory()
        self.assertIsNone(found["best single"])
        self.execute("design-bootstrap-display")
        self.assertEqual(len(self.env["rows"]), len(self.env["CONTRASTS"]))
        self.assertTrue(all(row["note"] == "missing" for row in self.env["rows"]))
        self.bootstrap.assert_not_called()

    def test_missing_invalid_or_duplicate_single_never_selects_partial_set(self):
        paths = self.save_singles()
        target = paths["unimodal:rna"]
        original = target.read_bytes()
        for mode in ("missing", "invalid", "duplicate"):
            with self.subTest(mode=mode):
                target.write_bytes(original)
                duplicate = self.references / "duplicate.json"
                duplicate.unlink(missing_ok=True)
                if mode == "missing":
                    target.unlink()
                elif mode == "invalid":
                    data = json.loads(original)
                    data["val_score"] = None
                    target.write_text(json.dumps(data), encoding="utf-8")
                else:
                    duplicate.write_bytes(original)
                found = self.inventory()
                self.assertIsNone(found["best single"])
                self.assertIn("unimodal:dna", found)
                self.assertIn("coverage incomplete", self.env["coverage_rows"][0]["selection issue"])

    def test_legacy_scores_remain_visible_but_selection_is_unverified(self):
        for variant, path in self.save_singles().items():
            record = invented_result(variant)
            record.pop("identity")
            path.write_text(json.dumps(record), encoding="utf-8")
        found = self.inventory()
        self.assertIsNone(found["best single"])
        self.assertEqual(len([v for v in self.env["SINGLE_VARIANTS"] if v in found]), 4)
        self.assertTrue(self.env["coverage_rows"][0]["selection issue"])

    def test_selection_refuses_different_recorded_cohorts(self):
        self.save_singles()
        changed = invented_result("unimodal:rna", val_score=0.99)
        changed["identity"]["inputs"]["rows"]["sha256"] = "b" * 64
        self.save("single-1.json", changed)
        self.assertIsNone(self.inventory()["best single"])
        self.assertIn("different recorded", self.env["coverage_rows"][0]["selection issue"])

    def test_approved_legacy_selection_needs_scores_not_missing_prediction_arrays(self):
        paths = self.save_legacy_singles(missing_predictions=True)
        found = self.inventory()
        self.assertEqual(found["best single name"], "unimodal:dna")
        self.assertEqual(self.env["coverage_rows"][0]["input provenance"],
                         "qualified legacy reference reuse")
        self.assertEqual(found["unimodal:rna"]["config"]["tconv"], "appendix")
        for path in paths:
            self.assertNotIn("identity", json.loads(path.read_bytes()))
        design = self.original_encoder_result("coupled_mamba")
        with self.assertRaisesRegex(ValueError, "targets or predictions absent"):
            self.env["paired_design_comparison"](
                design, found["unimodal:rna"], "regression",
                self.env["reference_sources"]["fixture"],
            )
        self.bootstrap.assert_not_called()

    def test_disabling_legacy_reuse_keeps_approved_scores_but_blocks_selection(self):
        self.save_legacy_singles()
        self.env["ALLOW_LEGACY_REFERENCE_REUSE"] = False
        found = self.inventory()
        self.assertIsNone(found["best single"])
        self.assertTrue(all(name in found for name in self.env["SINGLE_VARIANTS"]))
        self.assertTrue(self.env["coverage_rows"][0]["selection issue"])
        self.bootstrap.assert_not_called()

    def test_approved_file_with_invalid_modern_identity_cannot_use_legacy_fallback(self):
        paths = self.save_legacy_singles()
        target = paths[0]
        original = json.loads(target.read_bytes())
        for identity in (None, {}, {"schema": "broken"}):
            with self.subTest(identity=identity):
                target.write_text(json.dumps({**original, "identity": identity}), encoding="utf-8")
                self.approve_legacy(paths)
                self.assertIsNone(self.inventory()["best single"])
        self.bootstrap.assert_not_called()

    def test_qualified_reference_pair_retains_label_arrays_and_bootstrap_settings(self):
        legacy = self.original_encoder_result("concat", legacy=True)
        path = self.save("legacy-concat.json", legacy)
        self.approve_legacy([path])
        modern = self.original_encoder_result("coupled_mamba")
        self.save("coupled.json", modern, design=True)
        before = deepcopy((legacy, modern))
        files_before = {p: p.read_bytes() for p in self.folder.rglob("*.json")}
        self.inventory()
        self.execute("design-bootstrap-display")
        self.bootstrap.assert_called_once()
        args, kwargs = self.bootstrap.call_args
        self.assertEqual(args[0].test_prediction, modern["test_prediction"])
        self.assertEqual(args[1].test_truth, legacy["test_truth"])
        self.assertEqual(kwargs, {"draws": 2000, "seed": 0})
        row = next(r for r in self.env["rows"] if r["contrast"] == "coupled_mamba - concat")
        self.assertEqual(row["input provenance"], "qualified legacy reference reuse")
        self.assertEqual(row["note"], "")
        self.assertEqual((legacy, modern), before)
        self.assertEqual(files_before, {p: p.read_bytes() for p in self.folder.rglob("*.json")})

    def test_qualified_pair_still_rejects_changed_modern_rows_splits_targets_or_encoders(self):
        legacy = self.original_encoder_result("concat", legacy=True)
        path = self.save("legacy-concat.json", legacy)
        self.approve_legacy([path])
        sources = {"concat": path}
        for field in ("rows", "splits", "targets", "encoders"):
            with self.subTest(field=field):
                modern = self.original_encoder_result("coupled_mamba")
                inputs = modern["identity"]["inputs"]
                if field == "splits":
                    inputs[field]["test"]["sha256"] = "b" * 64
                elif field == "encoders":
                    inputs[field]["rna"] = "different_encoder"
                else:
                    inputs[field]["sha256"] = "b" * 64
                with self.assertRaisesRegex(ValueError, "ordered inputs"):
                    self.env["paired_design_comparison"](modern, legacy, "regression", sources)
        self.bootstrap.assert_not_called()

    def test_changed_approved_file_blocks_selection_without_changing_saved_scores(self):
        paths = self.save_legacy_singles()
        paths[0].write_bytes(paths[0].read_bytes() + b"\n")
        found = self.inventory()
        self.assertIsNone(found["best single"])
        self.assertIn("unimodal:dna", found)
        self.assertTrue(self.env["coverage_rows"][0]["selection issue"])
        self.bootstrap.assert_not_called()

    def test_approved_source_is_rechecked_after_inventory_before_bootstrap(self):
        reference = self.original_encoder_result("concat", legacy=True)
        path = self.save("legacy-concat.json", reference)
        self.approve_legacy([path])
        self.save("coupled.json", self.original_encoder_result("coupled_mamba"), design=True)
        self.inventory()
        path.write_bytes(path.read_bytes() + b"\n")
        self.execute("design-bootstrap-display")
        row = next(r for r in self.env["rows"] if r["contrast"] == "coupled_mamba - concat")
        self.assertTrue(row["note"])
        self.bootstrap.assert_not_called()

    def test_legacy_design_is_never_given_a_reference_source(self):
        reference = self.original_encoder_result("concat", legacy=True)
        reference_path = self.save("legacy-concat.json", reference)
        design = self.original_encoder_result("coupled_mamba", legacy=True)
        design_path = self.save("legacy-design.json", design, design=True)
        self.approve_legacy([reference_path, design_path])
        self.inventory()
        self.assertNotIn("coupled_mamba", self.env["reference_sources"]["fixture"])
        self.execute("design-bootstrap-display")
        row = next(r for r in self.env["rows"] if r["contrast"] == "coupled_mamba - concat")
        self.assertTrue(row["note"])
        self.bootstrap.assert_not_called()

    def test_reader_checks_content_not_filename_and_preserves_native_fallback(self):
        self.save_singles()
        self.save("coupled_mamba__seed0.json", invented_result("coupled_mamba", tconv="appendix"),
                  design=True)
        fallback = invented_result("unimodal:rna", tconv="appendix")
        self.save("single-1.json", fallback)
        found = self.inventory()
        self.assertEqual(found["unimodal:rna"]["config"]["tconv"], "appendix")
        self.assertNotIn("coupled_mamba", found)
        self.assertIsNotNone(found["best single"])
        self.save("preferred.json", invented_result("unimodal:rna"))
        self.assertEqual(self.inventory()["unimodal:rna"]["config"]["tconv"], "equation")

    def test_pairing_rejects_reordered_rows_even_when_all_test_labels_equal(self):
        first, second = invented_result("coupled_mamba"), invented_result("concat")
        self.assertEqual(first["test_truth"], second["test_truth"])
        second["identity"]["inputs"]["rows"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "ordered inputs"):
            self.env["paired_design_comparison"](first, second, "regression")
        self.bootstrap.assert_not_called()

    def test_pairing_rejects_different_splits_and_unknown_provenance(self):
        first = invented_result("coupled_mamba")
        for mode in ("split", "legacy", "partial"):
            with self.subTest(mode=mode):
                second = invented_result("concat")
                if mode == "split":
                    second["identity"]["inputs"]["splits"]["test"]["sha256"] = "b" * 64
                elif mode == "legacy":
                    second.pop("identity")
                else:
                    second["identity"]["inputs"].pop("rows")
                with self.assertRaises(ValueError):
                    self.env["paired_design_comparison"](first, second, "regression")
        self.bootstrap.assert_not_called()

    def test_pairing_checks_saved_target_fingerprint_and_prediction_arrays(self):
        for mode in ("wrong_targets", "missing_predictions", "nonfinite", "wrong_length"):
            with self.subTest(mode=mode):
                first, second = invented_result("coupled_mamba"), invented_result("concat")
                if mode == "wrong_targets":
                    first["test_truth"] = second["test_truth"] = [2.0, 2.0]
                elif mode == "missing_predictions":
                    second.pop("test_prediction")
                elif mode == "nonfinite":
                    second["test_prediction"] = [float("nan"), 1.0]
                else:
                    second["test_prediction"] = [1.0]
                with self.assertRaises(ValueError):
                    self.env["paired_design_comparison"](first, second, "regression")
        self.bootstrap.assert_not_called()

    def test_valid_pair_preserves_original_arrays_bootstrap_draws_and_seed(self):
        first, second = invented_result("coupled_mamba"), invented_result("concat")
        before = deepcopy((first, second))
        value = self.env["paired_design_comparison"](first, second, "regression")
        self.assertEqual(value, self.bootstrap.return_value)
        args, kwargs = self.bootstrap.call_args
        self.assertEqual(args[0].test_truth, first["test_truth"])
        self.assertEqual(args[1].test_prediction, second["test_prediction"])
        self.assertEqual(args[2], "regression")
        self.assertEqual(kwargs, {"draws": 2000, "seed": 0})
        self.assertEqual((first, second), before)

    def test_classification_requires_valid_class_ids_and_matching_task(self):
        first, second = invented_result("coupled_mamba"), invented_result("concat")
        for record in (first, second):
            inputs = record["identity"]["inputs"]
            labels = np.array([0, 1, 1, 0, 1, 1], dtype=np.int64)
            inputs["targets"].update(run_identity.array_record(labels), task="classification")
            inputs["split_targets"] = {
                key: run_identity.array_record(labels[indices])
                for key, indices in {"train": [0, 1], "val": [2, 3], "test": [4, 5]}.items()
            }
            record["test_truth"] = [1, 1]
            record["test_prediction"] = [0, 2]
        self.env["paired_design_comparison"](first, second, "classification")
        self.bootstrap.assert_called_once()
        self.bootstrap.reset_mock()
        for predictions in ([0.5, 1], [0, 3], [-1, 1]):
            second["test_prediction"] = predictions
            with self.assertRaises(ValueError):
                self.env["paired_design_comparison"](first, second, "classification")
        with self.assertRaisesRegex(ValueError, "task differs"):
            self.env["paired_design_comparison"](first, first, "regression")
        self.bootstrap.assert_not_called()

    def test_missing_best_does_not_block_independent_verified_contrast(self):
        self.save("concat.json", invented_result("concat"))
        self.save("coupled.json", invented_result("coupled_mamba"), design=True)
        self.inventory()
        self.execute("design-bootstrap-display")
        self.bootstrap.assert_called_once()
        rows = self.env["rows"]
        self.assertEqual(next(r for r in rows if r["contrast"] == "coupled_mamba - concat")["note"], "")
        self.assertEqual(next(r for r in rows if r["contrast"] == "coupled_mamba - best single")["note"], "missing")

    def test_saved_mode_never_opens_streams_or_requests_producer_identity(self):
        self.env["run_identity"] = SimpleNamespace(execution_records=Mock(
            side_effect=AssertionError("Producer identity requested in saved mode")))
        self.execute("12088db5")
        self.env["blf"].BenchmarkStreams.open.assert_not_called()
        self.env["train"].run.assert_not_called()
        self.env["run_identity"].execution_records.assert_not_called()

    def test_training_wires_existing_identity_adapter_without_changing_fit_requests(self):
        streams = SimpleNamespace(close=Mock())
        opened = Mock(return_value=streams)
        records = {"run_identity": {"fixture": True}, "producer": {"fixture": True}}
        producer = Mock(return_value=records)
        fit = Mock()
        self.env.update(READ_SAVED_RESULTS=False,
                        run_identity=SimpleNamespace(execution_records=producer),
                        blf=SimpleNamespace(BenchmarkStreams=SimpleNamespace(open=opened)),
                        train=SimpleNamespace(run=fit), DEVICE="cpu")
        self.execute("12088db5")
        self.assertEqual([call.args[1] for call in fit.call_args_list], list(self.env["DESIGNS"]))
        for call in fit.call_args_list:
            self.assertEqual(call.args[0], streams)
            self.assertEqual(call.args[2:], (self.env["CONFIG"], "cpu", self.designs))
            self.assertEqual(call.kwargs, records)
        self.assertEqual(producer.call_count, len(self.env["DESIGNS"]))
        streams.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
