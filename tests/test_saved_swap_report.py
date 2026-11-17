"""Read saved synthetic reports without scoring, datasets, caches, or writes."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np  # The project environment requires NumPy before PyTorch.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import swap_reporting
from shared_code.fusion.result_sets import NewtonSelectionSnapshot


class SavedSwapReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "returned/outputs"
        self.historical = self.root / "returned/historical_results/Code"
        self.reports = self.outputs / "phaseB/analyses/paired-report-v1"
        self.local = self.root / "returned/local_analyses/paired-report-v1"
        self.source = self.outputs / "phaseA/fixture/fit.json"
        self.selection = self.historical / "stage4/results/phaseA/fixture/selection.json"
        for path in (self.source, self.selection):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"synthetic": true}\n', encoding="utf-8")
        self.identity = {
            "schema": swap_reporting.SCHEMA, "benchmark": "fixture", "seeds": [0, 1, 2],
            "selection_seed": 0, "selection_policy": "fixed-historical-seed0-validation",
            "config": {"recipe": "synthetic"}, "draws": 2000, "bootstrap_seed": 0,
            "sources": {"/newton/project/outputs/phaseA/fixture/fit.json": self.sha(self.source)},
            "selection_sources": {
                "/newton/project/historical_results/Code/stage4/results/phaseA/fixture/selection.json": self.sha(self.selection)},
            "source_sha256": {"swap_reporting.py": "a" * 64, "swap_results.py": "b" * 64,
                              "run_identity.py": "c" * 64},
        }
        contrast = {"encoder": "synthetic", "kind": "replacement", "comparison": "synthetic",
                    "first": "synthetic/concat", "second": "original/concat"}
        self.analysis = {
            "per_seed": [{**contrast, "seed": seed, "difference": 1.0} for seed in (0, 1, 2)],
            "seed_summary": [{**contrast, "available_seeds": [0, 1, 2], "missing_seeds": [],
                              "status": "complete", "difference": 1.0}],
            "issues": [], "rows": 2, "groups": 2, "units": "accuracy percentage points",
        }
        self.path = self.publish_fixture(self.identity)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def publish_fixture(self, identity, *, directory=None, analysis=None):
        analysis = self.analysis if analysis is None else analysis
        saved = {"identity": identity, "analysis": analysis,
                 "analysis_sha256": swap_reporting._digest(analysis),
                 "producer": {"host": "newton-fixture", "git": "historical-fixture"}}
        directory = self.reports if directory is None else directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fixture__{swap_reporting._digest(identity)[:20]}.json"
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        return path

    def snapshot(self):
        return {path.relative_to(self.root): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def read(self, **changes):
        before = self.snapshot()
        try:
            with ExitStack() as blocked:
                for name in ("calculate", "report", "benchmark_context", "collect", "_score"):
                    blocked.enter_context(patch.object(swap_reporting, name,
                        side_effect=AssertionError("Saved-report reading attempted computation")))
                for name in ("mkdir", "write_text", "write_bytes"):
                    blocked.enter_context(patch.object(Path, name,
                        side_effect=AssertionError("Saved-report reading attempted a write")))
                return swap_reporting.read_existing_report(**{
                    "benchmark": "fixture", "report_dirs": (self.local, self.reports),
                    "seeds": (0, 1, 2),
                    "expected": {"config": {"recipe": "synthetic"}, "draws": 2000,
                                 "bootstrap_seed": 0,
                                 "selection_policy": "fixed-historical-seed0-validation"},
                    "source_roots": {"outputs": self.outputs,
                                     "historical_results/Code": self.historical}, **changes})
        finally:
            self.assertEqual(before, self.snapshot())

    def test_newton_sources_map_to_exact_local_trees_and_metadata_stays_original(self):
        saved, path = self.read()
        self.assertEqual(path, self.path)
        self.assertEqual(saved["identity"], self.identity)
        self.assertEqual(saved["producer"]["host"], "newton-fixture")
        self.assertEqual(saved["identity"]["source_sha256"]["swap_reporting.py"], "a" * 64)
        self.assertFalse(self.local.exists())

    def relocated_selection(self):
        relative = self.selection.relative_to(self.historical)
        canonical = self.root / "Code" / relative
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(self.selection.read_bytes())
        self.selection.unlink()
        digest = self.sha(canonical)
        path = self.root / "returned/provenance/source-snapshot.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema_version": 1, "files": [{
            "source": "Code/" + relative.as_posix(),
            "destination": "historical_results/Code/" + relative.as_posix(),
            "source_sha256": digest, "initial_copy_sha256": digest,
        }]}), encoding="utf-8")
        return NewtonSelectionSnapshot(path, self.root / "Code", self.sha(path)), canonical

    def test_relocated_selection_reads_original_report_without_rewriting_or_analysis(self):
        snapshot, canonical = self.relocated_selection()
        before = self.path.read_bytes()
        saved, path = self.read(selection_snapshot=snapshot)
        self.assertEqual(path, self.path)
        self.assertEqual(saved["identity"], self.identity)
        self.assertEqual(before, path.read_bytes())
        self.assertNotIn(str(canonical), saved["identity"]["selection_sources"])

    def test_old_local_return_path_inside_canonical_code_maps_by_original_marker(self):
        snapshot, canonical = self.relocated_selection()
        old_path = (self.root / "Code/stage4/results/newton_h100_seeds012/historical_results/Code"
                    / canonical.relative_to(self.root / "Code"))
        identity = {**self.identity, "selection_sources": {str(old_path): self.sha(canonical)}}
        path = self.publish_fixture(identity, directory=self.local)
        saved, actual = self.read(selection_snapshot=snapshot, selected_report=path)
        self.assertEqual(actual, path)
        self.assertEqual(saved["identity"], identity)

    def test_relocated_selection_refuses_missing_changed_and_unlisted_evidence(self):
        snapshot, canonical = self.relocated_selection()
        original = canonical.read_bytes()
        canonical.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection file changed"):
            self.read(selection_snapshot=snapshot)
        canonical.unlink()
        with self.assertRaisesRegex(ValueError, "selection file is missing"):
            self.read(selection_snapshot=snapshot)
        canonical.write_bytes(original)
        unlisted = canonical.with_name("unlisted.json")
        unlisted.write_bytes(original)
        identity = {**self.identity, "selection_sources": {
            "/newton/project/historical_results/Code/stage4/results/phaseA/fixture/unlisted.json": self.sha(unlisted)}}
        path = self.publish_fixture(identity, directory=self.local)
        with self.assertRaisesRegex(ValueError, "not in the original Newton snapshot"):
            self.read(selection_snapshot=snapshot, selected_report=path)

    def test_relocated_report_refuses_selection_path_traversal(self):
        snapshot, canonical = self.relocated_selection()
        identity = {**self.identity, "selection_sources": {
            "/newton/project/historical_results/Code/stage4/results/phaseA/fixture/../fixture/selection.json": self.sha(canonical)}}
        path = self.publish_fixture(identity, directory=self.local)
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            self.read(selection_snapshot=snapshot, selected_report=path)

    def test_local_absolute_sources_use_the_same_bounded_roots(self):
        identity = {**self.identity, "sources": {str(self.source): self.sha(self.source)},
                    "selection_sources": {str(self.selection): self.sha(self.selection)}}
        path = self.publish_fixture(identity, directory=self.local)
        saved, actual = self.read(selected_report=path)
        self.assertEqual(actual, path)
        self.assertEqual(saved["identity"], identity)

    def test_historical_reports_can_read_original_absolute_paths_without_mapping(self):
        identity = {**self.identity, "sources": {str(self.source): self.sha(self.source)}}
        identity.pop("selection_policy")
        identity.pop("selection_sources")
        path = self.publish_fixture(identity, directory=self.local)
        saved, actual = self.read(selected_report=path, source_roots=None,
                                  expected={"draws": 2000, "bootstrap_seed": 0})
        self.assertEqual(actual, path)
        self.assertNotIn("selection_sources", saved["identity"])

    def test_missing_report_is_unavailable_without_creating_directories(self):
        missing = self.root / "missing-reports"
        self.assertEqual(self.read(report_dirs=(missing,)), (None, None))
        self.assertEqual(self.read(selected_report=self.local / "absent.json"), (None, None))
        self.assertFalse(missing.exists())

    def test_multiple_eligible_reports_require_selection_even_across_directory_order(self):
        second = self.publish_fixture({**self.identity, "groups_sha256": "d" * 64}, directory=self.local)
        for directories in ((self.local, self.reports), (self.reports, self.local)):
            with self.subTest(directories=directories), self.assertRaisesRegex(ValueError, "Multiple eligible"):
                self.read(report_dirs=directories)
        self.assertEqual(self.read(selected_report=second)[1], second)
        self.assertEqual(self.read(selected_report=self.path.name)[1], self.path)

    def test_expected_parameters_choose_exact_match_and_ignore_other_draw_counts(self):
        self.publish_fixture({**self.identity, "draws": 100}, directory=self.local)
        self.assertEqual(self.read()[1], self.path)
        self.assertEqual(self.read(expected={"draws": 17}), (None, None))
        with self.assertRaisesRegex(ValueError, "differs from the requested identity"):
            self.read(selected_report=self.path, expected={"draws": 17})

    def test_other_seed_sets_do_not_fill_requested_seed_gaps(self):
        self.assertEqual(self.read(seeds=(0, 1)), (None, None))
        with self.assertRaisesRegex(ValueError, "requested identity"):
            self.read(selected_report=self.path, seeds=(0, 1))

    def test_analysis_corruption_is_rejected_and_bytes_preserved(self):
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        saved["analysis"]["per_seed"][0]["difference"] = 9.0
        self.path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "analysis.*hash"):
            self.read()

    def test_identity_corruption_cannot_keep_an_old_filename(self):
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        saved["identity"]["context"] = {"changed": True}
        self.path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity hash"):
            self.read()

    def test_unknown_schema_is_rejected(self):
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        saved["identity"]["schema"] = "future-unknown"
        self.path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema"):
            self.read()

    def test_changed_or_missing_sources_stop_reading_without_fallback(self):
        for source in (self.source, self.selection):
            before = source.read_bytes()
            with self.subTest(source=source):
                source.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, "source hash differs"):
                    self.read()
                source.unlink()
                with self.assertRaisesRegex(ValueError, "source unavailable"):
                    self.read()
                source.write_bytes(before)

    def test_parent_traversal_and_wrong_evidence_role_are_rejected(self):
        for source in ("/newton/outputs/../escape.json",
                       "/newton/historical_results/Code/fit.json",
                       "/newton/outputs/one/outputs/fit.json"):
            with self.subTest(source=source):
                path = self.publish_fixture({**self.identity, "sources": {source: self.sha(self.source)}},
                                            directory=self.local)
                with self.assertRaisesRegex(ValueError, "parent traversal|unique outputs root"):
                    self.read(selected_report=path)

    def test_invalid_source_hash_and_missing_historical_selection_are_rejected(self):
        for changes, error in (({"source_sha256": {"swap_reporting.py": "bad"}}, "invalid.*hash"),
                               ({"selection_sources": {}}, "selection sources")):
            path = self.publish_fixture({**self.identity, **changes}, directory=self.local)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                self.read(selected_report=path)

    def test_summary_cannot_claim_a_complete_mean_from_partial_requested_seeds(self):
        analysis = deepcopy(self.analysis)
        analysis["seed_summary"][0]["available_seeds"] = [0, 1]
        path = self.publish_fixture(self.identity, analysis=analysis)
        with self.assertRaisesRegex(ValueError, "seed summary"):
            self.read(selected_report=path)

    def test_analysis_display_fields_are_validated_before_returning(self):
        for key in ("units", "first", "encoder"):
            analysis = deepcopy(self.analysis)
            if key == "units":
                analysis.pop(key)
            else:
                analysis["per_seed"][0].pop(key)
            path = self.publish_fixture(self.identity, analysis=analysis)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "incomplete|invalid fields"):
                self.read(selected_report=path)

    def test_selected_report_must_remain_inside_declared_report_directories(self):
        path = self.publish_fixture(self.identity, directory=self.root / "unselected")
        with self.assertRaisesRegex(ValueError, "outside the declared"):
            self.read(selected_report=path)


if __name__ == "__main__":
    unittest.main()
