"""Synthetic path contracts; no real results, imports, caches or fitting."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion.result_sets import NewtonSelectionSnapshot, resolve_swap_results
from shared_code.fusion import swap_results


class ResultSetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.code = self.root / "Code"

    def test_historical_paths_and_fungal_sources_are_unchanged(self):
        selected = resolve_swap_results(self.code)
        self.assertEqual(selected.swaps, self.code / "stage4/results/phaseB")
        self.assertEqual(selected.originals["fungal_expression"], (
            self.code / "stage3/results/fungal_expression",
            self.code / "stage4/results/phaseA/fungal_expression",
        ))
        self.assertFalse(selected.fixed_historical_selection)
        selected.require_writable_fits()

    def test_newton_selects_scores_and_historical_choices_from_one_package(self):
        selected = resolve_swap_results(self.code, "newton_h100_seeds012")
        root = self.code / "stage4/results/newton_h100_seeds012"
        self.assertEqual(selected.originals["fungal_expression"],
                         (root / "outputs/phaseA/fungal_expression",))
        self.assertEqual(selected.selection_swaps,
                         self.code / "stage4/results/phaseB")
        self.assertEqual(selected.selection_snapshot.path, root / "provenance/source-snapshot.json")
        self.assertTrue(selected.fixed_historical_selection)
        self.assertNotEqual(selected.analysis_dir, selected.saved_report_dirs[1])
        with self.assertRaisesRegex(ValueError, "read-only"):
            selected.require_writable_fits()

    def test_resolving_absent_evidence_creates_nothing_and_does_not_fallback(self):
        before = list(self.root.iterdir())
        with patch.object(Path, "mkdir", side_effect=AssertionError("must not write")):
            selected = resolve_swap_results(self.code, "newton_h100_seeds012")
        self.assertEqual(before, list(self.root.iterdir()))
        self.assertFalse(selected.swaps.exists())
        self.assertTrue(all(selected.root in p.parents for v in selected.originals.values() for p in v))

    def test_explicit_return_location_and_invalid_or_overlapping_roots(self):
        selected = resolve_swap_results(self.code, "newton_h100_seeds012",
                                        workspace=self.root / "returned")
        self.assertEqual(selected.root, self.root / "returned")
        for bad in (self.code, self.code / "stage4/results/phaseB",
                    self.code / "stage4/results/phaseA/inside"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "overlaps"):
                resolve_swap_results(self.code, "newton_h100_seeds012", workspace=bad)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_swap_results(self.code, "latest")
        with self.assertRaisesRegex(ValueError, "override"):
            resolve_swap_results(self.code, workspace=self.root)


class NewtonSelectionSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "Code"
        self.path = self.root / "returned/provenance/source-snapshot.json"
        self.path.parent.mkdir(parents=True)
        self.entries = []

    def add_record(self, relative, data):
        path = self.code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.entries.append({"source": "Code/" + relative,
                             "destination": "historical_results/Code/" + relative,
                             "initial_copy_sha256": digest, "source_sha256": digest})
        return path

    def snapshot(self):
        self.path.write_text(json.dumps({"schema_version": 1, "files": self.entries}), encoding="utf-8")
        return NewtonSelectionSnapshot(self.path, self.code,
                                       hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_exact_files_map_to_canonical_paths_without_returned_duplicates(self):
        path = self.add_record("stage4/results/phaseA/fixture/fit.json", {"value": 1})
        snapshot = self.snapshot()
        with patch.object(Path, "glob", side_effect=AssertionError("No directory candidate discovery")):
            self.assertEqual(snapshot.file_hashes(), {path: hashlib.sha256(path.read_bytes()).hexdigest()})
        self.assertFalse((self.root / "returned/historical_results").exists())

    def test_missing_changed_or_changed_manifest_is_refused(self):
        path = self.add_record("stage4/results/phaseA/fixture/fit.json", {"value": 1})
        snapshot = self.snapshot()
        original = path.read_bytes()
        path.write_text('{"value": 2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            snapshot.file_hashes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            snapshot.file_hashes()
        path.write_bytes(original)
        self.path.write_bytes(self.path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "original recorded bytes"):
            snapshot.file_hashes()

    def test_manifest_path_traversal_and_unrelated_paths_are_refused(self):
        self.add_record("stage4/results/phaseA/fixture/fit.json", {"value": 1})
        for destination in ("historical_results/Code/../outside.json",
                            "historical_results/Code/stage4/results/phaseA/../../outside.json",
                            "historical_results/Code/stage4/results/phaseA\\outside.json",
                            "historical_results/Code/shared_code/outside.json"):
            with self.subTest(destination=destination):
                self.entries[0]["destination"] = destination
                self.entries[0]["source"] = destination.removeprefix("historical_results/")
                with self.assertRaisesRegex(ValueError, "Invalid Newton selection path"):
                    self.snapshot().file_hashes()

    def test_canonical_path_redirect_cannot_alias_fresh_outputs_even_with_same_bytes(self):
        path = self.add_record("stage4/results/phaseA/fixture/fit.json", {"value": 1})
        snapshot = self.snapshot()
        redirected = self.code / "stage4/results/newton_h100_seeds012/outputs/fit.json"
        redirected.parent.mkdir(parents=True)
        redirected.write_bytes(path.read_bytes())
        original_resolve = Path.resolve

        def resolve(candidate, *args, **kwargs):
            return redirected if candidate == path else original_resolve(candidate, *args, **kwargs)

        # Simulate a junction/symlink without requiring Windows link privileges.
        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "escapes or aliases"):
                snapshot.file_hashes()

    def test_new_unlisted_results_cannot_change_validation_selection(self):
        recipe = {"seed": 0, "tconv": "equation", "entropy_lambda": 0.0}
        encoders = {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"}
        folder = self.code / "stage4/results/phaseA/fixture"
        variants = ("unimodal:dna", "unimodal:rna", "unimodal:prot", "unimodal-grid:rna")
        for index, variant in enumerate(variants):
            self.add_record(f"stage4/results/phaseA/fixture/single{index}.json", {
                "benchmark": "fixture", "variant": variant, "config": recipe,
                "epochs": 2, "best_epoch": 1, "seconds": 1., "val_score": .8 - index / 10,
                "test_score": .2})
        for penalty in (.01, .5, 1.):
            self.add_record(f"stage4/results/phaseA/fixture/mil{penalty}.json", {
                "benchmark": "fixture", "variant": "mil",
                "config": {**recipe, "entropy_lambda": penalty},
                "epochs": 2, "best_epoch": 1, "seconds": 1., "val_score": -penalty,
                "test_score": .2})
        snapshot = self.snapshot()
        unlisted = json.loads((folder / "single1.json").read_text())
        unlisted["val_score"] = 100.
        (folder / "new-superior-rna.json").write_text(json.dumps(unlisted))
        with patch.object(Path, "glob", side_effect=AssertionError("No directory candidate discovery")):
            selected = swap_results.read_validation_selection(
                benchmark="fixture", config=recipe, original_folders=(folder,),
                original_encoders=encoders, swaps=(),
                swap_root=self.code / "stage4/results/phaseB/fixture", selection_snapshot=snapshot)
        self.assertEqual(selected["best_singles"], {"original": "unimodal:dna"})
        self.assertEqual(selected["mil_lambda"], .01)
        self.assertNotIn(str(folder / "new-superior-rna.json"), selected["sources"])


if __name__ == "__main__":
    unittest.main()
