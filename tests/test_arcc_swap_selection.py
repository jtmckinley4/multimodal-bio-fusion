"""Synthetic selection routing checks; no models, caches, notebooks or research runs.

The imported ARCC adapters freeze historical choices while keeping fitted cohorts
separate. These fixtures contain only invented results and use no research data.
"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np  # Preserve NumPy-before-PyTorch order in the project environment.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import run_identity, swap_plan, swap_reporting, swap_results


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5,
              entropy_lambda=0.0, channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="nt", rna="rnafm", prot="esm8")
SWAPS = (("dna", "nt500", True), ("dna", "hyena", True),
         ("rna", "rinalmo", True), ("rna", "mrnafm", False),
         ("rna", "calm", False), ("prot", "esm35", False),
         ("prot", "esm150", False), ("prot", "protbert", False))
SINGLES = ("unimodal:dna", "unimodal:rna", "unimodal:prot", "unimodal-grid:rna")


class HistoricalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.historical_original = self.root / "historical" / "original"
        self.historical_swaps = self.root / "historical" / "swaps"
        self.fresh_original = self.root / "fresh" / "original"
        self.fresh_swaps = self.root / "fresh" / "swaps"
        for variant in SINGLES:
            self.write(self.historical_original, variant, ORIGINAL,
                       score=0.8 if variant == "unimodal:dna" else 0.5)
        for penalty in (0.01, 0.5, 1.0):
            self.write(self.historical_original, "mil", ORIGINAL, penalty=penalty,
                       score=0.9 if penalty == 0.5 else 0.4)
        for modality, encoder, grid in SWAPS:
            for variant in swap_results.swap_variants(modality, grid=grid):
                if variant.startswith("unimodal"):
                    self.write(self.historical_swaps / encoder, variant,
                               {**ORIGINAL, modality: encoder}, score=0.6)

    @staticmethod
    def identity(benchmark, variant, recipe, encoders):
        # A fully versioned custom fixture identity; no real caches or loaders.
        return run_identity.make_identity(
            benchmark=benchmark, variant=variant, config=recipe,
            inputs={"encoders": encoders}, model={"contract": "synthetic-selection-test"},
            loader={"kind": "custom", "recovery": {name: {} for name in ("train", "val", "test")}})

    def write(self, folder, variant, encoders, *, seed=0, penalty=0.0,
              score=0.5, modern=False):
        recipe = {**RECIPE, "seed": seed, "entropy_lambda": penalty}
        path = swap_plan.result_path(folder, "fixture", variant, recipe)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = (self.identity("fixture", variant, recipe, encoders) if modern
                    else {"inputs": {"encoders": encoders}})
        record = dict(benchmark="fixture", variant=variant, config=recipe,
                      val_score=score, test_score=0.5, epochs=1, best_epoch=0, seconds=1,
                      identity=identity, test_truth=[0, 1], test_prediction=[0, 1])
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def selection_routes(self):
        return dict(selection_original_folders=(self.historical_original,),
                    selection_swap_root=self.historical_swaps)

    def plan(self, seed=0, **changes):
        return swap_plan.plan_benchmark(**{
            "benchmark": "fixture", "config": {**RECIPE, "seed": seed},
            "original_encoders": ORIGINAL, "original_folders": (self.fresh_original,),
            "original_destination": self.fresh_original, "swaps": SWAPS,
            "swap_root": self.fresh_swaps, **self.selection_routes(), **changes})

    def collect(self, **changes):
        return swap_reporting.collect(**{
            "benchmark": "fixture", "config": RECIPE, "original_encoders": ORIGINAL,
            "original_folders": (self.fresh_original,), "swaps": SWAPS,
            "swap_root": self.fresh_swaps, **self.selection_routes(), **changes})

    def test_historical_choices_do_not_satisfy_any_fresh_fit_at_any_seed(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                plan = self.plan(seed)
                self.assertTrue(plan.ready, plan.issues)
                self.assertEqual(len(plan.runs), 42)
                self.assertEqual({run.status for run in plan.runs}, {"new"})
                self.assertTrue(all(run.source_path is None for run in plan.runs))
                self.assertEqual(plan.selection_lambda, 0.5)
                self.assertEqual(plan.selected_singles["mrnafm"], "unimodal:dna")
                self.assertEqual(len(plan.selection_sources), 18)
        self.assertFalse(self.fresh_original.exists())
        self.assertFalse(self.fresh_swaps.exists())
        bundle = self.collect()
        self.assertEqual(bundle["models"], {})
        self.assertEqual(bundle["sources"], {})
        self.assertEqual(bundle["best_singles"]["mrnafm"], "unimodal:dna")
        self.assertEqual(len(bundle["selection_sources"]), 18)

    def test_new_seed_zero_scores_never_reselect_the_baseline_or_mil(self):
        for variant in swap_plan.ORIGINAL_VARIANTS:
            self.write(self.fresh_original, variant, ORIGINAL, modern=True,
                       penalty=0.5 if variant == "mil" else 0.0,
                       score=0.1 if variant == "unimodal:dna" else 0.99)
        for modality, encoder, grid in SWAPS:
            for variant in swap_results.swap_variants(modality, grid=grid):
                self.write(self.fresh_swaps / encoder, variant,
                           {**ORIGINAL, modality: encoder}, modern=True,
                           penalty=0.5 if variant == "mil" else 0.0, score=1.0)
        plan = self.plan(identity_factory=self.identity)
        self.assertTrue(plan.ready, [run.reason for run in plan.runs])
        self.assertEqual({run.status for run in plan.runs}, {"reuse"})
        self.assertTrue(all(not run.legacy_identity for run in plan.runs))
        self.assertEqual(plan.selected_singles["mrnafm"], "unimodal:dna")
        bundle = self.collect(seeds=(0,))
        self.assertEqual(bundle["best_singles"]["mrnafm"], "unimodal:dna")
        self.assertEqual(bundle["mil_lambda"], 0.5)
        self.assertEqual(len(bundle["sources"]), 42)
        self.assertTrue(all(Path(p).is_relative_to(self.root / "fresh") for p in bundle["sources"]))
        # No fresh alternative MIL penalties were needed for those 42 completed fits.
        for contrast in bundle["comparisons"]:
            if contrast["encoder"] == "mrnafm" and contrast["kind"] == "complementarity":
                self.assertEqual(contrast["second"], "mrnafm/unimodal:dna")

    def test_missing_historical_evidence_never_falls_back_to_fresh_scores(self):
        missing = swap_plan.result_path(self.historical_original, "fixture", "mil",
                                        {**RECIPE, "entropy_lambda": 1.0})
        missing.unlink()
        self.write(self.fresh_original, "mil", ORIGINAL, penalty=1.0, modern=True)
        plan = self.plan()
        self.assertFalse(plan.ready)
        self.assertEqual(plan.runs, [])
        self.assertIn("historical seed-0 positive MIL grid", " ".join(plan.issues))
        with self.assertRaisesRegex(ValueError, "historical seed-0 positive MIL grid"):
            self.collect()

    def test_missing_historical_single_stops_selection_for_all_training_seeds(self):
        missing = swap_plan.result_path(self.historical_swaps / "mrnafm", "fixture",
                                        "unimodal:rna", RECIPE)
        missing.unlink()
        plan = self.plan(seed=2)
        self.assertFalse(plan.ready)
        self.assertIn("mrnafm", " ".join(plan.issues))
        with self.assertRaisesRegex(ValueError, "mrnafm"):
            self.collect()

    def test_external_mode_refuses_legacy_fresh_records(self):
        self.write(self.fresh_original, "concat", ORIGINAL)
        plan = self.plan()
        concat = next(run for run in plan.runs if run.role == "original" and run.variant == "concat")
        self.assertEqual(concat.status, "blocked")
        self.assertIn("full versioned identity", concat.reason)
        with self.assertRaisesRegex(ValueError, "Fresh comparison result lacks full identity"):
            self.collect()

    def test_checkpoint_needs_full_requested_identity_in_external_mode(self):
        path = swap_plan.result_path(self.fresh_original, "fixture", "concat", RECIPE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".checkpoint").write_bytes(b"synthetic placeholder; do not deserialize")
        validator = unittest.mock.Mock(return_value=None)
        plan = self.plan(checkpoint_validator=validator)
        concat = next(run for run in plan.runs if run.role == "original" and run.variant == "concat")
        self.assertEqual(concat.status, "checkpoint_requires_review")
        validator.assert_not_called()
        plan = self.plan(checkpoint_validator=validator, identity_factory=self.identity)
        concat = next(run for run in plan.runs if run.role == "original" and run.variant == "concat")
        self.assertEqual(concat.status, "resume")
        self.assertEqual(validator.call_args.args[1]["schema"], run_identity.SCHEMA)

    def test_partial_or_overlapping_selection_routes_are_rejected(self):
        arguments = dict(original_folders=(self.fresh_original,), swap_root=self.fresh_swaps)
        with self.assertRaisesRegex(ValueError, "both historical"):
            swap_results.separate_selection_roots(**arguments,
                selection_original_folders=(self.historical_original,), selection_swap_root=None)
        for overlap in (self.fresh_original, self.fresh_original / "nested", self.root / "fresh"):
            with self.subTest(overlap=overlap), self.assertRaisesRegex(ValueError, "must be separate"):
                swap_results.separate_selection_roots(**arguments,
                    selection_original_folders=(overlap,), selection_swap_root=self.historical_swaps)

    def test_default_mode_still_reuses_historical_legacy_records(self):
        plan = self.plan(original_folders=(self.historical_original,),
                         original_destination=self.historical_original,
                         swap_root=self.historical_swaps,
                         selection_original_folders=None, selection_swap_root=None)
        self.assertTrue(plan.ready, plan.issues)
        self.assertEqual(sum(run.status == "reuse" for run in plan.runs), 16)
        self.assertEqual(plan.selected_singles, {})
        self.assertEqual(plan.selection_sources, {})

    def test_report_identity_tracks_historical_selection_hashes_separately(self):
        bundle = self.collect()
        # Stub only numerical reporting and producer collection, not the selection readers.
        with patch.object(swap_reporting, "calculate", return_value={"synthetic": True}) as calculate, \
             patch.object(run_identity, "producer_record", return_value={"synthetic": True}):
            saved, _, made = swap_reporting.report(bundle, truth=[0, 1], groups=["a", "b"],
                task="classification", context={}, output_dir=self.root / "reports", draws=1)
        self.assertTrue(made)
        calculate.assert_called_once()
        self.assertEqual(saved["identity"]["sources"], {})
        self.assertEqual(saved["identity"]["selection_sources"], bundle["selection_sources"])
        self.assertEqual(saved["identity"]["selection_policy"], "fixed-historical-seed0-validation")
        for path, digest in bundle["selection_sources"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)

    def test_changed_historical_source_stops_report_before_calculation(self):
        bundle = self.collect()
        path = Path(next(iter(bundle["selection_sources"])))
        path.write_bytes(path.read_bytes() + b"\n")
        with patch.object(swap_reporting, "calculate", side_effect=AssertionError("must not calculate")):
            with self.assertRaisesRegex(ValueError, "Saved result changed during reporting"):
                swap_reporting.report(bundle, truth=[0, 1], groups=["a", "b"],
                    task="classification", context={}, output_dir=self.root / "reports", draws=1)
        self.assertFalse((self.root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
