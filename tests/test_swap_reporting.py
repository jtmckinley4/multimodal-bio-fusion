"""Saved-swap analysis on tiny synthetic JSONs; no research files or models used."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import swap_reporting


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5,
              entropy_lambda=0.0, channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="dna_fixture", rna="rna_fixture", prot="protein_fixture")
TRUTH = [0, 1, 0, 1, 0, 1]


class SwapReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original"
        self.swaps = self.root / "swaps"
        self.frame = pd.DataFrame(dict(sequence=[f"sequence-{i}" for i in range(8)],
                                       length=[10] * 8, Value=[0, 1, *TRUTH],
                                       Split=["train", "val", *(["test"] * 6)]))
        self.context = swap_reporting.benchmark_context(
            self.frame, task="classification", label_column="Value", split_column="Split")
        self.groups = self.frame.sequence.to_numpy()[2:]
        for seed in (0, 1, 2):
            for name in ("unimodal:dna", "unimodal:rna", "unimodal:prot",
                         "unimodal-grid:rna", "concat", "cross"):
                self.write(self.original, name, seed, 0.0)
            for penalty in (0.01, 0.5, 1.0):
                self.write(self.original, "mil", seed, penalty,
                           val_score=0.8 if penalty == 0.01 else 0.4)
            for name in ("unimodal:rna", "concat", "mil", "cross"):
                prediction = TRUTH if name == "unimodal:rna" else [0, 1, 0, 1, 0, 0]
                self.write(self.swaps / "replacement", name, seed,
                           0.01 if name == "mil" else 0.0,
                           test_prediction=prediction, val_score=0.8 if seed == 0 else 0.1)

    def write(self, folder, variant, seed, penalty, **fields):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{variant.replace(':', '-')}__{penalty}__{seed}.json"
        result = dict(benchmark="fixture", variant=variant,
                      config={**RECIPE, "seed": seed, "entropy_lambda": penalty},
                      epochs=3, best_epoch=2, seconds=1.0,
                      val_score=0.6, test_score=0.5,
                      test_truth=TRUTH, test_prediction=[0] * 6)
        result.update(fields)
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def collect(self, **changes):
        return swap_reporting.collect(**{
            "benchmark": "fixture", "config": RECIPE,
            "original_folders": [self.original], "original_encoders": ORIGINAL,
            "swaps": [("rna", "replacement", False)], "swap_root": self.swaps,
            **changes})

    def calculate(self, bundle=None, **changes):
        return swap_reporting.calculate(self.collect() if bundle is None else bundle, **{
            "truth": TRUTH, "groups": self.groups, "task": "classification",
            "context": self.context, "draws": 30, "n_classes": 2, **changes})

    def test_fixed_selection_does_not_follow_test_scores_or_repeat_validation(self):
        self.write(self.original, "unimodal:dna", 0, 0.0, test_score=1.0)
        self.write(self.original, "unimodal:dna", 1, 0.0, val_score=0.99)
        bundle = self.collect()
        self.assertEqual(bundle["best_singles"]["replacement"], "unimodal:rna")
        self.assertEqual(bundle["mil_lambda"], 0.01)
        contrasts = bundle["comparisons"]
        self.assertEqual(len(contrasts), 9)
        self.assertEqual({c["kind"] for c in contrasts},
                         {"representation", "replacement", "complementarity", "method"})

    def test_known_difference_and_all_three_seed_mean(self):
        result = self.calculate()
        rows = [r for r in result["seed_summary"] if r["kind"] == "representation"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["available_seeds"], [0, 1, 2])
        self.assertEqual(rows[0]["difference"], 50.0)
        self.assertEqual(rows[0]["seed_sd"], 0.0)
        self.assertEqual(rows[0]["requested_draws"], 30)
        method = [r for r in result["seed_summary"] if r["kind"] == "method"]
        self.assertTrue(all(r["difference"] == r["ci_low"] == r["ci_high"] == 0 for r in method))

    def test_scores_are_averaged_not_prediction_ensemble(self):
        # Across seeds, opposite class predictions can average into a different ensemble.
        for seed, prediction in [(0, TRUTH), (1, [1-x for x in TRUTH]), (2, TRUTH)]:
            self.write(self.swaps / "replacement", "unimodal:rna", seed, 0.0,
                       test_prediction=prediction, val_score=0.8)
        result = self.calculate()
        row = next(r for r in result["seed_summary"] if r["kind"] == "representation")
        self.assertAlmostEqual(row["first_score"], 200 / 3)
        self.assertAlmostEqual(row["difference"], 100 / 6)
        self.assertAlmostEqual(row["seed_sd"], np.std([50, -50, 50], ddof=1))

    def test_missing_seed_never_reports_subset_mean(self):
        for path in (self.swaps / "replacement").glob('*__2.json'):
            path.unlink()
        result = self.calculate()
        self.assertEqual({r["seed"] for r in result["per_seed"]}, {0, 1})
        for row in result["seed_summary"]:
            self.assertEqual(row["status"], "incomplete")
            self.assertIsNone(row["difference"])
            self.assertIsNone(row["ci_low"])
            self.assertEqual(row["missing_seeds"], [2])

    def test_missing_native_arrays_do_not_block_unrelated_comparisons(self):
        self.write(self.original, "unimodal:rna", 0, 0.0,
                   test_truth=None, test_prediction=None)
        result = self.calculate()
        self.assertEqual(next(r for r in result["seed_summary"]
                              if r["kind"] == "representation")["status"], "incomplete")
        self.assertTrue(all(r["status"] == "complete" for r in result["seed_summary"]
                            if r["kind"] in {"complementarity", "method"}))

    def test_duplicate_identity_is_not_silently_selected(self):
        path = next(self.original.glob('concat*__0.json'))
        (self.original / "duplicate.json").write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.collect()

    def test_missing_penalty_grid_stops_selection(self):
        (self.original / "mil__0.5__0.json").unlink()
        with self.assertRaisesRegex(ValueError, "Complete original"):
            self.collect()

    def test_reordered_targets_are_reported_not_scored(self):
        self.write(self.swaps / "replacement", "cross", 0, 0.0,
                   test_truth=list(reversed(TRUTH)))
        result = self.calculate()
        self.assertTrue(any("targets differ" in r["reason"] for r in result["issues"]))
        self.assertTrue(all(r["seed"] != 0 for r in result["per_seed"]
                            if r["first"] == "replacement/cross"))

    def test_recorded_row_mismatch_blocks_even_with_identical_targets(self):
        bundle = self.collect()
        entry = bundle["models"]["replacement/cross"][0]
        entry["data"]["identity"] = {"inputs": {"rows": {"sha256": "different"}}}
        result = self.calculate(bundle)
        self.assertTrue(any("recorded rows differ" in x["reason"] for x in result["issues"]))

    def test_repeated_groups_are_resampled_together(self):
        result = self.calculate(groups=["a", "a", "b", "b", "c", "c"])
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["rows"], 6)

    def test_one_group_cannot_claim_a_resampling_interval(self):
        result = self.calculate(groups=["same"] * 6)
        row = result["seed_summary"][0]
        self.assertIsNotNone(row["difference"])
        self.assertIsNone(row["ci_low"])
        self.assertIn("fewer than two", row["interval_issue"])

    def test_regression_known_rank_difference_and_undefined_draws(self):
        bundle = self.collect(seeds=(0,))
        for name, entries in bundle["models"].items():
            for entry in entries.values():
                entry["data"]["test_truth"] = list(range(6))
                entry["data"]["test_prediction"] = list(range(6)) if name.startswith("replacement/") else list(reversed(range(6)))
        result = self.calculate(bundle, truth=list(range(6)), task="regression", n_classes=None)
        row = next(r for r in result["seed_summary"] if r["kind"] == "representation")
        self.assertAlmostEqual(row["difference"], 2.0)
        self.assertEqual(row["seed_sd"], None)

    def test_invalid_bootstrap_seed_is_rejected(self):
        for seed in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "bootstrap_seed"):
                self.calculate(bootstrap_seed=seed)

    def test_regression_scores_use_loader_precision_when_csv_values_round_to_ties(self):
        bundle = self.collect(seeds=(0,))
        raw_truth = np.array([1.0, 1.0 + 1e-8, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        cast_truth = raw_truth.astype(np.float32)
        first, second = list(range(6)), list(reversed(range(6)))
        for name, entries in bundle["models"].items():
            for entry in entries.values():
                entry["data"]["test_truth"] = cast_truth.tolist()
                entry["data"]["test_prediction"] = first if name.startswith("replacement/") else second
        result = self.calculate(bundle, truth=raw_truth, task="regression", n_classes=None)
        row = next(r for r in result["seed_summary"] if r["kind"] == "representation")
        expected = spearmanr(cast_truth, first).statistic - spearmanr(cast_truth, second).statistic
        wrong = spearmanr(raw_truth, first).statistic - spearmanr(raw_truth, second).statistic
        self.assertAlmostEqual(row["difference"], expected)
        self.assertNotAlmostEqual(row["difference"], wrong)

    def test_fractional_class_labels_are_rejected_before_casting(self):
        with self.assertRaisesRegex(ValueError, "integer class IDs"):
            self.calculate(truth=[0, 0.5, 0, 1, 0, 1])

    def test_immutable_analysis_reuse_skips_calculation(self):
        bundle = self.collect()
        args = dict(truth=TRUTH, groups=self.groups, task="classification",
                    context=self.context, output_dir=self.root / "analyses", draws=20,
                    n_classes=2)
        saved, path, computed = swap_reporting.report(bundle, **args)
        before = path.read_bytes()
        self.assertTrue(computed)
        with patch.object(swap_reporting, "calculate", side_effect=AssertionError("recalculated")):
            reused, same, computed = swap_reporting.report(bundle, **args)
        self.assertFalse(computed)
        self.assertEqual(path, same)
        self.assertEqual(saved, reused)
        self.assertEqual(before, path.read_bytes())

    def test_modified_saved_report_is_preserved_and_rejected(self):
        bundle = self.collect()
        args = dict(truth=TRUTH, groups=self.groups, task="classification",
                    context=self.context, output_dir=self.root / "analyses", draws=20)
        saved, path, _ = swap_reporting.report(bundle, **args)
        saved["analysis"]["rows"] = 10
        path.write_text(json.dumps(saved), encoding="utf8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "preserved"):
            swap_reporting.report(bundle, **args)
        self.assertEqual(before, path.read_bytes())

    def test_changed_input_after_collection_is_rejected_before_calculation(self):
        bundle = self.collect()
        path = Path(next(iter(bundle["sources"])))
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed during reporting"):
            swap_reporting.report(bundle, truth=TRUTH, groups=self.groups, task="classification",
                                  context=self.context, output_dir=self.root / "analyses")
        self.assertFalse((self.root / "analyses").exists())

    def test_notebook_switch_defaults_off_and_stops_training_combination_in_setup(self):
        notebook = json.loads((ROOT / "Code/stage4/tier2_encoder_swaps.ipynb").read_text(encoding="utf8"))
        cells = {c.get("id"): "".join(c["source"]) for c in notebook["cells"]}
        source = cells["swaps-settings"]
        guard = source[source.index("RUN_SAVED_COMPARISONS ="):source.index("# Cache chunking")]
        state = {"READ_SAVED_RESULTS": True, "RESULTS_ROOT": self.root}
        exec(guard, state)
        self.assertFalse(state["RUN_SAVED_COMPARISONS"])
        with redirect_stdout(io.StringIO()):
            exec(cells["swaps-paired-report-run"], state)
        self.assertEqual(state["SAVED_COMPARISON_REPORTS"], {})
        with self.assertRaisesRegex(ValueError, "training and extraction"):
            exec(guard.replace("RUN_SAVED_COMPARISONS = False", "RUN_SAVED_COMPARISONS = True"),
                 {"READ_SAVED_RESULTS": False, "RESULTS_ROOT": self.root})

    def test_enabled_notebook_report_uses_synthetic_saved_results_only(self):
        notebook = json.loads((ROOT / "Code/stage4/tier2_encoder_swaps.ipynb").read_text(encoding="utf8"))
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c.get("id") == "swaps-paired-report-run")
        def forbidden(*args, **kwargs):
            raise AssertionError("Reporting attempted training or extraction")
        state = dict(
            RUN_SAVED_COMPARISONS=True, READ_SAVED_RESULTS=True, BENCHMARKS=["fixture"],
            datasets=SimpleNamespace(blf_benchmark_rows=lambda key: self.frame,
                                     DATASETS={"fixture": SimpleNamespace(
                                         task="classification", label_column="Value", split_column="Split")}),
            np=np, pd=pd, display=lambda obj: None,
            result_recipe=lambda seed: {**RECIPE, "seed": seed},
            ORIGINAL_RESULTS={"fixture": [self.original]}, ORIGINAL=ORIGINAL,
            swaps_for=lambda key: [("rna", "replacement")],
            tokens=SimpleNamespace(grid_resolution=lambda e: "codon", extract=forbidden),
            train=SimpleNamespace(run=forbidden, fit=forbidden),
            RESULTS_ROOT=self.root, COMPARISON_SEEDS=(0, 1, 2),
            COMPARISON_DIR=self.root / "reports", COMPARISON_DRAWS=10,
            COMPARISON_BOOTSTRAP_SEED=0, CODE_DIR=self.root,
        )
        # The notebook derives its benchmark directory; use a fixture mapping only.
        original_collect = swap_reporting.collect
        def collect_fixture(**kwargs):
            kwargs["swap_root"] = self.swaps
            return original_collect(**kwargs)
        with redirect_stdout(io.StringIO()), patch.object(swap_reporting, "collect", collect_fixture):
            exec(source, state)
        self.assertEqual(len(list((self.root / "reports").glob("*.json"))), 1)
        self.assertIn("fixture", state["SAVED_COMPARISON_REPORTS"])


if __name__ == "__main__":
    unittest.main()
