"""Synthetic JSON reader checks; no notebook cells, models, or research results."""

import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import swap_results


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5,
              entropy_lambda=0.0, channels=100, tconv="equation", seed=0)
ENCODERS = dict(dna="dna_fixture", rna="rna_fixture", prot="protein_fixture")


class SwapResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def save(self, name, variant="concat", config=None, **fields):
        record = dict(benchmark="fixture", variant=variant,
                      config={**RECIPE, **(config or {})}, epochs=4, best_epoch=3, seconds=1.5,
                      val_score=0.5, test_score=0.4,
                      test_truth=[1.0, 2.0], test_prediction=[1.1, 1.9])
        record.update(fields)
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def read(self, **overrides):
        return swap_results.read_results(
            [self.folder], **{**dict(benchmark="fixture", config=RECIPE,
                                    encoders=ENCODERS), **overrides})

    def test_seed_filter_not_filename_order(self):
        first = self.save("a.json")
        self.save("z.json", config={"seed": 2}, test_score=0.99)
        result = self.read()
        self.assertEqual([r["path"] for r in result.require_unique()], [first])
        self.assertIn("seed", result.issues[0]["reason"])

    def test_full_recipe_and_benchmark_are_checked(self):
        good = self.save("good.json")
        for key, value in [("lr", 1e-4), ("batch_size", 16), ("max_epochs", 50),
                           ("channels", 1280), ("tconv", "appendix")]:
            self.save(key + ".json", config={key: value})
        self.save("benchmark.json", benchmark="another")
        self.assertEqual([r["path"] for r in self.read().require_unique()], [good])

    def test_missing_or_extra_configuration_never_fills_defaults(self):
        path = self.save("missing.json")
        data = json.loads(path.read_text())
        del data["config"]["lr"]
        path.write_text(json.dumps(data))
        self.save("extra.json", config={"new_method": True})
        self.assertEqual(self.read().records, [])
        self.assertEqual(len(self.read().issues), 2)

    def test_native_preference_and_unique_legacy_fallback(self):
        self.save("dna-old.json", "unimodal:dna", {"tconv": "appendix"})
        chosen = self.save("dna-new.json", "unimodal:dna")
        fallback = self.save("rna-old.json", "unimodal:rna", {"tconv": "appendix"})
        records = self.read(native_fallback=True).require_unique()
        self.assertEqual({r["path"] for r in records}, {chosen, fallback})
        self.assertTrue(all(r["legacy"] for r in records))

    def test_grid_and_fusion_do_not_get_native_exception(self):
        self.save("grid.json", "unimodal-grid:rna", {"tconv": "appendix"})
        self.save("fusion.json", config={"tconv": "appendix"})
        self.assertEqual(self.read(native_fallback=True).records, [])

    def test_duplicate_identical_files_are_not_arbitrarily_selected(self):
        self.save("one.json")
        self.save("two.json")
        result = self.read()
        self.assertEqual(result.records, [])
        self.assertIn("concat", result.coverage(["concat"])["missing"])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            result.require_unique()

    def test_duplicate_fallback_is_also_ambiguous(self):
        self.save("one.json", "unimodal:rna", {"tconv": "appendix"})
        self.save("two.json", "unimodal:rna", {"tconv": "appendix"})
        with self.assertRaises(ValueError):
            self.read(native_fallback=True).require_unique()

    def test_validation_selects_original_lambda_then_swap_matches_it(self):
        self.save("zero.json", "mil", {"entropy_lambda": 0}, val_score=0.99)
        self.save("small.json", "mil", {"entropy_lambda": 0.01}, val_score=0.7)
        self.save("large.json", "mil", {"entropy_lambda": 0.5}, val_score=0.6, test_score=0.95)
        self.assertEqual(self.read().positive_lambda(), 0.01)
        selected = self.read(mil_lambda=0.01).require_unique()
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["data"]["config"]["entropy_lambda"], 0.01)

    def test_missing_selected_penalty_not_counted_as_available(self):
        self.save("other.json", "mil", {"entropy_lambda": 0.5})
        self.assertEqual(self.read().coverage(["mil"], mil_lambda=0.01)["missing"], "mil")

    def test_recorded_identity_checked_without_inventing_legacy_identity(self):
        self.save("wrong.json", identity={"inputs": {"encoders": {**ENCODERS, "rna": "wrong"}}})
        self.save("contradict.json", identity={"config": {**RECIPE, "seed": 2}})
        path = self.save("right.json", identity={"inputs": {"encoders": ENCODERS}})
        records = self.read().require_unique()
        self.assertEqual([r["path"] for r in records], [path])
        self.assertFalse(records[0]["legacy"])

    def test_direct_children_only_and_input_bytes_unchanged(self):
        path = self.save("one.json")
        self.save("archive/duplicate.json")
        before = path.read_bytes()
        self.assertEqual(len(self.read().require_unique()), 1)
        self.assertEqual(path.read_bytes(), before)

    def test_malformed_incomplete_and_nonfinite_records_are_reported(self):
        (self.folder / "invalid.json").write_text("{")
        self.save("null.json", val_score=None)
        self.save("nan.json", test_score=float("nan"))
        self.save("incomplete.json", epochs=0)
        self.assertEqual(self.read().records, [])
        self.assertEqual(len(self.read().issues), 4)

    def test_score_only_baseline_remains_available_but_not_paired(self):
        self.save("old.json", "unimodal:rna", test_truth=None, test_prediction=None)
        report = self.read()
        coverage = report.coverage(["unimodal:rna", "concat"])
        self.assertEqual(coverage["available"], "unimodal:rna")
        self.assertEqual(coverage["missing"], "concat")
        self.assertEqual(coverage["without paired arrays"], "unimodal:rna")

    def test_missing_required_result_metadata_is_excluded(self):
        for key in ("best_epoch", "seconds"):
            path = self.save(key + ".json")
            payload = json.loads(path.read_text())
            del payload[key]
            path.write_text(json.dumps(payload))
        self.assertEqual(self.read().records, [])
        self.assertEqual(len(self.read().issues), 2)

    def test_float32_targets_use_loader_precision_without_changing_bootstrap_truth(self):
        original = [0.1, 0.2]
        record = dict(test_truth=[0.10000000149011612, 0.20000000298023224],
                      test_prediction=[0.11, 0.19])
        loader_truth = [struct.unpack("f", struct.pack("f", value))[0] for value in original]
        self.assertIsNone(swap_results.prediction_issue(record, loader_truth))
        self.assertIsNotNone(swap_results.prediction_issue(record, loader_truth[::-1]))
        self.assertEqual(original, [0.1, 0.2])

    def test_seed_coverage_does_not_borrow_missing_references(self):
        self.save("original-seed0.json", "unimodal:rna")
        self.save("fusion-seed1.json", "concat", {"seed": 1})
        report = self.read(config={**RECIPE, "seed": 1})
        coverage = report.coverage(["unimodal:rna", "unimodal-grid:rna", "concat"])
        self.assertEqual(coverage["available"], "concat")
        self.assertEqual(coverage["missing"], "unimodal:rna, unimodal-grid:rna")

    def test_target_order_and_prediction_shape_check(self):
        record = dict(test_truth=[1, 2], test_prediction=[1.1, 1.9])
        self.assertIsNone(swap_results.prediction_issue(record, [1, 2]))
        self.assertIn("order", swap_results.prediction_issue(record, [2, 1]))
        record["test_prediction"] = [[0.2, 0.8], [0.9]]
        self.assertIn("widths", swap_results.prediction_issue(record))
        record["test_prediction"] = [[0.2, 0.8], [0.9, 0.1]]
        self.assertIsNone(swap_results.prediction_issue(record))

    def test_grid_expectation_is_explicit(self):
        self.assertEqual(swap_results.swap_variants("rna", grid=False),
                         ("unimodal:rna", "concat", "mil", "cross"))
        self.assertIn("unimodal-grid:dna", swap_results.swap_variants("dna", grid=True))
        with self.assertRaises(ValueError):
            swap_results.swap_variants("unknown", grid=False)

    def test_declared_mil_grid_excludes_an_unrequested_penalty(self):
        self.save("declared.json", "mil", {"entropy_lambda": 0.01})
        self.save("other.json", "mil", {"entropy_lambda": 0.25}, val_score=0.99)
        report = self.read(mil_lambdas=(0.0, 0.01, 0.5, 1.0))
        self.assertEqual(report.positive_lambda(), 0.01)
        self.assertIn("candidate grid", report.issues[0]["reason"])

    def test_seed_zero_validation_choices_require_the_declared_candidates(self):
        self.save("dna.json", "unimodal:dna", val_score=0.6, test_score=0.9)
        self.save("rna.json", "unimodal:rna", val_score=0.8, test_score=0.2)
        self.save("mil-low.json", "mil", {"entropy_lambda": 0.01}, val_score=0.7)
        self.save("mil-high.json", "mil", {"entropy_lambda": 0.5}, val_score=0.5, test_score=0.99)
        selected = swap_results.validation_choices(
            self.read().require_unique(), singles=("unimodal:dna", "unimodal:rna"),
            positive_lambdas=(0.01, 0.5),
        )
        self.assertEqual(selected, ("unimodal:rna", 0.01))
        incomplete = swap_results.validation_choices(
            self.read().require_unique(), singles=("unimodal:dna", "unimodal:rna", "unimodal:prot"),
            positive_lambdas=(0.01, 0.5, 1.0),
        )
        self.assertEqual(incomplete, (None, None))

    def test_validation_choices_do_not_mix_training_seeds(self):
        self.save("seed1.json", "unimodal:rna", {"seed": 1})
        with self.assertRaisesRegex(ValueError, "seed-0"):
            swap_results.validation_choices(
                self.read(config={**RECIPE, "seed": 1}).require_unique(),
                singles=("unimodal:rna",), positive_lambdas=(0.01,),
            )

    def fixture_entries(self, variants, seeds):
        """Build already recipe-checked entries without any research data or model."""
        for variant in variants:
            for seed in seeds:
                self.save(f"{variant}-{seed}.json", variant, {"seed": seed})
        return [entry for seed in seeds for entry in self.read(
            config={**RECIPE, "seed": seed},
        ).require_unique()]

    def paired(self, entries, variants=("concat", "cross"), seeds=(0, 1, 2), **overrides):
        return swap_results.paired_records(
            entries, **{**dict(variants=variants, seeds=seeds, mil_lambda=0.01,
                              truth=[1.0, 2.0], task="regression"), **overrides},
        )

    def test_pairing_uses_matching_seed_ids_in_declared_order(self):
        entries = self.fixture_entries(("concat", "cross"), (0, 1, 2))
        paired = self.paired(entries[::-1], seeds=(2, 0, 1))
        self.assertEqual(paired.seeds, (2, 0, 1))
        for records in paired.records.values():
            self.assertEqual([r["data"]["config"]["seed"] for r in records], [2, 0, 1])

    def test_equal_file_counts_with_disjoint_seeds_do_not_pair(self):
        entries = self.fixture_entries(("concat", "cross"), (0, 1))
        entries = [r for r in entries if (r["data"]["variant"], r["data"]["config"]["seed"])
                   in {( "concat", 0), ("cross", 1)}]
        paired = self.paired(entries, seeds=(0, 1))
        self.assertEqual(paired.seeds, ())
        self.assertTrue(all(not records for records in paired.records.values()))

    def test_one_bad_target_order_removes_that_seed_from_every_model(self):
        entries = self.fixture_entries(("concat", "cross"), (0, 1))
        for entry in entries:
            if entry["data"]["variant"] == "cross" and entry["data"]["config"]["seed"] == 1:
                entry["data"]["test_truth"] = [2.0, 1.0]
        paired = self.paired(entries, seeds=(0, 1))
        self.assertEqual(paired.seeds, (0,))
        self.assertEqual(len(paired.issues), 1)
        self.assertIn("order", paired.issues[0]["reason"])

    def test_pairing_refuses_duplicates_even_if_one_has_bad_arrays(self):
        entries = self.fixture_entries(("concat", "cross"), (0,))
        entries[0]["data"]["test_prediction"] = None
        with self.assertRaisesRegex(ValueError, "Ambiguous paired"):
            self.paired([*entries, entries[0]], seeds=(0,))

    def test_pairing_classification_uses_class_ids_not_logits(self):
        entries = self.fixture_entries(("concat", "cross"), (0,))
        for entry in entries:
            entry["data"].update(test_truth=[0, 2], test_prediction=[0, 1])
        self.assertEqual(self.paired(entries, seeds=(0,), task="classification", truth=[0, 2]).seeds, (0,))
        entries[0]["data"]["test_prediction"] = [[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]]
        self.assertEqual(self.paired(entries, seeds=(0,), task="classification", truth=[0, 2]).seeds, ())
        entries[0]["data"]["test_prediction"] = [0.2, 1.1]
        self.assertIn("class IDs", self.paired(
            entries, seeds=(0,), task="classification", truth=[0, 2],
        ).issues[0]["reason"])

    def test_pairing_rejects_regression_panels_and_mismatched_expression_widths(self):
        entries = self.fixture_entries(("concat", "cross"), (0,))
        entries[0]["data"]["test_prediction"] = [[1.1], [1.9]]
        self.assertEqual(self.paired(entries, seeds=(0,)).seeds, ())
        panel = dict(test_truth=[[1, 2], [3, 4]], test_prediction=[[1.1], [3.1]])
        self.assertIn("widths", swap_results.prediction_issue(panel, task="expression"))

    def test_classification_upper_bound_checks_targets_and_predictions(self):
        record = dict(test_truth=[0, 2], test_prediction=[1, 2])
        self.assertIsNone(swap_results.prediction_issue(record, task="classification", n_classes=3))
        record["test_prediction"] = [1, 3]
        self.assertIn("class range", swap_results.prediction_issue(
            record, task="classification", n_classes=3,
        ))
        # An omitted bound retains the legacy generic classification contract.
        self.assertIsNone(swap_results.prediction_issue(record, task="classification"))
        record.update(test_truth=[0, 3], test_prediction=[1, 2])
        self.assertIn("class range", swap_results.prediction_issue(
            record, task="classification", n_classes=3,
        ))

    def test_invalid_class_bounds_are_rejected_even_without_records(self):
        record = dict(test_truth=[0, 2], test_prediction=[1, 2])
        for invalid in (0, -1, 3.0, True, "3"):
            with self.subTest(bound=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    swap_results.prediction_issue(record, task="classification", n_classes=invalid)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.paired([], task="classification", n_classes=invalid)

    def test_out_of_range_class_removes_seed_from_paired_cohort(self):
        entries = self.fixture_entries(("concat", "cross"), (0,))
        for entry in entries:
            entry["data"].update(test_truth=[0, 2], test_prediction=[0, 1])
        entries[0]["data"]["test_prediction"] = [0, 3]
        selected = self.paired(
            entries, seeds=(0,), task="classification", truth=[0, 2], n_classes=3,
        )
        self.assertEqual(selected.seeds, ())
        self.assertIn("class range", selected.issues[0]["reason"])


if __name__ == "__main__":
    unittest.main()
