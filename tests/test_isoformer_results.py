"""Check the GTEx result reader using invented JSONs, without research data or ML."""

import builtins
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code"))

from shared_code.fusion import isoformer_results  # noqa: E402


# Fixed fixture recipe, independent of TrainConfig and its model dependencies.
CONFIG = {
    "lr": 3e-5, "weight_decay": 1e-5, "batch_size": 16, "max_epochs": 500,
    "early_stopping_patience": 20, "plateau_patience": 5, "entropy_lambda": 0.0,
    "channels": 100, "tconv": "appendix", "seed": 0,
}
MODALITIES = ("dna", "rna", "prot", "dna+prot", "dna+rna", "dna+rna+prot")


class IsoFormerResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".isoformer-read-", dir=REPO / "tests")
        root = Path(self.temporary.name).resolve()
        self.assertTrue(root.is_relative_to((REPO / "tests").resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.reference, self.repeats = root / "reference", root / "repeats"
        self.reference.mkdir()
        self.repeats.mkdir()

    def write(self, modality="dna", seed=0, *, folder=None, legacy=False,
              heads=8, budget=True, score=0.2, **overrides):
        folder = folder or self.reference
        tag = f"gtex_pilot__{modality}__heads{heads}"
        if budget:
            tag += "__budget300M"
        path = folder / f"{tag}__seed{seed}.json"
        payload = {
            "benchmark": "gtex_pilot", "variant": "IsoFormer" if legacy else modality,
            "config": dict(CONFIG, seed=seed), "epochs": 2, "test_score": score,
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def read(self, **kwargs):
        kwargs.setdefault("train_config", CONFIG)
        return isoformer_results.read_cohort(self.reference, self.repeats, **kwargs)

    def test_six_legacy_labels_survive_and_unbudgeted_siblings_are_excluded(self):
        paths = [self.write(v, legacy=True, score=(i + 1) / 10)
                 for i, v in enumerate(MODALITIES)]
        paths.append(self.write("rna", legacy=True, budget=False, score=0.99))
        before = {p: p.read_bytes() for p in paths}

        result = self.read()

        self.assertEqual([r["variant"] for r in result.records], list(MODALITIES))
        self.assertEqual([r["Spearman"] for r in result.records],
                         [(i + 1) / 10 for i in range(6)])
        self.assertEqual([r["seed"] for r in result.records], [0] * 6)
        self.assertEqual(len([i for i in result.issues if i["status"] == "excluded"]), 1)
        self.assertEqual(before, {p: p.read_bytes() for p in paths})
        self.assertFalse(result.duplicate_keys)

    def test_coverage_distinguishes_partial_complete_and_missing_modalities(self):
        self.write("dna", legacy=True)
        self.write("rna", legacy=True)
        self.write("dna", 1, folder=self.repeats)
        self.write("dna", 2, folder=self.repeats)

        coverage = {r["variant"]: r for r in self.read().coverage}

        self.assertEqual(coverage["dna"]["available seeds"], (0, 1, 2))
        self.assertEqual(coverage["dna"]["missing seeds"], ())
        self.assertEqual(coverage["dna"]["status"], "complete")
        self.assertEqual(coverage["rna"]["count"], 1)
        self.assertEqual(coverage["rna"]["missing seeds"], (1, 2))
        self.assertEqual(coverage["rna"]["status"], "partial")
        self.assertEqual(coverage["prot"]["count"], 0)
        self.assertEqual(coverage["prot"]["missing seeds"], (0, 1, 2))
        self.assertEqual(coverage["prot"]["status"], "missing")

    def test_duplicate_modality_seed_is_blocked_instead_of_keeping_first(self):
        self.write("dna", legacy=True, score=0.2)
        for repeated_score in (0.2, 0.8):
            with self.subTest(repeated_score=repeated_score):
                self.write("dna", folder=self.repeats, score=repeated_score)
                result = self.read()
                self.assertEqual(result.duplicate_keys, [("dna", 0)])
                self.assertEqual(result.records, [])
                self.assertEqual(result.coverage[0]["status"], "duplicate; blocked")
                self.assertEqual(len([i for i in result.issues
                                      if i["status"] == "duplicate; blocked"]), 2)

    def test_nested_archives_do_not_enter_the_selected_cohort(self):
        self.write("dna", legacy=True)
        archive = self.reference / "archive"
        archive.mkdir()
        self.write("dna", folder=archive, score=0.9)
        result = self.read()
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["Spearman"], 0.2)
        self.assertFalse(result.duplicate_keys)

    def test_legacy_alias_requires_exact_canonical_filename(self):
        path = self.write("dna", legacy=True)
        path.rename(path.with_name(path.name.replace("seed0", "seed00")))
        result = self.read()
        self.assertFalse(result.records)
        self.assertTrue(any("historical mapping" in i["reason"] for i in result.issues))

    def test_generic_alias_requires_known_reference_seed_zero_recipe(self):
        cases = (
            ({"seed": 1}, {}),
            ({"folder": self.repeats}, {}),
            ({"heads": 4}, {"n_heads": 4}),
            ({"config": dict(CONFIG, batch_size=8)},
             {"train_config": dict(CONFIG, batch_size=8)}),
            ({}, {"val_fraction": 0.1}),
        )
        for write_kwargs, read_kwargs in cases:
            with self.subTest(write=write_kwargs, read=read_kwargs):
                path = self.write(legacy=True, **write_kwargs)
                try:
                    result = self.read(**read_kwargs)
                    self.assertFalse(result.records)
                    self.assertTrue(any("historical mapping" in i["reason"]
                                        for i in result.issues))
                finally:
                    path.unlink()

    def test_filename_modality_and_complete_training_config_must_agree(self):
        cases = (
            {"variant": "prot"},
            {"config": dict(CONFIG, seed=2)},
            {"config": dict(CONFIG, batch_size=32)},
            {"config": dict(CONFIG, seed=1, unknown_setting=1)},
            {"config": dict(CONFIG, seed=True)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.write("dna", 1, folder=self.repeats, **overrides)
                result = self.read()
                self.assertFalse(result.records)
                self.assertTrue(any(i["status"] == "excluded" for i in result.issues))

    def test_malformed_or_incomplete_results_are_reported(self):
        for overrides in ({"epochs": 0}, {"epochs": True}, {"test_score": None},
                          {"test_score": float("nan")}, {"test_score": True},
                          {"test_score": 1.1}, {"benchmark": "other"}, {"config": {}}):
            with self.subTest(overrides=overrides):
                path = self.write(**overrides)
                self.assertFalse(self.read().records)
        path.write_text("{", encoding="utf-8")
        result = self.read()
        self.assertFalse(result.records)
        self.assertEqual(result.issues[0]["status"], "excluded")

    def test_missing_destination_and_unrecorded_identity_are_visible(self):
        self.write(legacy=True)
        self.repeats.rmdir()
        result = self.read()
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["identity"], "historical label mapped")
        self.assertTrue(any(i["status"] == "missing directory" for i in result.issues))
        limits = [i["reason"] for i in result.issues if i["status"] == "provenance limit"]
        self.assertEqual(len(limits), 1)
        self.assertIn("not established", limits[0])
        self.assertIn("Validation fraction", limits[0])

    def test_request_does_not_collapse_distinct_seeds_or_budget_values(self):
        for kwargs in ({"expected_seeds": (0, 0)}, {"expected_seeds": (False, 1)},
                       {"budget": 300_000_001}, {"variants": ("IsoFormer",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.read(**kwargs)

    def panel_fields(self):
        return {"test_truth": [[1.0, 2.0], [3.0, 4.0]],
                "test_prediction": [[1.1, 2.1], [2.9, 4.2]]}

    def modern_identity(self, modality="dna", seed=0):
        """A versioned fixture identity with explicit requested loader/model settings."""
        return {
            "schema": "fusion.run.v1", "benchmark": "gtex_pilot", "variant": modality,
            "config": dict(CONFIG, seed=seed), "inputs": {"encoders": {"dna": "fixture"}},
            "model": {"contract": "synthetic", "kwargs": {"n_heads": 8},
                      "modalities": modality.split("+"), "n_outputs": 2},
            "loader": {"kind": "isoformer", "n_heads": 8, "budget": 300_000_000,
                       "val_fraction": 0.05, "batch_size": 16, "seed": seed,
                       "recovery": {split: {} for split in ("train", "val", "test")}},
        }

    def read_panels(self, **kwargs):
        options = dict(train_config=CONFIG, expected_seeds=(0,),
                       require_predictions=True, expected_shape=(2, 2))
        options.update(kwargs)
        return isoformer_results.read_cohort(self.reference, None, **options)

    def test_reference_only_read_does_not_invent_a_repeats_directory(self):
        self.write("dna", legacy=True)
        result = isoformer_results.read_cohort(
            self.reference, train_config=CONFIG, expected_seeds=(0,))
        self.assertEqual(len(result.records), 1)
        self.assertFalse(any(i["status"] == "missing directory" for i in result.issues))
        self.assertEqual(result.coverage[0]["status"], "complete")
        self.assertNotIn("result", result.records[0])

    def test_optional_prediction_validation_keeps_default_score_rows_unchanged(self):
        self.write("dna", legacy=True)
        score_only = self.read(expected_seeds=(0,))
        self.assertEqual(len(score_only.records), 1)
        self.assertEqual(set(score_only.records[0]),
                         {"variant", "seed", "Spearman", "path", "identity"})
        required = self.read_panels()
        self.assertFalse(required.records)
        self.assertEqual(required.coverage[0]["status"], "missing")

    def test_validated_payload_is_available_without_recomputing_its_score(self):
        paths = [self.write(name, legacy=True, score=-0.125, **self.panel_fields())
                 for name in MODALITIES]
        before = {path: path.read_bytes() for path in paths}
        result = self.read_panels()
        self.assertEqual([row["variant"] for row in result.records], list(MODALITIES))
        self.assertTrue(all(row["Spearman"] == -0.125 for row in result.records))
        self.assertTrue(all(row["result"]["test_truth"] == self.panel_fields()["test_truth"]
                            for row in result.records))
        self.assertTrue(all(row["status"] == "complete" for row in result.coverage))
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertTrue(any(i["status"] == "provenance limit" for i in result.issues))

    def test_invalid_prediction_panels_are_excluded_and_leave_missing_coverage(self):
        invalid = (
            None, [], [1.0, 2.0], [[], []], [[1.0, 2.0], [3.0]],
            [[1.0, 2.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, float("nan")], [3.0, 4.0]],
            [[1.0, float("inf")], [3.0, 4.0]],
            [[True, 2.0], [3.0, 4.0]], [["1", 2.0], [3.0, 4.0]],
        )
        for field in ("test_truth", "test_prediction"):
            for values in invalid:
                with self.subTest(field=field, values=values):
                    fields = {**self.panel_fields(), field: values}
                    path = self.write("dna", **fields)
                    before = path.read_bytes()
                    result = self.read_panels()
                    self.assertFalse(result.records)
                    self.assertEqual(result.coverage[0]["missing seeds"], (0,))
                    self.assertTrue(any(i["status"] == "excluded" for i in result.issues))
                    self.assertEqual(path.read_bytes(), before)

    def test_expected_panel_shape_checks_rows_and_target_width(self):
        self.write("dna", **self.panel_fields())
        for shape in ((3, 2), (2, 3)):
            with self.subTest(shape=shape):
                result = self.read_panels(expected_shape=shape)
                self.assertFalse(result.records)
                self.assertTrue(any("expected" in i["reason"] for i in result.issues))
        self.assertEqual(len(self.read_panels(expected_shape=None).records), 1)

    def test_conflicting_target_row_or_column_order_blocks_entire_cohort(self):
        for changed in ([[3.0, 4.0], [1.0, 2.0]], [[2.0, 1.0], [4.0, 3.0]]):
            with self.subTest(targets=changed):
                self.write("dna", **self.panel_fields())
                self.write("rna", **{**self.panel_fields(), "test_truth": changed})
                for variants in (("dna", "rna"), ("rna", "dna")):
                    result = self.read_panels(variants=variants)
                    self.assertFalse(result.records)
                    self.assertTrue(all(row["status"] == "missing" for row in result.coverage))
                    excluded = [i for i in result.issues if i["status"] == "excluded"]
                    self.assertEqual(len(excluded), 2)
                    self.assertTrue(all("entire prediction cohort" in i["reason"] for i in excluded))

    def test_one_malformed_panel_does_not_discard_independent_valid_panels(self):
        self.write("dna", **self.panel_fields())
        self.write("rna", **{**self.panel_fields(), "test_prediction": None})
        result = self.read_panels()
        self.assertEqual([row["variant"] for row in result.records], ["dna"])
        coverage = {row["variant"]: row for row in result.coverage}
        self.assertEqual(coverage["dna"]["status"], "complete")
        self.assertEqual(coverage["rna"]["status"], "missing")

    def test_prediction_mode_preserves_duplicate_and_legacy_alias_boundaries(self):
        self.write("dna", legacy=True, **self.panel_fields())
        self.write("dna", folder=self.repeats, **self.panel_fields())
        result = self.read(require_predictions=True, expected_shape=(2, 2), expected_seeds=(0,))
        self.assertEqual(result.duplicate_keys, [("dna", 0)])
        self.assertFalse(result.records)
        self.assertEqual(result.coverage[0]["status"], "duplicate; blocked")
        self.write("rna", 1, legacy=True, folder=self.repeats, **self.panel_fields())
        result = self.read(require_predictions=True, expected_shape=(2, 2))
        self.assertTrue(any("historical mapping" in i["reason"] for i in result.issues))

    def test_present_malformed_identity_cannot_fall_back_to_legacy_label_mapping(self):
        for identity in (None, {}, [], {"schema": "malformed"}):
            with self.subTest(identity=identity):
                self.write("dna", legacy=True, identity=identity, **self.panel_fields())
                self.assertFalse(self.read_panels().records)

    def test_versioned_identity_checks_request_settings_and_result_metadata(self):
        self.write("dna", identity=self.modern_identity(), **self.panel_fields())
        self.assertEqual(len(self.read_panels().records), 1)
        changes = (
            ("loader", "n_heads", 4), ("loader", "budget", 200_000_000),
            ("loader", "val_fraction", 0.1), ("loader", "batch_size", 8),
            ("loader", "seed", 1), ("loader", "kind", "blf"),
            ("model", "modalities", ["rna"]), ("model", "n_outputs", 3),
            ("model", "kwargs", {"n_heads": 4}), ("config", "batch_size", 8),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                identity = self.modern_identity()
                identity[section][key] = value
                self.write("dna", identity=identity, **self.panel_fields())
                self.assertFalse(self.read_panels().records)
        for key, value in (("variant", "rna"), ("benchmark", "other")):
            identity = self.modern_identity()
            identity[key] = value
            self.write("dna", identity=identity, **self.panel_fields())
            self.assertFalse(self.read_panels().records)

    def test_dataset_order_fingerprint_does_not_reject_budget_sorted_evaluation_order(self):
        from shared_code.fusion import run_identity

        dataset_targets = self.panel_fields()["test_truth"]
        # Invented cost-sorted batches emit the second dataset row before the first.
        budgeted_batches = ((1,), (0,))
        emitted_targets = [dataset_targets[index] for batch in budgeted_batches for index in batch]
        identity = self.modern_identity()
        identity["inputs"]["split_targets"] = {
            "test": run_identity.array_record(dataset_targets, "float32")}
        self.assertNotEqual(identity["inputs"]["split_targets"]["test"],
                            run_identity.array_record(emitted_targets, "float32"))
        fields = {**self.panel_fields(), "test_truth": emitted_targets,
                  "test_prediction": list(reversed(self.panel_fields()["test_prediction"]))}
        self.write("dna", identity=identity, **fields)
        rna_identity = deepcopy(identity)
        rna_identity["variant"] = "rna"
        rna_identity["model"]["modalities"] = ["rna"]
        self.write("rna", identity=rna_identity, **fields)
        result = self.read_panels()
        self.assertEqual([row["variant"] for row in result.records], ["dna", "rna"])
        self.assertTrue(all(row["result"]["test_truth"] == emitted_targets for row in result.records))

    def test_prediction_options_reject_ambiguous_or_invalid_requests(self):
        for options in ({"require_predictions": 1}, {"expected_shape": (2, 2)},
                        {"require_predictions": True, "expected_shape": (0, 2)},
                        {"require_predictions": True, "expected_shape": (2, True)},
                        {"require_predictions": True, "expected_shape": (2,)},
                        {"require_predictions": True, "expected_shape": "2,2"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.read(**options)

    def test_reader_import_has_no_model_or_numeric_library_dependency(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "pandas", "transformers"}:
                raise AssertionError(f"reader imported {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            importlib.reload(isoformer_results)


if __name__ == "__main__":
    unittest.main()
