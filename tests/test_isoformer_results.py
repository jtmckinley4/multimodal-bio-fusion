"""Check the GTEx result reader using invented JSONs, without research data or ML."""

import builtins
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
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
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
