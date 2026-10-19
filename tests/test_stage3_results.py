"""Check saved-result selection using invented records, without opening research caches."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Code/stage3/03_blf_build_check.ipynb"
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion.train import Result, TrainConfig


class Stage3ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    def setUp(self):
        # Only configuration and reader definitions are executed. This substitute exposes
        # no model, cache, optimizer or training entry points to the notebook's namespace.
        self.stored = {}
        self.values = {
            "CODE_DIR": ROOT / "Code", "asdict": asdict, "replace": replace, "np": np,
            "train": SimpleNamespace(TrainConfig=TrainConfig,
                                     Result=SimpleNamespace(load=lambda path: self.stored[path.name])),
        }
        for marker in ('BENCHMARK = "fungal_expression"', "def check_comparable_runs(",
                       "def select_mil_weight("):
            cells = [cell for cell in self.cells if marker in "".join(cell["source"])]
            self.assertEqual(len(cells), 1)
            with patch("builtins.open", side_effect=AssertionError("Unexpected file access")):
                exec(compile("".join(cells[0]["source"]), NOTEBOOK.name, "exec"), self.values)

    def record(self, variant, weight=0.0, **settings):
        config = asdict(replace(self.values["BASE_CONFIG"], entropy_lambda=weight, **settings))
        identity = {
            "version": 1, "model_kwargs": {}, "widths": {"dna": 12, "rna": 9, "prot": 6},
            "n_outputs": 1, "inputs": {"rows_sha256": "a" * 64},
            "environment": {"device": "cpu"}, "code_sha256": {"train.py": "b" * 64},
            "variant": variant, "config": deepcopy(config),
        }
        return Result(variant, "fungal_expression", config, 1, 0, 0.5, 0.4, 1.0,
                      history=[{"epoch": 0, "val_score": 0.5}], identity=identity)

    def load(self, records, config=None):
        self.stored = {f"{index:03d}.json": result for index, result in enumerate(records)}
        directory = SimpleNamespace(glob=lambda pattern: [Path(name) for name in self.stored])
        with redirect_stdout(io.StringIO()), patch(
                "builtins.open", side_effect=AssertionError("Unexpected file access")):
            return self.values["load_comparison_runs"](
                directory, config or self.values["BASE_CONFIG"])

    def test_selected_directory_and_configuration_groups_stay_separate(self):
        expected = ROOT / "Code/stage3/results/fungal_expression/reproducible"
        self.assertEqual(self.values["COMPARISON_DIR"], expected)
        appendix = [self.record(variant, overrides.get("entropy_lambda", 0.0))
                    for variant, overrides in self.values["RUNS"]]
        equation = [self.record(result.variant, result.config["entropy_lambda"], tconv="equation")
                    for result in appendix if not result.variant.startswith("unimodal:")]
        original = deepcopy(appendix + equation)
        self.assertEqual(len(appendix), 9)
        self.assertEqual(len(equation), 6)
        self.assertEqual(self.load(appendix + equation), appendix)
        self.assertEqual(self.load(appendix + equation,
                                   replace(self.values["BASE_CONFIG"], tconv="equation")), equation)
        self.assertEqual(appendix + equation, original)

    def test_duplicate_candidates_and_mixed_recorded_cohorts_are_rejected(self):
        first = self.record("concat")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load([first, deepcopy(first)])
        for field in ("inputs", "environment", "code_sha256"):
            with self.subTest(field=field):
                other = self.record("cross")
                other.identity[field] = {"different": "producer"}
                with self.assertRaisesRegex(ValueError, "different"):
                    self.load([first, other])
        legacy = self.record("cross")
        legacy.identity = None
        with self.assertRaisesRegex(ValueError, "separately"):
            self.load([first, legacy])

    def test_other_settings_are_excluded_without_replacing_selected_evidence(self):
        expected = self.record("concat")
        others = [self.record("cross", seed=1), self.record("cross", channels=1280),
                  self.record("mil", 0.25)]
        self.assertEqual(self.load([expected, *others]), [expected])

    def test_mil_selection_uses_validation_and_keeps_zero_weight_separate(self):
        runs = [self.record("mil", weight) for weight in (0.0, 0.01, 0.5, 1.0)]
        for result, validation, test in zip(runs, (0.99, 0.7, 0.8, 0.6), (0.99, 0.95, 0.4, 0.9)):
            result.val_score, result.test_score = validation, test
        weight = self.values["select_mil_weight"](runs)
        self.assertEqual(weight, 0.5)
        self.assertEqual(self.values["selected_scores"](runs, weight),
                         {"mil (lam 0)": 0.99, "mil": 0.4})

    def test_main_progress_reflects_empty_partial_and_complete_saved_groups(self):
        cell = next(cell for cell in self.cells if "main_results = load_comparison_runs(" in "".join(cell["source"]))
        paper_rows = ["Nucleotide Transformer", "RNA-FM", "ESM-2 (8M)",
                      "Concatenation", "MIL + Entropy", "Cross Attention"]
        self.values.update(
            pd=pd, display=lambda *args: None,
            TABLE1=pd.DataFrame({"fungal_expression": [0.5] * len(paper_rows)}, index=paper_rows),
            TABLE4=pd.DataFrame({"fungal_expression": [0.5]}, index=["MIL without entropy"]),
        )
        records = [self.record(variant, overrides.get("entropy_lambda", 0.0))
                   for variant, overrides in self.values["RUNS"]]
        for count in (0, 4, 9):
            with self.subTest(saved=count):
                self.values["load_comparison_runs"] = lambda *args: records[:count]
                with redirect_stdout(io.StringIO()) as output, patch(
                        "builtins.open", side_effect=AssertionError("Unexpected file access")):
                    exec(compile("".join(cell["source"]), NOTEBOOK.name, "exec"), self.values)
                self.assertIn(f"Main sweep: {count}/9 planned candidates recorded", output.getvalue())
                self.assertEqual(len(self.values["missing_runs"]), 9 - count)


if __name__ == "__main__":
    unittest.main()
