"""Cache-only LucaOne follow-up boundaries using invented rows and small arrays."""

import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import lucaone, matrix_cache


def cell(cell_id):
    book = json.loads((ROOT / "Code/stage1/lucaone_representation_comparison.ipynb").read_text(encoding="utf-8"))
    return "".join(next(c for c in book["cells"] if c.get("id") == cell_id)["source"])


class ReadOnlyLucaCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gene.npz"
        self.texts = ["ATG", "GCC", "TTA"]
        self.matrix = np.arange(12, dtype=float).reshape(3, 4)

    def write(self, matrix=None):
        matrix_cache.write(
            self.path, self.matrix if matrix is None else matrix,
            lucaone._cache_metadata("default", "gene", self.texts, lucaone.CONTEXT_TOKENS),
        )

    def test_missing_cache_never_loads_or_writes(self):
        with patch.object(lucaone, "load", side_effect=AssertionError("no inference")), \
                patch.object(matrix_cache, "write", side_effect=AssertionError("no publication")):
            with self.assertRaises(FileNotFoundError):
                lucaone.read_cached(self.texts, "gene", self.path)
        self.assertFalse(self.path.exists())

    def test_matching_cache_read_preserves_bytes(self):
        self.write()
        original = self.path.read_bytes()
        with patch.object(lucaone, "load", side_effect=AssertionError("no inference")):
            actual = lucaone.read_cached(self.texts, "gene", self.path)
        np.testing.assert_array_equal(actual, self.matrix)
        self.assertEqual(original, self.path.read_bytes())

    def test_reordered_inputs_or_changed_stream_preserve_cache(self):
        self.write()
        original = self.path.read_bytes()
        for texts, stream in [(self.texts[::-1], "gene"), (self.texts, "prot")]:
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                lucaone.read_cached(texts, stream, self.path)
        self.assertEqual(original, self.path.read_bytes())

    def test_nonfinite_matrix_is_not_accepted(self):
        bad = self.matrix.copy()
        bad[0, 0] = np.nan
        self.write(bad)
        with self.assertRaises(ValueError):
            lucaone.read_cached(self.texts, "gene", self.path)


class FocusedNotebookTests(unittest.TestCase):
    def test_phase_a_context_errors_are_isolated_only_in_read_mode(self):
        read_results = Mock(side_effect=ValueError("invented ambiguous records"))
        messages = Mock()
        namespace = {
            "BENCHMARK": "fixture", "STAGE4_DIR": Path("unused-fixture"),
            "READ_SAVED_RESULTS": True, "print": messages,
            "swap_results": SimpleNamespace(read_results=read_results),
            "train": SimpleNamespace(TrainConfig=lambda **settings: SimpleNamespace(**settings)),
            "blf": SimpleNamespace(ENCODER_OF_MODALITY={"dna": "fixture-encoder"}),
        }
        # Defining the shared display helper must not inspect files or compute results.
        exec(cell("phasea-context-def"), namespace)
        read_results.assert_not_called()
        exec(cell("dc6ff652"), namespace)
        self.assertIn("continuing to IsoFormer coverage", messages.call_args.args[0])
        namespace["READ_SAVED_RESULTS"] = False
        with self.assertRaisesRegex(ValueError, "invented ambiguous records"):
            exec(cell("dc6ff652"), namespace)
        self.assertEqual(read_results.call_count, 2)

    def test_disabled_followup_does_not_open_inputs(self):
        forbidden = Mock(side_effect=AssertionError("no input read"))
        namespace = {"RUN_LUCAONE_COMPARISON": False, "print": lambda *args: None,
                     "datasets": SimpleNamespace(blf_benchmark_rows=forbidden)}
        exec(cell("lucaone-complementarity-inputs"), namespace)
        self.assertIsNone(namespace["lucaone_followup_inputs"])
        forbidden.assert_not_called()

    def test_enabled_inputs_are_only_two_cached_streams_and_fixed_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            rows = pd.DataFrame({"sequence": ["ATG", "GCC", "TTA", "GCA", "ACC", "TTT"],
                                 "Split": ["train", "train", "val", "val", "test", "test"],
                                 "Value": [1., 2., 3., 4., 5., 6.]}, index=[10, 11, 12, 13, 14, 15])
            cached = Mock(side_effect=lambda texts, stream, path, **kw:
                          np.ones((6, 2)) * (1 if stream == "gene" else 2))
            namespace = {
                "RUN_LUCAONE_COMPARISON": True, "BENCHMARK": "fixture", "CHECKPOINT": "default",
                "LUCAONE_DIR": base, "CODE_DIR": base, "STAGE4_DIR": base, "np": np,
                "datasets": SimpleNamespace(blf_benchmark_rows=lambda key: rows,
                                            DEFAULT_DATA_DIR=base,
                                            DATASETS={"fixture": SimpleNamespace(path="data.csv")}),
                "lucaone": SimpleNamespace(read_cached=cached),
                "matched_probes": SimpleNamespace(notebook_source=lambda *args: {"source": "fixture"}),
                "tokens": SimpleNamespace(grid_protein=lambda s: "M", file_fingerprint=lambda p: "fixture"),
            }
            exec(cell("lucaone-complementarity-inputs"), namespace)
            prepared = namespace["lucaone_followup_inputs"]
            self.assertEqual(cached.call_count, 2)
            self.assertEqual(set(prepared["matrices"]), {"gene", "protein", "gene + protein"})
            np.testing.assert_array_equal(prepared["matrices"]["gene + protein"][0], [1, 1, 2, 2])
            self.assertEqual(prepared["row_ids"][0], "fixture:10")
            for actual, expected in zip(prepared["splits"]["published"], [[0, 1], [2, 3], [4, 5]]):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(prepared["contrasts"]), 2)

    def test_disabled_report_reads_only_completed_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new-report.json"
            namespace = {
                "RUN_LUCAONE_COMPARISON": False, "LUCAONE_COMPARISON_PATH": path,
                "matched_probes": SimpleNamespace(run=Mock(side_effect=AssertionError("no fit")),
                                                  read=Mock(return_value={"scores": [], "comparisons": []})),
                "pd": pd, "display": lambda value: None, "print": lambda *args: None,
            }
            exec(cell("lucaone-complementarity-results"), namespace)
            namespace["matched_probes"].read.assert_not_called()
            path.write_text("fixture", encoding="utf-8")
            exec(cell("lucaone-complementarity-results"), namespace)
            namespace["matched_probes"].read.assert_called_once_with(path)
            namespace["matched_probes"].run.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "fixture")

    def test_enabled_report_keeps_declared_metric_and_overlap_policy(self):
        run = Mock(return_value=({"scores": [], "comparisons": []}, False))
        namespace = {
            "RUN_LUCAONE_COMPARISON": True, "lucaone_followup_inputs": {"name": "fixture"},
            "LUCAONE_COMPARISON_PATH": Path("unused-new-report.json"),
            "matched_probes": SimpleNamespace(run=run), "pd": pd,
            "LUCAONE_ALPHAS": (.01, 1.), "BOOTSTRAP_DRAWS": 20,
            "print": lambda *args: None, "display": lambda value: None,
        }
        exec(cell("lucaone-complementarity-results"), namespace)
        self.assertEqual(run.call_args.kwargs["metric"], "spearman")
        self.assertFalse(run.call_args.kwargs["require_disjoint_groups"])
        self.assertEqual(run.call_args.kwargs["alphas"], (.01, 1.))

    def test_missing_cache_stops_before_result_calculation(self):
        rows = pd.DataFrame({"sequence": ["ATG"], "Split": ["train"], "Value": [1.]})
        namespace = {
            "RUN_LUCAONE_COMPARISON": True, "BENCHMARK": "fixture", "CHECKPOINT": "default",
            "LUCAONE_DIR": Path("unused"), "np": np,
            "datasets": SimpleNamespace(blf_benchmark_rows=lambda key: rows),
            "tokens": SimpleNamespace(grid_protein=lambda s: "M"),
            "lucaone": SimpleNamespace(read_cached=Mock(side_effect=FileNotFoundError("fixture"))),
        }
        with self.assertRaises(FileNotFoundError):
            exec(cell("lucaone-complementarity-inputs"), namespace)
        self.assertIsNone(namespace["lucaone_followup_inputs"])

    def test_conflicting_flags_are_rejected_inside_setup(self):
        book = json.loads((ROOT / "Code/stage1/lucaone_representation_comparison.ipynb").read_text(encoding="utf-8"))
        setup = next("".join(c["source"]) for c in book["cells"]
                     if c["cell_type"] == "code" and "RUN_LUCAONE_COMPARISON = False" in "".join(c["source"]))
        tree = ast.parse(setup)
        guard = next(node for node in tree.body if isinstance(node, ast.If)
                     and "RUN_LUCAONE_COMPARISON" in ast.unparse(node.test))
        with self.assertRaises(ValueError):
            exec(compile(ast.Module(body=[guard], type_ignores=[]), "guard", "exec"),
                 {"RUN_LUCAONE_COMPARISON": True, "READ_SAVED_RESULTS": False})


if __name__ == "__main__":
    unittest.main()
