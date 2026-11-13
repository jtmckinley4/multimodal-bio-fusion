"""Check split-notebook handoffs without data, encoders, fits or research metrics."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
LUCAONE = ROOT / "Code/stage1/lucaone_representation_comparison.ipynb"
GTEX = ROOT / "Code/stage3/gtex_frozen_fusion_reference.ipynb"


def cells(path):
    book = json.loads(path.read_text(encoding="utf-8"))
    return {cell["id"]: cell for cell in book["cells"]}


def source(cell):
    return "".join(cell["source"])


def execute(cell, namespace):
    with redirect_stdout(StringIO()):
        exec(compile(source(cell), f"split-fixture:{cell['id']}", "exec"), namespace)


class ReorganizedMethodNotebookTests(unittest.TestCase):
    def test_lucaone_read_mode_does_not_touch_moved_diagnostic_inputs(self):
        blocked = Mock(side_effect=AssertionError("Diagnostic input or Hub query reached"))
        namespace = {
            "READ_SAVED_RESULTS": True,
            "pd": SimpleNamespace(read_csv=blocked, DataFrame=blocked),
            "datasets": SimpleNamespace(blf_benchmark_rows=blocked),
            "__builtins__": {"print": print},
        }
        # No import builtin is supplied: the Hub branch must remain unentered too.
        for cell_id in ("s4-gtex-lengths", "s4c4", "s4c6"):
            execute(cells(LUCAONE)[cell_id], namespace)
        blocked.assert_not_called()
        self.assertNotIn("gtex", namespace)

    def test_pilot_has_an_independent_input_load_before_consumers(self):
        book = json.loads(GTEX.read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in book["cells"]]
        frame = object()
        read_csv = Mock(return_value=frame)
        data_dir = ROOT / "tests" / "invented-data"
        namespace = {
            "pd": SimpleNamespace(read_csv=read_csv),
            "datasets": SimpleNamespace(DEFAULT_DATA_DIR=data_dir),
            "isoformer": SimpleNamespace(GTEX_PILOT="invented-pilot.csv"),
        }
        execute(cells(GTEX)["gtex-pilot-inputs"], namespace)
        read_csv.assert_called_once_with(data_dir / "invented-pilot.csv")
        self.assertIs(namespace["gtex"], frame)
        self.assertLess(ids.index("gtex-pilot-inputs"), ids.index("s4c10"))
        self.assertNotIn("s4-gtex-lengths", ids)

    def test_repeat_settings_do_not_reconfigure_the_baseline(self):
        book = cells(GTEX)
        namespace = {
            "Path": Path, "CODE_DIR": ROOT / "Code", "token_embeddings": object(),
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            ),
            # Baseline prefers MPS when both backends exist; repeats retain CUDA first.
            "DEVICE": "mps",
        }
        execute(book["s4-pilot-settings"], namespace)
        names = ("DEVICE", "BENCHMARK", "CACHE_ROOT", "RESULTS_DIR", "MAX_BATCH",
                 "CELL_BUDGET", "N_HEADS", "VAL_FRACTION", "MAX_EPOCHS", "SEEDS",
                 "VARIANTS", "CROP", "CONTEXT", "GTEX_CACHE_RECOVERY")
        baseline = {name: namespace[name] for name in names}
        execute(book["gtex-repeat-setup"], namespace)
        execute(book["264945ea"], namespace)
        self.assertEqual({name: namespace[name] for name in names}, baseline)
        self.assertEqual(namespace["REPEAT_DEVICE"], "cuda")
        self.assertTrue(namespace["READ_SAVED_RESULTS"])
        self.assertFalse(namespace["RUN_ISOFORMER_SEEDS"])
        self.assertEqual(namespace["ISOFORMER_SEEDS"], (1, 2))
        self.assertEqual(namespace["REPEAT_CACHE_ROOT"], ROOT / "Code/stage3/embeddings")
        self.assertEqual(namespace["REPEAT_RESULTS_DIR"], ROOT / "Code/stage4/results/gtex_isoformer")
        self.assertEqual(namespace["REPEAT_CACHE_RECOVERY"], ROOT / "Code/stage3/gtex_cache_recovery.json")
        # Mutating repeat settings cannot change globals captured by baseline helpers.
        namespace["REPEAT_N_HEADS"] = 2
        namespace["REPEAT_MAX_BATCH"] = 3
        namespace["REPEAT_DEVICE"] = "cpu"
        self.assertEqual({name: namespace[name] for name in names}, baseline)
        loaded = {node.id for node in ast.walk(ast.parse(source(book["isoformer-fit-def"])))
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        self.assertFalse(loaded.intersection({"DEVICE", "MAX_BATCH", "CELL_BUDGET",
                                             "N_HEADS", "VAL_FRACTION", "MAX_EPOCHS"}))

    def test_repeat_read_branch_needs_no_baseline_data_or_cache(self):
        blocked = Mock(side_effect=AssertionError("Repeat computation reached"))
        namespace = {
            "READ_SAVED_RESULTS": True,
            "tokens": SimpleNamespace(CacheVerification=blocked),
            "isoformer": SimpleNamespace(GTExStreams=SimpleNamespace(open=blocked)),
            "fit_isoformer_variant": blocked,
        }
        execute(cells(GTEX)["f9d1f49b"], namespace)
        blocked.assert_not_called()

    def test_lucaone_producer_capture_resolves_its_moved_code_cells(self):
        book = cells(LUCAONE)
        tree = ast.parse(source(book["lucaone-complementarity-inputs"]))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "notebook_source"]
        self.assertEqual(len(calls), 1)
        observed = []

        def capture(path, ids):
            self.assertEqual(path, LUCAONE)
            producers = cells(path)
            self.assertTrue(all(producers[cell_id]["cell_type"] == "code" for cell_id in ids))
            observed.extend(ids)

        eval(compile(ast.Expression(calls[0]), "moved-producer", "eval"), {
            "CODE_DIR": ROOT / "Code",
            "matched_probes": SimpleNamespace(notebook_source=capture),
        })
        self.assertEqual(observed, ["lucaone-complementarity-inputs", "lucaone-complementarity-results"])


if __name__ == "__main__":
    unittest.main()
