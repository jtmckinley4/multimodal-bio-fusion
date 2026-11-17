"""Source-only contracts for the A6 notebook; no notebook cell is executed."""

import ast
import json
from pathlib import Path
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "Code/stage4/region_aware_fusion.ipynb"


class RegionNotebookTests(unittest.TestCase):
    def setUp(self):
        self.book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.code = [cell for cell in self.book["cells"] if cell["cell_type"] == "code"]

    def test_all_cells_compile_and_have_no_claimed_execution(self):
        for index, cell in enumerate(self.code):
            compile("".join(cell["source"]), f"region-cell-{index}", "exec")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_launcher_settings_are_tagged_once_with_read_only_defaults(self):
        settings = [cell for cell in self.code if "fusion-study-settings" in cell["metadata"].get("tags", [])]
        self.assertEqual(len(settings), 1)
        assignments = {target.id: node.value for node in ast.parse("".join(settings[0]["source"])).body
                       if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
        self.assertTrue(ast.literal_eval(assignments["READ_SAVED_RESULTS"]))
        for name in ("RUN_INPUT_AUDIT", "RUN_PREPARATION", "RUN_EXTRACTION", "RUN_TRAINING", "RUN_ANALYSIS", "RUN_DIAGNOSTICS"):
            self.assertFalse(ast.literal_eval(assignments[name]))
        for name in ("DATASET_KEY", "SEED", "DEVICE", "PRODUCER", "RESULTS_ROOT", "DATA_CACHE_ROOT",
                     "TOKEN_CACHE_ROOT", "BASE_CONFIG", "CONFIG", "UNMAPPED_GENE_POLICY"):
            self.assertIn(name, assignments)
        self.assertEqual(ast.literal_eval(assignments["UNMAPPED_GENE_POLICY"]), "gene")
        after = self.code[self.code.index(settings[0]) + 1:]
        for cell in after:
            assigned = {target.id for node in ast.walk(ast.parse("".join(cell["source"])))
                        if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
            self.assertFalse(assigned & {"DATASET_KEY", "SEED", "DEVICE", "RESULTS_ROOT", "DATA_CACHE_ROOT", "TOKEN_CACHE_ROOT", "CONFIG"})

    def test_input_acquisition_and_execution_calls_have_explicit_gates(self):
        source = "\n".join("".join(cell["source"]) for cell in self.code)
        self.assertIn('if needs_inputs:', source)
        self.assertIn('needs_inputs = RUN_PREPARATION or RUN_EXTRACTION or RUN_TRAINING or RUN_DIAGNOSTICS', source)
        self.assertIn('if RUN_EXTRACTION:', source)
        self.assertIn('if RUN_TRAINING or RUN_DIAGNOSTICS:', source)
        self.assertIn('run_analysis=RUN_ANALYSIS', source)
        self.assertNotIn('mrnabench.load(', source)
        self.assertNotIn('download(', source)
        self.assertIn('mode="read"', source)
        self.assertIn('mode="extract"', source)
        self.assertIn('unmapped_gene_policy=UNMAPPED_GENE_POLICY', source)

    def test_cohort_policy_is_explicit_and_homology_map_is_still_required(self):
        nodes = [node for cell in self.code for node in ast.walk(ast.parse("".join(cell["source"])))]
        calls = [node for node in nodes if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "prepare_cohort"]
        self.assertEqual(len(calls), 1)
        arguments = {argument.arg: argument.value for argument in calls[0].keywords}
        self.assertEqual(ast.literal_eval(arguments["grouping"]), "homology")
        self.assertIsInstance(arguments["unmapped_gene_policy"], ast.Name)
        self.assertEqual(arguments["unmapped_gene_policy"].id, "UNMAPPED_GENE_POLICY")
        self.assertIn("homology_map", arguments)
        self.assertFalse(isinstance(arguments["homology_map"], ast.Constant)
                         and arguments["homology_map"].value is None)
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "load_local_homology_map" for node in nodes))

    def test_input_audit_reads_existing_json_without_loading_inputs(self):
        # Exercise only the guarded display cell with a fake reader, never a kernel.
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        cell = next(c for c in self.code if c["id"] == "region-audit-read")
        reader = Mock(return_value=None)
        inspect = Mock(side_effect=AssertionError("Reading must not audit inputs"))
        publish = Mock(side_effect=AssertionError("Reading must not publish"))
        namespace = {"RUN_INPUT_AUDIT": False, "INPUT_AUDIT_PATH": Path("missing.json"),
                     "region": SimpleNamespace(read_input_audit=reader, input_audit=inspect,
                                               publish_input_audit=publish)}
        with patch("builtins.print"):
            exec(compile("".join(cell["source"]), "audit-read-guard", "exec"), namespace)
        reader.assert_called_once_with(Path("missing.json"))
        inspect.assert_not_called()
        publish.assert_not_called()

    def test_input_audit_does_not_publish_canonical_preparation(self):
        cell = next(c for c in self.code if c["id"] == "region-10")
        tree = ast.parse("".join(cell["source"]))
        guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Name) and node.test.id == "RUN_PREPARATION"]
        self.assertEqual(len(guarded), 1)
        writes = [node for node in ast.walk(guarded[0]) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "publish_preparation"]
        self.assertEqual(len(writes), 1)
        self.assertIn("and not RUN_INPUT_AUDIT", "".join(
            next(c for c in self.code if c["id"] == "region-12")["source"]))

    def test_scientific_choices_and_execution_limits_are_visible(self):
        prose = "\n".join("".join(cell["source"]) for cell in self.book["cells"] if cell["cell_type"] == "markdown")
        for phrase in ("294", "homology", "terminal stop", "negative control", "midpoint",
                       "not a published", "parameter counts", "per-seed scores", "task 42"):
            self.assertIn(phrase, prose)
        self.assertFalse(any(ord(character) < 32 and character not in "\n\t" for character in prose))


if __name__ == "__main__":
    unittest.main()
