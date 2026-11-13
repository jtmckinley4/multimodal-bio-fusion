"""Synthetic CSV and isolated read-branch checks; no notebook research execution."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code import mrfp_cache


class MrfpCacheTests(unittest.TestCase):
    def setUp(self):
        self.identity = mrfp_cache.fingerprint({"folds": 5, "sample_seed": 42})
        self.expected = {("encoder", seed): self.identity for seed in (0, 1)}
        self.arguments = dict(keys=("model", "seed"), expected=self.expected,
                              numeric=("seed", "r2"), source="fixture.csv")

    def row(self, seed=0, **changes):
        return {"model": "encoder", "seed": seed, "r2": 0.2,
                mrfp_cache.IDENTITY_COLUMN: self.identity, **changes}

    def test_exact_seed_keys_not_same_row_count(self):
        rows = [self.row(0), self.row(7)]
        found = mrfp_cache.checked_rows(rows, **self.arguments)
        self.assertEqual(len(found), len(self.expected))
        self.assertEqual(set(self.expected) - set(found), {("encoder", 1)})

    def test_duplicate_seed_is_not_complete_coverage(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mrfp_cache.checked_rows([self.row(), self.row()], **self.arguments)

    def test_seed_type_is_integer_not_bool_or_float_alias(self):
        for seed in (True, False, 1.0, "1", -1):
            with self.assertRaises(ValueError):
                mrfp_cache.checked_rows([self.row(seed)], **self.arguments)

    def test_legacy_cache_stops_without_mutation(self):
        rows = [self.row()]
        del rows[0][mrfp_cache.IDENTITY_COLUMN]
        before = deepcopy(rows)
        with self.assertRaisesRegex(ValueError, "pd.read_csv"):
            mrfp_cache.checked_rows(rows, **self.arguments)
        self.assertEqual(rows, before)

    def test_legacy_rows_outside_request_also_stop(self):
        with self.assertRaisesRegex(ValueError, "legacy"):
            mrfp_cache.checked_rows([self.row(), self.row(7, calculation_sha256=None)],
                                    **self.arguments)

    def test_changed_settings_reject_reuse(self):
        changed = mrfp_cache.fingerprint({"folds": 4, "sample_seed": 42})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            mrfp_cache.checked_rows([self.row(calculation_sha256=changed)], **self.arguments)

    def test_order_values_shapes_and_dtypes_affect_input_identity(self):
        values = np.array([[1, 2], [3, 4]], dtype=np.float32)
        original = mrfp_cache.fingerprint({"seed": 0}, arrays=(values,))
        for changed in (values[::-1], values + 1, values.reshape(1, 4), values.astype(np.float64)):
            self.assertNotEqual(original, mrfp_cache.fingerprint({"seed": 0}, arrays=(changed,)))
        self.assertEqual(original, mrfp_cache.fingerprint({"seed": 0}, arrays=(values.copy(),)))
        with self.assertRaisesRegex(ValueError, "object"):
            mrfp_cache.fingerprint({}, arrays=(values.astype(object),))

    def test_ordered_sequences_and_folding_windows_affect_identity(self):
        first = {"sequences": ["AAA", "CCC"], "window": 200}
        original = mrfp_cache.fingerprint(first)
        self.assertNotEqual(original, mrfp_cache.fingerprint({**first, "window": 60}))
        self.assertNotEqual(original, mrfp_cache.fingerprint({**first, "sequences": ["CCC", "AAA"]}))

    def test_upsert_replaces_once_and_preserves_other_seeds(self):
        rows = [self.row(0), self.row(1)]
        before = deepcopy(rows)
        combined = mrfp_cache.upsert(rows, [self.row(1, r2=0.4)], **self.arguments)
        self.assertEqual(len(combined), 2)
        self.assertEqual([r["r2"] for r in combined], [0.2, 0.4])
        self.assertEqual(rows, before)
        self.assertEqual(mrfp_cache.upsert(combined, [self.row(1, r2=0.4)], **self.arguments), combined)

    def test_upsert_rejects_unrequested_seed_and_conflicts_before_changes(self):
        rows = [self.row()]
        before = deepcopy(rows)
        with self.assertRaisesRegex(ValueError, "outside"):
            mrfp_cache.upsert(rows, [self.row(7)], **self.arguments)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            mrfp_cache.upsert(rows, [self.row(calculation_sha256="a" * 64)], **self.arguments)
        self.assertEqual(rows, before)

    def test_control_group_is_part_of_gain_key(self):
        rows = [self.row(controls="Track A"), self.row(controls="all controls")]
        expected = {(r["controls"], r["model"], r["seed"]): self.identity for r in rows}
        found = mrfp_cache.checked_rows(rows, keys=("controls", "model", "seed"),
                                        expected=expected, numeric=("seed", "r2"), source="gain.csv")
        self.assertEqual(len(found), 2)

    def test_sequence_key_and_required_values_are_checked(self):
        expected = {("AAA",): self.identity}
        arguments = dict(keys=("sequence",), expected=expected,
                         numeric=("energy",), source="structure.csv")
        row = {"sequence": "AAA", "energy": -1.0, "calculation_sha256": self.identity}
        self.assertEqual(len(mrfp_cache.checked_rows([row], **arguments)), 1)
        for changes in ({"energy": float("nan")}, {"sequence": None},
                        {"sequence": float("nan")}, {"energy": "-1.0"}):
            with self.assertRaises(ValueError):
                mrfp_cache.checked_rows([{**row, **changes}], **arguments)


class MrfpSavedReadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, rows, columns):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_probe(self, path):
        return mrfp_cache.read_saved(path, keys=("model", "seed"), numeric=("seed", "r2"))

    def test_legacy_read_does_not_certify_identity_or_rewrite(self):
        path = self.write("probe.csv", [{"model": "encoder", "seed": 0, "r2": .2}],
                          ["model", "seed", "r2"])
        before = path.read_bytes()
        table = self.read_probe(path)
        selected, coverage = mrfp_cache.complete_groups(table, {"encoder": [("encoder", 0)]})
        self.assertEqual(selected[0]["r2"], .2)
        self.assertNotIn(mrfp_cache.IDENTITY_COLUMN, selected[0])
        self.assertEqual(coverage[0]["identity"], "unknown for all saved rows")
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_key_withholds_all_copies_including_invalid_copy(self):
        for scores in ((.2, .3, .4), ("nan", .3), (.2, "nan")):
            path = self.write("probe.csv", [
                {"model": "encoder", "seed": 0, "r2": score} for score in scores
            ], ["model", "seed", "r2"])
            table = self.read_probe(path)
            self.assertFalse(table.rows)
            self.assertTrue(any("duplicate" in issue for issue in table.issues))

    def test_missing_schema_and_invalid_values_are_reported(self):
        missing = self.read_probe(self.root / "missing.csv")
        self.assertFalse(missing.rows)
        self.assertIn("unavailable", missing.issues[0])
        path = self.write("schema.csv", [{"model": "encoder"}], ["model"])
        self.assertIn("columns", self.read_probe(path).issues[0])
        path = self.write("invalid.csv", [
            {"model": "encoder", "seed": seed, "r2": score}
            for seed, score in (("1.0", .2), ("True", .2), (-1, .2), (0, "inf"), (1, ""))
        ], ["model", "seed", "r2"])
        table = self.read_probe(path)
        self.assertFalse(table.rows)
        self.assertEqual(len(table.issues), 5)

    def test_recorded_identity_is_not_revalidated_and_malformed_identity_is_withheld(self):
        path = self.write("identity.csv", [
            {"model": "encoder", "seed": seed, "r2": .2, "calculation_sha256": identity}
            for seed, identity in ((0, "a" * 64), (1, "bad"))
        ], ["model", "seed", "r2", "calculation_sha256"])
        table = self.read_probe(path)
        self.assertEqual(set(table.rows), {("encoder", 0)})
        _, coverage = mrfp_cache.complete_groups(table, {"encoder": [("encoder", 0)]})
        self.assertEqual(coverage[0]["identity"], "recorded; not revalidated")

    def test_duplicate_headers_and_unexpected_field_counts_are_withheld(self):
        path = self.root / "malformed.csv"
        path.write_text("model,seed,r2,r2\nencoder,0,.2,.3\n", encoding="utf-8")
        table = self.read_probe(path)
        self.assertFalse(table.rows)
        self.assertIn("duplicate column", table.issues[0])
        path.write_text("model,seed,r2\nencoder,0,.2,extra\nencoder,1\nencoder,2,.4\n",
                        encoding="utf-8")
        table = self.read_probe(path)
        self.assertEqual(set(table.rows), {("encoder", 2)})
        self.assertEqual(len(table.issues), 2)

    def test_missing_seed_omits_only_its_group_and_extra_seed_cannot_replace_it(self):
        table = mrfp_cache.SavedTable({
            ("a", seed): {"model": "a", "seed": seed, "r2": .2} for seed in (0, 7)
        } | {("b", seed): {"model": "b", "seed": seed, "r2": .4} for seed in (0, 1)}, [])
        rows, coverage = mrfp_cache.complete_groups(
            table, {name: [(name, seed) for seed in (0, 1)] for name in ("a", "b")},
        )
        self.assertEqual({row["model"] for row in rows}, {"b"})
        self.assertEqual(coverage[0]["missing keys"], [("a", 1)])
        with self.assertRaisesRegex(ValueError, "distinct"):
            mrfp_cache.complete_groups(table, {"a": [("a", 0), ("a", 0)]})

    def notebook_sources(self):
        notebook = Path(__file__).resolve().parents[1] / "Code/stage1/mrfp_representation_control.ipynb"
        return {cell["id"]: "".join(cell["source"])
                for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]}

    def execute_selected(self, source, namespace, *, assignment_names=(), read_branch=False):
        """Compile isolated orchestration only; exclude every research else body."""
        statements = []
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in assignment_names
                for target in statement.targets
            ):
                statements.append(statement)
            if (read_branch and isinstance(statement, ast.If)
                    and isinstance(statement.test, ast.Name)
                    and statement.test.id == "READ_SAVED_RESULTS"):
                statements.extend(statement.body)
        exec(compile(ast.Module(body=statements, type_ignores=[]), "isolated saved-results branch", "exec"), namespace)

    def check_notebook_read_branches(self, *, incomplete):
        sources = self.notebook_sources()
        settings = {}
        self.execute_selected(sources["3ccd3a1d"], settings, assignment_names=("READ_SAVED_RESULTS",))
        self.assertIs(settings["READ_SAVED_RESULTS"], True)
        forbidden = Mock(side_effect=AssertionError("read mode called a producer or writer"))
        structure_path = self.write("structure.csv", [
            {"sequence": "AAA", "nussinov_200": 0., "mfe_start_60": 0., "mfe_full": 0.},
        ], ["sequence", "nussinov_200", "mfe_start_60", "mfe_full"])
        probe_path = self.write("probe.csv", [
            {"model": model, "seed": seed, "r2": .2}
            for model in ("Encoder", "random features (640)")
            for seed in (42, 0, 1)
            if not (incomplete and model == "Encoder" and seed == 1)
        ], ["model", "seed", "r2"])
        gain_path = self.write("gain.csv", [
            {"controls": control, "model": "Encoder", "seed": seed,
             "controls R2": .1, "with encoder R2": .2}
            for control in ("Track A controls", "all controls") for seed in (0, 1)
        ], ["controls", "model", "seed", "controls R2", "with encoder R2"])
        if incomplete:
            structure_path.unlink()  # One absent input must not block independent score tables.
        before = {path: path.read_bytes() for path in self.root.iterdir()}
        ns = {
            "READ_SAVED_RESULTS": True, "pd": pd, "mrfp_cache": mrfp_cache,
            "STRUCTURE_CACHE": structure_path, "PROBE_CACHE": probe_path, "GAIN_CACHE": gain_path,
            "PROBE_SEED": 42, "N_ASSIGNMENTS": 2, "display": Mock(), "save": forbidden,
            "ENCODERS": {"encoder": SimpleNamespace(key="encoder", label="Encoder", modality="RNA")},
            "datasets": SimpleNamespace(load_dataset=forbidden),
            "embeddings": SimpleNamespace(read_saved=forbidden),
            "analysis": SimpleNamespace(probe_scores=forbidden, centroid_spread=forbidden),
            "structure_row": forbidden, "gain_over": forbidden,
            "RNA": SimpleNamespace(fold=forbidden),
        }
        with redirect_stdout(io.StringIO()), patch.object(Path, "mkdir", forbidden), \
                patch.object(pd.DataFrame, "to_csv", forbidden), \
                patch.object(mrfp_cache, "fingerprint", forbidden):
            self.execute_selected(sources["68a04a03"], ns,
                                  assignment_names=("CACHE_FIELDS",), read_branch=True)
            self.execute_selected(sources["26ae21c9"], ns,
                                  assignment_names=("MODALITY", "LABEL"), read_branch=True)
            for cell_id in ("57035a06", "2153fc8a", "67f402b3"):
                self.execute_selected(sources[cell_id], ns, read_branch=True)
            self.execute_selected(sources["704b742a"], ns,
                                  assignment_names=("track_a_members", "control_groups"), read_branch=True)
            self.execute_selected(sources["bd151ece"], ns, read_branch=True)
            self.execute_selected(sources["bd151ece"], ns, assignment_names=("KIND", "SEEDS"))
            self.execute_selected(sources["d9e3a32b"], ns, read_branch=True)
            self.execute_selected(sources["8b12ee61"], ns, read_branch=True)
            # These cells only summarize the tiny synthetic rows already read.
            for cell_id in ("d270dab9", "d5099ac5"):
                exec(compile(sources[cell_id], "synthetic saved-row summary", "exec"), ns)
        forbidden.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, before)
        self.assertEqual(set(ns["PROBES"].index),
                         {"random features (640)"} if incomplete else {"Encoder", "random features (640)"})
        self.assertEqual(set(ns["GAINS"].index.get_level_values("model")), {"Encoder"})
        self.assertEqual(len(ns["gains"]), 4)

    def test_complete_saved_branch_fixture_avoids_all_producers_and_writes(self):
        self.check_notebook_read_branches(incomplete=False)

    def test_missing_saved_branch_fixture_keeps_independent_displays(self):
        self.check_notebook_read_branches(incomplete=True)


if __name__ == "__main__":
    unittest.main()
