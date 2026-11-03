"""Synthetic CSVs and isolated read branches; no research notebook is executed."""

import ast
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import saved_results


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def forbidden(*args, **kwargs):
    raise AssertionError("A saved-results branch reached a producer or writer")


class NoProducer:
    def __getattr__(self, name):
        return forbidden


class HistoricalCSVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "summary.csv"
        self.messages = []

    def read(self, **changes):
        options = dict(
            required=("model", "score", "seeds"), key=("model",),
            numeric=("score",), integers=("seeds",),
            expected_keys=[("example",)], expected_values={"seeds": 3},
            bounds={"score": (-1, 1), "seeds": (1, None)},
            report=self.messages.append,
        )
        options.update(changes)
        return saved_results.read_historical_csv(self.path, **options)

    def test_valid_read_preserves_file_and_labels_unknown_identity(self):
        self.path.write_text("model,score,seeds\nexample,0.25,3\n", encoding="utf-8")
        before = self.path.read_bytes()
        self.assertEqual(self.read(), [{"model": "example", "score": 0.25, "seeds": 3}])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIn("actual seed IDs", self.messages[-1])
        self.assertIn("not fully recorded", self.messages[-1])

    def test_missing_summary_reports_without_creating_anything(self):
        self.assertIsNone(self.read())
        self.assertEqual(list(self.path.parent.iterdir()), [])
        self.assertIn("unavailable", self.messages[-1])

    def test_malformed_summaries_are_unavailable(self):
        fixtures = {
            "missing header": "model,score\nexample,0.2\n",
            "duplicate header": "model,score,seeds,seeds\nexample,0.2,3,3\n",
            "empty": "model,score,seeds\n",
            "short row": "model,score,seeds\nexample,0.2\n",
            "extra field": "model,score,seeds\nexample,0.2,3,extra\n",
            "empty key": "model,score,seeds\n,0.2,3\n",
            "duplicate": "model,score,seeds\nexample,0.2,3\nexample,0.3,3\n",
            "nan": "model,score,seeds\nexample,nan,3\n",
            "infinite": "model,score,seeds\nexample,inf,3\n",
            "not numeric": "model,score,seeds\nexample,unknown,3\n",
            "fractional count": "model,score,seeds\nexample,0.2,3.5\n",
            "wrong count": "model,score,seeds\nexample,0.2,2\n",
            "invalid domain": "model,score,seeds\nexample,1.2,3\n",
            "wrong entry": "model,score,seeds\nother,0.2,3\n",
        }
        for label, source in fixtures.items():
            with self.subTest(label=label):
                self.path.write_text(source, encoding="utf-8")
                before = self.path.read_bytes()
                self.assertIsNone(self.read())
                self.assertEqual(self.path.read_bytes(), before)

    def test_reader_contract_errors_are_not_disguised_as_bad_results(self):
        with self.assertRaises(ValueError):
            self.read(numeric=("undeclared",))
        with self.assertRaises(ValueError):
            self.read(bounds={"model": (0, 1)})


class SavedNotebookBranchesTests(unittest.TestCase):
    """Run only guarded reading statements with invented files and blocked producers."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.displayed = []
        self.messages = io.StringIO()

    def branch(self, notebook, cell_id, namespace):
        doc = json.loads((ROOT / "Code" / "stage4" / notebook).read_text(encoding="utf-8"))
        cell = next(c for c in doc["cells"] if c["id"] == cell_id)
        tree = ast.parse("".join(cell["source"]))
        self.assertEqual(len(tree.body), 1, cell_id)
        guard = tree.body[0]
        self.assertIsInstance(guard, ast.If, cell_id)
        self.assertIsInstance(guard.test, ast.Name, cell_id)
        self.assertEqual(guard.test.id, "READ_SAVED_RESULTS", cell_id)
        # The producer else branch is never compiled or evaluated by this fixture.
        isolated = ast.Module(body=guard.body, type_ignores=[])
        with redirect_stdout(self.messages), patch.object(Path, "mkdir", forbidden), \
                patch.object(pd.DataFrame, "to_csv", forbidden):
            exec(compile(isolated, f"fixture:{cell_id}", "exec"), namespace)

    def namespace(self):
        return {
            "READ_SAVED_RESULTS": True,
            "saved_results": saved_results,
            "pd": pd,
            "RESULTS_DIR": self.folder,
            "display": self.displayed.append,
            "lucaone": NoProducer(), "tokens": NoProducer(),
            "train": NoProducer(), "isoformer": NoProducer(),
            "probe_on_split": forbidden, "stream_texts": forbidden,
        }

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.folder.iterdir()}

    def test_default_settings_are_read_only(self):
        for notebook in ("tier2_lucaone_isoformer.ipynb", "tier3_halflife_te.ipynb"):
            doc = json.loads((ROOT / "Code/stage4" / notebook).read_text(encoding="utf-8"))
            assignments = [
                node.value.value
                for cell in doc["cells"] if cell["cell_type"] == "code"
                for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "READ_SAVED_RESULTS"
                        for t in node.targets)
            ]
            self.assertEqual(assignments, [True])

    def lucaone_fixture(self):
        env = self.namespace()
        env.update(BENCHMARK="invented", RUN_LUCAONE=True, RUN_ISOFORMER_SEEDS=True)
        env["encoders"] = SimpleNamespace(ENCODERS={
            key: SimpleNamespace(label=key) for key in ("a", "b", "c")
        })
        env["blf"] = SimpleNamespace(ENCODER_OF_MODALITY={"dna": "a", "rna": "b", "prot": "c"})
        names = [
            "Composition (GC, GC3, codon use, length)", "a", "b", "c",
            "BLF three encoders concatenated", "LucaOne, gene stream",
            "LucaOne, protein stream", "LucaOne, both streams",
        ]
        write_csv(self.folder / "invented_probe_scores.csv", ("", "probe Spearman", "width"),
                  [{"": name, "probe Spearman": 0.2, "width": 2} for name in names])
        comparisons = [
            "LucaOne both streams minus the best BLF single encoder",
            "LucaOne both streams minus the three concatenated",
            "LucaOne gene stream minus composition",
        ]
        write_csv(self.folder / "invented_comparisons.csv",
                  ("comparison", "difference", "ci_low", "ci_high", "first_wins", "rows", "groups"),
                  [dict(comparison=name, difference=0.1, ci_low=-0.1, ci_high=0.2,
                        first_wins=0.7, rows=10, groups=8) for name in comparisons])
        return env

    def test_lucaone_read_branches_block_producers_even_with_compute_flags_on(self):
        env = self.lucaone_fixture()
        before = self.snapshot()
        for cell_id in (
            "74c94f4f", "166c1b45", "1941755b", "9d482c82", "45007ed7", "6886c65d",
            "399c1745", "d5abb6f5", "0aee3c7f", "47fd36d1", "f9d1f49b",
        ):
            self.branch("tier2_lucaone_isoformer.ipynb", cell_id, env)
        self.assertEqual(len(self.displayed), 2)
        self.assertEqual(env["predictions"], {})
        self.assertEqual(before, self.snapshot())

    def test_lucaone_comparisons_still_display_when_probe_file_missing(self):
        env = self.lucaone_fixture()
        (self.folder / "invented_probe_scores.csv").unlink()
        for cell_id in ("0aee3c7f", "47fd36d1"):
            self.branch("tier2_lucaone_isoformer.ipynb", cell_id, env)
        self.assertIsNone(env["summary"])
        self.assertEqual(len(self.displayed), 1)
        self.assertIn("unavailable", self.messages.getvalue())

    def halflife_fixture(self):
        env = self.namespace()
        env.update(PRIMARY=("synthetic",), PROBE_SEEDS=(10, 20, 30, 40), RUN_EMBEDDINGS=True)
        env["mrnabench"] = SimpleNamespace(
            DATASETS={"synthetic": SimpleNamespace(targets=("target",))},
            load=forbidden, embed_with_cache=forbidden, probe_over_seeds=forbidden,
        )
        env["REGION_SETS"] = {"CDS only": ("cds",)}
        env["INPUTS"] = {"full transcript": ("rna", ("utr5", "cds", "utr3"))}
        gate = {
            "dataset": "synthetic", "rows": 10, "species": "invented", "measures": "invented",
            "targets present": "target", "has cds track": True, "has splice track": True,
            "rows with a coding region": 1, "rows with a 5' region": 1, "rows with a 3' region": 1,
            "coding sequence in frame": 1, "median transcript nt": 12, "longest transcript nt": 20,
            "median cds nt": 6, "distinct genes": 10, "largest gene group": 1,
            "repeated sequences": 0.2, "split rule": "invented", "verdict": "passes", "caveats": "",
        }
        write_csv(self.folder / "gate.csv", tuple(gate), [gate])
        baseline = dict(dataset="synthetic", target="target", representation="CDS only",
                        **{"test r": 0.2, "spread": 0.01, "seeds": 3})
        write_csv(self.folder / "composition_baselines.csv", tuple(baseline), [baseline])
        encoder_rows = [dict(dataset="synthetic", target="target", representation=name,
                             **{"test r": 0.3, "standard error": 0.01, "test Spearman": 0.2,
                                "seeds": 4, "grouping": "invented groups"})
                        for name in (*env["INPUTS"], "full transcript + protein", "three encoders fused")]
        write_csv(self.folder / "encoder_probes.csv", tuple(encoder_rows[0]), encoder_rows)
        gain_rows = [dict(dataset="synthetic", target="target",
                          **{"fused model": name, "fused r": 0.3, "best single r": 0.2,
                             "delta fusion": 0.1, "over composition": 0.1})
                     for name in ("full transcript + protein", "three encoders fused")]
        write_csv(self.folder / "fusion_gains.csv", tuple(gain_rows[0]), gain_rows)
        return env

    def test_halflife_read_branches_block_loads_fits_and_writes(self):
        env = self.halflife_fixture()
        env["usable"] = ["synthetic"]
        before = self.snapshot()
        for cell_id in (
            "8911ff86", "d2cdfe59", "9f809fa4", "d3c29f9e", "bc1b7b69",
            "6260d5de", "fc21afe3", "398d1d5f",
        ):
            self.branch("tier3_halflife_te.ipynb", cell_id, env)
        self.assertEqual(len(self.displayed), 5)
        self.assertEqual(env["embedded"], {})
        self.assertEqual(before, self.snapshot())

    def test_missing_gate_skips_baselines_but_reads_independent_tables(self):
        env = self.halflife_fixture()
        (self.folder / "gate.csv").unlink()
        for cell_id in ("8911ff86", "d3c29f9e", "fc21afe3", "398d1d5f"):
            self.branch("tier3_halflife_te.ipynb", cell_id, env)
        self.assertIsNone(env["gate"])
        self.assertIsNone(env["baselines"])
        self.assertEqual(len(self.displayed), 2)
        self.assertIn("valid saved gate", self.messages.getvalue())

    def test_invalid_encoder_table_does_not_prevent_saved_gain_display(self):
        env = self.halflife_fixture()
        path = self.folder / "encoder_probes.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("0.3", "nan"), encoding="utf-8")
        for cell_id in ("fc21afe3", "398d1d5f"):
            self.branch("tier3_halflife_te.ipynb", cell_id, env)
        self.assertIsNone(env["encoder_scores"])
        self.assertEqual(len(self.displayed), 1)


if __name__ == "__main__":
    unittest.main()
