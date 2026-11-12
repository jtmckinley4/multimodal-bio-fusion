"""Exercise encoder-swap reading cells against fixtures, with producers blocked."""

from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import swap_plan, swap_results


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5, entropy_lambda=0.0,
              channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="ntv2_100m", rna="rnafm", prot="esm2_8m")


def forbidden(*args, **kwargs):
    raise AssertionError("Read-saved mode attempted computation or publication")


class SwapNotebookReadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "Code"
        self.code.mkdir()
        self.cells = json.loads((ROOT / "Code/stage4/tier2_encoder_swaps.ipynb").read_text(
            encoding="utf-8"))["cells"]

    def result(self, relative, variant, **changes):
        path = self.code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(benchmark="cov_vac_degradation", variant=variant,
                    config={**RECIPE, "entropy_lambda": 0.01 if variant == "mil" else 0.0},
                    epochs=4, best_epoch=2, seconds=1.0, val_score=0.4, test_score=0.5)
        data.update(changes)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def execute_reading(self):
        labels = (*ORIGINAL.values(), "nt500m", "hyenadna", "rinalmo", "mrnafm",
                  "calm", "esm2_35m", "esm2_150m", "protbert")
        shown = []
        namespace = dict(
            CODE_DIR=self.code, DEVICE="cpu", Path=Path, np=np, pd=pd, json=json,
            datetime=datetime, time=time, display=shown.append,
            ENCODERS={key: SimpleNamespace(label=key) for key in labels},
            datasets=SimpleNamespace(blf_benchmark_rows=forbidden),
            blf=SimpleNamespace(ENCODER_OF_MODALITY=ORIGINAL,
                                BenchmarkStreams=SimpleNamespace(open=forbidden)),
            tokens=SimpleNamespace(DEFAULT_CHUNK=256, extract=forbidden,
                                   inspect_cache=forbidden, tokens_for_length=forbidden,
                                   CacheVerification=forbidden,
                                   grid_resolution=lambda k: "codon" if k in
                                   {"mrnafm", "calm", "esm2_8m", "esm2_35m",
                                    "esm2_150m", "protbert"} else "nucleotide"),
            train=SimpleNamespace(
                TrainConfig=lambda **kw: SimpleNamespace(**{**RECIPE, **kw}),
                Result=SimpleNamespace(load=lambda p: SimpleNamespace(
                    **json.loads(Path(p).read_text(encoding="utf-8")))),
                run=forbidden, fit=forbidden, seed_averaged_bootstrap=forbidden),
            swap_results=swap_results, swap_plan=swap_plan,
        )
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = io.StringIO()
        # Imports/device discovery are not research; inject their interfaces so this
        # test needs neither pretrained models nor a particular PyTorch installation.
        omitted = {"swaps-imports", "swaps-paths", "1a72c669"}
        with redirect_stdout(output), patch.object(pd.DataFrame, "to_csv", forbidden), \
                patch.object(Path, "mkdir", forbidden), patch.object(Path, "write_text", forbidden):
            for cell in self.cells:
                if cell["cell_type"] == "code" and cell["id"] not in omitted:
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
                    if cell["id"] == "swaps-settings":
                        # Test the reading branch even when the user last saved
                        # the notebook's settings for an authorized compute run.
                        namespace["READ_SAVED_RESULTS"] = True
        self.assertTrue(namespace["READ_SAVED_RESULTS"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertNotIn("STREAMS", namespace)
        self.assertEqual(namespace["RESOLVED"], {})
        return namespace, shown, output.getvalue()

    def test_empty_checkout_reports_gaps_and_finishes_without_producers(self):
        namespace, shown, output = self.execute_reading()
        self.assertIn("Training skipped", output)
        self.assertIn("Stage 1 alignment unavailable", output)
        self.assertIn("Paired intervals not recalculated", output)
        self.assertTrue(namespace["result_coverage"]["missing"].str.len().gt(0).all())

    def test_existing_scores_read_without_caches_and_without_recalculation(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "concat", "mil", "cross"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{variant.replace(':', '-')}.json",
                        variant)
        for variant in ("unimodal:dna", "unimodal-grid:dna", "concat", "mil", "cross"):
            self.result(f"stage4/results/phaseB/cov_vac_degradation/nt500m/{variant.replace(':', '-')}.json",
                        variant, test_score=0.7)
        namespace, shown, output = self.execute_reading()
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertAlmostEqual(table.loc[("dna", "nt500m"), "concat"], 0.7)
        self.assertIn("metadata unavailable", str(namespace["checks"]))

    def test_ambiguous_or_malformed_records_report_without_replacement(self):
        for name in ("first", "duplicate"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{name}.json", "mil")
        path = self.result("stage4/results/phaseB/cov_vac_degradation/nt500m/broken.json", "concat")
        path.write_text("not JSON", encoding="utf-8")
        namespace, shown, output = self.execute_reading()
        self.assertIn("seed-0 MIL selection unavailable", str(namespace["status"]))
        self.assertIn("Ambiguous saved results", str(namespace["FIT_PLANS"]["cov_vac_degradation"].issues))
        self.assertFalse(namespace["result_issues"].empty)

    def test_missing_mil_selection_does_not_hide_independent_scores(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "concat"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{variant.replace(':', '-')}.json",
                        variant)
        self.result("stage4/results/phaseB/cov_vac_degradation/nt500m/concat.json", "concat",
                    test_score=0.7)
        namespace, _, _ = self.execute_reading()
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.5)
        self.assertEqual(table.loc[("dna", "nt500m"), "concat"], 0.7)
        self.assertTrue(np.isnan(table.loc[("dna", "nt500m"), "mil"]))

    def test_duplicated_variant_does_not_hide_unique_scores(self):
        for name in ("first", "duplicate"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{name}.json", "mil")
        self.result("stage4/results/phaseA/cov_vac_degradation/concat.json", "concat")
        namespace, _, _ = self.execute_reading()
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.5)
        self.assertTrue(np.isnan(table.loc[("none", "BioLangFusion's own"), "mil"]))

    def test_cache_metadata_reports_valid_invalid_and_missing_without_recovery(self):
        benchmark = "cov_vac_degradation"
        metadata = {
            "ntv2_100m": '{"n_shards": 13}',
            "rnafm": "[]",
            "esm2_8m": "not JSON",
        }
        for encoder_key, contents in metadata.items():
            path = self.code / "stage3/embeddings" / benchmark / encoder_key / "meta.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")

        # The existing reading harness blocks recovery, producers and writers and
        # checks every fixture byte, even when cache metadata is unusable.
        namespace, _, _ = self.execute_reading()
        rows = {
            row["encoder"]: row for row in namespace["checks"]
            if row["benchmark"] == benchmark
        }
        self.assertEqual(rows["ntv2_100m"], {
            "benchmark": benchmark,
            "encoder": "ntv2_100m",
            "status": "saved metadata; shards not checked",
            "recorded shards": 13,
            "issues": "",
        })
        for encoder_key in ("rnafm", "esm2_8m", "nt500m"):
            with self.subTest(encoder=encoder_key):
                self.assertEqual(rows[encoder_key]["status"], "metadata unavailable")
                self.assertIsNone(rows[encoder_key]["recorded shards"])
                self.assertTrue(rows[encoder_key]["issues"])
        self.assertIn("metadata is not an object", rows["rnafm"]["issues"])
        self.assertIn("meta.json", rows["nt500m"]["issues"])

    def test_coverage_keeps_requested_seed_order_prerequisites_and_reference_gaps(self):
        benchmark = "cov_vac_degradation"
        self.result(f"stage4/results/phaseA/{benchmark}/concat.json", "concat")
        self.result(f"stage4/results/phaseB/{benchmark}/nt500m/concat.json", "concat")
        broken = self.result(f"stage4/results/phaseB/{benchmark}/nt500m/broken.json", "cross")
        broken.write_text("not JSON", encoding="utf-8")
        namespace, _, _ = self.execute_reading()
        namespace["swaps_for"] = lambda key: [("dna", "nt500m")]
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(Path, "mkdir", forbidden), patch.object(Path, "write_text", forbidden):
            coverage, issues = namespace["coverage_for_benchmark"](benchmark, (2, 0))

        self.assertEqual([(row["encoder"], row["seed"]) for row in coverage],
                         [("original", 2), ("nt500m", 2), ("original", 0), ("nt500m", 0)])
        self.assertTrue(all("seed-0 MIL selection unavailable" in row["prerequisite"]
                            for row in coverage))
        self.assertEqual(coverage[1]["available"], "")
        self.assertEqual(coverage[3]["available"], "concat")
        self.assertEqual(coverage[3]["without paired arrays"], "concat")
        self.assertEqual(coverage[3]["reference gaps"],
                         "unimodal:rna, unimodal:prot, unimodal-grid:rna, mil, cross")
        self.assertEqual(coverage[3]["reference arrays absent"], "concat")
        broken_issues = [row for row in issues if row["source"] == str(broken.resolve())]
        self.assertEqual([(row["benchmark"], row["encoder"], row["seed"])
                          for row in broken_issues],
                         [(benchmark, "nt500m", 2), (benchmark, "nt500m", 0)])
        self.assertEqual(namespace["SEED"], 0)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*")
                                  if path.is_file()})

    def test_bootstrap_plan_uses_selection_seed_and_requires_both_fusion_heads(self):
        namespace, _, _ = self.execute_reading()
        original = {"concat": SimpleNamespace(), "mil": SimpleNamespace()}
        current = {
            "unimodal:dna": SimpleNamespace(val_score=0.95),
            "unimodal:rna": SimpleNamespace(val_score=0.1),
            "unimodal:prot": SimpleNamespace(val_score=0.4),
            "concat": SimpleNamespace(),
            "cross": SimpleNamespace(),
        }
        selection = {
            "unimodal:dna": SimpleNamespace(val_score=0.2),
            "unimodal:rna": SimpleNamespace(val_score=0.8),
            "unimodal:prot": SimpleNamespace(val_score=0.4),
        }
        requested = []

        def result_set(benchmark, modality=None, encoder_key=None, seed=None):
            requested.append((benchmark, modality, encoder_key, seed))
            if encoder_key is None:
                return original
            return selection if seed == 0 else current

        namespace.update(SEED=2, SELECTION_SEED=0, results_for_encoder_set=result_set,
                         swaps_for=lambda key: [("rna", "mrnafm")])
        records, comparisons = namespace["bootstrap_comparison_plan"]("fixture")
        self.assertEqual(requested, [
            ("fixture", None, None, None),
            ("fixture", "rna", "mrnafm", None),
            ("fixture", "rna", "mrnafm", 0),
        ])
        self.assertEqual(comparisons, [
            ("mrnafm concat vs original", "mrnafm concat", "original concat"),
            ("mrnafm concat vs best single (unimodal:rna)",
             "mrnafm concat", "mrnafm unimodal:rna"),
        ])
        self.assertIs(records["mrnafm unimodal:rna"], current["unimodal:rna"])
        # Incomplete selection evidence cannot silently choose a different single.
        del selection["unimodal:rna"]
        _, incomplete = namespace["bootstrap_comparison_plan"]("fixture")
        self.assertEqual(incomplete, comparisons[:1])

    def test_alignment_uses_explicit_capture_labels_and_preserves_raw_values(self):
        namespace, _, _ = self.execute_reading()
        index = ["mrnafm", "esm2_8m", "nt500m", "ntv2_100m", "rnafm"]
        columns = ["ntv2_100m", "rnafm", "esm2_8m", "nt500m", "mrnafm"]
        values = [
            [0.51, 0.52, 0.53, 0.54, 1.0],
            [0.13, 0.23, 1.0, 0.43, 0.53],
            [0.41, 0.42004, 0.43, 1.0, 0.54],
            [1.0, 0.12, 0.13, 0.41, 0.51],
            [0.12, 1.0, 0.23, 0.42004, 0.52],
        ]
        capture = {"similarity": {
            "Linear CKA": {"index": index, "columns": columns, "data": values},
            "Linear CKA after control": {
                "index": index, "columns": columns, "data": (np.array(values) / 2).tolist(),
            },
        }}
        before = json.dumps(capture)
        namespace["stage1"] = {"similarity": {}}  # The explicit argument owns this lookup.
        matrix = namespace["alignment_matrix"](capture, "Linear CKA")
        self.assertEqual(matrix.index.tolist(), index)
        self.assertEqual(matrix.columns.tolist(), columns)
        actual = namespace["alignment_table"](capture, {("rna", "mrnafm"), ("dna", "nt500m")})
        expected = pd.DataFrame({
            "swap": ["nt500m"] * 3 + ["mrnafm"] * 3,
            "pair": ["nt500m + rnafm", "nt500m + esm2_8m",
                     "nt500m vs the ntv2_100m it replaces",
                     "mrnafm + ntv2_100m", "mrnafm + esm2_8m",
                     "mrnafm vs the rnafm it replaces"],
            "CKA": [0.42004, 0.43, 0.41, 0.51, 0.53, 0.52],
            "CKA after composition control": [0.21002, 0.215, 0.205, 0.255, 0.265, 0.26],
        })
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertEqual(json.dumps(capture), before)


if __name__ == "__main__":
    unittest.main()
