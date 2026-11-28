"""Exercise encoder-swap reading cells against fixtures, with producers blocked."""

import ast
from contextlib import ExitStack, redirect_stdout
import datetime
import hashlib
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
from shared_code.fusion import result_sets, swap_plan, swap_reporting, swap_results


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
        self.cells = json.loads((ROOT / "Code/stage4/encoder_swaps.ipynb").read_text(
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

    def saved_report(self, directory, source, *, selection=None):
        benchmark = "cov_vac_degradation"
        identity = {
            "schema": swap_reporting.SCHEMA, "benchmark": benchmark,
            "seeds": [0, 1, 2], "selection_seed": 0, "config": RECIPE,
            "draws": 2000, "bootstrap_seed": 0,
            "sources": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
            "source_sha256": {"synthetic_producer.py": "a" * 64},
        }
        if selection is not None:
            identity.update(
                selection_policy="fixed-historical-seed0-validation",
                selection_sources={str(selection): hashlib.sha256(
                    selection.read_bytes()).hexdigest()},
            )
        contrast = dict(encoder="nt500m", kind="replacement", comparison="synthetic",
                        first="nt500m/concat", second="original/concat")
        analysis = {
            "per_seed": [{**contrast, "seed": seed, "difference": 0.2}
                         for seed in (0, 1, 2)],
            "seed_summary": [{**contrast, "available_seeds": [0, 1, 2],
                              "missing_seeds": [], "status": "complete", "difference": 0.2}],
            "issues": [], "units": "synthetic score difference",
        }
        saved = {"identity": identity, "analysis": analysis,
                 "analysis_sha256": swap_reporting._digest(analysis),
                 "producer": {"host": "synthetic-fixture"}}
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{benchmark}__{swap_reporting._digest(identity)[:20]}.json"
        path.write_text(json.dumps(saved), encoding="utf-8")
        return saved

    def execute_reading(self, result_set=None):
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
        with ExitStack() as blocked, redirect_stdout(output):
            if hasattr(self, "snapshot_sha256"):
                blocked.enter_context(patch.object(result_sets, "NEWTON_SNAPSHOT_SHA256", self.snapshot_sha256))
            blocked.enter_context(patch.object(pd.DataFrame, "to_csv", forbidden))
            for name in ("mkdir", "write_text", "write_bytes"):
                blocked.enter_context(patch.object(Path, name, forbidden))
            for name in ("report", "calculate", "benchmark_context", "collect", "_score"):
                blocked.enter_context(patch.object(swap_reporting, name, forbidden))
            for cell in self.cells:
                if cell["cell_type"] == "code" and cell["id"] not in omitted:
                    source = "".join(cell["source"])
                    if cell["id"] == "swaps-locations" and result_set is not None:
                        # Override this assignment regardless of the saved default;
                        # None exercises the actual notebook settings unchanged.
                        source = ast.parse(source)
                        assignments = [node for node in source.body
                                       if isinstance(node, ast.Assign) and any(
                                           isinstance(target, ast.Name) and target.id == "RESULT_SET"
                                           for target in node.targets)]
                        self.assertEqual(len(assignments), 1)
                        assignments[0].value = ast.Constant(result_set)
                        ast.fix_missing_locations(source)
                    exec(compile(source, cell["id"], "exec"), namespace)
        self.assertTrue(namespace["READ_SAVED_RESULTS"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertNotIn("STREAMS", namespace)
        return namespace, shown, output.getvalue()

    def test_empty_checkout_reports_gaps_and_finishes_without_producers(self):
        namespace, shown, output = self.execute_reading("historical")
        self.assertIn("Training skipped", output)
        self.assertTrue(namespace["cache_summary"]["encoder metadata found"].eq("0/11").all())
        self.assertTrue(namespace["result_coverage"]["missing"].str.len().gt(0).all())

    def test_saved_default_does_not_replace_absent_newton_with_desktop_scores_or_reports(self):
        source = self.result("stage4/results/phaseA/cov_vac_degradation/desktop.json", "concat")
        self.saved_report(self.code / "stage4/results/phaseB/analyses/paired-report-v1", source)
        namespace, _, output = self.execute_reading()
        self.assertEqual(namespace["SAVED_COMPARISON_REPORTS"], {})
        self.assertFalse(namespace["select_original_fit_records"]("cov_vac_degradation", 0).records)
        self.assertIn("no matching saved report", output)
        self.assertIn("newton_h100_seeds012", output)
        self.assertTrue(namespace["result_coverage"]["missing"].str.len().gt(0).all())

    def test_saved_default_reads_newton_scores_and_reports_without_csv_copies_or_caches(self):
        root = "stage4/results/newton_h100_seeds012"
        source = self.result(f"{root}/outputs/phaseA/cov_vac_degradation/concat.json",
                             "concat", test_score=0.65)
        relative = "stage4/results/phaseA/cov_vac_degradation/concat.json"
        selection = self.result(relative, "concat", test_score=0.2)
        snapshot = self.code / root / "provenance/source-snapshot.json"
        snapshot.parent.mkdir(parents=True)
        digest = hashlib.sha256(selection.read_bytes()).hexdigest()
        snapshot.write_text(json.dumps({"schema_version": 1, "files": [{
            "source": "Code/" + relative,
            "destination": "historical_results/Code/" + relative,
            "source_sha256": digest, "initial_copy_sha256": digest,
        }]}), encoding="utf-8")
        self.snapshot_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        self.result("stage4/results/phaseA/cov_vac_degradation/desktop.json",
                    "concat", test_score=0.99)
        saved = self.saved_report(
            self.code / root / "outputs/phaseB/analyses/paired-report-v1", source,
            selection=selection)

        namespace, _, output = self.execute_reading()

        self.assertEqual(namespace["RESULT_SET"], "newton_h100_seeds012")
        self.assertTrue(namespace["RESULT_PATHS"].fixed_historical_selection)
        self.assertEqual((namespace["SEED"], namespace["SELECTION_SEED"]), (0, 0))
        self.assertEqual(namespace["COMPARISON_SEEDS"], (0, 1, 2))
        self.assertFalse(namespace["RUN_SAVED_COMPARISONS"])
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.65)
        self.assertEqual(namespace["SAVED_COMPARISON_REPORTS"], {"cov_vac_degradation": saved})
        self.assertIn("cov_vac_degradation: read", output)
        self.assertFalse(any(self.root.rglob("*.csv")))
        self.assertFalse(namespace["CACHE_ROOT"].exists())

    def test_existing_scores_read_without_caches_and_without_recalculation(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "concat", "mil", "cross"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{variant.replace(':', '-')}.json",
                        variant)
        for variant in ("unimodal:dna", "unimodal-grid:dna", "concat", "mil", "cross"):
            self.result(f"stage4/results/phaseB/cov_vac_degradation/nt500m/{variant.replace(':', '-')}.json",
                        variant, test_score=0.7)
        namespace, shown, output = self.execute_reading("historical")
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertAlmostEqual(table.loc[("dna", "nt500m"), "concat"], 0.7)
        self.assertIn("metadata unavailable", str(namespace["checks"]))

    def test_ambiguous_or_malformed_records_report_without_replacement(self):
        for name in ("first", "duplicate"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{name}.json", "mil")
        path = self.result("stage4/results/phaseB/cov_vac_degradation/nt500m/broken.json", "concat")
        path.write_text("not JSON", encoding="utf-8")
        namespace, shown, output = self.execute_reading("historical")
        prerequisites = namespace["result_coverage"]["prerequisite"]
        self.assertTrue(prerequisites.str.contains("seed-0 MIL entropy penalty unavailable").any())
        self.assertIn("Ambiguous saved results", str(namespace["FIT_PLANS"]["cov_vac_degradation"].issues))
        self.assertFalse(namespace["result_issues"].empty)

    def test_missing_mil_selection_does_not_hide_independent_scores(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "concat"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{variant.replace(':', '-')}.json",
                        variant)
        self.result("stage4/results/phaseB/cov_vac_degradation/nt500m/concat.json", "concat",
                    test_score=0.7)
        namespace, _, _ = self.execute_reading("historical")
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.5)
        self.assertEqual(table.loc[("dna", "nt500m"), "concat"], 0.7)
        self.assertTrue(np.isnan(table.loc[("dna", "nt500m"), "mil"]))

    def test_duplicated_variant_does_not_hide_unique_scores(self):
        for name in ("first", "duplicate"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{name}.json", "mil")
        self.result("stage4/results/phaseA/cov_vac_degradation/concat.json", "concat")
        namespace, _, _ = self.execute_reading("historical")
        table = namespace["benchmark_score_table"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.5)
        self.assertTrue(np.isnan(table.loc[("none", "BioLangFusion's own"), "mil"]))

    def test_cache_metadata_reports_valid_invalid_and_missing_without_recovery(self):
        benchmark = "cov_vac_degradation"
        metadata = {
            "ntv2_100m": '{"n_shards": 1013, "n_sequences": 12345}',
            "rnafm": "[]",
            "esm2_8m": "not JSON",
        }
        for encoder_key, contents in metadata.items():
            path = self.code / "stage3/embeddings" / benchmark / encoder_key / "meta.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")

        # The existing reading harness blocks recovery, producers and writers and
        # checks every fixture byte, even when cache metadata is unusable.
        namespace, shown, _ = self.execute_reading("historical")
        rows = {
            row["encoder"]: row for row in namespace["checks"]
            if row["benchmark"] == benchmark
        }
        self.assertEqual(rows["ntv2_100m"], {
            "benchmark": benchmark,
            "encoder": "ntv2_100m",
            "status": "cache metadata only",
            "recorded input rows": 12345,
            "recorded cache shards": 1013,
            "issues": "",
        })
        for encoder_key in ("rnafm", "esm2_8m", "nt500m"):
            with self.subTest(encoder=encoder_key):
                self.assertEqual(rows[encoder_key]["status"], "metadata unavailable")
                self.assertIsNone(rows[encoder_key]["recorded input rows"])
                self.assertIsNone(rows[encoder_key]["recorded cache shards"])
                self.assertTrue(rows[encoder_key]["issues"])
        self.assertIn("metadata is not an object", rows["rnafm"]["issues"])
        self.assertIn("meta.json", rows["nt500m"]["issues"])

        summary = namespace["cache_summary"]
        self.assertEqual(summary.index.name, "benchmark")
        self.assertEqual(summary.columns.tolist(), [
            "encoder metadata found", "recorded input rows", "recorded shards per encoder",
        ])
        self.assertEqual(summary.loc[benchmark].tolist(), ["1/11", "12,345", "1,013"])
        self.assertEqual(summary.loc["fungal_expression"].tolist(),
                         ["0/11", "unavailable", "unavailable"])
        self.assertTrue(any(summary is item for item in shown))

    def test_coverage_keeps_requested_seed_order_prerequisites_and_reference_gaps(self):
        benchmark = "cov_vac_degradation"
        self.result(f"stage4/results/phaseA/{benchmark}/concat.json", "concat")
        self.result(f"stage4/results/phaseB/{benchmark}/nt500m/concat.json", "concat")
        broken = self.result(f"stage4/results/phaseB/{benchmark}/nt500m/broken.json", "cross")
        broken.write_text("not JSON", encoding="utf-8")
        namespace, _, _ = self.execute_reading("historical")
        namespace["SWAPS"] = [("dna", "nt500m")]
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(Path, "mkdir", forbidden), patch.object(Path, "write_text", forbidden):
            coverage, issues = namespace["coverage_for_benchmark"](benchmark, (2, 0))

        self.assertEqual([(row["encoder"], row["seed"]) for row in coverage],
                         [("original", 2), ("nt500m", 2), ("original", 0), ("nt500m", 0)])
        self.assertTrue(all("seed-0 MIL entropy penalty unavailable" in row["prerequisite"]
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

    def test_coverage_display_counts_fits_and_only_shows_diagnostic_rows(self):
        source = next("".join(cell["source"]) for cell in self.cells
                      if cell["id"] == "swaps-result-coverage")
        benchmark = "cov_vac_degradation"
        healthy = {"benchmark": benchmark, "encoder": "original", "seed": 0,
                   "available": "concat, cross", "missing": "", "without paired arrays": "",
                   "ambiguous": "", "excluded files": 1, "legacy identity": 0,
                   "reference gaps": "", "reference arrays absent": "", "prerequisite": ""}
        flagged = [
            dict(healthy, encoder=column, **{column: value})
            for column, value in (
                ("missing", "mil"),
                ("without paired arrays", "cross"),
                ("ambiguous", "concat"),
                ("legacy identity", 1),
                ("reference gaps", "unimodal:rna"),
                ("reference arrays absent", "unimodal:prot"),
                ("prerequisite", "seed-0 MIL entropy penalty unavailable"),
            )
        ]
        issues = [{"benchmark": benchmark, "encoder": "original", "seed": 0,
                   "source": "other-seed.json", "reason": "configuration differs: seed"}]

        cases = (
            ("encoder_swaps", (2, 0), [healthy], [2, 2, 2]),
            ("encoder_swaps", (2, 0), [healthy, *flagged], [17, 16, 15]),
            ("ecoli_ernie", (1,), [dict(healthy, seed=1)], [2, 2, 2]),
        )
        for experiment, expected_seeds, rows, counts in cases:
            with self.subTest(experiment=experiment, coverage_rows=len(rows)):
                shown = []
                requests = []

                def collect_coverage(requested_benchmark, seeds):
                    requests.append((requested_benchmark, seeds))
                    return rows, issues

                env = dict(
                    BENCHMARKS=(benchmark,), EXPERIMENT=experiment, SEED=1,
                    COMPARISON_SEEDS=(2, 0), READ_SAVED_RESULTS=True,
                    FIT_PLANS={benchmark: SimpleNamespace(lambda_=0.01)}, pd=pd,
                    display=shown.append, coverage_for_benchmark=collect_coverage,
                )
                with redirect_stdout(io.StringIO()):
                    exec(compile(source, "swaps-result-coverage", "exec"), env)
                self.assertEqual(requests, [(benchmark, expected_seeds)])
                self.assertEqual(env["SEED"], 1)
                pd.testing.assert_frame_equal(
                    env["result_coverage"], pd.DataFrame(rows).set_index(["benchmark", "encoder", "seed"]),
                )
                pd.testing.assert_frame_equal(env["result_issues"], pd.DataFrame(issues))
                expected = pd.DataFrame(
                    [{"benchmark": benchmark, "expected fits": counts[0], "saved fits": counts[1],
                      "fits with prediction and target arrays": counts[2], "MIL entropy penalty": 0.01}],
                ).set_index("benchmark")
                pd.testing.assert_frame_equal(env["coverage_summary"], expected)
                self.assertEqual(len(shown), 1 if len(rows) == 1 else 2)
                pd.testing.assert_frame_equal(shown[0], expected)
                if len(shown) > 1:
                    self.assertEqual(set(shown[1].index.get_level_values("encoder")),
                                     {row["encoder"] for row in flagged})

    def test_execution_plan_displays_details_only_when_fits_need_attention(self):
        source = next("".join(cell["source"]) for cell in self.cells
                      if cell["id"] == "swaps-execution-plan")
        benchmark = "cov_vac_degradation"
        ready = dict(benchmark=benchmark, role="original", encoder="original",
                     variant="concat", seed=0, status="reuse")
        pending = dict(ready, role="replacement", encoder="mrnafm", status="new")
        run = SimpleNamespace(benchmark=benchmark, status="new", encoders={"rna": "mrnafm"})
        for rows, runs in (([ready], []), ([ready, pending], [run])):
            with self.subTest(has_pending=bool(runs)):
                shown = []
                plan = SimpleNamespace(rows=lambda: rows, runs=runs, issues=[], require_ready=forbidden)
                env = dict(BENCHMARKS=(benchmark,), SEED=0, SELECTION_SEED=0, READ_SAVED_RESULTS=True,
                           pd=pd, display=shown.append, execution_plan=lambda _: plan)
                with redirect_stdout(io.StringIO()):
                    exec(compile(source, "swaps-execution-plan", "exec"), env)
                self.assertEqual(len(shown), 1 + bool(runs))
                self.assertEqual(env["PENDING_RUNS"], runs)
                self.assertEqual(env["PENDING_CACHE_KEYS"], [(benchmark, "mrnafm")] if runs else [])
                if runs:
                    self.assertEqual(shown[1]["status"].tolist(), ["new"])

if __name__ == "__main__":
    unittest.main()
