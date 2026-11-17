"""Run BLF notebook reading flow only, with invented files and producers blocked.

These fixtures do not load research data, construct models, open token arrays,
score predictions or reproduce notebook results.
"""

from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import ast
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Preserve NumPy-before-PyTorch on Windows.
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import datasets, encoders
from shared_code.fusion import blf, codon_grid, heads, run_identity, textcnn, train
from shared_code.fusion import token_embeddings as tokens


class BLFReadingModesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".blf-modes-", dir=ROOT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.code = self.folder / "Code"
        self.results = self.code / "stage3/results/fungal_expression"
        self.results.mkdir(parents=True)
        self.cache = self.folder / "embeddings"
        self.audit = self.results / "records/verification.json"
        self.audit_sha = "missing-report"

    def snapshot(self):
        return {p.relative_to(self.folder): p.read_bytes()
                for p in self.folder.rglob("*") if p.is_file()}

    def run_reading(self, name):
        notebook = json.loads((ROOT / "Code/stage3" / f"{name}.ipynb").read_text(
            encoding="utf-8"))
        before = self.snapshot()
        blocked = Mock(side_effect=AssertionError("Producing entry point reached"))
        output, env = StringIO(), {}
        original_text, original_bytes = Path.read_text, Path.read_bytes

        def local_read(original):
            def read(path, *args, **kwargs):
                if not path.resolve().is_relative_to(self.folder.resolve()):
                    raise AssertionError(f"Read outside invented fixture: {path}")
                return original(path, *args, **kwargs)
            return read

        with ExitStack() as guards, redirect_stdout(output):
            guards.enter_context(patch("IPython.display.display", Mock()))
            guards.enter_context(patch.object(datasets, "DEFAULT_DATA_DIR", self.folder / "datasets"))
            guards.enter_context(patch.object(Path, "read_text", local_read(original_text)))
            guards.enter_context(patch.object(Path, "read_bytes", local_read(original_bytes)))
            for name_to_block in ("mkdir", "write_text", "write_bytes", "rename", "replace", "unlink"):
                guards.enter_context(patch.object(Path, name_to_block, blocked))
            for module, names in (
                (datasets, ("blf_benchmark_rows", "is_retained")),
                (encoders, ("load_encoder", "select_device")),
                (tokens, ("extract", "summarize", "token_states", "inputs_for_rows", "expected_tokens")),
                (blf.BenchmarkStreams, ("open",)),
                (blf, ("build", "loader")),
                (train, ("fit", "run", "score_rows", "paired_bootstrap", "seed_averaged_bootstrap")),
                (run_identity, ("execution_records",)),
                (torch.nn.Module, ("__init__",)),
                (torch, ("manual_seed", "randn")),
                (torch.optim, ("Adam",)),
                (np.random, ("default_rng",)),
                (np, ("load", "save", "savez", "savez_compressed")),
                (pd, ("read_csv",)),
                (pd.DataFrame, ("to_csv",)),
            ):
                for attr in names:
                    guards.enter_context(patch.object(module, attr, blocked))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source, f"fixture-{name}:{index}", "exec"), env)
                # Keep the notebook's actual default mode/settings; relocate only I/O roots.
                if "CODE_DIR" in env:
                    env["CODE_DIR"] = self.code
                if "CACHE_ROOT" in env:
                    env["CACHE_ROOT"] = self.cache
                if "RESULTS_DIR" in env:
                    env["RESULTS_DIR"] = self.results
                if "FUNGAL_VERIFICATION" in env:
                    env["FUNGAL_VERIFICATION"] = self.audit
                    env["FUNGAL_VERIFICATION_SHA256"] = self.audit_sha
        blocked.assert_not_called()
        self.assertEqual(before, self.snapshot())
        self.assertIs(env["READ_SAVED_RESULTS"], True)
        return env, output.getvalue()

    def test_dataset_missing_inputs_does_not_prepare_rows(self):
        env, output = self.run_reading("blf_datasets")
        self.assertTrue(all(row["status"] == "missing locally" for row in env["inventory"]))
        self.assertIn("no audit table is rebuilt", output)
        self.assertIn("PUBLISHED", env)
        self.assertNotIn("AUDIT", env)

    def test_dataset_inventory_is_metadata_only_and_preserves_file_bytes(self):
        path = self.folder / "datasets" / datasets.DATASETS[datasets.BLF_BENCHMARKS[0]].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"deliberately not a research CSV")
        env, _ = self.run_reading("blf_datasets")
        self.assertEqual(env["inventory"][0]["status"], "available locally")
        self.assertEqual(env["inventory"][0]["bytes"], path.stat().st_size)

    def test_token_missing_metadata_never_opens_arrays_or_models(self):
        env, output = self.run_reading("blf_token_embeddings")
        self.assertEqual(len(env["metadata_rows"]), 12)
        self.assertTrue(all("missing metadata" in row["status"] for row in env["metadata_rows"]))
        self.assertNotIn("ROWS", env)
        self.assertNotIn("RATE", env)
        self.assertIn("no arrays were opened", output)

    def test_token_inventory_reports_malformed_and_unvalidated_metadata_independently(self):
        for key, payload in (("ntv2_100m", '{"n_sequences": 2, "n_shards": 1}'),
                             ("rnafm", "[1, 2]"), ("esm2_8m", "{broken")):
            path = self.cache / "cov_vac_degradation" / key / "meta.json"
            path.parent.mkdir(parents=True)
            path.write_text(payload, encoding="utf-8")
        env, _ = self.run_reading("blf_token_embeddings")
        rows = env["metadata_rows"]
        self.assertEqual(rows[0]["status"], "metadata present; cache not validated")
        self.assertEqual(rows[0]["recorded sequences"], 2)
        self.assertTrue(all("unreadable metadata" in row["status"] for row in rows[1:3]))
        self.assertTrue(all("missing metadata" in row["status"] for row in rows[3:]))

    def save_result(self, variant="concat", *, seed=0, tconv="appendix", penalty=0.0):
        record = {
            "benchmark": "fungal_expression", "variant": variant,
            "config": asdict(train.TrainConfig(tconv=tconv, seed=seed, entropy_lambda=penalty)),
            "epochs": 4, "best_epoch": 3, "seconds": 1.0,
            "val_score": 0.5, "test_score": 0.4,
            "test_truth": [2.0, 4.0, 5.0, 6.0],
            "test_prediction": [2.1, 3.9, 5.1, 6.0],
        }
        path = self.results / f"{variant.replace(':', '-')}-{tconv}-{penalty}-{seed}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path, record

    def save_cohort_and_report(self):
        inputs = []
        variants = ("unimodal:dna", "concat", "mil", "cross")
        for variant in variants:
            for seed in (0, 1, 2):
                path, record = self.save_result(variant, seed=seed, tconv="equation",
                                               penalty=0.01 if variant == "mil" else 0.0)
                inputs.append({
                    "path": path.relative_to(self.folder).as_posix(), "variant": variant,
                    "config": record["config"], "status": "passed", "rows": 4,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        # Values deliberately invented; the reader must display rather than recalculate them.
        report = {"status": "passed_internal_consistency_only", "inputs": inputs,
                  "historical_method": {"seed_order": [0, 1, 2]},
                  "comparisons": [{"first": "concat", "second": "unimodal:dna",
                                   "difference": 0.123, "ci_low": -0.456, "ci_high": 0.789}]}
        self.audit.parent.mkdir(parents=True)
        self.audit.write_text(json.dumps(report), encoding="utf-8")
        self.audit_sha = hashlib.sha256(self.audit.read_bytes()).hexdigest()

    def test_build_missing_inputs_keeps_all_readers_available_without_producers(self):
        env, output = self.run_reading("blf_build_check")
        self.assertTrue(env["comparison"].empty)
        self.assertEqual(len(env["comparison_coverage"]), 7)
        self.assertEqual(len(env["spread"]), 4)
        self.assertIn("Saved paired comparison unavailable", output)
        self.assertNotIn("STREAMS", env)
        self.assertNotIn("probe", env)
        self.assertNotIn("timing", env)
        self.assertNotIn("interval", env)

    def test_build_partial_malformed_and_ambiguous_records_do_not_block_other_tables(self):
        self.save_result()
        path, _ = self.save_result("unimodal:dna")
        (self.results / "duplicate.json").write_bytes(path.read_bytes())
        (self.results / "malformed.json").write_text("{broken", encoding="utf-8")
        env, output = self.run_reading("blf_build_check")
        self.assertEqual(list(env["comparison"].index), ["concat"])
        self.assertIn("Ambiguous result withheld", output)
        self.assertIn("Saved paired comparison unavailable", output)
        self.assertEqual(len(env["settings"]), 4)

    def test_build_saved_report_is_displayed_without_scoring_or_resampling(self):
        self.save_cohort_and_report()
        env, output = self.run_reading("blf_build_check")
        self.assertEqual(env["saved_intervals"].iloc[0]["difference"], 0.123)
        self.assertEqual(env["saved_intervals"].iloc[0]["ci_low"], -0.456)
        self.assertIn("Saved historical paired report", output)
        self.assertTrue((env["spread"]["summary status"] == "ready").all())

    def test_build_changed_report_does_not_fall_back_to_bootstrap(self):
        self.save_cohort_and_report()
        self.audit.write_text(self.audit.read_text(encoding="utf-8") + " ", encoding="utf-8")
        env, output = self.run_reading("blf_build_check")
        self.assertNotIn("saved_intervals", env)
        self.assertIn("Saved paired comparison unavailable", output)
        self.assertFalse(env["settings"].empty)

    def test_build_changed_selected_result_withholds_report_and_keeps_score_table(self):
        self.save_cohort_and_report()
        path = next(self.results.glob("concat-equation*0.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["test_prediction"][0] += 0.1
        path.write_text(json.dumps(record), encoding="utf-8")
        env, output = self.run_reading("blf_build_check")
        self.assertNotIn("saved_intervals", env)
        self.assertIn("Saved paired comparison unavailable", output)
        self.assertFalse(env["settings"].empty)

    def test_producing_calls_and_stream_dependencies_have_explicit_mode_guards(self):
        producing = {
            "datasets.blf_benchmark_rows", "pd.read_csv", "encoders.select_device",
            "encoders.load_encoder", "tokens.extract", "tokens.summarize", "tokens.token_states",
            "cost_table", "time_encoder", "runtime_projection", "time_variant",
            "blf.BenchmarkStreams.open", "blf.build", "train.fit", "run_recorded",
            "bootstrap_contrast", "torch.randn", "codon_grid.CodonGrid",
            "textcnn.TextCNN", "heads.GatedMILFusion",
        }
        failures = []

        def inspect(node, guarded=False):
            # Defining a helper does not invoke it; whole-flow fixtures test its callers.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test)
                body_guard = condition in ("not READ_SAVED_RESULTS",
                                           "RUN_WIDE_HEAD and (not READ_SAVED_RESULTS)")
                inspect(node.test, guarded)
                for child in node.body:
                    inspect(child, guarded or body_guard)
                for child in node.orelse:
                    inspect(child, guarded or condition == "READ_SAVED_RESULTS")
                return
            if isinstance(node, ast.Call) and ast.unparse(node.func) in producing and not guarded:
                failures.append(ast.unparse(node.func))
            if isinstance(node, ast.Name) and node.id == "STREAMS" and not guarded:
                failures.append("unguarded STREAMS dependency")
            for child in ast.iter_child_nodes(node):
                inspect(child, guarded)

        for name in ("blf_datasets", "blf_token_embeddings", "blf_build_check"):
            notebook = json.loads((ROOT / "Code/stage3" / f"{name}.ipynb").read_text(
                encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    inspect(ast.parse("".join(cell["source"])))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
