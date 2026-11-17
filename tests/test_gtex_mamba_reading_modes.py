"""Execute default notebook control flow against only invented local records.

Every code cell runs in notebook order in a temporary repository layout. Research
loaders, model construction, fitting, metrics, resampling, downloads and filesystem
publication are blocked. This checks reading branches, not research execution or
scientific correctness. Audit fixtures retain its deliberate fail-closed reader.
"""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import builtins
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# NumPy must precede torch on the supported Windows environment.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import IPython.display
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import datasets, encoders
from shared_code.fusion import (
    blf, coupled_mamba, heads, isoformer, isoformer_cohort,
    isoformer_data_checks, isoformer_results, run_identity, swap_results,
    token_embeddings, train,
)
from test_isoformer_cohort import ProteinTokenizer, RNATokenizer


NOTEBOOKS = {
    name: json.loads((ROOT / "Code/stage3" / f"{name}.ipynb").read_text(encoding="utf-8"))
    for name in ("gtex_frozen_fusion_reference", "coupled_mamba_reference", "isoformer_gtex_audit")
}


class DefaultReadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="b3-reading-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.code = self.root / "Code"
        self.cwd = self.code / "stage3"
        self.cwd.mkdir(parents=True)
        (self.code / "shared_code").mkdir()
        self.producer = Mock(side_effect=AssertionError("Producer reached in default reading"))
        self.displays = []

    def save(self, relative, value):
        path = self.code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def inventory(self):
        return {p.relative_to(self.root): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def execute(self, name, *, configure=None):
        """Run the unmodified cell sources; only fixture identities may be injected."""
        env, output = {}, StringIO()
        original_cwd, original_path = Path.cwd(), list(sys.path)
        original_open, original_path_open = builtins.open, Path.open
        before = self.inventory()
        directories = {p.relative_to(self.root) for p in self.root.rglob("*") if p.is_dir()}

        def readonly_open(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                raise AssertionError(f"Publication attempted: {path}")
            return original_open(path, mode, *args, **kwargs)

        def readonly_path_open(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                raise AssertionError(f"Publication attempted: {path}")
            return original_path_open(path, mode, *args, **kwargs)

        blocked = (
            (datasets, "load_dataset"), (encoders, "load_encoder"),
            (encoders, "load_tokenizer"), (blf.BenchmarkStreams, "open"),
            (isoformer.GTExStreams, "open"), (isoformer, "loaders_for"),
            (isoformer, "builder"), (token_embeddings, "extract"),
            (token_embeddings, "summarize"), (token_embeddings, "open_shard"),
            (run_identity, "execution_records"), (isoformer_cohort, "audit_csv"),
            (isoformer_cohort, "save_audit"), (train, "run"), (train, "fit"),
            (train, "score"), (train, "r_squared"), (train, "paired_bootstrap"),
            (train, "seed_averaged_bootstrap"), (torch.nn.Module, "__init__"),
            (pd, "read_csv"), (np, "load"), (Path, "mkdir"), (Figure, "savefig"),
        )
        try:
            os.chdir(self.cwd)
            with ExitStack() as stack, redirect_stdout(output):
                for module, attribute in blocked:
                    stack.enter_context(patch.object(module, attribute, self.producer))
                stack.enter_context(patch("urllib.request.urlopen", self.producer))
                stack.enter_context(patch("socket.create_connection", self.producer))
                stack.enter_context(patch("builtins.open", readonly_open))
                stack.enter_context(patch.object(Path, "open", readonly_path_open))
                stack.enter_context(patch.object(IPython.display, "display", self.displays.append))
                stack.enter_context(patch.object(plt, "show", lambda: plt.close("all")))
                # Keep the audit's already-loaded helpers instrumented across its reload calls.
                stack.enter_context(patch.object(importlib, "reload", lambda module: module))
                for cell in NOTEBOOKS[name]["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    exec(compile("".join(cell["source"]), f"fixture:{name}:{cell['id']}", "exec"), env)
                    if configure is not None:
                        configure(cell, env)
        finally:
            os.chdir(original_cwd)
            sys.path[:] = original_path
            plt.close("all")
            self.assertEqual(self.inventory(), before)
            self.assertEqual({p.relative_to(self.root) for p in self.root.rglob("*") if p.is_dir()},
                             directories)
            self.producer.assert_not_called()
        return env, output.getvalue()

    def gtex_record(self, variant, seed, *, repeat=False, score=0.25):
        parent = "stage4/results/gtex_isoformer" if repeat else "stage3/results/gtex_pilot"
        return self.save(
            f"{parent}/gtex_pilot__{variant}__heads8__budget300M__seed{seed}.json",
            {"benchmark": "gtex_pilot", "variant": variant, "epochs": 3,
             "test_score": score,
             "config": vars(train.TrainConfig(batch_size=16, max_epochs=500, seed=seed))},
        )

    def test_gtex_empty_default_runs_all_cells_without_inputs_or_directories(self):
        env, output = self.execute("gtex_frozen_fusion_reference")
        self.assertTrue(env["READ_SAVED_RESULTS"])
        self.assertFalse(any(env[name] for name in (
            "RUN_PILOT", "RUN_ISOFORMER_SEEDS", "RUN_SAVED_ANALYSIS")))
        self.assertNotIn("gtex", env)
        self.assertNotIn("STREAMS", env)
        self.assertTrue(env["comparison"].empty)
        self.assertIn("No compatible saved pilot scores", output)
        self.assertIn("No compatible IsoFormer runs found", output)

    def test_gtex_complete_scores_are_read_without_predictions_or_new_statistics(self):
        for index, variant in enumerate(isoformer_results.VARIANTS):
            self.gtex_record(variant, 0, score=(index + 1) / 10)
            for seed in (1, 2):
                self.gtex_record(variant, seed, repeat=True, score=(index + 1) / 10)
        # Seed aggregation would be a new calculation even without calling train metrics.
        with patch.object(pd.core.groupby.generic.SeriesGroupBy, "agg", self.producer):
            env, output = self.execute("gtex_frozen_fusion_reference")
        self.assertEqual(len(env["pilot_cohort"].records), 6)
        self.assertEqual(len(env["isoformer_cohort"].records), 18)
        self.assertTrue(env["comparison"].empty)
        self.assertNotIn("ordering_checks", env)
        self.assertIn("aggregate report unavailable", output)
        self.assertTrue(all("result" not in row for row in env["pilot_cohort"].records))

    def test_gtex_malformed_pilot_does_not_block_independent_repeat_score(self):
        path = self.gtex_record("dna", 0)
        path.write_text("{broken", encoding="utf-8")
        self.gtex_record("rna", 1, repeat=True, score=0.37)
        env, _ = self.execute("gtex_frozen_fusion_reference")
        self.assertFalse(env["pilot_cohort"].records)
        self.assertTrue(env["pilot_cohort"].issues)
        self.assertEqual([(row["variant"], row["seed"], row["Spearman"])
                          for row in env["isoformer_cohort"].records], [("rna", 1, 0.37)])

    def test_gtex_duplicate_key_withholds_ambiguous_scores_without_repair(self):
        self.gtex_record("rna", 0)
        self.gtex_record("rna", 0, repeat=True, score=0.9)
        self.gtex_record("dna", 1, repeat=True)
        env, output = self.execute("gtex_frozen_fusion_reference")
        self.assertTrue(env["isoformer_cohort"].duplicate_keys)
        self.assertIn("summary skipped", output)
        self.assertEqual([(r["variant"], r["seed"])
                          for r in env["isoformer_cohort"].records], [("dna", 1)])

    def test_mamba_empty_default_runs_all_cells_without_bootstrap(self):
        env, output = self.execute("coupled_mamba_reference")
        self.assertTrue(env["READ_SAVED_RESULTS"])
        self.assertFalse(env["RUN_BOOTSTRAP"])
        self.assertEqual(env["rows"], [])
        self.assertEqual(len(env["coverage_rows"]), 4)
        self.assertIn("no standalone bootstrap report is saved", output)

    def test_mamba_saved_pairs_and_malformed_neighbor_never_trigger_bootstrap(self):
        for variant in ("concat", "coupled_mamba"):
            directory = ("phaseA" if variant == "concat" else "tier4_designs")
            self.save(f"stage4/results/{directory}/cov_vac_degradation/{variant}.json", {
                "benchmark": "cov_vac_degradation", "variant": variant,
                "config": vars(train.TrainConfig(channels=100, tconv="equation", seed=0)),
                "epochs": 3, "best_epoch": 2, "seconds": 1.0,
                "val_score": 0.3, "test_score": 0.4,
                "test_truth": [1.0, 2.0], "test_prediction": [1.1, 1.9],
            })
        self.save("stage4/results/tier4_designs/cov_vac_degradation/broken.json", {})
        env, output = self.execute("coupled_mamba_reference")
        self.assertEqual(env["runs"]["cov_vac_degradation"]["concat"]["test_score"], 0.4)
        self.assertEqual(env["runs"]["cov_vac_degradation"]["coupled_mamba"]["test_score"], 0.4)
        self.assertTrue(env["reader_issues"])
        self.assertEqual(env["rows"], [])
        self.assertIn("report unavailable", output)

    def audit_fixture(self):
        source = self.root / "invented.csv"
        source.write_text("transcript_id_gtex,split,RNA,Protein,chr,start,end,strand\n"
                          "fixture1,train,ACGTAC,M,chr1,300001,300010,+\n"
                          "fixture2,test,N,,chr1,400001,400010,+\n", encoding="utf-8")
        report = isoformer_cohort.audit_csv(
            source, RNATokenizer(), ProteinTokenizer(), chromosome_sizes={"chr1": 1_000_000})
        report.update({
            "started_utc": "2026-01-01T00:00:00+00:00",
            "finished_utc": "2026-01-01T00:00:01+00:00",
            "dataset_revision": "a" * 40, "model_revision": "b" * 40,
            "vocabulary_sha256": {"rna": "c" * 64, "protein": "d" * 64},
            "chromosome_sizes_sha256": "e" * 64,
            "transformers": "fixture", "host": "synthetic-host",
        })
        report["audit_identity"] = {
            key: report[key] for key in (
                "source_sha256", "source_bytes", "dataset_revision", "model_revision",
                "vocabulary_sha256", "chromosome_sizes_sha256", "transformers",
            )
        }
        report["audit_identity"]["method_version"] = isoformer_cohort.METHOD_VERSION
        return report

    def configure_audit_identity(self, report):
        def configure(cell, env):
            if cell["id"] == "source-identity":
                identity = {key: value for key, value in report["audit_identity"].items()
                            if key != "transformers"}
                env.update(SOURCE_IDENTITY=identity, PREVIOUS_IDENTITY=identity,
                           LEGACY_IDENTITY={key: value for key, value in identity.items()
                                            if key != "method_version"})
        return configure

    def test_audit_valid_saved_record_completes_without_csv_or_measurement(self):
        report = self.audit_fixture()
        path = self.save("stage4/results/isoformer_cohort_audit/gtex_audit_v2.json", report)
        # Make its CSV unavailable before reading; only the compact JSON remains.
        (self.root / "invented.csv").unlink()
        env, output = self.execute("isoformer_gtex_audit", configure=self.configure_audit_identity(report))
        self.assertEqual(env["AUDIT_MODE"], "read_results")
        self.assertEqual(env["audit"], report)
        self.assertEqual(env["selected_path"], path)
        self.assertIn("synthetic-host", str(self.displays[0]))
        self.assertIn("Historical reference is unavailable", output)

    def test_audit_missing_report_stops_without_starting_audit(self):
        with self.assertRaisesRegex(FileNotFoundError, "No completed audit"):
            self.execute("isoformer_gtex_audit")

    def test_audit_corrupt_preferred_record_does_not_fall_back(self):
        report = self.audit_fixture()
        self.save("stage4/results/isoformer_cohort_audit/gtex_audit_v2.json", {})
        self.save("stage4/results/isoformer_cohort_audit/cohort_audit.json", report)
        with self.assertRaises(ValueError):
            self.execute("isoformer_gtex_audit", configure=self.configure_audit_identity(report))


if __name__ == "__main__":
    unittest.main()
