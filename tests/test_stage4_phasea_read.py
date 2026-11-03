"""Execute only saved-reading branches against invented files and blocked producers.

Selected notebook source cells run only in a synthetic namespace. No actual
datasets, caches or research records are read, and no research calculations run.
"""

from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, dataclass, replace
import builtins
import hashlib
import io
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_phasea_entrypoints import FixtureResult, import_runner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import swap_results


@dataclass
class ReadingConfig:
    lr: float = 3e-5
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 500
    early_stopping_patience: int = 20
    plateau_patience: int = 5
    entropy_lambda: float = 0.0
    channels: int = 100
    tconv: str = "appendix"
    seed: int = 0


class Forbidden:
    """Any interaction with an input/producer/resampling entry point is a failure."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        raise AssertionError("Saved reading reached a dataset/cache/model/computation entry point")


def code_cells(name):
    book = json.loads((ROOT / "Code/stage4" / name).read_text(encoding="utf-8"))
    return [(cell["id"], "".join(cell["source"]))
            for cell in book["cells"] if cell["cell_type"] == "code"]


class PhaseASavedReadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.code = self.folder / "Code"
        self.runner = import_runner()
        self.runner.train.TrainConfig = ReadingConfig
        self.displayed = []
        self.output = StringIO()
        self.namespace = {
            "Path": Path, "CODE_DIR": self.code, "REPO_ROOT": self.folder,
            "np": np, "pd": pd, "json": json, "asdict": asdict, "replace": replace,
            "hashlib": hashlib, "BytesIO": BytesIO,
            "display": self.displayed.append, "phasea_runner": self.runner,
            "swap_results": swap_results,
            "train": SimpleNamespace(TrainConfig=ReadingConfig, Result=FixtureResult,
                                     run=Forbidden(), fit=Forbidden(), build=Forbidden(),
                                     bootstrap=Forbidden(), score_rows=Forbidden()),
            "blf": SimpleNamespace(ENCODER_OF_MODALITY={"dna": "dna", "rna": "rna", "prot": "prot"},
                                   BenchmarkStreams=Forbidden(), build=Forbidden(), loader=Forbidden(),
                                   datasets=Forbidden()),
            "datasets": Forbidden(), "token_embeddings": Forbidden(), "encoders": Forbidden(),
            "time": Forbidden(), "plt": Forbidden(), "file_fingerprint": Forbidden(),
            "input_fingerprint": Forbidden(), "inputs_for_rows": Forbidden(),
        }

    def snapshot(self):
        return {p.relative_to(self.folder): p.read_bytes() for p in self.folder.rglob("*") if p.is_file()}

    def execute_read_cells(self, name, benchmark=None):
        # Imports, path search and device discovery are excluded; their bindings are controlled above.
        first = "0965876d" if name == "tier2_blf_phaseA.ipynb" else "settings"
        cells = code_cells(name)
        start = next(i for i, (cell_id, _) in enumerate(cells) if cell_id == first)
        before = self.snapshot()
        def guarded_open(original):
            def reading_only(file, mode="r", *args, **kwargs):
                if any(flag in mode for flag in "wax+"):
                    raise AssertionError("No file writes")
                if not Path(file).resolve().is_relative_to(self.folder):
                    raise AssertionError("No files outside the invented fixture")
                return original(file, mode, *args, **kwargs)
            return reading_only

        with ExitStack() as guards, redirect_stdout(self.output):
            guards.enter_context(patch("builtins.open", side_effect=guarded_open(builtins.open)))
            guards.enter_context(patch("io.open", side_effect=guarded_open(io.open)))
            for method in ("write_text", "write_bytes", "mkdir", "rename", "replace", "unlink"):
                guards.enter_context(patch.object(Path, method, side_effect=AssertionError("No file mutation")))
            guards.enter_context(patch("os.replace", side_effect=AssertionError("No publication")))
            guards.enter_context(patch("pandas.DataFrame.to_csv", side_effect=AssertionError("No export")))
            for cell_id, source in cells[start:]:
                if cell_id == "settings" and benchmark is not None:
                    source = source.replace('BENCHMARK = "fungal_expression"',
                                            'BENCHMARK = ' + repr(benchmark))
                exec(compile(source, "synthetic-reading:" + cell_id, "exec"), self.namespace)
        self.assertTrue(self.namespace["READ_SAVED_RESULTS"])
        self.assertEqual(before, self.snapshot())
        return self.namespace

    def main_record(self, key="cov_vac_degradation", name="one.json", variant="concat", **updates):
        record = dict(benchmark=key, variant=variant, config=asdict(ReadingConfig(tconv="equation")),
                      epochs=1, best_epoch=0, seconds=1.0, val_score=0.5, test_score=0.4)
        record.update(updates)
        path = self.code / "stage4/results/phaseA" / key / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def appendix_records(self, benchmark="ecoli_proteins", complete=True):
        directory = (self.code / "stage4/results/blf_phaseA" / benchmark
                     if benchmark == "ecoli_proteins"
                     else self.code / "stage3/results/fungal_expression/reproducible")
        directory.mkdir(parents=True)
        runs = self.runner.planned_runs(ReadingConfig())
        for variant, config in runs if complete else [("concat", ReadingConfig())]:
            identity = dict(version=1, variant=variant, config=asdict(config),
                            model_kwargs={}, widths=self.runner.EXPECTED_WIDTHS,
                            n_outputs=3 if benchmark == "ecoli_proteins" else 1,
                            inputs={"rows_sha256": "synthetic"}, environment={"python": "fixture"},
                            code_sha256={"train.py": "synthetic"})
            record = dict(benchmark=benchmark, variant=variant, config=asdict(config), identity=identity,
                          epochs=1, best_epoch=0, val_score=0.5, test_score=0.4,
                          history=[{"epoch": 0, "val_score": 0.5}])
            self.runner.result_path(directory, benchmark, variant, config).write_text(
                json.dumps(record), encoding="utf-8")
        return directory

    def appendix_export(self, directory):
        output = directory / "records/analysis-20261020"
        output.mkdir(parents=True)
        reference_path = self.runner.result_path(directory, "ecoli_proteins", "concat", ReadingConfig())
        reference = FixtureResult.load(reference_path)
        labels = ["DNA", "RNA", "Protein", "Concat", "MIL lambda=0.01", "MIL lambda=0.5",
                  "MIL lambda=1", "MIL without entropy", "Cross"]
        runs = self.runner.planned_runs(ReadingConfig())
        candidates = [dict(seed=config.seed, candidate=labels[i % 9], variant=variant,
                           entropy_lambda=config.entropy_lambda) for i, (variant, config) in enumerate(runs)]
        manifest = dict(benchmark="ecoli_proteins", task="classification",
                        protocol=dict(base_config=asdict(ReadingConfig()), seeds=[0, 1, 2],
                                      positive_lambdas=[0.01, 0.5, 1.0], candidates=candidates),
                        recorded_producer_cohort=self.runner._historical_cohort(
                            reference, "ecoli_proteins", "concat", ReadingConfig()),
                        candidate_files=[dict(path=p.relative_to(self.folder).as_posix(),
                                              sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                         for p in sorted(directory.glob("*.json"))], artifacts=[])
        for name in ("validation_choices", "selected_scores", "seed_summary", "paired_intervals"):
            path = output / (name + ".csv")
            path.write_text("seed,fixture_value\n0,0.4\n", encoding="utf-8")
            manifest["artifacts"].append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (output / "analysis-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return output

    def test_main_empty_folders_report_missing_without_inputs_or_training(self):
        state = self.execute_read_cells("tier2_blf_phaseA.ipynb")
        self.assertTrue(state["saved_phaseA_scores"].empty)
        self.assertIn("missing", state["phaseA_result_coverage"]["status"].tolist())
        self.assertIn("no bootstrap is recomputed", self.output.getvalue())
        self.assertNotIn("STREAMS", state)

    def test_main_score_only_record_remains_readable_when_other_evidence_is_bad(self):
        self.main_record()
        bad = self.main_record(key="ecoli_proteins")
        bad.write_text("invalid JSON", encoding="utf-8")
        state = self.execute_read_cells("tier2_blf_phaseA.ipynb")
        self.assertEqual(state["saved_phaseA_scores"]["test score"].tolist(), [0.4])
        self.assertTrue(state["saved_score_issues"])
        self.assertIn("saved targets or predictions absent",
                      state["phaseA_result_coverage"]["array check"].tolist())

    def test_main_duplicates_are_reported_without_hiding_other_benchmarks(self):
        self.main_record(name="a.json")
        self.main_record(name="b.json")
        self.main_record(key="ecoli_proteins")
        state = self.execute_read_cells("tier2_blf_phaseA.ipynb")
        self.assertEqual(state["saved_phaseA_scores"]["benchmark"].tolist(), ["ecoli_proteins"])
        self.assertIn("ambiguous", state["phaseA_result_coverage"]["status"].tolist())

    def test_main_inaccessible_folder_does_not_hide_other_benchmarks(self):
        self.main_record(key="ecoli_proteins")
        original = swap_results.read_results

        def inaccessible(folders, **kwargs):
            if kwargs["benchmark"] == "cov_vac_degradation":
                raise OSError("Synthetic folder unavailable")
            return original(folders, **kwargs)

        with patch.object(swap_results, "read_results", side_effect=inaccessible):
            state = self.execute_read_cells("tier2_blf_phaseA.ipynb")
        self.assertEqual(state["saved_phaseA_scores"]["benchmark"].tolist(), ["ecoli_proteins"])
        self.assertIn("unavailable", state["phaseA_result_coverage"]["status"].tolist())

    def test_appendix_empty_and_partial_folders_need_no_rows_or_cache_headers(self):
        self.appendix_records("fungal_expression", complete=False)
        state = self.execute_read_cells("tier2_blf_phaseA_appendix.ipynb")
        self.assertFalse(state["RUN_TRAINING"])
        self.assertEqual(state["saved_scores"]["test score"].tolist(), [0.4])
        self.assertIn("not_found_locally", state["saved_coverage"]["status"].tolist())
        self.assertIn("intervals remain unavailable", self.output.getvalue())
        self.assertNotIn("cache_headers", state)
        self.assertNotIn("rows", state)

    def test_appendix_invalid_record_is_reported_and_does_not_hide_other_scores(self):
        directory = self.appendix_records()
        path = self.runner.result_path(directory, "ecoli_proteins", "cross", ReadingConfig())
        path.write_text("invalid JSON", encoding="utf-8")
        state = self.execute_read_cells("tier2_blf_phaseA_appendix.ipynb", "ecoli_proteins")
        self.assertEqual(len(state["saved_scores"]), 26)
        self.assertIn("blocked_preserve", state["saved_coverage"]["status"].tolist())
        self.assertTrue(state["saved_export_issues"])

    def test_appendix_missing_reference_keeps_individual_scores_labeled_pending(self):
        directory = self.appendix_records()
        self.runner.result_path(directory, "ecoli_proteins", "concat", ReadingConfig()).unlink()
        state = self.execute_read_cells("tier2_blf_phaseA_appendix.ipynb", "ecoli_proteins")
        self.assertEqual(len(state["saved_scores"]), 26)
        self.assertEqual(set(state["saved_scores"]["status"]), {"reference_pending"})
        self.assertEqual(state["saved_export_tables"], {})

    def test_appendix_existing_export_is_read_without_bootstrap(self):
        self.appendix_export(self.appendix_records())
        state = self.execute_read_cells("tier2_blf_phaseA_appendix.ipynb", "ecoli_proteins")
        self.assertEqual(len(state["saved_scores"]), 27)
        self.assertEqual(len(state["saved_export_tables"]), 4)
        self.assertEqual(state["saved_export_issues"], [])
        self.assertEqual(state["saved_export_tables"]["paired_intervals"]["fixture_value"].tolist(), [0.4])

    def test_appendix_corrupt_interval_preserves_independent_export_tables(self):
        output = self.appendix_export(self.appendix_records())
        (output / "paired_intervals.csv").write_text("changed", encoding="utf-8")
        state = self.execute_read_cells("tier2_blf_phaseA_appendix.ipynb", "ecoli_proteins")
        self.assertEqual(len(state["saved_export_tables"]), 3)
        self.assertIn("checksum", state["saved_export_issues"][0]["reason"])
        self.assertIn("no reanalysis", self.output.getvalue())

    def test_appendix_export_with_changed_source_does_not_hide_individual_scores(self):
        directory = self.appendix_records()
        self.appendix_export(directory)
        path = self.runner.result_path(directory, "ecoli_proteins", "cross", ReadingConfig())
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        state = self.execute_read_cells("tier2_blf_phaseA_appendix.ipynb", "ecoli_proteins")
        self.assertEqual(len(state["saved_scores"]), 27)
        self.assertEqual(state["saved_export_tables"], {})
        self.assertIn("candidate changed", state["saved_export_issues"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
