"""Check Stage 3 build readers with invented files and blocked research operations.

Only isolated definitions and reading branches run. Bootstrap guards use stubbed
scores, never scientific metrics, research results, models or token caches.
"""

from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
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
from shared_code.fusion import run_identity, swap_results


ENCODERS = {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"}
CLAIM_VARIANTS = ("unimodal:dna", "concat", "mil", "cross")
DEFINITION_CELLS = (
    "build-result-reader", "build-comparison-helpers", "build-convolution-reader",
    "build-pairing-evidence", "build-seed-reader", "build-pairing-coverage",
    "build-bootstrap-helper",
)


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


class Stage3BuildReadingTests(unittest.TestCase):
    def setUp(self):
        # Keep fixture cleanup within this checkout's writable tests directory.
        self.temp = tempfile.TemporaryDirectory(prefix=".stage3-read-", dir=ROOT / "tests")
        self.folder = Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to((ROOT / "tests").resolve()))
        self.addCleanup(self.temp.cleanup)
        self.code = self.folder / "Code"
        self.results = self.code / "stage3/results/fixture"
        self.results.mkdir(parents=True)
        self.audit = self.results / "records/verification.json"
        notebook = json.loads((ROOT / "Code/stage3/blf_build_check.ipynb").read_text(
            encoding="utf-8"))
        self.cells = {cell["id"]: "".join(cell["source"])
                      for cell in notebook["cells"] if cell["cell_type"] == "code"}
        self.blocked = Mock(side_effect=AssertionError("Research entry point reached"))
        self.scoring = Mock(side_effect=AssertionError("Metric calculation is blocked"))
        published = {
            "unimodal:dna": ("Nucleotide Transformer", 0.8004),
            "unimodal:rna": ("RNA-FM", 0.7),
            "unimodal:prot": ("ESM-2 (8M)", 0.6),
            "concat": ("Concatenation", 0.8005),
            "mil": ("MIL + Entropy", 0.83),
            "mil (lam 0)": ("MIL without entropy", 0.84),
            "cross": ("Cross Attention", 0.82),
        }
        self.env = {
            "Path": Path, "json": json, "hashlib": hashlib, "np": np, "pd": pd,
            "run_identity": run_identity, "swap_results": swap_results,
            "CODE_DIR": self.code, "BENCHMARK": "fixture", "RESULTS_DIR": self.results,
            "CHANNELS": 100, "BATCH_SIZE": 32, "MAX_EPOCHS": 500,
            "LAMBDAS": (0.01, 0.5, 1.0), "SELECTION_SEED": 0, "SEEDS": (0, 1, 2),
            "CLAIM_LAMBDA": 0.01,
            "CLAIM_RUNS": [(name, {"entropy_lambda": 0.01} if name == "mil" else {})
                           for name in CLAIM_VARIANTS],
            "PUBLISHED_OF": published,
            "TABLE1": pd.DataFrame({"fixture": {label: value
                for name, (label, value) in published.items() if name != "mil (lam 0)"}}),
            "TABLE4": pd.DataFrame({"fixture": {"MIL without entropy": 0.84}}),
            "FUNGAL_VERIFICATION": self.audit, "FUNGAL_VERIFICATION_SHA256": "unused",
            "display": Mock(),
            "train": SimpleNamespace(
                TrainConfig=ReadingConfig, score_rows=self.scoring,
                run=self.blocked, fit=self.blocked, build=self.blocked,
                paired_bootstrap=self.blocked, seed_averaged_bootstrap=self.blocked,
            ),
            "blf": SimpleNamespace(
                ENCODER_OF_MODALITY=ENCODERS,
                BenchmarkStreams=SimpleNamespace(open=self.blocked),
                build=self.blocked, loader=self.blocked,
            ),
            "tokens": SimpleNamespace(extract=self.blocked),
        }
        for cell_id in DEFINITION_CELLS:
            self.execute(cell_id)

    def snapshot(self):
        return {p.relative_to(self.folder): p.read_bytes()
                for p in self.folder.rglob("*") if p.is_file()}

    @contextmanager
    def reading_only(self):
        before = self.snapshot()
        original_text, original_bytes = Path.read_text, Path.read_bytes

        def local_read(original):
            def read(path, *args, **kwargs):
                if not path.resolve().is_relative_to(self.folder):
                    raise AssertionError("Read outside invented fixture")
                return original(path, *args, **kwargs)
            return read

        with ExitStack() as guards:
            guards.enter_context(patch.object(Path, "read_text", local_read(original_text)))
            guards.enter_context(patch.object(Path, "read_bytes", local_read(original_bytes)))
            for name in ("write_text", "write_bytes", "mkdir", "rename", "replace", "unlink"):
                guards.enter_context(patch.object(Path, name, side_effect=AssertionError("File mutation")))
            guards.enter_context(patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("Export")))
            yield
        self.assertEqual(before, self.snapshot())
        self.blocked.assert_not_called()

    def execute(self, cell_id):
        output = StringIO()
        with self.reading_only(), redirect_stdout(output):
            exec(compile(self.cells[cell_id], f"synthetic-stage3:{cell_id}", "exec"), self.env)
        return output.getvalue()

    def call(self, name, *args, **kwargs):
        with self.reading_only():
            return self.env[name](*args, **kwargs)

    def save(self, variant="concat", *, seed=0, tconv="equation", penalty=None,
             modern=False, name=None, config_updates=None, **fields):
        if penalty is None:
            penalty = 0.01 if variant == "mil" else 0.0
        config = {**asdict(ReadingConfig(tconv=tconv, seed=seed, entropy_lambda=penalty)),
                  **(config_updates or {})}
        record = {
            "benchmark": "fixture", "variant": variant, "config": config,
            "epochs": 4, "best_epoch": 3, "seconds": 1.0,
            "val_score": 0.5, "test_score": 0.4,
            "test_truth": [2.0, 4.0, 5.0, 6.0],
            "test_prediction": [2.1, 3.9, 5.1, 6.0],
            "alpha": {"dna": 0.2, "rna": 0.3, "prot": 0.5} if variant == "mil" else None,
        }
        record.update(fields)
        if modern:
            labels = np.asarray([1.0, 3.0, *record["test_truth"]], dtype=np.float32)
            splits = {"train": [0], "val": [1], "test": [2, 3, 4, 5]}
            inputs = {
                "encoders": dict(ENCODERS),
                "rows": {"columns": ["sequence", "length"], "count": 6, "sha256": "a" * 64},
                "targets": {**run_identity.array_record(labels), "columns": ["Value"],
                            "task": "regression", "transform": "as_supplied"},
                "splits": {key: run_identity.array_record(indices, np.int64)
                           for key, indices in splits.items()},
                "split_targets": {key: run_identity.array_record(labels[indices])
                                  for key, indices in splits.items()},
            }
            record["identity"] = run_identity.make_identity(
                benchmark=record["benchmark"], variant=variant, config=config, inputs=inputs,
                model={"contract": "synthetic"},
                loader=run_identity.blf_loader_spec(config, tuple(ENCODERS)),
            )
        filename = name or f"{variant.replace(':', '-')}__{tconv}__{penalty}__{seed}.json"
        path = self.results / filename
        path.write_text(json.dumps(record), encoding="utf-8")
        return {"path": path, "data": record, "legacy": not modern}

    def rewrite(self, entry):
        entry["path"].write_text(json.dumps(entry["data"]), encoding="utf-8")

    def cohort(self, *, modern=False):
        return {name: {seed: self.save(name, seed=seed, modern=modern)
                       for seed in self.env["SEEDS"]} for name in CLAIM_VARIANTS}

    def approve_legacy(self, results):
        entries = [entry for seeds in results.values() for entry in seeds.values()]
        report = {"status": "passed_internal_consistency_only", "inputs": [
            {"path": entry["path"].relative_to(self.folder).as_posix(),
             "sha256": hashlib.sha256(entry["path"].read_bytes()).hexdigest(),
             "variant": entry["data"]["variant"], "config": entry["data"]["config"],
             "rows": len(entry["data"]["test_truth"]), "status": "passed"}
            for entry in entries
        ]}
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        self.audit.write_text(json.dumps(report), encoding="utf-8")
        self.env["FUNGAL_VERIFICATION_SHA256"] = hashlib.sha256(self.audit.read_bytes()).hexdigest()
        return report

    def verdicts(self, scores):
        self.env["comparison"] = pd.DataFrame([
            {"variant": name, "ours": score,
             "difference": score - self.env["PUBLISHED_OF"][name][1]}
            for name, score in scores.items()
        ], columns=("variant", "ours", "difference")).set_index("variant")
        self.execute("d47122b2")
        return [row["result"] for row in self.env["verdict"]]

    def test_reader_filters_seed_full_recipe_benchmark_and_declared_penalties(self):
        good = self.save()
        for index, settings in enumerate((
            {"seed": 1}, {"lr": 1e-4}, {"batch_size": 16}, {"max_epochs": 50},
            {"channels": 1280}, {"tconv": "appendix"}, {"extra": True},
        )):
            self.save(name=f"foreign-{index}.json", config_updates=settings)
        self.save(name="foreign-benchmark.json", benchmark="other")
        self.save("mil", penalty=0.2)
        selection = self.call("read_build_results", "equation")
        self.assertEqual([e["path"] for e in selection.records], [good["path"]])
        self.assertEqual(len(selection.issues), 9)

    def test_duplicate_result_is_withheld_without_hiding_independent_scores(self):
        self.save(name="a.json")
        self.save(name="b.json")
        cross = self.save("cross")
        selection = self.call("read_build_results", "equation")
        chosen, penalty = self.call("selected_build_entries", selection)
        self.assertEqual(chosen, {"cross": selection.records[0]})
        self.assertEqual(chosen["cross"]["path"], cross["path"])
        self.assertEqual(len(selection.duplicates), 1)
        self.assertIsNone(penalty)

    def test_mil_selection_requires_complete_unique_positive_grid(self):
        self.save("mil", penalty=0.01, val_score=0.6)
        self.save("mil", penalty=0.5, val_score=0.9)
        self.save("mil", penalty=0.0, val_score=0.99)
        selected, penalty = self.call("selected_build_entries", self.call("read_build_results", "equation"))
        self.assertIsNone(penalty)
        self.assertEqual(set(selected), {"mil (lam 0)"})
        self.save("mil", penalty=1.0, val_score=0.4)
        selected, penalty = self.call("selected_build_entries", self.call("read_build_results", "equation"))
        self.assertEqual(penalty, 0.5)
        self.assertEqual(selected["mil"]["data"]["config"]["entropy_lambda"], 0.5)
        self.save("mil", penalty=0.01, name="duplicate-mil.json")
        _, penalty = self.call("selected_build_entries", self.call("read_build_results", "equation"))
        self.assertIsNone(penalty)

    def test_verdicts_use_full_precision_complete_coverage_and_strict_ordering(self):
        scores = {name: value for name, (_, value) in self.env["PUBLISHED_OF"].items()}
        verdicts = self.verdicts(scores)
        self.assertEqual(verdicts, ["ok"] * 5)
        # 0.8005 exceeds 0.8004 although rounding both first would conceal the difference.
        self.assertEqual(verdicts[2], "ok")
        for missing in ("unimodal:rna", "mil (lam 0)"):
            with self.subTest(missing=missing):
                partial = {name: value for name, value in scores.items() if name != missing}
                checks = self.verdicts(partial)
                self.assertEqual(checks[0], "not yet")
                self.assertEqual(checks[2 if missing.startswith("unimodal") else 4], "not yet")
        self.assertEqual(self.verdicts({})[0], "not yet")

    def test_verdict_ties_are_not_wins_or_decreases(self):
        scores = {name: value for name, (_, value) in self.env["PUBLISHED_OF"].items()}
        self.assertEqual(self.verdicts({**scores, "concat": scores["unimodal:dna"]})[2], "NO")
        self.assertEqual(self.verdicts({**scores, "cross": scores["mil"]})[3], "NO")
        self.assertEqual(self.verdicts({**scores, "unimodal:rna": scores["unimodal:dna"]})[1], "NO")
        self.assertEqual(self.verdicts({**scores, "mil (lam 0)": scores["mil"]})[4], "NO")

    def test_gate_display_uses_exact_selected_seed_and_recipe(self):
        for tconv in ("appendix", "equation"):
            for penalty in self.env["LAMBDAS"]:
                self.save("mil", tconv=tconv, penalty=penalty,
                          val_score=0.9 if penalty == 0.01 else 0.5)
        self.save("mil", seed=1, alpha={"dna": 99.0})
        self.save("mil", seed=0, name="wide.json", config_updates={"channels": 1280},
                  alpha={"dna": 88.0})
        output = self.execute("tconv4")
        self.assertIn("appendix seed 0", output)
        self.assertIn("equation seed 0", output)
        self.assertNotIn("seed 1 (", output)
        self.assertNotIn("dna 99.00", output)
        self.assertNotIn("dna 88.00", output)
        entry = self.env["equation_entries"]["mil"]
        self.assertEqual(self.env["equation"]["mil"], entry["data"]["test_score"])

    def test_empty_convolution_comparison_keeps_missing_rows_visible(self):
        self.execute("tconv4")
        self.assertEqual(set(self.env["settings"].index), {"concat", "mil", "mil (lam 0)", "cross"})
        self.assertTrue(self.env["settings"]["Equation 1 (k2 s2)"].isna().all())

    def test_claim_reader_preserves_requested_seed_ids_and_has_no_subset_means(self):
        results = self.cohort()
        self.approve_legacy(results)
        results["concat"][2]["path"].unlink()
        self.save("concat", seed=9)
        loaded, issues = self.call("read_claim_results")
        self.assertEqual(set(loaded["concat"]), {0, 1})
        self.assertTrue(issues)
        summary = self.call("summarize_claim_results", loaded)
        self.assertEqual(summary.loc["concat", "missing seeds"], (2,))
        for column in ("mean", "min", "max", "spread", "mean vs published"):
            self.assertTrue(pd.isna(summary.loc["concat", column]))
        self.assertIn("missing", summary.loc["concat", "summary status"])
        self.assertAlmostEqual(summary.loc["cross", "mean"], 0.4)

    def test_complete_compatible_scores_preserve_original_aggregations(self):
        for modern in (False, True):
            with self.subTest(modern=modern):
                results = self.cohort(modern=modern)
                scores = (0.125, 0.625, 0.875)
                for entries in results.values():
                    for seed, score in zip(self.env["SEEDS"], scores):
                        entries[seed]["data"]["test_score"] = score
                        self.rewrite(entries[seed])
                if not modern:
                    self.approve_legacy(results)
                summary = self.call("summarize_claim_results", results)
                self.assertTrue((summary["summary status"] == "ready").all())
                kind = "recorded inputs" if modern else "qualified historical files"
                self.assertTrue((summary["input evidence"] == kind).all())
                for variant in CLAIM_VARIANTS:
                    row = summary.loc[variant]
                    self.assertEqual(row["mean"], np.mean(scores))
                    self.assertEqual(row["min"], min(scores))
                    self.assertEqual(row["max"], max(scores))
                    self.assertEqual(row["spread"], max(scores) - min(scores))
                    self.assertEqual(row["mean vs published"], np.mean(scores) - row["published"])
                self.scoring.assert_not_called()

    def test_same_recipe_with_different_recorded_inputs_withholds_only_affected_summary(self):
        for mode in ("rows", "targets", "splits", "encoders"):
            with self.subTest(mode=mode):
                results = self.cohort(modern=True)
                entry = results["concat"][1]
                record = entry["data"]
                identity = record["identity"]
                inputs = deepcopy(identity["inputs"])
                if mode == "rows":
                    inputs["rows"]["sha256"] = hashlib.sha256(b"other invented row order").hexdigest()
                elif mode == "targets":
                    # Change an invented training label, preserving the saved test panel.
                    labels = np.asarray([7.0, 3.0, *record["test_truth"]], dtype=np.float32)
                    inputs["targets"].update(run_identity.array_record(labels))
                    inputs["split_targets"]["train"] = run_identity.array_record(labels[[0]])
                elif mode == "splits":
                    for field in ("splits", "split_targets"):
                        inputs[field]["train"], inputs[field]["val"] = (
                            inputs[field]["val"], inputs[field]["train"])
                else:
                    inputs["encoders"]["rna"] = "synthetic-alternate-rna"
                record["identity"] = run_identity.make_identity(
                    benchmark=record["benchmark"], variant=record["variant"],
                    config=record["config"], inputs=inputs,
                    model=identity["model"], loader=identity["loader"],
                )
                # Each record remains valid individually; it is cohort equality that fails.
                self.assertEqual(run_identity.recorded_blf_inputs(record), inputs)
                self.rewrite(entry)
                summary = self.call("summarize_claim_results", results)
                self.assertIn("input evidence differs", summary.loc["concat", "summary status"])
                for column in ("mean", "min", "max", "spread", "mean vs published"):
                    self.assertTrue(pd.isna(summary.loc["concat", column]))
                self.assertEqual(summary.loc["concat", "input evidence"], "")
                self.assertEqual(summary.loc["cross", "summary status"], "ready")
                self.assertAlmostEqual(summary.loc["cross", "mean"], 0.4)
                self.scoring.assert_not_called()

    def test_summary_does_not_mix_modern_and_qualified_historical_evidence(self):
        results = self.cohort(modern=True)
        results["concat"][1] = self.save("concat", seed=1)
        self.approve_legacy(results)
        summary = self.call("summarize_claim_results", results)
        self.assertIn("input evidence differs", summary.loc["concat", "summary status"])
        self.assertTrue(pd.isna(summary.loc["concat", "mean"]))
        self.assertEqual(summary.loc["cross", "summary status"], "ready")

    def test_score_only_modern_records_can_summarize_but_cannot_bootstrap(self):
        results = self.cohort(modern=True)
        for entries in results.values():
            for entry in entries.values():
                del entry["data"]["test_truth"]
                del entry["data"]["test_prediction"]
                self.rewrite(entry)
        loaded, issues = self.call("read_claim_results")
        self.assertTrue(all(set(entries) == set(self.env["SEEDS"])
                            for entries in loaded.values()))
        self.assertTrue(all(issue["reason"] == "configuration differs: seed" for issue in issues))
        summary = self.call("summarize_claim_results", loaded)
        self.assertTrue((summary["summary status"] == "ready").all())
        self.assertTrue((summary["input evidence"] == "recorded inputs").all())
        np.testing.assert_allclose(summary["mean"], 0.4)
        ready, coverage = self.call("prepare_paired_results", loaded)
        self.assertEqual(ready, {})
        self.assertTrue(all(row["paired status"] != "ready" for row in coverage))
        self.scoring.assert_not_called()

    def test_claim_reader_does_not_trust_a_filename_with_foreign_seed_metadata(self):
        results = self.cohort()
        entry = results["mil"][2]
        entry["data"]["config"]["seed"] = 8
        self.rewrite(entry)
        loaded, _ = self.call("read_claim_results")
        self.assertEqual(set(loaded["mil"]), {0, 1})
        self.assertTrue(pd.isna(self.call("summarize_claim_results", loaded).loc["mil", "mean"]))

    def test_duplicate_requested_seed_is_not_substituted_or_averaged(self):
        self.cohort()
        self.save("cross", seed=2, name="duplicate-cross.json")
        loaded, issues = self.call("read_claim_results")
        self.assertEqual(set(loaded["cross"]), {0, 1})
        self.assertTrue(any("cross" in issue["reason"] for issue in issues))
        self.assertTrue(pd.isna(self.call("summarize_claim_results", loaded).loc["cross", "mean"]))

    def test_complete_modern_cohort_pairs_without_historical_report(self):
        ready, coverage = self.call("prepare_paired_results", self.cohort(modern=True))
        self.assertEqual(set(ready), set(CLAIM_VARIANTS))
        self.assertTrue(all(row["paired status"] == "ready" for row in coverage))
        self.assertIsNone(self.call("paired_contrast_issue", ready, "cross", "unimodal:dna"))
        self.assertFalse(self.audit.exists())
        self.scoring.assert_not_called()

    def test_missing_seed_blocks_that_model_before_array_or_metric_work(self):
        results = self.cohort(modern=True)
        del results["concat"][1]
        ready, coverage = self.call("prepare_paired_results", results)
        self.assertNotIn("concat", ready)
        self.assertIn("missing", next(r["paired status"] for r in coverage if r["variant"] == "concat"))
        self.assertIsNotNone(self.call("paired_contrast_issue", ready, "concat", "cross"))
        self.scoring.assert_not_called()

    def test_mismatched_targets_missing_arrays_and_nonfinite_values_block_pairing(self):
        for mode in ("targets", "absent", "nonfinite", "foreign_seed"):
            with self.subTest(mode=mode):
                results = self.cohort(modern=True)
                entry = results["concat"][1]
                if mode == "targets":
                    entry["data"]["test_truth"][0] = 20.0
                elif mode == "absent":
                    entry["data"]["test_prediction"] = None
                elif mode == "nonfinite":
                    entry["data"]["test_prediction"][0] = float("nan")
                else:
                    entry["data"]["config"]["seed"] = 7
                self.rewrite(entry)
                ready, _ = self.call("prepare_paired_results", results)
                self.assertNotIn("concat", ready)
                self.scoring.assert_not_called()

    def test_modern_target_fingerprint_and_cross_seed_input_signature_are_checked(self):
        for mode in ("target", "rows"):
            with self.subTest(mode=mode):
                results = self.cohort(modern=True)
                entry = results["concat"][0]
                if mode == "target":
                    entry["data"]["test_truth"][0] += 1
                else:
                    entry["data"]["identity"]["inputs"]["rows"]["sha256"] = "b" * 64
                self.rewrite(entry)
                ready, _ = self.call("prepare_paired_results", results)
                self.assertNotIn("concat", ready)

    def test_pairing_requires_equal_input_evidence_and_ordered_targets_between_models(self):
        results = self.cohort(modern=True)
        for entry in results["concat"].values():
            entry["data"]["identity"]["inputs"]["rows"]["sha256"] = "b" * 64
            self.rewrite(entry)
        ready, _ = self.call("prepare_paired_results", results)
        self.assertIn("concat", ready)
        self.assertIn("input evidence", self.call("paired_contrast_issue", ready, "concat", "cross"))
        ready["concat"]["truth"] = ready["concat"]["truth"][::-1]
        self.assertIn("ordered targets", self.call("paired_contrast_issue", ready, "concat", "cross"))

    def test_legacy_reuse_requires_exact_report_and_result_bytes(self):
        results = self.cohort()
        self.assertTrue(self.call("summarize_claim_results", results)["mean"].isna().all())
        self.approve_legacy(results)
        ready, coverage = self.call("prepare_paired_results", results)
        self.assertEqual(set(ready), set(CLAIM_VARIANTS))
        self.assertTrue(all(row["input evidence"] == "qualified historical files" for row in coverage))
        entry = results["concat"][1]
        entry["data"]["seconds"] += 1
        self.rewrite(entry)
        ready, _ = self.call("prepare_paired_results", results)
        self.assertNotIn("concat", ready)
        summary = self.call("summarize_claim_results", results)
        self.assertTrue(pd.isna(summary.loc["concat", "mean"]))
        self.assertEqual(summary.loc["cross", "summary status"], "ready")
        self.audit.write_text("{}", encoding="utf-8")
        ready, _ = self.call("prepare_paired_results", results)
        self.assertEqual(ready, {})
        self.assertTrue(self.call("summarize_claim_results", results)["mean"].isna().all())
        self.scoring.assert_not_called()

    def test_report_entries_must_uniquely_match_path_config_rows_and_status(self):
        for mode in ("path", "duplicate", "config", "rows", "status"):
            with self.subTest(mode=mode):
                results = self.cohort()
                report = self.approve_legacy(results)
                match = report["inputs"][0]
                if mode == "path":
                    match["path"] = "Code/results/unlisted.json"
                elif mode == "duplicate":
                    report["inputs"].append(deepcopy(match))
                elif mode == "config":
                    match["config"] = {**match["config"], "seed": 99}
                elif mode == "rows":
                    match["rows"] += 1
                else:
                    match["status"] = "failed"
                self.audit.write_text(json.dumps(report), encoding="utf-8")
                self.env["FUNGAL_VERIFICATION_SHA256"] = hashlib.sha256(self.audit.read_bytes()).hexdigest()
                ready, _ = self.call("prepare_paired_results", results)
                self.assertNotIn("unimodal:dna", ready)

    def test_any_present_modern_identity_cannot_fall_back_to_approved_legacy_bytes(self):
        for identity in (None, {}, [], {"schema": "malformed"}):
            with self.subTest(identity=identity):
                results = self.cohort()
                entry = results["concat"][0]
                entry["data"]["identity"] = identity
                self.rewrite(entry)
                self.approve_legacy(results)
                ready, _ = self.call("prepare_paired_results", results)
                self.assertNotIn("concat", ready)
                self.assertIn("cross", ready)
                summary = self.call("summarize_claim_results", results)
                self.assertTrue(pd.isna(summary.loc["concat", "mean"]))
                self.assertNotEqual(summary.loc["concat", "summary status"], "ready")
                self.assertEqual(summary.loc["cross", "summary status"], "ready")

    def test_file_change_after_inventory_is_rejected(self):
        results = self.cohort(modern=True)
        entry = results["concat"][0]
        changed = deepcopy(entry["data"])
        changed["seconds"] += 1
        entry["path"].write_text(json.dumps(changed), encoding="utf-8")
        ready, coverage = self.call("prepare_paired_results", results)
        self.assertNotIn("concat", ready)
        issue = next(row["paired status"] for row in coverage if row["variant"] == "concat")
        self.assertIn("changed after", issue)
        summary = self.call("summarize_claim_results", results)
        self.assertTrue(pd.isna(summary.loc["concat", "mean"]))
        self.assertIn("changed after", summary.loc["concat", "summary status"])

    def test_undefined_observed_or_resampled_stub_score_withholds_interval(self):
        ready, _ = self.call("prepare_paired_results", self.cohort(modern=True))
        for values in ([float("nan")], [0.4] * 6 + [float("nan")]):
            with self.subTest(location="observed" if len(values) == 1 else "resampled"):
                self.scoring.reset_mock()
                self.scoring.side_effect = list(values) + [0.4] * 30
                with self.assertRaisesRegex(ValueError, "Undefined"):
                    self.call("bootstrap_contrast", ready, "concat", "cross", draws=2)

    def test_missing_pair_refuses_before_scoring_or_resampling(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            self.call("bootstrap_contrast", {}, "concat", "cross", draws=2)
        self.scoring.assert_not_called()

    def test_valid_stubbed_scores_are_averaged_with_paired_row_draws(self):
        ready, _ = self.call("prepare_paired_results", self.cohort(modern=True))
        for name in ("concat", "cross"):
            ready[name]["predictions"] = [
                prediction + index for index, prediction in enumerate(ready[name]["predictions"])
            ]
        # Scores are invented return values. No correlation is calculated.
        self.scoring.side_effect = [0.8, 0.6, 0.4, 0.3, 0.3, 0.3] * 3
        result = self.call("bootstrap_contrast", ready, "concat", "cross", draws=2, seed=5)
        self.assertAlmostEqual(result["difference"], 0.3)
        self.assertAlmostEqual(result["95% low"], 0.3)
        self.assertAlmostEqual(result["95% high"], 0.3)
        self.assertEqual(result["resolved"], "yes")
        calls = self.scoring.call_args_list
        self.assertEqual(len(calls), 18)
        # Fixed fixture draws from the original default_rng(5).integers(0, 4, 4)
        # sequence retain both the historical sampling order and shared row pairing.
        picks = (np.arange(4), np.array([2, 3, 0, 3]), np.array([1, 2, 2, 1]))
        for offset, pick in zip((0, 6, 12), picks):
            for model_index, name in enumerate(("concat", "cross")):
                for seed_index, prediction in enumerate(ready[name]["predictions"]):
                    call = calls[offset + model_index * 3 + seed_index]
                    np.testing.assert_array_equal(call.args[1], ready[name]["truth"][pick])
                    np.testing.assert_array_equal(call.args[2], prediction[pick])

    def test_analysis_display_reports_missing_evidence_without_scoring(self):
        self.env["claim_results"] = {name: {} for name in CLAIM_VARIANTS}
        self.env["spread"] = self.call("summarize_claim_results", self.env["claim_results"])
        self.execute("noise4")
        self.assertEqual(len(self.env["interval_rows"]), 6)
        self.assertTrue(all(row["status"] != "available" for row in self.env["interval_rows"]))
        self.scoring.assert_not_called()


if __name__ == "__main__":
    unittest.main()
