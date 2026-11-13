"""Check pilot ownership and comparison guards with invented objects and files.

Only notebook function definitions and one isolated ordering branch are loaded.
Models, cache payloads, fitting and statistical calculations are all substitutes.
"""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = (
    "s4-cache-lifetime", "s4-cache-preparation", "s4-probe-helper", "s4-result-tag",
    "s4-fit-helper", "s4-comparison-helper",
)
MODALITIES = ("dna", "rna", "prot")


class FixtureCache:
    width = 2
    meta = {"total_tokens": 4}

    def __init__(self):
        self.close = Mock()

    def __len__(self):
        return 2


class FixtureStreams:
    def __init__(self, benchmark, rows, caches, tissues):
        self.benchmark, self.rows, self.caches, self.tissues = benchmark, rows, caches, tissues
        self.check = Mock()
        self.close = Mock(side_effect=self.close_caches)

    def close_caches(self):
        for cache in self.caches.values():
            cache.close()


class PilotEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".pilot-entry-", dir=ROOT / "tests")
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to((ROOT / "tests").resolve()))
        self.addCleanup(self.temporary.cleanup)
        book = json.loads((ROOT / "Code/stage3/gtex_frozen_fusion_reference.ipynb").read_text(
            encoding="utf-8"))
        self.cells = {cell["id"]: "".join(cell["source"])
                      for cell in book["cells"] if cell["cell_type"] == "code"}
        self.streams = []
        self.caches = []
        self.verification = object()
        self.created_stream = Mock(side_effect=self.make_streams)
        self.extract = Mock(side_effect=lambda *args, **kwargs: (self.make_cache(), True))
        self.summarize = Mock(side_effect=lambda *args, **kwargs: self.make_cache())
        self.free_memory = Mock()
        self.verify_caches = Mock(return_value={"synthetic": "verified"})
        self.loader_bundle = {"fixture": "loaders"}
        self.make_loaders = Mock(return_value=self.loader_bundle)
        self.execution_records = Mock(return_value={"run_identity": "fixture-identity"})
        self.fit = Mock()
        self.metric = Mock(side_effect=AssertionError("Unexpected statistical calculation"))
        self.build_model = Mock(side_effect=AssertionError("Probe model was not configured"))
        self.collate = Mock(side_effect=AssertionError("Probe batch was not configured"))
        self.env = {
            "np": np, "pd": pd, "json": json, "Path": Path,
            "BENCHMARK": "fixture", "gtex": object(), "CODE_DIR": self.folder / "Code",
            "CACHE_ROOT": self.folder / "cache", "RESULTS_DIR": self.folder / "results",
            "DEVICE": "fixture-device", "MAX_BATCH": 16, "MAX_EPOCHS": 500,
            "VAL_FRACTION": 0.05, "N_HEADS": 8, "CELL_BUDGET": 300_000_000,
            "GTEX_CACHE_RECOVERY": self.folder / "recovery.json",
            "CROP": {"dna": None, "rna": 12, "prot": 6},
            "CONTEXT": {"nt": 100, "protein": 100},
            "VARIANTS": ("dna", "rna", "prot", "dna+rna+prot"), "SEEDS": (0,),
            "display": Mock(),
            "datasets": SimpleNamespace(GTEX_TISSUES=("a", "b")),
            "encoders": SimpleNamespace(
                ENCODERS={"nt": SimpleNamespace(modality="dna"),
                          "protein": SimpleNamespace(modality="protein")},
                free_memory=self.free_memory,
            ),
            "isoformer": SimpleNamespace(
                MODALITIES=MODALITIES, STREAM_ENCODER={"dna": "nt", "rna": "nt", "prot": "protein"},
                GTExStreams=self.created_stream, loaders_for=self.make_loaders,
                builder=self.build_model, collate=self.collate,
            ),
            "token_embeddings": SimpleNamespace(
                CacheVerification=Mock(return_value=self.verification),
                extract=self.extract, summarize=self.summarize,
            ),
            "run_identity": SimpleNamespace(
                gtex_cache_records=self.verify_caches, execution_records=self.execution_records,
            ),
            "stream_texts": Mock(return_value=(["AAAAAA", "CCCCCC"], 0)),
            "train": SimpleNamespace(TrainConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                                     run=self.fit, r_squared=self.metric),
            "time": SimpleNamespace(perf_counter=Mock(side_effect=range(100))),
        }
        self.env["ISOFORMER_TABLE2"] = pd.DataFrame(
            {"Spearman": 0.8, "R2": 0.5}, index=self.env["VARIANTS"])
        for cell_id in DEFINITIONS:
            tree = ast.parse(self.cells[cell_id], filename=cell_id)
            if not all(isinstance(node, ast.FunctionDef) for node in tree.body):
                raise AssertionError(f"Expected definitions only in {cell_id}")
            exec(compile(tree, f"synthetic-definitions:{cell_id}", "exec"), self.env)

    def make_streams(self, *args):
        streams = FixtureStreams(*args)
        self.streams.append(streams)
        return streams

    def make_cache(self):
        cache = FixtureCache()
        self.caches.append(cache)
        return cache

    def call(self, name, *args, **kwargs):
        with redirect_stdout(StringIO()):
            return self.env[name](*args, **kwargs)

    def metadata(self, modality, value):
        encoder = self.env["isoformer"].STREAM_ENCODER[modality]
        folder = self.env["CACHE_ROOT"] / "fixture" / f"{modality}_{encoder}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "meta.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def owned_stream(self):
        return self.make_streams("fixture", object(), {}, ("a", "b"))

    def assert_released(self, streams):
        streams.close.assert_called_once_with()
        self.free_memory.assert_called_once_with()

    def configure_probe(self, failure=None):
        error = RuntimeError(f"synthetic {failure} failure") if failure else None
        labels = SimpleNamespace(to=Mock(return_value="labels"))
        batch = SimpleNamespace(to=Mock(), labels=labels)
        batch.to.return_value = batch
        model = Mock(return_value=("prediction", "diagnostics"))
        model.to.return_value = model
        backward = Mock()
        loss = Mock(return_value=SimpleNamespace(backward=backward))
        self.env["torch"] = SimpleNamespace(nn=SimpleNamespace(MSELoss=Mock(return_value=loss)))
        self.build_model.side_effect = None
        self.build_model.return_value = model
        self.collate.side_effect = None
        self.collate.return_value = batch
        failures = {"constructor": self.build_model, "model_to": model.to,
                    "collate": self.collate, "batch_to": batch.to, "forward": model,
                    "labels_to": labels.to, "backward": backward}
        if failure:
            failures[failure].side_effect = error
        loaders = {"train": SimpleNamespace(
            dataset=["first", "second"], batch_sampler=SimpleNamespace(batches=[[0], [1, 0]]))}
        return loaders, error, model, batch, backward

    def test_fresh_and_modern_cache_paths_extract_then_verify_before_return(self):
        self.metadata("rna", {"input_sha256": "synthetic", "shards": []})
        streams = self.call("prepare_pilot_caches")
        self.assertEqual(self.extract.call_count, 3)
        self.summarize.assert_not_called()
        self.assertEqual(tuple(streams.caches), MODALITIES)
        streams.check.assert_called_once_with()
        self.verify_caches.assert_called_once_with(
            streams, crop=self.env["CROP"], verification=self.verification,
            gtex_recovery=self.env["GTEX_CACHE_RECOVERY"])
        streams.close.assert_not_called()
        self.free_memory.assert_not_called()
        self.call("release_pilot_resources", streams)
        self.assert_released(streams)
        for cache in self.caches:
            cache.close.assert_called_once_with()

    def test_complete_legacy_caches_are_only_summarized_and_verified(self):
        paths = [self.metadata(name, {"n_shards": 1, "n_sequences": 2}) for name in MODALITIES]
        before = {path: path.read_bytes() for path in paths}
        streams = self.call("prepare_pilot_caches")
        self.extract.assert_not_called()
        self.assertEqual(self.summarize.call_count, 3)
        self.verify_caches.assert_called_once()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.call("release_pilot_resources", streams)

    def test_partial_legacy_never_falls_back_to_extraction(self):
        self.metadata("dna", {"n_shards": 2})
        error = FileNotFoundError("synthetic missing legacy shard")
        self.summarize.side_effect = error
        with self.assertRaises(FileNotFoundError) as caught:
            self.call("prepare_pilot_caches")
        self.assertIs(caught.exception, error)
        self.extract.assert_not_called()
        self.verify_caches.assert_not_called()
        self.assert_released(self.streams[-1])

    def test_malformed_metadata_refuses_before_extracting_and_cleans_up(self):
        for content in ("[1, 2]", "{"):
            with self.subTest(content=content):
                path = self.metadata("dna", {})
                path.write_text(content, encoding="utf-8")
                self.free_memory.reset_mock()
                with self.assertRaises(ValueError):
                    self.call("prepare_pilot_caches")
                self.extract.assert_not_called()
                self.assert_released(self.streams[-1])

    def test_preparation_failure_releases_prior_caches_and_preserves_original_error(self):
        first = self.make_cache()
        error = KeyboardInterrupt("synthetic interrupted extraction")
        self.extract.side_effect = [(first, False), error]
        self.free_memory.side_effect = OSError("synthetic allocator cleanup failure")
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.call("prepare_pilot_caches")
        self.assertIs(caught.exception, error)
        first.close.assert_called_once_with()
        self.assert_released(self.streams[-1])
        self.assertIn("cleanup", " ".join(error.__notes__))

    def test_verification_failure_cleans_all_caches_without_replacement(self):
        for name in MODALITIES:
            self.metadata(name, {"n_shards": 1})
        error = ValueError("synthetic changed legacy hash")
        self.verify_caches.side_effect = error
        with self.assertRaises(ValueError) as caught:
            self.call("prepare_pilot_caches")
        self.assertIs(caught.exception, error)
        self.extract.assert_not_called()
        self.assert_released(self.streams[-1])
        for cache in self.caches:
            cache.close.assert_called_once_with()

    def test_probe_success_uses_selected_batch_and_releases_resources(self):
        streams = self.owned_stream()
        loaders, _, model, batch, backward = self.configure_probe()
        self.call("probe_largest_batch", streams, loaders, 1, 10, "fixture-device", 8)
        self.build_model.assert_called_once_with(streams, "dna+rna+prot", None, n_heads=8)
        self.collate.assert_called_once_with(["second", "first"])
        model.assert_called_once_with(batch)
        backward.assert_called_once_with()
        self.assert_released(streams)

    def test_probe_failures_and_cleanup_failures_preserve_the_original_error(self):
        for stage in ("constructor", "model_to", "collate", "batch_to", "forward", "labels_to", "backward"):
            for cleanup_fails in (False, True):
                with self.subTest(stage=stage, cleanup_fails=cleanup_fails):
                    self.free_memory.reset_mock()
                    self.free_memory.side_effect = OSError("cleanup failed") if cleanup_fails else None
                    streams = self.owned_stream()
                    loaders, error, _, _, _ = self.configure_probe(stage)
                    with self.assertRaises(RuntimeError) as caught:
                        self.call("probe_largest_batch", streams, loaders, 1, 10, "fixture-device", 8)
                    self.assertIs(caught.exception, error)
                    self.assert_released(streams)

    def test_cleanup_failure_after_success_propagates_and_attempts_both_cleanups(self):
        streams = self.owned_stream()
        error = OSError("mapping close failed")
        streams.close.side_effect = error
        loaders, _, _, _, _ = self.configure_probe()
        with self.assertRaises(OSError) as caught:
            self.call("probe_largest_batch", streams, loaders, 1, 10, "fixture-device", 8)
        self.assertIs(caught.exception, error)
        self.assert_released(streams)

    def test_fit_passes_selected_modalities_recovery_and_existing_recipe_then_closes(self):
        streams = self.owned_stream()
        self.call("fit_pilot_variant", streams, "dna+rna", 2, self.verification)
        self.make_loaders.assert_called_once_with(
            streams, batch_size=16, val_fraction=0.05, seed=2, budget=300_000_000,
            n_heads=8, modalities=("dna", "rna"))
        args, kwargs = self.execution_records.call_args
        self.assertEqual(args[:2], (streams, "dna+rna"))
        self.assertEqual(vars(args[2]), {"batch_size": 16, "max_epochs": 500, "seed": 2})
        self.assertIs(kwargs["verification"], self.verification)
        self.assertEqual(kwargs["gtex_recovery"], self.env["GTEX_CACHE_RECOVERY"])
        self.assertEqual(kwargs["crop"], self.env["CROP"])
        self.assertEqual(kwargs["model_kwargs"], {"n_heads": 8})
        self.assertEqual(kwargs["tag"], "fixture__dna+rna__heads8__budget300M__seed2")
        self.assertIs(self.fit.call_args.kwargs["loaders"], self.loader_bundle)
        self.assertEqual(self.fit.call_args.kwargs["run_identity"], "fixture-identity")
        self.assert_released(streams)

    def test_fit_loader_identity_and_training_failures_always_release(self):
        for stage in (self.make_loaders, self.execution_records, self.fit):
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(stage=stage, error=error_type):
                    self.free_memory.reset_mock()
                    self.free_memory.side_effect = OSError("cleanup failed")
                    for operation in (self.make_loaders, self.execution_records, self.fit):
                        operation.side_effect = None
                    error = error_type("synthetic operation failed")
                    stage.side_effect = error
                    streams = self.owned_stream()
                    with self.assertRaises(error_type) as caught:
                        self.call("fit_pilot_variant", streams, "prot", 0, self.verification)
                    self.assertIs(caught.exception, error)
                    self.assert_released(streams)

    def test_summary_requires_complete_variants_and_retains_declared_seed_order(self):
        self.env["SEEDS"] = (2, 0, 1)
        self.metric.side_effect = lambda truth, prediction: float(prediction[0, 0])
        records = [
            {"variant": "rna", "seed": seed,
             "result": {"test_score": (seed + 1) / 10, "epochs": seed + 2,
                        "test_truth": [[1, 2]], "test_prediction": [[seed, seed + 1]]}}
            for seed in (1, 0, 2)
        ]
        records.append({"variant": "dna", "seed": 0, "result": {"unread": True}})
        cohort = SimpleNamespace(records=records, coverage=[
            {"variant": "rna", "status": "complete"}, {"variant": "dna", "status": "partial"},
        ])
        summary = self.call("summarize_pilot_results", cohort)
        self.assertEqual(summary.index.tolist(), ["rna"])
        self.assertEqual([call.args[1][0, 0] for call in self.metric.call_args_list], [2, 0, 1])
        self.assertAlmostEqual(summary.loc["rna", "our Spearman"], 0.2)
        self.assertAlmostEqual(summary.loc["rna", "our R2"], 1.0)

    def test_empty_and_partial_summaries_do_not_call_metrics(self):
        for coverage in ([], [{"variant": "dna", "status": "partial"}]):
            with self.subTest(coverage=coverage):
                summary = self.call("summarize_pilot_results", SimpleNamespace(records=[], coverage=coverage))
                self.assertTrue(summary.empty)
                self.assertIn("our Spearman", summary.columns)
                self.assertIn("our R2", summary.columns)
        self.metric.assert_not_called()

    def test_ordering_claims_require_all_three_single_modalities(self):
        for scores, expected_count in (({}, 0), ({"dna": 0.4, "dna+rna+prot": 0.8}, 0),
                                       ({"dna": 0.4, "rna": 0.6, "prot": 0.5,
                                         "dna+rna+prot": 0.8}, 2)):
            with self.subTest(scores=scores):
                self.env["comparison"] = pd.DataFrame.from_dict(
                    scores, orient="index", columns=["our Spearman"])
                exec(compile(self.cells["s4c15"], "synthetic-pilot-ordering", "exec"), self.env)
                self.assertEqual(len(self.env["ordering_checks"]), expected_count)
                self.assertTrue(all(row["result"] == "ok" for row in self.env["ordering_checks"]))
        self.metric.assert_not_called()


if __name__ == "__main__":
    unittest.main()
