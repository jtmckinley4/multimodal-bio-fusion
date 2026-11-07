"""Synthetic metadata and stub-only swap orchestration; no scientific fits or caches."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import swap_plan, swap_results


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5, entropy_lambda=0.0,
              channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="ntv2_100m", rna="rnafm", prot="esm2_8m")
BENCHMARK = "mrna_stability"
SWAPS = [("rna", "rinalmo")]


class InjectedFailure(RuntimeError):
    pass


def forbidden(*args, **kwargs):
    raise AssertionError("Synthetic orchestration reached a real producer")


class SwapResumeNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "Code"
        self.original_dir = self.code / "stage4/results/phaseA" / BENCHMARK
        self.swap_root = self.code / "stage4/results/phaseB"
        notebook = json.loads((ROOT / "Code/stage4/tier2_encoder_swaps.ipynb").read_text(encoding="utf-8"))
        self.sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        self.calls, self.opened, self.closed = [], [], []
        self.fail_at = None
        self.close_error = False
        # The seed-0 tuning grid is metadata, never trained by these tests.
        for penalty, score in ((0.01, 0.8), (0.5, 0.6), (1.0, 0.4)):
            cfg = {**RECIPE, "entropy_lambda": penalty}
            self.write_result(swap_plan.result_path(self.original_dir, BENCHMARK, "mil", cfg),
                              "mil", cfg, ORIGINAL, val_score=score)

    def write_result(self, path, variant, config, encoders, val_score=0.5, test_score=0.4):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(benchmark=BENCHMARK, variant=variant, config=dict(config), epochs=4,
                    best_epoch=2, seconds=1.0, val_score=val_score, test_score=test_score,
                    identity=dict(benchmark=BENCHMARK, variant=variant, config=dict(config),
                                  inputs={"encoders": dict(encoders)}))
        path.write_text(json.dumps(data), encoding="utf-8")

    def plan(self, seed=0):
        return swap_plan.plan_benchmark(
            benchmark=BENCHMARK, config={**RECIPE, "seed": seed}, original_encoders=ORIGINAL,
            original_folders=[self.original_dir], original_destination=self.original_dir,
            swaps=[(m, e, True) for m, e in SWAPS], swap_root=self.swap_root / BENCHMARK,
        )

    def complete_seed_zero(self):
        for run in self.plan().require_ready():
            if run.status != "reuse":
                self.write_result(run.path, run.variant, run.config, run.encoders)
        self.assertTrue(all(run.status == "reuse" for run in self.plan().require_ready()))

    def fake_open(self, key, mapping):
        stream = SimpleNamespace(key=key, encoders=dict(mapping), rows=object())
        self.opened.append(stream)

        def close():
            self.closed.append(stream)
            if self.close_error:
                raise ValueError("injected cleanup failure")

        stream.close = close
        return stream

    def fake_run(self, stream, variant, config, device, folder, *, tag, run_identity, producer):
        self.assertEqual(run_identity["variant"], variant)
        self.assertEqual(run_identity["config"], vars(config))
        self.assertEqual(producer, {"fixture": True})
        path = Path(folder) / (tag + ".json")
        self.calls.append((variant, config.seed, path, dict(stream.encoders)))
        if self.fail_at == len(self.calls):
            raise InjectedFailure("injected fit failure")
        if path.exists():
            raise AssertionError("Orchestration called fitting for an existing result")
        self.write_result(path, variant, vars(config), stream.encoders)
        return SimpleNamespace(test_score=0.4)

    def namespace(self, seed=0):
        env = dict(
            READ_SAVED_RESULTS=False, SEED=seed, SELECTION_SEED=0, RUN_BOOTSTRAP=False,
            BATCH_SIZE=32, MAX_EPOCHS=500, CHANNELS=100, TCONV="equation", DEVICE="cpu",
            BENCHMARKS=(BENCHMARK,), ORIGINAL=ORIGINAL, CODE_DIR=self.code,
            CACHE_ROOT=self.code / "stage3/embeddings", RESULTS_ROOT=self.swap_root,
            ORIGINAL_RESULTS={BENCHMARK: [self.original_dir]}, BENCHMARK_ROWS={},
            swaps_for=lambda key: SWAPS, swap_plan=swap_plan,
            required_encoders=lambda key: [*ORIGINAL.values(), *(e for _, e in SWAPS)],
            CHUNK_SIZE=256, DTYPE=np.float16,
            tokens=SimpleNamespace(grid_resolution=lambda key: "nucleotide", summarize=forbidden,
                                   CacheVerification=lambda: object(), inspect_cache=forbidden,
                                   extract=forbidden, inputs_for_rows=forbidden,
                                   tokens_for_length=forbidden),
            train=SimpleNamespace(TrainConfig=lambda **kw: SimpleNamespace(**{**RECIPE, **kw}),
                                  checkpoint_preflight=forbidden, run=self.fake_run),
            datasets=SimpleNamespace(blf_benchmark_rows=forbidden),
            run_identity=SimpleNamespace(producer_record=lambda **kw: {"fixture": True}),
            CACHE_VERIFICATION=object(),
            time=time, pd=pd, np=np, json=json, display=lambda value: None,
        )
        exec(compile(self.sources["swaps-training-defs"], "fixture orchestration definitions", "exec"), env)
        env["open_fit_streams"] = self.fake_open
        return env

    def execute_plan(self, env):
        # Metadata inspection only, with invented folders and no real data paths.
        with redirect_stdout(io.StringIO()):
            exec(compile(self.sources["swaps-execution-plan"], "fixture metadata plan", "exec"), env)
        # Only the downstream fit identity is stubbed; the metadata plan above
        # must itself avoid current inputs for completed historical scores.
        env["requested_identity"] = lambda key, variant, config, mapping, **kw: {
            "benchmark": key, "variant": variant, "config": config,
            "inputs": {"encoders": mapping},
        }

    def execute_training_branch(self, env):
        guard = ast.parse(self.sources["fc1f77cc"]).body[0]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(guard.test.id, "READ_SAVED_RESULTS")
        # Only orchestration runs, under stub train.run and stub stream opening.
        code = compile(ast.Module(body=guard.orelse, type_ignores=[]), "fixture fit dispatch", "exec")
        with redirect_stdout(io.StringIO()):
            exec(code, env)

    def snapshot(self):
        return {p: p.read_bytes() for p in self.root.rglob("*.json")}

    def test_all_eight_stability_swaps_and_manual_seed_defaults(self):
        env = dict(np=np, tokens=SimpleNamespace(DEFAULT_CHUNK=256),
                   blf=SimpleNamespace(ENCODER_OF_MODALITY=ORIGINAL))
        exec(compile(self.sources["swaps-settings"], "fixture settings", "exec"), env)
        self.assertEqual(env["STABILITY_SWAPS"], env["SWAPS"])
        self.assertEqual(len(env["STABILITY_SWAPS"]), 8)
        self.assertEqual(env["SEED"], 0)
        self.assertEqual(env["SELECTION_SEED"], 0)
        self.assertFalse(env["RUN_BOOTSTRAP"])

    def test_complete_plan_opens_no_stream_and_calls_no_fit(self):
        self.complete_seed_zero()
        before = self.snapshot()
        env = self.namespace()
        self.execute_plan(env)
        self.assertEqual(env["PENDING_RUNS"], [])
        self.assertEqual(env["PENDING_CACHE_KEYS"], set())
        with redirect_stdout(io.StringIO()):
            # No dataset or token producer may be reached by these no-work branches.
            for cell_id in ("diskproj", "3e310c7e", "c397d229", "de27ae73"):
                exec(compile(self.sources[cell_id], "fixture no-work cache orchestration", "exec"), env)
        self.execute_training_branch(env)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.opened, [])
        self.assertEqual(before, self.snapshot())

    def test_compatible_native_historical_filename_is_reused_without_fitting(self):
        self.complete_seed_zero()
        run = next(r for r in self.plan().runs
                   if r.role == "original" and r.variant == "unimodal:prot")
        run.path.unlink()
        historical = self.original_dir / "historical_native_baseline.json"
        self.write_result(historical, run.variant, {**run.config, "tconv": "appendix"}, run.encoders)
        before = self.snapshot()
        env = self.namespace()
        self.execute_plan(env)
        native = next(r for r in env["FIT_PLANS"][BENCHMARK].runs
                      if r.role == "original" and r.variant == "unimodal:prot")
        self.assertEqual(native.path, historical.resolve())
        self.assertEqual(native.status, "reuse")
        self.execute_training_branch(env)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.opened, [])
        self.assertEqual(before, self.snapshot())

    def test_missing_same_seed_reference_precedes_missing_replacement(self):
        self.complete_seed_zero()
        removed = [r for r in self.plan().runs if
                   (r.role == "original" and r.variant == "unimodal:prot") or
                   (r.role == "replacement" and r.variant == "concat")]
        for run in removed:
            run.path.unlink()
        before = self.snapshot()
        env = self.namespace()
        self.execute_plan(env)
        self.assertEqual(len(env["PENDING_RUNS"]), 2)
        self.execute_training_branch(env)
        self.assertEqual([call[0] for call in self.calls], ["unimodal:prot", "concat"])
        self.assertEqual(len(self.opened), 2)
        self.assertEqual(self.opened, self.closed)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_newly_completed_result_after_preflight_is_reused(self):
        self.complete_seed_zero()
        run = next(r for r in self.plan().runs if r.role == "replacement" and r.variant == "concat")
        run.path.unlink()
        env = self.namespace()
        self.execute_plan(env)
        self.assertEqual(len(env["PENDING_RUNS"]), 1)
        self.write_result(run.path, run.variant, run.config, run.encoders)
        self.execute_training_branch(env)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.opened, [])

    def test_manual_seeds_use_seed_zero_selection_and_separate_destinations(self):
        self.complete_seed_zero()
        before = self.snapshot()
        for seed in (1, 2):
            env = self.namespace(seed)
            self.execute_plan(env)
            self.assertTrue(env["PENDING_RUNS"])
            self.assertEqual({r.config["seed"] for r in env["PENDING_RUNS"]}, {seed})
            self.assertEqual({r.config["entropy_lambda"] for r in env["PENDING_RUNS"]
                              if r.variant == "mil"}, {0.01})
            self.execute_training_branch(env)
            selected = [call for call in self.calls if call[1] == seed]
            self.assertEqual(len(selected), 12)  # Seven original + five nucleotide-replacement heads.
            self.assertTrue(all(f"__seed{seed}.json" in str(call[2]) for call in selected))
            self.assertEqual([call[3] for call in selected[:7]], [ORIGINAL] * 7)
            self.assertEqual(selected[-1][0], "cross")
        self.assertEqual({call[1] for call in self.calls}, {1, 2})
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_failure_closes_streams_and_rerun_reuses_completed_fixture_records(self):
        self.complete_seed_zero()
        before = self.snapshot()
        env = self.namespace(1)
        self.execute_plan(env)
        self.fail_at, self.close_error = 3, True
        with self.assertRaisesRegex(InjectedFailure, "injected fit failure") as caught:
            self.execute_training_branch(env)
        self.assertTrue(any("cleanup" in note.lower() for note in caught.exception.__notes__))
        self.assertEqual(self.opened, self.closed)
        completed = [call[2] for call in self.calls if call[2].exists()]
        self.assertEqual(len(completed), 2)
        self.fail_at, self.close_error = None, False
        previous_calls = len(self.calls)
        self.execute_plan(env)
        self.assertEqual(len(env["PENDING_RUNS"]), 10)
        self.execute_training_branch(env)
        retried_paths = [call[2] for call in self.calls[previous_calls:]]
        self.assertEqual(len(retried_paths), 10)
        self.assertTrue(set(completed).isdisjoint(retried_paths))
        self.assertEqual(self.opened, self.closed)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_seed_zero_baseline_choice_uses_current_seed_score_without_reselection(self):
        self.complete_seed_zero()
        # Seed 0 selects the replacement RNA representation. Seed 1 would prefer
        # original DNA if its validation scores were incorrectly used to reselect.
        for seed in (0, 1):
            for run in self.plan(seed).require_ready():
                if run.status != "reuse":
                    self.write_result(run.path, run.variant, run.config, run.encoders)
                if run.role == "original" and run.variant == "unimodal:dna":
                    self.write_result(run.path, run.variant, run.config, run.encoders,
                                      val_score=0.8 if seed == 0 else 0.99, test_score=0.88)
                if run.role == "replacement" and run.variant == "unimodal:rna":
                    self.write_result(run.path, run.variant, run.config, run.encoders,
                                      val_score=0.95 if seed == 0 else 0.02, test_score=0.23)
        env = self.namespace(1)
        self.execute_plan(env)
        env.update(swap_results=swap_results, ENCODERS={
            key: SimpleNamespace(label=key) for key in (*ORIGINAL.values(), "rinalmo")
        })
        env["train"].Result = SimpleNamespace(load=lambda path: SimpleNamespace(
            **json.loads(Path(path).read_text(encoding="utf-8"))))
        for cell_id in ("swaps-original-selectors", "swaps-result-selectors", "swaps-summary"):
            exec(compile(self.sources[cell_id], "fixture result selection", "exec"), env)
        current = env["triple"](BENCHMARK, "rna", "rinalmo")
        self.assertEqual(env["best_single"](current, "rna", "rinalmo"), "unimodal:dna")
        table = env["summary"](BENCHMARK)
        self.assertEqual(table.loc[("rna", "rinalmo"), "best single"], "unimodal:rna")
        self.assertEqual(table.loc[("rna", "rinalmo"), "best single score"], 0.23)
        self.assertEqual(self.calls, [])

    def test_partial_stream_open_preserves_primary_failure_and_closes_prior_caches(self):
        env = self.namespace()
        closed, attempted = [], []

        def summarize(path):
            attempted.append(path)
            if len(attempted) == 2:
                raise InjectedFailure("injected second cache failure")

            def close():
                closed.append(path)
                raise ValueError("injected cache-close failure")

            return SimpleNamespace(close=close)

        env.update(
            BENCHMARK_ROWS={BENCHMARK: object()},
            datasets=SimpleNamespace(blf_benchmark_rows=forbidden, DATASETS={BENCHMARK:
                SimpleNamespace(task="regression", label_column="Value", split_column="Split")}),
            tokens=SimpleNamespace(summarize=summarize),
            blf=SimpleNamespace(BenchmarkStreams=forbidden),
        )
        exec(compile(self.sources["swaps-training-defs"], "fixture cleanup definitions", "exec"), env)
        with self.assertRaisesRegex(InjectedFailure, "second cache") as caught:
            env["open_fit_streams"](BENCHMARK, ORIGINAL)
        self.assertEqual(closed, attempted[:1])
        self.assertTrue(any("cleanup" in note.lower() for note in caught.exception.__notes__))


if __name__ == "__main__":
    unittest.main()
