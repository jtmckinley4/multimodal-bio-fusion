"""Isolated notebook branches with invented paths/records; no research execution."""

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
from shared_code.fusion.result_sets import resolve_ernie_extension

ORIGINAL = dict(dna="ntv2_100m", rna="rnafm", prot="esm2_8m")
SWAPS = [("rna", "rnaernie"), ("rna", "ernierna")]
BENCHMARK = "ecoli_proteins"


def forbidden(*args, **kwargs):
    raise AssertionError("Synthetic branch reached a forbidden producer")


class FocusedNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        notebook = json.loads((ROOT / "Code/stage4/encoder_swaps.ipynb").read_text(encoding="utf-8"))
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        self.env = dict(
            EXPERIMENT="ecoli_ernie", READ_SAVED_RESULTS=False, RUN_TRAINING=False,
            ERNIE_ENCODERS=("rnaernie", "ernierna"),
            CACHE_REUSE_ONLY=False, CACHE_ROOT=self.root / "old", NEW_CACHE_ROOT=self.root / "new",
            REFERENCE_SWAP_ROOT=self.root / "old_results", RESULTS_ROOT=self.root / "extension",
            ORIGINAL=ORIGINAL, SWAPS=SWAPS, STABILITY_SWAPS=SWAPS, BENCHMARKS=(BENCHMARK,),
            SEED=0, CHUNK_SIZE=256, DTYPE=np.float16, DEVICE="cpu", pd=pd, np=np,
            json=json, time=time, display=lambda value: None,
            tokens=SimpleNamespace(DEFAULT_CHUNK=256, extract=forbidden, inspect_cache=forbidden,
                                   CacheVerification=lambda: object(), CacheRecoveryRequired=RuntimeError,
                                   inputs_for_rows=lambda encoder, rows: rows),
            ENCODERS={key: SimpleNamespace(label=key) for key in (*ORIGINAL.values(), "rinalmo", "rnaernie", "ernierna")},
            datasets=SimpleNamespace(blf_benchmark_rows=lambda key: pd.DataFrame({"sequence": ["ATG"]})),
        )
        self.execute("swaps-selection")

    def execute(self, cell_id):
        with redirect_stdout(io.StringIO()):
            exec(compile(self.cells[cell_id], cell_id, "exec"), self.env)

    def test_extension_settings_keep_budget_and_request_only_two_encoders(self):
        self.env["blf"] = SimpleNamespace(ENCODER_OF_MODALITY=ORIGINAL)
        self.execute("swaps-settings")
        self.assertEqual(self.env["SWAPS"], SWAPS)
        self.assertEqual((self.env["MAX_EPOCHS"], self.env["BATCH_SIZE"], self.env["CHANNELS"]), (500, 32, 100))
        self.assertTrue(self.env["CACHE_REUSE_ONLY"])
        self.assertFalse(self.env["RUN_SAVED_COMPARISONS"])

    def test_cache_routing_does_not_request_original_rna(self):
        self.assertEqual(self.env["required_encoders"](BENCHMARK),
                         ["ntv2_100m", "esm2_8m", "rnaernie", "ernierna"])
        for key in ("ntv2_100m", "esm2_8m"):
            self.assertEqual(self.env["cache_directory"](BENCHMARK, key), self.root / "old" / BENCHMARK / key)
            self.assertTrue(self.env["cache_is_read_only"](key))
        for key in ("rnaernie", "ernierna"):
            self.assertEqual(self.env["cache_directory"](BENCHMARK, key), self.root / "new" / BENCHMARK / key)
            self.assertFalse(self.env["cache_is_read_only"](key))

    def test_missing_original_cache_blocks_even_in_extraction_mode(self):
        self.env["PENDING_CACHE_KEYS"] = {(BENCHMARK, "ntv2_100m")}
        self.env["tokens"].inspect_cache = lambda *a, **kw: SimpleNamespace(
            ready=True, missing_shards=[0], restore_metadata=False, reusable_shards=[],
            status="missing", provenance="missing", issues=[], summary="missing original",
        )
        with self.assertRaisesRegex(RuntimeError, "missing original"):
            self.execute("swaps-cache-preflight")

    def test_extraction_visits_only_new_caches(self):
        self.env.update(PENDING_CACHE_KEYS={(BENCHMARK, k) for k in self.env["required_encoders"](BENCHMARK)},
                        BENCHMARK_ROWS={BENCHMARK: pd.DataFrame({"sequence": ["ATG"]})},
                        RECOVERY_RECORDS={}, CACHE_VERIFICATION=object())
        calls, closed = [], []

        class Cache:
            width = 768
            meta = dict(total_tokens=3, bytes=4608)

            def __len__(self):
                return 1

            def close(self):
                closed.append(self)

        def extract(encoder, texts, directory, device, **kwargs):
            calls.append((encoder.label, directory))
            return Cache(), True

        self.env["tokens"].extract = extract
        self.execute("c397d229")
        self.assertEqual([key for key, _ in calls], ["rnaernie", "ernierna"])
        self.assertTrue(all(path.is_relative_to(self.root / "new") for _, path in calls))
        self.assertEqual(len(closed), 2)

    def test_extract_only_never_reaches_fit_planner_or_training(self):
        self.env.update(execution_plan=forbidden, run_pending_group=forbidden)
        self.execute("fc1f77cc")

    def test_gated_for_both_encoders_precedes_concat(self):
        runs = [SimpleNamespace(role="replacement", encoder_key=key, variant=head)
                for key in ("rnaernie", "ernierna") for head in ("mil", "concat")]
        plan = SimpleNamespace(runs=runs, require_ready=lambda: runs)
        calls = []
        self.env.update(RUN_TRAINING=True, execution_plan=lambda key: plan,
                        run_pending_group=lambda b, key, selected: calls.extend((r.encoder_key, r.variant) for r in selected))
        self.execute("fc1f77cc")
        self.assertEqual(calls, [("rnaernie", "mil"), ("ernierna", "mil"),
                                 ("rnaernie", "concat"), ("ernierna", "concat")])

    def test_single_selection_remains_unavailable(self):
        self.execute("swaps-single-selection")
        self.assertIsNone(self.env["validation_selected_single"]({"unimodal:dna": object()}, benchmark=BENCHMARK))

    def test_focused_writable_display_uses_one_seed_and_leaves_missing_scores_blank(self):
        def reader(folders, **kwargs):
            records = ([{"data": {"variant": "mil", "test_score": 0.6}}]
                       if kwargs["encoders"]["rna"] == "rinalmo" else [])
            return SimpleNamespace(require_unique=lambda: records)

        displayed = []
        self.env.update(swap_results=SimpleNamespace(read_results=reader), display=displayed.append,
                        SEED=2, resolve_ernie_extension=forbidden,
                        selected_mil_penalty=lambda b: 0.01, result_recipe=lambda seed: {"seed": seed})
        self.execute("swaps-summary")
        self.execute("swaps-ernie-table")
        table, = displayed
        self.assertEqual(list(table.columns), ["Seed 2 (%)"])
        self.assertEqual(table.loc[("rinalmo", "MIL gated", "existing RiNALMo reference"), "Seed 2 (%)"], 60)
        self.assertTrue(pd.isna(table.loc[("rnaernie", "MIL gated", "focused extension"), "Seed 2 (%)"]))
        self.assertTrue(pd.isna(table.loc[("ernierna", "Concatenation", "focused extension"), "Seed 2 (%)"]))

    def test_saved_table_shows_all_three_seeds_without_averaging_partial_rows(self):
        def reader(folders, *, benchmark, config, encoders, mil_lambda, variants):
            encoder, seed = encoders["rna"], config["seed"]
            expected_root = self.env["REFERENCE_SWAP_ROOT"] if encoder == "rinalmo" else self.env["RESULTS_ROOT"]
            self.assertEqual(folders, [expected_root / BENCHMARK / encoder])
            self.assertEqual((benchmark, mil_lambda, variants), (BENCHMARK, 0.01, ("mil", "concat")))
            # One extension's middle concatenation seed is absent. Its gated fit is still complete.
            records = [{"data": {"variant": v, "test_score": (0.6 if v == "mil" else 0.7) + 0.05 * seed}}
                       for v in variants if (encoder, v, seed) != ("ernierna", "concat", 1)]
            return SimpleNamespace(require_unique=lambda: records)

        displayed = []
        self.env.update(READ_SAVED_RESULTS=True, SEED=2, SCALE={BENCHMARK: 100},
                        resolve_ernie_extension=forbidden, benchmark_score_table=forbidden,
                        swap_results=SimpleNamespace(read_results=reader), display=displayed.append,
                        selected_mil_penalty=lambda b: 0.01, result_recipe=lambda seed: {"seed": seed})
        self.execute("swaps-summary")
        self.execute("bcddefb4")
        self.execute("swaps-ernie-table")
        table, = displayed
        self.assertEqual(table.shape, (6, 4))
        self.assertEqual(list(table.columns), ["Seed 0 (%)", "Seed 1 (%)", "Seed 2 (%)", "Mean (%)"])
        self.assertEqual(table.loc[("rnaernie", "MIL gated", "focused extension")].tolist(), [60, 65, 70, 65])
        partial = table.loc[("ernierna", "Concatenation", "focused extension")]
        self.assertEqual((partial["Seed 0 (%)"], partial["Seed 2 (%)"]), (70, 80))
        self.assertTrue(partial[["Seed 1 (%)", "Mean (%)"]].isna().all())

    def test_default_newton_reader_displays_only_the_two_extensions_and_their_reference(self):
        code = self.root / "Code"
        extension = resolve_ernie_extension(code).swaps
        full_panel = self.root / "original Newton" / "outputs" / "phaseB"
        received = set()

        def reader(folders, *, benchmark, config, encoders, mil_lambda, variants):
            key, seed = encoders["rna"], config["seed"]
            self.assertEqual(benchmark, BENCHMARK)
            self.assertEqual(folders, [(full_panel if key == "rinalmo" else extension) / benchmark / key])
            received.add((key, seed))
            records = [{"data": {"variant": v, "test_score": 0.625}} for v in variants]
            return SimpleNamespace(require_unique=lambda: records)

        shown = []
        self.env.update(EXPERIMENT="full", READ_SAVED_RESULTS=True,
                        CODE_DIR=code, RETURNED_WORKSPACE=None, REFERENCE_SWAP_ROOT=full_panel,
                        RESULTS_ROOT=full_panel, RESULT_PATHS=SimpleNamespace(name="newton_h100_seeds012"),
                        resolve_ernie_extension=resolve_ernie_extension, blf=SimpleNamespace(ENCODER_OF_MODALITY=ORIGINAL),
                        swap_results=SimpleNamespace(read_results=reader), display=shown.append,
                        selected_mil_penalty=lambda b: 0.01, result_recipe=lambda seed: {"seed": seed})
        self.execute("swaps-settings")
        self.assertNotIn(("rna", "rnaernie"), self.env["SWAPS"])
        self.execute("swaps-summary")
        self.execute("swaps-ernie-table")
        table, = shown
        self.assertEqual(table.shape, (6, 4))
        self.assertEqual(received, {(k, s) for k in ("rinalmo", "rnaernie", "ernierna") for s in (0, 1, 2)})

    def test_other_full_panel_scopes_do_not_read_the_bundled_extension(self):
        for name, workspace, reading, benchmarks in [
            ("historical", None, True, (BENCHMARK,)),
            ("newton_h100_seeds012", self.root / "another return", True, (BENCHMARK,)),
            ("newton_h100_seeds012", None, False, (BENCHMARK,)),
            ("newton_h100_seeds012", None, True, ("fungal_expression",)),
        ]:
            with self.subTest(name=name, workspace=workspace, reading=reading, benchmarks=benchmarks):
                self.env.update(EXPERIMENT="full", READ_SAVED_RESULTS=reading, BENCHMARKS=benchmarks,
                                RESULT_PATHS=SimpleNamespace(name=name), RETURNED_WORKSPACE=workspace,
                                resolve_ernie_extension=forbidden, extension_score_table=forbidden, display=forbidden)
                self.execute("swaps-ernie-table")

    def test_extension_saved_report_branch_does_not_compute(self):
        self.env.update(READ_SAVED_RESULTS=True, RUN_SAVED_COMPARISONS=False, build_or_reuse_report=forbidden)
        self.execute("swaps-paired-report-run")
        self.assertEqual(self.env["SAVED_COMPARISON_REPORTS"], {})

    def test_returned_extension_is_immutable_and_cannot_overlap_old_results(self):
        code = self.root / "Code"
        reference = code / "stage4/results/newton_h100_seeds012"
        resolved = resolve_ernie_extension(code)
        self.assertEqual(resolved.swaps, resolved.root / "outputs/ecoli_ernie/phaseB")
        self.assertEqual(resolved.originals[BENCHMARK], (reference / "outputs/phaseA" / BENCHMARK,))
        with self.assertRaisesRegex(ValueError, "read-only"):
            resolved.require_writable_fits()
        for conflict in (reference, reference / "outputs", reference.parent):
            with self.subTest(conflict=conflict), self.assertRaisesRegex(ValueError, "overlaps"):
                resolve_ernie_extension(code, workspace=conflict)


if __name__ == "__main__":
    unittest.main()
