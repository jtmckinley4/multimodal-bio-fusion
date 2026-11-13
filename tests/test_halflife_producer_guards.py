"""Isolated producer-boundary fixtures with extraction/fitting/writing replaced by stubs."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import mrnabench as MB

CALCULATION_CELLS = {
    "8911ff86": "halflife-gate-calculation",
    "d3c29f9e": "halflife-baseline-calculation",
    "fc21afe3": "halflife-encoder-calculation",
    "398d1d5f": "halflife-gains-calculation",
}


def forbidden(*args, **kwargs):
    raise AssertionError("A fixture reached a real load or writer")


class ProducerGuardTests(unittest.TestCase):
    def setUp(self):
        doc = json.loads((ROOT / "Code/stage4/halflife_translation_efficiency.ipynb").read_text(encoding="utf-8"))
        self.cells = {c["id"]: "".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code"}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.calls, self.writes = [], []
        n = 120
        track = np.zeros(19, dtype=np.int8)
        track[5:14:3] = 1
        self.frame = pd.DataFrame({
            "sequence": ["AAAAAATGAAATAGCCCCC"] * n,
            "cds": [track.copy() for _ in range(n)],
            "gene": [f"gene{i}" for i in range(n)], "target": np.arange(n, dtype=float),
        })
        self.env = dict(
            READ_SAVED_RESULTS=False, RUN_EMBEDDINGS=True, PRIMARY=MB.HALF_LIFE,
            frames={key: self.frame.copy(deep=True) for key in MB.HALF_LIFE},
            usable=list(MB.HALF_LIFE), PROBE_SEEDS=(3, 7, 11, 13),
            RESULTS_DIR=self.folder, EMBED_DIR=self.folder, DATA_CACHE=self.folder,
            DEVICE="cpu", mrnabench=MB, np=np, pd=pd, ENCODERS=MB.ENCODERS,
            time=SimpleNamespace(perf_counter=lambda: 0), display=lambda value: None,
        )
        for cell in ("530b93c8", "95d7593a", "183c5b74"):
            exec(compile(self.cells[cell], f"fixture:{cell}", "exec"), self.env)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(MB, "MINIMUM_ROWS", 100))
        self.stack.enter_context(patch.object(MB, "load", forbidden))
        self.stack.enter_context(patch.object(MB, "grouping", self.grouping))
        self.stack.enter_context(patch.object(MB, "region_features", self.features))
        self.stack.enter_context(patch.object(MB, "embed_with_cache", self.embed))
        self.stack.enter_context(patch.object(MB, "probe_over_seeds", self.probe))
        self.stack.enter_context(patch.object(Path, "mkdir", forbidden))
        self.stack.enter_context(patch.object(pd.DataFrame, "to_csv", self.write))

    def grouping(self, regions, species, cache_dir):
        return regions["gene"].to_numpy(), "invented groups"

    def features(self, regions, parts):
        self.calls.append(("features", tuple(parts)))
        return np.column_stack([np.arange(len(regions)), np.ones(len(regions))])

    def embed(self, encoder, texts, path, device):
        self.calls.append(("embed", encoder, list(texts), path, device))
        stream = sum(call[0] == "embed" for call in self.calls)
        return np.column_stack([np.full(len(texts), stream), np.arange(len(texts))]), True

    def probe(self, X, y, groups, seeds):
        self.calls.append(("probe", X.copy(), y.copy(), groups.copy(), tuple(seeds)))
        return pd.DataFrame({"test_r": [0.4] * len(seeds), "test_spearman": [0.3] * len(seeds)})

    def write(self, path, **kwargs):
        self.writes.append(Path(path).name)

    def run_cell(self, cell):
        # Keep each reader and its separately guarded producer in notebook order.
        selected = [cell]
        if cell in CALCULATION_CELLS:
            selected.append(CALCULATION_CELLS[cell])
        with redirect_stdout(io.StringIO()):
            for cell_id in selected:
                exec(compile(self.cells[cell_id], f"fixture:{cell_id}", "exec"), self.env)

    def test_separate_calculation_cells_skip_without_inputs_in_saved_mode(self):
        for cell in CALCULATION_CELLS.values():
            with self.subTest(cell=cell):
                # No data, models or writer names are available if a guard fails.
                env = {"READ_SAVED_RESULTS": True}
                exec(compile(self.cells[cell], f"fixture:{cell}", "exec"), env)
                self.assertEqual(set(env) - {"__builtins__"}, {"READ_SAVED_RESULTS"})

    def test_every_producer_refuses_a_mixed_valid_invalid_cohort_before_work(self):
        self.env["frames"]["rnahl-mouse"].at[0, "target"] = np.nan
        for cell in ("9f809fa4", "d3c29f9e", "bc1b7b69", "6260d5de", "fc21afe3", "398d1d5f"):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, "rnahl-mouse"):
                self.run_cell(cell)
            self.assertEqual(self.calls, [])
            self.assertEqual(self.writes, [])

    def test_source_example_uses_the_inspected_frame_without_loading(self):
        self.run_cell("9f809fa4")
        expected = [("features", tuple(parts)) for parts in self.env["REGION_SETS"].values()]
        self.assertEqual(self.calls, expected)
        self.assertEqual(self.writes, [])

    def test_valid_pipeline_preserves_argument_order_and_declared_seeds(self):
        self.run_cell("d3c29f9e")
        baseline_calls = [call for call in self.calls if call[0] == "probe"]
        self.assertEqual(len(baseline_calls), len(MB.HALF_LIFE) * len(self.env["REGION_SETS"]))
        self.assertTrue(all(call[-1] == (3, 7, 11) for call in baseline_calls))
        self.calls.clear()
        self.run_cell("6260d5de")
        expected_paths = [key for key in self.env["PRIMARY"] for _ in self.env["INPUTS"]]
        self.assertEqual([call[3].parent.name for call in self.calls], expected_paths)
        expected_streams = (
            ("rnafm", "AAAAA", "5_UTR.npz"),
            ("rnafm", "AUGAAAUAG", "CDS.npz"),
            ("rnafm", "AAAAAAUGAAAUAGCCCCC", "full_transcript.npz"),
            ("ntv2_100m", "AAAAAATGAAATAGCCCCC", "full_transcript_DNA_encoder.npz"),
            ("esm2_8m", "MK", "protein_of_the_CDS.npz"),
        )
        for call, (encoder, text, filename) in zip(self.calls, expected_streams * 2):
            self.assertEqual((call[1], call[2], call[3].name, call[4]),
                             (encoder, [text] * len(self.frame), filename, "cpu"))
        self.run_cell("fc21afe3")
        probes = [call for call in self.calls if call[0] == "probe"]
        self.assertEqual(len(probes), 14)
        self.assertEqual([call[1].shape[1] for call in probes], [2, 2, 2, 2, 2, 4, 6] * 2)
        stream_order = ((1,), (2,), (3,), (4,), (5,), (3, 5), (4, 3, 5))
        expected_stream_order = [tuple(i + offset for i in order)
                                 for offset in (0, 5) for order in stream_order]
        for call, streams in zip(probes, expected_stream_order):
            expected = np.column_stack([
                column for stream in streams
                for column in (np.full(len(self.frame), stream), np.arange(len(self.frame)))
            ])
            np.testing.assert_array_equal(call[1], expected)
            np.testing.assert_array_equal(call[2], self.frame["target"].to_numpy())
            np.testing.assert_array_equal(call[3], self.frame["gene"].to_numpy())
            self.assertEqual(call[4], (3, 7, 11, 13))
        self.run_cell("398d1d5f")
        self.assertEqual(self.writes, ["composition_baselines.csv", "encoder_probes.csv", "fusion_gains.csv"])
        self.assertEqual(len(self.env["gains"]), 4)

    def test_interrupted_extraction_cannot_leave_old_or_partial_handoffs(self):
        self.env.update(
            embedded={"old": object()}, embedding_state="old",
            encoder_scores=pd.DataFrame({"old": [1]}), score_state={"old": "old"},
        )
        count = 0
        def interrupted(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            return self.embed(*args, **kwargs)
        with patch.object(MB, "embed_with_cache", interrupted), self.assertRaises(KeyboardInterrupt):
            self.run_cell("6260d5de")
        self.assertEqual(self.env["embedded"], {})
        self.assertIsNone(self.env["embedding_state"])
        self.assertIsNone(self.env["encoder_scores"])
        for cell in ("fc21afe3", "398d1d5f"):
            with self.assertRaises(ValueError):
                self.run_cell(cell)
        self.assertEqual(self.writes, [])

    def test_failed_probe_cannot_publish_gains_from_old_scores(self):
        self.run_cell("d3c29f9e")
        self.run_cell("6260d5de")
        self.env.update(encoder_scores=pd.DataFrame({"old": [1]}), score_state={"old": "old"})
        self.writes.clear()
        with patch.object(MB, "probe_over_seeds", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.run_cell("fc21afe3")
        self.assertIsNone(self.env["encoder_scores"])
        self.assertIsNone(self.env["score_state"])
        with self.assertRaises(ValueError):
            self.run_cell("398d1d5f")
        self.assertEqual(self.writes, [])

    def test_changed_targets_or_recipe_and_missing_matrices_stop_before_probes(self):
        self.run_cell("6260d5de")
        self.calls.clear()
        self.env["frames"]["rnahl-human"].at[0, "target"] = 0.25
        with self.assertRaisesRegex(ValueError, "current inputs/settings"):
            self.run_cell("fc21afe3")
        self.assertEqual(self.calls, [])
        self.env["frames"]["rnahl-human"].at[0, "target"] = 0.0
        self.env["STREAM_ENCODERS"]["rna"] = "mrnafm"
        with self.assertRaisesRegex(ValueError, "current inputs/settings"):
            self.run_cell("fc21afe3")
        self.assertEqual(self.calls, [])
        self.env["STREAM_ENCODERS"]["rna"] = "rnafm"
        del self.env["embedded"][("rnahl-human", "CDS")]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.run_cell("fc21afe3")
        self.assertEqual(self.calls, [])

    def test_changed_seed_or_grouping_refuses_gain_publication(self):
        self.run_cell("d3c29f9e")
        self.run_cell("6260d5de")
        self.run_cell("fc21afe3")
        self.writes.clear()
        self.env["PROBE_SEEDS"] = (3, 7, 11, 99)
        with self.assertRaisesRegex(ValueError, "stale inputs/settings"):
            self.run_cell("398d1d5f")
        self.assertEqual(self.writes, [])
        self.env["PROBE_SEEDS"] = (3, 7, 11, 13)
        def changed_groups(regions, *args):
            return np.repeat("changed", len(regions)), "changed groups"

        with patch.object(MB, "grouping", side_effect=changed_groups):
            with self.assertRaisesRegex(ValueError, "stale inputs/settings"):
                self.run_cell("398d1d5f")
        self.assertEqual(self.writes, [])

    def test_saved_table_reads_invalidate_previous_computation_handoffs(self):
        for cell, state in (("d3c29f9e", "baseline_state"), ("fc21afe3", "score_state")):
            with self.subTest(cell=cell):
                self.env["READ_SAVED_RESULTS"] = False
                self.run_cell("d3c29f9e")
                self.run_cell("6260d5de")
                self.run_cell("fc21afe3")
                self.env.update(READ_SAVED_RESULTS=True, gate=pd.DataFrame(),
                                saved_results=SimpleNamespace(read_historical_csv=lambda *a, **k: None))
                self.writes.clear()
                self.run_cell(cell)
                self.assertIsNone(self.env[state])
                self.env["READ_SAVED_RESULTS"] = False
                with self.assertRaisesRegex(ValueError, "completed current"):
                    self.run_cell("398d1d5f")
                self.assertEqual(self.writes, [])

    def test_failed_summary_write_leaves_no_completed_handoff(self):
        for cell, state in (("d3c29f9e", "baseline_state"), ("fc21afe3", "score_state")):
            with self.subTest(cell=cell):
                self.run_cell("d3c29f9e")
                self.run_cell("6260d5de")
                self.run_cell("fc21afe3")
                with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("fixture writer failure")):
                    with self.assertRaisesRegex(OSError, "fixture writer failure"):
                        self.run_cell(cell)
                self.assertIsNone(self.env[state])
                with self.assertRaisesRegex(ValueError, "completed current"):
                    self.run_cell("398d1d5f")

    def test_disabled_extraction_clears_prior_success_without_any_producer(self):
        self.env.update(
            RUN_EMBEDDINGS=False, embedded={"old": object()},
            embedding_state="old", score_state={"old": "old"},
        )
        for cell in ("6260d5de", "fc21afe3", "398d1d5f"):
            self.run_cell(cell)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.writes, [])
        self.assertEqual(self.env["embedded"], {})
        self.assertIsNone(self.env["score_state"])


if __name__ == "__main__":
    unittest.main()
