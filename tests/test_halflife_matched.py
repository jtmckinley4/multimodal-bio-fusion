"""Exercise the matched follow-up with invented tables and temporary caches only."""

import ast
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
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
from shared_code import mrnabench as MB


def forbidden(*args, **kwargs):
    raise AssertionError("A matched fixture attempted extraction, download or historical fitting")


class MatchedNotebookTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "Code/stage4/tier3_halflife_te.ipynb"
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.cells = {c["id"]: "".join(c["source"])
                      for c in doc["cells"] if c["cell_type"] == "code"}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.frames = {}
        self.report = {"scores": [], "comparisons": [], "summary": []}
        self.runner = Mock(return_value=(self.report, False))
        self.reader = Mock(return_value=self.report)
        self.env = {
            "np": np, "pd": pd, "Path": Path, "hashlib": hashlib, "CODE_DIR": ROOT / "Code",
            "mrnabench": MB, "ENCODERS": MB.ENCODERS,
            "READ_SAVED_RESULTS": True, "RUN_MATCHED_PROBES": False,
            "MATCHED_PRIMARY": ("rnahl-human",),
            "MATCHED_ALPHAS": (1e-3, 1e-2, 1e-1, 1, 10),
            "MATCHED_BOOTSTRAP_DRAWS": 20, "MATCHED_BOOTSTRAP_SEED": 0,
            "PROBE_SEEDS": (2541, 2547, 413, 412, 411, 321, 421, 2515, 2516),
            "DATA_CACHE": self.folder / "data", "EMBED_DIR": self.folder / "embeddings",
            "MATCHED_RESULTS_DIR": self.folder / "results", "display": lambda value: None,
            "matched_probes": SimpleNamespace(
                run=self.runner, read=self.reader,
                notebook_source=lambda path, cells: {"fixture_source_cells": list(cells)},
            ),
        }
        for cell in ("95d7593a", "183c5b74", "matched-local-inputs", "matched-cache-matrices"):
            exec(compile(self.cells[cell], f"fixture:{cell}", "exec"), self.env)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(MB, "MINIMUM_ROWS", 100))
        stack.enter_context(patch.object(MB, "download", forbidden))
        stack.enter_context(patch.object(MB, "load", forbidden))
        stack.enter_context(patch.object(MB, "embed_with_cache", forbidden))
        stack.enter_context(patch.object(MB, "probe_over_seeds", forbidden))
        stack.enter_context(patch.object(pd, "read_parquet", self.read_parquet))

    def read_parquet(self, path):
        return self.frames[Path(path).stem].copy(deep=True)

    def prepare(self, key="rnahl-human", *, missing_cache=None, invalid_first=False):
        n = 240
        track = np.zeros(19, dtype=np.int8)
        track[5:14:3] = 1
        frame = pd.DataFrame({
            "sequence": ["AAAAAATGAAATAGCCCCC"] * n,
            "cds": [track.copy() for _ in range(n)],
            "gene": [f"gene{i // 2}" for i in range(n)],
            "target": np.arange(n, dtype=float),
        })
        if invalid_first:
            frame.at[0, "cds"] = np.zeros(19, dtype=np.int8)
        self.frames[key] = frame
        self.env["DATA_CACHE"].mkdir(exist_ok=True)
        (self.env["DATA_CACHE"] / f"{key}.parquet").write_bytes(b"synthetic table fixture")
        regions, _ = self.env["load_matched_regions"]((key,))
        regions = regions[key]
        folder = self.env["EMBED_DIR"] / key
        folder.mkdir(parents=True, exist_ok=True)
        for name, modality, parts in (
            ("full transcript", "rna", ("utr5", "cds", "utr3")),
            ("protein of the CDS", "prot", ("cds",)),
        ):
            if name == missing_cache:
                continue
            encoder = MB.ENCODERS[self.env["STREAM_ENCODERS"][modality]]
            texts = self.env["stream_texts"](regions, modality, parts)
            matrix = np.column_stack([np.arange(len(regions)), np.ones(len(regions))])
            np.savez(
                folder / f"{name.replace(' ', '_')}.npz", embeddings=matrix,
                cache_version=1, encoder=encoder.key, checkpoint=encoder.checkpoint,
                revision=encoder.revision or "", window=MB.default_window(encoder),
                n_sequences=len(texts), fingerprint=MB.sequences_fingerprint(texts),
            )
        return regions

    def run_cell(self, cell="matched-probe-run"):
        with redirect_stdout(io.StringIO()):
            exec(compile(self.cells[cell], f"fixture:{cell}", "exec"), self.env)

    def test_default_off_reads_no_data_or_caches_and_creates_no_files(self):
        self.run_cell()
        self.runner.assert_not_called()
        self.reader.assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_saved_report_read_does_not_open_inputs_or_fit(self):
        folder = self.env["MATCHED_RESULTS_DIR"]
        folder.mkdir()
        path = folder / "rnahl-human__target.json"
        path.write_text("{}", encoding="utf-8")
        self.run_cell()
        self.reader.assert_called_once_with(path)
        self.runner.assert_not_called()

    def test_enabled_route_refuses_missing_local_data(self):
        self.env["RUN_MATCHED_PROBES"] = True
        with self.assertRaisesRegex(FileNotFoundError, "no download attempted"):
            self.run_cell()
        self.runner.assert_not_called()

    def test_all_selected_caches_are_checked_before_any_fit(self):
        self.prepare()
        self.prepare("translation-efficiency-human", missing_cache="protein of the CDS")
        self.env.update(RUN_MATCHED_PROBES=True,
                        MATCHED_PRIMARY=("rnahl-human", "translation-efficiency-human"))
        with self.assertRaisesRegex(FileNotFoundError, "no extraction attempted"):
            self.run_cell()
        self.runner.assert_not_called()

    def test_changed_cache_input_metadata_is_rejected_without_replacement(self):
        self.prepare()
        path = self.env["EMBED_DIR"] / "rnahl-human/full_transcript.npz"
        with np.load(path, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
        fields["fingerprint"] = "0" * 64
        np.savez(path, **fields)
        before = path.read_bytes()
        self.env["RUN_MATCHED_PROBES"] = True
        with self.assertRaisesRegex(ValueError, "different run"):
            self.run_cell()
        self.runner.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_fixed_inputs_common_partitions_and_source_row_ids_reach_helper(self):
        regions = self.prepare(invalid_first=True)
        self.env["RUN_MATCHED_PROBES"] = True
        self.run_cell()
        self.runner.assert_called_once()
        args, kwargs = self.runner.call_args
        self.assertEqual(args[0].parent, self.env["MATCHED_RESULTS_DIR"])
        self.assertEqual(set(kwargs["matrices"]), {
            "full transcript", "protein of the CDS", "full transcript + protein", "composition",
        })
        joined = np.concatenate([kwargs["matrices"]["full transcript"],
                                 kwargs["matrices"]["protein of the CDS"]], axis=1)
        np.testing.assert_array_equal(kwargs["matrices"]["full transcript + protein"], joined)
        self.assertEqual(kwargs["row_ids"][0], "rnahl-human:1")
        np.testing.assert_array_equal(kwargs["y"], regions["target"])
        self.assertEqual(tuple(kwargs["splits"]), tuple(map(str, self.env["PROBE_SEEDS"])))
        self.assertEqual(kwargs["metric"], "pearson")
        self.assertTrue(kwargs["require_disjoint_groups"])
        self.assertEqual(len(kwargs["contrasts"]), 3)
        for train, validation, test in kwargs["splits"].values():
            groups = kwargs["groups"]
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(set(kwargs["evidence"]["caches"]),
                         {"full transcript", "protein of the CDS"})

    def test_bad_flags_fail_in_setup_before_old_producer_cells(self):
        tree = ast.parse(self.cells["6880b3af"])
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                name = statement.targets[0]
                if isinstance(name, ast.Name) and name.id in {"READ_SAVED_RESULTS", "RUN_MATCHED_PROBES"}:
                    statement.value = ast.Constant(name.id == "RUN_MATCHED_PROBES")
        ast.fix_missing_locations(tree)
        with self.assertRaisesRegex(ValueError, "protect historical outputs"):
            exec(compile(tree, "fixture:setup", "exec"), self.env)
        self.runner.assert_not_called()

    def test_invalid_target_in_other_dataset_blocks_whole_batch(self):
        self.prepare()
        self.prepare("translation-efficiency-human")
        self.frames["translation-efficiency-human"].at[0, "target"] = np.nan
        self.env.update(RUN_MATCHED_PROBES=True,
                        MATCHED_PRIMARY=("rnahl-human", "translation-efficiency-human"))
        with self.assertRaisesRegex(ValueError, "translation-efficiency-human"):
            self.run_cell()
        self.runner.assert_not_called()

    def test_duplicate_partition_seeds_are_not_silently_collapsed(self):
        self.env.update(RUN_MATCHED_PROBES=True, PROBE_SEEDS=(1, 1))
        with self.assertRaisesRegex(ValueError, "distinct split-assignment"):
            self.run_cell()
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
