"""Check Stage 1 setup and global names without running the research bodies."""

import ast
import builtins
from contextlib import ExitStack, redirect_stdout
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import re
import socket
import symtable
import sys
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1] / "Code"
sys.path.insert(0, str(CODE))

# Preserve the notebook's NumPy-before-PyTorch import order on Windows.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import pandas as pd
import torch
from mbf import analysis, datasets, embeddings, encoders

NOTEBOOKS = ("Stage1_stability.ipynb", "Stage1_gtex.ipynb")


def read_notebook(name):
    return json.loads((CODE / name).read_text(encoding="utf-8"))["cells"]


def source(cell):
    return "".join(cell["source"])


def setup_cells(cells):
    """Require one Setup section and a following section before executing code."""
    headings = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        # Headings inside fenced examples are not notebook section boundaries.
        fence = None
        for line in source(cell).splitlines():
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                run = marker[1]
                if fence is None:
                    fence = run
                elif run[0] == fence[0] and len(run) >= len(fence):
                    fence = None
            elif fence is None:
                heading = re.match(r"^(#{1,2})\s+(.+?)\s*#*\s*$", line)
                if heading:
                    headings.append((index, heading[1], heading[2]))
    starts = [i for i, (_, level, title) in enumerate(headings)
              if level == "##" and title == "Setup"]
    if len(starts) != 1 or starts[0] + 1 >= len(headings):
        raise AssertionError("Require exactly one ## Setup and a following section")
    start, _, _ = headings[starts[0]]
    stop, _, _ = headings[starts[0] + 1]
    selected = [(i, c) for i, c in enumerate(cells[start + 1:stop], start + 1)
                if c["cell_type"] == "code"]
    if not selected:
        raise AssertionError("Setup contains no code cells")
    return selected


def global_references(table):
    names = {s.get_name() for s in table.get_symbols()
             if s.is_referenced() and (table.get_type() == "module" or s.is_global())}
    for child in table.get_children():
        names |= global_references(child)
    return names


def missing_globals(cells, name):
    """Conservative top-to-bottom scan, not branch or runtime validation.

    Names assigned anywhere in a compound statement are treated as available in
    that statement. Function bodies are checked where defined, not where called.
    """
    available = set(dir(builtins)) | {"__name__"}
    missing = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        location = f"{name}:cell {index + 1}"
        tree = ast.parse(source(cell), filename=location)
        compile(tree, location, "exec")
        for statement in tree.body:
            table = symtable.symtable(ast.unparse(statement), location, "exec")
            assigned = {s.get_name() for s in table.get_symbols()
                        if s.is_assigned() or s.is_imported()}
            unresolved = global_references(table) - available - assigned
            if unresolved:
                missing.append(f"{location}, line {statement.lineno}: {sorted(unresolved)}")
            available |= assigned
    return missing


class NotebookSetupTests(unittest.TestCase):
    def execute_setup(self, name):
        cells = setup_cells(read_notebook(name))
        namespace = {"__name__": "__main__"}
        called = []

        def forbidden(label):
            def fail(*args, **kwargs):
                called.append(label)
                raise AssertionError(f"Research/data/network operation in Setup: {label}")
            return fail

        with ExitStack() as guards, redirect_stdout(io.StringIO()):
            seeding = guards.enter_context(patch.object(torch, "manual_seed", wraps=torch.manual_seed))
            # Registries remain readable; computation and data entry points fail.
            for module in (analysis, datasets, embeddings):
                for key, value in list(vars(module).items()):
                    if inspect.isfunction(value) and value.__module__ == module.__name__:
                        guards.enter_context(patch.object(module, key, forbidden(f"{module.__name__}.{key}")))
            for key in ("load_encoder", "load_tokenizer", "_load_remote_model", "_load_checkpoint_state"):
                guards.enter_context(patch.object(encoders, key, forbidden(f"encoders.{key}")))
            for owner, key in ((embeddings, "load_encoder"), (pd, "read_csv"),
                               (np, "load"), (np, "save"), (np, "savez"),
                               (torch, "load"), (socket.socket, "connect"),
                               (socket, "create_connection")):
                guards.enter_context(patch.object(owner, key, forbidden(f"{owner.__name__}.{key}")))
            for index, cell in cells:
                exec(compile(source(cell), f"{name}:cell {index + 1}", "exec"), namespace)
        self.assertEqual(called, [], "Setup caught a blocked operation instead of avoiding it")
        seeding.assert_any_call(namespace["SEED"])
        return namespace

    def check_common_settings(self, values):
        keys = values["ENCODER_KEYS"]
        self.assertTrue(keys)
        self.assertEqual(len(keys), len(set(keys)), "Encoder keys must be unique")
        self.assertTrue(set(keys) <= encoders.ENCODERS.keys())
        self.assertEqual([e.key for e in values["ENCODERS"]], keys)
        self.assertEqual(set(values["LABELS"]), set(keys))
        self.assertEqual(set(values["MODALITY"]), set(keys))
        self.assertTrue(set(values["REFERENCE_KEYS"]) <= set(keys))
        self.assertTrue(set(values["ATTENTION_ENCODERS"]) <= set(keys))
        self.assertIsInstance(values["SEED"], int)
        self.assertEqual(torch.initial_seed(), values["SEED"])
        self.assertIs(values["SHOW_IMPLEMENTATION"], False)
        self.assertIn(str(values["DEVICE"]).split(":")[0], {"cpu", "cuda", "mps"})

    def test_stability_setup_is_independent(self):
        values = self.execute_setup("Stage1_stability.ipynb")
        self.check_common_settings(values)
        self.assertIn(values["DATASET"], datasets.DATASETS)
        self.assertIn(values["CONTROL_DATASET"], datasets.DATASETS)
        self.assertGreater(values["N_ROWS"], 0)
        self.assertNotEqual(values["EMBEDDING_DIR"], values["CONTROL_EMBEDDING_DIR"])
        self.assertNotEqual(values["BENCHMARK_TRAIN_DIR"], values["BENCHMARK_TEST_DIR"])

    def test_gtex_setup_is_independent(self):
        values = self.execute_setup("Stage1_gtex.ipynb")
        self.check_common_settings(values)
        self.assertEqual(values["PAIRS"], list(itertools.combinations(values["ENCODER_KEYS"], 2)))
        self.assertIn(values["GTEX_DATASET"], datasets.DATASETS)
        self.assertNotEqual(values["GTEX_TRAIN_DIR"], values["GTEX_TEST_DIR"])
        self.assertTrue(set(values["UTR_ATTENTION_ENCODERS"]) <= set(values["ATTENTION_ENCODERS"]))
        for key in values["UTR_ATTENTION_ENCODERS"]:
            self.assertEqual(encoders.ENCODERS[key].modality, "RNA")
            self.assertFalse(encoders.ENCODERS[key].codon)

    def test_analysis_cells_compile_and_names_are_supplied_locally(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                self.assertEqual(missing_globals(read_notebook(name), name), [])


if __name__ == "__main__":
    unittest.main()
