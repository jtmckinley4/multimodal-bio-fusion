"""Run Stage 1 reading control flow against invented files with producers blocked.

Every code cell is visited, including Setup and figure exports. These checks do
not execute the research branches or inspect the repository's research results.
"""

import base64
import builtins
from contextlib import ExitStack, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

# Preserve NumPy-before-PyTorch import order on Windows.
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
# Load display dependencies before the tests block file and network access.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import IPython.display as ipython_display

CODE = Path(__file__).resolve().parents[1] / "Code"
sys.path.insert(0, str(CODE))
from shared_code import analysis, datasets, embeddings, encoders, notebook, sequences, splits


NOTEBOOKS = ("stability", "gtex")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aH9kAAAAASUVORK5CYII="
)
FIGURE_FILES = {
    "stability": (
        "01_raw_selected_pairs.png", "02_codon_pair_comparison.png",
        "03_pca_variance_composition_probe.png", "04_layerwise_codon_heatmaps.png",
        "05_all_encoder_probes.png", "06_all_pairs_composition_control.png",
        "07_geometry_and_concatenation_gain.png", "08_whole_composition_vs_stability.png",
    ),
    "gtex": (
        "01_all_encoder_expression_probes.png", "02_pinned_baselines_and_encoders.png",
        "03_all_pairs_composition_control.png", "04_geometry_and_concatenation_gain.png",
    ),
}
FIRST_FIGURE = {dataset: files[0] for dataset, files in FIGURE_FILES.items()}
FIGURE_COUNTS = {dataset: len(files) for dataset, files in FIGURE_FILES.items()}


def code_cells(dataset):
    path = CODE / "stage1" / f"Stage1_{dataset}.ipynb"
    document = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in document["cells"] if cell["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"])


def files_under(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class Stage1ReadingModeTests(unittest.TestCase):
    def run_reading(self, dataset, root):
        cells = code_cells(dataset)
        blocked_calls, displayed, source_displays, opened = [], [], [], []
        output = io.StringIO()
        namespace = {"__name__": "__main__"}
        figure_dir = root / "figures"
        prefix = "STAGE1" if dataset == "stability" else "GTEX"
        options = {
            f"{prefix}_FIGURE_DIR": str(figure_dir),
            f"{prefix}_FIGURE_SOURCE": str(root / "snapshot.json"),
        }

        def forbidden(label):
            def fail(*args, **kwargs):
                blocked_calls.append(label)
                raise AssertionError(f"Producing or research-input operation in reading mode: {label}")
            return fail

        original_open = builtins.open

        def read_png_only(file, mode="r", *args, **kwargs):
            path = Path(file)
            if mode != "rb" or path.parent != figure_dir or path.suffix != ".png":
                return forbidden(f"open({path}, {mode})")()
            opened.append(path.name)
            return original_open(file, mode, *args, **kwargs)

        with ExitStack() as guards, redirect_stdout(output):
            guards.enter_context(patch.dict(os.environ, options))
            guards.enter_context(patch.object(notebook, "show_source", side_effect=lambda *a, **k: source_displays.append(a)))
            guards.enter_context(patch.object(ipython_display, "display", side_effect=lambda obj: displayed.append(obj)))
            for module in (analysis, datasets, embeddings, sequences, splits):
                for key, value in list(vars(module).items()):
                    if inspect.isfunction(value) and value.__module__ == module.__name__:
                        guards.enter_context(patch.object(module, key, forbidden(f"{module.__name__}.{key}")))
            for key in ("load_encoder", "load_tokenizer", "_load_remote_model", "_load_checkpoint_state"):
                guards.enter_context(patch.object(encoders, key, forbidden(f"encoders.{key}")))
            for owner, key in (
                (embeddings, "load_encoder"), (pd, "read_csv"), (pd.DataFrame, "to_csv"),
                (np, "load"), (np, "save"), (np, "savez"), (torch, "load"), (torch, "save"),
                (Figure, "savefig"), (Path, "mkdir"), (Path, "write_text"), (Path, "write_bytes"),
                (socket.socket, "connect"), (socket, "create_connection"),
            ):
                guards.enter_context(patch.object(owner, key, forbidden(f"{owner.__name__}.{key}")))
            guards.enter_context(patch.object(builtins, "open", side_effect=read_png_only))
            guards.enter_context(patch.object(io, "open", side_effect=read_png_only))
            for cell in cells:
                exec(compile(source(cell), f"{dataset}:{cell['id']}", "exec"), namespace)

        self.assertEqual(blocked_calls, [], "A caught forbidden call still violates reading mode")
        self.assertIs(namespace["READ_SAVED_RESULTS"], True)
        for name in ("dataset", "gtex_train", "EMBEDDINGS", "GTEX_EMBEDDINGS",
                     "figure_capture", "GTEX_FIGURE_SNAPSHOT", "figure_manifest", "gtex_figure_manifest"):
            self.assertNotIn(name, namespace, f"Reading mode unexpectedly created {name}")
        self.assertTrue(source_displays, "Implementation-display cells remain independently usable")
        return namespace, output.getvalue(), displayed, opened

    def test_all_cells_read_without_datasets_caches_snapshot_or_figure_directory(self):
        for dataset in NOTEBOOKS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                before = files_under(root)
                _, messages, displayed, opened = self.run_reading(dataset, root)
                self.assertEqual(displayed, [])
                self.assertEqual(opened, [])
                self.assertEqual(messages.count("Saved figure unavailable:"), FIGURE_COUNTS[dataset])
                self.assertIn("no live numeric result is reconstructed", messages)
                self.assertEqual(files_under(root), before)
                self.assertFalse((root / "figures").exists())

    def test_partial_figures_read_independently_of_missing_or_malformed_snapshot(self):
        for dataset in NOTEBOOKS:
            for snapshot in (None, "{this is not a valid numeric snapshot"):
                with self.subTest(dataset=dataset, snapshot=snapshot), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / "figures").mkdir()
                    (root / "figures" / FIRST_FIGURE[dataset]).write_bytes(PNG)
                    if snapshot is not None:
                        (root / "snapshot.json").write_text(snapshot, encoding="utf-8")
                    before = files_under(root)
                    _, messages, displayed, opened = self.run_reading(dataset, root)
                    self.assertEqual(len(displayed), 1)
                    self.assertEqual(displayed[0].data, PNG)
                    self.assertEqual(opened, [FIRST_FIGURE[dataset]])
                    self.assertEqual(messages.count("Saved figure unavailable:"), FIGURE_COUNTS[dataset] - 1)
                    self.assertIn("historical export; no recalculation", messages)
                    self.assertEqual(files_under(root), before)

    def test_complete_existing_figures_are_read_without_opening_numeric_evidence(self):
        for dataset in NOTEBOOKS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "figures").mkdir()
                for filename in FIGURE_FILES[dataset]:
                    (root / "figures" / filename).write_bytes(PNG)
                before = files_under(root)
                _, messages, displayed, opened = self.run_reading(dataset, root)
                self.assertEqual(opened, list(FIGURE_FILES[dataset]))
                self.assertEqual(len(displayed), FIGURE_COUNTS[dataset])
                self.assertNotIn("Saved figure unavailable:", messages)
                self.assertEqual(files_under(root), before)

    def test_unreadable_figure_does_not_block_an_independent_existing_figure(self):
        original_image = ipython_display.Image
        for dataset in NOTEBOOKS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "figures").mkdir()
                for filename in FIGURE_FILES[dataset][:2]:
                    (root / "figures" / filename).write_bytes(PNG)

                def image_or_error(*args, **kwargs):
                    if Path(kwargs["filename"]).name == FIRST_FIGURE[dataset]:
                        raise OSError("synthetic image read failure")
                    return original_image(*args, **kwargs)

                before = files_under(root)
                with patch.object(ipython_display, "Image", side_effect=image_or_error):
                    _, messages, displayed, opened = self.run_reading(dataset, root)
                self.assertIn("synthetic image read failure", messages)
                self.assertIn("No recovery was attempted", messages)
                self.assertEqual(len(displayed), 1)
                self.assertEqual(opened, [FIGURE_FILES[dataset][1]])
                self.assertEqual(files_under(root), before)

    def test_direct_local_producers_refuse_reading_mode_before_touching_inputs(self):
        functions = {
            "stability": (("published_split_scores", ([],)),
                          ("finish_stability_figure", (None, "fixture", None, [], "fixture"))),
            "gtex": (("gtex_published_split_scores", ([],)),
                     ("gtex_figure_finish", (None, "fixture", None, [], "fixture")),
                     ("gtex_figure_draw_expression", ()), ("gtex_figure_draw_baselines", ()),
                     ("gtex_figure_draw_controls", ()), ("gtex_figure_draw_gains", ())),
        }
        for dataset in NOTEBOOKS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                namespace, _, _, _ = self.run_reading(dataset, Path(temporary))
                for name, args in functions[dataset]:
                    with self.subTest(function=name), self.assertRaisesRegex(RuntimeError, "Reading mode"):
                        namespace[name](*args)

    def test_explicit_compute_mode_reaches_existing_data_entry_point(self):
        for dataset, entry in (("stability", "load_dataset"), ("gtex", "sample_rows")):
            cells = code_cells(dataset)
            matches = [cell for cell in cells if f"datasets.{entry}(" in source(cell)
                       and "if READ_SAVED_RESULTS:" in source(cell)]
            self.assertTrue(matches)
            sentinel = RuntimeError("synthetic data entry reached")
            namespace = {
                "READ_SAVED_RESULTS": False, "datasets": datasets,
                "DATASET": "fixture-stability", "GTEX_DATASET": "fixture-gtex",
                "N_ROWS": 7, "SEED": 42,
            }
            with self.subTest(dataset=dataset), patch.object(datasets, entry, side_effect=sentinel) as producer:
                with self.assertRaises(RuntimeError) as caught:
                    exec(compile(source(matches[0]), "synthetic-compute-branch", "exec"), namespace)
                self.assertIs(caught.exception, sentinel)
                if dataset == "stability":
                    producer.assert_called_once_with("fixture-stability", n_rows=7, seed=42)
                else:
                    producer.assert_called_once_with("fixture-gtex")

    def test_only_explicit_cli_namespace_overrides_default_for_tagged_exporter(self):
        # The existing CLI requests figure production and executes tagged cells only.
        # Check its settings with an invented snapshot; no export body is executed.
        for dataset in NOTEBOOKS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                snapshot = root / "snapshot.json"
                snapshot.write_text("{}", encoding="utf-8")
                prefix = "STAGE1" if dataset == "stability" else "GTEX"
                tagged = [cell for cell in code_cells(dataset)
                          if f"{dataset}-figure-export" in cell.get("metadata", {}).get("tags", [])]
                namespace = {"__name__": "__figure_export__"}
                options = {f"{prefix}_FIGURE_SOURCE": str(snapshot), f"{prefix}_FIGURE_DIR": str(root / "exports")}
                with patch.dict(os.environ, options):
                    exec(compile(source(tagged[0]), "explicit-cli-settings", "exec"), namespace)
                self.assertIs(namespace["READ_SAVED_RESULTS"], False)
                self.assertFalse((root / "exports").exists())


if __name__ == "__main__":
    unittest.main()
