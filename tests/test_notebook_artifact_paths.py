"""Exercise path settings against an invented checkout, without research inputs.

CODE_DIR is supplied as the already located project directory. These tests cover
artifact destinations, not the separate project-discovery/import bootstrap.
"""

from contextlib import chdir
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1] / "Code"
CASES = (
    ("stage1/Stage1_stability.ipynb", "stability-cache-paths", {
        "CACHE_ROOT": "stage1/embeddings",
        "EMBEDDING_DIR": "stage1/embeddings/fixture_stability_n11_seed7",
        "CONTROL_EMBEDDING_DIR": "stage1/embeddings/fixture_control_n11_seed7",
        "BENCHMARK_TRAIN_DIR": "stage1/embeddings/fixture_stability_published_train13_seed7",
        "BENCHMARK_TEST_DIR": "stage1/embeddings/fixture_stability_published_test5_seed7",
    }),
    ("stage1/Stage1_gtex.ipynb", "gtex-cache-paths", {
        "CACHE_ROOT": "stage1/embeddings",
        "GTEX_TRAIN_DIR": "stage1/embeddings/fixture_gtex_train",
        "GTEX_TEST_DIR": "stage1/embeddings/fixture_gtex_test",
    }),
    ("stage3/blf_token_embeddings.ipynb", "token-settings", {
        "CACHE_ROOT": "stage3/embeddings",
    }),
    ("stage3/blf_build_check.ipynb", "blf-build-settings", {
        "CACHE_ROOT": "stage3/embeddings",
        "RESULTS_DIR": "stage3/results/fungal_expression",
    }),
    ("stage3/gtex_frozen_fusion_reference.ipynb", "gtex-execution-paths", {
        "CACHE_ROOT": "stage3/embeddings",
        "RESULTS_DIR": "stage3/results/gtex_pilot",
        "GTEX_CACHE_RECOVERY": "stage3/gtex_cache_recovery.json",
    }),
)


class NotebookArtifactPathTests(unittest.TestCase):
    def test_settings_keep_existing_destinations_from_different_working_directories(self):
        # These directories model possible launch locations. Artifact folders are
        # intentionally absent: computing their paths must not create anything.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            code_dir = root / "Code"
            launch_dirs = (root, code_dir, code_dir / "stage1", code_dir / "stage3")
            for directory in launch_dirs:
                directory.mkdir(parents=True, exist_ok=True)
            before = set(root.rglob("*"))
            for notebook, cell_id, expected in CASES:
                document = json.loads((CODE / notebook).read_text(encoding="utf-8"))
                cell, = [cell for cell in document["cells"] if cell.get("id") == cell_id]
                source = compile("".join(cell["source"]), f"fixture:{notebook}:{cell_id}", "exec")
                for directory in launch_dirs:
                    with self.subTest(notebook=notebook, working_directory=directory.name):
                        namespace = {
                            "CODE_DIR": code_dir, "Path": Path, "os": os,
                            "np": SimpleNamespace(float16=object()),
                            "DATASET": "fixture_stability", "CONTROL_DATASET": "fixture_control",
                            "N_ROWS": 11, "SEED": 7, "BENCHMARK_N_TRAIN": 13,
                            "BENCHMARK_N_TEST": 5, "GTEX_DATASET": "fixture_gtex",
                        }
                        # Only settings cells execute. No project imports,
                        # models, datasets, analysis bodies or notebook executor.
                        with chdir(directory), patch.object(
                            Path, "mkdir", side_effect=AssertionError("Settings created a directory")
                        ), patch.object(
                            os, "makedirs", side_effect=AssertionError("Settings created a directory")
                        ):
                            exec(source, namespace)
                        for name, relative in expected.items():
                            self.assertEqual(Path(namespace[name]), code_dir / relative, name)
                        if notebook.startswith("stage1/"):
                            for name in expected.keys() - {"CACHE_ROOT"}:
                                self.assertIsInstance(namespace[name], str)
                        else:
                            self.assertTrue(namespace["READ_SAVED_RESULTS"])
            self.assertEqual(set(root.rglob("*")), before)


if __name__ == "__main__":
    unittest.main()
