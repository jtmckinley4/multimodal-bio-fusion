"""Synthetic input and fixed-selection boundaries; no research files are loaded."""

import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import study_inputs, study_selection


class InputTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame({"sequence": ["AAA", "AAC", "AAG", "AAT", "ACA", "ACC"],
                             "length": [3] * 6, "Value": np.arange(6, dtype=float),
                             "Split": ["train", "train", "val", "val", "test", "test"]},
                            index=[10, 12, 14, 16, 18, 20])

    def test_preserves_supplied_rows_and_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "Fungal_expression.csv").write_text("fixture")
            frame = self.rows()
            with patch.object(study_inputs.datasets, "blf_benchmark_rows", return_value=frame):
                got, column, groups, positions, identity = study_inputs.benchmark_rows(
                    "fungal_expression", folder,
                )
            pd.testing.assert_frame_equal(got, self.rows())
            self.assertEqual(column, "Split")
            self.assertEqual(positions["test"].tolist(), [4, 5])
            self.assertEqual(identity["policy"], "supplied")
            self.assertEqual(len(set(groups)), 6)

    def test_t_and_u_overlap_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "Fungal_expression.csv").write_text("fixture")
            frame = self.rows()
            frame.loc[10, "sequence"] = "AAU"
            with patch.object(study_inputs.datasets, "blf_benchmark_rows", return_value=frame):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    study_inputs.benchmark_rows("fungal_expression", folder)

    def test_stability_does_not_generate_a_missing_split(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "mRNA_Stability.csv").write_text("fixture")
            with patch.object(study_inputs.datasets, "blf_benchmark_rows", return_value=self.rows()), \
                    patch.object(study_inputs.grouped_stability, "build_split") as build:
                with self.assertRaisesRegex(ValueError, "saved grouped split"):
                    study_inputs.benchmark_rows("mrna_stability", folder)
                build.assert_not_called()

    def test_open_failure_closes_owned_cache(self):
        frame = self.rows()
        cache = Mock()
        with patch.object(study_inputs, "benchmark_rows", return_value=(frame, "Split", [], {}, {})), \
                patch.object(study_inputs.run_identity, "cache_record", side_effect=[{}, ValueError("broken")]), \
                patch.object(study_inputs.token_embeddings, "summarize", return_value=cache):
            with self.assertRaisesRegex(ValueError, "broken"):
                with study_inputs.open_streams("fungal_expression", {
                    "dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m",
                }, "unused", "unused"):
                    self.fail("A broken cache must not reach fitting.")
        cache.close.assert_called_once()


class SelectionTests(unittest.TestCase):
    def candidate(self, root, name, validation, testing, seed=0):
        path = root / name / "fit.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"config": {"seed": seed}, "val_score": validation,
                                    "test_score": testing}))
        return {"key": name, "modality": "rna", "layout": "native", "path": path}

    def test_validation_only_selection_and_immutable_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidates = [self.candidate(root, "a", .8, .1), self.candidate(root, "b", .7, .99)]
            path = root / "selection.json"
            chosen = study_selection.select_anchor(candidates, path=path, context={"study": "toy"})
            self.assertEqual(chosen["key"], "a")
            original = path.read_bytes()
            study_selection.select_anchor(candidates, path=path, context={"study": "toy"})
            self.assertEqual(path.read_bytes(), original)
            candidates[0]["path"].write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                study_selection.read_anchor(path, context={"study": "toy"})
            self.assertEqual(path.read_bytes(), original)

    def test_rejects_nonzero_seed_and_uses_declared_tie_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidates = [self.candidate(root, "a", .8, .1), self.candidate(root, "b", .8, .99)]
            selected = study_selection.select_anchor(candidates, path=root / "selection.json", context={})
            self.assertEqual(selected["key"], "a")
            invalid = self.candidate(root, "c", .9, .99, seed=1)
            with self.assertRaisesRegex(ValueError, "seed-zero"):
                study_selection.select_anchor([invalid], path=root / "other.json", context={})

    def test_named_roots_survive_returned_folder_names_without_copying_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            remote = root / "newton-workspace"
            remote.mkdir()
            evidence = self.candidate(remote, "source", .8, .3)
            selected = root / "study" / "selection.json"
            selected.parent.mkdir()
            study_selection.select_anchor([evidence], path=selected, context={},
                                          evidence_roots={"newton": remote})
            before = selected.read_bytes()
            returned = root / "newton_h100_seeds012"
            shutil.move(str(remote), str(returned))  # Owned temporary fixture only.
            choice = study_selection.read_anchor(selected, context={},
                                                 evidence_roots={"newton": returned})
            self.assertEqual(choice["root"], "newton")
            self.assertEqual(choice["path"], "source/fit.json")
            self.assertEqual(selected.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
