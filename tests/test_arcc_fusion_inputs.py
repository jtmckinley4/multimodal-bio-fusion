"""Small file fixtures for the Newton input checker; no model or research data."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "execution/arcc/fusion_studies"
if not RECIPE.is_dir():
    RECIPE = ROOT
spec = importlib.util.spec_from_file_location("fusion_input_check", RECIPE / "verify_inputs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class InputVerificationTests(unittest.TestCase):
    def entry(self, name, content):
        return {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def test_identical_bytes_pass_and_same_size_corruption_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            path.write_bytes(b"abc")
            entry = self.entry(path.name, b"abc")
            self.assertEqual(checker.check_files([entry], folder), (1, []))
            path.write_bytes(b"abd")
            count, errors = checker.check_files([entry], folder)
            self.assertEqual(count, 0)
            self.assertEqual(errors, ["SHA-256 differs: weights.pt"])
            self.assertEqual(path.read_bytes(), b"abd")

    def test_missing_and_size_mismatch_are_both_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "present").write_bytes(b"long")
            entries = [self.entry("missing", b"a"), self.entry("present", b"a")]
            self.assertEqual(checker.check_files(entries, folder),
                             (0, ["Missing: missing", "Size differs: present"]))

    def test_reference_path_cannot_escape_the_supplied_root(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("../outside", "/outside", "C:/outside", "..\\outside"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    checker.input_path(folder, name)

    def test_ref_main_is_checked_as_well_as_model_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "refs").mkdir()
            (root / "refs/main").write_bytes(b"new-revision")
            (root / "model.safetensors").write_bytes(b"weights")
            entries = [self.entry("refs/main", b"old-revision"),
                       self.entry("model.safetensors", b"weights")]
            count, errors = checker.check_files(entries, root)
            self.assertEqual(count, 1)
            self.assertEqual(errors, ["SHA-256 differs: refs/main"])

    def test_region_scopes_do_not_require_unselected_reference_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.parquet").write_bytes(b"fixture")
            manifest = {"schema": "fusion-studies.inputs.v1",
                        "references": [self.entry("absent.weights.pt", b"not present")],
                        "region_data": [self.entry("source.parquet", b"fixture")],
                        "encoder_files": []}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with redirect_stdout(io.StringIO()) as output:
                checker.main(["--manifest", str(path), "--scope", "regions", "--region-data-root", str(root)])
            self.assertIn("1/1", output.getvalue())
            self.assertFalse((root / "absent.weights.pt").exists())


if __name__ == "__main__":
    unittest.main()
