"""Exercise the local Newton preparation command without a cluster or scientific inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "execution/arcc/prepare.py"


def run_cli(*args):
    return subprocess.run([sys.executable, "-B", str(CLI), *map(str, args)],
                          capture_output=True, text=True, check=False)


class PrepareCommandTests(unittest.TestCase):
    def test_each_recipe_exposes_the_same_creation_and_update_commands(self):
        for recipe in ("encoder_swaps", "grouped_stability", "fusion_studies"):
            with self.subTest(recipe=recipe):
                result = run_cli(recipe, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                for option in ("--destination", "--update", "--apply", "--json"):
                    self.assertIn(option, result.stdout)
                error = run_cli(recipe, "--apply")
                self.assertNotEqual(error.returncode, 0)
                self.assertIn("--apply requires --update", error.stderr)

    def test_public_workflow_previews_and_publishes_only_local_source(self):
        # Four notebooks are inert text fixtures. No notebook libraries are imported.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            package = root.parent / "upload"
            from execution.arcc.fusion_studies import package_spec
            paths = [f"Code/stage4/{name}.ipynb" for name in package_spec.NOTEBOOKS]
            paths += [f"tests/{name}" for name in package_spec.TESTS]
            paths += ["execution/arcc/fusion_studies/README.md", "execution/arcc/requirements.txt",
                      "execution/arcc/encoder_swaps/run_encoder_swaps.py",
                      "execution/arcc/fusion_studies/run_fusion_study.py",
                      "Code/shared_code/fusion/train.py", "Code/stage3/embeddings/sample.npz"]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            def report(*args):
                result = run_cli("fusion_studies", "--source-root", root, "--json", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            preview = report()
            self.assertFalse(package.exists())
            self.assertTrue(all(preview[key] for key in ("packaged", "newton_needs", "verify")))
            self.assertEqual(preview["file_count"], len(paths) - 1)
            before = report("--destination", package)
            snapshot = package / "source-snapshot.json"
            self.assertEqual(before["snapshot_sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertFalse((package / "Code/stage3/embeddings/sample.npz").exists())
            saved = package / "outputs/finished.json"
            saved.write_bytes(b"completed fit")
            (root / "Code/shared_code/fusion/train.py").write_bytes(b"modified helper")
            change = report("--update", package)
            self.assertEqual(change["delta"]["changed"], ["Code/shared_code/fusion/train.py"])
            self.assertEqual(change["snapshot_sha256"], before["snapshot_sha256"])
            applied = report("--update", package, "--apply")
            self.assertEqual(applied["previous_snapshot_sha256"], before["snapshot_sha256"])
            self.assertEqual(applied["snapshot_sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertEqual((package / "Code/shared_code/fusion/train.py").read_bytes(), b"modified helper")
            self.assertEqual(saved.read_bytes(), b"completed fit")
            self.assertEqual(report("--update", package)["delta"]["changed"], [])
