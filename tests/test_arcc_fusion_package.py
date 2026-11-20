"""Package source, tests and small inventories while excluding research payloads."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "execution/arcc/fusion_studies"
if not RECIPE.is_dir():
    RECIPE = ROOT
spec = importlib.util.spec_from_file_location("fusion_package_check", RECIPE / "package_spec.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PackageInventoryTests(unittest.TestCase):
    def test_control_regressions_travel_with_the_execution_source(self):
        # Newton can verify the model controls using invented tensors, without
        # loading the large scientific inputs or executing a notebook.
        required = {"test_controlled_models.py", "test_controlled_experiment.py",
                    "test_region_training.py"}
        self.assertTrue(required <= set(prepare.TESTS))

    def fixture(self, root):
        names = [f"Code/stage4/{n}.ipynb" for n in prepare.NOTEBOOKS]
        names += [f"tests/{n}" for n in prepare.TESTS]
        names += ["Code/shared_code/fusion/train.py",
                  "execution/arcc/encoder_swaps/run_encoder_swaps.py",
                  "execution/arcc/fusion_studies/input-manifest.json",
                  "execution/arcc/fusion_studies/batches.tsv",
                  "execution/arcc/fusion_studies/fusion_studies_preflight.sbatch",
                  "execution/arcc/fusion_studies/README.md",
                  "execution/arcc/requirements.txt",
                  "execution/arcc/fusion_studies/run_fusion_study.py",
                  "Code/stage3/embeddings/cache.npz", "datasets/source.parquet"]
        for name in names:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture", encoding="utf-8")

    def test_small_manifests_and_tests_travel_without_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            self.fixture(root)
            plan = prepare.inventory(root)
            names = {p["destination"] for p in plan["files"]}
            self.assertTrue({"input-manifest.json", "batches.tsv", "README.md", "requirements.txt",
                             "fusion_studies_preflight.sbatch"} <= names)
            self.assertTrue({f"tests/{n}" for n in prepare.TESTS} <= names)
            self.assertFalse(any(n.endswith((".npz", ".parquet")) for n in names))
            target = root.parent / "prepared"
            prepare.write_package(plan, target)
            self.assertEqual((target / "input-manifest.json").read_text(), "fixture")
            self.assertEqual((target / "fusion_studies_preflight.sbatch").read_text(), "fixture")
            self.assertEqual((target / "requirements.txt").read_bytes(),
                             (root / "execution/arcc/requirements.txt").read_bytes())
            self.assertFalse((target / "datasets").exists())

    def test_duplicate_flat_runtime_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            self.fixture(root)
            (root / "execution/arcc/fusion_studies/arcc_execution_support.py").write_text("duplicate")
            with self.assertRaisesRegex(ValueError, "collide"):
                prepare.inventory(root)

    def test_missing_offline_test_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "repo"
            self.fixture(root)
            (root / "tests" / prepare.TESTS[0]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                prepare.inventory(root)


if __name__ == "__main__":
    unittest.main()
