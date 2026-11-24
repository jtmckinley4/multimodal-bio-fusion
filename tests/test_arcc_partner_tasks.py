"""Synthetic preparation and launch checks for the RPI2241 Newton recipe."""

from builtins import __import__ as real_import
from contextlib import redirect_stdout
import csv
import importlib.util
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "execution/arcc/partner_tasks"


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"partner_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_only_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "shared_code", "nbformat", "nbclient", "nbconvert"}:
        raise AssertionError(f"Source-only execution imported {name}")
    return real_import(name, *args, **kwargs)


class PartnerNewtonTests(unittest.TestCase):
    def test_schedule_declares_every_split_and_seed_once(self):
        with (RECIPE / "batches.tsv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertCountEqual(
            [(row["held_out"], int(row["seed"]), int(row["fits"])) for row in rows],
            [(split, seed, 6) for split in ("left", "right", "both") for seed in (0, 1, 2)],
        )

    def test_gpu_check_refuses_cpu_or_another_gpu(self):
        runtime = load_module(RECIPE / "run_partner_tasks.py")
        torch = Mock()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            torch.cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runtime.check_gpu(torch, "NVIDIA H100 PCIe")
            torch.cuda.get_device_name.assert_not_called()

            torch.cuda.is_available.return_value = True
            torch.cuda.device_count.return_value = 2
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runtime.check_gpu(torch, "NVIDIA H100 PCIe")

            torch.cuda.device_count.return_value = 1
            torch.cuda.get_device_name.return_value = "NVIDIA A100"
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                runtime.check_gpu(torch, "NVIDIA H100 PCIe")

            torch.cuda.get_device_name.return_value = "NVIDIA H100 PCIe"
            self.assertEqual(runtime.check_gpu(torch, "NVIDIA H100 PCIe"), "NVIDIA H100 PCIe")

    def test_training_refuses_login_shell_before_importing_models(self):
        runtime = load_module(RECIPE / "run_partner_tasks.py")
        with (
            patch.object(runtime, "verify_package", return_value="source"),
            patch.dict(os.environ, {}, clear=True),
            patch("builtins.__import__", side_effect=source_only_import),
            self.assertRaisesRegex(RuntimeError, "Slurm"),
        ):
            runtime.main(["--split", "both", "--seed", "0"])

    def test_output_root_cannot_resolve_into_historical_results(self):
        runtime = load_module(RECIPE / "run_partner_tasks.py")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            workspace = parent / "newton"
            historical = parent / "historical"
            historical.mkdir()
            saved = historical / "saved-fit.json"
            saved.write_bytes(b"historical fixture")
            self.assertEqual(runtime.workspace_path(workspace, runtime.OUTPUTS),
                             workspace / "outputs/rpi2241-newton")
            # Simulate an output-directory link on hosts that cannot create symlinks.
            with (
                patch.object(Path, "resolve", return_value=historical),
                self.assertRaisesRegex(ValueError, "Linked workspace path"),
            ):
                runtime.workspace_path(workspace, runtime.OUTPUTS)
            self.assertEqual(saved.read_bytes(), b"historical fixture")
            self.assertFalse(workspace.exists())

    def prepared_fixture(self):
        prepare = load_module(RECIPE / "package_spec.py")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "source"
        target = source.with_name("upload")
        git_state = {"source_commit": "a" * 40, "source_git_status": "", "source_dirty": False}
        git = patch.object(prepare, "git_state", return_value=git_state)
        git.start()
        self.addCleanup(git.stop)

        fixture_paths = [
            *prepare.SHARED, *(prepare.RECIPE + item for item in prepare.RUNTIME),
            "execution/arcc/requirements.txt",
            "Code/stage4/partner_tasks.ipynb",
            "Code/stage4/data/pairs/rpi2241/pairs.csv",
            "Code/stage4/cache/pair_entities/rpi2241/left__rna__rinalmo/shard-00000.npz",
            "Code/stage4/results/partner_tasks/rpi2241/held_out_both/concat__seed0.json",
        ]
        for relative in fixture_paths:
            file = source / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(f"Invented fixture for {relative}", encoding="utf-8")
        for name in ("run_partner_tasks.py", "batches.tsv", "input-identity.json"):
            (source / prepare.RECIPE / name).write_bytes((RECIPE / name).read_bytes())

        plan = prepare.inventory(source)
        self.assertFalse(target.exists())
        prepare.write_package(plan, target)
        return prepare, source, target, plan

    def test_package_inventory_and_source_only_verification_exclude_research_payload(self):
        prepare, source, target, plan = self.prepared_fixture()
        destinations = {item["destination"] for item in plan["files"]}
        self.assertEqual(destinations, {*prepare.SHARED, *prepare.RUNTIME, "requirements.txt"})
        self.assertIn("Code/shared_code/fusion/run_identity.py", destinations)
        self.assertEqual((target / "batches.tsv").read_bytes(),
                         (source / prepare.RECIPE / "batches.tsv").read_bytes())
        self.assertTrue((target / "source-snapshot.json").is_file())
        self.assertFalse((target / "Code/stage4").exists())

        before = sorted(target.rglob("*"))
        with (
            patch("builtins.__import__", side_effect=source_only_import),
            patch.object(sys, "dont_write_bytecode", True),
            redirect_stdout(StringIO()),
        ):
            runtime = load_module(target / "run_partner_tasks.py")
            runtime.main(["--verify-only"])
        self.assertEqual(before, sorted(target.rglob("*")))
        (target / "batches.tsv").write_bytes(b"changed declaration")
        with (
            patch("builtins.__import__", side_effect=source_only_import),
            self.assertRaisesRegex(ValueError, "Missing or changed prepared source"),
        ):
            runtime.main(["--verify-only"])

    def test_workspace_update_preserves_staged_inputs_and_fit_outputs(self):
        prepare, source, target, _ = self.prepared_fixture()
        table = target / "inputs/rpi2241/pairs.csv"
        fit = target / "outputs/rpi2241-newton/rpi2241/held_out_both/concat__seed0.json"
        for file in (table, fit):
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"existing payload")
        revised = b"updated instructions"
        (source / prepare.RECIPE / "README.md").write_bytes(revised)
        plan = prepare.inventory(source)
        roots = {"maintained_repository": source}
        update = prepare.support.update_package
        preview = update("partner_tasks", plan, target, roots)
        self.assertEqual(preview["changed"], ["README.md"])
        self.assertNotEqual((target / "README.md").read_bytes(), revised)
        self.assertTrue(update("partner_tasks", plan, target, roots, apply=True)["applied"])
        self.assertEqual((target / "README.md").read_bytes(), revised)
        self.assertEqual(table.read_bytes(), b"existing payload")
        self.assertEqual(fit.read_bytes(), b"existing payload")


if __name__ == "__main__":
    unittest.main()
