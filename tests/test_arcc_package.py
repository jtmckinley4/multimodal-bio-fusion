"""Test execution packaging with invented source, CSV and selection bytes only."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("arcc_prepare", ROOT / "execution/arcc/encoder_swaps/prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
GIT_STATE = {"source_commit": "a" * 40, "source_git_status": " M Code/shared_code/helper.py",
             "source_dirty": True}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class ArccPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.source = self.parent / "maintained"
        self.selection = self.parent / "original-selection"
        self.destination = self.parent / "new-execution"
        self.cache = self.parent / "large-caches"
        # Nonempty saved outputs/counts are deliberately preserved verbatim.
        self.notebook_bytes = b'{"cells":[{"cell_type":"code","execution_count":99,"outputs":[{"text":"historical fixture"}],"source":["READ_SAVED_RESULTS=True"]}]}\n'
        write(self.source / prepare.NOTEBOOK, self.notebook_bytes)
        self.context_bytes = b'{\n  "invented_historical_context": 1.2300\n}\n'
        for relative in prepare.HISTORICAL_CONTEXT:
            write(self.source / relative, self.context_bytes)
        write(self.source / "Code/shared_code/__init__.py", b'"""Fixture package."""\n')
        write(self.source / "Code/shared_code/fusion/helper.py", b"ANSWER = 3\n")
        for relative in prepare.REQUIRED_SHARED[1:]:
            write(self.source / relative, b'"""Invented shared execution source."""\n')
        for name in prepare.RUNTIME_FILES:
            write(self.source / "execution/arcc/encoder_swaps" / name, ("runtime fixture: " + name + "\n").encode())
        for name in prepare.RUNTIME_TESTS:
            write(self.source / "tests" / name, ("synthetic test fixture: " + name + "\n").encode())
        # Neither current repository inputs nor any cache payload may enter this package.
        for name in prepare.DATASETS:
            write(self.source / name, b"different-current-cohort\n")
        write(self.cache / "tokens.npy", b"not a real array; do not copy")
        write(self.selection / "Code/stage3/embeddings/recovery-records.json", b'{"do_not_copy":true}')
        self.selection_records = []
        for index, name in enumerate(prepare.DATASETS):
            self.add_selection(name, f"Sequence,Value\nfixture-{index},{index}\n".encode())
        for index, directory in enumerate(prepare.HISTORICAL_ROOTS):
            self.add_selection(directory + f"/fixture-{index}.json", b'{"invented_selection":true}\n')
        self.snapshot = {
            "schema_version": 1, "source_commit": "b" * 40, "source_root": "original-source",
            "captured_utc": "2026-01-01T00:00:00+00:00", "files": self.selection_records,
        }
        self.save_snapshot()
        self.git_patch = patch.object(prepare, "git_state", return_value=GIT_STATE.copy())
        self.git_patch.start()
        self.addCleanup(self.git_patch.stop)

    def add_selection(self, name, contents):
        write(self.selection / name, contents)
        digest = hashlib.sha256(contents).hexdigest()
        self.selection_records.append({"source": name.removeprefix("historical_results/"),
                                       "destination": name, "source_sha256": digest,
                                       "initial_copy_sha256": digest})

    def save_snapshot(self):
        write(self.selection / prepare.SNAPSHOT, json.dumps(self.snapshot, indent=2).encode())

    def plan(self):
        return prepare.build_plan(self.source, self.selection, self.cache)

    def test_inventory_is_deterministic_and_reads_no_cache_payload(self):
        before = tree_bytes(self.parent)
        original_identity = prepare.file_identity

        def no_cache(path):
            self.assertFalse(path.is_relative_to(self.cache))
            self.assertNotIn("embeddings", path.parts)
            return original_identity(path)

        with patch.object(prepare, "file_identity", side_effect=no_cache):
            first, second = self.plan(), self.plan()
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(first["source_commit"], "a" * 40)
        self.assertEqual(first["historical_selection"]["source_commit"], "b" * 40)
        self.assertEqual(Path(first["cache_reference"]["path"]), self.cache)
        self.assertFalse(first["cache_reference"]["copied"])
        self.assertTrue(first["source_dirty"])
        self.assertEqual(first["source_git_status"], GIT_STATE["source_git_status"])
        names = [item["destination"] for item in first["files"]]
        self.assertEqual(names, sorted(names))
        self.assertIn(prepare.SELECTION_EVIDENCE, names)
        self.assertEqual({item["destination"] for item in first["files"] if item["role"] == "historical_context"},
                         set(prepare.HISTORICAL_CONTEXT))
        self.assertEqual({item["destination"] for item in first["files"] if item["role"] == "synthetic_test"},
                         {"tests/" + name for name in prepare.RUNTIME_TESTS})
        self.assertNotIn("tests/test_arcc_package.py", names)
        self.assertFalse(any("embeddings" in name or name.endswith(".npy") for name in names))
        self.assertEqual(tree_bytes(self.parent), before)

    def test_default_cli_prints_inventory_without_making_a_package(self):
        output = io.StringIO()
        before = tree_bytes(self.parent)
        with redirect_stdout(output):
            prepare.main(["--source-root", str(self.source), "--selection-workspace", str(self.selection)])
        document = json.loads(output.getvalue())
        self.assertEqual(document["schema_version"], 2)
        self.assertNotIn("package_status", document)
        self.assertEqual(tree_bytes(self.parent), before)
        self.assertFalse(self.destination.exists())

    def test_cli_requires_an_explicit_selection_workspace(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            prepare.main(["--source-root", str(self.source)])
        self.assertEqual(caught.exception.code, 2)

    def test_cli_defaults_to_repository_root_from_nested_recipe(self):
        output = io.StringIO()
        before = tree_bytes(self.parent)
        recipe_script = self.source / "execution/arcc/encoder_swaps/prepare.py"
        with patch.object(prepare, "__file__", str(recipe_script)), redirect_stdout(output):
            prepare.main(["--selection-workspace", str(self.selection)])
        document = json.loads(output.getvalue())
        self.assertEqual(Path(document["source_root"]), self.source)
        self.assertEqual(tree_bytes(self.parent), before)
        self.assertFalse(self.destination.exists())

    def test_recipe_runtime_is_packaged_flat_without_other_arcc_files(self):
        # A general ARCC guide or another experiment cannot replace this recipe.
        for name in prepare.RUNTIME_FILES:
            write(self.source / "execution/arcc" / name, b"outside the encoder-swap recipe\n")
        write(self.source / "execution/arcc/other_experiment/run.py", b"unrelated experiment\n")
        plan = self.plan()
        runtime = [item for item in plan["files"] if item["role"] == "execution_runtime"]
        self.assertEqual({item["destination"]: item["source"] for item in runtime},
                         {name: "execution/arcc/encoder_swaps/" + name for name in prepare.RUNTIME_FILES})
        prepare.write_package(plan, self.destination)
        for item in runtime:
            self.assertEqual((self.destination / item["destination"]).read_bytes(),
                             (self.source / item["source"]).read_bytes())
        self.assertFalse((self.destination / "execution").exists())

    def test_new_package_keeps_exact_source_and_original_selection_bytes(self):
        plan = self.plan()
        originals = tree_bytes(self.source), tree_bytes(self.selection), tree_bytes(self.cache)
        manifest = prepare.write_package(plan, self.destination)
        self.assertEqual((self.destination / prepare.NOTEBOOK).read_bytes(), self.notebook_bytes)
        self.assertEqual(json.loads((self.destination / prepare.SNAPSHOT).read_text()), manifest)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertFalse(manifest["research_executed"])
        self.assertFalse(manifest["newton_validated"])
        for item in manifest["files"]:
            origin = self.source if item["source_origin"] == "maintained_repository" else self.selection
            actual = (self.destination / item["destination"]).read_bytes()
            self.assertEqual(actual, (origin / item["source"]).read_bytes())
            self.assertEqual(hashlib.sha256(actual).hexdigest(), item["initial_copy_sha256"])
            self.assertEqual(len(actual), item["bytes"])
        for dataset in prepare.DATASETS:
            self.assertNotEqual((self.destination / dataset).read_bytes(), (self.source / dataset).read_bytes())
        self.assertEqual((tree_bytes(self.source), tree_bytes(self.selection), tree_bytes(self.cache)), originals)
        self.assertEqual(tree_bytes(self.destination / "outputs"), {})
        self.assertTrue((self.destination / "outputs/logs").is_dir())
        self.assertTrue((self.destination / "outputs/notebooks").is_dir())
        self.assertFalse((self.destination / "Code/stage3/embeddings").exists())

    def test_missing_maintained_notebook_shared_initializer_or_runtime_stops_before_writing(self):
        missing_paths = [prepare.NOTEBOOK, *prepare.HISTORICAL_CONTEXT, *prepare.REQUIRED_SHARED,
                         *["execution/arcc/encoder_swaps/" + name for name in prepare.RUNTIME_FILES],
                         *["tests/" + name for name in prepare.RUNTIME_TESTS]]
        for relative in missing_paths:
            with self.subTest(path=relative):
                path = self.source / relative
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.plan()
                    self.assertFalse(self.destination.exists())
                finally:
                    path.write_bytes(original)

    def test_historical_context_is_copied_verbatim_without_regeneration(self):
        plan = self.plan()
        prepare.write_package(plan, self.destination)
        for relative in prepare.HISTORICAL_CONTEXT:
            self.assertEqual((self.source / relative).read_bytes(), self.context_bytes)
            self.assertEqual((self.destination / relative).read_bytes(), self.context_bytes)
            entry = next(item for item in plan["files"] if item["destination"] == relative)
            self.assertEqual(entry["role"], "historical_context")
            self.assertEqual(entry["source"], relative)
            self.assertEqual(entry["source_sha256"], hashlib.sha256(self.context_bytes).hexdigest())

    def test_changed_context_after_planning_stops_without_recovery_or_package(self):
        plan = self.plan()
        context = self.source / prepare.HISTORICAL_CONTEXT[0]
        changed_bytes = self.context_bytes + b"changed after inventory"
        context.write_bytes(changed_bytes)
        with self.assertRaisesRegex(ValueError, "Source changed after planning"):
            prepare.write_package(plan, self.destination)
        self.assertEqual(context.read_bytes(), changed_bytes)
        self.assertFalse(self.destination.exists())

    def test_modified_or_missing_preserved_inputs_are_refused(self):
        for entry in self.selection_records:
            relative = entry["destination"]
            path = self.selection / relative
            original = path.read_bytes()
            with self.subTest(path=relative):
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "Preserved input changed"):
                    self.plan()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    self.plan()
                path.write_bytes(original)
        self.assertFalse(self.destination.exists())

    def test_incomplete_or_duplicate_selection_inventory_is_refused(self):
        original = copy.deepcopy(self.selection_records)
        for index in range(len(original)):
            with self.subTest(omitted=index):
                self.snapshot["files"] = original[:index] + original[index + 1:]
                self.save_snapshot()
                with self.assertRaises(ValueError):
                    self.plan()
        self.snapshot["files"] = original + [dict(original[0])]
        self.save_snapshot()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.plan()

    def test_traversal_absolute_paths_and_unsupported_selection_sources_are_refused(self):
        invalid = ("../escape.json", "/escape.json", "C:/escape.json", "datasets/../escape.csv",
                   "datasets\\escape.csv", "datasets/./escape.csv", "datasets//escape.csv",
                   "datasets/new-cohort.csv", "historical_results/unknown.json",
                   prepare.HISTORICAL_ROOTS[0] + "/weights.pt")
        for relative in invalid:
            with self.subTest(relative=relative):
                self.snapshot["files"] = copy.deepcopy(self.selection_records)
                self.snapshot["files"].append({"destination": relative})
                self.save_snapshot()
                with self.assertRaises(ValueError):
                    self.plan()

    def test_absent_malformed_or_unbound_selection_manifest_is_refused(self):
        snapshot_path = self.selection / prepare.SNAPSHOT
        original = snapshot_path.read_bytes()
        snapshot_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()
        snapshot_path.write_bytes(b"not JSON")
        with self.assertRaises(ValueError):
            self.plan()
        for field, value in (("schema_version", 999), ("files", None), ("source_commit", None)):
            with self.subTest(field=field):
                document = json.loads(original)
                document[field] = value
                snapshot_path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    self.plan()
        document = json.loads(original)
        document["files"][0].pop("initial_copy_sha256")
        snapshot_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "initial-copy hash"):
            self.plan()

    def test_existing_and_overlapping_destinations_are_preserved(self):
        plan = self.plan()
        write(self.destination / "keep.txt", b"existing package evidence")
        before = tree_bytes(self.parent)
        for path in (self.destination, self.source, self.source / "child", self.selection,
                     self.selection / "child", self.cache, self.cache / "child", self.parent,
                     self.parent / "outside" / "nested"):
            with self.subTest(path=path), self.assertRaises((ValueError, FileExistsError)):
                prepare.write_package(plan, path)
        self.assertEqual(tree_bytes(self.parent), before)

    def test_new_destination_cannot_contain_a_declared_future_cache(self):
        plan = prepare.build_plan(self.source, self.selection, self.destination / "cache")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            prepare.write_package(plan, self.destination)
        self.assertFalse(self.destination.exists())

    def test_source_or_git_state_changed_after_inventory_stops_before_any_write(self):
        plan = self.plan()
        path = self.source / prepare.NOTEBOOK
        path.write_bytes(self.notebook_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "Source changed after planning"):
            prepare.write_package(plan, self.destination)
        path.write_bytes(self.notebook_bytes)
        with patch.object(prepare, "git_state", return_value={**GIT_STATE, "source_commit": "c" * 40}):
            with self.assertRaisesRegex(ValueError, "working-tree status changed"):
                prepare.write_package(plan, self.destination)
        self.assertFalse(self.destination.exists())

    def test_copy_failure_preserves_partial_files_without_a_completion_manifest(self):
        plan = self.plan()
        original_copy = prepare._copy_file
        calls = []

        def fail_second(source, destination):
            calls.append(destination)
            if len(calls) == 2:
                raise OSError("synthetic copy interruption")
            return original_copy(source, destination)

        with patch.object(prepare, "_copy_file", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "synthetic copy interruption"):
                prepare.write_package(plan, self.destination)
        self.assertTrue(calls[0].is_file())
        self.assertFalse((self.destination / prepare.SNAPSHOT).exists())
        with self.assertRaises(FileExistsError):
            prepare.write_package(plan, self.destination)

    def test_mutation_during_copy_is_caught_before_completion_evidence(self):
        plan = self.plan()
        original_copy = prepare._copy_file
        mutated = False

        def copy_then_change_source(source, destination):
            nonlocal mutated
            original_copy(source, destination)
            if not mutated:
                source.write_bytes(source.read_bytes() + b"changed during copy")
                mutated = True

        with patch.object(prepare, "_copy_file", side_effect=copy_then_change_source):
            with self.assertRaisesRegex(ValueError, "Source changed after planning"):
                prepare.write_package(plan, self.destination)
        self.assertFalse((self.destination / prepare.SNAPSHOT).exists())

    def test_linked_source_is_rejected_without_reading_its_bytes(self):
        linked = self.source / prepare.NOTEBOOK
        original_is_link = prepare._is_link
        with patch.object(prepare, "_is_link", side_effect=lambda path: path == linked or original_is_link(path)):
            with self.assertRaisesRegex(ValueError, "Symlinks and junctions"):
                self.plan()

    def test_escaping_symlink_source_is_refused_when_supported(self):
        source = self.source / prepare.NOTEBOOK
        outside = self.parent / "outside-notebook.ipynb"
        outside.write_bytes(b"must not be copied")
        source.unlink()
        try:
            source.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable on this host: {error}")
        with self.assertRaisesRegex(ValueError, "Symlinks and junctions"):
            self.plan()

    def test_git_identity_uses_only_read_commands_and_disables_optional_locks(self):
        self.git_patch.stop()
        outputs = [str(self.source), "d" * 40, " M fixture.py\n"]
        with patch.object(prepare.subprocess, "run", side_effect=[SimpleNamespace(stdout=value) for value in outputs]) as run:
            state = prepare.git_state(self.source)
        self.assertEqual(state["source_commit"], "d" * 40)
        self.assertTrue(state["source_dirty"])
        arguments = [call.args[0] for call in run.call_args_list]
        self.assertEqual([args[4:] for args in arguments], [
            ["rev-parse", "--show-toplevel"], ["rev-parse", "HEAD"],
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ])
        self.assertTrue(all(args[1] == "--no-optional-locks" for args in arguments))


if __name__ == "__main__":
    unittest.main()
