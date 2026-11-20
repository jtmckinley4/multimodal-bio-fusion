"""Local upload-folder update checks using tiny synthetic manifests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from execution.arcc import package_update as update


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PackageUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "repo"
        self.prepared = self.base / "prepared"
        self.repo.mkdir()
        self.prepared.mkdir()
        self.roots = {"maintained_repository": self.repo}
        self.recipe = "fusion_studies"
        self.names = ["README.md", "old.py", "stable.py"]
        for name in self.names:
            (self.repo / name).write_bytes(name.encode())
            shutil.copyfile(self.repo / name, self.prepared / name)
        self.snapshot = self.prepared / update.SNAPSHOT
        self.snapshot.write_bytes(json.dumps(self.plan(self.names), indent=2).encode())

    def plan(self, names):
        files = []
        for name in names:
            content = (self.repo / name).read_bytes()
            entry = dict(source=name, destination=name, bytes=len(content))
            if self.recipe == "fusion_studies":
                entry.update(sha256=digest(content))
            else:
                entry.update(source_sha256=digest(content), initial_copy_sha256=digest(content),
                             source_origin="maintained_repository")
            files.append(entry)
        schema, version = update.SCHEMAS[self.recipe]
        plan = dict(schema=schema, files=files, source_root=str(self.repo))
        if version is not None:
            plan.update(schema_version=version, experiment=self.recipe)
        return plan

    def run_update(self, plan, apply=False):
        return update.update_package(self.recipe, plan, self.prepared, self.roots, apply=apply)

    def change_sources(self):
        (self.repo / "README.md").write_bytes(b"new")
        (self.repo / "new.py").write_bytes(b"added")
        return self.plan(["README.md", "stable.py", "new.py"])

    def test_preview_update_and_no_op_preserve_results_and_provenance(self):
        original = self.snapshot.read_bytes()
        for name in ("outputs/fits/seed0.json", "Code/stage3/embeddings/shard.npz",
                     "outputs/logs/trial.out", "extra.py"):
            target = self.prepared / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(b"keep")
        plan = self.change_sources()
        preview = self.run_update(plan)
        self.assertEqual(preview, dict(added=["new.py"], changed=["README.md"], removed=["old.py"],
                                       unchanged=1, applied=False))
        self.assertEqual(self.snapshot.read_bytes(), original)
        history = self.prepared / update.HISTORY
        self.assertFalse(history.exists())
        self.assertEqual(self.run_update(plan, apply=True), {**preview, "applied": True})
        current = self.snapshot.read_bytes()
        self.assertEqual(json.loads(current)["previous_snapshot_sha256"], digest(original))
        self.assertEqual((history / "snapshots" / (digest(original) + ".json")).read_bytes(), original)
        for content in (b"README.md", b"old.py"):
            self.assertEqual((history / "objects" / digest(content)).read_bytes(), content)
        self.assertEqual(len(list((history / "objects").iterdir())), 2)
        self.assertEqual((self.prepared / "README.md").read_bytes(), b"new")
        self.assertFalse((self.prepared / "old.py").exists())
        self.assertFalse(self.run_update(plan, apply=True)["applied"])
        self.assertEqual(self.snapshot.read_bytes(), current)
        self.assertEqual(sum(p.read_bytes() == b"keep" for p in self.prepared.rglob("*")
                             if p.is_file()), 4)

    def test_preflight_rename_updates_existing_workspace_and_archives_old_record(self):
        old_name = "newton_fusion_preflight.sbatch.example"
        new_name = "fusion_studies_preflight.sbatch"
        script = b"#!/usr/bin/env bash\n"
        (self.repo / old_name).write_bytes(script)
        (self.prepared / old_name).write_bytes(script)
        original = json.dumps(self.plan([*self.names, old_name]), indent=2).encode()
        self.snapshot.write_bytes(original)
        (self.repo / old_name).rename(self.repo / new_name)

        plan = self.plan([*self.names, new_name])
        preview = self.run_update(plan)
        self.assertEqual(preview["added"], [new_name])
        self.assertEqual(preview["removed"], [old_name])
        self.assertTrue(self.run_update(plan, apply=True)["applied"])
        self.assertEqual((self.prepared / new_name).read_bytes(), script)
        self.assertFalse((self.prepared / old_name).exists())
        archive = self.prepared / update.HISTORY / "snapshots" / f"{digest(original)}.json"
        self.assertEqual(archive.read_bytes(), original)

    def test_refuse_drift_and_unmanaged_collision_before_replacing_sources(self):
        plan = self.change_sources()
        for name in ("stable.py", "new.py"):
            with self.subTest(name=name):
                path = self.prepared / name
                original = path.read_bytes() if path.exists() else None
                # Even identical content cannot assign ownership of an extra file.
                path.write_bytes(b"added")
                with self.assertRaisesRegex(ValueError, "collision"):
                    self.run_update(plan, apply=True)
                self.assertEqual((self.prepared / "README.md").read_bytes(), b"README.md")
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)
        (self.repo / "new.py").write_bytes(b"changed after inventory")
        with self.assertRaises(ValueError):
            self.run_update(plan, apply=True)
        self.assertFalse((self.prepared / "new.py").exists())

    def test_invalid_inventory_cannot_delete_or_redirect_files(self):
        plan = self.plan(self.names)
        for name in ("../outside.py", "outputs/seed0.json", "Code/stage3/embeddings/x.py",
                     ".arcc-history/objects/data", "source-snapshot.json", "README.md", "NUL.py"):
            with self.subTest(name=name):
                old = self.plan(self.names)
                old["files"][-1]["destination"] = name
                self.snapshot.write_text(json.dumps(old))
                with self.assertRaises(ValueError):
                    self.run_update(plan, apply=True)
        self.snapshot.write_text(json.dumps(plan))
        plan["files"][-1]["source"] = "../outside.py"
        with self.assertRaises(ValueError):
            self.run_update(plan, apply=True)

    def test_profiles_schemas_and_source_roots(self):
        for recipe in ("encoder_swaps", "grouped_stability"):
            with self.subTest(recipe=recipe):
                self.recipe = recipe
                plan = self.plan(self.names)
                self.snapshot.write_text(json.dumps(plan))
                self.assertEqual(self.run_update(plan)["unchanged"], 3)
                for incompatible in ({"experiment": "other"}, {"schema_version": 0}):
                    with self.assertRaises(ValueError):
                        self.run_update({**plan, **incompatible})
                external = self.base / "historical"
                external.mkdir(exist_ok=True)
                self.roots["historical_selection_workspace"] = external
                plan["files"][0]["source_origin"] = "historical_selection_workspace"
                (external / "README.md").write_bytes(b"README.md")
                self.assertEqual(self.run_update(plan)["unchanged"], 3)
        with self.assertRaises(ValueError):
            update.update_package(self.recipe, plan, self.repo, self.roots)
        with self.assertRaises(ValueError):
            self.run_update({**plan, "cache_reference": {"path": str(self.prepared)}})

    def test_source_updates_retain_and_protect_the_recorded_external_cache(self):
        self.recipe = "grouped_stability"
        cache = self.base / "embeddings"
        cache.mkdir()
        shard = cache / "shard.npz"
        shard.write_bytes(b"existing embeddings")
        old = self.plan(self.names)
        old["cache_reference"] = {"path": str(cache), "copied": False}
        self.snapshot.write_text(json.dumps(old))
        plan = self.plan(self.names)
        other = self.base / "other"
        with self.assertRaisesRegex(ValueError, "retains its recorded cache"):
            self.run_update({**plan, "cache_reference": {"path": str(other)}}, apply=True)
        self.assertFalse(self.run_update({**plan, "cache_reference": {"path": str(cache)}})["applied"])
        plan = self.change_sources()
        plan["cache_reference"] = {"path": None, "copied": False}
        self.assertEqual(self.run_update(plan)["changed"], ["README.md"])
        self.assertTrue(self.run_update(plan, apply=True)["applied"])
        self.assertEqual(json.loads(self.snapshot.read_bytes())["cache_reference"], old["cache_reference"])
        self.assertEqual(shard.read_bytes(), b"existing embeddings")
        old["cache_reference"]["path"] = str(self.prepared / "runtime-cache")
        self.snapshot.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "overlaps a declared cache"):
            self.run_update(plan, apply=True)

    def test_malformed_entries_stop_before_writing(self):
        plan = self.plan(self.names)
        for files in (None, [], ["a.py"], [{"source":"a.py"}], [{"destination":"a.py"}]):
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.run_update({**plan, "files": files}, apply=True)
        self.assertFalse((self.prepared / update.HISTORY).exists())

    def test_linked_directory_rejected(self):
        plan = self.plan(self.names)
        linked_source = self.base / "alternate"
        linked_source.mkdir()
        (linked_source / "README.md").write_bytes(b"README.md")
        link = self.repo / "alias"
        plan["files"][0]["source"] = "alias/README.md"
        if os.name == "nt":
            # The sandbox cannot create junctions. Supply the lstat fields of a directory junction.
            link.mkdir()
            lstat = Path.lstat

            def directory_status(path):
                if path == link:
                    return SimpleNamespace(st_mode=stat.S_IFDIR,
                                           st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
                return lstat(path)

            with patch.object(Path, "lstat", directory_status), self.assertRaisesRegex(ValueError, "Linked path"):
                self.run_update(plan)
        else:
            link.symlink_to(linked_source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Linked path"):
                self.run_update(plan)

    def test_recover_partial_update_without_replacing_an_unrelated_file(self):
        original = self.snapshot.read_bytes()
        plan = self.change_sources()
        copy = update._copy

        def interrupt(source, target, identity):
            if target == self.prepared / "README.md":
                raise KeyboardInterrupt()
            copy(source, target, identity)

        with patch.object(update, "_copy", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_update(plan, apply=True)
        self.assertTrue((self.prepared / "new.py").exists())
        self.assertEqual(self.snapshot.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "Interrupted"):
            self.run_update(plan)
        (self.prepared / "new.py").write_bytes(b"personal edit")
        with self.assertRaisesRegex(ValueError, "Unrecognized edits"):
            self.run_update(plan, apply=True)
        self.assertEqual((self.prepared / "new.py").read_bytes(), b"personal edit")
        (self.prepared / "new.py").write_bytes(b"added")
        self.assertTrue(self.run_update(plan, apply=True)["applied"])
        self.assertFalse((self.prepared / update.HISTORY / "pending-snapshot.json").exists())

    def test_failed_write_restores_previous_source_immediately(self):
        original = self.snapshot.read_bytes()
        plan = self.change_sources()
        copy = update._copy

        def fail(source, target, identity):
            if target == self.prepared / "README.md":
                raise OSError("simulated storage failure")
            copy(source, target, identity)

        with patch.object(update, "_copy", side_effect=fail), self.assertRaises(OSError):
            self.run_update(plan, apply=True)
        self.assertEqual(self.snapshot.read_bytes(), original)
        self.assertEqual((self.prepared / "README.md").read_bytes(), b"README.md")
        self.assertFalse((self.prepared / "new.py").exists())
        self.assertFalse((self.prepared / update.HISTORY / "pending-snapshot.json").exists())


if __name__ == "__main__":
    unittest.main()
