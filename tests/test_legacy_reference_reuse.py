"""Qualified historical comparisons using invented files, never research results."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import run_identity


ENCODERS = {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"}


class LegacyReferenceReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result_path = self.root / "Code/results/reference.json"
        self.audit_path = self.root / "Code/reports/paired.json"
        self.recovery_path = self.root / "Code/legacy_reference_reuse.json"
        self.result = {"benchmark": "fixture", "variant": "unimodal:rna",
                       "config": {"seed": 0, "batch_size": 32}, "val_score": 0.5}
        labels = np.array([2, 3, 5], dtype=np.float32)
        self.inputs = {
            "encoders": ENCODERS,
            "rows": {"columns": ["sequence", "length"], "count": 3, "sha256": "a" * 64},
            "targets": {**run_identity.array_record(labels), "columns": ["Value"],
                        "task": "regression", "transform": "as_supplied"},
            "splits": {split: run_identity.array_record([index], np.int64)
                       for index, split in enumerate(run_identity.SPLITS)},
            "split_targets": {split: run_identity.array_record(labels[[index]])
                              for index, split in enumerate(run_identity.SPLITS)},
        }
        self.publish()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def publish(self):
        source_hash = self.write(self.result_path, self.result)
        relative = self.result_path.relative_to(self.root).as_posix()
        self.report = {"identity": {
            "schema": "swap.paired-report.v1", "benchmark": "fixture",
            "sources": {"C:\\\\old\\\\repository\\\\" + relative.replace("/", "\\\\"): source_hash},
            "context": {key: deepcopy(value) for key, value in self.inputs.items()
                        if key != "encoders"},
        }}
        report_hash = self.write(self.audit_path, self.report)
        self.manifest = {
            "schema": "blf.reference-reuse.v1", "scope": "tier4_designs.references",
            "encoders": dict(ENCODERS), "qualification": "qualified legacy reference",
            "references": [{"path": relative, "sha256": source_hash,
                            "benchmark": "fixture", "variant": "unimodal:rna"}],
            "audits": {"fixture": {"path": self.audit_path.relative_to(self.root).as_posix(),
                                   "sha256": report_hash}},
        }
        self.write(self.recovery_path, self.manifest)

    def reuse(self, record=None, path=None):
        return run_identity.legacy_reference_inputs(
            self.result if record is None else record,
            path=self.result_path if path is None else path,
            repository=self.root, recovery_path=self.recovery_path,
        )

    def write_changed_report(self):
        self.manifest["audits"]["fixture"]["sha256"] = self.write(self.audit_path, self.report)
        self.write(self.recovery_path, self.manifest)

    def test_approved_legacy_signature_without_predictions_preserves_every_file(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        original = deepcopy(self.result)
        self.assertEqual(self.reuse(), self.inputs)
        self.assertNotIn("test_prediction", self.result)
        self.assertNotIn("identity", self.result)
        self.assertEqual(self.result, original)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*.json")})

    def test_byte_change_and_stale_in_memory_record_are_rejected(self):
        with self.result_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.reuse()
        self.publish()
        changed = {**self.result, "val_score": 0.9}
        with self.assertRaisesRegex(ValueError, "file contents"):
            self.reuse(changed)

    def test_copy_at_another_path_and_outside_path_are_rejected(self):
        copied = self.root / "Code/results/copied.json"
        copied.write_bytes(self.result_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "not an approved"):
            self.reuse(path=copied)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.reuse(path=self.root.parent / "outside.json")

    def test_manifest_paths_cannot_escape_or_use_absolute_paths(self):
        for bad_path in ("../outside.json", "/outside.json", "C:/outside.json", "Code\\results\\reference.json"):
            with self.subTest(path=bad_path):
                self.publish()
                self.manifest["references"][0]["path"] = bad_path
                self.write(self.recovery_path, self.manifest)
                with self.assertRaisesRegex(ValueError, "relative POSIX"):
                    self.reuse()

    def test_symlink_resolving_outside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as external:
            target = Path(external) / "target.json"
            target.write_bytes(self.result_path.read_bytes())
            link = self.root / "Code/results/link.json"
            try:
                link.symlink_to(target)
            except OSError as error:
                self.skipTest(f"Symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "outside"):
                self.reuse(path=link)

    def test_wrong_schema_scope_encoders_and_duplicate_entries_fail(self):
        for mode in ("schema", "scope", "encoders", "duplicate_path", "duplicate_variant"):
            with self.subTest(mode=mode):
                self.publish()
                if mode in ("schema", "scope"):
                    self.manifest[mode] = "other"
                elif mode == "encoders":
                    self.manifest["encoders"]["rna"] = "other"
                else:
                    duplicate = deepcopy(self.manifest["references"][0])
                    duplicate["variant" if mode == "duplicate_path" else "path"] = (
                        "concat" if mode == "duplicate_path" else "Code/results/other.json")
                    self.manifest["references"].append(duplicate)
                self.write(self.recovery_path, self.manifest)
                with self.assertRaises(ValueError):
                    self.reuse()

    def test_changed_benchmark_variant_seed_and_missing_manifest_fail(self):
        for key, value in (("benchmark", "other"), ("variant", "coupled_mamba"),
                           ("config", {"seed": 1})):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.reuse({**self.result, key: value})
        self.recovery_path.unlink()
        with self.assertRaisesRegex(ValueError, "cannot read"):
            self.reuse()

    def test_audit_hash_schema_benchmark_context_and_binding_are_checked(self):
        for mode in ("hash", "schema", "benchmark", "context", "binding", "ambiguous", "encoders"):
            with self.subTest(mode=mode):
                self.publish()
                identity = self.report["identity"]
                if mode == "hash":
                    self.audit_path.write_text("{}", encoding="utf-8")
                else:
                    if mode in ("schema", "benchmark"):
                        identity[mode] = "other"
                    elif mode == "context":
                        identity["context"]["splits"]["test"]["shape"] = [0]
                    elif mode == "binding":
                        identity["sources"] = {"C:/other/file.json": "a" * 64}
                    elif mode == "ambiguous":
                        identity["sources"]["Code/results/reference.json"] = self.manifest["references"][0]["sha256"]
                    else:
                        identity["context"]["encoders"] = {**ENCODERS, "dna": "other"}
                    self.write_changed_report()
                with self.assertRaises(ValueError):
                    self.reuse()

    def test_malformed_modern_identity_never_uses_legacy_approval(self):
        for identity in (None, {}, [], {"schema": "fusion.run.v1"}):
            with self.subTest(identity=identity):
                self.result["identity"] = identity
                self.publish()
                with self.assertRaisesRegex(ValueError, "Recorded BLF inputs"):
                    self.reuse()

    def test_valid_modern_identity_does_not_need_recovery_files(self):
        self.result["identity"] = run_identity.make_identity(
            benchmark="fixture", variant=self.result["variant"], config=self.result["config"],
            inputs=self.inputs, model={"contract": "synthetic"},
            loader=run_identity.blf_loader_spec(self.result["config"], ("dna", "rna", "prot")),
        )
        self.recovery_path.unlink()
        self.result_path.unlink()
        self.assertEqual(self.reuse(), self.inputs)


if __name__ == "__main__":
    unittest.main()
