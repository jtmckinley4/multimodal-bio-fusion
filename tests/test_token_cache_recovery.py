"""Token-cache readiness and historical recovery use synthetic shards and no models."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.encoders import Encoder  # noqa: E402
from shared_code.fusion import token_embeddings as TE  # noqa: E402


ENCODER = Encoder("rnafm", "synthetic RNA-FM", "RNA", "test/rnafm", "rna", 1024)
TEXTS = ["ACG", "CA", "GGGG", "U", "AAA"]


class FakeModel:
    def eval(self):
        return self


class TokenCacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name) / "rnafm"
        self.verification = TE.CacheVerification()

    def extract(self, texts=TEXTS, **kwargs):
        def states(text, *args, **unused):
            return np.full((len(text), 3), ord(text[0]), dtype=np.float16)

        with patch.object(TE, "load_encoder", return_value=(object(), FakeModel())), \
                patch.object(TE, "token_states", side_effect=states):
            return TE.extract(ENCODER, texts, self.cache, "cpu", chunk_size=2, progress=None,
                              verification=self.verification, **kwargs)

    def inspect(self, texts=TEXTS, **kwargs):
        return TE.inspect_cache(ENCODER, texts, self.cache, chunk_size=2,
                                verification=self.verification, **kwargs)

    def metadata(self):
        return json.loads((self.cache / "meta.json").read_text())

    def evidence(self):
        return {"metadata": self.metadata(), "input_sha256": TE.input_fingerprint(TEXTS),
                "shards": [{"name": path.name, "sha256": TE.file_fingerprint(path)}
                           for path in self.cache.glob("shard_*.npz")],
                "source": "recorded historical run.json"}

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.cache.iterdir() if path.is_file()}

    def make_legacy(self):
        legacy = {key: value for key, value in self.metadata().items()
                  if key in ("shard_version", "encoder", "checkpoint", "revision", "max_len",
                             "dtype", "n_sequences", "chunk_size", "n_shards")}
        (self.cache / "meta.json").write_text(json.dumps(legacy))

    def test_new_inspection_does_not_create_directory_or_load_model(self):
        with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")):
            plan = self.inspect()
        self.assertEqual(plan.status, "new")
        self.assertEqual(plan.missing_shards, (0, 1, 2))
        self.assertFalse(self.cache.exists())

    def test_complete_cache_uses_bound_input_identity_without_model_or_writes(self):
        self.extract()
        before = self.snapshot()
        self.assertEqual(self.metadata()["input_sha256"], TE.input_fingerprint(TEXTS))
        self.assertEqual(len(self.metadata()["extraction_code_sha256"]), 64)
        with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")):
            cache, computed = TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2,
                                        progress=None, verification=self.verification)
        cache.close()
        self.assertFalse(computed)
        self.assertEqual(self.inspect().status, "complete")
        self.assertEqual(self.snapshot(), before)

    def test_same_size_input_reordering_is_rejected_before_model_load(self):
        self.extract()
        before = self.snapshot()
        with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")), \
                self.assertRaisesRegex(TE.CacheRecoveryRequired, "ordered inputs"):
            TE.extract(ENCODER, list(reversed(TEXTS)), self.cache, "cpu", chunk_size=2)
        self.assertEqual(self.snapshot(), before)

    def test_declared_empty_or_malformed_input_identity_cannot_use_legacy_opt_in(self):
        self.extract()[0].close()
        original = self.metadata()
        for declared in ("", None, False, []):
            with self.subTest(declared=declared):
                metadata = {**original, "input_sha256": declared}
                (self.cache / "meta.json").write_text(json.dumps(metadata))
                before = self.snapshot()
                self.assertFalse(self.inspect(allow_legacy=True).ready)
                with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")), \
                        self.assertRaisesRegex(TE.CacheRecoveryRequired, "input_sha256 differs"):
                    TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2, allow_legacy=True)
                self.assertEqual(self.snapshot(), before)

    def test_malformed_empty_metadata_and_wrong_cache_dtype_fail_without_mutation(self):
        self.extract()[0].close()
        original = self.metadata()
        for content in ("", "{", "[]", "{}", json.dumps({**original, "dtype": "float32"})):
            with self.subTest(content=content):
                (self.cache / "meta.json").write_text(content)
                before = self.snapshot()
                self.assertFalse(self.inspect(allow_legacy=True).ready)
                with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")), \
                        self.assertRaises(TE.CacheRecoveryRequired):
                    TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2, allow_legacy=True)
                self.assertEqual(self.snapshot(), before)

    def test_partial_cache_computes_only_missing_chunk_and_keeps_other_bytes(self):
        cache, _ = self.extract()
        cache.close()
        kept = {path.name: path.read_bytes() for path in self.cache.glob("shard_*.npz")}
        (self.cache / "shard_00001.npz").unlink()
        self.assertEqual(self.inspect().status, "partial")
        calls = []

        def states(text, *args, **kwargs):
            calls.append(text)
            return np.ones((len(text), 3), dtype=np.float16)

        with patch.object(TE, "load_encoder", return_value=(object(), FakeModel())) as loader, \
                patch.object(TE, "token_states", side_effect=states):
            cache, computed = TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2,
                                        progress=None, verification=self.verification)
        cache.close()
        self.assertTrue(computed)
        loader.assert_called_once()
        self.assertEqual(calls, TEXTS[2:4])
        for name in ("shard_00000.npz", "shard_00002.npz"):
            self.assertEqual((self.cache / name).read_bytes(), kept[name])

    def test_missing_metadata_cannot_be_invented_from_current_inputs(self):
        self.extract()[0].close()
        (self.cache / "meta.json").unlink()
        before = self.snapshot()
        self.assertEqual(self.inspect().status, "recovery_required")
        with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")), \
                self.assertRaisesRegex(TE.CacheRecoveryRequired, "historical metadata"):
            TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2)
        self.assertEqual(self.snapshot(), before)

    def test_verified_historical_record_restores_original_producer_and_partial_run(self):
        self.extract()[0].close()
        evidence = self.evidence()
        evidence["metadata"]["extraction_code_sha256"] = "a" * 64
        (self.cache / "meta.json").unlink()
        (self.cache / "shard_00001.npz").unlink()
        before = self.snapshot()
        plan = self.inspect(recovery_evidence=evidence)
        self.assertTrue(plan.ready)
        self.assertTrue(plan.restore_metadata)
        self.assertFalse((self.cache / "meta.json").exists(), "inspection must be read-only")
        cache, computed = self.extract(recovery_evidence=evidence)
        cache.close()
        self.assertTrue(computed)
        metadata = self.metadata()
        self.assertEqual(metadata["extraction_code_sha256"], "a" * 64)
        self.assertEqual(metadata["shard_extraction_code_sha256"]["shard_00001.npz"],
                         TE.file_fingerprint(TE.__file__))
        for name, content in before.items():
            self.assertEqual((self.cache / name).read_bytes(), content)

    def test_complete_historical_recovery_retains_historical_producer_without_model_load(self):
        self.extract()[0].close()
        evidence = self.evidence()
        evidence["metadata"]["extraction_code_sha256"] = "a" * 64
        evidence["metadata"].pop("shard_extraction_code_sha256")
        self.make_legacy()
        shards = {path.name: path.read_bytes() for path in self.cache.glob("shard_*.npz")}
        with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")):
            cache, computed = TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2,
                                        recovery_evidence=evidence, progress=None)
        cache.close()
        self.assertFalse(computed)
        self.assertEqual(self.metadata()["extraction_code_sha256"], "a" * 64)
        self.assertNotIn("shard_extraction_code_sha256", self.metadata())
        for name, content in shards.items():
            self.assertEqual((self.cache / name).read_bytes(), content)

    def test_historical_input_or_shard_sha_mismatch_preserves_all_files(self):
        self.extract()[0].close()
        evidence = self.evidence()
        (self.cache / "meta.json").unlink()
        before = self.snapshot()
        for change in ("input", "shard"):
            invalid = json.loads(json.dumps(evidence))
            if change == "input":
                invalid["input_sha256"] = "0" * 64
            else:
                invalid["shards"][0]["sha256"] = "0" * 64
            with self.assertRaises(TE.CacheRecoveryRequired):
                self.extract(recovery_evidence=invalid)
            self.assertEqual(self.snapshot(), before)

    def test_legacy_metadata_is_kept_unless_complete_historical_evidence_matches(self):
        self.extract()[0].close()
        evidence = self.evidence()
        self.make_legacy()
        before = self.snapshot()
        self.assertFalse(self.inspect().ready)
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(self.inspect(recovery_evidence=evidence).ready)
        evidence["metadata"]["checkpoint"] = "wrong/checkpoint"
        self.assertFalse(self.inspect(recovery_evidence=evidence).ready)
        self.assertEqual(self.snapshot(), before)

    def test_explicit_complete_legacy_reuse_keeps_reordering_unverified_and_bytes_unchanged(self):
        texts = ["AAA", "CCC", "GGG", "UUU", "ACG"]
        self.extract(texts=texts)[0].close()
        self.make_legacy()
        before = self.snapshot()
        reordered = list(reversed(texts))  # Same lengths do not prove row correspondence.
        plan = self.inspect(texts=reordered, allow_legacy=True)
        self.assertTrue(plan.ready)
        self.assertEqual(plan.provenance, "legacy_unverified")
        self.assertIn("UNVERIFIED", plan.summary)
        self.assertNotIn("input_sha256", plan.metadata)
        warnings = []
        with patch.object(TE, "load_encoder", side_effect=AssertionError("model loaded")):
            cache, computed = TE.extract(ENCODER, reordered, self.cache, "cpu", chunk_size=2,
                                        allow_legacy=True, progress=warnings.append)
        cache.close()
        self.assertFalse(computed)
        self.assertTrue(any("UNVERIFIED" in warning for warning in warnings))
        self.assertEqual(self.snapshot(), before)

    def test_legacy_opt_in_cannot_continue_partial_or_override_recorded_input_identity(self):
        self.extract()[0].close()
        self.assertFalse(self.inspect(texts=list(reversed(TEXTS)), allow_legacy=True).ready)
        self.make_legacy()
        (self.cache / "shard_00001.npz").unlink()
        before = self.snapshot()
        with self.assertRaises(TE.CacheRecoveryRequired):
            self.extract(allow_legacy=True)
        self.assertEqual(self.snapshot(), before)

    def test_legacy_opt_in_cannot_admit_damaged_shards(self):
        self.extract()[0].close()
        self.make_legacy()
        (self.cache / "shard_00000.npz").write_bytes(b"incomplete archive")
        before = self.snapshot()
        self.assertFalse(self.inspect(allow_legacy=True).ready)
        with self.assertRaises(TE.CacheRecoveryRequired):
            self.extract(allow_legacy=True)
        self.assertEqual(self.snapshot(), before)

    def test_header_and_length_validation_never_loads_whole_token_arrays(self):
        self.extract()[0].close()
        metadata = self.metadata()
        metadata.pop("shards")
        (self.cache / "meta.json").write_text(json.dumps(metadata))
        with patch.object(np, "load", side_effect=AssertionError("whole-array np.load")):
            self.assertTrue(self.inspect().ready)
        np.savez(self.cache / "shard_00000.npz", tokens=np.ones((5, 3), dtype=np.float16),
                 lengths=np.array([2, 3], dtype=np.int32))
        self.assertIn("token lengths", self.inspect().issues[0])

    def test_new_token_count_mismatch_stops_before_publishing_shard(self):
        with patch.object(TE, "load_encoder", return_value=(object(), FakeModel())), \
                patch.object(TE, "token_states", return_value=np.ones((1, 3), dtype=np.float16)), \
                self.assertRaisesRegex(ValueError, "expected 3"):
            TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2, progress=None)
        self.assertFalse(list(self.cache.glob("shard_*.npz")))
        self.assertEqual(self.metadata()["input_sha256"], TE.input_fingerprint(TEXTS))

    def test_interruption_records_completed_shard_and_resumes_remaining_rows(self):
        calls = []

        def states(text, *args, **kwargs):
            calls.append(text)
            if len(calls) == 3:
                raise KeyboardInterrupt("interrupted between chunks")
            return np.ones((len(text), 3), dtype=np.float16)

        with patch.object(TE, "load_encoder", return_value=(object(), FakeModel())), \
                patch.object(TE, "token_states", side_effect=states), \
                self.assertRaises(KeyboardInterrupt):
            TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2, progress=None)
        plan = self.inspect()
        self.assertEqual(plan.status, "partial")
        self.assertEqual(plan.reusable_shards, (0,))
        first = (self.cache / "shard_00000.npz").read_bytes()
        self.extract()[0].close()
        self.assertEqual(self.inspect().status, "complete")
        self.assertEqual((self.cache / "shard_00000.npz").read_bytes(), first)

    def test_empty_legacy_metadata_is_bound_before_model_load_and_interrupted_shard_update(self):
        self.extract()[0].close()
        self.make_legacy()
        for path in self.cache.glob("shard_*.npz"):
            path.unlink()
        original_write = TE._write_metadata
        writes = []

        def metadata_write(cache_dir, metadata):
            writes.append(dict(metadata))
            if len(writes) == 2:
                raise KeyboardInterrupt("interrupted after first shard, before metadata update")
            return original_write(cache_dir, metadata)

        def loader(*args, **kwargs):
            self.assertEqual(self.metadata()["input_sha256"], TE.input_fingerprint(TEXTS))
            self.assertEqual(len(self.metadata()["extraction_code_sha256"]), 64)
            return object(), FakeModel()

        with patch.object(TE, "_write_metadata", side_effect=metadata_write), \
                patch.object(TE, "load_encoder", side_effect=loader), \
                patch.object(TE, "token_states", side_effect=lambda text, *args:
                             np.ones((len(text), 3), dtype=np.float16)), \
                self.assertRaises(KeyboardInterrupt):
            TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2, progress=None)
        self.assertEqual(len(writes), 2)
        self.assertEqual(self.inspect().status, "partial")
        first = (self.cache / "shard_00000.npz").read_bytes()
        self.extract()[0].close()
        self.assertEqual(self.inspect().status, "complete")
        self.assertEqual((self.cache / "shard_00000.npz").read_bytes(), first)

    def test_new_nonfinite_token_values_are_rejected_before_shard_publication(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), \
                    patch.object(TE, "load_encoder", return_value=(object(), FakeModel())), \
                    patch.object(TE, "token_states", side_effect=lambda text, *args:
                                 np.full((len(text), 3), value, dtype=np.float16)), \
                    self.assertRaisesRegex(ValueError, "non-finite"):
                TE.extract(ENCODER, TEXTS, self.cache, "cpu", chunk_size=2, progress=None)
            self.assertFalse(list(self.cache.glob("shard_*.npz")))
            self.assertEqual(self.metadata()["input_sha256"], TE.input_fingerprint(TEXTS))

    def test_gpu_cleanup_preserves_inference_error_and_reports_cleanup_failure(self):
        primary = RuntimeError("inference failed")
        with patch.object(TE, "load_encoder", return_value=(object(), FakeModel())), \
                patch.object(TE, "token_states", side_effect=primary), \
                patch.object(TE.torch.cuda, "empty_cache", side_effect=OSError("cleanup failed")), \
                self.assertRaisesRegex(RuntimeError, "inference failed") as raised:
            TE.extract(ENCODER, TEXTS, self.cache, "cuda:0", chunk_size=2, progress=None)
        self.assertIs(raised.exception, primary)
        self.assertTrue(any("cleanup failed" in note for note in getattr(primary, "__notes__", [])))
        self.assertFalse(list(self.cache.glob("shard_*.npz")))

    def test_torch_mps_device_object_runs_cleanup_without_gpu_work(self):
        with patch.object(TE, "load_encoder", return_value=(object(), FakeModel())), \
                patch.object(TE, "token_states", side_effect=lambda text, *args:
                             np.ones((len(text), 3), dtype=np.float16)), \
                patch.object(TE.torch.mps, "empty_cache") as cleanup:
            cache, computed = TE.extract(ENCODER, TEXTS, self.cache, TE.torch.device("mps"),
                                        chunk_size=2, progress=None)
        cache.close()
        self.assertTrue(computed)
        cleanup.assert_called_once()

    def test_failed_recovery_metadata_publication_preserves_shards(self):
        self.extract()[0].close()
        evidence = self.evidence()
        (self.cache / "meta.json").unlink()
        before = self.snapshot()
        with patch.object(TE.os, "replace", side_effect=OSError("publication failed")), \
                self.assertRaisesRegex(OSError, "publication failed"):
            self.extract(recovery_evidence=evidence)
        self.assertEqual(self.snapshot(), before)

    def test_session_sha_is_memoized_and_stat_change_invalidates_it(self):
        self.extract()[0].close()
        evidence = self.evidence()
        (self.cache / "meta.json").unlink()
        verification = TE.CacheVerification()
        with patch.object(TE, "file_fingerprint", wraps=TE.file_fingerprint) as digest:
            self.assertTrue(TE.inspect_cache(ENCODER, TEXTS, self.cache, chunk_size=2,
                                            recovery_evidence=evidence, verification=verification).ready)
            first = digest.call_count
            self.assertTrue(TE.inspect_cache(ENCODER, TEXTS, self.cache, chunk_size=2,
                                            recovery_evidence=evidence, verification=verification).ready)
            self.assertEqual(digest.call_count, first)
            path = self.cache / "shard_00000.npz"
            with path.open("ab") as handle:
                handle.write(b"changed")
            self.assertFalse(TE.inspect_cache(ENCODER, TEXTS, self.cache, chunk_size=2,
                                             recovery_evidence=evidence, verification=verification).ready)
            self.assertEqual(digest.call_count, first + 1)


if __name__ == "__main__":
    unittest.main()
