"""Embedding cache identity and failure behavior with tiny arrays and no real models."""

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Preserve the notebook environment's NumPy-before-PyTorch import order on Windows.
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code import embeddings
from shared_code.encoders import Encoder


def expected_fingerprint(seqs):
    """Independent encoding specified by cache version 1."""
    text = json.dumps(seqs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TrackingArchive:
    """Observe closure while retaining NumPy's real archive parsing and arrays."""

    def __init__(self, archive):
        self.archive = archive
        self.closed = False

    def __enter__(self):
        return self.archive.__enter__()

    def __exit__(self, *args):
        try:
            return self.archive.__exit__(*args)
        finally:
            self.closed = True

    def __getitem__(self, key):
        return self.archive[key]

    def __getattr__(self, name):
        return getattr(self.archive, name)

    def close(self):
        self.archive.close()
        self.closed = True


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name)
        self.encoder = Encoder(
            "fixture", "Fixture RNA", "RNA", "fixture/checkpoint", "rna", 12,
            revision="requested-revision",
        )
        self.seqs = ["AUG", "GCU"]
        self.path = self.cache_dir / "fixture.npz"
        self.matrix = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        self.load = self.start_patch("load_encoder", return_value=(object(), object()))
        self.embed = self.start_patch("embed", side_effect=self.vector)
        self.free = self.start_patch("free_memory")
        real_load = np.load

        def load_with_fixture_cleanup(*args, **kwargs):
            archive = real_load(*args, **kwargs)
            if hasattr(archive, "close"):
                # Fallback only after assertions: leaked handles must not strand test files.
                self.addCleanup(archive.close)
            return archive

        patcher = patch.object(embeddings.np, "load", side_effect=load_with_fixture_cleanup)
        patcher.start()
        self.addCleanup(patcher.stop)

    def start_patch(self, name, **options):
        patcher = patch.object(embeddings, name, **options)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    @staticmethod
    def vector(text, *_args, **_kwargs):
        return np.array([len(text), sum(map(ord, text))], dtype=np.float32)

    def call(self, seqs=None, encoder=None, prepared=True, device="consumer-device"):
        return embeddings.embed_with_cache(
            encoder or self.encoder, self.seqs if seqs is None else seqs,
            self.cache_dir, device, progress_every=0, prepared=prepared,
        )

    def metadata(self, seqs=None):
        texts = self.seqs if seqs is None else seqs
        return {
            "cache_version": 1,
            "checkpoint": self.encoder.checkpoint,
            "revision": self.encoder.revision,
            "max_len": self.encoder.max_len,
            "fingerprint": expected_fingerprint(texts),
            "n_sequences": len(texts),
            "device": "producer-device",
        }

    def save_fixture(self, matrix=None, metadata=None):
        values = self.metadata() if metadata is None else metadata.copy()
        values["embeddings"] = self.matrix if matrix is None else matrix
        np.savez(self.path, **values)

    def assert_recovery_error(self, seqs=None):
        before = self.path.read_bytes()
        self.load.reset_mock()
        self.embed.reset_mock()
        with self.assertRaises(ValueError) as raised:
            self.call(seqs)
        message = str(raised.exception)
        self.assertIn(str(self.path), message)
        self.assertRegex(message.lower(), r"preserve|fresh cache directory")
        self.load.assert_not_called()
        self.embed.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_effective_raw_and_prepared_inputs_share_cache_and_keep_producer_device(self):
        matrix, reused = self.call(["ATG", "GCT"], prepared=False, device="producer-device")
        self.assertFalse(reused)
        np.testing.assert_array_equal(matrix, [self.vector(s) for s in self.seqs])
        self.assertEqual([call.args[0] for call in self.embed.call_args_list], self.seqs)
        with np.load(self.path, allow_pickle=False) as saved:
            for key, value in self.metadata().items():
                self.assertEqual(saved[key].shape, ())
                self.assertEqual(saved[key].item(), value)
        before = self.path.read_bytes()
        self.load.reset_mock()
        self.embed.reset_mock()
        second, reused = self.call(self.seqs, prepared=True, device="different-consumer-device")
        self.assertTrue(reused)
        np.testing.assert_array_equal(second, matrix)
        self.load.assert_not_called()
        self.embed.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_same_raw_strings_cannot_collide_across_preparation_modes(self):
        raw = ["ATG", "GCT"]
        first, reused = self.call(raw, prepared=False)
        self.assertFalse(reused)
        self.load.reset_mock()
        self.embed.reset_mock()
        second, reused = self.call(raw, prepared=True)
        self.assertFalse(reused)
        self.load.assert_called_once()
        self.assertEqual([call.args[0] for call in self.embed.call_args_list], raw)
        np.testing.assert_array_equal(second, [self.vector(s) for s in raw])
        self.assertFalse(np.array_equal(first, second))

    def test_fingerprint_is_ordered_json_sha256_without_join_boundary_collision(self):
        cases = (["A\nB", "C"], ["A", "B\nC"], ["C", "A\nB"], ["A", "é"])
        for seqs in cases:
            with self.subTest(seqs=seqs):
                self.assertEqual(embeddings.sequences_fingerprint(seqs), expected_fingerprint(seqs))
        self.assertNotEqual(
            embeddings.sequences_fingerprint(cases[0]), embeddings.sequences_fingerprint(cases[1]),
        )

    def test_valid_identity_mismatches_recompute_including_different_row_counts(self):
        cases = {
            "order": (self.seqs[::-1], self.encoder),
            "text": (["AAA", "CCC"], self.encoder),
            "count": (["AUG"], self.encoder),
            "checkpoint": (self.seqs, replace(self.encoder, checkpoint="different/checkpoint")),
            "revision": (self.seqs, replace(self.encoder, revision="other-revision")),
            "unpinned revision": (self.seqs, replace(self.encoder, revision=None)),
            "token limit": (self.seqs, replace(self.encoder, max_len=24)),
        }
        for label, (seqs, encoder) in cases.items():
            with self.subTest(changed=label):
                self.save_fixture()
                self.load.reset_mock()
                self.embed.reset_mock()
                matrix, reused = self.call(seqs, encoder)
                self.assertFalse(reused)
                self.load.assert_called_once_with(encoder, "consumer-device")
                np.testing.assert_array_equal(matrix, [self.vector(s) for s in seqs])
                with np.load(self.path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["embeddings"], matrix)
                    self.assertEqual(saved["fingerprint"].item(), expected_fingerprint(seqs))
                    self.assertEqual(saved["n_sequences"].item(), len(seqs))
                    self.assertEqual(saved["checkpoint"].item(), encoder.checkpoint)
                    self.assertEqual(saved["revision"].item(), encoder.revision or "")
                    self.assertEqual(saved["max_len"].item(), encoder.max_len)
                self.assertEqual(list(self.cache_dir.iterdir()), [self.path])

    def test_empty_request_fails_before_loading_or_replacing_completed_cache(self):
        self.save_fixture()
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.call([])
        self.load.assert_not_called()
        self.embed.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_and_invalid_metadata_require_recovery_before_loading(self):
        cases = {}
        legacy = self.metadata()
        del legacy["cache_version"]
        del legacy["n_sequences"]
        cases["legacy"] = legacy
        for field, value in (
            ("cache_version", 2), ("checkpoint", np.array([self.encoder.checkpoint])),
            ("n_sequences", 0), ("n_sequences", 2.0), ("n_sequences", True),
        ):
            values = self.metadata()
            values[field] = value
            cases[f"{field}={value!r}"] = values
        missing = self.metadata()
        del missing["revision"]
        cases["missing revision"] = missing
        for label, metadata in cases.items():
            with self.subTest(metadata=label):
                self.save_fixture(metadata=metadata)
                self.assert_recovery_error()

    def test_corrupt_archives_and_invalid_matrices_require_recovery(self):
        self.path.write_bytes(b"not a numpy archive")
        self.assert_recovery_error()
        npy = io.BytesIO()
        np.save(npy, self.matrix)
        self.path.write_bytes(npy.getvalue())
        self.assert_recovery_error()
        np.savez(self.path, **self.metadata())
        self.assert_recovery_error()
        cases = {
            "one dimensional": np.ones(2), "three dimensional": np.ones((2, 1, 2)),
            "zero rows": np.empty((0, 2)), "zero columns": np.empty((2, 0)),
            "wrong stored row count": np.ones((3, 2)),
            "text": np.array([["A"], ["B"]]), "object": np.array([[object()], [object()]]),
            "bool": np.ones((2, 2), dtype=bool), "complex": np.ones((2, 2), dtype=complex),
        }
        for label, matrix in cases.items():
            with self.subTest(matrix=label):
                self.save_fixture(matrix)
                # Structural corruption must still error when the requested identity differs.
                self.assert_recovery_error(["different"])

    def test_archives_close_on_hit_mismatch_and_validation_error(self):
        real_load = np.load
        for outcome in ("hit", "mismatch", "error"):
            with self.subTest(outcome=outcome):
                self.save_fixture(matrix=np.ones((3, 2)) if outcome == "error" else None)
                observed = []

                def track(*args, **kwargs):
                    archive = TrackingArchive(real_load(*args, **kwargs))
                    observed.append(archive)
                    self.addCleanup(archive.close)
                    return archive

                def load_after_close(*_args, **_kwargs):
                    self.assertTrue(observed)
                    self.assertTrue(all(archive.closed for archive in observed))
                    return object(), object()

                with patch.object(embeddings.np, "load", side_effect=track):
                    with patch.object(embeddings, "load_encoder", side_effect=load_after_close):
                        if outcome == "error":
                            with self.assertRaises(ValueError):
                                self.call()
                        else:
                            _matrix, reused = self.call(["different"] if outcome == "mismatch" else None)
                            self.assertEqual(reused, outcome == "hit")
                self.assertTrue(observed)
                self.assertTrue(all(archive.closed for archive in observed))

    @staticmethod
    def write_partial(destination):
        if hasattr(destination, "write"):
            destination.write(b"partial archive")
            destination.flush()
        else:
            Path(destination).write_bytes(b"partial archive")

    def test_partial_save_and_replace_failures_preserve_completed_cache(self):
        for failure_stage in ("save", "replace"):
            with self.subTest(stage=failure_stage):
                self.save_fixture()
                before = self.path.read_bytes()
                failure = OSError(f"controlled {failure_stage} failure")

                def partial_save(destination, *_args, **_kwargs):
                    self.write_partial(destination)
                    raise failure

                owner, name = (embeddings.np, "savez") if failure_stage == "save" else (embeddings.os, "replace")
                effect = partial_save if failure_stage == "save" else failure
                with patch.object(owner, name, side_effect=effect):
                    with self.assertRaises(OSError) as raised:
                        self.call(["different"])
                self.assertIs(raised.exception, failure)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual(list(self.cache_dir.iterdir()), [self.path])

    def test_failed_temporary_cleanup_retains_primary_error_and_completed_cache(self):
        self.save_fixture()
        before = self.path.read_bytes()
        failure = OSError("controlled partial save failure")
        real_unlink = os.unlink

        def partial_save(destination, *_args, **_kwargs):
            self.write_partial(destination)
            raise failure

        def failed_cleanup(path, *args, **kwargs):
            if Path(path).name.startswith(f".{self.path.name}."):
                raise PermissionError("controlled temporary cleanup failure")
            return real_unlink(path, *args, **kwargs)

        with patch.object(embeddings.np, "savez", side_effect=partial_save):
            with patch.object(embeddings.os, "unlink", side_effect=failed_cleanup):
                with self.assertRaises(OSError) as raised:
                    self.call(["different"])
        self.assertIs(raised.exception, failure)
        self.assertTrue(getattr(raised.exception, "__notes__", []))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
