"""Token publication under Windows file locks, using tiny local fixtures only."""

import ctypes
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.encoders import Encoder  # noqa: E402
from shared_code.fusion import token_embeddings as TE  # noqa: E402


def windows_error(code):
    error = PermissionError(13, "fixture file lock")
    error.winerror = code
    return error


class TokenPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.rows = [np.ones((3, 2), dtype=np.float16),
                     np.full((2, 2), 7, dtype=np.float16)]

    def test_transient_sharing_and_lock_errors_publish_without_reserializing(self):
        replace = os.replace
        for code in (32, 33):
            with self.subTest(code=code):
                path = self.cache / f"shard_{code}.npz"

                def first_locked(source, destination):
                    if publication.call_count == 1:
                        raise windows_error(code)
                    return replace(source, destination)

                with patch.object(TE.os, "replace", side_effect=first_locked) as publication, \
                        patch.object(TE.time, "sleep") as sleep, \
                        patch.object(TE.np, "savez", wraps=np.savez) as serialize:
                    TE._write_shard(path, self.rows)
                self.assertEqual(publication.call_count, 2)
                sleep.assert_called_once()
                serialize.assert_called_once()
                self.assertFalse(path.with_name(path.name + ".partial").exists())
                with np.load(path) as archive:
                    np.testing.assert_array_equal(archive["tokens"], np.concatenate(self.rows))
                    np.testing.assert_array_equal(archive["lengths"], [3, 2])

    def test_persistent_lock_retains_partial_and_prior_shard_with_bounded_wait(self):
        prior = self.cache / "shard_00000.npz"
        TE._write_shard(prior, self.rows)
        before = prior.read_bytes()
        path = self.cache / "shard_00001.npz"
        error = windows_error(32)
        with patch.object(TE.os, "replace", side_effect=error) as publication, \
                patch.object(TE.time, "sleep") as sleep, \
                self.assertRaises(PermissionError) as caught:
            TE._write_shard(path, self.rows)
        self.assertIs(caught.exception, error)
        self.assertEqual(publication.call_count, 10)
        self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), 11.1)
        self.assertTrue(any("10 attempts" in note for note in error.__notes__))
        self.assertEqual(prior.read_bytes(), before)
        self.assertFalse(path.exists())
        with np.load(path.with_name(path.name + ".partial")) as archive:
            np.testing.assert_array_equal(archive["tokens"], np.concatenate(self.rows))

    def test_other_errors_and_cancellation_are_not_retried(self):
        for index, error in enumerate((windows_error(5), PermissionError("access denied"),
                                       OSError("disk failure"), KeyboardInterrupt())):
            with self.subTest(error=type(error).__name__, index=index):
                path = self.cache / f"shard_{index}.npz"
                with patch.object(TE.os, "replace", side_effect=error) as publication, \
                        patch.object(TE.time, "sleep") as sleep, \
                        self.assertRaises(type(error)) as caught:
                    TE._write_shard(path, self.rows)
                self.assertIs(caught.exception, error)
                publication.assert_called_once()
                sleep.assert_not_called()
                self.assertFalse(path.exists())
                self.assertTrue(path.with_name(path.name + ".partial").exists())

    def test_destination_appearing_during_retry_is_preserved(self):
        path = self.cache / "shard_00000.npz"
        with patch.object(TE.os, "replace", side_effect=windows_error(32)) as publication, \
                patch.object(TE.time, "sleep", side_effect=lambda _: path.write_bytes(b"other writer")), \
                self.assertRaisesRegex(TE.CacheRecoveryRequired, "appeared before publication"):
            TE._write_shard(path, self.rows)
        publication.assert_called_once()
        self.assertEqual(path.read_bytes(), b"other writer")
        self.assertTrue(path.with_name(path.name + ".partial").exists())

    def test_metadata_retries_then_publishes_and_preserves_old_bytes_on_failure(self):
        path = self.cache / "meta.json"
        path.write_text('{"previous": true}', encoding="utf-8")
        replace = os.replace

        def first_locked(source, destination):
            if publication.call_count == 1:
                raise windows_error(32)
            return replace(source, destination)

        with patch.object(TE.os, "replace", side_effect=first_locked) as publication, \
                patch.object(TE.time, "sleep"):
            TE._write_metadata(self.cache, {"updated": True})
        self.assertEqual(json.loads(path.read_text()), {"updated": True})
        self.assertEqual(publication.call_count, 2)
        before = path.read_bytes()
        with patch.object(TE.os, "replace", side_effect=windows_error(32)), \
                patch.object(TE.time, "sleep"), self.assertRaises(PermissionError):
            TE._write_metadata(self.cache, {"should_not_publish": True})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.cache.glob(".meta.*.tmp")), [])

    def test_metadata_cleanup_does_not_hide_publication_failure(self):
        path = self.cache / "meta.json"
        path.write_bytes(b'{"previous": true}')
        error = windows_error(32)
        with patch.object(TE.os, "replace", side_effect=error), \
                patch.object(TE.time, "sleep"), \
                patch.object(Path, "unlink", side_effect=PermissionError("cleanup lock")), \
                self.assertRaises(PermissionError) as caught:
            TE._write_metadata(self.cache, {"should_not_publish": True})
        self.assertIs(caught.exception, error)
        self.assertTrue(any("Metadata temporary file retained" in note for note in error.__notes__))
        self.assertEqual(path.read_bytes(), b'{"previous": true}')
        self.assertEqual(len(list(self.cache.glob(".meta.*.tmp"))), 1)

    def test_failed_publication_resumes_missing_chunks_and_preserves_completed_bytes(self):
        encoder = Encoder("rnafm", "fixture", "RNA", "test/rnafm", "rna", 1024)
        texts = ["ACG", "CA", "GGGG", "U", "AAA"]
        replace = os.replace

        class Model:
            def eval(self):
                return self

        def states(text, *args, **kwargs):
            return np.ones((len(text), 2), dtype=np.float16)

        def lock_second_shard(source, destination):
            if Path(destination).name == "shard_00001.npz":
                raise windows_error(32)
            return replace(source, destination)

        with patch.object(TE, "load_encoder", return_value=(object(), Model())), \
                patch.object(TE, "token_states", side_effect=states) as inference, \
                patch.object(TE.os, "replace", side_effect=lock_second_shard), \
                patch.object(TE.time, "sleep"), self.assertRaises(PermissionError):
            TE.extract(encoder, texts, self.cache, "cpu", chunk_size=2, progress=None)
        self.assertEqual(inference.call_count, 4)
        first = self.cache / "shard_00000.npz"
        before = first.read_bytes()
        plan = TE.inspect_cache(encoder, texts, self.cache, chunk_size=2)
        self.assertEqual(plan.reusable_shards, (0,))
        self.assertEqual(plan.missing_shards, (1, 2))
        with patch.object(TE, "load_encoder", return_value=(object(), Model())), \
                patch.object(TE, "token_states", side_effect=states) as inference:
            cache, computed = TE.extract(encoder, texts, self.cache, "cpu",
                                         chunk_size=2, progress=None)
        self.assertTrue(computed)
        self.assertEqual(len(cache), len(texts))
        cache.close()
        self.assertEqual([call.args[0] for call in inference.call_args_list], texts[2:])
        self.assertEqual(first.read_bytes(), before)
        self.assertEqual(TE.inspect_cache(encoder, texts, self.cache, chunk_size=2).status, "complete")

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing semantics")
    def test_real_windows_locks_retry_sharing_errors_and_preserve_access_denied(self):
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        for target in ("source", "destination"):
            with self.subTest(locked=target):
                source = self.cache / f"{target}.partial"
                destination = self.cache / f"{target}.complete"
                source.write_bytes(b"new contents")
                destination.write_bytes(b"previous contents")
                locked = source if target == "source" else destination
                # Read/write sharing is allowed; omitting FILE_SHARE_DELETE blocks rename.
                handle = kernel.CreateFileW(str(locked), 0x80000000, 0x1 | 0x2,
                                            None, 3, 0x80, None)
                if handle == ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                released = False

                def release(_):
                    nonlocal released
                    if not released:
                        self.assertTrue(kernel.CloseHandle(handle))
                        released = True

                try:
                    with patch.object(TE.os, "replace", wraps=os.replace) as publication, \
                            patch.object(TE.time, "sleep", side_effect=release) as sleep:
                        try:
                            TE._replace_with_retry(source, destination)
                        except PermissionError as error:
                            # Windows can report a locked existing destination as access
                            # denied (5), which cannot safely be assumed to be transient.
                            if target != "destination" or error.winerror != 5:
                                raise
                            publication.assert_called_once()
                            sleep.assert_not_called()
                            self.assertEqual(source.read_bytes(), b"new contents")
                            self.assertEqual(destination.read_bytes(), b"previous contents")
                            continue
                    self.assertTrue(released, "The real file lock must exercise the retry")
                    self.assertGreaterEqual(publication.call_count, 2)
                    sleep.assert_called()
                    self.assertEqual(destination.read_bytes(), b"new contents")
                    self.assertFalse(source.exists())
                finally:
                    if not released:
                        kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
