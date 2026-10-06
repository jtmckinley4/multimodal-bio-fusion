"""Owned model cleanup on successful and failed tiny substitute inference."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from mbf import embeddings
from mbf.encoders import ENCODERS


class EmbeddingResourcesTests(unittest.TestCase):
    def owning_calls(self, cache):
        return (
            ('pooled', 'embed', np.array([1., 2.]),
             lambda: embeddings.embed_with_cache(ENCODERS['esm2_8m'], ['ATGGCT'], cache, 'cpu')),
            ('layers', 'embed_all_layers', [np.array([1., 2.]), np.array([3., 4.])],
             lambda: embeddings.embed_layers(ENCODERS['esm2_8m'], ['ATGGCT'], 'cpu')),
            ('attention', 'nucleotide_attention', np.array([0., 1.]),
             lambda: embeddings.attention_profiles(ENCODERS['rnafm'], ['ATGGCT'], 'cpu')),
        )

    def test_pooled_cleanup_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as cache:
                error = RuntimeError('substitute inference failed')
                with patch.object(embeddings, 'load_encoder', return_value=(object(), object())), \
                     patch.object(embeddings, 'free_memory') as cleanup, \
                     patch.object(embeddings, 'embed', side_effect=error if fail else None,
                                  return_value=np.array([1., 2.])):
                    if fail:
                        with self.assertRaises(RuntimeError) as raised:
                            embeddings.embed_with_cache(ENCODERS['esm2_8m'], ['ATGGCT'], cache, 'cpu')
                        self.assertIs(raised.exception, error)
                        self.assertEqual(list(Path(cache).iterdir()), [])
                    else:
                        matrix, reused = embeddings.embed_with_cache(ENCODERS['esm2_8m'], ['ATGGCT'], cache, 'cpu')
                        np.testing.assert_array_equal(matrix, [[1., 2.]])
                        self.assertFalse(reused)
                    cleanup.assert_called_once_with()

    def test_layer_cleanup_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                error = RuntimeError('substitute layer inference failed')
                with patch.object(embeddings, 'load_encoder', return_value=(object(), object())), \
                     patch.object(embeddings, 'free_memory') as cleanup, \
                     patch.object(embeddings, 'embed_all_layers', side_effect=error if fail else None,
                                  return_value=[np.array([1., 2.]), np.array([3., 4.])]):
                    if fail:
                        with self.assertRaises(RuntimeError) as raised:
                            embeddings.embed_layers(ENCODERS['esm2_8m'], ['ATGGCT'], 'cpu')
                        self.assertIs(raised.exception, error)
                    else:
                        result = embeddings.embed_layers(ENCODERS['esm2_8m'], ['ATGGCT'], 'cpu')
                        np.testing.assert_array_equal(result, [[[1., 2.]], [[3., 4.]]])
                    cleanup.assert_called_once_with()

    def test_primary_inference_errors_survive_cleanup_failure_for_each_owner(self):
        with tempfile.TemporaryDirectory() as cache:
            for name, inference_name, _result, call in self.owning_calls(cache):
                for error_type in (RuntimeError, KeyboardInterrupt):
                    with self.subTest(owner=name, primary=error_type.__name__):
                        primary = error_type('controlled inference failure')
                        cleanup_error = OSError('controlled memory cleanup failure')
                        with patch.object(embeddings, 'load_encoder', return_value=(object(), object())), \
                             patch.object(embeddings, inference_name, side_effect=primary), \
                             patch.object(embeddings, 'free_memory', side_effect=cleanup_error) as cleanup:
                            with self.assertRaises(error_type) as raised:
                                call()
                        self.assertIs(raised.exception, primary)
                        if hasattr(primary, 'add_note'):
                            self.assertTrue(any(str(cleanup_error) in note for note in primary.__notes__))
                        cleanup.assert_called_once_with()
                        self.assertEqual(list(Path(cache).iterdir()), [])

    def test_successful_work_cleanup_errors_propagate_despite_unrelated_caller_error(self):
        with tempfile.TemporaryDirectory() as cache:
            for name, inference_name, result, call in self.owning_calls(cache):
                with self.subTest(owner=name):
                    cleanup_error = OSError('controlled memory cleanup failure')
                    with patch.object(embeddings, 'load_encoder', return_value=(object(), object())), \
                         patch.object(embeddings, inference_name, return_value=result), \
                         patch.object(embeddings, 'free_memory', side_effect=cleanup_error) as cleanup:
                        try:
                            raise LookupError('unrelated active error in the caller')
                        except LookupError:
                            with self.assertRaises(OSError) as raised:
                                call()
                    self.assertIs(raised.exception, cleanup_error)
                    cleanup.assert_called_once_with()
                    self.assertEqual(list(Path(cache).iterdir()), [])

    def test_attention_cleanup_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                error = RuntimeError('substitute attention inference failed')
                with patch.object(embeddings, 'load_encoder', return_value=(object(), object())), \
                     patch.object(embeddings, 'free_memory') as cleanup, \
                     patch.object(embeddings, 'nucleotide_attention', side_effect=error if fail else None,
                                  return_value=np.array([0., 1.])):
                    if fail:
                        with self.assertRaises(RuntimeError) as raised:
                            embeddings.attention_profiles(ENCODERS['rnafm'], ['ATGGCT'], 'cpu')
                        self.assertIs(raised.exception, error)
                    else:
                        result = embeddings.attention_profiles(ENCODERS['rnafm'], ['ATGGCT'], 'cpu')
                        np.testing.assert_array_equal(result, [[0., 1.]])
                    cleanup.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
