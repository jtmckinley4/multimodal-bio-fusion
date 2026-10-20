"""Check the LucaOne baseline's arithmetic, chunking and cache with stubs: no 1.8B download."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code import lucaone  # noqa: E402

WIDTH = 4


class StubTokenizer:
    """One token per character between [CLS] and [SEP], and it records every seq_type seen."""

    def __init__(self):
        self.seen = []

    def __call__(self, text, seq_type="gene", return_tensors=None, add_special_tokens=True,
                 truncation=False, max_length=None, **kwargs):
        self.seen.append((text, seq_type))
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = [2] + ids + [3]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        types = [0 if seq_type == "gene" else 1] * len(ids)
        return {"input_ids": torch.tensor([ids]), "token_type_ids": torch.tensor([types])}


class StubModel(torch.nn.Module):
    """A state per token that depends only on the token, so pooling can be checked by value."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, **kwargs):
        return (input_ids.float().unsqueeze(-1).repeat(1, 1, WIDTH) * self.scale,)


def stub_load(checkpoint="default", device="cpu", dtype=None):
    return StubTokenizer(), StubModel()


class ContextArithmeticTests(unittest.TestCase):
    """The limits Stage 3 measured, as functions rather than as numbers in a notebook."""

    def test_a_sequence_costs_one_token_per_character_plus_two(self):
        self.assertEqual(lucaone.tokens_for("A" * 999), 1001)
        self.assertTrue(lucaone.fits("A" * 1278))
        self.assertFalse(lucaone.fits("A" * 1279))

    def test_the_joint_input_caps_the_coding_sequence_at_957_nucleotides(self):
        self.assertEqual(lucaone.longest_joint_cds(), 957)
        self.assertLessEqual(lucaone.joint_tokens_for(957), lucaone.CONTEXT_TOKENS)
        self.assertGreater(lucaone.joint_tokens_for(960), lucaone.CONTEXT_TOKENS)

    def test_coverage_reports_each_mode_separately(self):
        found = lucaone.coverage([81, 999, 1500])
        self.assertEqual(found["rows"], 3)
        self.assertEqual(found["longest nt"], 1500)
        # 1,500 nucleotides overrun the gene stream but their 500 residues do not.
        self.assertAlmostEqual(found["gene stream"], 2 / 3)
        self.assertAlmostEqual(found["protein stream"], 1.0)
        self.assertAlmostEqual(found["joint input"], 1 / 3)


class PoolingTests(unittest.TestCase):
    def test_pooling_drops_the_framing_tokens(self):
        tokenizer, model = stub_load()
        vector = lucaone.pooled("ACGT", "gene", tokenizer, model)
        self.assertEqual(vector.shape, (WIDTH,))
        self.assertAlmostEqual(float(vector[0]), np.mean([ord(c) for c in "ACGT"]), places=4)

    def test_the_stream_type_reaches_the_tokenizer(self):
        tokenizer, model = stub_load()
        lucaone.pooled("MKT", "prot", tokenizer, model)
        self.assertEqual(tokenizer.seen[-1][1], "prot")

    def test_a_sequence_past_the_context_is_pooled_over_all_its_windows(self):
        tokenizer, model = stub_load()
        text = "ACGT" * 4                       # 16 characters
        vector = lucaone.pooled(text, "gene", tokenizer, model, limit=8)
        # Windows of six characters each, every character counted exactly once.
        self.assertEqual([t for t, _ in tokenizer.seen], ["ACGTAC", "GTACGT", "ACGT"])
        self.assertAlmostEqual(float(vector[0]), np.mean([ord(c) for c in text]), places=4)


class CacheTests(unittest.TestCase):
    def test_a_matching_cache_is_reused_and_never_loads_the_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gene.npz"
            with patch.object(lucaone, "load", stub_load):
                first, reused = lucaone.embed_with_cache(["ACG", "TTT"], "gene", path,
                                                         progress=None)
            self.assertFalse(reused)

            def refuse(*args, **kwargs):
                raise AssertionError("the model must not be loaded when the cache matches")

            with patch.object(lucaone, "load", refuse):
                again, reused = lucaone.embed_with_cache(["ACG", "TTT"], "gene", path,
                                                         progress=None)
            self.assertTrue(reused)
            np.testing.assert_allclose(first, again)

    def test_the_same_rows_under_a_different_stream_are_refused_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stream.npz"
            with patch.object(lucaone, "load", stub_load):
                lucaone.embed_with_cache(["ACG"], "gene", path, progress=None)
                with self.assertRaises(ValueError) as raised:
                    lucaone.embed_with_cache(["ACG"], "prot", path, progress=None)
        self.assertIn("seq_type", str(raised.exception))

    def test_different_rows_under_the_same_name_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.npz"
            with patch.object(lucaone, "load", stub_load):
                lucaone.embed_with_cache(["ACG"], "gene", path, progress=None)
                with self.assertRaises(ValueError) as raised:
                    lucaone.embed_with_cache(["TTT"], "gene", path, progress=None)
        self.assertIn("fingerprint", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
