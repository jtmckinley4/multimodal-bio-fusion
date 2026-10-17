"""Check the token-embedding shard cache with a stub encoder: no checkpoints, no network."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.encoders import Encoder  # noqa: E402
from shared_code.fusion import token_embeddings as TE  # noqa: E402

WIDTH = 8
STUB = Encoder(key="stub", label="Stub", modality="RNA", checkpoint="stub/checkpoint",
               alphabet="rna", max_len=32)


class StubTokenizer:
    """One token per character, wrapped in a start and an end framing token.

    overlong_mask mimics Nucleotide Transformer v2, whose special_tokens_mask comes back
    one element longer than its own input_ids. Nothing should depend on that mask.
    """

    cls_token_id, eos_token_id, pad_token_id = 0, 1, 2
    unk_token_id = 3

    def __init__(self, overlong_mask=False):
        self.overlong_mask = overlong_mask

    def __call__(self, text, return_tensors=None, return_special_tokens_mask=False, **kwargs):
        ids = [self.cls_token_id] + [ord(c) for c in text] + [self.eos_token_id]
        encoded = {"input_ids": torch.tensor([ids])}
        if return_special_tokens_mask:
            mask = [1] + [0] * len(text) + [1] + ([1] if self.overlong_mask else [])
            encoded["special_tokens_mask"] = torch.tensor([mask])
        return encoded


class StubModel(torch.nn.Module):
    """Returns a deterministic state per token so rows can be checked by value."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, **kwargs):
        base = input_ids.float().unsqueeze(-1)
        return (base.repeat(1, 1, WIDTH) * self.scale,)


def stub_loader(encoder, device, **kwargs):
    return StubTokenizer(), StubModel()


class TokenCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = Path(self.dir.name) / "stub"
        self.texts = ["A" * 3, "C" * 5, "G" * 2, "U" * 7, "A" * 4]
        self.addCleanup(self.dir.cleanup)

    def extract(self, texts=None, **kwargs):
        with patch.object(TE, "load_encoder", stub_loader):
            return TE.extract(STUB, texts or self.texts, self.cache, "cpu",
                              chunk_size=kwargs.pop("chunk_size", 2), progress=None, **kwargs)

    def test_rows_hold_only_content_tokens_in_order(self):
        cache, computed = self.extract()
        self.assertTrue(computed)
        self.assertEqual(len(cache), len(self.texts))
        self.assertEqual(list(cache.lengths), [len(t) for t in self.texts])
        self.assertEqual(cache.width, WIDTH)
        for i, text in enumerate(self.texts):
            row = cache[i]
            self.assertEqual(row.shape, (len(text), WIDTH))
            # The stub encodes each character as its code point, so specials are visibly absent.
            self.assertTrue(np.array_equal(row[:, 0], np.array([ord(c) for c in text], dtype=row.dtype)))

    def test_a_second_run_recomputes_only_missing_shards(self):
        self.extract()
        shards = sorted(self.cache.glob("shard_*.npz"))
        self.assertEqual(len(shards), 3)  # five sequences in chunks of two
        before = {p.name: p.stat().st_mtime_ns for p in shards}
        shards[1].unlink()

        calls = []
        def counting_loader(encoder, device, **kwargs):
            calls.append(encoder.key)
            return stub_loader(encoder, device, **kwargs)
        with patch.object(TE, "load_encoder", counting_loader):
            cache, computed = TE.extract(STUB, self.texts, self.cache, "cpu", chunk_size=2, progress=None)

        self.assertEqual(calls, ["stub"], "the encoder should load once, for the missing shard")
        self.assertTrue(computed)
        self.assertEqual(list(cache.lengths), [len(t) for t in self.texts])
        kept = {p.name: p.stat().st_mtime_ns for p in sorted(self.cache.glob("shard_*.npz"))}
        self.assertEqual(kept["shard_00000.npz"], before["shard_00000.npz"], "untouched shard rewritten")
        self.assertEqual(kept["shard_00002.npz"], before["shard_00002.npz"], "untouched shard rewritten")

    def test_a_complete_cache_loads_no_model(self):
        self.extract()
        def refuse(*args, **kwargs):
            raise AssertionError("a complete cache must not load an encoder")
        with patch.object(TE, "load_encoder", refuse):
            cache, computed = TE.extract(STUB, self.texts, self.cache, "cpu", chunk_size=2, progress=None)
        self.assertFalse(computed)
        self.assertEqual(len(cache), len(self.texts))

    def test_a_different_extraction_is_refused_rather_than_mixed(self):
        self.extract()
        with self.assertRaises(ValueError) as raised:
            self.extract(texts=self.texts + ["AAA"])
        self.assertIn("n_sequences", str(raised.exception))

    def test_truncating_input_raises_instead_of_losing_the_end(self):
        with patch.object(TE, "load_encoder", stub_loader), self.assertRaises(ValueError) as raised:
            TE.extract(STUB, ["A" * 64], self.cache, "cpu", chunk_size=2, progress=None)
        self.assertIn("token limit", str(raised.exception))

    def test_summary_reports_size_without_reading_every_token(self):
        cache, _ = self.extract()
        self.assertEqual(cache.meta["total_tokens"], sum(len(t) for t in self.texts))
        self.assertEqual(cache.meta["bytes"], sum(len(t) for t in self.texts) * WIDTH * 2)
        self.assertEqual(json.loads((self.cache / "meta.json").read_text())["encoder"], "stub")

    def test_a_tokenizer_whose_mask_is_too_long_is_still_handled(self):
        """Nucleotide Transformer v2 returns a mask longer than its own input_ids."""
        def loader(encoder, device, **kwargs):
            return StubTokenizer(overlong_mask=True), StubModel()
        with patch.object(TE, "load_encoder", loader):
            cache, _ = TE.extract(STUB, self.texts, self.cache, "cpu", chunk_size=2, progress=None)
        self.assertEqual(list(cache.lengths), [len(t) for t in self.texts])
        for i, text in enumerate(self.texts):
            row = cache[i]
            self.assertTrue(np.array_equal(row[:, 0], np.array([ord(c) for c in text], dtype=row.dtype)))

    def test_an_unknown_token_is_kept_rather_than_stripped(self):
        """Dropping <unk> would silently shorten a row; only framing tokens are removed."""
        tokenizer = StubTokenizer()
        self.assertEqual(TE.structural_token_ids(tokenizer), {0, 1, 2})
        self.assertNotIn(tokenizer.unk_token_id, TE.structural_token_ids(tokenizer))

    def test_expected_tokens_counts_from_the_coding_sequence(self):
        from shared_code.encoders import ENCODERS
        cds = "A" * 999                                  # 999 nucleotides, 333 codons
        self.assertEqual(TE.expected_tokens(ENCODERS["rnafm"], cds), 999)      # per nucleotide
        self.assertEqual(TE.expected_tokens(ENCODERS["calm"], cds), 333)       # per codon
        # A k-mer encoder falls back to single nucleotides for a partial final k-mer:
        # 999 = 166 six-mers + 3 singles = 169 tokens, not the 167 of a plain division.
        self.assertEqual(TE.expected_tokens(ENCODERS["ntv2_100m"], cds), 169)
        self.assertEqual(TE.expected_tokens(ENCODERS["ntv2_100m"], "A" * 996), 166)  # exact fit
        self.assertEqual(TE.expected_tokens(ENCODERS["esm2_8m"], cds), 333)    # per residue
        # ProtBERT reads a spaced alphabet, so a character count would be wrong.
        self.assertEqual(TE.expected_tokens(ENCODERS["protbert"], cds), 333)
        self.assertIsNone(TE.expected_tokens(ENCODERS["dnabert2"], cds))

    def test_grid_protein_gives_one_token_per_codon(self):
        self.assertEqual(TE.grid_protein("AUGUUUUAAGGG"), "MFXG")   # the stop becomes X, not an end
        self.assertEqual(len(TE.grid_protein("AUG" * 27)), 27)


if __name__ == "__main__":
    unittest.main()
