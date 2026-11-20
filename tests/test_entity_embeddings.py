"""Check the per-entity embedding cache: what it refuses, what it records, what it reuses.

No pretrained weights are loaded. A stub tokenizer and a stub model stand in for an encoder,
which is enough to check the properties a result depends on: an encoder that cannot read a
side is refused before anything is written, every entity that reached the token cap is named,
an interrupted extraction recomputes only what is missing, a cache built from different inputs
is reported rather than overwritten, and a lookup by name returns that entity's own rows.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import entity_embeddings as EE, pair_tasks as PT  # noqa: E402


class StubTokenizer:
    """One token per character, framed by a start and an end token."""

    cls_token_id, pad_token_id, eos_token_id = 0, 1, 2

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [self.cls_token_id] + [ord(c) % 23 + 5 for c in text] + [self.eos_token_id]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_token_id]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class StubModel(nn.Module):
    """A state per token whose first column is the token id, so rows are checkable."""

    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, **_):
        states = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.width)
        states[:, :, 0] = input_ids.float()
        states[:, :, 1] = torch.arange(input_ids.shape[1]).float()
        return (states,)


class StubLoader:
    """Stands in for encoders.load_encoder and counts how often weights were loaded."""

    def __init__(self, width=8):
        self.width, self.calls = width, 0

    def __call__(self, encoder, device, **_):
        self.calls += 1
        return StubTokenizer(), StubModel(self.width).eval()


def sequences(n=5, length=12, prefix="R"):
    return {f"{prefix}{i:02d}": "ACGU"[i % 4] * length for i in range(n)}


def frame(kind="binding", left_modality="rna", right_modality="protein"):
    lefts, rights = sequences(4, 10, "R"), sequences(3, 9, "P")
    rows = [{"left": a, "right": b, "label": (i + j) % 2,
             "provenance": "measured_positive" if (i + j) % 2 else "measured_negative"}
            for i, a in enumerate(lefts) for j, b in enumerate(rights)]
    return PT.PairFrame("stub", pd.DataFrame(rows), lefts, rights, {}, {}, kind,
                        left_modality, right_modality)


class EncoderChoiceTests(unittest.TestCase):
    def test_each_modality_has_a_default(self):
        for modality in PT.MODALITIES:
            self.assertIn(modality, EE.MODALITY_DEFAULTS)
            encoder = EE.encoder_for(modality)
            self.assertIn(encoder.alphabet, EE.MODALITY_ALPHABETS[modality])

    def test_an_encoder_that_cannot_read_the_side_is_refused(self):
        """The mistake this prevents writes a cache of nonsense without ever raising."""
        with self.assertRaises(ValueError):
            EE.encoder_for("ligand", "rinalmo")
        with self.assertRaises(ValueError):
            EE.encoder_for("protein", "hyenadna")
        with self.assertRaises(ValueError):
            EE.encoder_for("rna", "chemberta")

    def test_a_codon_encoder_is_refused_because_an_entity_is_not_a_coding_sequence(self):
        with self.assertRaises(ValueError):
            EE.encoder_for("rna", "mrnafm")

    def test_an_unknown_modality_or_encoder_is_refused(self):
        with self.assertRaises(ValueError):
            EE.encoder_for("peptide")
        with self.assertRaises(ValueError):
            EE.encoder_for("rna", "not_an_encoder")


class EntityInputTests(unittest.TestCase):
    def test_a_side_is_handed_over_in_its_own_alphabet_and_never_translated(self):
        protein = EE.encoder_for("protein", "esm2_150m")
        self.assertEqual(EE.entity_input(protein, "mkvla"), "MKVLA")
        rna = EE.encoder_for("rna", "rinalmo")
        self.assertEqual(EE.entity_input(rna, "ACGT"), "ACGU")
        dna = EE.encoder_for("dna", "hyenadna")
        self.assertEqual(EE.entity_input(dna, "acgu"), "ACGT")

    def test_a_smiles_string_is_not_uppercased_because_case_carries_meaning(self):
        """Lower-case atoms mark aromatic rings; upper-casing a SMILES changes the molecule."""
        ligand = EE.encoder_for("ligand", "chemberta")
        smiles = "Cc1ccc(NC(=O)c2ccccc2)cc1"
        self.assertEqual(EE.entity_input(ligand, smiles), smiles)

    def test_protbert_gets_its_spaced_form(self):
        spaced = EE.entity_input(EE.encoder_for("protein", "protbert"), "MKUZOB")
        self.assertEqual(spaced, "M K X X X X")

    def test_an_empty_entity_is_refused(self):
        with self.assertRaises(ValueError):
            EE.entity_input(EE.encoder_for("rna", "rinalmo"), "   ")

    def test_inputs_are_returned_in_sorted_name_order(self):
        names, texts = EE.inputs_for(EE.encoder_for("rna"), {"B": "ACGU", "A": "GGGG"})
        self.assertEqual(names, ["A", "B"])
        self.assertEqual(texts, ["GGGG", "ACGU"])

    def test_the_fingerprint_separates_the_same_sequences_under_different_names(self):
        first = EE.identity_fingerprint(["A", "B"], ["ACGU", "GGGG"])
        renamed = EE.identity_fingerprint(["A", "C"], ["ACGU", "GGGG"])
        reordered = EE.identity_fingerprint(["B", "A"], ["GGGG", "ACGU"])
        self.assertNotEqual(first, renamed)
        self.assertNotEqual(first, reordered)
        self.assertEqual(first, EE.identity_fingerprint(["A", "B"], ["ACGU", "GGGG"]))


class TokenLimitTests(unittest.TestCase):
    def test_the_encoders_own_limit_leaves_room_for_its_framing_tokens(self):
        self.assertEqual(EE.token_limit(EE.encoder_for("protein", "esm2_150m")), 1022)

    def test_an_explicit_cap_is_honoured_only_when_it_is_the_smaller(self):
        protein = EE.encoder_for("protein", "esm2_150m")
        self.assertEqual(EE.token_limit(protein, 256), 256)
        self.assertEqual(EE.token_limit(protein, 4096), 1022)

    def test_hyenadnas_million_token_window_is_capped_in_practice(self):
        """Padded pair batches cannot hold a million positions, whatever the encoder allows."""
        self.assertEqual(EE.token_limit(EE.encoder_for("dna", "hyenadna"), 1024), 1024)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.loader = StubLoader()
        self._real_loader = EE.load_encoder
        EE.load_encoder = self.loader
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(setattr, EE, "load_encoder", self._real_loader)

    def extract(self, seqs, *, modality="rna", encoder="rinalmo", **kwargs):
        return EE.extract(seqs, self.root / "cache", "cpu", modality=modality,
                          encoder=encoder, progress=None, **kwargs)

    def test_a_cache_holds_one_row_per_entity_with_the_encoders_width(self):
        summary = self.extract(sequences(5, 12), chunk_size=2)
        self.assertEqual(summary["entities"], 5)
        self.assertEqual(summary["width"], 8)
        self.assertEqual(summary["tokens"]["total"], 5 * 12)
        cache = EE.load(self.root / "cache")
        self.assertEqual(len(cache), 5)
        self.assertEqual(cache.width, 8)
        self.assertEqual(cache.names, sorted(sequences(5, 12)))

    def test_a_lookup_by_name_returns_that_entitys_own_rows(self):
        seqs = {"A": "ACGU", "B": "GGGGGGGG", "C": "UU"}
        self.extract(seqs, chunk_size=2)
        cache = EE.load(self.root / "cache")
        for name, sequence in seqs.items():
            row = cache.row(name)
            self.assertEqual(row.shape, (len(sequence), 8))
            # The stub writes the token id into column 0, so the row is checkable.
            expected = [ord(c) % 23 + 5 for c in sequence]
            self.assertEqual([int(v) for v in row[:, 0]], expected)
        self.assertEqual(tuple(cache.tokens("A").shape), (4, 8))
        self.assertIsInstance(cache.tokens("A"), torch.Tensor)

    def test_an_absent_name_is_a_clear_error_rather_than_a_wrong_row(self):
        self.extract(sequences(3, 6))
        cache = EE.load(self.root / "cache")
        with self.assertRaises(KeyError):
            cache.row("not_an_entity")
        self.assertEqual(cache.missing(["R00", "ghost"]), ["ghost"])

    def test_every_entity_at_the_token_cap_is_named(self):
        """A result has to be able to say how much of each side was actually read."""
        seqs = {"short": "ACGU", "long": "ACGU" * 10}
        summary = self.extract(seqs, max_tokens=8)
        self.assertEqual(summary["at_token_limit"], 1)
        self.assertEqual(summary["at_token_limit_share"], 0.5)
        cache = EE.load(self.root / "cache")
        self.assertEqual(sorted(cache.truncated), ["long"])
        self.assertEqual(cache.truncated["long"], 8)
        self.assertEqual(cache.row("long").shape, (8, 8))
        self.assertEqual(cache.row("short").shape, (4, 8))

    def test_a_complete_cache_is_reused_without_loading_weights(self):
        seqs = sequences(6, 10)
        self.extract(seqs, chunk_size=2)
        self.assertEqual(self.loader.calls, 1)
        again = self.extract(seqs, chunk_size=2)
        self.assertEqual(self.loader.calls, 1)  # nothing was missing, so nothing was loaded
        self.assertEqual(again["entities"], 6)

    def test_an_interrupted_extraction_recomputes_only_the_missing_shard(self):
        seqs = sequences(6, 10)
        self.extract(seqs, chunk_size=2)
        before = {path.name: path.read_bytes()
                  for path in sorted((self.root / "cache").glob("shard-*.npz"))}
        self.assertEqual(len(before), 3)
        (self.root / "cache" / "shard-00001.npz").unlink()
        plan = EE.inspect(seqs, self.root / "cache", modality="rna", encoder="rinalmo",
                          chunk_size=2)
        self.assertEqual(plan["status"], "partial")
        self.assertEqual(plan["missing"], [1])
        self.extract(seqs, chunk_size=2)
        after = {path.name: path.read_bytes()
                 for path in sorted((self.root / "cache").glob("shard-*.npz"))}
        self.assertEqual(set(after), set(before))
        for name in ("shard-00000.npz", "shard-00002.npz"):
            self.assertEqual(after[name], before[name])  # untouched, not rewritten
        self.assertEqual(EE.load(self.root / "cache").row("R02").shape, (10, 8))

    def test_a_truncated_shard_is_treated_as_missing_rather_than_read(self):
        seqs = sequences(4, 10)
        self.extract(seqs, chunk_size=2)
        path = self.root / "cache" / "shard-00001.npz"
        payload = bytearray(path.read_bytes())
        path.write_bytes(payload[: len(payload) // 2])
        plan = EE.inspect(seqs, self.root / "cache", modality="rna", encoder="rinalmo",
                          chunk_size=2)
        self.assertEqual(plan["missing"], [1])
        with self.assertRaises(ValueError):
            EE.load(self.root / "cache")

    def test_a_cache_built_from_different_inputs_is_reported_not_overwritten(self):
        seqs = sequences(4, 10)
        self.extract(seqs, chunk_size=2)
        before = (self.root / "cache" / "shard-00000.npz").read_bytes()
        changed = dict(seqs)
        changed["R00"] = "GGGGGGGGGG"
        with self.assertRaises(EE.EntityCacheMismatch):
            self.extract(changed, chunk_size=2)
        self.assertEqual((self.root / "cache" / "shard-00000.npz").read_bytes(), before)

    def test_a_different_encoder_is_a_different_cache_rather_than_a_silent_reuse(self):
        seqs = sequences(3, 10)
        self.extract(seqs)
        with self.assertRaises(EE.EntityCacheMismatch):
            self.extract(seqs, encoder="rnafm")

    def test_an_added_entity_is_a_mismatch_because_the_cache_is_keyed_by_position(self):
        seqs = sequences(3, 10)
        self.extract(seqs, chunk_size=2)
        with self.assertRaises(EE.EntityCacheMismatch):
            self.extract({**seqs, "R09": "ACGUACGUAC"}, chunk_size=2)

    def test_the_metadata_records_the_encoder_revision_it_was_built_under(self):
        self.extract(sequences(2, 6))
        stored = json.loads((self.root / "cache" / "entities.json").read_text())
        self.assertEqual(stored["schema"], EE.SCHEMA)
        self.assertEqual(stored["encoder"], "rinalmo")
        self.assertEqual(stored["revision"],
                         EE.ENCODERS["rinalmo"].revision)
        self.assertEqual(stored["modality"], "rna")

    def test_a_cache_whose_metadata_disagrees_with_its_shards_is_refused(self):
        seqs = sequences(4, 10)
        self.extract(seqs, chunk_size=2)
        path = self.root / "cache" / "entities.json"
        stored = json.loads(path.read_text())
        stored["lengths"] = stored["lengths"][:-1]
        path.write_text(json.dumps(stored))
        with self.assertRaises(ValueError):
            EE.load(self.root / "cache")


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.loader = StubLoader()
        self._real_loader = EE.load_encoder
        EE.load_encoder = self.loader
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(setattr, EE, "load_encoder", self._real_loader)

    def test_both_sides_are_built_and_reported_with_their_encoders(self):
        task = frame()
        caches, summaries = EE.build(task, self.root, "cpu", max_tokens=64, progress=None)
        self.assertEqual(set(caches), set(PT.SIDES))
        self.assertEqual(summaries["left"]["modality"], "rna")
        self.assertEqual(summaries["right"]["modality"], "protein")
        self.assertEqual(summaries["left"]["encoder"], EE.MODALITY_DEFAULTS["rna"])
        self.assertEqual(summaries["right"]["encoder"], EE.MODALITY_DEFAULTS["protein"])
        self.assertEqual(caches["left"].width, caches["right"].width)
        for side in PT.SIDES:
            self.assertEqual(caches[side].missing(task.pairs[side]), [])

    def test_the_two_sides_are_cached_separately_even_at_the_same_modality(self):
        """A protein-and-ligand table has a protein on the left; it must not share a cache."""
        task = frame(kind="binding", left_modality="protein", right_modality="protein")
        caches, _ = EE.build(task, self.root, "cpu", progress=None)
        self.assertNotEqual(caches["left"].cache_dir, caches["right"].cache_dir)

    def test_an_override_names_the_encoder_for_one_side_only(self):
        task = frame()
        _, summaries = EE.build(task, self.root, "cpu", encoders={"left": "rnafm"},
                                progress=None)
        self.assertEqual(summaries["left"]["encoder"], "rnafm")
        self.assertEqual(summaries["right"]["encoder"], EE.MODALITY_DEFAULTS["protein"])

    def test_a_per_side_token_cap_is_honoured(self):
        task = frame()
        _, summaries = EE.build(task, self.root, "cpu",
                                max_tokens={"left": 4, "right": 64}, progress=None)
        self.assertEqual(summaries["left"]["token_limit"], 4)
        self.assertEqual(summaries["right"]["token_limit"], 64)
        self.assertEqual(summaries["left"]["at_token_limit"], 4)  # every left entity is longer

    def test_a_ligand_side_goes_through_the_chemical_encoder(self):
        task = frame(kind="binding", left_modality="protein", right_modality="ligand")
        task.right_sequences = {name: "Cc1ccccc1" for name in task.right_sequences}
        _, summaries = EE.build(task, self.root, "cpu", progress=None)
        self.assertEqual(summaries["right"]["encoder"], "chemberta")
        self.assertEqual(summaries["right"]["modality"], "ligand")

    def test_a_cache_missing_an_entity_the_table_names_is_refused(self):
        task = frame()
        caches, summaries = EE.build(task, self.root, "cpu", progress=None)
        caches["left"].close()
        extra = task.pairs.copy()
        extra.loc[len(extra)] = {"left": "R09", "right": "P00", "label": 1,
                                 "provenance": "measured_positive"}
        wider = PT.PairFrame("stub", extra, {**task.left_sequences, "R09": "ACGUACGUAC"},
                             task.right_sequences)
        with self.assertRaises(Exception) as caught:
            EE.build(wider, self.root, "cpu", progress=None)
        self.assertTrue(isinstance(caught.exception, (ValueError, EE.EntityCacheMismatch)))


class CacheLayoutTests(unittest.TestCase):
    def test_a_cache_directory_names_the_table_side_modality_and_encoder(self):
        path = EE.cache_dir_for("/root", frame(), "left", "rinalmo")
        self.assertEqual(path.parts[-2:], ("stub", "left__rna__rinalmo"))

    def test_the_two_sides_of_a_same_modality_table_do_not_collide(self):
        task = frame(left_modality="protein", right_modality="protein")
        left = EE.cache_dir_for("/root", task, "left", "esm2_150m")
        right = EE.cache_dir_for("/root", task, "right", "esm2_150m")
        self.assertNotEqual(left, right)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
