"""Check the transcript regions, the gate, the grouped split and the probe: no downloads."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code import mrnabench as MB  # noqa: E402
from shared_code.encoders import Encoder  # noqa: E402

WIDTH = 4


def transcript(utr5, cds, utr3):
    """One row as the benchmark publishes it: a sequence and a codon-start track."""
    sequence = utr5 + cds + utr3
    track = np.zeros(len(sequence), dtype=np.int8)
    track[len(utr5):len(utr5) + len(cds):3] = 1
    return sequence, track


def frame_of(rows, gene_names=None, target=None):
    sequences, tracks = zip(*[transcript(*r) for r in rows])
    n = len(rows)
    return pd.DataFrame({
        "gene": gene_names if gene_names is not None else [f"G{i}" for i in range(n)],
        "sequence": list(sequences),
        "cds": list(tracks),
        "splice": [np.zeros(len(s), dtype=np.int8) for s in sequences],
        "target": target if target is not None else np.arange(n, dtype=float),
    })


class RegionTests(unittest.TestCase):
    def test_the_codon_track_recovers_the_three_regions(self):
        sequence, track = transcript("AAAAA", "ATGAAATAG", "CCC")
        found = MB.regions(sequence, track)
        self.assertEqual(found["utr5"], "AAAAA")
        self.assertEqual(found["cds"], "ATGAAATAG")
        self.assertEqual(found["utr3"], "CCC")

    def test_the_coding_sequence_recovered_this_way_is_always_in_frame(self):
        sequence, track = transcript("AA", "ATGAAA", "T")
        self.assertEqual(len(MB.regions(sequence, track)["cds"]) % 3, 0)

    def test_a_track_whose_marks_are_not_one_per_codon_raises(self):
        sequence = "ATGAAATAG"
        broken = np.zeros(len(sequence), dtype=np.int8)
        broken[[0, 2]] = 1          # two marks two apart: no reading frame
        with self.assertRaises(ValueError):
            MB.cds_bounds(broken)

    def test_a_row_with_no_marked_coding_region_raises(self):
        with self.assertRaises(ValueError):
            MB.cds_bounds(np.zeros(9, dtype=np.int8))

    def test_the_region_columns_are_strings_and_no_column_is_doubled(self):
        found = MB.region_frame(frame_of([("AA", "ATGAAA", "TT")]))
        self.assertEqual(list(found.columns), list(dict.fromkeys(found.columns)))
        self.assertIsInstance(found.at[0, "cds"], str)
        self.assertEqual(found.at[0, "cds"], "ATGAAA")

    def test_region_frame_keeps_an_unusable_row_and_names_the_problem(self):
        frame = frame_of([("AA", "ATGAAA", "TT")])
        frame.at[0, "cds"] = np.zeros(len(frame.at[0, "sequence"]), dtype=np.int8)
        found = MB.region_frame(frame)
        self.assertEqual(len(found), 1)
        self.assertNotEqual(found.at[0, "frame_problem"], "")
        self.assertEqual(found.at[0, "cds_len"], 0)


class GateTests(unittest.TestCase):
    """The plan's rule: no untranslated regions, no full-transcript claim."""

    spec = MB.DATASETS["rnahl-human"]

    def test_a_dataset_with_both_regions_and_many_genes_passes(self):
        frame = frame_of([("AAAAA", "ATGAAATAG", "CCCCC")] * 200,
                         gene_names=[f"G{i}" for i in range(200)])
        answer = MB.gate(frame, self.spec)
        self.assertEqual(answer["rows with a 5' region"], 1.0)
        self.assertEqual(answer["rows with a 3' region"], 1.0)
        self.assertEqual(answer["coding sequence in frame"], 1.0)
        self.assertIn("passes", answer["verdict"])

    def test_coding_sequences_with_no_regions_do_not_count_as_full_transcripts(self):
        frame = frame_of([("", "ATGAAATAG", "")] * 200,
                         gene_names=[f"G{i}" for i in range(200)])
        answer = MB.gate(frame, self.spec)
        self.assertEqual(answer["rows with a 5' region"], 0.0)
        self.assertIn("not a full-transcript test", answer["verdict"])

    def test_too_few_gene_groups_fails_the_split_check(self):
        frame = frame_of([("AAAAA", "ATGAAATAG", "CCCCC")] * 10, gene_names=["G"] * 10)
        answer = MB.gate(frame, self.spec)
        self.assertIn("non-leaking split", answer["verdict"])

    def test_the_gate_counts_repeated_sequences(self):
        frame = frame_of([("AAAAA", "ATGAAATAG", "CCCCC")] * 4,
                         gene_names=["A", "B", "C", "D"])
        self.assertAlmostEqual(MB.gate(frame, self.spec)["repeated sequences"], 0.75)


class FeatureTests(unittest.TestCase):
    """The composition floor: in-frame codon counting, and nested region sets."""

    def setUp(self):
        self.regions = MB.region_frame(frame_of([("GGGGG", "ATGAAACCCTAG", "TTTTT")] * 3))

    def test_the_full_transcript_features_contain_the_coding_ones(self):
        cds_only = MB.region_features(self.regions, ("cds",))
        full = MB.region_features(self.regions, ("utr5", "cds", "utr3"))
        self.assertGreater(full.shape[1], cds_only.shape[1])
        # The coding block is the tail of the full vector, unchanged by the regions beside it.
        np.testing.assert_allclose(full[:, -cds_only.shape[1] + 2:], cds_only[:, 2:])

    def test_codons_are_counted_in_frame_not_from_the_start_of_the_transcript(self):
        # Two transcripts with the same coding sequence and different 5' regions must have
        # the same codon counts; counting from position zero would make them differ.
        one = MB.region_frame(frame_of([("G", "ATGAAACCCTAG", "T")]))
        two = MB.region_frame(frame_of([("GGGG", "ATGAAACCCTAG", "T")]))
        codons_one = MB.region_features(one, ("cds",))
        codons_two = MB.region_features(two, ("cds",))
        np.testing.assert_allclose(codons_one, codons_two)

    def test_an_empty_region_contributes_a_defined_value(self):
        empty = MB.region_frame(frame_of([("", "ATGAAA", "")]))
        found = MB.region_features(empty, ("utr5", "cds", "utr3"))
        self.assertTrue(np.isfinite(found).all())


class SplitTests(unittest.TestCase):
    def test_a_gene_never_lands_on_two_sides_of_the_split(self):
        groups = np.array([f"g{i // 3}" for i in range(300)])
        train, validation, test = MB.grouped_split(groups, seed=0)
        self.assertEqual(len(train) + len(validation) + len(test), 300)
        for a, b in ((train, test), (train, validation), (validation, test)):
            self.assertEqual(set(groups[a]) & set(groups[b]), set())

    def test_the_split_is_reproducible_and_the_seed_changes_it(self):
        groups = np.array([f"g{i}" for i in range(200)])
        first = MB.grouped_split(groups, seed=3)[2]
        np.testing.assert_array_equal(first, MB.grouped_split(groups, seed=3)[2])
        self.assertFalse(np.array_equal(first, MB.grouped_split(groups, seed=4)[2]))

    def test_the_ratios_are_respected_within_a_group(self):
        groups = np.array([f"g{i}" for i in range(1000)])
        train, validation, test = MB.grouped_split(groups, seed=0)
        self.assertAlmostEqual(len(train) / 1000, 0.7, delta=0.02)
        self.assertAlmostEqual(len(test) / 1000, 0.15, delta=0.02)

    def test_ratios_that_do_not_sum_to_one_are_refused(self):
        with self.assertRaises(ValueError):
            MB.grouped_split(np.array(["a", "b"]), seed=0, ratios=(0.5, 0.2, 0.2))

    def test_grouping_falls_back_to_the_gene_and_says_so(self):
        frame = frame_of([("A", "ATG", "T")] * 3, gene_names=["X", "X", "Y"])
        with tempfile.TemporaryDirectory() as folder:
            groups, described = MB.grouping(frame, "human", cache_dir=folder)
        self.assertEqual(list(groups), ["X", "X", "Y"])
        self.assertIn("gene", described)
        self.assertIn("weaker", described)

    def test_a_downloaded_homology_map_groups_paralogues_together(self):
        frame = frame_of([("A", "ATG", "T")] * 3, gene_names=["X", "Y", "Z"])
        with tempfile.TemporaryDirectory() as folder:
            maps = Path(folder) / "homology_maps"
            maps.mkdir()
            pd.DataFrame({"gene_name": ["X", "Y"], "gene_group": [7, 7]}).to_csv(
                maps / "human_homology_map.csv", index=False)
            groups, described = MB.grouping(frame, "human", cache_dir=folder)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[1], groups[2])
        self.assertIn("homology", described)


class ProbeTests(unittest.TestCase):
    def test_the_probe_recovers_a_signal_it_is_given(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        y = X[:, 0] * 2 + rng.normal(scale=0.1, size=300)
        split = MB.grouped_split(np.arange(300), seed=0)
        scores = MB.probe(X, y, split)
        self.assertGreater(scores["test_r"], 0.9)
        self.assertIn("test_spearman", scores)
        self.assertIn("alpha", scores)

    def test_one_row_per_seed_comes_back(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4))
        y = rng.normal(size=200)
        table = MB.probe_over_seeds(X, y, np.arange(200), seeds=(0, 1, 2))
        self.assertEqual(list(table["seed"]), [0, 1, 2])
        self.assertEqual(len(set(table["test_r"])), 3)


class StubTokenizer:
    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, **kwargs):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids])}


class StubModel(torch.nn.Module):
    class config:
        max_position_embeddings = 10_000

    def forward(self, input_ids, **kwargs):
        return (input_ids.float().unsqueeze(-1).repeat(1, 1, WIDTH),)


class ChunkedEmbeddingTests(unittest.TestCase):
    STUB = Encoder(key="rnafm", label="Stub", modality="RNA", checkpoint="stub/checkpoint",
                   alphabet="rna", max_len=12)

    def test_a_transcript_longer_than_the_context_is_pooled_over_every_nucleotide(self):
        text = "ACGU" * 10
        vector = MB.pooled_chunked(text, StubTokenizer(), StubModel(), "cpu", window=7)
        self.assertAlmostEqual(float(vector[0]), np.mean([ord(c) for c in text]), places=3)

    def test_the_window_follows_the_encoder_s_own_token_rate(self):
        self.assertEqual(MB.default_window(self.STUB), 10)             # 1 nt per token
        kmer = Encoder(key="ntv2_100m", label="k", modality="DNA", checkpoint="c",
                       alphabet="dna", max_len=12)
        self.assertEqual(MB.default_window(kmer), 60)                  # 6 nt per token

    def test_a_cache_written_under_a_different_window_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.npz"
            with patch.object(MB, "ENCODERS", {"rnafm": self.STUB}), \
                 patch("shared_code.encoders.load_encoder",
                       lambda *a, **k: (StubTokenizer(), StubModel())), \
                 patch("shared_code.encoders.free_memory", lambda: None):
                matrix, reused = MB.embed_with_cache("rnafm", ["ACGU"], path, window=10,
                                                     progress=None)
                self.assertFalse(reused)
                self.assertEqual(matrix.shape, (1, WIDTH))
                _, reused = MB.embed_with_cache("rnafm", ["ACGU"], path, window=10,
                                                progress=None)
                self.assertTrue(reused)
                with self.assertRaises(ValueError) as raised:
                    MB.embed_with_cache("rnafm", ["ACGU"], path, window=5, progress=None)
        self.assertIn("window", str(raised.exception))


class CatalogTests(unittest.TestCase):
    def test_half_life_and_translation_efficiency_stay_separate_tasks(self):
        self.assertEqual(set(MB.HALF_LIFE) & set(MB.TRANSLATION_EFFICIENCY), set())
        self.assertEqual(set(MB.HALF_LIFE) & set(MB.RIBOSOME_LOAD), set())
        # The one dataset carrying both targets names them separately.
        self.assertEqual(len(MB.DATASETS["mrl-hl-lbkwk"].targets), 2)

    def test_mean_ribosome_load_is_not_filed_as_half_life(self):
        self.assertIn("NOT half-life", MB.DATASETS["mrl-sugimoto"].measures)


if __name__ == "__main__":
    unittest.main()
