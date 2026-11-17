"""A6 input/cache/batch checks on invented sequences and mocked encoder states only."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import region_experiment as region
from shared_code.fusion import token_embeddings as tokens


def track(length, start, end):
    values = np.zeros(length, dtype=np.int8)
    values[start:end:3] = 1
    return values


def example_frame():
    # Both CDSs include a terminal stop; the first also has two-base UTRs.
    return pd.DataFrame({"sequence": ["GGATGAAATAACC", "ATGGGGTAA"],
                         "cds": [track(13, 2, 11), track(9, 0, 9)],
                         "gene": ["g1", "g2"], "target": [1.25, -0.75]})


def cohort(frame=None, **kwargs):
    return region.prepare_cohort(example_frame() if frame is None else frame,
                                 "rnahl-human", grouping="gene", **kwargs)


def fake_states(text, tokenizer, model, max_len, dtype):
    width = 3 if tokenizer == "esm2_8m" else 2
    return np.repeat(np.array([ord(letter) for letter in text], dtype=dtype)[:, None],
                     width, axis=1)


class RegionInputTests(unittest.TestCase):
    def test_local_reader_never_downloads_and_records_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(region.mrnabench, "download", side_effect=AssertionError("network")):
                with self.assertRaises(FileNotFoundError):
                    region.load_local_frame("rnahl-human", directory)
                path = Path(directory) / "rnahl-human.parquet"
                path.write_bytes(b"synthetic file, read by a test callback")
                frame, source = region.load_local_frame("rnahl-human", directory,
                                                        reader=lambda _: example_frame())
            self.assertEqual(source["sha256"], tokens.file_fingerprint(path))
            self.assertEqual(source["bytes"], path.stat().st_size)
            self.assertEqual(len(frame), 2)

    def test_source_change_during_read_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rnahl-human.parquet"
            path.write_bytes(b"before")

            def changing_reader(source):
                source.write_bytes(b"after")
                return example_frame()

            with self.assertRaisesRegex(ValueError, "changed"):
                region.load_local_frame("rnahl-human", directory, reader=changing_reader)

    def test_common_cohort_retains_stop_and_records_all_rejected_rows(self):
        frame = example_frame()
        bad = pd.DataFrame({"sequence": ["ATGTAAGGG", "ATGNNNTAA", "ATG"],
                            "cds": [track(9, 0, 9), track(9, 0, 9), track(3, 0, 3)],
                            "gene": ["g3", "g4", ""], "target": [1., 2., np.nan]})
        source = pd.concat([frame, bad], ignore_index=True)
        before = region._frame_hash(source)
        result = cohort(source)
        self.assertEqual(len(result.frame), 2)
        self.assertEqual([r["source_position"] for r in result.exclusions], [2, 3, 4])
        self.assertIn("internal stop", result.exclusions[0]["reasons"][0])
        self.assertIn("ambiguous", result.exclusions[1]["reasons"][0])
        self.assertEqual(len(result.exclusions[2]["reasons"]), 2)
        self.assertTrue(result.layouts[0].has_terminal_stop)
        self.assertEqual(result.layouts[0].protein, "MK")
        self.assertEqual(result.layouts[0].cds_mask.sum(), 9)
        self.assertEqual(result.layouts[0].protein_mask.sum(), 6)
        self.assertEqual(region._frame_hash(source), before)

    def test_protein_mismatch_excludes_same_row_for_every_encoder(self):
        frame = example_frame().assign(protein=["MK", "WRONG"])
        result = cohort(frame, supplied_protein_column="protein")
        plans = region.plan_chunks(result, window_lengths={key: 4 for key in region.ENCODER_KEYS})
        self.assertEqual(len(result.frame), 1)
        self.assertIn("does not match", result.exclusions[0]["reasons"][0])
        self.assertEqual({plan.identity["cohort_sha256"] for plan in plans.values()},
                         {result.identity["sha256"]})

    def test_frame_target_and_row_order_changes_have_distinct_identity(self):
        original = cohort()
        changed = cohort(example_frame().assign(target=[1.5, -0.75]))
        reordered = cohort(example_frame().iloc[::-1])
        self.assertNotEqual(original.identity["sha256"], changed.identity["sha256"])
        self.assertNotEqual(original.identity["targets"]["sha256"], changed.identity["targets"]["sha256"])
        self.assertNotEqual(original.identity["source_frame_sha256"], reordered.identity["source_frame_sha256"])

    def test_source_location_is_portable_and_prepared_mutation_is_refused(self):
        first = cohort(source={"path": "desktop/table.parquet", "sha256": "a" * 64})
        moved = cohort(source={"path": "/newton/table.parquet", "sha256": "a" * 64})
        self.assertEqual(first.identity, moved.identity)
        self.assertNotEqual(first.source, moved.source)
        first.frame.loc[0, "target"] = 9
        with self.assertRaisesRegex(ValueError, "Prepared cohort changed"):
            region.plan_chunks(first, window_lengths={"rnafm": 4})

    def test_standalone_dataset_cannot_skip_changed_cohort_validation(self):
        selected = cohort()
        selected.frame.at[0, "target"] += 1
        # The direct constructor must reject before even consulting a cache object.
        with self.assertRaisesRegex(ValueError, "Prepared cohort changed"):
            region.RegionDataset(selected, None, rna_encoder="rnafm", grid="native")

    def test_utr_ambiguity_is_excluded_and_rna_alphabet_is_coordinate_preserving(self):
        frame = example_frame()
        frame.at[0, "sequence"] = "NGATGAAATAACC"
        frame.at[1, "sequence"] = "augggguaa"
        accepted = cohort(frame)
        self.assertEqual(accepted.row_ids, ("rnahl-human:1",))
        self.assertEqual(accepted.frame.loc[0, "sequence"], "ATGGGGTAA")
        self.assertIn("ambiguous", accepted.exclusions[0]["reasons"][0])

    def test_homology_is_explicit_and_has_no_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "nonempty map"):
            region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology")
        with self.assertRaisesRegex(ValueError, "No homology assignment"):
            region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                   homology_map={"g1": "family"})
        result = region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                       homology_map={"g1": "family"}, unmapped_gene_policy="gene")
        self.assertEqual(result.groups.tolist(), ["homology:family", "gene:g2"])
        together = region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                         homology_map={"g1": "family", "g2": "family"})
        self.assertEqual(len(set(together.groups)), 1)

    def test_unmapped_gene_policy_preserves_rows_targets_and_group_namespaces(self):
        source = pd.concat([example_frame()] * 3, ignore_index=True)
        source["gene"] = ["mapped_a", "shared", "shared", "mapped_b",
                          "homology:shared", "gene:shared"]
        source["target"] = [1.25, -.75, 9.5, 2.0, -4.0, 0.125]
        source.index = [20, 10, 40, 30, 60, 50]
        before = source.copy(deep=True)
        before_hash = region._frame_hash(source)
        prepared = region.prepare_cohort(
            source, "rnahl-human", grouping="homology", unmapped_gene_policy="gene",
            homology_map={"mapped_a": "shared", "mapped_b": "gene:shared"},
        )
        self.assertEqual(prepared.groups.tolist(), [
            "homology:shared", "gene:shared", "gene:shared", "homology:gene:shared",
            "gene:homology:shared", "gene:gene:shared",
        ])
        self.assertEqual(len(set(prepared.groups)), 5)
        self.assertEqual(prepared.row_ids, tuple(f"rnahl-human:{i}" for i in range(6)))
        self.assertEqual(prepared.exclusions, ())
        pd.testing.assert_frame_equal(prepared.frame, before.reset_index(drop=True))
        np.testing.assert_array_equal(prepared.labels[:, 0], before["target"].to_numpy())
        pd.testing.assert_frame_equal(source, before)
        self.assertEqual(region._frame_hash(source), before_hash)

    def test_gene_fallback_still_requires_nonempty_map_and_explicit_opt_in(self):
        for missing in (None, {}):
            with self.subTest(mapping=missing), self.assertRaisesRegex(ValueError, "nonempty map"):
                region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                       homology_map=missing, unmapped_gene_policy="gene")
        with self.assertRaisesRegex(ValueError, "No homology assignment"):
            region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                   homology_map={"g1": "family"})
        mapping = {"g1": "family1", "g2": "family2"}
        strict = region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                       homology_map=mapping)
        fallback = region.prepare_cohort(example_frame(), "rnahl-human", grouping="homology",
                                         homology_map=mapping, unmapped_gene_policy="gene")
        np.testing.assert_array_equal(strict.groups, fallback.groups)
        self.assertNotEqual(strict.identity["sha256"], fallback.identity["sha256"])

    def test_local_homology_map_refuses_duplicate_and_missing_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                region.load_local_homology_map(directory)
            folder = Path(directory) / "homology_maps"
            folder.mkdir()
            path = folder / "human_homology_map.csv"
            path.write_text("gene_name,gene_group\ng1,family\ng2,family\n", encoding="utf-8")
            mapping, source = region.load_local_homology_map(directory)
            self.assertEqual(mapping, {"g1": "family", "g2": "family"})
            self.assertEqual(source["sha256"], tokens.file_fingerprint(path))
            path.write_text("gene_name,gene_group\ng1,family\ng1,other\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeated"):
                region.load_local_homology_map(directory)

    def test_window_plan_retains_every_base_and_residue_in_order(self):
        result = cohort()
        plans = region.plan_chunks(result, window_lengths={"rnafm": 4, "rinalmo": 5, "esm2_8m": 1})
        self.assertEqual(plans["rnafm"].texts[:4], ["GGAU", "GAAA", "UAAC", "C"])
        self.assertEqual(plans["esm2_8m"].texts, ["M", "K", "M", "G"])
        for key, plan in plans.items():
            for row, indices in enumerate(plan.row_chunks):
                chunks = [plan.chunks[i] for i in indices]
                expected = result.layouts[row].protein if key == "esm2_8m" else result.frame.iloc[row]["sequence"].replace("T", "U")
                self.assertEqual("".join(chunk.text for chunk in chunks), expected)
                self.assertEqual(chunks[0].start, 0)
                self.assertEqual(chunks[-1].end, len(expected))
                self.assertEqual([chunk.start for chunk in chunks[1:]], [chunk.end for chunk in chunks[:-1]])
        for windows in ({}, {"nt500m": 10}, {"rnafm": 1023}, {"rnafm": True}):
            with self.assertRaises(ValueError):
                region.plan_chunks(result, window_lengths=windows)

    def test_pooling_respects_cds_frame_and_partial_utr_edges(self):
        layout = cohort().layouts[0]
        self.assertEqual(region.codon_pool_spans(layout), ((0, 2), (2, 5), (5, 8), (8, 11), (11, 13)))


class RegionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cohort = cohort()
        self.plans = region.plan_chunks(self.cohort, window_lengths={key: 4 for key in region.ENCODER_KEYS})
        self.encoder_patch = patch.object(tokens, "load_encoder", side_effect=lambda enc, device: (enc.key, Mock()))
        self.state_patch = patch.object(tokens, "token_states", side_effect=fake_states)
        self.encoder_patch.start()
        self.state_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        self.addCleanup(self.state_patch.stop)

    def extract(self):
        def quiet_extract(*args, **kwargs):
            return tokens.extract(*args, progress=None, **kwargs)
        return region.open_token_caches(self.plans, self.root, mode="extract", device="cpu", chunk_size=2,
                                        extractor=quiet_extract)

    def test_metadata_inventory_does_not_create_or_verify_token_files(self):
        empty = self.root / "missing"
        with patch.object(tokens, "inspect_cache", side_effect=AssertionError("tokens inspected")):
            rows = region.cache_inventory(self.plans, empty, chunk_size=2)
        self.assertFalse(empty.exists())
        self.assertEqual({row["status"] for row in rows}, {"not_ready"})
        with self.extract():
            pass
        with patch.object(tokens, "inspect_cache", side_effect=AssertionError("tokens inspected")):
            rows = region.cache_inventory(self.plans, self.root, chunk_size=2)
        self.assertEqual({row["status"] for row in rows}, {"recorded_complete_unverified"})
        self.assertFalse(any(row["verified"] for row in rows))

    def test_read_mode_never_extracts_and_requires_complete_identity(self):
        extractor = Mock(side_effect=AssertionError("extraction called in read mode"))
        with self.assertRaisesRegex(ValueError, "input record is missing"):
            region.open_token_caches(self.plans, self.root, mode="read", extractor=extractor, chunk_size=2)
        with self.extract():
            pass
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with region.open_token_caches(self.plans, self.root, mode="read", extractor=extractor, chunk_size=2) as caches:
            self.assertEqual(caches.row_features("rnafm", 0, 13).shape, (13, 2))
        self.assertEqual(original, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})
        extractor.assert_not_called()
        next(self.root.glob("**/shard_00000.npz")).unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            region.open_token_caches(self.plans, self.root, mode="read", extractor=extractor, chunk_size=2)

    def test_reuse_preserves_complete_bytes_and_no_model_is_reloaded(self):
        with self.extract():
            pass
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(tokens, "load_encoder", side_effect=AssertionError("unnecessary model load")):
            with self.extract():
                pass
        self.assertEqual(original, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_changed_window_or_target_refuses_old_cache_without_overwrite(self):
        with self.extract():
            pass
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        changed = region.plan_chunks(self.cohort, window_lengths={"rnafm": 3})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            region.open_token_caches(changed, self.root, mode="extract", device="cpu", chunk_size=2)
        changed_targets = cohort(example_frame().assign(target=[1., 2.]))
        changed = region.plan_chunks(changed_targets, window_lengths={"rnafm": 4})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            region.open_token_caches(changed, self.root, mode="read", chunk_size=2)
        self.assertEqual(original, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_shard_corruption_is_refused_without_repair(self):
        with self.extract():
            pass
        path = next(self.root.glob("**/shard_00000.npz"))
        path.write_bytes(b"broken synthetic shard")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            region.open_token_caches(self.plans, self.root, mode="read", chunk_size=2)
        self.assertEqual(path.read_bytes(), b"broken synthetic shard")

    def test_failed_second_reader_closes_first_and_preserves_primary_error(self):
        with self.extract():
            pass
        cache = Mock(lengths=np.array([c.end - c.start for c in self.plans["rnafm"].chunks]))
        cache.__len__ = Mock(return_value=len(cache.lengths))
        cache.close.side_effect = RuntimeError("cleanup secondary")
        reader = Mock(side_effect=[cache, ValueError("reader primary")])
        with self.assertRaisesRegex(ValueError, "reader primary") as caught:
            region.open_token_caches(self.plans, self.root, mode="read", reader=reader, chunk_size=2)
        cache.close.assert_called_once()
        self.assertIn("cleanup secondary", " ".join(caught.exception.__notes__))

    def test_native_alignment_stop_utr_masks_and_padding(self):
        with self.extract() as caches:
            dataset = region.RegionDataset(self.cohort, caches, rna_encoder="rnafm", grid="native")
            first = dataset[0]
            self.assertTrue(np.all(first["streams"]["prot"][:2] == 0))
            self.assertTrue(np.all(first["streams"]["prot"][8:] == 0))
            np.testing.assert_array_equal(first["streams"]["prot"][2:5], np.full((3, 3), ord("M")))
            batch = region.collate_region_rows([first, dataset[1]])
            self.assertEqual(batch.streams["rna"].shape, (2, 13, 2))
            self.assertEqual(batch.streams["prot"].shape, (2, 13, 3))
            self.assertEqual(batch.lengths.tolist(), [13, 9])
            self.assertEqual(batch.region[0].sum().item(), 9)
            self.assertEqual(batch.masks["prot"][0].sum().item(), 6)
            self.assertFalse(batch.region[1, 9:].any())
            self.assertFalse(batch.masks["rna"][1, 9:].any())
            self.assertEqual(batch.row_ids, self.cohort.row_ids)
            self.assertEqual(batch.to("cpu").labels.tolist(), [[1.25], [-0.75]])

    def test_codon_pool_averages_only_actual_bases_and_retains_stop_mask(self):
        with self.extract() as caches:
            native = region.RegionDataset(self.cohort, caches, rna_encoder="rinalmo", grid="native")[0]
            pooled = region.RegionDataset(self.cohort, caches, rna_encoder="rinalmo", grid="codon_pool")[0]
            self.assertEqual(pooled["positions"], ((0, 2), (2, 5), (5, 8), (8, 11), (11, 13)))
            expected = np.repeat(np.array([71, 221 / 3, 65, 215 / 3, 67])[:, None], 2, axis=1)
            np.testing.assert_allclose(pooled["streams"]["rna"], expected, rtol=1e-6)
            self.assertEqual(pooled["region"].tolist(), [False, True, True, True, False])
            self.assertEqual(pooled["masks"]["prot"].tolist(), [False, True, True, False, False])

    def test_loader_order_and_subset_identity_are_explicit(self):
        with self.extract() as caches:
            dataset = region.RegionDataset(self.cohort, caches, rna_encoder="rnafm", grid="native", indices=[1, 0])
            loader = region.make_loader(dataset, batch_size=2, shuffle=False, seed=0)
            self.assertEqual(next(iter(loader)).row_ids, tuple(reversed(self.cohort.row_ids)))
            for indices in ([0, 0], [-1], [2], [0.5]):
                with self.assertRaises(ValueError):
                    region.RegionDataset(self.cohort, caches, rna_encoder="rnafm", grid="native", indices=indices)
            with self.assertRaisesRegex(ValueError, "num_workers"):
                region.make_loader(dataset, batch_size=2, shuffle=False, seed=0, num_workers=1)

    def test_rna_only_batches_do_not_read_protein_tokens(self):
        with self.extract() as caches:
            rna_only = region.RegionCaches({"rnafm": caches.plans["rnafm"]},
                                            {"rnafm": caches.caches["rnafm"]})
            dataset = region.RegionDataset(self.cohort, rna_only, rna_encoder="rnafm",
                                            grid="codon_pool", include_protein=False)
            batch = region.collate_region_rows([dataset[0], dataset[1]])
            self.assertEqual(set(batch.streams), {"rna"})
            self.assertEqual(set(batch.masks), {"rna"})
            self.assertEqual(batch.lengths.tolist(), [5, 3])

    def test_mask_donors_are_supplied_not_invented_or_rescaled(self):
        with self.assertRaisesRegex(ValueError, "different donor"):
            region.validate_mask_donors(self.cohort, [0, 1])
        with self.assertRaisesRegex(ValueError, "equal transcript length"):
            region.validate_mask_donors(self.cohort, [1, 0])
        same_length = example_frame().iloc[[0, 0]].copy().reset_index(drop=True)
        same_length.at[1, "gene"] = "g2"
        same_length.at[1, "sequence"] = "ATGAAATAACCCC"
        same_length.at[1, "cds"] = track(13, 0, 9)
        accepted = cohort(same_length)
        self.assertEqual(region.validate_mask_donors(accepted, [1, 0]).tolist(), [1, 0])
        for donors in ([1, 1], [1.0, 0.0], [False, True]):
            with self.assertRaises(ValueError):
                region.validate_mask_donors(accepted, donors)


if __name__ == "__main__":
    unittest.main()
