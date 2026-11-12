"""Synthetic identity boundaries; no notebook, encoder or research artifacts are run."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import blf, isoformer, run_identity as identity, train  # noqa: E402
from shared_code import encoders  # noqa: E402
from shared_code.fusion import token_embeddings as tokens  # noqa: E402


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.config = train.TrainConfig(batch_size=2)
        self.rows = pd.DataFrame({"sequence": ["ATGAAA", "ATGCCC", "ATGGGG", "ATGTTT"],
                                  "length": [6] * 4, "Value": [1., 2., 3., 4.],
                                  "Split": ["train", "train", "val", "test"]})
        self.encoders = dict(blf.ENCODER_OF_MODALITY)
        self.caches = {m: {"width": 8, "encoder": key, "n_sequences": len(self.rows),
                           "input_sha256": tokens.input_fingerprint(tokens.inputs_for_rows(
                               encoders.ENCODERS[key], self.rows["sequence"].tolist())), "revision": None,
                           "shards": [{"name": "shard_00000.npz", "sha256": "a" * 64}]}
                       for m, key in self.encoders.items()}

    def make(self, **overrides):
        args = dict(rows=self.rows, benchmark="synthetic", variant="mil", config=self.config,
                    encoders=self.encoders, caches=self.caches)
        return identity.blf_identity(**{**args, **overrides})

    def test_order_labels_membership_and_cache_contents_change_identity(self):
        original = self.make()
        with self.assertRaisesRegex(ValueError, "ordered-input"):
            self.make(rows=self.rows.iloc[[1, 0, 2, 3]])
        for column, value in (("Value", 9.), ("Split", "test")):
            rows = self.rows.copy()
            rows.loc[0, column] = value
            self.assertNotEqual(original, self.make(rows=rows))
        changed_rows = self.rows.copy()
        changed_rows.loc[0, "sequence"] = "ATGCCA"
        with self.assertRaisesRegex(ValueError, "ordered-input"):
            self.make(rows=changed_rows)
        caches = deepcopy(self.caches)
        caches["rna"]["shards"][0]["sha256"] = "b" * 64
        self.assertNotEqual(original, self.make(caches=caches))
        self.assertIsNone(original["inputs"]["caches"]["rna"]["revision"])
        caches["rna"]["input_sha256"] = None
        with self.assertRaisesRegex(ValueError, "provenance is missing"):
            self.make(caches=caches)

    def test_defaults_are_explicit_and_producer_is_separate(self):
        made = self.make()
        self.assertEqual(made, self.make(model_kwargs={"attention_width": 100}))
        self.assertNotEqual(made, self.make(model_kwargs={"attention_width": 200}))
        self.assertNotEqual(made, self.make(config=replace(self.config, batch_size=3)))
        first, second = {"host": "a", "git": "docs-only-commit"}, {"host": "b"}
        self.assertEqual({"identity": made, "producer": first}["identity"],
                         {"identity": made, "producer": second}["identity"])
        with self.assertRaises(ValueError):
            self.make(model_kwargs={"unknown": True})
        with self.assertRaises(ValueError):
            identity.validate_model_request(made, model_kwargs={"attention_width": 200})
        identity.validate_model_request(made)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with self.assertRaisesRegex(ValueError, "float32"):
                identity.validate_model_request(made)
        finally:
            torch.set_default_dtype(previous)

    def test_cache_record_reads_headers_and_optional_memoized_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard_00000.npz"
            np.savez(shard, tokens=np.zeros((4, 8), dtype=np.float16), lengths=[2, 2])
            sha = hashlib.sha256(shard.read_bytes()).hexdigest()
            meta = {"n_shards": 1, "n_sequences": 2, "encoder": "fixture",
                    "shards": [{"name": shard.name, "sha256": sha}]}
            (root / "meta.json").write_text(json.dumps(meta))
            with patch.object(np, "load", side_effect=AssertionError("must not load arrays")):
                made = identity.cache_record(root)
            self.assertEqual(made["width"], 8)
            self.assertIsNone(made["input_sha256"])
            self.assertIsNone(made["revision"])
            from shared_code.fusion.token_embeddings import CacheVerification
            verification = CacheVerification()
            self.assertEqual(made, identity.cache_record(root, verification=verification))
            self.assertEqual(made, identity.cache_record(root, verification=verification))
            self.assertEqual(len(verification._sha256), 1)
            meta["shards"][0]["sha256"] = "0" * 64
            (root / "meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "contents differ"):
                identity.cache_record(root, verification=verification)

    def test_blf_loader_validation_never_iterates_or_reads_tokens(self):
        class NoTokens:
            width = 8
            def __getitem__(self, index):
                raise AssertionError("must not read cached tokens")
        streams = blf.BenchmarkStreams("synthetic", self.rows,
                                      {m: NoTokens() for m in self.encoders}, "regression")
        made = self.make(variant="unimodal:rna")
        loaders = {name: blf.loader(streams, name, batch_size=2, seed=0, modalities=("rna",))
                   for name in identity.SPLITS}
        before = loaders["train"].generator.get_state().clone()
        identity.validate_loaders(made, loaders)
        self.assertTrue(torch.equal(before, loaders["train"].generator.get_state()))
        loaders["val"] = blf.loader(streams, "val", batch_size=3, modalities=("rna",))
        with self.assertRaisesRegex(ValueError, "batch size"):
            identity.validate_loaders(made, loaders)
        loaders["val"] = blf.loader(streams, "val", batch_size=2, modalities=("rna",))
        loaders["val"].collate_fn = lambda rows: rows
        with self.assertRaisesRegex(ValueError, "collation"):
            identity.validate_loaders(made, loaders)

    def test_isoformer_tissue_order_actual_splits_and_budget(self):
        rows = pd.DataFrame({"DNA": ["AC"] * 4, "RNA": ["AU"] * 4,
                             "Protein": ["M"] * 4, "a": [1., 2., 3., 4.],
                             "b": [4., 3., 2., 1.]})
        caches = deepcopy(self.caches)
        for m, column in isoformer.STREAM_COLUMN.items():
            caches[m].update(encoder=isoformer.STREAM_ENCODER[m],
                             input_sha256=tokens.input_fingerprint(
                                 rows[column].str.replace("U", "T").tolist()))
        args = dict(rows=rows, benchmark="gtex", variant="dna+rna", config=self.config,
                    encoders=isoformer.STREAM_ENCODER, caches=caches, tissues=("a", "b"),
                    split_indices={"train": [1, 0], "val": [2], "test": [3]},
                    loader_settings={"budget": 1000})
        made = identity.isoformer_identity(**args)
        self.assertNotEqual(made, identity.isoformer_identity(**{**args, "tissues": ("b", "a")}))
        self.assertNotEqual(made, identity.isoformer_identity(
            **{**args, "loader_settings": {"budget": 2000}}))
        with self.assertRaisesRegex(ValueError, "ordered-input"):
            identity.isoformer_identity(**{**args, "crop": {"dna": None, "rna": 1, "prot": 1022}})
        changed = {"train": [0, 1], "val": [2], "test": [3]}
        self.assertNotEqual(made, identity.isoformer_identity(**{**args, "split_indices": changed}))
        identity.validate_model_request(made, isoformer.builder)
        with self.assertRaisesRegex(ValueError, "builder"):
            identity.validate_model_request(made)

    def test_all_blf_variants_have_matching_actual_loader_contracts(self):
        caches = {m: SimpleNamespace(width=8) for m in self.encoders}
        streams = blf.BenchmarkStreams("synthetic", self.rows, caches, "regression")
        variants = ["concat", "mil", "cross"] + [f"{kind}:{m}" for kind in
                    ("unimodal", "unimodal-grid") for m in identity.MODALITIES]
        for variant in variants:
            with self.subTest(variant=variant):
                made = self.make(variant=variant)
                modalities = made["model"]["modalities"]
                loaders = {s: blf.loader(streams, s, batch_size=2, modalities=modalities)
                           for s in identity.SPLITS}
                identity.validate_model_request(made)
                identity.validate_loaders(made, loaders)
                model = blf.build(streams, variant, channels=self.config.channels,
                                  tconv=self.config.tconv)
                self.assertEqual(list(getattr(model, "required_modalities", identity.MODALITIES)),
                                 modalities)

    def test_actual_isoformer_budgeted_loader_contract_before_iteration(self):
        rows = pd.DataFrame({"DNA": ["AC"] * 4, "RNA": ["AU"] * 4,
                             "Protein": ["M"] * 4, "a": [1., 2., 3., 4.],
                             "split": ["train", "train", "train", "test"]})
        caches = {m: SimpleNamespace(width=8, lengths=np.array([2, 2, 2, 2]))
                  for m in identity.MODALITIES}
        streams = isoformer.GTExStreams("gtex", rows, caches, ("a",))
        loaders = isoformer.loaders_for(streams, batch_size=2, seed=0, budget=2000)
        records = deepcopy(self.caches)
        for m, column in isoformer.STREAM_COLUMN.items():
            records[m].update(encoder=isoformer.STREAM_ENCODER[m],
                              input_sha256=tokens.input_fingerprint(
                                  rows[column].str.replace("U", "T").tolist()))
        made = identity.isoformer_identity(
            rows=rows, benchmark="gtex", variant="dna+rna+prot", config=self.config,
            encoders=isoformer.STREAM_ENCODER, caches=records, tissues=("a",),
            split_indices={s: loader.dataset.indices for s, loader in loaders.items()},
            loader_settings={"batch_size": 2, "budget": 2000})
        identity.validate_loaders(made, loaders)
        self.assertEqual(loaders["train"].batch_sampler.epoch, 0)
        loaders["train"].batch_sampler.budget = 1000
        with self.assertRaisesRegex(ValueError, "budget"):
            identity.validate_loaders(made, loaders)

    def test_legacy_completed_branch_does_not_inspect_caches_or_relabel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.json"
            path.write_text('{"variant": "mil"}', encoding="utf-8")
            before = path.read_bytes()
            with (patch.object(identity, "identity_for_streams", side_effect=AssertionError),
                  self.assertWarns(UserWarning)):
                records = identity.execution_records(SimpleNamespace(key="synthetic"), "mil",
                                                     self.config, directory, tag="historical")
            self.assertEqual(records, {"run_identity": None, "producer": None})
            self.assertEqual(before, path.read_bytes())
            path.write_text('{"identity": null}', encoding="utf-8")
            with self.assertWarns(UserWarning):
                identity.execution_records(SimpleNamespace(key="synthetic"), "mil",
                                           self.config, directory, tag="historical")
            path.write_text('{"identity": {"schema": "fusion.run.v2"}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                identity.execution_records(SimpleNamespace(key="synthetic"), "mil",
                                           self.config, directory, tag="historical")
            for invalid in ([], False, 0, "", {"schema": None}):
                path.write_text(json.dumps({"identity": invalid}), encoding="utf-8")
                with self.subTest(identity=invalid), self.assertRaises(ValueError):
                    identity.execution_records(SimpleNamespace(key="synthetic"), "mil",
                                               self.config, directory, tag="historical")

    def test_invalid_contracts_rejected_without_changing_input(self):
        made = self.make()
        before = deepcopy(made)
        self.assertEqual(identity.validate_identity(made, config=asdict(self.config)), made)
        self.assertEqual(made, before)
        for field in ("inputs", "model", "loader"):
            changed = deepcopy(made)
            del changed[field]
            with self.assertRaises(ValueError):
                identity.validate_identity(changed)
        changed = deepcopy(made)
        changed["config"]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            identity.validate_identity(changed)


class RecordedBLFInputsTests(unittest.TestCase):
    """Saved comparison metadata needs no model, cache or prediction loading."""

    def record(self, *, sequences=None, splits=None, task="regression"):
        sequences = sequences or ["ATGAAA", "ATGCCC", "ATGGGG", "ATGTTT"]
        rows = pd.DataFrame({"sequence": sequences, "length": [6] * 4})
        labels = np.ones(4, dtype=np.float32 if task == "regression" else np.int64)
        splits = splits or {"train": [0, 1], "val": [2], "test": [3]}
        inputs = {
            "encoders": {"dna": "dna-fixture", "rna": "rna-fixture", "prot": "protein-fixture"},
            "rows": identity._rows_record(rows, ("sequence", "length")),
            "targets": {**identity.array_record(labels), "columns": ["Value"],
                        "task": task, "transform": "as_supplied"},
            "splits": {name: identity.array_record(indices, np.int64)
                       for name, indices in splits.items()},
            "split_targets": {name: identity.array_record(labels[indices])
                              for name, indices in splits.items()},
            "caches": {"fixture": "never opened"},
        }
        config = {"batch_size": 2, "seed": 0}
        made = identity.make_identity(
            benchmark="fixture", variant="concat", config=config, inputs=inputs,
            model={"contract": "synthetic-model"},
            loader=identity.blf_loader_spec(config, identity.MODALITIES),
        )
        return {"benchmark": "fixture", "variant": "concat", "config": config, "identity": made}

    def test_signature_needs_no_predictions_and_does_not_mutate_or_alias_record(self):
        record = self.record()
        before = deepcopy(record)
        signature = identity.recorded_blf_inputs(record)
        self.assertEqual(set(signature), {"encoders", "rows", "targets", "splits", "split_targets"})
        self.assertEqual(record, before)
        signature["rows"]["sha256"] = "0" * 64
        self.assertEqual(record, before)
        self.assertNotIn("test_truth", record)
        self.assertNotIn("test_prediction", record)

    def test_identity_must_match_result_metadata(self):
        for field, value in (("benchmark", "other"), ("variant", "cross"),
                             ("config", {"batch_size": 2, "seed": 1})):
            with self.subTest(field=field):
                record = self.record()
                record[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    identity.recorded_blf_inputs(record)
        for field in ("benchmark", "variant", "config", "identity"):
            record = self.record()
            del record[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                identity.recorded_blf_inputs(record)

    def test_missing_or_partial_input_metadata_is_rejected(self):
        for field in ("encoders", "rows", "targets", "splits", "split_targets"):
            for replacement in (None, {}, []):
                record = self.record()
                record["identity"]["inputs"][field] = replacement
                with self.subTest(field=field, value=replacement), self.assertRaises(ValueError):
                    identity.recorded_blf_inputs(record)
        for field in ("splits", "split_targets"):
            record = self.record()
            del record["identity"]["inputs"][field]["val"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                identity.recorded_blf_inputs(record)

    def test_each_recorded_hash_must_be_valid_sha256(self):
        paths = [("rows",), ("targets",)] + [
            (field, name) for field in ("splits", "split_targets") for name in identity.SPLITS
        ]
        for path in paths:
            for invalid in (None, "", "a" * 63, "z" * 64, "A" * 64):
                record = self.record()
                descriptor = record["identity"]["inputs"]
                for key in path:
                    descriptor = descriptor[key]
                descriptor["sha256"] = invalid
                with self.subTest(path=path, sha=invalid), self.assertRaisesRegex(ValueError, "SHA-256"):
                    identity.recorded_blf_inputs(record)

    def test_scalar_shapes_counts_dtypes_and_declared_target_are_checked(self):
        changes = [
            (("rows", "count"), True), (("rows", "count"), 0),
            (("rows", "columns"), ["length", "sequence"]),
            (("targets", "columns"), ["other"]), (("targets", "transform"), "log"),
            (("targets", "task"), "expression"), (("targets", "dtype"), "<f8"),
            (("targets", "shape"), [4, 1]), (("targets", "shape"), [3]),
            (("splits", "train", "shape"), [True]),
            (("splits", "train", "dtype"), "<f4"),
            (("split_targets", "test", "shape"), [2]),
            (("split_targets", "test", "dtype"), "<i8"),
            (("encoders", "rna"), ""), (("encoders", "prot"), None),
        ]
        for path, value in changes:
            record = self.record()
            target = record["identity"]["inputs"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                identity.recorded_blf_inputs(record)
        record = self.record()
        for field in ("splits", "split_targets"):
            record["identity"]["inputs"][field]["train"]["shape"] = [1]
        with self.assertRaisesRegex(ValueError, "total rows.count"):
            identity.recorded_blf_inputs(record)

    def test_classification_requires_int64_targets(self):
        record = self.record(task="classification")
        self.assertEqual(identity.recorded_blf_inputs(record)["targets"]["dtype"], np.dtype(np.int64).str)
        record["identity"]["inputs"]["targets"]["dtype"] = np.dtype(np.float32).str
        with self.assertRaisesRegex(ValueError, "dtype"):
            identity.recorded_blf_inputs(record)

    def test_each_completed_fit_partition_must_be_nonempty(self):
        for empty in identity.SPLITS:
            splits = {name: [] for name in identity.SPLITS}
            other = [name for name in identity.SPLITS if name != empty]
            splits[other[0]], splits[other[1]] = [0, 1, 2], [3]
            record = self.record(splits=splits)
            inputs = record["identity"]["inputs"]
            self.assertEqual(sum(item["shape"][0] for item in inputs["splits"].values()), 4)
            self.assertEqual(inputs["split_targets"][empty]["shape"], [0])
            with self.subTest(empty=empty), self.assertRaisesRegex(ValueError, empty + " must contain rows"):
                identity.recorded_blf_inputs(record)

    def test_malformed_nested_records_raise_value_error(self):
        paths = [
            ("identity", "inputs"), ("identity", "model"), ("identity", "loader"),
            ("identity", "loader", "recovery"),
            ("identity", "inputs", "rows"), ("identity", "inputs", "targets"),
        ] + [("identity", "inputs", field, name)
             for field in ("splits", "split_targets") for name in identity.SPLITS]
        for path in paths:
            for invalid in (None, [], {}, False, "invalid", 1):
                record = self.record()
                parent = record
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = invalid
                with self.subTest(path=path, value=invalid), self.assertRaises(ValueError):
                    identity.recorded_blf_inputs(record)

    def test_equal_target_values_do_not_establish_equal_rows_or_partitions(self):
        original = identity.recorded_blf_inputs(self.record())
        reordered = identity.recorded_blf_inputs(self.record(
            sequences=["ATGCCC", "ATGAAA", "ATGGGG", "ATGTTT"],
        ))
        repartitioned = identity.recorded_blf_inputs(self.record(
            splits={"train": [0, 2], "val": [1], "test": [3]},
        ))
        for changed in (reordered, repartitioned):
            self.assertEqual(original["targets"], changed["targets"])
            self.assertEqual(original["split_targets"], changed["split_targets"])
            self.assertNotEqual(original, changed)
        self.assertNotEqual(original["rows"], reordered["rows"])
        self.assertNotEqual(original["splits"], repartitioned["splits"])

    def test_model_and_cache_differences_are_excluded_but_encoders_are_retained(self):
        record = self.record()
        signature = identity.recorded_blf_inputs(record)
        record["variant"] = record["identity"]["variant"] = "cross"
        record["identity"]["model"] = {"contract": "another-head"}
        record["identity"]["inputs"]["caches"] = {"different": "still never opened"}
        self.assertEqual(signature, identity.recorded_blf_inputs(record))
        record["identity"]["inputs"]["encoders"]["rna"] = "other-rna"
        self.assertNotEqual(signature, identity.recorded_blf_inputs(record))

    def test_non_blf_or_malformed_record_is_not_promoted(self):
        for record in (None, [], SimpleNamespace(), {"identity": {}}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                identity.recorded_blf_inputs(record)
        for loader_kind in ("isoformer", "custom"):
            record = self.record()
            record["identity"]["loader"]["kind"] = loader_kind
            with self.subTest(kind=loader_kind), self.assertRaisesRegex(ValueError, "BLF loader"):
                identity.recorded_blf_inputs(record)
        record = self.record()
        record["identity"]["config"]["unsupported"] = object()
        with self.assertRaises(ValueError):
            identity.recorded_blf_inputs(record)


class GTExRecoveryTests(unittest.TestCase):
    """A qualified receipt admits only its exact tiny pilot fixtures, without writes."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = pd.DataFrame({
            "DNA": ["ACGTAC", "TTTTTT", "GGGGGG", "CCCCCC"],
            "RNA": ["ACGUAC", "UUUUUU", "GGGGGG", "CCCCCC"],
            "Protein": ["MA", "MK", "ML", "MF"],
            "split": ["train", "train", "train", "test"],
            "a": [1., 2., 3., 4.], "b": [4., 3., 2., 1.],
        })
        self.crop = {"dna": None, "rna": 12000, "prot": 1022}
        _, texts = identity._gtex_inputs(self.rows, isoformer.STREAM_ENCODER, self.crop)
        caches, records = {}, {}
        for modality, key in isoformer.STREAM_ENCODER.items():
            encoder = encoders.ENCODERS[key]
            directory = self.root / "gtex_pilot" / f"{modality}_{key}"
            directory.mkdir(parents=True)
            lengths = np.full(4, 2 if modality == "prot" else 1, dtype=np.int32)
            shard = directory / "shard_00000.npz"
            np.savez(shard, tokens=np.zeros((int(lengths.sum()), 8), dtype=np.float16),
                     lengths=lengths)
            meta = {"shard_version": 1, "encoder": key, "checkpoint": encoder.checkpoint,
                    "revision": encoder.revision or "", "max_len": encoder.max_len,
                    "dtype": "float16", "n_sequences": 4, "chunk_size": 4, "n_shards": 1}
            metadata = directory / "meta.json"
            metadata.write_text(json.dumps(meta), encoding="utf-8")
            caches[modality] = SimpleNamespace(cache_dir=directory, lengths=lengths, width=8)
            records[modality] = {
                "directory": directory.name,
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                "reconstructed_input_sha256": tokens.input_fingerprint(texts[modality]),
                "width": 8, "total_tokens": int(lengths.sum()),
                "shards": [{"name": shard.name,
                            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}],
            }
        self.streams = isoformer.GTExStreams("gtex_pilot", self.rows, caches, ("a", "b"))
        self.record = {
            "schema": "gtex-pilot-recovery.v1", "status": "historically_reconstructed",
            "record_id": "synthetic-gtex-reconstruction", "benchmark": "gtex_pilot",
            "encoders": dict(isoformer.STREAM_ENCODER), "crop": self.crop,
            "rows": identity._rows_record(self.rows, ("DNA", "RNA", "Protein", "split")),
            "targets": identity.array_record(self.streams.labels), "tissues": ["a", "b"],
            "caches": records, "limitations": ["Original producer is unknown."],
        }
        self.path = self.root / "recovery.json"
        self.save_record()

    def save_record(self):
        self.path.write_text(json.dumps(self.record), encoding="utf-8")

    def snapshot(self):
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def make_fresh(self, modalities=identity.MODALITIES):
        for modality in modalities:
            path = self.streams.caches[modality].cache_dir / "meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update(input_sha256=self.record["caches"][modality]["reconstructed_input_sha256"],
                        shards=self.record["caches"][modality]["shards"])
            path.write_text(json.dumps(meta), encoding="utf-8")

    def test_fresh_caches_ignore_missing_or_corrupt_recovery_record_without_warning(self):
        self.make_fresh()
        config = train.TrainConfig(batch_size=2)
        loaders = isoformer.loaders_for(self.streams, batch_size=2)
        for record_path in (self.root / "missing.json", self.path):
            if record_path == self.path:
                self.path.write_text("not valid JSON", encoding="utf-8")
            before = self.snapshot()
            with (patch.object(identity, "gtex_recovered_cache_records", side_effect=AssertionError),
                  warnings.catch_warnings(record=True) as seen):
                warnings.simplefilter("always")
                made = identity.identity_for_streams(
                    self.streams, "dna+rna+prot", config, loaders=loaders,
                    verification=tokens.CacheVerification(), gtex_recovery=record_path)
            self.assertEqual(seen, [])
            self.assertTrue(all("input_provenance" not in c for c in made["inputs"]["caches"].values()))
            self.assertEqual(before, self.snapshot())

    def test_mixed_triple_qualifies_only_the_exact_historical_caches(self):
        self.make_fresh(("dna",))
        before = self.snapshot()
        config = train.TrainConfig(batch_size=2)
        loaders = isoformer.loaders_for(self.streams, batch_size=2)
        with self.assertWarnsRegex(UserWarning, "legacy caches"):
            made = identity.identity_for_streams(
                self.streams, "dna+rna+prot", config, loaders=loaders,
                verification=tokens.CacheVerification(), gtex_recovery=self.path)
        caches = made["inputs"]["caches"]
        self.assertNotIn("input_provenance", caches["dna"])
        for modality in ("rna", "prot"):
            self.assertEqual(caches[modality]["input_provenance"]["status"], "historically_reconstructed")
            self.assertEqual(caches[modality]["input_provenance"]["record_sha256"],
                             hashlib.sha256(self.path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "shard checksums"):
            identity.identity_for_streams(self.streams, "dna", config, loaders=loaders,
                                          verification=tokens.CacheVerification(), gtex_recovery=None)
        self.assertEqual(before, self.snapshot())

    def test_bad_modern_provenance_never_falls_back_even_in_a_mixed_triple(self):
        self.make_fresh(("dna",))
        path = self.streams.caches["dna"].cache_dir / "meta.json"
        valid = json.loads(path.read_text(encoding="utf-8"))
        missing_input = {k: v for k, v in valid.items() if k != "input_sha256"}
        missing_shards = {k: v for k, v in valid.items() if k != "shards"}
        bad_hash = deepcopy(valid)
        bad_hash["shards"][0]["sha256"] = "0" * 64
        cases = [[], missing_input, missing_shards, {**valid, "input_sha256": None},
                 {**valid, "input_sha256": "wrong"}, {**valid, "shards": None},
                 {**valid, "shards": []}, bad_hash]
        config = train.TrainConfig(batch_size=2)
        loaders = isoformer.loaders_for(self.streams, batch_size=2)
        for number, metadata in enumerate(cases):
            with self.subTest(case=number):
                path.write_text(json.dumps(metadata), encoding="utf-8")
                before = self.snapshot()
                with (patch.object(identity, "gtex_recovered_cache_records", side_effect=AssertionError),
                      self.assertRaises(ValueError)):
                    identity.identity_for_streams(
                        self.streams, "dna", config, loaders=loaders,
                        verification=tokens.CacheVerification(), gtex_recovery=self.path)
                self.assertEqual(before, self.snapshot())

    def test_recovery_helper_cannot_relabel_modern_metadata_as_historical(self):
        self.make_fresh(("dna",))
        path = self.streams.caches["dna"].cache_dir / "meta.json"
        self.record["caches"]["dna"]["metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_record()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "modern provenance fields"):
            identity.gtex_recovered_cache_records(self.streams, self.path, modalities=("dna",))
        self.assertEqual(before, self.snapshot())

    def test_default_is_strict_and_qualified_recovery_preserves_every_byte(self):
        config = train.TrainConfig(batch_size=2)
        loaders = isoformer.loaders_for(self.streams, batch_size=2)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "shard checksums"):
            identity.identity_for_streams(self.streams, "dna+rna+prot", config, loaders=loaders)
        verification = tokens.CacheVerification()
        with self.assertWarnsRegex(UserWarning, "historical reconstruction"):
            made = identity.identity_for_streams(
                self.streams, "dna+rna+prot", config, loaders=loaders,
                gtex_recovery=self.path, verification=verification)
        identity.validate_loaders(made, loaders)
        persisted = json.loads(json.dumps(made))
        for cache in persisted["inputs"]["caches"].values():
            self.assertEqual(cache["input_provenance"], {
                "status": "historically_reconstructed",
                "record_id": self.record["record_id"],
                "record_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            })
        self.assertEqual(persisted["inputs"]["caches"]["prot"]["revision"], "")
        with self.assertWarns(UserWarning):
            identity.gtex_recovered_cache_records(self.streams, self.path, verification=verification)
        self.assertEqual(len(verification._sha256), 3)
        self.assertEqual(before, self.snapshot())

    def test_rows_targets_tissues_and_crop_must_match(self):
        before = self.snapshot()
        for column, value, reason in (("DNA", "AAAAAA", "ordered rows"),
                                      ("split", "test", "split membership"),
                                      ("a", 9., "ordered targets")):
            with self.subTest(column=column):
                changed = self.rows.copy()
                changed.loc[0, column] = value
                streams = isoformer.GTExStreams("gtex_pilot", changed, self.streams.caches, ("a", "b"))
                with self.assertRaisesRegex(ValueError, reason):
                    identity.gtex_recovered_cache_records(streams, self.path)
        changed = isoformer.GTExStreams("gtex_pilot", self.rows.iloc[::-1], self.streams.caches, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "ordered rows"):
            identity.gtex_recovered_cache_records(changed, self.path)
        changed = isoformer.GTExStreams("gtex_pilot", self.rows, self.streams.caches, ("b", "a"))
        with self.assertRaisesRegex(ValueError, "tissue order"):
            identity.gtex_recovered_cache_records(changed, self.path)
        with self.assertRaisesRegex(ValueError, "crop"):
            identity.gtex_recovered_cache_records(self.streams, self.path,
                                                 crop={**self.crop, "rna": 6})
        self.assertEqual(before, self.snapshot())

    def test_metadata_and_shard_bytes_are_checked_even_without_verifier(self):
        directory = self.streams.caches["dna"].cache_dir
        metadata = directory / "meta.json"
        metadata.write_bytes(metadata.read_bytes() + b"\n")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "metadata bytes"):
            identity.gtex_recovered_cache_records(self.streams, self.path)
        self.assertEqual(before, self.snapshot())
        metadata.write_bytes(metadata.read_bytes()[:-1])
        shard = directory / "shard_00000.npz"
        shard.write_bytes(shard.read_bytes() + b"changed")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "contents differ"):
            identity.gtex_recovered_cache_records(self.streams, self.path)
        self.assertEqual(before, self.snapshot())

    def test_record_cannot_admit_other_benchmarks_encoders_or_directories(self):
        before = self.snapshot()
        other = isoformer.GTExStreams("another_benchmark", self.rows, self.streams.caches, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "only gtex_pilot"):
            identity.gtex_recovered_cache_records(other, self.path)
        for field, value, reason in (
                ("encoders", {**isoformer.STREAM_ENCODER, "rna": "rnafm"}, "encoder selection"),
                ("status", "recorded_at_extraction", "contract")):
            old = self.record[field]
            self.record[field] = value
            self.save_record()
            with self.assertRaisesRegex(ValueError, reason):
                identity.gtex_recovered_cache_records(self.streams, self.path)
            self.record[field] = old
        self.record["caches"]["dna"]["directory"] = "another_cache"
        self.save_record()
        with self.assertRaisesRegex(ValueError, "directory"):
            identity.gtex_recovered_cache_records(self.streams, self.path)
        self.record["caches"]["dna"]["directory"] = "dna_ntv2_100m"
        self.save_record()
        self.assertEqual(before, self.snapshot())

    def test_legacy_completed_result_does_not_require_the_recovery_record(self):
        destination = self.root / "historical.json"
        destination.write_text('{"variant":"IsoFormer"}', encoding="utf-8")
        before = self.snapshot()
        with self.assertWarnsRegex(UserWarning, "historical completed"):
            made = identity.execution_records(
                self.streams, "dna", train.TrainConfig(), self.root, tag="historical",
                gtex_recovery=self.root / "missing-record.json")
        self.assertEqual(made, {"run_identity": None, "producer": None})
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
