"""Tiny invented OOF/correction execution, controls and preservation checks.

The few optimizer updates use two-feature synthetic token arrays on CPU. No research
dataset, pretrained encoder, notebook, network operation or cluster job is involved.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import anchor_correction as evidence
from shared_code.fusion import blf, correction_experiment as experiment, run_identity, train
from shared_code.fusion import token_embeddings as tokens
from shared_code.encoders import ENCODERS


class Cache:
    def __init__(self, values):
        self.values = values
        self.width = values[0].shape[1]
        self.lengths = np.asarray([len(value) for value in values])
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.values[index]


def fixture():
    rng = np.random.default_rng(4)
    sequences = ["ATG" + "".join("ACGT"[(index // 4 ** place) % 4] for place in range(12)) + "TAA"
                 for index in range(12)]
    rows = pd.DataFrame({
        "sequence": sequences, "length": [18] * 12, "Value": np.linspace(-1, 1, 12),
        "Split": ["train"] * 6 + ["val"] * 3 + ["test"] * 3,
    })
    encoders = dict(blf.ENCODER_OF_MODALITY)
    caches = {name: Cache([rng.normal(size=(count, 2)).astype(np.float16) for _ in sequences])
              for name, count in (("dna", 3), ("rna", 18), ("prot", 6))}
    streams = blf.BenchmarkStreams("invented", rows, caches, "regression", encoders=encoders)
    descriptors = {
        name: {"encoder": key, "n_sequences": len(rows), "width": 2,
               "input_sha256": tokens.input_fingerprint(tokens.inputs_for_rows(ENCODERS[key], sequences)),
               "shards": [{"name": "invented.npz", "sha256": name[0] * 64}]}
        for name, key in encoders.items()
    }
    context = {"row_ids": tuple(range(12)), "groups": tuple(sequences),
               "split_positions": {"train": tuple(range(6)), "val": (6, 7, 8), "test": (9, 10, 11)}}
    return streams, descriptors, context


def protocol():
    return {"question": "synthetic correction", "offset_sources": {
        "final_anchor_sha256": "a" * 64, "oof_artifact_sha256": ["b" * 64],
    }}


def config():
    return train.TrainConfig(batch_size=3, max_epochs=2, early_stopping_patience=1,
                             plateau_patience=1, channels=2, seed=0, tconv="equation")


class FoldPlanTests(unittest.TestCase):
    def test_five_folds_have_group_exclusion_and_inner_stopping(self):
        groups = tuple(f"group-{index // 2}" for index in range(240)) + ("val", "test")
        options = dict(n_splits=5, seed=42, inner_seed=42, stop_fraction=.2)
        plans = experiment.fold_plan(range(240), groups, **options)
        evidence.validate_plan(plans, row_ids=range(242), groups=groups, train_positions=range(240),
                               validation_positions=(240,), test_positions=(241,))
        self.assertEqual(len(plans), 5)
        self.assertTrue(all(plan.fit and plan.stop and plan.predict for plan in plans))
        self.assertEqual(plans, experiment.fold_plan(range(240), groups, **options))

    def test_assignments_are_group_stable_under_row_reordering(self):
        groups = tuple(f"group-{index}" for index in range(120))
        options = dict(n_splits=5, seed=42, inner_seed=12, stop_fraction=.2)
        first = experiment.fold_plan(range(120), groups, **options)
        second = experiment.fold_plan(range(120), groups[::-1], **options)
        for original, reverse in zip(first, second):
            for role in ("fit", "stop", "predict"):
                self.assertEqual({groups[index] for index in getattr(original, role)},
                                 {groups[::-1][index] for index in getattr(reverse, role)})

    def test_empty_or_invalid_choices_are_not_automatically_repaired(self):
        for options in (
            dict(n_splits=True, seed=42, inner_seed=42, stop_fraction=.2),
            dict(n_splits=5, seed=-1, inner_seed=42, stop_fraction=.2),
            dict(n_splits=5, seed=42, inner_seed=42, stop_fraction=0),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                experiment.fold_plan(range(20), tuple(map(str, range(20))), **options)
        with self.assertRaises(ValueError):
            experiment.fold_plan((0, 1), ("one", "two"), n_splits=5, seed=42,
                                 inner_seed=42, stop_fraction=.2)

    def test_oof_orchestration_passes_only_inner_roles_and_retains_completed_folds(self):
        streams, records, context = fixture()
        plans = (
            evidence.FoldRows((2, 3), (4, 5), (1, 0)),
            evidence.FoldRows((0, 1), (4, 5), (3, 2)),
            evidence.FoldRows((0, 1), (2, 3), (5, 4)),
        )
        calls = []

        def fit_stub(actual, **kwargs):
            calls.append(kwargs)
            positions = kwargs["split_positions"]["test"]
            values = [float(index) + .2 for index in positions]
            return experiment.FitArtifact(Path("unused"), {"predictions": {"test": {
                "row_ids": [context["row_ids"][index] for index in positions], "values": values,
            }}})

        with patch.object(experiment, "run_neural", side_effect=fit_stub):
            oof, artifacts = experiment.run_anchor_oof(
                streams, plans=plans, row_ids=context["row_ids"], groups=context["groups"],
                train_positions=range(6), validation_positions=(6, 7, 8), test_positions=(9, 10, 11),
                output_dir=Path("unused"), input_contract={"test": True},
            )
        np.testing.assert_allclose(oof.values, np.arange(6) + .2)
        self.assertEqual(len(artifacts), 3)
        for plan, call in zip(plans, calls):
            self.assertEqual(call["split_positions"], {"train": plan.fit, "val": plan.stop, "test": plan.predict})
            self.assertEqual(call["role"], "anchor")
            self.assertEqual(call["control"], {"kind": "real"})
            self.assertEqual(call["input_contract"]["outer_partitions"]["test_positions"], [9, 10, 11])
            self.assertFalse(np.any(call["offsets"]))


class ControlTests(unittest.TestCase):
    def test_row_shuffle_preserves_every_donor_and_duplicate_frequencies(self):
        groups = ("a", "a", "a", "b", "c", "c", "untouched")
        strata = {group: "all" for group in groups[:6]}
        original = list(groups)
        for seed in (0, 1, 7):
            mapping = experiment.row_permutation(groups, range(6), strata_by_group=strata, seed=seed)
            self.assertEqual(sorted(mapping[:6]), list(range(6)))
            self.assertEqual(mapping[6], 6)
            self.assertTrue(all(groups[row] != groups[mapping[row]] for row in range(6)))
            self.assertEqual(sorted(groups[row] for row in mapping[:6]), sorted(groups[:6]))
            self.assertEqual(len({groups[row] for row in mapping[:3]}), 2)
            np.testing.assert_array_equal(mapping, experiment.row_permutation(
                groups, reversed(range(6)), strata_by_group=dict(reversed(list(strata.items()))), seed=seed))
        self.assertEqual(list(groups), original)

    def test_row_shuffle_is_partition_and_stratum_local(self):
        groups = ("a", "a", "b", "b", "c", "d", "outside", "elsewhere")
        strata = {"a": "one", "b": "one", "c": "two", "d": "two"}
        mapping = experiment.row_permutation(groups, range(6), strata_by_group=strata, seed=3)
        self.assertEqual(set(mapping[:4]), set(range(4)))
        self.assertEqual(mapping[4:6].tolist(), [5, 4])
        self.assertEqual(mapping[6:].tolist(), [6, 7])

    def test_row_shuffle_rejects_infeasible_stratum_without_dropping_rows(self):
        cases = [
            (("a", "a", "a", "b", "c"), {"a": "all", "b": "all", "c": "all"}),
            (("a", "a", "b", "c", "d", "e"),
             {"a": "small", "b": "small", "c": "other", "d": "other", "e": "other"}),
            (("a",), {"a": "all"}),
        ]
        for groups, strata in cases:
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "infeasible"):
                experiment.row_permutation(groups, range(len(groups)), strata_by_group=strata, seed=0)
        with self.assertRaisesRegex(ValueError, "exactly one explicit stratum"):
            experiment.row_permutation(("a", "b"), (0, 1), strata_by_group={"a": "all"}, seed=0)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            experiment.row_permutation(("a", "b"), (0, 1),
                                       strata_by_group={"a": "all", "b": "all"}, seed=True)

    def test_row_shuffle_validation_and_historical_contract_stay_distinct(self):
        streams, _, context = fixture()
        groups = ("a", "a", "a", "b", "c", "c", "d", "e", "f", "g", "h", "i")
        splits = {key: np.asarray(value) for key, value in context["split_positions"].items()}
        mapping, strata = np.arange(12), {}
        for name, indices in splits.items():
            strata[name] = {groups[index]: "all" for index in indices}
            mapping[indices] = experiment.row_permutation(
                groups, indices, strata_by_group=strata[name], seed=0)[indices]
        control = {"kind": experiment.ROW_SHUFFLE, "row_map": mapping.tolist(), "seed": 0, "strata": strata}
        self.assertEqual(experiment._control(control, streams, splits, groups, ("rna",)), control)
        original = json.dumps(control, sort_keys=True)
        for mutation, message in (
            (lambda item: item["row_map"].__setitem__(0, 9), "permitted partition"),
            (lambda item: item["row_map"].__setitem__(0, item["row_map"][1]), "every donor row"),
            (lambda item: item.update(row_map=list(range(12))), "exact-sequence group"),
            (lambda item: item["strata"]["train"].update(a="different"), "permutation stratum"),
            (lambda item: item.update(kind="shuffle"), "whole group"),
        ):
            changed = json.loads(original)
            mutation(changed)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                experiment._control(changed, streams, splits, groups, ("rna",))
        self.assertEqual(json.dumps(control, sort_keys=True), original)
        recorded = experiment._control_record(control)
        self.assertEqual(recorded["kind"], experiment.ROW_SHUFFLE)
        self.assertEqual(recorded["row_map"], mapping.tolist())
        self.assertNotIn("a", recorded["strata"]["train"])
        old_mapping = mapping.copy()
        for name, indices in splits.items():
            old_mapping[indices] = experiment.group_permutation(
                groups, indices, strata_by_group=strata[name], seed=0)[indices]
        old_control = {**control, "kind": "shuffle", "row_map": old_mapping.tolist()}
        experiment._control(old_control, streams, splits, groups, ("rna",))
        with self.assertRaisesRegex(ValueError, "every donor row"):
            experiment._control({**old_control, "kind": experiment.ROW_SHUFFLE}, streams, splits, groups, ("rna",))
        report = experiment.inspect_shuffle(streams, groups=groups, split_positions=splits,
                                             control=control, sample_positions=(0, 6, 9))
        self.assertTrue(all(row["donor_exposure_total_variation"] == 0 for row in report["metadata"]))

    def test_bounded_shuffle_inspection_detects_unchanged_protein_and_group_reweighting(self):
        streams, _, context = fixture()
        groups = ("a", "a", "a", "b", "c", "c", "d", "e", "f", "g", "h", "i")
        streams.caches["prot"].values = [np.ones((6, 2), dtype=np.float16) for _ in groups]
        mapping, strata = np.arange(12), {}
        for split, indices in context["split_positions"].items():
            strata[split] = {groups[index]: "one" for index in indices}
            moved = experiment.group_permutation(groups, indices, strata_by_group=strata[split], seed=0)
            mapping[list(indices)] = moved[list(indices)]
        control = {"kind": "shuffle", "row_map": mapping.tolist(), "seed": 0, "strata": strata}
        report = experiment.inspect_shuffle(
            streams, groups=groups, split_positions=context["split_positions"],
            control=control, sample_positions=(0, 6, 9))
        self.assertAlmostEqual(report["metadata"][0]["donor_exposure_total_variation"], 1 / 3)
        protein = [row for row in report["sample"] if row["modality"] == "prot"]
        self.assertEqual(len(protein), 3)
        self.assertTrue(all(row["identical_features"] for row in protein))
        allowed = {0, 6, 9, *mapping[[0, 6, 9]]}
        for cache in streams.caches.values():
            self.assertEqual(len(cache.reads), 6)
            self.assertTrue(set(cache.reads) <= allowed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            experiment.inspect_shuffle(streams, groups=groups, split_positions=context["split_positions"],
                                       control=control, sample_positions=(0, 0))

    def test_group_shuffle_is_deterministic_deranged_and_partition_local(self):
        groups = ("a", "a", "b", "c", "x", "y")
        strata = {"a": "same", "b": "same", "c": "same"}
        mapping = experiment.group_permutation(groups, (0, 1, 2, 3), strata_by_group=strata, seed=7)
        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(mapping[4:].tolist(), [4, 5])
        self.assertTrue(all(groups[index] != groups[mapping[index]] for index in range(4)))
        np.testing.assert_array_equal(mapping, experiment.group_permutation(
            groups, (3, 2, 1, 0), strata_by_group=strata, seed=7))

    def test_singleton_or_missing_strata_are_rejected(self):
        for strata in ({"a": "one", "b": "two"}, {"a": "one"}):
            with self.subTest(strata=strata), self.assertRaises(ValueError):
                experiment.group_permutation(("a", "b"), (0, 1), strata_by_group=strata, seed=0)

    def test_random_statistics_use_training_tokens_only(self):
        streams, _, context = fixture()
        stats = experiment.training_random_statistics(streams, (0, 1), ("rna",))
        expected = np.concatenate(streams.caches["rna"].values[:2]).astype(np.float64)
        np.testing.assert_allclose(stats["rna"]["mean"], expected.mean(axis=0))
        np.testing.assert_allclose(stats["rna"]["scale"], expected.std(axis=0))
        self.assertEqual(streams.caches["rna"].reads, [0, 1])
        self.assertEqual(stats["rna"]["training_positions"], [0, 1])

    def test_random_features_preserve_shape_dtype_and_group_identity(self):
        streams, _, context = fixture()
        streams.caches["rna"].values[1] = streams.caches["rna"].values[0].copy()
        groups = ("same", "same") + context["groups"][2:]
        stats = experiment.training_random_statistics(streams, (0, 1, 2), ("rna",))
        control = {"kind": "random", "seed": 3, "statistics": stats}
        first = experiment.CorrectionDataset(streams, np.array([0, 1]), ("rna",), np.zeros(12), groups, control)
        reverse = experiment.CorrectionDataset(streams, np.array([1, 0]), ("rna",), np.zeros(12), groups, control)
        one, two = first[0][0]["rna"], first[1][0]["rna"]
        torch.testing.assert_close(one, two, rtol=0, atol=0)
        torch.testing.assert_close(one, reverse[1][0]["rna"], rtol=0, atol=0)
        self.assertEqual(one.shape, streams.caches["rna"].values[0].shape)
        self.assertEqual(one.dtype, torch.float16)

    def test_joint_shuffle_uses_donor_lengths_but_receiver_labels_and_offset(self):
        streams, _, context = fixture()
        streams.rows.loc[2, "length"] = 24
        streams.caches["rna"].values[2] = np.ones((24, 2), dtype=np.float16)
        streams.caches["prot"].values[2] = np.ones((8, 2), dtype=np.float16) * 2
        mapping = list(range(12))
        mapping[0] = 2
        dataset = experiment.CorrectionDataset(
            streams, np.array([0]), ("rna", "prot"), np.arange(12) + .5, context["groups"],
            {"kind": experiment.ROW_SHUFFLE, "row_map": mapping},
        )
        features, codons, label, offset = dataset[0]
        self.assertEqual(features["rna"].shape[0], 24)
        self.assertEqual(features["prot"].shape[0], 8)
        self.assertEqual(streams.caches["rna"].reads, [2])
        self.assertEqual(streams.caches["prot"].reads, [2])
        self.assertEqual(codons, 8)
        self.assertEqual(label, np.float32(streams.labels[0]))
        self.assertEqual(offset, .5)

    def test_legacy_result_identity_cannot_satisfy_row_shuffle_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            old_identity = {"protocol": {"control": {"kind": "shuffle"}}}
            path.write_text(json.dumps({"contract": experiment.CONTRACT, "identity": old_identity}))
            original = path.read_bytes()
            new_identity = {"protocol": {"control": {"kind": experiment.ROW_SHUFFLE}}}
            with self.assertRaisesRegex(ValueError, "incompatible completed correction evidence"):
                experiment._read_artifact(path, new_identity, np.zeros(1), (0,), {})
            self.assertEqual(path.read_bytes(), original)

    def test_controls_reject_cross_partition_or_cross_stratum_mapping(self):
        streams, _, context = fixture()
        splits = {key: np.asarray(values) for key, values in context["split_positions"].items()}
        mapping = np.arange(12)
        strata = {}
        for split, positions in splits.items():
            groups = {context["groups"][index]: "all" for index in positions}
            changed = experiment.group_permutation(context["groups"], positions, strata_by_group=groups, seed=5)
            mapping[positions] = changed[positions]
            strata[split] = groups
        control = {"kind": "shuffle", "row_map": mapping.tolist(), "strata": strata, "seed": 5}
        experiment._control(control, streams, splits, context["groups"], ("rna",))
        control["row_map"][0] = 9
        with self.assertRaisesRegex(ValueError, "permitted partition"):
            experiment._control(control, streams, splits, context["groups"], ("rna",))
        control["row_map"] = mapping.tolist()
        control["strata"]["train"][context["groups"][0]] = "different"
        with self.assertRaisesRegex(ValueError, "permutation stratum"):
            experiment._control(control, streams, splits, context["groups"], ("rna",))


class RidgeTests(unittest.TestCase):
    def test_excluded_nan_source_rows_do_not_block_ridge_or_exact_reuse(self):
        features = np.array([[0.], [1.], [np.nan], [2.], [3.], [4.], [5.]])
        targets = 3 * features[:, 0] + 2
        offsets = np.full(7, 2.)
        offsets[2] = np.nan
        options = dict(row_ids=tuple(range(7)),
                       split_positions={"train": (0, 1, 3), "val": (4,), "test": (5, 6)},
                       alphas=(.1, 10.), input_contract=protocol(), producer={"synthetic": True})
        with tempfile.TemporaryDirectory() as directory:
            artifact = experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
            self.assertEqual(artifact.record["fitted"]["scaler_mean"], [1.])
            self.assertEqual(artifact.prediction("test").row_ids, (5, 6))
            before = artifact.path.read_bytes()
            with patch.object(experiment.Ridge, "fit", side_effect=AssertionError("must reuse")):
                reused = experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
            np.testing.assert_array_equal(reused.prediction("test").values, artifact.prediction("test").values)
            self.assertEqual(artifact.path.read_bytes(), before)
            self.assertTrue(np.isnan(features[2, 0]) and np.isnan(targets[2]) and np.isnan(offsets[2]))
            invalid_offsets = offsets.copy()
            invalid_offsets[0] = np.nan
            with self.assertRaises(ValueError):
                experiment.run_ridge(features, targets, invalid_offsets, output_dir=directory, **options)
            self.assertEqual(artifact.path.read_bytes(), before)

    def test_malformed_ridge_evidence_is_refused_without_refitting_or_overwriting(self):
        features = np.arange(7., dtype=float).reshape(-1, 1)
        targets, offsets = 3 * features[:, 0] + 2, np.full(7, 2.)
        options = dict(row_ids=tuple(range(7)),
                       split_positions={"train": (0, 1, 2), "val": (3, 4), "test": (5, 6)},
                       alphas=(.1, 10.), input_contract=protocol(), producer={"synthetic": True})
        with tempfile.TemporaryDirectory() as directory:
            artifact = experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
            original = artifact.path.read_text(encoding="utf-8")
            mutations = [
                (("fitted",), {}),
                (("fitted", "coefficient"), [0., 0.]),
                (("fitted", "coefficient"), [artifact.record["fitted"]["coefficient"][0] + 1]),
                (("fitted", "scaler_scale"), [0.]),
                (("selected_alpha",), 123.),
                (("validation_mse",), artifact.record["validation_mse"] + 1),
                (("candidates",), []),
                (("predictions", "val", "values"), (artifact.prediction("val").values + 1).tolist()),
                (("predictions", "test", "values"), (artifact.prediction("test").values + 1).tolist()),
            ]
            for fields, replacement in mutations:
                with self.subTest(fields=fields, replacement=replacement):
                    damaged = json.loads(original)
                    destination = damaged
                    for field in fields[:-1]:
                        destination = destination[field]
                    destination[fields[-1]] = replacement
                    artifact.path.write_text(json.dumps(damaged), encoding="utf-8")
                    before = artifact.path.read_bytes()
                    with patch.object(experiment.Ridge, "fit", side_effect=AssertionError("must refuse")):
                        with self.assertRaises(ValueError):
                            experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
                    self.assertEqual(artifact.path.read_bytes(), before)

    def test_ridge_is_train_only_and_reuses_exact_completed_evidence(self):
        features = np.array([[0.], [1.], [2.], [100.], [101.], [-50.], [-51.]])
        targets = 3 * features[:, 0] + 2
        offsets = np.full(7, 2.)
        splits = {"train": (0, 1, 2), "val": (3, 4), "test": (5, 6)}
        options = dict(row_ids=tuple(range(7)), split_positions=splits, alphas=(.1, 10.),
                       input_contract=protocol(), producer={"synthetic": True})
        with tempfile.TemporaryDirectory() as directory:
            artifact = experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
            self.assertEqual(artifact.record["fitted"]["scaler_mean"], [1.])
            self.assertEqual(artifact.record["selected_alpha"], .1)
            before = artifact.path.read_bytes()
            with patch.object(experiment.Ridge, "fit", side_effect=AssertionError("must reuse")):
                reused = experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
            np.testing.assert_array_equal(artifact.prediction("test").values, reused.prediction("test").values)
            with self.assertRaises(ValueError):
                experiment.run_ridge(features, targets, offsets + 1, output_dir=directory, **options)
            self.assertEqual(artifact.path.read_bytes(), before)

    def test_ridge_ties_follow_declared_alpha_order_and_require_anchor_bindings(self):
        features = np.zeros((6, 2))
        targets, offsets = np.arange(6.), np.arange(6.)
        options = dict(row_ids=tuple(range(6)), split_positions={"train": (0, 1), "val": (2, 3), "test": (4, 5)},
                       alphas=(10., .1), input_contract=protocol(), producer={})
        with tempfile.TemporaryDirectory() as directory:
            artifact = experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)
            self.assertEqual(artifact.record["selected_alpha"], 10.)
            np.testing.assert_array_equal(artifact.prediction("test").values, offsets[[4, 5]])
            options["input_contract"] = {}
            with self.assertRaisesRegex(ValueError, "SHA-256 bindings"):
                experiment.run_ridge(features, targets, offsets, output_dir=directory, **options)


class NewtonAnchorTests(unittest.TestCase):
    def test_zero_offset_adapter_matches_blf_native_and_grid_singles(self):
        streams, _, context = fixture()
        for layout in ("native", "codon"):
            for modality in ("dna", "rna", "prot"):
                with self.subTest(layout=layout, modality=modality):
                    variant = f"{'unimodal' if layout == 'native' else 'unimodal-grid'}:{modality}"
                    original = blf.build(streams, variant, channels=2, tconv="equation").eval()
                    adapter = experiment.CorrectionTextCNN(
                        streams.widths, streams.resolutions, (modality,), layout, config()).eval()
                    adapter.load_state_dict(original.state_dict(), strict=True)
                    batch = next(iter(blf.loader(streams, "val", batch_size=3, modalities=(modality,))))
                    corrected_batch = experiment.CorrectionBatch(
                        batch.streams, batch.tokens, batch.codons, batch.labels, torch.zeros(3))
                    with torch.no_grad():
                        np.testing.assert_allclose(original(batch)[0], adapter(corrected_batch)[0], atol=0, rtol=0)

    def source_fixture(self, root):
        streams, caches, context = fixture()
        variant = "unimodal:rna"
        recipe = config()
        model = blf.build(streams, variant, channels=recipe.channels, tconv=recipe.tconv).eval()
        truth, prediction, _ = train.predict(model, blf.loader(streams, "test", modalities=("rna",)),
                                             "cpu", "regression")
        identity = {"config": asdict(recipe), "inputs": {"splits": {
            name: run_identity.array_record(rows, np.int64) for name, rows in context["split_positions"].items()}}}
        weights = root / "original.weights.pt"
        torch.save({"format": "fusion-state-dict-v1", "identity": identity, "config": asdict(recipe),
                    "variant": variant, "benchmark": streams.key, "best_epoch": 0,
                    "state_dict": model.state_dict()}, weights)
        original = {"identity": identity, "config": asdict(recipe), "best_epoch": 0,
                    "test_truth": truth.tolist(), "test_prediction": prediction.tolist(),
                    "producer": {"host": "synthetic-newton"}, "model_artifact": {
                        "file": weights.name, "size_bytes": weights.stat().st_size,
                        "sha256": experiment._sha(weights)}}
        source = root / "original.json"
        source.write_text(json.dumps(original), encoding="utf8")
        reference = SimpleNamespace(extra_fields={"reference_source": {
            "path": str(source), "sha256": experiment._sha(source), "current_inputs_verified": True}})
        options = {**context, "reference_root": root, "bundle": "original", "modality": "rna",
                   "layout": "native", "config": recipe, "cache_records": caches,
                   "output_dir": root / "new-a2", "input_contract": {"candidate": "rnafm_native"},
                   "device": "cpu", "producer": {"host": "synthetic-inference"}}
        return streams, reference, source, weights, original, options

    def test_reuse_infers_validation_without_training_or_altering_original_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, reference, source, weights, original, options = self.source_fixture(Path(directory))
            before = source.read_bytes(), weights.read_bytes()
            with patch("shared_code.fusion.study_references.read_reference", return_value=reference), \
                 patch.object(train, "run", side_effect=AssertionError("must not train")):
                artifact = experiment.reuse_newton_anchor(streams, **options)
                with patch.object(train, "predict", side_effect=AssertionError("completed inference must reuse")):
                    repeated = experiment.reuse_newton_anchor(streams, **options)
            self.assertEqual(artifact.sha256, repeated.sha256)
            self.assertEqual(before, (source.read_bytes(), weights.read_bytes()))
            self.assertEqual(artifact.record["identity"], original["identity"])
            self.assertEqual(artifact.record["producer"], original["producer"])
            self.assertEqual(artifact.record["prediction_producer"], options["producer"])
            self.assertEqual(artifact.record["predictions"]["test"]["values"], original["test_prediction"])
            self.assertEqual(artifact.prediction("val").row_ids, (6, 7, 8))
            self.assertFalse((artifact.path.parent / "fit.json").exists())
            oof = evidence.Predictions("regression", tuple(range(6)), np.zeros(6))
            offsets = experiment.correction_offsets(oof, artifact, row_ids=range(12),
                                                     split_positions=options["split_positions"])
            np.testing.assert_array_equal(offsets[9:], original["test_prediction"])

    def test_missing_weights_and_changed_source_hold_instead_of_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            streams, reference, source, weights, _, options = self.source_fixture(Path(directory))
            with patch("shared_code.fusion.study_references.read_reference", return_value=reference), \
                 patch.object(train, "run", side_effect=AssertionError("must not train")):
                weights.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "best weights"):
                    experiment.reuse_newton_anchor(streams, **options)
                source.write_text("{}", encoding="utf8")
                with self.assertRaisesRegex(ValueError, "verified source"):
                    experiment.reuse_newton_anchor(streams, **options)
            self.assertFalse(Path(options["output_dir"]).exists())

    def test_grouped_reference_requires_saved_panel_and_exact_current_identity(self):
        from shared_code.fusion import grouped_stability
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            streams, caches, context = fixture()
            streams.key = "mrna_stability"
            path = root / "fits" / "rnafm_native" / "seed0.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"identity": {"invented": "identity"}}), encoding="utf8")
            options = {**context, "reference_root": root, "candidate": "rnafm_native",
                       "config": config(), "cache_records": caches, "output_dir": root / "a2",
                       "input_contract": {}, "device": "cpu", "producer": {"synthetic": True}}
            reference = SimpleNamespace(extra_fields={})
            with patch.object(grouped_stability, "read_split", return_value={"split": "verified"}), \
                 patch.object(grouped_stability, "validate_result") as validate, \
                 patch.object(grouped_stability, "make_identity", return_value={"invented": "identity"}) as identity, \
                 patch.object(train.Result, "load", return_value=reference), \
                 patch.object(experiment, "_reference_inference", return_value="reused") as inference, \
                 patch.object(train, "run", side_effect=AssertionError("must not fit")):
                self.assertEqual(experiment.reuse_grouped_anchor(streams, **options), "reused")
                validate.assert_called_once()
                self.assertEqual(set(identity.call_args.args[4]), set(streams.encoders.values()))
                self.assertEqual(inference.call_args.kwargs["reference_location"], "grouped_stability")
                identity.return_value = {"invented": "changed"}
                with self.assertRaisesRegex(ValueError, "current.*identity"):
                    experiment.reuse_grouped_anchor(streams, **options)
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    experiment.reuse_grouped_anchor(streams, **options)
                self.assertEqual(inference.call_count, 1)


class NeuralAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def options(self, streams, records, context):
        return dict(**context, modalities=("rna",), layout="native", offsets=np.zeros(12),
                    config=config(), cache_records=records, input_contract={"synthetic": True},
                    device="cpu", producer={"synthetic": True}, role="anchor", control={"kind": "real"}, log=None)

    def test_summary_recovery_preserves_fit_producer_and_records_prediction_producer(self):
        streams, records, context = fixture()
        options = {**self.options(streams, records, context), "producer": {"host": "original-fit"}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment, "_publish", side_effect=RuntimeError("summary interrupted")):
                with self.assertRaisesRegex(RuntimeError, "summary interrupted"):
                    experiment.run_neural(streams, output_dir=directory, **options)
            self.assertFalse((Path(directory) / "predictions.json").exists())
            preserved = {name: (Path(directory) / name).read_bytes() for name in ("fit.json", "fit.weights.pt")}
            recovery_producer = {"host": "summary-recovery"}
            with patch.object(train, "fit", side_effect=AssertionError("completed fit must be reused")):
                recovered = experiment.run_neural(
                    streams, output_dir=directory, **{**options, "producer": recovery_producer})
            self.assertEqual(recovered.record["producer"], options["producer"])
            self.assertEqual(recovered.record["prediction_producer"], recovery_producer)
            for name, before in preserved.items():
                self.assertEqual((Path(directory) / name).read_bytes(), before)
            saved_fit = json.loads(preserved["fit.json"])
            np.testing.assert_array_equal(recovered.prediction("test").values, saved_fit["test_prediction"])
            self.assertEqual(recovered.prediction("val").row_ids, (6, 7, 8))

    def test_native_and_codon_predictions_ignore_longer_batch_neighbors(self):
        streams, _, context = fixture()
        rng = np.random.default_rng(15)
        # Donor lengths differ from the receiver's original length, including a short
        # protein stream and appendix DNA upsampling that needs per-row zero padding.
        for donor, length in ((1, 12), (2, 30)):
            streams.rows.loc[donor, "length"] = length
            for name, count in (("dna", length // 6), ("rna", length), ("prot", length // 3)):
                streams.caches[name].values[donor] = rng.normal(size=(count, 2)).astype(np.float16)
        mapping = list(range(12))
        mapping[0], mapping[1] = 1, 2
        controls = {"kind": "shuffle", "row_map": mapping}
        cases = [("native", ("dna",), "equation"), ("native", ("rna",), "equation"),
                 ("native", ("prot",), "equation"),
                 ("codon", ("dna", "rna", "prot"), "equation"),
                 ("codon", ("dna", "rna", "prot"), "appendix")]
        for layout, modalities, tconv in cases:
            with self.subTest(layout=layout, modalities=modalities, tconv=tconv):
                dataset = experiment.CorrectionDataset(
                    streams, np.array([0, 1]), modalities, np.arange(12) * .1, context["groups"], controls)
                alone = experiment._collate([dataset[0]])
                together = experiment._collate([dataset[0], dataset[1]])
                torch.manual_seed(7)
                model = experiment.CorrectionTextCNN(
                    streams.widths, streams.resolutions, modalities, layout, replace(config(), tconv=tconv)).eval()
                with torch.no_grad():
                    expected, _ = model(alone)
                    actual, _ = model(together)
                torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)

    def test_real_adapter_trains_preserves_reuses_and_rejects_changed_inputs(self):
        streams, records, context = fixture()
        options = self.options(streams, records, context)
        with tempfile.TemporaryDirectory() as directory:
            artifact = experiment.run_neural(streams, output_dir=directory, **options)
            self.assertEqual(artifact.prediction("test").row_ids, (9, 10, 11))
            self.assertTrue((Path(directory) / "fit.weights.pt").is_file())
            original = {path.name: path.read_bytes() for path in Path(directory).iterdir()}
            with patch.object(train, "run", side_effect=AssertionError("completed fit must be reused")):
                reused = experiment.run_neural(streams, output_dir=directory, **options)
            np.testing.assert_array_equal(reused.prediction("test").values, artifact.prediction("test").values)
            with self.assertRaises(ValueError):
                experiment.run_neural(streams, output_dir=directory, **{**options, "config": replace(config(), seed=1)})
            self.assertEqual(original, {path.name: path.read_bytes() for path in Path(directory).iterdir()})
            (Path(directory) / "fit.weights.pt").write_bytes(b"damaged fixture")
            with self.assertRaises(ValueError):
                experiment.run_neural(streams, output_dir=directory, **options)

    def test_correction_mse_uses_fixed_offsets_and_joint_codon_inputs(self):
        streams, records, context = fixture()
        options = {**self.options(streams, records, context), "role": "correction",
                   "modalities": ("rna", "prot"), "layout": "codon", "offsets": np.arange(12) * .1,
                   "input_contract": protocol()}
        dataset = experiment.CorrectionDataset(streams, np.array([0, 1]), ("rna", "prot"),
                                               options["offsets"], context["groups"], {"kind": "real"})
        batch = experiment._collate([dataset[0], dataset[1]])
        model = experiment.CorrectionTextCNN(streams.widths, streams.resolutions, ("rna", "prot"), "codon", config())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        predicted, aux = model(batch)
        torch.testing.assert_close(predicted[:, 0], batch.offsets)
        loss_original = torch.nn.functional.mse_loss(predicted[:, 0], batch.labels)
        loss_residual = torch.nn.functional.mse_loss(predicted[:, 0] - batch.offsets, batch.labels - batch.offsets)
        torch.testing.assert_close(loss_original, loss_residual)
        with tempfile.TemporaryDirectory() as directory:
            artifact = experiment.run_neural(streams, output_dir=directory, **options)
            self.assertEqual(artifact.record["identity"]["inputs"]["role"], "correction")
            self.assertEqual(artifact.record["identity"]["model"]["layout"], "codon")
            self.assertEqual(artifact.prediction("val").row_ids, (6, 7, 8))

    def test_corrupted_cache_identity_or_nonzero_anchor_offset_stops_before_training(self):
        streams, records, context = fixture()
        options = self.options(streams, records, context)
        with patch.object(train, "run", side_effect=AssertionError("must refuse before fitting")):
            with self.assertRaises(ValueError):
                experiment.run_neural(streams, output_dir="unused", **{**options, "offsets": np.ones(12)})
            records["rna"]["input_sha256"] = "bad"
            with self.assertRaises(ValueError):
                experiment.run_neural(streams, output_dir="unused", **options)

    def test_same_final_anchor_supplies_baseline_and_correction_offsets(self):
        streams, records, context = fixture()
        options = self.options(streams, records, context)
        with tempfile.TemporaryDirectory() as directory:
            final = experiment.run_neural(streams, output_dir=directory, **options)
            oof = evidence.Predictions("regression", tuple(range(6)), np.arange(6) * .2)
            offsets = experiment.correction_offsets(oof, final, row_ids=context["row_ids"],
                                                     split_positions=context["split_positions"])
            np.testing.assert_allclose(offsets[:6], oof.values)
            np.testing.assert_allclose(offsets[6:9], final.prediction("val").values)
            np.testing.assert_allclose(offsets[9:], final.prediction("test").values)
            changed = {"train": tuple(range(7)), "val": (7, 8), "test": (9, 10, 11)}
            expanded_oof = evidence.Predictions("regression", tuple(range(7)), np.zeros(7))
            with self.assertRaisesRegex(ValueError, r"train\+val refit"):
                experiment.correction_offsets(expanded_oof, final, row_ids=context["row_ids"], split_positions=changed)


if __name__ == "__main__":
    unittest.main()
