"""Tiny invented token tensors exercise fitting/reuse; no benchmark or encoder runs."""

from copy import deepcopy
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
from shared_code.encoders import ENCODERS
from shared_code.fusion import blf, controlled_experiment as experiment, run_identity, train
from shared_code.fusion import token_embeddings as tokens


class SyntheticCache:
    def __init__(self, count, length, width, rng):
        self.width = width
        self.lengths = np.full(count, length)
        self.values = rng.normal(size=(count, length, width)).astype(np.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def fixture():
    count = 12
    rows = pd.DataFrame({"sequence": ["ATGAAA" * 4] * count, "length": [24] * count,
                         "Value": np.linspace(-1, 1, count),
                         "Split": ["train"] * 8 + ["val"] * 2 + ["test"] * 2})
    rng = np.random.default_rng(42)
    caches = {"dna": SyntheticCache(count, 4, 3, rng),
              "rna": SyntheticCache(count, 24, 4, rng),
              "prot": SyntheticCache(count, 8, 2, rng)}
    streams = blf.BenchmarkStreams("synthetic", rows, caches, "regression")
    records = {m: {"encoder": key, "n_sequences": count, "width": caches[m].width,
                   "input_sha256": tokens.input_fingerprint(tokens.inputs_for_rows(
                       ENCODERS[key], rows["sequence"].tolist())),
                   "shards": [{"name": "synthetic.npz", "sha256": "1" * 64}]}
               for m, key in streams.encoders.items()}
    config = train.TrainConfig(batch_size=4, max_epochs=1, channels=100, tconv="equation", seed=0)
    return streams, records, config


def settings(variant):
    found = {"projection_width": 4, "projection_activation": "tanh", "projection_norm": "none"}
    if variant == "anchor_only":
        return {**found, "anchor": "rna"}
    if variant == "position_control":
        return {**found, "control_bottleneck": 2}
    if variant == "position_residual_control":
        return {**found, "control_bottleneck": 2, "control_scale": .5,
                "control_initialization": "zero_output_random_input"}
    found["scorer_width"] = 3
    if variant in experiment.SEQUENCE_VARIANTS:
        found.update(shared_scorer=False, temperature_range=(
            [.02, 20.] if experiment.SEQUENCE_VARIANTS[variant][1] == "softmax" else None))
    if variant in experiment.ATTENTION_VARIANTS:
        found.update(anchor="rna", n_heads=2, dropout=0., update_norm="none")
    return found


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ControlledExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.streams, self.records, self.config = fixture()
        self.producer = {"scope": "synthetic test", "device": "cpu"}
        # Isolate the native hard-link primitive: the Windows app sandbox refuses
        # it in its temporary directory. Preserve exclusive creation semantics on
        # these owned fixture files; this is not a native publication integration test.
        def exclusive_publication(source, destination):
            self.assertEqual(Path(source).parent, self.output)
            self.assertEqual(Path(destination).parent, self.output)
            with Path(destination).open("xb") as handle:
                handle.write(Path(source).read_bytes())
        publication = patch.object(experiment.os, "link", side_effect=exclusive_publication)
        publication.start()
        self.addCleanup(publication.stop)

    def fit(self, variant="position_sigmoid"):
        return experiment.run_fit(self.streams, variant, self.config, settings(variant),
                                  self.records, self.output, "cpu", self.producer, "tiny")

    def collect(self, variant="position_sigmoid"):
        return experiment.collect_diagnostics(self.streams, variant, self.config, settings(variant),
            self.records, result_path=self.output / "tiny.json", output_stem=self.output / "diagnostics",
            device="cpu", producer=self.producer)

    def test_group_paired_intervals_validate_actual_rows_and_average_complete_seeds(self):
        rows = pd.DataFrame({"sequence": ["AAA", "CCC", "GGG", "GGG", "TTT", "TTT"],
                             "length": [3] * 6, "Value": [0, 1, 0, 1, 0, 1],
                             "Split": ["train", "val", "test", "test", "test", "test"]})
        labels = rows.Value.to_numpy(dtype=np.int64)
        splits = {name: np.flatnonzero(rows.Split == name) for name in ("train", "val", "test")}
        inputs = run_identity._inputs(rows, ("sequence", "length"), labels, {}, {}, splits,
                                      ("Value",), "classification")
        def result(prediction, seed):
            return SimpleNamespace(config={"seed": seed}, test_truth=labels[2:].tolist(),
                test_prediction=prediction,
                test_score=float(np.mean(labels[2:] == prediction)),
                extra_fields={"identity": {"inputs": inputs}})
        records = {"candidate": {seed: result([0, 1, 0, 1], seed) for seed in (0, 1, 2)},
                   "baseline": {seed: result([1, 0, 1, 0], seed) for seed in (0, 1, 2)}}
        def analyze():
            return experiment.paired_intervals(records, rows=rows, split_column="Split",
                task="classification", groups=rows.sequence.to_numpy(), seeds=(0, 1, 2),
                comparisons=[("candidate minus baseline", "candidate", "baseline")],
                draws=20, bootstrap_seed=0)
        measured = analyze()[0]
        self.assertEqual((measured["difference"], measured["ci_low"], measured["ci_high"]), (1., 1., 1.))
        self.assertEqual(measured["groups"], 2)
        removed = records["candidate"].pop(2)
        with self.assertRaisesRegex(ValueError, "all declared seeds"):
            analyze()
        records["candidate"][2] = removed
        rows.loc[2, "sequence"] = "AAC"
        with self.assertRaisesRegex(ValueError, "cohort identity"):
            analyze()

    def test_all_declared_variants_construct_and_preserve_prediction_shape(self):
        batch = next(iter(blf.loader(self.streams, "test", batch_size=2)))
        for variant in experiment.VARIANTS:
            with self.subTest(variant=variant):
                model = experiment.build_model(self.streams, variant, self.config, settings=settings(variant))
                predicted, aux = model(batch)
                self.assertEqual(predicted.shape, (2, 1))
                self.assertTrue(torch.isfinite(predicted).all())
                self.assertEqual("alpha" in aux, variant == "sequence_softmax")

    def test_legacy_mil_wrapper_matches_original_blf_state_and_predictions(self):
        controlled = experiment.build_model(self.streams, "sequence_softmax", self.config,
                                            settings=settings("sequence_softmax"))
        original = blf.BioLangFusion(self.streams.widths, "mil", 1, channels=100, out_width=4,
                                    attention_width=3, tconv="equation", resolutions=self.streams.resolutions)
        controlled.load_state_dict(original.state_dict(), strict=True)
        batch = next(iter(blf.loader(self.streams, "test", batch_size=2)))
        original.eval()
        controlled.eval()
        expected, _ = original(batch)
        actual, _ = controlled(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_identity_records_parameters_and_preserves_rng(self):
        before = torch.get_rng_state().clone()
        identity = experiment.make_identity(self.streams, "position_sigmoid", self.config,
                                            settings("position_sigmoid"), self.records)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(identity["loader"]["kind"], "custom")
        counts = identity["model"]["parameter_counts"]
        self.assertEqual(counts["total"], counts["grid"] + counts["fusion"] + counts["prediction"])
        self.assertEqual(identity["inputs"]["splits"]["test"], run_identity.array_record(np.array([10, 11]), np.int64))

    def test_projected_anchor_control_materializes_only_anchor_stream(self):
        variant = "anchor_only"
        identity = experiment.make_identity(self.streams, variant, self.config, settings(variant), self.records)
        self.assertEqual(identity["loader"]["modalities"], ["rna"])
        model = experiment.build_model(self.streams, variant, self.config, settings=settings(variant))
        loaders = experiment._loaders(self.streams, self.config, identity)
        batch = next(iter(loaders["test"]))
        self.assertEqual(tuple(batch.streams), ("rna",))
        predicted, aux = model(batch)
        self.assertEqual(predicted.shape, (2, 1))
        self.assertEqual(aux, {})

    def test_tiny_fit_is_reused_without_overwriting_or_training(self):
        first = self.fit()
        result_path, weight_path = self.output / "tiny.json", self.output / "tiny.weights.pt"
        originals = (sha(result_path), sha(weight_path))
        with patch.object(train, "run", side_effect=AssertionError("No second fit")):
            second = self.fit()
        self.assertEqual(first.test_prediction, second.test_prediction)
        self.assertEqual(originals, (sha(result_path), sha(weight_path)))
        self.assertEqual(first.best_epoch, 0)

    def test_incompatible_completed_fit_refuses_replacement(self):
        self.fit()
        original = sha(self.output / "tiny.json")
        self.streams.rows.loc[0, "Value"] += .5
        with patch.object(train, "run", side_effect=AssertionError("No replacement fit")):
            with self.assertRaisesRegex(ValueError, "different computation identity"):
                self.fit()
        self.assertEqual(original, sha(self.output / "tiny.json"))

    def panel_fixture(self, variant="position_sigmoid"):
        identity = experiment.make_identity(self.streams, variant, self.config,
                                            settings(variant), self.records)
        truth = self.streams.labels[self.streams.split_rows("test")].tolist()
        result = train.Result(variant, self.streams.key, asdict(self.config), 1, 0, 1., 1., 0.,
                              test_truth=truth, test_prediction=truth,
                              extra_fields={"identity": identity})
        path = self.output / "saved-panel.json"
        result.save(path)
        request = dict(benchmark=self.streams.key, variant=variant, config=self.config,
                       encoders=dict(self.streams.encoders), settings=settings(variant))
        return path, result, request

    def test_residual_control_has_separate_identity_and_refuses_legacy_relabeling(self):
        path, result, request = self.panel_fixture("position_control")
        original = sha(path)
        changed = {**request, "variant": "position_residual_control",
                   "settings": settings("position_residual_control")}
        with self.assertRaisesRegex(ValueError, "variant"):
            experiment.read_panel_fit(path, **changed)
        self.assertEqual(sha(path), original)
        experiment.read_panel_fit(path, **request)
        path, result, request = self.panel_fixture("position_residual_control")
        formula = result.extra_fields["identity"]["model"]["control_formula"]
        self.assertEqual(formula["scale"], .5)
        self.assertEqual(formula["update_rank"], 2)
        self.assertEqual(formula["update_initialization"], "zero_output_random_input")
        experiment.read_panel_fit(path, **request)
        result.extra_fields["identity"]["model"].pop("control_formula")
        result.save(path)
        with self.assertRaisesRegex(ValueError, "architecture"):
            experiment.read_panel_fit(path, **request)

    def test_saved_panel_refuses_wrong_bundle_without_cache_access(self):
        path, result, request = self.panel_fixture()
        with patch.object(experiment, "build_model", side_effect=AssertionError("Saved reads cannot build a model")):
            loaded = experiment.read_panel_fit(path, **request)
            self.assertEqual(loaded.test_prediction, result.test_prediction)
            request["encoders"]["rna"] = "mrnafm"
            with self.assertRaisesRegex(ValueError, "encoder bundle"):
                experiment.read_panel_fit(path, **request)

    def test_saved_panel_refuses_changed_architecture_loader_and_source_contract(self):
        path, result, request = self.panel_fixture()
        original = deepcopy(result.extra_fields["identity"])
        changes = (
            lambda identity: identity["model"]["settings"].update(scorer_width=99),
            lambda identity: identity["model"].update(modalities=["rna"]),
            lambda identity: identity["model"].pop("gate_formula"),
            lambda identity: identity["loader"].update(seed=2),
            lambda identity: identity["model"]["implementation_sha256"].pop("heads.py"),
        )
        for change in changes:
            result.extra_fields["identity"] = deepcopy(original)
            change(result.extra_fields["identity"])
            result.save(path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment.read_panel_fit(path, **request)
        # A retained, well-formed producer fingerprint may differ from today's reader.
        result.extra_fields["identity"] = deepcopy(original)
        result.extra_fields["identity"]["model"]["implementation_sha256"] = {
            name: "a" * 64 for name in experiment.IMPLEMENTATION_FILES
        }
        result.save(path)
        experiment.read_panel_fit(path, **request)

    def test_settings_and_entropy_are_explicit(self):
        for variant in experiment.VARIANTS:
            if variant == "sequence_softmax":
                continue
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "entropy"):
                experiment.validate_settings(variant, replace(self.config, entropy_lambda=.01), settings(variant))
        altered = settings("position_sigmoid")
        altered["hidden_choice"] = 99
        with self.assertRaisesRegex(ValueError, "unused"):
            experiment.validate_settings("position_sigmoid", self.config, altered)
        with self.assertRaisesRegex(ValueError, "equation"):
            experiment.validate_settings("position_sigmoid", replace(self.config, tconv="appendix"), settings("position_sigmoid"))

    def test_actual_loader_row_order_is_verified_before_fit(self):
        original = blf.loader
        def reversed_test(*args, **kwargs):
            loader = original(*args, **kwargs)
            if args[1] == "test":
                loader.dataset.indices = loader.dataset.indices[::-1].copy()
                loader.dataset.labels = loader.dataset.labels[::-1].copy()
            return loader
        with patch.object(blf, "loader", side_effect=reversed_test), patch.object(train, "run") as run:
            with self.assertRaisesRegex(ValueError, "row order"):
                self.fit()
            run.assert_not_called()

    def test_diagnostics_use_best_weights_without_fitting_and_preserve_source_files(self):
        result = self.fit()
        originals = {path.name: sha(path) for path in self.output.iterdir()}
        with patch.object(train, "run", side_effect=AssertionError("Diagnostics cannot fit")):
            record, arrays, reused = self.collect()
        self.assertFalse(reused)
        self.assertEqual(arrays["row_ids"].tolist(), ["synthetic:10", "synthetic:11"])
        self.assertEqual(arrays["gates"].shape, (16, 3))
        self.assertEqual(arrays["gates_lengths"].tolist(), [8, 8])
        self.assertEqual(arrays["gates_positions"].tolist(), list(range(8)) * 2)
        np.testing.assert_allclose(arrays["prediction_outputs"][:, 0], result.test_prediction, rtol=0, atol=0)
        for name, fingerprint in originals.items():
            self.assertEqual(fingerprint, sha(self.output / name))
        with patch.object(torch, "load", side_effect=AssertionError("Reuse should not infer")):
            reread, repeated, reused = self.collect()
        self.assertTrue(reused)
        self.assertEqual(reread, record)
        np.testing.assert_array_equal(repeated["gates"], arrays["gates"])

    def test_sequence_attention_records_raw_and_effective_gate_resolutions(self):
        variant = "attention_sequence_sigmoid"
        self.fit(variant)
        record, arrays, _ = self.collect(variant)
        self.assertEqual(record["gate_modalities"], ["dna", "prot"])
        self.assertEqual(arrays["gates"].shape, (2, 2))
        self.assertNotIn("gates_positions", arrays)
        self.assertEqual(arrays["update_weights"].shape, (16, 2))
        self.assertEqual(arrays["update_weights_lengths"].tolist(), [8, 8])
        self.assertEqual(record["norm_columns"]["anchor_norm"], ["rna"])
        self.assertEqual(record["norm_columns"]["update_norms"], ["dna", "prot"])
        np.testing.assert_allclose(arrays["update_norms"],
                                   arrays["raw_update_norms"] * arrays["update_weights"], rtol=1e-6)

    def test_joint_gate_identity_and_norms_are_explicit(self):
        self.fit()
        record, arrays, _ = self.collect()
        formula = record["identity"]["fit_identity"]["model"]["gate_formula"]
        self.assertEqual(formula, {"input": "all_projected_modalities", "scorer": "affine_sigmoid",
                                   "scorer_width_active": False})
        self.assertEqual(record["schema"], experiment.DIAGNOSTIC_SCHEMA)
        self.assertEqual(arrays["norm_positions"].tolist(), list(range(8)) * 2)
        np.testing.assert_allclose(arrays["contribution_norms"],
                                   arrays["feature_norms"] * arrays["gates"], rtol=1e-6)
        self.assertEqual(arrays["fused_norm"].shape, (16, 1))

    def test_legacy_diagnostics_remain_readable_but_cannot_be_overwritten_as_v2(self):
        self.fit()
        record, arrays, _ = self.collect()
        record["schema"] = "controlled-fusion-diagnostics-v1"
        record.pop("norm_columns")
        record.pop("temperature")
        record["identity"].pop("diagnostic_schema")
        record["identity"].pop("observer_sha256")
        for field in (*experiment.NORM_FIELDS, "norm_lengths", "norm_positions"):
            arrays.pop(field, None)
        array_path, record_path = self.output / "diagnostics.npz", self.output / "diagnostics.json"
        np.savez_compressed(array_path, **arrays)
        record["arrays"].update(sha256=sha(array_path), bytes=array_path.stat().st_size)
        record_path.write_text(json.dumps(record))
        originals = sha(array_path), sha(record_path)
        with patch.object(torch, "load", side_effect=AssertionError("Reading cannot infer")):
            legacy, saved = experiment.read_diagnostics(self.output / "diagnostics")
            summary = experiment.summarize_diagnostics(legacy, saved)
            self.assertEqual(summary["norm_summary"], [])
            self.assertIn("legacy", summary["norm_status"])
            with self.assertRaisesRegex(ValueError, "different evidence"):
                self.collect()
        self.assertEqual(originals, (sha(array_path), sha(record_path)))

    def test_norm_validation_refuses_negative_values_and_reordered_positions(self):
        self.fit()
        record, arrays, _ = self.collect()
        array_path, record_path = self.output / "diagnostics.npz", self.output / "diagnostics.json"
        for field, message in (("fused_norm", "nonnegative"), ("norm_positions", "Norm positions")):
            changed = {name: values.copy() for name, values in arrays.items()}
            changed[field].flat[0] = -1
            np.savez_compressed(array_path, **changed)
            record["arrays"].update(sha256=sha(array_path), bytes=array_path.stat().st_size)
            record_path.write_text(json.dumps(record))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                experiment.read_diagnostics(self.output / "diagnostics")

    def test_saved_summary_preserves_local_variation_and_undefined_norm_ratio(self):
        record = {"gate_modalities": ["rna"], "norm_columns": {
            "anchor_norm": ["dna"], "update_norms": ["rna"], "fused_norm": ["fused"]}}
        arrays = {"gates": np.array([[0.], [1.], [.5], [.5]]),
                  "gates_lengths": np.array([2, 2]), "gates_positions": np.array([0, 1, 0, 1]),
                  "norm_lengths": np.array([2, 2]), "anchor_norm": np.array([[0.], [0.], [2.], [2.]]),
                  "update_norms": np.ones((4, 1)), "fused_norm": np.ones((4, 1))}
        found = experiment.summarize_diagnostics(record, arrays)
        self.assertEqual(found["summary"][0]["median"], .5)
        self.assertEqual(found["positional_variation"][0]["within_row_sd"], .25)
        self.assertEqual(found["positional_variation"][0]["within_row_range"], .5)
        self.assertEqual(found["update_anchor_ratios"][0]["undefined_rows"], 1)
        self.assertEqual(found["update_anchor_ratios"][0]["median"], .5)
        self.assertEqual(found["weight_mass"][0]["median"], .5)
        arrays["anchor_norm"][:] = 0
        self.assertIsNone(experiment.summarize_diagnostics(record, arrays)["update_anchor_ratios"][0]["median"])

    def test_reference_mil_diagnostics_preserve_its_original_identity(self):
        identity = run_identity.blf_identity(rows=self.streams.rows, benchmark="synthetic",
            variant="mil", config=self.config, encoders=self.streams.encoders, caches=self.records)
        train.run(self.streams, "mil", self.config, "cpu", self.output, log=None,
                  tag="reference", run_identity=identity, require_predictions=True, producer=self.producer)
        path = self.output / "reference.json"
        original = sha(path)
        result = train.Result.load(path)
        result.extra_fields["reference_source"] = {
            "path": str(path.resolve()), "sha256": original, "current_inputs_verified": True}
        requested = {**settings("sequence_softmax"), "projection_width": 600, "scorer_width": 100}
        with patch.object(train, "run", side_effect=AssertionError("Reference inference cannot fit")):
            record, arrays, _ = experiment.collect_diagnostics(
                self.streams, "sequence_softmax", self.config, requested, self.records,
                result_path=path, output_stem=self.output / "reference-diagnostics", device="cpu",
                producer=self.producer, reference_result=result)
        self.assertEqual(record["identity"]["fit_identity"], identity)
        self.assertEqual(record["identity"]["fit_identity"]["variant"], "mil")
        self.assertEqual(arrays["gates"].shape, (2, 3))
        self.assertEqual(sha(path), original)

    def test_missing_weights_or_partial_diagnostics_cannot_trigger_fit(self):
        self.fit()
        (self.output / "diagnostics.npz").write_bytes(b"preserve incomplete evidence")
        with patch.object(train, "run", side_effect=AssertionError("Cannot fit")):
            with self.assertRaises(FileExistsError):
                self.collect()
        self.assertEqual((self.output / "diagnostics.npz").read_bytes(), b"preserve incomplete evidence")
        weights = self.output / "tiny.weights.pt"
        weights.rename(self.output / "moved.weights.pt")
        with patch.object(train, "run", side_effect=AssertionError("Cannot fit")):
            with self.assertRaises(FileNotFoundError):
                self.collect()

    def test_diagnostic_checksum_and_invalid_row_order_are_detected(self):
        self.fit()
        self.collect()
        arrays_path, metadata_path = self.output / "diagnostics.npz", self.output / "diagnostics.json"
        with arrays_path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            experiment.read_diagnostics(self.output / "diagnostics")
        # Even a rewritten manifest must not excuse structurally invalid position records.
        with np.load(arrays_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["gates_positions"][0] = 4
        np.savez_compressed(arrays_path, **arrays)
        record = json.loads(metadata_path.read_text())
        record["arrays"].update(sha256=sha(arrays_path), bytes=arrays_path.stat().st_size)
        metadata_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "codon-grid positions"):
            experiment.read_diagnostics(self.output / "diagnostics")


if __name__ == "__main__":
    unittest.main()
