"""Synthetic A6 protocols, model arithmetic and mocked execution boundaries."""

from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import region_experiment as inputs, region_training as study
from shared_code.fusion import run_identity, token_embeddings as tokens, train


SETTINGS = {"projection_width": 8, "scorer_width": 3, "n_heads": 2, "dropout": 0.,
            "projection_activation": "tanh", "projection_norm": "none", "update_norm": "layer",
            "control_bottleneck": 3}


def fixture():
    rows = []
    for index in range(24):
        prefix = "".join("ACGT"[(index // 4 ** power) % 4] for power in range(4))
        suffix = "C" * (index % 3)
        sequence = prefix + "ATGAAATAA" + suffix
        track = np.zeros(len(sequence), dtype=np.int8)
        track[4:13:3] = 1
        rows.append({"sequence": sequence, "cds": track, "gene": f"g{index}", "target": index / 10})
    return inputs.prepare_cohort(pd.DataFrame(rows), "rnahl-human", grouping="homology",
                                 homology_map={f"g{i}": f"family{i // 2}" for i in range(24)},
                                 unmapped_gene_policy="gene")


class ArrayCache:
    def __init__(self, arrays):
        self.arrays = arrays
        self.width = arrays[0].shape[1]
        self.lengths = np.array([len(a) for a in arrays])
        self.cache_dir = "unused synthetic directory"

    def __getitem__(self, index):
        return self.arrays[index]

    def __len__(self):
        return len(self.arrays)

    def close(self):
        pass


def streams(spec=None):
    cohort = fixture()
    preparation = study.prepare_study(cohort, split_seed=42, mask_seed=0)
    plans = inputs.plan_chunks(cohort, window_lengths={"rnafm": 6, "rinalmo": 6, "esm2_8m": 6})
    caches = {key: ArrayCache([np.full((c.end - c.start, 3 if key == "esm2_8m" else 4),
                                     (i + 1) / 10, dtype=np.float32)
                              for i, c in enumerate(plan.chunks)]) for key, plan in plans.items()}
    return study.RegionStreams(cohort, inputs.RegionCaches(plans, caches), preparation,
                                spec or study.FitSpec("rnafm", "native", "position_gate", "true"))


def cache_records(stream):
    required = study.required_encoders(stream.spec)
    return {key: {"input_sha256": stream.caches.plans[key].identity["input_sha256"],
                  "width": stream.caches.caches[key].width} for key in required}


def fake_result(stream, config, identity, prediction=None):
    truth = stream.cohort.labels[stream.preparation["positions"]["test"], 0].astype(np.float32).tolist()
    prediction = truth if prediction is None else prediction
    test_score = train.score_rows("regression", np.asarray(truth), np.asarray(prediction))
    result = train.Result(stream.spec.method, stream.key, asdict(config), 1, 0, .5, test_score, .01,
                          history=[{"epoch": 0}], test_truth=truth, test_prediction=prediction)
    result.extra_fields = {"identity": identity, "producer": {"kind": "synthetic-test"}}
    return result


def save_report_panel(root, stream, config, template):
    """Write a complete invented panel without fitting or opening research inputs."""
    for spec in study.fit_panel():
        arm = replace(stream, spec=spec)
        for seed in (0, 1, 2):
            identity = deepcopy(template)
            identity["variant"] = spec.method
            identity["config"]["seed"] = seed
            identity["loader"]["seed"] = seed
            identity["inputs"]["spec"] = asdict(spec)
            identity["inputs"]["encoders"] = cache_records(arm)
            result = fake_result(arm, replace(config, seed=seed), identity)
            directory = Path(root) / "fits" / spec.key
            directory.mkdir(parents=True, exist_ok=True)
            result.save(directory / f"seed{seed}.json")


class RegionProtocolTests(unittest.TestCase):
    def test_core_panel_has_294_planned_fits_optional_is_explicit(self):
        self.assertEqual(len(study.fit_panel()), 49)
        self.assertEqual(49 * 2 * 3, 294)
        self.assertEqual(len(study.fit_panel(include_optional=True)), 61)
        self.assertEqual(len({spec.key for spec in study.fit_panel()}), 49)

    def test_group_split_donors_cover_same_cohort_without_cross_partition_leakage(self):
        cohort = fixture()
        record = study.prepare_study(cohort, split_seed=42, mask_seed=0)
        self.assertEqual(record, study.prepare_study(cohort, split_seed=42, mask_seed=0))
        for name, members in record["positions"].items():
            donors = [record["donor_positions"][i] for i in members]
            self.assertEqual(set(donors), set(members))
            self.assertTrue(all(a != b for a, b in zip(members, donors)))
            other = set().union(*(set(v) for k, v in record["positions"].items() if k != name))
            self.assertFalse(set(cohort.groups[members]) & set(cohort.groups[list(other)]))
        self.assertEqual(sorted(sum(record["positions"].values(), [])), list(range(24)))
        # A donor starting at 4/15 mapped to a 13-base recipient starts at floor(52/15)=3.
        recipient = next(i for i, l in enumerate(cohort.layouts) if l.sequence_length == 13)
        donor = next(i for i, l in enumerate(cohort.layouts) if l.sequence_length == 15)
        self.assertEqual(study._interval(cohort, recipient, donor), [3, 12])

    def test_whole_cohort_gene_grouping_or_changed_protocol_is_not_accepted(self):
        cohort = fixture()
        gene = inputs.prepare_cohort(cohort.frame, "rnahl-human", grouping="gene")
        with self.assertRaisesRegex(ValueError, "homology map"):
            study.prepare_study(gene, split_seed=42, mask_seed=0)
        with self.assertRaisesRegex(ValueError, "seed 42"):
            study.prepare_study(cohort, split_seed=1, mask_seed=0)

    def test_partial_map_split_keeps_repeated_genes_and_mapped_families_together(self):
        source = fixture().frame
        repeats = source.iloc[[12, 12, 17]].copy()
        repeats["target"] = [4.25, -2.5, 6.75]
        source = pd.concat([source, repeats], ignore_index=True)
        before = source.copy(deep=True)
        cohort = inputs.prepare_cohort(
            source, "rnahl-human", grouping="homology", unmapped_gene_policy="gene",
            homology_map={f"g{i}": f"family{i // 2}" for i in range(12)},
        )
        record = study.prepare_study(cohort, split_seed=42, mask_seed=0)
        self.assertEqual(record, study.prepare_study(cohort, split_seed=42, mask_seed=0))
        assignment = {position: split for split, positions in record["positions"].items()
                      for position in positions}
        self.assertEqual(set(assignment), set(range(len(source))))
        for group in set(cohort.groups):
            positions = np.flatnonzero(cohort.groups == group)
            self.assertEqual(len({assignment[int(i)] for i in positions}), 1)
        self.assertEqual(assignment[12], assignment[24])
        self.assertEqual(assignment[12], assignment[25])
        self.assertEqual(assignment[17], assignment[26])
        self.assertEqual(assignment[0], assignment[1])
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(set(cohort.groups[record["positions"][first]])
                             & set(cohort.groups[record["positions"][second]]))
        self.assertEqual(cohort.groups[12], "gene:g12")
        self.assertEqual(cohort.groups[0], "homology:family0")
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(cohort.frame, before)
        np.testing.assert_array_equal(cohort.labels[:, 0], before["target"].to_numpy())

    def test_old_unmapped_policy_preparation_is_refused_without_replacing_it(self):
        cohort = fixture()
        current = study.prepare_study(cohort, split_seed=42, mask_seed=0)
        legacy_cohort = inputs.prepare_cohort(
            cohort.frame, "rnahl-human", grouping="homology",
            homology_map={f"g{i}": f"family{i // 2}" for i in range(24)},
        )
        with self.assertRaisesRegex(ValueError, "homology map"):
            study.prepare_study(legacy_cohort, split_seed=42, mask_seed=0)
        # Recreate an old completed record without executing its superseded policy.
        legacy = deepcopy(current)
        legacy["cohort"] = legacy_cohort.identity
        legacy["sha256"] = inputs._digest({key: value for key, value in legacy.items() if key != "sha256"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preparation.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "identity|protocol"):
                study.read_preparation(path, cohort)
            with self.assertRaises(ValueError):
                study.publish_preparation(path, current)
            self.assertEqual(path.read_bytes(), before)

    def test_preparation_publish_is_immutable_and_portable(self):
        cohort = fixture()
        record = study.prepare_study(cohort, split_seed=42, mask_seed=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preparation.json"
            study.publish_preparation(path, record)
            before = path.read_bytes()
            study.publish_preparation(path, record)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(study.read_preparation(path, cohort), record)
            changed = study.prepare_study(cohort, split_seed=42, mask_seed=1)
            with self.assertRaises(ValueError):
                study.publish_preparation(path, changed)
            self.assertEqual(path.read_bytes(), before)

    def test_transcript_permuted_masks_do_not_move_protein_availability(self):
        stream = streams(study.FitSpec("rnafm", "codon_pool", "position_gate", "permuted"))
        dataset = inputs.RegionDataset(stream.cohort, stream.caches, rna_encoder="rnafm", grid="codon_pool",
                                        region_intervals=stream.preparation["permuted_intervals"])
        original = inputs.RegionDataset(stream.cohort, stream.caches, rna_encoder="rnafm", grid="codon_pool")
        differs = False
        for index in range(len(dataset)):
            row, true = dataset[index], original[index]
            np.testing.assert_array_equal(row["masks"]["prot"], true["masks"]["prot"])
            np.testing.assert_array_equal(row["streams"]["prot"], true["streams"]["prot"])
            differs |= not np.array_equal(row["region"], true["region"])
        self.assertTrue(differs)


class RegionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_sequence_and_position_local_gates_share_scorers_and_readout(self):
        stream = streams()
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=3))["test"]))
        position = study.RegionModel(stream.widths, "position_gate", SETTINGS, 100).eval()
        sequence = study.RegionModel(stream.widths, "sequence_gate", SETTINGS, 100).eval()
        sequence.load_state_dict(position.state_dict(), strict=True)
        projected = {name: sequence.project[name](batch.streams[name]).masked_fill(
            ~batch.masks[name].unsqueeze(-1), 0) for name in ("rna", "prot")}
        joint = torch.cat(list(projected.values()), dim=-1)
        pooled = (joint * batch.masks["rna"].unsqueeze(-1)).sum(1) / batch.masks["rna"].sum(1)[:, None]
        _, _, aux = sequence.features(batch)
        for index, name in enumerate(("rna", "prot")):
            expected = torch.sigmoid(sequence.gate[name](pooled)).expand(-1, joint.shape[1])
            allowed = batch.masks["rna"] if name == "rna" else batch.region & batch.masks["prot"]
            torch.testing.assert_close(aux["gates"][..., index], expected.masked_fill(~allowed, 0))
        for name in batch.streams:
            batch.streams[name][~batch.masks[name]] = 1e6
        _, _, changed = sequence.features(batch)
        torch.testing.assert_close(changed["gates"], aux["gates"])

    def test_full_width_affine_control_retains_signal_and_active_adjustment(self):
        stream = streams()
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=2))["test"]))
        model = study.RegionModel(stream.widths, "position_control_residual", SETTINGS, 100).eval()
        width = SETTINGS["projection_width"]
        self.assertEqual(sum(p.numel() for p in model.adjust_down.parameters()) +
                         sum(p.numel() for p in model.adjust_up.parameters()), 4 * width + 1)
        self.assertIsNone(model.adjust_up.bias)
        projected = {m: model.project[m](batch.streams[m]).masked_fill(~batch.masks[m].unsqueeze(-1), 0)
                     for m in ("rna", "prot")}
        projected["prot"] = projected["prot"].masked_fill(~batch.region.unsqueeze(-1), 0)
        expected = model.mix(torch.cat((.5 * projected["rna"], .5 * projected["prot"],
                                       batch.region.unsqueeze(-1).float()), -1))
        expected = expected.masked_fill(~batch.masks["rna"].unsqueeze(-1), 0)
        actual, _, _ = model.features(batch)
        torch.testing.assert_close(actual, expected)
        # At zero-up initialization only up has an immediate gradient; after a
        # nonzero update both trainable factors influence the output.
        actual.square().sum().backward()
        self.assertGreater(model.adjust_up.weight.grad.abs().sum(), 0)
        with torch.no_grad():
            model.adjust_up.weight.fill_(.1)
        model.zero_grad()
        model.features(batch)[0].square().sum().backward()
        for parameter in (*model.adjust_down.parameters(), *model.adjust_up.parameters()):
            self.assertGreater(parameter.grad.abs().sum(), 0)
        self.assertEqual(4 * 600 + 1, 2401)
        self.assertEqual(2 * (2 * 600 + 1), 2402)

    def test_all_cds_attention_controls_share_masks_and_skip(self):
        stream = streams()
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=2))["test"]))
        models = {name: study.RegionModel(stream.widths, name, SETTINGS, 100).eval()
                  for name in ("attention_cds", "attention_cds_sequence", "attention_cds_ungated")}
        position = models["attention_cds"]
        models["attention_cds_sequence"].load_state_dict(position.state_dict())
        ungated = {key: value for key, value in position.state_dict().items() if not key.startswith("attention.gate.")}
        models["attention_cds_ungated"].load_state_dict(ungated)
        allowed = batch.region & batch.masks["prot"]
        for method, model in models.items():
            features, real, aux = model.features(batch)
            self.assertFalse(aux["update_weights"][..., 0][~allowed].any())
            if method == "attention_cds_ungated":
                torch.testing.assert_close(aux["update_weights"][..., 0], allowed.float())
            projected = model.attention.project["rna"](batch.streams["rna"]).masked_fill(~real.unsqueeze(-1), 0)
            torch.testing.assert_close(features[~allowed], projected[~allowed])
        batch.region.zero_()
        for model in models.values():
            features, real, _ = model.features(batch)
            projected = model.attention.project["rna"](batch.streams["rna"]).masked_fill(~real.unsqueeze(-1), 0)
            torch.testing.assert_close(features, projected)

    def test_native_protein_single_is_deduplicated_and_never_reads_rna(self):
        spec, = [spec for spec in study.fit_panel() if spec.method == "protein_only"]
        stream = streams(spec)
        stream.caches.caches = {"esm2_8m": stream.caches.caches["esm2_8m"]}
        stream.caches.plans = {"esm2_8m": stream.caches.plans["esm2_8m"]}
        config = train.TrainConfig(batch_size=2)
        with patch.object(inputs, "RegionDataset", side_effect=AssertionError("RNA dataset requested")):
            loaders = study.make_loaders(stream, config)
            study.validate_loaders(stream, config, loaders)
            batch = next(iter(loaders["test"]))
        self.assertEqual(set(batch.streams), {"prot"})
        self.assertEqual(batch.positions, ())
        self.assertEqual(batch.lengths.tolist(), [2, 2])
        model = study.build_model(stream, "protein_only", config, settings=SETTINGS)
        prediction, _ = model(batch)
        prediction.square().sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        study.make_identity(stream, config, SETTINGS, loaders, cache_records(stream))
        with self.assertRaisesRegex(ValueError, "no RNA-region"):
            study.collect_diagnostics(stream, config, SETTINGS, "unused", "cpu", producer={"test": True})

    def test_single_selection_uses_only_seed_zero_validation_and_requires_all_candidates(self):
        panel = study.fit_panel()
        candidates = [spec for spec in panel if spec.method in ("rna_only", "protein_only")]
        fits = {spec.key: [SimpleNamespace(config={"seed": seed}, val_score=index if seed == 0 else -index,
                                         test_score=100-index) for seed in (0, 1, 2)]
                for index, spec in enumerate(candidates)}
        selected = study.validation_selected_single(fits, panel)
        self.assertEqual(selected["selected"], candidates[-1].key)
        self.assertEqual(selected["held_fixed_for_training_seeds"], [0, 1, 2])
        self.assertEqual(selected["attention_anchor"], "rna")
        fits.pop(candidates[0].key)
        with self.assertRaisesRegex(ValueError, "requires seeds"):
            study.validation_selected_single(fits, panel)
        with self.assertRaisesRegex(ValueError, "all four RNA"):
            study.validation_selected_single(fits, panel[:-1])

    def test_sequence_attention_diagnostics_broadcast_over_real_positions(self):
        stream = streams(study.FitSpec("rnafm", "native", "attention_cds_sequence", "true"))
        config = train.TrainConfig(batch_size=2)
        loader = study.make_loaders(stream, config)["test"]
        model = study.build_model(stream, stream.spec.method, config, settings=SETTINGS).eval()
        model.collect_norms = True
        rows, prediction, truth = study._diagnostic_rows(model, loader, stream.cohort, "cpu")
        self.assertEqual(len(rows), len(stream.preparation["positions"]["test"]))
        self.assertEqual(len(prediction), len(truth))
        for row in rows:
            gates = [region["gates"] for region in row["regions"].values() if region["positions"]]
            for region in row["regions"].values():
                if region["positions"]:
                    self.assertIsInstance(region["anchor_norm"], float)
                    self.assertIsInstance(region["fused_norm"], float)
            for gate in gates[1:]:
                np.testing.assert_allclose(gate, gates[0], rtol=1e-6)
            self.assertEqual(row["regions"]["utr5"]["update_weights"], [0.])

    def test_old_narrow_report_cannot_satisfy_repaired_panel(self):
        stream = streams()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / stream.key
            study.publish_preparation(root / "preparation.json", stream.preparation)
            old = root / "reports" / "three_seed_core.json"
            old.parent.mkdir()
            old.write_text('{"schema":"region-three-seed-report.v1","context":{}}')
            before = old.read_bytes()
            with patch.object(train, "seed_averaged_bootstrap", side_effect=AssertionError("statistics")):
                with self.assertRaises(FileNotFoundError):
                    study.saved_comparisons(temporary, dataset_key=stream.key, draws=5,
                                             bootstrap_seed=0, producer=None)
            self.assertEqual(old.read_bytes(), before)
            self.assertFalse((old.parent / "three_seed_controls_v2.json").exists())

    def test_every_core_and_optional_arm_has_finite_forward_and_gradients(self):
        config = train.TrainConfig(seed=0, batch_size=2, channels=100, tconv="equation")
        for method, mask in study.CORE + study.OPTIONAL:
            with self.subTest(method=method, mask=mask):
                stream = streams(study.FitSpec("rnafm", "native", method, mask))
                batch = next(iter(study.make_loaders(stream, config)["train"]))
                model = study.build_model(stream, method, config, settings=SETTINGS)
                prediction, aux = model(batch)
                self.assertEqual(prediction.shape, (2, 1))
                self.assertEqual(batch.labels.shape, (2,))
                self.assertNotIn("alpha", aux)
                prediction.square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_zero_protein_updates_recover_projected_rna_for_cds_attention(self):
        stream = streams(study.FitSpec("rnafm", "native", "attention_cds", "true"))
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=2))["test"]))
        batch.region.zero_()
        model = study.build_model(stream, "attention_cds", train.TrainConfig(), settings=SETTINGS).eval()
        features, real, aux = model.features(batch)
        projected = model.attention.project["rna"](batch.streams["rna"]).masked_fill(~real.unsqueeze(-1), 0)
        torch.testing.assert_close(features, projected)
        self.assertFalse(aux["update_weights"].any())

    def test_protein_gates_cannot_create_residues_in_utrs_or_stop_codon(self):
        stream = streams(study.FitSpec("rnafm", "native", "position_gate", "permuted"))
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=2))["test"]))
        model = study.build_model(stream, "position_gate", train.TrainConfig(), settings=SETTINGS)
        _, _, aux = model.features(batch)
        allowed = batch.region & batch.masks["prot"]
        self.assertFalse(aux["gates"][:, :, 1][~allowed].any())

    def test_region_d_joint_gate_equation_retains_concat_mask_readout(self):
        settings = {**SETTINGS, "projection_width": 1, "n_heads": 1, "update_norm": "none"}
        model = study.RegionModel({"rna": 1, "prot": 1}, "position_gate", settings, 100)
        model.collect_norms = True
        with torch.no_grad():
            for projection in model.project.values():
                projection[0].weight.fill_(1)
                projection[0].bias.zero_()
            model.gate["rna"].weight.copy_(torch.tensor([[0., 2.]]))
            model.gate["rna"].bias.zero_()
            model.gate["prot"].weight.copy_(torch.tensor([[3., 0.]]))
            model.gate["prot"].bias.zero_()
            model.mix.weight.copy_(torch.tensor([[1., 2., .25]]))
            model.mix.bias.zero_()
        batch = SimpleNamespace(
            streams={"rna": torch.tensor([[[.2], [.4], [99.]]]),
                     "prot": torch.tensor([[[.5], [99.], [99.]]])},
            masks={"rna": torch.tensor([[True, True, False]]),
                   "prot": torch.tensor([[True, False, False]])},
            region=torch.tensor([[True, False, False]]),
        )
        features, _, aux = model.features(batch)
        r, p = torch.tanh(torch.tensor(.2)), torch.tanh(torch.tensor(.5))
        gr, gp = torch.sigmoid(2 * p), torch.sigmoid(3 * r)
        expected = torch.tensor([[[gr * r + 2 * gp * p + .25], [.5 * torch.tanh(torch.tensor(.4))], [0.]]])
        torch.testing.assert_close(features, expected)
        torch.testing.assert_close(aux["gates"][0, 0], torch.stack((gr, gp)))
        self.assertFalse(aux["contribution_norms"][0, 1:, 1].any())
        batch.streams["prot"][0, 0] += 1
        _, _, changed = model.features(batch)
        self.assertGreater(changed["gates"][0, 0, 0], aux["gates"][0, 0, 0])
        torch.testing.assert_close(changed["gates"][..., 1], aux["gates"][..., 1])

    def test_region_gate_masks_block_unavailable_protein_gradients(self):
        stream = streams(study.FitSpec("rnafm", "native", "position_gate", "true"))
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=2))["test"]))
        batch.streams = {name: value.requires_grad_() for name, value in batch.streams.items()}
        model = study.build_model(stream, "position_gate", train.TrainConfig(), settings=SETTINGS)
        features, _, _ = model.features(batch)
        features.square().sum().backward()
        for name, tensor in batch.streams.items():
            self.assertFalse(tensor.grad[~batch.masks[name]].any())
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_padded_values_cannot_change_predictions(self):
        stream = streams(study.FitSpec("rnafm", "native", "attention_cds", "true"))
        batch = next(iter(study.make_loaders(stream, train.TrainConfig(batch_size=5))["val"]))
        model = study.build_model(stream, "attention_cds", train.TrainConfig(), settings=SETTINGS).eval()
        first = model(batch)[0].detach()
        for name in batch.streams:
            batch.streams[name][~batch.masks[name]] = 9999
        second = model(batch)[0].detach()
        torch.testing.assert_close(first, second)

    def test_identity_includes_split_donors_grid_source_and_loader_recovery(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        loaders = study.make_loaders(stream, config)
        identity = study.make_identity(stream, config, SETTINGS, loaders, cache_records(stream))
        run_identity.validate_loaders(identity, loaders)
        self.assertEqual(identity["inputs"]["preparation_sha256"], stream.preparation["sha256"])
        self.assertIn("region_training.py", identity["model"]["implementation_sha256"])
        self.assertGreater(identity["model"]["parameter_counts"]["total"], 0)
        self.assertEqual(identity["model"]["gate"], {
            "scorer": "affine_sigmoid", "scorer_width_active": False,
            "input": "joint_projected_rna_protein", "readout": "gated_concat_mask_mixer",
        })
        self.assertIn("generator", identity["loader"]["recovery"]["train"]["loader"]["fields"])
        loaders["train"].dataset.indices = loaders["train"].dataset.indices[::-1]
        with self.assertRaisesRegex(ValueError, "loader differs"):
            study.validate_loaders(stream, config, loaders)

    def test_declared_counts_and_loader_contract_match_actual_components(self):
        base = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        methods = sorted({spec.method for spec in study.fit_panel(include_optional=True)})
        for method in methods:
            for width, normalization in ((8, "none"), (12, "layer")):
                with self.subTest(method=method, width=width, normalization=normalization):
                    spec = (study.fit_panel()[-1] if method == "protein_only" else
                            study.FitSpec("rnafm", "native", method, "true"))
                    stream = replace(base, spec=spec)
                    settings = {**SETTINGS, "projection_width": width, "update_norm": normalization}
                    loaders = study.make_loaders(stream, config)
                    model = study.build_model(stream, method, config, settings=settings)
                    identity = study.make_identity(stream, config, settings, loaders, cache_records(stream))
                    self.assertEqual(identity["model"]["parameter_counts"], {
                        "total": sum(p.numel() for p in model.parameters()),
                        "prediction": sum(p.numel() for p in model.head.parameters()),
                    })
                    self.assertEqual(identity["loader"]["recovery"], {
                        name: run_identity._component_contract(loader) for name, loader in loaders.items()
                    })

    def test_direct_helpers_still_check_mutated_cohort(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2)
        loaders = study.make_loaders(stream, config)
        stream.cohort.frame.at[0, "target"] += 1
        for call in (
            lambda: study.make_loaders(stream, config),
            lambda: study.make_identity(stream, config, SETTINGS, loaders, cache_records(stream)),
            lambda: study.validate_preparation(stream.cohort, stream.preparation),
        ):
            with self.subTest(call=call), self.assertRaisesRegex(ValueError, "Prepared cohort changed"):
                call()

    def test_saved_inventory_does_not_open_datasets_or_token_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(inputs, "load_local_frame", side_effect=AssertionError("data read")), \
                 patch.object(inputs, "open_token_caches", side_effect=AssertionError("cache read")):
                rows = study.saved_inventory(temporary, dataset_key="rnahl-human")
            self.assertEqual(len(rows), 147)
            self.assertEqual({r["status"] for r in rows}, {"missing"})
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_reader_refuses_loader_seed_or_batch_mismatch(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        original = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), cache_records(stream))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed0.json"
            for field, value in (("seed", 1), ("batch_size", 3), ("seed", True)):
                with self.subTest(field=field, value=value):
                    identity = deepcopy(original)
                    identity["loader"][field] = value
                    fake_result(stream, config, identity).save(path)
                    before = path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "loader seed or batch size"):
                        study.read_fit(path)
                    self.assertEqual(path.read_bytes(), before)

    def test_historical_gate_fit_remains_readable_but_cannot_be_reused_as_current(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        current = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), cache_records(stream))
        historical = deepcopy(current)
        del historical["model"]["gate"]
        historical["model"]["implementation_sha256"]["controlled_models.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed0.json"
            fake_result(stream, config, historical).save(path)
            before = path.read_bytes()
            self.assertEqual(study.read_fit(path).extra_fields["identity"], historical)
            with self.assertRaisesRegex(ValueError, "different region experiment"):
                study.read_fit(path, expected_identity=current)
            self.assertEqual(before, path.read_bytes())

    def test_reader_requires_numeric_finite_validation_and_test_scores(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        identity = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), cache_records(stream))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed0.json"
            for field, value in (("test_score", True), ("val_score", None), ("val_score", float("nan"))):
                with self.subTest(field=field, value=value):
                    result = fake_result(stream, config, identity)
                    setattr(result, field, value)
                    result.save(path)
                    with self.assertRaisesRegex(ValueError, "complete valid"):
                        study.read_fit(path)

    def test_report_refuses_mixed_code_before_reading_later_fits_or_statistics(self):
        stream = streams(study.fit_panel()[0])
        config = train.TrainConfig(batch_size=2, tconv="equation")
        original = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), cache_records(stream))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / stream.key
            study.publish_preparation(root / "preparation.json", stream.preparation)
            folder = root / "fits" / stream.spec.key
            folder.mkdir(parents=True)
            for seed in (0, 1):
                identity = deepcopy(original)
                identity["config"]["seed"] = seed
                identity["loader"]["seed"] = seed
                if seed == 1:
                    identity["model"]["implementation_sha256"]["region_training.py"] = "0" * 64
                fake_result(stream, replace(config, seed=seed), identity).save(folder / f"seed{seed}.json")
            before = {path: path.read_bytes() for path in folder.iterdir()}
            with patch.object(train, "seed_averaged_bootstrap", side_effect=AssertionError("statistics ran")):
                with self.assertRaisesRegex(ValueError, "different implementation revisions"):
                    study.saved_comparisons(temporary, dataset_key=stream.key, draws=5,
                                             bootstrap_seed=0, producer={"test": True}, run_analysis=True)
            self.assertFalse((root / "reports").exists())
            self.assertEqual(before, {path: path.read_bytes() for path in folder.iterdir()})

    def test_fit_handoff_reuses_completed_record_and_never_replaces_it(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        records = cache_records(stream)

        # run_fit iterates a set, so identify each synthetic cache by a unique path.
        for key, cache in stream.caches.caches.items():
            cache.cache_dir = key
        with tempfile.TemporaryDirectory() as temporary:
            def fake_run(actual_streams, method, actual_config, device, destination, **kwargs):
                self.assertIs(kwargs["builder"], study.build_model)
                self.assertTrue(kwargs["require_predictions"])
                run_identity.validate_loaders(kwargs["run_identity"], kwargs["loaders"])
                Path(destination).mkdir(parents=True)
                result = fake_result(actual_streams, actual_config, kwargs["run_identity"])
                result.save(Path(destination) / "seed0.json")
                return result

            with patch.object(run_identity, "cache_record", side_effect=lambda key: records[key]), \
                 patch.object(train, "run", side_effect=fake_run) as runner:
                with patch.object(inputs, "_frame_hash", wraps=inputs._frame_hash) as full_hash:
                    study.run_fit(stream, config, SETTINGS, temporary, "cpu", producer={"test": True})
                    self.assertEqual(full_hash.call_count, 1)
                path = Path(temporary) / stream.spec.key / "seed0.json"
                before = path.read_bytes()
                with patch.object(inputs, "_frame_hash", wraps=inputs._frame_hash) as full_hash, \
                     patch.object(inputs.RegionDataset, "_from_validated_cohort", side_effect=AssertionError("dataset constructed")), \
                     patch.object(study, "_make_loaders", side_effect=AssertionError("loaders constructed")), \
                     patch.object(study, "build_model", side_effect=AssertionError("model constructed")):
                    study.run_fit(stream, config, SETTINGS, temporary, "cpu", producer=None)
                    self.assertEqual(full_hash.call_count, 1)
                runner.assert_called_once()
                self.assertEqual(path.read_bytes(), before)
                changed = replace(config, lr=config.lr * 2)
                with self.assertRaisesRegex(ValueError, "different region experiment"):
                    study.run_fit(stream, changed, SETTINGS, temporary, "cpu", producer={"test": True})
                self.assertEqual(path.read_bytes(), before)

    def test_reuse_rejects_changed_cohort_counts_settings_and_cache_before_construction(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        records = cache_records(stream)
        for key, cache in stream.caches.caches.items():
            cache.cache_dir = key
        identity = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), records)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / stream.spec.key / "seed0.json"
            path.parent.mkdir()
            fake_result(stream, config, identity).save(path)
            before = path.read_bytes()
            with patch.object(run_identity, "cache_record", side_effect=lambda key: records[key]), \
                 patch.object(study, "_make_loaders", side_effect=AssertionError("loaders constructed")), \
                 patch.object(study, "build_model", side_effect=AssertionError("model constructed")), \
                 patch.object(train, "run", side_effect=AssertionError("training started")):
                for changed_config, settings in (
                    (replace(config, seed=1), SETTINGS),
                    (config, {**SETTINGS, "projection_width": 10}),
                    (replace(config, channels=100.), SETTINGS),
                    (replace(config, batch_size=True), SETTINGS),
                ):
                    with self.subTest(config=changed_config, settings=settings), self.assertRaises(ValueError):
                        study.run_fit(stream, changed_config, settings, temporary, "cpu", producer=None)
                corrupt = deepcopy(identity)
                corrupt["model"]["parameter_counts"]["total"] += 1
                fake_result(stream, config, corrupt).save(path)
                with self.assertRaisesRegex(ValueError, "different region experiment"):
                    study.run_fit(stream, config, SETTINGS, temporary, "cpu", producer=None)
                path.write_bytes(before)
                records["rnafm"]["input_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "planned ordered windows"):
                    study.run_fit(stream, config, SETTINGS, temporary, "cpu", producer=None)
                records["rnafm"]["input_sha256"] = stream.caches.plans["rnafm"].identity["input_sha256"]
                stream.cohort.frame.at[0, "target"] += 1
                with self.assertRaisesRegex(ValueError, "Prepared cohort changed"):
                    study.run_fit(stream, config, SETTINGS, temporary, "cpu", producer=None)
            self.assertEqual(path.read_bytes(), before)

    def test_diagnostics_use_saved_weights_and_cannot_retrain(self):
        stream = streams(study.FitSpec("rnafm", "native", "attention_cds", "true"))
        config = train.TrainConfig(batch_size=2, tconv="equation")
        records = cache_records(stream)
        for key, cache in stream.caches.caches.items():
            cache.cache_dir = key
        loaders = study.make_loaders(stream, config)
        identity = study.make_identity(stream, config, SETTINGS, loaders, records)
        model = study.build_model(stream, "attention_cds", config, settings=SETTINGS).eval()
        with torch.no_grad():
            prediction = np.concatenate([model(batch)[0].squeeze(-1).numpy() for batch in loaders["test"]]).tolist()
        result = fake_result(stream, config, identity, prediction)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / stream.spec.key
            directory.mkdir()
            weights = directory / "seed0.weights.pt"
            torch.save({"format": "fusion-state-dict-v1", "identity": identity,
                        "best_epoch": 0, "config": result.config, "state_dict": model.state_dict()}, weights)
            result.extra_fields["model_artifact"] = {"file": weights.name, "sha256": study.tokens_sha(weights),
                                                      "size_bytes": weights.stat().st_size}
            result.save(directory / "seed0.json")
            legacy_path = directory / "seed0.diagnostics.json"
            legacy = {"schema": "region-diagnostics.v1", "context": {
                "fit_sha256": study.tokens_sha(directory / "seed0.json"),
                "weights_sha256": result.extra_fields["model_artifact"]["sha256"], "identity": identity},
                "producer": {"test": True}, "summary": "legacy gate-only fixture", "rows": []}
            for index in stream.preparation["positions"]["test"]:
                legacy["rows"].append({"row_id": stream.cohort.row_ids[index], "positions": 1,
                    "regions": {"utr5": {"positions": 0, "gates": None},
                                "cds": {"positions": 1, "gates": [.5]},
                                "utr3": {"positions": 0, "gates": None}}})
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            legacy_bytes = legacy_path.read_bytes()
            with patch.object(study, "build_model", side_effect=AssertionError("model constructed")), \
                 patch.object(inputs, "open_token_caches", side_effect=AssertionError("cache opened")), \
                 patch.object(torch, "load", side_effect=AssertionError("weights read")):
                historical = study.read_diagnostics(directory / "seed0.json")
                self.assertEqual(historical["version"], 1)
                self.assertEqual(historical["norm_status"], "legacy_unavailable")
                self.assertEqual(historical["integrity"], "legacy_structure_and_fit_binding")
                self.assertNotIn("update_norms", historical["record"]["rows"][0]["regions"]["cds"])
            with patch.object(run_identity, "cache_record", side_effect=lambda key: records[key]), \
                 patch.object(train, "run", side_effect=AssertionError("retraining")):
                with patch.object(inputs, "_frame_hash", wraps=inputs._frame_hash) as full_hash:
                    saved = study.collect_diagnostics(stream, config, SETTINGS, temporary, "cpu", producer={"test": True})
                    self.assertEqual(full_hash.call_count, 1)
                self.assertEqual(len(saved["rows"]), len(stream.preparation["positions"]["test"]))
                self.assertEqual(set(saved["rows"][0]["regions"]), {"utr5", "cds", "utr3"})
                cds = saved["rows"][0]["regions"]["cds"]
                self.assertIn("raw_update_norms", cds)
                self.assertIn("update_norms", cds)
                self.assertIn("anchor_norm", cds)
                self.assertIn("fused_norm", cds)
                path = directory / "seed0.diagnostics-v2.json"
                before = path.read_bytes()
                with patch.object(inputs, "_frame_hash", wraps=inputs._frame_hash) as full_hash, \
                     patch.object(study, "_make_loaders", side_effect=AssertionError("loaders constructed")), \
                     patch.object(study, "build_model", side_effect=AssertionError("model constructed")), \
                     patch.object(torch, "load", side_effect=AssertionError("weights loaded")):
                    self.assertEqual(saved, study.collect_diagnostics(stream, config, SETTINGS, temporary, "cpu", producer=None))
                    self.assertEqual(full_hash.call_count, 1)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
                self.assertEqual(saved["schema"], "region-diagnostics.v2")
                self.assertEqual(saved["context"]["diagnostic_definition"], study.DIAGNOSTIC_DEFINITION)
                self.assertIn("region_training.py", saved["context"]["implementation_sha256"])
            with patch.object(study, "build_model", side_effect=AssertionError("model constructed")), \
                 patch.object(inputs, "open_token_caches", side_effect=AssertionError("cache opened")), \
                 patch.object(torch, "load", side_effect=AssertionError("weights read")):
                current = study.read_diagnostics(directory / "seed0.json")
                self.assertEqual(current["version"], 2)
                self.assertEqual(current["integrity"], "record_sha256_and_fit_binding")
                self.assertEqual(current["record"], saved)
                corrupt = deepcopy(saved)
                corrupt["rows"][0]["regions"]["cds"]["update_norms"] = [999.]
                path.write_text(json.dumps(corrupt), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "content hash"):
                    study.read_diagnostics(directory / "seed0.json")
                # Even a newly hashed record must still bind to this unchanged fit.
                corrupt = deepcopy(saved)
                corrupt["context"]["fit_sha256"] = "0" * 64
                corrupt["sha256"] = inputs._digest({k: v for k, v in corrupt.items() if k != "sha256"})
                path.write_text(json.dumps(corrupt), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unchanged source fit"):
                    study.read_diagnostics(directory / "seed0.json")
                self.assertEqual(legacy_path.read_bytes(), legacy_bytes)

    def test_missing_diagnostics_cannot_trigger_inputs_or_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(study, "read_fit", side_effect=AssertionError("fit opened")), \
                 patch.object(study, "build_model", side_effect=AssertionError("model constructed")), \
                 patch.object(study, "collect_diagnostics", side_effect=AssertionError("inference")):
                self.assertIsNone(study.read_diagnostics(Path(temporary) / "seed0.json"))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_diagnostic_failures_release_model_and_preserve_primary_error(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        records = cache_records(stream)
        for key, cache in stream.caches.caches.items():
            cache.cache_dir = key
        identity = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), records)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / stream.spec.key
            directory.mkdir()
            weights = directory / "seed0.weights.pt"
            result = fake_result(stream, config, identity)
            torch.save({"format": "fusion-state-dict-v1", "identity": identity,
                        "best_epoch": 0, "config": result.config, "state_dict": {}}, weights)
            result.extra_fields["model_artifact"] = {
                "file": weights.name, "sha256": study.tokens_sha(weights), "size_bytes": weights.stat().st_size}
            path = directory / "seed0.json"
            result.save(path)
            before = {p: p.read_bytes() for p in directory.iterdir()}
            for phase in ("state-load", "inference", "cleanup-only"):
                with self.subTest(phase=phase):
                    model = Mock()
                    model.to.return_value = model
                    model.eval.return_value = model
                    original = RuntimeError(phase)
                    if phase == "state-load":
                        model.load_state_dict.side_effect = original
                    model.cpu.side_effect = original if phase == "cleanup-only" else RuntimeError("cleanup failure")
                    inferred = ([], result.test_prediction, result.test_truth)
                    with patch.object(run_identity, "cache_record", side_effect=lambda key: records[key]), \
                         patch.object(study, "build_model", return_value=model), \
                         patch.object(study, "_diagnostic_rows", return_value=inferred,
                                      side_effect=original if phase == "inference" else None), \
                         patch.object(train, "run", side_effect=AssertionError("retraining")):
                        with self.assertRaises(RuntimeError) as failure:
                            study.collect_diagnostics(stream, config, SETTINGS, temporary, "cpu",
                                                      producer={"test": True})
                    self.assertIs(failure.exception, original)
                    model.cpu.assert_called_once()
                    if phase != "cleanup-only":
                        self.assertIn("cleanup also failed", " ".join(failure.exception.__notes__))
                    self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir()})

    def test_three_seed_report_refuses_missing_fits_reuses_and_binds_sources(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        template = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), cache_records(stream))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / stream.key
            study.publish_preparation(root / "preparation.json", stream.preparation)
            with patch.object(train, "seed_averaged_bootstrap", side_effect=AssertionError("premature statistics")):
                with self.assertRaises(FileNotFoundError):
                    study.saved_comparisons(temporary, dataset_key=stream.key, draws=5, bootstrap_seed=0,
                                             producer={"test": True}, run_analysis=True)
            save_report_panel(root, stream, config, template)
            with patch.object(train, "score_rows", side_effect=AssertionError("read-only scoring")), \
                 patch.object(train, "seed_averaged_bootstrap", side_effect=AssertionError("read-only bootstrap")), \
                 patch.object(inputs, "_publish_once", side_effect=AssertionError("read-only publication")):
                self.assertIsNone(study.saved_comparisons(temporary, dataset_key=stream.key, draws=5,
                                                          bootstrap_seed=0, producer=None))
            def bootstrap(task, truth, predictions, comparisons, **kwargs):
                self.assertEqual({len(v) for v in predictions.values()}, {3})
                np.testing.assert_array_equal(kwargs["groups"], stream.cohort.groups[stream.preparation["positions"]["test"]])
                return [{"comparison": label, "difference": 0., "ci_low": -.1, "ci_high": .1}
                        for label, _, _ in comparisons]
            with patch.object(train, "seed_averaged_bootstrap", side_effect=bootstrap) as compute:
                report = study.saved_comparisons(temporary, dataset_key=stream.key, draws=5, bootstrap_seed=0,
                                                  producer={"test": True}, run_analysis=True)
                before = {path: path.read_bytes() for path in root.rglob("*.json")}
                with patch.object(train, "score_rows", side_effect=AssertionError("saved report rescored")), \
                     patch.object(inputs, "_publish_once", side_effect=AssertionError("saved report rewritten")):
                    again = study.saved_comparisons(temporary, dataset_key=stream.key, draws=5,
                                                    bootstrap_seed=0, producer=None)
                self.assertEqual(report, again)
                self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*.json")})
                compute.assert_called_once()
            self.assertEqual(len(report["summary"]), 49)
            self.assertEqual(len(report["contrasts"]), 112)
            np.testing.assert_allclose([row["mean"] for row in report["summary"]], 1., rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose([row["seed_sd"] for row in report["summary"]], 0., rtol=1e-7, atol=1e-8)
            source = root / "fits" / study.fit_panel()[0].key / "seed0.json"
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Existing report differs"):
                study.saved_comparisons(temporary, dataset_key=stream.key, draws=5, bootstrap_seed=0, producer=None)


    def test_report_refuses_inconsistent_finite_scores_before_bootstrap_or_publication(self):
        stream = streams()
        config = train.TrainConfig(batch_size=2, tconv="equation")
        template = study.make_identity(stream, config, SETTINGS, study.make_loaders(stream, config), cache_records(stream))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / stream.key
            study.publish_preparation(root / "preparation.json", stream.preparation)
            save_report_panel(root, stream, config, template)
            source = root / "fits" / study.fit_panel()[0].key / "seed0.json"
            result = train.Result.load(source)
            # Perfect predictions have Spearman 1. Both values exceed rtol=1e-7, atol=1e-8.
            for inconsistent in (.5, 1. - 2e-7):
                with self.subTest(test_score=inconsistent):
                    result.test_score = inconsistent
                    result.save(source)
                    before = {path: path.read_bytes() for path in root.rglob("*.json")}
                    with patch.object(train, "seed_averaged_bootstrap") as bootstrap, \
                         patch.object(inputs, "_publish_once") as publish:
                        with self.assertRaisesRegex(ValueError, "score.*prediction|prediction.*score"):
                            study.saved_comparisons(temporary, dataset_key=stream.key, draws=5,
                                                     bootstrap_seed=0, producer={"test": True}, run_analysis=True)
                        bootstrap.assert_not_called()
                        publish.assert_not_called()
                    self.assertFalse((root / "reports").exists())
                    self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*.json")})


if __name__ == "__main__":
    unittest.main()
