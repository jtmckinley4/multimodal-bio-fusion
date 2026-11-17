"""Synthetic checks for controlled feature heads, without fitting or research inputs."""

from pathlib import Path
import sys
import unittest

import numpy as np  # Keep NumPy before PyTorch in the Windows project environment.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import controlled_models as models
from shared_code.fusion.heads import GatedMILFusion, TEMPERATURE_RANGE


WIDTHS = {"dna": 3, "rna": 4, "prot": 2}


def sequence_head(**overrides):
    settings = dict(projection_width=4, scorer_width=3, summary="mean",
                    normalization="softmax", shared_scorer=False,
                    temperature_range=TEMPERATURE_RANGE,
                    projection_activation="tanh", projection_norm="none")
    settings.update(overrides)
    return models.SequenceGateFusion(WIDTHS, **settings)


def aligned_inputs():
    torch.manual_seed(23)
    streams = {m: torch.randn(2, 3, width) for m, width in WIDTHS.items()}
    mask = torch.tensor([[True, True, False], [True, True, True]])
    return streams, mask


def attention_head(mode, **overrides):
    settings = dict(anchor="rna", projection_width=4, scorer_width=3,
                    n_heads=2, dropout=0., mode=mode, projection_activation="tanh",
                    projection_norm="none", update_norm="none")
    settings.update(overrides)
    return models.AnchorAttentionFusion(WIDTHS, **settings)


def attention_inputs():
    torch.manual_seed(31)
    streams = {"dna": torch.randn(2, 5, 3), "rna": torch.randn(2, 3, 4),
               "prot": torch.randn(2, 2, 2)}
    masks = {"dna": torch.tensor([[True, True, True, False, False], [True] * 5]),
             "rna": torch.tensor([[True, True, False], [True, True, True]]),
             "prot": torch.tensor([[True, False], [False, False]])}
    return streams, masks


class ControlledModelsTests(unittest.TestCase):
    def test_joint_sequence_gate_has_same_scorers_and_independent_masked_mean_equation(self):
        settings = dict(projection_width=4, scorer_width=3,
                        projection_activation="tanh", projection_norm="none")
        position = models.PositionGateFusion(WIDTHS, **settings)
        sequence = models.JointSequenceGateFusion(WIDTHS, **settings)
        sequence.load_state_dict(position.state_dict(), strict=True)
        self.assertEqual(models.parameter_count(sequence), models.parameter_count(position))
        streams, mask = aligned_inputs()
        actual, aux = sequence(streams, mask)
        projected = {m: sequence.project[m](streams[m]) for m in WIDTHS}
        joint = torch.cat(list(projected.values()), -1)
        mean = torch.stack([joint[i, mask[i]].mean(0) for i in range(len(mask))])
        logits = torch.cat([mean @ sequence.gate[m].weight.T + sequence.gate[m].bias for m in WIDTHS], -1)
        weights = torch.sigmoid(logits)
        expected = sum(weights[:, i, None, None] * projected[m] for i, m in enumerate(WIDTHS))
        torch.testing.assert_close(actual, expected.masked_fill(~mask.unsqueeze(-1), 0))
        torch.testing.assert_close(aux["gates"], weights)
        self.assertNotIn("alpha", aux)
        self.assertFalse(any("temperature" in key for key in sequence.state_dict()))
        changed = {m: value.masked_fill(~mask.unsqueeze(-1), 1e4) for m, value in streams.items()}
        repeated, repeated_aux = sequence(changed, mask)
        torch.testing.assert_close(repeated, actual)
        torch.testing.assert_close(repeated_aux["gates"], weights)
        # Constant inputs give identical sequence and positional affine gates.
        constant = {m: value[:, :1].expand_as(value) for m, value in streams.items()}
        torch.testing.assert_close(sequence(constant, mask)[0], position(constant, mask)[0])

    def test_joint_sequence_gate_uses_other_modalities_and_all_active_parameters(self):
        head = models.JointSequenceGateFusion(WIDTHS, projection_width=4, scorer_width=3,
            projection_activation="linear", projection_norm="none")
        streams, mask = aligned_inputs()
        with torch.no_grad():
            for gate in head.gate.values():
                gate.weight.fill_(.1)
                gate.bias.fill_(.2)
            head.project["prot"][0].weight.fill_(.1)
        original = head(streams, mask)[1]["gates"]
        changed = {**streams, "prot": streams["prot"] + 1}
        actual, aux = head(changed, mask)
        self.assertFalse(torch.equal(original[:, 0], aux["gates"][:, 0]))
        actual.square().sum().backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_attention_control_budget_and_active_mixing_counts(self):
        budget = models.attention_control_budget(600, 3)
        self.assertEqual(budget, {"attention_mixing": 2888402, "hidden_width": 1203,
                                  "control_mixing": 2889003})
        gate = models.JointSequenceGateFusion(WIDTHS, projection_width=600, scorer_width=100,
            projection_activation="tanh", projection_norm="none")
        self.assertEqual(models.parameter_count(gate.gate), 5403)
        control = models.ResidualConcatFusion(WIDTHS, anchor="rna", projection_width=600,
            hidden_width=1203, projection_activation="tanh", projection_norm="none")
        self.assertEqual(models.parameter_count(control.mix), 2889003)
        attention = attention_head("position_sigmoid", projection_width=600, n_heads=4)
        mixing = models.parameter_count(attention) - models.parameter_count(attention.project)
        self.assertEqual(mixing, 2888402)
        self.assertLess(abs(mixing - models.parameter_count(control.mix)) / mixing, .0003)
        with self.assertRaisesRegex(ValueError, "hidden_width"):
            models.ResidualConcatFusion(WIDTHS, anchor="rna", projection_width=600,
                hidden_width=1202, projection_activation="tanh", projection_norm="none")

    def test_residual_concat_equation_padding_gradients_and_zero_update(self):
        head = models.ResidualConcatFusion(WIDTHS, anchor="rna", projection_width=4,
            hidden_width=11, projection_activation="tanh", projection_norm="none")
        streams, mask = aligned_inputs()
        actual, aux = head(streams, mask)
        projected = {m: head.project[m](streams[m]) for m in WIDTHS}
        first, _, second = head.mix
        hidden = torch.nn.functional.gelu(torch.cat(list(projected.values()), -1) @ first.weight.T + first.bias)
        expected = projected["rna"] + hidden @ second.weight.T + second.bias
        torch.testing.assert_close(actual, expected.masked_fill(~mask.unsqueeze(-1), 0))
        self.assertEqual(aux, {})
        actual.square().sum().backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        changed = {m: value.masked_fill(~mask.unsqueeze(-1), -1e4) for m, value in streams.items()}
        torch.testing.assert_close(head(changed, mask)[0], actual)
        with torch.no_grad():
            second.weight.zero_()
            second.bias.zero_()
        torch.testing.assert_close(head(streams, mask)[0], projected["rna"].masked_fill(~mask.unsqueeze(-1), 0))

    def test_mean_softmax_exactly_matches_legacy_with_transferred_state(self):
        streams, mask = aligned_inputs()
        for shared in (False, True):
            with self.subTest(shared=shared):
                legacy = GatedMILFusion(WIDTHS, out_width=4, attention_width=3,
                                        shared_projection=shared)
                head = sequence_head(shared_scorer=shared)
                head.load_state_dict(legacy.state_dict(), strict=True)
                expected, old_aux = legacy(streams, mask)
                actual, new_aux = head(streams, mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(new_aux["alpha"], old_aux["alpha"], rtol=0, atol=0)
                self.assertEqual(models.parameter_count(head), models.parameter_count(legacy))

    def test_seed_controls_initialization_without_global_seed_choice_in_models(self):
        torch.manual_seed(17)
        first = sequence_head()
        torch.manual_seed(17)
        second = sequence_head()
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
        torch.manual_seed(18)
        changed = sequence_head()
        self.assertFalse(torch.equal(first.project["rna"][0].weight,
                                     changed.project["rna"][0].weight))

    def test_sigmoid_never_exposes_entropy_alpha_or_renormalizes(self):
        streams, mask = aligned_inputs()
        head = sequence_head(normalization="sigmoid", temperature_range=None)
        with torch.no_grad():
            head.score.weight.zero_()
        actual, aux = head(streams, mask)
        self.assertNotIn("alpha", aux)
        self.assertNotIn("log_temperature", dict(head.named_parameters()))
        torch.testing.assert_close(aux["gates"], torch.full((2, 3), .5))
        expected = sum(.5 * head.project[m](streams[m]) for m in WIDTHS)
        torch.testing.assert_close(actual, expected.masked_fill(~mask.unsqueeze(-1), 0))

    def test_mean_max_summary_and_padding_invariance_for_two_modalities(self):
        widths = {"rna": 4, "prot": 2}
        head = models.SequenceGateFusion(widths, projection_width=4, scorer_width=3,
            summary="mean_max", normalization="sigmoid", shared_scorer=False,
            temperature_range=None, projection_activation="tanh", projection_norm="layer")
        streams, mask = aligned_inputs()
        streams = {m: streams[m] for m in widths}
        actual, aux = head(streams, mask)
        padded = {m: torch.cat((value, torch.full((2, 4, value.shape[-1]), 1000.)), dim=1)
                  for m, value in streams.items()}
        padded_mask = torch.cat((mask, torch.zeros((2, 4), dtype=torch.bool)), dim=1)
        repeated, padded_aux = head(padded, padded_mask)
        torch.testing.assert_close(actual, repeated[:, :3])
        torch.testing.assert_close(aux["gates"], padded_aux["gates"])
        self.assertEqual(head.gate_v["rna"].in_features, 8)

    def test_two_modality_softmax_does_not_use_legacy_three_modality_alpha(self):
        head = models.SequenceGateFusion({"rna": 4, "prot": 2}, projection_width=4,
            scorer_width=3, summary="mean", normalization="softmax", shared_scorer=False,
            temperature_range=TEMPERATURE_RANGE, projection_activation="tanh", projection_norm="none")
        streams, mask = aligned_inputs()
        _, aux = head({m: streams[m] for m in ("rna", "prot")}, mask)
        self.assertNotIn("alpha", aux)
        torch.testing.assert_close(aux["gates"].sum(-1), torch.ones(2))

    def test_position_gates_vary_and_padding_gradients_are_zero(self):
        widths = {"rna": 1, "prot": 1}
        head = models.PositionGateFusion(widths, projection_width=1, scorer_width=1,
                                        projection_activation="linear", projection_norm="none")
        with torch.no_grad():
            for m in widths:
                head.project[m][0].weight.fill_(1)
                head.project[m][0].bias.zero_()
                head.gate[m].weight.fill_(1)
                head.gate[m].bias.zero_()
        streams = {m: torch.tensor([[[-2.], [2.], [99.]]], requires_grad=True) for m in widths}
        mask = torch.tensor([[True, True, False]])
        output, aux = head(streams, mask)
        self.assertLess(aux["gates"][0, 0, 0], aux["gates"][0, 1, 0])
        self.assertNotIn("alpha", aux)
        torch.testing.assert_close(aux["gates"][0, 2], torch.zeros(2))
        output.sum().backward()
        for stream in streams.values():
            torch.testing.assert_close(stream.grad[0, 2], torch.zeros(1))
            self.assertGreater(stream.grad[0, :2].abs().sum().item(), 0)

    def test_low_rank_control_count_is_reported_without_claiming_capacity_match(self):
        head = models.PositionalControl(WIDTHS, projection_width=4, bottleneck_width=2,
                                       projection_activation="linear", projection_norm="none")
        # Projection affine maps: (3+4+2)*4 + 3*4 = 48; mixer: 12*2+2 + 2*4+4 = 38.
        self.assertEqual(models.parameter_count(head), 86)
        self.assertNotEqual(models.parameter_count(head), models.parameter_count(sequence_head()))
        streams, mask = aligned_inputs()
        output, aux = head(streams, mask)
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(aux, {})
        torch.testing.assert_close(output[0, 2], torch.zeros(4))

    def test_retained_control_is_no_longer_parameter_matched_after_joint_linear_correction(self):
        for rna_width in (640, 1280):
            widths = {"dna": 512, "rna": rna_width, "prot": 320}
            gate = models.PositionGateFusion(widths, projection_width=600, scorer_width=100,
                                            projection_activation="tanh", projection_norm="none")
            control = models.PositionalControl(widths, projection_width=600, bottleneck_width=150,
                                                projection_activation="tanh", projection_norm="none")
            # Joint gate: 3*(1800+1)=5403. Existing rank150 mixer: 360750.
            self.assertEqual(models.parameter_count(control) - models.parameter_count(gate), 355347)

    def test_residual_control_starts_as_half_sum_with_a_full_width_path(self):
        widths = {"rna": 4, "prot": 4}
        head = models.ResidualPositionalControl(widths, projection_width=4, bottleneck_width=2,
            fixed_scale=.5, update_initialization="zero_output_random_input",
            projection_activation="linear", projection_norm="none")
        with torch.no_grad():
            for projection in head.project.values():
                projection[0].weight.copy_(torch.eye(4))
                projection[0].bias.zero_()
        streams = {"rna": torch.eye(4)[None], "prot": torch.zeros(1, 4, 4)}
        output, aux = head(streams, torch.ones(1, 4, dtype=torch.bool))
        torch.testing.assert_close(output, .5 * torch.eye(4)[None], rtol=0, atol=0)
        self.assertEqual(torch.linalg.matrix_rank(output[0]).item(), 4)
        self.assertGreater(torch.count_nonzero(head.update[0].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(head.update[1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(head.update[1].bias).item(), 0)
        self.assertEqual(aux, {})

    def test_residual_control_affine_update_matches_hand_calculation(self):
        head = models.ResidualPositionalControl({"rna": 1, "prot": 1}, projection_width=1,
            bottleneck_width=2, fixed_scale=.5, update_initialization="zero_output_random_input",
            projection_activation="linear", projection_norm="none")
        with torch.no_grad():
            for projection in head.project.values():
                projection[0].weight.fill_(1)
                projection[0].bias.zero_()
            head.update[0].weight.copy_(torch.tensor([[1., 2.], [-1., 3.]]))
            head.update[0].bias.copy_(torch.tensor([.5, -.5]))
            head.update[1].weight.copy_(torch.tensor([[2., -1.]]))
            head.update[1].bias.fill_(.25)
        # x=1,y=2: fixed=1.5; hidden=(5.5,4.5); update=11-4.5+.25=6.75.
        streams = {"rna": torch.tensor([[[1.], [9.]]]), "prot": torch.tensor([[[2.], [99.]]])}
        result, _ = head(streams, torch.tensor([[True, False]]))
        torch.testing.assert_close(result, torch.tensor([[[8.25], [0.]]]), rtol=0, atol=0)

    def test_residual_control_gradients_reach_output_factor_and_fixed_path_not_padding(self):
        head = models.ResidualPositionalControl(WIDTHS, projection_width=4, bottleneck_width=2,
            fixed_scale=.5, update_initialization="zero_output_random_input",
            projection_activation="linear", projection_norm="none")
        streams, mask = aligned_inputs()
        streams = {name: value.requires_grad_() for name, value in streams.items()}
        head(streams, mask)[0].sum().backward()
        self.assertGreater(head.update[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.update[1].bias.grad.abs().sum().item(), 0)
        torch.testing.assert_close(head.update[0].weight.grad, torch.zeros_like(head.update[0].weight))
        for stream in streams.values():
            torch.testing.assert_close(stream.grad[0, 2], torch.zeros_like(stream.grad[0, 2]))
            self.assertGreater(stream.grad[0, :2].abs().sum().item(), 0)
        # Once the output factor moves, gradients can reach the random input factor.
        head.zero_grad()
        with torch.no_grad():
            head.update[1].weight.fill_(.25)
        head(streams, mask)[0].sum().backward()
        self.assertGreater(head.update[0].weight.grad.abs().sum().item(), 0)

    def test_residual_control_count_and_declared_initialization(self):
        for rna_width in (640, 1280):
            widths = {"dna": 512, "rna": rna_width, "prot": 320}
            head = models.ResidualPositionalControl(widths, projection_width=600, bottleneck_width=2,
                fixed_scale=.5, update_initialization="zero_output_random_input",
                projection_activation="tanh", projection_norm="none")
            gate = models.PositionGateFusion(widths, projection_width=600, scorer_width=100,
                projection_activation="tanh", projection_norm="none")
            self.assertEqual(models.parameter_count(head.update), 5402)
            self.assertEqual(models.parameter_count(head), models.parameter_count(gate) - 1)
        for scale, initialization in ((1., "zero_output_random_input"), (.5, "both_zero")):
            with self.assertRaises(ValueError):
                models.ResidualPositionalControl(WIDTHS, projection_width=4, bottleneck_width=2,
                    fixed_scale=scale, update_initialization=initialization,
                    projection_activation="linear", projection_norm="none")

    def test_joint_position_formula_and_other_modality_dependency(self):
        head = models.PositionGateFusion({"rna": 1, "prot": 1}, projection_width=1,
            scorer_width=99, projection_activation="linear", projection_norm="none")
        with torch.no_grad():
            for projection in head.project.values():
                projection[0].weight.fill_(1)
                projection[0].bias.zero_()
            head.gate["rna"].weight.copy_(torch.tensor([[0., 2.]]))
            head.gate["rna"].bias.fill_(-1.)
            head.gate["prot"].weight.copy_(torch.tensor([[-1., 0.]]))
            head.gate["prot"].bias.fill_(.5)
        streams = {"rna": torch.tensor([[[1.], [2.]]]), "prot": torch.tensor([[[3.], [5.]]])}
        mask = torch.ones((1, 2), dtype=torch.bool)
        output, aux = head(streams, mask)
        expected = torch.sigmoid(torch.tensor([[[5., -.5], [9., -1.5]]]))
        torch.testing.assert_close(aux["gates"], expected)
        torch.testing.assert_close(output.squeeze(-1),
            expected[..., 0] * torch.tensor([[1., 2.]]) + expected[..., 1] * torch.tensor([[3., 5.]]))
        _, changed = head({**streams, "prot": streams["prot"] + 1}, mask)
        self.assertTrue((changed["gates"][..., 0] > aux["gates"][..., 0]).all())
        torch.testing.assert_close(changed["gates"][..., 1], aux["gates"][..., 1])

    def test_attention_joint_formula_allows_one_auxiliary_to_change_another_gate(self):
        for mode in ("sequence_sigmoid", "position_sigmoid"):
            head = models.AnchorAttentionFusion({"dna": 1, "rna": 1, "prot": 1}, anchor="rna",
                projection_width=1, scorer_width=99, n_heads=1, dropout=0., mode=mode,
                projection_activation="linear", projection_norm="none", update_norm="none")
            with torch.no_grad():
                for projection in head.project.values():
                    projection[0].weight.fill_(1)
                    projection[0].bias.zero_()
                for attention in head.attend.values():
                    attention.in_proj_weight.copy_(torch.tensor([[0.], [0.], [1.]]))
                    attention.in_proj_bias.zero_()
                    attention.out_proj.weight.fill_(1)
                    attention.out_proj.bias.zero_()
                head.gate["dna"].weight.copy_(torch.tensor([[.1, 0., .5]]))
                head.gate["dna"].bias.zero_()
                head.gate["prot"].weight.copy_(torch.tensor([[0., .2, 0.]]))
                head.gate["prot"].bias.zero_()
            streams = {"rna": torch.tensor([[[1.], [2.]]]), "dna": torch.tensor([[[3.]]]),
                       "prot": torch.tensor([[[4.]]])}
            masks = {m: torch.ones(value.shape[:2], dtype=torch.bool) for m, value in streams.items()}
            output, aux = head(streams, masks)
            logits = (torch.tensor([[2.15, .6]]) if mode == "sequence_sigmoid"
                      else torch.tensor([[[2.1, .6], [2.2, .6]]]))
            gates = torch.sigmoid(logits)
            torch.testing.assert_close(aux["gates"], gates)
            effective = gates[:, None] if gates.ndim == 2 else gates
            expected = torch.tensor([[1., 2.]]) + 3 * effective[..., 0] + 4 * effective[..., 1]
            torch.testing.assert_close(output.squeeze(-1), expected)
            _, changed = head({**streams, "prot": streams["prot"] + 1}, masks)
            self.assertTrue((changed["gates"][..., 0] > gates[..., 0]).all())
            torch.testing.assert_close(changed["gates"][..., 1], gates[..., 1])

    def test_norm_diagnostics_are_opt_in_detached_and_preserve_predictions(self):
        streams, masks = attention_inputs()
        head = attention_head("position_sigmoid")
        first, aux = head(streams, masks)
        self.assertNotIn("raw_update_norms", aux)
        head.collect_norms = True
        second, aux = head(streams, masks)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for key in ("anchor_norm", "raw_update_norms", "update_norms", "fused_norm"):
            self.assertFalse(aux[key].requires_grad)
            self.assertFalse(aux[key][~masks["rna"]].any())
        torch.testing.assert_close(aux["update_norms"], aux["raw_update_norms"] * aux["update_weights"])
        self.assertFalse(aux["update_norms"][1, :, 1].any())
        self.assertEqual(aux["update_norms"].shape, (2, 3, 2))

    def test_gate_norms_preserve_sequence_and_position_weights_without_entropy_changes(self):
        streams, mask = aligned_inputs()
        position = models.PositionGateFusion(WIDTHS, projection_width=4, scorer_width=3,
                                            projection_activation="tanh", projection_norm="none")
        for head in (sequence_head(), position):
            first, initial = head(streams, mask)
            self.assertNotIn("feature_norms", initial)
            head.collect_norms = True
            second, aux = head(streams, mask)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            weights = aux["gates"][:, None] if aux["gates"].ndim == 2 else aux["gates"]
            torch.testing.assert_close(aux["contribution_norms"], aux["feature_norms"] * weights)
            self.assertEqual(aux["feature_norms"].shape, (2, 3, 3))
            for key in ("feature_norms", "contribution_norms", "fused_norm"):
                self.assertFalse(aux[key].requires_grad)
                self.assertFalse(aux[key][~mask].any())
            self.assertEqual("alpha" in aux, "alpha" in initial)
            self.assertEqual("temperature" in aux, isinstance(head, models.SequenceGateFusion))

    def test_attention_all_modes_support_unequal_lengths_and_missing_auxiliary_rows(self):
        streams, masks = attention_inputs()
        for mode in ("ordinary", "skip", "sequence_sigmoid", "position_sigmoid"):
            with self.subTest(mode=mode):
                output, aux = attention_head(mode)(streams, masks)
                self.assertEqual(output.shape, (2, 3, 4))
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output[0, 2], torch.zeros(4))
                torch.testing.assert_close(aux["update_weights"][1, :, 1], torch.zeros(3))
                self.assertNotIn("alpha", aux)
                if mode.endswith("sigmoid"):
                    expected = (2, 2) if mode == "sequence_sigmoid" else (2, 3, 2)
                    self.assertEqual(aux["gates"].shape, expected)

    def test_skip_and_ordinary_differ_exactly_by_projected_anchor(self):
        streams, masks = attention_inputs()
        ordinary, skip = attention_head("ordinary"), attention_head("skip")
        skip.load_state_dict(ordinary.state_dict(), strict=True)
        first, _ = ordinary(streams, masks)
        second, _ = skip(streams, masks)
        anchor = skip.project["rna"](streams["rna"]).masked_fill(~masks["rna"].unsqueeze(-1), 0)
        torch.testing.assert_close(second, first + anchor)

    def test_padding_changes_do_not_change_attention_real_positions(self):
        streams, masks = attention_inputs()
        head = attention_head("position_sigmoid", projection_norm="layer", update_norm="layer")
        output, _ = head(streams, masks)
        padded = {m: torch.cat((value, torch.full((2, 2, value.shape[-1]), 900.)), dim=1)
                  for m, value in streams.items()}
        padded_masks = {m: torch.cat((mask, torch.zeros((2, 2), dtype=torch.bool)), dim=1)
                        for m, mask in masks.items()}
        repeated, _ = head(padded, padded_masks)
        torch.testing.assert_close(output, repeated[:, :3], rtol=1e-5, atol=1e-6)

    def test_zero_availability_blocks_biases_and_recovers_anchor_exactly(self):
        streams, masks = attention_inputs()
        availability = {m: torch.zeros_like(masks["rna"]) for m in ("dna", "prot")}
        for mode in ("skip", "sequence_sigmoid", "position_sigmoid"):
            head = attention_head(mode, projection_norm="layer", update_norm="layer")
            with torch.no_grad():
                for m in head.names:
                    head.attend[m].out_proj.bias.fill_(10)
                    head.update_norm[m].bias.fill_(20)
            output, aux = head(streams, masks, availability)
            expected = head.project["rna"](streams["rna"]).masked_fill(~masks["rna"].unsqueeze(-1), 0)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(aux["update_weights"], torch.zeros((2, 3, 2)))

    def test_zero_length_auxiliaries_bypass_attention_without_nan(self):
        streams, masks = attention_inputs()
        for m in ("dna", "prot"):
            streams[m] = streams[m][:, :0]
            masks[m] = masks[m][:, :0]
        head = attention_head("skip")
        output, _ = head(streams, masks)
        expected = head.project["rna"](streams["rna"]).masked_fill(~masks["rna"].unsqueeze(-1), 0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_partial_availability_blocks_updates_after_attention_and_norm_biases(self):
        streams, masks = attention_inputs()
        availability = {"dna": torch.zeros_like(masks["rna"]),
                        "prot": torch.tensor([[True, False, False], [False, False, False]])}
        head = attention_head("skip", update_norm="layer")
        with torch.no_grad():
            head.attend["prot"].out_proj.bias.fill_(10)
            head.update_norm["prot"].bias.fill_(20)
        output, aux = head(streams, masks, availability)
        anchor = head.project["rna"](streams["rna"]).masked_fill(~masks["rna"].unsqueeze(-1), 0)
        torch.testing.assert_close(output[0, 1:], anchor[0, 1:], rtol=0, atol=0)
        torch.testing.assert_close(output[1], anchor[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(output[0, 0], anchor[0, 0]))
        torch.testing.assert_close(aux["update_weights"][..., 1], availability["prot"].float())

    def test_zero_learned_gates_recover_anchor_without_residual_normalization(self):
        streams, masks = attention_inputs()
        for mode in ("sequence_sigmoid", "position_sigmoid"):
            head = attention_head(mode, update_norm="layer")
            with torch.no_grad():
                for gate in head.gate.values():
                    gate.weight.zero_()
                    gate.bias.fill_(-1000)
            output, _ = head(streams, masks)
            anchor = head.project["rna"](streams["rna"]).masked_fill(~masks["rna"].unsqueeze(-1), 0)
            torch.testing.assert_close(output, anchor, rtol=0, atol=0)

    def test_unavailable_protein_and_padding_have_zero_gradients(self):
        streams, masks = attention_inputs()
        streams = {m: tensor.requires_grad_() for m, tensor in streams.items()}
        availability = {"dna": masks["rna"].clone(), "prot": torch.zeros_like(masks["rna"])}
        head = attention_head("position_sigmoid")
        output, _ = head(streams, masks, availability)
        output.square().sum().backward()
        protein_gradient = streams["prot"].grad
        self.assertTrue(protein_gradient is None or not protein_gradient.any())
        for m in ("dna", "rna"):
            self.assertIsNotNone(streams[m].grad)
            torch.testing.assert_close(streams[m].grad[~masks[m]], torch.zeros_like(streams[m].grad[~masks[m]]))
        self.assertGreater(streams["dna"].grad[masks["dna"]].abs().sum().item(), 0)

    def test_invalid_architecture_and_input_contracts_are_rejected(self):
        with self.assertRaises(ValueError):
            sequence_head(normalization="sigmoid")
        with self.assertRaises(ValueError):
            attention_head("skip", projection_width=3, n_heads=2)
        with self.assertRaises(ValueError):
            attention_head("unknown")
        streams, masks = attention_inputs()
        head = attention_head("skip")
        with self.assertRaises(ValueError):
            head(streams, masks, {"prot": masks["rna"]})
        with self.assertRaises(ValueError):
            head(streams, masks, {m: masks["rna"].float() for m in ("dna", "prot")})
        masks["rna"] = torch.zeros_like(masks["rna"])
        with self.assertRaises(ValueError):
            head(streams, masks)


if __name__ == "__main__":
    unittest.main()
