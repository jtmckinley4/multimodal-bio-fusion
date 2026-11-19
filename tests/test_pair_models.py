"""Check the pair scorers on small tensors: no checkpoints, no cached embeddings.

What is asserted is what the comparison between arms depends on: every arm returns one logit
per pair and trains; padding a batch cannot change a row's score; the single-partner baselines
really do ignore their partner while the fusion arms do not; a gated cross-attention arm
returns the gate it applied, per sequence or per token; and the interleaved Mamba carries state
across the boundary between the two molecules, in the order it was given.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import pair_models as PM  # noqa: E402

WIDTHS = {"left": 7, "right": 5}
ARMS = [("left_only", {}), ("right_only", {}), ("concat", {}), ("bilinear", {}),
        ("cross_attention", {"direction": "both"}),
        ("cross_attention", {"direction": "left", "gate": "sequence"}),
        ("cross_attention", {"direction": "left", "gate": "position"}),
        ("interleaved_mamba", {"bidirectional": False}),
        ("interleaved_mamba", {"bidirectional": True})]
SMALL = {"hidden": 16}
MAMBA = {"d_model": 8, "d_state": 4, "layers": 1, "hidden": 16, "chunk": 3}


def options(name, extra):
    return {**(MAMBA if name == "interleaved_mamba" else SMALL), **extra}


def batch(rows=3, left_len=6, right_len=4, seed=0, lengths=None):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(rows, left_len, WIDTHS["left"], generator=generator)
    right = torch.randn(rows, right_len, WIDTHS["right"], generator=generator)
    left_mask = torch.ones(rows, left_len, dtype=torch.bool)
    right_mask = torch.ones(rows, right_len, dtype=torch.bool)
    if lengths is not None:
        for row, (a, b) in enumerate(lengths):
            left_mask[row, a:] = False
            right_mask[row, b:] = False
            left[row, a:] = 0.0
            right[row, b:] = 0.0
    return left, right, left_mask, right_mask


class PairScorerTests(unittest.TestCase):
    def build(self, name, extra):
        torch.manual_seed(0)
        return PM.build(name, WIDTHS, **options(name, extra))

    def test_every_arm_returns_one_finite_logit_per_pair(self):
        for name, extra in ARMS:
            with self.subTest(arm=name, **extra):
                model = self.build(name, extra).eval()
                with torch.no_grad():
                    logit, _ = model(*batch())
                self.assertEqual(tuple(logit.shape), (3,))
                self.assertTrue(bool(torch.isfinite(logit).all()))

    def test_every_arm_takes_a_training_step(self):
        labels = torch.tensor([1.0, 0.0, 1.0])
        for name, extra in ARMS:
            with self.subTest(arm=name, **extra):
                model = self.build(name, extra).train()
                logit, _ = model(*batch())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, labels)
                loss.backward()
                trained = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(trained)
                self.assertTrue(bool(torch.isfinite(loss)))

    def test_padding_a_batch_cannot_change_a_row(self):
        """A short row scored beside a long one must equal the same row scored alone."""
        lengths = [(2, 2), (6, 4), (4, 3)]
        padded = batch(lengths=lengths, seed=1)
        for name, extra in ARMS:
            with self.subTest(arm=name, **extra):
                model = self.build(name, extra).eval()
                with torch.no_grad():
                    together, _ = model(*padded)
                    a, b = lengths[0]
                    alone, _ = model(padded[0][:1, :a], padded[1][:1, :b],
                                     padded[2][:1, :a], padded[3][:1, :b])
                torch.testing.assert_close(together[:1], alone, atol=2e-5, rtol=2e-5,
                                           msg=f"{name} row changed when batched")

    def test_the_baselines_ignore_their_partner_and_the_fusion_arms_do_not(self):
        left, right, left_mask, right_mask = batch(seed=2)
        other = torch.randn_like(right)
        for name, extra in ARMS:
            with self.subTest(arm=name, **extra):
                model = self.build(name, extra).eval()
                with torch.no_grad():
                    first, _ = model(left, right, left_mask, right_mask)
                    second, _ = model(left, other, left_mask, right_mask)
                changed = bool((first - second).abs().max() > 1e-6)
                self.assertEqual(changed, name != "left_only",
                                 f"{name} responded to the right partner: {changed}")

    def test_a_gated_arm_returns_the_gate_it_applied(self):
        left, right, left_mask, right_mask = batch(lengths=[(2, 2), (6, 4), (4, 3)])
        sequence = self.build("cross_attention", {"direction": "left", "gate": "sequence"}).eval()
        position = self.build("cross_attention", {"direction": "left", "gate": "position"}).eval()
        with torch.no_grad():
            _, from_sequence = sequence(left, right, left_mask, right_mask)
            _, from_position = position(left, right, left_mask, right_mask)
        self.assertEqual(tuple(from_sequence["left_gate"].shape), (3, 1))
        self.assertEqual(tuple(from_position["left_gate"].shape), (3, left.shape[1]))
        for gate in (from_sequence["left_gate"], from_position["left_gate"]):
            self.assertTrue(bool(((gate >= 0) & (gate <= 1)).all()))

    def test_a_position_gate_does_not_compete_across_tokens(self):
        """Independent sigmoids: the gates of one row need not sum to anything."""
        model = self.build("cross_attention", {"direction": "left", "gate": "position"}).eval()
        with torch.no_grad():
            _, extras = model(*batch(seed=3))
        total = extras["left_gate"].sum(dim=1)
        self.assertTrue(bool((total > 0).all()))
        self.assertFalse(bool(torch.allclose(total, torch.ones_like(total))))

    def test_the_interleaved_mamba_carries_state_across_the_boundary(self):
        """A change in the molecule read first must reach the read-out of the second."""
        model = self.build("interleaved_mamba", {"bidirectional": False}).eval()
        left, right, left_mask, right_mask = batch(rows=1, seed=4)
        changed = left.clone()
        changed[0, 0] += 3.0
        with torch.no_grad():
            joined, mask = model.join(left, right, left_mask, right_mask)
            other, _ = model.join(changed, right, left_mask, right_mask)
            first = model._scan(model.layers[0][0], joined, mask, reverse=False)
            second = model._scan(model.layers[0][0], other, mask, reverse=False)
        tail = left.shape[1]  # positions from here on belong to the right molecule
        self.assertGreater(float((first[:, tail:] - second[:, tail:]).abs().max()), 1e-6)

    def test_the_order_of_the_two_molecules_changes_a_one_way_scan(self):
        left, right, left_mask, right_mask = batch(seed=5)
        forward = self.build("interleaved_mamba", {"bidirectional": False,
                                                   "order": "left_right"}).eval()
        reverse = self.build("interleaved_mamba", {"bidirectional": False,
                                                   "order": "right_left"}).eval()
        reverse.load_state_dict(forward.state_dict())
        with torch.no_grad():
            a, _ = forward(left, right, left_mask, right_mask)
            b, _ = reverse(left, right, left_mask, right_mask)
        self.assertGreater(float((a - b).abs().max()), 1e-6)

    def test_the_joined_sequence_holds_both_molecules_without_padding_between_them(self):
        left, right, left_mask, right_mask = batch(lengths=[(2, 2), (6, 4), (4, 3)], seed=6)
        model = self.build("interleaved_mamba", {}).eval()
        with torch.no_grad():
            joined, mask = model.join(left, right, left_mask, right_mask)
        for row, (a, b) in enumerate([(2, 2), (6, 4), (4, 3)]):
            self.assertEqual(int(mask[row].sum()), a + b)
            self.assertTrue(bool(mask[row, :a + b].all()))
            self.assertFalse(bool(mask[row, a + b:].any()))
        self.assertEqual(joined.shape[1], mask.shape[1])

    def test_reversing_a_row_keeps_its_padding_at_the_end(self):
        """The reverse direction must not move padding in front of a row's real tokens.

        The causal convolution reads the positions before each token, so padding placed first
        would make a row's result depend on the longest row in its batch.
        """
        model = self.build("interleaved_mamba", {}).eval()
        left, right, left_mask, right_mask = batch(lengths=[(2, 2), (6, 4), (4, 3)], seed=7)
        joined, mask = model.join(left, right, left_mask, right_mask)
        flipped, _ = model._reversed_rows(joined, mask)
        twice, _ = model._reversed_rows(flipped, mask)
        torch.testing.assert_close(twice, joined)
        for row, length in enumerate(mask.sum(dim=1).tolist()):
            torch.testing.assert_close(flipped[row, :length], joined[row, :length].flip(0))
            self.assertEqual(float(flipped[row, length:].detach().abs().sum()), 0.0)

    def test_parameter_counts_separate_the_fusion_path_from_the_scorer(self):
        counts = {}
        for name in ("concat", "bilinear", "cross_attention"):
            model = self.build(name, {})
            counts[name] = PM.parameter_counts(model)
            self.assertEqual(counts[name]["total"],
                             counts[name]["fusion"] + counts[name]["scorer"])
        self.assertGreater(counts["cross_attention"]["fusion"], counts["concat"]["fusion"])

    def test_an_unknown_arm_or_a_missing_width_is_refused(self):
        with self.assertRaises(ValueError):
            PM.build("gated_mil", WIDTHS)
        with self.assertRaises(ValueError):
            PM.build("concat", {"left": 7})


if __name__ == "__main__":
    unittest.main()
