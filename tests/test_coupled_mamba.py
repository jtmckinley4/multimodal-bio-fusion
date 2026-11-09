"""Check the Coupled Mamba fusion head on small tensors: no checkpoints, no cached embeddings.

The properties asserted are the ones the comparison depends on: coupling carries one
modality's input into another modality's later codons and never into earlier ones; the
uncoupled ablation never does; padding cannot reach a real codon; a full-length row stays
finite; and the coupled head and its ablation have the same parameters, so a difference
between them is the coupling and not the size.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_code.fusion import blf, coupled_mamba as CM, heads, train  # noqa: E402
import test_fusion_model  # noqa: E402  (its stub cache, built with the real shard writer)

WIDTHS = {"dna": 12, "rna": 9, "prot": 6}
DESIGNS = ("coupled_mamba", "uncoupled_mamba", "mamba_bottleneck")


def random_streams(batch, codons, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {m: torch.randn(batch, codons, w, generator=generator) for m, w in WIDTHS.items()}


def full_mask(batch, codons):
    return torch.ones(batch, codons, dtype=torch.bool)


class RegistrationTests(unittest.TestCase):
    def test_the_constants_repeated_here_match_the_other_heads(self):
        self.assertEqual(CM.MODALITIES, heads.MODALITIES)
        self.assertEqual(CM.SHARED_WIDTH, heads.SHARED_WIDTH)

    def test_all_three_designs_are_reachable_by_name(self):
        for name in DESIGNS:
            self.assertIs(heads.FUSIONS[name], CM.DESIGNS[name])
        self.assertEqual(set(heads.FUSIONS) - set(DESIGNS), {"concat", "mil", "cross"})

    def test_the_pipeline_builds_each_design_by_name(self):
        for name in DESIGNS:
            model = blf.BioLangFusion(WIDTHS, name, n_outputs=1, channels=3)
            self.assertIsInstance(model.fusion, CM.DESIGNS[name])


class ScanTests(unittest.TestCase):
    def test_the_state_before_each_step_follows_the_recurrence(self):
        # Constant decay a and drive 1: h_t = (1 - a^(t+1)) / (1 - a), and position t holds h_{t-1}.
        a = 0.5
        decay = torch.full((1, 6, 1, 1), a)
        drive = torch.ones(1, 6, 1, 1)
        before, last = CM.previous_states(decay, drive)
        expected = [0.0] + [(1 - a ** (t + 1)) / (1 - a) for t in range(6)]
        torch.testing.assert_close(before.flatten(), torch.tensor(expected[:6]))
        torch.testing.assert_close(last.flatten(), torch.tensor(expected[6:]))

    def test_a_scan_split_in_two_matches_one_scan(self):
        generator = torch.Generator().manual_seed(0)
        decay = torch.rand(2, 10, 3, 4, generator=generator)
        drive = torch.randn(2, 10, 3, 4, generator=generator)
        whole, _ = CM.previous_states(decay, drive)
        first, carried = CM.previous_states(decay[:, :4], drive[:, :4])
        second, _ = CM.previous_states(decay[:, 4:], drive[:, 4:], carried)
        torch.testing.assert_close(torch.cat([first, second], dim=1), whole)

    def test_the_pooled_decay_stays_below_one_where_a_sum_would_not(self):
        # Mamba initializes many decays near 1. Summing three such states, as Eq. 6 writes it,
        # multiplies the state by nearly 3 per codon; averaging keeps it below 1.
        torch.manual_seed(0)
        block = CM.MambaBlock(16, 32, 8, 4, 1)
        x = torch.randn(2, 50, 16)
        with torch.no_grad():
            delta = block.prepare(x, full_mask(2, 50))[2]
            decay = torch.exp(delta.unsqueeze(-1) * block.transition)
        self.assertGreater(float(decay.max()), 0.99)
        self.assertLess(float(((decay + decay + decay) / 3).max()), 1.0)
        self.assertGreater(float((decay + decay + decay).max()), 2.9)

    def test_the_chunk_size_never_changes_the_output(self):
        for name in ("coupled_mamba", "uncoupled_mamba"):
            torch.manual_seed(0)
            model = CM.DESIGNS[name](WIDTHS).eval()
            streams, mask = random_streams(2, 45), full_mask(2, 45)
            with torch.no_grad():
                one_chunk = model.modality_streams(streams, mask, chunk=1000)
                for chunk in (1, 7, CM.CHUNK):
                    split = model.modality_streams(streams, mask, chunk=chunk)
                    for m in WIDTHS:
                        torch.testing.assert_close(split[m], one_chunk[m], msg=f"{name} {chunk}")

    def test_recomputing_chunks_in_the_backward_pass_gives_the_stored_gradients(self):
        for name in ("coupled_mamba", "uncoupled_mamba"):
            torch.manual_seed(0)
            model = CM.DESIGNS[name](WIDTHS)
            streams, mask = random_streams(2, 45), full_mask(2, 45)
            grads = []
            for training in (True, False):        # train mode recomputes; eval mode stores
                model.train(training)
                model.zero_grad()
                model(streams, mask)[0].pow(2).mean().backward()
                grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
            for n in grads[0]:
                torch.testing.assert_close(grads[0][n], grads[1][n], msg=f"{name} {n}")


class CouplingTests(unittest.TestCase):
    """Perturb the DNA input at one codon and watch the RNA stream before the final fusion."""

    CODONS, AT = 20, 8

    def rna_change(self, name):
        torch.manual_seed(0)
        model = CM.DESIGNS[name](WIDTHS).eval()
        streams = random_streams(1, self.CODONS)
        moved = {m: s.clone() for m, s in streams.items()}
        moved["dna"][:, self.AT] += 5.0
        mask = full_mask(1, self.CODONS)
        with torch.no_grad():
            before = model.modality_streams(streams, mask)["rna"]
            after = model.modality_streams(moved, mask)["rna"]
        return (after - before).abs().amax(dim=-1).flatten()

    def test_coupling_carries_dna_into_later_rna_codons_and_never_earlier(self):
        change = self.rna_change("coupled_mamba")
        self.assertEqual(float(change[: self.AT + 1].max()), 0.0,
                         "an RNA codon at or before the perturbed codon changed")
        self.assertGreater(float(change[self.AT + 1:].min()), 1e-6,
                           "a later RNA codon did not change")

    def test_the_uncoupled_ablation_keeps_the_rna_stream_its_own(self):
        self.assertEqual(float(self.rna_change("uncoupled_mamba").max()), 0.0)

    def test_the_bottleneck_control_keeps_the_rna_stream_its_own(self):
        self.assertEqual(float(self.rna_change("mamba_bottleneck").max()), 0.0)

    def test_no_design_lets_a_codon_see_downstream(self):
        for name in DESIGNS:
            torch.manual_seed(0)
            model = CM.DESIGNS[name](WIDTHS).eval()
            streams = random_streams(1, self.CODONS)
            moved = {m: s.clone() for m, s in streams.items()}
            for m in moved:
                moved[m][:, self.AT] += 5.0
            mask = full_mask(1, self.CODONS)
            with torch.no_grad():
                change = (model(moved, mask)[0] - model(streams, mask)[0]).abs().amax(dim=-1)
            self.assertEqual(float(change[0, : self.AT].max()), 0.0, name)


class ShapeAndStabilityTests(unittest.TestCase):
    def test_padding_cannot_reach_a_real_codon(self):
        # The same row, alone and padded with noise to a longer batch length.
        for name in DESIGNS:
            torch.manual_seed(0)
            model = CM.DESIGNS[name](WIDTHS).eval()
            real = random_streams(1, 30, seed=1)
            junk = random_streams(1, 12, seed=2)
            padded = {m: torch.cat([real[m], junk[m]], dim=1) for m in WIDTHS}
            mask = torch.zeros(1, 42, dtype=torch.bool)
            mask[:, :30] = True
            with torch.no_grad():
                alone = model(real, full_mask(1, 30))[0]
                inside = model(padded, mask)[0]
            torch.testing.assert_close(inside[:, :30], alone)
            self.assertEqual(float(inside[:, 30:].abs().max()), 0.0, name)

    def test_a_full_length_row_stays_finite_forward_and_backward(self):
        # 333 codons is the longest row under the 999-nucleotide rule.
        torch.manual_seed(0)
        model = CM.CoupledMambaFusion(WIDTHS)
        fused, _ = model(random_streams(2, 333), full_mask(2, 333))
        self.assertTrue(bool(torch.isfinite(fused).all()))
        fused.pow(2).mean().backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(bad, [])

    def test_the_coupled_head_and_its_ablation_have_the_same_parameters(self):
        coupled = CM.CoupledMambaFusion(WIDTHS)
        uncoupled = CM.UncoupledMambaFusion(WIDTHS)
        self.assertEqual([(n, p.shape) for n, p in coupled.named_parameters()],
                         [(n, p.shape) for n, p in uncoupled.named_parameters()])

    def test_the_bottleneck_control_has_no_mamba_layers(self):
        self.assertEqual(len(CM.MambaBottleneckFusion(WIDTHS).layers), 0)
        self.assertEqual(len(CM.CoupledMambaFusion(WIDTHS).layers), CM.N_LAYERS)

    def test_a_seed_fixes_the_initial_weights(self):
        outputs = []
        for _ in range(2):
            torch.manual_seed(7)
            model = CM.CoupledMambaFusion(WIDTHS).eval()
            with torch.no_grad():
                outputs.append(model(random_streams(1, 15), full_mask(1, 15))[0])
        torch.testing.assert_close(outputs[0], outputs[1])

    def test_initial_step_sizes_fall_in_mambas_range(self):
        torch.manual_seed(0)
        block = CM.MambaBlock(32, 64, 16, 4, 2)
        steps = torch.nn.functional.softplus(block.dt_proj.bias.detach())
        low, high = CM.DT_RANGE
        self.assertGreaterEqual(float(steps.min()), low * 0.999)
        self.assertLessEqual(float(steps.max()), high * 1.001)


class TrainerTests(unittest.TestCase):
    """The path the Stage 4 notebook takes: train by name, save, reuse, compare."""

    def setUp(self):
        self.fixture = test_fusion_model.PipelineTests("test_a_batch_carries_padded_streams_and_their_real_token_counts")
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_a_design_is_trained_saved_reused_and_compared_by_name(self):
        streams = self.fixture.streams()
        config = train.TrainConfig(max_epochs=2, batch_size=2, channels=3,
                                   early_stopping_patience=1, tconv="equation")
        with tempfile.TemporaryDirectory() as out:
            first = train.run(streams, "coupled_mamba", config, "cpu", out, log=None)
            self.assertEqual(first.variant, "coupled_mamba")
            saved = sorted(p.name for p in Path(out).iterdir())
            self.assertIn("stub__coupled_mamba__lambda0.0__ch3__equation__seed0.json", saved)
            again = train.run(streams, "coupled_mamba", config, "cpu", out, log=None)
            self.assertEqual(again.test_score, first.test_score)       # reused, not refitted
            other = train.run(streams, "mamba_bottleneck", config, "cpu", out, log=None)
            # Three stub test rows are too few for a defined interval; this checks only that
            # two designs saved on the same rows can be paired.
            contrast = train.paired_bootstrap(first, other, "regression", draws=20)
            self.assertEqual(set(contrast), {"difference", "ci_low", "ci_high", "first_wins"})


if __name__ == "__main__":
    unittest.main()
