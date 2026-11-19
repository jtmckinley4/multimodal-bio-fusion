"""Focused swap plans use invented records, without models or research inputs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import run_identity, swap_plan


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5,
              entropy_lambda=0.0, channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="nt", rna="rnafm", prot="esm8")
NEW_SWAPS = (("rna", "rnaernie", True), ("rna", "ernierna", True))


class SwapExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.historical_originals = self.root / "historical/originals"
        self.historical_swaps = self.root / "historical/swaps"
        self.fresh_originals = self.root / "extension/originals"
        self.fresh_swaps = self.root / "extension/swaps"
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "unimodal-grid:rna"):
            self.write(self.historical_originals, variant, ORIGINAL)
        for penalty in (0.01, 0.5, 1.0):
            self.write(self.historical_originals, "mil", ORIGINAL, penalty=penalty,
                       val_score=0.8 if penalty == 0.01 else 0.4)

    @staticmethod
    def identity(benchmark, variant, recipe, encoders):
        return run_identity.make_identity(
            benchmark=benchmark, variant=variant, config=recipe,
            inputs={"encoders": encoders, "ordered_rows": "invented-row-identity"},
            model={"contract": "synthetic-extension"},
            loader={"kind": "custom", "recovery": {name: {} for name in ("train", "val", "test")}},
        )

    def write(self, directory, variant, encoders, *, seed=0, penalty=0.0,
              val_score=0.5, modern=False):
        recipe = {**RECIPE, "seed": seed, "entropy_lambda": penalty}
        identity = (self.identity("fixture", variant, recipe, encoders) if modern
                    else {"inputs": {"encoders": encoders}})
        record = dict(benchmark="fixture", variant=variant, config=recipe,
                      val_score=val_score, test_score=0.5, epochs=1, best_epoch=0,
                      seconds=1.0, identity=identity)
        path = swap_plan.result_path(directory, "fixture", variant, recipe)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def snapshot(self):
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def plan(self, *, seed=0, **changes):
        request = dict(
            benchmark="fixture", config={**RECIPE, "seed": seed}, original_encoders=ORIGINAL,
            original_folders=(self.fresh_originals,), original_destination=self.fresh_originals,
            swaps=NEW_SWAPS, swap_root=self.fresh_swaps,
            selection_original_folders=(self.historical_originals,),
            selection_swap_root=self.historical_swaps,
            fit_variants=("mil", "concat"), include_originals=False, select_singles=False,
        )
        request.update(changes)
        before = self.snapshot()
        unchanged = deepcopy({k: v for k, v in request.items()
                              if k not in {"identity_factory", "checkpoint_validator"}})
        try:
            return swap_plan.plan_benchmark(**request)
        finally:
            self.assertEqual(self.snapshot(), before)
            self.assertEqual({k: v for k, v in request.items()
                              if k not in {"identity_factory", "checkpoint_validator"}}, unchanged)

    def test_exact_four_requested_fits_with_mil_first_and_no_invented_single(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                plan = self.plan(seed=seed)
                self.assertEqual(
                    [(run.encoder_key, run.variant) for run in plan.require_ready()],
                    [("rnaernie", "mil"), ("ernierna", "mil"),
                     ("rnaernie", "concat"), ("ernierna", "concat")],
                )
                self.assertEqual({run.role for run in plan.runs}, {"replacement"})
                self.assertEqual({run.status for run in plan.runs}, {"new"})
                self.assertEqual({run.config["seed"] for run in plan.runs}, {seed})
                self.assertTrue(all(f"__seed{seed}" in run.path.name for run in plan.runs))
                self.assertEqual(plan.selected_singles, {})
                self.assertEqual(plan.selection_lambda, 0.01)
                self.assertEqual({run.config["entropy_lambda"] for run in plan.runs
                                  if run.variant == "mil"}, {0.01})
                self.assertEqual({run.config["entropy_lambda"] for run in plan.runs
                                  if run.variant == "concat"}, {0.0})
        self.assertFalse(self.fresh_originals.exists())
        self.assertFalse(self.fresh_swaps.exists())
        self.assertFalse(self.historical_swaps.exists())

    def test_original_selection_bytes_remain_the_only_selection_evidence(self):
        plan = self.plan()
        self.assertEqual(len(plan.selection_sources), 7)
        for name, digest in plan.selection_sources.items():
            path = Path(name)
            self.assertTrue(path.is_relative_to(self.historical_originals))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_completed_fit_is_reused_and_other_training_seeds_remain_new(self):
        completed = self.write(self.fresh_swaps / "rnaernie", "mil",
                               {**ORIGINAL, "rna": "rnaernie"}, penalty=0.01, modern=True)
        plan = self.plan(identity_factory=self.identity)
        self.assertTrue(plan.ready)
        self.assertEqual([run.status for run in plan.runs], ["reuse", "new", "new", "new"])
        self.assertEqual(plan.runs[0].source_path, completed)
        for seed in (1, 2):
            self.assertEqual({run.status for run in self.plan(seed=seed).runs}, {"new"})

    def test_mismatched_encoder_or_current_input_identity_blocks_and_preserves_result(self):
        path = self.write(self.fresh_swaps / "rnaernie", "mil",
                          {**ORIGINAL, "rna": "different-encoder"}, penalty=0.01, modern=True)
        plan = self.plan(identity_factory=self.identity)
        self.assertFalse(plan.ready)
        self.assertIn("encoder mapping differs", plan.runs[0].reason)
        self.assertTrue(path.is_file())

        self.write(self.fresh_swaps / "rnaernie", "mil",
                   {**ORIGINAL, "rna": "rnaernie"}, penalty=0.01, modern=True)

        def changed_identity(*args):
            identity = self.identity(*args)
            identity["inputs"]["ordered_rows"] = "different-invented-rows"
            return identity

        plan = self.plan(identity_factory=changed_identity)
        self.assertFalse(plan.ready)
        self.assertIn("full input/model/loader identity differs", plan.runs[0].reason)

    def test_focused_mode_does_not_weaken_versioned_result_or_checkpoint_checks(self):
        path = self.write(self.fresh_swaps / "rnaernie", "concat",
                          {**ORIGINAL, "rna": "rnaernie"})
        plan = self.plan()
        self.assertFalse(plan.ready)
        self.assertIn("full versioned identity", plan.runs[2].reason)
        path.unlink()  # This test fixture now represents an interrupted new fit.
        checkpoint = path.with_suffix(".checkpoint")
        checkpoint.write_bytes(b"synthetic checkpoint, never deserialized")
        validator = Mock(return_value=None)
        plan = self.plan(checkpoint_validator=validator)
        self.assertEqual(plan.runs[2].status, "checkpoint_requires_review")
        validator.assert_not_called()
        plan = self.plan(identity_factory=self.identity, checkpoint_validator=validator)
        self.assertEqual(plan.runs[2].status, "resume")
        validator.assert_called_once_with(
            checkpoint, self.identity("fixture", "concat", RECIPE, {**ORIGINAL, "rna": "rnaernie"}),
        )

    def test_missing_historical_penalty_cannot_be_replaced_by_new_scores(self):
        missing = swap_plan.result_path(self.historical_originals, "fixture", "mil",
                                       {**RECIPE, "entropy_lambda": 1.0})
        missing.unlink()
        self.write(self.fresh_originals, "mil", ORIGINAL, penalty=1.0, modern=True, val_score=1.0)
        plan = self.plan()
        self.assertFalse(plan.ready)
        self.assertEqual(plan.runs, [])
        self.assertIn("historical seed-0 positive MIL grid", plan.issues[0])

    def test_default_single_selection_still_requires_new_encoder_baselines(self):
        plan = self.plan(fit_variants=None, include_originals=True, select_singles=True)
        self.assertFalse(plan.ready)
        self.assertIn("single-encoder grid required for rnaernie", plan.issues[0])

    def test_explicit_one_head_limits_every_replacement(self):
        plan = self.plan(fit_variants=("mil",))
        self.assertTrue(plan.ready)
        self.assertEqual([(run.encoder_key, run.variant) for run in plan.runs],
                         [("rnaernie", "mil"), ("ernierna", "mil")])

    def test_unsupported_or_ambiguous_scope_is_rejected(self):
        for changes in (
            {"fit_variants": None}, {"fit_variants": ()}, {"fit_variants": "mil"},
            {"fit_variants": {"mil", "concat"}}, {"fit_variants": ("mil", "mil")},
            {"fit_variants": ("unimodal:rna",)}, {"fit_variants": ("unknown",)},
            {"include_originals": True}, {"include_originals": 0}, {"select_singles": 0},
            {"selection_original_folders": None}, {"selection_swap_root": None}, {"swaps": ()},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)


if __name__ == "__main__":
    unittest.main()
