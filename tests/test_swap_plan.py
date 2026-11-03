"""Metadata-only execution planning with invented JSONs; no models or research data."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import swap_plan


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5,
              entropy_lambda=0.0, channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="nt", rna="rnafm", prot="esm8")
SWAPS = (("dna", "nt500", True), ("dna", "hyena", True), ("rna", "rinalmo", True),
         ("rna", "mrnafm", False), ("rna", "calm", False),
         ("prot", "esm35", False), ("prot", "esm150", False), ("prot", "protbert", False))
PREVIOUS_STABILITY = tuple(item for item in SWAPS if item[1] in {"mrnafm", "calm", "esm150"})


class SwapPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.originals = self.root / "historical"
        self.destination = self.root / "phaseA/fixture"
        self.swaps = self.root / "phaseB/fixture"

    def snapshot(self):
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def write(self, directory, variant, *, seed=0, penalty=0.0, encoders=ORIGINAL,
              recipe=None, name=None, **updates):
        config = {**RECIPE, "seed": seed, "entropy_lambda": penalty, **(recipe or {})}
        record = dict(benchmark="fixture", variant=variant, config=config,
                      val_score=0.8 - penalty / 10, test_score=0.4,
                      epochs=1, best_epoch=0, seconds=1.0,
                      identity={"inputs": {"encoders": encoders}})
        record.update(updates)
        path = (Path(directory) / name if name else
                swap_plan.result_path(directory, "fixture", variant, config))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def original_grid(self, seed=0, variants=None):
        for variant in variants or swap_plan.ORIGINAL_VARIANTS:
            self.write(self.originals, variant, seed=seed, penalty=0.01 if variant == "mil" else 0.0)
        if seed == 0:
            self.write(self.originals, "mil", penalty=0.5)
            self.write(self.originals, "mil", penalty=1.0)

    def replacements(self, swaps=SWAPS, seed=0, singles_only=False):
        for modality, encoder, grid in swaps:
            for variant in swap_plan.swap_results.swap_variants(modality, grid=grid):
                if singles_only and not variant.startswith("unimodal"):
                    continue
                self.write(self.swaps / encoder, variant, seed=seed,
                           penalty=0.01 if variant == "mil" else 0.0,
                           encoders={**ORIGINAL, modality: encoder})

    def plan(self, *, seed=0, swaps=SWAPS, **overrides):
        request = dict(benchmark="fixture", config={**RECIPE, "seed": seed},
                       original_encoders=ORIGINAL, original_folders=[self.originals],
                       original_destination=self.destination, swaps=swaps, swap_root=self.swaps)
        request.update(overrides)
        untouched = deepcopy({k: v for k, v in request.items() if k != "checkpoint_validator"})
        before = self.snapshot()
        plan = swap_plan.plan_benchmark(**request)
        self.assertEqual({k: v for k, v in request.items() if k != "checkpoint_validator"}, untouched)
        self.assertEqual(before, self.snapshot())
        return plan

    def test_remaining_five_stability_swaps_need_23_fits_and_reuse_completed_subset(self):
        self.original_grid()
        self.replacements(PREVIOUS_STABILITY)
        plan = self.plan()
        self.assertTrue(plan.ready)
        self.assertEqual(len(plan.require_ready()), 42)
        self.assertEqual(sum(r.status == "new" for r in plan.runs), 23)
        self.assertEqual(sum(r.status == "reuse" for r in plan.runs), 19)
        self.assertEqual({r.encoder_key for r in plan.runs if r.status == "new"},
                         {"nt500", "hyena", "rinalmo", "esm35", "protbert"})
        self.assertFalse(self.destination.exists())

    def test_complete_selected_seed_plans_no_new_fits(self):
        self.original_grid()
        self.replacements()
        plan = self.plan()
        self.assertEqual({r.status for r in plan.runs}, {"reuse"})
        self.assertEqual(plan.lambda_, 0.01)
        self.assertTrue(all(r.path == r.source_path for r in plan.runs))

    def test_seed1_and_seed2_hold_seed0_selection_and_keep_existing_seed_files(self):
        self.original_grid()
        self.replacements()
        for seed in (1, 2):
            self.original_grid(seed, variants=("unimodal:dna", "concat", "mil", "cross"))
            # A better validation score at another repeat's penalty cannot retune the comparison.
            self.write(self.originals, "mil", seed=seed, penalty=0.5, val_score=0.99)
            plan = self.plan(seed=seed)
            self.assertTrue(plan.ready)
            self.assertEqual(plan.lambda_, 0.01)
            self.assertEqual(sum(r.status == "new" for r in plan.runs), 38)
            self.assertEqual({r.config["seed"] for r in plan.runs}, {seed})
            self.assertEqual({r.config["entropy_lambda"] for r in plan.runs if r.variant == "mil"}, {0.01})
            self.assertTrue(all(f"__seed{seed}" in r.path.name for r in plan.runs))

    def test_stability_repeat_schedules_all_seven_absent_original_references(self):
        self.original_grid()
        self.replacements()
        plan = self.plan(seed=1)
        originals = [r for r in plan.runs if r.role == "original"]
        self.assertEqual(len(originals), 7)
        self.assertEqual({r.status for r in originals}, {"new"})
        self.assertEqual({r.variant for r in originals}, set(swap_plan.ORIGINAL_VARIANTS))

    def test_repeat_missing_seed0_replacement_single_blocks_before_any_work(self):
        self.original_grid()
        self.replacements(singles_only=True)
        path = swap_plan.result_path(self.swaps / "hyena", "fixture", "unimodal-grid:dna", RECIPE)
        path.unlink()
        plan = self.plan(seed=1)
        with self.assertRaisesRegex(ValueError, "Seed-0 single-encoder selection unavailable for hyena"):
            plan.require_ready()
        self.assertTrue(any("unimodal-grid:dna" in issue for issue in plan.issues))

    def test_repeat_missing_original_rna_grid_blocks_unchanged_rna_reference_selection(self):
        self.original_grid(variants=("unimodal:dna", "unimodal:rna", "unimodal:prot", "mil"))
        self.replacements(singles_only=True)
        plan = self.plan(seed=2)
        self.assertFalse(plan.ready)
        self.assertIn("unimodal-grid:rna", plan.issues[0])

    def test_incomplete_or_duplicate_seed0_mil_grid_never_selects_from_repeats(self):
        self.write(self.originals, "mil", penalty=0.01)
        self.write(self.originals, "mil", seed=1, penalty=0.5)
        plan = self.plan(seed=1)
        self.assertFalse(plan.ready)
        self.assertIsNone(plan.lambda_)
        self.assertEqual(plan.runs, [])
        self.assertIn("missing penalties", plan.issues[0])
        self.original_grid()
        self.write(self.originals, "mil", penalty=0.01, name="duplicate.json")
        plan = self.plan()
        self.assertFalse(plan.ready)
        self.assertIn("Ambiguous", plan.issues[0])

    def test_native_fallback_uses_actual_historical_path_not_new_canonical_name(self):
        self.original_grid()
        normal = swap_plan.result_path(self.originals, "fixture", "unimodal:prot", RECIPE)
        normal.unlink()
        fallback = self.write(self.originals, "unimodal:prot", recipe={"tconv": "appendix"}, identity=None)
        plan = self.plan()
        protein = next(r for r in plan.runs if r.role == "original" and r.variant == "unimodal:prot")
        self.assertEqual(protein.status, "reuse")
        self.assertEqual(protein.path, fallback.resolve())
        self.assertNotEqual(protein.path, protein.destination)
        self.assertTrue(protein.legacy_identity)

    def test_duplicate_replacement_records_are_blocked_not_arbitrarily_chosen(self):
        self.original_grid()
        self.replacements(PREVIOUS_STABILITY)
        self.write(self.swaps / "calm", "concat", name="duplicate.json", encoders={**ORIGINAL, "rna": "calm"})
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "Ambiguous compatible results"):
            plan.require_ready()

    def test_wrong_recipe_or_malformed_canonical_target_is_preserved_and_blocked(self):
        self.original_grid()
        path = self.write(self.swaps / "calm", "concat", recipe={"lr": 0.01},
                          encoders={**ORIGINAL, "rna": "calm"})
        plan = self.plan()
        target = next(r for r in plan.runs if r.encoder_key == "calm" and r.variant == "concat")
        self.assertEqual(target.status, "blocked")
        self.assertIn("configuration differs", target.reason)
        path.write_text("{", encoding="utf-8")
        plan = self.plan()
        self.assertFalse(plan.ready)

    def test_recorded_wrong_encoder_mapping_blocks_existing_canonical_target(self):
        self.original_grid()
        self.write(self.swaps / "calm", "concat", encoders={**ORIGINAL, "rna": "other"})
        plan = self.plan()
        target = next(r for r in plan.runs if r.encoder_key == "calm" and r.variant == "concat")
        self.assertEqual(target.status, "blocked")
        self.assertIn("encoder mapping differs", target.reason)

    def test_checkpoint_is_metadata_only_unless_explicit_validator_succeeds(self):
        self.original_grid()
        target = swap_plan.result_path(self.swaps / "calm", "fixture", "concat", RECIPE)
        target.parent.mkdir(parents=True)
        checkpoint = target.with_suffix(".checkpoint")
        checkpoint.write_bytes(b"not deserialized by planner")
        plan = self.plan()
        run = next(r for r in plan.runs if r.encoder_key == "calm" and r.variant == "concat")
        self.assertEqual(run.status, "checkpoint_requires_review")
        self.assertFalse(plan.ready)
        validator = Mock(return_value=None)
        plan = self.plan(checkpoint_validator=validator)
        validator.assert_called_once_with(checkpoint.resolve(),
                                          {"config": RECIPE, "variant": "concat", "benchmark": "fixture"})
        self.assertEqual(next(r for r in plan.runs if r.path == target.resolve()).status, "resume")
        self.assertTrue(plan.ready)

    def test_incompatible_checkpoint_and_orphan_final_weights_block_before_work(self):
        self.original_grid()
        target = swap_plan.result_path(self.swaps / "calm", "fixture", "concat", RECIPE)
        target.parent.mkdir(parents=True)
        target.with_suffix(".checkpoint").write_bytes(b"fixture")
        plan = self.plan(checkpoint_validator=Mock(side_effect=ValueError("different seed")))
        self.assertFalse(plan.ready)
        self.assertTrue(any("different seed" in r.reason for r in plan.runs))
        target.with_suffix(".weights.pt").write_bytes(b"fixture")
        validator = Mock(return_value=None)
        plan = self.plan(checkpoint_validator=validator)
        validator.assert_not_called()
        self.assertTrue(any("Final weights exist" in r.reason for r in plan.runs))

    def test_completed_score_only_result_does_not_retrain_for_missing_arrays_or_checkpoint(self):
        self.original_grid()
        path = self.write(self.swaps / "calm", "concat", encoders={**ORIGINAL, "rna": "calm"})
        path.with_suffix(".checkpoint").write_bytes(b"leftover")
        validator = Mock(side_effect=AssertionError("Completed result needs no checkpoint load"))
        plan = self.plan(checkpoint_validator=validator)
        run = next(r for r in plan.runs if r.encoder_key == "calm" and r.variant == "concat")
        self.assertEqual(run.status, "reuse")
        validator.assert_not_called()

    def test_failure_then_completion_replans_only_unfinished_fits(self):
        self.original_grid()
        initial = self.plan(swaps=PREVIOUS_STABILITY)
        first = next(r for r in initial.runs if r.status == "new")
        self.write(first.destination.parent, first.variant, seed=first.config["seed"],
                   penalty=first.config["entropy_lambda"], encoders=first.encoders)
        after = self.plan(swaps=PREVIOUS_STABILITY)
        self.assertEqual(sum(r.status == "new" for r in after.runs), 11)
        self.assertEqual(next(r for r in after.runs if r.destination == first.destination).status, "reuse")

    def test_full_config_and_fixed_seed_contract_are_explicit(self):
        for changes in (dict(selection_seed=1), dict(config={**RECIPE, "seed": True}),
                        dict(config={k: v for k, v in RECIPE.items() if k != "lr"}),
                        dict(config={**RECIPE, "tconv": "appendix"})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)


if __name__ == "__main__":
    unittest.main()
