"""Check the partner-task panel: what identifies a fit, and what the reader refuses.

The property that matters is that saved evidence cannot be used for a question it does not
answer, and cannot be silently replaced. A fit is identified by the table, the split's rows,
both token caches, the architecture and the optimizer settings; change any of them and the
reader reports an incompatible record instead of reusing or overwriting it.

No arm is fitted here. ``run_panel`` is exercised against a stub fit, which is enough to check
the reuse, refusal and saving decisions without a training loop.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import pair_study as PS, pair_tasks as PT, pair_training as PTR  # noqa: E402


def frame(n_left=10, n_right=10, seed=0, kind="binding"):
    rng = np.random.default_rng(seed)
    lefts = [f"R{i:02d}" for i in range(n_left)]
    rights = [f"P{i:02d}" for i in range(n_right)]
    rows = [{"left": a, "right": b, "label": int(rng.random() < 0.4),
             "provenance": "measured_positive"} for a in lefts for b in rights]
    for row in rows:
        row["provenance"] = "measured_positive" if row["label"] else "measured_negative"
    table = pd.DataFrame(rows)
    if kind == "affinity":
        table["label"] = rng.normal(size=len(table))
        table["provenance"] = "measured_positive"
        table["endpoint"] = "pKd"
    return PT.PairFrame("stub", table,
                        {a: "ACGU" * 5 for a in lefts}, {b: "MKV" * 5 for b in rights},
                        {a: f"lg{i // 2}" for i, a in enumerate(lefts)},
                        {b: f"rg{i // 2}" for i, b in enumerate(rights)},
                        kind)


def summaries(left_encoder="rinalmo", right_encoder="esm2_150m"):
    return {"left": {"modality": "rna", "encoder": left_encoder, "revision": "abc",
                     "width": 640, "token_limit": 1024, "entities": 10,
                     "at_token_limit": 0, "at_token_limit_share": 0.0,
                     "input_sha256": "left-hash"},
            "right": {"modality": "protein", "encoder": right_encoder, "revision": "def",
                      "width": 320, "token_limit": 1024, "entities": 10,
                      "at_token_limit": 2, "at_token_limit_share": 0.2,
                      "input_sha256": "right-hash"}}


def stub_record(arm, split, *, auroc=0.8, seed=0):
    rows = split["indices"]["test"]
    rng = np.random.default_rng(seed)
    return {"arm": arm, "kind": "binding", "selection_metric": "auprc",
            "model_kwargs": {}, "config": {}, "widths": {"left": 640, "right": 320},
            "best_epoch": 3, "history": [],
            "val": {"auroc": auroc, "auprc": auroc},
            "test": {"auroc": auroc, "auprc": auroc - 0.1},
            "test_rows": rows.tolist(), "test_truth": [0, 1] * (len(rows) // 2) + [0] * (len(rows) % 2),
            "test_score": rng.random(len(rows)).tolist(),
            "positive_weight": 1.0, "split": {"held_out": split["held_out"]},
            "model": object()}


class PanelTests(unittest.TestCase):
    def test_the_declared_panel_is_six_arms_with_two_baselines(self):
        self.assertEqual(len(PS.PANEL), 6)
        self.assertIn("left_only", PS.PANEL)
        self.assertIn("right_only", PS.PANEL)

    def test_every_panel_key_names_a_real_scorer(self):
        from shared_code.fusion import pair_models

        for key in PS.panel_keys(include_optional=True):
            arm, kwargs = PS.arm_of(key)
            self.assertIn(arm, pair_models.PAIR_SCORERS)
            self.assertIsInstance(kwargs, dict)

    def test_an_optional_key_carries_its_model_settings(self):
        arm, kwargs = PS.arm_of("cross_attention__position_gate")
        self.assertEqual(arm, "cross_attention")
        self.assertEqual(kwargs, {"gate": "position"})

    def test_an_unknown_key_is_refused(self):
        with self.assertRaises(ValueError):
            PS.arm_of("mystery_arm")

    def test_optional_arms_are_excluded_unless_asked_for(self):
        self.assertEqual(PS.panel_keys(), list(PS.PANEL))
        self.assertGreater(len(PS.panel_keys(include_optional=True)), len(PS.PANEL))


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.split = PT.disjoint_split(self.frame, held_out="both", seed=42)
        self.config = PTR.PairConfig(max_epochs=1)
        self.summaries = summaries()

    def identity(self, **changes):
        parts = {"frame": self.frame, "split": self.split, "key": "concat",
                 "config": self.config, "summaries": self.summaries, **changes}
        return PS.study_identity(parts["frame"], parts["split"], parts["key"],
                                 parts["config"], parts["summaries"])

    def test_the_same_question_gives_the_same_identity(self):
        self.assertEqual(self.identity()["sha256"], self.identity()["sha256"])

    def test_the_identity_records_what_the_table_is(self):
        record = self.identity()["table"]
        self.assertEqual(record["name"], "stub")
        self.assertEqual(record["kind"], "binding")
        self.assertEqual(record["modalities"], {"left": "rna", "right": "protein"})
        self.assertEqual(record["pairs"], len(self.frame.pairs))

    def test_a_different_arm_is_a_different_fit(self):
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(key="bilinear")["sha256"])

    def test_a_gate_variant_is_not_the_same_fit_as_the_ungated_arm(self):
        """Both are cross-attention; the gate is the thing being tested, so it must count."""
        plain = self.identity(key="cross_attention")["sha256"]
        gated = self.identity(key="cross_attention__position_gate")["sha256"]
        self.assertNotEqual(plain, gated)

    def test_a_different_split_is_a_different_fit(self):
        other = PT.disjoint_split(self.frame, held_out="left", seed=42)
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(split=other)["sha256"])

    def test_the_same_split_rule_with_a_different_seed_is_a_different_fit(self):
        other = PT.disjoint_split(self.frame, held_out="both", seed=7)
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(split=other)["sha256"])

    def test_a_different_encoder_on_either_side_is_a_different_fit(self):
        """The three-seed cohort found the encoder mattered more than the operation did."""
        left = self.identity(summaries=summaries(left_encoder="rnafm"))
        right = self.identity(summaries=summaries(right_encoder="esm2_8m"))
        self.assertNotEqual(self.identity()["sha256"], left["sha256"])
        self.assertNotEqual(self.identity()["sha256"], right["sha256"])

    def test_different_cache_inputs_are_a_different_fit_even_under_one_encoder(self):
        changed = summaries()
        changed["left"]["input_sha256"] = "other-hash"
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(summaries=changed)["sha256"])

    def test_a_different_optimizer_setting_is_a_different_fit(self):
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(config=PTR.PairConfig(max_epochs=1, lr=1e-4))["sha256"])

    def test_a_different_seed_is_a_different_fit(self):
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(config=PTR.PairConfig(max_epochs=1, seed=1))["sha256"])

    def test_a_relabelled_table_of_the_same_shape_is_a_different_fit(self):
        other = frame(seed=5)
        self.assertNotEqual(self.identity()["sha256"],
                            self.identity(frame=other)["sha256"])

    def test_the_token_limit_note_names_both_sides(self):
        note = PS.token_limit_note(self.summaries)
        self.assertIn("rinalmo", note)
        self.assertIn("esm2_150m", note)
        self.assertIn("2 of 10", note)


class SavedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.frame = frame()
        self.split = PT.disjoint_split(self.frame, held_out="both", seed=42)
        self.config = PTR.PairConfig(max_epochs=1)
        self.summaries = summaries()
        self.identity = PS.study_identity(self.frame, self.split, "concat", self.config,
                                          self.summaries)
        self.path = PS.result_path(self.root, self.frame, "concat", self.config,
                                   held_out="both")

    def test_a_result_path_separates_table_split_arm_and_seed(self):
        self.assertEqual(self.path.name, "concat__seed0.json")
        self.assertEqual(self.path.parent.name, "held_out_both")
        self.assertEqual(self.path.parent.parent.name, "stub")

    def test_a_saved_fit_drops_the_live_module_and_keeps_the_scores(self):
        record = stub_record("concat", self.split)
        PS.save_result(self.path, record, self.identity)
        stored = json.loads(self.path.read_text())
        self.assertNotIn("model", stored["result"])
        self.assertEqual(stored["result"]["test"]["auroc"], 0.8)
        self.assertEqual(stored["identity"]["sha256"], self.identity["sha256"])

    def test_a_saved_fit_is_read_back_when_the_question_is_the_same(self):
        PS.save_result(self.path, stub_record("concat", self.split), self.identity)
        payload, issue = PS.read_result(self.path, self.identity)
        self.assertIsNone(issue)
        self.assertEqual(payload["result"]["arm"], "concat")

    def test_a_missing_file_is_reported_rather_than_raising(self):
        payload, issue = PS.read_result(self.path, self.identity)
        self.assertIsNone(payload)
        self.assertEqual(issue, "missing")

    def test_a_fit_saved_under_another_identity_is_reported_as_incompatible(self):
        other = PS.study_identity(self.frame, self.split, "concat",
                                  PTR.PairConfig(max_epochs=1, lr=1e-4), self.summaries)
        PS.save_result(self.path, stub_record("concat", self.split), other)
        payload, issue = PS.read_result(self.path, self.identity)
        self.assertIsNotNone(payload)
        self.assertTrue(issue.startswith("incompatible"))
        self.assertIn("config", issue)

    def test_an_unreadable_file_is_reported_rather_than_raising(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json")
        payload, issue = PS.read_result(self.path, self.identity)
        self.assertIsNone(payload)
        self.assertIn("unreadable", issue)

    def test_an_unsupported_schema_is_refused(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"schema": "pair-study.v0"}))
        payload, issue = PS.read_result(self.path, self.identity)
        self.assertIsNone(payload)
        self.assertIn("unsupported schema", issue)

    def test_the_inventory_reports_one_row_per_planned_fit(self):
        splits = {held_out: PT.disjoint_split(self.frame, held_out=held_out, seed=42)
                  for held_out in PT.HELD_OUT}
        rows = PS.saved_inventory(self.root, self.frame, self.config, self.summaries, splits)
        self.assertEqual(len(rows), len(PT.HELD_OUT) * len(PS.PANEL))
        self.assertTrue(all(row["status"] == "missing" for row in rows))
        PS.save_result(self.path, stub_record("concat", self.split), self.identity)
        rows = PS.saved_inventory(self.root, self.frame, self.config, self.summaries, splits)
        saved = [row for row in rows if row["status"] == "saved"]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["key"], "concat")
        self.assertEqual(saved[0]["held_out"], "both")
        self.assertEqual(saved[0]["test_auroc"], 0.8)


class RunPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.frame = frame()
        self.split = PT.disjoint_split(self.frame, held_out="both", seed=42)
        self.config = PTR.PairConfig(max_epochs=1)
        self.summaries = summaries()
        self.fitted = []

        def stub_fit(arm, frame_, split_, embeddings, config, *, device="cpu",
                     model_kwargs=None, log=None):
            self.fitted.append(arm)
            return stub_record(arm, split_, auroc=0.6 + 0.02 * len(self.fitted),
                               seed=len(self.fitted))

        self._real_fit = PTR.fit
        PTR.fit = stub_fit
        self.addCleanup(setattr, PTR, "fit", self._real_fit)

    def run_panel(self, **kwargs):
        return PS.run_panel(self.frame, self.split, {}, self.config, self.summaries,
                            self.root, controls=False, log=None, **kwargs)

    def test_every_panel_arm_is_fitted_once_and_saved(self):
        panel = self.run_panel()
        self.assertEqual(len(panel["results"]), len(PS.PANEL))
        self.assertEqual(len(self.fitted), len(PS.PANEL))
        for key in PS.PANEL:
            path = PS.result_path(self.root, self.frame, key, self.config, held_out="both")
            self.assertTrue(path.exists())

    def test_a_second_run_reuses_the_saved_evidence_instead_of_refitting(self):
        self.run_panel()
        self.fitted.clear()
        panel = self.run_panel()
        self.assertEqual(self.fitted, [])
        self.assertEqual(sorted(panel["reused"]), sorted(PS.PANEL))

    def test_saved_evidence_under_another_identity_is_left_alone_and_reported(self):
        """An accidental rerun must not replace the file a comparison was computed from."""
        other = PS.study_identity(self.frame, self.split, "concat",
                                  PTR.PairConfig(max_epochs=1, lr=1e-4), self.summaries)
        path = PS.result_path(self.root, self.frame, "concat", self.config, held_out="both")
        PS.save_result(path, stub_record("concat", self.split, auroc=0.99), other)
        before = path.read_bytes()
        panel = self.run_panel()
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn("concat", panel["results"])
        self.assertEqual([entry["key"] for entry in panel["refused"]], ["concat"])
        self.assertNotIn("concat", self.fitted)

    def test_a_named_subset_fits_only_those_arms(self):
        panel = self.run_panel(keys=["left_only", "concat"])
        self.assertEqual(sorted(panel["results"]), ["concat", "left_only"])
        self.assertEqual(sorted(self.fitted), ["concat", "left_only"])

    def test_the_panel_table_reports_each_arm_with_its_controls(self):
        panel = self.run_panel(keys=["left_only", "concat"])
        panel["results"]["concat"]["controls"] = {
            "degree": {"auroc": 0.55, "auprc": 0.4, "reads_sequences": False},
            "partner_reassigned": {"rows": 10, "mean_score_real": 0.8,
                                   "mean_score_reassigned": 0.3}}
        rows = PS.panel_table(panel, self.frame)
        concat = next(row for row in rows if row["key"] == "concat")
        self.assertEqual(concat["degree_auroc"], 0.55)
        self.assertAlmostEqual(concat["real_minus_reassigned"], 0.5)

    def test_the_degree_gap_names_the_best_arm_and_what_it_beat(self):
        panel = self.run_panel(keys=["left_only", "concat"])
        for payload in panel["results"].values():
            payload["controls"] = {"degree": {"auroc": 0.55, "auprc": 0.4}}
        gap = PS.degree_gap(panel, self.frame)
        self.assertEqual(gap["metric"], "auroc")
        self.assertAlmostEqual(gap["degree"], 0.55)
        self.assertAlmostEqual(gap["gap"], gap["best"] - 0.55)
        self.assertIn(gap["best_arm"], panel["results"])

    def test_an_affinity_table_is_read_on_its_own_metric(self):
        affinity = frame(kind="affinity")
        split = PT.disjoint_split(affinity, held_out="both", seed=42)
        identity = PS.study_identity(affinity, split, "concat", self.config, self.summaries)
        path = PS.result_path(self.root, affinity, "concat", self.config, held_out="both")
        record = stub_record("concat", split)
        record["kind"] = "affinity"
        record["test"] = {"spearman": 0.42, "concordance": 0.7, "rmse": 1.1}
        PS.save_result(path, record, identity, controls={"degree": {"spearman": 0.1}})
        panel = {"results": {"concat": PS.read_result(path, identity)[0]}}
        gap = PS.degree_gap(panel, affinity)
        self.assertEqual(gap["metric"], "spearman")
        self.assertAlmostEqual(gap["gap"], 0.42 - 0.1)

    def test_the_comparison_shape_carries_the_panel_key_as_the_arm(self):
        panel = self.run_panel(keys=["left_only", "concat"])
        records = PS.results_for_comparison(panel)
        self.assertEqual(sorted(r["arm"] for r in records), ["concat", "left_only"])
        self.assertTrue(all("test_score" in r for r in records))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
