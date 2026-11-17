"""Decision inspections on invented transcripts; no datasets, weights or caches."""

from copy import deepcopy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import region_experiment as inputs, region_inputs, region_training as study


def audit_fixture():
    rows = []
    for i, prefix in enumerate(itertools.islice(itertools.product("ACGT", repeat=3), 24)):
        sequence = "".join(prefix) + "ATGAAATAA" + "CC"
        track = np.zeros(len(sequence), dtype=np.int8)
        track[3:12:3] = 1
        rows.append({"sequence": sequence, "cds": track, "gene": f"gene{i}", "target": i / 10})
    rows.append({**rows[0], "sequence": "N" + rows[0]["sequence"][1:], "target": 0.})
    frame = pd.DataFrame(rows)
    cohort = inputs.prepare_cohort(frame, "rnahl-human", grouping="homology",
                                   homology_map={f"gene{i}": str(i // 2) for i in range(12)},
                                   unmapped_gene_policy="gene")
    preparation = study.prepare_study(cohort, split_seed=42, mask_seed=0)
    plans = inputs.plan_chunks(cohort, window_lengths={"rnafm": 5, "rinalmo": 5, "esm2_8m": 2})
    return frame, cohort, plans, preparation


class RegionDecisionTests(unittest.TestCase):
    def test_mask_change_counts_actual_update_positions_and_stop(self):
        layout = region_inputs.CodingLayout(12, 3, 12, 9, "MK", True)
        native = inputs.mask_change_summary(layout, (0, 6), "native")
        pooled = inputs.mask_change_summary(layout, (0, 6), "codon_pool")
        self.assertEqual(native["protein_available_positions"], 6)
        self.assertEqual(native["changed_update_positions"], 3)
        self.assertEqual(pooled["protein_available_positions"], 2)
        self.assertEqual(pooled["changed_update_positions"], 1)
        self.assertEqual(native["retained_update_fraction"], .5)
        self.assertEqual(native["mask_overlap"], .25)
        self.assertEqual(pooled["mask_overlap"], .25)

    def test_changed_annotation_can_leave_effective_control_unchanged(self):
        layout = region_inputs.CodingLayout(12, 3, 12, 9, "MK", True)
        changed = inputs.mask_change_summary(layout, (0, 12), "native")
        self.assertEqual(changed["changed_mask_positions"], 3)
        self.assertEqual(changed["changed_update_positions"], 0)
        self.assertTrue(changed["effective_mask_unchanged"])
        self.assertEqual(changed["retained_update_fraction"], 1.)

    def test_audit_retains_rows_and_reports_geometry_without_target_values(self):
        frame, cohort, plans, preparation = audit_fixture()
        original = inputs._frame_hash(frame)
        with patch.object(inputs, "open_token_caches", side_effect=AssertionError("cache access")):
            report = inputs.input_audit(frame, cohort, plans, preparation,
                                       widths={"rnafm": 2, "rinalmo": 2, "esm2_8m": 3}, dtype="float16")
        self.assertEqual(inputs._frame_hash(frame), original)
        self.assertEqual(report["raw_rows"], 25)
        self.assertEqual(report["retained_rows"], 24)
        self.assertEqual(len(report["masks"]), 48)
        self.assertEqual(sum(g["rows"] for g in report["groups"]), 24)
        self.assertEqual({r["group_kind"] for r in report["rows"] if r["status"] == "retained"}, {"mapped", "unmapped"})
        self.assertFalse(any("target" in r or "label" in r for r in report["rows"]))
        rna = next(r for r in report["windows"] if r["encoder"] == "rnafm")
        self.assertEqual(rna["tokens"], 24 * 14)
        self.assertEqual(rna["payload_bytes"], 24 * 14 * 2 * 2)
        self.assertEqual(rna["chunks"], 24 * 3)
        self.assertEqual(rna["multiwindow_rows"], 24)
        self.assertEqual(report["rows"][0]["rnafm_boundaries_in_cds"], 2)
        self.assertEqual(report["rows"][-1]["status"], "excluded")
        self.assertEqual(report["context"]["preparation_policy"]["split_seed"], 42)
        self.assertEqual(report["context"]["cohort_policy"]["homology_map_sha256"],
                         cohort.identity["homology_map_sha256"])
        self.assertEqual(report["context"]["plans"]["rnafm"]["checkpoint"],
                         plans["rnafm"].identity["checkpoint"])

    def test_audit_refuses_reordered_source_or_wrong_widths(self):
        frame, cohort, plans, preparation = audit_fixture()
        with self.assertRaisesRegex(ValueError, "source differs"):
            inputs.input_audit(frame.iloc[::-1], cohort, plans, preparation,
                               widths={key: 2 for key in plans}, dtype="float16")
        with self.assertRaisesRegex(ValueError, "positive width"):
            inputs.input_audit(frame, cohort, plans, preparation, widths={"rnafm": 2}, dtype="float16")

    def test_audit_refuses_changed_window_contents_or_row_mapping(self):
        frame, cohort, plans, preparation = audit_fixture()
        plan = plans["rnafm"]
        changes = (
            replace(plan, chunks=(replace(plan.chunks[0], text=plan.chunks[0].text[:-1]), *plan.chunks[1:])),
            replace(plan, row_chunks=tuple(reversed(plan.row_chunks))),
        )
        for changed in changes:
            with self.subTest(changed=changed.row_chunks[0]):
                with self.assertRaisesRegex(ValueError, "window contents"):
                    inputs.input_audit(frame, cohort, {**plans, "rnafm": changed}, preparation,
                                       widths={key: 2 for key in plans}, dtype="float16")

    def test_saved_audit_reads_without_inputs_and_preserves_incompatible_evidence(self):
        frame, cohort, plans, preparation = audit_fixture()
        audit = inputs.input_audit(frame, cohort, plans, preparation,
                                   widths={key: 2 for key in plans}, dtype="float16")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            self.assertIsNone(inputs.read_input_audit(path))
            saved = inputs.publish_input_audit(path, audit, producer={"host": "synthetic"})
            before = path.read_bytes()
            self.assertEqual(inputs.publish_input_audit(path, audit, producer={"host": "other"}), saved)
            different = deepcopy(audit)
            different["context"]["dtype"] = "float32"
            with self.assertRaisesRegex(ValueError, "differs"):
                inputs.publish_input_audit(path, different, producer={"host": "synthetic"})
            self.assertEqual(path.read_bytes(), before)
            with patch.object(inputs, "load_local_frame", side_effect=AssertionError("input load")):
                self.assertEqual(inputs.read_input_audit(path), saved)
            damaged = json.loads(path.read_text())
            damaged["retained_rows"] = 999
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "changed"):
                inputs.read_input_audit(path)


if __name__ == "__main__":
    unittest.main()
