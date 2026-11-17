"""Synthetic in-place reference reuse; no completed research records are opened."""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import tempfile
import unittest

from test_controlled_experiment import fixture
from shared_code.fusion import run_identity, study_references, study_selection


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.streams, self.caches, self.config = fixture()
        self.streams.key = "fungal_expression"

    def save(self, variant="unimodal:dna"):
        path = self.root / "outputs/phaseA/fungal_expression/fit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = run_identity.blf_identity(rows=self.streams.rows,
            benchmark=self.streams.key, variant=variant, config=self.config,
            encoders=self.streams.encoders, caches=self.caches, task="regression")
        record = dict(benchmark=self.streams.key, variant=variant, config=asdict(self.config),
            identity=identity, epochs=1, best_epoch=0, val_score=.4, test_score=.3,
            seconds=1., test_truth=self.streams.rows.loc[
                self.streams.rows.Split == "test", "Value"].to_numpy(dtype=np.float32).tolist(),
            test_prediction=[.1, .9])
        path.write_text(json.dumps(record))
        return path

    def read(self, bundle="original", current=True):
        args = dict(streams=self.streams, cache_records=self.caches) if current else {}
        return study_references.read_reference(self.root, self.streams.key, bundle,
            "unimodal:dna", 0, self.config, **args)

    def test_reads_without_writing_or_loading_weights(self):
        path = self.save()
        before = path.read_bytes()
        result = self.read()
        self.assertEqual(result.val_score, .4)
        self.assertTrue(result.extra_fields["reference_source"]["current_inputs_verified"])
        self.assertFalse(self.read(current=False).extra_fields["reference_source"]["current_inputs_verified"])
        self.assertEqual(path.read_bytes(), before)

    def test_unused_replacement_rna_does_not_invalidate_dna_single(self):
        self.save()
        self.streams.encoders = dict(study_references.BUNDLES["mrnafm"])
        self.caches["rna"] = {**self.caches["rna"], "encoder": "mrnafm"}
        from shared_code.encoders import ENCODERS
        from shared_code.fusion import token_embeddings as tokens
        self.caches["rna"]["input_sha256"] = tokens.input_fingerprint(tokens.inputs_for_rows(
            ENCODERS["mrnafm"], self.streams.rows["sequence"].tolist()))
        self.assertIsNotNone(self.read(bundle="mrnafm"))

    def test_missing_is_reported_but_incompatible_is_not_missing(self):
        self.assertIsNone(self.read())
        path = self.save()
        record = json.loads(path.read_text())
        record["epochs"] = 0
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.read()

    def test_changed_cohort_or_used_cache_refused(self):
        self.save()
        self.streams.rows.loc[0, "Value"] = 80.
        with self.assertRaisesRegex(ValueError, "targets differ"):
            self.read()
        self.streams, self.caches, self.config = fixture()
        self.streams.key = "fungal_expression"
        self.caches["dna"]["shards"][0]["sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "dna cache"):
            self.read()

    def test_grouped_stability_never_uses_published_split(self):
        self.streams.key = "mrna_stability"
        with self.assertRaisesRegex(ValueError, "Published-split"):
            self.read()


if __name__ == "__main__":
    unittest.main()
