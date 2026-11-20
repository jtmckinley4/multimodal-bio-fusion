"""Check the qualified recovery for pre-checksum BioLangFusion caches, on stub caches.

These caches are written here with the real shard writer, so the checks exercise the same
readers a benchmark run uses. What is asserted is the boundary the recovery has to hold:
a legacy cache is refused by strict validation; a reviewed record admits it and the run's
identity then carries the reconstruction label; and every way the cache or its inputs can
stop matching the record is an error rather than a silently accepted fit.

Nothing here establishes that a recovered cache holds the vectors its metadata claims. That
is a property of the extraction, not of these files, and the module says so.
"""

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import blf, blf_cache_recovery as recovery, run_identity, train  # noqa: E402
from shared_code.fusion import token_embeddings as TE  # noqa: E402

ENCODERS = dict(blf.ENCODER_OF_MODALITY)          # ntv2_100m, rnafm, esm2_8m
WIDTHS = {"dna": 6, "rna": 5, "prot": 4}
LEGACY_META = {"shard_version": 1, "chunk_size": 64, "max_len": 1024, "revision": "stub",
               "checkpoint": "stub/checkpoint"}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        rng = np.random.default_rng(0)
        self.lengths = [99, 150, 201, 102, 96, 309, 153, 198, 105, 249, 300, 93]
        letters = np.array(list("ACGT"))
        self.rows = pd.DataFrame({
            "sequence": ["".join(rng.choice(letters, size=n)) for n in self.lengths],
            "length": self.lengths,
            "Value": rng.normal(size=len(self.lengths)),
            "Split": ["train"] * 6 + ["val"] * 3 + ["test"] * 3,
        })
        self.caches, self.record_path = {}, self.root / "recovery.json"
        for modality, key in ENCODERS.items():
            directory = self.root / "caches" / modality
            directory.mkdir(parents=True)
            counts = [TE.tokens_for_length(TE.ENCODERS[key], n) for n in self.lengths]
            TE._write_shard(directory / "shard_00000.npz",
                            [rng.normal(size=(c, WIDTHS[modality])).astype(np.float16)
                             for c in counts])
            self.write_meta(modality, {**LEGACY_META, "n_shards": 1, "dtype": "float16",
                                       "encoder": key, "n_sequences": len(self.lengths)})
            self.caches[modality] = TE.summarize(directory)

    def tearDown(self):
        for cache in self.caches.values():
            cache.close()  # Windows cannot remove a shard while its mapping is open.
        self.dir.cleanup()

    def write_meta(self, modality, meta):
        (self.root / "caches" / modality / "meta.json").write_text(
            json.dumps(meta), encoding="utf-8")

    def streams(self):
        return blf.BenchmarkStreams("stub", self.rows, self.caches, "regression",
                                    encoders=dict(ENCODERS))

    def config(self):
        return train.TrainConfig(max_epochs=1, batch_size=3, channels=3,
                                 early_stopping_patience=1, tconv="equation")

    def build_record(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return recovery.write_recovery_record(self.streams(), self.record_path, **kwargs)

    def recovered(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return recovery.recovered_cache_records(self.streams(), self.record_path, **kwargs)

    # ---------------------------------------------------------------- the blocked state

    def test_strict_validation_refuses_a_cache_without_shard_checksums(self):
        """The failure a benchmark run hits today, before any recovery record exists."""
        with self.assertRaises(ValueError) as caught:
            run_identity.identity_for_streams(self.streams(), "concat", self.config())
        self.assertIn("shard checksums", str(caught.exception))

    # ---------------------------------------------------------------- the recovery route

    def test_a_reviewed_record_admits_the_cache_and_labels_the_association(self):
        record = self.build_record()
        self.assertEqual(record["schema"], recovery.RECOVERY_SCHEMA)
        self.assertEqual(set(record["caches"]), set(ENCODERS))
        caches = self.recovered()
        for modality, key in ENCODERS.items():
            cache = caches[modality]
            self.assertEqual(cache["encoder"], key)
            self.assertEqual(len(cache["shards"]), 1)
            self.assertEqual(cache["input_provenance"]["status"], recovery.RECOVERY_STATUS)
            self.assertEqual(cache["input_sha256"],
                             record["caches"][modality]["reconstructed_input_sha256"])

    def test_recovery_warns_that_producer_evidence_stays_unknown(self):
        self.build_record()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recovery.recovered_cache_records(self.streams(), self.record_path)
        self.assertTrue(any("producer details remain unknown" in str(w.message) for w in caught))

    def test_a_recovered_identity_validates_and_carries_the_label(self):
        self.build_record()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            identity = recovery.identity_for_streams(self.streams(), "concat", self.config(),
                                                     recovery=self.record_path)
        run_identity.validate_identity(identity, variant="concat", benchmark="stub")
        for cache in identity["inputs"]["caches"].values():
            self.assertEqual(cache["input_provenance"]["status"], recovery.RECOVERY_STATUS)

    def test_a_completed_result_still_reads_without_the_recovery_record(self):
        """Recovery qualifies new fits; a saved legacy result keeps its existing checks."""
        results = self.root / "results"
        results.mkdir()
        tag = "stub__concat__lambda0.0__ch3__equation__seed0"
        (results / f"{tag}.json").write_text(json.dumps({"variant": "concat", "test_score": 0.1}),
                                             encoding="utf-8")
        self.build_record()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            records = recovery.execution_records(self.streams(), "concat", self.config(), results,
                                                 recovery=self.record_path)
        self.assertIsNone(records["run_identity"])

    # ---------------------------------------------------------------- what must still fail

    def test_changed_shard_bytes_are_caught(self):
        self.build_record()
        directory = self.root / "caches" / "rna"
        self.caches["rna"].close()
        # One flipped byte in the stored vectors: shape and metadata stay as recorded.
        shard = directory / "shard_00000.npz"
        payload = bytearray(shard.read_bytes())
        payload[-1] ^= 0xFF
        shard.write_bytes(bytes(payload))
        self.caches["rna"] = TE.summarize(directory)
        with self.assertRaises(ValueError):
            self.recovered()

    def test_changed_metadata_is_caught(self):
        self.build_record()
        meta = json.loads((self.root / "caches" / "dna" / "meta.json").read_text())
        self.write_meta("dna", {**meta, "max_len": 512})
        with self.assertRaises(ValueError):
            self.recovered()

    def test_different_sequences_cannot_reuse_the_record(self):
        self.build_record()
        changed = self.rows.copy()
        changed.loc[0, "sequence"] = "T" * self.lengths[0]
        self.rows = changed
        with self.assertRaises(ValueError):
            self.recovered()

    def test_a_missing_shard_is_caught(self):
        self.build_record()
        meta = json.loads((self.root / "caches" / "prot" / "meta.json").read_text())
        self.write_meta("prot", {**meta, "n_shards": 2})
        with self.assertRaises(ValueError):
            self.recovered()

    def test_a_cache_with_extraction_provenance_is_not_a_recovery_candidate(self):
        meta = json.loads((self.root / "caches" / "dna" / "meta.json").read_text())
        self.write_meta("dna", {**meta, "input_sha256": "0" * 64,
                                "shards": [{"name": "shard_00000.npz", "sha256": "0" * 64}]})
        with self.assertRaises(ValueError):
            recovery.inspect_caches(self.streams())

    def test_a_record_written_for_another_benchmark_is_refused(self):
        record = self.build_record()
        other = self.root / "other.json"
        other.write_text(json.dumps({**record, "benchmark": "elsewhere"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            recovery.recovered_cache_records(self.streams(), other)

    def test_the_gtex_pilot_keeps_its_own_route(self):
        streams = self.streams()
        streams.tissues = ("Liver",)
        with self.assertRaises(ValueError):
            recovery.inspect_caches(streams)

    # ---------------------------------------------------------------- corroboration

    def corroborating_result(self, path, *, fingerprints):
        caches = {modality: {"encoder": key, "input_sha256": fingerprints[modality],
                             "checkpoint": "stub/checkpoint", "revision": "stub", "max_len": 1024,
                             "dtype": "float16", "n_sequences": len(self.lengths)}
                  for modality, key in ENCODERS.items()}
        path.write_text(json.dumps({"identity": {"inputs": {"caches": caches}}}), encoding="utf-8")
        return path

    def fingerprints(self):
        return {modality: TE.input_fingerprint(
                    TE.inputs_for_rows(TE.ENCODERS[key], self.rows["sequence"].tolist()))
                for modality, key in ENCODERS.items()}

    def test_an_agreeing_completed_result_is_recorded_as_corroboration(self):
        result = self.corroborating_result(self.root / "newton.json",
                                           fingerprints=self.fingerprints())
        record = self.build_record(corroborating_results=[result])
        self.assertEqual(record["corroborated_modalities"], sorted(ENCODERS))
        for modality in ENCODERS:
            self.assertEqual(record["caches"][modality]["corroboration"]["source"], "newton.json")
        caches = self.recovered()
        self.assertEqual(caches["rna"]["input_provenance"]["corroboration"]["source"],
                         "newton.json")

    def test_a_disagreeing_completed_result_stops_the_inspection(self):
        fingerprints = {**self.fingerprints(), "rna": "1" * 64}
        result = self.corroborating_result(self.root / "newton.json", fingerprints=fingerprints)
        with self.assertRaises(ValueError) as caught:
            recovery.inspect_caches(self.streams(), corroborating_results=[result])
        self.assertIn("ordered inputs", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
