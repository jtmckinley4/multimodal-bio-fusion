"""Check historical Equation 1 verification using invented JSON and prediction arrays."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd  # Keep NumPy/pandas before the shared modules import PyTorch.
from scipy.stats import ConstantInputWarning, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import blf, token_embeddings, train

SCRIPT = ROOT / "Code/stage3/results/fungal_expression/records/verify_equation_results.py"
SPEC = importlib.util.spec_from_file_location("equation_verifier_fixture", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)  # A named import never enters the script's main report.


class EquationVerificationTests(unittest.TestCase):
    def setUp(self):
        for owner, name in (
            (train, "run"), (train, "fit"), (train, "build"), (blf, "build"),
            (blf.BenchmarkStreams, "open"), (token_embeddings, "open_shard"),
            (train.torch, "load"), (train.torch, "save"),
            (verifier, "blf_benchmark_rows"), (verifier, "git_output"), (verifier, "main"),
            (Path, "open"), (Path, "read_text"), (Path, "read_bytes"),
            (Path, "write_text"), (Path, "write_bytes"),
        ):
            self.enterContext(mock.patch.object(
                owner, name, side_effect=AssertionError("Unexpected research, artifact, or report access")))
        self.config = train.TrainConfig(tconv="equation")
        self.truth = np.array([0.1, 0.2, 0.3, 0.4])
        self.payload = {
            "benchmark": "fungal_expression", "variant": "concat",
            "config": asdict(self.config), "epochs": 3, "best_epoch": 1,
            "val_score": 0.6, "test_score": 1.0, "seconds": 1.0,
            "history": [{"epoch": 0, "val_score": 0.3}, {"epoch": 1, "val_score": 0.6},
                        {"epoch": 2, "val_score": 0.4}],
            "test_truth": self.truth.astype(np.float32).tolist(),
            "test_prediction": [1.0, 2.0, 3.0, 4.0],
        }

    def verify(self, payload):
        # Exercise Result.load's actual JSON parsing without opening a research file.
        with mock.patch.object(Path, "read_text", return_value=json.dumps(payload)):
            return verifier.verify_record(Path("invented-result.json"), "concat", self.config, self.truth)

    def predictions(self):
        # Deliberately unequal seed rankings distinguish averaging scores from averaging predictions.
        return {
            "unimodal:dna": np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                     [7, 6, 5, 4, 3, 2, 1, 0],
                                     [0, 2, 1, 3, 5, 4, 7, 6]], dtype=float),
            "concat": np.array([[0, 1, 3, 2, 4, 5, 7, 6],
                                [0, 2, 1, 4, 3, 5, 6, 7],
                                [1, 0, 2, 3, 4, 6, 5, 7]], dtype=float),
            "mil": np.array([[0, 1, 2, 4, 3, 6, 5, 7],
                             [2, 0, 1, 3, 4, 5, 6, 7],
                             [0, 1, 3, 4, 2, 5, 7, 6]], dtype=float),
            "cross": np.array([[0, 2, 1, 3, 4, 5, 6, 7],
                               [1, 0, 3, 2, 5, 4, 7, 6],
                               [0, 1, 2, 3, 5, 4, 6, 7]], dtype=float),
        }

    def test_valid_legacy_json_is_verified_without_inventing_execution_identity(self):
        before = deepcopy(self.payload)
        result = self.verify(self.payload)
        self.assertEqual(result.test_prediction, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result.test_truth, self.truth.astype(np.float32).tolist())
        self.assertIsNone(result.identity)
        self.assertEqual(self.payload, before)

    def test_nonobject_json_is_rejected_as_malformed_evidence(self):
        for payload in (None, []):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.verify(payload)

    def test_wrong_configuration_targets_score_history_or_missing_predictions_are_rejected(self):
        wrong_config = deepcopy(self.payload)
        wrong_config["config"]["tconv"] = "appendix"
        wrong_truth = deepcopy(self.payload)
        wrong_truth["test_truth"].reverse()
        wrong_score = deepcopy(self.payload)
        wrong_score["test_score"] = 0.5
        wrong_history = deepcopy(self.payload)
        wrong_history["history"][2]["epoch"] = 1
        missing_predictions = deepcopy(self.payload)
        del missing_predictions["test_prediction"]
        for name, payload in (("configuration", wrong_config), ("target order", wrong_truth),
                              ("score", wrong_score), ("history", wrong_history),
                              ("predictions", missing_predictions)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.verify(payload)

    def test_partial_seed_or_variant_cohort_is_rejected(self):
        predictions = self.predictions()
        predictions["concat"] = predictions["concat"][:2]
        with self.assertRaisesRegex(ValueError, "all three seed"):
            verifier.seed_average_bootstrap(np.arange(8.0), predictions, draws=3)
        predictions = self.predictions()
        del predictions["cross"]
        with self.assertRaisesRegex(ValueError, "four declared variants"):
            verifier.seed_average_bootstrap(np.arange(8.0), predictions, draws=3)

    def test_bootstrap_matches_an_independent_loop_over_per_seed_rank_correlations(self):
        truth, predictions = np.arange(8.0), self.predictions()
        pairs = (("concat", "unimodal:dna"), ("mil", "unimodal:dna"), ("cross", "unimodal:dna"),
                 ("cross", "mil"), ("cross", "concat"), ("mil", "concat"))
        draws, seed = 7, 23
        random = np.random.default_rng(seed)
        samples = [np.arange(8), *[random.integers(0, 8, 8) for _ in range(draws)]]
        differences = []
        for positions in samples:
            scores = {}
            for variant, seed_predictions in predictions.items():
                total = 0.0
                for values in seed_predictions:
                    total += np.corrcoef(rankdata(truth[positions]), rankdata(values[positions]))[0, 1]
                scores[variant] = total / 3
            differences.append([scores[first] - scores[second] for first, second in pairs])
        differences = np.asarray(differences)
        self.assertTrue(np.isfinite(differences).all())

        actual = verifier.seed_average_bootstrap(truth, predictions, draws=draws, seed=seed)
        self.assertEqual([(row["first"], row["second"]) for row in actual], list(pairs))
        for index, row in enumerate(actual):
            low, high = np.percentile(differences[1:, index], [2.5, 97.5])
            expected = [differences[0, index], low, high, np.mean(differences[1:, index] > 0)]
            np.testing.assert_allclose(
                [row["difference"], row["ci_low"], row["ci_high"], row["fraction_above_zero"]],
                expected, rtol=0, atol=1e-14,
            )

    def test_undefined_resampled_score_stops_instead_of_producing_an_interval(self):
        generator = mock.Mock()
        generator.integers.return_value = np.zeros(8, dtype=int)
        with mock.patch.object(verifier.np.random, "default_rng", return_value=generator), \
                self.assertWarns(ConstantInputWarning), \
                self.assertRaisesRegex(ValueError, "Bootstrap draw 0:.*undefined"):
            verifier.seed_average_bootstrap(np.arange(8.0), self.predictions(), draws=2)


if __name__ == "__main__":
    unittest.main()
