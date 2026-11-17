"""Exercise only the loading-check cell with invented encoders and stub operations."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "Code/stage1/Stage1_stability.ipynb"


def loading_check_code():
    """Locate the check by its behavior; no setup or other notebook cell runs."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    candidates = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "check_rows.append(" in "".join(cell["source"])
        and "encoders.load_encoder(" in "".join(cell["source"])
    ]
    if len(candidates) != 1:
        raise AssertionError("Expected one encoder-loading check")
    return compile(ast.parse(candidates[0]), "isolated-loading-check", "exec")


class LoadingCheckTests(unittest.TestCase):
    def fixture(self, failure_stage=None, primary_error=None, cleanup_error=None):
        events = []
        namespace = {}
        encoders = [
            SimpleNamespace(key="first", label="First", max_len=3),
            SimpleNamespace(key="second", label="Second", max_len=3),
        ]

        def fail(stage):
            if failure_stage == stage:
                raise primary_error

        class Tokenizer:
            def __call__(self, text, **options):
                fail("example_tokens" if options else "full_tokens")
                count = min(len(text), options["max_length"]) if options else len(text)
                return {"input_ids": list(range(count))}

            def convert_ids_to_tokens(self, ids):
                fail("token_names")
                return [f"t{i}" for i in ids]

        def load_encoder(encoder, device):
            events.append(("load", encoder.key, device))
            fail("load")
            return Tokenizer(), SimpleNamespace(key=encoder.key)

        def encoder_input(encoder, text):
            fail("input")
            return text

        def embed(text, tokenizer, model, max_len, device):
            events.append(("embed", model.key, text, max_len, device))
            fail("embed")
            return SimpleNamespace(shape=(4,))

        def free_memory():
            # This checks owned references, not actual GPU-memory reclamation.
            self.assertNotIn("model", namespace)
            self.assertNotIn("tokenizer", namespace)
            events.append(("cleanup",))
            if cleanup_error is not None:
                raise cleanup_error

        def dataframe(rows):
            return SimpleNamespace(set_index=lambda name: rows)

        namespace.update(
            READ_SAVED_RESULTS=False,
            display=lambda value: None,
            ENCODERS=encoders,
            DEVICE="synthetic-device",
            kept_seqs=["AC", "ACGTA"],
            encoders=SimpleNamespace(
                load_encoder=load_encoder,
                encoder_input=encoder_input,
                free_memory=free_memory,
            ),
            embeddings=SimpleNamespace(embed=embed),
            pd=SimpleNamespace(DataFrame=dataframe),
        )
        return namespace, events

    def test_success_keeps_row_order_counts_and_cleanup_between_models(self):
        namespace, events = self.fixture()
        exec(loading_check_code(), namespace)
        self.assertEqual(
            namespace["check_rows"],
            [
                {
                    "encoder": label,
                    "embedding width": 4,
                    "first tokens": "t0 t1",
                    "longest input (tokens)": 5,
                    "sequences truncated": 1,
                }
                for label in ("First", "Second")
            ],
        )
        self.assertEqual([event[0] for event in events],
                         ["load", "embed", "cleanup", "load", "embed", "cleanup"])

    def test_failure_at_each_postload_step_cleans_up_and_stops(self):
        for stage in ("input", "embed", "example_tokens", "full_tokens", "token_names"):
            with self.subTest(stage=stage):
                error = RuntimeError(stage)
                namespace, events = self.fixture(stage, error)
                with self.assertRaises(RuntimeError) as caught:
                    exec(loading_check_code(), namespace)
                self.assertIs(caught.exception, error)
                self.assertEqual(namespace["check_rows"], [])
                self.assertEqual([event[0] for event in events].count("cleanup"), 1)
                self.assertEqual([event[0] for event in events].count("load"), 1)

    def test_interruption_cleans_up_and_remains_the_original_interruption(self):
        error = KeyboardInterrupt("cancelled check")
        namespace, events = self.fixture("full_tokens", error)
        with self.assertRaises(KeyboardInterrupt) as caught:
            exec(loading_check_code(), namespace)
        self.assertIs(caught.exception, error)
        self.assertEqual(events[-1], ("cleanup",))

    def test_secondary_cleanup_failure_does_not_mask_primary_failure(self):
        for error in (RuntimeError("check failed"), KeyboardInterrupt("interrupted")):
            with self.subTest(error=type(error).__name__):
                namespace, _ = self.fixture("embed", error, OSError("cleanup failed"))
                with self.assertRaises(type(error)) as caught:
                    exec(loading_check_code(), namespace)
                self.assertIs(caught.exception, error)
                if hasattr(error, "add_note"):
                    self.assertIn("Memory cleanup also failed: cleanup failed", error.__notes__)

    def test_cleanup_failure_after_success_is_reported(self):
        error = OSError("cleanup failed")
        namespace, events = self.fixture(cleanup_error=error)
        with self.assertRaises(OSError) as caught:
            exec(loading_check_code(), namespace)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(namespace["check_rows"]), 1)
        self.assertEqual([event[0] for event in events].count("load"), 1)

    def test_load_failure_does_not_cleanup_unassigned_resources(self):
        error = RuntimeError("load failed")
        namespace, events = self.fixture("load", error)
        with self.assertRaises(RuntimeError) as caught:
            exec(loading_check_code(), namespace)
        self.assertIs(caught.exception, error)
        self.assertEqual([event[0] for event in events], ["load"])


if __name__ == "__main__":
    unittest.main()
