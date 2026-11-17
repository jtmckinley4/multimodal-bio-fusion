"""Synthetic resource lifetimes only; no models, caches or research fits are opened."""

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def cell_code(notebook, cell_id):
    document = json.loads((ROOT / "Code" / notebook).read_text(encoding="utf-8"))
    cell = next(cell for cell in document["cells"] if cell.get("id") == cell_id)
    return compile("".join(cell["source"]), f"isolated-{cell_id}", "exec")


class LucaOneSmokeCleanupTests(unittest.TestCase):
    def fixture(self, fail_at=None, primary=None, cleanup_error=None):
        events = []
        namespace = {}

        def fail(stage):
            if fail_at == stage:
                raise primary

        def parameters():
            fail("parameters")
            return [SimpleNamespace(numel=lambda: 11)]

        def load(*args):
            events.append(("load", args))
            fail("load")
            return object(), SimpleNamespace(parameters=parameters)

        def pooled(text, *args):
            events.append(("pooled", text))
            fail("inference")

        def cleanup():
            self.assertNotIn("model", namespace)
            self.assertNotIn("tokenizer", namespace)
            events.append(("cleanup",))
            if cleanup_error is not None:
                raise cleanup_error

        texts = ["AC", "ACG", "ACGT", "ACGTA"]

        class Rows:
            def __len__(self):
                return len(texts)

            def __getitem__(self, name):
                if name == "sequence":
                    return SimpleNamespace(head=lambda n: SimpleNamespace(tolist=lambda: texts[:n]))
                return SimpleNamespace(min=lambda: 2, max=lambda: 5)

        ticks = iter((10, 12, 20, 28))
        namespace.update(
            READ_SAVED_RESULTS=False, BENCHMARK="invented", rows=Rows(),
            CHECKPOINT="stub", DEVICE="fake-device", ENCODER_DTYPE="fake-dtype",
            lucaone=SimpleNamespace(load=load, pooled=pooled),
            encoders=SimpleNamespace(free_memory=cleanup),
            time=SimpleNamespace(perf_counter=lambda: next(ticks)), print=lambda *args: None,
        )
        return namespace, events

    def execute(self, namespace):
        try:
            exec(cell_code("stage1/lucaone_representation_comparison.ipynb", "1941755b"), namespace)
        finally:
            self.assertNotIn("smoke_error", namespace)

    def test_success_preserves_order_and_timing_without_counting_cleanup(self):
        namespace, events = self.fixture()
        self.execute(namespace)
        self.assertEqual([event[0] for event in events], ["load"] + ["pooled"] * 4 + ["cleanup"])
        self.assertEqual([event[1] for event in events if event[0] == "pooled"], namespace["sample"])
        self.assertEqual(namespace["parameters"], 11)
        self.assertEqual(namespace["load_seconds"], 2)
        self.assertEqual(namespace["per_sequence"], 2)

    def test_failure_or_interruption_releases_references(self):
        for stage, error in (("parameters", ValueError("parameters")),
                             ("inference", RuntimeError("inference")),
                             ("inference", KeyboardInterrupt("interrupted"))):
            with self.subTest(stage=stage, error=type(error).__name__):
                namespace, events = self.fixture(stage, error)
                with self.assertRaises(type(error)) as caught:
                    self.execute(namespace)
                self.assertIs(caught.exception, error)
                self.assertEqual(events[-1], ("cleanup",))

    def test_cleanup_failure_keeps_the_primary_error(self):
        primary = RuntimeError("inference")
        namespace, _ = self.fixture("inference", primary, OSError("cleanup"))
        with self.assertRaises(RuntimeError) as caught:
            self.execute(namespace)
        self.assertIs(caught.exception, primary)
        if hasattr(primary, "add_note"):
            self.assertIn("Memory cleanup also failed: cleanup", primary.__notes__)

    def test_cleanup_failure_after_success_is_reported(self):
        error = OSError("cleanup")
        namespace, _ = self.fixture(cleanup_error=error)
        with self.assertRaises(OSError) as caught:
            self.execute(namespace)
        self.assertIs(caught.exception, error)

    def test_load_failure_has_no_returned_resources_to_release(self):
        error = RuntimeError("load")
        namespace, events = self.fixture("load", error)
        with self.assertRaises(RuntimeError) as caught:
            self.execute(namespace)
        self.assertIs(caught.exception, error)
        self.assertEqual([event[0] for event in events], ["load"])

    def test_reading_does_not_load_or_cleanup_models(self):
        namespace, events = self.fixture()
        namespace["READ_SAVED_RESULTS"] = True
        self.execute(namespace)
        self.assertEqual(events, [])


class MambaStreamCleanupTests(unittest.TestCase):
    def fixture(self, fit_error=None, cleanup_error=None, identity_error=None):
        events = []

        def opened(benchmark, root):
            events.append(("open", benchmark))

            def close():
                events.append(("close", benchmark))
                if cleanup_error is not None:
                    raise cleanup_error

            return SimpleNamespace(key=benchmark, close=close)

        def records(streams, design, *args, **kwargs):
            if identity_error is not None:
                raise identity_error
            return {"producer": design}

        def fit(streams, design, config, device, destination, **kwargs):
            events.append(("fit", streams.key, design))
            if fit_error is not None:
                raise fit_error

        namespace = dict(
            READ_SAVED_RESULTS=False, BENCHMARKS=("first", "second"), DESIGNS=("a", "b"),
            CACHE_ROOT=Path("unused-cache"), RESULTS_ROOT=Path("unused-results"),
            CODE_DIR=Path("unused-code"), CONFIG=object(), DEVICE="fake-device",
            blf=SimpleNamespace(BenchmarkStreams=SimpleNamespace(open=opened)),
            run_identity=SimpleNamespace(execution_records=records),
            train=SimpleNamespace(run=fit), time=SimpleNamespace(time=lambda: 0),
            print=lambda *args: None,
        )
        return namespace, events

    def execute(self, namespace):
        exec(cell_code("stage3/coupled_mamba_reference.ipynb", "12088db5"), namespace)

    def test_success_keeps_benchmark_and_design_order(self):
        namespace, events = self.fixture()
        self.execute(namespace)
        self.assertEqual(events, [
            ("open", "first"), ("fit", "first", "a"), ("fit", "first", "b"), ("close", "first"),
            ("open", "second"), ("fit", "second", "a"), ("fit", "second", "b"), ("close", "second"),
        ])

    def test_fit_failure_or_interruption_survives_cleanup_failure(self):
        for error in (RuntimeError("fit"), KeyboardInterrupt("cancel")):
            with self.subTest(error=type(error).__name__):
                namespace, events = self.fixture(error, OSError("close"))
                with self.assertRaises(type(error)) as caught:
                    self.execute(namespace)
                self.assertIs(caught.exception, error)
                self.assertEqual(events, [("open", "first"), ("fit", "first", "a"), ("close", "first")])
                if hasattr(error, "add_note"):
                    self.assertIn("Stream cleanup also failed: close", error.__notes__)

    def test_identity_failure_also_closes_streams(self):
        error = ValueError("identity")
        namespace, events = self.fixture(identity_error=error)
        with self.assertRaises(ValueError) as caught:
            self.execute(namespace)
        self.assertIs(caught.exception, error)
        self.assertEqual(events, [("open", "first"), ("close", "first")])

    def test_cleanup_failure_after_success_is_reported(self):
        error = OSError("close")
        namespace, events = self.fixture(cleanup_error=error)
        with self.assertRaises(OSError) as caught:
            self.execute(namespace)
        self.assertIs(caught.exception, error)
        self.assertEqual(events[-1], ("close", "first"))
        self.assertNotIn(("open", "second"), events)


def stream_class(summarize):
    """Load only the actual stream owner definitions, with invented external inputs."""
    source = (ROOT / "Code/shared_code/fusion/blf.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BenchmarkStreams"]
    namespace = dict(
        dataclass=dataclass, field=field, summarize=summarize,
        ENCODER_OF_MODALITY={"dna": "dna_encoder", "rna": "rna_encoder", "prot": "protein_encoder"},
        datasets_module=SimpleNamespace(
            DATASETS={"invented": SimpleNamespace(task="regression", label_column="Value", split_column="Split")},
            blf_benchmark_rows=lambda key: object(),
        ),
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), "isolated-blf-streams", "exec"), namespace)
    return namespace["BenchmarkStreams"]


class BlfStreamCleanupTests(unittest.TestCase):
    def fixture(self, fail_open=None, primary=None, close_errors=None, check_error=None):
        events = []
        close_errors = close_errors or {}

        def summarize(path):
            encoder = path.rsplit("/", 1)[-1]
            events.append(("open", encoder))
            if encoder == fail_open:
                raise primary

            def close():
                events.append(("close", encoder))
                if encoder in close_errors:
                    raise close_errors[encoder]

            return SimpleNamespace(close=close)

        cls = stream_class(summarize)

        def check(streams):
            events.append(("check", tuple(streams.caches)))
            if check_error is not None:
                raise check_error

        cls.check = check
        return cls, events

    def test_success_leaves_caches_open_until_owner_closes(self):
        cls, events = self.fixture()
        streams = cls.open("invented", "unused")
        self.assertEqual(events, [("open", "dna_encoder"), ("open", "rna_encoder"),
                                  ("open", "protein_encoder"), ("check", ("dna", "rna", "prot"))])
        streams.close()
        self.assertEqual(events[-3:], [("close", "dna_encoder"), ("close", "rna_encoder"),
                                       ("close", "protein_encoder")])

    def test_failed_later_open_closes_only_previously_returned_caches(self):
        for encoder, closed in (("rna_encoder", ["dna_encoder"]),
                                ("protein_encoder", ["dna_encoder", "rna_encoder"])):
            with self.subTest(encoder=encoder):
                error = RuntimeError("open")
                cls, events = self.fixture(encoder, error)
                with self.assertRaises(RuntimeError) as caught:
                    cls.open("invented", "unused")
                self.assertIs(caught.exception, error)
                self.assertEqual([name for action, name in events if action == "close"], closed)

    def test_check_failure_or_interruption_closes_every_cache(self):
        for error in (ValueError("check"), KeyboardInterrupt("cancel")):
            with self.subTest(error=type(error).__name__):
                cls, events = self.fixture(check_error=error)
                with self.assertRaises(type(error)) as caught:
                    cls.open("invented", "unused")
                self.assertIs(caught.exception, error)
                self.assertEqual([name for action, name in events if action == "close"],
                                 ["dna_encoder", "rna_encoder", "protein_encoder"])

    def test_failed_open_keeps_primary_error_when_multiple_closes_fail(self):
        error = RuntimeError("open")
        cls, events = self.fixture("protein_encoder", error,
                                  {"dna_encoder": OSError("dna close"), "rna_encoder": OSError("rna close")})
        with self.assertRaises(RuntimeError) as caught:
            cls.open("invented", "unused")
        self.assertIs(caught.exception, error)
        self.assertEqual(events[-2:], [("close", "dna_encoder"), ("close", "rna_encoder")])
        if hasattr(error, "add_note"):
            self.assertTrue(any("dna close" in note for note in error.__notes__))
            self.assertTrue(any("rna close" in note for note in error.__notes__))

    def test_close_attempts_all_and_raises_first_failure(self):
        first = OSError("dna close")
        cls, events = self.fixture(close_errors={"dna_encoder": first, "rna_encoder": OSError("rna close")})
        streams = cls.open("invented", "unused")
        with self.assertRaises(OSError) as caught:
            streams.close()
        self.assertIs(caught.exception, first)
        self.assertEqual(events[-3:], [("close", "dna_encoder"), ("close", "rna_encoder"),
                                       ("close", "protein_encoder")])


if __name__ == "__main__":
    unittest.main()
