"""Check Step 3 execution switches without opening research caches or training."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass, replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Code/stage3/03_blf_build_check.ipynb"
SWITCHES = ("RUN_TIMING", "RUN_PROBE", "RUN_MAIN_SWEEP", "RUN_EQUATION_SWEEP", "RUN_WIDE_HEAD")


@dataclass
class Config:
    """Only configuration fields used by these guarded notebook cells."""

    batch_size: int
    max_epochs: int
    channels: int
    tconv: str
    seed: int
    entropy_lambda: float = 0.0
    early_stopping_patience: int = 50


class OperationReached(Exception):
    """Stop an enabled branch at its first intended compute operation."""


class Stage3ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    def cell(self, marker):
        matches = [cell for cell in self.cells if cell["cell_type"] == "code"
                   and marker in "".join(cell["source"])]
        self.assertEqual(len(matches), 1, f"Expected one code cell containing {marker!r}")
        return matches[0]

    def execute(self, cell, values):
        # Unexpected file publication fails even if a future edit bypasses the stubs.
        with ExitStack() as guards, redirect_stdout(io.StringIO()) as output:
            for owner, name in ((Path, "open"), (Path, "mkdir"),
                                (Path, "write_text"), (Path, "write_bytes")):
                guards.enter_context(patch.object(owner, name, side_effect=AssertionError("File I/O")))
            guards.enter_context(patch("builtins.open", side_effect=AssertionError("File I/O")))
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec"), values)
        return output.getvalue()

    def namespace(self, *, mps=False, cuda=False):
        forbidden = lambda: Mock(side_effect=AssertionError("Unexpected research operation"))
        values = {
            "__name__": "__main__", "CODE_DIR": ROOT / "Code", "replace": replace,
            "torch": SimpleNamespace(
                __version__="test", manual_seed=Mock(), device=lambda name: name,
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                cuda=SimpleNamespace(is_available=lambda: cuda,
                                     get_device_name=Mock(return_value="Test CUDA GPU"))),
            "train": SimpleNamespace(TrainConfig=Config, run=forbidden(), fit=forbidden(),
                                     run_identity=forbidden()),
            "blf": SimpleNamespace(BenchmarkStreams=SimpleNamespace(open=forbidden()),
                                   build=forbidden(), loader=forbidden()),
            "time": SimpleNamespace(perf_counter=Mock(return_value=0.0)),
        }
        self.execute(self.cell("RUN_TIMING ="), values)
        self.execute(self.cell('BENCHMARK = "fungal_expression"'), values)
        return values

    def test_code_compiles_and_controls_are_in_setup(self):
        for cell in self.cells:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec")
        controls = self.cell("RUN_TIMING =")
        preceding_sections = ["".join(cell["source"]).splitlines()[0]
                              for cell in self.cells[:self.cells.index(controls)]
                              if cell["cell_type"] == "markdown"
                              and "".join(cell["source"]).startswith("## ")]
        self.assertEqual(preceding_sections[-1], "## Setup")

    def test_device_availability_and_preserved_preference(self):
        for mps, cuda, expected in ((False, False, "cpu"), (False, True, "cuda"),
                                    (True, False, "mps"), (True, True, "mps")):
            with self.subTest(mps=mps, cuda=cuda):
                values = self.namespace(mps=mps, cuda=cuda)
                output = self.execute(self.cell("RUN_TIMING ="), values)
                self.assertEqual(values["DEVICE"], expected)
                self.assertIn("CUDA", output)
                self.assertIn("MPS", output)
                self.assertIn(expected, output)
                if expected == "cuda":
                    self.assertIn("Test CUDA GPU", output)

    def test_defaults_skip_compute_and_clear_stale_state(self):
        values = self.namespace()
        self.assertTrue(all(values[name] is False for name in SWITCHES))
        stale_streams = SimpleNamespace(close=Mock())
        values.update(STREAMS=stale_streams, timing=object(), timing_rows=[object()],
                      variants_to_time=["old"], sweep_minutes_per_epoch=99,
                      probe=object(), probe_curve=object(), elapsed_hours=99,
                      equation_elapsed_hours=99, wide_result=object())
        output = self.execute(self.cell("STREAMS = blf.BenchmarkStreams.open("), values)
        stale_streams.close.assert_called_once_with()
        self.assertIsNone(values["STREAMS"])
        for path in ("CACHE_ROOT", "RESULTS_DIR", "COMPARISON_DIR"):
            self.assertIn(str(values[path]), output)
        self.execute(self.cell("def time_variant("), values)
        for switch in SWITCHES:
            self.execute(self.cell(f"if {switch}:"), values)
        for name in ("timing", "sweep_minutes_per_epoch", "probe", "probe_curve",
                     "elapsed_hours", "equation_elapsed_hours", "wide_result"):
            self.assertIsNone(values[name], name)
        self.assertEqual(values["timing_rows"], [])
        self.assertEqual(values["variants_to_time"], [])
        for operation in (values["blf"].BenchmarkStreams.open, values["blf"].build,
                          values["blf"].loader, values["train"].run, values["train"].fit,
                          values["train"].run_identity):
            operation.assert_not_called()

    def test_each_switch_opens_only_the_requested_cache(self):
        for switch in SWITCHES:
            with self.subTest(switch=switch):
                values = self.namespace()
                values[switch] = True
                streams = SimpleNamespace(rows=[0], task="regression", widths={},
                                          split_rows=lambda name: [0])
                opener = Mock(return_value=streams)
                values["blf"].BenchmarkStreams.open = opener
                self.execute(self.cell("STREAMS = blf.BenchmarkStreams.open("), values)
                opener.assert_called_once_with(values["BENCHMARK"], values["CACHE_ROOT"],
                                               allow_legacy_cache=False)
                self.assertIs(values["STREAMS"], streams)

    def test_enabled_compute_requires_open_streams(self):
        for switch in SWITCHES:
            with self.subTest(switch=switch):
                values = self.namespace()
                values.update({switch: True, "STREAMS": None})
                with self.assertRaisesRegex(RuntimeError, "[Cc]ache|STREAMS"):
                    self.execute(self.cell(f"if {switch}:"), values)

    def test_enabled_switches_reach_their_intended_operations(self):
        for switch in SWITCHES:
            with self.subTest(switch=switch):
                values = self.namespace()
                values.update({switch: True, "STREAMS": object()})
                stop = Mock(side_effect=OperationReached)
                if switch == "RUN_TIMING":
                    values["time_variant"] = stop
                elif switch == "RUN_PROBE":
                    values["blf"].build = stop
                else:
                    values["train"].run = stop
                if switch == "RUN_WIDE_HEAD":
                    identity = {"test": "matching input identity"}
                    values.update(chosen_lambda=0.5, missing_weights=[], main_results=[
                        SimpleNamespace(variant="mil", config={"entropy_lambda": 0.5},
                                        identity=identity)])
                    values["train"].run_identity = Mock(return_value=identity)
                with self.assertRaises(OperationReached):
                    self.execute(self.cell(f"if {switch}:"), values)
                stop.assert_called_once()
                if switch == "RUN_TIMING":
                    self.assertEqual(stop.call_args.args, ("unimodal:prot",))
                elif switch == "RUN_PROBE":
                    self.assertEqual(stop.call_args.args[1], "unimodal:prot")
                else:
                    expected = "concat" if switch == "RUN_EQUATION_SWEEP" else "unimodal:prot"
                    if switch == "RUN_WIDE_HEAD":
                        expected = "mil"
                        self.assertEqual(stop.call_args.args[2].channels, 1280)
                    self.assertEqual(stop.call_args.args[1], expected)
                    if switch == "RUN_EQUATION_SWEEP":
                        self.assertEqual(stop.call_args.args[2].tconv, "equation")


if __name__ == "__main__":
    unittest.main()
