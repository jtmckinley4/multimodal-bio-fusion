"""Synthetic kernel messages only: no kernels, notebooks or research runs execute."""

import io
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import nbformat
from nbclient.exceptions import CellExecutionComplete
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT if (ROOT / "run_encoder_swaps.py").is_file() else ROOT / "execution/arcc/encoder_swaps"
RUNTIME = WORKSPACE / "run_encoder_swaps.py"
SPEC = importlib.util.spec_from_file_location("arcc_live_output_runtime", RUNTIME)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)
LiveOutputPreprocessor = runtime.LiveOutputPreprocessor


class RecordingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def message(kind, **content):
    return {
        "msg_type": kind, "header": {"msg_type": kind},
        "parent_header": {}, "content": content,
    }


class LiveOutputTests(unittest.TestCase):
    def setUp(self):
        self.cell = nbformat.v4.new_code_cell("pass")
        notebook = nbformat.v4.new_notebook(cells=[self.cell])
        self.processor = LiveOutputPreprocessor(record_timing=False)
        self.processor.nb = notebook
        # nbclient normally initializes this when starting a cell execution.
        self.processor.clear_before_next_output = False
        self.stdout = RecordingStream()
        self.stderr = RecordingStream()
        self.enterContext(patch.object(sys, "stdout", self.stdout))
        self.enterContext(patch.object(sys, "stderr", self.stderr))

    def send(self, kind, **content):
        return self.processor.process_message(message(kind, **content), self.cell, 0)

    def test_stdout_is_live_before_cell_completion_and_keeps_partial_lines(self):
        chunks = ("    epoch  10", "  train loss 0.1234\r", "\n")
        for index, chunk in enumerate(chunks, start=1):
            output = self.send("stream", name="stdout", text=chunk)
            self.assertEqual(self.stdout.getvalue(), "".join(chunks[:index]))
            self.assertEqual(self.stdout.flush_count, index)
            self.assertIs(output, self.cell.outputs[-1])
        self.assertEqual(self.stderr.getvalue(), "")
        self.assertEqual(len(self.cell.outputs), len(chunks))
        self.assertEqual("".join(item.text for item in self.cell.outputs), "".join(chunks))

    def test_stderr_stays_separate_and_is_saved(self):
        output = self.send("stream", name="stderr", text="warning: synthetic fixture\n")
        self.assertEqual(self.stderr.getvalue(), "warning: synthetic fixture\n")
        self.assertEqual(self.stderr.flush_count, 1)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(output.name, "stderr")
        self.assertEqual(output.text, self.stderr.getvalue())

    def test_rich_display_remains_in_notebook_without_dumping_into_log(self):
        output = self.send("display_data", data={"text/plain": "synthetic table"}, metadata={})
        self.assertEqual(output.data["text/plain"], "synthetic table")
        self.assertEqual(len(self.cell.outputs), 1)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(self.stderr.getvalue(), "")

    def test_clearing_cell_output_preserves_log_history(self):
        self.send("stream", name="stdout", text="first\n")
        self.send("clear_output", wait=False)
        self.assertEqual(self.cell.outputs, [])
        self.send("stream", name="stdout", text="second\n")
        self.assertEqual(self.stdout.getvalue(), "first\nsecond\n")
        self.assertEqual([item.text for item in self.cell.outputs], ["second\n"])

    def test_idle_keeps_completion_signal(self):
        with self.assertRaises(CellExecutionComplete):
            self.send("status", execution_state="idle")

    def test_upstream_failure_propagates_before_forwarding(self):
        failure = RuntimeError("synthetic output failure")
        with patch.object(ExecutePreprocessor, "process_message", side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                self.send("stream", name="stdout", text="not processed")
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
