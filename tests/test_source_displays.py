"""Check source displays without calling the displayed research functions."""

import ast
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
from urllib.parse import quote, unquote


REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "Code"
sys.path.insert(0, str(CODE))

# Preserve the notebook environment's NumPy-before-PyTorch import order on Windows.
import numpy as np  # noqa: F401, E402
from shared_code import embeddings, encoders, sequences  # noqa: E402
from shared_code.notebook import show_source  # noqa: E402


def _outside_package():
    raise AssertionError("Source-display targets must not execute.")


def _definition(path, qualified_name):
    """Find a named definition and its first decorator directly in the source."""
    text = path.read_text(encoding="utf-8")
    body = ast.parse(text, filename=str(path)).body
    for name in qualified_name.split("."):
        matches = [node for node in body if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and node.name == name]
        if len(matches) != 1:
            raise AssertionError(f"Expected one definition of {qualified_name} in {path}.")
        node = matches[0]
        body = node.body
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    source = "\n".join(text.splitlines()[start - 1:node.end_lineno])
    return start, source


def _markdown(value):
    return "".join(value) if isinstance(value, list) else value


class SourceDisplayTests(unittest.TestCase):
    # These cover an ordinary function, a decorated function, and a decorated class.
    targets = (sequences.gc_content, embeddings.embed, encoders.Encoder)

    def capture(self, *objects, **options):
        with patch("IPython.display.display") as display:
            show_source(*objects, **options)
        display.assert_called_once()
        return display.call_args.args[0].data

    def expected_full(self):
        blocks = []
        for obj in self.targets:
            path = CODE.joinpath(*obj.__module__.split(".")).with_suffix(".py")
            _, source = _definition(path, obj.__qualname__)
            blocks.append(f"```python\n{source}\n```")
        return "\n\n".join(blocks)

    def test_full_source_is_the_default(self):
        self.assertEqual(self.capture(*self.targets), self.expected_full())

    def test_explicit_full_source(self):
        self.assertEqual(self.capture(*self.targets, full=True), self.expected_full())

    def test_compact_links_name_the_definition_and_first_decorator(self):
        links = []
        for obj in self.targets:
            path = CODE.joinpath(*obj.__module__.split(".")).with_suffix(".py")
            line, _ = _definition(path, obj.__qualname__)
            name = f"{obj.__module__.removeprefix('shared_code.')}.{obj.__qualname__}"
            url = quote(path.relative_to(CODE).as_posix())
            links.append(f"[`{name}` (line {line})]({url}#L{line})")
        self.assertEqual(self.capture(*self.targets, full=False),
                         "Implementation: " + ", ".join(links))

    def test_unsupported_compact_sources_fail_without_display(self):
        namespace = {}
        exec(compile("def missing_source():\n    pass\n", "<missing-source>", "exec"), namespace)
        cases = ((len, TypeError), (object(), TypeError),
                 (_outside_package, ValueError), (namespace["missing_source"], ValueError))
        for obj, error in cases:
            with self.subTest(source=repr(obj)), patch("IPython.display.display") as display:
                with self.assertRaises(error):
                    show_source(obj, full=False)
                display.assert_not_called()


class SavedSourceLinkTests(unittest.TestCase):
    def test_saved_compact_links_still_identify_the_displayed_definitions(self):
        pattern = re.compile(r"\[`([^`]+)` \(line (\d+)\)\]\(([^)]+)#L(\d+)\)")
        for notebook_name in ("Stage1_stability.ipynb", "Stage1_gtex.ipynb"):
            notebook = json.loads((CODE / notebook_name).read_text(encoding="utf-8"))
            checked = 0
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse(_markdown(cell["source"]))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "show_source"]
                if not calls:
                    continue
                with self.subTest(notebook=notebook_name, cell=index):
                    expected_names = [ast.unparse(arg).removeprefix("shared_code.")
                                      for call in calls for arg in call.args]
                    saved = [_markdown(output.get("data", {}).get("text/markdown", ""))
                             for output in cell.get("outputs", [])]
                    compact = [text for text in saved if text.startswith("Implementation: ")]
                    self.assertTrue(compact, "Refresh this cell's compact source display.")
                    links = [match for text in compact for match in pattern.findall(text)]
                    self.assertEqual([link[0] for link in links], expected_names)
                    for name, label_line, url, anchor_line in links:
                        path = (CODE / unquote(url)).resolve()
                        self.assertTrue(path.is_relative_to((CODE / "shared_code").resolve()))
                        module = ".".join(path.relative_to(CODE / "shared_code").with_suffix("").parts)
                        self.assertTrue(name.startswith(module + "."), f"Wrong source file for {name}.")
                        line, _ = _definition(path, name[len(module) + 1:])
                        self.assertEqual(int(label_line), line, f"Stale source label for {name}.")
                        self.assertEqual(int(anchor_line), line, f"Stale source anchor for {name}.")
                    checked += len(links)
            self.assertGreater(checked, 0, f"No source links checked in {notebook_name}.")


if __name__ == "__main__":
    unittest.main()
