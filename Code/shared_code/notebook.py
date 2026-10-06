"""Helpers for presenting shared code inside notebooks."""

from __future__ import annotations

import inspect
from pathlib import Path
from urllib.parse import quote


def show_source(*objects, full=True):
    """Display shared_code functions/classes as Python source, or compact source links.

    full=True preserves the original source-block display. With full=False, links
    are relative to a notebook in Code/ and use #L anchors for source line numbers;
    jumping to those lines depends on the notebook viewer. Compact mode requires
    readable Python source inside this shared_code package and raises for unsupported
    objects or sources rather than displaying a misleading link.
    """
    from IPython.display import Markdown, display

    if full:
        blocks = [f"```python\n{inspect.getsource(o).rstrip()}\n```" for o in objects]
        text = "\n\n".join(blocks)
    else:
        package_dir = Path(__file__).resolve().parent
        links = []
        for obj in objects:
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                raise TypeError("Compact source links require functions or classes from shared_code.")
            try:
                source = inspect.unwrap(obj)
                path = Path(inspect.getsourcefile(source)).resolve()
                relative = path.relative_to(package_dir)
                _, line = inspect.getsourcelines(source)
            except (OSError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot link {obj.__qualname__}: source must be a readable file in shared_code."
                ) from error
            name = f"{obj.__module__.removeprefix('shared_code.')}.{obj.__qualname__}"
            links.append(f"[`{name}` (line {line})](shared_code/{quote(relative.as_posix())}#L{line})")
        text = "Implementation: " + ", ".join(links) if links else ""
    display(Markdown(text))
