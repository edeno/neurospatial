#!/usr/bin/env python
"""Write ``docs/llms.txt``, a plain-text map of the public API for assistants.

The file lists every public name pinned in ``tests/data/public_api.txt`` with
the first line of its docstring, after the scientific conventions and the
canonical argument order from ``CLAUDE.md``. ``tests/test_llms_txt.py`` fails
when the committed file is out of date.

Usage
-----
    uv run python scripts/generate_llms_txt.py
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "data" / "public_api.txt"
CLAUDE_MD = ROOT / "CLAUDE.md"
OUTPUT = ROOT / "docs" / "llms.txt"
SECTIONS = ("Scientific conventions", "Canonical argument order")
INTRO = (
    "# neurospatial\n\n"
    "> Discretize spatial environments into bins with connectivity graphs and run "
    "spatial neuroscience analyses on them: neural encoding, decoding, behavior, "
    "peri-event analysis, simulation, animation and NWB I/O.\n\n"
    "Import every name from the package path listed below. Documentation: "
    "https://edeno.github.io/neurospatial/\n"
)


def _section(text: str, title: str) -> str:
    """Return the body of the ``## title`` section of ``CLAUDE.md``."""
    match = re.search(
        rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    if match is None:
        msg = f"CLAUDE.md has no '## {title}' section"
        raise ValueError(msg)
    # Drop the sentence that points at contributor-only rules.
    body = re.sub(
        r"\s*The full public API contract.*", "", match.group(1), flags=re.DOTALL
    )
    return body.strip()


def _summary(obj: object, name: str) -> str:
    """First docstring paragraph of a class, function or module; else empty."""
    # Aliases and constants carry the docstring of their type, which varies by
    # Python version.
    if not (inspect.isclass(obj) or inspect.isroutine(obj) or inspect.ismodule(obj)):
        return ""
    doc = obj.__doc__
    if not isinstance(doc, str):
        return ""
    first = inspect.cleandoc(doc).split("\n\n", 1)[0].replace("\n", " ").strip()
    # A dataclass without a docstring reports its generated signature.
    return "" if first.startswith(f"{name}(") else first


def render() -> str:
    """Return the text of ``llms.txt``."""
    claude = CLAUDE_MD.read_text(encoding="utf-8")
    parts = [INTRO]
    parts.extend(f"## {title}\n\n{_section(claude, title)}\n" for title in SECTIONS)
    parts.append("## Public API\n")
    names: dict[str, list[str]] = {}
    for line in SNAPSHOT.read_text(encoding="utf-8").splitlines():
        path = re.split(r"\(| \(", line, maxsplit=1)[0].strip()
        module_name, name = path.rsplit(".", 1)
        names.setdefault(module_name, []).append(name)
    for namespace in sorted(names):
        parts.append(f"\n### {namespace}\n")
        parts.extend(_entry(namespace, name) for name in sorted(names[namespace]))
    return "\n".join(parts) + "\n"


def _entry(module_name: str, name: str) -> str:
    """One list line: the name, and its summary when the docstring has one."""
    obj = getattr(importlib.import_module(module_name), name)
    label = f"`{name}`" + (" (class)" if inspect.isclass(obj) else "")
    summary = _summary(obj, name)
    return f"- {label}: {summary}" if summary else f"- {label}"


if __name__ == "__main__":
    OUTPUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
