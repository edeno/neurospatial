"""Backstop test: no user-facing source file may reference the internal CLAUDE.md doc.

CLAUDE.md is an AI-assistant/developer guide that ships in the repository but
is meaningless to a library user.  Leaking its name into error messages or
docstrings that users see is a documentation boundary violation.

This test walks every .py file under src/neurospatial/ and asserts that none
of them contain the literal substring "CLAUDE.md".  It is intentionally
broad — even a comment that says "# see CLAUDE.md" would surface here so that
developers address it rather than silently shipping it.
"""

from __future__ import annotations

import pathlib


def _all_py_files() -> list[pathlib.Path]:
    src_root = pathlib.Path(__file__).parents[1] / "src" / "neurospatial"
    return sorted(src_root.rglob("*.py"))


def test_no_claude_md_references_in_source() -> None:
    """No .py file under src/neurospatial/ may contain the substring 'CLAUDE.md'."""
    offending: list[str] = []
    for py_file in _all_py_files():
        try:
            text = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if "CLAUDE.md" in text:
            offending.append(str(py_file))

    assert not offending, (
        "The following source files contain a reference to the internal "
        "developer doc 'CLAUDE.md'.  Remove or reword the reference so "
        "that users never see the internal file name in error messages, "
        "docstrings, or comments:\n" + "\n".join(f"  {f}" for f in offending)
    )
