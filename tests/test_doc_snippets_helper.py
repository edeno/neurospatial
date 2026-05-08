"""Regression tests for ``scripts/test_doc_snippets.py``.

The doc-snippet smoke runner is itself a piece of CI infrastructure: it
needs to stay strict (fail loudly on a broken snippet), accurate (treat a
missing block as a hard failure rather than a silent skip), and stable
(its argparse / manifest plumbing should keep working). These tests
cover each of those three properties with small synthetic manifests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "test_doc_snippets.py"


def _load_runner_module():
    """Import scripts/test_doc_snippets.py as a module, once per session."""
    if "test_doc_snippets" in sys.modules:
        return sys.modules["test_doc_snippets"]
    spec = importlib.util.spec_from_file_location("test_doc_snippets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_doc_snippets"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    """The imported runner module — reused across tests in this file."""
    return _load_runner_module()


def _write_manifest(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_pass_on_runnable_snippet(tmp_path, runner):
    """A trivial snippet that prints 'ok' should report status='pass'."""
    md = tmp_path / "doc.md"
    md.write_text("```python\nprint('ok')\n```\n", encoding="utf-8")
    manifest = tmp_path / "snippets.yml"
    _write_manifest(
        manifest,
        f"snippets:\n  - id: ok\n    source: {md}\n    kind: markdown\n    index: 0\n",
    )
    specs = runner.load_manifest(manifest)
    assert len(specs) == 1
    result = runner.run_snippet(specs[0])
    assert result.status == "pass", result.detail


def test_fail_on_broken_snippet(tmp_path, runner):
    """A snippet that raises must report status='fail' with a non-empty detail."""
    md = tmp_path / "doc.md"
    md.write_text("```python\nraise RuntimeError('boom')\n```\n", encoding="utf-8")
    manifest = tmp_path / "snippets.yml"
    _write_manifest(
        manifest,
        f"snippets:\n  - id: broken\n    source: {md}\n    kind: markdown\n    index: 0\n",
    )
    specs = runner.load_manifest(manifest)
    result = runner.run_snippet(specs[0])
    assert result.status == "fail"
    assert "RuntimeError" in result.detail or "boom" in result.detail


def test_missing_block_index_reports_missing(tmp_path, runner):
    """Pointing at index N when the file has fewer than N+1 blocks must not pass."""
    md = tmp_path / "doc.md"
    md.write_text("```python\nprint('only one block')\n```\n", encoding="utf-8")
    manifest = tmp_path / "snippets.yml"
    _write_manifest(
        manifest,
        f"snippets:\n  - id: oob\n    source: {md}\n    kind: markdown\n    index: 5\n",
    )
    specs = runner.load_manifest(manifest)
    result = runner.run_snippet(specs[0])
    assert result.status == "missing"
    assert "index 5" in result.detail


def test_skip_entry_is_honored(tmp_path, runner):
    """An entry with a ``skip`` reason must be reported as skip, not fail."""
    md = tmp_path / "doc.md"
    md.write_text("```python\nprint('would-have-run')\n```\n", encoding="utf-8")
    manifest = tmp_path / "snippets.yml"
    _write_manifest(
        manifest,
        f"snippets:\n  - id: skipme\n    source: {md}\n    kind: markdown\n    index: 0\n"
        f"    skip: 'requires external dataset'\n",
    )
    specs = runner.load_manifest(manifest)
    result = runner.run_snippet(specs[0])
    assert result.status == "skip"
    assert "external dataset" in result.detail


def test_setup_is_prepended_to_snippet(tmp_path, runner):
    """The ``setup`` block runs before the snippet, providing fixture vars."""
    md = tmp_path / "doc.md"
    md.write_text(
        "```python\nassert MAGIC == 42, 'setup did not run'\n```\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "snippets.yml"
    _write_manifest(
        manifest,
        "snippets:\n"
        f"  - id: with_setup\n    source: {md}\n    kind: markdown\n    index: 0\n"
        "    setup: |\n      MAGIC = 42\n",
    )
    specs = runner.load_manifest(manifest)
    result = runner.run_snippet(specs[0])
    assert result.status == "pass", result.detail


def test_package_docstring_extraction(tmp_path, runner):
    """REPL example groups in a module docstring are extractable by index."""
    py = tmp_path / "tiny_module.py"
    py.write_text(
        '"""Module docstring.\n\n'
        "First example::\n\n"
        "    >>> x = 1\n"
        "    >>> assert x == 1\n\n"
        "Second example::\n\n"
        "    >>> y = 2\n"
        "    >>> assert y == 2\n"
        '"""\n',
        encoding="utf-8",
    )
    body = runner.extract_package_docstring_example(py, index=1)
    assert body is not None
    assert "y = 2" in body and "y == 2" in body
    out_of_range = runner.extract_package_docstring_example(py, index=5)
    assert out_of_range is None


def test_real_manifest_round_trip(runner):
    """The shipped manifest parses cleanly and lists the curated entries."""
    manifest_path = ROOT / "docs" / "snippets.yml"
    specs = runner.load_manifest(manifest_path)
    ids = {s.id for s in specs}
    # Sanity: at least the canonical first-run entries are present. Other
    # entries are allowed to come and go without breaking this test.
    assert "readme_quickstart" in ids
    assert "docs_quickstart_basic_creation" in ids
