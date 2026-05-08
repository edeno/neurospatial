"""Run a curated set of documentation code snippets to verify they execute.

This is the smoke test that backstops the v0.4 onboarding-hotfix milestone
(M0). It is *not* a doctest replacement — it deliberately checks only the
first-run snippets a new user will copy-paste from the README, the
`.claude/QUICKSTART.md` AI guide, the documentation quickstart, and the
top-level package docstring. Each snippet is enumerated explicitly in
``docs/snippets.yml`` so adding a new fragment to the docs is a deliberate
opt-in rather than a fragile auto-collection.

Usage
-----
::

    uv run python scripts/test_doc_snippets.py

Exit code
---------
0 if all non-skipped snippets execute, 1 otherwise. The script prints a
summary table at the end.

Manifest format
---------------
``docs/snippets.yml`` is a YAML mapping with a single ``snippets`` list.
Each entry has the keys:

- ``id`` (str, required): short identifier used in output.
- ``source`` (str, required): path relative to the repo root.
- ``kind`` (str, required): one of:
  - ``markdown``: extract a fenced ```python``` block from a Markdown file.
  - ``package_docstring``: extract Python REPL examples (``>>>`` / ``...``
    lines) from the top-level package docstring of ``source``.
- ``index`` (int, required for ``markdown``): zero-based index of the
  ```python``` block in the file, or zero-based index of the example
  group in the package docstring.
- ``setup`` (str, optional): Python prelude prepended to the snippet
  before execution. Use this to define fixture variables that the
  snippet depends on (e.g. ``positions`` arrays, ``times`` arrays).
- ``timeout`` (float, optional): per-snippet timeout in seconds. Default 60.
- ``skip`` (str, optional): if present, the snippet is reported as
  ``skip`` with this reason instead of executed. Use sparingly; the
  point of the test is to keep first-run snippets executable.

Adding a new entry
------------------
1. Decide on a stable ``id`` (e.g. ``readme_quickstart``).
2. Pick the lowest-cost extraction mode (``markdown`` block index is
   simpler than ``package_docstring`` group index).
3. Provide minimal ``setup`` so the snippet runs in isolation. Any fixture
   that requires external files or services either belongs in setup or
   warrants a ``skip:`` with reason.
4. Run ``uv run python scripts/test_doc_snippets.py`` locally and confirm
   the new snippet passes.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "docs" / "snippets.yml"
DEFAULT_TIMEOUT_SECONDS = 60.0

PYTHON_FENCE_RE = re.compile(
    r"^[ \t]*```python\s*$\n(?P<body>.*?)\n^[ \t]*```\s*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class SnippetSpec:
    """One entry in the snippet manifest, normalized."""

    id: str
    source: Path
    kind: str
    index: int
    setup: str
    timeout: float
    skip: str | None


@dataclass
class SnippetResult:
    """Outcome of running one snippet."""

    spec: SnippetSpec
    status: str  # "pass", "fail", "skip", "missing"
    detail: str  # short message; empty for pass


def load_manifest(path: Path) -> list[SnippetSpec]:
    """Parse the YAML manifest into normalized snippet specs.

    Raises a ValueError on missing required keys so the manifest itself
    is validated up-front rather than at execution time.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "snippets" not in raw:
        raise ValueError(f"{path} must be a YAML mapping with a 'snippets' list")
    specs: list[SnippetSpec] = []
    for entry in raw["snippets"]:
        for key in ("id", "source", "kind"):
            if key not in entry:
                raise ValueError(f"snippet entry missing '{key}': {entry!r}")
        kind = entry["kind"]
        if kind not in ("markdown", "package_docstring"):
            raise ValueError(f"snippet {entry['id']!r}: unknown kind {kind!r}")
        if "index" not in entry:
            raise ValueError(f"snippet {entry['id']!r}: 'index' is required")
        specs.append(
            SnippetSpec(
                id=str(entry["id"]),
                source=ROOT / entry["source"],
                kind=kind,
                index=int(entry["index"]),
                setup=textwrap.dedent(entry.get("setup", "") or ""),
                timeout=float(entry.get("timeout", DEFAULT_TIMEOUT_SECONDS)),
                skip=entry.get("skip"),
            )
        )
    return specs


def extract_markdown_block(text: str, index: int) -> str | None:
    """Return the body of the Nth ```python fenced block in ``text``.

    Returns None if there are fewer than ``index + 1`` blocks. Leading and
    trailing newlines on the body are stripped; internal indentation is
    preserved verbatim because list-nested fences sometimes carry a leading
    indent that is meaningful inside the snippet.
    """
    blocks = [m.group("body") for m in PYTHON_FENCE_RE.finditer(text)]
    if index < 0 or index >= len(blocks):
        return None
    return blocks[index].strip("\n")


DOCTEST_SKIP_RE = re.compile(r"#\s*doctest:\s*\+SKIP\b", re.IGNORECASE)


def extract_package_docstring_example(path: Path, index: int) -> str | None:
    """Return the Nth REPL example group from a Python module docstring.

    A "group" is a maximal run of consecutive lines that begin with ``>>> `` or
    ``... ``. Lines that are purely expected-output (no prompt) are dropped,
    matching how the Examples sections in our package docstrings are
    structured for human reading rather than strict doctest harvest.

    Lines carrying a ``# doctest: +SKIP`` directive are dropped from the
    extracted group rather than executed: the directive's whole purpose is to
    flag "do not run this in a vacuum", and silently executing such lines as
    bare Python (the ``+SKIP`` becomes a comment) was producing
    undefined-name failures the moment the manifest indexed into a Map /
    Compute / Save group. A group consisting entirely of ``+SKIP`` lines
    leaves no surviving content and is therefore not registered as a group
    at all: indexing into such a group via the manifest reports ``missing``
    (the deliberate signal "this group is not runnable in CI; mark it
    ``skip:`` in the manifest").

    SKIP semantics extend to multi-line statements. When a ``>>>`` line
    carries the directive, every following ``... `` continuation line is
    also dropped: those continuations are part of the same Python statement
    and would be syntactically orphaned without their ``>>>`` opener. The
    skip-mode flag clears at the next ``>>>`` line (the next statement) or
    at a group boundary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree)
    if docstring is None:
        return None
    lines = docstring.splitlines()
    groups: list[list[str]] = []
    current: list[str] = []
    in_skip_continuation = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(">>> "):
            payload = stripped[4:]
            if DOCTEST_SKIP_RE.search(payload):
                in_skip_continuation = True
                continue
            in_skip_continuation = False
            current.append(payload)
        elif stripped.startswith("... "):
            if in_skip_continuation:
                continue
            current.append(stripped[4:])
        else:
            in_skip_continuation = False
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    if index < 0 or index >= len(groups):
        return None
    return "\n".join(groups[index])


def build_snippet_source(spec: SnippetSpec) -> tuple[str | None, str]:
    """Return (snippet_body, error_message) for a spec.

    On success, error_message is the empty string. On failure
    (missing file, missing block, etc.) snippet_body is None and
    error_message describes the cause.
    """
    if not spec.source.exists():
        return None, f"source not found: {spec.source}"
    if spec.kind == "markdown":
        body = extract_markdown_block(
            spec.source.read_text(encoding="utf-8"), spec.index
        )
        if body is None:
            return None, f"no ```python block at index {spec.index} in {spec.source}"
        return body, ""
    if spec.kind == "package_docstring":
        body = extract_package_docstring_example(spec.source, spec.index)
        if body is None:
            return None, f"no REPL example group at index {spec.index} in {spec.source}"
        return body, ""
    return None, f"unknown kind {spec.kind!r}"


def run_snippet(spec: SnippetSpec) -> SnippetResult:
    """Execute one snippet in an isolated subprocess and return the result."""
    if spec.skip:
        return SnippetResult(spec, "skip", spec.skip)
    body, err = build_snippet_source(spec)
    if body is None:
        return SnippetResult(spec, "missing", err)
    program = (spec.setup + "\n" if spec.setup else "") + body
    # Inherit os.environ rather than scrub it: snippets need PATH for any
    # imported binary, HOME for matplotlib's font cache and several scientific
    # libraries, and SYSTEMROOT/USERPROFILE on Windows. We only force
    # MPLBACKEND=Agg so plt.show() in a snippet does not try to open a
    # display.
    snippet_env = {**os.environ, "MPLBACKEND": "Agg"}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=spec.timeout,
            text=True,
            env=snippet_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SnippetResult(spec, "fail", f"timeout after {spec.timeout:.0f}s")
    if completed.returncode != 0:
        last_lines = completed.stderr.strip().splitlines()[-6:]
        detail = "\n  ".join(last_lines) if last_lines else "non-zero exit"
        return SnippetResult(spec, "fail", detail)
    return SnippetResult(spec, "pass", "")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Parses args, runs all manifest snippets, prints summary."""
    description = (__doc__ or "Run curated doc snippets.").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="path to docs/snippets.yml (default: %(default)s)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="run only snippets whose id contains this substring",
    )
    args = parser.parse_args(argv)

    specs = load_manifest(args.manifest)
    if args.filter:
        specs = [s for s in specs if args.filter in s.id]

    results: list[SnippetResult] = []
    for spec in specs:
        print(
            f"running {spec.id} ({spec.source.relative_to(ROOT)}) ... ",
            end="",
            flush=True,
        )
        result = run_snippet(spec)
        print(result.status)
        if result.status not in ("pass", "skip"):
            indented = "\n  ".join(result.detail.splitlines())
            print(f"  {indented}")
        results.append(result)

    total = len(results)
    passed = sum(r.status == "pass" for r in results)
    failed = sum(r.status == "fail" for r in results)
    skipped = sum(r.status == "skip" for r in results)
    missing = sum(r.status == "missing" for r in results)
    print()
    print(
        f"summary: {passed} passed, {failed} failed, {skipped} skipped, "
        f"{missing} missing, {total} total"
    )
    return 0 if (failed == 0 and missing == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
