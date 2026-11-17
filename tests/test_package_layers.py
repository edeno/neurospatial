"""Packages meet at their public names (docs/contributing.md, "Layers").

Two parts of that rule are checked here, over every module in ``src``:

- a package imports another package's names from that package's
  ``__init__`` (or a public submodule), never from one of its ``_`` modules;
- code outside ``neurospatial.environment`` reads an ``Environment`` only
  through its public members.

``tests/data/layering_exceptions.txt`` lists the places that do not follow
the rule yet. The list only shrinks: a place that is not listed fails, and so
does a listed place that no longer exists, so a fixed exception is removed
from the list in the same change. The direction of the layers (which package
may import which) is still reviewed by hand.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "neurospatial"
EXCEPTIONS = ROOT / "tests" / "data" / "layering_exceptions.txt"

# The state an Environment owns; everything else reads its public members.
PRIVATE_ENVIRONMENT_MEMBERS = frozenset(
    {
        "_arena",
        "_binning",
        "_cache",
        "_geometry",
        "_landmarks",
        "_metadata",
        "_provenance",
        "_regions",
    }
)


def _packages() -> set[str]:
    return {path.name for path in SRC.iterdir() if (path / "__init__.py").exists()}


def _is_environment(expression: ast.expr) -> bool:
    """Whether ``expression`` is named as an environment (``env``, ``rec.env``)."""
    if isinstance(expression, ast.Attribute):
        name = expression.attr
    elif isinstance(expression, ast.Name):
        name = expression.id
    else:
        return False
    return "env" in name.lower()


def _found() -> set[str]:
    """Every place in ``src`` that breaks one of the two rules, one per line."""
    packages = _packages()
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        owner = relative.parts[0] if relative.parts[0] in packages else None
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                modules = []
            for module in modules:
                parts = module.split(".")
                if (
                    len(parts) > 2
                    and parts[0] == "neurospatial"
                    and parts[1] in packages
                    and parts[1] != owner
                    and any(part.startswith("_") for part in parts[2:])
                ):
                    found.add(f"import {relative.as_posix()} {module}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in PRIVATE_ENVIRONMENT_MEMBERS
                and owner != "environment"
                and _is_environment(node.value)
            ):
                found.add(f"attribute {relative.as_posix()} {ast.unparse(node)}")
    return found


def _listed() -> set[str]:
    lines = (
        line.strip() for line in EXCEPTIONS.read_text(encoding="utf-8").splitlines()
    )
    return {line for line in lines if line and not line.startswith("#")}


def test_no_new_private_imports_or_private_environment_reads() -> None:
    new = sorted(_found() - _listed())
    assert not new, (
        "These places reach into another package's private module, or read an "
        "Environment's private state, and are not in "
        "tests/data/layering_exceptions.txt:\n  "
        + "\n  ".join(new)
        + "\nImport the name from the other package's __init__ (re-export it "
        "there under its own name if another package needs it), and read the "
        "environment through a public member such as env.binning or "
        "env.get_diffusion_geometry()."
    )


def test_fixed_exceptions_leave_the_list() -> None:
    stale = sorted(_listed() - _found())
    assert not stale, (
        "These lines of tests/data/layering_exceptions.txt no longer match the "
        "source, so the exception is fixed; remove them:\n  " + "\n  ".join(stale)
    )
