"""The documentation teaches public import paths only."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest
from test_public_api_snapshot import NAMESPACES

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "neurospatial"
# Not built by mkdocs (see ``exclude_docs``): history and research notes.
UNBUILT_DOCS = ("plans", "reviews", "research")
# The per-module reference pages are generated at build time, keyed by module;
# the contributor and profiling notes describe internals.
INTERNAL_DOCS = ("api/neurospatial/", "contributing.md", "PERFORMANCE_PROFILING.md")


def _modules() -> set[str]:
    """Every dotted module and package name under ``src/neurospatial``."""
    names = {"neurospatial"}
    for path in SRC.rglob("*.py"):
        parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.add(".".join(parts))
    return names


MODULES = _modules()
PRIVATE = MODULES - set(NAMESPACES)

_DOTTED = re.compile(r"\bneurospatial((?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
# A backticked dotted name that starts at a package of neurospatial without the
# ``neurospatial.`` prefix, such as ``simulation.models``. Requiring the first
# part to be a real package keeps ordinary attribute access (``result.plot``)
# from matching.
_PACKAGES = {name.split(".")[1] for name in MODULES if name.count(".") >= 1}
_BARE_DOTTED = re.compile(
    rf"`((?:{'|'.join(sorted(_PACKAGES))})(?:\.[a-z_][a-z0-9_]*)+)[^`]*`"
)
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_FROM_IMPORT = re.compile(
    r"(?m)^[ \t>]*from\s+(neurospatial(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+"
    r"(\([^)]*\)|[^\n(]+)"
)


def _documents() -> list[tuple[str, str]]:
    """``(label, text)`` for the README, built docs and package docstrings."""
    documents = [("README.md", (ROOT / "README.md").read_text(encoding="utf-8"))]
    for path in sorted((ROOT / "docs").rglob("*.md")):
        relative = path.relative_to(ROOT / "docs")
        if relative.parts[0] in UNBUILT_DOCS or relative.as_posix().startswith(
            INTERNAL_DOCS
        ):
            continue
        documents.append(
            (f"docs/{relative.as_posix()}", path.read_text(encoding="utf-8"))
        )
    for path in sorted(SRC.rglob("__init__.py")):
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        if docstring:
            documents.append((str(path.relative_to(ROOT)), docstring))
    return documents


def _owner(parts: list[str]) -> str:
    """The longest prefix of a dotted name that is a module of the package."""
    return next(
        ".".join(parts[:n])
        for n in range(len(parts), 0, -1)
        if ".".join(parts[:n]) in MODULES
    )


def _exported(module: str, name: str) -> bool:
    return name in importlib.import_module(module).__all__


def private_paths(text: str) -> set[str]:
    """Return what a document names that is not public.

    That is a private module, written with or without the ``neurospatial.``
    prefix, or a name imported from a public module that its ``__all__`` does
    not list.
    """
    text = _LINK_TARGET.sub("]", text)
    found = set()
    for match in _DOTTED.finditer(text):
        owner = _owner(("neurospatial" + match.group(1)).split("."))
        if owner in PRIVATE:
            found.add(owner)
    for match in _BARE_DOTTED.finditer(text):
        owner = _owner(["neurospatial", *match.group(1).split(".")])
        if owner in PRIVATE:
            found.add(owner)
    for match in _FROM_IMPORT.finditer(text):
        package = match.group(1)
        listed = re.sub(r"#[^\n]*", "", match.group(2))
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", listed):
            if name == "as":
                continue
            if f"{package}.{name}" in PRIVATE:
                found.add(f"{package}.{name}")
            elif package in NAMESPACES and not _exported(package, name):
                found.add(f"{package}.{name} (not in __all__)")
    return found


def test_private_path_scanner_flags_private_modules() -> None:
    text = (
        "from neurospatial.encoding.grid import grid_score\n"
        "from neurospatial.behavior import segmentation, detect_laps\n"
        "see neurospatial.ops.egocentric.allocentric_to_egocentric\n"
    )
    assert private_paths(text) == {
        "neurospatial.encoding.grid",
        "neurospatial.behavior.segmentation",
        "neurospatial.ops.egocentric",
    }
    assert private_paths(
        "the `simulation.models` and `ops.mapping.regions_to_mask`"
    ) == {
        "neurospatial.simulation.models",
        "neurospatial.ops.mapping",
    }
    assert private_paths("from neurospatial.replay.methods import detect_replay") == {
        "neurospatial.replay.methods.detect_replay (not in __all__)"
    }
    assert not private_paths(
        "from neurospatial.encoding import compute_spatial_rate\n"
        "from neurospatial.replay.methods import DAVIDSON_2009\n"
        "neurospatial.layout.ScaleBarConfig\n"
        "`result.plot` and `simulation.PlaceCellModel` and `np.random.default_rng`\n"
    )


@pytest.mark.parametrize(
    ("label", "text"), _documents(), ids=[label for label, _ in _documents()]
)
def test_docs_teach_public_paths(label: str, text: str) -> None:
    assert not private_paths(text), (
        f"{label} names private modules {sorted(private_paths(text))}; use the "
        "public package path (the name listed in the package's __all__)."
    )
