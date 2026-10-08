"""Pin the public API: every ``__all__`` name of every public namespace."""

from __future__ import annotations

import difflib
import importlib
import inspect
import os
import re
from pathlib import Path

NAMESPACES = (
    "neurospatial",
    "neurospatial.encoding",
    "neurospatial.decoding",
    "neurospatial.behavior",
    "neurospatial.events",
    "neurospatial.ops",
    "neurospatial.stats",
    "neurospatial.simulation",
    "neurospatial.io",
    "neurospatial.io.nwb",
    "neurospatial.animation",
    "neurospatial.annotation",
    "neurospatial.regions",
    "neurospatial.layout",
)
SNAPSHOT = Path(__file__).parent / "data" / "public_api.txt"
UPDATE_ENV = "NEUROSPATIAL_UPDATE_API_SNAPSHOT"
UPDATE_CMD = f"{UPDATE_ENV}=1 uv run pytest tests/test_public_api_snapshot.py"
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _describe(obj: object) -> str:
    """Return the snapshot suffix for one exported object."""
    if inspect.ismodule(obj):
        return " (module)"
    if inspect.isclass(obj):
        return " (class)"
    if not callable(obj):
        return " (constant)"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(<signature unavailable>)"
    # Annotation text depends on the Python and NumPy versions (NDArray's repr),
    # so it is dropped; names, kinds, defaults and order are what break callers.
    sig = sig.replace(
        parameters=[p.replace(annotation=p.empty) for p in sig.parameters.values()],
        return_annotation=inspect.Signature.empty,
    )
    return _ADDRESS.sub("", str(sig))


def render_public_api() -> str:
    """Render the sorted snapshot text for every namespace in ``NAMESPACES``."""
    lines = []
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            lines.append(f"{namespace}.{name}{_describe(getattr(module, name))}")
    return "\n".join(sorted(lines)) + "\n"


def test_public_api_matches_snapshot() -> None:
    current = render_public_api()
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(current, encoding="utf-8")
        return
    expected = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
    if current != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile="tests/data/public_api.txt (committed)",
                tofile="public API (this checkout)",
            )
        )
        raise AssertionError(
            "The public API differs from tests/data/public_api.txt.\n"
            f"{diff}\nIf this change is intended, regenerate the snapshot and "
            f"commit it:\n    {UPDATE_CMD}"
        )
