"""Tests for top-level lazy submodule access (PEP 562 ``__getattr__``).

Accessing ``neurospatial.encoding`` (etc.) should import the submodule on first
use without eagerly importing it when the package loads. ``dir(neurospatial)``
should reveal both the eager exports and the lazily importable submodules, and
unknown attributes must still raise ``AttributeError`` so typos fail loudly.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

import neurospatial as ns

# Submodules expected to be lazily accessible from the top-level package.
_EXPECTED_SUBMODULES = (
    "encoding",
    "decoding",
    "behavior",
    "events",
    "ops",
    "layout",
    "regions",
    "stats",
    "simulation",
    "annotation",
    "animation",
    "io",
)

# Eager exports that must remain importable directly from the package.
_EAGER_EXPORTS = (
    "Environment",
    "Region",
    "Regions",
    "CompositeEnvironment",
    "bin_spikes_in_time",
)


@pytest.mark.parametrize("name", _EXPECTED_SUBMODULES)
def test_lazy_submodule_access(name: str) -> None:
    """Accessing a submodule returns the module object."""
    submodule = getattr(ns, name)
    assert isinstance(submodule, ModuleType)
    assert submodule.__name__ == f"neurospatial.{name}"


def test_dir_lists_submodules() -> None:
    """``dir(ns)`` includes every lazy submodule and every eager export."""
    listed = set(dir(ns))
    for name in _EXPECTED_SUBMODULES:
        assert name in listed, f"{name!r} missing from dir(neurospatial)"
    for name in _EAGER_EXPORTS:
        assert name in listed, f"{name!r} missing from dir(neurospatial)"


def test_unknown_attr_raises() -> None:
    """A typo / unknown attribute raises AttributeError (not silently swallowed)."""
    name = "nonexistent_submodule"
    with pytest.raises(AttributeError):
        getattr(ns, name)


@pytest.mark.parametrize("name", _EAGER_EXPORTS)
def test_eager_exports_unchanged(name: str) -> None:
    """Eager exports remain accessible directly from the package."""
    assert hasattr(ns, name)


def test_no_eager_submodule_import() -> None:
    """Importing neurospatial does not eagerly import a lazy-only submodule.

    Uses ``animation`` as a representative submodule that is not pulled in by the
    package's eager imports, then confirms it loads on first access.
    """

    # Re-import the package in a clean module state so the assertion is robust to
    # whatever other tests already imported. Snapshot the original module objects
    # first and restore them afterward: purging ``neurospatial.*`` without
    # restoring would split class identity for any test later in this worker that
    # imported a neurospatial class at collection time (e.g.
    # ``isinstance(x, SpikeTrains)`` would then compare against a stale class).
    def _neurospatial_modnames() -> list[str]:
        return [
            name
            for name in sys.modules
            if name == "neurospatial" or name.startswith("neurospatial.")
        ]

    saved = {name: sys.modules[name] for name in _neurospatial_modnames()}
    for mod_name in list(saved):
        del sys.modules[mod_name]

    try:
        fresh = importlib.import_module("neurospatial")

        assert "neurospatial.animation" not in sys.modules, (
            "neurospatial.animation should not be imported until first accessed"
        )

        # First access triggers the lazy import.
        module = fresh.animation
        assert isinstance(module, ModuleType)
        assert "neurospatial.animation" in sys.modules
    finally:
        # Restore the original module objects so later tests in this worker see
        # the same class identities they imported at collection time.
        for mod_name in _neurospatial_modnames():
            del sys.modules[mod_name]
        sys.modules.update(saved)
