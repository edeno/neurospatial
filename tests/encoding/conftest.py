"""Shared fixtures for encoding tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_numpy_random_state() -> Generator[None, None, None]:
    """Make legacy global np.random use deterministic and order-independent."""
    previous_state: tuple[Any, ...] = np.random.get_state()
    np.random.seed(0)
    yield
    np.random.set_state(previous_state)


@pytest.fixture(autouse=True)
def restore_jax_x64_config() -> Generator[None, None, None]:
    """Keep tests that enable JAX x64 from leaking global config state."""
    # Resolve ``is_jax_available`` dynamically. ``restore_backend_availability_cache``
    # below calls ``importlib.reload(backend_module)`` on teardown, which
    # rebinds ``is_jax_available`` to a fresh function inside the module —
    # a module-level ``from … import is_jax_available`` here would silently
    # point at the *pre-reload* function (with its own stale LRU cache) on
    # every test after the first.
    import neurospatial.encoding._backend as backend_module

    if not backend_module.is_jax_available():
        yield
        return

    import jax

    previous_value = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous_value)


@pytest.fixture(autouse=True)
def restore_backend_availability_cache() -> Generator[None, None, None]:
    """Isolate the ``_backend`` module's LRU cache across tests.

    Tests that monkeypatch ``sys.platform`` or call ``cache_clear()``
    leave the module holding stale availability state once
    ``sys.platform`` is restored. Without a teardown, subsequent tests
    run against that perturbed module — order-dependent under xdist.

    Autouse so any test in the encoding suite that mutates platform or
    reloads ``_backend`` is cleaned up, even if it forgets to request
    the fixture. Clears pre-yield and reloads the module on teardown,
    giving every test a fresh ``is_jax_available()`` lookup.
    """
    import importlib

    import neurospatial.encoding._backend as backend_module

    backend_module.is_jax_available.cache_clear()
    yield
    importlib.reload(backend_module)
