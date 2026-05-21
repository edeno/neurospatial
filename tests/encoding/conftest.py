"""Shared fixtures for encoding tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from neurospatial.encoding._backend import is_jax_available


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
    if not is_jax_available():
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

    Tests that monkeypatch ``sys.platform`` and call ``cache_clear()``
    (or ``importlib.reload(backend_module)``) leave the module holding
    stale state once ``sys.platform`` is restored. Without a teardown,
    subsequent tests run against that perturbed module — order-dependent
    under xdist. Clearing pre-yield and reloading on teardown gives
    every test in the ``encoding`` suite a clean cache.
    """
    import importlib

    import neurospatial.encoding._backend as backend_module

    backend_module.is_jax_available.cache_clear()
    yield
    importlib.reload(backend_module)
