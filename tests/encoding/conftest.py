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
