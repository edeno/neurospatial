"""Tests for the ``neurospatial._typing`` Protocol surface and adapters.

Covers, without pynapple installed:

- ``EnvironmentLike`` is a narrow Protocol that Cartesian and polar
  ``Environment`` instances both structurally satisfy, matched to
  ``is_environment_like``.
- ``is_environment_like`` accepts an ``Environment`` and the polar sibling
  (regression for the ``isinstance(env, Environment)``-False surprise) and
  rejects arbitrary objects.
- ``import neurospatial._typing`` does not drag in pynapple / pynwb.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._typing import (
    EnvironmentLike,
    is_environment_like,
)


@pytest.fixture
def env() -> Environment:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (200, 2))
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def polar_env() -> Environment:
    return Environment.from_polar_egocentric(
        (0.0, 50.0), (-np.pi, np.pi), 10.0, np.pi / 6
    )


def test_environmentlike_is_narrow_protocol(
    env: Environment, polar_env: Environment
) -> None:
    # Environment and its polar sibling both structurally satisfy the narrow
    # surface (bin_centers / connectivity / neighbors), and it agrees with the
    # runtime ``is_environment_like`` duck-check.
    def _accepts_env_like(e: EnvironmentLike) -> tuple[object, object, object]:
        return e.bin_centers, e.connectivity, e.neighbors(0)

    _accepts_env_like(env)
    _accepts_env_like(polar_env)
    assert is_environment_like(env)
    assert is_environment_like(polar_env)


def test_is_environment_like_accepts_environment(env: Environment) -> None:
    assert is_environment_like(env)


def test_is_environment_like_accepts_polar_regression(
    polar_env: Environment,
) -> None:
    # Both coordinate choices use Environment. The structural check also
    # accepts compatible providers that do not inherit this class.
    assert isinstance(polar_env, Environment) is True
    assert is_environment_like(polar_env)


@pytest.mark.parametrize("obj", [None, np.array([1, 2, 3]), object(), "env"])
def test_is_environment_like_rejects_non_environment(obj: object) -> None:
    assert is_environment_like(obj) is False


def test_combine_refuses_polar_charts(polar_env: Environment) -> None:
    """A Cartesian union cannot interpret polar columns as Cartesian axes."""
    with pytest.raises(TypeError, match="Cartesian"):
        Environment.combine({"polar": polar_env})


def test_combine_rejects_non_environment() -> None:
    with pytest.raises(TypeError, match="Environment"):
        Environment.combine({"invalid": object()})


def test_typing_import_is_light() -> None:
    """``import neurospatial._typing`` must not import pynapple / pynwb."""
    code = (
        "import sys; import neurospatial._typing; "
        "assert 'pynapple' not in sys.modules, 'pynapple imported'; "
        "assert 'pynwb' not in sys.modules, 'pynwb imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
