"""Tests for the ``neurospatial._typing`` Protocol surface and adapters.

Covers, without pynapple installed:

- ``EnvironmentLike`` is a purpose-built NARROW Protocol (not the internal
  ``EnvironmentProtocol`` mixin re-export) that ``Environment`` and the polar
  sibling both structurally satisfy, matched to ``is_environment_like``.
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
from neurospatial.environment._protocols import EnvironmentProtocol
from neurospatial.environment.polar import EgocentricPolarEnvironment


@pytest.fixture
def env() -> Environment:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (200, 2))
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def polar_env() -> EgocentricPolarEnvironment:
    return EgocentricPolarEnvironment.create(
        (0.0, 50.0), (-np.pi, np.pi), 10.0, np.pi / 6
    )


def test_environmentlike_is_narrow_protocol(
    env: Environment, polar_env: EgocentricPolarEnvironment
) -> None:
    # EnvironmentLike is now a purpose-built NARROW Protocol, no longer the
    # internal mixin ``EnvironmentProtocol`` re-export (which published ~14
    # private members and disagreed with the 3-attr runtime check).
    assert EnvironmentLike is not EnvironmentProtocol

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
    polar_env: EgocentricPolarEnvironment,
) -> None:
    # Regression: the polar sibling is NOT an Environment subclass, so the old
    # ``isinstance(env, Environment)`` check was False for it. ``is_environment_like``
    # accepts it via the shared structural surface.
    assert isinstance(polar_env, Environment) is False
    assert is_environment_like(polar_env)


@pytest.mark.parametrize("obj", [None, np.array([1, 2, 3]), object(), "env"])
def test_is_environment_like_rejects_non_environment(obj: object) -> None:
    assert is_environment_like(obj) is False


def test_composite_validate_subenvs_accepts_polar_regression(
    polar_env: EgocentricPolarEnvironment,
) -> None:
    """Regression at the composite call site: the polar sibling is no longer
    rejected by ``isinstance(env, Environment)`` (now duck-typed)."""
    from neurospatial.composite import _validate_subenvs

    validated = _validate_subenvs([polar_env])
    assert validated == [polar_env]


def test_composite_validate_subenvs_rejects_non_environment() -> None:
    """The duck-typed check still rejects a genuinely non-environment object."""
    from neurospatial.composite import _validate_subenvs

    with pytest.raises(TypeError, match="Environment-like"):
        _validate_subenvs([object()])


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
