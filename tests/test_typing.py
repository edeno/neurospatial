"""Tests for the ``neurospatial._typing`` Protocol surface and adapters.

Covers, without pynapple installed:

- ``EnvironmentLike`` is a purpose-built NARROW Protocol (not the internal
  ``EnvironmentProtocol`` mixin re-export) that ``Environment`` and the polar
  sibling both structurally satisfy, matched to ``is_environment_like``.
- ``as_times_positions`` normalizes both the array pair and a duck-typed
  ``PositionLike`` object.
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
    as_times_positions,
    is_environment_like,
)
from neurospatial.environment._protocols import EnvironmentProtocol
from neurospatial.environment.polar import EgocentricPolarEnvironment


class _FakeTsd:
    """Minimal duck-typed ``PositionLike`` exposing ``.t`` and ``.values``."""

    def __init__(self, t: np.ndarray, values: np.ndarray) -> None:
        self._t = t
        self._values = values

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def values(self) -> np.ndarray:
        return self._values


class _FakeTsdD:
    """Duck-typed ``PositionLike`` exposing ``.t`` and the pynapple ``.d`` alias."""

    def __init__(self, t: np.ndarray, d: np.ndarray) -> None:
        self.t = t
        self.d = d


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


def test_as_times_positions_array_pair_returns_float64() -> None:
    times = np.array([0, 1, 2], dtype=np.int64)
    positions = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)

    out_times, out_positions = as_times_positions(times, positions)

    assert out_times.dtype == np.float64
    assert out_positions.dtype == np.float64
    np.testing.assert_array_equal(out_times, times.astype(np.float64))
    np.testing.assert_array_equal(out_positions, positions.astype(np.float64))


def test_as_times_positions_positionlike_values() -> None:
    times = np.linspace(0.0, 1.0, 5)
    positions = np.column_stack([times, times * 2])

    out_times, out_positions = as_times_positions(_FakeTsd(times, positions))

    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(out_positions, positions)


def test_as_times_positions_positionlike_d_alias() -> None:
    times = np.linspace(0.0, 1.0, 4)
    positions = np.column_stack([times, times])

    out_times, out_positions = as_times_positions(_FakeTsdD(times, positions))

    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(out_positions, positions)


def test_as_times_positions_positionlike_with_positions_raises() -> None:
    times = np.array([0.0, 1.0])
    positions = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="EITHER a single PositionLike"):
        as_times_positions(_FakeTsd(times, positions), positions)


def test_as_times_positions_array_without_positions_raises() -> None:
    with pytest.raises(ValueError, match="no `positions`"):
        as_times_positions(np.array([0.0, 1.0, 2.0]))


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
