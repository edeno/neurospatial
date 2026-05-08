"""Tests for backend dispatch in compute functions.

This module tests that all compute functions properly support the backend parameter:
- compute_spatial_rate(s)
- compute_directional_rate(s)
- compute_view_rate(s)
- compute_egocentric_rate(None, s)

TDD approach: Tests written first, implementation follows.

Design requirements (from TASKS.md Task 6.4):
- All compute functions should accept backend parameter
- backend should be validated against SUPPORTED_BACKENDS
- "numpy" backend should always work
- "jax" backend should raise NotImplementedError (until implemented)
- "auto" backend should use NumPy fallback (until JAX is fully implemented)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(-50, 50, (1000, 2))
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def trajectory_times() -> NDArray[np.float64]:
    """Trajectory timestamps (10 seconds at 100 Hz)."""
    return np.linspace(0, 10, 1000)


@pytest.fixture
def trajectory_positions() -> NDArray[np.float64]:
    """Trajectory positions covering the environment."""
    rng = np.random.default_rng(42)
    return rng.uniform(-50, 50, (1000, 2))


@pytest.fixture
def trajectory_headings() -> NDArray[np.float64]:
    """Head direction data (random walk)."""
    rng = np.random.default_rng(42)
    return np.cumsum(rng.uniform(-0.1, 0.1, 1000)) % (2 * np.pi)


@pytest.fixture
def spike_times() -> NDArray[np.float64]:
    """Simple spike train for testing."""
    rng = np.random.default_rng(42)
    return np.sort(rng.uniform(0, 10, 50))


@pytest.fixture
def multi_spike_times() -> list[NDArray[np.float64]]:
    """Multiple spike trains for batch testing."""
    rng = np.random.default_rng(42)
    return [np.sort(rng.uniform(0, 10, 50)) for _ in range(3)]


@pytest.fixture
def object_positions() -> NDArray[np.float64]:
    """Object positions for egocentric tests."""
    return np.array([[10.0, 10.0], [-20.0, 15.0]])


# ==============================================================================
# Test compute_spatial_rate backend parameter
# ==============================================================================


class TestComputeSpatialRateBackend:
    """Tests for backend parameter in compute_spatial_rate."""

    def test_accepts_numpy_backend(
        self,
        simple_env: Environment,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rate accepts backend='numpy'."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            simple_env,
            spike_times,
            trajectory_times,
            trajectory_positions,
            backend="numpy",
        )
        assert result is not None
        assert np.asarray(result.firing_rate).shape == (simple_env.n_bins,)

    def test_accepts_auto_backend(
        self,
        simple_env: Environment,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rate accepts backend='auto'."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            simple_env,
            spike_times,
            trajectory_times,
            trajectory_positions,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        simple_env: Environment,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rate raises ValueError for invalid backend."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_spatial_rate(
                simple_env,
                spike_times,
                trajectory_times,
                trajectory_positions,
                backend="invalid",  # type: ignore[arg-type]
            )


class TestComputeSpatialRatesBackend:
    """Tests for backend parameter in compute_spatial_rates."""

    def test_accepts_numpy_backend(
        self,
        simple_env: Environment,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rates accepts backend='numpy'."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            simple_env,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            backend="numpy",
        )
        assert result is not None
        assert np.asarray(result.firing_rates).shape == (3, simple_env.n_bins)

    def test_accepts_auto_backend(
        self,
        simple_env: Environment,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rates accepts backend='auto'."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            simple_env,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        simple_env: Environment,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rates raises ValueError for invalid backend."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_spatial_rates(
                simple_env,
                multi_spike_times,
                trajectory_times,
                trajectory_positions,
                backend="invalid",  # type: ignore[arg-type]
            )


# ==============================================================================
# Test compute_directional_rate backend parameter
# ==============================================================================


class TestComputeDirectionalRateBackend:
    """Tests for backend parameter in compute_directional_rate."""

    def test_accepts_numpy_backend(
        self,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_directional_rate accepts backend='numpy'."""
        from neurospatial.encoding.directional import compute_directional_rate

        result = compute_directional_rate(
            spike_times,
            trajectory_times,
            trajectory_headings,
            backend="numpy",
        )
        assert result is not None

    def test_accepts_auto_backend(
        self,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_directional_rate accepts backend='auto'."""
        from neurospatial.encoding.directional import compute_directional_rate

        result = compute_directional_rate(
            spike_times,
            trajectory_times,
            trajectory_headings,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_directional_rate raises ValueError for invalid backend."""
        from neurospatial.encoding.directional import compute_directional_rate

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_directional_rate(
                spike_times,
                trajectory_times,
                trajectory_headings,
                backend="invalid",  # type: ignore[arg-type]
            )


class TestComputeDirectionalRatesBackend:
    """Tests for backend parameter in compute_directional_rates."""

    def test_accepts_numpy_backend(
        self,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_directional_rates accepts backend='numpy'."""
        from neurospatial.encoding.directional import compute_directional_rates

        result = compute_directional_rates(
            multi_spike_times,
            trajectory_times,
            trajectory_headings,
            backend="numpy",
        )
        assert result is not None
        assert np.asarray(result.firing_rates).shape[0] == 3

    def test_accepts_auto_backend(
        self,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_directional_rates accepts backend='auto'."""
        from neurospatial.encoding.directional import compute_directional_rates

        result = compute_directional_rates(
            multi_spike_times,
            trajectory_times,
            trajectory_headings,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_directional_rates raises ValueError for invalid backend."""
        from neurospatial.encoding.directional import compute_directional_rates

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_directional_rates(
                multi_spike_times,
                trajectory_times,
                trajectory_headings,
                backend="invalid",  # type: ignore[arg-type]
            )


# ==============================================================================
# Test compute_view_rate backend parameter
# ==============================================================================


class TestComputeViewRateBackend:
    """Tests for backend parameter in compute_view_rate."""

    def test_accepts_numpy_backend(
        self,
        simple_env: Environment,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_view_rate accepts backend='numpy'."""
        from neurospatial.encoding.view import compute_view_rate

        result = compute_view_rate(
            simple_env,
            spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            backend="numpy",
        )
        assert result is not None

    def test_accepts_auto_backend(
        self,
        simple_env: Environment,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_view_rate accepts backend='auto'."""
        from neurospatial.encoding.view import compute_view_rate

        result = compute_view_rate(
            simple_env,
            spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        simple_env: Environment,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_view_rate raises ValueError for invalid backend."""
        from neurospatial.encoding.view import compute_view_rate

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_view_rate(
                simple_env,
                spike_times,
                trajectory_times,
                trajectory_positions,
                trajectory_headings,
                backend="invalid",  # type: ignore[arg-type]
            )


class TestComputeViewRatesBackend:
    """Tests for backend parameter in compute_view_rates."""

    def test_accepts_numpy_backend(
        self,
        simple_env: Environment,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_view_rates accepts backend='numpy'."""
        from neurospatial.encoding.view import compute_view_rates

        result = compute_view_rates(
            simple_env,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            backend="numpy",
        )
        assert result is not None
        assert np.asarray(result.firing_rates).shape[0] == 3

    def test_accepts_auto_backend(
        self,
        simple_env: Environment,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_view_rates accepts backend='auto'."""
        from neurospatial.encoding.view import compute_view_rates

        result = compute_view_rates(
            simple_env,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        simple_env: Environment,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
    ) -> None:
        """compute_view_rates raises ValueError for invalid backend."""
        from neurospatial.encoding.view import compute_view_rates

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_view_rates(
                simple_env,
                multi_spike_times,
                trajectory_times,
                trajectory_positions,
                trajectory_headings,
                backend="invalid",  # type: ignore[arg-type]
            )


# ==============================================================================
# Test compute_egocentric_rate backend parameter
# ==============================================================================


class TestComputeEgocentricRateBackend:
    """Tests for backend parameter in compute_egocentric_rate."""

    def test_accepts_numpy_backend(
        self,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
        object_positions: NDArray[np.float64],
    ) -> None:
        """compute_egocentric_rate accepts backend='numpy'."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        result = compute_egocentric_rate(
            None,
            spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            object_positions,
            backend="numpy",
        )
        assert result is not None

    def test_accepts_auto_backend(
        self,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
        object_positions: NDArray[np.float64],
    ) -> None:
        """compute_egocentric_rate accepts backend='auto'."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        result = compute_egocentric_rate(
            None,
            spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            object_positions,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        spike_times: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
        object_positions: NDArray[np.float64],
    ) -> None:
        """compute_egocentric_rate raises ValueError for invalid backend."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_egocentric_rate(
                None,
                spike_times,
                trajectory_times,
                trajectory_positions,
                trajectory_headings,
                object_positions,
                backend="invalid",  # type: ignore[arg-type]
            )


class TestComputeEgocentricRatesBackend:
    """Tests for backend parameter in compute_egocentric_rates."""

    def test_accepts_numpy_backend(
        self,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
        object_positions: NDArray[np.float64],
    ) -> None:
        """compute_egocentric_rates accepts backend='numpy'."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            object_positions,
            backend="numpy",
        )
        assert result is not None
        assert np.asarray(result.firing_rates).shape[0] == 3

    def test_accepts_auto_backend(
        self,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
        object_positions: NDArray[np.float64],
    ) -> None:
        """compute_egocentric_rates accepts backend='auto'."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            multi_spike_times,
            trajectory_times,
            trajectory_positions,
            trajectory_headings,
            object_positions,
            backend="auto",
        )
        assert result is not None

    def test_rejects_invalid_backend(
        self,
        multi_spike_times: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
        trajectory_headings: NDArray[np.float64],
        object_positions: NDArray[np.float64],
    ) -> None:
        """compute_egocentric_rates raises ValueError for invalid backend."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        with pytest.raises(ValueError, match="Unknown backend"):
            compute_egocentric_rates(
                None,
                multi_spike_times,
                trajectory_times,
                trajectory_positions,
                trajectory_headings,
                object_positions,
                backend="invalid",  # type: ignore[arg-type]
            )


# ==============================================================================
# Test backend consistency across modules
# ==============================================================================


class TestBackendConsistency:
    """Test that backend parameter is consistent across all compute functions."""

    def test_all_functions_have_backend_parameter(self) -> None:
        """All compute functions should have backend parameter."""
        import inspect

        from neurospatial.encoding.directional import (
            compute_directional_rate,
            compute_directional_rates,
        )
        from neurospatial.encoding.egocentric import (
            compute_egocentric_rate,
            compute_egocentric_rates,
        )
        from neurospatial.encoding.spatial import (
            compute_spatial_rate,
            compute_spatial_rates,
        )
        from neurospatial.encoding.view import compute_view_rate, compute_view_rates

        functions = [
            compute_spatial_rate,
            compute_spatial_rates,
            compute_directional_rate,
            compute_directional_rates,
            compute_view_rate,
            compute_view_rates,
            compute_egocentric_rate,
            compute_egocentric_rates,
        ]

        for func in functions:
            sig = inspect.signature(func)  # type: ignore[arg-type]
            assert "backend" in sig.parameters, (
                f"{func.__name__} should have 'backend' parameter"
            )

    def test_all_functions_default_to_numpy(self) -> None:
        """All compute functions should default to backend='numpy'."""
        import inspect

        from neurospatial.encoding.directional import (
            compute_directional_rate,
            compute_directional_rates,
        )
        from neurospatial.encoding.egocentric import (
            compute_egocentric_rate,
            compute_egocentric_rates,
        )
        from neurospatial.encoding.spatial import (
            compute_spatial_rate,
            compute_spatial_rates,
        )
        from neurospatial.encoding.view import compute_view_rate, compute_view_rates

        functions = [
            compute_spatial_rate,
            compute_spatial_rates,
            compute_directional_rate,
            compute_directional_rates,
            compute_view_rate,
            compute_view_rates,
            compute_egocentric_rate,
            compute_egocentric_rates,
        ]

        for func in functions:
            sig = inspect.signature(func)  # type: ignore[arg-type]
            backend_param = sig.parameters["backend"]
            assert backend_param.default == "numpy", (
                f"{func.__name__} should default to backend='numpy', "
                f"got default={backend_param.default}"
            )
