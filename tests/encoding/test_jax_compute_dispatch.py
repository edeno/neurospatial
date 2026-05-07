"""Tests for JAX backend dispatch in compute functions.

This module verifies that compute functions correctly dispatch to JAX backend
for core rate/metric computations while keeping binning on CPU.

TDD approach: Tests written first, implementation follows.

Design principle (from PLAN.md):
- Binning layer: CPU, joblib (stays NumPy)
- Core rate/metrics layer: dispatches to NumPy or JAX based on backend parameter

Success criteria:
- backend="jax" should use JAX for rate computation (after NumPy binning)
- backend="auto" should select JAX when available (except on Windows)
- Results from JAX backend should match NumPy backend numerically
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.encoding._backend import is_jax_available
from neurospatial.encoding._base import _is_jax_array

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(
    not is_jax_available(),
    reason="JAX is not available on this platform",
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, size=(500, 2))
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def spike_times() -> np.ndarray:
    """Create spike times for a single neuron."""
    rng = np.random.default_rng(42)
    # About 100 spikes over 10 seconds
    return np.sort(rng.uniform(0, 10, size=100))


@pytest.fixture
def spike_times_batch() -> list[np.ndarray]:
    """Create spike times for multiple neurons."""
    rng = np.random.default_rng(42)
    return [np.sort(rng.uniform(0, 10, size=n)) for n in [50, 80, 120, 60, 90]]


@pytest.fixture
def times() -> np.ndarray:
    """Create timestamps for trajectory sampling."""
    return np.linspace(0, 10, 1000)


@pytest.fixture
def positions(times: np.ndarray) -> np.ndarray:
    """Create trajectory positions."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, size=(len(times), 2))


# =============================================================================
# Test compute_spatial_rate with JAX backend
# =============================================================================


class TestComputeSpatialRateJaxBackend:
    """Tests for compute_spatial_rate with backend='jax'."""

    def test_jax_backend_does_not_raise(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """compute_spatial_rate with backend='jax' should not raise NotImplementedError."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        # This should NOT raise NotImplementedError
        result = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",  # Simple case
            backend="jax",
        )

        # Basic sanity checks
        assert result is not None
        assert result.firing_rate.shape == (simple_env.n_bins,)
        assert _is_jax_array(result.firing_rate)
        assert _is_jax_array(result.occupancy)

    def test_jax_backend_matches_numpy(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """JAX backend should produce numerically equivalent results to NumPy."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.spatial import compute_spatial_rate

        # Compute with NumPy backend
        result_numpy = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rate),
            result_numpy.firing_rate,
            rtol=1e-10,
            atol=1e-14,
        )
        assert_allclose(
            np.asarray(result_jax.occupancy),
            result_numpy.occupancy,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_jax_backend_with_diffusion_kde(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """JAX backend should work with diffusion_kde smoothing."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.spatial import compute_spatial_rate

        # Compute with NumPy backend
        result_numpy = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rate),
            result_numpy.firing_rate,
            rtol=1e-10,
            atol=1e-14,
        )


# =============================================================================
# Test compute_spatial_rates (batch) with JAX backend
# =============================================================================


class TestComputeSpatialRatesJaxBackend:
    """Tests for compute_spatial_rates with backend='jax'."""

    def test_jax_backend_does_not_raise(
        self,
        simple_env: Environment,
        spike_times_batch: list[np.ndarray],
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """compute_spatial_rates with backend='jax' should not raise NotImplementedError."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # This should NOT raise NotImplementedError
        result = compute_spatial_rates(
            simple_env,
            spike_times_batch,
            times,
            positions,
            smoothing_method="binned",
            backend="jax",
        )

        # Basic sanity checks
        assert result is not None
        assert result.firing_rates.shape == (len(spike_times_batch), simple_env.n_bins)
        assert _is_jax_array(result.firing_rates)
        assert _is_jax_array(result.occupancy)

    def test_jax_backend_matches_numpy(
        self,
        simple_env: Environment,
        spike_times_batch: list[np.ndarray],
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """JAX backend should produce numerically equivalent results to NumPy for batch."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.spatial import compute_spatial_rates

        # Compute with NumPy backend
        result_numpy = compute_spatial_rates(
            simple_env,
            spike_times_batch,
            times,
            positions,
            smoothing_method="binned",
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_spatial_rates(
            simple_env,
            spike_times_batch,
            times,
            positions,
            smoothing_method="binned",
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rates),
            result_numpy.firing_rates,
            rtol=1e-10,
            atol=1e-14,
        )


# =============================================================================
# Test 'auto' backend behavior
# =============================================================================


class TestAutoBackendBehavior:
    """Tests for backend='auto' selecting JAX when available."""

    def test_auto_backend_uses_jax_when_available(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """backend='auto' should use JAX when available (not on Windows)."""
        import sys

        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.spatial import compute_spatial_rate

        # Compute with auto backend
        result_auto = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="auto",
        )

        # Compute with explicit jax backend
        result_jax = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="jax",
        )

        # On non-Windows platforms, auto should use JAX
        if sys.platform != "win32":
            assert_allclose(
                np.asarray(result_auto.firing_rate),
                np.asarray(result_jax.firing_rate),
                rtol=1e-10,
                atol=1e-14,
            )


# =============================================================================
# Test metrics with JAX backend
# =============================================================================


class TestMetricsWithJaxBackend:
    """Tests that result class metrics work correctly with JAX backend."""

    def test_spatial_information_with_jax_backend(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """spatial_information() should work and match NumPy for JAX backend results."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.spatial import compute_spatial_rate

        result_numpy = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="numpy",
        )

        result_jax = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="jax",
        )

        info_numpy = result_numpy.spatial_information()
        info_jax = result_jax.spatial_information()

        # Metrics should match
        assert_allclose(info_jax, info_numpy, rtol=1e-10)

    def test_sparsity_with_jax_backend(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """sparsity() should work and match NumPy for JAX backend results."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.spatial import compute_spatial_rate

        result_numpy = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="numpy",
        )

        result_jax = compute_spatial_rate(
            simple_env,
            spike_times,
            times,
            positions,
            smoothing_method="binned",
            backend="jax",
        )

        spars_numpy = result_numpy.sparsity()
        spars_jax = result_jax.sparsity()

        # Metrics should match
        assert_allclose(spars_jax, spars_numpy, rtol=1e-10)


# =============================================================================
# Test compute_view_rate with JAX backend
# =============================================================================


@pytest.fixture
def headings(times: np.ndarray) -> np.ndarray:
    """Create headings for view/egocentric tests."""
    rng = np.random.default_rng(42)
    return rng.uniform(-np.pi, np.pi, size=len(times))


class TestComputeViewRateJaxBackend:
    """Tests for compute_view_rate with backend='jax'."""

    def test_jax_backend_does_not_raise(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
    ) -> None:
        """compute_view_rate with backend='jax' should not raise NotImplementedError."""
        from neurospatial.encoding.view import compute_view_rate

        # This should NOT raise NotImplementedError
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="binned",
            backend="jax",
        )

        # Basic sanity checks
        assert result is not None
        assert result.firing_rate.shape == (simple_env.n_bins,)
        assert _is_jax_array(result.firing_rate)
        assert _is_jax_array(result.view_occupancy)

    def test_jax_backend_matches_numpy(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
    ) -> None:
        """JAX backend should produce numerically equivalent results to NumPy."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.view import compute_view_rate

        # Compute with NumPy backend
        result_numpy = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="binned",
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="binned",
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rate),
            result_numpy.firing_rate,
            rtol=1e-10,
            atol=1e-14,
        )
        assert_allclose(
            np.asarray(result_jax.view_occupancy),
            result_numpy.view_occupancy,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_jax_backend_with_diffusion_kde(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
    ) -> None:
        """JAX backend should work with diffusion_kde smoothing."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.view import compute_view_rate

        # Compute with NumPy backend
        result_numpy = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rate),
            result_numpy.firing_rate,
            rtol=1e-10,
            atol=1e-14,
        )


# =============================================================================
# Test compute_view_rates (batch) with JAX backend
# =============================================================================


class TestComputeViewRatesJaxBackend:
    """Tests for compute_view_rates with backend='jax'."""

    def test_jax_backend_does_not_raise(
        self,
        simple_env: Environment,
        spike_times_batch: list[np.ndarray],
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
    ) -> None:
        """compute_view_rates with backend='jax' should not raise NotImplementedError."""
        from neurospatial.encoding.view import compute_view_rates

        # This should NOT raise NotImplementedError
        result = compute_view_rates(
            simple_env,
            spike_times_batch,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="binned",
            backend="jax",
        )

        # Basic sanity checks
        assert result is not None
        assert result.firing_rates.shape == (len(spike_times_batch), simple_env.n_bins)
        assert _is_jax_array(result.firing_rates)
        assert _is_jax_array(result.view_occupancy)

    def test_jax_backend_matches_numpy(
        self,
        simple_env: Environment,
        spike_times_batch: list[np.ndarray],
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
    ) -> None:
        """JAX backend should produce numerically equivalent results to NumPy for batch."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.view import compute_view_rates

        # Compute with NumPy backend
        result_numpy = compute_view_rates(
            simple_env,
            spike_times_batch,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="binned",
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_view_rates(
            simple_env,
            spike_times_batch,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="binned",
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rates),
            result_numpy.firing_rates,
            rtol=1e-10,
            atol=1e-14,
        )


# =============================================================================
# Test compute_egocentric_rate with JAX backend
# =============================================================================


@pytest.fixture
def object_positions() -> np.ndarray:
    """Create object positions for egocentric tests."""
    return np.array([[50.0, 50.0], [25.0, 75.0], [75.0, 25.0]])


class TestComputeEgocentricRateJaxBackend:
    """Tests for compute_egocentric_rate with backend='jax'."""

    def test_jax_backend_does_not_raise(
        self,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
        object_positions: np.ndarray,
    ) -> None:
        """compute_egocentric_rate with backend='jax' should not raise NotImplementedError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        # This should NOT raise NotImplementedError
        result = compute_egocentric_rate(
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 100.0),
            n_distance_bins=5,
            n_direction_bins=8,
            smoothing_method="binned",
            backend="jax",
        )

        # Basic sanity checks
        assert result is not None
        assert result.firing_rate.shape == (result.ego_env.n_bins,)
        assert _is_jax_array(result.firing_rate)
        assert _is_jax_array(result.occupancy)

    def test_jax_backend_matches_numpy(
        self,
        spike_times: np.ndarray,
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
        object_positions: np.ndarray,
    ) -> None:
        """JAX backend should produce numerically equivalent results to NumPy."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.egocentric import compute_egocentric_rate

        # Compute with NumPy backend
        result_numpy = compute_egocentric_rate(
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 100.0),
            n_distance_bins=5,
            n_direction_bins=8,
            smoothing_method="binned",
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_egocentric_rate(
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 100.0),
            n_distance_bins=5,
            n_direction_bins=8,
            smoothing_method="binned",
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rate),
            result_numpy.firing_rate,
            rtol=1e-10,
            atol=1e-14,
        )
        assert_allclose(
            np.asarray(result_jax.occupancy),
            result_numpy.occupancy,
            rtol=1e-10,
            atol=1e-14,
        )


# =============================================================================
# Test compute_egocentric_rates (batch) with JAX backend
# =============================================================================


class TestComputeEgocentricRatesJaxBackend:
    """Tests for compute_egocentric_rates with backend='jax'."""

    def test_jax_backend_does_not_raise(
        self,
        spike_times_batch: list[np.ndarray],
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
        object_positions: np.ndarray,
    ) -> None:
        """compute_egocentric_rates with backend='jax' should not raise NotImplementedError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # This should NOT raise NotImplementedError
        result = compute_egocentric_rates(
            spike_times_batch,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 100.0),
            n_distance_bins=5,
            n_direction_bins=8,
            smoothing_method="binned",
            backend="jax",
        )

        # Basic sanity checks
        assert result is not None
        assert result.firing_rates.shape == (
            len(spike_times_batch),
            result.ego_env.n_bins,
        )
        assert _is_jax_array(result.firing_rates)
        assert _is_jax_array(result.occupancy)

    def test_jax_backend_matches_numpy(
        self,
        spike_times_batch: list[np.ndarray],
        times: np.ndarray,
        positions: np.ndarray,
        headings: np.ndarray,
        object_positions: np.ndarray,
    ) -> None:
        """JAX backend should produce numerically equivalent results to NumPy for batch."""
        import jax

        jax.config.update("jax_enable_x64", True)

        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # Compute with NumPy backend
        result_numpy = compute_egocentric_rates(
            spike_times_batch,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 100.0),
            n_distance_bins=5,
            n_direction_bins=8,
            smoothing_method="binned",
            backend="numpy",
        )

        # Compute with JAX backend
        result_jax = compute_egocentric_rates(
            spike_times_batch,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 100.0),
            n_distance_bins=5,
            n_direction_bins=8,
            smoothing_method="binned",
            backend="jax",
        )

        # Results should match
        assert_allclose(
            np.asarray(result_jax.firing_rates),
            result_numpy.firing_rates,
            rtol=1e-10,
            atol=1e-14,
        )
