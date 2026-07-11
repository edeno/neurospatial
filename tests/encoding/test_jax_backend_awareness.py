"""Tests for backend awareness of result class methods with JAX arrays.

This module verifies that all result class methods correctly handle JAX arrays:
- _to_numpy() converts JAX arrays to NumPy
- _get_array_module() detects JAX arrays
- plot() methods work with JAX array data
- metric methods work with JAX array data

TDD approach: Tests written first, implementation follows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.encoding._backend import is_jax_available

# ==============================================================================
# Skip marker
# ==============================================================================


skip_without_jax = pytest.mark.skipif(
    not is_jax_available(),
    reason="JAX not installed or not available on this platform",
)


# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    positions = np.array(
        [
            [10.0, 10.0],
            [20.0, 10.0],
            [30.0, 10.0],
            [10.0, 20.0],
            [20.0, 20.0],
            [30.0, 20.0],
            [10.0, 30.0],
            [20.0, 30.0],
            [30.0, 30.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=10.0)


# ==============================================================================
# Test _to_numpy with JAX arrays
# ==============================================================================


@skip_without_jax
class TestToNumpyWithJax:
    """Tests for _to_numpy helper with JAX arrays."""

    def test_jax_1d_array_conversion(self) -> None:
        """1D JAX arrays should be converted to NumPy."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import _to_numpy

        jax_arr = jnp.array([1.0, 2.0, 3.0])
        result = _to_numpy(jax_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_jax_2d_array_conversion(self) -> None:
        """2D JAX arrays should be converted to NumPy."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import _to_numpy

        jax_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = _to_numpy(jax_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_jax_preserves_values(self) -> None:
        """JAX to NumPy conversion should preserve values exactly."""
        import jax.numpy as jnp

        # Enable 64-bit precision for exact comparison
        from neurospatial.encoding._base import _to_numpy

        values = np.array([1.5, 2.7, 3.14159, -0.5, 0.0])
        jax_arr = jnp.array(values)
        result = _to_numpy(jax_arr)
        np.testing.assert_array_almost_equal(result, values)


# ==============================================================================
# Test _get_array_module with JAX arrays
# ==============================================================================


@skip_without_jax
class TestGetArrayModuleWithJax:
    """Tests for _get_array_module helper with JAX arrays."""


@skip_without_jax
class TestSpatialResultMixinWithJax:
    """Tests for SpatialResultMixin methods with JAX arrays."""

    def test_peak_locations_with_jax_single(self, simple_env: Environment) -> None:
        """peak_location() should work with JAX arrays for single neuron."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        firing_rate_np = np.zeros(n_bins)
        peak_bin = n_bins // 2
        firing_rate_np[peak_bin] = 10.0

        # Convert to JAX array
        firing_rate_jax = jnp.array(firing_rate_np)

        result = MockSingleResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak = result.peak_location()
        # Should return NumPy array
        assert isinstance(peak, np.ndarray)
        np.testing.assert_array_equal(peak, simple_env.bin_centers[peak_bin])

    def test_peak_locations_with_jax_batch(self, simple_env: Environment) -> None:
        """peak_location() should work with JAX arrays for batch."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockBatchResult(SpatialResultMixin):
            firing_rates: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_np = np.zeros((n_neurons, n_bins))
        peak_bins = [1, n_bins // 2, n_bins - 2]
        for i, peak_bin in enumerate(peak_bins):
            firing_rates_np[i, peak_bin] = 10.0

        # Convert to JAX array
        firing_rates_jax = jnp.array(firing_rates_np)

        result = MockBatchResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peaks = result.peak_location()
        # Should return NumPy array
        assert isinstance(peaks, np.ndarray)
        assert peaks.shape == (n_neurons, simple_env.n_dims)

    def test_peak_firing_rate_with_jax_single(self, simple_env: Environment) -> None:
        """peak_firing_rate() should work with JAX arrays for single neuron."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        firing_rate_np = np.zeros(n_bins)
        peak_value = 15.0
        firing_rate_np[n_bins // 2] = peak_value

        # Convert to JAX array
        firing_rate_jax = jnp.array(firing_rate_np)

        result = MockSingleResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak_rate = result.peak_firing_rate()
        assert isinstance(peak_rate, float)
        assert peak_rate == peak_value

    def test_peak_firing_rate_with_jax_batch(self, simple_env: Environment) -> None:
        """peak_firing_rate() should work with JAX arrays for batch."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockBatchResult(SpatialResultMixin):
            firing_rates: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_np = np.zeros((n_neurons, n_bins))
        peak_values = [10.0, 15.0, 20.0]
        for i, peak_val in enumerate(peak_values):
            firing_rates_np[i, i + 1] = peak_val

        # Convert to JAX array
        firing_rates_jax = jnp.array(firing_rates_np)

        result = MockBatchResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak_rates = result.peak_firing_rate()
        assert isinstance(peak_rates, np.ndarray)
        assert peak_rates.shape == (n_neurons,)


# ==============================================================================
# Test SpatialRateResult with JAX arrays
# ==============================================================================


@skip_without_jax
class TestSpatialRateResultWithJax:
    """Tests for SpatialRateResult methods with JAX arrays."""

    def test_plot_with_jax_array(self, simple_env: Environment) -> None:
        """plot() should work with JAX array data."""
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate_jax = jnp.ones(n_bins) * 5.0
        occupancy_jax = jnp.ones(n_bins) * 1.0

        result = SpatialRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        # Should not raise
        ax = result.plot()
        assert ax is not None
        plt.close("all")

    def test_spatial_information_with_jax(self, simple_env: Environment) -> None:
        """spatial_information() should work with JAX arrays and return JAX scalar."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        # Uniform firing - should have ~0 spatial info
        firing_rate_jax = jnp.ones(n_bins) * 5.0
        occupancy_jax = jnp.ones(n_bins) * 1.0

        result = SpatialRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        info = result.spatial_information()
        # Backend-aware: JAX input -> JAX output
        assert isinstance(info, jax.Array)
        assert float(info) >= 0.0

    def test_peak_location_with_jax(self, simple_env: Environment) -> None:
        """peak_location() should work with JAX arrays."""
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate_np = np.zeros(n_bins)
        peak_bin = n_bins // 2
        firing_rate_np[peak_bin] = 10.0
        firing_rate_jax = jnp.array(firing_rate_np)
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        peak = result.peak_location()
        assert isinstance(peak, np.ndarray)


# ==============================================================================
# Test SpatialRatesResult with JAX arrays
# ==============================================================================


@skip_without_jax
class TestSpatialRatesResultWithJax:
    """Tests for SpatialRatesResult methods with JAX arrays."""

    def test_plot_with_jax_array(self, simple_env: Environment) -> None:
        """plot() should work with JAX array data."""
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_jax = jnp.ones((n_neurons, n_bins)) * 5.0
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRatesResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        # Should not raise
        ax = result.plot(idx=0)
        assert ax is not None
        plt.close("all")

    def test_spatial_information_with_jax(self, simple_env: Environment) -> None:
        """spatial_information() should work with JAX arrays and return JAX array."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_jax = jnp.ones((n_neurons, n_bins)) * 5.0
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRatesResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        info = result.spatial_information()
        # Backend-aware: JAX input -> JAX output
        assert isinstance(info, jax.Array)
        assert info.shape == (n_neurons,)

    def test_getitem_with_jax(self, simple_env: Environment) -> None:
        """__getitem__ should work with JAX arrays."""
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_jax = jnp.ones((n_neurons, n_bins)) * 5.0
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRatesResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        single_result = result[0]
        assert isinstance(single_result, SpatialRateResult)


# ==============================================================================
# Test DirectionalRateResult with JAX arrays
# ==============================================================================


@skip_without_jax
class TestDirectionalRateResultWithJax:
    """Tests for DirectionalRateResult methods with JAX arrays."""

    def test_plot_with_jax_array(self) -> None:
        """plot() should work with JAX array data."""
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rate_jax = jnp.ones(n_bins) * 5.0
        occupancy_jax = jnp.ones(n_bins)
        bin_centers_jax = jnp.array(bin_centers)

        result = DirectionalRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            bin_centers=bin_centers_jax,  # type: ignore[arg-type]
            bin_size=np.pi / 30,
            bandwidth=None,
        )

        # Should not raise
        ax = result.plot()
        assert ax is not None
        plt.close("all")

    def test_preferred_direction_with_jax(self) -> None:
        """preferred_direction() should work with JAX arrays."""
        import jax.numpy as jnp

        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        # Create tuning curve with peak at pi/2
        firing_rate_np = 10.0 * np.exp(2.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        firing_rate_jax = jnp.array(firing_rate_np)
        occupancy_jax = jnp.ones(n_bins)
        bin_centers_jax = jnp.array(bin_centers)

        result = DirectionalRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            bin_centers=bin_centers_jax,  # type: ignore[arg-type]
            bin_size=np.pi / 30,
            bandwidth=None,
        )

        pref_dir = result.preferred_direction()
        assert isinstance(pref_dir, float)

    def test_mean_vector_length_with_jax(self) -> None:
        """mean_vector_length() should work with JAX arrays."""
        import jax.numpy as jnp

        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        # Sharply tuned cell
        firing_rate_np = 10.0 * np.exp(4.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        firing_rate_jax = jnp.array(firing_rate_np)
        occupancy_jax = jnp.ones(n_bins)
        bin_centers_jax = jnp.array(bin_centers)

        result = DirectionalRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            bin_centers=bin_centers_jax,  # type: ignore[arg-type]
            bin_size=np.pi / 30,
            bandwidth=None,
        )

        mvl = result.mean_vector_length()
        assert isinstance(mvl, float)
        assert 0.0 <= mvl <= 1.0


# ==============================================================================
# Test ViewRateResult with JAX arrays
# ==============================================================================


@skip_without_jax
class TestViewRateResultWithJax:
    """Tests for ViewRateResult methods with JAX arrays."""

    def test_plot_with_jax_array(self, simple_env: Environment) -> None:
        """plot() should work with JAX array data."""
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        from neurospatial.encoding.view import ViewRateResult

        n_bins = simple_env.n_bins
        firing_rate_jax = jnp.ones(n_bins) * 5.0
        occupancy_jax = jnp.ones(n_bins)

        result = ViewRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            method="binned",
            bandwidth=5.0,
        )

        # Should not raise
        ax = result.plot()
        assert ax is not None
        plt.close("all")


# ==============================================================================
# Test EgocentricRateResult with JAX arrays
# ==============================================================================


@skip_without_jax
class TestEgocentricRateResultWithJax:
    """Tests for EgocentricRateResult methods with JAX arrays."""

    def test_plot_with_jax_array(self) -> None:
        """plot() should work with JAX array data."""
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create egocentric polar environment
        n_distance_bins = 5
        n_direction_bins = 12
        distance_bin_size = 10.0  # 50.0 / 5 bins
        angle_bin_size = 2 * np.pi / n_direction_bins
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        n_bins = env.n_bins
        firing_rate_jax = jnp.ones(n_bins) * 5.0
        occupancy_jax = jnp.ones(n_bins)

        result = EgocentricRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=env,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )

        # Should not raise
        ax = result.plot()
        assert ax is not None
        plt.close("all")

    def test_preferred_distance_with_jax(self) -> None:
        """preferred_distance() should work with JAX arrays."""
        import jax.numpy as jnp

        from neurospatial.encoding.egocentric import EgocentricRateResult

        n_distance_bins = 5
        n_direction_bins = 12
        distance_bin_size = 10.0  # 50.0 / 5 bins
        angle_bin_size = 2 * np.pi / n_direction_bins
        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 50.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=distance_bin_size,
            angle_bin_size=angle_bin_size,
        )

        n_bins = env.n_bins
        # Create firing rate with peak at middle bin
        firing_rate_np = np.zeros(n_bins)
        peak_bin = n_bins // 2
        firing_rate_np[peak_bin] = 10.0
        firing_rate_jax = jnp.array(firing_rate_np)
        occupancy_jax = jnp.ones(n_bins)

        result = EgocentricRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=env,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )

        pref_dist = result.preferred_distance()
        assert isinstance(pref_dist, float)
        assert pref_dist >= 0.0


# ==============================================================================
# Test consistency: NumPy vs JAX results should match
# ==============================================================================


@skip_without_jax
class TestNumpyJaxConsistency:
    """Verify that results are consistent between NumPy and JAX arrays."""

    def test_spatial_information_consistency(self, simple_env: Environment) -> None:
        """spatial_information() should give same result for NumPy and JAX."""
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        # Create spatially selective firing
        firing_rate_np = np.zeros(n_bins)
        firing_rate_np[n_bins // 2] = 10.0
        firing_rate_np[n_bins // 2 + 1] = 5.0
        occupancy_np = np.ones(n_bins)

        # NumPy result
        result_np = SpatialRateResult(
            firing_rate=firing_rate_np,
            occupancy=occupancy_np,
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )
        info_np = result_np.spatial_information()

        # JAX result
        result_jax = SpatialRateResult(
            firing_rate=jnp.array(firing_rate_np),  # type: ignore[arg-type]
            occupancy=jnp.array(occupancy_np),  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )
        info_jax = result_jax.spatial_information()

        # Should be the same
        np.testing.assert_almost_equal(info_np, info_jax, decimal=10)

    def test_peak_location_consistency(self, simple_env: Environment) -> None:
        """peak_location() should give same result for NumPy and JAX."""
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate_np = np.zeros(n_bins)
        peak_bin = n_bins // 2
        firing_rate_np[peak_bin] = 10.0
        occupancy_np = np.ones(n_bins)

        # NumPy result
        result_np = SpatialRateResult(
            firing_rate=firing_rate_np,
            occupancy=occupancy_np,
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )
        peak_np = result_np.peak_location()

        # JAX result
        result_jax = SpatialRateResult(
            firing_rate=jnp.array(firing_rate_np),  # type: ignore[arg-type]
            occupancy=jnp.array(occupancy_np),  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )
        peak_jax = result_jax.peak_location()

        # Should be the same
        np.testing.assert_array_equal(peak_np, peak_jax)


# ==============================================================================
# Test JAX array preservation through metric calls (Task 8.3)
# ==============================================================================


@skip_without_jax
class TestJaxArrayPreservation:
    """Verify that JAX arrays are preserved through backend-aware metric calls.

    These tests verify the backend-aware behavior: when JAX arrays are passed
    to metric functions via result class methods, the output should also be
    JAX arrays (not converted to NumPy).

    This is critical for JAX-traced compute graphs and GPU workflows.
    """

    def test_spatial_information_preserves_jax_single(
        self, simple_env: Environment
    ) -> None:
        """spatial_information() on SpatialRateResult should return JAX scalar."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate_jax = jnp.zeros(n_bins).at[n_bins // 2].set(10.0)
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        info = result.spatial_information()
        # Should be a JAX array (scalar), not a Python float
        assert isinstance(info, jax.Array), (
            f"Expected jax.Array, got {type(info)}. "
            "JAX arrays should be preserved through metric calls."
        )

    def test_sparsity_preserves_jax_single(self, simple_env: Environment) -> None:
        """sparsity() on SpatialRateResult should return JAX scalar."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate_jax = jnp.zeros(n_bins).at[n_bins // 2].set(10.0)
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRateResult(
            firing_rate=firing_rate_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        spars = result.sparsity()
        # Should be a JAX array (scalar), not a Python float
        assert isinstance(spars, jax.Array), (
            f"Expected jax.Array, got {type(spars)}. "
            "JAX arrays should be preserved through metric calls."
        )

    def test_spatial_information_preserves_jax_batch(
        self, simple_env: Environment
    ) -> None:
        """spatial_information() on SpatialRatesResult should return JAX array."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_np = np.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            firing_rates_np[i, i + 1] = 10.0
        firing_rates_jax = jnp.array(firing_rates_np)
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRatesResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        info = result.spatial_information()
        # Should be a JAX array, not NumPy
        assert isinstance(info, jax.Array), (
            f"Expected jax.Array, got {type(info)}. "
            "JAX arrays should be preserved through metric calls."
        )
        assert info.shape == (n_neurons,)

    def test_sparsity_preserves_jax_batch(self, simple_env: Environment) -> None:
        """sparsity() on SpatialRatesResult should return JAX array."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates_np = np.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            firing_rates_np[i, i + 1] = 10.0
        firing_rates_jax = jnp.array(firing_rates_np)
        occupancy_jax = jnp.ones(n_bins)

        result = SpatialRatesResult(
            firing_rates=firing_rates_jax,  # type: ignore[arg-type]
            occupancy=occupancy_jax,  # type: ignore[arg-type]
            env=simple_env,
            method="binned",
            bandwidth=5.0,
        )

        spars = result.sparsity()
        # Should be a JAX array, not NumPy
        assert isinstance(spars, jax.Array), (
            f"Expected jax.Array, got {type(spars)}. "
            "JAX arrays should be preserved through metric calls."
        )
        assert spars.shape == (n_neurons,)

    def test_metrics_module_preserves_jax_direct(self) -> None:
        """_metrics module should preserve JAX arrays when called directly."""
        import jax
        import jax.numpy as jnp

        from neurospatial.encoding._metrics import (
            batch_sparsity,
            batch_spatial_information,
            sparsity,
            spatial_information,
        )

        n_bins = 10
        n_neurons = 3

        # Single neuron test
        firing_rate_jax = jnp.zeros(n_bins).at[5].set(10.0)
        occupancy_jax = jnp.ones(n_bins)

        info = spatial_information(firing_rate_jax, occupancy_jax)
        spars = sparsity(firing_rate_jax, occupancy_jax)

        assert isinstance(info, jax.Array), f"Expected jax.Array, got {type(info)}"
        assert isinstance(spars, jax.Array), f"Expected jax.Array, got {type(spars)}"

        # Batch test
        firing_rates_jax = jnp.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            firing_rates_jax = firing_rates_jax.at[i, i + 1].set(10.0)

        info_batch = batch_spatial_information(firing_rates_jax, occupancy_jax)
        spars_batch = batch_sparsity(firing_rates_jax, occupancy_jax)

        assert isinstance(info_batch, jax.Array), (
            f"Expected jax.Array, got {type(info_batch)}"
        )
        assert isinstance(spars_batch, jax.Array), (
            f"Expected jax.Array, got {type(spars_batch)}"
        )
