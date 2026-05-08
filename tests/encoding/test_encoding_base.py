"""Tests for neurospatial.encoding._base module.

This module tests the shared infrastructure for encoding result classes:
- _to_numpy: Convert arrays (NumPy or JAX) to NumPy
- _get_array_module: Detect array backend (numpy or jax.numpy)
- HasOccupancy protocol
- HasEnvironment protocol
- SpatialResultMixin: Shared methods for spatial result classes

TDD approach: Tests written first, implementation follows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment

# ==============================================================================
# Helper function (must be defined before used in decorators)
# ==============================================================================


def _has_jax() -> bool:
    """Check if JAX is available on this platform."""
    import importlib.util

    return importlib.util.find_spec("jax") is not None


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


@pytest.fixture
def simple_1d_env() -> Environment:
    """Create a simple 1D-like environment for testing.

    Note: This creates a "narrow" 2D environment with minimal y-extent,
    simulating a 1D linear track. True 1D environments via from_graph
    require more complex setup with edge distances.
    """
    # Create a narrow strip that behaves like 1D
    positions = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=10.0)


# ==============================================================================
# Test _to_numpy helper
# ==============================================================================


class TestToNumpy:
    """Tests for _to_numpy helper function."""

    def test_numpy_array_passthrough(self) -> None:
        """NumPy arrays should pass through unchanged."""
        from neurospatial.encoding._base import _to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = _to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_list_to_numpy(self) -> None:
        """Python lists should be converted to NumPy arrays."""
        from neurospatial.encoding._base import _to_numpy

        arr = [1.0, 2.0, 3.0]
        result = _to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(arr))

    def test_tuple_to_numpy(self) -> None:
        """Python tuples should be converted to NumPy arrays."""
        from neurospatial.encoding._base import _to_numpy

        arr = (1.0, 2.0, 3.0)
        result = _to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(arr))

    def test_2d_array(self) -> None:
        """2D arrays should be preserved."""
        from neurospatial.encoding._base import _to_numpy

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.skipif(
        not _has_jax(), reason="JAX not installed or not available on this platform"
    )
    def test_jax_array_conversion(self) -> None:
        """JAX arrays should be converted to NumPy arrays."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import _to_numpy

        jax_arr = jnp.array([1.0, 2.0, 3.0])
        result = _to_numpy(jax_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


# ==============================================================================
# Test _get_array_module helper
# ==============================================================================


class TestGetArrayModule:
    """Tests for _get_array_module helper function."""

    def test_numpy_array_returns_numpy(self) -> None:
        """NumPy arrays should return numpy module."""
        from neurospatial.encoding._base import _get_array_module

        arr = np.array([1.0, 2.0, 3.0])
        xp = _get_array_module(arr)
        assert xp is np

    def test_list_returns_numpy(self) -> None:
        """Python lists should return numpy module (after conversion)."""
        from neurospatial.encoding._base import _get_array_module

        arr = [1.0, 2.0, 3.0]
        xp = _get_array_module(arr)
        # Lists don't have __jax_array__, so should return numpy
        assert xp is np

    @pytest.mark.skipif(
        not _has_jax(), reason="JAX not installed or not available on this platform"
    )
    def test_jax_array_returns_jax_numpy(self) -> None:
        """JAX arrays should return jax.numpy module."""
        import jax.numpy as jnp

        from neurospatial.encoding._base import _get_array_module

        jax_arr = jnp.array([1.0, 2.0, 3.0])
        xp = _get_array_module(jax_arr)
        assert xp is jnp


# ==============================================================================
# Test Protocols
# ==============================================================================


class TestHasOccupancyProtocol:
    """Tests for HasOccupancy protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """HasOccupancy protocol should be runtime checkable."""
        from neurospatial.encoding._base import HasOccupancy

        @dataclass
        class MockResult:
            occupancy: NDArray[np.float64]

        result = MockResult(occupancy=np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, HasOccupancy)

    def test_non_conforming_class_fails(self) -> None:
        """Classes without occupancy attribute should not satisfy protocol."""
        from neurospatial.encoding._base import HasOccupancy

        @dataclass
        class NoOccupancy:
            other_field: int

        result = NoOccupancy(other_field=42)
        assert not isinstance(result, HasOccupancy)


class TestHasEnvironmentProtocol:
    """Tests for HasEnvironment protocol."""

    def test_protocol_is_runtime_checkable(self, simple_env: Environment) -> None:
        """HasEnvironment protocol should be runtime checkable."""
        from neurospatial.encoding._base import HasEnvironment

        @dataclass
        class MockResult:
            env: Environment

        result = MockResult(env=simple_env)
        assert isinstance(result, HasEnvironment)

    def test_non_conforming_class_fails(self) -> None:
        """Classes without env attribute should not satisfy protocol."""
        from neurospatial.encoding._base import HasEnvironment

        @dataclass
        class NoEnv:
            other_field: int

        result = NoEnv(other_field=42)
        assert not isinstance(result, HasEnvironment)


# ==============================================================================
# Test SpatialResultMixin
# ==============================================================================


class TestSpatialResultMixin:
    """Tests for SpatialResultMixin class."""

    def test_peak_locations_single_neuron(self, simple_env: Environment) -> None:
        """peak_locations() should return (n_dims,) for single neuron result."""
        from neurospatial.encoding._base import SpatialResultMixin

        # Create a mock single-neuron result class
        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        # Create firing rate with peak at a specific bin
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins)
        peak_bin = n_bins // 2
        firing_rate[peak_bin] = 10.0

        result = MockSingleResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak = result.peak_locations()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (simple_env.n_dims,)
        # Peak should be at the bin center of the peak bin
        np.testing.assert_array_equal(peak, simple_env.bin_centers[peak_bin])

    def test_peak_locations_multiple_neurons(self, simple_env: Environment) -> None:
        """peak_locations() should return (n_neurons, n_dims) for batch result."""
        from neurospatial.encoding._base import SpatialResultMixin

        # Create a mock batch result class
        @dataclass
        class MockBatchResult(SpatialResultMixin):
            firing_rates: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        n_neurons = 3

        # Create firing rates with peaks at different bins
        firing_rates = np.zeros((n_neurons, n_bins))
        peak_bins = [1, n_bins // 2, n_bins - 2]
        for i, peak_bin in enumerate(peak_bins):
            firing_rates[i, peak_bin] = 10.0

        result = MockBatchResult(
            firing_rates=firing_rates,
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peaks = result.peak_locations()
        assert isinstance(peaks, np.ndarray)
        assert peaks.shape == (n_neurons, simple_env.n_dims)
        for i, peak_bin in enumerate(peak_bins):
            np.testing.assert_array_equal(peaks[i], simple_env.bin_centers[peak_bin])

    def test_peak_firing_rates_single_neuron(self, simple_env: Environment) -> None:
        """peak_firing_rates() should return scalar for single neuron result."""
        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        firing_rate = np.random.rand(n_bins) * 5.0
        peak_value = 15.0
        firing_rate[n_bins // 2] = peak_value

        result = MockSingleResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak_rate = result.peak_firing_rates()
        assert isinstance(peak_rate, float)
        assert peak_rate == peak_value

    def test_peak_firing_rates_multiple_neurons(self, simple_env: Environment) -> None:
        """peak_firing_rates() should return (n_neurons,) for batch result."""
        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockBatchResult(SpatialResultMixin):
            firing_rates: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        n_neurons = 3

        firing_rates = np.random.rand(n_neurons, n_bins) * 5.0
        peak_values = [10.0, 15.0, 20.0]
        for i, peak_val in enumerate(peak_values):
            firing_rates[i, i + 1] = peak_val

        result = MockBatchResult(
            firing_rates=firing_rates,
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak_rates = result.peak_firing_rates()
        assert isinstance(peak_rates, np.ndarray)
        assert peak_rates.shape == (n_neurons,)
        np.testing.assert_array_equal(peak_rates, peak_values)

    def test_peak_locations_with_nan(self, simple_env: Environment) -> None:
        """peak_locations() should handle NaN values correctly using nanargmax."""
        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins)
        peak_bin = n_bins // 2
        firing_rate[peak_bin] = 10.0
        # Add some NaN values
        firing_rate[0] = np.nan
        firing_rate[1] = np.nan

        result = MockSingleResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak = result.peak_locations()
        # Should still find the peak correctly, ignoring NaNs
        np.testing.assert_array_equal(peak, simple_env.bin_centers[peak_bin])

    def test_peak_firing_rates_with_nan(self, simple_env: Environment) -> None:
        """peak_firing_rates() should handle NaN values correctly using nanmax."""
        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins)
        peak_value = 15.0
        firing_rate[n_bins // 2] = peak_value
        # Add some NaN values
        firing_rate[0] = np.nan

        result = MockSingleResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins),
            env=simple_env,
        )

        peak_rate = result.peak_firing_rates()
        assert peak_rate == peak_value

    def test_peak_locations_1d_environment(self, simple_1d_env: Environment) -> None:
        """peak_locations() should work correctly for 1D environments."""
        from neurospatial.encoding._base import SpatialResultMixin

        @dataclass
        class MockSingleResult(SpatialResultMixin):
            firing_rate: NDArray[np.float64]
            occupancy: NDArray[np.float64]
            env: Environment

        n_bins = simple_1d_env.n_bins
        firing_rate = np.zeros(n_bins)
        peak_bin = 2
        firing_rate[peak_bin] = 10.0

        result = MockSingleResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins),
            env=simple_1d_env,
        )

        peak = result.peak_locations()
        assert isinstance(peak, np.ndarray)
        # 1D env still has (n_dims,) shape for bin_centers
        np.testing.assert_array_equal(peak, simple_1d_env.bin_centers[peak_bin])


# ==============================================================================
# Test imports
# ==============================================================================


class TestBaseImports:
    """Test that all expected items are importable from _base module."""

    def test_to_numpy_importable(self) -> None:
        """_to_numpy should be importable from encoding._base."""
        from neurospatial.encoding._base import _to_numpy

        assert callable(_to_numpy)

    def test_get_array_module_importable(self) -> None:
        """_get_array_module should be importable from encoding._base."""
        from neurospatial.encoding._base import _get_array_module

        assert callable(_get_array_module)

    def test_has_occupancy_importable(self) -> None:
        """HasOccupancy should be importable from encoding._base."""
        from neurospatial.encoding._base import HasOccupancy

        assert HasOccupancy is not None

    def test_has_environment_importable(self) -> None:
        """HasEnvironment should be importable from encoding._base."""
        from neurospatial.encoding._base import HasEnvironment

        assert HasEnvironment is not None

    def test_spatial_result_mixin_importable(self) -> None:
        """SpatialResultMixin should be importable from encoding._base."""
        from neurospatial.encoding._base import SpatialResultMixin

        assert SpatialResultMixin is not None


class TestValidateTrajectoryRejectsBadTimes:
    """Regression: validate_trajectory must reject empty / decreasing / NaN times.

    A previous version only checked ndim and length cross-alignment, so
    public ``compute_*`` entry points silently accepted degenerate inputs.
    """

    def test_empty_times_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_trajectory

        empty = np.empty(0, dtype=np.float64)
        with pytest.raises(ValueError, match=r"At least 2 samples"):
            validate_trajectory(empty)

    def test_single_sample_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_trajectory

        with pytest.raises(ValueError, match=r"At least 2 samples"):
            validate_trajectory(np.array([0.0]))

    def test_decreasing_times_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_trajectory

        decreasing = np.array([0.0, 1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match=r"monotonically non-decreasing"):
            validate_trajectory(decreasing)

    def test_nan_times_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_trajectory

        with_nan = np.array([0.0, 0.5, np.nan, 1.0])
        with pytest.raises(ValueError, match=r"finite"):
            validate_trajectory(with_nan)

    def test_compute_spatial_rate_rejects_decreasing_times(self) -> None:
        """End-to-end: the public entry point inherits the time validation."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial import compute_spatial_rate

        positions = np.linspace(0.0, 100.0, 10).reshape(-1, 1)
        env = Environment.from_samples(positions, bin_size=10.0)
        decreasing = np.array([0.0, 1.0, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        with pytest.raises(ValueError, match=r"monotonically non-decreasing"):
            compute_spatial_rate(env, np.array([0.5]), decreasing, positions)


class TestValidateSpikeTimes:
    """Regression: validate_spike_times rejects out-of-order / negative / NaN spikes.

    Internal binning paths use ``np.searchsorted`` against the spike-time
    array, so an out-of-order spike train silently produces wrong bin
    assignments. The public ``compute_*_rate(s)`` entry points should call
    this once on user input.
    """

    def test_empty_spike_train_accepted_by_default(self) -> None:
        """A neuron with zero spikes is a valid input (silent neuron)."""
        from neurospatial.encoding._validation import validate_spike_times

        validate_spike_times(np.empty(0, dtype=np.float64))  # does not raise

    def test_empty_rejected_when_disallowed(self) -> None:
        from neurospatial.encoding._validation import validate_spike_times

        with pytest.raises(ValueError, match=r"empty \(no spikes\)"):
            validate_spike_times(np.empty(0, dtype=np.float64), allow_empty=False)

    def test_two_dim_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_spike_times

        with pytest.raises(ValueError, match=r"1-D"):
            validate_spike_times(np.array([[0.5, 1.0], [1.5, 2.0]]))

    def test_nan_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_spike_times

        with pytest.raises(ValueError, match=r"finite"):
            validate_spike_times(np.array([0.0, np.nan, 1.0]))

    def test_inf_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_spike_times

        with pytest.raises(ValueError, match=r"finite"):
            validate_spike_times(np.array([0.0, 1.0, np.inf]))

    def test_negative_rejected(self) -> None:
        from neurospatial.encoding._validation import validate_spike_times

        with pytest.raises(ValueError, match=r"non-negative"):
            validate_spike_times(np.array([-0.1, 0.5, 1.0]))

    def test_decreasing_rejected_with_actionable_message(self) -> None:
        from neurospatial.encoding._validation import validate_spike_times

        with pytest.raises(ValueError, match=r"monotonically non-decreasing.*np\.sort"):
            validate_spike_times(np.array([0.0, 2.0, 1.0, 3.0]))

    def test_sorted_with_duplicates_accepted(self) -> None:
        """Equal-valued adjacent samples (simultaneous spikes) are allowed."""
        from neurospatial.encoding._validation import validate_spike_times

        validate_spike_times(np.array([0.0, 0.5, 0.5, 0.5, 1.0]))  # does not raise

    def test_compute_spatial_rate_rejects_unsorted_spike_times(self) -> None:
        """End-to-end: compute_spatial_rate inherits the spike-time validation."""
        from neurospatial import Environment
        from neurospatial.encoding.spatial import compute_spatial_rate

        positions = np.linspace(0.0, 100.0, 10).reshape(-1, 1)
        env = Environment.from_samples(positions, bin_size=10.0)
        times = np.linspace(0.0, 1.0, 10)
        unsorted_spikes = np.array([0.5, 0.1, 0.7])
        with pytest.raises(ValueError, match=r"monotonically non-decreasing"):
            compute_spatial_rate(env, unsorted_spikes, times, positions)

    def test_compute_directional_rate_rejects_unsorted_spike_times(self) -> None:
        """End-to-end: compute_directional_rate inherits the validation."""
        from neurospatial.encoding.directional import compute_directional_rate

        times = np.linspace(0.0, 10.0, 100)
        headings = np.linspace(-np.pi, np.pi, 100)
        unsorted_spikes = np.array([5.0, 1.0, 7.0])
        with pytest.raises(ValueError, match=r"monotonically non-decreasing"):
            compute_directional_rate(unsorted_spikes, times, headings)


class TestComputeRateRequiresFittedEnv:
    """Regression: env-consuming compute_*_rate(s) must reject unfitted envs.

    Previously these public entry points accepted a bare ``Environment()``
    or any object that quacked like one and crashed deep in a binning
    helper with a confusing AttributeError. They should fail at the
    boundary with EnvironmentNotFittedError so the user gets the
    canonical "use a factory method" guidance.
    """

    def _bare_env(self):
        """Construct an Environment without going through a factory.

        ``Environment()`` requires layout=..., so build a partially-
        initialized instance via the dataclass default path and then
        force ``_is_fitted = False``. This mirrors what users hit when
        they restore from a partial copy/subset/serialization that
        doesn't run ``_setup_from_layout``.
        """
        from neurospatial import Environment

        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        env = Environment.from_samples(positions, bin_size=1.0)
        env._is_fitted = False  # Simulate an unfitted env at function entry
        return env

    def test_compute_spatial_rate_rejects_unfitted_env(self) -> None:
        from neurospatial.encoding.spatial import compute_spatial_rate
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        times = np.linspace(0.0, 1.0, 10)
        positions = np.linspace(0.0, 2.0, 10).reshape(-1, 1)
        with pytest.raises(EnvironmentNotFittedError):
            compute_spatial_rate(env, np.array([0.5]), times, positions)

    def test_compute_spatial_rates_rejects_unfitted_env(self) -> None:
        from neurospatial.encoding.spatial import compute_spatial_rates
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        times = np.linspace(0.0, 1.0, 10)
        positions = np.linspace(0.0, 2.0, 10).reshape(-1, 1)
        with pytest.raises(EnvironmentNotFittedError):
            compute_spatial_rates(env, [np.array([0.5])], times, positions)

    def test_compute_view_rate_rejects_unfitted_env(self) -> None:
        from neurospatial.encoding.view import compute_view_rate
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        times = np.linspace(0.0, 1.0, 10)
        positions = np.column_stack([np.linspace(0, 2, 10), np.linspace(0, 2, 10)])
        headings = np.zeros(10)
        with pytest.raises(EnvironmentNotFittedError):
            compute_view_rate(env, np.array([0.5]), times, positions, headings)

    def test_compute_egocentric_rate_rejects_unfitted_env(self) -> None:
        """When env is supplied to egocentric, it must be fitted.

        compute_egocentric_rate accepts ``env=None`` for the euclidean
        distance path (the geodesic path needs the env-derived graph), so
        the fitted-state check is conditional on a non-None env.
        """
        from neurospatial.encoding.egocentric import compute_egocentric_rate
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        times = np.linspace(0.0, 1.0, 10)
        positions = np.column_stack([np.linspace(0, 2, 10), np.linspace(0, 2, 10)])
        headings = np.zeros(10)
        objects = np.array([[1.0, 1.0]])
        with pytest.raises(EnvironmentNotFittedError):
            compute_egocentric_rate(
                env, np.array([0.5]), times, positions, headings, objects
            )

    def test_compute_view_rates_rejects_unfitted_env(self) -> None:
        from neurospatial.encoding.view import compute_view_rates
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        times = np.linspace(0.0, 1.0, 10)
        positions = np.column_stack([np.linspace(0, 2, 10), np.linspace(0, 2, 10)])
        headings = np.zeros(10)
        with pytest.raises(EnvironmentNotFittedError):
            compute_view_rates(env, [np.array([0.5])], times, positions, headings)

    def test_compute_egocentric_rates_rejects_unfitted_env(self) -> None:
        from neurospatial.encoding.egocentric import compute_egocentric_rates
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        times = np.linspace(0.0, 1.0, 10)
        positions = np.column_stack([np.linspace(0, 2, 10), np.linspace(0, 2, 10)])
        headings = np.zeros(10)
        objects = np.array([[1.0, 1.0]])
        with pytest.raises(EnvironmentNotFittedError):
            compute_egocentric_rates(
                env, [np.array([0.5])], times, positions, headings, objects
            )

    def test_decode_position_rejects_unfitted_env(self) -> None:
        from neurospatial.decoding.posterior import decode_position
        from neurospatial.environment.decorators import EnvironmentNotFittedError

        env = self._bare_env()
        # Sentinel inputs sized for one neuron x one bin: the env-fitted
        # check should fire before downstream shape/dtype checks.
        spike_counts = np.zeros((1, 1), dtype=np.int64)
        encoding_models = np.zeros((1, 1), dtype=np.float64)
        with pytest.raises(EnvironmentNotFittedError):
            decode_position(env, spike_counts, encoding_models, dt=0.01)
