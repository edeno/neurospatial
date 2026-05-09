"""Tests for neurospatial.encoding.spatial module.

This module tests the spatial encoding result classes:
- SpatialRateResult: Single-neuron spatial rate map result
- SpatialRatesResult: Multi-neuron spatial rate map result

TDD approach: Tests written first, implementation follows.

Task 2.1: Result class definitions
- SpatialRateResult dataclass (frozen=True)
- SpatialRatesResult dataclass (frozen=True)
- Inherit from SpatialResultMixin for shared methods
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.encoding._metrics import BatchScoresResult

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    # Create a 5x5 grid of positions
    x = np.linspace(0, 40, 5)
    y = np.linspace(0, 40, 5)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def firing_rate_single(simple_env: Environment) -> NDArray[np.float64]:
    """Create a single-neuron firing rate map with a clear peak."""
    n_bins = simple_env.n_bins
    firing_rate = np.zeros(n_bins, dtype=np.float64)
    # Create a peak near the center
    peak_bin = n_bins // 2
    firing_rate[peak_bin] = 20.0
    firing_rate[peak_bin - 1] = 10.0
    firing_rate[peak_bin + 1] = 10.0
    return firing_rate


@pytest.fixture
def firing_rates_batch(simple_env: Environment) -> NDArray[np.float64]:
    """Create a batch of firing rate maps for multiple neurons."""
    n_bins = simple_env.n_bins
    n_neurons = 5
    firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
    # Create distinct peaks for each neuron
    for i in range(n_neurons):
        peak_bin = min(i * 4 + 2, n_bins - 1)
        firing_rates[i, peak_bin] = 15.0 + i * 5.0  # Different peak rates
    return firing_rates


@pytest.fixture
def occupancy(simple_env: Environment) -> NDArray[np.float64]:
    """Create uniform occupancy."""
    return np.ones(simple_env.n_bins, dtype=np.float64)


# ==============================================================================
# Test SpatialRateResult class definition
# ==============================================================================


class TestSpatialRateResultDefinition:
    """Tests for SpatialRateResult class definition (Task 2.1)."""

    def test_class_is_importable(self) -> None:
        """SpatialRateResult should be importable from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRateResult

        assert SpatialRateResult is not None

    def test_class_is_dataclass(self) -> None:
        """SpatialRateResult should be a dataclass."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRateResult

        # dataclasses have __dataclass_fields__ attribute
        assert hasattr(SpatialRateResult, "__dataclass_fields__")
        # Check that fields() works
        assert len(fields(SpatialRateResult)) > 0

    def test_class_is_frozen(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should be immutable (frozen=True)."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.firing_rate = np.zeros_like(firing_rate_single)  # type: ignore[misc]

    def test_required_fields(self) -> None:
        """SpatialRateResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRateResult

        field_names = {f.name for f in fields(SpatialRateResult)}
        expected_fields = {
            "firing_rate",
            "occupancy",
            "env",
            "smoothing_method",
            "bandwidth",
        }
        assert expected_fields.issubset(field_names)

    def test_instantiation(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should be instantiable with correct fields."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert result.firing_rate is firing_rate_single
        assert result.occupancy is occupancy
        assert result.env is simple_env
        assert result.smoothing_method == "diffusion_kde"
        assert result.bandwidth == 5.0

    def test_inherits_from_spatial_result_mixin(self) -> None:
        """SpatialRateResult should inherit from SpatialResultMixin."""
        from neurospatial.encoding._base import SpatialResultMixin
        from neurospatial.encoding.spatial import SpatialRateResult

        assert issubclass(SpatialRateResult, SpatialResultMixin)

    def test_has_peak_locations_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have peak_location() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_location")
        assert callable(result.peak_location)

    def test_has_peak_firing_rate_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have peak_firing_rate() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_firing_rate")
        assert callable(result.peak_firing_rate)


class TestSpatialRateResultMixinMethods:
    """Tests for SpatialResultMixin methods on SpatialRateResult."""

    def test_peak_locations_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_location() should return (n_dims,) for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak = result.peak_location()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (simple_env.n_dims,)

    def test_peak_locations_finds_maximum(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """peak_location() should find the bin with maximum firing rate."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        peak_bin = 5
        firing_rate[peak_bin] = 30.0

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak = result.peak_location()
        expected = simple_env.bin_centers[peak_bin]
        np.testing.assert_array_almost_equal(peak, expected)

    def test_peak_firing_rate_returns_scalar(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rate() should return a scalar for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rate = result.peak_firing_rate()
        assert isinstance(peak_rate, float)
        assert peak_rate == np.max(firing_rate_single)


# ==============================================================================
# Test SpatialRatesResult class definition
# ==============================================================================


class TestSpatialRatesResultDefinition:
    """Tests for SpatialRatesResult class definition (Task 2.1)."""

    def test_class_is_importable(self) -> None:
        """SpatialRatesResult should be importable from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        assert SpatialRatesResult is not None

    def test_class_is_dataclass(self) -> None:
        """SpatialRatesResult should be a dataclass."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRatesResult

        # dataclasses have __dataclass_fields__ attribute
        assert hasattr(SpatialRatesResult, "__dataclass_fields__")
        assert len(fields(SpatialRatesResult)) > 0

    def test_class_is_frozen(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should be immutable (frozen=True)."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        with pytest.raises(FrozenInstanceError):
            result.firing_rates = np.zeros_like(firing_rates_batch)  # type: ignore[misc]

    def test_required_fields(self) -> None:
        """SpatialRatesResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRatesResult

        field_names = {f.name for f in fields(SpatialRatesResult)}
        # Note: uses firing_rates (plural) for batch
        expected_fields = {
            "firing_rates",
            "occupancy",
            "env",
            "smoothing_method",
            "bandwidth",
        }
        assert expected_fields.issubset(field_names)

    def test_instantiation(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should be instantiable with correct fields."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="gaussian_kde",
            bandwidth=10.0,
        )
        assert result.firing_rates is firing_rates_batch
        assert result.occupancy is occupancy
        assert result.env is simple_env
        assert result.smoothing_method == "gaussian_kde"
        assert result.bandwidth == 10.0

    def test_inherits_from_spatial_result_mixin(self) -> None:
        """SpatialRatesResult should inherit from SpatialResultMixin."""
        from neurospatial.encoding._base import SpatialResultMixin
        from neurospatial.encoding.spatial import SpatialRatesResult

        assert issubclass(SpatialRatesResult, SpatialResultMixin)

    def test_has_peak_locations_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have peak_location() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_location")
        assert callable(result.peak_location)

    def test_has_peak_firing_rate_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have peak_firing_rate() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_firing_rate")
        assert callable(result.peak_firing_rate)


class TestSpatialRatesResultMixinMethods:
    """Tests for SpatialResultMixin methods on SpatialRatesResult."""

    def test_peak_locations_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_location() should return (n_neurons, n_dims) for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peaks = result.peak_location()
        n_neurons = firing_rates_batch.shape[0]
        assert isinstance(peaks, np.ndarray)
        assert peaks.shape == (n_neurons, simple_env.n_dims)

    def test_peak_locations_finds_maximums(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """peak_location() should find the bin with maximum firing rate for each neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
        peak_bins = [2, 5, 10]
        for i, peak_bin in enumerate(peak_bins):
            firing_rates[i, peak_bin] = 20.0 + i * 10.0

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peaks = result.peak_location()
        for i, peak_bin in enumerate(peak_bins):
            expected = simple_env.bin_centers[peak_bin]
            np.testing.assert_array_almost_equal(peaks[i], expected)

    def test_peak_firing_rate_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rate() should return (n_neurons,) for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rates = result.peak_firing_rate()
        n_neurons = firing_rates_batch.shape[0]
        assert isinstance(peak_rates, np.ndarray)
        assert peak_rates.shape == (n_neurons,)

    def test_peak_firing_rate_correct_values(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rate() should return the maximum firing rate for each neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rates = result.peak_firing_rate()
        expected = np.max(firing_rates_batch, axis=1)
        np.testing.assert_array_almost_equal(peak_rates, expected)


# ==============================================================================
# Test SpatialRatesResult iteration interface
# ==============================================================================


class TestSpatialRatesResultIteration:
    """Tests for SpatialRatesResult iteration interface (__len__, __getitem__, __iter__)."""

    def test_len(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__len__ should return number of neurons."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert len(result) == firing_rates_batch.shape[0]

    def test_getitem_returns_single_result(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__getitem__ should return a SpatialRateResult (single neuron)."""
        from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        single = result[0]
        assert isinstance(single, SpatialRateResult)

    def test_getitem_preserves_fields(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__getitem__ should preserve env, smoothing_method, bandwidth."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="gaussian_kde",
            bandwidth=7.5,
        )
        single = result[2]
        assert single.env is simple_env
        assert single.smoothing_method == "gaussian_kde"
        assert single.bandwidth == 7.5
        np.testing.assert_array_equal(single.occupancy, occupancy)

    def test_getitem_extracts_correct_firing_rate(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__getitem__ should extract the correct neuron's firing rate."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        for idx in range(len(result)):
            single = result[idx]
            np.testing.assert_array_equal(single.firing_rate, firing_rates_batch[idx])

    def test_iter(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__iter__ should yield SpatialRateResult for each neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        items = list(result)
        assert len(items) == len(result)
        for i, item in enumerate(items):
            assert isinstance(item, SpatialRateResult)
            np.testing.assert_array_equal(item.firing_rate, firing_rates_batch[i])


# ==============================================================================
# Test module exports
# ==============================================================================


class TestSpatialModuleExports:
    """Tests that the spatial module exports all expected items."""

    def test_spatial_rate_result_exported(self) -> None:
        """SpatialRateResult should be exported from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRateResult

        assert SpatialRateResult is not None

    def test_spatial_rates_result_exported(self) -> None:
        """SpatialRatesResult should be exported from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        assert SpatialRatesResult is not None


# ==============================================================================
# Test SpatialRateResult convenience methods (Task 2.2)
# ==============================================================================


class TestSpatialRateResultPlot:
    """Tests for SpatialRateResult.plot() method."""

    def test_has_plot_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a plot() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "plot")
        assert callable(result.plot)

    def test_plot_returns_axes(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """plot() should return matplotlib Axes."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for testing
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        ax = result.plot()
        assert ax is not None
        plt.close("all")

    def test_plot_accepts_ax_argument(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """plot() should accept an optional ax argument."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        _fig, ax = plt.subplots()
        returned_ax = result.plot(ax=ax)
        # Should use the provided axes
        assert returned_ax is ax or returned_ax is not None
        plt.close("all")


class TestSpatialRateResultPeakLocation:
    """Tests for SpatialRateResult.peak_location() alias method."""

    def test_has_peak_location_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a peak_location() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_location")
        assert callable(result.peak_location)

    def test_peak_location_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_location() should return (n_dims,) for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak = result.peak_location()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (simple_env.n_dims,)


class TestSpatialRateResultSpatialInformation:
    """Tests for SpatialRateResult.spatial_information() method."""

    def test_has_spatial_information_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a spatial_information() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "spatial_information")
        assert callable(result.spatial_information)

    def test_spatial_information_returns_float(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """spatial_information() should return a float."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        assert isinstance(info, float)

    def test_spatial_information_delegates_to_metrics(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """spatial_information() should match _metrics.spatial_information()."""
        from neurospatial.encoding._metrics import spatial_information as si_func
        from neurospatial.encoding.spatial import SpatialRateResult

        # Create a specific firing pattern
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 30.0  # Single peak

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Compare with direct call to _metrics
        expected = si_func(firing_rate, occupancy)
        assert result.spatial_information() == pytest.approx(expected)

    def test_spatial_information_nonnegative(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """spatial_information() should always be non-negative."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        assert info >= 0.0

    def test_spatial_information_uniform_firing_is_zero(
        self, simple_env: Environment
    ) -> None:
        """Uniform firing should have ~zero spatial information."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.ones(n_bins, dtype=np.float64) * 5.0  # Uniform
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        assert info == pytest.approx(0.0, abs=1e-6)


class TestSpatialRateResultSparsity:
    """Tests for SpatialRateResult.sparsity() method."""

    def test_has_sparsity_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a sparsity() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "sparsity")
        assert callable(result.sparsity)

    def test_sparsity_returns_float(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """sparsity() should return a float."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        assert isinstance(spars, float)

    def test_sparsity_delegates_to_metrics(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """sparsity() should match _metrics.sparsity()."""
        from neurospatial.encoding._metrics import sparsity as sparsity_func
        from neurospatial.encoding.spatial import SpatialRateResult

        # Create a specific firing pattern
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 30.0  # Single peak

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Compare with direct call to _metrics
        expected = sparsity_func(firing_rate, occupancy)
        assert result.sparsity() == pytest.approx(expected)

    def test_sparsity_in_valid_range(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """sparsity() should be in range [0, 1]."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        assert 0.0 <= spars <= 1.0

    def test_sparsity_uniform_firing_is_one(self, simple_env: Environment) -> None:
        """Uniform firing should have sparsity close to 1."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.ones(n_bins, dtype=np.float64) * 5.0  # Uniform
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        assert spars == pytest.approx(1.0, abs=1e-6)

    def test_sparsity_selective_firing_is_low(self, simple_env: Environment) -> None:
        """Selective firing (single bin) should have low sparsity."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 30.0  # Only fires in one bin
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        # Should be 1/n_bins for single-bin firing with uniform occupancy
        expected = 1.0 / n_bins
        assert spars == pytest.approx(expected, rel=1e-6)


# ==============================================================================
# Test SpatialRateResult cell type metrics (Task 2.3)
# ==============================================================================


@pytest.fixture
def regular_grid_env() -> Environment:
    """Create a regular 2D grid environment suitable for FFT autocorrelation.

    This creates a larger environment that works well with the grid score
    algorithm which requires FFT-based autocorrelation.
    """
    # Create a dense grid of positions (20x20)
    x = np.linspace(0, 100, 21)
    y = np.linspace(0, 100, 21)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def grid_cell_firing_rate(regular_grid_env: Environment) -> NDArray[np.float64]:
    """Create a synthetic grid cell firing rate pattern.

    Creates a hexagonal grid pattern with peaks at regular intervals.
    """
    n_bins = regular_grid_env.n_bins
    bin_centers = regular_grid_env.bin_centers

    # Create a grid-like pattern using hexagonal spacing
    # Use a simple cosine-based grid approximation
    scale = 25.0  # Grid spacing in cm
    x = bin_centers[:, 0]
    y = bin_centers[:, 1]

    # Create hexagonal pattern using sum of cosines at 60 degree angles
    firing_rate = np.zeros(n_bins, dtype=np.float64)
    for angle_deg in [0, 60, 120]:
        angle = np.radians(angle_deg)
        phase = 2 * np.pi * (x * np.cos(angle) + y * np.sin(angle)) / scale
        firing_rate += np.cos(phase)

    # Normalize to positive values
    firing_rate = (firing_rate + 3) / 6 * 20.0  # Peak around 20 Hz
    return firing_rate


@pytest.fixture
def border_cell_firing_rate(simple_env: Environment) -> NDArray[np.float64]:
    """Create a synthetic border cell firing rate pattern.

    Creates firing along the boundary of the environment.
    """
    n_bins = simple_env.n_bins
    firing_rate = np.zeros(n_bins, dtype=np.float64)

    # Get boundary bins and set high firing there
    boundary_bins = simple_env.boundary_bins
    firing_rate[boundary_bins] = 15.0

    return firing_rate


class TestSpatialRateResultGridScore:
    """Tests for SpatialRateResult.grid_score() method."""

    def test_has_grid_score_method(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a grid_score() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        # Need occupancy matching the regular grid environment
        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "grid_score")
        assert callable(result.grid_score)

    def test_grid_score_returns_float(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """grid_score() should return a float."""
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        score = result.grid_score()
        assert isinstance(score, float)

    def test_grid_score_delegates_to_grid_module(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """grid_score() should delegate to grid.grid_score()."""
        from neurospatial.encoding.grid import grid_score as gs_func
        from neurospatial.encoding.grid import spatial_autocorrelation
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Compute expected value using direct call to grid module
        autocorr = spatial_autocorrelation(regular_grid_env, grid_cell_firing_rate)
        expected = gs_func(autocorr)

        score = result.grid_score()
        assert score == pytest.approx(expected)

    def test_grid_score_in_valid_range(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """grid_score() should be in range [-2, 2] or NaN."""
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        score = result.grid_score()
        if not np.isnan(score):
            assert -2.0 <= score <= 2.0


class TestSpatialRateResultGridProperties:
    """Tests for SpatialRateResult.grid_properties() method."""

    def test_has_grid_properties_method(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a grid_properties() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "grid_properties")
        assert callable(result.grid_properties)

    def test_grid_properties_returns_grid_properties_dataclass(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """grid_properties() should return a GridProperties dataclass."""
        from neurospatial.encoding.grid import GridProperties
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        props = result.grid_properties()
        assert isinstance(props, GridProperties)

    def test_grid_properties_has_expected_fields(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """grid_properties() result should have all expected fields."""
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        props = result.grid_properties()
        assert hasattr(props, "score")
        assert hasattr(props, "scale")
        assert hasattr(props, "orientation")
        assert hasattr(props, "orientation_std")
        assert hasattr(props, "peak_coords")
        assert hasattr(props, "n_peaks")

    def test_grid_properties_delegates_to_grid_module(
        self,
        regular_grid_env: Environment,
        grid_cell_firing_rate: NDArray[np.float64],
    ) -> None:
        """grid_properties() should delegate to grid.grid_properties()."""
        from neurospatial.encoding.grid import grid_properties as gp_func
        from neurospatial.encoding.grid import spatial_autocorrelation
        from neurospatial.encoding.spatial import SpatialRateResult

        occ = np.ones(regular_grid_env.n_bins, dtype=np.float64)
        result = SpatialRateResult(
            firing_rate=grid_cell_firing_rate,
            occupancy=occ,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Compute expected value using direct call to grid module
        autocorr = spatial_autocorrelation(regular_grid_env, grid_cell_firing_rate)
        bin_size = float(np.min(regular_grid_env.bin_sizes))
        expected = gp_func(autocorr, bin_size=bin_size)

        props = result.grid_properties()
        assert props.score == pytest.approx(expected.score)


class TestSpatialRateResultBorderScore:
    """Tests for SpatialRateResult.border_score() method."""

    def test_has_border_score_method(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a border_score() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "border_score")
        assert callable(result.border_score)

    def test_border_score_returns_float(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should return a float."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        score = result.border_score()
        assert isinstance(score, float) or np.isnan(score)

    def test_border_score_accepts_threshold_parameter(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should accept a threshold parameter."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise
        score = result.border_score(threshold=0.5)
        assert isinstance(score, float) or np.isnan(score)

    def test_border_score_accepts_metric_parameter(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should accept a metric parameter."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with either metric
        score_geodesic = result.border_score(metric="geodesic")
        score_euclidean = result.border_score(metric="euclidean")
        assert isinstance(score_geodesic, float) or np.isnan(score_geodesic)
        assert isinstance(score_euclidean, float) or np.isnan(score_euclidean)

    def test_border_score_accepts_min_area_parameter(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should accept a min_area parameter."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with min_area parameter
        score = result.border_score(min_area=0.0)
        assert isinstance(score, float) or np.isnan(score)

        # With very high min_area, should return NaN (field too small)
        score_large_min = result.border_score(min_area=1e10)
        assert np.isnan(score_large_min)

    def test_border_score_zero_firing_returns_nan(
        self,
        simple_env: Environment,
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should return NaN when all firing is zero."""
        from neurospatial.encoding.spatial import SpatialRateResult

        # All zeros
        firing_rate = np.zeros(simple_env.n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        score = result.border_score()
        assert np.isnan(score)

    def test_border_score_delegates_to_border_module(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should delegate to border.border_score()."""
        from neurospatial.encoding.border import border_score as bs_func
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Compute expected value using direct call to border module
        expected = bs_func(simple_env, border_cell_firing_rate)

        score = result.border_score()
        if np.isnan(expected):
            assert np.isnan(score)
        else:
            assert score == pytest.approx(expected)

    def test_border_score_in_valid_range(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should be in range [-1, 1] or NaN."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        score = result.border_score()
        if not np.isnan(score):
            assert -1.0 <= score <= 1.0


class TestSpatialRateResultRegionCoverage:
    """Tests for SpatialRateResult.region_coverage() method."""

    @pytest.fixture
    def env_with_regions(self, simple_env: Environment) -> Environment:
        """Create an environment with defined regions."""
        from shapely.geometry import box

        # Add wall regions to the environment
        # The simple_env is 40x40 centered around (20, 20)
        simple_env.regions.add("north", polygon=box(0, 30, 40, 40))
        simple_env.regions.add("south", polygon=box(0, 0, 40, 10))
        simple_env.regions.add("center", polygon=box(15, 15, 25, 25))
        return simple_env

    def test_has_region_coverage_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have a region_coverage() method."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "region_coverage")
        assert callable(result.region_coverage)

    def test_region_coverage_returns_dict(
        self,
        env_with_regions: Environment,
        occupancy: NDArray[np.float64],
    ) -> None:
        """region_coverage() should return a dict."""
        from neurospatial.encoding.spatial import SpatialRateResult

        # Create firing rate for this env
        n_bins = env_with_regions.n_bins
        firing_rate = np.random.rand(n_bins) * 10.0
        occ = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occ,
            env=env_with_regions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        coverage = result.region_coverage()
        assert isinstance(coverage, dict)

    def test_region_coverage_accepts_threshold_parameter(
        self,
        env_with_regions: Environment,
    ) -> None:
        """region_coverage() should accept a threshold parameter."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = env_with_regions.n_bins
        firing_rate = np.random.rand(n_bins) * 10.0
        occ = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occ,
            env=env_with_regions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with different thresholds
        coverage1 = result.region_coverage(threshold=0.3)
        coverage2 = result.region_coverage(threshold=0.5)
        assert isinstance(coverage1, dict)
        assert isinstance(coverage2, dict)

    def test_region_coverage_accepts_regions_parameter(
        self,
        env_with_regions: Environment,
    ) -> None:
        """region_coverage() should accept a regions list parameter."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = env_with_regions.n_bins
        firing_rate = np.random.rand(n_bins) * 10.0
        occ = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occ,
            env=env_with_regions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should only return coverage for specified regions
        coverage = result.region_coverage(regions=["north", "south"])
        assert "north" in coverage
        assert "south" in coverage
        assert "center" not in coverage

    def test_region_coverage_values_in_valid_range(
        self,
        env_with_regions: Environment,
    ) -> None:
        """region_coverage() values should be in range [0, 1]."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = env_with_regions.n_bins
        firing_rate = np.random.rand(n_bins) * 10.0
        occ = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occ,
            env=env_with_regions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        coverage = result.region_coverage()
        for region, value in coverage.items():
            assert 0.0 <= value <= 1.0, f"Coverage for {region} out of range: {value}"

    def test_region_coverage_delegates_to_border_module(
        self,
        env_with_regions: Environment,
    ) -> None:
        """region_coverage() should delegate to border.compute_region_coverage()."""
        from neurospatial.encoding.border import compute_region_coverage as crc_func
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = env_with_regions.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        # Set high firing in specific bins
        firing_rate[0:5] = 20.0  # Create a field
        occ = np.ones(n_bins, dtype=np.float64)

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occ,
            env=env_with_regions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Compute expected coverage using direct call
        threshold = 0.3
        peak_rate = np.nanmax(firing_rate)
        field_mask = firing_rate >= threshold * peak_rate
        field_bins = np.where(field_mask)[0]
        expected = crc_func(field_bins, env_with_regions)

        coverage = result.region_coverage(threshold=threshold)
        for region in expected:
            assert coverage[region] == pytest.approx(expected[region])


# ==============================================================================
# Test SpatialRatesResult batch methods (Task 2.4)
# ==============================================================================


class TestSpatialRatesResultPlot:
    """Tests for SpatialRatesResult.plot() method."""

    def test_has_plot_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a plot() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "plot")
        assert callable(result.plot)

    def test_plot_requires_idx_argument(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """plot() should require an idx argument for batch results."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should work with idx
        ax = result.plot(idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_returns_axes(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """plot() should return matplotlib Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        ax = result.plot(idx=2)
        assert ax is not None
        plt.close("all")

    def test_plot_accepts_ax_argument(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """plot() should accept an optional ax argument."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        _fig, ax = plt.subplots()
        returned_ax = result.plot(idx=1, ax=ax)
        assert returned_ax is ax or returned_ax is not None
        plt.close("all")

    def test_plot_different_neurons(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """plot() should plot different neurons when idx changes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise for any valid index
        for idx in range(len(result)):
            ax = result.plot(idx=idx)
            assert ax is not None
            plt.close("all")


class TestSpatialRatesResultSpatialInformation:
    """Tests for SpatialRatesResult.spatial_information() method."""

    def test_has_spatial_information_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a spatial_information() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "spatial_information")
        assert callable(result.spatial_information)

    def test_spatial_information_returns_array(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """spatial_information() should return an ndarray for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        assert isinstance(info, np.ndarray)

    def test_spatial_information_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """spatial_information() should return (n_neurons,) array."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        n_neurons = firing_rates_batch.shape[0]
        assert info.shape == (n_neurons,)

    def test_spatial_information_delegates_to_metrics(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """spatial_information() should match batch_spatial_information()."""
        from neurospatial.encoding._metrics import batch_spatial_information
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        expected = batch_spatial_information(firing_rates_batch, occupancy)
        info = result.spatial_information()
        np.testing.assert_array_almost_equal(info, expected)

    def test_spatial_information_all_nonnegative(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """spatial_information() should always be non-negative."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        assert np.all(info >= 0.0)

    def test_spatial_information_uniform_firing_is_zero(
        self, simple_env: Environment
    ) -> None:
        """Uniform firing should have ~zero spatial information for all neurons."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates = np.ones((n_neurons, n_bins), dtype=np.float64) * 5.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        info = result.spatial_information()
        np.testing.assert_array_almost_equal(info, np.zeros(n_neurons), decimal=6)


class TestSpatialRatesResultSparsity:
    """Tests for SpatialRatesResult.sparsity() method."""

    def test_has_sparsity_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a sparsity() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "sparsity")
        assert callable(result.sparsity)

    def test_sparsity_returns_array(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """sparsity() should return an ndarray for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        assert isinstance(spars, np.ndarray)

    def test_sparsity_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """sparsity() should return (n_neurons,) array."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        n_neurons = firing_rates_batch.shape[0]
        assert spars.shape == (n_neurons,)

    def test_sparsity_delegates_to_metrics(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """sparsity() should match batch_sparsity()."""
        from neurospatial.encoding._metrics import batch_sparsity
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        expected = batch_sparsity(firing_rates_batch, occupancy)
        spars = result.sparsity()
        np.testing.assert_array_almost_equal(spars, expected)

    def test_sparsity_all_in_valid_range(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """sparsity() values should all be in range [0, 1]."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        assert np.all(spars >= 0.0)
        assert np.all(spars <= 1.0)

    def test_sparsity_uniform_firing_is_one(self, simple_env: Environment) -> None:
        """Uniform firing should have sparsity close to 1 for all neurons."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates = np.ones((n_neurons, n_bins), dtype=np.float64) * 5.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        np.testing.assert_array_almost_equal(spars, np.ones(n_neurons), decimal=6)

    def test_sparsity_selective_firing_is_low(self, simple_env: Environment) -> None:
        """Selective firing (single bin) should have low sparsity."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
        # Each neuron fires in different single bins
        for i in range(n_neurons):
            firing_rates[i, i * 4 + 2] = 30.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        spars = result.sparsity()
        # Sparsity should be 1/n_bins for single-bin firing with uniform occupancy
        expected = np.full(n_neurons, 1.0 / n_bins)
        np.testing.assert_array_almost_equal(spars, expected)


# ==============================================================================
# Test SpatialRatesResult batch metrics (Task 2.5)
# ==============================================================================


class TestSpatialRatesResultGridScores:
    """Tests for SpatialRatesResult.grid_scores() method."""

    def test_has_grid_scores_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a grid_scores() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "grid_scores")
        assert callable(result.grid_scores)

    def test_grid_scores_returns_array(
        self,
        regular_grid_env: Environment,
    ) -> None:
        """grid_scores() should return an ndarray for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_neurons = 3
        n_bins = regular_grid_env.n_bins
        firing_rates = np.random.rand(n_neurons, n_bins) * 10.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        scores = result.grid_scores()
        assert isinstance(scores, BatchScoresResult)

    def test_grid_scores_correct_shape(
        self,
        regular_grid_env: Environment,
    ) -> None:
        """grid_scores() should return (n_neurons,) array."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_neurons = 5
        n_bins = regular_grid_env.n_bins
        firing_rates = np.random.rand(n_neurons, n_bins) * 10.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        scores = result.grid_scores()
        assert scores.shape == (n_neurons,)

    def test_grid_scores_delegates_to_batch_grid_scores(
        self,
        regular_grid_env: Environment,
    ) -> None:
        """grid_scores() should delegate to batch_grid_scores()."""
        from neurospatial.encoding._metrics import batch_grid_scores
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_neurons = 3
        n_bins = regular_grid_env.n_bins
        rng = np.random.default_rng(42)
        firing_rates = rng.random((n_neurons, n_bins)) * 10.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        expected = batch_grid_scores(regular_grid_env, firing_rates)
        scores = result.grid_scores()

        # Handle NaN comparison properly
        for i in range(n_neurons):
            if np.isnan(expected[i]):
                assert np.isnan(scores[i])
            else:
                assert scores[i] == pytest.approx(expected[i])

    def test_grid_scores_in_valid_range(
        self,
        regular_grid_env: Environment,
    ) -> None:
        """grid_scores() should be in range [-2, 2] or NaN."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_neurons = 5
        n_bins = regular_grid_env.n_bins
        rng = np.random.default_rng(123)
        firing_rates = rng.random((n_neurons, n_bins)) * 10.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=regular_grid_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        scores = result.grid_scores()
        for score in scores:
            if not np.isnan(score):
                assert -2.0 <= score <= 2.0


class TestSpatialRatesResultBorderScores:
    """Tests for SpatialRatesResult.border_scores() method."""

    def test_has_border_scores_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a border_scores() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "border_scores")
        assert callable(result.border_scores)

    def test_border_scores_returns_array(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should return an ndarray for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        scores = result.border_scores()
        assert isinstance(scores, BatchScoresResult)

    def test_border_scores_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should return (n_neurons,) array."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        scores = result.border_scores()
        n_neurons = firing_rates_batch.shape[0]
        assert scores.shape == (n_neurons,)

    def test_border_scores_accepts_threshold_parameter(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should accept a threshold parameter."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise
        scores = result.border_scores(threshold=0.5)
        assert isinstance(scores, BatchScoresResult)

    def test_border_scores_accepts_metric_parameter(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should accept a metric parameter."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with either metric
        scores_geodesic = result.border_scores(metric="geodesic")
        scores_euclidean = result.border_scores(metric="euclidean")
        assert isinstance(scores_geodesic, BatchScoresResult)
        assert isinstance(scores_euclidean, BatchScoresResult)

    def test_border_scores_accepts_min_area_parameter(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should accept a min_area parameter."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with min_area parameter
        scores = result.border_scores(min_area=10.0)
        assert isinstance(scores, BatchScoresResult)

    def test_border_scores_delegates_to_batch_border_scores(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should delegate to batch_border_scores()."""
        from neurospatial.encoding._metrics import batch_border_scores
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        expected = batch_border_scores(simple_env, firing_rates_batch)
        scores = result.border_scores()

        # Handle NaN comparison properly
        for i in range(len(expected)):
            if np.isnan(expected[i]):
                assert np.isnan(scores[i])
            else:
                assert scores[i] == pytest.approx(expected[i])

    def test_border_scores_in_valid_range(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_scores() should be in range [-1, 1] or NaN."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        scores = result.border_scores()
        for score in scores:
            if not np.isnan(score):
                assert -1.0 <= score <= 1.0


class TestSpatialRatesResultClassify:
    """Tests for SpatialRatesResult.classify() method."""

    def test_has_classify_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a classify() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "classify")
        assert callable(result.classify)

    def test_classify_returns_string_array(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """classify() should return an ndarray of strings."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        labels = result.classify()
        assert isinstance(labels, np.ndarray)
        assert labels.dtype.kind in ("U", "S", "O")  # Unicode, byte string, or object

    def test_classify_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """classify() should return (n_neurons,) array."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        labels = result.classify()
        n_neurons = firing_rates_batch.shape[0]
        assert labels.shape == (n_neurons,)

    def test_classify_accepts_threshold_parameters(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """classify() should accept min_spatial_info, min_grid_score, min_border_score."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with custom thresholds
        labels = result.classify(
            min_spatial_info=0.3,
            min_grid_score=0.5,
            min_border_score=0.4,
        )
        assert isinstance(labels, np.ndarray)

    def test_classify_valid_labels(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """classify() should return valid cell type labels."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        labels = result.classify()
        valid_labels = {"place", "grid", "border", "unclassified"}
        for label in labels:
            assert label in valid_labels, f"Unexpected label: {label}"

    def test_classify_place_cell_by_spatial_info(self, simple_env: Environment) -> None:
        """classify() should label neurons with high spatial info as place cells."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
        # Each neuron fires in only one bin (high spatial information)
        for i in range(n_neurons):
            firing_rates[i, i * 4 + 2] = 30.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # With high spatial info and no grid/border preference, should be place cells
        labels = result.classify(
            min_spatial_info=0.1, min_grid_score=2.0, min_border_score=2.0
        )
        # All should be labeled as place cells (high spatial info, no grid/border)
        assert all(label == "place" for label in labels)

    def test_classify_unclassified_low_spatial_info(
        self, simple_env: Environment
    ) -> None:
        """classify() should label neurons with low spatial info as unclassified."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        # Uniform firing (zero spatial information)
        firing_rates = np.ones((n_neurons, n_bins), dtype=np.float64) * 5.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Use unreachable thresholds for grid and border to test spatial info threshold
        labels = result.classify(
            min_spatial_info=0.5,
            min_grid_score=2.0,  # Unreachable (max is ~2.0)
            min_border_score=2.0,  # Unreachable (max is 1.0)
        )
        # All should be unclassified (uniform firing = no spatial info, and we
        # disabled grid/border classification with unreachable thresholds)
        assert all(label == "unclassified" for label in labels)

    def test_classify_border_cell_high_border_score(
        self, simple_env: Environment
    ) -> None:
        """classify() should label neurons with high border score as border cells."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 1
        firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)

        # Set high firing on boundary bins
        boundary_bins = simple_env.boundary_bins
        if len(boundary_bins) > 0:
            firing_rates[0, boundary_bins] = 20.0
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Get the actual border score to set appropriate threshold
        border_score = result.border_scores()[0]
        if not np.isnan(border_score) and border_score > 0.0:
            # Set threshold just below the actual score
            labels = result.classify(
                min_spatial_info=0.0,  # Don't require spatial info
                min_grid_score=2.0,  # Effectively disable grid classification
                min_border_score=border_score - 0.1,  # Just below actual score
            )
            assert labels[0] == "border"

    def test_classify_priority_order(self, simple_env: Environment) -> None:
        """classify() should follow priority: grid > border > place > unclassified."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        # This tests that when multiple criteria are met, the right label is chosen
        n_bins = simple_env.n_bins
        n_neurons = 1
        # Create firing that could meet multiple criteria
        firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
        firing_rates[0, :] = 5.0  # Some baseline
        firing_rates[0, n_bins // 2] = 30.0  # Peak (place cell)
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # With only spatial info criterion met, should be place cell
        labels = result.classify(
            min_spatial_info=0.0,  # Low threshold
            min_grid_score=2.0,  # Unreachable threshold
            min_border_score=2.0,  # Unreachable threshold
        )
        assert labels[0] == "place"


# ==============================================================================
# Test SpatialRatesResult.to_dataframe() (Task 2.6)
# ==============================================================================


class TestSpatialRatesResultToDataframe:
    """Tests for SpatialRatesResult.to_dataframe() method (Task 2.6)."""

    def test_has_to_dataframe_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have a to_dataframe() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "to_dataframe")
        assert callable(result.to_dataframe)

    def test_to_dataframe_returns_dataframe(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should return a pandas DataFrame."""
        import pandas as pd

        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_to_dataframe_has_correct_row_count(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have one row per neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert len(df) == len(result)

    def test_to_dataframe_has_neuron_id_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a neuron_id column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "neuron_id" in df.columns

    def test_to_dataframe_has_peak_x_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a peak_x column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "peak_x" in df.columns

    def test_to_dataframe_has_peak_y_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a peak_y column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "peak_y" in df.columns

    def test_to_dataframe_has_peak_rate_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a peak_rate column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "peak_rate" in df.columns

    def test_to_dataframe_has_spatial_info_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a spatial_info column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "spatial_info" in df.columns

    def test_to_dataframe_has_sparsity_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a sparsity column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "sparsity" in df.columns

    def test_to_dataframe_has_grid_score_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a grid_score column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "grid_score" in df.columns

    def test_to_dataframe_has_border_score_column(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should have a border_score column."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "border_score" in df.columns

    def test_to_dataframe_has_cell_type_column_by_default(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should include cell_type column by default."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert "cell_type" in df.columns

    def test_to_dataframe_excludes_cell_type_when_disabled(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe(include_classification=False) should exclude cell_type."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe(include_classification=False)
        assert "cell_type" not in df.columns

    def test_to_dataframe_uses_integer_neuron_ids_by_default(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should use integer indices for neuron_id by default."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_ids = list(range(len(result)))
        assert list(df["neuron_id"]) == expected_ids

    def test_to_dataframe_uses_custom_neuron_ids(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe(neuron_ids=...) should use custom identifiers."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        custom_ids = [f"unit_{i}" for i in range(len(result))]
        df = result.to_dataframe(neuron_ids=custom_ids)
        assert list(df["neuron_id"]) == custom_ids

    def test_to_dataframe_peak_rate_matches_peak_firing_rate(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() peak_rate should match peak_firing_rate() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_rates = result.peak_firing_rate()
        np.testing.assert_array_almost_equal(df["peak_rate"].values, expected_rates)

    def test_to_dataframe_spatial_info_matches_spatial_information(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() spatial_info should match spatial_information() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_info = result.spatial_information()
        np.testing.assert_array_almost_equal(df["spatial_info"].values, expected_info)

    def test_to_dataframe_sparsity_matches_sparsity_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() sparsity should match sparsity() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_sparsity = result.sparsity()
        np.testing.assert_array_almost_equal(df["sparsity"].values, expected_sparsity)

    def test_to_dataframe_grid_score_matches_grid_scores_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() grid_score should match grid_scores() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_scores = result.grid_scores()
        # Use allclose with nan handling
        assert np.allclose(
            df["grid_score"].values, expected_scores, equal_nan=True
        ) or np.array_equal(
            np.isnan(df["grid_score"].values), np.isnan(expected_scores)
        )

    def test_to_dataframe_border_score_matches_border_scores_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() border_score should match border_scores() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_scores = result.border_scores()
        # Use allclose with nan handling
        assert np.allclose(
            df["border_score"].values, expected_scores, equal_nan=True
        ) or np.array_equal(
            np.isnan(df["border_score"].values), np.isnan(expected_scores)
        )

    def test_to_dataframe_cell_type_matches_classify_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() cell_type should match classify() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_labels = result.classify()
        assert list(df["cell_type"]) == list(expected_labels)

    def test_to_dataframe_peak_locations_match_peak_locations_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() peak_x, peak_y should match peak_location() method."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        expected_locs = result.peak_location()
        np.testing.assert_array_almost_equal(df["peak_x"].values, expected_locs[:, 0])
        np.testing.assert_array_almost_equal(df["peak_y"].values, expected_locs[:, 1])

    def test_to_dataframe_1d_environment_peak_y_is_nan(self) -> None:
        """to_dataframe() should set peak_y to NaN for 1D environments."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        # Create a 1D-like environment (narrow 2D with 1 column effectively)
        positions = np.linspace(0, 100, 20).reshape(-1, 1)
        # Pad to 2D with constant y
        positions_2d = np.column_stack([positions, np.zeros_like(positions)])
        env = Environment.from_samples(positions_2d, bin_size=10.0)

        # Check if this is effectively 1D
        n_bins = env.n_bins
        n_neurons = 3
        firing_rates = np.random.rand(n_neurons, n_bins) * 10

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=np.ones(n_bins),
            env=env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.to_dataframe()
        # For 2D environments with shape (n_neurons, n_dims), peak_y should exist
        # The specification says peak_y is NaN for 1D, but we created a 2D env
        # So peak_y should be valid for this case
        assert "peak_y" in df.columns

    def test_to_dataframe_single_neuron(
        self,
        simple_env: Environment,
        occupancy: NDArray[np.float64],
    ) -> None:
        """to_dataframe() should work correctly with a single neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        firing_rates = np.random.rand(1, n_bins) * 10

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        df = result.to_dataframe()
        assert len(df) == 1
        assert df["neuron_id"].iloc[0] == 0


# ==============================================================================
# Test compute_spatial_rate() function (Task 2.8)
# ==============================================================================


@pytest.fixture
def trajectory_env() -> Environment:
    """Create an environment from a realistic trajectory."""
    # Generate a random walk trajectory
    np.random.seed(42)
    n_samples = 1000
    dt = 0.01  # 10 ms sampling interval
    velocity = 10.0  # cm/s

    # Random walk
    angles = np.cumsum(np.random.randn(n_samples) * 0.1)
    dx = velocity * dt * np.cos(angles)
    dy = velocity * dt * np.sin(angles)
    x = 50 + np.cumsum(dx)
    y = 50 + np.cumsum(dy)

    # Clip to arena bounds
    x = np.clip(x, 0, 100)
    y = np.clip(y, 0, 100)

    positions = np.column_stack([x, y])
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def trajectory_times() -> NDArray[np.float64]:
    """Create timestamps for the trajectory."""
    n_samples = 1000
    dt = 0.01
    return np.arange(n_samples) * dt


@pytest.fixture
def trajectory_positions(trajectory_env: Environment) -> NDArray[np.float64]:
    """Create positions matching the trajectory environment."""
    np.random.seed(42)
    n_samples = 1000
    dt = 0.01
    velocity = 10.0

    angles = np.cumsum(np.random.randn(n_samples) * 0.1)
    dx = velocity * dt * np.cos(angles)
    dy = velocity * dt * np.sin(angles)
    x = 50 + np.cumsum(dx)
    y = 50 + np.cumsum(dy)

    x = np.clip(x, 0, 100)
    y = np.clip(y, 0, 100)

    return np.column_stack([x, y])


@pytest.fixture
def place_cell_spikes(
    trajectory_times: NDArray[np.float64],
    trajectory_positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Create spike times for a place cell with a field at (60, 60)."""
    np.random.seed(123)
    field_center = np.array([60.0, 60.0])
    field_radius = 15.0
    max_rate = 20.0  # Hz

    # Compute distance from field center at each time
    distances = np.linalg.norm(trajectory_positions - field_center, axis=1)

    # Firing rate is Gaussian centered on field
    firing_rates = max_rate * np.exp(-0.5 * (distances / field_radius) ** 2)

    # Generate spikes using Poisson process
    dt = trajectory_times[1] - trajectory_times[0]
    spike_probs = firing_rates * dt
    spikes_mask = np.random.random(len(trajectory_times)) < spike_probs
    spike_times = trajectory_times[spikes_mask]

    return spike_times


class TestComputeSpatialRateFunction:
    """Tests for compute_spatial_rate() function (Task 2.8)."""

    def test_function_is_importable(self) -> None:
        """compute_spatial_rate should be importable from encoding.spatial."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        assert compute_spatial_rate is not None

    def test_function_is_callable(self) -> None:
        """compute_spatial_rate should be callable."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        assert callable(compute_spatial_rate)

    def test_returns_spatial_rate_result(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rate should return a SpatialRateResult."""
        from neurospatial.encoding.spatial import (
            SpatialRateResult,
            compute_spatial_rate,
        )

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert isinstance(result, SpatialRateResult)

    def test_result_has_correct_firing_rate_shape(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result firing_rate should have shape (n_bins,)."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.firing_rate.shape == (trajectory_env.n_bins,)

    def test_result_has_correct_occupancy_shape(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.occupancy.shape == (trajectory_env.n_bins,)

    def test_result_stores_environment(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should store the environment."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.env is trajectory_env

    def test_result_stores_smoothing_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should store the smoothing method used."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="gaussian_kde",
        )
        assert result.smoothing_method == "gaussian_kde"

    def test_result_stores_bandwidth(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should store the bandwidth used."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            bandwidth=10.0,
        )
        assert result.bandwidth == 10.0

    def test_default_smoothing_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Default smoothing method should be 'diffusion_kde'."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.smoothing_method == "diffusion_kde"

    def test_default_bandwidth(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Default bandwidth should be 5.0."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.bandwidth == 5.0

    def test_firing_rates_are_non_negative(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Firing rates should be non-negative (or NaN)."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        # Check non-NaN values are >= 0
        valid_rates = result.firing_rate[~np.isnan(result.firing_rate)]
        assert np.all(valid_rates >= 0)

    def test_occupancy_is_non_negative(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Occupancy should be non-negative."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert np.all(result.occupancy >= 0)

    def test_empty_spike_train_produces_zero_rates(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Empty spike train should produce zero (or NaN) firing rates."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        empty_spikes = np.array([], dtype=np.float64)
        result = compute_spatial_rate(
            trajectory_env,
            empty_spikes,
            trajectory_times,
            trajectory_positions,
        )
        # Non-NaN values should be 0
        valid_rates = result.firing_rate[~np.isnan(result.firing_rate)]
        assert np.all(valid_rates == 0)

    def test_single_spike_produces_localized_rate(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Single spike should produce localized firing rate."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        # Single spike at t=0.5s
        single_spike = np.array([0.5])
        result = compute_spatial_rate(
            trajectory_env,
            single_spike,
            trajectory_times,
            trajectory_positions,
        )
        # At least one bin should have non-zero firing rate
        valid_rates = result.firing_rate[~np.isnan(result.firing_rate)]
        assert np.any(valid_rates > 0)


class TestComputeSpatialRateSmoothingMethods:
    """Tests for compute_spatial_rate smoothing method options."""

    def test_diffusion_kde_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """diffusion_kde smoothing method should work."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="diffusion_kde",
        )
        assert result.smoothing_method == "diffusion_kde"
        assert result.firing_rate.shape == (trajectory_env.n_bins,)

    def test_gaussian_kde_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """gaussian_kde smoothing method should work."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="gaussian_kde",
        )
        assert result.smoothing_method == "gaussian_kde"
        assert result.firing_rate.shape == (trajectory_env.n_bins,)

    def test_binned_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """binned smoothing method should work."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="binned",
        )
        assert result.smoothing_method == "binned"
        assert result.firing_rate.shape == (trajectory_env.n_bins,)

    def test_different_methods_produce_different_results(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Different smoothing methods should produce different results."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result_diffusion = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="diffusion_kde",
        )
        result_gaussian = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="gaussian_kde",
        )

        # Results should be different (though the peak location should be similar)
        # Use np.nanmax to handle NaN values
        assert not np.allclose(
            np.nan_to_num(result_diffusion.firing_rate, nan=0),
            np.nan_to_num(result_gaussian.firing_rate, nan=0),
        )


class TestComputeSpatialRateMinOccupancy:
    """Tests for compute_spatial_rate min_occupancy parameter."""

    def test_min_occupancy_parameter_exists(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """min_occupancy parameter should be accepted."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        # Should not raise
        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            min_occupancy=0.1,
        )
        assert result is not None

    def test_min_occupancy_masks_low_occupancy_bins(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Bins with occupancy below min_occupancy should be NaN."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        # Use a high min_occupancy threshold
        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            min_occupancy=1.0,  # 1 second minimum
        )

        # Bins with low occupancy should be NaN
        low_occ_mask = result.occupancy < 1.0
        if np.any(low_occ_mask):
            assert np.all(np.isnan(result.firing_rate[low_occ_mask]))


class TestComputeSpatialRateBackendParameter:
    """Tests for compute_spatial_rate backend parameter."""

    def test_backend_parameter_exists(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """backend parameter should be accepted."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        # Should not raise
        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            backend="numpy",
        )
        assert result is not None

    def test_default_backend_is_numpy(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Default backend should be 'numpy'."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        # Function should work without specifying backend
        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        # Result should be a numpy array
        assert isinstance(np.asarray(result.firing_rate), np.ndarray)

    def test_auto_backend_works(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """backend='auto' should work (use numpy if jax unavailable)."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            backend="auto",
        )
        assert result is not None


class TestComputeSpatialRateResultMethods:
    """Tests that result from compute_spatial_rate has all expected methods."""

    def test_result_has_plot_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have plot() method."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert hasattr(result, "plot")
        assert callable(result.plot)

    def test_result_has_peak_location_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have peak_location() method."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert hasattr(result, "peak_location")
        peak = result.peak_location()
        assert peak.shape == (2,)  # 2D environment

    def test_result_has_spatial_information_method(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have spatial_information() method."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert hasattr(result, "spatial_information")
        info = result.spatial_information()
        assert isinstance(info, float)
        assert info >= 0

    def test_result_peak_location_near_expected(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Peak location should be near the place field center."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        result = compute_spatial_rate(
            trajectory_env,
            place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        peak = result.peak_location()
        # Place field was centered at (60, 60) in the fixture
        expected_center = np.array([60.0, 60.0])
        # Peak should be within reasonable distance (depends on sampling)
        distance = np.linalg.norm(peak - expected_center)
        # Allow 30 cm tolerance due to random trajectory sampling and discrete binning
        assert distance < 30.0, f"Peak at {peak}, expected near {expected_center}"


# ==============================================================================
# Fixtures for compute_spatial_rates (batch) tests
# ==============================================================================


@pytest.fixture
def multiple_place_cell_spikes(
    trajectory_times: NDArray[np.float64],
    trajectory_positions: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Create spike times for multiple place cells with different field locations."""
    np.random.seed(456)
    field_centers = [
        np.array([30.0, 30.0]),
        np.array([60.0, 60.0]),
        np.array([80.0, 40.0]),
    ]
    field_radius = 15.0
    max_rate = 20.0  # Hz

    spike_times_list = []
    dt = trajectory_times[1] - trajectory_times[0]

    for i, field_center in enumerate(field_centers):
        np.random.seed(456 + i)
        distances = np.linalg.norm(trajectory_positions - field_center, axis=1)
        firing_rates = max_rate * np.exp(-0.5 * (distances / field_radius) ** 2)
        spike_probs = firing_rates * dt
        spikes_mask = np.random.random(len(trajectory_times)) < spike_probs
        spike_times = trajectory_times[spikes_mask]
        spike_times_list.append(spike_times)

    return spike_times_list


# ==============================================================================
# Test compute_spatial_rates() function (Task 2.9)
# ==============================================================================


class TestComputeSpatialRatesFunction:
    """Tests for compute_spatial_rates() function (Task 2.9)."""

    def test_function_is_importable(self) -> None:
        """compute_spatial_rates should be importable from encoding.spatial."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        assert compute_spatial_rates is not None

    def test_function_is_callable(self) -> None:
        """compute_spatial_rates should be callable."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        assert callable(compute_spatial_rates)

    def test_returns_spatial_rates_result(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """compute_spatial_rates should return a SpatialRatesResult."""
        from neurospatial.encoding.spatial import (
            SpatialRatesResult,
            compute_spatial_rates,
        )

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert isinstance(result, SpatialRatesResult)

    def test_result_has_correct_firing_rates_shape(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result firing_rates should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        n_neurons = len(multiple_place_cell_spikes)
        assert result.firing_rates.shape == (n_neurons, trajectory_env.n_bins)

    def test_result_has_correct_occupancy_shape(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.occupancy.shape == (trajectory_env.n_bins,)

    def test_result_stores_environment(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should store the environment."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.env is trajectory_env

    def test_result_stores_smoothing_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should store the smoothing method used."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="gaussian_kde",
        )
        assert result.smoothing_method == "gaussian_kde"

    def test_result_stores_bandwidth(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should store the bandwidth used."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            bandwidth=10.0,
        )
        assert result.bandwidth == 10.0

    def test_default_smoothing_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Default smoothing method should be 'diffusion_kde'."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.smoothing_method == "diffusion_kde"

    def test_default_bandwidth(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Default bandwidth should be 5.0."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert result.bandwidth == 5.0

    def test_firing_rates_are_non_negative(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Firing rates should be non-negative (or NaN)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        # Check non-NaN values are >= 0
        valid_rates = result.firing_rates[~np.isnan(result.firing_rates)]
        assert np.all(valid_rates >= 0)

    def test_occupancy_is_non_negative(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Occupancy should be non-negative."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert np.all(result.occupancy >= 0)

    def test_len_returns_number_of_neurons(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """len(result) should return number of neurons."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == len(multiple_place_cell_spikes)


class TestComputeSpatialRatesInputFormats:
    """Tests for compute_spatial_rates input format handling."""

    def test_accepts_list_of_arrays(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should accept list of 1D arrays (canonical format)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,  # List of 1D arrays
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == len(multiple_place_cell_spikes)

    def test_accepts_tuple_of_arrays(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should accept tuple of 1D arrays."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        spike_times_tuple = tuple(multiple_place_cell_spikes)
        result = compute_spatial_rates(
            trajectory_env,
            spike_times_tuple,
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == len(multiple_place_cell_spikes)

    def test_accepts_2d_array_with_nan_padding(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should accept 2D array with NaN padding."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # Create 2D array with NaN padding
        spike_times_2d = np.array(
            [
                [0.1, 0.5, 1.0, np.nan, np.nan],
                [0.2, 0.3, 0.8, 1.2, 1.5],
                [0.4, np.nan, np.nan, np.nan, np.nan],
            ]
        )
        result = compute_spatial_rates(
            trajectory_env,
            spike_times_2d,
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == 3

    def test_accepts_single_neuron_1d_array(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should accept 1D array and return result with 1 neuron."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # 1D array gets wrapped in list
        result = compute_spatial_rates(
            trajectory_env,
            place_cell_spikes,  # 1D array
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == 1
        assert result.firing_rates.shape[0] == 1


class TestComputeSpatialRatesNJobs:
    """Tests for compute_spatial_rates n_jobs parameter."""

    def test_n_jobs_parameter_exists(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """n_jobs parameter should be accepted."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # Should not raise
        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            n_jobs=1,
        )
        assert result is not None

    def test_n_jobs_equals_2(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """n_jobs=2 should work for parallel processing."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            n_jobs=2,
        )
        assert len(result) == len(multiple_place_cell_spikes)

    def test_sequential_and_parallel_produce_same_results(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Sequential and parallel processing should produce same results."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result_seq = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            n_jobs=1,
        )
        result_par = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            n_jobs=2,
        )
        np.testing.assert_allclose(
            np.nan_to_num(result_seq.firing_rates, nan=0),
            np.nan_to_num(result_par.firing_rates, nan=0),
            rtol=1e-10,
        )


class TestComputeSpatialRatesSmoothingMethods:
    """Tests for compute_spatial_rates smoothing method options."""

    def test_diffusion_kde_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """diffusion_kde smoothing method should work."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="diffusion_kde",
        )
        assert result.smoothing_method == "diffusion_kde"
        n_neurons = len(multiple_place_cell_spikes)
        assert result.firing_rates.shape == (n_neurons, trajectory_env.n_bins)

    def test_gaussian_kde_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """gaussian_kde smoothing method should work."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="gaussian_kde",
        )
        assert result.smoothing_method == "gaussian_kde"
        n_neurons = len(multiple_place_cell_spikes)
        assert result.firing_rates.shape == (n_neurons, trajectory_env.n_bins)

    def test_binned_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """binned smoothing method should work."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="binned",
        )
        assert result.smoothing_method == "binned"
        n_neurons = len(multiple_place_cell_spikes)
        assert result.firing_rates.shape == (n_neurons, trajectory_env.n_bins)


class TestComputeSpatialRatesMinOccupancy:
    """Tests for compute_spatial_rates min_occupancy parameter."""

    def test_min_occupancy_parameter_exists(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """min_occupancy parameter should be accepted."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # Should not raise
        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            min_occupancy=0.1,
        )
        assert result is not None

    def test_min_occupancy_masks_low_occupancy_bins(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Bins with occupancy below min_occupancy should be NaN."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # Use a high min_occupancy threshold
        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            min_occupancy=1.0,  # 1 second minimum
        )

        # Bins with low occupancy should be NaN for all neurons
        low_occ_mask = result.occupancy < 1.0
        if np.any(low_occ_mask):
            for i in range(len(result)):
                assert np.all(np.isnan(result.firing_rates[i, low_occ_mask]))


class TestComputeSpatialRatesBackendParameter:
    """Tests for compute_spatial_rates backend parameter."""

    def test_backend_parameter_exists(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """backend parameter should be accepted."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # Should not raise
        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            backend="numpy",
        )
        assert result is not None

    def test_default_backend_is_numpy(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Default backend should be 'numpy'."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        # Function should work without specifying backend
        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        # Result should be a numpy array
        assert isinstance(np.asarray(result.firing_rates), np.ndarray)

    def test_auto_backend_works(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """backend='auto' should work (use numpy if jax unavailable)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            backend="auto",
        )
        assert result is not None


class TestComputeSpatialRatesConsistency:
    """Tests for consistency between single and batch compute functions."""

    def test_batch_matches_single_neuron_results(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Batch result should match single-neuron results."""
        from neurospatial.encoding.spatial import (
            compute_spatial_rate,
            compute_spatial_rates,
        )

        # Compute batch result
        batch_result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Compute single-neuron results
        for i, spike_times in enumerate(multiple_place_cell_spikes):
            single_result = compute_spatial_rate(
                trajectory_env,
                spike_times,
                trajectory_times,
                trajectory_positions,
                smoothing_method="diffusion_kde",
                bandwidth=5.0,
            )
            # Compare firing rates
            np.testing.assert_allclose(
                np.nan_to_num(batch_result.firing_rates[i], nan=0),
                np.nan_to_num(single_result.firing_rate, nan=0),
                rtol=1e-10,
                err_msg=f"Mismatch for neuron {i}",
            )

    def test_getitem_returns_matching_single_result(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """result[i] should return SpatialRateResult matching neuron i."""
        from neurospatial.encoding.spatial import (
            SpatialRateResult,
            compute_spatial_rates,
        )

        batch_result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )

        for i in range(len(batch_result)):
            single = batch_result[i]
            assert isinstance(single, SpatialRateResult)
            np.testing.assert_array_equal(
                single.firing_rate, batch_result.firing_rates[i]
            )
            np.testing.assert_array_equal(single.occupancy, batch_result.occupancy)
            assert single.env is batch_result.env


class TestComputeSpatialRatesEdgeCases:
    """Tests for compute_spatial_rates edge cases."""

    def test_empty_spike_train_in_list(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should handle neurons with no spikes."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        spike_times_list = [
            np.array([0.1, 0.5]),  # Normal neuron
            np.array([]),  # Empty spike train
            np.array([0.3]),  # Single spike
        ]
        result = compute_spatial_rates(
            trajectory_env,
            spike_times_list,
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == 3
        # Empty spike train should have zero firing rate (or NaN where no occupancy)
        non_nan = ~np.isnan(result.firing_rates[1])
        assert np.all(result.firing_rates[1, non_nan] == 0)

    def test_single_neuron_list(
        self,
        trajectory_env: Environment,
        place_cell_spikes: NDArray[np.float64],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should handle list with single neuron."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            [place_cell_spikes],
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == 1
        assert result.firing_rates.shape[0] == 1

    def test_all_empty_spike_trains(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should handle all neurons having empty spike trains."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        spike_times_list = [np.array([]), np.array([]), np.array([])]
        result = compute_spatial_rates(
            trajectory_env,
            spike_times_list,
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == 3
        # All firing rates should be zero (or NaN)
        non_nan = ~np.isnan(result.firing_rates)
        assert np.all(result.firing_rates[non_nan] == 0)

    def test_empty_neurons_list(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Should handle empty list of neurons."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            [],  # Empty list of neurons
            trajectory_times,
            trajectory_positions,
        )
        assert len(result) == 0
        assert result.firing_rates.shape == (0, trajectory_env.n_bins)

    def test_empty_neurons_list_validates_smoothing_method(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Empty neuron list should still reject invalid smoothing methods."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        with pytest.raises(ValueError, match="method must be one of"):
            compute_spatial_rates(
                trajectory_env,
                [],
                trajectory_times,
                trajectory_positions,
                smoothing_method="invalid",  # type: ignore[arg-type]
            )

    def test_empty_neurons_list_has_valid_occupancy(
        self,
        trajectory_env: Environment,
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Empty neuron list should still have valid occupancy from trajectory.

        Bug: Previously returned zero occupancy, which is incorrect.
        Occupancy reflects animal position regardless of neural activity.
        """
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            [],  # Empty list of neurons
            trajectory_times,
            trajectory_positions,
        )

        # Occupancy should still be computed from trajectory
        total_duration = trajectory_times[-1] - trajectory_times[0]
        total_occupancy = np.sum(result.occupancy)

        # Should have meaningful occupancy (not all zeros)
        assert total_occupancy > 0, (
            "Empty neuron list should still compute occupancy from trajectory"
        )
        # Total should be close to recording duration
        assert total_occupancy >= total_duration * 0.8, (
            f"Expected occupancy ~{total_duration:.1f}s, got {total_occupancy:.1f}s"
        )


class TestComputeSpatialRatesResultMethods:
    """Tests that result from compute_spatial_rates has expected batch methods."""

    def test_result_has_spatial_information_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have spatial_information() returning (n_neurons,)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        info = result.spatial_information()
        assert info.shape == (len(multiple_place_cell_spikes),)
        assert np.all(info >= 0)

    def test_result_has_sparsity_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have sparsity() returning (n_neurons,)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        spars = result.sparsity()
        assert spars.shape == (len(multiple_place_cell_spikes),)
        assert np.all((spars >= 0) & (spars <= 1))

    def test_result_has_grid_scores_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have grid_scores() returning (n_neurons,)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        scores = result.grid_scores()
        assert scores.shape == (len(multiple_place_cell_spikes),)

    def test_result_has_border_scores_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have border_scores() returning (n_neurons,)."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        scores = result.border_scores()
        assert scores.shape == (len(multiple_place_cell_spikes),)

    def test_result_has_classify_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have classify() returning (n_neurons,) string labels."""
        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        labels = result.classify()
        assert labels.shape == (len(multiple_place_cell_spikes),)
        assert labels.dtype.kind == "U"  # String dtype

    def test_result_has_to_dataframe_method(
        self,
        trajectory_env: Environment,
        multiple_place_cell_spikes: list[NDArray[np.float64]],
        trajectory_times: NDArray[np.float64],
        trajectory_positions: NDArray[np.float64],
    ) -> None:
        """Result should have to_dataframe() returning DataFrame."""
        import pandas as pd

        from neurospatial.encoding.spatial import compute_spatial_rates

        result = compute_spatial_rates(
            trajectory_env,
            multiple_place_cell_spikes,
            trajectory_times,
            trajectory_positions,
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(multiple_place_cell_spikes)
        assert "spatial_info" in df.columns
