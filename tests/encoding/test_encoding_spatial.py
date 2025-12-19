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
        """SpatialRateResult should have peak_locations() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_locations")
        assert callable(result.peak_locations)

    def test_has_peak_firing_rates_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have peak_firing_rates() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_firing_rates")
        assert callable(result.peak_firing_rates)


class TestSpatialRateResultMixinMethods:
    """Tests for SpatialResultMixin methods on SpatialRateResult."""

    def test_peak_locations_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_locations() should return (n_dims,) for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak = result.peak_locations()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (simple_env.n_dims,)

    def test_peak_locations_finds_maximum(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """peak_locations() should find the bin with maximum firing rate."""
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
        peak = result.peak_locations()
        expected = simple_env.bin_centers[peak_bin]
        np.testing.assert_array_almost_equal(peak, expected)

    def test_peak_firing_rates_returns_scalar(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rates() should return a scalar for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rate = result.peak_firing_rates()
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
        """SpatialRatesResult should have peak_locations() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_locations")
        assert callable(result.peak_locations)

    def test_has_peak_firing_rates_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have peak_firing_rates() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_firing_rates")
        assert callable(result.peak_firing_rates)


class TestSpatialRatesResultMixinMethods:
    """Tests for SpatialResultMixin methods on SpatialRatesResult."""

    def test_peak_locations_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_locations() should return (n_neurons, n_dims) for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peaks = result.peak_locations()
        n_neurons = firing_rates_batch.shape[0]
        assert isinstance(peaks, np.ndarray)
        assert peaks.shape == (n_neurons, simple_env.n_dims)

    def test_peak_locations_finds_maximums(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """peak_locations() should find the bin with maximum firing rate for each neuron."""
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
        peaks = result.peak_locations()
        for i, peak_bin in enumerate(peak_bins):
            expected = simple_env.bin_centers[peak_bin]
            np.testing.assert_array_almost_equal(peaks[i], expected)

    def test_peak_firing_rates_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rates() should return (n_neurons,) for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rates = result.peak_firing_rates()
        n_neurons = firing_rates_batch.shape[0]
        assert isinstance(peak_rates, np.ndarray)
        assert peak_rates.shape == (n_neurons,)

    def test_peak_firing_rates_correct_values(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rates() should return the maximum firing rate for each neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rates = result.peak_firing_rates()
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

    def test_peak_location_returns_same_as_peak_locations(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_location() should return the same result as peak_locations()."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_location = result.peak_location()
        peak_locations = result.peak_locations()
        np.testing.assert_array_equal(peak_location, peak_locations)

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
        autocorr = spatial_autocorrelation(
            regular_grid_env, grid_cell_firing_rate, method="fft"
        )
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
        autocorr = spatial_autocorrelation(
            regular_grid_env, grid_cell_firing_rate, method="fft"
        )
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

    def test_border_score_accepts_distance_metric_parameter(
        self,
        simple_env: Environment,
        border_cell_firing_rate: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """border_score() should accept a distance_metric parameter."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=border_cell_firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Should not raise with either metric
        score_geodesic = result.border_score(distance_metric="geodesic")
        score_euclidean = result.border_score(distance_metric="euclidean")
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
