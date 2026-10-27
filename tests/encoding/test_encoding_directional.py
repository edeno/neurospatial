"""Tests for directional rate computation result classes.

This module tests DirectionalRateResult and DirectionalRatesResult dataclasses
defined in encoding/directional.py for head direction cell analysis.

The result classes need no Environment, only bin_centers.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_array_equal

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def n_bins() -> int:
    """Number of angular bins (60 bins = 6 degree resolution)."""
    return 60


@pytest.fixture
def n_neurons() -> int:
    """Number of neurons for batch tests."""
    return 5


@pytest.fixture
def bin_size() -> float:
    """Bin size in radians (6 degrees)."""
    return np.pi / 30


@pytest.fixture
def bin_centers(n_bins: int) -> np.ndarray:
    """Bin centers in radians [0, 2π)."""
    return np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + np.pi / n_bins


@pytest.fixture
def single_firing_rate(n_bins: int) -> np.ndarray:
    """Firing rate for a single neuron with a clear peak."""
    # Create tuning curve with peak at π/2 (90 degrees)
    preferred_dir = np.pi / 2
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    # Von Mises-like tuning curve
    kappa = 2.0  # Concentration parameter
    firing_rate = 10.0 * np.exp(kappa * (np.cos(angles - preferred_dir) - 1))
    return firing_rate


@pytest.fixture
def single_occupancy(n_bins: int) -> np.ndarray:
    """Occupancy for a single neuron (uniform)."""
    return np.ones(n_bins) * 0.5  # 0.5 seconds per bin


@pytest.fixture
def batch_firing_rates(n_neurons: int, n_bins: int) -> np.ndarray:
    """Firing rates for multiple neurons with different preferred directions."""
    rates = np.zeros((n_neurons, n_bins))
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)

    for i in range(n_neurons):
        # Each neuron has a different preferred direction
        preferred_dir = i * 2 * np.pi / n_neurons
        kappa = 2.0
        rates[i] = 10.0 * np.exp(kappa * (np.cos(angles - preferred_dir) - 1))

    return rates


# ==============================================================================
# DirectionalRateResult Tests
# ==============================================================================


class TestDirectionalRateResultCreation:
    """Test DirectionalRateResult dataclass creation."""

    def test_is_frozen_dataclass(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """DirectionalRateResult is a frozen dataclass (immutable)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.bin_size = 0.2  # type: ignore[misc]


# ==============================================================================
# DirectionalRatesResult Tests
# ==============================================================================


class TestDirectionalRatesResultCreation:
    """Test DirectionalRatesResult dataclass creation."""

    def test_is_frozen_dataclass(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """DirectionalRatesResult is a frozen dataclass (immutable)."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        with pytest.raises(FrozenInstanceError):
            result.bin_size = 0.2  # type: ignore[misc]


class TestDirectionalRatesResultIteration:
    """Test DirectionalRatesResult iteration interface."""

    def test_len(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """len() returns number of units."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        assert len(result) == n_neurons

    def test_getitem_returns_single_result(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """__getitem__ returns DirectionalRateResult for single neuron."""
        from neurospatial.encoding.directional import (
            DirectionalRateResult,
            DirectionalRatesResult,
        )

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        single = result[0]
        assert isinstance(single, DirectionalRateResult)

    def test_getitem_preserves_metadata(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """__getitem__ preserves occupancy, bin_centers, bin_size, bandwidth."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        bandwidth = np.pi / 6

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=bandwidth,
        )

        single = result[0]

        assert_array_equal(np.asarray(single.occupancy), single_occupancy)
        assert_array_equal(np.asarray(single.bin_centers), bin_centers)
        assert single.bin_size == bin_size
        assert single.bandwidth == bandwidth

    def test_getitem_extracts_correct_firing_rate(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """__getitem__ extracts correct neuron's firing rate."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        for i in range(batch_firing_rates.shape[0]):
            single = result[i]
            assert_array_equal(np.asarray(single.firing_rate), batch_firing_rates[i])

    def test_iter(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """__iter__ yields DirectionalRateResult for each neuron."""
        from neurospatial.encoding.directional import (
            DirectionalRateResult,
            DirectionalRatesResult,
        )

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        items = list(result)
        assert len(items) == n_neurons
        for item in items:
            assert isinstance(item, DirectionalRateResult)

    def test_iter_order(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """__iter__ yields neurons in order."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        for i, single in enumerate(result):
            assert_array_equal(np.asarray(single.firing_rate), batch_firing_rates[i])


class TestDirectionalRatesResultEdgeCases:
    """Test edge cases for DirectionalRatesResult."""

    def test_single_neuron(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """DirectionalRatesResult works with single neuron (n_neurons=1)."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        # Reshape to (1, n_bins)
        rates = single_firing_rate.reshape(1, -1)

        result = DirectionalRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        assert len(result) == 1
        single = result[0]
        assert_array_equal(np.asarray(single.firing_rate), single_firing_rate)

    def test_empty_result(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """DirectionalRatesResult works with zero neurons."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        # Empty firing_rates with shape (0, n_bins)
        rates = np.empty((0, n_bins), dtype=np.float64)

        result = DirectionalRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        assert len(result) == 0
        assert list(result) == []


# ==============================================================================
# DirectionalRateResult Convenience Methods Tests
# ==============================================================================


class TestDirectionalRateResultPlot:
    """Test DirectionalRateResult.plot() method."""

    def test_plot_returns_axes(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot() returns matplotlib Axes object."""
        pytest.importorskip("matplotlib")
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        ax = result.plot()
        assert ax is not None
        # Check that it's an Axes-like object
        assert hasattr(ax, "plot")

    def test_plot_polar_default(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot() creates polar plot by default."""
        pytest.importorskip("matplotlib")
        from matplotlib.projections.polar import PolarAxes

        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        ax = result.plot(polar=True)
        assert isinstance(ax, PolarAxes)

    def test_plot_cartesian(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot(polar=False) creates Cartesian plot."""
        pytest.importorskip("matplotlib")
        from matplotlib.projections.polar import PolarAxes

        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        ax = result.plot(polar=False)
        # Not a polar plot
        assert not isinstance(ax, PolarAxes)

    def test_plot_with_ax(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot() accepts existing axes."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        _, ax_provided = plt.subplots(subplot_kw={"projection": "polar"})
        ax_returned = result.plot(ax=ax_provided)
        assert ax_returned is ax_provided

    def test_plot_with_kwargs(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot() passes through kwargs."""
        pytest.importorskip("matplotlib")
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Should not raise an error with valid kwargs
        ax = result.plot(color="red", linewidth=2)
        assert ax is not None


class TestDirectionalRateResultPreferredDirection:
    """Test DirectionalRateResult.preferred_direction() method."""

    def test_preferred_direction_in_valid_range(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_direction() returns value in [-π, π]."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dir = result.preferred_direction()
        assert -np.pi <= pref_dir <= np.pi

    def test_preferred_direction_near_expected(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """preferred_direction() is near the tuning curve peak."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Create tuning curve with clear peak at π (180 degrees)
        expected_dir = np.pi
        angles = bin_centers
        kappa = 5.0  # High concentration for clear peak
        firing_rate = 10.0 * np.exp(kappa * (np.cos(angles - expected_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dir = result.preferred_direction()
        # Should be close to expected direction (within ~15 degrees)
        # Note: circular_mean returns [-π, π], so expected_dir (π) should be close
        # to either π or -π (they're the same direction on a circle)
        angular_diff = np.abs(np.angle(np.exp(1j * (pref_dir - expected_dir))))
        assert angular_diff < np.pi / 12  # Within 15 degrees

    def test_preferred_direction_uniform_firing(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """preferred_direction() handles uniform firing rate (returns NaN or value)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate - no clear preferred direction
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # For uniform firing, circular_mean with uniform weights should return
        # a value (though it may not be meaningful)
        pref_dir = result.preferred_direction()
        # Should return a float (either NaN or a value)
        assert isinstance(pref_dir, float)


class TestDirectionalRateResultPreferredDirectionDeg:
    """Test DirectionalRateResult.preferred_direction_deg() method."""

    def test_preferred_direction_deg_conversion(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_direction_deg() equals degrees conversion of preferred_direction()."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dir_rad = result.preferred_direction()
        pref_dir_deg = result.preferred_direction_deg()
        np.testing.assert_allclose(pref_dir_deg, np.degrees(pref_dir_rad))

    def test_preferred_direction_deg_in_valid_range(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_direction_deg() returns value in [-180, 180]."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dir_deg = result.preferred_direction_deg()
        assert -180 <= pref_dir_deg <= 180


class TestDirectionalRateResultPeakFiringRate:
    """Test DirectionalRateResult.peak_firing_rate() method."""

    def test_peak_firing_rate_correct_value(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rate() returns maximum of firing_rate."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peak = result.peak_firing_rate()
        expected = float(np.max(single_firing_rate))
        np.testing.assert_allclose(peak, expected)

    def test_peak_firing_rate_with_nan(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """peak_firing_rate() handles NaN values correctly."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Create firing rate with some NaN values
        firing_rate = np.ones(n_bins) * 5.0
        firing_rate[0] = np.nan
        firing_rate[10] = 15.0  # Peak

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peak = result.peak_firing_rate()
        assert peak == 15.0

    def test_peak_firing_rate_nonnegative(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rate() returns non-negative value."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peak = result.peak_firing_rate()
        assert peak >= 0


# ==============================================================================
# DirectionalRateResult Tuning Metrics Tests
# ==============================================================================


class TestDirectionalRateResultMeanVectorLength:
    """Test DirectionalRateResult.mean_vector_length() method."""

    def test_mean_vector_length_in_valid_range(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """mean_vector_length() returns value in [0, 1]."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvl = result.mean_vector_length()
        assert 0 <= mvl <= 1

    def test_mean_vector_length_tuned_neuron_higher(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """mean_vector_length() is higher for tuned than untuned neurons."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Sharply tuned neuron (kappa=5)
        preferred_dir = np.pi / 2
        tuned_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        tuned_result = DirectionalRateResult(
            firing_rate=tuned_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Uniform firing rate (untuned)
        uniform_rate = np.ones(n_bins) * 5.0

        uniform_result = DirectionalRateResult(
            firing_rate=uniform_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        assert tuned_result.mean_vector_length() > uniform_result.mean_vector_length()

    def test_mean_vector_length_uniform_firing(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """mean_vector_length() is low for uniform firing."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvl = result.mean_vector_length()
        # Uniform firing should have near-zero MVL (weighted by equal rates)
        # With uniform weights (bins evenly spaced), MVL should be near 0
        assert mvl < 0.1

    def test_mean_vector_length_sharply_tuned(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """mean_vector_length() is high for sharply tuned neurons."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Very sharply tuned neuron (kappa=10)
        preferred_dir = 0.0
        firing_rate = 20.0 * np.exp(10.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvl = result.mean_vector_length()
        # Sharply tuned should have high MVL (typically > 0.5)
        assert mvl > 0.5


class TestDirectionalRateResultTuningWidth:
    """Test DirectionalRateResult.tuning_width() method."""

    def test_tuning_width_in_valid_range(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_width() returns value in (0, π]."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        width = result.tuning_width()
        # HWHM should be at most π (half circle)
        # Could be NaN for edge cases, so check if valid
        if not np.isnan(width):
            assert 0 < width <= np.pi

    def test_tuning_width_sharper_tuning_smaller_width(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_width() is smaller for sharper tuning."""
        from neurospatial.encoding.directional import DirectionalRateResult

        preferred_dir = np.pi / 2

        # Sharply tuned (kappa=5)
        sharp_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))
        sharp_result = DirectionalRateResult(
            firing_rate=sharp_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Broadly tuned (kappa=1)
        broad_rate = 10.0 * np.exp(1.0 * (np.cos(bin_centers - preferred_dir) - 1))
        broad_result = DirectionalRateResult(
            firing_rate=broad_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        assert sharp_result.tuning_width() < broad_result.tuning_width()

    def test_tuning_width_uniform_returns_nan(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """tuning_width() returns NaN for uniform firing (no tuning)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing - no tuning curve
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        width = result.tuning_width()
        # Uniform firing has no peak, so HWHM is undefined
        assert np.isnan(width)


class TestDirectionalRateResultTuningWidthDeg:
    """Test DirectionalRateResult.tuning_width_deg() method."""

    def test_tuning_width_deg_conversion(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_width_deg() equals degrees conversion of tuning_width()."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        width_rad = result.tuning_width()
        width_deg = result.tuning_width_deg()
        np.testing.assert_allclose(width_deg, np.degrees(width_rad))

    def test_tuning_width_deg_valid_range(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_width_deg() returns value in (0, 180]."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        width_deg = result.tuning_width_deg()
        # Should be at most 180 degrees
        if not np.isnan(width_deg):
            assert 0 < width_deg <= 180


@pytest.fixture(scope="module")
def biased_heading_session() -> tuple[np.ndarray, np.ndarray]:
    """Ten minutes at 30 Hz with headings concentrated near 0.5 rad.

    Headings follow a von Mises distribution (kappa = 1.5), so the animal
    faces some directions about 20 times longer than others.
    """
    times = np.arange(0.0, 600.0, 1 / 30)
    headings = np.random.default_rng(0).vonmises(0.5, 1.5, times.size)
    return times, headings


def _frame_poisson_spikes(times: np.ndarray, rate: np.ndarray, seed: int) -> np.ndarray:
    """Spike times from per-frame Poisson counts at ``rate`` (Hz) per frame."""
    rng = np.random.default_rng(seed)
    dt = np.diff(times)
    counts = rng.poisson(rate[:-1] * dt)
    starts = np.repeat(times[:-1], counts)
    widths = np.repeat(dt, counts)
    return np.sort(starts + rng.uniform(0.0, 1.0, starts.size) * widths)


class TestTuningPvalue:
    """tuning_pvalue tests one preferred direction given heading occupancy."""

    @staticmethod
    def _result(counts, occupancy):
        from neurospatial.encoding.directional import DirectionalRateResult

        counts = np.asarray(counts, dtype=np.float64)
        occupancy = np.asarray(occupancy, dtype=np.float64)
        n = len(counts)
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(occupancy > 0, counts / occupancy, np.nan)
        return DirectionalRateResult(
            firing_rate=rate,
            occupancy=occupancy,
            bin_centers=centers,
            bin_size=2 * np.pi / n,
            bandwidth=None,
            spike_count_dispersion=1.0,
            spike_counts=counts,
        )

    def test_flat_cell_with_biased_headings_is_calibrated(
        self, biased_heading_session
    ) -> None:
        """A constant-rate cell is not called tuned because of where the animal faced.

        Spike headings of a flat cell follow the heading occupancy, so a test
        of the spike headings alone (count-weighted Rayleigh) rejects in
        every run. Correcting for occupancy keeps the false-positive rate
        near alpha.
        """
        from neurospatial.encoding.directional import compute_directional_rates

        times, headings = biased_heading_session
        rng = np.random.default_rng(1)
        duration = times[-1] - times[0]
        trains = [
            np.sort(rng.uniform(times[0], times[-1], rng.poisson(2.0 * duration)))
            for _ in range(400)
        ]
        result = compute_directional_rates(trains, times, headings)
        pvalues = result.tuning_pvalues()
        assert np.all(np.isfinite(pvalues))
        false_positive_rate = np.mean(pvalues < 0.05)
        assert 0.02 <= false_positive_rate <= 0.09
        # The screen built on it flags no flat cell.
        assert not result.classify().any()

    def test_hd_cell_with_biased_headings_is_detected(
        self, biased_heading_session
    ) -> None:
        from neurospatial.encoding.directional import compute_directional_rate

        times, headings = biased_heading_session
        rate = 0.5 + 8.0 * np.exp(2.0 * (np.cos(headings - 2.0) - 1.0))
        spikes = _frame_poisson_spikes(times, rate, seed=2)
        result = compute_directional_rate(spikes, times, headings)
        assert result.tuning_pvalue() < 1e-10
        assert result.is_head_direction_cell()
        # Occupancy normalization recovers the cell's preference (2.0 rad),
        # not the animal's preferred heading (0.5 rad).
        error = abs(np.angle(np.exp(1j * (result.preferred_direction() - 2.0))))
        assert error < np.radians(6)

    def test_uniform_occupancy_equals_large_sample_rayleigh(self) -> None:
        """With equal time at every heading, T = 2 N R²: the Rayleigh test."""
        from neurospatial.stats.circular import mean_resultant_length, rayleigh_test

        n = 36
        rng = np.random.default_rng(3)
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        counts = rng.poisson(6.0 * np.exp(1.0 * (np.cos(centers - 1.0) - 1.0)))
        result = self._result(counts, np.full(n, 0.4))

        n_spikes = counts.sum()
        assert n_spikes >= 50  # Rayleigh's small-sample correction is off
        r = mean_resultant_length(centers, weights=counts)
        expected = np.exp(-n_spikes * r**2)
        assert result.tuning_pvalue() == pytest.approx(expected, rel=1e-10)
        _, rayleigh_p = rayleigh_test(centers[counts > 0], weights=counts[counts > 0])
        assert result.tuning_pvalue() == pytest.approx(rayleigh_p, rel=1e-10)

    def test_occupancy_scale_does_not_change_pvalue(self) -> None:
        """Doubling every bin's time with the same counts tests the same tuning."""
        n = 24
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        counts = np.round(10.0 * np.exp(2.0 * (np.cos(centers) - 1.0))) + 1.0
        occupancy = 0.2 + 0.6 * np.abs(np.sin(centers))
        p = self._result(counts, occupancy).tuning_pvalue()
        assert np.isfinite(p)
        assert self._result(counts, 2.0 * occupancy).tuning_pvalue() == pytest.approx(
            p, rel=1e-10
        )

    def test_spikes_follow_occupancy_is_not_tuned(self) -> None:
        """Counts proportional to occupancy give the null statistic exactly."""
        n = 24
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        occupancy = np.exp(1.5 * np.cos(centers - 0.5))
        counts = 10.0 * occupancy
        assert self._result(counts, occupancy).tuning_pvalue() == pytest.approx(1.0)

    def test_spikes_in_unvisited_bins_are_ignored(self) -> None:
        """Spikes assigned to a heading the animal never faced cannot be tested."""
        occupancy = np.array([2.0, 0.0, 0.0, 0.0, 2.0, 2.0])
        counts = np.array([0.0, 50.0, 50.0, 0.0, 0.0, 0.0])
        assert np.isnan(self._result(counts, occupancy).tuning_pvalue())

    def test_two_visited_directions_cannot_be_tested(self) -> None:
        """Cosine and sine terms need at least three visited directions."""
        occupancy = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        counts = np.array([30.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        assert np.isnan(self._result(counts, occupancy).tuning_pvalue())

    def test_spikes_concentrated_in_two_bins_are_tuned(self) -> None:
        n = 36
        counts = np.zeros(n)
        counts[10], counts[11] = 60.0, 40.0
        assert self._result(counts, np.full(n, 0.5)).tuning_pvalue() < 1e-10

    def test_counts_rebuilt_from_unsmoothed_rate(self) -> None:
        """Without spike_counts, counts are firing_rate * occupancy."""
        from neurospatial.encoding.directional import DirectionalRateResult

        n = 24
        occupancy = np.linspace(0.2, 1.0, n)
        counts = np.arange(n, dtype=float) % 7
        with_counts = self._result(counts, occupancy)
        without = DirectionalRateResult(
            firing_rate=with_counts.firing_rate,
            occupancy=occupancy,
            bin_centers=with_counts.bin_centers,
            bin_size=with_counts.bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )
        assert without.tuning_pvalue() == pytest.approx(
            with_counts.tuning_pvalue(), rel=1e-12
        )

    def test_population_matches_single_units(self, biased_heading_session) -> None:
        from neurospatial.encoding.directional import (
            compute_directional_rate,
            compute_directional_rates,
        )

        times, headings = biased_heading_session
        trains = [
            _frame_poisson_spikes(
                times, 1.0 + 4.0 * np.exp(np.cos(headings - pref) - 1.0), seed=10 + k
            )
            for k, pref in enumerate((-2.0, 0.0, 2.5))
        ]
        result = compute_directional_rates(trains, times, headings)
        np.testing.assert_allclose(
            result.tuning_pvalues(), [unit.tuning_pvalue() for unit in result]
        )
        table = result.summary_table()
        np.testing.assert_allclose(table["tuning_pvalue"], result.tuning_pvalues())
        np.testing.assert_allclose(
            table["spike_count_dispersion"], result.spike_count_dispersion
        )
        single = compute_directional_rate(trains[1], times, headings)
        assert single.spike_count_dispersion == pytest.approx(
            result[1].spike_count_dispersion, rel=1e-12
        )
        assert single.tuning_pvalue() == pytest.approx(
            result.tuning_pvalues()[1], rel=1e-12
        )

    def test_fewer_than_fifty_spikes_is_nan(self) -> None:
        """Below 50 spikes the χ² reference distribution is not trusted."""
        n = 12
        counts = np.zeros(n)
        counts[3] = 49.0
        assert np.isnan(self._result(counts, np.full(n, 1.0)).tuning_pvalue())
        counts[3] = 50.0
        assert self._result(counts, np.full(n, 1.0)).tuning_pvalue() < 1e-10


# Heading diffusion constant (rad² / s) of the wandering-heading session.
_HEADING_DIFFUSION = 2.0


@pytest.fixture(scope="module")
def wandering_heading_session() -> tuple[np.ndarray, np.ndarray]:
    """Ten minutes at 30 Hz with a heading that diffuses around the circle.

    For a Brownian heading with diffusion constant D, the heading unit
    vector has autocorrelation ``exp(-D |τ| / 2)``, so headings stay
    correlated for about a second, as in a freely moving animal.
    """
    times = np.arange(0.0, 600.0, 1 / 30)
    steps = np.random.default_rng(20).normal(
        0.0, np.sqrt(_HEADING_DIFFUSION / 30), times.size
    )
    return times, np.angle(np.exp(1j * np.cumsum(steps)))


def _bursting_spikes(
    times: np.ndarray, rate: np.ndarray, mean_burst: float, seed: int
) -> np.ndarray:
    """Bursts at ``rate / mean_burst``, each a geometric number of spikes 4 ms apart."""
    events = _frame_poisson_spikes(times, rate / mean_burst, seed=seed)
    sizes = np.random.default_rng(seed + 1).geometric(1.0 / mean_burst, events.size)
    spikes = np.repeat(events, sizes) + 0.004 * (
        np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    )
    return np.sort(spikes[spikes < times[-1]])


class TestTuningPvalueDispersion:
    """The tuning test stays calibrated for bursting and drifting cells."""

    def test_heading_correlation_time_matches_diffusion(
        self, wandering_heading_session
    ) -> None:
        """∫ exp(-D|τ|/2) dτ = 4 / D for a diffusing heading."""
        from neurospatial.encoding._directional_counts import (
            prepare_directional_binning,
        )
        from neurospatial.encoding._directional_dispersion import (
            heading_correlation_time,
        )

        times, headings = wandering_heading_session
        prepared, centers = prepare_directional_binning(times, headings, None)
        window = heading_correlation_time(prepared, centers)
        assert window == pytest.approx(4.0 / _HEADING_DIFFUSION, rel=0.25)

    def test_independent_headings_give_one_sample_window(
        self, biased_heading_session
    ) -> None:
        from neurospatial.encoding._directional_counts import (
            prepare_directional_binning,
        )
        from neurospatial.encoding._directional_dispersion import (
            heading_correlation_time,
        )

        times, headings = biased_heading_session
        prepared, centers = prepare_directional_binning(times, headings, None)
        window = heading_correlation_time(prepared, centers)
        assert window == pytest.approx(1 / 30, rel=0.1)

    @pytest.mark.parametrize("mean_burst", [1.0, 2.5, 4.0])
    def test_dispersion_matches_burst_size(
        self, wandering_heading_session, mean_burst
    ) -> None:
        """Geometric bursts of mean b have count variance / mean = 2b - 1."""
        from neurospatial.encoding import compute_directional_rates

        times, headings = wandering_heading_session
        rate = np.full(times.size, 3.0)
        trains = [
            _bursting_spikes(times, rate, mean_burst, seed=100 + 2 * k)
            for k in range(20)
        ]
        result = compute_directional_rates(trains, times, headings)
        median = np.median(result.spike_count_dispersion)
        assert median == pytest.approx(2 * mean_burst - 1, rel=0.15)
        assert result.analysis_settings["dispersion_window"] == pytest.approx(
            4.0 / _HEADING_DIFFUSION, rel=0.25
        )

    def test_bursting_flat_cells_are_calibrated(
        self, wandering_heading_session
    ) -> None:
        """Untuned cells firing bursts of 2.5 spikes reject near alpha.

        Treating burst spikes as independent rejects about half of these
        cells at alpha = 0.05.
        """
        from neurospatial.encoding import compute_directional_rates

        times, headings = wandering_heading_session
        rate = np.full(times.size, 3.0)
        trains = [
            _bursting_spikes(times, rate, 2.5, seed=1000 + 2 * k) for k in range(200)
        ]
        pvalues = compute_directional_rates(trains, times, headings).tuning_pvalues()
        assert np.all(np.isfinite(pvalues))
        assert 0.01 <= np.mean(pvalues < 0.05) <= 0.10

    def test_slow_rate_drift_is_calibrated(self, wandering_heading_session) -> None:
        """A rate that drifts over 30 s is not mistaken for heading tuning."""
        from neurospatial.encoding import compute_directional_rates

        times, headings = wandering_heading_session
        dt = times[1] - times[0]
        rng = np.random.default_rng(7)
        decay = np.exp(-dt / 30.0)
        trains = []
        for k in range(100):
            log_rate = np.empty(times.size)
            log_rate[0] = rng.normal(0.0, 0.5)
            noise = rng.normal(0.0, 0.5 * np.sqrt(1 - decay**2), times.size)
            for i in range(1, times.size):
                log_rate[i] = decay * log_rate[i - 1] + noise[i]
            trains.append(
                _frame_poisson_spikes(times, 3.0 * np.exp(log_rate), seed=5000 + k)
            )
        pvalues = compute_directional_rates(trains, times, headings).tuning_pvalues()
        assert np.mean(pvalues < 0.05) <= 0.10

    def test_bursting_hd_cells_are_detected(self, wandering_heading_session) -> None:
        """Broad tuning (κ = 0.5) is still detected in bursting cells."""
        from neurospatial.encoding import compute_directional_rates

        times, headings = wandering_heading_session
        trains = [
            _bursting_spikes(
                times,
                1.0 + 5.0 * np.exp(0.5 * (np.cos(headings - pref) - 1.0)),
                2.5,
                seed=3000 + 2 * k,
            )
            for k, pref in enumerate(np.linspace(-3.0, 3.0, 20))
        ]
        result = compute_directional_rates(trains, times, headings)
        assert np.all(result.spike_count_dispersion > 3.0)
        assert np.all(result.tuning_pvalues() < 0.05)

    def test_one_burst_is_not_a_head_direction_cell(
        self, wandering_heading_session
    ) -> None:
        """Three spikes of one burst have MVL 1 but cannot show tuning."""
        from neurospatial.encoding import compute_directional_rate

        times, headings = wandering_heading_session
        spikes = 100.0 + np.array([0.0, 0.004, 0.008])
        result = compute_directional_rate(spikes, times, headings)
        assert result.mean_vector_length() == pytest.approx(1.0)
        assert np.isnan(result.tuning_pvalue())
        assert not result.is_head_direction_cell()

    def test_statistic_is_divided_by_dispersion(self) -> None:
        """p = (1 + T / (φ dof))^(-dof/2): the quasi-Poisson F test."""
        from dataclasses import replace

        n = 36
        rng = np.random.default_rng(3)
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        counts = rng.poisson(6.0 * np.exp(0.3 * (np.cos(centers - 1.0) - 1.0)))
        poisson = TestTuningPvalue._result(counts, np.full(n, 0.4))
        statistic = -2.0 * np.log(poisson.tuning_pvalue())

        corrected = replace(poisson, spike_count_dispersion=4.0, dispersion_dof=100)
        expected = (1.0 + statistic / (4.0 * 100)) ** (-50)
        assert corrected.tuning_pvalue() == pytest.approx(expected, rel=1e-10)
        # Underdispersion is not credited: φ is floored at 1.
        under = replace(poisson, spike_count_dispersion=0.5, dispersion_dof=100)
        one = replace(poisson, spike_count_dispersion=1.0, dispersion_dof=100)
        assert under.tuning_pvalue() == pytest.approx(one.tuning_pvalue())
        # An unestimable dispersion leaves the test undefined.
        missing = replace(poisson, spike_count_dispersion=np.nan, dispersion_dof=100)
        assert np.isnan(missing.tuning_pvalue())

    def test_result_without_dispersion_is_untestable(self) -> None:
        """A hand-built result does not fall back to the Poisson test.

        Without a dispersion the Poisson test rejects most bursting cells, so
        the p-value is NaN and the screen flags nothing until a dispersion
        (1.0 for Poisson spiking) is given.
        """
        from dataclasses import replace

        n = 36
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        counts = np.round(20.0 * np.exp(3.0 * (np.cos(centers - 1.0) - 1.0)))
        poisson = TestTuningPvalue._result(counts, np.full(n, 0.5))
        assert poisson.tuning_pvalue() < 1e-10
        assert poisson.is_head_direction_cell()

        unknown = replace(poisson, spike_count_dispersion=None)
        assert np.isnan(unknown.tuning_pvalue())
        assert not unknown.is_head_direction_cell()
        assert "dispersion" in unknown.interpretation()


# ==============================================================================
# DirectionalRateResult Classification Methods Tests
# ==============================================================================


class TestDirectionalRateResultIsHdCell:
    """Test DirectionalRateResult.is_head_direction_cell() method."""

    def test_is_hd_cell_true_for_sharply_tuned(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_head_direction_cell() returns True for sharply tuned neurons."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Very sharply tuned neuron (kappa=5)
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )

        # Should be classified as HD cell (high MVL, low p-value)
        assert result.is_head_direction_cell() is True

    def test_is_hd_cell_false_for_uniform(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """is_head_direction_cell() returns False for uniform firing."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Should NOT be classified as HD cell
        assert result.is_head_direction_cell() is False

    def test_is_hd_cell_uses_min_mvl_threshold(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_head_direction_cell() respects min_mvl parameter."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Moderately tuned neuron (kappa=2 gives MVL around 0.3-0.4)
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(2.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )

        mvl = result.mean_vector_length()

        # With high threshold (above MVL), should be False
        assert result.is_head_direction_cell(min_mvl=mvl + 0.1) is False

        # With low threshold (below MVL), should be True
        assert result.is_head_direction_cell(min_mvl=mvl - 0.1) is True

    def test_is_hd_cell_uses_alpha_threshold(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_head_direction_cell() respects alpha parameter."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Sharply tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )

        pval = result.tuning_pvalue()

        # With very strict alpha (below p-value), should be False
        # Note: sharply tuned will have very low p-value, so use even lower alpha
        assert result.is_head_direction_cell(alpha=pval / 10) is False

        # With permissive alpha (above p-value), should be True
        assert result.is_head_direction_cell(alpha=0.1) is True

    def test_is_hd_cell_default_thresholds(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_head_direction_cell() uses default thresholds of min_mvl=0.4, alpha=0.05."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Calling with no arguments should use defaults
        is_hd_default = result.is_head_direction_cell()
        is_hd_explicit = result.is_head_direction_cell(min_mvl=0.4, alpha=0.05)
        assert is_hd_default == is_hd_explicit

    def test_is_hd_cell_requires_both_criteria(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_head_direction_cell() requires BOTH MVL > threshold AND p-value < alpha."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Create tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvl = result.mean_vector_length()
        pval = result.tuning_pvalue()

        # Set thresholds so only MVL criterion is met
        # (high alpha threshold that is below the p-value)
        if pval > 0:
            assert (
                result.is_head_direction_cell(min_mvl=mvl - 0.1, alpha=pval / 10)
                is False
            )

        # Set thresholds so only p-value criterion is met
        # (high MVL threshold that is above the actual MVL)
        assert result.is_head_direction_cell(min_mvl=mvl + 0.1, alpha=0.1) is False


class TestDirectionalRateResultInterpretation:
    """Test DirectionalRateResult.interpretation() method."""

    def test_interpretation_hd_cell_format(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """interpretation() for HD cell includes key metrics."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Very sharply tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )

        interp = result.interpretation()

        # Should indicate it's an HD cell
        assert "HEAD DIRECTION CELL" in interp

        # Should include preferred direction
        assert "Preferred direction" in interp

        # Should include MVL
        assert "Mean vector length" in interp or "MVL" in interp

        # Should include peak firing rate
        assert "Peak" in interp or "peak" in interp or "firing" in interp.lower()

    def test_interpretation_non_hd_cell_format(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """interpretation() for non-HD cell explains why."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        interp = result.interpretation()

        # Should NOT say it's an HD cell
        assert "*** HEAD DIRECTION CELL ***" not in interp

        # Should indicate it's not an HD cell
        assert "Not classified as HD cell" in interp or "not" in interp.lower()

    def test_interpretation_explains_low_mvl(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """interpretation() explains when MVL is too low."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate (low MVL)
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        interp = result.interpretation()

        # Should mention MVL being too low
        assert "vector length" in interp.lower() or "mvl" in interp.lower()

    def test_interpretation_uses_min_mvl_threshold(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """interpretation() respects min_mvl parameter."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Moderately tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(2.5 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )

        mvl = result.mean_vector_length()

        # With low threshold, should be HD cell
        interp_low = result.interpretation(min_mvl=mvl - 0.1)
        assert "HEAD DIRECTION CELL" in interp_low

        # With high threshold, should NOT be HD cell
        interp_high = result.interpretation(min_mvl=mvl + 0.1)
        assert "HEAD DIRECTION CELL" not in interp_high

    def test_interpretation_includes_threshold_value(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """interpretation() includes the threshold value used."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )

        custom_threshold = 0.5
        interp = result.interpretation(min_mvl=custom_threshold)

        # Should mention the threshold somewhere
        assert str(custom_threshold) in interp or "0.5" in interp


# ==============================================================================
# DirectionalRatesResult Batch Methods Tests
# ==============================================================================


class TestDirectionalRatesResultPlot:
    """Test DirectionalRatesResult.plot() method."""

    def test_plot_returns_axes(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot(idx) returns matplotlib Axes object."""
        pytest.importorskip("matplotlib")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        ax = result.plot(0)
        assert ax is not None
        assert hasattr(ax, "plot")

    def test_plot_requires_idx(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot() requires idx parameter."""
        pytest.importorskip("matplotlib")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Should raise TypeError if idx not provided
        with pytest.raises(TypeError):
            result.plot()  # type: ignore[call-arg]

    def test_plot_polar_default(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot(idx) creates polar plot by default."""
        pytest.importorskip("matplotlib")
        from matplotlib.projections.polar import PolarAxes

        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        ax = result.plot(0, polar=True)
        assert isinstance(ax, PolarAxes)

    def test_plot_cartesian(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot(idx, polar=False) creates Cartesian plot."""
        pytest.importorskip("matplotlib")
        from matplotlib.projections.polar import PolarAxes

        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        ax = result.plot(0, polar=False)
        assert not isinstance(ax, PolarAxes)

    def test_plot_with_ax(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot(idx) accepts existing axes."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        _, ax_provided = plt.subplots(subplot_kw={"projection": "polar"})
        ax_returned = result.plot(0, ax=ax_provided)
        assert ax_returned is ax_provided

    def test_plot_with_kwargs(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """plot(idx) passes through kwargs."""
        pytest.importorskip("matplotlib")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Should not raise an error with valid kwargs
        ax = result.plot(0, color="red", linewidth=2)
        assert ax is not None


class TestDirectionalRatesResultPreferredDirections:
    """Test DirectionalRatesResult.preferred_directions() method."""

    def test_preferred_directions_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_directions() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dirs = result.preferred_directions()
        assert isinstance(pref_dirs, np.ndarray)

    def test_preferred_directions_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """preferred_directions() returns (n_neurons,) array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dirs = result.preferred_directions()
        assert pref_dirs.shape == (n_neurons,)

    def test_preferred_directions_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_directions() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dirs = result.preferred_directions()

        for i, single in enumerate(result):
            np.testing.assert_allclose(pref_dirs[i], single.preferred_direction())

    def test_preferred_directions_in_valid_range(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_directions() values are in [-π, π]."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        pref_dirs = result.preferred_directions()
        assert np.all(pref_dirs >= -np.pi)
        assert np.all(pref_dirs <= np.pi)


class TestDirectionalRatesResultMeanVectorLengths:
    """Test DirectionalRatesResult.mean_vector_lengths() method."""

    def test_mean_vector_lengths_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """mean_vector_lengths() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvls = result.mean_vector_lengths()
        assert isinstance(mvls, np.ndarray)

    def test_mean_vector_lengths_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """mean_vector_lengths() returns (n_neurons,) array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvls = result.mean_vector_lengths()
        assert mvls.shape == (n_neurons,)

    def test_mean_vector_lengths_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """mean_vector_lengths() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvls = result.mean_vector_lengths()

        for i, single in enumerate(result):
            np.testing.assert_allclose(mvls[i], single.mean_vector_length())

    def test_mean_vector_lengths_in_valid_range(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """mean_vector_lengths() values are in [0, 1]."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        mvls = result.mean_vector_lengths()
        assert np.all(mvls >= 0)
        assert np.all(mvls <= 1)


class TestDirectionalRatesResultTuningWidths:
    """Test DirectionalRatesResult.tuning_widths() method."""

    def test_tuning_widths_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_widths() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        widths = result.tuning_widths()
        assert isinstance(widths, np.ndarray)

    def test_tuning_widths_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """tuning_widths() returns (n_neurons,) array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        widths = result.tuning_widths()
        assert widths.shape == (n_neurons,)

    def test_tuning_widths_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_widths() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        widths = result.tuning_widths()

        for i, single in enumerate(result):
            np.testing.assert_allclose(widths[i], single.tuning_width())

    def test_tuning_widths_in_valid_range(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_widths() values are in (0, π] or NaN."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        widths = result.tuning_widths()
        # Filter out NaN values for range check
        valid_widths = widths[~np.isnan(widths)]
        if len(valid_widths) > 0:
            assert np.all(valid_widths > 0)
            assert np.all(valid_widths <= np.pi)


class TestDirectionalRatesResultClassify:
    """Test DirectionalRatesResult.classify() method."""

    def test_classify_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """classify() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        is_hd = result.classify()
        assert isinstance(is_hd, np.ndarray)

    def test_classify_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """classify() returns (n_neurons,) bool array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        is_hd = result.classify()
        assert is_hd.shape == (n_neurons,)
        assert is_hd.dtype == np.bool_

    def test_classify_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """classify() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        is_hd = result.classify()

        for i, single in enumerate(result):
            assert is_hd[i] == single.is_head_direction_cell()

    def test_classify_respects_min_mvl(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """classify() respects min_mvl parameter."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Very permissive threshold
        is_hd_permissive = result.classify(min_mvl=0.0)

        # Very strict threshold
        is_hd_strict = result.classify(min_mvl=0.99)

        # Strict should have fewer or equal HD cells
        assert np.sum(is_hd_strict) <= np.sum(is_hd_permissive)

    def test_classify_respects_alpha(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """classify() respects alpha parameter."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Very permissive alpha
        is_hd_permissive = result.classify(alpha=1.0)

        # Very strict alpha
        is_hd_strict = result.classify(alpha=1e-10)

        # Strict should have fewer or equal HD cells
        assert np.sum(is_hd_strict) <= np.sum(is_hd_permissive)

    def test_classify_default_thresholds(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """classify() uses default thresholds of min_mvl=0.4, alpha=0.05."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        is_hd_default = result.classify()
        is_hd_explicit = result.classify(min_mvl=0.4, alpha=0.05)

        np.testing.assert_array_equal(is_hd_default, is_hd_explicit)


class TestDirectionalRatesResultPeakFiringRates:
    """Test DirectionalRatesResult.peak_firing_rate() method."""

    def test_peak_firing_rate_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rate() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peaks = result.peak_firing_rate()
        assert isinstance(peaks, np.ndarray)

    def test_peak_firing_rate_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """peak_firing_rate() returns (n_neurons,) array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peaks = result.peak_firing_rate()
        assert peaks.shape == (n_neurons,)

    def test_peak_firing_rate_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rate() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peaks = result.peak_firing_rate()

        for i, single in enumerate(result):
            np.testing.assert_allclose(peaks[i], single.peak_firing_rate())

    def test_peak_firing_rate_nonnegative(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rate() values are non-negative."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        peaks = result.peak_firing_rate()
        assert np.all(peaks >= 0)


# ==============================================================================
# DirectionalRatesResult.summary_table() Tests
# ==============================================================================


class TestDirectionalRatesResultSummaryTable:
    """Test DirectionalRatesResult.summary_table() method."""

    @pytest.fixture
    def population(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ):
        from neurospatial.encoding.directional import DirectionalRatesResult

        return DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

    def test_summary_table_one_row_per_unit(self, population, n_neurons) -> None:
        df = population.summary_table()
        assert len(df) == n_neurons
        assert df.index.name == "unit_id"

    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ("peak_firing_rate", lambda r: r.peak_firing_rate()),
            ("mean_vector_length", lambda r: r.mean_vector_lengths()),
            ("preferred_direction", lambda r: r.preferred_directions()),
            ("preferred_direction_deg", lambda r: np.degrees(r.preferred_directions())),
            ("tuning_width", lambda r: r.tuning_widths()),
            ("tuning_width_deg", lambda r: np.degrees(r.tuning_widths())),
            ("tuning_pvalue", lambda r: r.tuning_pvalues()),
            ("is_head_direction_cell", lambda r: r.classify()),
        ],
    )
    def test_summary_table_column_matches_method(
        self, population, column, expected
    ) -> None:
        """Each column equals the batch method it summarizes."""
        df = population.summary_table()
        np.testing.assert_allclose(
            df[column].to_numpy(dtype=float),
            np.asarray(expected(population), dtype=float),
        )

    def test_summary_table_unit_id_default_integers(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """summary_table() uses integer unit_id index labels by default."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        expected_ids = list(range(n_neurons))
        assert list(df.index) == expected_ids

    def test_summary_table_custom_unit_ids(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """summary_table() accepts custom unit_ids."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        custom_ids = [f"unit_{i}" for i in range(n_neurons)]
        df = result.summary_table(unit_ids=custom_ids)
        assert list(df.index) == custom_ids

    def test_summary_table_unit_id_length_mismatch(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() raises ValueError for wrong unit_ids length."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        # Wrong number of unit_ids
        with pytest.raises(ValueError, match="unit_ids"):
            result.summary_table(unit_ids=["only_one"])

    def test_summary_table_preferred_direction_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() preferred_direction matches preferred_directions()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        pref_dirs = result.preferred_directions()
        np.testing.assert_allclose(df["preferred_direction"].values, pref_dirs)

    def test_summary_table_preferred_direction_deg_is_conversion(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() preferred_direction_deg is degrees conversion."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        np.testing.assert_allclose(
            df["preferred_direction_deg"].values,
            np.degrees(df["preferred_direction"].values),
        )

    def test_summary_table_mean_vector_length_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() mean_vector_length matches mean_vector_lengths()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        mvls = result.mean_vector_lengths()
        np.testing.assert_allclose(df["mean_vector_length"].values, mvls)

    def test_summary_table_tuning_width_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() tuning_width matches tuning_widths()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        widths = result.tuning_widths()
        np.testing.assert_allclose(df["tuning_width"].values, widths, equal_nan=True)

    def test_summary_table_tuning_width_deg_is_conversion(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() tuning_width_deg is degrees conversion."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        np.testing.assert_allclose(
            df["tuning_width_deg"].values,
            np.degrees(df["tuning_width"].values),
            equal_nan=True,
        )

    def test_summary_table_peak_rate_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() peak_rate matches peak_firing_rate()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        peaks = result.peak_firing_rate()
        np.testing.assert_allclose(df["peak_firing_rate"].values, peaks)

    def test_summary_table_is_hd_cell_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() is_head_direction_cell matches classify()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        is_hd = result.classify()
        np.testing.assert_array_equal(df["is_head_direction_cell"].values, is_hd)

    def test_summary_table_empty_result(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """summary_table() works with zero neurons."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        # Empty result (0 neurons)
        rates = np.empty((0, n_bins), dtype=np.float64)

        result = DirectionalRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        assert len(df) == 0

    def test_summary_table_single_neuron(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """summary_table() works with single neuron."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        # Single neuron
        rates = single_firing_rate.reshape(1, -1)

        result = DirectionalRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            bandwidth=None,
        )

        df = result.summary_table()
        assert len(df) == 1


# ==============================================================================
# compute_directional_rate() Tests
# ==============================================================================


class TestComputeDirectionalRateBasic:
    """Test basic compute_directional_rate functionality."""

    def test_returns_directional_rate_result(self) -> None:
        """compute_directional_rate returns DirectionalRateResult."""
        from neurospatial.encoding.directional import (
            DirectionalRateResult,
            compute_directional_rate,
        )

        # Create trajectory with varying head directions
        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        # Create spike times
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)
        assert isinstance(result, DirectionalRateResult)

    def test_bin_centers_in_radians(self) -> None:
        """compute_directional_rate produces bin_centers in radians [-π, π)."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        centers = np.asarray(result.bin_centers)
        assert np.all(centers >= -np.pi)
        assert np.all(centers < np.pi)

    def test_metadata_defaults(self) -> None:
        """compute_directional_rate stores correct default metadata."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        assert result.bin_size == np.pi / 30
        assert result.bandwidth is None

    def test_firing_rate_non_negative(self) -> None:
        """compute_directional_rate produces non-negative firing rates."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        rates = np.asarray(result.firing_rate)
        # Allow NaN for unvisited bins, but finite values should be non-negative
        assert np.all(rates[~np.isnan(rates)] >= 0)


class TestComputeDirectionalRateBinSize:
    """Test compute_directional_rate bin_size parameter."""

    def test_custom_bin_size_radians(self) -> None:
        """compute_directional_rate respects bin_size parameter in radians."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        # Use 15 degree bins = π/12 radians
        bin_size = np.pi / 12  # 24 bins
        result = compute_directional_rate(
            spike_times, times, headings, bin_size=bin_size
        )

        assert result.bin_size == bin_size
        # 2π / (π/12) = 24 bins
        assert np.asarray(result.firing_rate).shape == (24,)

    def test_custom_bin_size_degrees(self) -> None:
        """compute_directional_rate respects bin_size parameter in degrees."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 360, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        # Use 15 degree bins
        bin_size = 15.0  # 24 bins
        result = compute_directional_rate(
            spike_times, times, headings, bin_size=bin_size, angle_unit="deg"
        )

        # bin_size is stored in radians internally
        np.testing.assert_allclose(result.bin_size, np.radians(bin_size))
        # 360 / 15 = 24 bins
        assert np.asarray(result.firing_rate).shape == (24,)


class TestComputeDirectionalRateAngleUnit:
    """Test compute_directional_rate angle_unit parameter."""

    def test_angle_unit_rad_default(self) -> None:
        """compute_directional_rate uses radians by default."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)  # Radians
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        # Should work without errors
        assert result is not None
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_angle_unit_deg(self) -> None:
        """compute_directional_rate handles degrees correctly."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 360, 100)  # Degrees
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(
            spike_times, times, headings, bin_size=6.0, angle_unit="deg"
        )

        # Should produce 60 bins (360 / 6)
        assert np.asarray(result.firing_rate).shape == (60,)
        # bin_size stored in radians
        np.testing.assert_allclose(result.bin_size, np.radians(6.0))

    def test_angle_unit_rad_and_deg_equivalent(self) -> None:
        """compute_directional_rate produces equivalent results for rad and deg."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings_rad = np.random.uniform(0, 2 * np.pi, 100)
        headings_deg = np.degrees(headings_rad)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result_rad = compute_directional_rate(
            spike_times, times, headings_rad, bin_size=np.pi / 30, angle_unit="rad"
        )

        result_deg = compute_directional_rate(
            spike_times, times, headings_deg, bin_size=6.0, angle_unit="deg"
        )

        # Firing rates should be equivalent
        np.testing.assert_allclose(
            result_rad.firing_rate, result_deg.firing_rate, rtol=1e-10
        )


class TestComputeDirectionalRateSmoothing:
    """Test compute_directional_rate bandwidth parameter."""

    def test_bandwidth_none_default(self) -> None:
        """compute_directional_rate returns None bandwidth by default."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)
        assert result.bandwidth is None

    def test_bandwidth_stored(self) -> None:
        """compute_directional_rate stores bandwidth in result."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        bandwidth = np.pi / 6  # 30 degrees
        result = compute_directional_rate(
            spike_times, times, headings, bandwidth=bandwidth
        )
        assert result.bandwidth == bandwidth

    def test_smoothing_produces_smoother_curve(self) -> None:
        """compute_directional_rate with smoothing produces smoother curves."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result_raw = compute_directional_rate(spike_times, times, headings)
        result_smooth = compute_directional_rate(
            spike_times, times, headings, bandwidth=np.pi / 6
        )

        rates_raw = np.asarray(result_raw.firing_rate)
        rates_smooth = np.asarray(result_smooth.firing_rate)

        # Replace NaN with 0 for variance calculation
        rates_raw_filled = np.nan_to_num(rates_raw, nan=0.0)
        rates_smooth_filled = np.nan_to_num(rates_smooth, nan=0.0)

        # Smoothed curve should have lower bin-to-bin variance
        var_raw = np.var(np.diff(rates_raw_filled))
        var_smooth = np.var(np.diff(rates_smooth_filled))

        # Smoothed curve has smaller local variance (more gradual changes)
        assert var_smooth <= var_raw

    def test_bandwidth_degrees(self) -> None:
        """compute_directional_rate bandwidth with angle_unit=deg."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 360, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        # 30 degree smoothing
        result = compute_directional_rate(
            spike_times,
            times,
            headings,
            bin_size=6.0,
            bandwidth=30.0,
            angle_unit="deg",
        )

        # bandwidth is stored in radians
        np.testing.assert_allclose(result.bandwidth, np.radians(30.0))


class TestComputeDirectionalRateEdgeCases:
    """Test edge cases for compute_directional_rate."""

    def test_empty_spike_train(self) -> None:
        """compute_directional_rate handles empty spike train."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([])  # Empty

        result = compute_directional_rate(spike_times, times, headings)

        # Should return result with zero firing rates (or NaN)
        rates = np.asarray(result.firing_rate)
        assert np.all(rates[~np.isnan(rates)] == 0)

    def test_single_spike(self) -> None:
        """compute_directional_rate handles single spike."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([5.0])  # Single spike

        result = compute_directional_rate(spike_times, times, headings)

        # Should return valid result
        rates = np.asarray(result.firing_rate)
        # At least one bin should have non-zero rate
        assert np.sum(rates[~np.isnan(rates)] > 0) >= 1

    def test_spikes_outside_time_range_excluded(self) -> None:
        """compute_directional_rate excludes spikes outside time range."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(1, 9, 100)  # Time range: 1 to 9
        headings = np.random.uniform(0, 2 * np.pi, 100)
        # Spikes at 0.5 and 9.5 are outside time range
        spike_times = np.array([0.5, 2.0, 5.0, 9.5])

        result = compute_directional_rate(spike_times, times, headings)

        # Only the spikes at 2.0 and 5.0 fall inside the tracked time.
        assert result.spike_counts.sum() == 2
        rates = np.asarray(result.firing_rate)
        assert np.nansum(rates * result.occupancy) == pytest.approx(2.0)

    def test_all_same_heading(self) -> None:
        """compute_directional_rate handles constant head direction."""
        from neurospatial.encoding.directional import compute_directional_rate

        times = np.linspace(0, 10, 100)
        headings = np.ones(100) * np.pi  # All at 180 degrees
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        # All occupancy should be in one bin
        occ = np.asarray(result.occupancy)
        assert np.sum(occ > 0) == 1


class TestComputeDirectionalRateResultMethods:
    """Test that result from compute_directional_rate has working methods."""

    def test_preferred_direction_method(self) -> None:
        """Result has working preferred_direction method."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        # Use dense trajectory that covers all directions uniformly
        times = np.linspace(0, 60, 3600)  # 60 seconds at 60 Hz
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        # Create spikes correlated with heading near π/2
        spike_mask = np.abs(headings - np.pi / 2) < 0.5
        spike_times = times[spike_mask]

        result = compute_directional_rate(spike_times, times, headings)

        pref_dir = result.preferred_direction()
        assert isinstance(pref_dir, float)
        # Should be near π/2
        assert np.abs(pref_dir - np.pi / 2) < 1.0  # Within ~57 degrees

    def test_mean_vector_length_method(self) -> None:
        """Result has working mean_vector_length method."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        # Use dense trajectory that covers all directions
        times = np.linspace(0, 60, 3600)  # 60 seconds at 60 Hz
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        # Spikes at regular intervals
        spike_times = np.array([1.0, 2.5, 4.0, 7.5, 15.0, 30.0, 45.0])

        result = compute_directional_rate(spike_times, times, headings)

        mvl = result.mean_vector_length()
        assert isinstance(mvl, float)
        assert 0 <= mvl <= 1

    def test_is_hd_cell_method(self) -> None:
        """Result has working is_head_direction_cell method."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        # Use dense trajectory that covers all directions
        times = np.linspace(0, 60, 3600)  # 60 seconds at 60 Hz
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5, 15.0, 30.0])

        result = compute_directional_rate(spike_times, times, headings)

        is_hd = result.is_head_direction_cell()
        assert isinstance(is_hd, bool)

    def test_plot_method(self) -> None:
        """Result has working plot method."""
        pytest.importorskip("matplotlib")
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        # Use dense trajectory that covers all directions
        times = np.linspace(0, 60, 3600)  # 60 seconds at 60 Hz
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5, 15.0, 30.0])

        result = compute_directional_rate(spike_times, times, headings)

        ax = result.plot()
        assert ax is not None


class TestComputeDirectionalRateInputValidation:
    """Test input validation for compute_directional_rate."""

    def test_invalid_angle_unit(self) -> None:
        """compute_directional_rate raises ValueError for invalid angle_unit."""
        from neurospatial.encoding.directional import compute_directional_rate

        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0])

        with pytest.raises(ValueError, match="angle_unit"):
            compute_directional_rate(
                spike_times,
                times,
                headings,
                angle_unit="invalid",  # type: ignore[arg-type]
            )


# ==============================================================================
# compute_directional_rates() Tests
# ==============================================================================


class TestComputeDirectionalRatesBasic:
    """Test basic compute_directional_rates functionality."""

    def test_returns_directional_rates_result(self) -> None:
        """compute_directional_rates returns DirectionalRatesResult."""
        from neurospatial.encoding.directional import (
            DirectionalRatesResult,
            compute_directional_rates,
        )

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        # Create spike times for 3 neurons
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 3.0, 5.5, 8.0]),
            np.array([2.0, 6.0]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        assert isinstance(result, DirectionalRatesResult)

    def test_bin_size_stored_in_radians(self) -> None:
        """compute_directional_rates stores bin_size in radians."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5])]

        result = compute_directional_rates(spike_times, times, headings)

        # Default bin_size is π/30
        np.testing.assert_allclose(result.bin_size, np.pi / 30)

    def test_bandwidth_none_default(self) -> None:
        """compute_directional_rates returns None bandwidth by default."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5])]

        result = compute_directional_rates(spike_times, times, headings)
        assert result.bandwidth is None

    def test_len_returns_n_neurons(self) -> None:
        """len() of result returns number of units."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5]),
            np.array([0.5, 3.0]),
            np.array([2.0]),
            np.array([4.0, 7.0]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 4


class TestComputeDirectionalRatesInputFormats:
    """Test compute_directional_rates accepts various input formats."""

    def test_list_of_arrays(self) -> None:
        """compute_directional_rates accepts list of arrays."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5]),
            np.array([0.5, 3.0, 5.5]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 2

    def test_tuple_of_arrays(self) -> None:
        """compute_directional_rates accepts tuple of arrays."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = (
            np.array([1.0, 2.5]),
            np.array([0.5, 3.0, 5.5]),
        )

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 2

    def test_2d_array_with_nan_padding(self) -> None:
        """compute_directional_rates accepts 2D array with NaN padding."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        # 2D array with NaN padding
        spike_times = np.array(
            [
                [1.0, 2.5, 4.0, np.nan],
                [0.5, 3.0, np.nan, np.nan],
                [2.0, 6.0, 8.0, 9.0],
            ]
        )

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 3

    def test_1d_array_single_neuron(self) -> None:
        """compute_directional_rates accepts 1D array (single neuron)."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        # Single neuron as 1D array
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 1


class TestComputeDirectionalRatesNJobs:
    """Test compute_directional_rates n_jobs parameter."""

    def test_n_jobs_default_one(self) -> None:
        """compute_directional_rates defaults to n_jobs=1."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5]), np.array([0.5, 3.0])]

        # Should work without passing n_jobs
        result = compute_directional_rates(spike_times, times, headings)
        assert result is not None

    def test_n_jobs_parallel(self) -> None:
        """compute_directional_rates works with n_jobs > 1."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5]),
            np.array([0.5, 3.0]),
            np.array([2.0, 6.0]),
            np.array([4.0, 8.0]),
        ]

        result = compute_directional_rates(spike_times, times, headings, n_jobs=2)
        assert len(result) == 4

    def test_n_jobs_consistency(self) -> None:
        """compute_directional_rates produces same results with different n_jobs."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5]),
            np.array([0.5, 3.0]),
            np.array([2.0, 6.0]),
        ]

        result_seq = compute_directional_rates(spike_times, times, headings, n_jobs=1)
        result_par = compute_directional_rates(spike_times, times, headings, n_jobs=2)

        np.testing.assert_array_almost_equal(
            np.asarray(result_seq.firing_rates),
            np.asarray(result_par.firing_rates),
        )


class TestComputeDirectionalRatesAngleUnit:
    """Test compute_directional_rates angle_unit parameter."""

    def test_angle_unit_rad_default(self) -> None:
        """compute_directional_rates uses radians by default."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)  # Radians
        spike_times = [np.array([1.0, 2.5, 4.0])]

        result = compute_directional_rates(spike_times, times, headings)
        assert result is not None
        assert np.asarray(result.firing_rates).shape[1] > 0

    def test_angle_unit_deg(self) -> None:
        """compute_directional_rates handles degrees correctly."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 360, 100)  # Degrees
        spike_times = [np.array([1.0, 2.5, 4.0])]

        result = compute_directional_rates(
            spike_times, times, headings, bin_size=6.0, angle_unit="deg"
        )

        # Should produce 60 bins (360 / 6)
        assert np.asarray(result.firing_rates).shape == (1, 60)
        # bin_size stored in radians
        np.testing.assert_allclose(result.bin_size, np.radians(6.0))

    def test_angle_unit_rad_and_deg_equivalent(self) -> None:
        """compute_directional_rates produces equivalent results for rad and deg."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings_rad = np.random.uniform(0, 2 * np.pi, 100)
        headings_deg = np.degrees(headings_rad)
        spike_times = [np.array([1.0, 2.5, 4.0])]

        result_rad = compute_directional_rates(
            spike_times, times, headings_rad, bin_size=np.pi / 30, angle_unit="rad"
        )

        result_deg = compute_directional_rates(
            spike_times, times, headings_deg, bin_size=6.0, angle_unit="deg"
        )

        np.testing.assert_array_almost_equal(
            np.asarray(result_rad.firing_rates),
            np.asarray(result_deg.firing_rates),
        )


class TestComputeDirectionalRatesSmoothing:
    """Test compute_directional_rates bandwidth parameter."""

    def test_bandwidth_stored(self) -> None:
        """compute_directional_rates stores bandwidth in result."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5, 4.0])]

        bandwidth = np.pi / 6  # 30 degrees
        result = compute_directional_rates(
            spike_times, times, headings, bandwidth=bandwidth
        )
        assert result.bandwidth == bandwidth

    def test_smoothing_produces_smoother_curves(self) -> None:
        """compute_directional_rates with smoothing produces smoother curves."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5, 4.0, 7.5])]

        result_raw = compute_directional_rates(spike_times, times, headings)
        result_smooth = compute_directional_rates(
            spike_times, times, headings, bandwidth=np.pi / 6
        )

        rates_raw = np.asarray(result_raw.firing_rates[0])
        rates_smooth = np.asarray(result_smooth.firing_rates[0])

        # Replace NaN with 0 for variance calculation
        rates_raw_filled = np.nan_to_num(rates_raw, nan=0.0)
        rates_smooth_filled = np.nan_to_num(rates_smooth, nan=0.0)

        # Smoothed should have lower variance
        assert np.var(rates_smooth_filled) < np.var(rates_raw_filled)


class TestComputeDirectionalRatesConsistency:
    """Test consistency between batch and single-neuron computation."""

    def test_batch_matches_single_neuron_results(self) -> None:
        """compute_directional_rates matches single-neuron compute_directional_rate."""
        from neurospatial.encoding.directional import (
            compute_directional_rate,
            compute_directional_rates,
        )

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 3.0, 5.5, 8.0]),
            np.array([2.0, 6.0]),
        ]

        # Compute batch
        batch_result = compute_directional_rates(spike_times, times, headings)

        # Compute single-neuron for each
        for i, spikes in enumerate(spike_times):
            single_result = compute_directional_rate(spikes, times, headings)

            np.testing.assert_array_almost_equal(
                np.asarray(batch_result[i].firing_rate),
                np.asarray(single_result.firing_rate),
                decimal=10,
            )

    def test_batch_shares_occupancy(self) -> None:
        """compute_directional_rates shares occupancy across neurons."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 3.0, 5.5, 8.0]),
        ]

        result = compute_directional_rates(spike_times, times, headings)

        # Occupancy should be identical for all neurons
        occ_0 = np.asarray(result[0].occupancy)
        occ_1 = np.asarray(result[1].occupancy)
        np.testing.assert_array_equal(occ_0, occ_1)


class TestComputeDirectionalRatesEdgeCases:
    """Test edge cases for compute_directional_rates."""

    def test_empty_spike_train(self) -> None:
        """compute_directional_rates handles neurons with empty spike trains."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([]),  # Empty
            np.array([2.0, 6.0]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 3

        # Neuron with empty spikes should have zero (or NaN) firing rates
        rates_empty = np.asarray(result[1].firing_rate)
        assert np.all(rates_empty[~np.isnan(rates_empty)] == 0)

    def test_single_neuron(self) -> None:
        """compute_directional_rates works with single neuron."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5, 4.0])]

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 1

    def test_empty_list(self) -> None:
        """compute_directional_rates handles empty list (0 neurons)."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times: list[np.ndarray] = []

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 0

    def test_all_empty_spike_trains(self) -> None:
        """compute_directional_rates handles all neurons having empty spikes."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([]), np.array([]), np.array([])]

        result = compute_directional_rates(spike_times, times, headings)
        assert len(result) == 3


class TestComputeDirectionalRatesResultMethods:
    """Test that result from compute_directional_rates has working methods."""

    def test_preferred_directions_method(self) -> None:
        """Result has working preferred_directions method."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 60, 3600)  # 60 seconds at 60 Hz
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)

        # Create spikes for neurons with different preferred directions
        spike_times = []
        for pref_dir in [0, np.pi / 2, np.pi]:
            mask = np.abs(headings - pref_dir) < 0.5
            spike_times.append(times[mask])

        result = compute_directional_rates(spike_times, times, headings)
        pref_dirs = result.preferred_directions()

        assert isinstance(pref_dirs, np.ndarray)
        assert pref_dirs.shape == (3,)

    def test_mean_vector_lengths_method(self) -> None:
        """Result has working mean_vector_lengths method."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 60, 3600)
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 3.0, 5.5]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        mvls = result.mean_vector_lengths()

        assert isinstance(mvls, np.ndarray)
        assert mvls.shape == (2,)
        assert np.all((mvls >= 0) & (mvls <= 1))

    def test_classify_method(self) -> None:
        """Result has working classify method."""
        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 60, 3600)
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 3.0, 5.5]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        is_hd = result.classify()

        assert isinstance(is_hd, np.ndarray)
        assert is_hd.dtype == np.bool_
        assert is_hd.shape == (2,)

    def test_summary_table_method(self) -> None:
        """Result has working summary_table method."""
        pytest.importorskip("pandas")
        import pandas as pd

        from neurospatial.encoding.directional import compute_directional_rates

        np.random.seed(42)
        times = np.linspace(0, 60, 3600)
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        spike_times = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 3.0, 5.5]),
        ]

        result = compute_directional_rates(spike_times, times, headings)
        df = result.summary_table()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.index.name == "unit_id"
        assert "preferred_direction" in df.columns
        assert "mean_vector_length" in df.columns
        assert "is_head_direction_cell" in df.columns


class TestComputeDirectionalRatesInputValidation:
    """Test input validation for compute_directional_rates."""

    def test_invalid_angle_unit(self) -> None:
        """compute_directional_rates raises ValueError for invalid angle_unit."""
        from neurospatial.encoding.directional import compute_directional_rates

        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = [np.array([1.0, 2.5, 4.0])]

        with pytest.raises(ValueError, match="angle_unit"):
            compute_directional_rates(
                spike_times,
                times,
                headings,
                angle_unit="invalid",  # type: ignore[arg-type]
            )


class TestDirectionalRateResultNaNHandling:
    """Test that DirectionalRateResult handles NaN values gracefully.

    When bins have zero occupancy, firing_rate contains NaN for those bins.
    The circular statistics methods should handle this gracefully by masking
    NaN values before computation.
    """

    def test_preferred_direction_with_nans(self) -> None:
        """preferred_direction handles NaN values in firing_rate."""
        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        # Create tuning curve with some NaN values (unvisited bins)
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        firing_rate[::5] = np.nan  # Set every 5th bin to NaN

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins) * 0.5,
            bin_centers=bin_centers,
            bin_size=np.pi / 30,
            bandwidth=None,
        )
        pref_dir = result.preferred_direction()

        # Should still compute a valid preferred direction
        assert not np.isnan(pref_dir)
        # Should be close to π/2 (90 degrees)
        assert np.abs(pref_dir - np.pi / 2) < 0.2

    def test_mean_vector_length_with_nans(self) -> None:
        """mean_vector_length handles NaN values in firing_rate."""
        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        firing_rate[::5] = np.nan  # Set every 5th bin to NaN

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins) * 0.5,
            bin_centers=bin_centers,
            bin_size=np.pi / 30,
            bandwidth=None,
        )
        mvl = result.mean_vector_length()

        # Should still compute a valid MVL
        assert not np.isnan(mvl)
        assert 0 <= mvl <= 1

    def test_tuning_pvalue_with_nans(self) -> None:
        """tuning_pvalue handles NaN values in firing_rate."""
        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        firing_rate[::5] = np.nan  # Set every 5th bin to NaN

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=np.full(n_bins, 2.0),
            bin_centers=bin_centers,
            bin_size=np.pi / 30,
            bandwidth=None,
            spike_count_dispersion=1.0,
        )
        pval = result.tuning_pvalue()

        # Should still compute a valid p-value
        assert not np.isnan(pval)
        assert 0 <= pval <= 1

    def test_is_hd_cell_with_nans(self) -> None:
        """is_head_direction_cell handles NaN values in firing_rate."""
        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        firing_rate[::5] = np.nan  # Set every 5th bin to NaN

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=np.ones(n_bins) * 0.5,
            bin_centers=bin_centers,
            bin_size=np.pi / 30,
            bandwidth=None,
        )
        is_hd = result.is_head_direction_cell()

        # Should return a boolean (not error or NaN comparison issue)
        assert isinstance(is_hd, bool)

    def test_all_nans_returns_nan(self) -> None:
        """Metrics return NaN when all firing_rate values are NaN."""
        from neurospatial.encoding.directional import DirectionalRateResult

        n_bins = 60
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        firing_rate = np.full(n_bins, np.nan)

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=np.zeros(n_bins),  # All unvisited
            bin_centers=bin_centers,
            bin_size=np.pi / 30,
            bandwidth=None,
        )

        # All metrics should return NaN
        assert np.isnan(result.preferred_direction())
        assert np.isnan(result.mean_vector_length())
        assert np.isnan(result.tuning_pvalue())


class TestDirectionalBatchNaNHandling:
    """Regression: NaN bins must be dropped (not propagated) by the batched
    circular-stats reductions.

    A previous refactor used `rates.sum(axis=1)` and a plain matmul, which
    propagates NaN to every neuron with even one NaN bin. The single-neuron
    path masks NaNs first; the two paths must agree.
    """

    def test_preferred_directions_match_singletons_with_nan_bins(self) -> None:
        from neurospatial.encoding.directional import DirectionalRatesResult

        n_bins = 4
        bin_centers = np.linspace(0.0, 2 * np.pi, n_bins, endpoint=False)
        firing_rates = np.array(
            [
                [1.0, np.nan, 0.0, 0.0],
                [0.0, 2.0, np.nan, 0.0],
            ]
        )
        result = DirectionalRatesResult(
            firing_rates=firing_rates,
            occupancy=np.ones(n_bins) * 0.5,
            bin_centers=bin_centers,
            bin_size=2 * np.pi / n_bins,
            bandwidth=None,
        )

        batch = result.preferred_directions()
        singles = np.array([result[i].preferred_direction() for i in range(2)])
        np.testing.assert_allclose(batch, singles)
        assert np.all(np.isfinite(batch))

    def test_mean_vector_lengths_match_singletons_with_nan_bins(self) -> None:
        from neurospatial.encoding.directional import DirectionalRatesResult

        n_bins = 4
        bin_centers = np.linspace(0.0, 2 * np.pi, n_bins, endpoint=False)
        firing_rates = np.array(
            [
                [1.0, np.nan, 0.0, 0.0],
                [0.0, 2.0, np.nan, 0.0],
            ]
        )
        result = DirectionalRatesResult(
            firing_rates=firing_rates,
            occupancy=np.ones(n_bins) * 0.5,
            bin_centers=bin_centers,
            bin_size=2 * np.pi / n_bins,
            bandwidth=None,
        )

        batch = result.mean_vector_lengths()
        singles = np.array([result[i].mean_vector_length() for i in range(2)])
        np.testing.assert_allclose(batch, singles)
        assert np.all(np.isfinite(batch))


class TestDirectionalRateRecovery:
    """End-to-end recovery of a simulated HD tuning curve through compute_directional_rate.

    Drives the public ``compute_directional_rate`` on Poisson spikes from a known
    von Mises head-direction cell and asserts the recovered preferred direction
    matches the simulated one within two angular bins, including across the ±π
    wrap. Exercises the real circular binning + occupancy normalization, not a
    hand-built result object.
    """

    BIN_SIZE = 2 * np.pi / 36  # 36 angular bins over the full circle

    @pytest.fixture(scope="class")
    def headings_and_times(self):
        # Headings use their own RNG stream (seed 0); generate_poisson_spikes
        # uses seed 42. Seeding both identically would make the heading values
        # and the spike-acceptance uniforms the same underlying stream, biasing
        # spike acceptance against the tuning peak and corrupting recovery.
        times = np.arange(0.0, 600.0, 0.02)
        headings = np.random.default_rng(0).uniform(-np.pi, np.pi, times.size)
        return headings, times

    @pytest.mark.parametrize(
        "true_pref", [0.0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 3]
    )
    def test_preferred_direction_recovered(self, headings_and_times, true_pref):
        from neurospatial.encoding.directional import compute_directional_rate
        from neurospatial.simulation import (
            HeadDirectionCellModel,
            generate_poisson_spikes,
        )

        headings, times = headings_and_times
        model = HeadDirectionCellModel(
            preferred_direction=true_pref,
            concentration=4.0,
            max_rate=15.0,
            baseline_rate=1.0,
        )
        rates = model.firing_rate(headings)
        spike_times = generate_poisson_spikes(rates, times, seed=42)

        result = compute_directional_rate(
            spike_times, times, headings, bin_size=self.BIN_SIZE
        )

        # Circular distance between recovered and true preferred direction,
        # within two angular bins. Wraps correctly across ±π.
        circ_dist = abs(
            np.angle(np.exp(1j * (result.preferred_direction() - true_pref)))
        )
        assert circ_dist < 2 * self.BIN_SIZE


class TestComputeDirectionalRateSpikeCounts:
    """compute_directional_rate / _rates plumb spike counts through."""

    def test_compute_directional_rate_stores_spike_counts(
        self, uniform_heading_trajectory, von_mises_hd_spikes
    ) -> None:
        from neurospatial.encoding.directional import compute_directional_rate

        times, headings = uniform_heading_trajectory
        spikes = von_mises_hd_spikes(times, headings, np.pi / 2, 4.0, seed=1)
        result = compute_directional_rate(spikes, times, headings)

        assert result.spike_counts is not None
        # Spikes inside the recording window (last frame excluded from binning).
        in_window = np.sum((spikes >= times[0]) & (spikes <= times[-1]))
        assert result.spike_counts.sum() == pytest.approx(float(in_window))

    def test_directional_rates_getitem_forwards_counts(
        self, uniform_heading_trajectory, von_mises_hd_spikes
    ) -> None:
        from neurospatial.encoding.directional import compute_directional_rates

        times, headings = uniform_heading_trajectory
        s0 = von_mises_hd_spikes(times, headings, np.pi / 2, 4.0, seed=2)
        s1 = von_mises_hd_spikes(times, headings, np.pi, 4.0, seed=3)
        batch = compute_directional_rates([s0, s1], times, headings)

        assert batch.spike_counts is not None
        np.testing.assert_array_equal(
            batch[1].spike_counts, np.asarray(batch.spike_counts)[1]
        )
        # Round-trips through tuning_pvalue without error.
        assert np.isfinite(batch[1].tuning_pvalue())


# ==============================================================================
# tuning_width with NaN bins
# ==============================================================================


class TestTuningWidthNaN:
    """tuning_width interpolates across NaN bins at the half-max crossing."""

    @staticmethod
    def _sharp_curve(n=60, kappa=5.0):
        from neurospatial.encoding.directional import DirectionalRateResult

        centers = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rate = 10.0 * np.exp(kappa * (np.cos(centers - np.pi / 2) - 1.0))
        return (
            DirectionalRateResult(
                firing_rate=rate,
                occupancy=np.ones(n) * 0.5,
                bin_centers=centers,
                bin_size=2 * np.pi / n,
                bandwidth=None,
            ),
            rate,
            centers,
        )

    def test_tuning_width_skips_nan_neighbor(self) -> None:
        from neurospatial.encoding.directional import DirectionalRateResult

        result, rate, centers = self._sharp_curve()
        baseline = result.tuning_width()
        assert np.isfinite(baseline)

        # Insert a single NaN between the peak and the right half-max crossing.
        peak_idx = int(np.nanargmax(rate))
        half_max = rate[peak_idx] / 2.0
        # Find first finite bin below half-max on the right.
        n = len(rate)
        cross = None
        for off in range(1, n // 2):
            idx = (peak_idx + off) % n
            if rate[idx] < half_max:
                cross = off
                break
        assert cross is not None and cross >= 2
        nan_rate = rate.copy()
        nan_rate[(peak_idx + 1) % n] = np.nan
        nan_result = DirectionalRateResult(
            firing_rate=nan_rate,
            occupancy=np.ones(n) * 0.5,
            bin_centers=centers,
            bin_size=2 * np.pi / n,
            bandwidth=None,
        )
        width = nan_result.tuning_width()
        assert np.isfinite(width)
        assert width == pytest.approx(baseline, abs=2 * (2 * np.pi / n))

    def test_tuning_width_one_sided_nan(self) -> None:
        from neurospatial.encoding.directional import DirectionalRateResult

        result, rate, centers = self._sharp_curve()
        n = len(rate)
        peak_idx = int(np.nanargmax(rate))
        # NaN out the entire left half (decreasing index) of the curve.
        masked = rate.copy()
        for off in range(1, n // 2 + 1):
            masked[(peak_idx - off) % n] = np.nan
        one_sided = DirectionalRateResult(
            firing_rate=masked,
            occupancy=np.ones(n) * 0.5,
            bin_centers=centers,
            bin_size=2 * np.pi / n,
            bandwidth=None,
        )
        width = one_sided.tuning_width()
        assert np.isfinite(width)
        # Equals the right-side half-width of the original (single finite side).
        assert width == pytest.approx(result.tuning_width(), abs=2 * (2 * np.pi / n))


# ==============================================================================
# is_head_direction_cell: validation outside try
# ==============================================================================


class TestIsHeadDirectionCellValidation:
    """Genuine input errors raise; only non-significance returns False."""

    def test_is_head_direction_cell_raises_on_bad_input(
        self, uniform_heading_trajectory
    ) -> None:
        from neurospatial.encoding.directional import is_head_direction_cell

        times, headings = uniform_heading_trajectory
        spikes = np.sort(np.random.default_rng(0).uniform(0, 60, 50))
        with pytest.raises(ValueError):
            is_head_direction_cell(spikes, times, headings[:-5])

    def test_is_head_direction_cell_raises_on_invalid_angle_unit(
        self, uniform_heading_trajectory
    ) -> None:
        # A typo such as angle_unit="degrees" must surface as a ValueError
        # rather than being swallowed by the except into a False
        # classification.
        from neurospatial.encoding.directional import is_head_direction_cell

        times, headings = uniform_heading_trajectory
        spikes = np.sort(np.random.default_rng(0).uniform(0, 60, 50))
        with pytest.raises(ValueError, match="angle_unit"):
            is_head_direction_cell(
                spikes,
                times,
                headings,
                angle_unit="degrees",  # type: ignore[arg-type]
            )

    def test_is_head_direction_cell_false_on_uniform(
        self, uniform_heading_trajectory
    ) -> None:
        from neurospatial.encoding.directional import is_head_direction_cell

        times, headings = uniform_heading_trajectory
        # Uniform random spikes -> no directional preference.
        spikes = np.sort(np.random.default_rng(7).uniform(0, 60, 300))
        assert is_head_direction_cell(spikes, times, headings) is False

    @pytest.mark.slow
    def test_is_head_direction_cell_recovers_hd_cell(
        self, uniform_heading_trajectory, von_mises_hd_spikes
    ) -> None:
        from neurospatial.encoding.directional import (
            compute_directional_rate,
            is_head_direction_cell,
        )

        times, headings = uniform_heading_trajectory
        true_pref = np.pi / 2
        spikes = von_mises_hd_spikes(
            times, headings, true_pref, 6.0, peak_rate=40.0, seed=11
        )
        assert is_head_direction_cell(spikes, times, headings) is True
        result = compute_directional_rate(spikes, times, headings)
        circ_dist = abs(
            np.angle(np.exp(1j * (result.preferred_direction() - true_pref)))
        )
        assert circ_dist < np.radians(10)


@pytest.mark.parametrize("caller_axis", [False, True], ids=["new axis", "caller axis"])
def test_head_direction_plot_north_is_up(caller_axis, polar_display_offset) -> None:
    """Head directions are drawn as in the arena: 0 = East, π/2 = North (up).

    A caller-supplied polar axis configured North-up/clockwise is normalized.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    from neurospatial.encoding.directional import (
        DirectionalRateResult,
        plot_head_direction_tuning,
    )

    bin_centers = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    result = DirectionalRateResult(
        firing_rate=10.0 * np.exp(2.0 * (np.cos(bin_centers - np.pi / 2) - 1)),
        occupancy=np.full(60, 0.5),
        bin_centers=bin_centers,
        bin_size=np.pi / 30,
        bandwidth=None,
    )
    ax = None
    if caller_axis:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    ax = plot_head_direction_tuning(result, ax=ax)

    north_dx, north_dy = polar_display_offset(ax, np.pi / 2, 5.0)
    east_dx, east_dy = polar_display_offset(ax, 0.0, 5.0)
    assert abs(north_dx) < 1e-6 * north_dy
    assert north_dy > 0
    assert abs(east_dy) < 1e-6 * east_dx
    assert east_dx > 0
    plt.close("all")


# ==============================================================================
# One angle range for every directional output
# ==============================================================================


def _circular_distance(a: float, b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (a - b)))))


class TestHeadingRange:
    """Bin centers, peaks and preferred directions share the range [-π, π)."""

    @pytest.fixture(scope="class")
    def session(self) -> tuple[np.ndarray, np.ndarray]:
        times = np.arange(0.0, 600.0, 1 / 30)
        headings = np.random.default_rng(0).uniform(-np.pi, np.pi, times.size)
        return times, headings

    @staticmethod
    def _hd_spikes(times, headings, preferred, seed):
        rate = 0.5 + 15.0 * np.exp(4.0 * (np.cos(headings - preferred) - 1.0))
        return _frame_poisson_spikes(times, rate, seed=seed)

    def test_preferred_direction_near_pi_recovered(self, session) -> None:
        """A cell preferring π - 0.05 gives one answer from every accessor."""
        from neurospatial.encoding.directional import compute_directional_rate

        times, headings = session
        preferred = np.pi - 0.05
        spikes = self._hd_spikes(times, headings, preferred, seed=4)
        result = compute_directional_rate(spikes, times, headings)

        centers = np.asarray(result.bin_centers)
        assert centers.min() >= -np.pi
        assert centers.max() < np.pi
        assert _circular_distance(result.preferred_direction(), preferred) < np.radians(
            6
        )
        peak = float(result.peak_location())
        assert -np.pi <= peak < np.pi
        # Same number, not merely the same angle modulo 2π.
        assert abs(peak - result.preferred_direction()) < 2 * result.bin_size
        df = result.to_dataframe()
        assert df["bin_center_angle"].between(-np.pi, np.pi, inclusive="left").all()

    @pytest.mark.parametrize("delta", [0.7, -2.0, np.pi])
    def test_rotating_headings_rotates_preferred_direction(
        self, session, delta
    ) -> None:
        from neurospatial.encoding.directional import compute_directional_rate

        times, headings = session
        spikes = self._hd_spikes(times, headings, 0.3, seed=5)
        base = compute_directional_rate(spikes, times, headings)
        rotated = compute_directional_rate(spikes, times, headings + delta)
        shift = rotated.preferred_direction() - base.preferred_direction()
        assert _circular_distance(shift, delta) < np.radians(1)
        assert rotated.mean_vector_length() == pytest.approx(
            base.mean_vector_length(), abs=0.01
        )

    def test_zero_to_two_pi_headings_match_signed_headings(self, session) -> None:
        from neurospatial.encoding.directional import compute_directional_rate

        times, headings = session
        spikes = self._hd_spikes(times, headings, -2.5, seed=6)
        signed = compute_directional_rate(spikes, times, headings)
        unsigned = compute_directional_rate(spikes, times, headings % (2 * np.pi))
        assert_array_equal(unsigned.firing_rate, signed.firing_rate)
        assert_array_equal(unsigned.bin_centers, signed.bin_centers)
        assert unsigned.preferred_direction() == pytest.approx(
            signed.preferred_direction()
        )

    def test_summary_records_allocentric_frame(self, session) -> None:
        from neurospatial.encoding.directional import compute_directional_rates

        times, headings = session
        result = compute_directional_rates(
            [self._hd_spikes(times, headings, 1.0, seed=7)], times, headings
        )
        assert result.summary()["direction_frame"] == "allocentric"
        assert result[0].summary()["direction_frame"] == "allocentric"
        assert result.summary_table().attrs["direction_frame"] == "allocentric"


# ==============================================================================
# Heading coverage
# ==============================================================================


class TestHeadingCoverage:
    """MVL over a partial heading range is inflated; the screen says so."""

    @pytest.fixture(scope="class")
    def limited_session(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Headings within ±60° of East and 20 untuned 2 Hz units."""
        times = np.arange(0.0, 600.0, 1 / 30)
        headings = np.random.default_rng(0).uniform(-np.pi / 3, np.pi / 3, times.size)
        rng = np.random.default_rng(1)
        trains = [np.sort(rng.uniform(0.0, times[-1], 1200)) for _ in range(20)]
        return times, headings, trains

    def test_largest_unvisited_arc(self) -> None:
        from neurospatial.encoding.directional import DirectionalRateResult

        n = 12
        occupancy = np.ones(n)
        occupancy[[3, 4, 5, 11, 0]] = 0.0  # runs of 3 and of 2 across the wrap
        result = DirectionalRateResult(
            firing_rate=np.where(occupancy > 0, 1.0, np.nan),
            occupancy=occupancy,
            bin_centers=np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n,
            bin_size=2 * np.pi / n,
            bandwidth=None,
        )
        assert result.largest_unvisited_arc() == pytest.approx(3 * 2 * np.pi / n)

    def test_wrapping_run_and_full_coverage(self) -> None:
        from neurospatial.encoding.directional import DirectionalRatesResult

        n = 12
        centers = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
        occupancy = np.ones(n)
        occupancy[[10, 11, 0, 1]] = 0.0
        result = DirectionalRatesResult(
            firing_rates=np.ones((2, n)),
            occupancy=occupancy,
            bin_centers=centers,
            bin_size=2 * np.pi / n,
            bandwidth=None,
        )
        assert result.largest_unvisited_arc() == pytest.approx(4 * 2 * np.pi / n)
        full = DirectionalRatesResult(
            firing_rates=np.ones((2, n)),
            occupancy=np.ones(n),
            bin_centers=centers,
            bin_size=2 * np.pi / n,
            bandwidth=None,
        )
        assert full.largest_unvisited_arc() == 0.0

    def test_limited_headings_warn_where_mvl_verdict_is_made(
        self, limited_session
    ) -> None:
        from neurospatial.encoding.directional import (
            compute_directional_rates,
            is_head_direction_cell,
        )

        times, headings, trains = limited_session
        result = compute_directional_rates(trains, times, headings)
        # Untuned cells over a 120° arc look tuned by MVL alone.
        assert np.median(result.mean_vector_lengths()) > 0.7
        assert result.largest_unvisited_arc() == pytest.approx(
            np.radians(240), abs=result.bin_size
        )
        with pytest.warns(UserWarning, match="unvisited"):
            flags = result.classify()
        # The occupancy-corrected test keeps the screen from passing them.
        assert flags.sum() <= 2
        with pytest.warns(UserWarning, match='criterion="shuffle"'):
            table = result.summary_table()
        assert table.attrs["largest_unvisited_arc"] == pytest.approx(
            result.largest_unvisited_arc()
        )
        with pytest.warns(UserWarning, match="unvisited"):
            result[0].is_head_direction_cell()
        with pytest.warns(UserWarning, match="unvisited"):
            is_head_direction_cell(trains[0], times, headings)

    def test_full_heading_coverage_does_not_warn(self, biased_heading_session) -> None:
        import warnings

        from neurospatial.encoding.directional import compute_directional_rate

        times, headings = biased_heading_session
        spikes = np.sort(np.random.default_rng(3).uniform(0.0, times[-1], 600))
        result = compute_directional_rate(spikes, times, headings)
        assert result.largest_unvisited_arc() < np.pi / 2
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result.is_head_direction_cell()
            result.summary_table()


class TestDegreeDefaults:
    """The default bin width is 6° whatever unit the headings use."""

    @pytest.fixture(scope="class")
    def session(self):
        times = np.arange(0.0, 300.0, 1 / 30)
        headings = np.random.default_rng(0).uniform(-np.pi, np.pi, times.size)
        rate = 0.5 + 10.0 * np.exp(3.0 * (np.cos(headings - 1.0) - 1.0))
        return times, headings, _frame_poisson_spikes(times, rate, seed=9)

    def test_rate_default_bin_size_matches_radians(self, session) -> None:
        from neurospatial.encoding.directional import (
            compute_directional_rate,
            compute_directional_rates,
        )

        times, headings, spikes = session
        rad = compute_directional_rate(spikes, times, headings)
        deg = compute_directional_rate(
            spikes, times, np.degrees(headings), angle_unit="deg"
        )
        assert len(rad.firing_rate) == len(deg.firing_rate) == 60
        np.testing.assert_allclose(deg.firing_rate, rad.firing_rate)
        assert deg.tuning_width() == pytest.approx(rad.tuning_width())
        rates = compute_directional_rates(
            [spikes], times, np.degrees(headings), angle_unit="deg"
        )
        np.testing.assert_allclose(rates.firing_rates[0], rad.firing_rate)

    def test_predicates_default_bin_size_matches_radians(self, session) -> None:
        from neurospatial.encoding.directional import (
            head_direction_cell_significance,
            is_head_direction_cell,
        )

        times, headings, spikes = session
        assert is_head_direction_cell(spikes, times, headings) is True
        assert (
            is_head_direction_cell(
                spikes, times, np.degrees(headings), angle_unit="deg"
            )
            is True
        )
        kwargs = {"n_shuffles": 9, "min_shift": 20.0, "rng": 0}
        deg = head_direction_cell_significance(
            [spikes], times, np.degrees(headings), angle_unit="deg", **kwargs
        )[0]
        rad = head_direction_cell_significance([spikes], times, headings, **kwargs)[0]
        assert deg.observed_score == pytest.approx(rad.observed_score)


@pytest.mark.parametrize(
    ("rate", "occupancy"),
    [(np.full(12, np.nan), np.zeros(12)), (np.zeros(12), np.ones(12))],
    ids=["never-visited", "silent"],
)
def test_undefined_tuning_metrics_are_nan(rate, occupancy) -> None:
    """A map with no finite or no positive rate has undefined tuning, not an error."""
    from neurospatial.encoding.directional import (
        DirectionalRateResult,
        DirectionalRatesResult,
    )

    centers = np.linspace(-np.pi, np.pi, 12, endpoint=False) + np.pi / 12
    result = DirectionalRateResult(
        firing_rate=rate,
        occupancy=occupancy,
        bin_centers=centers,
        bin_size=np.pi / 6,
        bandwidth=None,
    )
    assert np.isnan(result.preferred_direction())
    assert np.isnan(result.preferred_direction_deg())
    assert np.isnan(result.tuning_width())
    assert np.isnan(result.tuning_width_deg())
    assert "undefined" in result.interpretation()
    population = DirectionalRatesResult(
        firing_rates=np.vstack([rate, 1.0 + np.cos(centers)]),
        occupancy=occupancy,
        bin_centers=centers,
        bin_size=np.pi / 6,
        bandwidth=None,
    )
    widths = population.tuning_widths()
    assert np.isnan(widths[0])
    assert np.isnan(population.summary_table()["tuning_width"].iloc[0])


class TestBackendParity:
    """backend="jax" changes the output array type, never the statistics."""

    @pytest.fixture(scope="class")
    def session(self):
        times = np.arange(0.0, 120.0, 1 / 30)
        headings = np.random.default_rng(0).uniform(-np.pi, np.pi, times.size)
        rate = 0.5 + 10.0 * np.exp(3.0 * (np.cos(headings + 1.0) - 1.0))
        flat = np.sort(np.random.default_rng(2).uniform(0.0, times[-1], 200))
        return times, headings, [_frame_poisson_spikes(times, rate, seed=8), flat]

    def test_rates_and_metrics_match(self, session) -> None:
        pytest.importorskip("jax")
        from neurospatial.encoding.directional import compute_directional_rates

        times, headings, trains = session
        ref = compute_directional_rates(trains, times, headings, bandwidth=0.2)
        jax_result = compute_directional_rates(
            trains, times, headings, bandwidth=0.2, backend="jax"
        )
        np.testing.assert_allclose(
            np.asarray(jax_result.firing_rates), ref.firing_rates, rtol=1e-6
        )
        np.testing.assert_allclose(
            jax_result.tuning_pvalues(), ref.tuning_pvalues(), rtol=1e-6
        )
        assert_array_equal(jax_result.classify(), ref.classify())
        assert ref.classify().tolist() == [True, False]

    @pytest.mark.parametrize("criterion", ["threshold", "shuffle"])
    def test_predicate_matches(self, session, criterion) -> None:
        pytest.importorskip("jax")
        from neurospatial.encoding.directional import is_head_direction_cell

        times, headings, trains = session
        kwargs = {"n_shuffles": 19, "rng": 0} if criterion == "shuffle" else {}
        for train in trains:
            assert is_head_direction_cell(
                train, times, headings, criterion=criterion, backend="jax", **kwargs
            ) == is_head_direction_cell(
                train, times, headings, criterion=criterion, **kwargs
            )

    def test_significance_matches(self, session) -> None:
        pytest.importorskip("jax")
        from neurospatial.encoding.directional import head_direction_cell_significance

        times, headings, trains = session
        ref = head_direction_cell_significance(
            trains, times, headings, n_shuffles=19, rng=0
        )
        jax_result = head_direction_cell_significance(
            trains, times, headings, n_shuffles=19, rng=0, backend="jax"
        )
        for unit in ref:
            assert jax_result[unit].observed_score == pytest.approx(
                ref[unit].observed_score, rel=1e-6
            )
            np.testing.assert_allclose(
                jax_result[unit].null_scores, ref[unit].null_scores, rtol=1e-6
            )
            assert jax_result[unit].p_value == ref[unit].p_value


@pytest.mark.parametrize("bandwidth", [0.0, -0.1, np.nan, np.inf])
@pytest.mark.parametrize("angle_unit", ["rad", "deg"])
def test_directional_rates_refuse_a_bandwidth_that_cannot_smooth(
    bandwidth: float, angle_unit: str
) -> None:
    """A zero, negative or non-finite width is refused, never read as unsmoothed."""
    from neurospatial.encoding import (
        compute_directional_rate,
        compute_directional_rates,
    )

    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 100.0, 1000)
    headings = rng.uniform(-np.pi, np.pi, 1000)
    spikes = np.sort(np.random.default_rng(1).uniform(0.0, 100.0, 50))
    if angle_unit == "deg":
        headings = np.degrees(headings)
    for compute, spike_times in (
        (compute_directional_rate, spikes),
        (compute_directional_rates, [spikes]),
    ):
        with pytest.raises(ValueError, match="is not a smoothing width"):
            compute(
                spike_times,
                times,
                headings,
                bandwidth=bandwidth,
                angle_unit=angle_unit,
            )
