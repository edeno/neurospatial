"""Tests for directional rate computation result classes.

This module tests DirectionalRateResult and DirectionalRatesResult dataclasses
defined in encoding/directional.py for head direction cell analysis.

Task 3.1: Result class definitions (no Environment dependency, only bin_centers)
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
# DirectionalRateResult Tests - Task 3.1
# ==============================================================================


class TestDirectionalRateResultImport:
    """Test that DirectionalRateResult can be imported."""

    def test_import_from_directional(self) -> None:
        """DirectionalRateResult can be imported from encoding.directional."""
        from neurospatial.encoding.directional import DirectionalRateResult

        assert DirectionalRateResult is not None

    def test_import_from_encoding(self) -> None:
        """DirectionalRateResult can be imported from encoding package."""
        from neurospatial.encoding import DirectionalRateResult

        assert DirectionalRateResult is not None


class TestDirectionalRateResultCreation:
    """Test DirectionalRateResult dataclass creation."""

    def test_create_result(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """DirectionalRateResult can be created with all required fields."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        assert result is not None
        assert result.firing_rate is not None
        assert result.occupancy is not None
        assert result.bin_centers is not None
        assert result.bin_size == bin_size
        assert result.smoothing_sigma is None

    def test_create_result_with_smoothing(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """DirectionalRateResult can be created with smoothing_sigma."""
        from neurospatial.encoding.directional import DirectionalRateResult

        smoothing_sigma = np.pi / 6  # 30 degrees

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=smoothing_sigma,
        )

        assert result.smoothing_sigma == smoothing_sigma

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
            smoothing_sigma=None,
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.bin_size = 0.2  # type: ignore[misc]


class TestDirectionalRateResultFields:
    """Test DirectionalRateResult field access and shapes."""

    def test_firing_rate_shape(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """firing_rate has correct shape (n_bins,)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        rate = np.asarray(result.firing_rate)
        assert rate.shape == (n_bins,)

    def test_occupancy_shape(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """occupancy has correct shape (n_bins,)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        occ = np.asarray(result.occupancy)
        assert occ.shape == (n_bins,)

    def test_bin_centers_shape(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """bin_centers has correct shape (n_bins,)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        centers = np.asarray(result.bin_centers)
        assert centers.shape == (n_bins,)

    def test_bin_centers_in_radians(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """bin_centers are in radians [0, 2π)."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        centers = np.asarray(result.bin_centers)
        assert np.all(centers >= 0)
        assert np.all(centers < 2 * np.pi)


# ==============================================================================
# DirectionalRatesResult Tests - Task 3.1
# ==============================================================================


class TestDirectionalRatesResultImport:
    """Test that DirectionalRatesResult can be imported."""

    def test_import_from_directional(self) -> None:
        """DirectionalRatesResult can be imported from encoding.directional."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        assert DirectionalRatesResult is not None

    def test_import_from_encoding(self) -> None:
        """DirectionalRatesResult can be imported from encoding package."""
        from neurospatial.encoding import DirectionalRatesResult

        assert DirectionalRatesResult is not None


class TestDirectionalRatesResultCreation:
    """Test DirectionalRatesResult dataclass creation."""

    def test_create_result(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """DirectionalRatesResult can be created with all required fields."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        assert result is not None
        assert result.firing_rates is not None
        assert result.occupancy is not None
        assert result.bin_centers is not None
        assert result.bin_size == bin_size
        assert result.smoothing_sigma is None

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
            smoothing_sigma=None,
        )

        with pytest.raises(FrozenInstanceError):
            result.bin_size = 0.2  # type: ignore[misc]


class TestDirectionalRatesResultFields:
    """Test DirectionalRatesResult field access and shapes."""

    def test_firing_rates_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
        n_bins: int,
    ) -> None:
        """firing_rates has correct shape (n_neurons, n_bins)."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        rates = np.asarray(result.firing_rates)
        assert rates.shape == (n_neurons, n_bins)

    def test_occupancy_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """occupancy has correct shape (n_bins,) - shared across neurons."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        occ = np.asarray(result.occupancy)
        assert occ.shape == (n_bins,)


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
        """len() returns number of neurons."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
        """__getitem__ preserves occupancy, bin_centers, bin_size, smoothing_sigma."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        smoothing_sigma = np.pi / 6

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=smoothing_sigma,
        )

        single = result[0]

        assert_array_equal(np.asarray(single.occupancy), single_occupancy)
        assert_array_equal(np.asarray(single.bin_centers), bin_centers)
        assert single.bin_size == bin_size
        assert single.smoothing_sigma == smoothing_sigma

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        assert len(result) == 0
        assert list(result) == []


# ==============================================================================
# DirectionalRateResult Convenience Methods Tests - Task 3.2
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        # Should not raise an error with valid kwargs
        ax = result.plot(color="red", linewidth=2)
        assert ax is not None


class TestDirectionalRateResultPreferredDirection:
    """Test DirectionalRateResult.preferred_direction() method."""

    def test_preferred_direction_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_direction() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pref_dir = result.preferred_direction()
        assert isinstance(pref_dir, float)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        # For uniform firing, circular_mean with uniform weights should return
        # a value (though it may not be meaningful)
        pref_dir = result.preferred_direction()
        # Should return a float (either NaN or a value)
        assert isinstance(pref_dir, float)


class TestDirectionalRateResultPreferredDirectionDeg:
    """Test DirectionalRateResult.preferred_direction_deg() method."""

    def test_preferred_direction_deg_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """preferred_direction_deg() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pref_dir_deg = result.preferred_direction_deg()
        assert isinstance(pref_dir_deg, float)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        pref_dir_deg = result.preferred_direction_deg()
        assert -180 <= pref_dir_deg <= 180


class TestDirectionalRateResultPeakFiringRate:
    """Test DirectionalRateResult.peak_firing_rate() method."""

    def test_peak_firing_rate_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rate() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        peak = result.peak_firing_rate()
        assert isinstance(peak, float)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        peak = result.peak_firing_rate()
        assert peak >= 0


# ==============================================================================
# DirectionalRateResult Tuning Metrics Tests - Task 3.3
# ==============================================================================


class TestDirectionalRateResultMeanVectorLength:
    """Test DirectionalRateResult.mean_vector_length() method."""

    def test_mean_vector_length_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """mean_vector_length() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        mvl = result.mean_vector_length()
        assert isinstance(mvl, float)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        # Uniform firing rate (untuned)
        uniform_rate = np.ones(n_bins) * 5.0

        uniform_result = DirectionalRateResult(
            firing_rate=uniform_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        mvl = result.mean_vector_length()
        # Sharply tuned should have high MVL (typically > 0.5)
        assert mvl > 0.5


class TestDirectionalRateResultTuningWidth:
    """Test DirectionalRateResult.tuning_width() method."""

    def test_tuning_width_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_width() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        width = result.tuning_width()
        assert isinstance(width, float)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        # Broadly tuned (kappa=1)
        broad_rate = 10.0 * np.exp(1.0 * (np.cos(bin_centers - preferred_dir) - 1))
        broad_result = DirectionalRateResult(
            firing_rate=broad_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        width = result.tuning_width()
        # Uniform firing has no peak, so HWHM is undefined
        assert np.isnan(width)


class TestDirectionalRateResultTuningWidthDeg:
    """Test DirectionalRateResult.tuning_width_deg() method."""

    def test_tuning_width_deg_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """tuning_width_deg() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        width_deg = result.tuning_width_deg()
        assert isinstance(width_deg, float)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        width_deg = result.tuning_width_deg()
        # Should be at most 180 degrees
        if not np.isnan(width_deg):
            assert 0 < width_deg <= 180


class TestDirectionalRateResultRayleighPvalue:
    """Test DirectionalRateResult.rayleigh_pvalue() method."""

    def test_rayleigh_pvalue_returns_float(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """rayleigh_pvalue() returns a float."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pval = result.rayleigh_pvalue()
        assert isinstance(pval, float)

    def test_rayleigh_pvalue_in_valid_range(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """rayleigh_pvalue() returns value in [0, 1]."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pval = result.rayleigh_pvalue()
        assert 0 <= pval <= 1

    def test_rayleigh_pvalue_tuned_neuron_low(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """rayleigh_pvalue() is low for sharply tuned neurons."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Sharply tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pval = result.rayleigh_pvalue()
        # Tuned neuron should have low p-value (significant non-uniformity)
        assert pval < 0.05

    def test_rayleigh_pvalue_uniform_high(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """rayleigh_pvalue() is high for uniform firing."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pval = result.rayleigh_pvalue()
        # Uniform firing should have high p-value (non-significant)
        assert pval > 0.5


# ==============================================================================
# DirectionalRateResult Classification Methods Tests - Task 3.4
# ==============================================================================


class TestDirectionalRateResultIsHdCell:
    """Test DirectionalRateResult.is_hd_cell() method."""

    def test_is_hd_cell_returns_bool(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_hd_cell() returns a boolean."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        is_hd = result.is_hd_cell()
        assert isinstance(is_hd, bool)

    def test_is_hd_cell_true_for_sharply_tuned(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_hd_cell() returns True for sharply tuned neurons."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Very sharply tuned neuron (kappa=5)
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        # Should be classified as HD cell (high MVL, low p-value)
        assert result.is_hd_cell() is True

    def test_is_hd_cell_false_for_uniform(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """is_hd_cell() returns False for uniform firing."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Uniform firing rate
        firing_rate = np.ones(n_bins) * 5.0

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        # Should NOT be classified as HD cell
        assert result.is_hd_cell() is False

    def test_is_hd_cell_uses_min_mvl_threshold(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_hd_cell() respects min_mvl parameter."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Moderately tuned neuron (kappa=2 gives MVL around 0.3-0.4)
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(2.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        mvl = result.mean_vector_length()

        # With high threshold (above MVL), should be False
        assert result.is_hd_cell(min_mvl=mvl + 0.1) is False

        # With low threshold (below MVL), should be True
        assert result.is_hd_cell(min_mvl=mvl - 0.1) is True

    def test_is_hd_cell_uses_alpha_threshold(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_hd_cell() respects alpha parameter."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Sharply tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        pval = result.rayleigh_pvalue()

        # With very strict alpha (below p-value), should be False
        # Note: sharply tuned will have very low p-value, so use even lower alpha
        assert result.is_hd_cell(alpha=pval / 10) is False

        # With permissive alpha (above p-value), should be True
        assert result.is_hd_cell(alpha=0.1) is True

    def test_is_hd_cell_default_thresholds(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_hd_cell() uses default thresholds of min_mvl=0.4, alpha=0.05."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        # Calling with no arguments should use defaults
        is_hd_default = result.is_hd_cell()
        is_hd_explicit = result.is_hd_cell(min_mvl=0.4, alpha=0.05)
        assert is_hd_default == is_hd_explicit

    def test_is_hd_cell_requires_both_criteria(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """is_hd_cell() requires BOTH MVL > threshold AND p-value < alpha."""
        from neurospatial.encoding.directional import DirectionalRateResult

        # Create tuned neuron
        preferred_dir = np.pi / 2
        firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - preferred_dir) - 1))

        result = DirectionalRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        mvl = result.mean_vector_length()
        pval = result.rayleigh_pvalue()

        # Set thresholds so only MVL criterion is met
        # (high alpha threshold that is below the p-value)
        if pval > 0:
            assert result.is_hd_cell(min_mvl=mvl - 0.1, alpha=pval / 10) is False

        # Set thresholds so only p-value criterion is met
        # (high MVL threshold that is above the actual MVL)
        assert result.is_hd_cell(min_mvl=mvl + 0.1, alpha=0.1) is False


class TestDirectionalRateResultInterpretation:
    """Test DirectionalRateResult.interpretation() method."""

    def test_interpretation_returns_string(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """interpretation() returns a string."""
        from neurospatial.encoding.directional import DirectionalRateResult

        result = DirectionalRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        interp = result.interpretation()
        assert isinstance(interp, str)

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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        custom_threshold = 0.5
        interp = result.interpretation(min_mvl=custom_threshold)

        # Should mention the threshold somewhere
        assert str(custom_threshold) in interp or "0.5" in interp


# ==============================================================================
# DirectionalRatesResult Batch Methods Tests - Task 3.5
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
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
            smoothing_sigma=None,
        )

        widths = result.tuning_widths()
        # Filter out NaN values for range check
        valid_widths = widths[~np.isnan(widths)]
        if len(valid_widths) > 0:
            assert np.all(valid_widths > 0)
            assert np.all(valid_widths <= np.pi)


class TestDirectionalRatesResultDetectHdCells:
    """Test DirectionalRatesResult.detect_hd_cells() method."""

    def test_detect_hd_cells_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """detect_hd_cells() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        is_hd = result.detect_hd_cells()
        assert isinstance(is_hd, np.ndarray)

    def test_detect_hd_cells_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """detect_hd_cells() returns (n_neurons,) bool array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        is_hd = result.detect_hd_cells()
        assert is_hd.shape == (n_neurons,)
        assert is_hd.dtype == np.bool_

    def test_detect_hd_cells_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """detect_hd_cells() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        is_hd = result.detect_hd_cells()

        for i, single in enumerate(result):
            assert is_hd[i] == single.is_hd_cell()

    def test_detect_hd_cells_respects_min_mvl(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """detect_hd_cells() respects min_mvl parameter."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        # Very permissive threshold
        is_hd_permissive = result.detect_hd_cells(min_mvl=0.0)

        # Very strict threshold
        is_hd_strict = result.detect_hd_cells(min_mvl=0.99)

        # Strict should have fewer or equal HD cells
        assert np.sum(is_hd_strict) <= np.sum(is_hd_permissive)

    def test_detect_hd_cells_respects_alpha(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """detect_hd_cells() respects alpha parameter."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        # Very permissive alpha
        is_hd_permissive = result.detect_hd_cells(alpha=1.0)

        # Very strict alpha
        is_hd_strict = result.detect_hd_cells(alpha=1e-10)

        # Strict should have fewer or equal HD cells
        assert np.sum(is_hd_strict) <= np.sum(is_hd_permissive)

    def test_detect_hd_cells_default_thresholds(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """detect_hd_cells() uses default thresholds of min_mvl=0.4, alpha=0.05."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        is_hd_default = result.detect_hd_cells()
        is_hd_explicit = result.detect_hd_cells(min_mvl=0.4, alpha=0.05)

        np.testing.assert_array_equal(is_hd_default, is_hd_explicit)


class TestDirectionalRatesResultPeakFiringRates:
    """Test DirectionalRatesResult.peak_firing_rates() method."""

    def test_peak_firing_rates_returns_array(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rates() returns numpy array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        peaks = result.peak_firing_rates()
        assert isinstance(peaks, np.ndarray)

    def test_peak_firing_rates_shape(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """peak_firing_rates() returns (n_neurons,) array."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        peaks = result.peak_firing_rates()
        assert peaks.shape == (n_neurons,)

    def test_peak_firing_rates_matches_single(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rates() matches iteration over single results."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        peaks = result.peak_firing_rates()

        for i, single in enumerate(result):
            np.testing.assert_allclose(peaks[i], single.peak_firing_rate())

    def test_peak_firing_rates_nonnegative(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """peak_firing_rates() values are non-negative."""
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        peaks = result.peak_firing_rates()
        assert np.all(peaks >= 0)


# ==============================================================================
# DirectionalRatesResult.to_dataframe() Tests - Task 3.6
# ==============================================================================


class TestDirectionalRatesResultToDataframe:
    """Test DirectionalRatesResult.to_dataframe() method."""

    def test_to_dataframe_returns_dataframe(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() returns a pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_to_dataframe_has_correct_row_count(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """to_dataframe() returns one row per neuron."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert len(df) == n_neurons

    def test_to_dataframe_has_neuron_id_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes neuron_id column."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "neuron_id" in df.columns

    def test_to_dataframe_has_preferred_direction_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes preferred_direction column (radians)."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "preferred_direction" in df.columns

    def test_to_dataframe_has_preferred_direction_deg_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes preferred_direction_deg column."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "preferred_direction_deg" in df.columns

    def test_to_dataframe_has_mean_vector_length_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes mean_vector_length column."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "mean_vector_length" in df.columns

    def test_to_dataframe_has_tuning_width_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes tuning_width column (radians)."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "tuning_width" in df.columns

    def test_to_dataframe_has_tuning_width_deg_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes tuning_width_deg column."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "tuning_width_deg" in df.columns

    def test_to_dataframe_has_peak_rate_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes peak_rate column."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "peak_rate" in df.columns

    def test_to_dataframe_has_is_hd_cell_column(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() includes is_hd_cell column."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert "is_hd_cell" in df.columns

    def test_to_dataframe_neuron_id_default_integers(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """to_dataframe() uses integer indices as default neuron_id."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        expected_ids = list(range(n_neurons))
        assert list(df["neuron_id"]) == expected_ids

    def test_to_dataframe_custom_neuron_ids(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_neurons: int,
    ) -> None:
        """to_dataframe() accepts custom neuron_ids."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        custom_ids = [f"unit_{i}" for i in range(n_neurons)]
        df = result.to_dataframe(neuron_ids=custom_ids)
        assert list(df["neuron_id"]) == custom_ids

    def test_to_dataframe_neuron_id_length_mismatch(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() raises ValueError for wrong neuron_ids length."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        # Wrong number of neuron_ids
        with pytest.raises(ValueError, match="neuron_ids"):
            result.to_dataframe(neuron_ids=["only_one"])

    def test_to_dataframe_preferred_direction_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() preferred_direction matches preferred_directions()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        pref_dirs = result.preferred_directions()
        np.testing.assert_allclose(df["preferred_direction"].values, pref_dirs)

    def test_to_dataframe_preferred_direction_deg_is_conversion(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() preferred_direction_deg is degrees conversion."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        np.testing.assert_allclose(
            df["preferred_direction_deg"].values,
            np.degrees(df["preferred_direction"].values),
        )

    def test_to_dataframe_mean_vector_length_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() mean_vector_length matches mean_vector_lengths()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        mvls = result.mean_vector_lengths()
        np.testing.assert_allclose(df["mean_vector_length"].values, mvls)

    def test_to_dataframe_tuning_width_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() tuning_width matches tuning_widths()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        widths = result.tuning_widths()
        np.testing.assert_allclose(df["tuning_width"].values, widths, equal_nan=True)

    def test_to_dataframe_tuning_width_deg_is_conversion(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() tuning_width_deg is degrees conversion."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        np.testing.assert_allclose(
            df["tuning_width_deg"].values,
            np.degrees(df["tuning_width"].values),
            equal_nan=True,
        )

    def test_to_dataframe_peak_rate_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() peak_rate matches peak_firing_rates()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        peaks = result.peak_firing_rates()
        np.testing.assert_allclose(df["peak_rate"].values, peaks)

    def test_to_dataframe_is_hd_cell_matches_method(
        self,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() is_hd_cell matches detect_hd_cells()."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        result = DirectionalRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        is_hd = result.detect_hd_cells()
        np.testing.assert_array_equal(df["is_hd_cell"].values, is_hd)

    def test_to_dataframe_empty_result(
        self,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
        n_bins: int,
    ) -> None:
        """to_dataframe() works with zero neurons."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        # Empty result (0 neurons)
        rates = np.empty((0, n_bins), dtype=np.float64)

        result = DirectionalRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_single_neuron(
        self,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
        bin_centers: np.ndarray,
        bin_size: float,
    ) -> None:
        """to_dataframe() works with single neuron."""
        pytest.importorskip("pandas")
        from neurospatial.encoding.directional import DirectionalRatesResult

        # Single neuron
        rates = single_firing_rate.reshape(1, -1)

        result = DirectionalRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            bin_centers=bin_centers,
            bin_size=bin_size,
            smoothing_sigma=None,
        )

        df = result.to_dataframe()
        assert len(df) == 1


# ==============================================================================
# compute_directional_rate() Tests - Task 3.8
# ==============================================================================


class TestComputeDirectionalRateImport:
    """Test that compute_directional_rate can be imported."""

    def test_import_from_directional(self) -> None:
        """compute_directional_rate can be imported from encoding.directional."""
        from neurospatial.encoding.directional import compute_directional_rate

        assert compute_directional_rate is not None

    def test_import_from_encoding(self) -> None:
        """compute_directional_rate can be imported from encoding package."""
        from neurospatial.encoding import compute_directional_rate

        assert compute_directional_rate is not None


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

    def test_firing_rate_shape(self) -> None:
        """compute_directional_rate produces firing_rate with correct shape."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        # Default bin_size is π/30 = 60 bins
        n_bins = 60
        assert np.asarray(result.firing_rate).shape == (n_bins,)

    def test_occupancy_shape(self) -> None:
        """compute_directional_rate produces occupancy with correct shape."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        n_bins = 60
        assert np.asarray(result.occupancy).shape == (n_bins,)

    def test_bin_centers_shape(self) -> None:
        """compute_directional_rate produces bin_centers with correct shape."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        n_bins = 60
        assert np.asarray(result.bin_centers).shape == (n_bins,)

    def test_bin_centers_in_radians(self) -> None:
        """compute_directional_rate produces bin_centers in radians [0, 2π)."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        centers = np.asarray(result.bin_centers)
        assert np.all(centers >= 0)
        assert np.all(centers < 2 * np.pi)

    def test_metadata_defaults(self) -> None:
        """compute_directional_rate stores correct default metadata."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)

        assert result.bin_size == np.pi / 30
        assert result.smoothing_sigma is None

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
    """Test compute_directional_rate smoothing_sigma parameter."""

    def test_smoothing_sigma_none_default(self) -> None:
        """compute_directional_rate returns None smoothing_sigma by default."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result = compute_directional_rate(spike_times, times, headings)
        assert result.smoothing_sigma is None

    def test_smoothing_sigma_stored(self) -> None:
        """compute_directional_rate stores smoothing_sigma in result."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        smoothing_sigma = np.pi / 6  # 30 degrees
        result = compute_directional_rate(
            spike_times, times, headings, smoothing_sigma=smoothing_sigma
        )
        assert result.smoothing_sigma == smoothing_sigma

    def test_smoothing_produces_smoother_curve(self) -> None:
        """compute_directional_rate with smoothing produces smoother curves."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5])

        result_raw = compute_directional_rate(spike_times, times, headings)
        result_smooth = compute_directional_rate(
            spike_times, times, headings, smoothing_sigma=np.pi / 6
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

    def test_smoothing_sigma_degrees(self) -> None:
        """compute_directional_rate smoothing_sigma with angle_unit=deg."""
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
            smoothing_sigma=30.0,
            angle_unit="deg",
        )

        # smoothing_sigma is stored in radians
        np.testing.assert_allclose(result.smoothing_sigma, np.radians(30.0))


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

        # Should have valid result (only 2 spikes counted: 2.0 and 5.0)
        rates = np.asarray(result.firing_rate)
        # Total spike count should be approximately 2 (not 4)
        total_rate = np.nansum(rates * result.occupancy)
        # Total spikes = integral of rate * occupancy
        # With only 2 spikes included, the integral should be roughly 2
        assert 1 <= total_rate <= 3  # Allow some tolerance

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
        """Result has working is_hd_cell method."""
        from neurospatial.encoding.directional import compute_directional_rate

        np.random.seed(42)
        # Use dense trajectory that covers all directions
        times = np.linspace(0, 60, 3600)  # 60 seconds at 60 Hz
        headings = np.linspace(0, 2 * np.pi, 3600) % (2 * np.pi)
        spike_times = np.array([1.0, 2.5, 4.0, 7.5, 15.0, 30.0])

        result = compute_directional_rate(spike_times, times, headings)

        is_hd = result.is_hd_cell()
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
