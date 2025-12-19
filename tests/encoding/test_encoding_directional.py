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
