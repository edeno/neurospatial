"""Tests for view rate computation result classes.

This module tests ViewRateResult and ViewRatesResult dataclasses
defined in encoding/view.py for spatial view cell analysis.

Task 4.1: Result class definitions
- ViewRateResult dataclass (frozen=True)
- ViewRatesResult dataclass (frozen=True)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment

# ==============================================================================
# Test Fixtures
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
def n_neurons() -> int:
    """Number of neurons for batch tests."""
    return 5


@pytest.fixture
def single_firing_rate(simple_env: Environment) -> np.ndarray:
    """Firing rate for a single neuron with a clear peak."""
    n_bins = simple_env.n_bins
    firing_rate = np.zeros(n_bins, dtype=np.float64)
    # Create a peak near the center
    peak_bin = n_bins // 2
    firing_rate[peak_bin] = 20.0
    firing_rate[peak_bin - 1] = 10.0
    firing_rate[peak_bin + 1] = 10.0
    return firing_rate


@pytest.fixture
def single_view_occupancy(simple_env: Environment) -> np.ndarray:
    """View occupancy for a single neuron (uniform viewing)."""
    return np.ones(simple_env.n_bins, dtype=np.float64) * 0.5  # 0.5 seconds per bin


@pytest.fixture
def batch_firing_rates(simple_env: Environment, n_neurons: int) -> np.ndarray:
    """Firing rates for multiple neurons with different peak locations."""
    n_bins = simple_env.n_bins
    rates = np.zeros((n_neurons, n_bins), dtype=np.float64)

    for i in range(n_neurons):
        # Each neuron has a different peak location
        peak_bin = min(i * 4 + 2, n_bins - 1)
        rates[i, peak_bin] = 15.0 + i * 5.0  # Different peak rates

    return rates


# ==============================================================================
# ViewRateResult Tests - Task 4.1
# ==============================================================================


class TestViewRateResultImport:
    """Test that ViewRateResult can be imported."""

    def test_import_from_view(self) -> None:
        """ViewRateResult can be imported from encoding.view."""
        from neurospatial.encoding.view import ViewRateResult

        assert ViewRateResult is not None

    def test_import_from_encoding(self) -> None:
        """ViewRateResult can be imported from encoding package."""
        from neurospatial.encoding import ViewRateResult

        assert ViewRateResult is not None


class TestViewRateResultCreation:
    """Test ViewRateResult dataclass creation."""

    def test_create_result(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """ViewRateResult can be created with all required fields."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert result is not None
        assert result.firing_rate is not None
        assert result.view_occupancy is not None
        assert result.env is simple_env
        assert result.gaze_model == "fixed_distance"
        assert result.view_distance == 10.0
        assert result.smoothing_method == "diffusion_kde"
        assert result.bandwidth == 5.0

    def test_is_frozen_dataclass(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """ViewRateResult should be immutable (frozen=True)."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.firing_rate = np.zeros_like(single_firing_rate)  # type: ignore[misc]


class TestViewRateResultFields:
    """Test ViewRateResult field attributes."""

    def test_firing_rate_shape(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """firing_rate should have shape (n_bins,)."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.firing_rate).shape == (simple_env.n_bins,)

    def test_view_occupancy_shape(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.view_occupancy).shape == (simple_env.n_bins,)

    def test_required_fields_exist(self) -> None:
        """ViewRateResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.view import ViewRateResult

        field_names = {f.name for f in fields(ViewRateResult)}
        expected_fields = {
            "firing_rate",
            "view_occupancy",
            "env",
            "gaze_model",
            "view_distance",
            "smoothing_method",
            "bandwidth",
        }
        assert expected_fields.issubset(field_names)


# ==============================================================================
# ViewRatesResult Tests - Task 4.1
# ==============================================================================


class TestViewRatesResultImport:
    """Test that ViewRatesResult can be imported."""

    def test_import_from_view(self) -> None:
        """ViewRatesResult can be imported from encoding.view."""
        from neurospatial.encoding.view import ViewRatesResult

        assert ViewRatesResult is not None

    def test_import_from_encoding(self) -> None:
        """ViewRatesResult can be imported from encoding package."""
        from neurospatial.encoding import ViewRatesResult

        assert ViewRatesResult is not None


class TestViewRatesResultCreation:
    """Test ViewRatesResult dataclass creation."""

    def test_create_result(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult can be created with all required fields."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert result is not None
        assert result.firing_rates is not None
        assert result.view_occupancy is not None
        assert result.env is simple_env
        assert result.gaze_model == "fixed_distance"
        assert result.view_distance == 10.0
        assert result.smoothing_method == "diffusion_kde"
        assert result.bandwidth == 5.0

    def test_is_frozen_dataclass(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult should be immutable (frozen=True)."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.firing_rates = np.zeros_like(batch_firing_rates)  # type: ignore[misc]


class TestViewRatesResultFields:
    """Test ViewRatesResult field attributes."""

    def test_firing_rates_shape(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """firing_rates should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.firing_rates).shape == (n_neurons, simple_env.n_bins)

    def test_view_occupancy_shape(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_occupancy should have shape (n_bins,) - shared across neurons."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.view_occupancy).shape == (simple_env.n_bins,)

    def test_required_fields_exist(self) -> None:
        """ViewRatesResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.view import ViewRatesResult

        field_names = {f.name for f in fields(ViewRatesResult)}
        expected_fields = {
            "firing_rates",
            "view_occupancy",
            "env",
            "gaze_model",
            "view_distance",
            "smoothing_method",
            "bandwidth",
        }
        assert expected_fields.issubset(field_names)


class TestViewRatesResultIteration:
    """Test ViewRatesResult iteration interface."""

    def test_len_returns_n_neurons(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """len(result) should return the number of neurons."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert len(result) == n_neurons

    def test_getitem_returns_single_result(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """result[idx] should return a ViewRateResult for that neuron."""
        from neurospatial.encoding.view import ViewRateResult, ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        single = result[0]
        assert isinstance(single, ViewRateResult)

    def test_getitem_preserves_firing_rate(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """result[idx].firing_rate should match firing_rates[idx]."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        for i in range(len(result)):
            single = result[i]
            assert_array_equal(single.firing_rate, batch_firing_rates[i])

    def test_getitem_shares_metadata(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """result[idx] should share metadata with parent."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        single = result[0]
        assert single.env is result.env
        assert single.gaze_model == result.gaze_model
        assert single.view_distance == result.view_distance
        assert single.smoothing_method == result.smoothing_method
        assert single.bandwidth == result.bandwidth
        assert_array_equal(single.view_occupancy, result.view_occupancy)

    def test_iter_yields_all_neurons(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """Iterating over result should yield all neurons."""
        from neurospatial.encoding.view import ViewRateResult, ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        neurons = list(result)
        assert len(neurons) == n_neurons
        for single in neurons:
            assert isinstance(single, ViewRateResult)

    def test_iter_order_matches_indexing(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """Iteration order should match indexing order."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        for i, single in enumerate(result):
            assert_array_equal(single.firing_rate, result[i].firing_rate)


class TestViewRatesResultEdgeCases:
    """Test ViewRatesResult edge cases."""

    def test_single_neuron(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult should work with a single neuron."""
        from neurospatial.encoding.view import ViewRatesResult

        # Reshape single neuron to batch format
        batch_rates = single_firing_rate.reshape(1, -1)

        result = ViewRatesResult(
            firing_rates=batch_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert len(result) == 1
        single = result[0]
        assert_array_equal(single.firing_rate, single_firing_rate)

    def test_empty_firing_rates(
        self,
        simple_env: Environment,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult should handle empty firing rates (zero neurons)."""
        from neurospatial.encoding.view import ViewRatesResult

        # Create empty batch
        empty_rates = np.zeros((0, simple_env.n_bins), dtype=np.float64)

        result = ViewRatesResult(
            firing_rates=empty_rates,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert len(result) == 0
        assert list(result) == []


# ==============================================================================
# ViewRateResult Convenience Methods Tests - Task 4.2
# ==============================================================================


class TestViewRateResultPlot:
    """Test ViewRateResult.plot() method."""

    def test_plot_returns_axes(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """plot() should return matplotlib Axes."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing
        import matplotlib.pyplot as plt

        ax = result.plot()
        assert ax is not None
        plt.close()

    def test_plot_accepts_ax_argument(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """plot() should accept existing axes."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, ax_in = plt.subplots()
        ax_out = result.plot(ax=ax_in)
        assert ax_out is ax_in
        plt.close()

    def test_plot_accepts_kwargs(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """plot() should pass through kwargs to env.plot_field()."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Should not raise - kwargs are passed through
        ax = result.plot(cmap="hot", vmax=25.0)
        assert ax is not None
        plt.close()


class TestViewRateResultPeakViewLocation:
    """Test ViewRateResult.peak_view_location() method."""

    def test_peak_view_location_returns_ndarray(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should return ndarray."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_view_location()
        assert isinstance(peak, np.ndarray)

    def test_peak_view_location_shape(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should return (n_dims,) array."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_view_location()
        # 2D environment should have 2 dimensions
        assert peak.shape == (2,)

    def test_peak_view_location_at_max_firing(
        self,
        simple_env: Environment,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should return location of maximum firing rate."""
        from neurospatial.encoding.view import ViewRateResult

        # Create firing rate with known peak at a specific bin
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        peak_bin = 5  # Known peak bin
        firing_rate[peak_bin] = 100.0

        result = ViewRateResult(
            firing_rate=firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_view_location()
        expected = simple_env.bin_centers[peak_bin]
        assert_array_equal(peak, expected)

    def test_peak_view_location_handles_nan(
        self,
        simple_env: Environment,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should handle NaN values correctly."""
        from neurospatial.encoding.view import ViewRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.full(n_bins, np.nan, dtype=np.float64)
        peak_bin = 3
        firing_rate[peak_bin] = 50.0  # Only non-NaN value

        result = ViewRateResult(
            firing_rate=firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_view_location()
        expected = simple_env.bin_centers[peak_bin]
        assert_array_equal(peak, expected)


class TestViewRateResultViewSpatialInformation:
    """Test ViewRateResult.view_spatial_information() method."""

    def test_view_spatial_information_returns_float(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should return float."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert isinstance(info, float)

    def test_view_spatial_information_non_negative(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should be non-negative."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert info >= 0.0

    def test_view_spatial_information_uses_view_occupancy(
        self,
        simple_env: Environment,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should use view_occupancy, not standard occupancy."""
        from neurospatial.encoding.view import ViewRateResult

        # Create a peaked firing rate
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 20.0

        # Create non-uniform view occupancy that emphasizes the peak
        view_occupancy = np.ones(n_bins, dtype=np.float64) * 0.1
        view_occupancy[n_bins // 2] = 1.0  # More time viewing the peak

        result = ViewRateResult(
            firing_rate=firing_rate,
            view_occupancy=view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        # Should produce meaningful spatial information
        assert info >= 0.0

    def test_view_spatial_information_uniform_firing_is_zero(
        self,
        simple_env: Environment,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should return 0 for uniform firing."""
        from neurospatial.encoding.view import ViewRateResult

        # Uniform firing rate across all bins
        firing_rate = np.ones(simple_env.n_bins, dtype=np.float64) * 5.0

        result = ViewRateResult(
            firing_rate=firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        # Uniform firing should have zero spatial information
        assert info == pytest.approx(0.0, abs=1e-10)

    def test_view_spatial_information_peaked_is_positive(
        self,
        simple_env: Environment,
        single_view_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should be positive for peaked firing."""
        from neurospatial.encoding.view import ViewRateResult

        # Peaked firing rate
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 50.0  # Strong peak

        result = ViewRateResult(
            firing_rate=firing_rate,
            view_occupancy=single_view_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert info > 0.0
