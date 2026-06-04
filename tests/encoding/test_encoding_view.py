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
def single_occupancy(simple_env: Environment) -> np.ndarray:
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


class TestViewRateResultCreation:
    """Test ViewRateResult dataclass creation."""

    def test_create_result(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """ViewRateResult can be created with all required fields."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert result is not None
        assert result.firing_rate is not None
        assert result.occupancy is not None
        assert result.env is simple_env
        assert result.gaze_model == "fixed_distance"
        assert result.view_distance == 10.0
        assert result.smoothing_method == "diffusion_kde"
        assert result.bandwidth == 5.0

    def test_is_frozen_dataclass(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """ViewRateResult should be immutable (frozen=True)."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """firing_rate should have shape (n_bins,)."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.firing_rate).shape == (simple_env.n_bins,)

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.occupancy).shape == (simple_env.n_bins,)

    def test_required_fields_exist(self) -> None:
        """ViewRateResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.view import ViewRateResult

        field_names = {f.name for f in fields(ViewRateResult)}
        expected_fields = {
            "firing_rate",
            "occupancy",
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


class TestViewRatesResultCreation:
    """Test ViewRatesResult dataclass creation."""

    def test_create_result(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult can be created with all required fields."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert result is not None
        assert result.firing_rates is not None
        assert result.occupancy is not None
        assert result.env is simple_env
        assert result.gaze_model == "fixed_distance"
        assert result.view_distance == 10.0
        assert result.smoothing_method == "diffusion_kde"
        assert result.bandwidth == 5.0

    def test_is_frozen_dataclass(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult should be immutable (frozen=True)."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """firing_rates should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.firing_rates).shape == (n_neurons, simple_env.n_bins)

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """occupancy should have shape (n_bins,) - shared across neurons."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        assert np.asarray(result.occupancy).shape == (simple_env.n_bins,)

    def test_required_fields_exist(self) -> None:
        """ViewRatesResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.view import ViewRatesResult

        field_names = {f.name for f in fields(ViewRatesResult)}
        expected_fields = {
            "firing_rates",
            "occupancy",
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
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """len(result) should return the number of neurons."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """result[idx] should return a ViewRateResult for that neuron."""
        from neurospatial.encoding.view import ViewRateResult, ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """result[idx].firing_rate should match firing_rates[idx]."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """result[idx] should share metadata with parent."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        assert_array_equal(single.occupancy, result.occupancy)

    def test_iter_yields_all_neurons(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """Iterating over result should yield all neurons."""
        from neurospatial.encoding.view import ViewRateResult, ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """Iteration order should match indexing order."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult should work with a single neuron."""
        from neurospatial.encoding.view import ViewRatesResult

        # Reshape single neuron to batch format
        batch_rates = single_firing_rate.reshape(1, -1)

        result = ViewRatesResult(
            firing_rates=batch_rates,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """ViewRatesResult should handle empty firing rates (zero neurons)."""
        from neurospatial.encoding.view import ViewRatesResult

        # Create empty batch
        empty_rates = np.zeros((0, simple_env.n_bins), dtype=np.float64)

        result = ViewRatesResult(
            firing_rates=empty_rates,
            occupancy=single_occupancy,
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
    """Test ViewRateResult.plot() method.

    Only the ax-passthrough test is kept. The previous file had three
    tests in this class — ``test_plot_returns_axes``, this one, and
    ``test_plot_accepts_kwargs`` — and all three ended with
    ``assert ax is not None`` after a call that has a non-Optional
    return type. ``test_plot_accepts_ax_argument`` is the only one
    that pins meaningful behavior (the user-supplied ax is reused).
    """

    def test_plot_accepts_ax_argument(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """plot() should reuse the provided axes."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
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


class TestViewRateResultPeakViewLocation:
    """Test ViewRateResult.peak_view_location() method."""

    def test_peak_view_location_shape(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should return (n_dims,) array."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_location()
        # 2D environment should have 2 dimensions
        assert peak.shape == (2,)

    def test_peak_view_location_at_max_firing(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
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
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_location()
        expected = simple_env.bin_centers[peak_bin]
        assert_array_equal(peak, expected)

    def test_peak_view_location_handles_nan(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should handle NaN values correctly."""
        from neurospatial.encoding.view import ViewRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.full(n_bins, np.nan, dtype=np.float64)
        peak_bin = 3
        firing_rate[peak_bin] = 50.0  # Only non-NaN value

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peak = result.peak_location()
        expected = simple_env.bin_centers[peak_bin]
        assert_array_equal(peak, expected)


class TestViewRateResultViewSpatialInformation:
    """Test ViewRateResult.view_spatial_information() method."""

    def test_view_spatial_information_returns_float(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should return float."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should be non-negative."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert info >= 0.0

    def test_view_spatial_information_uses_occupancy(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should use occupancy, not standard occupancy."""
        from neurospatial.encoding.view import ViewRateResult

        # Create a peaked firing rate
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 20.0

        # Create non-uniform view occupancy that emphasizes the peak
        occupancy = np.ones(n_bins, dtype=np.float64) * 0.1
        occupancy[n_bins // 2] = 1.0  # More time viewing the peak

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should return 0 for uniform firing."""
        from neurospatial.encoding.view import ViewRateResult

        # Uniform firing rate across all bins
        firing_rate = np.ones(simple_env.n_bins, dtype=np.float64) * 5.0

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
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
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should be positive for peaked firing."""
        from neurospatial.encoding.view import ViewRateResult

        # Peaked firing rate
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 50.0  # Strong peak

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert info > 0.0


# ==============================================================================
# ViewRateResult Classification Tests - Task 4.3
# ==============================================================================


class TestViewRateResultIsViewCell:
    """Test ViewRateResult.is_spatial_view_cell() method."""

    def test_is_view_cell_returns_bool(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """is_spatial_view_cell() should return a boolean."""
        from neurospatial.encoding.view import ViewRateResult

        result = ViewRateResult(
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        classification = result.is_spatial_view_cell()
        assert isinstance(classification, bool)

    def test_is_view_cell_true_for_high_info(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """is_spatial_view_cell() should return True for neurons with high view spatial info."""
        from neurospatial.encoding.view import ViewRateResult

        # Create a sharply peaked firing rate (high spatial info)
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 50.0  # Strong peak

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # With a sharp peak, spatial info should be high
        # Use a low threshold to ensure this passes
        assert result.is_spatial_view_cell(min_info=0.1) is True

    def test_is_view_cell_false_for_uniform_firing(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """is_spatial_view_cell() should return False for uniform firing (zero info)."""
        from neurospatial.encoding.view import ViewRateResult

        # Uniform firing rate
        firing_rate = np.ones(simple_env.n_bins, dtype=np.float64) * 5.0

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Uniform firing has zero spatial information, should be False
        assert result.is_spatial_view_cell(min_info=0.1) is False

    def test_is_view_cell_respects_min_info_parameter(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """is_spatial_view_cell() should respect the min_info threshold parameter."""
        from neurospatial.encoding.view import ViewRateResult

        # Create moderately peaked firing rate
        n_bins = simple_env.n_bins
        firing_rate = np.ones(n_bins, dtype=np.float64) * 2.0
        firing_rate[n_bins // 2] = 10.0  # Moderate peak

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()

        # With very low threshold, should be True
        assert result.is_spatial_view_cell(min_info=0.0) is True

        # With threshold higher than actual info, should be False
        assert result.is_spatial_view_cell(min_info=info + 10.0) is False

    def test_is_view_cell_default_threshold(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """is_spatial_view_cell() should use default min_info=0.5."""
        from neurospatial.encoding.view import ViewRateResult

        # Uniform firing with zero info
        firing_rate = np.ones(simple_env.n_bins, dtype=np.float64) * 5.0

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # With default threshold (0.5), uniform firing should be False
        assert result.is_spatial_view_cell() is False

    def test_is_view_cell_uses_view_spatial_information(
        self,
        simple_env: Environment,
    ) -> None:
        """is_spatial_view_cell() should use view_spatial_information() for classification."""
        from neurospatial.encoding.view import ViewRateResult

        # Create peaked firing rate with custom view occupancy
        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        firing_rate[n_bins // 2] = 30.0

        # Custom view occupancy that emphasizes the peak
        occupancy = np.ones(n_bins, dtype=np.float64)

        result = ViewRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Get actual info value
        info = result.view_spatial_information()

        # Classification should be consistent with the threshold
        assert result.is_spatial_view_cell(min_info=info - 0.01) is True
        assert result.is_spatial_view_cell(min_info=info + 0.01) is False


# ==============================================================================
# ViewRatesResult Batch Methods Tests - Task 4.4
# ==============================================================================


class TestViewRatesResultPlot:
    """Test ViewRatesResult.plot() method.

    Same trim as ``TestViewRateResultPlot`` plus one regression: ``idx``
    must be passed (the batch result has no single neuron to plot
    without it). The ``test_plot_returns_axes`` and
    ``test_plot_accepts_kwargs`` variants only asserted
    ``assert ax is not None``.
    """

    def test_plot_requires_idx(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """plot() should require an explicit ``idx`` argument."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        with pytest.raises(TypeError):
            result.plot()  # type: ignore[call-arg]

    def test_plot_accepts_ax_argument(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """plot() should reuse the provided axes."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
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
        ax_out = result.plot(idx=0, ax=ax_in)
        assert ax_out is ax_in
        plt.close()


class TestViewRatesResultPeakViewLocations:
    """Test ViewRatesResult.peak_view_location() method."""

    def test_peak_view_location_shape(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """peak_view_location() should return (n_neurons, n_dims) array."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peaks = result.peak_locations()
        # 2D environment with n_neurons neurons
        assert peaks.shape == (n_neurons, 2)

    def test_peak_view_location_matches_single_neuron(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """peak_view_location() should match single-neuron peak_view_location()."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        batch_peaks = result.peak_locations()

        # Verify each neuron's peak matches single-neuron result
        for i in range(n_neurons):
            single = result[i]
            single_peak = single.peak_location()
            assert_array_equal(batch_peaks[i], single_peak)

    def test_peak_view_location_handles_nan(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should handle NaN values correctly."""
        from neurospatial.encoding.view import ViewRatesResult

        n_bins = simple_env.n_bins
        # Create rates with NaN, but one non-NaN peak
        rates = np.full((2, n_bins), np.nan, dtype=np.float64)
        rates[0, 3] = 50.0  # Peak at bin 3
        rates[1, 7] = 30.0  # Peak at bin 7

        result = ViewRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peaks = result.peak_locations()
        assert peaks.shape == (2, 2)
        assert_array_equal(peaks[0], simple_env.bin_centers[3])
        assert_array_equal(peaks[1], simple_env.bin_centers[7])

    def test_peak_view_location_all_nan(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """peak_view_location() should return NaN for neurons with all-NaN rates."""
        from neurospatial.encoding.view import ViewRatesResult

        n_bins = simple_env.n_bins
        # Create rates where one neuron has all NaN, one has valid data
        rates = np.full((2, n_bins), np.nan, dtype=np.float64)
        rates[1, 5] = 30.0  # Only second neuron has valid data

        result = ViewRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        peaks = result.peak_locations()
        assert peaks.shape == (2, 2)
        # First neuron: all NaN -> NaN coordinates
        assert np.all(np.isnan(peaks[0]))
        # Second neuron: has valid data -> valid coordinates
        assert_array_equal(peaks[1], simple_env.bin_centers[5])


class TestViewRatesResultViewSpatialInformation:
    """Test ViewRatesResult.view_spatial_information() method."""

    def test_view_spatial_information_returns_ndarray(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should return ndarray."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert isinstance(info, np.ndarray)

    def test_view_spatial_information_shape(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """view_spatial_information() should return (n_neurons,) array."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert info.shape == (n_neurons,)

    def test_view_spatial_information_matches_single_neuron(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """view_spatial_information() should match single-neuron view_spatial_information()."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        batch_info = result.view_spatial_information()

        # Verify each neuron's info matches single-neuron result
        for i in range(n_neurons):
            single = result[i]
            single_info = single.view_spatial_information()
            assert batch_info[i] == pytest.approx(single_info, rel=1e-10)

    def test_view_spatial_information_non_negative(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """view_spatial_information() should be non-negative for all neurons."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert np.all(info >= 0.0)

    def test_view_spatial_information_uniform_is_zero(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """view_spatial_information() should return 0 for uniform firing."""
        from neurospatial.encoding.view import ViewRatesResult

        # Uniform firing rates
        uniform_rates = np.ones((n_neurons, simple_env.n_bins), dtype=np.float64) * 5.0

        result = ViewRatesResult(
            firing_rates=uniform_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        info = result.view_spatial_information()
        assert np.all(np.abs(info) < 1e-10)


class TestViewRatesResultDetectViewCells:
    """Test ViewRatesResult.detect_view_cells() method."""

    def test_detect_view_cells_returns_ndarray(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """detect_view_cells() should return ndarray."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        classification = result.classify()
        assert isinstance(classification, np.ndarray)

    def test_detect_view_cells_shape(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """detect_view_cells() should return (n_neurons,) bool array."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        classification = result.classify()
        assert classification.shape == (n_neurons,)
        assert classification.dtype == np.bool_

    def test_detect_view_cells_matches_single_neuron(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """detect_view_cells() should match single-neuron is_spatial_view_cell()."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        min_info = 0.1  # Use a consistent threshold
        batch_classification = result.classify(min_info=min_info)

        # Verify each neuron's classification matches single-neuron result
        for i in range(n_neurons):
            single = result[i]
            single_classification = single.is_spatial_view_cell(min_info=min_info)
            assert batch_classification[i] == single_classification

    def test_detect_view_cells_respects_min_info(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """detect_view_cells() should respect min_info parameter."""
        from neurospatial.encoding.view import ViewRatesResult

        n_bins = simple_env.n_bins

        # Create neurons with different spatial info levels
        rates = np.zeros((3, n_bins), dtype=np.float64)
        # Neuron 0: uniform (zero info)
        rates[0, :] = 5.0
        # Neuron 1: moderate peak
        rates[1, n_bins // 2] = 20.0
        rates[1, :] += 1.0  # Add baseline
        # Neuron 2: sharp peak (high info)
        rates[2, n_bins // 3] = 50.0

        result = ViewRatesResult(
            firing_rates=rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # With very low threshold, most should pass
        low_thresh = result.classify(min_info=0.0)
        # At least the peaked neurons should be True
        assert low_thresh[1] or low_thresh[2]

        # With high threshold, uniform should definitely fail
        high_thresh = result.classify(min_info=10.0)
        assert high_thresh[0] is np.False_

    def test_detect_view_cells_default_threshold(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """detect_view_cells() should use default min_info=0.5."""
        from neurospatial.encoding.view import ViewRatesResult

        # Uniform firing - should all be False with default threshold
        uniform_rates = np.ones((n_neurons, simple_env.n_bins), dtype=np.float64) * 5.0

        result = ViewRatesResult(
            firing_rates=uniform_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        classification = result.classify()
        # Uniform firing has zero info, should all be False
        assert not np.any(classification)


# ==============================================================================
# ViewRatesResult.to_dataframe() Tests - Task 4.5
# ==============================================================================


class TestViewRatesResultToDataframe:
    """Test ViewRatesResult.to_dataframe() method."""

    def test_summary_table_returns_dataframe(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should return a pandas DataFrame."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        import pandas as pd

        df = result.summary_table()
        assert isinstance(df, pd.DataFrame)

    def test_summary_table_row_count(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """to_dataframe() should have one row per neuron."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert len(df) == n_neurons

    def test_summary_table_has_unit_id_column(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """summary_table() should be indexed by unit_id."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert df.index.name == "unit_id"

    def test_summary_table_has_peak_view_x_column(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should have peak_view_x column."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert "peak_view_x" in df.columns

    def test_summary_table_has_peak_view_y_column(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should have peak_view_y column."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert "peak_view_y" in df.columns

    def test_summary_table_has_peak_rate_column(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should have peak_rate column."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert "peak_rate" in df.columns

    def test_summary_table_has_view_spatial_info_column(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should have view_spatial_info column."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert "view_spatial_info" in df.columns

    def test_summary_table_has_is_view_cell_column(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should have is_spatial_view_cell column."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert "is_spatial_view_cell" in df.columns

    def test_summary_table_default_unit_ids(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """to_dataframe() should use integer indices by default for neuron_id."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        expected_ids = list(range(n_neurons))
        assert list(df.index) == expected_ids

    def test_summary_table_custom_unit_ids(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """summary_table() should accept custom unit_ids."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        custom_ids = [f"unit_{i}" for i in range(n_neurons)]
        df = result.summary_table(unit_ids=custom_ids)
        assert list(df.index) == custom_ids

    def test_summary_table_unit_ids_length_mismatch(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """summary_table() should raise ValueError for wrong unit_ids length."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        # Wrong number of ids
        with pytest.raises(ValueError):
            result.summary_table(unit_ids=["a", "b"])  # Only 2, but 5 neurons

    def test_summary_table_peak_view_x_matches_batch_method(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """to_dataframe() peak_view_x should match peak_view_location()[:, 0]."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        expected = result.peak_locations()[:, 0]
        np.testing.assert_array_almost_equal(df["peak_view_x"].values, expected)

    def test_summary_table_peak_view_y_matches_batch_method(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
        n_neurons: int,
    ) -> None:
        """to_dataframe() peak_view_y should match peak_view_location()[:, 1]."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        expected = result.peak_locations()[:, 1]
        np.testing.assert_array_almost_equal(df["peak_view_y"].values, expected)

    def test_summary_table_peak_rate_matches_batch_method(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() peak_rate should match np.nanmax(firing_rates, axis=1)."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        expected = np.nanmax(batch_firing_rates, axis=1)
        np.testing.assert_array_almost_equal(df["peak_rate"].values, expected)

    def test_summary_table_view_spatial_info_matches_batch_method(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() view_spatial_info should match view_spatial_information()."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        expected = result.view_spatial_information()
        np.testing.assert_array_almost_equal(df["view_spatial_info"].values, expected)

    def test_summary_table_is_view_cell_matches_batch_method(
        self,
        simple_env: Environment,
        batch_firing_rates: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() is_spatial_view_cell should match detect_view_cells()."""
        from neurospatial.encoding.view import ViewRatesResult

        result = ViewRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        expected = result.classify()
        np.testing.assert_array_equal(df["is_spatial_view_cell"].values, expected)

    def test_summary_table_empty_result(
        self,
        simple_env: Environment,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should handle empty result (zero neurons)."""
        import pandas as pd

        from neurospatial.encoding.view import ViewRatesResult

        empty_rates = np.zeros((0, simple_env.n_bins), dtype=np.float64)

        result = ViewRatesResult(
            firing_rates=empty_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Should still have all columns
        expected_columns = {
            "peak_view_x",
            "peak_view_y",
            "peak_rate",
            "view_spatial_info",
            "is_spatial_view_cell",
        }
        assert expected_columns.issubset(set(df.columns))

    def test_summary_table_single_neuron(
        self,
        simple_env: Environment,
        single_firing_rate: np.ndarray,
        single_occupancy: np.ndarray,
    ) -> None:
        """to_dataframe() should work with a single neuron."""
        import pandas as pd

        from neurospatial.encoding.view import ViewRatesResult

        # Reshape single neuron to batch format
        batch_rates = single_firing_rate.reshape(1, -1)

        result = ViewRatesResult(
            firing_rates=batch_rates,
            occupancy=single_occupancy,
            env=simple_env,
            gaze_model="fixed_distance",
            view_distance=10.0,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )

        df = result.summary_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.index[0] == 0


# ==============================================================================
# Free is_spatial_view_cell parameter-order parity with compute_view_rate
# ==============================================================================


class TestIsSpatialViewCellParamOrder:
    """The free is_spatial_view_cell mirrors compute_view_rate's kw order."""

    def test_is_spatial_view_cell_param_order(
        self,
        simple_env: Environment,
    ) -> None:
        """gaze_model precedes view_distance, matching compute_view_rate."""
        import inspect

        from neurospatial.encoding.view import (
            compute_view_rate,
            is_spatial_view_cell,
        )

        rng = np.random.default_rng(0)
        n_samples = 500
        times = np.linspace(0.0, 50.0, n_samples)
        positions = rng.uniform(0.0, 40.0, (n_samples, 2))
        headings = rng.uniform(-np.pi, np.pi, n_samples)
        spike_times = np.sort(rng.uniform(0.0, 50.0, 60))

        # Runs with both keywords supplied (any order, since keyword-only).
        result = is_spatial_view_cell(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="ray_cast",
            view_distance=15.0,
        )
        assert isinstance(result, bool)

        # Signature inspection: gaze_model precedes view_distance, matching
        # the primary compute_view_rate function.
        sig = inspect.signature(is_spatial_view_cell)
        names = list(sig.parameters)
        assert names.index("gaze_model") < names.index("view_distance")

        view_sig = inspect.signature(compute_view_rate)
        view_names = list(view_sig.parameters)
        assert view_names.index("gaze_model") < view_names.index("view_distance")
