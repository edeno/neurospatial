"""Tests for compute_view_rate() function.

This module tests the compute_view_rate() function for computing spatial
view fields from spike trains and trajectory data.

Task 4.7: Implement compute_view_rate() function
- Accept single spike_times array
- Support gaze_model parameter
- Support view_distance parameter
- Apply smoothing via _smoothing.py
- Return ViewRateResult
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

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
def trajectory_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:  # times, positions, headings
    """Create sample trajectory data with positions and headings."""
    np.random.seed(42)
    n_samples = 1000
    times = np.linspace(0, 10, n_samples)
    # Random walk within environment
    positions = np.column_stack(
        [
            20 + np.cumsum(np.random.randn(n_samples) * 0.5),  # x
            20 + np.cumsum(np.random.randn(n_samples) * 0.5),  # y
        ]
    )
    positions = np.clip(positions, 5, 35)  # Keep within bounds
    # Random headings (0 to 2*pi)
    headings = np.random.uniform(0, 2 * np.pi, n_samples)
    return times, positions, headings


@pytest.fixture
def spike_times() -> np.ndarray:
    """Spike times within the trajectory time range."""
    np.random.seed(123)
    return np.sort(np.random.uniform(0.1, 9.9, 50))  # 50 spikes


@pytest.fixture
def empty_spike_times() -> np.ndarray:
    """Empty spike train."""
    return np.array([], dtype=np.float64)


# ==============================================================================
# Import Tests
# ==============================================================================


class TestComputeViewRateImport:
    """Test that compute_view_rate can be imported."""

    def test_import_from_view(self) -> None:
        """compute_view_rate can be imported from encoding.view."""
        from neurospatial.encoding.view import compute_view_rate

        assert compute_view_rate is not None
        assert callable(compute_view_rate)

    def test_import_from_encoding(self) -> None:
        """compute_view_rate can be imported from encoding package."""
        from neurospatial.encoding import compute_view_rate

        assert compute_view_rate is not None
        assert callable(compute_view_rate)


# ==============================================================================
# Return Type Tests
# ==============================================================================


class TestComputeViewRateReturnsResult:
    """Test that compute_view_rate returns ViewRateResult."""

    def test_returns_view_rate_result(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """compute_view_rate should return a ViewRateResult object."""
        from neurospatial.encoding.view import ViewRateResult, compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert isinstance(result, ViewRateResult)

    def test_firing_rate_shape(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """firing_rate should have shape (n_bins,)."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert np.asarray(result.firing_rate).shape == (simple_env.n_bins,)

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert np.asarray(result.occupancy).shape == (simple_env.n_bins,)

    def test_result_has_correct_env(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should contain the same environment."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert result.env is simple_env


# ==============================================================================
# Gaze Model Parameter Tests
# ==============================================================================


class TestComputeViewRateGazeModel:
    """Test gaze_model parameter."""

    @pytest.mark.parametrize("gaze_model", ["fixed_distance", "ray_cast", "boundary"])
    def test_accepts_valid_gaze_models(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
        gaze_model: Literal["fixed_distance", "ray_cast", "boundary"],
    ) -> None:
        """compute_view_rate should accept all valid gaze models."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model=gaze_model,
        )
        assert result.gaze_model == gaze_model

    def test_default_gaze_model_is_fixed_distance(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default gaze_model should be 'fixed_distance'."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert result.gaze_model == "fixed_distance"

    def test_invalid_gaze_model_raises(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Invalid gaze_model should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        with pytest.raises(ValueError, match="gaze_model"):
            compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
                gaze_model="invalid_model",  # type: ignore[arg-type]
            )


# ==============================================================================
# View Distance Parameter Tests
# ==============================================================================


class TestComputeViewRateViewDistance:
    """Test view_distance parameter."""

    def test_stores_view_distance_in_result(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """view_distance should be stored in the result."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            view_distance=15.0,
        )
        assert result.view_distance == 15.0

    def test_default_view_distance_is_10(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default view_distance should be 10.0."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert result.view_distance == 10.0


# ==============================================================================
# Smoothing Parameter Tests
# ==============================================================================


class TestComputeViewRateSmoothing:
    """Test smoothing parameters."""

    @pytest.mark.parametrize(
        "smoothing_method", ["diffusion_kde", "gaussian_kde", "binned"]
    )
    def test_accepts_valid_smoothing_methods(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
        smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    ) -> None:
        """compute_view_rate should accept all valid smoothing methods."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            smoothing_method=smoothing_method,
        )
        assert result.smoothing_method == smoothing_method

    def test_default_smoothing_method_is_diffusion_kde(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default smoothing_method should be 'diffusion_kde'."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert result.smoothing_method == "diffusion_kde"

    def test_stores_bandwidth_in_result(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """bandwidth should be stored in the result."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            bandwidth=8.0,
        )
        assert result.bandwidth == 8.0

    def test_default_bandwidth_is_5(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default bandwidth should be 5.0."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert result.bandwidth == 5.0


# ==============================================================================
# Empty Spike Train Tests
# ==============================================================================


class TestComputeViewRateEmptySpikes:
    """Test handling of empty spike trains."""

    def test_empty_spikes_returns_zero_firing_rate(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        empty_spike_times: np.ndarray,
    ) -> None:
        """Empty spike train should produce firing rate of zero or NaN."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            empty_spike_times,
            times,
            positions,
            headings,
        )
        firing_rate = np.asarray(result.firing_rate)
        # All values should be 0 or NaN (low occupancy bins are NaN)
        valid_mask = ~np.isnan(firing_rate)
        if np.any(valid_mask):
            assert np.all(firing_rate[valid_mask] == 0.0)

    def test_empty_spikes_still_has_occupancy(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        empty_spike_times: np.ndarray,
    ) -> None:
        """Empty spike train should still have positive view occupancy."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            empty_spike_times,
            times,
            positions,
            headings,
        )
        occupancy = np.asarray(result.occupancy)
        # Should have some positive occupancy from trajectory
        assert np.sum(occupancy) > 0


# ==============================================================================
# Correctness Tests
# ==============================================================================


class TestComputeViewRateCorrectness:
    """Test correctness of compute_view_rate computation."""

    def test_firing_rate_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Firing rate should be non-negative (or NaN)."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        firing_rate = np.asarray(result.firing_rate)
        valid_mask = ~np.isnan(firing_rate)
        assert np.all(firing_rate[valid_mask] >= 0)

    def test_occupancy_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """View occupancy should be non-negative."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        occupancy = np.asarray(result.occupancy)
        assert np.all(occupancy >= 0)

    def test_uses_view_binning_not_spatial_binning(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """View occupancy should differ from standard spatial occupancy.

        This test verifies that compute_view_rate uses VIEW occupancy
        (time viewing each bin) not standard occupancy (time at each bin).
        """
        from neurospatial.encoding._binning import compute_occupancy
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )

        # Compute standard spatial occupancy
        standard_occupancy = compute_occupancy(simple_env, times, positions)

        # View occupancy and spatial occupancy should be different
        # because they measure different things
        occupancy = np.asarray(result.occupancy)
        # They should not be identical (unless trajectory is contrived)
        assert not np.allclose(occupancy, standard_occupancy)


# ==============================================================================
# Result Method Tests
# ==============================================================================


class TestComputeViewRateResultMethods:
    """Test that result has working methods."""

    def test_result_has_plot_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should have plot() method."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert hasattr(result, "plot")
        assert callable(result.plot)

    def test_result_has_peak_location_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should have peak_location() method that works."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        peak = result.peak_location()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (2,)  # 2D environment

    def test_result_has_view_spatial_information_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should have view_spatial_information() method that works."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        info = result.view_spatial_information()
        assert isinstance(info, float)
        assert info >= 0.0

    def test_result_has_is_view_cell_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should have is_spatial_view_cell() method that works."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        is_cell = result.is_spatial_view_cell()
        assert isinstance(is_cell, bool)


# ==============================================================================
# Min Occupancy Parameter Tests
# ==============================================================================


class TestComputeViewRateMinOccupancy:
    """Test min_occupancy parameter."""

    def test_min_occupancy_sets_low_occupancy_to_nan(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Bins with occupancy below min_occupancy should have NaN firing rate."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result_high = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            min_occupancy=1.0,  # High threshold
        )
        result_low = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            min_occupancy=0.0,  # No threshold
        )
        # High threshold should have more NaNs
        high_nan_count = np.sum(np.isnan(np.asarray(result_high.firing_rate)))
        low_nan_count = np.sum(np.isnan(np.asarray(result_low.firing_rate)))
        assert high_nan_count >= low_nan_count

    def test_default_min_occupancy_is_zero(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default min_occupancy should be 0.0."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        # This should work without min_occupancy parameter
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        # With zero threshold, we should get computed rates for all bins with data
        assert result is not None


# ==============================================================================
# Signature Tests
# ==============================================================================


class TestComputeViewRateInputValidation:
    """Test input validation."""

    def test_mismatched_times_positions_raises(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched times and positions should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rate

        times = np.linspace(0, 10, 1000)
        positions = np.random.rand(500, 2) * 100  # Wrong length
        headings = np.random.uniform(0, 2 * np.pi, 1000)

        with pytest.raises(ValueError, match=r"times length.*positions length"):
            compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
            )

    def test_mismatched_times_headings_raises(
        self,
        simple_env: Environment,
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched times and headings should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rate

        times = np.linspace(0, 10, 1000)
        positions = np.random.rand(1000, 2) * 100
        headings = np.random.uniform(0, 2 * np.pi, 500)  # Wrong length

        with pytest.raises(ValueError, match=r"times length.*headings length"):
            compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
            )


class TestComputeViewRateSignature:
    """Test function signature follows conventions."""

    def test_canonical_argument_order(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Arguments should follow canonical order: env, spike_times, times, positions, headings."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        # Positional arguments should work in this order
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert result is not None

    def test_keyword_only_parameters(self) -> None:
        """gaze_model, view_distance, smoothing_method, etc. should be keyword-only."""
        import inspect

        from neurospatial.encoding.view import compute_view_rate

        sig = inspect.signature(compute_view_rate)
        params = sig.parameters

        # These should be keyword-only (after the *)
        keyword_only_params = {
            "gaze_model",
            "view_distance",
            "smoothing_method",
            "bandwidth",
            "min_occupancy",
        }
        for param_name in keyword_only_params:
            if param_name in params:
                assert params[param_name].kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"{param_name} should be keyword-only"
                )


# =============================================================================
# Tests for compute_view_rates() (batch version)
# =============================================================================


class TestComputeViewRatesImport:
    """Test that compute_view_rates can be imported correctly."""

    def test_import_from_view_module(self) -> None:
        """compute_view_rates should be importable from view module."""
        from neurospatial.encoding.view import compute_view_rates

        assert compute_view_rates is not None

    def test_in_module_all(self) -> None:
        """compute_view_rates should be in __all__ list."""
        from neurospatial.encoding import view

        assert "compute_view_rates" in view.__all__


class TestComputeViewRatesReturnsResult:
    """Test that compute_view_rates returns ViewRatesResult."""

    def test_returns_view_rates_result(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """compute_view_rates should return ViewRatesResult."""
        from neurospatial.encoding.view import ViewRatesResult, compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 1.5, 2.5]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert isinstance(result, ViewRatesResult)

    def test_result_has_correct_firing_rates_shape(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """firing_rates should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
            np.array([5.0]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert result.firing_rates.shape == (3, simple_env.n_bins)

    def test_result_has_correct_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """occupancy should have shape (n_bins,) - shared across neurons."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert result.occupancy.shape == (simple_env.n_bins,)


class TestComputeViewRatesSpikeTimeFormats:
    """Test that compute_view_rates accepts different spike time formats."""

    def test_list_of_arrays(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should accept list of 1D arrays (canonical format)."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert len(result) == 2

    def test_2d_array_with_nan_padding(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should accept 2D array with NaN padding."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_2d = np.array(
            [
                [1.0, 2.5, np.nan],
                [0.5, 1.5, 3.5],
            ]
        )

        result = compute_view_rates(
            simple_env, spike_times_2d, times, positions, headings
        )

        assert len(result) == 2


class TestComputeViewRatesParameters:
    """Test compute_view_rates parameter handling."""

    def test_gaze_model_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """gaze_model parameter should be passed to underlying functions."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            gaze_model="ray_cast",
        )

        assert result.gaze_model == "ray_cast"

    def test_view_distance_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """view_distance parameter should be stored in result."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            view_distance=20.0,
        )

        assert result.view_distance == 20.0

    def test_smoothing_method_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """smoothing_method parameter should be stored in result."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            smoothing_method="gaussian_kde",
        )

        assert result.smoothing_method == "gaussian_kde"

    def test_bandwidth_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """bandwidth parameter should be stored in result."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            bandwidth=10.0,
        )

        assert result.bandwidth == 10.0

    def test_n_jobs_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """n_jobs parameter should work without error."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
            np.array([5.0]),
        ]

        # Test with n_jobs=2
        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            n_jobs=2,
        )

        assert len(result) == 3


class TestComputeViewRatesNeuronIteration:
    """Test that ViewRatesResult supports iteration over neurons."""

    def test_len(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """len() should return number of neurons."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
            np.array([5.0]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert len(result) == 3

    def test_getitem(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """result[i] should return ViewRateResult for neuron i."""
        from neurospatial.encoding.view import ViewRateResult, compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        single = result[0]
        assert isinstance(single, ViewRateResult)
        assert single.firing_rate.shape == (simple_env.n_bins,)

    def test_iteration(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Iteration should yield ViewRateResult for each neuron."""
        from neurospatial.encoding.view import ViewRateResult, compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        count = 0
        for single in result:
            assert isinstance(single, ViewRateResult)
            count += 1

        assert count == 2


class TestComputeViewRatesEdgeCases:
    """Test edge cases for compute_view_rates."""

    def test_empty_spike_times_list(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Empty list should return result with zero neurons."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list: list[np.ndarray] = []

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert len(result) == 0
        assert result.firing_rates.shape == (0, simple_env.n_bins)

    def test_empty_spike_times_list_validates_smoothing_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Empty list should still reject invalid smoothing methods."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data

        with pytest.raises(ValueError, match="method must be one of"):
            compute_view_rates(
                simple_env,
                [],
                times,
                positions,
                headings,
                smoothing_method="invalid",  # type: ignore[arg-type]
            )

    def test_single_neuron(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Single neuron should work correctly."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5, 4.0])]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        assert len(result) == 1
        assert result.firing_rates.shape == (1, simple_env.n_bins)

    def test_neuron_with_no_spikes(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Neuron with empty spike train should produce zeros."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([]),  # No spikes
        ]

        result = compute_view_rates(
            simple_env, spike_times_list, times, positions, headings
        )

        # Second neuron should have all zeros
        assert np.all(result.firing_rates[1] == 0)


class TestComputeViewRatesInputValidation:
    """Test input validation for compute_view_rates."""

    def test_invalid_gaze_model_raises(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Invalid gaze_model should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        with pytest.raises(ValueError, match=r"Invalid gaze_model"):
            compute_view_rates(
                simple_env,
                spike_times_list,
                times,
                positions,
                headings,
                gaze_model="invalid_model",  # type: ignore[arg-type]
            )

    def test_mismatched_times_positions_raises(
        self,
        simple_env: Environment,
    ) -> None:
        """Mismatched times and positions should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rates

        times = np.linspace(0, 10, 1000)
        positions = np.random.rand(500, 2) * 100  # Wrong length
        headings = np.random.uniform(0, 2 * np.pi, 1000)
        spike_times_list = [np.array([1.0, 2.5])]

        with pytest.raises(ValueError, match=r"times length.*positions length"):
            compute_view_rates(
                simple_env,
                spike_times_list,
                times,
                positions,
                headings,
            )

    def test_mismatched_times_headings_raises(
        self,
        simple_env: Environment,
    ) -> None:
        """Mismatched times and headings should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rates

        times = np.linspace(0, 10, 1000)
        positions = np.random.rand(1000, 2) * 100
        headings = np.random.uniform(0, 2 * np.pi, 500)  # Wrong length
        spike_times_list = [np.array([1.0, 2.5])]

        with pytest.raises(ValueError, match=r"times length.*headings length"):
            compute_view_rates(
                simple_env,
                spike_times_list,
                times,
                positions,
                headings,
            )


class TestComputeViewRatesConsistencyWithSingle:
    """Test that compute_view_rates is consistent with compute_view_rate."""

    def test_single_neuron_matches(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Single neuron result should match compute_view_rate."""
        from neurospatial.encoding.view import compute_view_rate, compute_view_rates

        times, positions, headings = trajectory_data

        # Compute with single function
        single_result = compute_view_rate(
            simple_env, spike_times, times, positions, headings
        )

        # Compute with batch function
        batch_result = compute_view_rates(
            simple_env, [spike_times], times, positions, headings
        )

        # Should match
        np.testing.assert_array_almost_equal(
            batch_result.firing_rates[0], single_result.firing_rate
        )
        np.testing.assert_array_almost_equal(
            batch_result.occupancy, single_result.occupancy
        )


class TestComputeViewRatesSignature:
    """Test function signature follows conventions."""

    def test_canonical_argument_order(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Arguments should follow canonical order: env, spike_times, times, positions, headings."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        # Positional arguments should work in this order
        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
        )
        assert result is not None

    def test_keyword_only_parameters(self) -> None:
        """gaze_model, view_distance, etc. should be keyword-only."""
        import inspect

        from neurospatial.encoding.view import compute_view_rates

        sig = inspect.signature(compute_view_rates)
        params = sig.parameters

        # These should be keyword-only (after the *)
        keyword_only_params = {
            "gaze_model",
            "view_distance",
            "smoothing_method",
            "bandwidth",
            "min_occupancy",
            "n_jobs",
            "gaze_offsets",
        }
        for param_name in keyword_only_params:
            if param_name in params:
                assert params[param_name].kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"{param_name} should be keyword-only"
                )


# =============================================================================
# Tests for gaze_offsets parameter (Task 8.4)
# =============================================================================


class TestComputeViewRateGazeOffsets:
    """Test gaze_offsets parameter for compute_view_rate."""

    def test_accepts_gaze_offsets_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """compute_view_rate should accept gaze_offsets parameter."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        # Create gaze offsets - e.g., simulate looking 30 degrees to the left
        gaze_offsets = np.full_like(headings, np.pi / 6)

        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        assert result is not None
        assert result.firing_rate.shape == (simple_env.n_bins,)

    def test_gaze_offsets_default_is_none(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default gaze_offsets should be None (gaze aligned with heading)."""
        import inspect

        from neurospatial.encoding.view import compute_view_rate

        sig = inspect.signature(compute_view_rate)
        params = sig.parameters

        assert "gaze_offsets" in params
        assert params["gaze_offsets"].default is None

    def test_gaze_offsets_changes_view_field(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Non-zero gaze offsets should produce different view field than zero offsets."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data

        # No gaze offset (aligned with heading)
        result_no_offset = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_offsets=None,
        )

        # 90-degree gaze offset (looking left)
        gaze_offsets = np.full_like(headings, np.pi / 2)
        result_with_offset = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        # View occupancy should be different (looking in different directions)
        assert not np.allclose(result_no_offset.occupancy, result_with_offset.occupancy)

    def test_gaze_offsets_mismatched_length_raises(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched gaze_offsets length should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        # Wrong length gaze_offsets
        gaze_offsets = np.zeros(len(times) // 2)

        with pytest.raises(ValueError, match=r"gaze_offsets.*length"):
            compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
                gaze_offsets=gaze_offsets,
            )

    def test_gaze_offsets_with_different_gaze_models(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """gaze_offsets should work with all gaze models."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        gaze_offsets = np.full_like(headings, np.pi / 6)

        for gaze_model in ["fixed_distance", "ray_cast", "boundary"]:
            result = compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
                gaze_model=gaze_model,  # type: ignore[arg-type]
                gaze_offsets=gaze_offsets,
            )
            assert result is not None

    def test_zero_gaze_offsets_equals_no_offsets(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Zero gaze offsets should produce same result as None."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data

        # No offset (None)
        result_none = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_offsets=None,
        )

        # Explicit zero offsets
        gaze_offsets = np.zeros_like(headings)
        result_zero = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        # Should be identical
        np.testing.assert_array_almost_equal(
            result_none.firing_rate, result_zero.firing_rate
        )
        np.testing.assert_array_almost_equal(
            result_none.occupancy, result_zero.occupancy
        )


class TestComputeViewRatePrecomputation:
    """Regression tests for single-neuron precomputation efficiency."""

    def test_view_coordinates_computed_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """compute_view_rate should reuse viewed locations for counts and occupancy."""
        from neurospatial.encoding import _view_binning
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        call_count = 0
        original = _view_binning.compute_viewed_location

        def counting_compute_viewed_location(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            _view_binning,
            "compute_viewed_location",
            counting_compute_viewed_location,
        )

        compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="ray_cast",
            smoothing_method="binned",
        )

        assert call_count == 1


class TestComputeViewRatesGazeOffsets:
    """Test gaze_offsets parameter for compute_view_rates (batch version)."""

    def test_accepts_gaze_offsets_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """compute_view_rates should accept gaze_offsets parameter."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5, 4.0]),
            np.array([0.5, 1.5, 2.5]),
        ]
        gaze_offsets = np.full_like(headings, np.pi / 6)

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        assert result is not None
        assert result.firing_rates.shape == (2, simple_env.n_bins)

    def test_gaze_offsets_default_is_none(self) -> None:
        """Default gaze_offsets should be None."""
        import inspect

        from neurospatial.encoding.view import compute_view_rates

        sig = inspect.signature(compute_view_rates)
        params = sig.parameters

        assert "gaze_offsets" in params
        assert params["gaze_offsets"].default is None

    def test_gaze_offsets_changes_view_fields(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Non-zero gaze offsets should produce different view fields."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5, 4.0])]

        # No gaze offset
        result_no_offset = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            gaze_offsets=None,
        )

        # 90-degree gaze offset
        gaze_offsets = np.full_like(headings, np.pi / 2)
        result_with_offset = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        # View occupancy should differ
        assert not np.allclose(result_no_offset.occupancy, result_with_offset.occupancy)

    def test_gaze_offsets_mismatched_length_raises(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Mismatched gaze_offsets length should raise ValueError."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]
        gaze_offsets = np.zeros(len(times) // 2)

        with pytest.raises(ValueError, match=r"gaze_offsets.*length"):
            compute_view_rates(
                simple_env,
                spike_times_list,
                times,
                positions,
                headings,
                gaze_offsets=gaze_offsets,
            )

    def test_gaze_offsets_consistency_with_single(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Batch with gaze_offsets should match single neuron version."""
        from neurospatial.encoding.view import compute_view_rate, compute_view_rates

        times, positions, headings = trajectory_data
        gaze_offsets = np.full_like(headings, np.pi / 4)

        # Single neuron
        single_result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        # Batch with one neuron
        batch_result = compute_view_rates(
            simple_env,
            [spike_times],
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
        )

        # Should match
        np.testing.assert_array_almost_equal(
            batch_result.firing_rates[0], single_result.firing_rate
        )
        np.testing.assert_array_almost_equal(
            batch_result.occupancy, single_result.occupancy
        )

    def test_gaze_offsets_keyword_only(self) -> None:
        """gaze_offsets should be keyword-only parameter."""
        import inspect

        from neurospatial.encoding.view import compute_view_rates

        sig = inspect.signature(compute_view_rates)
        params = sig.parameters

        assert "gaze_offsets" in params
        assert params["gaze_offsets"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_gaze_offsets_with_n_jobs(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """gaze_offsets should work with parallel processing."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [
            np.array([1.0, 2.5]),
            np.array([0.5, 1.5]),
            np.array([3.0, 4.5]),
        ]
        gaze_offsets = np.full_like(headings, np.pi / 3)

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
            n_jobs=2,
        )

        assert len(result) == 3


class TestComputeViewRateNaNHandling:
    """NaN positions/headings are treated as missing data, not errors.

    Per the compute_view_rate docstring contract, NaN samples (tracking
    dropouts) are excluded rather than raising or corrupting the map.
    """

    @pytest.fixture(scope="class")
    def session(self):
        from neurospatial.ops.egocentric import heading_from_velocity
        from neurospatial.simulation import (
            generate_poisson_spikes,
            simulate_trajectory_ou,
        )

        samples = np.random.default_rng(0).uniform(0, 40, (3000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)
        env.units = "cm"
        positions, times = simulate_trajectory_ou(
            env, duration=400.0, speed_units="cm", seed=42
        )
        dt = float(times[1] - times[0])
        headings = heading_from_velocity(positions, dt, min_speed=2.0)
        rng = np.random.default_rng(3)
        rates = 5.0 + 5.0 * rng.random(len(times))
        spike_times = generate_poisson_spikes(rates, times, seed=7)
        return env, spike_times, times, positions, headings

    @pytest.mark.parametrize("nan_field", ["positions", "headings"])
    def test_nan_inputs_excluded_not_raised(self, session, nan_field):
        from neurospatial.encoding.view import compute_view_rate

        env, spike_times, times, positions, headings = session
        rng = np.random.default_rng(5)
        drop_idx = rng.choice(len(times), size=len(times) // 10, replace=False)
        positions_in = positions.copy()
        headings_in = headings.copy()
        if nan_field == "positions":
            positions_in[drop_idx] = np.nan
        else:
            headings_in[drop_idx] = np.nan

        result = compute_view_rate(
            env,
            spike_times,
            times,
            positions_in,
            headings_in,
            view_distance=10.0,
            gaze_model="fixed_distance",
            smoothing_method="binned",
        )

        # Did not raise; produced a usable (not all-NaN) map with a finite peak.
        assert not np.isnan(result.firing_rate).all()
        assert np.isfinite(np.nanmax(result.firing_rate))
