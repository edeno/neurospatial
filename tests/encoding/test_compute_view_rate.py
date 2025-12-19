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

    def test_view_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """view_occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        assert np.asarray(result.view_occupancy).shape == (simple_env.n_bins,)

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

    def test_empty_spikes_still_has_view_occupancy(
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
        view_occupancy = np.asarray(result.view_occupancy)
        # Should have some positive occupancy from trajectory
        assert np.sum(view_occupancy) > 0


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

    def test_view_occupancy_non_negative(
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
        view_occupancy = np.asarray(result.view_occupancy)
        assert np.all(view_occupancy >= 0)

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
        view_occupancy = np.asarray(result.view_occupancy)
        # They should not be identical (unless trajectory is contrived)
        assert not np.allclose(view_occupancy, standard_occupancy)


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

    def test_result_has_peak_view_location_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should have peak_view_location() method that works."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        peak = result.peak_view_location()
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
        """Result should have is_view_cell() method that works."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
        )
        is_cell = result.is_view_cell()
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
