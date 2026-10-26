"""Tests for compute_view_rate() function.

This module tests the compute_view_rate() function for computing spatial
view fields from spike trains and trajectory data.

compute_view_rate():
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
    return Environment.from_samples(positions, bin_size=10.0, units="cm")


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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
        )
        assert result.env is simple_env


# ==============================================================================
# Gaze Model Parameter Tests
# ==============================================================================


class TestComputeViewRateGazeModel:
    """Test gaze_model parameter."""

    @pytest.mark.parametrize("gaze_model", ["fixed_distance", "ray_cast"])
    def test_accepts_valid_gaze_models(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
        gaze_model: Literal["fixed_distance", "ray_cast"],
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
            boundary_source="analysis",
        )
        assert result.gaze_model == gaze_model

    def test_default_gaze_model_is_ray_cast(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """The default views the first wall, so it has no fixed distance."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            boundary_source="analysis",
        )
        assert result.gaze_model == "ray_cast"
        assert result.view_distance is None

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
                gaze_model="invalid_model",
                boundary_source="analysis",  # type: ignore[arg-type]
            )


# ==============================================================================
# View Distance Parameter Tests
# ==============================================================================


class TestComputeViewRateViewDistance:
    """view_distance belongs to the fixed-distance gaze and carries length units."""

    def test_stores_view_distance_in_result(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=15.0,
            boundary_source="analysis",
        )
        assert result.view_distance == 15.0

    @pytest.mark.parametrize(("units", "expected"), [("cm", 10.0), ("m", 0.1)])
    def test_default_view_distance_is_10_cm_in_env_units(
        self, units: str, expected: float
    ) -> None:
        from shapely.geometry import box

        from neurospatial.encoding.view import compute_view_rate

        side = 40.0 if units == "cm" else 0.4
        env = Environment.from_polygon(
            box(0, 0, side, side), bin_size=side / 8, units=units
        )
        rng = np.random.default_rng(0)
        times = np.arange(300) / 30.0
        positions = rng.uniform(0.1 * side, 0.9 * side, (300, 2))
        headings = rng.uniform(-np.pi, np.pi, 300)
        result = compute_view_rate(
            env, [], times, positions, headings, gaze_model="fixed_distance"
        )
        assert result.view_distance == pytest.approx(expected)
        # Every gaze lands in the arena: a short gaze or the nearer wall.
        assert np.sum(result.occupancy) == pytest.approx(times[-1] - times[0])

    def test_unknown_units_need_an_explicit_view_distance(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        env = Environment.from_samples(positions, bin_size=10.0)
        with pytest.raises(ValueError, match="needs view_distance"):
            compute_view_rate(
                env,
                [],
                times,
                positions,
                headings,
                gaze_model="fixed_distance",
                boundary_source="analysis",
            )

    def test_view_distance_is_refused_with_ray_cast(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        with pytest.raises(ValueError, match="no meaning with gaze_model='ray_cast'"):
            compute_view_rate(
                simple_env,
                [],
                times,
                positions,
                headings,
                view_distance=15.0,
                boundary_source="analysis",
            )


# ==============================================================================
# Smoothing Parameter Tests
# ==============================================================================


class TestComputeViewRateSmoothing:
    """Test smoothing parameters."""

    @pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "binned"])
    def test_accepts_valid_methods(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
        method: Literal["diffusion_kde", "gaussian_kde", "binned"],
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
            method=method,
            boundary_source="analysis",
        )
        assert result.method == method

    def test_default_method_is_diffusion_kde(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Default method should be 'diffusion_kde'."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            boundary_source="analysis",
        )
        assert result.method == "diffusion_kde"

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
            boundary_source="analysis",
        )
        assert result.bandwidth == 8.0

    @pytest.mark.parametrize("units, expected", [(None, 25.0), ("cm", 5.0)])
    def test_default_bandwidth_uses_units_or_bin_spacing(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
        units: str | None,
        expected: float,
    ) -> None:
        """Use 5 cm with metric units, otherwise 2.5 of the 10-unit grid steps."""
        from neurospatial.encoding.view import compute_view_rate

        simple_env = simple_env.with_metadata(units=units)
        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            boundary_source="analysis",
        )
        assert result.bandwidth == expected


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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
        from neurospatial.encoding._spatial_counts import compute_occupancy
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
        )
        peak = result.peak_location()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (2,)  # 2D environment

    def test_result_has_spatial_information_method(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
    ) -> None:
        """Result should have spatial_information() method that works."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        result = compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            boundary_source="analysis",
        )
        info = result.spatial_information()
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
            boundary_source="analysis",
        )
        is_cell = result.is_spatial_view_cell()
        assert isinstance(is_cell, bool)


# ==============================================================================
# Min Occupancy Parameter Tests
# ==============================================================================


class TestComputeViewRateMinOccupancy:
    """Test min_occupancy parameter."""

    @pytest.mark.parametrize("method", ["binned", "diffusion_kde", "gaussian_kde"])
    def test_nan_exactly_where_view_occupancy_is_below_min_occupancy(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        spike_times: np.ndarray,
        method: str,
    ) -> None:
        """min_occupancy is seconds of viewing per bin: below it, unmeasured."""
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        args = (simple_env, spike_times, times, positions, headings)
        settings = {"method": method, "boundary_source": "analysis"}
        occupancy = np.asarray(compute_view_rate(*args, **settings).occupancy)
        threshold = float(np.median(occupancy[occupancy > 0]))
        result = compute_view_rate(*args, min_occupancy=threshold, **settings)
        unsupported = (occupancy == 0) | (occupancy < threshold)
        assert unsupported.any() and not unsupported.all()
        np.testing.assert_array_equal(
            np.isnan(np.asarray(result.firing_rate)), unsupported
        )

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
            boundary_source="analysis",
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

        with pytest.raises(ValueError, match=r"times has.*positions has"):
            compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
                boundary_source="analysis",
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
                boundary_source="analysis",
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
            boundary_source="analysis",
        )
        assert result is not None

    def test_keyword_only_parameters(self) -> None:
        """gaze_model, view_distance, method, etc. should be keyword-only."""
        import inspect

        from neurospatial.encoding.view import compute_view_rate

        sig = inspect.signature(compute_view_rate)
        params = sig.parameters

        # These should be keyword-only (after the *)
        keyword_only_params = {
            "gaze_model",
            "view_distance",
            "method",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_2d,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            gaze_model="fixed_distance",
            view_distance=20.0,
            boundary_source="analysis",
        )

        assert result.view_distance == 20.0

    def test_method_parameter(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """method parameter should be stored in result."""
        from neurospatial.encoding.view import compute_view_rates

        times, positions, headings = trajectory_data
        spike_times_list = [np.array([1.0, 2.5])]

        result = compute_view_rates(
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            method="gaussian_kde",
            boundary_source="analysis",
        )

        assert result.method == "gaussian_kde"

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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
        )

        assert len(result) == 0
        assert result.firing_rates.shape == (0, simple_env.n_bins)

    def test_empty_spike_times_list_validates_method(
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
                method="invalid",
                boundary_source="analysis",  # type: ignore[arg-type]
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            simple_env,
            spike_times_list,
            times,
            positions,
            headings,
            boundary_source="analysis",
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
                gaze_model="invalid_model",
                boundary_source="analysis",  # type: ignore[arg-type]
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

        with pytest.raises(ValueError, match=r"times has.*positions has"):
            compute_view_rates(
                simple_env,
                spike_times_list,
                times,
                positions,
                headings,
                boundary_source="analysis",
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
                boundary_source="analysis",
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
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            boundary_source="analysis",
        )

        # Compute with batch function
        batch_result = compute_view_rates(
            simple_env,
            [spike_times],
            times,
            positions,
            headings,
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            "method",
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
# Tests for gaze_offsets parameter
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
            boundary_source="analysis",
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
                boundary_source="analysis",
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

        for gaze_model in ["fixed_distance", "ray_cast"]:
            result = compute_view_rate(
                simple_env,
                spike_times,
                times,
                positions,
                headings,
                gaze_model=gaze_model,  # type: ignore[arg-type]
                gaze_offsets=gaze_offsets,
                boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
        from neurospatial.encoding import _view_counts
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = trajectory_data
        call_count = 0
        original = _view_counts.gaze_points

        def counting_gaze_points(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            _view_counts,
            "gaze_points",
            counting_gaze_points,
        )

        compute_view_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            gaze_model="ray_cast",
            method="binned",
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
            boundary_source="analysis",
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
                boundary_source="analysis",
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
            boundary_source="analysis",
        )

        # Batch with one neuron
        batch_result = compute_view_rates(
            simple_env,
            [spike_times],
            times,
            positions,
            headings,
            gaze_offsets=gaze_offsets,
            boundary_source="analysis",
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
            boundary_source="analysis",
        )

        assert len(result) == 3


class TestComputeViewRateNaNHandling:
    """NaN positions/headings are missing samples: dropped, not errors."""

    @pytest.mark.parametrize("nan_field", ["positions", "headings"])
    def test_nan_samples_drop_exactly_their_intervals(
        self, ou_2min, ou_2min_env, noise_trains, nan_field
    ):
        from neurospatial.encoding.view import compute_view_rate

        times, positions, headings = ou_2min
        drop = np.random.default_rng(5).choice(len(times), len(times) // 10, False)
        data = {"positions": positions.copy(), "headings": headings.copy()}
        data[nan_field][drop] = np.nan
        # Every observed gaze ends on the edge of the binned area.
        result = compute_view_rate(
            ou_2min_env,
            noise_trains(120)[0],
            times,
            data["positions"],
            data["headings"],
            method="binned",
            boundary_source="analysis",
        )
        kept = np.ones(len(times) - 1, dtype=bool)
        kept[drop[drop < len(times) - 1]] = False
        assert np.sum(result.occupancy) == pytest.approx(np.diff(times)[kept].sum())
        assert np.isfinite(np.nanmax(result.firing_rate))


def _walled_room(shift=0.0):
    """A 40 x 24 cm room of 4 cm bins whose outline is a physical wall."""
    from shapely.geometry import box

    from neurospatial.arena import OpenArena

    edges_x, edges_y = np.arange(0.0, 41.0, 4.0) + shift, np.arange(0.0, 25.0, 4.0)
    return Environment.from_grid_mask(
        np.ones((10, 6), dtype=bool),
        (edges_x, edges_y),
        units="cm",
        frame="arena",
        arena=OpenArena(box(shift, 0, shift + 40, 24)),
    )


@pytest.mark.parametrize("boundary_source", ["analysis", "physical"])
def test_ray_cast_view_rates_turn_with_a_rotated_room(boundary_source):
    """Rotating room, positions and headings together changes no viewed bin.

    The rotated room keeps its grid in a stored affine frame, so a gaze ray
    that ends on a wall is assigned to the same observer-side bin as in the
    original room. (A ray running exactly along a cell face is left out: the
    rotated heading and position round, so its end falls on either side of
    the corner it would reach.)
    """
    from neurospatial.encoding._view_counts import _precompute_view_bins
    from neurospatial.encoding.view import compute_view_rate
    from neurospatial.ops.transforms import AffineND

    env = _walled_room()
    rng = np.random.default_rng(0)
    positions = rng.uniform([1.0, 1.0], [39.0, 23.0], size=(400, 2))
    headings = rng.uniform(-np.pi, np.pi, size=400)
    times = np.arange(len(positions)) * 0.02
    spike_times = np.sort(np.random.default_rng(1).uniform(0.0, times[-1], 200))

    angle = 0.7
    turn = np.array(
        [
            [np.cos(angle), -np.sin(angle), 13.0],
            [np.sin(angle), np.cos(angle), -5.0],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = env.apply_transform(AffineND(turn))
    turned_positions = positions @ turn[:2, :2].T + turn[:2, 2]
    turned_headings = headings + angle

    bins = _precompute_view_bins(
        env, positions, headings, gaze_model="ray_cast", boundary_source=boundary_source
    )
    turned_bins = _precompute_view_bins(
        turned,
        turned_positions,
        turned_headings,
        gaze_model="ray_cast",
        boundary_source=boundary_source,
    )
    assert np.all(bins >= 0)
    np.testing.assert_array_equal(turned_bins, bins)

    settings = {
        "gaze_model": "ray_cast",
        "method": "binned",
        "boundary_source": boundary_source,
    }
    rates = compute_view_rate(env, spike_times, times, positions, headings, **settings)
    turned_rates = compute_view_rate(
        turned, spike_times, times, turned_positions, turned_headings, **settings
    )
    np.testing.assert_array_equal(turned_rates.occupancy, rates.occupancy)
    np.testing.assert_array_equal(turned_rates.firing_rate, rates.firing_rate)


def test_physical_ray_cast_refuses_a_combined_room_with_its_own_frame():
    """Physical wall bins need one source frame; a rotated component has its own."""
    from neurospatial.encoding._view_counts import _precompute_view_bins
    from neurospatial.ops.transforms import AffineND

    angle = 0.4
    turn = np.array(
        [
            [np.cos(angle), -np.sin(angle), 60.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = _walled_room().apply_transform(AffineND(turn))
    combined = Environment.combine(
        {"a": _walled_room(), "b": turned.with_metadata(frame="arena")},
        auto_bridge=False,
    )
    positions = np.array([[20.0, 12.0], [10.0, 6.0]])
    headings = np.array([0.3, 2.0])
    analysis = _precompute_view_bins(
        combined, positions, headings, gaze_model="ray_cast", boundary_source="analysis"
    )
    assert np.all(analysis >= 0)
    with pytest.raises(NotImplementedError, match="shared source frame"):
        _precompute_view_bins(
            combined,
            positions,
            headings,
            gaze_model="ray_cast",
            boundary_source="physical",
        )


def _u_maze() -> Environment:
    """Two 10 cm wide arms joined at the bottom, with a 10 cm divider between."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    outline = unary_union([box(0, 0, 10, 100), box(20, 0, 30, 100), box(0, 0, 30, 10)])
    return Environment.from_polygon(outline, bin_size=2.0, units="cm")


@pytest.mark.parametrize(
    ("gaze_model", "settings"),
    [("fixed_distance", {"view_distance": 17.0}), ("ray_cast", {})],
)
def test_gaze_stops_at_the_first_wall(gaze_model, settings):
    """An animal facing the divider views the divider, never the far arm.

    The left arm spans x in [0, 10] and the right arm x in [20, 30]. Facing
    East from x in [3, 7], a 17 cm gaze would land in the right arm if it
    passed through the divider at x = 10.
    """
    from neurospatial.encoding.view import compute_view_rate

    env = _u_maze()
    rng = np.random.default_rng(0)
    times = np.arange(600) / 30.0
    positions = np.c_[rng.uniform(3, 7, 600), rng.uniform(40, 60, 600)]
    result = compute_view_rate(
        env,
        [],
        times,
        positions,
        np.zeros(600),
        gaze_model=gaze_model,
        method="binned",
        **settings,
    )
    x = env.bin_centers[:, 0]
    occupancy = np.asarray(result.occupancy)
    assert occupancy[x > 10].sum() == 0.0
    # All viewing time lands in the bins against the divider (centers x = 9).
    assert occupancy[np.isclose(x, 9.0)].sum() == pytest.approx(times[-1])


def test_fixed_distance_short_of_the_wall_views_open_space():
    """A gaze shorter than the distance to the wall is not pulled to the wall."""
    from neurospatial.encoding.view import compute_view_rate

    env = _u_maze()
    times = np.arange(300) / 30.0
    positions = np.tile([[3.0, 51.0]], (300, 1))
    result = compute_view_rate(
        env,
        [],
        times,
        positions,
        np.zeros(300),
        gaze_model="fixed_distance",
        view_distance=4.0,
        method="binned",
    )
    viewed = env.bin_centers[np.asarray(result.occupancy) > 0]
    np.testing.assert_allclose(viewed, [[7.0, 51.0]])


@pytest.fixture(scope="module")
def left_gaze_cell():
    """A cell that fires while the animal looks LEFT at the middle of the North wall.

    A 100 x 100 cm box; 10 minutes of uniform positions and headings at
    30 Hz. The viewed point is the first wall along heading + pi/2, and the
    rate is a 10 cm Gaussian around (50, 100) on that wall.
    """
    from shapely.geometry import box

    from neurospatial.ops.visibility import compute_viewed_location
    from neurospatial.simulation import generate_poisson_spikes

    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
    rng = np.random.default_rng(0)
    times = np.arange(0, 600, 1 / 30)
    positions = rng.uniform(5, 95, (len(times), 2))
    headings = rng.uniform(-np.pi, np.pi, len(times))
    viewed = compute_viewed_location(
        positions,
        headings,
        method="ray_cast",
        env=env,
        gaze_offsets=np.full(len(times), np.pi / 2),
    )
    target = np.array([50.0, 100.0])
    rate = 0.5 + 20 * np.exp(-0.5 * (np.linalg.norm(viewed - target, axis=1) / 10) ** 2)
    spikes = generate_poisson_spikes(rate, times, seed=7)
    return env, spikes, times, positions, headings, target


def test_gaze_offsets_recover_a_left_gaze_cell_only_on_the_left(left_gaze_cell):
    """+pi/2 is left of the heading: it recovers the viewed spot; -pi/2 does not."""
    from neurospatial.encoding.view import compute_view_rate

    env, spikes, times, positions, headings, target = left_gaze_cell
    left, right = (
        compute_view_rate(
            env,
            spikes,
            times,
            positions,
            headings,
            gaze_offsets=np.full(len(times), offset),
            method="binned",
            min_occupancy=1.0,
        )
        for offset in (np.pi / 2, -np.pi / 2)
    )
    assert np.linalg.norm(left.peak_location() - target) <= 10.0
    assert np.linalg.norm(right.peak_location() - target) > 20.0
    assert left.spatial_information() > 3 * right.spatial_information()


def test_view_rate_peaks_at_the_viewed_location_not_the_position(left_gaze_cell):
    """The position map of a view cell has no field at the viewed spot."""
    from neurospatial.encoding import compute_spatial_rate
    from neurospatial.encoding.view import compute_view_rate

    env, spikes, times, positions, headings, _target = left_gaze_cell
    view = compute_view_rate(
        env,
        spikes,
        times,
        positions,
        headings,
        gaze_offsets=np.full(len(times), np.pi / 2),
        method="binned",
        min_occupancy=1.0,
    )
    place = compute_spatial_rate(
        env, spikes, times, positions, method="binned", min_occupancy=1.0
    )
    assert view.spatial_information() > 5 * place.spatial_information()
