"""Tests for compute_egocentric_rate(None) function.

This module tests the compute_egocentric_rate(None) function for computing egocentric
firing rate maps for object-vector cells.

Task 5.7: Implement compute_egocentric_rate(None) function
- Accept single spike_times array
- Accept object_positions array
- Support distance_range and bin count parameters
- Support metric parameter
- Optional env parameter (required for geodesic)
- Apply smoothing via _smoothing.py
- Return EgocentricRateResult
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
    """Create a simple 2D environment for testing (required for geodesic)."""
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 100, 11)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def trajectory_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sample trajectory data (times, positions, headings)."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    times = np.linspace(0, 100, n_samples)
    # Random walk within environment
    positions = np.column_stack(
        [
            50 + np.cumsum(rng.standard_normal(n_samples) * 0.5),
            50 + np.cumsum(rng.standard_normal(n_samples) * 0.5),
        ]
    )
    positions = np.clip(positions, 10, 90)  # Keep within bounds
    # Random headings
    headings = rng.uniform(-np.pi, np.pi, n_samples)
    return times, positions, headings


@pytest.fixture
def object_positions() -> np.ndarray:
    """Object positions for testing."""
    return np.array([[50.0, 50.0], [25.0, 75.0]])


@pytest.fixture
def spike_times() -> np.ndarray:
    """Spike times within the trajectory time range."""
    rng = np.random.default_rng(123)
    return np.sort(rng.uniform(0.1, 99.9, 100))


@pytest.fixture
def empty_spike_times() -> np.ndarray:
    """Empty spike train."""
    return np.array([], dtype=np.float64)


# ==============================================================================
# Import Tests
# ==============================================================================


class TestComputeEgocentricRateImport:
    """Test that compute_egocentric_rate can be imported."""

    def test_import_from_egocentric(self) -> None:
        """compute_egocentric_rate can be imported from encoding.egocentric."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        assert compute_egocentric_rate is not None
        assert callable(compute_egocentric_rate)

    def test_import_from_encoding(self) -> None:
        """compute_egocentric_rate can be imported from encoding package."""
        from neurospatial.encoding import compute_egocentric_rate

        assert compute_egocentric_rate is not None
        assert callable(compute_egocentric_rate)

    def test_in_all(self) -> None:
        """compute_egocentric_rate is in __all__."""
        from neurospatial.encoding import egocentric

        assert "compute_egocentric_rate" in egocentric.__all__


# ==============================================================================
# Return Type Tests
# ==============================================================================


class TestComputeEgocentricRateReturnsResult:
    """Test that compute_egocentric_rate returns EgocentricRateResult."""

    def test_returns_egocentric_rate_result(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """compute_egocentric_rate should return an EgocentricRateResult object."""
        from neurospatial.encoding.egocentric import (
            EgocentricRateResult,
            compute_egocentric_rate,
        )

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert isinstance(result, EgocentricRateResult)

    def test_firing_rate_shape(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """firing_rate should have shape (n_bins,)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert np.asarray(result.firing_rate).ndim == 1
        assert np.asarray(result.firing_rate).shape == (result.env.n_bins,)

    def test_occupancy_shape(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """occupancy should have shape (n_bins,)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert np.asarray(result.occupancy).shape == (result.env.n_bins,)

    def test_env_is_environment(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """env should be an Environment."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        from neurospatial.environment.polar import EgocentricPolarEnvironment

        # Egocentric rate runs on a distinct polar environment type, not the
        # Cartesian Environment.
        assert isinstance(result.env, EgocentricPolarEnvironment)
        assert not isinstance(result.env, Environment)


# ==============================================================================
# Distance Range and Bin Count Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateDistanceRange:
    """Test distance_range parameter."""

    def test_default_distance_range(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default distance_range should be (0.0, 50.0)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert result.distance_range == (0.0, 50.0)

    def test_custom_distance_range(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Custom distance_range should be stored in result."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(5.0, 60.0),
        )
        assert result.distance_range == (5.0, 60.0)


class TestComputeEgocentricRateBinCounts:
    """Test n_distance_bins and n_direction_bins parameters."""

    def test_default_n_distance_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default n_distance_bins should be 10."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert result.n_distance_bins == 10

    def test_default_n_direction_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default n_direction_bins should be 12."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert result.n_direction_bins == 12

    def test_custom_n_distance_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Custom n_distance_bins should be stored in result."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            n_distance_bins=20,
        )
        assert result.n_distance_bins == 20

    def test_custom_n_direction_bins(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Custom n_direction_bins should be stored in result."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            n_direction_bins=24,
        )
        assert result.n_direction_bins == 24

    def test_n_bins_equals_product(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """env.n_bins should equal n_distance_bins * n_direction_bins."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        n_dist = 8
        n_dir = 16
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            n_distance_bins=n_dist,
            n_direction_bins=n_dir,
        )
        assert result.env.n_bins == n_dist * n_dir


# ==============================================================================
# Distance Metric Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateDistanceMetric:
    """Test metric parameter."""

    def test_default_metric_euclidean(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default metric should be 'euclidean' (no env needed)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        # Should work without env parameter
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        assert isinstance(result.firing_rate, np.ndarray) or hasattr(
            result.firing_rate, "__array__"
        )

    def test_euclidean_metric_explicit(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Explicit euclidean metric works without env."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="euclidean",
        )
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_geodesic_requires_env(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """geodesic metric requires env parameter."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        with pytest.raises(ValueError, match=r"geodesic.*requires.*env"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                metric="geodesic",
            )

    def test_geodesic_with_env(
        self,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """geodesic metric works with env parameter."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="geodesic",
        )
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_invalid_metric_raises(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Invalid metric raises ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        with pytest.raises(ValueError, match="Invalid metric"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
                metric="manhattan",  # type: ignore[arg-type]
            )


# ==============================================================================
# Smoothing Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateSmoothing:
    """Test smoothing parameters."""

    @pytest.mark.parametrize(
        "smoothing_method", ["diffusion_kde", "gaussian_kde", "binned"]
    )
    def test_accepts_valid_smoothing_methods(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
        smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    ) -> None:
        """compute_egocentric_rate should accept all valid smoothing methods."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            smoothing_method=smoothing_method,
        )
        assert np.asarray(result.firing_rate).shape[0] > 0

    def test_default_smoothing_method(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default smoothing_method should be 'binned'."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        explicit_result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            smoothing_method="binned",
        )

        np.testing.assert_allclose(
            np.asarray(result.firing_rate),
            np.asarray(explicit_result.firing_rate),
            equal_nan=True,
        )

    def test_bandwidth_parameter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """bandwidth parameter should be accepted."""
        from neurospatial.encoding import _smoothing
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        captured: dict[str, object] = {}
        original_smooth_rate_map = _smoothing.smooth_rate_map

        def capture_smooth_rate_map(*args, **kwargs):
            captured["method"] = kwargs["method"]
            captured["bandwidth"] = kwargs["bandwidth"]
            return original_smooth_rate_map(*args, **kwargs)

        monkeypatch.setattr(_smoothing, "smooth_rate_map", capture_smooth_rate_map)

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            smoothing_method="gaussian_kde",
            bandwidth=10.0,
        )
        assert np.asarray(result.firing_rate).shape[0] > 0
        assert captured == {"method": "gaussian_kde", "bandwidth": 10.0}


# ==============================================================================
# Empty Spike Train Tests
# ==============================================================================


class TestComputeEgocentricRateEmptySpikes:
    """Test behavior with empty spike train."""

    def test_empty_spikes_returns_zero_rate(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        empty_spike_times: np.ndarray,
    ) -> None:
        """Empty spike train should return zero firing rate."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            empty_spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        firing_rate = np.asarray(result.firing_rate)
        # All non-NaN values should be zero
        assert np.allclose(firing_rate[~np.isnan(firing_rate)], 0.0)

    def test_empty_spikes_has_positive_occupancy(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        empty_spike_times: np.ndarray,
    ) -> None:
        """Empty spike train should still have positive occupancy."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            empty_spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        occupancy = np.asarray(result.occupancy)
        assert np.sum(occupancy) > 0


class TestComputeEgocentricRatePrecomputation:
    """Regression tests for single-neuron precomputation efficiency."""

    def test_egocentric_coordinates_computed_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_env: Environment,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """compute_egocentric_rate should reuse coordinates for counts and occupancy."""
        from neurospatial.encoding import _egocentric_binning
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        call_count = 0
        original = _egocentric_binning._compute_egocentric_coords

        def counting_compute_egocentric_coords(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            _egocentric_binning,
            "_compute_egocentric_coords",
            counting_compute_egocentric_coords,
        )

        compute_egocentric_rate(
            simple_env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="geodesic",
        )

        assert call_count == 1


# ==============================================================================
# Correctness Tests
# ==============================================================================


class TestComputeEgocentricRateCorrectness:
    """Test correctness of computed values."""

    def test_firing_rate_non_negative(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Firing rate should be non-negative."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        firing_rate = np.asarray(result.firing_rate)
        # Exclude NaN values
        non_nan = firing_rate[~np.isnan(firing_rate)]
        assert np.all(non_nan >= 0)

    def test_occupancy_non_negative(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Occupancy should be non-negative."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        occupancy = np.asarray(result.occupancy)
        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximates_duration(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Total occupancy should approximately equal trajectory duration."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        duration = times[-1] - times[0]
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        occupancy = np.asarray(result.occupancy)
        total_occupancy = np.sum(occupancy)
        # Should be within 10% (some bins may be outside distance range)
        assert total_occupancy <= duration * 1.1
        assert total_occupancy > duration * 0.5  # At least half should be covered


# ==============================================================================
# Result Methods Tests
# ==============================================================================


class TestComputeEgocentricRateResultMethods:
    """Test that result methods work correctly."""

    def test_plot_returns_axes(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """plot() should return matplotlib Axes."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        import matplotlib.pyplot as plt

        ax = result.plot()
        assert ax is not None
        plt.close()

    def test_preferred_distance_returns_float(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """preferred_distance() should return a float."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        dist = result.preferred_distance()
        assert isinstance(dist, float)

    def test_preferred_direction_returns_float(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """preferred_direction() should return a float."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        direction = result.preferred_direction()
        assert isinstance(direction, float)

    def test_egocentric_spatial_information_returns_float(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """egocentric_spatial_information() should return a float."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        info = result.egocentric_spatial_information()
        assert isinstance(info, float)

    def test_is_ovc_returns_bool(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """is_object_vector_cell() should return a bool."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        is_object_vector_cell = result.is_object_vector_cell()
        assert isinstance(is_object_vector_cell, (bool, np.bool_))


# ==============================================================================
# min_occupancy Parameter Tests
# ==============================================================================


class TestComputeEgocentricRateMinOccupancy:
    """Test min_occupancy parameter."""

    def test_min_occupancy_threshold(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Higher min_occupancy should produce more NaN bins."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data

        # Compute with high min_occupancy threshold
        result_high = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            min_occupancy=5.0,  # High threshold
        )
        # Compute with no threshold
        result_low = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            min_occupancy=0.0,  # No threshold
        )

        # High threshold should have more NaNs (or equal)
        high_nan_count = np.sum(np.isnan(np.asarray(result_high.firing_rate)))
        low_nan_count = np.sum(np.isnan(np.asarray(result_low.firing_rate)))
        assert high_nan_count >= low_nan_count

    def test_default_min_occupancy_zero(
        self,
        trajectory_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Default min_occupancy should be 0.0."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times, positions, headings = trajectory_data
        # Should work without specifying min_occupancy
        result = compute_egocentric_rate(
            None,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
        )
        # With min_occupancy=0, bins with occupancy>0 should have values
        occupancy = np.asarray(result.occupancy)
        firing_rate = np.asarray(result.firing_rate)
        occupied_mask = occupancy > 0
        # At least some occupied bins should have non-NaN values
        assert np.any(~np.isnan(firing_rate[occupied_mask]))


# ==============================================================================
# Input Validation Tests
# ==============================================================================


class TestComputeEgocentricRateInputValidation:
    """Test input validation."""

    def test_mismatched_times_positions_raises(
        self,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched times and positions lengths should raise ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times = np.linspace(0, 100, 1000)
        rng = np.random.default_rng(42)
        positions = rng.random((500, 2)) * 100  # Wrong length
        headings = rng.uniform(-np.pi, np.pi, 1000)

        with pytest.raises(ValueError, match=r"times.*positions"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
            )

    def test_mismatched_times_headings_raises(
        self,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ) -> None:
        """Mismatched times and headings lengths should raise ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        times = np.linspace(0, 100, 1000)
        rng = np.random.default_rng(42)
        positions = rng.random((1000, 2)) * 100
        headings = rng.uniform(-np.pi, np.pi, 500)  # Wrong length

        with pytest.raises(ValueError, match=r"times.*headings"):
            compute_egocentric_rate(
                None,
                spike_times,
                times,
                positions,
                headings,
                object_positions,
            )


# ==============================================================================
# Signature Tests
# ==============================================================================


class TestComputeEgocentricRateSignature:
    """Test function signature follows conventions."""

    def test_argument_order(self) -> None:
        """Arguments should follow canonical order from CLAUDE.md (env first)."""
        import inspect

        from neurospatial.encoding.egocentric import compute_egocentric_rate

        sig = inspect.signature(compute_egocentric_rate)
        params = list(sig.parameters.keys())

        # First 6 positional args should be in this order (env first per CLAUDE.md)
        assert params[0] == "env"
        assert params[1] == "spike_times"
        assert params[2] == "times"
        assert params[3] == "positions"
        assert params[4] == "headings"
        assert params[5] == "object_positions"

    def test_keyword_only_parameters(self) -> None:
        """Optional parameters should be keyword-only."""
        import inspect

        from neurospatial.encoding.egocentric import compute_egocentric_rate

        sig = inspect.signature(compute_egocentric_rate)

        # These should be keyword-only (after *).
        # `env` is now first positional (canonical order); excluded here.
        keyword_only_params = [
            "distance_range",
            "n_distance_bins",
            "n_direction_bins",
            "metric",
            "smoothing_method",
            "bandwidth",
            "min_occupancy",
        ]

        for param_name in keyword_only_params:
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"{param_name} should be keyword-only"
                )


class TestEgocentricRateConvention:
    """End-to-end recovery of the egocentric direction and distance tuning.

    Drives the public ``compute_egocentric_rate`` on Poisson spikes from an
    object-vector cell with a known egocentric ``preferred_direction`` and
    ``preferred_distance`` and asserts the recovered peak matches both within
    one bin. This pins the shared egocentric convention (0=ahead, +pi/2=left,
    -pi/2=right) between the simulator and the analysis: a sign flip in
    ``compute_egocentric_bearing`` or a swapped left/right convention would
    fail at least one parametrization.

    The animal moves on an OU trajectory (the analysis needs many sampled
    egocentric configurations; a stationary animal collapses occupancy into a
    single bin and the recovered map is dominated by smoothing artifacts).

    Distance recovery exercises the default ``"binned"`` method, which reports
    the raw bin rate (spikes / occupancy). Graph-diffusion smoothing on the
    polar environment bleeds rate radially across distance rings and flattens
    the distance marginal, so it cannot recover ``preferred_distance``; the
    binned rate can.
    """

    N_DIRECTION_BINS = 12
    DISTANCE_RANGE = (0.0, 50.0)

    @pytest.fixture(scope="class")
    def trajectory(self):
        from neurospatial import Environment
        from neurospatial.ops.egocentric import heading_from_velocity
        from neurospatial.simulation import simulate_trajectory_ou

        samples = np.random.default_rng(0).uniform(0, 40, (3000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)
        env.units = "cm"
        positions, times = simulate_trajectory_ou(
            env, duration=1500.0, speed_units="cm", seed=7
        )
        dt = float(times[1] - times[0])
        headings = heading_from_velocity(positions, dt, min_speed=2.0)
        return env, positions, times, headings

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "preferred_direction",
        [0.0, np.pi / 2, -np.pi / 2],
        ids=["ahead", "left", "right"],
    )
    def test_preferred_direction_recovered(self, trajectory, preferred_direction):
        from neurospatial.encoding.egocentric import compute_egocentric_rate
        from neurospatial.simulation import generate_poisson_spikes
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        env, positions, times, headings = trajectory
        object_positions = np.array([[20.0, 20.0]])  # object at arena center
        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=10.0,
            distance_width=4.0,
            preferred_direction=preferred_direction,
            direction_kappa=4.0,
            max_rate=25.0,
            baseline_rate=0.1,
            metric="euclidean",
        )
        rates = model.firing_rate(positions, headings=headings)
        # Independent seed from the trajectory RNG so spike-acceptance uniforms
        # are not correlated with the simulated state.
        spike_times = generate_poisson_spikes(rates, times, seed=99)

        result = compute_egocentric_rate(
            env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            metric="euclidean",
            distance_range=self.DISTANCE_RANGE,
            n_distance_bins=10,
            n_direction_bins=self.N_DIRECTION_BINS,
        )

        # Direction convention recovered within one direction bin.
        direction_bin = 2 * np.pi / self.N_DIRECTION_BINS
        circ_dist = abs(
            np.angle(np.exp(1j * (result.preferred_direction() - preferred_direction)))
        )
        assert circ_dist < direction_bin

        # Distance tuning recovered within one distance bin. DISTANCE_RANGE
        # spans 0-50 cm over 10 bins -> 5 cm/bin; simulated peak is 10 cm.
        # Observed recovery is ~12.5 cm (the 10-15 cm bin center), so the
        # one-bin tolerance is intentional: it is tight enough that the old
        # diffusion-smoothed behavior (peak collapsed to ~2.5 cm) fails it.
        distance_bin = (self.DISTANCE_RANGE[1] - self.DISTANCE_RANGE[0]) / 10
        assert abs(result.preferred_distance() - 10.0) < distance_bin


class TestEgocentricRateNaNHandling:
    """NaN positions/headings are treated as missing data, not errors.

    Per the compute_egocentric_rate docstring contract, NaN samples (tracking
    dropouts) are excluded rather than raising or corrupting the map.
    """

    @pytest.fixture(scope="class")
    def session(self):
        from neurospatial import Environment
        from neurospatial.ops.egocentric import heading_from_velocity
        from neurospatial.simulation import (
            generate_poisson_spikes,
            simulate_trajectory_ou,
        )
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        samples = np.random.default_rng(0).uniform(0, 40, (3000, 2))
        env = Environment.from_samples(samples, bin_size=2.0)
        env.units = "cm"
        positions, times = simulate_trajectory_ou(
            env, duration=400.0, speed_units="cm", seed=42
        )
        dt = float(times[1] - times[0])
        headings = heading_from_velocity(positions, dt, min_speed=2.0)
        object_positions = np.array([[20.0, 20.0]])
        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=10.0,
            distance_width=4.0,
            preferred_direction=0.0,
            direction_kappa=4.0,
            max_rate=25.0,
            baseline_rate=0.1,
            metric="euclidean",
        )
        spike_times = generate_poisson_spikes(
            model.firing_rate(positions, headings=headings), times, seed=7
        )
        return env, spike_times, times, positions, headings, object_positions

    @pytest.mark.parametrize("nan_field", ["positions", "headings"])
    def test_nan_inputs_excluded_not_raised(self, session, nan_field):
        from neurospatial.encoding.egocentric import compute_egocentric_rate

        env, spike_times, times, positions, headings, object_positions = session
        rng = np.random.default_rng(5)
        drop_idx = rng.choice(len(times), size=len(times) // 10, replace=False)
        positions_in = positions.copy()
        headings_in = headings.copy()
        if nan_field == "positions":
            positions_in[drop_idx] = np.nan
        else:
            headings_in[drop_idx] = np.nan

        result = compute_egocentric_rate(
            env,
            spike_times,
            times,
            positions_in,
            headings_in,
            object_positions,
            metric="euclidean",
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Did not raise; produced a usable (not all-NaN) map with a finite peak.
        assert not np.isnan(result.firing_rate).all()
        assert np.isfinite(np.nanmax(result.firing_rate))
