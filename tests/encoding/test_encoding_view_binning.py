"""Tests for neurospatial.encoding._view_counts module.

This module tests the view binning layer that converts spike trains
and gaze data to spike counts and view occupancy arrays, through the path
the encoders use: ``prepare_view_binning`` followed by ``FrameBinning.count``.

The binning layer:
- Computes view occupancy (time viewing each spatial bin)
- Supports gaze models: "fixed_distance", "ray_cast"
- Converts (spike_times, times, positions, headings, gaze_model) -> (spike_counts, occupancy)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.encoding._view_counts import prepare_view_binning
from neurospatial.encoding.view import compute_view_rate


def _bin(
    env: Environment,
    spike_trains: list[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Counts ``(n_neurons, n_bins)`` and view occupancy ``(n_bins,)``."""
    prepared = prepare_view_binning(env, times, positions, headings, **kwargs)
    return prepared.count(spike_trains, n_jobs=n_jobs)


def _counts(env, spikes, times, positions, headings, **kwargs):
    """One neuron's spike counts, shape ``(n_bins,)``."""
    return _bin(env, [spikes], times, positions, headings, **kwargs)[0][0]


def _occupancy(env, times, positions, headings, **kwargs):
    """View occupancy, shape ``(n_bins,)``."""
    return _bin(env, [], times, positions, headings, **kwargs)[1]


def _trajectory(data):
    return data["times"], data["positions"], data["headings"]


# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    # Create a 10x10 grid covering 0-100 in each dimension
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 100, 11)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def trajectory_data(simple_env: Environment) -> dict:
    """Create trajectory data with positions and headings for testing.

    Returns dict with:
        - times: array of timestamps (10 seconds at 100 Hz)
        - positions: array of positions (centered in environment)
        - headings: array of headings (radians, random walk)
    """
    np.random.seed(42)
    n_samples = 1000
    times = np.linspace(0, 10.0, n_samples)

    # Stay in center of environment so viewed locations are valid
    positions = np.zeros((n_samples, 2))
    positions[:, 0] = 50 + np.cumsum(np.random.randn(n_samples) * 0.5)
    positions[:, 1] = 50 + np.cumsum(np.random.randn(n_samples) * 0.5)
    positions = np.clip(positions, 20, 80)  # Stay away from edges

    # Random walk in heading
    headings = np.zeros(n_samples)
    headings[0] = 0.0
    for i in range(1, n_samples):
        headings[i] = headings[i - 1] + np.random.randn() * 0.1
    headings = headings % (2 * np.pi)

    return {"times": times, "positions": positions, "headings": headings}


@pytest.fixture
def single_neuron_spikes() -> NDArray[np.float64]:
    """Create spike times for a single neuron."""
    return np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])


@pytest.fixture
def multiple_neuron_spikes() -> list[NDArray[np.float64]]:
    """Create spike times for multiple neurons."""
    return [
        np.array([0.5, 1.5, 2.5]),  # Neuron 0: 3 spikes
        np.array([1.0, 2.0, 3.0, 4.0]),  # Neuron 1: 4 spikes
        np.array([0.25]),  # Neuron 2: 1 spike
        np.array([]),  # Neuron 3: 0 spikes (silent)
    ]


# ==============================================================================
# View occupancy
# ==============================================================================


class TestViewOccupancy:
    """Time spent viewing each spatial bin."""

    def test_occupancy_shape_and_dtype(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """View occupancy is a float64 array with shape (n_bins,)."""
        occupancy = _occupancy(
            simple_env, *_trajectory(trajectory_data), boundary_source="analysis"
        )

        assert isinstance(occupancy, np.ndarray)
        assert occupancy.dtype == np.float64
        assert occupancy.shape == (simple_env.n_bins,)

    def test_occupancy_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """View occupancy should all be non-negative."""
        occupancy = _occupancy(
            simple_env, *_trajectory(trajectory_data), boundary_source="analysis"
        )

        assert np.all(occupancy >= 0)

    def test_total_occupancy_less_than_duration(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Total view occupancy should be approximately equal to recording duration.

        Note: May differ slightly due to:
        - Viewed locations falling outside the environment (reduces occupancy)
        - Floating point accumulation in dt computation (may slightly exceed)
        """
        occupancy = _occupancy(
            simple_env, *_trajectory(trajectory_data), boundary_source="analysis"
        )

        total_duration = trajectory_data["times"][-1] - trajectory_data["times"][0]
        total_occupancy = np.sum(occupancy)

        # Allow 2% tolerance for floating point accumulation
        assert total_occupancy <= total_duration * 1.02

    def test_occupancy_differs_from_position_occupancy(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """View occupancy should differ from position occupancy.

        This tests the key concept: view occupancy is time *viewing* each bin,
        not time *at* each bin.
        """
        occupancy = _occupancy(
            simple_env, *_trajectory(trajectory_data), boundary_source="analysis"
        )

        # Compute position occupancy for comparison
        position_occupancy = simple_env.occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # They should be different arrays
        assert not np.allclose(occupancy, position_occupancy)


class TestViewGazeModels:
    """Tests for the gaze_model parameter."""

    def test_ray_cast_default(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """The default gaze model (ray_cast) views some bins."""
        counts, occupancy = _bin(
            simple_env,
            [single_neuron_spikes],
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        assert counts.shape == (1, simple_env.n_bins)
        assert occupancy.shape == (simple_env.n_bins,)
        assert np.any(occupancy > 0)  # Some bins should be viewed

    def test_fixed_distance_explicit(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Explicit 'fixed_distance' should work."""
        counts, occupancy = _bin(
            simple_env,
            [single_neuron_spikes],
            *_trajectory(trajectory_data),
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )

        assert counts.shape == (1, simple_env.n_bins)
        assert occupancy.shape == (simple_env.n_bins,)

    def test_different_gaze_models_give_different_results(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Different gaze models should produce different view occupancies."""
        occ_fixed = _occupancy(
            simple_env,
            *_trajectory(trajectory_data),
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )
        occ_ray = _occupancy(
            simple_env,
            *_trajectory(trajectory_data),
            gaze_model="ray_cast",
            boundary_source="analysis",
        )

        # Results should differ
        assert not np.allclose(occ_fixed, occ_ray)

    def test_view_distance_parameter(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Different view_distance values should give different results."""
        occ_short = _occupancy(
            simple_env,
            *_trajectory(trajectory_data),
            gaze_model="fixed_distance",
            view_distance=5.0,
            boundary_source="analysis",
        )
        occ_long = _occupancy(
            simple_env,
            *_trajectory(trajectory_data),
            gaze_model="fixed_distance",
            view_distance=20.0,
            boundary_source="analysis",
        )

        # Results should differ
        assert not np.allclose(occ_short, occ_long)


# ==============================================================================
# Spike counts for one neuron
# ==============================================================================


class TestViewSpikeCounts:
    """One neuron's spike counts per viewed bin."""

    def test_spike_counts_shape_and_dtype(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts are float64 with shape (n_bins,)."""
        spike_counts = _counts(
            simple_env,
            single_neuron_spikes,
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        assert isinstance(spike_counts, np.ndarray)
        assert spike_counts.dtype == np.float64
        assert spike_counts.shape == (simple_env.n_bins,)

    def test_spike_counts_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts should all be non-negative."""
        spike_counts = _counts(
            simple_env,
            single_neuron_spikes,
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        assert np.all(spike_counts >= 0)

    def test_total_spike_count(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Total spike count should be at most equal to number of valid spikes.

        Note: May be less if some viewed locations at spike times fall outside
        the environment.
        """
        spike_counts = _counts(
            simple_env,
            single_neuron_spikes,
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        # Filter spikes to valid time range
        times = trajectory_data["times"]
        valid_spikes = single_neuron_spikes[
            (single_neuron_spikes >= times.min())
            & (single_neuron_spikes <= times.max())
        ]

        # Total should be <= number of valid spikes
        assert np.sum(spike_counts) <= len(valid_spikes)

    def test_empty_spike_train(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Empty spike train should produce all-zero spike counts."""
        spike_counts = _counts(
            simple_env,
            np.array([]),
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        assert spike_counts.shape == (simple_env.n_bins,)
        assert np.all(spike_counts == 0)

    def test_spikes_outside_time_range(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Spikes outside time range are excluded, with a warning."""
        # Spikes at -1.0 and 20.0, both outside the 0-10 second range
        spikes_outside = np.array([-1.0, 20.0])

        with pytest.warns(UserWarning, match="outside"):
            spike_counts = _counts(
                simple_env,
                spikes_outside,
                *_trajectory(trajectory_data),
                boundary_source="analysis",
            )

        assert np.sum(spike_counts) == 0


# ==============================================================================
# Populations
# ==============================================================================


class TestViewPopulationCounts:
    """Spike counts for several neurons sharing one view occupancy."""

    def test_population_shapes_and_dtypes(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Counts are (n_neurons, n_bins); occupancy is (n_bins,); both float64."""
        spike_counts, occupancy = _bin(
            simple_env,
            multiple_neuron_spikes,
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        n_neurons = len(multiple_neuron_spikes)
        assert spike_counts.shape == (n_neurons, simple_env.n_bins)
        assert occupancy.shape == (simple_env.n_bins,)
        assert spike_counts.dtype == np.float64
        assert occupancy.dtype == np.float64

    def test_empty_neuron_has_zero_counts(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Neuron with no spikes should have all-zero counts."""
        spike_counts, _ = _bin(
            simple_env,
            multiple_neuron_spikes,
            *_trajectory(trajectory_data),
            boundary_source="analysis",
        )

        # Neuron 3 has no spikes (empty array)
        assert np.all(spike_counts[3] == 0)

    @pytest.mark.parametrize(
        "gaze",
        [{}, {"gaze_model": "fixed_distance", "view_distance": 10.0}],
        ids=["ray_cast", "fixed_distance"],
    )
    def test_each_row_matches_the_neuron_alone(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
        gaze: dict,
    ) -> None:
        """A neuron's counts do not depend on the other neurons counted with it."""
        spike_counts_population, _ = _bin(
            simple_env,
            multiple_neuron_spikes,
            *_trajectory(trajectory_data),
            **gaze,
            boundary_source="analysis",
        )

        for i, spikes in enumerate(multiple_neuron_spikes):
            spike_counts_single = _counts(
                simple_env,
                spikes,
                *_trajectory(trajectory_data),
                **gaze,
                boundary_source="analysis",
            )
            np.testing.assert_array_equal(
                spike_counts_population[i],
                spike_counts_single,
                err_msg=f"Neuron {i} counts differ",
            )

    def test_n_jobs_parameter(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """n_jobs parameter should not change results."""
        spike_counts_serial, occupancy_serial = _bin(
            simple_env,
            multiple_neuron_spikes,
            *_trajectory(trajectory_data),
            n_jobs=1,
            boundary_source="analysis",
        )
        spike_counts_parallel, occupancy_parallel = _bin(
            simple_env,
            multiple_neuron_spikes,
            *_trajectory(trajectory_data),
            n_jobs=2,
            boundary_source="analysis",
        )

        np.testing.assert_array_equal(spike_counts_serial, spike_counts_parallel)
        np.testing.assert_array_equal(occupancy_serial, occupancy_parallel)

    def test_view_bins_precomputed_once(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Viewed bins are one index per sample, shared by every neuron.

        Supplying them skips the gaze computation and gives the same binning.
        """
        from neurospatial.encoding._view_counts import _precompute_view_bins

        view_bins = _precompute_view_bins(
            simple_env,
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )

        # Should return bin indices for each timepoint
        assert view_bins.shape == (len(trajectory_data["times"]),)
        assert view_bins.dtype == np.intp
        # Invalid views should be -1
        assert np.all((view_bins >= -1) & (view_bins < simple_env.n_bins))

        gaze = {"gaze_model": "fixed_distance", "view_distance": 10.0}
        spikes = [np.array([0.5, 1.5, 2.5])]
        computed = _bin(
            simple_env,
            spikes,
            *_trajectory(trajectory_data),
            **gaze,
            boundary_source="analysis",
        )
        supplied = _bin(
            simple_env,
            spikes,
            *_trajectory(trajectory_data),
            **gaze,
            view_bins=view_bins,
            boundary_source="analysis",
        )
        for a, b in zip(computed, supplied, strict=True):
            np.testing.assert_array_equal(a, b)


# ==============================================================================
# Test edge cases and error handling
# ==============================================================================


class TestViewBinningEdgeCases:
    """Tests for edge cases in view binning layer."""

    def test_constant_heading_forward(
        self,
        simple_env: Environment,
    ) -> None:
        """Constant heading should view along one direction."""
        times = np.linspace(0, 10.0, 1000)
        # Animal at center, looking east (heading=0)
        positions = np.tile([50, 50], (1000, 1))
        headings = np.zeros(1000)  # Constant heading east

        occupancy = _occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )

        # Every sample views (60, 50), 10 cm east of the animal.
        viewed_bin = simple_env.bin_at(np.array([[60.0, 50.0]]))[0]
        assert occupancy[viewed_bin] == pytest.approx(times[-1] - times[0])
        assert np.count_nonzero(occupancy) == 1

    def test_gaze_toward_the_edge_stops_at_the_edge(
        self,
        simple_env: Environment,
    ) -> None:
        """A gaze that would leave the environment views its edge instead."""
        times = np.linspace(0, 10.0, 1000)
        positions = np.tile([5.0, 50.0], (1000, 1))
        headings = np.full(1000, np.pi)  # West; 20 cm ahead is outside

        occupancy = _occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=20.0,
            boundary_source="analysis",
        )

        viewed = simple_env.bin_centers[occupancy > 0]
        assert len(viewed) == 1
        np.testing.assert_allclose(
            viewed[0],
            simple_env.bin_centers[simple_env.bin_at(np.array([[0.0, 50.0]]))[0]],
        )
        assert np.sum(occupancy) == pytest.approx(times[-1] - times[0])

    def test_all_spikes_at_same_viewed_location(
        self,
        simple_env: Environment,
    ) -> None:
        """Spikes at times with same view should be binned together."""
        times = np.linspace(0, 10.0, 1000)
        # Constant position and heading
        positions = np.tile([50, 50], (1000, 1))
        headings = np.zeros(1000)  # Looking east

        # Multiple spikes at various times
        spikes = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        spike_counts = _counts(
            simple_env,
            spikes,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )

        # All 5 spikes should be in one bin
        assert np.sum(spike_counts) == 5
        assert np.max(spike_counts) == 5


class TestViewBinningInputValidation:
    """Tests for input validation of view binning."""

    def test_mismatched_times_positions_length(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error if times and positions have different lengths."""
        times = np.linspace(0, 10, 100)
        positions = np.random.rand(50, 2) * 100  # Different length
        headings = np.random.uniform(0, 2 * np.pi, 100)

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_view_rate(
                simple_env,
                np.array([1.0]),
                times,
                positions,
                headings,
                boundary_source="analysis",
            )

    def test_mismatched_times_headings_length(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error if times and headings have different lengths."""
        times = np.linspace(0, 10, 100)
        positions = np.random.rand(100, 2) * 100
        headings = np.random.uniform(0, 2 * np.pi, 50)  # Different length

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_view_rate(
                simple_env,
                np.array([1.0]),
                times,
                positions,
                headings,
                boundary_source="analysis",
            )

    def test_invalid_gaze_model(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should raise error for invalid gaze_model."""
        with pytest.raises(ValueError, match=r"gaze_model|invalid"):
            compute_view_rate(
                simple_env,
                np.array([1.0]),
                *_trajectory(trajectory_data),
                gaze_model="invalid_model",  # type: ignore[arg-type]
                boundary_source="analysis",
            )

    def test_insufficient_samples(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error with fewer than 2 samples."""
        times = np.array([0.0])  # Only 1 sample - need at least 2
        positions = np.array([[50, 50]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="At least 2 samples"):
            prepare_view_binning(
                simple_env, times, positions, headings, boundary_source="analysis"
            )

    def test_unsorted_times_raises_error(
        self,
        simple_env: Environment,
    ) -> None:
        """Unsorted times should raise ValueError.

        searchsorted assumes sorted input. Unsorted times will silently
        produce incorrect results without validation.
        """
        times = np.array([0.0, 2.0, 1.0, 3.0])  # Not sorted
        positions = np.tile([50, 50], (4, 1))
        headings = np.zeros(4)

        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            prepare_view_binning(
                simple_env, times, positions, headings, boundary_source="analysis"
            )

    def test_duplicate_times_allowed(
        self,
        simple_env: Environment,
    ) -> None:
        """Duplicate times (monotonically non-decreasing) should be allowed.

        Times like [0, 1, 1, 2] are valid (non-decreasing but not strictly
        increasing). The duplicate contributes 0 seconds.
        """
        times = np.array([0.0, 1.0, 1.0, 2.0])  # Duplicate at t=1
        positions = np.tile([50, 50], (4, 1))
        headings = np.zeros(4)

        occupancy = _occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            max_gap=None,
            boundary_source="analysis",  # Coarse sampling is incidental to this test.
        )

        # 3 intervals: 1s, 0s, 1s = 2s total
        total_occupancy = np.sum(occupancy)
        assert abs(total_occupancy - 2.0) < 0.01


# ==============================================================================
# Tests for bugfixes from code review (2025-12-19)
# ==============================================================================


class TestViewOccupancyNonUniformSampling:
    """Tests for correct handling of non-uniform time sampling.

    Bug: occupancy used median(dt) constant instead of per-sample deltas.
    This inflates occupancy when sampling is non-uniform.
    """

    def test_nonuniform_times_uses_actual_deltas(
        self,
        simple_env: Environment,
    ) -> None:
        """View occupancy should use actual time deltas, not median.

        With non-uniform sampling, using median(dt) instead of actual dt
        gives incorrect total occupancy.
        """
        # Create non-uniform time samples: first half at 10Hz, second half at 100Hz
        times_slow = np.linspace(0, 5, 51)  # 50 intervals of 0.1s
        times_fast = np.linspace(5.01, 10, 500)  # 499 intervals of ~0.01s
        times = np.concatenate([times_slow, times_fast])

        n_samples = len(times)
        positions = np.tile([50, 50], (n_samples, 1))  # Stationary
        headings = np.zeros(n_samples)  # Looking east

        occupancy = _occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )

        # Expected total: sum of all actual time deltas
        # = (5-0) + (10-5.01) = 5 + 4.99 = ~9.99 seconds
        total_duration = times[-1] - times[0]  # ~10s
        total_occupancy = np.sum(occupancy)

        # With correct implementation, total should equal sum of deltas
        # (which equals times[-1] - times[0] minus any invalid frames)
        # Allow 1% tolerance for floating point
        assert abs(total_occupancy - total_duration) < total_duration * 0.01, (
            f"Expected occupancy ~{total_duration:.3f}s, got {total_occupancy:.3f}s. "
            "This may indicate using median(dt) instead of actual deltas."
        )

    def test_last_frame_not_counted(
        self,
        simple_env: Environment,
    ) -> None:
        """The last frame should not contribute to occupancy.

        Occupancy uses (n-1) intervals from n samples. The last sample
        has no following interval to contribute.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 5 samples, 4 intervals
        positions = np.tile([50, 50], (5, 1))
        headings = np.zeros(5)

        occupancy = _occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            max_gap=None,
            boundary_source="analysis",  # Coarse sampling is incidental to this test.
        )

        # With 4 intervals of 1s each, total should be 4s (not 5s)
        total_occupancy = np.sum(occupancy)
        assert abs(total_occupancy - 4.0) < 0.01, (
            f"Expected 4.0s (4 intervals), got {total_occupancy:.3f}s. "
            "Last frame may be incorrectly contributing to occupancy."
        )


class TestGazeModelGeometry:
    """Geometric correctness of the gaze models used by the view-binning layer.

    The view binning maps a viewed location (computed by
    ``compute_viewed_location`` with the chosen ``gaze_model``) into a spatial
    bin. These tests pin the geometry the audit found only smoke-tested:
    ``ray_cast`` clips the gaze ray at the environment wall (valid
    bins), while ``fixed_distance`` projects a fixed distance ahead and can land
    outside the environment (bin index -1).
    """

    @pytest.fixture(scope="class")
    def rectangular_env(self) -> Environment:
        rng = np.random.default_rng(0)
        return Environment.from_samples(rng.uniform(0, 100, (8000, 2)), bin_size=5.0)

    def test_boundary_clips_at_wall_fixed_distance_overshoots(self, rectangular_env):
        from neurospatial.ops.visibility import compute_viewed_location

        # Poses hugging the left wall, all facing -x toward the x=0 boundary.
        positions = np.column_stack([np.linspace(3.0, 8.0, 12), np.full(12, 50.0)])
        headings = np.full(12, np.pi)  # facing -x (West)

        viewed_boundary = compute_viewed_location(
            positions,
            headings,
            method="ray_cast",
            env=rectangular_env,
            max_distance=150.0,
            boundary_source="analysis",
        )
        viewed_fixed = compute_viewed_location(
            positions,
            headings,
            method="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        )

        bins_boundary = rectangular_env.bin_at(viewed_boundary)
        bins_fixed = rectangular_env.bin_at(viewed_fixed)

        # boundary clips at the wall -> every viewed location is a valid bin.
        assert np.all(bins_boundary >= 0)
        # fixed_distance projects 10 cm past the wall -> mostly outside the env.
        assert np.mean(bins_fixed < 0) >= 0.5
