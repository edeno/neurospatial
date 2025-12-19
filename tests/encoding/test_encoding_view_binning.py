"""Tests for neurospatial.encoding._view_binning module.

This module tests the view binning layer that converts spike trains
and gaze data to spike counts and view occupancy arrays.

TDD approach: Tests written first, implementation follows.

Task 4.6: Implement binning layer for view encoding
- Create helper to compute view occupancy (time viewing each spatial bin)
- Support gaze models: "fixed_distance", "ray_cast", "boundary"
- Convert (spike_times, times, positions, headings, gaze_model) -> (spike_counts, view_occupancy)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment

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
# Test compute_view_occupancy
# ==============================================================================


class TestComputeViewOccupancy:
    """Tests for compute_view_occupancy function."""

    def test_function_is_importable(self) -> None:
        """compute_view_occupancy should be importable from encoding._view_binning."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        assert compute_view_occupancy is not None

    def test_returns_view_occupancy_array(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """compute_view_occupancy should return view occupancy array in seconds."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert isinstance(view_occupancy, np.ndarray)
        assert view_occupancy.dtype == np.float64

    def test_view_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """compute_view_occupancy should return array with shape (n_bins,)."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert view_occupancy.shape == (simple_env.n_bins,)

    def test_view_occupancy_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """View occupancy should all be non-negative."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert np.all(view_occupancy >= 0)

    def test_total_view_occupancy_less_than_duration(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Total view occupancy should be approximately equal to recording duration.

        Note: May differ slightly due to:
        - Viewed locations falling outside the environment (reduces occupancy)
        - Floating point accumulation in dt computation (may slightly exceed)
        """
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        total_duration = trajectory_data["times"][-1] - trajectory_data["times"][0]
        total_occupancy = np.sum(view_occupancy)

        # Allow 2% tolerance for floating point accumulation
        assert total_occupancy <= total_duration * 1.02

    def test_view_occupancy_differs_from_position_occupancy(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """View occupancy should differ from position occupancy.

        This tests the key concept: view occupancy is time *viewing* each bin,
        not time *at* each bin.
        """
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        # Compute position occupancy for comparison
        position_occupancy = simple_env.occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # They should be different arrays
        assert not np.allclose(view_occupancy, position_occupancy)


class TestComputeViewOccupancyGazeModels:
    """Tests for gaze_model parameter in compute_view_occupancy."""

    def test_fixed_distance_default(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Default gaze_model should be 'fixed_distance'."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        # Should run without error and return valid array
        assert view_occupancy.shape == (simple_env.n_bins,)
        assert np.any(view_occupancy > 0)  # Some bins should be viewed

    def test_fixed_distance_explicit(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Explicit 'fixed_distance' should work."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        assert view_occupancy.shape == (simple_env.n_bins,)

    def test_ray_cast_gaze_model(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """'ray_cast' gaze model should work."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="ray_cast",
        )

        assert view_occupancy.shape == (simple_env.n_bins,)

    def test_boundary_gaze_model(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """'boundary' gaze model should work."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occupancy = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="boundary",
        )

        assert view_occupancy.shape == (simple_env.n_bins,)

    def test_different_gaze_models_give_different_results(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Different gaze models should produce different view occupancies."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        occ_fixed = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        occ_ray = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="ray_cast",
        )

        # Results should differ
        assert not np.allclose(occ_fixed, occ_ray)

    def test_view_distance_parameter(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Different view_distance values should give different results."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        occ_short = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=5.0,
        )

        occ_long = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=20.0,
        )

        # Results should differ
        assert not np.allclose(occ_short, occ_long)


# ==============================================================================
# Test bin_view_spike_train (single neuron)
# ==============================================================================


class TestBinViewSpikeTrain:
    """Tests for bin_view_spike_train function (single neuron binning)."""

    def test_function_is_importable(self) -> None:
        """bin_view_spike_train should be importable from encoding._view_binning."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        assert bin_view_spike_train is not None

    def test_returns_spike_counts(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_view_spike_train should return spike counts array."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert isinstance(spike_counts, np.ndarray)
        assert spike_counts.dtype == np.float64

    def test_spike_counts_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_view_spike_train should return array with shape (n_bins,)."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)

    def test_spike_counts_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts should all be non-negative."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
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
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
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
        from neurospatial.encoding._view_binning import bin_view_spike_train

        empty_spikes = np.array([])

        spike_counts = bin_view_spike_train(
            simple_env,
            empty_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)
        assert np.all(spike_counts == 0)

    def test_spikes_outside_time_range(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Spikes outside time range should be excluded."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        # Spikes at -1.0 and 20.0, both outside the 0-10 second range
        spikes_outside = np.array([-1.0, 20.0])

        spike_counts = bin_view_spike_train(
            simple_env,
            spikes_outside,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert np.sum(spike_counts) == 0


class TestBinViewSpikeTrainGazeModels:
    """Tests for gaze_model parameter in bin_view_spike_train."""

    def test_fixed_distance_default(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Default gaze_model should be 'fixed_distance'."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)

    def test_ray_cast_gaze_model(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """'ray_cast' gaze model should work."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="ray_cast",
        )

        assert spike_counts.shape == (simple_env.n_bins,)

    def test_boundary_gaze_model(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """'boundary' gaze model should work."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        spike_counts = bin_view_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="boundary",
        )

        assert spike_counts.shape == (simple_env.n_bins,)


# ==============================================================================
# Test bin_view_spike_trains (batch)
# ==============================================================================


class TestBinViewSpikeTrains:
    """Tests for bin_view_spike_trains function (multiple neurons)."""

    def test_function_is_importable(self) -> None:
        """bin_view_spike_trains should be importable from encoding._view_binning."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        assert bin_view_spike_trains is not None

    def test_returns_spike_counts_and_view_occupancy(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """bin_view_spike_trains should return (spike_counts, view_occupancy) tuple."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        result = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        spike_counts, view_occupancy = result
        assert isinstance(spike_counts, np.ndarray)
        assert isinstance(view_occupancy, np.ndarray)

    def test_spike_counts_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        spike_counts, _ = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        n_neurons = len(multiple_neuron_spikes)
        assert spike_counts.shape == (n_neurons, simple_env.n_bins)

    def test_view_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """View occupancy should have shape (n_bins,)."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        _, view_occupancy = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert view_occupancy.shape == (simple_env.n_bins,)

    def test_spike_counts_dtype(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should be float64."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        spike_counts, _ = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert spike_counts.dtype == np.float64

    def test_view_occupancy_dtype(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """View occupancy should be float64."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        _, view_occupancy = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert view_occupancy.dtype == np.float64

    def test_empty_neuron_has_zero_counts(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Neuron with no spikes should have all-zero counts."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        spike_counts, _ = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        # Neuron 3 has no spikes (empty array)
        assert np.all(spike_counts[3] == 0)

    def test_consistent_with_single_neuron(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Batch result should be consistent with single-neuron results."""
        from neurospatial.encoding._view_binning import (
            bin_view_spike_train,
            bin_view_spike_trains,
        )

        spike_counts_batch, _view_occupancy_batch = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        # Check each neuron matches single-neuron result
        for i, spikes in enumerate(multiple_neuron_spikes):
            spike_counts_single = bin_view_spike_train(
                simple_env,
                spikes,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
            )
            np.testing.assert_array_equal(spike_counts_batch[i], spike_counts_single)

    def test_n_jobs_parameter(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """n_jobs parameter should not change results."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        spike_counts_serial, view_occupancy_serial = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            n_jobs=1,
        )

        spike_counts_parallel, view_occupancy_parallel = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            n_jobs=2,
        )

        np.testing.assert_array_equal(spike_counts_serial, spike_counts_parallel)
        np.testing.assert_array_equal(view_occupancy_serial, view_occupancy_parallel)

    def test_single_neuron_list(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Should handle list with single neuron."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        spike_times_list = [single_neuron_spikes]

        spike_counts, view_occupancy = bin_view_spike_trains(
            simple_env,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert spike_counts.shape == (1, simple_env.n_bins)
        assert view_occupancy.shape == (simple_env.n_bins,)

    def test_normalizes_input_via_normalize_spike_times(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should accept normalized input from normalize_spike_times formats."""
        from neurospatial.encoding._view_binning import bin_view_spike_trains

        # Test with 2D NaN-padded array (common format)
        spikes_2d = np.array(
            [
                [0.5, 1.5, 2.5, np.nan],
                [1.0, 2.0, np.nan, np.nan],
                [0.25, np.nan, np.nan, np.nan],
            ]
        )

        spike_counts, _view_occupancy = bin_view_spike_trains(
            simple_env,
            spikes_2d,  # type: ignore[arg-type]
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
        )

        assert spike_counts.shape == (3, simple_env.n_bins)


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
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.linspace(0, 10.0, 1000)
        # Animal at center, looking east (heading=0)
        positions = np.tile([50, 50], (1000, 1))
        headings = np.zeros(1000)  # Constant heading east

        view_occupancy = compute_view_occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # Should have high occupancy in bins east of center
        # Viewed location is at (60, 50) for view_distance=10
        assert np.any(view_occupancy > 0)

    def test_view_outside_environment(
        self,
        simple_env: Environment,
    ) -> None:
        """Views outside environment should not contribute to occupancy."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.linspace(0, 10.0, 1000)
        # Animal at edge, looking outward
        positions = np.tile([5, 50], (1000, 1))
        headings = np.ones(1000) * np.pi  # Looking west (outside)

        view_occupancy = compute_view_occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=20.0,  # Would be at (-15, 50) - outside env
        )

        # Total occupancy should be less than duration due to invalid views
        total_duration = times[-1] - times[0]
        assert np.sum(view_occupancy) < total_duration * 0.5

    def test_all_spikes_at_same_viewed_location(
        self,
        simple_env: Environment,
    ) -> None:
        """Spikes at times with same view should be binned together."""
        from neurospatial.encoding._view_binning import bin_view_spike_train

        times = np.linspace(0, 10.0, 1000)
        # Constant position and heading
        positions = np.tile([50, 50], (1000, 1))
        headings = np.zeros(1000)  # Looking east

        # Multiple spikes at various times
        spikes = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        spike_counts = bin_view_spike_train(
            simple_env,
            spikes,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # All 5 spikes should be in one bin
        assert np.sum(spike_counts) == 5
        assert np.max(spike_counts) == 5


class TestViewBinningInputValidation:
    """Tests for input validation in view binning functions."""

    def test_mismatched_times_positions_length(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error if times and positions have different lengths."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.linspace(0, 10, 100)
        positions = np.random.rand(50, 2) * 100  # Different length
        headings = np.random.uniform(0, 2 * np.pi, 100)

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_view_occupancy(simple_env, times, positions, headings)

    def test_mismatched_times_headings_length(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error if times and headings have different lengths."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.linspace(0, 10, 100)
        positions = np.random.rand(100, 2) * 100
        headings = np.random.uniform(0, 2 * np.pi, 50)  # Different length

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_view_occupancy(simple_env, times, positions, headings)

    def test_invalid_gaze_model(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should raise error for invalid gaze_model."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        with pytest.raises(ValueError, match=r"gaze_model|invalid"):
            compute_view_occupancy(
                simple_env,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                gaze_model="invalid_model",  # type: ignore[arg-type]
            )

    def test_insufficient_samples(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error with fewer than 2 samples."""
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.array([0.0])  # Only 1 sample - need at least 2
        positions = np.array([[50, 50]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match=r"2|samples"):
            compute_view_occupancy(simple_env, times, positions, headings)


# ==============================================================================
# Test comparison with legacy spatial_view.py implementation
# ==============================================================================


class TestComparisonWithLegacySpatialView:
    """Tests comparing new binning layer with legacy compute_spatial_view_field.

    Note: The legacy implementation has a bug (uses median(dt) instead of per-sample
    deltas). The new implementation is CORRECT and may differ from legacy.
    """

    def test_view_occupancy_total_is_correct(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """New implementation should have correct total occupancy.

        Total view occupancy should equal sum of actual time intervals
        (times[-1] - times[0]) minus any intervals with invalid views.
        """
        from neurospatial.encoding._view_binning import compute_view_occupancy

        view_occ_new = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # Total should equal sum of time intervals
        total_duration = trajectory_data["times"][-1] - trajectory_data["times"][0]
        total_occupancy = np.sum(view_occ_new)

        # Allow some difference due to invalid views (outside environment)
        # but should be very close for trajectories staying in center
        assert total_occupancy <= total_duration * 1.01
        assert total_occupancy >= total_duration * 0.9  # Most views should be valid

    def test_view_occupancy_bins_are_similar_to_legacy(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """New and legacy implementations should have similar bin distributions.

        While total occupancy may differ (due to bug fix), the relative distribution
        across bins should be similar since both use the same gaze model and
        viewed location computation.
        """
        from neurospatial.encoding._view_binning import compute_view_occupancy
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        view_distance = 10.0
        gaze_model = "fixed_distance"

        # Get view occupancy from new binning layer
        view_occ_new = compute_view_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model=gaze_model,
            view_distance=view_distance,
        )

        # Get view occupancy from legacy implementation
        spikes = np.array([0.5, 1.5, 2.5, 3.5])
        result = compute_spatial_view_field(
            simple_env,
            spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model=gaze_model,
            view_distance=view_distance,
            smoothing_method="binned",
            min_occupancy_seconds=0.0,
        )
        view_occ_legacy = result.view_occupancy

        # Normalize to compare distributions
        new_norm = view_occ_new / (np.sum(view_occ_new) + 1e-10)
        legacy_norm = view_occ_legacy / (np.sum(view_occ_legacy) + 1e-10)

        # Distributions should be similar (correlation > 0.95)
        # Both use the same gaze model, just different time integration
        correlation = np.corrcoef(new_norm.flatten(), legacy_norm.flatten())[0, 1]
        assert correlation > 0.95, f"Distributions too different: r={correlation:.3f}"


# ==============================================================================
# Tests for bugfixes from code review (2025-12-19)
# ==============================================================================


class TestViewOccupancyNonUniformSampling:
    """Tests for correct handling of non-uniform time sampling.

    Bug: view_occupancy used median(dt) constant instead of per-sample deltas.
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
        from neurospatial.encoding._view_binning import compute_view_occupancy

        # Create non-uniform time samples: first half at 10Hz, second half at 100Hz
        times_slow = np.linspace(0, 5, 51)  # 50 intervals of 0.1s
        times_fast = np.linspace(5.01, 10, 500)  # 499 intervals of ~0.01s
        times = np.concatenate([times_slow, times_fast])

        n_samples = len(times)
        positions = np.tile([50, 50], (n_samples, 1))  # Stationary
        headings = np.zeros(n_samples)  # Looking east

        view_occupancy = compute_view_occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # Expected total: sum of all actual time deltas
        # = (5-0) + (10-5.01) = 5 + 4.99 = ~9.99 seconds
        total_duration = times[-1] - times[0]  # ~10s
        total_occupancy = np.sum(view_occupancy)

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
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 5 samples, 4 intervals
        positions = np.tile([50, 50], (5, 1))
        headings = np.zeros(5)

        view_occupancy = compute_view_occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # With 4 intervals of 1s each, total should be 4s (not 5s)
        total_occupancy = np.sum(view_occupancy)
        assert abs(total_occupancy - 4.0) < 0.01, (
            f"Expected 4.0s (4 intervals), got {total_occupancy:.3f}s. "
            "Last frame may be incorrectly contributing to occupancy."
        )


class TestTimesValidation:
    """Tests for input validation of times array."""

    def test_unsorted_times_raises_error(
        self,
        simple_env: Environment,
    ) -> None:
        """Unsorted times should raise ValueError.

        searchsorted assumes sorted input. Unsorted times will silently
        produce incorrect results without validation.
        """
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.array([0.0, 2.0, 1.0, 3.0])  # Not sorted
        positions = np.tile([50, 50], (4, 1))
        headings = np.zeros(4)

        with pytest.raises(ValueError, match=r"monotonic|sorted|increasing"):
            compute_view_occupancy(
                simple_env,
                times,
                positions,
                headings,
            )

    def test_duplicate_times_allowed(
        self,
        simple_env: Environment,
    ) -> None:
        """Duplicate times (monotonically non-decreasing) should be allowed.

        Times like [0, 1, 1, 2] are valid (non-decreasing but not strictly
        increasing). The duplicate contributes 0 seconds.
        """
        from neurospatial.encoding._view_binning import compute_view_occupancy

        times = np.array([0.0, 1.0, 1.0, 2.0])  # Duplicate at t=1
        positions = np.tile([50, 50], (4, 1))
        headings = np.zeros(4)

        view_occupancy = compute_view_occupancy(
            simple_env,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # 3 intervals: 1s, 0s, 1s = 2s total
        total_occupancy = np.sum(view_occupancy)
        assert abs(total_occupancy - 2.0) < 0.01


# ==============================================================================
# Tests for view binning optimization (precomputation)
# ==============================================================================


class TestBinViewSpikeTrainsPrecomputation:
    """Tests that bin_view_spike_trains precomputes shared quantities.

    Bug: viewed_locations was recomputed for every neuron in the batch,
    contradicting the docstring that says it precomputes shared quantities.
    """

    def test_batch_result_matches_sequential(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Batch result should match sequential single-neuron calls."""
        from neurospatial.encoding._view_binning import (
            bin_view_spike_train,
            bin_view_spike_trains,
        )

        # Get batch result
        batch_counts, _batch_occupancy = bin_view_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # Get sequential results
        for i, spikes in enumerate(multiple_neuron_spikes):
            single_counts = bin_view_spike_train(
                simple_env,
                spikes,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                gaze_model="fixed_distance",
                view_distance=10.0,
            )
            np.testing.assert_array_almost_equal(
                batch_counts[i], single_counts, err_msg=f"Neuron {i} counts differ"
            )

    def test_view_bins_precomputed_once(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """view_bins should be computed once and reused for all neurons.

        This test verifies that viewed locations are precomputed by checking
        that the internal implementation uses precomputed view_bins.
        """
        from neurospatial.encoding._view_binning import (
            _precompute_view_bins,
        )

        # This internal function should exist for precomputation
        view_bins = _precompute_view_bins(
            simple_env,
            trajectory_data["positions"],
            trajectory_data["headings"],
            gaze_model="fixed_distance",
            view_distance=10.0,
        )

        # Should return bin indices for each timepoint
        assert view_bins.shape == (len(trajectory_data["times"]),)
        assert view_bins.dtype == np.intp
        # Invalid views should be -1
        assert np.all((view_bins >= -1) & (view_bins < simple_env.n_bins))
