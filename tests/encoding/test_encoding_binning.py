"""Tests for neurospatial.encoding._binning module.

This module tests the binning layer that converts spike trains to
spike counts and occupancy arrays.

TDD approach: Tests written first, implementation follows.

Task 2.7: Implement binning layer for spatial encoding
- Create helper to convert (env, spike_times, times, positions) -> (spike_counts, occupancy)
- Spike counts shape: (n_neurons, n_bins)
- Occupancy shape: (n_bins,)
- Parallelize over neurons with joblib
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
    """Create trajectory data for testing.

    Returns dict with:
        - times: array of timestamps (1 second total, 100 samples at 100 Hz)
        - positions: array of positions (random walk within environment)
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    times = np.linspace(0, 1.0, n_samples)  # 1 second, 100 samples

    # Random walk within environment bounds
    positions = np.zeros((n_samples, 2))
    positions[0] = [50, 50]  # Start in center
    for i in range(1, n_samples):
        step = rng.normal(size=2) * 5  # Random step
        positions[i] = positions[i - 1] + step
        # Clip to environment bounds
        positions[i] = np.clip(positions[i], 0, 100)

    return {"times": times, "positions": positions}


@pytest.fixture
def single_neuron_spikes() -> NDArray[np.float64]:
    """Create spike times for a single neuron."""
    # Spikes at 0.1, 0.3, 0.5, 0.7, 0.9 seconds
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


@pytest.fixture
def multiple_neuron_spikes() -> list[NDArray[np.float64]]:
    """Create spike times for multiple neurons."""
    return [
        np.array([0.1, 0.3, 0.5]),  # Neuron 0: 3 spikes
        np.array([0.2, 0.4, 0.6, 0.8]),  # Neuron 1: 4 spikes
        np.array([0.15]),  # Neuron 2: 1 spike
        np.array([]),  # Neuron 3: 0 spikes (silent)
    ]


# ==============================================================================
# Test bin_spike_train (single neuron)
# ==============================================================================


class TestBinSpikeTrain:
    """Tests for bin_spike_train function (single neuron binning)."""

    def test_function_is_importable(self) -> None:
        """bin_spike_train should be importable from encoding._binning."""
        from neurospatial.encoding._binning import bin_spike_train

        assert bin_spike_train is not None

    def test_returns_spike_counts(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_spike_train should return spike counts array."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert isinstance(spike_counts, np.ndarray)
        assert spike_counts.dtype == np.float64

    def test_spike_counts_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_spike_train should return array with shape (n_bins,)."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)

    def test_spike_counts_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts should all be non-negative."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert np.all(spike_counts >= 0)

    def test_total_spike_count(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Total spike count should equal number of spikes (within time range)."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Filter spikes to valid time range
        valid_spikes = single_neuron_spikes[
            (single_neuron_spikes >= trajectory_data["times"].min())
            & (single_neuron_spikes <= trajectory_data["times"].max())
        ]

        # Some spikes may fall outside active bins, so sum should be <= n_valid_spikes
        # but for a well-covered environment, should be close
        assert np.sum(spike_counts) <= len(valid_spikes)

    def test_empty_spike_train(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Empty spike train should produce all-zero spike counts."""
        from neurospatial.encoding._binning import bin_spike_train

        empty_spikes = np.array([])

        spike_counts = bin_spike_train(
            simple_env,
            empty_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)
        assert np.all(spike_counts == 0)

    def test_spikes_outside_time_range(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Spikes outside time range should be excluded."""
        from neurospatial.encoding._binning import bin_spike_train

        # Spikes at -1.0 and 5.0, both outside the 0-1 second range
        spikes_outside = np.array([-1.0, 5.0])

        spike_counts = bin_spike_train(
            simple_env,
            spikes_outside,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert np.sum(spike_counts) == 0


# ==============================================================================
# Test compute_occupancy
# ==============================================================================


class TestComputeOccupancy:
    """Tests for compute_occupancy function."""

    def test_function_is_importable(self) -> None:
        """compute_occupancy should be importable from encoding._binning."""
        from neurospatial.encoding._binning import compute_occupancy

        assert compute_occupancy is not None

    def test_returns_occupancy_array(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """compute_occupancy should return occupancy array in seconds."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert isinstance(occupancy, np.ndarray)
        assert occupancy.dtype == np.float64

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """compute_occupancy should return array with shape (n_bins,)."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert occupancy.shape == (simple_env.n_bins,)

    def test_occupancy_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Occupancy should all be non-negative."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximately_equals_duration(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Total occupancy should approximately equal recording duration."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        total_duration = trajectory_data["times"][-1] - trajectory_data["times"][0]
        total_occupancy = np.sum(occupancy)

        # Should be close to total duration (may differ slightly due to
        # invalid bin positions or max_gap handling)
        assert total_occupancy <= total_duration + 0.01  # Allow small tolerance


# ==============================================================================
# Test bin_spike_trains (batch)
# ==============================================================================


class TestBinSpikeTrains:
    """Tests for bin_spike_trains function (multiple neurons)."""

    def test_function_is_importable(self) -> None:
        """bin_spike_trains should be importable from encoding._binning."""
        from neurospatial.encoding._binning import bin_spike_trains

        assert bin_spike_trains is not None

    def test_returns_spike_counts_and_occupancy(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """bin_spike_trains should return (spike_counts, occupancy) tuple."""
        from neurospatial.encoding._binning import bin_spike_trains

        result = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        spike_counts, occupancy = result
        assert isinstance(spike_counts, np.ndarray)
        assert isinstance(occupancy, np.ndarray)

    def test_spike_counts_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts, _ = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        n_neurons = len(multiple_neuron_spikes)
        assert spike_counts.shape == (n_neurons, simple_env.n_bins)

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Occupancy should have shape (n_bins,)."""
        from neurospatial.encoding._binning import bin_spike_trains

        _, occupancy = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert occupancy.shape == (simple_env.n_bins,)

    def test_spike_counts_dtype(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should be float64."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts, _ = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.dtype == np.float64

    def test_occupancy_dtype(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Occupancy should be float64."""
        from neurospatial.encoding._binning import bin_spike_trains

        _, occupancy = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert occupancy.dtype == np.float64

    def test_empty_neuron_has_zero_counts(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Neuron with no spikes should have all-zero counts."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts, _ = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
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
        from neurospatial.encoding._binning import (
            bin_spike_train,
            bin_spike_trains,
        )

        spike_counts_batch, _occupancy_batch = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Check each neuron matches single-neuron result
        for i, spikes in enumerate(multiple_neuron_spikes):
            spike_counts_single = bin_spike_train(
                simple_env,
                spikes,
                trajectory_data["times"],
                trajectory_data["positions"],
            )
            np.testing.assert_array_equal(spike_counts_batch[i], spike_counts_single)

    def test_n_jobs_parameter(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """n_jobs parameter should not change results."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts_serial, occupancy_serial = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            n_jobs=1,
        )

        spike_counts_parallel, occupancy_parallel = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            n_jobs=2,
        )

        np.testing.assert_array_equal(spike_counts_serial, spike_counts_parallel)
        np.testing.assert_array_equal(occupancy_serial, occupancy_parallel)

    def test_single_neuron_list(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Should handle list with single neuron."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_times_list = [single_neuron_spikes]

        spike_counts, occupancy = bin_spike_trains(
            simple_env,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (1, simple_env.n_bins)
        assert occupancy.shape == (simple_env.n_bins,)

    def test_normalizes_input_via_normalize_spike_times(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should accept normalized input from normalize_spike_times formats."""
        from neurospatial.encoding._binning import bin_spike_trains

        # Test with 2D NaN-padded array (common format)
        spikes_2d = np.array(
            [
                [0.1, 0.3, 0.5, np.nan],
                [0.2, 0.4, np.nan, np.nan],
                [0.15, np.nan, np.nan, np.nan],
            ]
        )

        spike_counts, _occupancy = bin_spike_trains(
            simple_env,
            spikes_2d,  # type: ignore[arg-type]
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (3, simple_env.n_bins)


# ==============================================================================
# Test edge cases and error handling
# ==============================================================================


class TestBinningEdgeCases:
    """Tests for edge cases in binning layer."""

    def test_all_spikes_in_one_bin(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """All spikes in one location should be binned together."""
        from neurospatial.encoding._binning import bin_spike_train

        # Create spikes at times when we know the position
        # Position at t=0.5 (midpoint)
        t_mid = 0.5
        mid_idx = np.argmin(np.abs(trajectory_data["times"] - t_mid))
        pos_at_mid = trajectory_data["positions"][mid_idx]

        # Multiple spikes all at t=0.5
        spikes = np.array([0.5, 0.5001, 0.5002])

        spike_counts = bin_spike_train(
            simple_env,
            spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Should have counts in just one bin
        target_bin = simple_env.bin_at(pos_at_mid.reshape(1, -1))[0]
        if target_bin >= 0:  # Valid bin
            assert spike_counts[target_bin] == 3
            # All other bins should be 0
            assert np.sum(spike_counts) == 3

    def test_handles_narrow_2d_environment(self) -> None:
        """Should handle narrow 2D environments correctly."""
        from neurospatial import Environment
        from neurospatial.encoding._binning import bin_spike_trains

        # Create a narrow 2D environment (effectively 1D-like behavior)
        # This is a linear track approximation
        x = np.linspace(0, 100, 11)
        y = np.ones(11) * 5  # Narrow in y-direction
        positions_env = np.column_stack([x, y])
        env = Environment.from_samples(positions_env, bin_size=10.0)

        # Create trajectory along the track
        times = np.linspace(0, 1.0, 100)
        positions = np.column_stack(
            [np.linspace(5, 95, 100), np.ones(100) * 5]  # Move along x
        )

        spikes = [np.array([0.1, 0.5]), np.array([0.3])]

        spike_counts, occupancy = bin_spike_trains(
            env,
            spikes,
            times,
            positions,
        )

        assert spike_counts.shape == (2, env.n_bins)
        assert occupancy.shape == (env.n_bins,)


class TestBinningInputValidation:
    """Tests for input validation in binning functions."""

    def test_mismatched_times_positions_length(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error if times and positions have different lengths."""
        from neurospatial.encoding._binning import compute_occupancy

        times = np.linspace(0, 1, 100)
        rng = np.random.default_rng(42)
        positions = rng.random((50, 2)) * 100  # Different length

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_occupancy(simple_env, times, positions)

    def test_position_dims_mismatch(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should raise error if position dimensions don't match env."""
        from neurospatial.encoding._binning import compute_occupancy

        # simple_env is 2D, but provide 3D positions
        positions_3d = np.column_stack(
            [trajectory_data["positions"], np.zeros(len(trajectory_data["times"]))]
        )

        with pytest.raises(ValueError, match=r"dimension|shape"):
            compute_occupancy(simple_env, trajectory_data["times"], positions_3d)


class TestSpikeInterpolationRegression:
    """Regression tests guarding against snapshot-instead-of-interpolated binning.

    A previous refactor mapped each spike to the bin of its most recent
    trajectory frame instead of its interpolated position. With sparse
    sampling or fast movement this shifts spikes to the previous bin —
    a silent rate-map error.
    """

    def test_spike_between_samples_uses_interpolated_position(self) -> None:
        """A spike at t=0.75 with samples at 0/10/20 maps to the 7.5 bin, not 0."""
        from neurospatial import Environment
        from neurospatial.encoding._binning import bin_spike_train

        # 1D environment with bins of width 5 spanning [0, 20].
        sample_positions = np.linspace(0.0, 20.0, 100).reshape(-1, 1)
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Sparse trajectory: three samples at t=0, 1, 2 with positions 0, 10, 20.
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0], [10.0], [20.0]])

        # Single spike at t=0.75; linear interpolation gives position 7.5
        # which falls in the same bin as 7.5 (a different bin from position 0).
        spike_times = np.array([0.75])

        counts = bin_spike_train(env, spike_times, times, positions)

        # Identify the bin that interpolated position 7.5 lives in, vs the
        # bin that the snapshot position 0.0 lives in. The two must differ
        # for this regression test to be meaningful, and the spike count
        # must land in the interpolated bin.
        interp_bin = env.bin_at(np.array([[7.5]]))[0]
        snapshot_bin = env.bin_at(np.array([[0.0]]))[0]
        assert interp_bin != snapshot_bin, "test setup not exercising the bug"
        assert counts[interp_bin] == 1.0
        assert counts[snapshot_bin] == 0.0

    def test_batch_matches_single_under_interpolation(self) -> None:
        """bin_spike_trains must match bin_spike_train per neuron exactly."""
        from neurospatial import Environment
        from neurospatial.encoding._binning import bin_spike_train, bin_spike_trains

        sample_positions = np.linspace(0.0, 20.0, 100).reshape(-1, 1)
        env = Environment.from_samples(sample_positions, bin_size=5.0)
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0], [10.0], [20.0]])
        spike_lists = [
            np.array([0.75]),
            np.array([0.25, 1.5]),
            np.array([]),
        ]

        batch_counts, _ = bin_spike_trains(env, spike_lists, times, positions)
        for i, spikes in enumerate(spike_lists):
            single = bin_spike_train(env, spikes, times, positions)
            np.testing.assert_array_equal(batch_counts[i], single)
