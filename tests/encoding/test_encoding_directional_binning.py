"""Tests for neurospatial.encoding._directional_binning module.

This module tests the directional binning layer that converts spike trains
and head direction data to spike counts and occupancy arrays.

TDD approach: Tests written first, implementation follows.

Task 3.7: Implement binning layer for directional encoding
- Create helper to convert (spike_times, times, headings, bin_size) -> (spike_counts, occupancy)
- Handle circular binning (0 to 2π)
- Support `angle_unit` parameter for input conversion
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def simple_trajectory() -> dict:
    """Create simple trajectory data for testing.

    Returns dict with:
        - times: array of timestamps (10 seconds at 100 Hz)
        - headings: array of head directions (random walk in radians)
    """
    np.random.seed(42)
    n_samples = 1000
    times = np.linspace(0, 10.0, n_samples)  # 10 seconds, 100 samples/s

    # Random walk in heading with wrap-around
    headings = np.zeros(n_samples)
    headings[0] = np.pi  # Start facing south
    for i in range(1, n_samples):
        step = np.random.randn() * 0.1  # Random angular step
        headings[i] = (headings[i - 1] + step) % (2 * np.pi)

    return {"times": times, "headings": headings}


@pytest.fixture
def single_neuron_spikes() -> NDArray[np.float64]:
    """Create spike times for a single neuron."""
    # Spikes at various times
    return np.array([0.5, 1.2, 2.5, 3.8, 5.0, 6.3, 7.5, 8.2, 9.0])


@pytest.fixture
def multiple_neuron_spikes() -> list[NDArray[np.float64]]:
    """Create spike times for multiple neurons."""
    return [
        np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 0: 4 spikes
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Neuron 1: 5 spikes
        np.array([0.25]),  # Neuron 2: 1 spike
        np.array([]),  # Neuron 3: 0 spikes (silent)
    ]


# ==============================================================================
# Test bin_directional_spike_train (single neuron)
# ==============================================================================


class TestBinDirectionalSpikeTrain:
    """Tests for bin_directional_spike_train function (single neuron binning)."""

    def test_function_is_importable(self) -> None:
        """bin_directional_spike_train should be importable from encoding._directional_binning."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        assert bin_directional_spike_train is not None

    def test_returns_spike_counts(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_directional_spike_train should return spike counts array."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        spike_counts = bin_directional_spike_train(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,  # 6 degrees
        )

        assert isinstance(spike_counts, np.ndarray)
        assert spike_counts.dtype == np.float64

    def test_spike_counts_shape(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_directional_spike_train should return array with shape (n_bins,)."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        bin_size = np.pi / 30  # 6 degrees -> 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))

        spike_counts = bin_directional_spike_train(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=bin_size,
        )

        assert spike_counts.shape == (n_bins,)

    def test_spike_counts_non_negative(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts should all be non-negative."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        spike_counts = bin_directional_spike_train(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert np.all(spike_counts >= 0)

    def test_total_spike_count(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Total spike count should equal number of spikes (within time range)."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        spike_counts = bin_directional_spike_train(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        # Filter spikes to valid time range
        times = simple_trajectory["times"]
        valid_spikes = single_neuron_spikes[
            (single_neuron_spikes >= times.min())
            & (single_neuron_spikes <= times.max())
        ]

        # Total should equal number of valid spikes
        assert np.sum(spike_counts) == len(valid_spikes)

    def test_empty_spike_train(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Empty spike train should produce all-zero spike counts."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        empty_spikes = np.array([])

        spike_counts = bin_directional_spike_train(
            empty_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        n_bins = int(np.round(2 * np.pi / (np.pi / 30)))
        assert spike_counts.shape == (n_bins,)
        assert np.all(spike_counts == 0)

    def test_spikes_outside_time_range(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Spikes outside time range should be excluded."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        # Spikes at -1.0 and 20.0, both outside the 0-10 second range
        spikes_outside = np.array([-1.0, 20.0])

        spike_counts = bin_directional_spike_train(
            spikes_outside,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert np.sum(spike_counts) == 0


class TestBinDirectionalSpikeTrainAngleUnit:
    """Tests for angle_unit parameter handling."""

    def test_angle_unit_rad_default(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Default angle_unit should be 'rad'."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        # Headings in radians
        spike_counts = bin_directional_spike_train(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],  # Already in radians
            bin_size=np.pi / 30,
        )

        assert isinstance(spike_counts, np.ndarray)

    def test_angle_unit_deg(
        self,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Should handle headings in degrees."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        np.random.seed(42)
        times = np.linspace(0, 10.0, 100)
        headings_deg = np.random.uniform(0, 360, 100)  # Degrees

        spike_counts = bin_directional_spike_train(
            single_neuron_spikes,
            times,
            headings_deg,
            bin_size=6.0,  # 6 degrees
            angle_unit="deg",
        )

        n_bins = 60  # 360 / 6 = 60 bins
        assert spike_counts.shape == (n_bins,)

    def test_rad_and_deg_give_same_result(
        self,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Same data in rad/deg should give same spike counts."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        np.random.seed(42)
        times = np.linspace(0, 10.0, 100)
        headings_rad = np.random.uniform(0, 2 * np.pi, 100)
        headings_deg = np.degrees(headings_rad)

        counts_rad = bin_directional_spike_train(
            single_neuron_spikes,
            times,
            headings_rad,
            bin_size=np.pi / 30,  # 6 degrees in radians
            angle_unit="rad",
        )

        counts_deg = bin_directional_spike_train(
            single_neuron_spikes,
            times,
            headings_deg,
            bin_size=6.0,  # 6 degrees
            angle_unit="deg",
        )

        np.testing.assert_array_equal(counts_rad, counts_deg)


# ==============================================================================
# Test compute_directional_occupancy
# ==============================================================================


class TestComputeDirectionalOccupancy:
    """Tests for compute_directional_occupancy function."""

    def test_function_is_importable(self) -> None:
        """compute_directional_occupancy should be importable from encoding._directional_binning."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        assert compute_directional_occupancy is not None

    def test_returns_occupancy_and_bin_centers(
        self,
        simple_trajectory: dict,
    ) -> None:
        """compute_directional_occupancy should return (occupancy, bin_centers) tuple."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        result = compute_directional_occupancy(
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        occupancy, bin_centers = result
        assert isinstance(occupancy, np.ndarray)
        assert isinstance(bin_centers, np.ndarray)

    def test_occupancy_shape(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Occupancy should have shape (n_bins,)."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        bin_size = np.pi / 30  # 6 degrees -> 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))

        occupancy, bin_centers = compute_directional_occupancy(
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=bin_size,
        )

        assert occupancy.shape == (n_bins,)
        assert bin_centers.shape == (n_bins,)

    def test_bin_centers_range(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Bin centers should be in [0, 2π) range."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        _, bin_centers = compute_directional_occupancy(
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert np.all(bin_centers >= 0)
        assert np.all(bin_centers < 2 * np.pi)

    def test_occupancy_dtype(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Occupancy should be float64."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        occupancy, _ = compute_directional_occupancy(
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert occupancy.dtype == np.float64

    def test_occupancy_non_negative(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Occupancy should all be non-negative."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        occupancy, _ = compute_directional_occupancy(
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximately_equals_duration(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Total occupancy should approximately equal recording duration."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        occupancy, _ = compute_directional_occupancy(
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        # Duration is approximately 10 seconds (last frame excluded)
        # times range from 0 to ~10 seconds
        expected_duration = (
            simple_trajectory["times"][-2] - simple_trajectory["times"][0]
        )
        total_occupancy = np.sum(occupancy)

        # Should be close to expected duration
        assert np.isclose(total_occupancy, expected_duration, rtol=0.01)

    def test_angle_unit_deg(self) -> None:
        """Should handle headings in degrees with angle_unit='deg'."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        times = np.linspace(0, 10.0, 100)
        headings_deg = np.random.uniform(0, 360, 100)

        occupancy, bin_centers = compute_directional_occupancy(
            times,
            headings_deg,
            bin_size=6.0,  # 6 degrees
            angle_unit="deg",
        )

        n_bins = 60
        assert occupancy.shape == (n_bins,)
        assert bin_centers.shape == (n_bins,)
        # Bin centers should always be in radians
        assert np.all(bin_centers >= 0)
        assert np.all(bin_centers < 2 * np.pi)


# ==============================================================================
# Test bin_directional_spike_trains (batch)
# ==============================================================================


class TestBinDirectionalSpikeTrains:
    """Tests for bin_directional_spike_trains function (multiple neurons)."""

    def test_function_is_importable(self) -> None:
        """bin_directional_spike_trains should be importable from encoding._directional_binning."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        assert bin_directional_spike_trains is not None

    def test_returns_spike_counts_occupancy_and_bin_centers(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """bin_directional_spike_trains should return (spike_counts, occupancy, bin_centers) tuple."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        result = bin_directional_spike_trains(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        spike_counts, occupancy, bin_centers = result
        assert isinstance(spike_counts, np.ndarray)
        assert isinstance(occupancy, np.ndarray)
        assert isinstance(bin_centers, np.ndarray)

    def test_spike_counts_shape(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        bin_size = np.pi / 30  # 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))
        n_neurons = len(multiple_neuron_spikes)

        spike_counts, _, _ = bin_directional_spike_trains(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=bin_size,
        )

        assert spike_counts.shape == (n_neurons, n_bins)

    def test_occupancy_shape(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Occupancy should have shape (n_bins,)."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        bin_size = np.pi / 30  # 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))

        _, occupancy, _ = bin_directional_spike_trains(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=bin_size,
        )

        assert occupancy.shape == (n_bins,)

    def test_spike_counts_dtype(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should be float64."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        spike_counts, _, _ = bin_directional_spike_trains(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        assert spike_counts.dtype == np.float64

    def test_empty_neuron_has_zero_counts(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Neuron with no spikes should have all-zero counts."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        spike_counts, _, _ = bin_directional_spike_trains(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=np.pi / 30,
        )

        # Neuron 3 has no spikes (empty array)
        assert np.all(spike_counts[3] == 0)

    def test_consistent_with_single_neuron(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Batch result should be consistent with single-neuron results."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
            bin_directional_spike_trains,
        )

        bin_size = np.pi / 30

        spike_counts_batch, _occupancy_batch, _bin_centers_batch = (
            bin_directional_spike_trains(
                multiple_neuron_spikes,
                simple_trajectory["times"],
                simple_trajectory["headings"],
                bin_size=bin_size,
            )
        )

        # Check each neuron matches single-neuron result
        for i, spikes in enumerate(multiple_neuron_spikes):
            spike_counts_single = bin_directional_spike_train(
                spikes,
                simple_trajectory["times"],
                simple_trajectory["headings"],
                bin_size=bin_size,
            )
            np.testing.assert_array_equal(spike_counts_batch[i], spike_counts_single)

    def test_n_jobs_parameter(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """n_jobs parameter should not change results."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        spike_counts_serial, occupancy_serial, bin_centers_serial = (
            bin_directional_spike_trains(
                multiple_neuron_spikes,
                simple_trajectory["times"],
                simple_trajectory["headings"],
                bin_size=np.pi / 30,
                n_jobs=1,
            )
        )

        spike_counts_parallel, occupancy_parallel, bin_centers_parallel = (
            bin_directional_spike_trains(
                multiple_neuron_spikes,
                simple_trajectory["times"],
                simple_trajectory["headings"],
                bin_size=np.pi / 30,
                n_jobs=2,
            )
        )

        np.testing.assert_array_equal(spike_counts_serial, spike_counts_parallel)
        np.testing.assert_array_equal(occupancy_serial, occupancy_parallel)
        np.testing.assert_array_equal(bin_centers_serial, bin_centers_parallel)

    def test_single_neuron_list(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Should handle list with single neuron."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        spike_times_list = [single_neuron_spikes]
        bin_size = np.pi / 30
        n_bins = int(np.round(2 * np.pi / bin_size))

        spike_counts, occupancy, bin_centers = bin_directional_spike_trains(
            spike_times_list,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=bin_size,
        )

        assert spike_counts.shape == (1, n_bins)
        assert occupancy.shape == (n_bins,)
        assert bin_centers.shape == (n_bins,)

    def test_normalizes_input_via_normalize_spike_times(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Should accept normalized input from normalize_spike_times formats."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_trains,
        )

        # Test with 2D NaN-padded array (common format)
        spikes_2d = np.array(
            [
                [0.5, 1.5, 2.5, np.nan],
                [1.0, 2.0, np.nan, np.nan],
                [0.25, np.nan, np.nan, np.nan],
            ]
        )

        bin_size = np.pi / 30
        n_bins = int(np.round(2 * np.pi / bin_size))

        spike_counts, _occupancy, _bin_centers = bin_directional_spike_trains(
            spikes_2d,  # type: ignore[arg-type]
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size=bin_size,
        )

        assert spike_counts.shape == (3, n_bins)


# ==============================================================================
# Test edge cases and error handling
# ==============================================================================


class TestDirectionalBinningEdgeCases:
    """Tests for edge cases in directional binning layer."""

    def test_all_spikes_at_one_direction(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Spikes at times with same heading should be binned together."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        # Create trajectory with constant heading at π/2
        times = np.linspace(0, 10.0, 100)
        headings = np.ones(100) * np.pi / 2  # Constant heading

        # Multiple spikes at various times (all at same heading)
        spikes = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        spike_counts = bin_directional_spike_train(
            spikes,
            times,
            headings,
            bin_size=np.pi / 30,  # 6 degrees -> 60 bins
        )

        # All 5 spikes should be in one bin (around π/2)
        assert np.sum(spike_counts) == 5
        assert np.max(spike_counts) == 5

    def test_uniform_heading_distribution(self) -> None:
        """Uniform heading should give roughly uniform occupancy."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        times = np.linspace(0, 10.0, 1001)  # 10 seconds, many samples
        # Linearly increasing heading (covers full circle uniformly)
        headings = np.linspace(0, 2 * np.pi, 1001)

        occupancy, _ = compute_directional_occupancy(
            times,
            headings,
            bin_size=np.pi / 6,  # 30 degrees -> 12 bins
        )

        # Each bin should have roughly equal occupancy
        # (with small variation due to edge effects)
        mean_occ = np.mean(occupancy)
        assert np.all(np.abs(occupancy - mean_occ) < 0.1 * mean_occ)

    def test_headings_wrap_around(self) -> None:
        """Headings crossing 0/2π boundary should be handled correctly."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        times = np.linspace(0, 10.0, 100)
        # Heading crosses from ~2π to ~0 (wraps around)
        headings = np.linspace(1.9 * np.pi, 2.1 * np.pi, 100) % (2 * np.pi)

        spikes = np.array([5.0])  # One spike in the middle of the wrap

        spike_counts = bin_directional_spike_train(
            spikes,
            times,
            headings,
            bin_size=np.pi / 30,
        )

        # Spike should be counted in one of the bins near 0 or 2π
        assert np.sum(spike_counts) == 1


class TestDirectionalBinningInputValidation:
    """Tests for input validation in directional binning functions."""

    def test_mismatched_times_headings_length(self) -> None:
        """Should raise error if times and headings have different lengths."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 50)  # Different length

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_directional_occupancy(times, headings, bin_size=np.pi / 30)

    def test_insufficient_samples(self) -> None:
        """Should raise error with fewer than 3 samples."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        times = np.array([0.0, 1.0])  # Only 2 samples
        headings = np.array([0.0, np.pi])

        with pytest.raises(ValueError, match=r"3|samples"):
            compute_directional_occupancy(times, headings, bin_size=np.pi / 30)

    def test_non_monotonic_times(self) -> None:
        """Should raise error if times are not monotonically increasing."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        times = np.array([0.0, 2.0, 1.0, 3.0])  # Not monotonic
        headings = np.random.uniform(0, 2 * np.pi, 4)

        with pytest.raises(ValueError, match=r"monotonic|increasing"):
            compute_directional_occupancy(times, headings, bin_size=np.pi / 30)

    def test_invalid_angle_unit_occupancy(self) -> None:
        """Should raise error for invalid angle_unit in compute_directional_occupancy."""
        from neurospatial.encoding._directional_binning import (
            compute_directional_occupancy,
        )

        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        with pytest.raises(ValueError, match=r"angle_unit"):
            compute_directional_occupancy(
                times,
                headings,
                bin_size=np.pi / 30,
                angle_unit="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_angle_unit_spike_train(self) -> None:
        """Should raise error for invalid angle_unit in bin_directional_spike_train."""
        from neurospatial.encoding._directional_binning import (
            bin_directional_spike_train,
        )

        spike_times = np.array([0.5, 1.5, 2.5])
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        with pytest.raises(ValueError, match=r"angle_unit"):
            bin_directional_spike_train(
                spike_times,
                times,
                headings,
                bin_size=np.pi / 30,
                angle_unit="invalid",  # type: ignore[arg-type]
            )
