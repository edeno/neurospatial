"""Tests for neurospatial.encoding._directional_counts module.

This module tests the directional binning layer that converts spike trains
and head direction data to spike counts and occupancy arrays, through the
path the encoders use: ``prepare_directional_binning`` followed by
``FrameBinning.count``.

The binning layer:
- Converts (spike_times, times, headings, bin_size) -> (spike_counts, occupancy)
- Handles circular binning over [-π, π)
- Supports the `angle_unit` parameter for input conversion
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial.encoding._directional_counts import prepare_directional_binning


def _bin(
    spike_trains: list[NDArray[np.float64]],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Counts ``(n_neurons, n_bins)``, occupancy and bin centers."""
    prepared, bin_centers = prepare_directional_binning(
        times, headings, bin_size, **kwargs
    )
    counts, occupancy = prepared.count(spike_trains, n_jobs=n_jobs)
    return counts, occupancy, bin_centers


def _counts(spikes, times, headings, bin_size, **kwargs) -> NDArray[np.float64]:
    """One neuron's spike counts, shape ``(n_bins,)``."""
    return _bin([spikes], times, headings, bin_size, **kwargs)[0][0]


def _occupancy(times, headings, bin_size, **kwargs):
    """Occupancy and bin centers, each shape ``(n_bins,)``."""
    _, occupancy, bin_centers = _bin([], times, headings, bin_size, **kwargs)
    return occupancy, bin_centers


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
# Spike counts for one neuron
# ==============================================================================


class TestDirectionalSpikeCounts:
    """One neuron's spike counts per heading bin."""

    def test_spike_counts_shape_and_dtype(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts are float64 with shape (n_bins,)."""
        bin_size = np.pi / 30  # 6 degrees -> 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))

        spike_counts = _counts(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size,
        )

        assert isinstance(spike_counts, np.ndarray)
        assert spike_counts.dtype == np.float64
        assert spike_counts.shape == (n_bins,)

    def test_spike_counts_non_negative(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts should all be non-negative."""
        spike_counts = _counts(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            np.pi / 30,
        )

        assert np.all(spike_counts >= 0)

    def test_total_spike_count(
        self,
        simple_trajectory: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Total spike count should equal number of spikes (within time range)."""
        spike_counts = _counts(
            single_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            np.pi / 30,
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
        spike_counts = _counts(
            np.array([]),
            simple_trajectory["times"],
            simple_trajectory["headings"],
            np.pi / 30,
        )

        n_bins = int(np.round(2 * np.pi / (np.pi / 30)))
        assert spike_counts.shape == (n_bins,)
        assert np.all(spike_counts == 0)

    def test_spikes_outside_time_range(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Spikes outside time range are excluded, with a warning."""
        # Spikes at -1.0 and 20.0, both outside the 0-10 second range
        spikes_outside = np.array([-1.0, 20.0])

        with pytest.warns(UserWarning, match="outside"):
            spike_counts = _counts(
                spikes_outside,
                simple_trajectory["times"],
                simple_trajectory["headings"],
                np.pi / 30,
            )

        assert np.sum(spike_counts) == 0


class TestDirectionalAngleUnit:
    """Tests for angle_unit parameter handling."""

    def test_angle_unit_deg(
        self,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Should handle headings in degrees."""
        np.random.seed(42)
        times = np.linspace(0, 10.0, 100)
        headings_deg = np.random.uniform(0, 360, 100)  # Degrees

        spike_counts = _counts(
            single_neuron_spikes,
            times,
            headings_deg,
            6.0,  # 6 degrees
            angle_unit="deg",
        )

        n_bins = 60  # 360 / 6 = 60 bins
        assert spike_counts.shape == (n_bins,)

    def test_rad_and_deg_give_same_result(
        self,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Same data in rad/deg should give same spike counts and occupancy."""
        np.random.seed(42)
        times = np.linspace(0, 10.0, 100)
        headings_rad = np.random.uniform(0, 2 * np.pi, 100)
        headings_deg = np.degrees(headings_rad)

        counts_rad, occupancy_rad, centers_rad = _bin(
            [single_neuron_spikes],
            times,
            headings_rad,
            np.pi / 30,  # 6 degrees in radians
            angle_unit="rad",
        )
        counts_deg, occupancy_deg, centers_deg = _bin(
            [single_neuron_spikes],
            times,
            headings_deg,
            6.0,  # 6 degrees
            angle_unit="deg",
        )

        np.testing.assert_array_equal(counts_rad, counts_deg)
        np.testing.assert_array_equal(occupancy_rad, occupancy_deg)
        np.testing.assert_allclose(centers_rad, centers_deg)


# ==============================================================================
# Occupancy
# ==============================================================================


class TestDirectionalOccupancy:
    """Time spent facing each heading bin."""

    def test_occupancy_and_bin_center_shapes(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Occupancy and bin centers should have shape (n_bins,)."""
        bin_size = np.pi / 30  # 6 degrees -> 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))

        occupancy, bin_centers = _occupancy(
            simple_trajectory["times"], simple_trajectory["headings"], bin_size
        )

        assert occupancy.shape == (n_bins,)
        assert bin_centers.shape == (n_bins,)

    def test_bin_centers_range(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Bin centers should be in [-π, π) range."""
        _, bin_centers = _occupancy(
            simple_trajectory["times"], simple_trajectory["headings"], np.pi / 30
        )

        assert np.all(bin_centers >= -np.pi)
        assert np.all(bin_centers < np.pi)

    def test_occupancy_dtype(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Occupancy should be float64."""
        occupancy, _ = _occupancy(
            simple_trajectory["times"], simple_trajectory["headings"], np.pi / 30
        )

        assert occupancy.dtype == np.float64

    def test_occupancy_non_negative(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Occupancy should all be non-negative."""
        occupancy, _ = _occupancy(
            simple_trajectory["times"], simple_trajectory["headings"], np.pi / 30
        )

        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximately_equals_duration(
        self,
        simple_trajectory: dict,
    ) -> None:
        """Total occupancy should approximately equal recording duration."""
        occupancy, _ = _occupancy(
            simple_trajectory["times"], simple_trajectory["headings"], np.pi / 30
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
        times = np.linspace(0, 10.0, 100)
        headings_deg = np.random.uniform(0, 360, 100)

        occupancy, bin_centers = _occupancy(
            times,
            headings_deg,
            6.0,  # 6 degrees
            angle_unit="deg",
        )

        n_bins = 60
        assert occupancy.shape == (n_bins,)
        assert bin_centers.shape == (n_bins,)
        # Bin centers should always be in radians
        assert np.all(bin_centers >= -np.pi)
        assert np.all(bin_centers < np.pi)


# ==============================================================================
# Populations
# ==============================================================================


class TestDirectionalPopulationCounts:
    """Spike counts for several neurons sharing one occupancy."""

    def test_population_shapes_and_dtype(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Counts are (n_neurons, n_bins) float64; occupancy is (n_bins,)."""
        bin_size = np.pi / 30  # 60 bins
        n_bins = int(np.round(2 * np.pi / bin_size))
        n_neurons = len(multiple_neuron_spikes)

        spike_counts, occupancy, bin_centers = _bin(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size,
        )

        assert spike_counts.shape == (n_neurons, n_bins)
        assert spike_counts.dtype == np.float64
        assert occupancy.shape == (n_bins,)
        assert bin_centers.shape == (n_bins,)

    def test_empty_neuron_has_zero_counts(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Neuron with no spikes should have all-zero counts."""
        spike_counts, _, _ = _bin(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            np.pi / 30,
        )

        # Neuron 3 has no spikes (empty array)
        assert np.all(spike_counts[3] == 0)

    def test_each_row_matches_the_neuron_alone(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """A neuron's counts do not depend on the other neurons counted with it."""
        bin_size = np.pi / 30

        spike_counts_population, _, _ = _bin(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            bin_size,
        )

        for i, spikes in enumerate(multiple_neuron_spikes):
            spike_counts_single = _counts(
                spikes,
                simple_trajectory["times"],
                simple_trajectory["headings"],
                bin_size,
            )
            np.testing.assert_array_equal(
                spike_counts_population[i], spike_counts_single
            )

    def test_n_jobs_parameter(
        self,
        simple_trajectory: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """n_jobs parameter should not change results."""
        serial = _bin(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            np.pi / 30,
            n_jobs=1,
        )
        parallel = _bin(
            multiple_neuron_spikes,
            simple_trajectory["times"],
            simple_trajectory["headings"],
            np.pi / 30,
            n_jobs=2,
        )

        for a, b in zip(serial, parallel, strict=True):
            np.testing.assert_array_equal(a, b)


# ==============================================================================
# Test edge cases and error handling
# ==============================================================================


class TestDirectionalBinningEdgeCases:
    """Tests for edge cases in directional binning layer."""

    def test_all_spikes_at_one_direction(self) -> None:
        """Spikes at times with same heading should be binned together."""
        # Create trajectory with constant heading at π/2
        times = np.linspace(0, 10.0, 100)
        headings = np.ones(100) * np.pi / 2  # Constant heading

        # Multiple spikes at various times (all at same heading)
        spikes = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        spike_counts = _counts(spikes, times, headings, np.pi / 30)

        # All 5 spikes should be in one bin (around π/2)
        assert np.sum(spike_counts) == 5
        assert np.max(spike_counts) == 5

    def test_uniform_heading_distribution(self) -> None:
        """Uniform heading should give roughly uniform occupancy."""
        times = np.linspace(0, 10.0, 1001)  # 10 seconds, many samples
        # Linearly increasing heading (covers full circle uniformly)
        headings = np.linspace(0, 2 * np.pi, 1001)

        occupancy, _ = _occupancy(times, headings, np.pi / 6)  # 30° -> 12 bins

        # Each bin should have roughly equal occupancy
        # (with small variation due to edge effects)
        mean_occ = np.mean(occupancy)
        assert np.all(np.abs(occupancy - mean_occ) < 0.1 * mean_occ)

    def test_headings_wrap_around(self) -> None:
        """Headings crossing 0/2π boundary should be handled correctly."""
        times = np.linspace(0, 10.0, 100)
        # Heading crosses from ~2π to ~0 (wraps around)
        headings = np.linspace(1.9 * np.pi, 2.1 * np.pi, 100) % (2 * np.pi)

        spikes = np.array([5.0])  # One spike in the middle of the wrap

        spike_counts = _counts(spikes, times, headings, np.pi / 30)

        # Spike should be counted in one of the bins near 0 or 2π
        assert np.sum(spike_counts) == 1


class TestDirectionalBinningInputValidation:
    """Tests for input validation in directional binning."""

    def test_mismatched_times_headings_length(self) -> None:
        """Should raise error if times and headings have different lengths."""
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 50)  # Different length

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            prepare_directional_binning(times, headings, np.pi / 30)

    def test_insufficient_samples(self) -> None:
        """Should raise error with fewer than 3 samples."""
        times = np.array([0.0, 1.0])  # Only 2 samples
        headings = np.array([0.0, np.pi])

        with pytest.raises(ValueError, match=r"3|samples"):
            prepare_directional_binning(times, headings, np.pi / 30)

    def test_non_monotonic_times(self) -> None:
        """Should raise error if times are not monotonically increasing."""
        times = np.array([0.0, 2.0, 1.0, 3.0])  # Not monotonic
        headings = np.random.uniform(0, 2 * np.pi, 4)

        with pytest.raises(ValueError, match=r"monotonic|increasing"):
            prepare_directional_binning(times, headings, np.pi / 30)

    def test_invalid_angle_unit(self) -> None:
        """Should raise error for an invalid angle_unit."""
        times = np.linspace(0, 10, 100)
        headings = np.random.uniform(0, 2 * np.pi, 100)

        with pytest.raises(ValueError, match=r"angle_unit"):
            prepare_directional_binning(
                times,
                headings,
                np.pi / 30,
                angle_unit="invalid",  # type: ignore[arg-type]
            )


# ==============================================================================
# Non-finite headings must not be folded into bin 0
# ==============================================================================


class TestNonFiniteHeadingMasking:
    """NaN/Inf headings are excluded from occupancy AND spike counts."""

    def test_nan_heading_not_folded_into_bin0(
        self, nan_block_heading_trajectory
    ) -> None:
        d = nan_block_heading_trajectory
        occ_bad, _ = _occupancy(d["times"], d["headings"], np.pi / 30)
        occ_clean, _ = _occupancy(d["times_clean"], d["headings_clean"], np.pi / 30)
        # Bin-0 occupancy must match the trajectory with bad frames removed.
        assert occ_bad[0] == pytest.approx(occ_clean[0], abs=1e-9)

    def test_inf_heading_rejected_or_dropped(
        self, nan_block_heading_trajectory
    ) -> None:
        # Parametrized fixture also covers the Inf variant; assert the
        # non-finite frames are excluded from occupancy (no bin-0 inflation,
        # and total occupancy excludes the masked frames' deltas).
        d = nan_block_heading_trajectory
        occ_bad, _ = _occupancy(d["times"], d["headings"], np.pi / 30)
        occ_clean, _ = _occupancy(d["times_clean"], d["headings_clean"], np.pi / 30)
        # Non-finite frames are treated exactly like deleted rows: bin 0 is
        # not inflated, and the 3.3 s block (longer than max_gap) adds no time.
        np.testing.assert_allclose(occ_bad, occ_clean, rtol=0, atol=1e-9)
        times = d["times"]
        assert occ_bad.sum() == pytest.approx(
            times[-1] - times[0] - (times[600] - times[499]), abs=1e-9
        )

    def test_nan_heading_spikes_excluded(self, nan_block_heading_trajectory) -> None:
        d = nan_block_heading_trajectory
        times = d["times"]
        block = d["block"]
        # Place distinct spikes squarely inside the non-finite-heading block.
        block_times = times[block]
        spikes = np.linspace(block_times[1], block_times[-2], 10)
        counts_bad = _counts(spikes, times, d["headings"], np.pi / 30)
        # All spikes fall on non-finite-heading frames -> none counted.
        assert counts_bad.sum() == 0.0

        # Spikes on finite frames are counted as normal.
        finite_spike = np.array([times[10]])
        counts_finite = _counts(finite_spike, times, d["headings"], np.pi / 30)
        assert counts_finite.sum() == 1.0
