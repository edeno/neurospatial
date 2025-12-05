"""Tests for neurospatial.events.alignment module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray


class TestAlignSpikesToEvents:
    """Tests for align_spikes_to_events() function."""

    # --- Basic functionality ---

    def test_single_event_single_spike(self):
        """Single spike aligned to single event."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.5])
        event_times = np.array([10.0])
        window = (-1.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 1
        assert_array_almost_equal(result[0], [0.5])  # 10.5 - 10.0 = 0.5

    def test_single_event_multiple_spikes(self):
        """Multiple spikes aligned to single event."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([9.5, 10.0, 10.5, 11.0, 11.5])
        event_times = np.array([10.0])
        window = (-1.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 1
        assert_array_almost_equal(result[0], [-0.5, 0.0, 0.5, 1.0, 1.5])

    def test_multiple_events_single_spike_train(self):
        """Single spike train aligned to multiple events."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([0.5, 1.5, 10.5, 11.5, 20.5, 21.5])
        event_times = np.array([0.0, 10.0, 20.0])
        window = (-1.0, 3.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 3
        assert_array_almost_equal(result[0], [0.5, 1.5])
        assert_array_almost_equal(result[1], [0.5, 1.5])
        assert_array_almost_equal(result[2], [0.5, 1.5])

    def test_spike_at_event_time(self):
        """Spike exactly at event time should have relative time 0."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [0.0])

    def test_negative_relative_times(self):
        """Spikes before event have negative relative times."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([8.0, 9.0, 9.5])
        event_times = np.array([10.0])
        window = (-3.0, 0.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [-2.0, -1.0, -0.5])

    # --- Window boundary handling ---

    def test_window_inclusive_boundaries(self):
        """Spikes at exact window boundaries are included."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([9.0, 10.0, 12.0])  # At -1.0, 0.0, +2.0
        event_times = np.array([10.0])
        window = (-1.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [-1.0, 0.0, 2.0])

    def test_spikes_outside_window_excluded(self):
        """Spikes outside window are not included."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([5.0, 9.0, 10.0, 11.0, 15.0])
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [0.0])  # Only spike at 10.0

    def test_asymmetric_window(self):
        """Asymmetric window (more time before or after event)."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([8.0, 9.0, 10.0, 10.5])
        event_times = np.array([10.0])
        window = (-3.0, 1.0)  # More time before

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [-2.0, -1.0, 0.0, 0.5])

    # --- Empty cases ---

    def test_no_spikes_in_window(self):
        """Event with no spikes in window returns empty array."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([5.0, 15.0])  # All outside window
        event_times = np.array([10.0])
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 1
        assert len(result[0]) == 0
        assert result[0].dtype == np.float64

    def test_empty_spike_times(self):
        """Empty spike array returns list of empty arrays."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([])
        event_times = np.array([10.0, 20.0])
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == 0

    def test_empty_event_times(self):
        """Empty event array returns empty list."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0, 20.0])
        event_times = np.array([])
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 0
        assert isinstance(result, list)

    def test_both_empty(self):
        """Both empty returns empty list."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([])
        event_times = np.array([])
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 0

    # --- Overlapping events ---

    def test_overlapping_events_spike_counted_multiple_times(self):
        """Spike in overlapping windows appears in both event results."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0])
        event_times = np.array([9.5, 10.5])  # Windows overlap at spike
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert len(result) == 2
        assert_array_almost_equal(result[0], [0.5])  # 10.0 - 9.5 = 0.5
        assert_array_almost_equal(result[1], [-0.5])  # 10.0 - 10.5 = -0.5

    # --- Input validation ---

    def test_invalid_window_inverted(self):
        """Inverted window (start > end) raises ValueError."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (1.0, -1.0)  # Inverted

        with pytest.raises(ValueError, match=r"start.*end"):
            align_spikes_to_events(spike_times, event_times, window)

    def test_invalid_spike_times_nan(self):
        """NaN in spike times raises ValueError."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0, np.nan, 11.0])
        event_times = np.array([10.0])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match="NaN"):
            align_spikes_to_events(spike_times, event_times, window)

    def test_invalid_event_times_nan(self):
        """NaN in event times raises ValueError."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0])
        event_times = np.array([10.0, np.nan])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match="NaN"):
            align_spikes_to_events(spike_times, event_times, window)

    def test_invalid_spike_times_inf(self):
        """Inf in spike times raises ValueError."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.0, np.inf])
        event_times = np.array([10.0])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match="Inf"):
            align_spikes_to_events(spike_times, event_times, window)

    # --- Output properties ---

    def test_output_is_list_of_arrays(self):
        """Result is list of numpy arrays."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.5, 11.0])
        event_times = np.array([10.0, 20.0])
        window = (-1.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert isinstance(result, list)
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_output_dtype_float64(self):
        """Output arrays have float64 dtype."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([10.5])
        event_times = np.array([10.0])
        window = (-1.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert result[0].dtype == np.float64

    def test_output_sorted(self):
        """Output arrays are sorted by relative time."""
        from neurospatial.events.alignment import align_spikes_to_events

        # Unsorted spike times
        spike_times = np.array([11.5, 9.5, 10.5, 10.0])
        event_times = np.array([10.0])
        window = (-1.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        expected = np.array([-0.5, 0.0, 0.5, 1.5])
        assert_array_almost_equal(result[0], expected)

    # --- Unsorted inputs ---

    def test_unsorted_spike_times(self):
        """Unsorted spike times are handled correctly."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([11.0, 9.0, 10.0])  # Unsorted
        event_times = np.array([10.0])
        window = (-2.0, 2.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [-1.0, 0.0, 1.0])

    def test_unsorted_event_times(self):
        """Unsorted event times are handled correctly."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([0.5, 10.5, 20.5])
        event_times = np.array([20.0, 0.0, 10.0])  # Unsorted
        window = (-1.0, 1.0)

        result = align_spikes_to_events(spike_times, event_times, window)

        # Result order should match input event order
        assert len(result) == 3
        assert_array_almost_equal(result[0], [0.5])  # Event at 20.0
        assert_array_almost_equal(result[1], [0.5])  # Event at 0.0
        assert_array_almost_equal(result[2], [0.5])  # Event at 10.0

    # --- Typical use cases ---

    def test_raster_plot_data(self):
        """Typical raster plot preparation with realistic data."""
        from neurospatial.events.alignment import align_spikes_to_events

        # Simulate 30 Hz firing rate for 100 seconds
        rng = np.random.default_rng(42)
        n_spikes = 3000
        spike_times = np.sort(rng.uniform(0, 100, n_spikes))

        # 10 stimulus events at 10-second intervals
        event_times = np.arange(5, 100, 10)
        window = (-0.5, 1.0)  # 500ms before to 1s after

        result = align_spikes_to_events(spike_times, event_times, window)

        # Should have result for each event
        assert len(result) == len(event_times)

        # Each event should have some spikes (probabilistically)
        total_spikes = sum(len(arr) for arr in result)
        assert total_spikes > 0

        # All relative times should be within window
        for arr in result:
            if len(arr) > 0:
                assert arr.min() >= window[0]
                assert arr.max() <= window[1]

    def test_zero_width_window(self):
        """Zero-width window only captures exact matches."""
        from neurospatial.events.alignment import align_spikes_to_events

        spike_times = np.array([9.9, 10.0, 10.1])
        event_times = np.array([10.0])
        window = (0.0, 0.0)  # Zero width

        result = align_spikes_to_events(spike_times, event_times, window)

        assert_array_almost_equal(result[0], [0.0])  # Only exact match


class TestPeriEventHistogram:
    """Tests for peri_event_histogram() function."""

    # --- Basic functionality ---

    def test_single_event_uniform_spikes(self):
        """Uniform spikes around single event."""
        from neurospatial.events import PeriEventResult
        from neurospatial.events.alignment import peri_event_histogram

        # 10 spikes uniformly spaced in 1-second window
        spike_times = np.linspace(9.5, 10.5, 10)
        event_times = np.array([10.0])
        window = (-0.5, 0.5)
        bin_size = 0.1

        result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        assert isinstance(result, PeriEventResult)
        assert result.n_events == 1
        assert result.bin_size == bin_size
        assert result.window == window
        assert len(result.bin_centers) == 10  # 1.0 / 0.1 = 10 bins

    def test_multiple_events_consistent_response(self):
        """Multiple events with consistent spike pattern."""
        from neurospatial.events.alignment import peri_event_histogram

        # Create consistent spike pattern around each event
        event_times = np.array([10.0, 20.0, 30.0])
        spike_times = np.concatenate(
            [
                event_times - 0.3,  # Spike 300ms before each event
                event_times + 0.2,  # Spike 200ms after each event
            ]
        )
        window = (-0.5, 0.5)
        bin_size = 0.1

        result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        assert result.n_events == 3
        # Should have 1 spike per event in bins around -0.3 and +0.2
        assert result.histogram.sum() > 0

    def test_bin_centers_correct(self):
        """Bin centers are correctly computed."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (-1.0, 1.0)
        bin_size = 0.5

        result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        # Window is 2.0s, bin_size 0.5s → 4 bins
        # Centers at: -0.75, -0.25, 0.25, 0.75
        expected_centers = np.array([-0.75, -0.25, 0.25, 0.75])
        assert_array_almost_equal(result.bin_centers, expected_centers)

    def test_firing_rate_conversion(self):
        """firing_rate() correctly converts counts to Hz."""
        from neurospatial.events.alignment import peri_event_histogram

        # Place 10 spikes in first bin (0.1s bin)
        spike_times = np.full(10, 9.55)  # All in first bin around -0.45
        event_times = np.array([10.0])
        window = (-0.5, 0.5)
        bin_size = 0.1

        result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        # First bin has 10 spikes in 0.1s → 100 Hz
        rates = result.firing_rate()
        assert rates[0] == pytest.approx(100.0, rel=0.01)

    def test_sem_with_multiple_events(self):
        """SEM is computed correctly across events."""
        from neurospatial.events.alignment import peri_event_histogram

        # Create variable response across events
        event_times = np.array([10.0, 20.0, 30.0])
        # Event 1: 2 spikes, Event 2: 4 spikes, Event 3: 3 spikes (in first bin)
        spike_times = np.array(
            [
                9.55,
                9.56,  # Event 1
                19.55,
                19.56,
                19.57,
                19.58,  # Event 2
                29.55,
                29.56,
                29.57,  # Event 3
            ]
        )
        window = (-0.5, 0.5)
        bin_size = 0.1

        result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        # SEM should be non-zero for first bin (variable counts)
        assert result.sem[0] > 0

        # SEM = std / sqrt(n_events) for first bin
        # Counts: [2, 4, 3], mean=3, std=1, sem=1/sqrt(3)≈0.577
        expected_sem = np.std([2, 4, 3], ddof=1) / np.sqrt(3)
        assert result.sem[0] == pytest.approx(expected_sem, rel=0.1)

    def test_sem_single_event_is_nan(self):
        """SEM is NaN for single event (undefined)."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        with pytest.warns(UserWarning, match="single event"):
            result = peri_event_histogram(spike_times, event_times, window)

        assert np.all(np.isnan(result.sem))

    # --- Edge cases ---

    def test_empty_spike_times(self):
        """Empty spike array returns zero histogram."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([])
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 0.5)

        result = peri_event_histogram(spike_times, event_times, window)

        assert np.all(result.histogram == 0)
        assert result.n_events == 2

    def test_empty_event_times(self):
        """Empty event array raises ValueError."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([])
        window = (-0.5, 0.5)

        with pytest.raises(ValueError, match="event"):
            peri_event_histogram(spike_times, event_times, window)

    def test_no_spikes_in_any_window(self):
        """No spikes in any window returns zero histogram."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([0.0, 100.0])  # Far from events
        event_times = np.array([50.0])
        window = (-0.5, 0.5)

        result = peri_event_histogram(spike_times, event_times, window)

        assert np.all(result.histogram == 0)

    # --- Validation ---

    def test_invalid_window_inverted(self):
        """Inverted window raises ValueError."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (0.5, -0.5)  # Inverted

        with pytest.raises(ValueError, match=r"start.*end"):
            peri_event_histogram(spike_times, event_times, window)

    def test_invalid_bin_size_zero(self):
        """Zero bin_size raises ValueError."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        with pytest.raises(ValueError, match="bin_size"):
            peri_event_histogram(spike_times, event_times, window, bin_size=0.0)

    def test_invalid_bin_size_negative(self):
        """Negative bin_size raises ValueError."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        with pytest.raises(ValueError, match="bin_size"):
            peri_event_histogram(spike_times, event_times, window, bin_size=-0.1)

    # --- Output properties ---

    def test_output_shapes_match(self):
        """All output arrays have consistent shapes."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.linspace(9.0, 11.0, 100)
        event_times = np.array([10.0])
        window = (-1.0, 1.0)
        bin_size = 0.1

        result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        n_bins = len(result.bin_centers)
        assert len(result.histogram) == n_bins
        assert len(result.sem) == n_bins

    def test_histogram_non_negative(self):
        """Histogram values are non-negative."""
        from neurospatial.events.alignment import peri_event_histogram

        rng = np.random.default_rng(42)
        spike_times = rng.uniform(0, 100, 1000)
        event_times = np.arange(10, 100, 10)
        window = (-0.5, 0.5)

        result = peri_event_histogram(spike_times, event_times, window)

        assert np.all(result.histogram >= 0)

    # --- Typical use case ---

    def test_psth_realistic_data(self):
        """Realistic PSTH with stimulus-evoked response."""
        from neurospatial.events.alignment import peri_event_histogram

        rng = np.random.default_rng(42)

        # 10 stimulus events
        stim_times = np.arange(5, 100, 10)

        # Generate spikes with elevated rate after stimulus
        baseline_spikes = rng.uniform(0, 100, 500)  # 5 Hz baseline
        evoked_spikes = []
        for stim_t in stim_times:
            # Elevated response 0-200ms after stimulus
            n_evoked = rng.poisson(20)  # ~100 Hz burst
            evoked_spikes.extend(stim_t + rng.uniform(0, 0.2, n_evoked))

        spike_times = np.sort(np.concatenate([baseline_spikes, evoked_spikes]))

        result = peri_event_histogram(
            spike_times, stim_times, window=(-0.5, 1.0), bin_size=0.05
        )

        assert result.n_events == len(stim_times)
        # Post-stimulus bins should have higher rate than pre-stimulus
        pre_stim_rate = result.firing_rate()[: len(result.bin_centers) // 3].mean()
        post_stim_rate = result.firing_rate()[
            len(result.bin_centers) // 3 : 2 * len(result.bin_centers) // 3
        ].mean()
        assert post_stim_rate > pre_stim_rate

    def test_default_bin_size(self):
        """Default bin_size is used when not specified."""
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        # Should not raise - uses default bin_size
        result = peri_event_histogram(spike_times, event_times, window)

        assert result.bin_size > 0


class TestPopulationPeriEventHistogram:
    """Tests for population_peri_event_histogram() function."""

    # --- Basic functionality ---

    def test_two_units_basic(self):
        """Basic test with two units."""
        from neurospatial.events import PopulationPeriEventResult
        from neurospatial.events.alignment import population_peri_event_histogram

        # Two units with different spike patterns
        spike_trains = [
            np.array([9.5, 10.5]),  # Unit 1
            np.array([9.8, 10.2, 10.8]),  # Unit 2
        ]
        event_times = np.array([10.0])
        window = (-1.0, 1.0)
        bin_size = 0.5

        result = population_peri_event_histogram(
            spike_trains, event_times, window, bin_size=bin_size
        )

        assert isinstance(result, PopulationPeriEventResult)
        assert result.n_units == 2
        assert result.n_events == 1
        assert result.histograms.shape == (2, 4)  # 2 units, 4 bins

    def test_population_mean_histogram(self):
        """Mean histogram is average across units."""
        from neurospatial.events.alignment import population_peri_event_histogram

        # Two units with identical patterns
        spike_trains = [
            np.array([10.1]),  # Unit 1: 1 spike in second bin
            np.array([10.1]),  # Unit 2: same pattern
        ]
        event_times = np.array([10.0, 20.0])  # 2 events, so mean is 0.5 per event
        window = (-0.5, 0.5)
        bin_size = 0.25

        result = population_peri_event_histogram(
            spike_trains, event_times, window, bin_size=bin_size
        )

        # Mean histogram should equal per-unit histogram (since identical)
        assert_array_almost_equal(result.mean_histogram, result.histograms[0])

    def test_firing_rates_conversion(self):
        """firing_rates() correctly converts to Hz for all units."""
        from neurospatial.events.alignment import population_peri_event_histogram

        spike_trains = [
            np.array([10.05] * 10),  # 10 spikes in one bin
            np.array([10.05] * 5),  # 5 spikes in same bin
        ]
        event_times = np.array([10.0])
        window = (-0.5, 0.5)
        bin_size = 0.1

        result = population_peri_event_histogram(
            spike_trains, event_times, window, bin_size=bin_size
        )

        rates = result.firing_rates()
        assert rates.shape == result.histograms.shape
        # Unit 1: 10 spikes in 0.1s = 100 Hz
        # Unit 2: 5 spikes in 0.1s = 50 Hz
        assert rates[0, 5] == pytest.approx(100.0, rel=0.1)
        assert rates[1, 5] == pytest.approx(50.0, rel=0.1)

    def test_sem_per_unit(self):
        """SEM is computed per unit across events."""
        from neurospatial.events.alignment import population_peri_event_histogram

        # Variable response across events for each unit
        spike_trains = [
            np.array([10.05, 20.05, 20.06]),  # Unit 1: 1, 2 spikes
            np.array([10.05, 10.06, 10.07, 20.05]),  # Unit 2: 3, 1 spikes
        ]
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 0.5)
        bin_size = 0.1

        result = population_peri_event_histogram(
            spike_trains, event_times, window, bin_size=bin_size
        )

        # SEM should have same shape as histograms
        assert result.sem.shape == result.histograms.shape
        # SEM should be non-zero where there's variance
        # Unit 1 in bin 5: counts [1, 2], mean=1.5, sem > 0
        assert result.sem[0, 5] > 0

    # --- Edge cases ---

    def test_empty_spike_train(self):
        """Unit with no spikes returns zero histogram."""
        from neurospatial.events.alignment import population_peri_event_histogram

        spike_trains = [
            np.array([10.0]),  # Unit 1: has spikes
            np.array([]),  # Unit 2: no spikes
        ]
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        result = population_peri_event_histogram(spike_trains, event_times, window)

        assert result.n_units == 2
        assert np.all(result.histograms[1] == 0)

    def test_single_unit(self):
        """Single unit behaves like peri_event_histogram."""
        from neurospatial.events.alignment import (
            peri_event_histogram,
            population_peri_event_histogram,
        )

        spike_times = np.array([9.8, 10.2, 10.5])
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 1.0)
        bin_size = 0.1

        # Single unit population
        pop_result = population_peri_event_histogram(
            [spike_times], event_times, window, bin_size=bin_size
        )

        # Should match single-unit PSTH
        single_result = peri_event_histogram(
            spike_times, event_times, window, bin_size=bin_size
        )

        assert pop_result.n_units == 1
        assert_array_almost_equal(pop_result.histograms[0], single_result.histogram)

    def test_empty_event_times(self):
        """Empty event times raises ValueError."""
        from neurospatial.events.alignment import population_peri_event_histogram

        spike_trains = [np.array([10.0])]
        event_times = np.array([])
        window = (-0.5, 0.5)

        with pytest.raises(ValueError, match="event"):
            population_peri_event_histogram(spike_trains, event_times, window)

    def test_empty_spike_trains_list(self):
        """Empty spike trains list raises ValueError."""
        from neurospatial.events.alignment import population_peri_event_histogram

        spike_trains: list[NDArray[np.float64]] = []
        event_times = np.array([10.0])
        window = (-0.5, 0.5)

        with pytest.raises(ValueError, match="spike_trains"):
            population_peri_event_histogram(spike_trains, event_times, window)

    # --- Output properties ---

    def test_output_shapes_consistent(self):
        """All output arrays have consistent shapes."""
        from neurospatial.events.alignment import population_peri_event_histogram

        spike_trains = [
            np.array([10.0, 10.5]),
            np.array([10.2, 10.7, 10.9]),
            np.array([10.1]),
        ]
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 1.0)
        bin_size = 0.1

        result = population_peri_event_histogram(
            spike_trains, event_times, window, bin_size=bin_size
        )

        n_units = 3
        n_bins = len(result.bin_centers)

        assert result.histograms.shape == (n_units, n_bins)
        assert result.sem.shape == (n_units, n_bins)
        assert result.mean_histogram.shape == (n_bins,)
        assert result.n_units == n_units

    # --- Typical use case ---

    def test_multi_unit_recording(self):
        """Typical multi-unit recording scenario."""
        from neurospatial.events.alignment import population_peri_event_histogram

        rng = np.random.default_rng(42)

        # 5 units with different baseline rates
        n_units = 5
        spike_trains = []
        for i in range(n_units):
            baseline_rate = 5 + i * 2  # 5, 7, 9, 11, 13 Hz
            n_spikes = int(baseline_rate * 100)  # 100 seconds
            spike_trains.append(np.sort(rng.uniform(0, 100, n_spikes)))

        # 10 stimulus events
        event_times = np.arange(5, 100, 10)
        window = (-0.5, 1.0)

        result = population_peri_event_histogram(
            spike_trains, event_times, window, bin_size=0.05
        )

        assert result.n_units == n_units
        assert result.n_events == len(event_times)
        # Higher-rate units should have higher mean firing
        rates = result.firing_rates().mean(axis=1)  # Mean rate per unit
        assert np.all(np.diff(rates) > 0)  # Should increase with unit index


class TestAlignEvents:
    """Tests for align_events() function."""

    # --- Basic functionality ---

    def test_single_reference_single_event(self):
        """Single event aligned to single reference."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [10.5]})
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (-1.0, 2.0)

        result = align_events(events, reference, window)

        assert len(result) == 1
        assert result["relative_time"].iloc[0] == pytest.approx(0.5)
        assert result["reference_index"].iloc[0] == 0

    def test_multiple_events_single_reference(self):
        """Multiple events around single reference."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [9.5, 10.5, 11.0]})
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (-1.0, 2.0)

        result = align_events(events, reference, window)

        assert len(result) == 3
        assert_array_almost_equal(result["relative_time"].values, [-0.5, 0.5, 1.0])
        assert all(result["reference_index"] == 0)

    def test_multiple_references(self):
        """Events aligned to multiple references."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [10.5, 20.5]})
        reference = pd.DataFrame({"timestamp": [10.0, 20.0]})
        window = (-1.0, 2.0)

        result = align_events(events, reference, window)

        assert len(result) == 2
        # Event at 10.5 aligned to reference at 10.0
        assert result["relative_time"].iloc[0] == pytest.approx(0.5)
        assert result["reference_index"].iloc[0] == 0
        # Event at 20.5 aligned to reference at 20.0
        assert result["relative_time"].iloc[1] == pytest.approx(0.5)
        assert result["reference_index"].iloc[1] == 1

    def test_event_in_overlapping_windows_counted_twice(self):
        """Event in overlapping windows appears for both references."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [15.0]})
        reference = pd.DataFrame({"timestamp": [10.0, 20.0]})
        window = (-10.0, 10.0)  # Overlapping windows

        result = align_events(events, reference, window)

        assert len(result) == 2
        # Event at 15.0: relative to ref 10.0 is +5.0
        mask_ref0 = result["reference_index"] == 0
        assert result.loc[mask_ref0, "relative_time"].iloc[0] == pytest.approx(5.0)
        # Event at 15.0: relative to ref 20.0 is -5.0
        mask_ref1 = result["reference_index"] == 1
        assert result.loc[mask_ref1, "relative_time"].iloc[0] == pytest.approx(-5.0)

    def test_preserves_event_columns(self):
        """Original event columns are preserved."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame(
            {
                "timestamp": [10.5],
                "event_type": ["reward"],
                "value": [100],
            }
        )
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (-1.0, 2.0)

        result = align_events(events, reference, window)

        assert "event_type" in result.columns
        assert "value" in result.columns
        assert result["event_type"].iloc[0] == "reward"
        assert result["value"].iloc[0] == 100

    # --- Edge cases ---

    def test_no_events_in_window(self):
        """No events within window returns empty DataFrame."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [0.0, 100.0]})  # Far from reference
        reference = pd.DataFrame({"timestamp": [50.0]})
        window = (-1.0, 1.0)

        result = align_events(events, reference, window)

        assert len(result) == 0
        assert "relative_time" in result.columns
        assert "reference_index" in result.columns

    def test_empty_events(self):
        """Empty events DataFrame returns empty result."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": []})
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (-1.0, 1.0)

        result = align_events(events, reference, window)

        assert len(result) == 0

    def test_empty_reference(self):
        """Empty reference DataFrame returns empty result."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [10.0]})
        reference = pd.DataFrame({"timestamp": []})
        window = (-1.0, 1.0)

        result = align_events(events, reference, window)

        assert len(result) == 0

    def test_event_at_window_boundary_included(self):
        """Events at exact window boundaries are included."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [9.0, 12.0]})  # At -1.0 and +2.0
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (-1.0, 2.0)

        result = align_events(events, reference, window)

        assert len(result) == 2
        assert_array_almost_equal(result["relative_time"].values, [-1.0, 2.0])

    # --- Custom column names ---

    def test_custom_timestamp_columns(self):
        """Custom timestamp column names work."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"event_time": [10.5]})
        reference = pd.DataFrame({"ref_time": [10.0]})
        window = (-1.0, 2.0)

        result = align_events(
            events,
            reference,
            window,
            event_column="event_time",
            reference_column="ref_time",
        )

        assert len(result) == 1
        assert result["relative_time"].iloc[0] == pytest.approx(0.5)

    # --- Validation ---

    def test_missing_event_column(self):
        """Missing event column raises ValueError."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"time": [10.0]})  # Wrong column name
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match="timestamp"):
            align_events(events, reference, window)

    def test_missing_reference_column(self):
        """Missing reference column raises ValueError."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [10.0]})
        reference = pd.DataFrame({"time": [10.0]})  # Wrong column name
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match="timestamp"):
            align_events(events, reference, window)

    def test_invalid_window_inverted(self):
        """Inverted window raises ValueError."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        events = pd.DataFrame({"timestamp": [10.0]})
        reference = pd.DataFrame({"timestamp": [10.0]})
        window = (1.0, -1.0)  # Inverted

        with pytest.raises(ValueError, match=r"start.*end"):
            align_events(events, reference, window)

    # --- Typical use case ---

    def test_align_licks_to_rewards(self):
        """Typical use case: align lick events to reward delivery."""
        import pandas as pd

        from neurospatial.events.alignment import align_events

        # Licks around reward times
        licks = pd.DataFrame(
            {
                "timestamp": [9.8, 10.1, 10.2, 10.5, 19.9, 20.2, 20.3],
                "lick_strength": [0.8, 0.9, 0.7, 0.5, 0.9, 0.8, 0.6],
            }
        )
        rewards = pd.DataFrame({"timestamp": [10.0, 20.0]})
        window = (-0.5, 1.0)

        result = align_events(licks, rewards, window)

        # Should have 7 licks aligned
        assert len(result) == 7
        # Lick strength should be preserved
        assert "lick_strength" in result.columns
        # All relative times should be within window
        assert all(result["relative_time"] >= window[0])
        assert all(result["relative_time"] <= window[1])


class TestPlotPeriEventHistogram:
    """Tests for plot_peri_event_histogram() function."""

    def test_plot_creates_axes(self):
        """Plot returns matplotlib axes."""
        import matplotlib.pyplot as plt

        from neurospatial.events._core import plot_peri_event_histogram
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([9.8, 10.2, 10.5, 20.1, 20.3])
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 1.0)

        result = peri_event_histogram(spike_times, event_times, window)
        ax = plot_peri_event_histogram(result)

        assert ax is not None
        assert hasattr(ax, "plot")  # Is a matplotlib axes
        plt.close("all")

    def test_plot_with_custom_axes(self):
        """Plot on provided axes."""
        import matplotlib.pyplot as plt

        from neurospatial.events._core import plot_peri_event_histogram
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0, 20.0])
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 1.0)

        result = peri_event_histogram(spike_times, event_times, window)

        _fig, custom_ax = plt.subplots()
        returned_ax = plot_peri_event_histogram(result, ax=custom_ax)

        assert returned_ax is custom_ax
        plt.close("all")

    def test_plot_as_rate(self):
        """Plot as firing rate (Hz)."""
        import matplotlib.pyplot as plt

        from neurospatial.events._core import plot_peri_event_histogram
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 0.5)

        result = peri_event_histogram(spike_times, event_times, window)
        ax = plot_peri_event_histogram(result, as_rate=True)

        # Check y-axis label indicates Hz
        ylabel = ax.get_ylabel()
        assert "Hz" in ylabel or "rate" in ylabel.lower()
        plt.close("all")

    def test_plot_as_counts(self):
        """Plot as spike counts."""
        import matplotlib.pyplot as plt

        from neurospatial.events._core import plot_peri_event_histogram
        from neurospatial.events.alignment import peri_event_histogram

        spike_times = np.array([10.0])
        event_times = np.array([10.0, 20.0])
        window = (-0.5, 0.5)

        result = peri_event_histogram(spike_times, event_times, window)
        ax = plot_peri_event_histogram(result, as_rate=False)

        # Check y-axis label indicates counts
        ylabel = ax.get_ylabel()
        assert "count" in ylabel.lower()
        plt.close("all")
