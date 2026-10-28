"""Tests for events module GLM regressors (time-based)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# Test time_to_nearest_event
# =============================================================================


class TestTimeToNearestEvent:
    """Tests for time_to_nearest_event function."""

    def test_basic_signed_functionality(self):
        """Test basic signed time to nearest event calculation."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        event_times = np.array([2.0])  # Single event at t=2.0

        result = time_to_nearest_event(sample_times, event_times)

        # Before event: negative (approaching)
        assert_allclose(result[0], -2.0)  # t=0.0, 2s before event
        assert_allclose(result[1], -1.0)  # t=1.0, 1s before event
        # At event: zero
        assert_allclose(result[2], 0.0)  # t=2.0, at event
        # After event: positive (elapsed)
        assert_allclose(result[3], 1.0)  # t=3.0, 1s after event
        assert_allclose(result[4], 2.0)  # t=4.0, 2s after event
        assert_allclose(result[5], 3.0)  # t=5.0, 3s after event

    def test_multiple_events_nearest_wins(self):
        """Test that nearest event is selected when multiple events exist."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        event_times = np.array([2.0, 5.0])  # Events at t=2.0 and t=5.0

        result = time_to_nearest_event(sample_times, event_times)

        # Closer to first event (2.0)
        assert_allclose(result[0], -2.0)  # t=0.0, nearest is 2.0
        assert_allclose(result[1], -1.0)  # t=1.0, nearest is 2.0
        assert_allclose(result[2], 0.0)  # t=2.0, at first event
        assert_allclose(result[3], 1.0)  # t=3.0, nearest is 2.0 (1s after)
        # Midpoint between events - closer to second
        # t=3.5 would be equidistant, t=4.0 is closer to 5.0
        assert_allclose(result[4], -1.0)  # t=4.0, nearest is 5.0 (1s before)
        assert_allclose(result[5], 0.0)  # t=5.0, at second event
        assert_allclose(result[6], 1.0)  # t=6.0, 1s after second event

    def test_midpoint_tie_breaking(self):
        """Test behavior at exact midpoint between events."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([3.0])  # Exactly between 2.0 and 4.0
        event_times = np.array([2.0, 4.0])

        result = time_to_nearest_event(sample_times, event_times)

        # At midpoint, either +1.0 or -1.0 is valid
        # Implementation should pick one consistently (we'll use the earlier event)
        assert np.abs(result[0]) == 1.0

    def test_unsigned_mode(self):
        """Test unsigned (absolute distance) mode."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([2.0])

        result = time_to_nearest_event(sample_times, event_times, signed=False)

        # All positive (absolute distance)
        assert_allclose(result[0], 2.0)  # |0.0 - 2.0|
        assert_allclose(result[1], 1.0)  # |1.0 - 2.0|
        assert_allclose(result[2], 0.0)  # |2.0 - 2.0|
        assert_allclose(result[3], 1.0)  # |3.0 - 2.0|
        assert_allclose(result[4], 2.0)  # |4.0 - 2.0|

    def test_sample_at_event_time(self):
        """Test sample exactly at event time returns zero."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([1.0, 2.0, 3.0])
        event_times = np.array([2.0])

        result = time_to_nearest_event(sample_times, event_times)

        assert_allclose(result[1], 0.0)  # Exactly at event

    def test_max_time_clips_values(self):
        """Test max_time parameter clips large values."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 5.0, 10.0, 15.0])
        event_times = np.array([5.0])

        result = time_to_nearest_event(sample_times, event_times, max_time=3.0)

        assert_allclose(result[0], -3.0)  # -5.0 clipped to -3.0
        assert_allclose(result[1], 0.0)  # At event
        assert_allclose(result[2], 3.0)  # 5.0 clipped to 3.0
        assert_allclose(result[3], 3.0)  # 10.0 clipped to 3.0

    def test_max_time_clips_unsigned(self):
        """Test max_time clipping in unsigned mode."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 5.0, 10.0])
        event_times = np.array([5.0])

        result = time_to_nearest_event(
            sample_times, event_times, max_time=3.0, signed=False
        )

        assert_allclose(result[0], 3.0)  # 5.0 clipped to 3.0
        assert_allclose(result[1], 0.0)  # At event
        assert_allclose(result[2], 3.0)  # 5.0 clipped to 3.0

    def test_empty_events_returns_nan(self):
        """Test empty events array returns all NaN."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([])

        result = time_to_nearest_event(sample_times, event_times)

        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5])

        result = time_to_nearest_event(sample_times, event_times)

        assert_allclose(result[0], -1.5)  # Before
        assert_allclose(result[1], -0.5)  # Before
        assert_allclose(result[2], 0.5)  # After
        assert_allclose(result[3], 1.5)  # After

    def test_unsorted_events_handled(self):
        """Test unsorted event times are handled correctly."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 2.5, 5.0])
        event_times = np.array([4.0, 1.0])  # Unsorted: should be [1.0, 4.0]

        result = time_to_nearest_event(sample_times, event_times)

        # t=0.0: nearest is 1.0, time = -1.0
        assert_allclose(result[0], -1.0)
        # t=2.5: midpoint between 1.0 and 4.0, nearest could be either
        # 2.5-1.0=1.5, 4.0-2.5=1.5, tie - pick earlier = 1.5 after 1.0
        assert_allclose(np.abs(result[1]), 1.5)
        # t=5.0: nearest is 4.0, time = 1.0
        assert_allclose(result[2], 1.0)

    def test_multiple_events_same_time(self):
        """Test multiple events at same time."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5, 1.5, 1.5])  # Three events at same time

        result = time_to_nearest_event(sample_times, event_times)

        # Should behave as if single event at 1.5
        assert_allclose(result[0], -1.5)
        assert_allclose(result[1], -0.5)
        assert_allclose(result[2], 0.5)
        assert_allclose(result[3], 1.5)

    def test_nan_in_sample_times_raises(self):
        """Test NaN in sample_times raises ValueError."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, np.nan, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"^times.*NaN"):
            time_to_nearest_event(sample_times, event_times)

    def test_nan_in_event_times_raises(self):
        """Test NaN in event_times raises ValueError."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.nan])

        with pytest.raises(ValueError, match=r"event_times.*NaN"):
            time_to_nearest_event(sample_times, event_times)

    def test_inf_in_sample_times_raises(self):
        """Test Inf in sample_times raises ValueError."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, np.inf, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"^times.*inf"):
            time_to_nearest_event(sample_times, event_times)

    def test_inf_in_event_times_raises(self):
        """Test Inf in event_times raises ValueError."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.inf])

        with pytest.raises(ValueError, match=r"event_times.*inf"):
            time_to_nearest_event(sample_times, event_times)

    def test_negative_max_time_raises(self):
        """Test negative max_time raises ValueError."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"max_time.*non-negative"):
            time_to_nearest_event(sample_times, event_times, max_time=-1.0)

    def test_nan_max_time_raises(self):
        """NaN would otherwise clip every value to NaN."""
        from neurospatial.events.regressors import time_to_nearest_event

        with pytest.raises(ValueError, match=r"max_time.*non-negative"):
            time_to_nearest_event(np.arange(3.0), np.array([1.0]), max_time=np.nan)

    @pytest.mark.parametrize("max_time", [True, np.inf, "1.0"])
    def test_non_finite_or_non_numeric_max_time_raises(self, max_time):
        """True silently clipped to 1 s; inf clipped nothing, which None says."""
        from neurospatial.events.regressors import time_to_nearest_event

        with pytest.raises(ValueError, match=r"max_time.*finite non-negative"):
            time_to_nearest_event(np.arange(3.0), np.array([1.0]), max_time=max_time)

    def test_zero_max_time_clips_to_zero(self):
        """Test max_time=0.0 clips all values to zero."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5])

        result = time_to_nearest_event(sample_times, event_times, max_time=0.0)

        assert_allclose(result, np.array([0.0, 0.0, 0.0, 0.0]))

    def test_dense_events(self):
        """Test with many closely spaced events."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        event_times = np.array([0.25, 0.75, 1.25, 1.75])

        result = time_to_nearest_event(sample_times, event_times)

        # Each sample should be within 0.25 of nearest event
        assert np.all(np.abs(result) <= 0.25)

    def test_sign_convention_like_psth(self):
        """Test that sign convention matches PSTH x-axis (negative before, positive after)."""
        from neurospatial.events.regressors import time_to_nearest_event

        # This is the key use case: creating PSTH-like time axis
        sample_times = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # Centered around 0
        event_times = np.array([0.0])

        result = time_to_nearest_event(sample_times, event_times)

        # Should match input since event is at 0
        assert_allclose(result, sample_times)

    def test_consistent_with_peri_event_window(self):
        """Test that output can be used to filter peri-event windows."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.linspace(0, 10, 101)  # 0 to 10 in 0.1s steps
        event_times = np.array([3.0, 7.0])

        result = time_to_nearest_event(sample_times, event_times)

        # Filter to +/- 1s window around events
        window_mask = np.abs(result) <= 1.0

        # Should include samples near events
        assert window_mask[30]  # t=3.0, at event
        assert window_mask[25]  # t=2.5, 0.5s before event
        assert window_mask[35]  # t=3.5, 0.5s after event
        assert window_mask[70]  # t=7.0, at event

        # Should exclude samples far from events
        assert not window_mask[0]  # t=0.0, far from events
        assert not window_mask[50]  # t=5.0, equidistant but >1s from both


# =============================================================================
# Test event_count_in_window
# =============================================================================


class TestEventCountInWindow:
    """Tests for event_count_in_window function."""

    def test_basic_backward_window(self):
        """Test counting events in backward window (past events)."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        event_times = np.array([1.0, 2.5])
        window = (-2.0, 0.0)  # Look 2s into the past

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=0.0: window [-2, 0), no events
        assert result[0] == 0
        # t=1.0: window [-1, 1), the event at 1.0 is the stop (excluded)
        assert result[1] == 0
        # t=2.0: window [0, 2), event at 1.0 included
        assert result[2] == 1
        # t=3.0: window [1, 3), events at 1.0 (the start) and 2.5 included
        assert result[3] == 2
        # t=4.0: window [2, 4), event at 2.5 included
        assert result[4] == 1
        # t=5.0: window [3, 5), no events
        assert result[5] == 0

    def test_basic_forward_window(self):
        """Test counting events in forward window (future events)."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([2.0, 3.5])
        window = (0.0, 2.0)  # Look 2s into the future

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=0.0: window [0, 2), the event at 2.0 is the stop (excluded)
        assert result[0] == 0
        # t=1.0: window [1, 3), event at 2.0 included
        assert result[1] == 1
        # t=2.0: window [2, 4), events at 2.0 (the start) and 3.5 included
        assert result[2] == 2
        # t=3.0: window [3, 5), event at 3.5 included
        assert result[3] == 1
        # t=4.0: window [4, 6), no events
        assert result[4] == 0

    def test_symmetric_window(self):
        """Test counting events in symmetric window around sample."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 2.0, 5.0])
        event_times = np.array([1.0, 3.0, 4.0])
        window = (-1.5, 1.5)  # +/- 1.5s around each sample

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=0.0: window [-1.5, 1.5], event at 1.0 included
        assert result[0] == 1
        # t=2.0: window [0.5, 3.5], events at 1.0, 3.0 included
        assert result[1] == 2
        # t=5.0: window [3.5, 6.5], event at 4.0 included
        assert result[2] == 1

    def test_empty_events_returns_zeros(self):
        """Test empty events array returns all zeros."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([])
        window = (-1.0, 1.0)

        result = event_count_in_window(sample_times, event_times, window=window)

        assert np.all(result == 0)
        assert len(result) == 3

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5])
        window = (-1.0, 1.0)  # +/- 1s

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=0.0: window [-1, 1], event at 1.5 not in window
        assert result[0] == 0
        # t=1.0: window [0, 2], event at 1.5 included
        assert result[1] == 1
        # t=2.0: window [1, 3], event at 1.5 included
        assert result[2] == 1
        # t=3.0: window [2, 4], event at 1.5 not in window
        assert result[3] == 0

    def test_multiple_events_same_time(self):
        """Test multiple events at same timestamp are counted separately."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, 1.0, 1.0])  # Three events at same time
        window = (-0.5, 0.5)

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=0.0: window [-0.5, 0.5], no events
        assert result[0] == 0
        # t=1.0: window [0.5, 1.5], 3 events at 1.0 included
        assert result[1] == 3
        # t=2.0: window [1.5, 2.5], no events
        assert result[2] == 0

    def test_window_is_half_open(self):
        """An event at the window start is counted; one at the stop is not."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([2.0])
        event_times = np.array([1.0, 3.0])  # Events at exact boundaries
        window = (-1.0, 1.0)

        result = event_count_in_window(sample_times, event_times, window=window)

        # window [1.0, 3.0): the start is in, the stop out
        assert result[0] == 1

    def test_bin_centered_windows_count_each_event_once(self):
        """Half-open windows tile time, so no event is counted at two samples."""
        from neurospatial.events.regressors import event_count_in_window

        dt = 0.25
        sample_times = np.arange(41) * dt
        # Events on bin edges (sample +/- dt/2) and between them.
        event_times = np.r_[np.arange(1, 40) * dt - dt / 2, [1.1, 3.3, 7.9]]

        counts = event_count_in_window(
            sample_times, event_times, window=(-dt / 2, dt / 2)
        )

        assert counts.sum() == len(event_times)

    @pytest.mark.parametrize("clock_start", [0.0, 5000.0])
    def test_bin_edges_on_a_non_dyadic_clock_count_each_event_once(self, clock_start):
        """Events on a 20 Hz camera clock fall on the edges of 10 Hz bins.

        0.1 s is not exact in binary, so each edge is computed twice with
        different rounding; every event must still land at one sample, the
        one whose bin holds it.
        """
        from neurospatial.events.regressors import event_count_in_window

        dt = 0.1
        sample_times = clock_start + np.arange(1000) * dt
        frames = clock_start + np.arange(1, 1999) * (dt / 2)
        midpoints = sample_times[:-1] + dt / 2

        for event_times in (frames, midpoints):
            counts = event_count_in_window(
                sample_times, event_times, window=(-dt / 2, dt / 2)
            )
            assert counts.sum() == event_times.size
            assert counts.max() <= 2

    @pytest.mark.parametrize("window", [(np.nan, 1.0), (-1.0, np.inf)])
    def test_non_finite_window_is_refused(self, window):
        """A NaN bound would otherwise produce negative counts."""
        from neurospatial.events.regressors import event_count_in_window

        with pytest.raises(ValueError, match="finite bounds"):
            event_count_in_window(np.arange(5.0), np.array([1.0, 2.0]), window=window)

    def test_two_dimensional_times_are_refused(self):
        from neurospatial.events.regressors import event_count_in_window

        with pytest.raises(ValueError, match="1-D"):
            event_count_in_window(
                np.arange(4.0).reshape(2, 2), np.array([1.0]), window=(-1.0, 0.0)
            )

    def test_long_history_window_does_not_warn(self, recwarn):
        """Reward history over minutes is a valid regressor, not milliseconds."""
        from neurospatial.events.regressors import event_count_in_window

        counts = event_count_in_window(
            np.array([0.0, 200.0]), np.array([50.0, 150.0]), window=(-120.0, 0.0)
        )
        np.testing.assert_array_equal(counts, [0, 1])
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_unsorted_events_handled(self):
        """Test unsorted event times are handled correctly."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([2.0])
        event_times = np.array([3.0, 1.0, 2.5])  # Unsorted
        window = (-1.5, 1.5)

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=2.0: window [0.5, 3.5], all events included
        assert result[0] == 3

    def test_nan_in_sample_times_raises(self):
        """Test NaN in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, np.nan, 2.0])
        event_times = np.array([1.0])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"^times.*NaN"):
            event_count_in_window(sample_times, event_times, window=window)

    def test_nan_in_event_times_raises(self):
        """Test NaN in event_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.nan])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"event_times.*NaN"):
            event_count_in_window(sample_times, event_times, window=window)

    def test_inf_in_sample_times_raises(self):
        """Test Inf in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, np.inf, 2.0])
        event_times = np.array([1.0])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"^times.*inf"):
            event_count_in_window(sample_times, event_times, window=window)

    def test_inf_in_event_times_raises(self):
        """Test Inf in event_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.inf])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"event_times.*inf"):
            event_count_in_window(sample_times, event_times, window=window)

    def test_inverted_window_raises(self):
        """Test window with start > end raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])
        window = (1.0, -1.0)  # Inverted

        with pytest.raises(ValueError, match=r"window.*start.*stop"):
            event_count_in_window(sample_times, event_times, window=window)

    def test_zero_width_window_is_refused(self):
        """A half-open window with stop == start holds no time."""
        from neurospatial.events.regressors import event_count_in_window

        with pytest.raises(ValueError, match="before its stop"):
            event_count_in_window(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.5]), window=(0.0, 0.0)
            )

    def test_dense_events(self):
        """Test with many closely spaced events."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([5.0])
        event_times = np.arange(101) / 10  # Events every 0.1 s, exact tenths
        window = (-1.0, 1.0)  # 2s total window

        result = event_count_in_window(sample_times, event_times, window=window)

        # t=5.0: window [4.0, 6.0): 4.0, 4.1, ..., 5.9 = 20 events
        assert result[0] == 20

    def test_typical_spike_counting_use_case(self):
        """Test typical use case: counting spikes in recent history."""
        from neurospatial.events.regressors import event_count_in_window

        # Continuous time at 100 Hz, on exact hundredths
        sample_times = np.arange(500) / 100
        # Reward events at 1s and 3s
        event_times = np.array([1.0, 3.0])
        # Count rewards in last 0.5s
        window = (-0.5, 0.0)

        result = event_count_in_window(sample_times, event_times, window=window)

        # Before any reward
        assert result[0] == 0  # t=0.0
        assert result[50] == 0  # t=0.5

        # The reward at 1.0 is in the past from the next sample on
        assert result[100] == 0  # t=1.0, window [0.5, 1.0) ends at the reward
        assert result[101] == 1  # t=1.01
        assert result[110] == 1  # t=1.1, reward within 0.5s window
        assert result[150] == 1  # t=1.5, window [1.0, 1.5) starts at the reward

        # After first reward fades from window
        assert result[151] == 0  # t=1.51, reward at 1.0 is outside [1.01, 1.51)
        assert result[160] == 0  # t=1.6

        # Second reward
        assert result[300] == 0  # t=3.0, the reward is now, not in the past
        assert result[301] == 1  # t=3.01

    def test_non_negative_counts(self):
        """Test that counts are always non-negative."""
        from neurospatial.events.regressors import event_count_in_window

        rng = np.random.default_rng(42)
        sample_times = np.linspace(0, 100, 1000)
        event_times = rng.random(50) * 100
        window = (-2.0, 2.0)

        result = event_count_in_window(sample_times, event_times, window=window)

        assert np.all(result >= 0)


# =============================================================================
# Test event_indicator
# =============================================================================


class TestEventIndicator:
    """Tests for event_indicator function."""

    def test_bin_centered_window_marks_each_event_once(self):
        """With window=(-dt/2, dt/2), each event marks the sample whose bin holds it."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([1.0, 3.0])

        result = event_indicator(sample_times, event_times, window=(-0.5, 0.5))

        assert result[0] is np.False_  # t=0.0, no event
        assert result[1] is np.True_  # t=1.0, event here
        assert result[2] is np.False_  # t=2.0, no event
        assert result[3] is np.True_  # t=3.0, event here
        assert result[4] is np.False_  # t=4.0, no event

    def test_window_around_event(self):
        """Test event_indicator with window parameter."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        event_times = np.array([1.5])
        window = (-0.5, 0.5)  # +/- 0.5s around event

        result = event_indicator(sample_times, event_times, window=window)

        # The event at 1.5 is in [t - 0.5, t + 0.5) for t in (1.0, 2.0].
        assert not result[0]  # t=0.0
        assert not result[1]  # t=0.5
        assert not result[2]  # t=1.0: window [0.5, 1.5) stops at the event
        assert result[3]  # t=1.5
        assert result[4]  # t=2.0: window [1.5, 2.5) starts at the event
        # t=2.5: |2.5 - 1.5| = 1.0 > 0.5, False
        assert not result[5]
        # t=3.0: |3.0 - 1.5| = 1.5 > 0.5, False
        assert not result[6]

    def test_multiple_events(self):
        """Test with multiple events."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        event_times = np.array([1.0, 4.0])
        window = (-0.5, 0.5)

        result = event_indicator(sample_times, event_times, window=window)

        # Near first event (1.0)
        assert not result[0]  # t=0.0, too far
        assert result[1]  # t=1.0, at event
        assert not result[2]  # t=2.0, too far (|2.0-1.0|=1.0 > 0.5)
        # Near second event (4.0)
        assert not result[3]  # t=3.0, too far (|3.0-4.0|=1.0 > 0.5)
        assert result[4]  # t=4.0, at event
        assert not result[5]  # t=5.0, too far

    def test_empty_events_returns_all_false(self):
        """Test empty events array returns all False."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([])

        result = event_indicator(sample_times, event_times, window=(-0.5, 0.5))

        assert np.all(~result)  # All False
        assert len(result) == 3

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        event_times = np.array([1.0])
        window = (-0.3, 0.3)

        result = event_indicator(sample_times, event_times, window=window)

        # t=0.0: |0.0-1.0|=1.0 > 0.3, False
        assert not result[0]
        # t=0.5: |0.5-1.0|=0.5 > 0.3, False
        assert not result[1]
        # t=1.0: |1.0-1.0|=0.0 <= 0.3, True
        assert result[2]
        # t=1.5: |1.5-1.0|=0.5 > 0.3, False
        assert not result[3]
        # t=2.0: |2.0-1.0|=1.0 > 0.3, False
        assert not result[4]

    def test_output_dtype_is_bool(self):
        """Test output dtype is bool."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        result = event_indicator(sample_times, event_times, window=(-0.5, 0.5))

        assert result.dtype == np.bool_

    def test_unsorted_events_handled(self):
        """Test unsorted event times are handled correctly."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([3.0, 1.0])  # Unsorted
        window = (-0.5, 0.5)

        result = event_indicator(sample_times, event_times, window=window)

        # Should find exact matches regardless of event order
        assert not result[0]  # t=0.0
        assert result[1]  # t=1.0
        assert not result[2]  # t=2.0
        assert result[3]  # t=3.0

    def test_multiple_events_same_time(self):
        """Test multiple events at same timestamp."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, 1.0, 1.0])  # Three events at same time

        result = event_indicator(sample_times, event_times, window=(-0.5, 0.5))

        # Should behave same as single event at 1.0
        assert not result[0]
        assert result[1]
        assert not result[2]

    def test_nan_in_sample_times_raises(self):
        """Test NaN in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, np.nan, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"^times.*NaN"):
            event_indicator(sample_times, event_times, window=(-0.5, 0.5))

    def test_nan_in_event_times_raises(self):
        """Test NaN in event_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.nan])

        with pytest.raises(ValueError, match=r"event_times.*NaN"):
            event_indicator(sample_times, event_times, window=(-0.5, 0.5))

    def test_inf_in_sample_times_raises(self):
        """Test Inf in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, np.inf, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"^times.*inf"):
            event_indicator(sample_times, event_times, window=(-0.5, 0.5))

    def test_inf_in_event_times_raises(self):
        """Test Inf in event_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.inf])

        with pytest.raises(ValueError, match=r"event_times.*inf"):
            event_indicator(sample_times, event_times, window=(-0.5, 0.5))

    def test_inverted_window_raises(self):
        """Test window with start > end raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"window.*start.*stop"):
            event_indicator(sample_times, event_times, window=(0.5, -0.5))

    def test_zero_width_window_is_refused(self):
        """A half-open window with stop == start holds no time."""
        from neurospatial.events.regressors import event_indicator

        with pytest.raises(ValueError, match="before its stop"):
            event_indicator(np.array([0.9, 1.0]), np.array([1.0]), window=(0.0, 0.0))

    def test_large_window_includes_all(self):
        """Test large window includes all samples near any event."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([2.0])
        window = (-10.0, 10.0)  # Very large window

        result = event_indicator(sample_times, event_times, window=window)

        # All samples within 10s of event at t=2.0
        assert np.all(result)

    def test_window_is_half_open(self):
        """An event at the window start counts; one at the stop does not."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.5, 1.0, 1.5])
        event_times = np.array([1.0])
        window = (-0.5, 0.5)  # Boundary at 0.5 and 1.5

        result = event_indicator(sample_times, event_times, window=window)

        assert not result[0]  # window [0.0, 1.0) stops at the event
        assert result[1]  # window [0.5, 1.5)
        assert result[2]  # window [1.0, 2.0) starts at the event

    def test_typical_time_bin_use_case(self):
        """Test typical use case: marking time bins with events."""
        from neurospatial.events.regressors import event_indicator

        # Time bins at 10ms resolution, on exact hundredths
        sample_times = np.arange(100) / 100
        # Events at 200ms, 500ms, 800ms
        event_times = np.array([0.2, 0.5, 0.8])
        # Window of +/- 50ms (one time bin)
        window = (-0.05, 0.05)

        result = event_indicator(sample_times, event_times, window=window)

        # Check around 200ms
        # The event at 200 ms is in [t - 50 ms, t + 50 ms) for t in (150, 250] ms.
        assert not result[15]  # t=150ms: window stops at the event
        assert result[16]  # t=160ms
        assert result[20]  # t=200ms, at event
        assert result[25]  # t=250ms: window starts at the event
        assert not result[26]  # t=260ms, too far

        # Total True values: ~6 per event * 3 events
        # (150-250ms, 450-550ms, 750-850ms)
        n_true = result.sum()
        assert n_true > 0  # Some True values
        assert n_true < len(sample_times)  # Not all True

    def test_impulse_regressor_marks_the_nearest_sample(self):
        """A bin-centred window marks each reward at the sample nearest it."""
        from neurospatial.events.regressors import event_indicator

        dt = 1 / 30
        sample_times = np.arange(3000) * dt
        event_times = np.array([10.01, 30.02, 50.0, 70.03, 90.0])

        result = event_indicator(sample_times, event_times, window=(-dt / 2, dt / 2))

        nearest = np.abs(sample_times[:, None] - event_times).argmin(axis=0)
        np.testing.assert_array_equal(np.flatnonzero(result), np.sort(nearest))


# =============================================================================
# Window-contract unification tests
# =============================================================================


class TestWindowContract:
    """window follows the canonical events order f(times, event_times, window)."""

    @pytest.mark.parametrize("name", ["event_count_in_window", "event_indicator"])
    def test_window_is_required_and_positional(self, name):
        import inspect

        from neurospatial.events import regressors

        function = getattr(regressors, name)
        parameters = inspect.signature(function).parameters
        assert list(parameters)[:3] == ["times", "event_times", "window"]
        assert parameters["window"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["window"].default is inspect.Parameter.empty
        assert parameters["epochs"].kind is inspect.Parameter.KEYWORD_ONLY

        times = np.array([0.0, 3.0, 6.0, 10.0])
        rewards = np.array([1.0, 2.0, 5.0])
        np.testing.assert_array_equal(
            function(times, rewards, (-5.0, 0.0)),
            function(times, rewards, window=(-5.0, 0.0)),
        )


class TestRecordedEpochs:
    """A sample whose value depends on unrecorded time is NaN, not 0."""

    def test_history_before_the_recording_is_missing(self):
        from neurospatial.events.regressors import event_count_in_window

        times = np.array([0.0, 3.0, 6.0, 10.0])
        rewards = np.array([1.0, 2.0, 5.0])

        counts = event_count_in_window(times, rewards, (-5.0, 0.0), epochs=(0.0, 20.0))

        np.testing.assert_array_equal(counts, [np.nan, np.nan, 3.0, 1.0])
        assert counts.dtype == np.float64

    def test_window_across_a_gap_is_missing_and_abutting_epochs_merge(self):
        from neurospatial.events.regressors import event_count_in_window

        times = np.array([1.0, 4.0, 9.0, 12.0])
        epochs = [(0.0, 5.0), (8.0, 10.0), (10.0, 20.0)]

        counts = event_count_in_window(
            times, np.array([9.5]), (-1.0, 1.0), epochs=epochs
        )

        # [3, 5) fits, [8, 10) fits, [11, 13) is inside the merged [8, 20).
        np.testing.assert_array_equal(counts, [0.0, 0.0, 1.0, 0.0])
        counts = event_count_in_window(
            np.array([4.5, 7.0]), np.array([9.5]), (-1.0, 1.0), epochs=epochs
        )
        np.testing.assert_array_equal(counts, [np.nan, np.nan])

    def test_indicator_is_nan_where_the_count_is(self):
        from neurospatial.events.regressors import event_indicator

        times = np.array([0.0, 1.0, 2.0, 3.0])
        result = event_indicator(
            times, np.array([1.0, 3.0]), (-0.5, 0.5), epochs=(0.25, 10.0)
        )
        np.testing.assert_array_equal(result, [np.nan, 1.0, 0.0, 1.0])

    def test_without_events_unrecorded_samples_are_still_missing(self):
        from neurospatial.events.regressors import event_count_in_window

        counts = event_count_in_window(
            np.array([0.5, 5.0]), np.array([]), (-1.0, 0.0), epochs=(0.0, 10.0)
        )
        np.testing.assert_array_equal(counts, [np.nan, 0.0])

    def test_epochs_none_keeps_integer_and_bool_dtypes(self):
        from neurospatial.events.regressors import (
            event_count_in_window,
            event_indicator,
        )

        times, events = np.arange(4.0), np.array([1.0])
        assert event_count_in_window(times, events, (-0.5, 0.5)).dtype == np.int64
        assert event_indicator(times, events, (-0.5, 0.5)).dtype == np.bool_

    def test_time_to_event_is_missing_when_an_unseen_event_could_be_nearer(self):
        from neurospatial.events.regressors import time_to_nearest_event

        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 9.5])
        result = time_to_nearest_event(times, np.array([2.0]), epochs=(1.0, 10.0))

        # (t - d, t + d) must be recorded: fails for t = 0, 1 and 9.5.
        np.testing.assert_array_equal(result, [np.nan, np.nan, 0.0, 1.0, 2.0, np.nan])

    def test_clipped_time_to_event_needs_only_max_time_of_recording(self):
        from neurospatial.events.regressors import time_to_nearest_event

        result = time_to_nearest_event(
            np.array([1.5, 9.5]), np.array([20.0]), max_time=1.0, epochs=(0.0, 10.0)
        )
        # "at least 1 s from any event" is known where (t - 1, t + 1) is recorded.
        np.testing.assert_array_equal(result, [-1.0, np.nan])

    def test_invalid_epochs_are_refused(self):
        from neurospatial.events.regressors import event_count_in_window

        with pytest.raises(ValueError, match="epochs"):
            event_count_in_window(
                np.arange(3.0), np.array([1.0]), (-1.0, 0.0), epochs=(5.0, 1.0)
            )
