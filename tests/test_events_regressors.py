"""Tests for events module GLM regressors."""

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

    def test_output_shape_matches_sample_times(self):
        """Test output shape matches sample_times."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.linspace(0, 10, 100)
        event_times = np.array([2.0, 5.0, 8.0])

        result = time_to_nearest_event(sample_times, event_times)

        assert result.shape == sample_times.shape

    def test_output_dtype_is_float64(self):
        """Test output dtype is float64."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        result = time_to_nearest_event(sample_times, event_times)

        assert result.dtype == np.float64

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

        with pytest.raises(ValueError, match=r"sample_times.*NaN"):
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

        with pytest.raises(ValueError, match=r"sample_times.*inf"):
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

    def test_zero_max_time_clips_to_zero(self):
        """Test max_time=0.0 clips all values to zero."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5])

        result = time_to_nearest_event(sample_times, event_times, max_time=0.0)

        assert_allclose(result, np.array([0.0, 0.0, 0.0, 0.0]))

    def test_empty_sample_times(self):
        """Test empty sample_times returns empty array."""
        from neurospatial.events.regressors import time_to_nearest_event

        sample_times = np.array([])
        event_times = np.array([1.0, 2.0])

        result = time_to_nearest_event(sample_times, event_times)

        assert len(result) == 0
        assert result.dtype == np.float64

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

        result = event_count_in_window(sample_times, event_times, window)

        # t=0.0: window [-2, 0], no events
        assert result[0] == 0
        # t=1.0: window [-1, 1], event at 1.0 is at boundary (included)
        assert result[1] == 1
        # t=2.0: window [0, 2], event at 1.0 is at start (included)
        assert result[2] == 1
        # t=3.0: window [1, 3], events at 1.0, 2.5 both included
        assert result[3] == 2
        # t=4.0: window [2, 4], event at 2.5 included
        assert result[4] == 1
        # t=5.0: window [3, 5], no events in [3, 5]
        assert result[5] == 0

    def test_basic_forward_window(self):
        """Test counting events in forward window (future events)."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([2.0, 3.5])
        window = (0.0, 2.0)  # Look 2s into the future

        result = event_count_in_window(sample_times, event_times, window)

        # t=0.0: window [0, 2], event at 2.0 at boundary (included)
        assert result[0] == 1
        # t=1.0: window [1, 3], event at 2.0 included
        assert result[1] == 1
        # t=2.0: window [2, 4], events at 2.0, 3.5 both included
        assert result[2] == 2
        # t=3.0: window [3, 5], event at 3.5 included
        assert result[3] == 1
        # t=4.0: window [4, 6], no events
        assert result[4] == 0

    def test_symmetric_window(self):
        """Test counting events in symmetric window around sample."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 2.0, 5.0])
        event_times = np.array([1.0, 3.0, 4.0])
        window = (-1.5, 1.5)  # +/- 1.5s around each sample

        result = event_count_in_window(sample_times, event_times, window)

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

        result = event_count_in_window(sample_times, event_times, window)

        assert np.all(result == 0)
        assert len(result) == 3

    def test_empty_sample_times(self):
        """Test empty sample_times returns empty array."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([])
        event_times = np.array([1.0, 2.0])
        window = (-1.0, 1.0)

        result = event_count_in_window(sample_times, event_times, window)

        assert len(result) == 0
        assert result.dtype == np.int64

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5])
        window = (-1.0, 1.0)  # +/- 1s

        result = event_count_in_window(sample_times, event_times, window)

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

        result = event_count_in_window(sample_times, event_times, window)

        # t=0.0: window [-0.5, 0.5], no events
        assert result[0] == 0
        # t=1.0: window [0.5, 1.5], 3 events at 1.0 included
        assert result[1] == 3
        # t=2.0: window [1.5, 2.5], no events
        assert result[2] == 0

    def test_event_at_window_boundary_included(self):
        """Test events exactly at window boundaries are included."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([2.0])
        event_times = np.array([1.0, 3.0])  # Events at exact boundaries
        window = (-1.0, 1.0)

        result = event_count_in_window(sample_times, event_times, window)

        # window [1.0, 3.0], both boundary events included
        assert result[0] == 2

    def test_output_dtype_is_int64(self):
        """Test output dtype is int64."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])
        window = (-1.0, 1.0)

        result = event_count_in_window(sample_times, event_times, window)

        assert result.dtype == np.int64

    def test_output_shape_matches_sample_times(self):
        """Test output shape matches sample_times."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.linspace(0, 10, 100)
        event_times = np.array([2.0, 5.0, 8.0])
        window = (-1.0, 1.0)

        result = event_count_in_window(sample_times, event_times, window)

        assert result.shape == sample_times.shape

    def test_unsorted_events_handled(self):
        """Test unsorted event times are handled correctly."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([2.0])
        event_times = np.array([3.0, 1.0, 2.5])  # Unsorted
        window = (-1.5, 1.5)

        result = event_count_in_window(sample_times, event_times, window)

        # t=2.0: window [0.5, 3.5], all events included
        assert result[0] == 3

    def test_nan_in_sample_times_raises(self):
        """Test NaN in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, np.nan, 2.0])
        event_times = np.array([1.0])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"sample_times.*NaN"):
            event_count_in_window(sample_times, event_times, window)

    def test_nan_in_event_times_raises(self):
        """Test NaN in event_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.nan])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"event_times.*NaN"):
            event_count_in_window(sample_times, event_times, window)

    def test_inf_in_sample_times_raises(self):
        """Test Inf in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, np.inf, 2.0])
        event_times = np.array([1.0])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"sample_times.*inf"):
            event_count_in_window(sample_times, event_times, window)

    def test_inf_in_event_times_raises(self):
        """Test Inf in event_times raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.inf])
        window = (-1.0, 1.0)

        with pytest.raises(ValueError, match=r"event_times.*inf"):
            event_count_in_window(sample_times, event_times, window)

    def test_inverted_window_raises(self):
        """Test window with start > end raises ValueError."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])
        window = (1.0, -1.0)  # Inverted

        with pytest.raises(ValueError, match=r"window.*start.*end"):
            event_count_in_window(sample_times, event_times, window)

    def test_zero_width_window(self):
        """Test zero-width window only counts exact matches."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, 1.5])
        window = (0.0, 0.0)  # Zero width

        result = event_count_in_window(sample_times, event_times, window)

        # Only exact matches counted
        assert result[0] == 0  # t=0, no event at exactly 0
        assert result[1] == 1  # t=1, event at exactly 1
        assert result[2] == 0  # t=2, no event at exactly 2

    def test_dense_events(self):
        """Test with many closely spaced events."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.array([5.0])
        event_times = np.linspace(0, 10, 101)  # Events every 0.1s
        window = (-1.0, 1.0)  # 2s total window

        result = event_count_in_window(sample_times, event_times, window)

        # t=5.0: window [4.0, 6.0], should have ~21 events (4.0 to 6.0 inclusive)
        # 4.0, 4.1, ..., 5.9, 6.0 = 21 events
        assert result[0] == 21

    def test_typical_spike_counting_use_case(self):
        """Test typical use case: counting spikes in recent history."""
        from neurospatial.events.regressors import event_count_in_window

        # Continuous time at 100 Hz
        sample_times = np.arange(0, 5, 0.01)
        # Reward events at 1s and 3s
        event_times = np.array([1.0, 3.0])
        # Count rewards in last 0.5s
        window = (-0.5, 0.0)

        result = event_count_in_window(sample_times, event_times, window)

        # Before any reward
        assert result[0] == 0  # t=0.0
        assert result[50] == 0  # t=0.5

        # Just after first reward
        assert result[100] == 1  # t=1.0, reward at boundary
        assert result[110] == 1  # t=1.1, reward within 0.5s window
        assert result[150] == 1  # t=1.5, reward at edge of window

        # After first reward fades from window
        assert result[160] == 0  # t=1.6, reward at 1.0 is 0.6s ago (outside window)

        # At second reward
        assert result[300] == 1  # t=3.0

    def test_non_negative_counts(self):
        """Test that counts are always non-negative."""
        from neurospatial.events.regressors import event_count_in_window

        sample_times = np.linspace(0, 100, 1000)
        event_times = np.random.rand(50) * 100
        window = (-2.0, 2.0)

        result = event_count_in_window(sample_times, event_times, window)

        assert np.all(result >= 0)


# =============================================================================
# Test event_indicator
# =============================================================================


class TestEventIndicator:
    """Tests for event_indicator function."""

    def test_basic_exact_match(self):
        """Test event_indicator returns True only at exact event times (window=0)."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([1.0, 3.0])

        result = event_indicator(sample_times, event_times)

        # Only exact matches when window=0 (default)
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
        window = 0.5  # +/- 0.5s around event

        result = event_indicator(sample_times, event_times, window=window)

        # t=0.0: |0.0 - 1.5| = 1.5 > 0.5, False
        assert not result[0]
        # t=0.5: |0.5 - 1.5| = 1.0 > 0.5, False
        assert not result[1]
        # t=1.0: |1.0 - 1.5| = 0.5 <= 0.5, True
        assert result[2]
        # t=1.5: |1.5 - 1.5| = 0.0 <= 0.5, True
        assert result[3]
        # t=2.0: |2.0 - 1.5| = 0.5 <= 0.5, True
        assert result[4]
        # t=2.5: |2.5 - 1.5| = 1.0 > 0.5, False
        assert not result[5]
        # t=3.0: |3.0 - 1.5| = 1.5 > 0.5, False
        assert not result[6]

    def test_multiple_events(self):
        """Test with multiple events."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        event_times = np.array([1.0, 4.0])
        window = 0.5

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

        result = event_indicator(sample_times, event_times)

        assert np.all(~result)  # All False
        assert len(result) == 3

    def test_empty_sample_times(self):
        """Test empty sample_times returns empty array."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([])
        event_times = np.array([1.0, 2.0])

        result = event_indicator(sample_times, event_times)

        assert len(result) == 0
        assert result.dtype == np.bool_

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        event_times = np.array([1.0])
        window = 0.3

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

        result = event_indicator(sample_times, event_times)

        assert result.dtype == np.bool_

    def test_output_shape_matches_sample_times(self):
        """Test output shape matches sample_times."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.linspace(0, 10, 100)
        event_times = np.array([2.0, 5.0, 8.0])

        result = event_indicator(sample_times, event_times)

        assert result.shape == sample_times.shape

    def test_unsorted_events_handled(self):
        """Test unsorted event times are handled correctly."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([3.0, 1.0])  # Unsorted
        window = 0.0

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

        result = event_indicator(sample_times, event_times)

        # Should behave same as single event at 1.0
        assert not result[0]
        assert result[1]
        assert not result[2]

    def test_nan_in_sample_times_raises(self):
        """Test NaN in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, np.nan, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"sample_times.*NaN"):
            event_indicator(sample_times, event_times)

    def test_nan_in_event_times_raises(self):
        """Test NaN in event_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.nan])

        with pytest.raises(ValueError, match=r"event_times.*NaN"):
            event_indicator(sample_times, event_times)

    def test_inf_in_sample_times_raises(self):
        """Test Inf in sample_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, np.inf, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"sample_times.*inf"):
            event_indicator(sample_times, event_times)

    def test_inf_in_event_times_raises(self):
        """Test Inf in event_times raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.inf])

        with pytest.raises(ValueError, match=r"event_times.*inf"):
            event_indicator(sample_times, event_times)

    def test_negative_window_raises(self):
        """Test negative window raises ValueError."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"window.*non-negative"):
            event_indicator(sample_times, event_times, window=-0.5)

    def test_zero_window_exact_match_only(self):
        """Test window=0.0 requires exact match."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.9, 1.0, 1.1])
        event_times = np.array([1.0])
        window = 0.0

        result = event_indicator(sample_times, event_times, window=window)

        assert not result[0]  # 0.9 != 1.0
        assert result[1]  # 1.0 == 1.0
        assert not result[2]  # 1.1 != 1.0

    def test_large_window_includes_all(self):
        """Test large window includes all samples near any event."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([2.0])
        window = 10.0  # Very large window

        result = event_indicator(sample_times, event_times, window=window)

        # All samples within 10s of event at t=2.0
        assert np.all(result)

    def test_boundary_inclusion(self):
        """Test samples exactly at window boundary are included."""
        from neurospatial.events.regressors import event_indicator

        sample_times = np.array([0.5, 1.0, 1.5])
        event_times = np.array([1.0])
        window = 0.5  # Boundary at 0.5 and 1.5

        result = event_indicator(sample_times, event_times, window=window)

        # All samples are exactly within or at boundary
        assert result[0]  # |0.5 - 1.0| = 0.5 <= 0.5, included
        assert result[1]  # |1.0 - 1.0| = 0.0 <= 0.5, included
        assert result[2]  # |1.5 - 1.0| = 0.5 <= 0.5, included

    def test_typical_time_bin_use_case(self):
        """Test typical use case: marking time bins with events."""
        from neurospatial.events.regressors import event_indicator

        # Time bins at 10ms resolution
        sample_times = np.arange(0, 1.0, 0.01)
        # Events at 200ms, 500ms, 800ms
        event_times = np.array([0.2, 0.5, 0.8])
        # Window of +/- 50ms (one time bin)
        window = 0.05

        result = event_indicator(sample_times, event_times, window=window)

        # Check around 200ms
        assert not result[14]  # t=140ms, too far
        assert result[15]  # t=150ms, within window of 200ms
        assert result[20]  # t=200ms, at event
        assert result[25]  # t=250ms, within window
        assert not result[26]  # t=260ms, too far

        # Total True values: ~6 per event * 3 events
        # (150-250ms, 450-550ms, 750-850ms)
        n_true = result.sum()
        assert n_true > 0  # Some True values
        assert n_true < len(sample_times)  # Not all True

    def test_glm_design_matrix_use_case(self):
        """Test use in GLM design matrix context."""
        from neurospatial.events.regressors import event_indicator

        # Typical neural data: 30Hz sampling over 100s
        sample_times = np.linspace(0, 100, 3000)
        # Reward events
        event_times = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        # Narrow window for impulse-like regressor
        window = 0.1

        result = event_indicator(sample_times, event_times, window=window)

        # Result should be sparse (mostly False)
        sparsity = result.sum() / len(result)
        assert sparsity < 0.1  # Less than 10% True

        # Can be used directly in design matrix
        X = result.astype(float)
        assert X.dtype == np.float64


# =============================================================================
# Test distance_to_reward
# =============================================================================


@pytest.fixture
def simple_grid_env():
    """Create a simple 10x10 grid environment for testing."""
    from neurospatial import Environment

    # Create positions covering 10x10 grid
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(positions, bin_size=1.0)
    return env


class TestDistanceToReward:
    """Tests for distance_to_reward function."""

    def test_basic_functionality(self, simple_grid_env):
        """Test basic distance to reward calculation."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        # Trajectory moving from (0,0) to (9,9)
        n_samples = 50
        times = np.linspace(0, 10, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=5 (midpoint)
        reward_times = np.array([5.0])

        result = distance_to_reward(
            env, positions, times, reward_times, metric="euclidean"
        )

        assert result.shape == (n_samples,)
        assert not np.all(np.isnan(result))
        # At t=5, animal is at midpoint, reward is there, distance ~0
        mid_idx = n_samples // 2
        assert result[mid_idx] < result[0]  # Closer at reward time

    def test_mode_nearest(self, simple_grid_env):
        """Test mode='nearest' selects closest reward in time."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.column_stack(
            [
                np.full(n_samples, 5.0),  # Stay at center
                np.full(n_samples, 5.0),
            ]
        )

        # Two rewards at different locations and times
        reward_times = np.array([2.0, 8.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            positions,
            times,
            reward_times,
            reward_positions=reward_positions,
            mode="nearest",
            metric="euclidean",
        )

        # At t=2, should be distance to (0,5) which is 5 units
        idx_at_2 = 20  # t=2.0
        # At t=8, should be distance to (9,5) which is 4 units
        idx_at_8 = 80  # t=8.0

        assert result[idx_at_2] == pytest.approx(5.0, abs=0.5)
        assert result[idx_at_8] == pytest.approx(4.0, abs=0.5)

    def test_mode_last(self, simple_grid_env):
        """Test mode='last' uses most recent reward."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.full((n_samples, 2), 5.0)

        reward_times = np.array([3.0, 7.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            positions,
            times,
            reward_times,
            reward_positions=reward_positions,
            mode="last",
            metric="euclidean",
        )

        # Before first reward (t<3): NaN
        assert np.isnan(result[0])
        assert np.isnan(result[10])

        # After first reward (3<=t<7): distance to (0,5)
        assert not np.isnan(result[30])  # t=3.0
        assert not np.isnan(result[50])  # t=5.0

        # After second reward (t>=7): distance to (9,5)
        assert not np.isnan(result[70])  # t=7.0
        assert not np.isnan(result[90])  # t=9.0

    def test_mode_next(self, simple_grid_env):
        """Test mode='next' uses upcoming reward."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.full((n_samples, 2), 5.0)

        reward_times = np.array([3.0, 7.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            positions,
            times,
            reward_times,
            reward_positions=reward_positions,
            mode="next",
            metric="euclidean",
        )

        # Before first reward: distance to (0,5)
        assert not np.isnan(result[0])
        assert not np.isnan(result[20])

        # Between rewards (3<=t<7): distance to (9,5)
        assert not np.isnan(result[50])

        # After last reward: NaN
        assert np.isnan(result[80])
        assert np.isnan(result[99])

    def test_empty_rewards_returns_nan(self, simple_grid_env):
        """Test empty reward_times returns all NaN."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        reward_times = np.array([])

        result = distance_to_reward(env, positions, times, reward_times)

        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_empty_positions(self, simple_grid_env):
        """Test empty positions returns empty array."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        times = np.array([])
        positions = np.empty((0, 2))
        reward_times = np.array([1.0, 2.0])

        result = distance_to_reward(env, positions, times, reward_times)

        assert len(result) == 0

    def test_reward_positions_interpolated(self, simple_grid_env):
        """Test reward positions are interpolated when not provided."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0, 10, n_samples)
        # Trajectory from (0,0) to (9,9)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=5 (position should be ~(4.5, 4.5))
        reward_times = np.array([5.0])

        # Without explicit reward_positions
        result = distance_to_reward(
            env, positions, times, reward_times, metric="euclidean"
        )

        # At t=5, animal is at ~(4.5, 4.5), reward interpolated to same
        mid_idx = n_samples // 2
        assert result[mid_idx] < 1.0  # Close to zero at reward time

    def test_position_times_mismatch_raises(self, simple_grid_env):
        """Test mismatched positions and times raises ValueError."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        times = np.array([0.0, 1.0])  # Wrong length
        reward_times = np.array([0.5])

        with pytest.raises(ValueError, match=r"positions and times.*same length"):
            distance_to_reward(env, positions, times, reward_times)

    def test_reward_positions_times_mismatch_raises(self, simple_grid_env):
        """Test mismatched reward_positions and reward_times raises ValueError."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        times = np.array([0.0, 1.0])
        reward_times = np.array([0.5, 0.8])
        reward_positions = np.array([[1.0, 1.0]])  # Wrong length

        with pytest.raises(ValueError, match=r"reward_positions and reward_times"):
            distance_to_reward(
                env, positions, times, reward_times, reward_positions=reward_positions
            )

    def test_output_dtype_is_float64(self, simple_grid_env):
        """Test output dtype is float64."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        times = np.array([0.0, 1.0])
        reward_times = np.array([0.5])

        result = distance_to_reward(env, positions, times, reward_times)

        assert result.dtype == np.float64

    def test_geodesic_respects_graph(self, simple_grid_env):
        """Test geodesic metric uses graph distances."""
        from neurospatial.events.regressors import distance_to_reward

        env = simple_grid_env

        # Position at (0, 0), reward at (1, 1)
        positions = np.array([[0.0, 0.0]])
        times = np.array([0.0])
        reward_times = np.array([0.0])
        reward_positions = np.array([[1.0, 1.0]])

        # Geodesic distance (Manhattan-like on grid)
        result_geodesic = distance_to_reward(
            env,
            positions,
            times,
            reward_times,
            reward_positions=reward_positions,
            metric="geodesic",
        )

        # Euclidean distance (sqrt(2) ~= 1.41)
        result_euclidean = distance_to_reward(
            env,
            positions,
            times,
            reward_times,
            reward_positions=reward_positions,
            metric="euclidean",
        )

        # Geodesic should be >= Euclidean (diagonal not directly connected)
        assert result_geodesic[0] >= result_euclidean[0]


# =============================================================================
# Test distance_to_boundary
# =============================================================================


class TestDistanceToBoundary:
    """Tests for distance_to_boundary function."""

    def test_basic_edge_distance(self, simple_grid_env):
        """Test basic distance to edge calculation."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        # Positions: center and corner
        positions = np.array([[5.0, 5.0], [0.0, 0.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        assert result.shape == (2,)
        # Center is farther from edge than corner
        assert result[0] > result[1]
        # Corner is at edge, distance should be 0 or very small
        assert result[1] < 1.0

    def test_region_boundary_distance(self, simple_grid_env):
        """Test distance to region boundary."""
        from shapely.geometry import box as shapely_box

        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        # Add a region (simple polygon using shapely)
        goal_polygon = shapely_box(7.0, 7.0, 9.0, 9.0)  # minx, miny, maxx, maxy
        env.regions.add("goal", polygon=goal_polygon)

        # Positions: one far from region, one near region
        positions = np.array([[1.0, 1.0], [6.0, 6.0]])

        result = distance_to_boundary(
            env, positions, boundary_type="region", region_name="goal"
        )

        assert result.shape == (2,)
        # Position at (1,1) is farther from goal region than (6,6)
        assert result[0] > result[1]

    def test_missing_region_raises(self, simple_grid_env):
        """Test missing region name raises ValueError."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"Region.*not found"):
            distance_to_boundary(
                env, positions, boundary_type="region", region_name="nonexistent"
            )

    def test_region_name_required_for_region_type(self, simple_grid_env):
        """Test region_name is required when boundary_type='region'."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"region_name.*required"):
            distance_to_boundary(env, positions, boundary_type="region")

    def test_empty_positions(self, simple_grid_env):
        """Test empty positions returns empty array."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        positions = np.empty((0, 2))

        result = distance_to_boundary(env, positions, boundary_type="edge")

        assert len(result) == 0

    def test_output_dtype_is_float64(self, simple_grid_env):
        """Test output dtype is float64."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        assert result.dtype == np.float64

    def test_geodesic_vs_euclidean(self, simple_grid_env):
        """Test geodesic and euclidean metrics give different results."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        result_geodesic = distance_to_boundary(
            env, positions, boundary_type="edge", metric="geodesic"
        )

        result_euclidean = distance_to_boundary(
            env, positions, boundary_type="edge", metric="euclidean"
        )

        # Both should return valid distances
        assert not np.isnan(result_geodesic[0])
        assert not np.isnan(result_euclidean[0])

    def test_invalid_boundary_type_raises(self, simple_grid_env):
        """Test invalid boundary_type raises ValueError."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"boundary_type"):
            distance_to_boundary(env, positions, boundary_type="invalid")

    def test_position_outside_env_returns_nan(self, simple_grid_env):
        """Test positions outside environment return NaN."""
        from neurospatial.events.regressors import distance_to_boundary

        env = simple_grid_env

        # Position far outside the 10x10 grid
        positions = np.array([[100.0, 100.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        # Position outside env → invalid bin → NaN
        assert np.isnan(result[0])
