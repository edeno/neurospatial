"""Tests for events module GLM regressors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# Test time_since_event
# =============================================================================


class TestTimeSinceEvent:
    """Tests for time_since_event function."""

    def test_basic_functionality(self):
        """Test basic time since event calculation."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        event_times = np.array([1.5, 3.5])

        result = time_since_event(sample_times, event_times)

        # Before first event: NaN
        assert np.isnan(result[0])  # t=0.0
        assert np.isnan(result[1])  # t=1.0
        # After first event (1.5): time since = sample_time - 1.5
        assert_allclose(result[2], 0.5)  # t=2.0, since 1.5 = 0.5
        assert_allclose(result[3], 1.5)  # t=3.0, since 1.5 = 1.5
        # After second event (3.5): time since = sample_time - 3.5
        assert_allclose(result[4], 0.5)  # t=4.0, since 3.5 = 0.5
        assert_allclose(result[5], 1.5)  # t=5.0, since 3.5 = 1.5

    def test_sample_at_event_time(self):
        """Test sample exactly at event time."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.0, 3.0])

        result = time_since_event(sample_times, event_times)

        assert np.isnan(result[0])  # t=0.0, before first event
        assert_allclose(result[1], 0.0)  # t=1.0, at first event
        assert_allclose(result[2], 1.0)  # t=2.0, 1s since event at 1.0
        assert_allclose(result[3], 0.0)  # t=3.0, at second event

    def test_max_time_clips_values(self):
        """Test max_time parameter clips large values."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 5.0, 10.0])
        event_times = np.array([1.0])

        result = time_since_event(sample_times, event_times, max_time=3.0)

        assert np.isnan(result[0])  # Before event
        assert_allclose(result[1], 0.0)  # At event
        assert_allclose(result[2], 3.0)  # 4s since, clipped to 3.0
        assert_allclose(result[3], 3.0)  # 9s since, clipped to 3.0

    def test_fill_before_first(self):
        """Test fill_before_first parameter."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([2.0])

        result = time_since_event(sample_times, event_times, fill_before_first=999.0)

        assert_allclose(result[0], 999.0)  # Filled
        assert_allclose(result[1], 999.0)  # Filled
        assert_allclose(result[2], 0.0)  # At event
        assert_allclose(result[3], 1.0)  # After event

    def test_fill_before_first_with_max_time(self):
        """Test fill_before_first combined with max_time."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 10.0])
        event_times = np.array([2.0])

        result = time_since_event(
            sample_times, event_times, max_time=5.0, fill_before_first=5.0
        )

        assert_allclose(result[0], 5.0)  # fill_before_first
        assert_allclose(result[1], 5.0)  # fill_before_first
        assert_allclose(result[2], 0.0)  # At event
        assert_allclose(result[3], 5.0)  # 8s since, clipped to max_time

    def test_empty_events_returns_nan(self):
        """Test empty events array returns all NaN."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([])

        result = time_since_event(sample_times, event_times)

        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_empty_events_with_fill(self):
        """Test empty events with fill_before_first."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([])

        result = time_since_event(sample_times, event_times, fill_before_first=100.0)

        assert_allclose(result, np.array([100.0, 100.0, 100.0]))

    def test_single_event(self):
        """Test with single event."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5])

        result = time_since_event(sample_times, event_times)

        assert np.isnan(result[0])  # Before
        assert np.isnan(result[1])  # Before
        assert_allclose(result[2], 0.5)  # After: 2.0 - 1.5
        assert_allclose(result[3], 1.5)  # After: 3.0 - 1.5

    def test_nan_policy_raise(self):
        """Test nan_policy='raise' raises when output has NaN."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.5])

        with pytest.raises(ValueError, match="NaN"):
            time_since_event(sample_times, event_times, nan_policy="raise")

    def test_nan_policy_fill_requires_fill_value(self):
        """Test nan_policy='fill' requires fill_before_first."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.5])

        with pytest.raises(ValueError, match="fill_before_first"):
            time_since_event(sample_times, event_times, nan_policy="fill")

    def test_nan_policy_fill_uses_fill_value(self):
        """Test nan_policy='fill' uses fill_before_first value."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.5])

        result = time_since_event(
            sample_times, event_times, nan_policy="fill", fill_before_first=999.0
        )

        assert_allclose(result[0], 999.0)
        assert_allclose(result[1], 999.0)
        assert_allclose(result[2], 0.5)

    def test_nan_policy_propagate_default(self):
        """Test nan_policy='propagate' is default and keeps NaN."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.5])

        result = time_since_event(sample_times, event_times)  # Default

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert_allclose(result[2], 0.5)

    def test_unsorted_events_handled(self):
        """Test unsorted event times are handled correctly."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        event_times = np.array([3.0, 1.0, 2.0])  # Unsorted

        result = time_since_event(sample_times, event_times)

        # Events are [1.0, 2.0, 3.0] when sorted
        assert np.isnan(result[0])  # t=0.0, before first
        assert_allclose(result[1], 0.0)  # t=1.0, at event
        assert_allclose(result[2], 0.0)  # t=2.0, at event
        assert_allclose(result[3], 0.0)  # t=3.0, at event
        assert_allclose(result[4], 1.0)  # t=4.0, 1s since 3.0

    def test_output_shape_matches_sample_times(self):
        """Test output shape matches sample_times."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.linspace(0, 10, 100)
        event_times = np.array([2.0, 5.0, 8.0])

        result = time_since_event(sample_times, event_times)

        assert result.shape == sample_times.shape

    def test_output_dtype_is_float64(self):
        """Test output dtype is float64."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        result = time_since_event(sample_times, event_times)

        assert result.dtype == np.float64

    def test_all_samples_after_all_events(self):
        """Test when all samples are after all events."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([5.0, 6.0, 7.0])
        event_times = np.array([1.0, 2.0])

        result = time_since_event(sample_times, event_times)

        # Last event is at 2.0
        assert_allclose(result[0], 3.0)  # 5.0 - 2.0
        assert_allclose(result[1], 4.0)  # 6.0 - 2.0
        assert_allclose(result[2], 5.0)  # 7.0 - 2.0

    def test_all_samples_before_all_events(self):
        """Test when all samples are before all events."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([5.0, 6.0])

        result = time_since_event(sample_times, event_times)

        assert np.all(np.isnan(result))

    def test_multiple_events_same_time(self):
        """Test multiple events at same time."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.5, 1.5, 1.5])  # Three events at same time

        result = time_since_event(sample_times, event_times)

        # Should behave as if single event at 1.5
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert_allclose(result[2], 0.5)
        assert_allclose(result[3], 1.5)

    def test_nan_in_sample_times_raises(self):
        """Test NaN in sample_times raises ValueError."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, np.nan, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"sample_times.*NaN"):
            time_since_event(sample_times, event_times)

    def test_nan_in_event_times_raises(self):
        """Test NaN in event_times raises ValueError."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.nan])

        with pytest.raises(ValueError, match=r"event_times.*NaN"):
            time_since_event(sample_times, event_times)

    def test_inf_in_sample_times_raises(self):
        """Test Inf in sample_times raises ValueError."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, np.inf, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"sample_times.*inf"):
            time_since_event(sample_times, event_times)

    def test_inf_in_event_times_raises(self):
        """Test Inf in event_times raises ValueError."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, np.inf])

        with pytest.raises(ValueError, match=r"event_times.*inf"):
            time_since_event(sample_times, event_times)

    def test_negative_max_time_raises(self):
        """Test negative max_time raises ValueError."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0])

        with pytest.raises(ValueError, match=r"max_time.*positive"):
            time_since_event(sample_times, event_times, max_time=-1.0)

    def test_zero_max_time_clips_to_zero(self):
        """Test max_time=0.0 clips all values to zero."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([0.0, 1.0, 2.0, 3.0])
        event_times = np.array([1.0])

        result = time_since_event(sample_times, event_times, max_time=0.0)

        assert np.isnan(result[0])  # Before event
        assert_allclose(result[1], 0.0)  # At event
        assert_allclose(result[2], 0.0)  # Clipped
        assert_allclose(result[3], 0.0)  # Clipped

    def test_empty_sample_times(self):
        """Test empty sample_times returns empty array."""
        from neurospatial.events.regressors import time_since_event

        sample_times = np.array([])
        event_times = np.array([1.0, 2.0])

        result = time_since_event(sample_times, event_times)

        assert len(result) == 0
        assert result.dtype == np.float64
