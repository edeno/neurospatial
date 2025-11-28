"""Tests for directional place field computation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neurospatial.spike_field import (
    DirectionalPlaceFields,
    _subset_spikes_by_time_mask,
)


class TestDirectionalPlaceFieldsDataclass:
    """Tests for the DirectionalPlaceFields dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test basic creation of DirectionalPlaceFields."""
        fields = {
            "A→B": np.array([1.0, 2.0, 3.0]),
            "B→A": np.array([3.0, 2.0, 1.0]),
        }
        labels = ("A→B", "B→A")

        result = DirectionalPlaceFields(fields=fields, labels=labels)

        assert result.fields == fields
        assert result.labels == labels

    def test_dataclass_is_frozen(self) -> None:
        """Test that dataclass is immutable (frozen)."""
        fields = {"A→B": np.array([1.0, 2.0, 3.0])}
        labels = ("A→B",)

        result = DirectionalPlaceFields(fields=fields, labels=labels)

        # Should raise FrozenInstanceError when trying to modify
        with pytest.raises(AttributeError):
            result.labels = ("B→A",)  # type: ignore[misc]

        with pytest.raises(AttributeError):
            result.fields = {}  # type: ignore[misc]

    def test_labels_is_tuple(self) -> None:
        """Test that labels preserves iteration order as tuple."""
        fields = {
            "first": np.array([1.0]),
            "second": np.array([2.0]),
            "third": np.array([3.0]),
        }
        labels = ("first", "second", "third")

        result = DirectionalPlaceFields(fields=fields, labels=labels)

        assert isinstance(result.labels, tuple)
        assert result.labels == ("first", "second", "third")

    def test_fields_is_mapping(self) -> None:
        """Test that fields is a mapping from string labels to arrays."""
        fields = {"A→B": np.array([1.0, 2.0])}
        labels = ("A→B",)

        result = DirectionalPlaceFields(fields=fields, labels=labels)

        # Should support dict-like access
        assert "A→B" in result.fields
        assert_array_equal(result.fields["A→B"], np.array([1.0, 2.0]))

    def test_empty_fields(self) -> None:
        """Test creation with empty fields."""
        result = DirectionalPlaceFields(fields={}, labels=())

        assert len(result.fields) == 0
        assert len(result.labels) == 0

    def test_single_direction(self) -> None:
        """Test with a single direction label."""
        fields = {"forward": np.array([1.0, 2.0, 3.0, 4.0])}
        labels = ("forward",)

        result = DirectionalPlaceFields(fields=fields, labels=labels)

        assert len(result.fields) == 1
        assert len(result.labels) == 1
        assert result.labels[0] == "forward"


class TestSubsetSpikesByTimeMask:
    """Tests for the _subset_spikes_by_time_mask helper function."""

    def test_single_contiguous_segment(self) -> None:
        """Test with a single contiguous True segment in mask."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        spike_times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        mask = np.array([False, True, True, True, False, False])

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        # Should return times[mask]
        assert_array_equal(times_sub, np.array([1.0, 2.0, 3.0]))
        # Spikes between t=1.0 and t=3.0: 1.5, 2.5 (not 0.5, 3.5, 4.5)
        assert_array_almost_equal(spike_times_sub, np.array([1.5, 2.5]))

    def test_multiple_contiguous_segments(self) -> None:
        """Test with multiple non-contiguous True segments."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        spike_times = np.array([0.5, 1.5, 2.5, 4.5, 5.5, 6.5])
        # Two segments: [1.0, 2.0] and [5.0, 6.0]
        mask = np.array([False, True, True, False, False, True, True, False])

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        # Should return times[mask]
        assert_array_equal(times_sub, np.array([1.0, 2.0, 5.0, 6.0]))
        # Spikes in [1.0, 2.0]: 1.5
        # Spikes in [5.0, 6.0]: 5.5
        assert_array_almost_equal(spike_times_sub, np.array([1.5, 5.5]))

    def test_empty_mask_all_false(self) -> None:
        """Test with all-False mask (no selected timepoints)."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([0.5, 1.5, 2.5])
        mask = np.array([False, False, False, False])

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        assert len(times_sub) == 0
        assert len(spike_times_sub) == 0

    def test_all_true_mask(self) -> None:
        """Test with all-True mask (all timepoints selected)."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([0.5, 1.5, 2.5])
        mask = np.array([True, True, True, True])

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        assert_array_equal(times_sub, times)
        # All spikes are within [0.0, 3.0]
        assert_array_almost_equal(spike_times_sub, spike_times)

    def test_no_spikes(self) -> None:
        """Test with empty spike train."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([])
        mask = np.array([True, True, True, True])

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        assert_array_equal(times_sub, times)
        assert len(spike_times_sub) == 0

    def test_no_spikes_in_segment(self) -> None:
        """Test when spikes exist but not in masked segment."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        spike_times = np.array([0.5, 4.5])  # Spikes outside mask
        mask = np.array([False, True, True, True, False, False])  # [1.0, 3.0]

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        assert_array_equal(times_sub, np.array([1.0, 2.0, 3.0]))
        assert len(spike_times_sub) == 0

    def test_spike_at_boundary(self) -> None:
        """Test behavior with spikes exactly at segment boundaries."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        spike_times = np.array([1.0, 2.0, 3.0])  # Spikes at boundaries
        mask = np.array([False, True, True, True])  # [1.0, 3.0]

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        assert_array_equal(times_sub, np.array([1.0, 2.0, 3.0]))
        # All boundary spikes should be included (inclusive boundaries)
        assert_array_almost_equal(spike_times_sub, np.array([1.0, 2.0, 3.0]))

    def test_single_timepoint_segment(self) -> None:
        """Test with a single True timepoint."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        spike_times = np.array([2.0])  # Spike at the single selected time
        mask = np.array([False, False, True, False, False])

        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )

        assert_array_equal(times_sub, np.array([2.0]))
        # For single point, segment is [2.0, 2.0] - spike at 2.0 should be included
        assert_array_almost_equal(spike_times_sub, np.array([2.0]))

    def test_times_sub_length_equals_mask_sum(self) -> None:
        """Test that returned times_sub has length equal to np.sum(mask)."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        spike_times = np.array([0.5, 2.5, 4.5])
        mask = np.array([True, False, True, True, False, True])

        times_sub, _ = _subset_spikes_by_time_mask(times, spike_times, mask)

        assert len(times_sub) == np.sum(mask)
