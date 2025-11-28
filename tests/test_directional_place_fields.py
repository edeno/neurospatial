"""Tests for directional place field computation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neurospatial import Environment, compute_place_field
from neurospatial.spike_field import (
    DirectionalPlaceFields,
    _subset_spikes_by_time_mask,
    compute_directional_place_fields,
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


class TestComputeDirectionalPlaceFields:
    """Tests for the compute_directional_place_fields function."""

    @pytest.fixture
    def sample_env(self) -> Environment:
        """Create a simple 2D environment for testing."""
        positions = np.column_stack(
            [np.linspace(0, 100, 200), np.linspace(0, 100, 200)]
        )
        return Environment.from_samples(positions, bin_size=10.0)

    @pytest.fixture
    def sample_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample trajectory data."""
        times = np.linspace(0, 20, 200)  # 20 seconds
        positions = np.column_stack(
            [np.linspace(0, 100, 200), np.linspace(0, 100, 200)]
        )
        return times, positions

    def test_constant_labels_equals_compute_place_field(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """If all labels are the same, result equals compute_place_field."""
        times, positions = sample_trajectory
        spike_times = np.array([2.0, 5.0, 10.0, 15.0])

        # All labels the same (not "other")
        labels = np.full(len(times), "forward", dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        # Compare with compute_place_field
        expected = compute_place_field(
            sample_env,
            spike_times,
            times,
            positions,
            method="binned",
            bandwidth=10.0,
        )

        assert len(result.fields) == 1
        assert "forward" in result.fields
        assert result.labels == ("forward",)
        # Should be numerically close
        assert_array_almost_equal(result.fields["forward"], expected, decimal=5)

    def test_two_directions_partition(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Two non-overlapping directions produce independent fields."""
        times, positions = sample_trajectory
        spike_times = np.array([2.0, 5.0, 12.0, 15.0])

        # First half is "A", second half is "B"
        labels = np.array(
            ["A"] * 100 + ["B"] * 100, dtype=object
        )  # 200 total like times

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert len(result.fields) == 2
        assert "A" in result.fields
        assert "B" in result.fields
        assert set(result.labels) == {"A", "B"}

        # Each field should have shape (n_bins,)
        assert result.fields["A"].shape == (sample_env.n_bins,)
        assert result.fields["B"].shape == (sample_env.n_bins,)

    def test_other_label_excluded(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The 'other' label is excluded from results."""
        times, positions = sample_trajectory
        spike_times = np.array([2.0, 5.0, 15.0])

        # Mix of "forward", "other", and "backward"
        labels = np.array(
            ["forward"] * 50 + ["other"] * 100 + ["backward"] * 50, dtype=object
        )

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        # "other" should NOT be in results
        assert "other" not in result.fields
        assert "other" not in result.labels
        assert len(result.fields) == 2
        assert "forward" in result.fields
        assert "backward" in result.fields

    def test_no_spikes_returns_zero_or_nan_fields(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Empty spike train produces zero/NaN fields."""
        times, positions = sample_trajectory
        spike_times = np.array([])  # No spikes

        labels = np.full(len(times), "forward", dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert "forward" in result.fields
        # Field should be all zeros or NaN (depending on occupancy)
        field = result.fields["forward"]
        assert np.all(np.isnan(field) | (field == 0))

    def test_result_structure(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """DirectionalPlaceFields has correct structure."""
        times, positions = sample_trajectory
        spike_times = np.array([5.0, 10.0])

        labels = np.array(["A"] * 100 + ["B"] * 100, dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        # Result should be DirectionalPlaceFields
        assert isinstance(result, DirectionalPlaceFields)

        # fields should be a mapping
        assert hasattr(result.fields, "__getitem__")

        # labels should be a tuple
        assert isinstance(result.labels, tuple)

        # All fields should have correct shape
        for label in result.labels:
            assert result.fields[label].shape == (sample_env.n_bins,)

    def test_length_mismatch_raises_error(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Raises ValueError if direction_labels length doesn't match times."""
        times, positions = sample_trajectory
        spike_times = np.array([5.0])

        # Wrong length labels
        wrong_labels = np.array(["A", "B", "C"], dtype=object)

        with pytest.raises(ValueError, match="direction_labels"):
            compute_directional_place_fields(
                sample_env,
                spike_times,
                times,
                positions,
                wrong_labels,
                method="binned",
                bandwidth=10.0,
            )

    def test_all_other_labels_returns_empty(
        self, sample_env: Environment, sample_trajectory: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """If all labels are 'other', returns empty fields."""
        times, positions = sample_trajectory
        spike_times = np.array([5.0, 10.0])

        labels = np.full(len(times), "other", dtype=object)

        result = compute_directional_place_fields(
            sample_env,
            spike_times,
            times,
            positions,
            labels,
            method="binned",
            bandwidth=10.0,
        )

        assert len(result.fields) == 0
        assert len(result.labels) == 0
