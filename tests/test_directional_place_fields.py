"""Tests for directional place field computation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.spike_field import DirectionalPlaceFields


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
