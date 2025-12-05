"""Tests for ObjectVectorOverlay animation overlay.

Tests the ObjectVectorOverlay class which visualizes vectors from animal
positions to objects (e.g., for object-vector cell analysis).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal


class TestObjectVectorOverlayModule:
    """Test module structure and imports."""

    def test_import_from_overlays(self):
        """Test ObjectVectorOverlay can be imported from overlays module."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        assert ObjectVectorOverlay is not None

    def test_import_from_animation(self):
        """Test ObjectVectorOverlay can be imported from animation package."""
        from neurospatial.animation import ObjectVectorOverlay

        assert ObjectVectorOverlay is not None

    def test_implements_overlay_protocol(self):
        """Test ObjectVectorOverlay implements OverlayProtocol."""
        from neurospatial.animation.overlays import (
            ObjectVectorOverlay,
            OverlayProtocol,
        )

        # Create minimal overlay instance
        object_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
        )

        # Check protocol compliance
        assert isinstance(overlay, OverlayProtocol)
        assert hasattr(overlay, "times")
        assert hasattr(overlay, "interp")
        assert hasattr(overlay, "convert_to_data")


class TestObjectVectorOverlayCreation:
    """Test ObjectVectorOverlay dataclass creation."""

    def test_basic_creation(self):
        """Test creating ObjectVectorOverlay with required fields only."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
        )

        assert_array_equal(overlay.object_positions, object_positions)
        assert_array_equal(overlay.animal_positions, animal_positions)
        # Check defaults
        assert overlay.times is None
        assert overlay.firing_rates is None
        assert overlay.color == "white"
        assert overlay.linewidth == 2.0
        assert overlay.show_objects is True
        assert overlay.object_marker == "o"
        assert overlay.object_size == 15.0
        assert overlay.interp == "linear"

    def test_with_times(self):
        """Test ObjectVectorOverlay with timestamps."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[10.0, 10.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0]])
        times = np.array([0.0, 1.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        assert_array_equal(overlay.times, times)

    def test_with_firing_rates(self):
        """Test ObjectVectorOverlay with firing rate modulation."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[10.0, 10.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
        firing_rates = np.array([1.0, 5.0, 2.0])  # One per timepoint

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            firing_rates=firing_rates,
        )

        assert_array_equal(overlay.firing_rates, firing_rates)

    def test_custom_styling(self):
        """Test ObjectVectorOverlay with custom visual styling."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[10.0, 10.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0]])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            color="cyan",
            linewidth=3.0,
            show_objects=False,
            object_marker="s",
            object_size=20.0,
        )

        assert overlay.color == "cyan"
        assert overlay.linewidth == 3.0
        assert overlay.show_objects is False
        assert overlay.object_marker == "s"
        assert overlay.object_size == 20.0

    def test_interp_nearest(self):
        """Test ObjectVectorOverlay with nearest interpolation."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[10.0, 10.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0]])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            interp="nearest",
        )

        assert overlay.interp == "nearest"

    def test_single_object(self):
        """Test ObjectVectorOverlay with single object."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[15.0, 15.0]])
        animal_positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
        )

        assert overlay.object_positions.shape == (1, 2)

    def test_multiple_objects(self):
        """Test ObjectVectorOverlay with multiple objects."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        animal_positions = np.array([[0.0, 0.0], [5.0, 5.0]])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
        )

        assert overlay.object_positions.shape == (3, 2)


class TestObjectVectorOverlayConversion:
    """Test ObjectVectorOverlay.convert_to_data() method."""

    @pytest.fixture
    def mock_env(self) -> Any:
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims = 2
            dimension_ranges = np.array([[0.0, 100.0], [0.0, 100.0]])

        return MockEnv()

    def test_convert_to_data_basic(self, mock_env: Any):
        """Test basic conversion to internal data representation."""
        from neurospatial.animation.overlays import (
            ObjectVectorData,
            ObjectVectorOverlay,
        )

        object_positions = np.array([[50.0, 50.0]])
        animal_positions = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        times = np.array([0.0, 1.0, 2.0])
        frame_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        n_frames = len(frame_times)

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert isinstance(result, ObjectVectorData)
        assert result.animal_positions.shape == (n_frames, 2)
        assert_array_equal(result.object_positions, object_positions)
        assert result.color == "white"
        assert result.linewidth == 2.0
        assert result.show_objects is True

    def test_convert_to_data_with_firing_rates(self, mock_env: Any):
        """Test conversion includes interpolated firing rates."""
        from neurospatial.animation.overlays import (
            ObjectVectorData,
            ObjectVectorOverlay,
        )

        object_positions = np.array([[50.0, 50.0]])
        animal_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        times = np.array([0.0, 1.0])
        firing_rates = np.array([0.0, 10.0])
        frame_times = np.array([0.0, 0.5, 1.0])
        n_frames = len(frame_times)

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
            firing_rates=firing_rates,
        )

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert isinstance(result, ObjectVectorData)
        assert result.firing_rates is not None
        assert result.firing_rates.shape == (n_frames,)
        # Check interpolation: at t=0.5, firing rate should be 5.0 (midpoint)
        np.testing.assert_allclose(result.firing_rates[1], 5.0)

    def test_convert_to_data_no_times(self, mock_env: Any):
        """Test conversion when no times provided (assumes uniform spacing)."""
        from neurospatial.animation.overlays import (
            ObjectVectorData,
            ObjectVectorOverlay,
        )

        object_positions = np.array([[50.0, 50.0]])
        animal_positions = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = len(frame_times)

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
        )

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert isinstance(result, ObjectVectorData)
        # Should map directly when n_samples == n_frames
        assert result.animal_positions.shape == (n_frames, 2)

    def test_convert_preserves_object_positions(self, mock_env: Any):
        """Test that object positions are preserved without modification."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[25.0, 75.0], [75.0, 25.0]])
        animal_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        times = np.array([0.0, 1.0])
        frame_times = np.array([0.0, 0.5, 1.0])
        n_frames = len(frame_times)

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        result = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Object positions should be unchanged
        assert_array_equal(result.object_positions, object_positions)


class TestObjectVectorDataContainer:
    """Test ObjectVectorData internal data container."""

    def test_data_container_creation(self):
        """Test creating ObjectVectorData container."""
        from neurospatial.animation.overlays import ObjectVectorData

        object_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])

        data = ObjectVectorData(
            object_positions=object_positions,
            animal_positions=animal_positions,
            firing_rates=None,
            color="white",
            linewidth=2.0,
            show_objects=True,
            object_marker="o",
            object_size=15.0,
        )

        assert_array_equal(data.object_positions, object_positions)
        assert_array_equal(data.animal_positions, animal_positions)
        assert data.firing_rates is None
        assert data.color == "white"
        assert data.linewidth == 2.0
        assert data.show_objects is True
        assert data.object_marker == "o"
        assert data.object_size == 15.0

    def test_data_container_with_firing_rates(self):
        """Test ObjectVectorData with firing rates."""
        from neurospatial.animation.overlays import ObjectVectorData

        object_positions = np.array([[10.0, 10.0]])
        animal_positions = np.array([[5.0, 5.0], [6.0, 6.0]])
        firing_rates = np.array([1.0, 2.0])

        data = ObjectVectorData(
            object_positions=object_positions,
            animal_positions=animal_positions,
            firing_rates=firing_rates,
            color="white",
            linewidth=2.0,
            show_objects=True,
            object_marker="o",
            object_size=15.0,
        )

        assert_array_equal(data.firing_rates, firing_rates)


class TestObjectVectorOverlayValidation:
    """Test validation in ObjectVectorOverlay.convert_to_data()."""

    @pytest.fixture
    def mock_env(self) -> Any:
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims = 2
            dimension_ranges = np.array([[0.0, 100.0], [0.0, 100.0]])

        return MockEnv()

    def test_validates_animal_positions_shape(self, mock_env: Any):
        """Test that animal_positions shape is validated."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[50.0, 50.0]])
        animal_positions = np.array([[10.0, 10.0, 10.0]])  # 3D, but env is 2D
        times = np.array([0.0])
        frame_times = np.array([0.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        with pytest.raises(ValueError, match="n_dims"):
            overlay.convert_to_data(frame_times, len(frame_times), mock_env)

    def test_validates_object_positions_shape(self, mock_env: Any):
        """Test that object_positions shape is validated."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[50.0, 50.0, 50.0]])  # 3D, but env is 2D
        animal_positions = np.array([[10.0, 10.0]])
        times = np.array([0.0])
        frame_times = np.array([0.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        with pytest.raises(ValueError, match="n_dims"):
            overlay.convert_to_data(frame_times, len(frame_times), mock_env)

    def test_validates_firing_rates_length(self, mock_env: Any):
        """Test that firing_rates length matches animal_positions."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[50.0, 50.0]])
        animal_positions = np.array([[10.0, 10.0], [20.0, 20.0]])  # 2 samples
        times = np.array([0.0, 1.0])
        firing_rates = np.array([1.0, 2.0, 3.0])  # 3 samples - mismatch!
        frame_times = np.array([0.0, 0.5, 1.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
            firing_rates=firing_rates,
        )

        with pytest.raises(ValueError, match="firing_rates"):
            overlay.convert_to_data(frame_times, len(frame_times), mock_env)

    def test_validates_firing_rates_finite(self, mock_env: Any):
        """Test that firing_rates with NaN/Inf are rejected."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[50.0, 50.0]])
        animal_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        times = np.array([0.0, 1.0])
        firing_rates = np.array([1.0, np.nan])  # Contains NaN
        frame_times = np.array([0.0, 1.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
            firing_rates=firing_rates,
        )

        with pytest.raises(ValueError, match="finite"):
            overlay.convert_to_data(frame_times, len(frame_times), mock_env)

    def test_validates_object_positions_finite(self, mock_env: Any):
        """Test that object_positions with NaN/Inf are rejected."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[50.0, np.inf]])  # Contains Inf
        animal_positions = np.array([[10.0, 10.0]])
        times = np.array([0.0])
        frame_times = np.array([0.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        with pytest.raises(ValueError, match="finite"):
            overlay.convert_to_data(frame_times, len(frame_times), mock_env)

    def test_warns_for_positions_outside_bounds(self, mock_env: Any):
        """Test warning when positions are outside environment bounds."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        object_positions = np.array([[50.0, 50.0]])
        # Animal positions outside [0, 100] bounds
        animal_positions = np.array([[-10.0, 10.0], [110.0, 50.0]])
        times = np.array([0.0, 1.0])
        frame_times = np.array([0.0, 1.0])

        overlay = ObjectVectorOverlay(
            object_positions=object_positions,
            animal_positions=animal_positions,
            times=times,
        )

        # Should warn but not raise
        with pytest.warns(UserWarning, match="outside.*bounds"):
            overlay.convert_to_data(frame_times, len(frame_times), mock_env)


class TestObjectVectorOverlayDocstring:
    """Test that ObjectVectorOverlay has proper documentation."""

    def test_class_has_docstring(self):
        """Test ObjectVectorOverlay has a docstring."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        assert ObjectVectorOverlay.__doc__ is not None
        assert len(ObjectVectorOverlay.__doc__) > 100

    def test_docstring_describes_purpose(self):
        """Test docstring describes the overlay's purpose."""
        from neurospatial.animation.overlays import ObjectVectorOverlay

        docstring = ObjectVectorOverlay.__doc__
        # Should mention vectors, objects, and animal
        assert "vector" in docstring.lower() or "object" in docstring.lower()

    def test_data_container_has_docstring(self):
        """Test ObjectVectorData has a docstring."""
        from neurospatial.animation.overlays import ObjectVectorData

        assert ObjectVectorData.__doc__ is not None
