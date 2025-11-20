"""Tests for animation overlay dataclasses and validation functions.

This module tests the public API dataclasses (PositionOverlay, BodypartOverlay,
HeadDirectionOverlay) and their validation/conversion pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
)


class TestPositionOverlay:
    """Test PositionOverlay dataclass."""

    def test_basic_creation(self):
        """Test creating a PositionOverlay with required fields."""
        data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        overlay = PositionOverlay(data=data)

        assert_array_equal(overlay.data, data)
        assert overlay.times is None
        assert overlay.color == "red"
        assert overlay.size == 10.0
        assert overlay.trail_length is None

    def test_with_times(self):
        """Test PositionOverlay with timestamps."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        times = np.array([0.0, 1.0])
        overlay = PositionOverlay(data=data, times=times)

        assert_array_equal(overlay.data, data)
        assert_array_equal(overlay.times, times)

    def test_custom_styling(self):
        """Test PositionOverlay with custom color, size, trail."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        overlay = PositionOverlay(data=data, color="blue", size=20.0, trail_length=10)

        assert overlay.color == "blue"
        assert overlay.size == 20.0
        assert overlay.trail_length == 10

    def test_3d_data(self):
        """Test PositionOverlay with 3D coordinates."""
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        overlay = PositionOverlay(data=data)

        assert overlay.data.shape == (2, 3)
        assert_array_equal(overlay.data, data)


class TestBodypartOverlay:
    """Test BodypartOverlay dataclass."""

    def test_basic_creation(self):
        """Test creating a BodypartOverlay with required fields."""
        data = {
            "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        overlay = BodypartOverlay(data=data)

        assert "head" in overlay.data
        assert "body" in overlay.data
        assert_array_equal(overlay.data["head"], data["head"])
        assert_array_equal(overlay.data["body"], data["body"])
        assert overlay.times is None
        assert overlay.skeleton is None
        assert overlay.colors is None
        assert overlay.skeleton_color == "white"
        assert overlay.skeleton_width == 2.0

    def test_with_skeleton(self):
        """Test BodypartOverlay with skeleton connections."""
        data = {
            "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "tail": np.array([[2.0, 3.0], [4.0, 5.0]]),
        }
        skeleton = [("head", "body"), ("body", "tail")]
        overlay = BodypartOverlay(data=data, skeleton=skeleton)

        assert overlay.skeleton == skeleton
        assert len(overlay.skeleton) == 2

    def test_with_custom_colors(self):
        """Test BodypartOverlay with per-part colors."""
        data = {
            "head": np.array([[0.0, 1.0]]),
            "body": np.array([[1.0, 2.0]]),
        }
        colors = {"head": "red", "body": "blue"}
        overlay = BodypartOverlay(data=data, colors=colors)

        assert overlay.colors == colors
        assert overlay.colors["head"] == "red"
        assert overlay.colors["body"] == "blue"

    def test_custom_skeleton_styling(self):
        """Test BodypartOverlay with custom skeleton appearance."""
        data = {"head": np.array([[0.0, 1.0]])}
        overlay = BodypartOverlay(
            data=data, skeleton_color="yellow", skeleton_width=3.0
        )

        assert overlay.skeleton_color == "yellow"
        assert overlay.skeleton_width == 3.0

    def test_with_times(self):
        """Test BodypartOverlay with timestamps."""
        data = {
            "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        times = np.array([0.0, 1.0])
        overlay = BodypartOverlay(data=data, times=times)

        assert_array_equal(overlay.times, times)


class TestHeadDirectionOverlay:
    """Test HeadDirectionOverlay dataclass."""

    def test_basic_creation_with_angles(self):
        """Test creating HeadDirectionOverlay with angle data."""
        data = np.array([0.0, np.pi / 2, np.pi])
        overlay = HeadDirectionOverlay(data=data)

        assert_array_equal(overlay.data, data)
        assert overlay.times is None
        assert overlay.color == "yellow"
        assert overlay.length == 20.0

    def test_with_unit_vectors(self):
        """Test HeadDirectionOverlay with unit vector data."""
        data = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        overlay = HeadDirectionOverlay(data=data)

        assert overlay.data.shape == (3, 2)
        assert_array_equal(overlay.data, data)

    def test_custom_styling(self):
        """Test HeadDirectionOverlay with custom appearance."""
        data = np.array([0.0, np.pi])
        overlay = HeadDirectionOverlay(data=data, color="red", length=30.0)

        assert overlay.color == "red"
        assert overlay.length == 30.0

    def test_with_times(self):
        """Test HeadDirectionOverlay with timestamps."""
        data = np.array([0.0, np.pi / 2])
        times = np.array([0.0, 1.0])
        overlay = HeadDirectionOverlay(data=data, times=times)

        assert_array_equal(overlay.times, times)

    def test_3d_vectors(self):
        """Test HeadDirectionOverlay with 3D unit vectors."""
        data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        overlay = HeadDirectionOverlay(data=data)

        assert overlay.data.shape == (2, 3)
        assert_array_equal(overlay.data, data)


class TestOverlayDataclassDefaults:
    """Test that overlay dataclasses have correct default values."""

    def test_position_overlay_defaults(self):
        """Test PositionOverlay default values."""
        data = np.array([[0.0, 1.0]])
        overlay = PositionOverlay(data=data)

        assert overlay.color == "red"
        assert overlay.size == 10.0
        assert overlay.trail_length is None

    def test_bodypart_overlay_defaults(self):
        """Test BodypartOverlay default values."""
        data = {"head": np.array([[0.0, 1.0]])}
        overlay = BodypartOverlay(data=data)

        assert overlay.skeleton_color == "white"
        assert overlay.skeleton_width == 2.0

    def test_head_direction_overlay_defaults(self):
        """Test HeadDirectionOverlay default values."""
        data = np.array([0.0])
        overlay = HeadDirectionOverlay(data=data)

        assert overlay.color == "yellow"
        assert overlay.length == 20.0


class TestMultiAnimalSupport:
    """Test that multiple overlays can be created for multi-animal scenarios."""

    def test_multiple_position_overlays(self):
        """Test creating multiple PositionOverlay instances for multiple animals."""
        animal1_data = np.array([[0.0, 1.0], [2.0, 3.0]])
        animal2_data = np.array([[5.0, 6.0], [7.0, 8.0]])

        overlay1 = PositionOverlay(data=animal1_data, color="red")
        overlay2 = PositionOverlay(data=animal2_data, color="blue")

        assert overlay1.color == "red"
        assert overlay2.color == "blue"
        assert not np.array_equal(overlay1.data, overlay2.data)

    def test_multiple_bodypart_overlays(self):
        """Test creating multiple BodypartOverlay instances."""
        animal1_data = {"head": np.array([[0.0, 1.0]])}
        animal2_data = {"head": np.array([[5.0, 6.0]])}

        overlay1 = BodypartOverlay(data=animal1_data)
        overlay2 = BodypartOverlay(data=animal2_data)

        assert not np.array_equal(overlay1.data["head"], overlay2.data["head"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
