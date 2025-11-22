"""Tests for animation coordinate transforms.

These tests verify the coordinate transformation functions used to convert
between environment (x, y) coordinates and napari pixel (row, col) coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 10x10 environment for testing transforms."""
    # Create positions that span a 10x10 grid
    positions = np.array(
        [
            [0.0, 0.0],  # Bottom-left
            [10.0, 0.0],  # Bottom-right
            [0.0, 10.0],  # Top-left
            [10.0, 10.0],  # Top-right
        ]
    )
    return Environment.from_samples(positions, bin_size=1.0)


@pytest.fixture
def rectangular_env() -> Environment:
    """Create a rectangular 20x10 environment for testing transforms."""
    positions = np.array(
        [
            [0.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [20.0, 10.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=1.0)


class TestEnvScale:
    """Tests for _EnvScale helper class."""

    def test_env_scale_creation(self, simple_env: Environment) -> None:
        """Test _EnvScale creates correctly from environment."""
        from neurospatial.animation.transforms import EnvScale

        scale = EnvScale.from_env(simple_env)

        assert scale is not None
        assert scale.n_x > 0
        assert scale.n_y > 0
        assert scale.x_scale > 0
        assert scale.y_scale > 0

    def test_env_scale_returns_none_for_missing_attributes(self) -> None:
        """Test _EnvScale returns None when environment lacks required attributes."""
        from neurospatial.animation.transforms import EnvScale

        # None environment
        assert EnvScale.from_env(None) is None

        # Object without dimension_ranges
        class FakeEnv:
            pass

        assert EnvScale.from_env(FakeEnv()) is None

    def test_env_scale_from_rectangular_env(self, rectangular_env: Environment) -> None:
        """Test _EnvScale handles rectangular environments correctly."""
        from neurospatial.animation.transforms import EnvScale

        scale = EnvScale.from_env(rectangular_env)

        assert scale is not None
        # X range is 20, Y range is 10 - scales should differ
        assert scale.x_max - scale.x_min == pytest.approx(20.0, rel=0.1)
        assert scale.y_max - scale.y_min == pytest.approx(10.0, rel=0.1)


class TestTransformCoordsForNapari:
    """Tests for transform_coords_for_napari function."""

    def test_basic_coordinate_transform(self, simple_env: Environment) -> None:
        """Test basic coordinate transformation from env to napari space."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        # Test a single point
        coords = np.array([[5.0, 5.0]])  # Center of 10x10 env
        result = transform_coords_for_napari(coords, simple_env)

        # Result should be in (row, col) format
        assert result.shape == coords.shape

    def test_origin_transforms_to_bottom_left(self, simple_env: Environment) -> None:
        """Test that (0, 0) in env transforms to high row (bottom of napari image)."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[0.0, 0.0]])  # Bottom-left in env
        result = transform_coords_for_napari(coords, simple_env)

        # In napari, bottom-left has high row index, low col index
        # Row should be near max (n_y - 1), col should be near 0
        _row, col = result[0]
        assert col == pytest.approx(0.0, abs=0.5)
        # Row should be high (bottom of image in napari)

    def test_top_right_transforms_correctly(self, simple_env: Environment) -> None:
        """Test that (max_x, max_y) transforms to low row, high col."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        # Get environment bounds
        (_x_min, x_max), (_y_min, y_max) = simple_env.dimension_ranges
        coords = np.array([[x_max, y_max]])  # Top-right in env

        result = transform_coords_for_napari(coords, simple_env)

        row, _col = result[0]
        # Top-right in env → low row (top of napari), high col (right)
        assert row == pytest.approx(0.0, abs=0.5)  # Top of napari image
        # Col should be near max

    def test_batch_coordinate_transform(self, simple_env: Environment) -> None:
        """Test batch transformation of multiple coordinates."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array(
            [
                [0.0, 0.0],
                [5.0, 5.0],
                [10.0, 10.0],
            ]
        )
        result = transform_coords_for_napari(coords, simple_env)

        assert result.shape == coords.shape
        # Results should all be finite
        assert np.all(np.isfinite(result))

    def test_3d_coords_transform(self, simple_env: Environment) -> None:
        """Test transformation of 3D array (n_frames, n_points, 2)."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        # Shape: (3 frames, 2 points, 2 dims)
        coords = np.array(
            [
                [[0.0, 0.0], [5.0, 5.0]],
                [[2.0, 3.0], [7.0, 8.0]],
                [[1.0, 1.0], [9.0, 9.0]],
            ]
        )
        result = transform_coords_for_napari(coords, simple_env)

        assert result.shape == coords.shape
        assert np.all(np.isfinite(result))

    def test_higher_dims_unchanged(self, simple_env: Environment) -> None:
        """Test that 3D or higher spatial coordinates are returned unchanged."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords_3d = np.array([[1.0, 2.0, 3.0]])  # 3D spatial
        result = transform_coords_for_napari(coords_3d, simple_env)

        assert_allclose(result, coords_3d)

    def test_nan_handling(self, simple_env: Environment) -> None:
        """Test that NaN values are preserved through transformation."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[np.nan, 5.0], [5.0, np.nan], [np.nan, np.nan]])
        result = transform_coords_for_napari(coords, simple_env)

        # NaN should propagate
        assert np.isnan(result[0, 0]) or np.isnan(result[0, 1])
        assert np.isnan(result[1, 0]) or np.isnan(result[1, 1])
        assert np.all(np.isnan(result[2]))


class TestTransformDirectionForNapari:
    """Tests for transform_direction_for_napari function."""

    def test_basic_direction_transform(self, simple_env: Environment) -> None:
        """Test basic direction vector transformation."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        # Direction pointing right and up in env
        direction = np.array([[1.0, 1.0]])
        result = transform_direction_for_napari(direction, simple_env)

        assert result.shape == direction.shape
        assert np.all(np.isfinite(result))

    def test_direction_x_axis_only(self, simple_env: Environment) -> None:
        """Test direction along X axis (right in env → right in napari col)."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        # Direction pointing right (positive X)
        direction = np.array([[1.0, 0.0]])
        result = transform_direction_for_napari(direction, simple_env)

        # In napari: X becomes col (unchanged sign)
        # result[0, 1] should be positive (pointing right)
        _row_dir, col_dir = result[0]
        assert col_dir > 0  # Positive X → positive col direction

    def test_direction_y_axis_only(self, simple_env: Environment) -> None:
        """Test direction along Y axis (up in env → up in napari, so negative row)."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        # Direction pointing up (positive Y in env)
        direction = np.array([[0.0, 1.0]])
        result = transform_direction_for_napari(direction, simple_env)

        # In napari: positive Y env → negative row direction (up in napari)
        row_dir, _col_dir = result[0]
        assert row_dir < 0  # Positive Y → negative row direction (up)

    def test_direction_no_translation(self, simple_env: Environment) -> None:
        """Test that direction vectors are not affected by position offset."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        # Same direction at different starting points should give same result
        direction = np.array([[2.0, 3.0]])

        result = transform_direction_for_napari(direction, simple_env)

        # Direction transformation should be consistent regardless of "position"
        # (though this function doesn't take position - it only transforms the direction)
        assert result.shape == direction.shape

    def test_batch_direction_transform(self, simple_env: Environment) -> None:
        """Test batch transformation of multiple directions."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        directions = np.array(
            [
                [1.0, 0.0],  # Right
                [0.0, 1.0],  # Up
                [-1.0, 0.0],  # Left
                [0.0, -1.0],  # Down
            ]
        )
        result = transform_direction_for_napari(directions, simple_env)

        assert result.shape == directions.shape
        assert np.all(np.isfinite(result))

    def test_3d_direction_unchanged(self, simple_env: Environment) -> None:
        """Test that 3D direction vectors are returned unchanged."""
        from neurospatial.animation.transforms import transform_direction_for_napari

        direction_3d = np.array([[1.0, 2.0, 3.0]])
        result = transform_direction_for_napari(direction_3d, simple_env)

        assert_allclose(result, direction_3d)


class TestTransformConsistency:
    """Tests for consistency between coordinate and direction transforms."""

    def test_transform_consistency_with_env_scale(
        self, simple_env: Environment
    ) -> None:
        """Test that using pre-computed EnvScale gives same results."""
        from neurospatial.animation.transforms import (
            EnvScale,
            transform_coords_for_napari,
            transform_direction_for_napari,
        )

        scale = EnvScale.from_env(simple_env)
        coords = np.array([[5.0, 5.0]])
        direction = np.array([[1.0, 1.0]])

        # Transform with env
        result_coords_env = transform_coords_for_napari(coords, simple_env)
        result_dir_env = transform_direction_for_napari(direction, simple_env)

        # Transform with pre-computed scale
        result_coords_scale = transform_coords_for_napari(coords, scale)
        result_dir_scale = transform_direction_for_napari(direction, scale)

        assert_allclose(result_coords_env, result_coords_scale)
        assert_allclose(result_dir_env, result_dir_scale)


class TestFallbackBehavior:
    """Tests for fallback behavior when environment lacks required attributes."""

    @pytest.fixture(autouse=True)
    def reset_warning_flag(self) -> None:
        """Reset the fallback warning flag before each test."""
        from neurospatial.animation.transforms import reset_transform_warning

        reset_transform_warning()

    def test_fallback_without_env(self) -> None:
        """Test fallback behavior when env is None."""
        from neurospatial.animation.transforms import transform_coords_for_napari

        coords = np.array([[5.0, 5.0]])

        with pytest.warns(UserWarning, match="falling back"):
            result = transform_coords_for_napari(coords, None)

        # Fallback just swaps axes
        assert result.shape == coords.shape

    def test_fallback_direction_without_env(self) -> None:
        """Test direction fallback behavior when env is None."""
        from neurospatial.animation.transforms import (
            reset_transform_warning,
            transform_direction_for_napari,
        )

        # Reset warning again to ensure this test triggers its own warning
        reset_transform_warning()
        direction = np.array([[1.0, 2.0]])

        with pytest.warns(UserWarning, match="falling back"):
            result = transform_direction_for_napari(direction, None)

        # Fallback swaps axes and inverts Y
        assert result.shape == direction.shape
