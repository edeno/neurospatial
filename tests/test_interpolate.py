"""Tests for Environment.interpolate() method."""

import numpy as np
import pytest

from neurospatial import Environment


class TestInterpolateBasic:
    """Basic interpolation functionality tests."""

    def test_interpolate_nearest_2d_grid(self):
        """Test nearest-neighbor interpolation on 2D grid."""
        # Create a simple 3x3 grid
        data = np.array(
            [[0, 0], [0, 2], [0, 4], [2, 0], [2, 2], [2, 4], [4, 0], [4, 2], [4, 4]],
            dtype=np.float64,
        )
        env = Environment.from_samples(data, bin_size=2.0)

        # Create a field with known values: field[i] = i
        field = np.arange(env.n_bins, dtype=np.float64)

        # Query at bin centers - should return exact values
        bin_centers = env.bin_centers
        result = env.interpolate(field, bin_centers, mode="nearest")

        assert result.shape == (env.n_bins,)
        np.testing.assert_array_almost_equal(result, field)

    def test_interpolate_nearest_between_bins(self):
        """Test nearest-neighbor picks closest bin."""
        # Create 1D-like environment for simplicity
        data = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=2.0)

        # Field values
        field = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        # Query point at 1.0 - closest to bin 0 (at 0.0)
        points = np.array([[1.0]])
        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (1,)
        assert result[0] == 10.0

        # Query point at 3.5 - closest to bin 1 (at 2.0) or bin 2 (at 4.0)
        # Should pick one deterministically
        points = np.array([[3.5]])
        result = env.interpolate(field, points, mode="nearest")
        assert result[0] in [20.0, 30.0]

    def test_interpolate_linear_2d_grid(self):
        """Test bilinear interpolation on 2D grid with known function."""
        # Create regular grid
        x = np.linspace(0, 10, 6)
        y = np.linspace(0, 10, 6)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=2.0)

        # Known linear function: f(x, y) = 2*x + 3*y
        field = 2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]

        # Query at intermediate points
        query_points = np.array([[1.5, 2.5], [5.0, 5.0], [7.3, 3.8]], dtype=np.float64)

        result = env.interpolate(field, query_points, mode="linear")
        expected = 2 * query_points[:, 0] + 3 * query_points[:, 1]

        # Linear interpolation of linear function should be exact
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_interpolate_multiple_points(self):
        """Test interpolation with multiple query points."""
        data = np.random.uniform(0, 10, size=(100, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.random.rand(env.n_bins)
        query_points = np.random.uniform(0, 10, size=(50, 2))

        result = env.interpolate(field, query_points, mode="nearest")

        assert result.shape == (50,)
        assert np.all(np.isfinite(result) | np.isnan(result))


class TestInterpolateOutsideBehavior:
    """Test handling of points outside environment."""

    def test_interpolate_outside_returns_nan(self):
        """Points outside environment should return NaN."""
        data = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=2.0)

        field = np.ones(env.n_bins, dtype=np.float64)

        # Query far outside
        outside_points = np.array([[100.0, 100.0], [-100.0, -100.0]])
        result = env.interpolate(field, outside_points, mode="nearest")

        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    def test_interpolate_mixed_inside_outside(self):
        """Mix of inside and outside points."""
        data = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=2.0)

        field = np.arange(env.n_bins, dtype=np.float64)

        # Mix inside and outside
        points = np.array(
            [
                [1.0, 1.0],  # inside
                [100.0, 100.0],  # outside
                [1.5, 0.5],  # inside
            ]
        )

        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (3,)
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])


class TestInterpolateLinearGridOnly:
    """Test that linear mode only works on grids."""

    def test_linear_mode_requires_grid(self):
        """Linear mode should raise NotImplementedError for non-grid layouts."""
        from shapely.geometry import box

        # Polygon layout is not a RegularGridLayout
        polygon = box(0, 0, 10, 10)
        env = Environment.from_polygon(polygon, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.array([[5.0, 5.0]])

        with pytest.raises(NotImplementedError, match=r"RegularGridLayout"):
            env.interpolate(field, points, mode="linear")


class TestInterpolateValidation:
    """Test input validation."""

    def test_interpolate_field_shape_mismatch(self):
        """Field shape must match n_bins."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # Wrong shape
        field = np.ones(env.n_bins + 5)
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"shape.*n_bins"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_field_not_1d(self):
        """Field must be 1-dimensional."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # 2D field
        field = np.ones((env.n_bins, 2))
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"1-D array"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_points_wrong_ndim(self):
        """Points must be 2D array (n_points, n_dims)."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)

        # 1D points array
        points = np.array([5.0, 5.0])

        with pytest.raises(ValueError, match=r"2-D array"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_points_dimension_mismatch(self):
        """Points dimensionality must match environment."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)

        # 3D points for 2D environment
        points = np.array([[5.0, 5.0, 5.0]])

        with pytest.raises(ValueError, match=r"dimension"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"mode.*nearest.*linear"):
            env.interpolate(field, points, mode="cubic")

    def test_interpolate_nan_in_field(self):
        """NaN values in field should raise ValueError."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        field[0] = np.nan
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"NaN"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_inf_in_field(self):
        """Inf values in field should raise ValueError."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        field[0] = np.inf
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"Inf"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_empty_points(self):
        """Empty points array should return empty result."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.empty((0, 2), dtype=np.float64)

        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (0,)

    def test_interpolate_nan_in_points(self):
        """NaN values in points should raise ValueError."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.array([[5.0, 5.0], [np.nan, 3.0]])

        with pytest.raises(ValueError, match=r"non-finite.*NaN"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_inf_in_points(self):
        """Inf values in points should raise ValueError."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.array([[5.0, 5.0], [np.inf, 3.0]])

        with pytest.raises(ValueError, match=r"non-finite.*Inf"):
            env.interpolate(field, points, mode="nearest")


class TestInterpolateEdgeCases:
    """Edge case tests."""

    def test_interpolate_single_point(self):
        """Single query point."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.random.rand(env.n_bins)
        points = np.array([[5.0, 5.0]])

        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_interpolate_constant_field(self):
        """Constant field should interpolate to constant."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.full(env.n_bins, 42.0)
        points = np.random.uniform(0, 10, size=(20, 2))

        result = env.interpolate(field, points, mode="nearest")

        # Points inside environment should be 42.0
        inside_mask = ~np.isnan(result)
        assert np.all(result[inside_mask] == 42.0)

    def test_interpolate_at_bin_centers_exact(self):
        """Interpolation at bin centers should return exact field values."""
        data = np.random.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.random.rand(env.n_bins)

        # Query at exact bin centers
        result = env.interpolate(field, env.bin_centers, mode="nearest")

        np.testing.assert_array_almost_equal(result, field)


class TestInterpolateMultipleLayouts:
    """Test interpolation across different layout types."""

    def test_interpolate_nearest_on_hexagonal(self):
        """Nearest-neighbor should work on hexagonal layout."""
        # Create sample data for hexagonal grid
        np.random.seed(42)
        data_points = np.random.uniform(0, 10, size=(100, 2))

        env = Environment.from_samples(
            data_samples=data_points,
            bin_size=2.0,
            layout_kind="Hexagonal",
        )

        # Create a simple field (e.g., distance from origin)
        field = np.linalg.norm(env.bin_centers, axis=1)

        # Query points
        query_points = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])

        result = env.interpolate(field, query_points, mode="nearest")

        assert result.shape == (3,)
        assert not np.any(np.isnan(result))  # All points should be inside environment

    def test_interpolate_nearest_on_polygon(self):
        """Nearest-neighbor should work on polygon layout."""
        from shapely.geometry import box

        polygon = box(0, 0, 10, 10)

        env = Environment.from_polygon(polygon=polygon, bin_size=1.0)

        field = np.random.rand(env.n_bins)
        points = np.random.uniform(0, 10, size=(20, 2))

        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (20,)


class TestInterpolateLinearAccuracy:
    """Test accuracy of linear interpolation."""

    def test_linear_interpolation_of_plane(self):
        """Linear interpolation should exactly recover plane f(x,y) = ax + by + c."""
        # Create fine grid
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=1.0)

        # Plane: f(x, y) = 3*x - 2*y + 5
        a, b, c = 3.0, -2.0, 5.0
        field = a * env.bin_centers[:, 0] + b * env.bin_centers[:, 1] + c

        # Query at random points
        np.random.seed(42)
        query_points = np.random.uniform(0, 10, size=(50, 2))

        result = env.interpolate(field, query_points, mode="linear")
        expected = a * query_points[:, 0] + b * query_points[:, 1] + c

        # Should be exact for linear function
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_linear_extrapolation_returns_nan(self):
        """Linear mode should return NaN for points outside grid bounds."""
        x = np.linspace(0, 10, 6)
        y = np.linspace(0, 10, 6)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=2.0)
        field = np.ones(env.n_bins)

        # Points outside grid
        outside_points = np.array([[15.0, 5.0], [5.0, -5.0]])
        result = env.interpolate(field, outside_points, mode="linear")

        assert np.all(np.isnan(result))


class TestInterpolateDeterminism:
    """Test deterministic behavior."""

    def test_interpolate_deterministic(self):
        """Repeated calls should give identical results."""
        data = np.random.RandomState(42).uniform(0, 10, size=(100, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.random.RandomState(43).rand(env.n_bins)
        points = np.random.RandomState(44).uniform(0, 10, size=(50, 2))

        result1 = env.interpolate(field, points, mode="nearest")
        result2 = env.interpolate(field, points, mode="nearest")

        np.testing.assert_array_equal(result1, result2)
