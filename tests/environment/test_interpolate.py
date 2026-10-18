"""Tests for Environment.interpolate() method."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_linear_interpolation_recovers_centers_next_to_holes(n_dims: int) -> None:
    """Zero-weight missing neighbors cannot erase a known active-bin value."""
    mask = np.ones((3,) * n_dims, dtype=bool)
    mask[(1,) * n_dims] = False
    env = Environment.from_grid_mask(mask, (np.arange(4.0),) * n_dims)
    field = np.arange(env.n_bins, dtype=float) + 1.0
    actual = env.interpolate(field, env.bin_centers, mode="linear")
    np.testing.assert_allclose(actual, field)


def test_linear_interpolation_ignores_only_zero_weight_holes() -> None:
    """An active grid line interpolates; missing positive-weight corners stay NaN."""
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    env = Environment.from_grid_mask(mask, (np.arange(4.0),) * 2)
    field = 2.0 * env.bin_centers[:, 0] + 3.0 * env.bin_centers[:, 1]
    points = np.array([[0.5, 1.0], [1.0, 0.5], [1.0, 1.0], [1.5, 1.5], [-1, 1]])
    actual = env.interpolate(field, points, mode="linear")
    np.testing.assert_allclose(actual[:2], [4.0, 3.5])
    assert np.isnan(actual[2:]).all()


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
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(100, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = rng.random(env.n_bins)
        query_points = rng.uniform(0, 10, size=(50, 2))

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
        env = Environment.linear_track(
            endpoints=((0.0, 0.0), (10.0, 0.0)), bin_size=1.0
        )

        field = np.ones(env.n_bins)
        points = np.array([[5.0, 5.0]])

        with pytest.raises(
            NotImplementedError, match=r"needs a grid of axis-aligned rectangular cells"
        ):
            env.interpolate(field, points, mode="linear")


def test_linear_interpolation_matches_masked_grid_twin(masked_grid_twins):
    """The mask factory gives identical values, including holes and outside points."""
    env, twin = masked_grid_twins
    inactive = np.argwhere(~env.active_mask)[0]
    hole = np.array(
        [
            (edges[index] + edges[index + 1]) / 2
            for edges, index in zip(env.grid_edges, inactive, strict=True)
        ]
    )
    points = np.vstack([env.bin_centers[::10], hole, env.bin_centers.max(axis=0) + 100])
    field = np.arange(env.n_bins, dtype=float)
    expected = env.interpolate(field, points, mode="linear")
    np.testing.assert_array_equal(
        twin.interpolate(field, points, mode="linear"), expected
    )
    assert np.isnan(expected[-2:]).all()


class TestInterpolateValidation:
    """Test input validation."""

    def test_interpolate_field_shape_mismatch(self):
        """Field shape must match n_bins."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # Wrong shape
        field = np.ones(env.n_bins + 5)
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"shape.*n_bins"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_field_not_1d(self):
        """Field must be 1-dimensional."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        # 2D field
        field = np.ones((env.n_bins, 2))
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"1-D array"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"mode.*nearest.*linear"):
            env.interpolate(field, points, mode="cubic")

    def test_interpolate_nan_in_field_reads_nan_only_there(self):
        """A NaN bin value is missing: only points that need it read NaN."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.arange(env.n_bins, dtype=float)
        field[0] = np.nan
        values = env.interpolate(field, env.bin_centers, mode="nearest")

        assert np.isnan(values[0])
        np.testing.assert_array_equal(values[1:], field[1:])

    def test_interpolate_inf_in_field(self):
        """Inf values in field should raise ValueError."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        field[0] = np.inf
        points = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"Inf"):
            env.interpolate(field, points, mode="nearest")

    def test_interpolate_empty_points(self):
        """Empty points array should return empty result."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.ones(env.n_bins)
        points = np.empty((0, 2), dtype=np.float64)

        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (0,)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    @pytest.mark.parametrize("mode", ["nearest", "linear"])
    def test_interpolate_nonfinite_points_read_nan(self, bad, mode):
        """A NaN point is a missing sample and an inf point is outside: both NaN."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = rng.random(env.n_bins)
        points = np.array([[5.0, 5.0], [bad, 3.0], [2.5, 7.5]])
        values = env.interpolate(field, points, mode=mode)

        assert np.isnan(values[1])
        np.testing.assert_array_equal(
            values[[0, 2]], env.interpolate(field, points[[0, 2]], mode=mode)
        )


class TestInterpolateEdgeCases:
    """Edge case tests."""

    def test_interpolate_single_point(self):
        """Single query point."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = rng.random(env.n_bins)
        points = env.bin_centers[:1]

        result = env.interpolate(field, points, mode="nearest")

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_interpolate_constant_field(self):
        """Constant field should interpolate to constant."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = np.full(env.n_bins, 42.0)
        points = rng.uniform(0, 10, size=(20, 2))

        result = env.interpolate(field, points, mode="nearest")

        # Points inside environment should be 42.0
        inside_mask = ~np.isnan(result)
        assert np.all(result[inside_mask] == 42.0)

    def test_interpolate_at_bin_centers_exact(self):
        """Interpolation at bin centers should return exact field values."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, size=(50, 2))
        env = Environment.from_samples(data, bin_size=1.0)

        field = rng.random(env.n_bins)

        # Query at exact bin centers
        result = env.interpolate(field, env.bin_centers, mode="nearest")

        np.testing.assert_array_almost_equal(result, field)


class TestInterpolateMultipleLayouts:
    """Test interpolation across different layout types."""

    def test_interpolate_nearest_on_hexagonal(self):
        """Nearest-neighbor should work on hexagonal layout."""
        # Create sample data for hexagonal grid
        rng = np.random.default_rng(42)
        data_points = rng.uniform(0, 10, size=(100, 2))

        env = Environment.from_samples(
            positions=data_points,
            bin_size=2.0,
            layout="Hexagonal",
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

        rng = np.random.default_rng(42)
        field = rng.random(env.n_bins)
        points = rng.uniform(0, 10, size=(20, 2))

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
        rng = np.random.default_rng(42)
        query_points = rng.uniform(0, 10, size=(50, 2))

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


class TestInterpolateLinearHoledGrid:
    """Regression: linear interpolation works on holed grids (no reshape crash)."""

    def test_interpolate_linear_holed_grid_returns_nan_over_hole(self, holed_grid_env):
        """Linear mode returns finite values over active bins and NaN over a hole."""
        env = holed_grid_env

        # A smooth plane field over the active bins.
        field = (2.0 * env.bin_centers[:, 0] + 3.0 * env.bin_centers[:, 1]).astype(
            np.float64
        )

        edges_x = env.grid_edges[0]
        edges_y = env.grid_edges[1]
        centers_x = (edges_x[:-1] + edges_x[1:]) / 2
        centers_y = (edges_y[:-1] + edges_y[1:]) / 2

        # Point over an active bin, and a point centered on the hole (col/row 5).
        active_point = [float(centers_x[1]), float(centers_y[1])]
        hole_point = [float(centers_x[5]), float(centers_y[5])]
        query_points = np.array([active_point, hole_point])

        # Must not raise (previously: ValueError: cannot reshape ...).
        values = env.interpolate(field, query_points, mode="linear")

        assert np.isfinite(values[0])
        assert np.isnan(values[1])

    def test_interpolate_linear_fully_active_unchanged(self, full_grid_env):
        """On an all-active grid, scatter path matches a direct reshape."""
        env = full_grid_env
        field = (2.0 * env.bin_centers[:, 0] + 3.0 * env.bin_centers[:, 1]).astype(
            np.float64
        )

        rng = np.random.default_rng(7)
        # Query points within the interior so they stay in-bounds.
        lo = env.bin_centers.min(axis=0)
        hi = env.bin_centers.max(axis=0)
        query_points = rng.uniform(lo, hi, size=(50, 2))

        values = env.interpolate(field, query_points, mode="linear")

        finite = np.isfinite(values)
        # The interpolated plane should match the analytic plane where defined.
        expected = 2.0 * query_points[:, 0] + 3.0 * query_points[:, 1]
        np.testing.assert_allclose(values[finite], expected[finite], rtol=1e-6)

    def test_nan_field_value_is_masked_like_a_hole(self, full_grid_env):
        """A NaN bin reads NaN exactly where its interpolation weight is positive."""
        env = full_grid_env
        plane = 2.0 * env.bin_centers[:, 0] + 3.0 * env.bin_centers[:, 1]
        missing = env.n_bins // 2
        field = plane.copy()
        field[missing] = np.nan
        indicator = np.zeros(env.n_bins)
        indicator[missing] = 1.0

        lo, hi = env.bin_centers.min(axis=0), env.bin_centers.max(axis=0)
        query = np.vstack(
            [env.bin_centers, np.random.default_rng(3).uniform(lo, hi, (200, 2))]
        )
        values = env.interpolate(field, query, mode="linear")
        weight = env.interpolate(indicator, query, mode="linear")
        reference = env.interpolate(plane, query, mode="linear")

        needs_missing = weight > 0
        assert needs_missing.any() and (~needs_missing).any()
        assert np.isnan(values[needs_missing]).all()
        np.testing.assert_allclose(
            values[~needs_missing], reference[~needs_missing], rtol=1e-12
        )


def test_linear_interpolation_on_polygon_preserves_inactive_cells():
    """A polygon grid interpolates inside its arena and masks corner cells."""
    from shapely.geometry import Point

    env = Environment.from_polygon(Point(15, 15).buffer(15), bin_size=2.0)
    result = env.interpolate(
        np.ones(env.n_bins), np.array([[15.0, 15.0], [1.0, 1.0]]), mode="linear"
    )
    assert result[0] == pytest.approx(1.0)
    assert np.isnan(result[1])
