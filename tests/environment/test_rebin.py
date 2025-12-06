"""Tests for Environment.rebin() - grid coarsening operations."""

import numpy as np
import pytest

from neurospatial import Environment


class TestRebinBasic:
    """Basic rebin functionality tests."""

    def test_rebin_factor_2_shape(self):
        """Rebinning by factor 2 reduces grid dimensions by half."""
        # Create grid with explicit range to control shape
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
        )

        # Get original shape
        original_shape = env.layout.grid_shape

        # Rebin by factor 2 (may warn if not evenly divisible)
        coarse = env.rebin(factor=2)

        # Check new shape is original // 2 (truncated if not evenly divisible)
        expected_shape = (original_shape[0] // 2, original_shape[1] // 2)
        assert coarse.layout.grid_shape == expected_shape
        assert coarse.n_bins == expected_shape[0] * expected_shape[1]

    def test_rebin_factor_2_bin_positions(self):
        """Rebinned bin centers are at correct positions."""
        # Create simple 4x4 grid
        data = np.array(
            [
                [0, 0],
                [10, 0],
                [20, 0],
                [30, 0],
                [0, 10],
                [10, 10],
                [20, 10],
                [30, 10],
                [0, 20],
                [10, 20],
                [20, 20],
                [30, 20],
                [0, 30],
                [10, 30],
                [20, 30],
                [30, 30],
            ]
        )
        env = Environment.from_samples(data, bin_size=10.0)
        assert env.layout.grid_shape == (4, 4)

        # Rebin by factor 2 → 2x2 grid
        coarse = env.rebin(factor=2)

        # Should have 2x2 grid with 4 bins
        assert coarse.layout.grid_shape == (2, 2)
        assert coarse.n_bins == 4

        # Bin centers should be within the original data range
        assert np.all(coarse.bin_centers[:, 0] >= 0)
        assert np.all(coarse.bin_centers[:, 0] <= 40)
        assert np.all(coarse.bin_centers[:, 1] >= 0)
        assert np.all(coarse.bin_centers[:, 1] <= 40)

    def test_rebin_coarse_grid_geometry(self):
        """Rebinning creates correct coarse grid geometry."""
        # Create grid with known shape (use dimension_ranges to control)
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
        )

        # Get grid shape
        grid_shape = env.layout.grid_shape

        # Rebin (may warn if not evenly divisible)
        coarse = env.rebin(factor=2)

        # Verify geometry is correct (truncated to nearest divisible size)
        expected_shape = (grid_shape[0] // 2, grid_shape[1] // 2)
        assert coarse.n_bins == expected_shape[0] * expected_shape[1]
        assert coarse.layout.grid_shape == expected_shape

    def test_rebin_creates_valid_environment(self):
        """Rebinning creates a valid Environment instance."""
        # Create 4x4 grid with known values
        data = np.array(
            [
                [0, 0],
                [10, 0],
                [20, 0],
                [30, 0],
                [0, 10],
                [10, 10],
                [20, 10],
                [30, 10],
                [0, 20],
                [10, 20],
                [20, 20],
                [30, 20],
                [0, 30],
                [10, 30],
                [20, 30],
                [30, 30],
            ]
        )
        env = Environment.from_samples(data, bin_size=10.0)

        # Rebin
        coarse = env.rebin(factor=2)

        # Verify shape is correct
        assert coarse.layout.grid_shape == (2, 2)
        assert coarse.n_bins == 4

        # Verify it's a valid environment
        assert isinstance(coarse, Environment)
        assert coarse._is_fitted


class TestRebinFactorVariations:
    """Test different factor specifications."""

    def test_rebin_factor_tuple(self):
        """Factor can be specified as tuple for anisotropic coarsening."""
        rng = np.random.default_rng(42)
        # Create 10x6 grid
        x = rng.random(1000) * 100
        y = rng.random(1000) * 60
        data = np.column_stack([x, y])
        env = Environment.from_samples(data, bin_size=10.0)

        original_shape = env.layout.grid_shape

        # Coarsen by (2, 3)
        coarse = env.rebin(factor=(2, 3))

        # Expected shape: (original[0]//2, original[1]//3)
        expected_shape = (original_shape[0] // 2, original_shape[1] // 3)
        assert coarse.layout.grid_shape == expected_shape

    def test_rebin_factor_3(self):
        """Factor 3 works correctly."""
        # Create grid with explicit range to get 9x9
        env = Environment.from_samples(
            positions=np.array([[0, 0], [90, 90]]),
            bin_size=10.0,
        )

        grid_shape = env.layout.grid_shape

        # Rebin by 3 (may warn if not evenly divisible)
        coarse = env.rebin(factor=3)

        # Should be shape // 3 (truncated if not evenly divisible)
        expected_shape = (grid_shape[0] // 3, grid_shape[1] // 3)
        assert coarse.layout.grid_shape == expected_shape
        assert coarse.n_bins == expected_shape[0] * expected_shape[1]


class TestRebinConnectivity:
    """Test that connectivity is correctly rebuilt."""

    def test_rebin_preserves_connectivity_structure(self):
        """Rebinned grid has correct connectivity."""
        rng = np.random.default_rng(42)
        # Create 4x4 grid
        data = rng.random((200, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        # Rebin to 2x2
        coarse = env.rebin(factor=2)

        # 2x2 grid should have 4 nodes
        assert coarse.connectivity.number_of_nodes() == 4

        # In a 2x2 grid with diagonal connectivity (default for from_samples):
        # Each corner has 3 neighbors (2 edges + 1 diagonal)
        degrees = [coarse.connectivity.degree(n) for n in coarse.connectivity.nodes()]

        # 2x2 grid with diagonal: all 4 corners have 3 neighbors each
        assert sorted(degrees) == [3, 3, 3, 3]

    def test_rebin_edge_attributes(self):
        """Rebinned connectivity has required edge attributes."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        coarse = env.rebin(factor=2)

        # Check edges have required attributes
        for _u, _v, attrs in coarse.connectivity.edges(data=True):
            assert "distance" in attrs
            assert "vector" in attrs
            assert "edge_id" in attrs
            assert isinstance(attrs["distance"], (int, float))
            assert isinstance(attrs["vector"], tuple)


class TestRebinValidation:
    """Input validation tests."""

    def test_rebin_non_grid_layout_raises(self):
        """Rebinning non-grid layout raises NotImplementedError."""
        # Create a polygon layout (not a regular grid)
        from shapely.geometry import box

        polygon = box(0, 0, 50, 50)
        env = Environment.from_polygon(polygon, bin_size=10.0)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match=r"only.*RegularGridLayout"):
            env.rebin(factor=2)

    def test_rebin_invalid_factor_raises(self):
        """Invalid factor values raise ValueError."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)

        # Factor must be positive
        with pytest.raises(ValueError, match=r"positive"):
            env.rebin(factor=0)

        with pytest.raises(ValueError, match=r"positive"):
            env.rebin(factor=-2)

    def test_rebin_factor_too_large_raises(self):
        """Factor larger than grid dimension raises ValueError."""
        rng = np.random.default_rng(42)
        # Create 4x4 grid
        data = rng.random((100, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        # Factor 10 is too large for 4x4 grid
        with pytest.raises(ValueError, match=r"too large.*grid shape"):
            env.rebin(factor=10)

    def test_rebin_non_divisible_warns(self):
        """Non-divisible factor warns and truncates."""
        # Create grid with known non-divisible shape
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
        )

        grid_shape = env.layout.grid_shape

        # Only run if shape is not divisible by 3
        if grid_shape[0] % 3 == 0 and grid_shape[1] % 3 == 0:
            pytest.skip(f"Grid shape {grid_shape} is divisible by 3")

        # Factor 3 doesn't divide evenly
        with pytest.warns(UserWarning, match=r"not evenly divisible"):
            coarse = env.rebin(factor=3)

        # Should truncate
        expected_shape = (grid_shape[0] // 3, grid_shape[1] // 3)
        assert coarse.layout.grid_shape == expected_shape


class TestRebinEdgeCases:
    """Edge cases and boundary conditions."""

    def test_rebin_factor_1_returns_copy(self):
        """Factor 1 returns a new environment with full grid."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)

        coarse = env.rebin(factor=1)

        # Should have same grid shape
        assert coarse.layout.grid_shape == env.layout.grid_shape

        # May have more bins if original had inactive bins
        # (rebin creates a full grid with all bins active)
        assert coarse.n_bins >= env.n_bins

        # But should be different object
        assert coarse is not env

    def test_rebin_2d_vs_3d(self):
        """Rebinning works for 2D and 3D grids."""
        rng = np.random.default_rng(42)
        # 2D grid
        data_2d = rng.random((200, 2)) * 40
        env_2d = Environment.from_samples(data_2d, bin_size=10.0)
        coarse_2d = env_2d.rebin(factor=2)
        assert coarse_2d.n_dims == 2

        # 3D grid
        data_3d = rng.random((500, 3)) * 40
        env_3d = Environment.from_samples(data_3d, bin_size=10.0)
        coarse_3d = env_3d.rebin(factor=2)
        assert coarse_3d.n_dims == 3

    def test_rebin_preserves_units_and_frame(self):
        """Rebinning preserves units and frame attributes."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)
        env.units = "cm"
        env.frame = "session1"

        coarse = env.rebin(factor=2)

        assert coarse.units == "cm"
        assert coarse.frame == "session1"


class TestRebinIntegration:
    """Integration tests with other operations."""

    def test_rebin_then_smooth(self):
        """Rebinning then smoothing works correctly."""
        rng = np.random.default_rng(42)
        data = rng.random((500, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)

        # Rebin first
        coarse = env.rebin(factor=2)

        # Then smooth a field
        field = rng.random(coarse.n_bins)
        smoothed = coarse.smooth(field, bandwidth=5.0)

        assert smoothed.shape == (coarse.n_bins,)
        assert not np.any(np.isnan(smoothed))

    def test_rebin_bin_centers_mappable(self):
        """Can map original bin centers to rebinned environment."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        coarse = env.rebin(factor=2)

        # All original bin centers should map to some bin in coarse
        from neurospatial.ops import map_points_to_bins

        indices = map_points_to_bins(env.bin_centers, coarse)

        # Should all be valid (not -1)
        assert np.all(indices >= 0)
        assert np.all(indices < coarse.n_bins)
