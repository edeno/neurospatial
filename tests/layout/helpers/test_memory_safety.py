"""Tests for memory safety checks in layout helpers."""

import warnings

import numpy as np
import pytest

from neurospatial.layout.helpers.utils import (
    check_grid_size_safety,
    estimate_grid_memory,
)


class TestEstimateGridMemory:
    """Tests for estimate_grid_memory function."""

    def test_2d_small_grid(self):
        """Test memory estimation for small 2D grid."""
        grid_shape = (10, 10)
        n_dims = 2
        mem_mb = estimate_grid_memory(grid_shape, n_dims)

        # Should be very small (< 1 MB)
        assert mem_mb < 1.0
        assert mem_mb > 0.0

    def test_2d_medium_grid(self):
        """Test memory estimation for medium 2D grid."""
        grid_shape = (100, 100)
        n_dims = 2
        mem_mb = estimate_grid_memory(grid_shape, n_dims)

        # Should be a few MB (less than 20 MB)
        assert 5.0 < mem_mb < 20.0

    def test_3d_grid(self):
        """Test memory estimation for 3D grid."""
        grid_shape = (50, 50, 50)
        n_dims = 3
        mem_mb = estimate_grid_memory(grid_shape, n_dims)

        # 3D grids use more memory
        assert mem_mb > 100.0

    def test_1d_grid(self):
        """Test memory estimation for 1D grid."""
        grid_shape = (1000,)
        n_dims = 1
        mem_mb = estimate_grid_memory(grid_shape, n_dims)

        # 1D should be small
        assert mem_mb < 5.0

    def test_memory_estimate_increases_with_size(self):
        """Test that memory estimate increases with grid size."""
        n_dims = 2
        mem_10 = estimate_grid_memory((10, 10), n_dims)
        mem_100 = estimate_grid_memory((100, 100), n_dims)
        mem_1000 = estimate_grid_memory((1000, 1000), n_dims)

        # Should increase with size
        assert mem_10 < mem_100 < mem_1000

    def test_memory_estimate_increases_with_dims(self):
        """Test that memory estimate increases with dimensionality."""
        grid_size = 20
        mem_1d = estimate_grid_memory((grid_size,), 1)
        mem_2d = estimate_grid_memory((grid_size, grid_size), 2)
        mem_3d = estimate_grid_memory((grid_size, grid_size, grid_size), 3)

        # Should increase with dimensionality (more bins and edges)
        assert mem_1d < mem_2d < mem_3d


class TestCheckGridSizeSafety:
    """Tests for check_grid_size_safety function."""

    def test_small_grid_no_warning(self):
        """Test that small grids don't trigger warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not raise any warning
            check_grid_size_safety((50, 50), n_dims=2)

    def test_medium_grid_warning(self):
        """Test that medium-sized grids trigger warnings."""
        # Grid that should exceed warn_threshold_mb
        with pytest.warns(ResourceWarning, match="Creating large grid"):
            check_grid_size_safety((500, 500), n_dims=2, warn_threshold_mb=50.0)

    def test_large_grid_still_proceeds(self):
        """Test that very large grids only warn, don't error."""
        # This should warn but NOT raise an error
        with pytest.warns(ResourceWarning, match="Creating large grid"):
            check_grid_size_safety((2000, 2000), n_dims=2, warn_threshold_mb=100.0)

    def test_warning_message_contains_diagnostics(self):
        """Test that warning message includes helpful diagnostics."""
        with pytest.warns(ResourceWarning) as warning_list:
            check_grid_size_safety((500, 500), n_dims=2, warn_threshold_mb=50.0)

        warning_msg = str(warning_list[0].message)
        # Should contain diagnostics
        assert "shape (500, 500)" in warning_msg
        assert "bins" in warning_msg
        assert "Estimated memory usage" in warning_msg
        # Should contain suggestions
        assert "bin_size" in warning_msg
        assert "infer_active_bins" in warning_msg

    def test_custom_thresholds(self):
        """Test using custom warning threshold."""
        # With very low threshold, even small grid should warn
        with pytest.warns(ResourceWarning):
            check_grid_size_safety((100, 100), n_dims=2, warn_threshold_mb=1.0)

        # With very high threshold, large grid should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            check_grid_size_safety((100, 100), n_dims=2, warn_threshold_mb=1000.0)

    def test_disable_warning(self):
        """Test disabling warning with inf threshold."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not warn with inf threshold
            check_grid_size_safety(
                (1000, 1000),
                n_dims=2,
                warn_threshold_mb=float("inf"),
            )


class TestMemorySafetyIntegration:
    """Integration tests with actual Environment creation."""

    def test_small_environment_no_warning(self):
        """Test creating small environment doesn't trigger warnings."""
        from neurospatial import Environment

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise warning
            env = Environment.from_samples(positions, bin_size=5.0)

        assert env.n_bins > 0

    @pytest.mark.slow
    def test_large_environment_warning(self):
        """Test creating large environment triggers warning.

        Marked as slow because creating 2.25M bins takes significant time on CI.
        """
        from neurospatial import Environment

        rng = np.random.default_rng(42)
        # Create positions that will result in large grid
        # 1500 x 1500 grid with bin_size=1.0 -> 2.25M bins ≈ 241MB (> 100MB warn)
        positions = rng.uniform(0, 1500, (1000, 2))

        with pytest.warns(ResourceWarning, match="Creating large grid"):
            # Small bin_size relative to range -> large grid
            env = Environment.from_samples(
                positions, bin_size=1.0, infer_active_bins=False
            )

        assert env.n_bins > 0
