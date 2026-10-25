"""Tests for memory safety checks in layout helpers."""

import warnings

import numpy as np
import pytest

from neurospatial.layout.helpers.utils import (
    check_grid_size_safety,
    estimate_grid_memory,
)


class TestEstimateGridMemory:
    """The estimate is a heuristic: only its ordering is tested, not its values."""

    def test_memory_estimate_increases_with_size(self):
        """Test that memory estimate increases with grid size."""
        mem_10 = estimate_grid_memory((10, 10))
        mem_100 = estimate_grid_memory((100, 100))
        mem_1000 = estimate_grid_memory((1000, 1000))

        # Should increase with size
        assert mem_10 < mem_100 < mem_1000


class TestCheckGridSizeSafety:
    """Tests for check_grid_size_safety function."""

    def test_small_grid_no_warning(self):
        """Test that small grids don't trigger warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not raise any warning
            check_grid_size_safety((50, 50))

    def test_medium_grid_warning(self):
        """Test that medium-sized grids trigger warnings."""
        # Grid that should exceed warn_threshold_mb
        with pytest.warns(UserWarning, match="Creating large grid"):
            check_grid_size_safety((500, 500), warn_threshold_mb=50.0)

    def test_large_grid_still_proceeds(self):
        """Test that very large grids only warn, don't error."""
        # This should warn but NOT raise an error
        with pytest.warns(UserWarning, match="Creating large grid"):
            check_grid_size_safety((2000, 2000), warn_threshold_mb=100.0)

    def test_warning_message_contains_diagnostics(self):
        """Test that warning message includes helpful diagnostics."""
        with pytest.warns(UserWarning) as warning_list:
            check_grid_size_safety((500, 500), warn_threshold_mb=50.0)

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
        with pytest.warns(UserWarning):
            check_grid_size_safety((100, 100), warn_threshold_mb=1.0)

        # With very high threshold, large grid should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            check_grid_size_safety((100, 100), warn_threshold_mb=1000.0)

    def test_disable_warning(self):
        """Test disabling warning with inf threshold."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not warn with inf threshold
            check_grid_size_safety(
                (1000, 1000),
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

        with pytest.warns(UserWarning, match="Creating large grid"):
            # Small bin_size relative to range -> large grid
            env = Environment.from_samples(
                positions, bin_size=1.0, infer_active_bins=False
            )

        assert env.n_bins > 0
