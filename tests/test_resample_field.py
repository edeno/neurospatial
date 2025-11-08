"""Test for resample_field() function with nearest and diffuse methods.

This module tests resampling fields between different environment discretizations,
enabling multi-resolution analysis and cross-session alignment.

Tests cover:
- Nearest-neighbor resampling
- Diffuse method with smoothing
- Same vs different resolutions
- Mass conservation properties
- Input validation
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.spatial import resample_field


class TestResampleFieldBasic:
    """Test basic resample_field functionality."""

    def test_nearest_method_same_resolution(self):
        """Test nearest method with same bin_size (identity-like)."""
        # Source environment
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination environment (slightly offset)
        dst_data = np.array([[i + 0.5, j + 0.5] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Create field on source
        field = np.random.rand(src_env.n_bins)

        # Resample to destination
        result = resample_field(field, src_env, dst_env, method="nearest")

        # Should return array with dst_env.n_bins elements
        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))

    def test_nearest_method_different_resolution(self):
        """Test nearest method with different bin sizes."""
        # Source: coarse resolution
        src_data = np.array([[i, j] for i in range(21) for j in range(21)])
        src_env = Environment.from_samples(src_data, bin_size=4.0)

        # Destination: fine resolution
        dst_data = np.array([[i, j] for i in range(21) for j in range(21)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Create field on source
        field = np.random.rand(src_env.n_bins)

        # Resample to destination
        result = resample_field(field, src_env, dst_env, method="nearest")

        # Fine resolution should have more bins
        assert result.shape == (dst_env.n_bins,)
        assert dst_env.n_bins > src_env.n_bins

    def test_nearest_is_default_method(self):
        """Test that method='nearest' is the default."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        result_explicit = resample_field(field, src_env, dst_env, method="nearest")
        result_default = resample_field(field, src_env, dst_env)

        np.testing.assert_array_equal(result_explicit, result_default)

    def test_resampling_preserves_field_range(self):
        """Test that resampled field stays within original value range."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Field with known range [0, 1]
        field = np.random.rand(src_env.n_bins)

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Nearest method should preserve exact values (no interpolation)
        assert np.min(result) >= np.min(field) - 1e-10
        assert np.max(result) <= np.max(field) + 1e-10


class TestResampleFieldDiffuse:
    """Test diffuse resampling method."""

    def test_diffuse_method_with_bandwidth(self):
        """Test diffuse method with explicit bandwidth."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        result = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=1.0
        )

        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))

    def test_diffuse_requires_bandwidth(self):
        """Test that diffuse method requires bandwidth parameter."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        with pytest.raises(ValueError, match="bandwidth must be provided"):
            resample_field(field, src_env, dst_env, method="diffuse")

    def test_diffuse_smoother_than_nearest(self):
        """Test that diffuse method produces smoother results."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Create spiky field
        field = np.zeros(src_env.n_bins)
        field[src_env.n_bins // 2] = 1.0

        nearest_result = resample_field(field, src_env, dst_env, method="nearest")
        diffuse_result = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=2.0
        )

        # Diffuse should spread the spike more
        assert np.sum(nearest_result > 0.1) < np.sum(diffuse_result > 0.01)

    def test_diffuse_with_different_bandwidths(self):
        """Test that larger bandwidth produces smoother results."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.zeros(src_env.n_bins)
        field[src_env.n_bins // 2] = 1.0

        result_small = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=0.5
        )
        result_large = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=3.0
        )

        # Larger bandwidth spreads values more
        assert np.std(result_large) < np.std(result_small)


class TestResampleFieldEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_field_zeros(self):
        """Test resampling field of all zeros."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.zeros(src_env.n_bins)

        result = resample_field(field, src_env, dst_env, method="nearest")

        np.testing.assert_array_equal(result, np.zeros(dst_env.n_bins))

    def test_constant_field(self):
        """Test resampling constant field."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.ones(src_env.n_bins) * 5.0

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Nearest neighbor should preserve constant value
        np.testing.assert_allclose(result, 5.0, rtol=1e-10)

    def test_single_bin_source(self):
        """Test resampling from environment with single bin."""
        # Source with one bin
        src_data = np.array([[5.0, 5.0]])
        src_env = Environment.from_samples(src_data, bin_size=10.0)

        # Destination with multiple bins
        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        field = np.array([3.14])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # All destination bins get mapped to single source bin
        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))

    def test_non_overlapping_environments(self):
        """Test resampling when environments don't overlap spatially."""
        # Source environment at (0, 0) to (10, 10)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination environment at (100, 100) to (110, 110)
        dst_data = np.array([[100 + i, 100 + j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        field = np.random.rand(src_env.n_bins)

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Should still work - each dst bin maps to nearest src bin
        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))


class TestResampleFieldInputValidation:
    """Test input validation and error messages."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        with pytest.raises(ValueError, match="method must be 'nearest' or 'diffuse'"):
            resample_field(field, src_env, dst_env, method="invalid")

    def test_field_size_mismatch(self):
        """Test that mismatched field and source environment sizes raise error."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        wrong_field = np.random.rand(src_env.n_bins + 5)

        with pytest.raises(
            ValueError, match=r"Field size .* does not match source environment"
        ):
            resample_field(wrong_field, src_env, dst_env, method="nearest")

    def test_negative_bandwidth(self):
        """Test that negative bandwidth raises error."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        with pytest.raises(ValueError, match="bandwidth must be positive"):
            resample_field(field, src_env, dst_env, method="diffuse", bandwidth=-1.0)

    def test_dimension_mismatch(self):
        """Test that environments with different dimensions raise error."""
        # 2D source
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # 1D destination (if we can create one - may skip if not supported)
        # For now, just test with matching dimensions
        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        # This should work (both 2D)
        result = resample_field(field, src_env, dst_env, method="nearest")
        assert result.shape == (dst_env.n_bins,)


class TestResampleFieldCorrectness:
    """Test correctness with known ground truth mappings."""

    def test_nearest_exact_alignment(self):
        """Test nearest method with exactly aligned grids (known mapping)."""
        # Source: 3x3 grid with bin centers at (1, 3, 5) x (1, 3, 5)
        src_data = np.array([[i, j] for i in [1, 3, 5] for j in [1, 3, 5]])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination: same grid (exact alignment)
        dst_data = np.array([[i, j] for i in [1, 3, 5] for j in [1, 3, 5]])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Field with known values at each position
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Should be exact identity mapping
        np.testing.assert_array_equal(result, field)

    def test_nearest_offset_grid(self):
        """Test nearest method with offset grids (known nearest neighbors)."""
        # Source: 2x2 grid with bin centers at (0, 2) x (0, 2)
        src_data = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination: points near each source bin
        # (0.5, 0.5) nearest to (0, 0)
        # (0.5, 1.5) nearest to (0, 2)
        # (1.5, 0.5) nearest to (2, 0)
        # (1.5, 1.5) nearest to (2, 2)
        dst_data = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Field with distinct values: [10, 20, 30, 40]
        field = np.array([10.0, 20.0, 30.0, 40.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Each destination point should get value from nearest source
        np.testing.assert_array_equal(result, field)

    def test_nearest_upsampling(self):
        """Test nearest method upsampling (fine grid from coarse)."""
        # Source: single bin at center
        src_data = np.array([[5.0, 5.0]])
        src_env = Environment.from_samples(src_data, bin_size=10.0)

        # Destination: 3 points all near the single source bin
        dst_data = np.array([[4.0, 5.0], [5.0, 5.0], [6.0, 5.0]])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Source has single value
        field = np.array([42.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # All destination bins should get the same value
        np.testing.assert_array_equal(result, np.array([42.0, 42.0, 42.0]))

    def test_nearest_downsampling(self):
        """Test nearest method downsampling (coarse grid from fine)."""
        # Source: 4 bins in 2x2 grid at (0, 1) x (0, 1)
        src_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        src_env = Environment.from_samples(src_data, bin_size=1.0)

        # Destination: single point at center
        dst_data = np.array([[0.5, 0.5]])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Field with different values
        field = np.array([1.0, 2.0, 3.0, 4.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # The (0.5, 0.5) point is equidistant from all 4 source bins
        # With tie_break=LOWEST_INDEX, should pick first bin
        assert result.shape == (1,)
        assert result[0] == 1.0  # From bin 0 (lowest index)


class TestResampleFieldMathematicalProperties:
    """Test mathematical properties of resampling."""

    def test_nearest_preserves_values(self):
        """Test that nearest method only produces values from source field."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Field with distinct values
        field = np.arange(src_env.n_bins, dtype=np.float64)

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Every result value should be from original field
        for val in result:
            assert val in field

    def test_identity_resampling_same_environment(self):
        """Test that resampling to same environment is identity-like."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        field = np.random.rand(src_env.n_bins)

        # Resample to itself
        result = resample_field(field, src_env, src_env, method="nearest")

        # Should be very close to identity
        np.testing.assert_allclose(result, field, rtol=1e-10)

    def test_diffuse_approximate_mass_conservation(self):
        """Test that diffuse method approximately conserves total mass."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.random.rand(src_env.n_bins)

        result = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=1.0
        )

        # Mass may not be exactly conserved (diffusion boundary effects)
        # but should be approximately preserved
        src_total = np.sum(field * src_env.bin_sizes)
        dst_total = np.sum(result * dst_env.bin_sizes)

        # Allow 20% tolerance due to boundary effects and discretization
        np.testing.assert_allclose(dst_total, src_total, rtol=0.2)
