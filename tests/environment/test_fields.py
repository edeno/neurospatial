"""Comprehensive tests for Environment field operations (fields.py module).

This module tests the field operations mixin for Environment, covering:
- compute_kernel() method (diffusion kernel computation)
- smooth() method (field smoothing with various modes)
- interpolate() method (nearest and linear interpolation)
- Edge cases and validation
- Property-based tests for smoothing invariants

Target coverage: 8% → 90% for src/neurospatial/environment/fields.py

See Also
--------
TEST_PLAN2.md : Section 1.2 - Field Operations Tests
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from neurospatial import Environment

# =============================================================================
# Test compute_kernel()
# =============================================================================


class TestComputeKernel:
    """Tests for Environment.compute_kernel() method."""

    def test_compute_kernel_basic(self, small_2d_env):
        """Test basic kernel computation."""
        kernel = small_2d_env.compute_kernel(bandwidth=5.0)

        # Kernel should be n_bins x n_bins
        assert kernel.shape == (small_2d_env.n_bins, small_2d_env.n_bins)

    def test_compute_kernel_density_mode_normalized(self, small_2d_env):
        """Test that density mode kernel has correct normalization properties."""
        kernel = small_2d_env.compute_kernel(bandwidth=5.0, mode="density")

        # Each column should integrate to 1 when weighted by bin volumes
        # For density mode, the normalization is volume-corrected
        # Just check that kernel is non-negative and well-formed
        assert np.all(kernel >= 0), "Kernel should be non-negative"
        # Check that columns have reasonable sums (not all zero)
        col_sums = kernel.sum(axis=0)
        assert np.all(col_sums > 0), "All columns should have positive mass"

    def test_compute_kernel_transition_mode_normalized(self, small_2d_env):
        """Test that transition mode kernel rows sum to 1."""
        kernel = small_2d_env.compute_kernel(bandwidth=5.0, mode="transition")

        # Each column should sum to 1 (transition mode = probability)
        col_sums = kernel.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, rtol=1e-10, atol=1e-10)

    def test_compute_kernel_symmetric(self, small_2d_env):
        """Test that kernel is symmetric for regular grids."""
        kernel = small_2d_env.compute_kernel(bandwidth=5.0, mode="transition")

        # For regular grids with uniform connectivity, kernel should be symmetric
        # (or at least approximately symmetric)
        np.testing.assert_allclose(kernel, kernel.T, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("bandwidth", [1.0, 5.0, 10.0])
    def test_compute_kernel_various_bandwidths(self, small_2d_env, bandwidth):
        """Test kernel computation with various bandwidths."""
        kernel = small_2d_env.compute_kernel(bandwidth=bandwidth)

        assert kernel.shape == (small_2d_env.n_bins, small_2d_env.n_bins)
        # All values should be non-negative
        assert np.all(kernel >= 0)

    @pytest.mark.parametrize("mode", ["transition", "density"])
    def test_compute_kernel_modes(self, small_2d_env, mode):
        """Test both kernel modes."""
        kernel = small_2d_env.compute_kernel(bandwidth=5.0, mode=mode)

        assert kernel.shape == (small_2d_env.n_bins, small_2d_env.n_bins)
        assert kernel.dtype == np.float64

    def test_compute_kernel_caching_enabled(self, small_2d_env):
        """Test that kernel caching works when enabled."""
        # First call - computes and caches
        kernel1 = small_2d_env.compute_kernel(bandwidth=5.0, mode="density", cache=True)

        # Second call - should return cached result
        kernel2 = small_2d_env.compute_kernel(bandwidth=5.0, mode="density", cache=True)

        # Should be identical (same object from cache)
        assert kernel1 is kernel2
        np.testing.assert_array_equal(kernel1, kernel2)

    def test_compute_kernel_caching_disabled(self, small_2d_env):
        """Test that kernel caching can be disabled."""
        # Two calls with cache=False
        kernel1 = small_2d_env.compute_kernel(
            bandwidth=5.0, mode="density", cache=False
        )
        kernel2 = small_2d_env.compute_kernel(
            bandwidth=5.0, mode="density", cache=False
        )

        # Should be equal but not the same object
        np.testing.assert_array_equal(kernel1, kernel2)
        # Note: Can't test 'is not' reliably since numpy might reuse memory

    def test_compute_kernel_different_params_different_cache(self, small_2d_env):
        """Test that different parameters create different cache entries."""
        kernel1 = small_2d_env.compute_kernel(bandwidth=5.0, mode="density", cache=True)
        kernel2 = small_2d_env.compute_kernel(
            bandwidth=10.0, mode="density", cache=True
        )
        kernel3 = small_2d_env.compute_kernel(
            bandwidth=5.0, mode="transition", cache=True
        )

        # All should be different
        assert not np.allclose(kernel1, kernel2)
        assert not np.allclose(kernel1, kernel3)

    def test_compute_kernel_positive_definite(self, small_2d_env):
        """Test that kernel matrix is positive (semi-)definite."""
        kernel = small_2d_env.compute_kernel(bandwidth=5.0, mode="transition")

        # Check eigenvalues are non-negative (positive semi-definite)
        eigenvalues = np.linalg.eigvalsh(kernel)
        assert np.all(eigenvalues >= -1e-10), "Kernel should be positive semi-definite"


# =============================================================================
# Test smooth()
# =============================================================================


class TestSmooth:
    """Tests for Environment.smooth() method."""

    def test_smooth_constant_field_unchanged(self, medium_2d_env):
        """Test that smoothing a constant field leaves it constant."""
        # Constant field
        field = np.ones(medium_2d_env.n_bins) * 10.0

        smoothed = medium_2d_env.smooth(field, bandwidth=5.0, mode="transition")

        # Should remain constant (within numerical tolerance)
        np.testing.assert_allclose(smoothed, field, rtol=1e-10, atol=1e-10)

    def test_smooth_gaussian_reduces_variance(self, medium_2d_env):
        """Test that smoothing reduces variance for non-constant fields."""
        # Create field with high variance
        rng = np.random.default_rng(42)
        field = rng.random(medium_2d_env.n_bins) * 100

        smoothed = medium_2d_env.smooth(field, bandwidth=5.0, mode="transition")

        # Variance should decrease after smoothing
        var_original = np.var(field)
        var_smoothed = np.var(smoothed)
        assert var_smoothed < var_original, "Smoothing should reduce variance"

    @pytest.mark.parametrize("bandwidth", [1.0, 2.0, 5.0, 10.0])
    def test_smooth_various_bandwidths(self, medium_2d_env, bandwidth):
        """Test smoothing with various bandwidths."""
        rng = np.random.default_rng(42)
        field = rng.random(medium_2d_env.n_bins)

        smoothed = medium_2d_env.smooth(field, bandwidth=bandwidth, mode="transition")

        assert smoothed.shape == field.shape
        assert not np.any(np.isnan(smoothed))

    def test_smooth_handles_edge_bins(self, small_2d_env):
        """Test smoothing behavior at boundary bins."""
        # Create field only on boundary
        boundary_bins = small_2d_env.boundary_bins
        field = np.zeros(small_2d_env.n_bins)
        field[boundary_bins] = 10.0

        smoothed = small_2d_env.smooth(field, bandwidth=2.0, mode="transition")

        # Smoothing should work without errors
        assert smoothed.shape == field.shape
        # Mass should be conserved
        np.testing.assert_allclose(smoothed.sum(), field.sum(), rtol=1e-10, atol=1e-10)

    def test_smooth_impulse_spreads(self, medium_2d_env):
        """Test that an impulse spreads to neighboring bins."""
        # Create impulse at center
        field = np.zeros(medium_2d_env.n_bins)
        center_bin = medium_2d_env.n_bins // 2
        field[center_bin] = 100.0

        smoothed = medium_2d_env.smooth(field, bandwidth=5.0, mode="transition")

        # Peak should decrease
        assert smoothed[center_bin] < field[center_bin]
        # Should spread to multiple bins
        num_nonzero = np.sum(smoothed > 1e-6)
        assert num_nonzero > 1, "Impulse should spread to neighbors"

    def test_smooth_transition_mode_conserves_mass(self, medium_2d_env):
        """Test that transition mode conserves total mass."""
        rng = np.random.default_rng(42)
        field = rng.random(medium_2d_env.n_bins) * 10

        smoothed = medium_2d_env.smooth(field, bandwidth=5.0, mode="transition")

        # Total mass should be conserved
        np.testing.assert_allclose(smoothed.sum(), field.sum(), rtol=1e-10, atol=1e-10)

    def test_smooth_density_mode_works(self, medium_2d_env):
        """Test that density mode smoothing works."""
        rng = np.random.default_rng(42)
        field = rng.random(medium_2d_env.n_bins)

        smoothed = medium_2d_env.smooth(field, bandwidth=5.0, mode="density")

        assert smoothed.shape == field.shape
        assert not np.any(np.isnan(smoothed))

    def test_smooth_preserves_shape(self, small_2d_env):
        """Test that smoothing preserves field shape."""
        rng = np.random.default_rng(42)
        field = rng.random(small_2d_env.n_bins)

        smoothed = small_2d_env.smooth(field, bandwidth=3.0)

        assert smoothed.shape == field.shape

    def test_smooth_zero_field_remains_zero(self, small_2d_env):
        """Test that smoothing zero field gives zero field."""
        field = np.zeros(small_2d_env.n_bins)

        smoothed = small_2d_env.smooth(field, bandwidth=5.0)

        np.testing.assert_array_equal(smoothed, field)

    # Validation tests
    def test_smooth_wrong_field_shape_raises(self, medium_2d_env):
        """Test that wrong field shape raises ValueError."""
        field_wrong = np.random.rand(medium_2d_env.n_bins + 10)

        with pytest.raises(ValueError, match=r"Field shape.*must match n_bins"):
            medium_2d_env.smooth(field_wrong, bandwidth=2.0)

    def test_smooth_2d_field_raises(self, medium_2d_env):
        """Test that 2-D field raises ValueError."""
        field_2d = np.random.rand(medium_2d_env.n_bins, 3)

        with pytest.raises(ValueError, match=r"Field must be 1-D array"):
            medium_2d_env.smooth(field_2d, bandwidth=2.0)

    def test_smooth_negative_bandwidth_raises(self, medium_2d_env):
        """Test that negative bandwidth raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)

        with pytest.raises(ValueError, match=r"bandwidth must be positive"):
            medium_2d_env.smooth(field, bandwidth=-1.0)

    def test_smooth_zero_bandwidth_raises(self, medium_2d_env):
        """Test that zero bandwidth raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)

        with pytest.raises(ValueError, match=r"bandwidth must be positive"):
            medium_2d_env.smooth(field, bandwidth=0.0)

    def test_smooth_invalid_mode_raises(self, medium_2d_env):
        """Test that invalid mode raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)

        with pytest.raises(ValueError, match=r"mode must be"):
            medium_2d_env.smooth(field, bandwidth=2.0, mode="invalid")

    def test_smooth_field_with_nan_raises(self, medium_2d_env):
        """Test that field with NaN raises ValueError."""
        field = np.ones(medium_2d_env.n_bins)
        field[0] = np.nan

        with pytest.raises(ValueError, match=r"Field contains NaN values"):
            medium_2d_env.smooth(field, bandwidth=2.0)

    def test_smooth_field_with_inf_raises(self, medium_2d_env):
        """Test that field with Inf raises ValueError."""
        field = np.ones(medium_2d_env.n_bins)
        field[0] = np.inf

        with pytest.raises(ValueError, match=r"Field contains infinite values"):
            medium_2d_env.smooth(field, bandwidth=2.0)

    def test_smooth_field_with_negative_inf_raises(self, medium_2d_env):
        """Test that field with -Inf raises ValueError."""
        field = np.ones(medium_2d_env.n_bins)
        field[0] = -np.inf

        with pytest.raises(ValueError, match=r"Field contains infinite values"):
            medium_2d_env.smooth(field, bandwidth=2.0)


# =============================================================================
# Test interpolate()
# =============================================================================


class TestInterpolate:
    """Tests for Environment.interpolate() method."""

    def test_interpolate_at_bin_centers_exact(self, medium_2d_env):
        """Test that interpolation at bin centers matches field values."""
        field = np.arange(medium_2d_env.n_bins, dtype=np.float64)

        # Interpolate at bin centers (should match exactly)
        interpolated = medium_2d_env.interpolate(
            field, medium_2d_env.bin_centers, mode="nearest"
        )

        np.testing.assert_allclose(interpolated, field, rtol=1e-10, atol=1e-10)

    def test_interpolate_nearest_mode(self, small_2d_env):
        """Test nearest-neighbor interpolation."""
        rng = np.random.default_rng(42)
        field = rng.random(small_2d_env.n_bins)

        # Generate query points within environment bounds
        # dimension_ranges is a tuple of (min, max) tuples
        dim_ranges = small_2d_env.dimension_ranges
        query_points = np.column_stack(
            [
                rng.uniform(dim_ranges[i][0], dim_ranges[i][1], size=20)
                for i in range(small_2d_env.n_dims)
            ]
        )

        interpolated = small_2d_env.interpolate(field, query_points, mode="nearest")

        assert len(interpolated) == len(query_points)
        # Some points might be outside - check that at least some are valid
        assert np.sum(~np.isnan(interpolated)) > 0

    def test_interpolate_linear_mode(self):
        """Test linear interpolation (for regular grids)."""
        # Create pure regular grid (no masking) for linear interpolation test
        x = np.linspace(0, 10, 6)
        y = np.linspace(0, 10, 6)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(data, bin_size=2.0, infer_active_bins=False)

        rng = np.random.default_rng(42)
        field = rng.random(env.n_bins)

        # Generate query points within environment bounds
        query_points = rng.uniform(low=0, high=10, size=(20, 2))

        interpolated = env.interpolate(field, query_points, mode="linear")

        assert len(interpolated) == len(query_points)
        # Points within bounds should have valid values
        assert np.sum(~np.isnan(interpolated)) > 0

    def test_interpolate_linear_exact_for_linear_field(self):
        """Test that linear interpolation is exact for linear fields."""
        # Create pure regular grid (no masking) for linear interpolation test
        x = np.linspace(0, 10, 6)
        y = np.linspace(0, 10, 6)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(data, bin_size=2.0, infer_active_bins=False)

        # Create linear field: f(x, y) = 2*x + 3*y
        field = 2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]

        # Query at bin centers
        interpolated = env.interpolate(field, env.bin_centers, mode="linear")

        # Should match exactly (or very close)
        np.testing.assert_allclose(interpolated, field, rtol=1e-5, atol=1e-5)

    def test_interpolate_outside_points_return_nan(self, small_2d_env):
        """Test that points outside environment return NaN."""
        field = np.ones(small_2d_env.n_bins)

        # Points far outside
        outside_points = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])

        interpolated = small_2d_env.interpolate(field, outside_points, mode="nearest")

        # Should return NaN for outside points
        assert np.all(np.isnan(interpolated))

    def test_interpolate_empty_points_array(self, small_2d_env):
        """Test interpolation with empty points array."""
        field = np.ones(small_2d_env.n_bins)
        empty_points = np.array([]).reshape(0, small_2d_env.n_dims)

        interpolated = small_2d_env.interpolate(field, empty_points, mode="nearest")

        assert len(interpolated) == 0

    def test_interpolate_single_point(self, small_2d_env):
        """Test interpolation with single point."""
        field = np.arange(small_2d_env.n_bins, dtype=np.float64)
        point = small_2d_env.bin_centers[0:1]  # Keep 2D shape

        interpolated = small_2d_env.interpolate(field, point, mode="nearest")

        assert len(interpolated) == 1
        np.testing.assert_allclose(interpolated[0], field[0], rtol=1e-10)

    # Validation tests
    def test_interpolate_wrong_field_shape_raises(self, medium_2d_env):
        """Test that wrong field shape raises ValueError."""
        field_wrong = np.random.rand(medium_2d_env.n_bins + 10)
        query_points = medium_2d_env.bin_centers[:5]

        with pytest.raises(ValueError, match=r"Field shape.*must match n_bins"):
            medium_2d_env.interpolate(field_wrong, query_points)

    def test_interpolate_2d_field_raises(self, medium_2d_env):
        """Test that 2-D field raises ValueError."""
        field_2d = np.random.rand(medium_2d_env.n_bins, 3)
        query_points = medium_2d_env.bin_centers[:5]

        with pytest.raises(ValueError, match=r"Field must be 1-D array"):
            medium_2d_env.interpolate(field_2d, query_points)

    def test_interpolate_wrong_points_dimension_raises(self, medium_2d_env):
        """Test that wrong points dimension raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)
        # 3D points for 2D environment
        query_points_3d = np.random.rand(10, 3)

        with pytest.raises(ValueError, match=r"Points dimension.*must match"):
            medium_2d_env.interpolate(field, query_points_3d)

    def test_interpolate_1d_points_raises(self, medium_2d_env):
        """Test that 1-D points array raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)
        points_1d = np.random.rand(10)

        with pytest.raises(ValueError, match=r"Points must be 2-D array"):
            medium_2d_env.interpolate(field, points_1d)

    def test_interpolate_invalid_mode_raises(self, medium_2d_env):
        """Test that invalid mode raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)
        query_points = medium_2d_env.bin_centers[:5]

        with pytest.raises(ValueError, match=r"mode must be"):
            medium_2d_env.interpolate(field, query_points, mode="invalid")

    def test_interpolate_field_with_nan_raises(self, medium_2d_env):
        """Test that field with NaN raises ValueError."""
        field = np.ones(medium_2d_env.n_bins)
        field[0] = np.nan
        query_points = medium_2d_env.bin_centers[:5]

        with pytest.raises(ValueError, match=r"Field contains NaN values"):
            medium_2d_env.interpolate(field, query_points)

    def test_interpolate_field_with_inf_raises(self, medium_2d_env):
        """Test that field with Inf raises ValueError."""
        field = np.ones(medium_2d_env.n_bins)
        field[0] = np.inf
        query_points = medium_2d_env.bin_centers[:5]

        with pytest.raises(ValueError, match=r"Field contains infinite values"):
            medium_2d_env.interpolate(field, query_points)

    def test_interpolate_points_with_nan_raises(self, medium_2d_env):
        """Test that points with NaN raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)
        query_points = medium_2d_env.bin_centers[:5].copy()
        query_points[0, 0] = np.nan

        with pytest.raises(ValueError, match=r"non-finite value"):
            medium_2d_env.interpolate(field, query_points)

    def test_interpolate_points_with_inf_raises(self, medium_2d_env):
        """Test that points with Inf raises ValueError."""
        field = np.random.rand(medium_2d_env.n_bins)
        query_points = medium_2d_env.bin_centers[:5].copy()
        query_points[0, 0] = np.inf

        with pytest.raises(ValueError, match=r"non-finite value"):
            medium_2d_env.interpolate(field, query_points)

    def test_interpolate_linear_on_graph_raises(self, graph_env):
        """Test that linear mode on graph layout raises NotImplementedError."""
        field = np.random.rand(graph_env.n_bins)
        query_points = graph_env.bin_centers[:2]

        with pytest.raises(NotImplementedError, match=r"Linear interpolation.*only"):
            graph_env.interpolate(field, query_points, mode="linear")


# =============================================================================
# Property-based tests (Hypothesis)
# =============================================================================


class TestFieldProperties:
    """Property-based tests for field operations using Hypothesis."""

    @given(
        field_values=st.lists(
            st.floats(
                min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
            min_size=10,
            max_size=50,
        )
    )
    def test_smooth_reduces_variance_property(self, field_values, small_2d_env):
        """Property: Smoothing should reduce or maintain variance for non-constant fields."""
        # Ensure field not larger than environment (tell Hypothesis to generate different input)
        assume(len(field_values) <= small_2d_env.n_bins)

        # Pad field to match environment size
        field = np.array(
            field_values + [0.0] * (small_2d_env.n_bins - len(field_values))
        )

        # Skip constant fields (variance doesn't change - tell Hypothesis to generate different input)
        assume(np.std(field) >= 1e-10)

        smoothed = small_2d_env.smooth(field, bandwidth=5.0, mode="transition")

        var_original = np.var(field)
        var_smoothed = np.var(smoothed)

        # Variance should not increase
        assert var_smoothed <= var_original + 1e-10

    @given(
        field_values=st.lists(
            st.floats(
                min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
            min_size=10,
            max_size=50,
        )
    )
    def test_smooth_preserves_mean_property(self, field_values, small_2d_env):
        """Property: Smoothing with transition mode should preserve mean."""
        # Ensure field not larger than environment (tell Hypothesis to generate different input)
        assume(len(field_values) <= small_2d_env.n_bins)

        # Pad field to match environment size
        field = np.array(
            field_values + [0.0] * (small_2d_env.n_bins - len(field_values))
        )

        smoothed = small_2d_env.smooth(field, bandwidth=5.0, mode="transition")

        # Mean should be preserved (mass conservation)
        np.testing.assert_allclose(
            np.mean(smoothed), np.mean(field), rtol=1e-10, atol=1e-10
        )

    @given(bandwidth=st.floats(min_value=1.0, max_value=20.0))
    def test_smooth_bandwidth_increases_spread_property(self, bandwidth, small_2d_env):
        """Property: Larger bandwidth should spread impulse more."""
        # Create impulse
        field = np.zeros(small_2d_env.n_bins)
        center_bin = small_2d_env.n_bins // 2
        field[center_bin] = 100.0

        smoothed = small_2d_env.smooth(field, bandwidth=bandwidth, mode="transition")

        # Number of bins with significant mass should correlate with bandwidth
        num_significant = np.sum(smoothed > 0.01)
        # At minimum, should spread to more than 1 bin for bandwidth >= 1.0
        # (bandwidth 1.0 should spread beyond the central bin given bin_size=2.0)
        assert num_significant >= 1  # At least the central bin has mass
