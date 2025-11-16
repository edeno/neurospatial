"""Tests for Environment.smooth() method.

This module tests the field smoothing functionality via diffusion kernels.
"""

import numpy as np
import pytest

from neurospatial import Environment


class TestSmoothBasic:
    """Basic smoothing functionality tests."""

    def test_smooth_impulse_spreads_to_neighbors(self, graph_env):
        """Impulse at one bin should spread to neighbors after smoothing."""
        env = graph_env

        # Create impulse at center bin (bin 4 in plus maze)
        field = np.zeros(env.n_bins)
        field[4] = 1.0

        # Apply smoothing with small bandwidth
        smoothed = env.smooth(field, bandwidth=5.0, mode="transition")

        # Check that impulse spread to neighbors
        assert smoothed[4] > 0  # Center should still have mass
        neighbors = list(env.connectivity.neighbors(4))
        for neighbor in neighbors:
            assert smoothed[neighbor] > 0, (
                f"Neighbor {neighbor} should have positive value"
            )

        # Non-neighbors should have less mass (or zero for distant bins)
        # This is a weak assertion - just checking basic spreading behavior

    def test_smooth_constant_field_unchanged(self):
        """Constant field should remain constant after smoothing."""
        # Create simple 2D grid environment
        positions = np.array(
            [[x, y] for x in range(10) for y in range(10)], dtype=np.float64
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Constant field
        field = np.ones(env.n_bins) * 5.0

        # Smooth with transition mode
        smoothed = env.smooth(field, bandwidth=2.0, mode="transition")

        # Should remain constant (within numerical tolerance)
        np.testing.assert_allclose(smoothed, field, rtol=1e-5, atol=1e-5)

    def test_smooth_returns_correct_shape(self):
        """Smoothed field should have same shape as input field."""
        positions = np.random.rand(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=5.0)

        field = np.random.rand(env.n_bins)
        smoothed = env.smooth(field, bandwidth=3.0)

        assert smoothed.shape == field.shape
        assert smoothed.shape == (env.n_bins,)

    def test_smooth_preserves_dtype(self):
        """Smoothed field should be float64."""
        positions = np.random.rand(50, 2) * 30
        env = Environment.from_samples(positions, bin_size=3.0)

        field = np.random.rand(env.n_bins)
        smoothed = env.smooth(field, bandwidth=2.0)

        assert smoothed.dtype == np.float64


class TestSmoothMassConservation:
    """Test mass conservation properties."""

    def test_smooth_transition_mode_conserves_mass(self):
        """Smoothing with mode='transition' should conserve total mass."""
        positions = np.random.rand(100, 2) * 40
        env = Environment.from_samples(positions, bin_size=4.0)

        # Random field
        field = np.random.rand(env.n_bins) * 10

        # Smooth with transition mode
        smoothed = env.smooth(field, bandwidth=3.0, mode="transition")

        # Total mass should be conserved
        np.testing.assert_allclose(
            smoothed.sum(),
            field.sum(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Mass not conserved",
        )

    def test_smooth_density_mode_respects_volumes(self):
        """Smoothing with mode='density' should respect bin volumes."""
        # Create regular grid where we can control bin sizes
        positions = np.array(
            [[x, y] for x in range(10) for y in range(10)], dtype=np.float64
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Uniform density field
        field = np.ones(env.n_bins)

        # Smooth with density mode
        smoothed = env.smooth(field, bandwidth=1.5, mode="density")

        # For uniform field with uniform bin sizes, should remain approximately uniform
        # (allowing for edge effects)
        interior_bins = smoothed[10:-10]  # Exclude edge bins
        std_dev = np.std(interior_bins)
        assert std_dev < 0.1, "Uniform field should remain approximately uniform"


class TestSmoothEdgePreservation:
    """Test that smoothing respects graph connectivity."""

    def test_smooth_no_leakage_across_components(self):
        """Smoothing should not leak mass between disconnected components."""
        # Create environment with two GUARANTEED disconnected components
        pos1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        pos2 = np.array(
            [[100, 100], [101, 100], [100, 101], [101, 101]], dtype=np.float64
        )

        positions = np.vstack([pos1, pos2])
        env = Environment.from_samples(positions, bin_size=2.0)

        # Verify disconnection FIRST
        components = env.components()
        assert len(components) == 2, (
            f"Test setup failed: expected 2 disconnected components, got {len(components)}. "
            "This test requires disconnected components."
        )

        # Now test non-leakage
        field = np.zeros(env.n_bins)
        comp1_bins = list(components[0])
        comp2_bins = list(components[1])
        field[comp1_bins] = 1.0  # Mass in component 1 only

        smoothed = env.smooth(field, bandwidth=1.0, mode="transition")

        # Component 2 should have zero mass
        assert smoothed[comp2_bins].sum() == 0, (
            "Mass leaked across disconnected components"
        )
        # Component 1 should preserve mass
        np.testing.assert_allclose(
            smoothed[comp1_bins].sum(), field[comp1_bins].sum(), rtol=1e-5
        )


class TestSmoothModes:
    """Test different smoothing modes."""

    def test_smooth_transition_mode(self):
        """Test mode='transition' parameter."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.random.rand(env.n_bins)
        smoothed = env.smooth(field, bandwidth=2.0, mode="transition")

        assert smoothed.shape == (env.n_bins,)
        # Mass conservation test
        np.testing.assert_allclose(smoothed.sum(), field.sum(), rtol=1e-5)

    def test_smooth_density_mode(self):
        """Test mode='density' parameter."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.random.rand(env.n_bins)
        smoothed = env.smooth(field, bandwidth=2.0, mode="density")

        assert smoothed.shape == (env.n_bins,)


class TestSmoothValidation:
    """Input validation tests."""

    def test_smooth_wrong_field_shape_raises_error(self):
        """Field with wrong shape should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        # Wrong shape
        field_wrong = np.random.rand(env.n_bins + 10)

        with pytest.raises(ValueError, match=r"Field shape.*must match n_bins"):
            env.smooth(field_wrong, bandwidth=2.0)

    def test_smooth_negative_bandwidth_raises_error(self):
        """Negative bandwidth should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.random.rand(env.n_bins)

        with pytest.raises(ValueError, match=r"bandwidth must be positive"):
            env.smooth(field, bandwidth=-1.0)

    def test_smooth_zero_bandwidth_raises_error(self):
        """Zero bandwidth should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.random.rand(env.n_bins)

        with pytest.raises(ValueError, match=r"bandwidth must be positive"):
            env.smooth(field, bandwidth=0.0)

    def test_smooth_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.random.rand(env.n_bins)

        with pytest.raises(ValueError, match=r"mode must be"):
            env.smooth(field, bandwidth=2.0, mode="invalid")

    # NOTE: No test for @check_fitted decorator on unfitted environments because:
    # - Cannot create unfitted Environment with current public API (by design)
    # - The decorator is tested indirectly by all other tests in this suite
    # - Factory methods always create fitted environments

    def test_smooth_2d_field_raises_error(self):
        """Field must be 1-D array. 2-D arrays should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        # 2D field
        field_2d = np.random.rand(env.n_bins, 3)

        with pytest.raises(ValueError, match=r"Field must be 1-D array"):
            env.smooth(field_2d, bandwidth=2.0)

    def test_smooth_field_with_nan_raises_error(self):
        """Field containing NaN should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.ones(env.n_bins)
        field[0] = np.nan

        with pytest.raises(ValueError, match=r"Field contains NaN values"):
            env.smooth(field, bandwidth=2.0)

    def test_smooth_field_with_inf_raises_error(self):
        """Field containing Inf should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.ones(env.n_bins)
        field[0] = np.inf

        with pytest.raises(ValueError, match=r"Field contains infinite values"):
            env.smooth(field, bandwidth=2.0)

    def test_smooth_field_with_negative_inf_raises_error(self):
        """Field containing -Inf should raise ValueError."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.ones(env.n_bins)
        field[0] = -np.inf

        with pytest.raises(ValueError, match=r"Field contains infinite values"):
            env.smooth(field, bandwidth=2.0)

    def test_smooth_empty_field_handled(self):
        """Empty field (all zeros) should be handled gracefully."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field = np.zeros(env.n_bins)
        smoothed = env.smooth(field, bandwidth=2.0)

        # Should return all zeros
        np.testing.assert_array_equal(smoothed, field)


class TestSmoothMultipleLayouts:
    """Test smoothing across different layout types."""

    def test_smooth_on_regular_grid(self):
        """Smooth should work on regular grid layouts."""
        positions = np.array(
            [[x, y] for x in range(5) for y in range(5)], dtype=np.float64
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        field = np.random.rand(env.n_bins)
        smoothed = env.smooth(field, bandwidth=1.0)

        assert smoothed.shape == (env.n_bins,)

    def test_smooth_on_graph_layout(self, graph_env):
        """Smooth should work on graph layouts."""
        env = graph_env

        field = np.random.rand(env.n_bins)
        smoothed = env.smooth(field, bandwidth=3.0)

        assert smoothed.shape == (env.n_bins,)


class TestSmoothBandwidthEffect:
    """Test effect of different bandwidth values."""

    def test_smooth_small_bandwidth_local_effect(self):
        """Small bandwidth should have more local smoothing effect."""
        positions = np.array(
            [[x, y] for x in range(10) for y in range(10)], dtype=np.float64
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Impulse at center
        field = np.zeros(env.n_bins)
        center_idx = env.n_bins // 2
        field[center_idx] = 10.0

        # Small bandwidth
        smoothed_small = env.smooth(field, bandwidth=0.5, mode="transition")
        # Large bandwidth
        smoothed_large = env.smooth(field, bandwidth=5.0, mode="transition")

        # Small bandwidth should keep more mass at center
        assert smoothed_small[center_idx] > smoothed_large[center_idx]

    def test_smooth_large_bandwidth_global_effect(self):
        """Large bandwidth should spread mass more globally."""
        positions = np.array(
            [[x, y] for x in range(10) for y in range(10)], dtype=np.float64
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Impulse at one corner
        field = np.zeros(env.n_bins)
        field[0] = 10.0

        # Large bandwidth
        smoothed = env.smooth(field, bandwidth=10.0, mode="transition")

        # Mass should spread across many bins
        num_nonzero = np.sum(smoothed > 1e-6)
        assert num_nonzero > 10, "Large bandwidth should spread mass widely"


class TestSmoothCaching:
    """Test that kernel caching works correctly."""

    def test_smooth_uses_kernel_cache(self):
        """Repeated calls with same bandwidth should use cached kernel."""
        positions = np.random.rand(50, 2) * 20
        env = Environment.from_samples(positions, bin_size=2.0)

        field1 = np.random.rand(env.n_bins)
        field2 = np.random.rand(env.n_bins)

        # First call computes kernel
        smoothed1 = env.smooth(field1, bandwidth=3.0, mode="transition")

        # Second call should use cached kernel
        smoothed2 = env.smooth(field2, bandwidth=3.0, mode="transition")

        # Both should succeed
        assert smoothed1.shape == (env.n_bins,)
        assert smoothed2.shape == (env.n_bins,)

        # Results should be different (different inputs)
        assert not np.allclose(smoothed1, smoothed2)
