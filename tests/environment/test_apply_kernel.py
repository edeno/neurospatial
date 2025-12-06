"""Test for apply_kernel() function with forward and adjoint modes.

This module tests the application of diffusion kernels in both forward and
adjoint modes, including proper handling of mass-weighted adjoints for
density-preserving kernels.

Tests cover:
- Forward mode (standard kernel application)
- Adjoint mode (transpose operation)
- Density mode with mass-weighted adjoint
- Mass conservation properties
- Input validation
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.smoothing import apply_kernel


class TestApplyKernelForward:
    """Test forward mode kernel application."""

    def test_forward_mode_basic(self):
        """Test basic forward kernel application."""
        # Create simple 5x5 environment
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Compute a simple kernel
        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        # Create a field with single spike
        field = np.zeros(env.n_bins)
        center_bin = env.n_bins // 2
        field[center_bin] = 1.0

        # Apply kernel in forward mode
        result = apply_kernel(field, K, mode="forward")

        # Result should be diffused (spread out)
        assert result.shape == (env.n_bins,)
        assert result[center_bin] < 1.0  # Original spike reduced
        assert np.sum(result > 0) > 1  # Spread to neighbors

    def test_forward_mode_is_default(self):
        """Test that mode='forward' is the default."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")
        field = rng.random(env.n_bins)

        result_explicit = apply_kernel(field, K, mode="forward")
        result_default = apply_kernel(field, K)

        np.testing.assert_array_equal(result_explicit, result_default)

    def test_forward_preserves_normalization_transition(self):
        """Test forward mode with transition kernel preserves normalization."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        # Normalized field (probability distribution)
        field = rng.random(env.n_bins)
        field /= field.sum()

        result = apply_kernel(field, K, mode="forward")

        # Transition kernel preserves normalization
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-10)


class TestApplyKernelAdjoint:
    """Test adjoint mode kernel application."""

    def test_adjoint_mode_basic(self):
        """Test basic adjoint kernel application."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        field = rng.random(env.n_bins)

        # Apply in adjoint mode
        result = apply_kernel(field, K, mode="adjoint")

        assert result.shape == (env.n_bins,)

    def test_adjoint_is_transpose_for_transition(self):
        """Test that adjoint is transpose for transition kernel without measure."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        field = rng.random(env.n_bins)

        # Adjoint without measure is just transpose
        result_adjoint = apply_kernel(field, K, mode="adjoint")
        result_manual = K.T @ field

        np.testing.assert_allclose(result_adjoint, result_manual)

    def test_adjoint_preserves_normalization(self):
        """Test adjoint preserves normalization for stochastic kernels."""
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        # Uniform distribution
        field = np.ones(env.n_bins) / env.n_bins

        result = apply_kernel(field, K, mode="adjoint")

        # For transition kernel, adjoint preserves total mass
        np.testing.assert_allclose(result.sum(), field.sum(), rtol=1e-10)


class TestApplyKernelDensityMode:
    """Test density mode with mass-weighted adjoint."""

    def test_density_mode_with_bin_sizes(self):
        """Test forward density mode with bin sizes."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Get bin sizes
        bin_sizes = env.bin_sizes

        # Compute density kernel
        K = env.compute_kernel(bandwidth=1.0, mode="density")

        # Create density field
        field = rng.random(env.n_bins)

        result = apply_kernel(field, K, mode="forward", bin_sizes=bin_sizes)

        assert result.shape == (env.n_bins,)

    def test_adjoint_density_mode_with_bin_sizes(self):
        """Test adjoint for density mode is mass-weighted."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        bin_sizes = env.bin_sizes
        K = env.compute_kernel(bandwidth=1.0, mode="density")

        field = rng.random(env.n_bins)

        # Adjoint with bin_sizes
        result = apply_kernel(field, K, mode="adjoint", bin_sizes=bin_sizes)

        # Manual computation: M^{-1} K^T M field
        M = np.diag(bin_sizes)
        M_inv = np.diag(1.0 / bin_sizes)
        expected = M_inv @ K.T @ M @ field

        np.testing.assert_allclose(result, expected)

    def test_density_mode_forward(self):
        """Test forward mode works with density kernels."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="density")

        # Density field (arbitrary values)
        field = rng.random(env.n_bins)

        # Apply forward
        result = apply_kernel(field, K, mode="forward")

        # Check result is valid
        assert result.shape == (env.n_bins,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Kernel is non-negative


class TestApplyKernelInputValidation:
    """Test input validation and error messages."""

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")
        field = rng.random(env.n_bins)

        with pytest.raises(ValueError, match="mode must be 'forward' or 'adjoint'"):
            apply_kernel(field, K, mode="invalid")

    def test_field_kernel_size_mismatch(self):
        """Test that mismatched field and kernel sizes raise error."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")
        wrong_field = rng.random(env.n_bins + 5)

        with pytest.raises(ValueError, match=r"Field size .* does not match kernel"):
            apply_kernel(wrong_field, K, mode="forward")

    def test_bin_sizes_size_mismatch(self):
        """Test that mismatched bin_sizes size raises error."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="density")
        field = rng.random(env.n_bins)
        wrong_bin_sizes = rng.random(env.n_bins + 5)

        with pytest.raises(ValueError, match=r"bin_sizes size .* does not match"):
            apply_kernel(field, K, mode="adjoint", bin_sizes=wrong_bin_sizes)

    def test_kernel_not_square(self):
        """Test that non-square kernel raises error."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        # Non-square matrix
        K = rng.random((env.n_bins, env.n_bins + 5))
        field = rng.random(env.n_bins)

        with pytest.raises(ValueError, match="Kernel must be square"):
            apply_kernel(field, K, mode="forward")


class TestApplyKernelMathematicalProperties:
    """Test mathematical properties of forward/adjoint pair."""

    def test_adjoint_inner_product_property(self):
        """Test that <Kx, y> = <x, K^T y> for transition mode."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        x = rng.random(env.n_bins)
        y = rng.random(env.n_bins)

        # Forward: K @ x
        Kx = apply_kernel(x, K, mode="forward")

        # Adjoint: K^T @ y
        KTy = apply_kernel(y, K, mode="adjoint")

        # Inner products should match
        left = np.dot(Kx, y)
        right = np.dot(x, KTy)

        np.testing.assert_allclose(left, right, rtol=1e-10)

    def test_adjoint_inner_product_with_bin_sizes(self):
        """Test weighted inner product property for density mode."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        bin_sizes = env.bin_sizes
        K = env.compute_kernel(bandwidth=1.0, mode="density")

        x = rng.random(env.n_bins)
        y = rng.random(env.n_bins)

        # Forward
        Kx = apply_kernel(x, K, mode="forward", bin_sizes=bin_sizes)

        # Adjoint
        KTy = apply_kernel(y, K, mode="adjoint", bin_sizes=bin_sizes)

        # Weighted inner product: <Kx, y>_M = <x, K^* y>_M
        # where <u, v>_M = sum(u * M * v)
        left = np.sum(Kx * bin_sizes * y)
        right = np.sum(x * bin_sizes * KTy)

        np.testing.assert_allclose(left, right, rtol=1e-10)

    def test_forward_adjoint_roundtrip(self):
        """Test applying forward then adjoint."""
        rng = np.random.default_rng(42)
        data = np.array([[i, j] for i in range(11) for j in range(11)])
        env = Environment.from_samples(data, bin_size=2.0)

        K = env.compute_kernel(bandwidth=1.0, mode="transition")

        field = rng.random(env.n_bins)

        # K^T @ K @ field
        forward_result = apply_kernel(field, K, mode="forward")
        roundtrip = apply_kernel(forward_result, K, mode="adjoint")

        # Should be different from original (smoothed twice)
        assert not np.allclose(roundtrip, field)

        # But should have same shape and be finite
        assert roundtrip.shape == field.shape
        assert np.all(np.isfinite(roundtrip))
