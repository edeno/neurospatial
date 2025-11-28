"""Tests for directional_field_index function.

Tests directional index metric: (forward - reverse) / (forward + reverse + eps).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestDirectionalFieldIndex:
    """Tests for directional_field_index function."""

    def test_all_forward(self) -> None:
        """Test: field_forward >> field_reverse → index ≈ +1."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Forward field is much larger than reverse
        field_forward = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        field_reverse = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = directional_field_index(field_forward, field_reverse)

        # Should be close to +1 (forward dominant)
        assert_allclose(result, np.ones(5), atol=1e-7)

    def test_all_reverse(self) -> None:
        """Test: field_reverse >> field_forward → index ≈ -1."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Reverse field is much larger than forward
        field_forward = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        field_reverse = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        result = directional_field_index(field_forward, field_reverse)

        # Should be close to -1 (reverse dominant)
        assert_allclose(result, -np.ones(5), atol=1e-7)

    def test_equal_fields(self) -> None:
        """Test: Equal fields → index ≈ 0."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Equal firing in both directions
        field_forward = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        field_reverse = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

        result = directional_field_index(field_forward, field_reverse)

        # Should be close to 0 (no preference)
        assert_allclose(result, np.zeros(5), atol=1e-6)

    def test_nan_propagation(self) -> None:
        """Test: NaN in input → NaN in output at that position."""
        from neurospatial.metrics.place_fields import directional_field_index

        field_forward = np.array([10.0, np.nan, 30.0, 40.0])
        field_reverse = np.array([5.0, 10.0, np.nan, 20.0])

        result = directional_field_index(field_forward, field_reverse)

        # NaN should propagate
        assert np.isnan(result[1])  # NaN in forward
        assert np.isnan(result[2])  # NaN in reverse
        # Valid positions should have valid results
        assert not np.isnan(result[0])
        assert not np.isnan(result[3])

    def test_eps_prevents_division_by_zero(self) -> None:
        """Test: Both fields zero → finite result (eps prevents div by 0)."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Both fields are zero at some positions
        field_forward = np.array([10.0, 0.0, 30.0])
        field_reverse = np.array([5.0, 0.0, 10.0])

        result = directional_field_index(field_forward, field_reverse)

        # Should be finite (eps prevents division by zero)
        assert np.all(np.isfinite(result))
        # Position 1: (0 - 0) / (0 + 0 + eps) = 0 / eps = 0
        assert_allclose(result[1], 0.0, atol=1e-6)

    def test_shape_preserved(self) -> None:
        """Test: Output shape matches input shape."""
        from neurospatial.metrics.place_fields import directional_field_index

        # 1D input
        field_forward_1d = np.array([10.0, 20.0, 30.0])
        field_reverse_1d = np.array([5.0, 10.0, 15.0])
        result_1d = directional_field_index(field_forward_1d, field_reverse_1d)
        assert result_1d.shape == field_forward_1d.shape

        # Larger 1D input
        field_forward_large = np.random.rand(100)
        field_reverse_large = np.random.rand(100)
        result_large = directional_field_index(field_forward_large, field_reverse_large)
        assert result_large.shape == (100,)

    def test_values_in_valid_range(self) -> None:
        """Test: Values in range [-1, 1] for non-NaN inputs."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Random positive values
        np.random.seed(42)
        field_forward = np.random.rand(50) * 10
        field_reverse = np.random.rand(50) * 10

        result = directional_field_index(field_forward, field_reverse)

        # All values should be in [-1, 1]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_custom_eps(self) -> None:
        """Test: Custom eps parameter is used."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Both fields zero
        field_forward = np.array([0.0])
        field_reverse = np.array([0.0])

        # With default eps=1e-9, result should be close to 0
        result_default = directional_field_index(field_forward, field_reverse)
        assert_allclose(result_default[0], 0.0, atol=1e-6)

        # With large eps=1.0, result should still be 0 (0/eps = 0)
        result_large_eps = directional_field_index(
            field_forward, field_reverse, eps=1.0
        )
        assert_allclose(result_large_eps[0], 0.0, atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        """Test: Mismatched shapes raise ValueError."""
        from neurospatial.metrics.place_fields import directional_field_index

        field_forward = np.array([10.0, 20.0, 30.0])
        field_reverse = np.array([5.0, 10.0])  # Different length

        with pytest.raises(ValueError, match="shape"):
            directional_field_index(field_forward, field_reverse)

    def test_no_environment_dependency(self) -> None:
        """Test: Function works without any Environment object."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Should work with just arrays, no Environment needed
        field_forward = np.array([10.0, 20.0, 30.0])
        field_reverse = np.array([5.0, 10.0, 15.0])

        # This should not raise any errors
        result = directional_field_index(field_forward, field_reverse)
        assert result.shape == (3,)

    def test_mixed_direction_bins(self) -> None:
        """Test: Some bins forward-dominant, others reverse-dominant."""
        from neurospatial.metrics.place_fields import directional_field_index

        # Bin 0: forward dominant, Bin 1: equal, Bin 2: reverse dominant
        field_forward = np.array([10.0, 5.0, 0.0])
        field_reverse = np.array([0.0, 5.0, 10.0])

        result = directional_field_index(field_forward, field_reverse)

        # Check each bin
        assert result[0] > 0.9  # Forward dominant → close to +1
        assert_allclose(result[1], 0.0, atol=1e-6)  # Equal → 0
        assert result[2] < -0.9  # Reverse dominant → close to -1
