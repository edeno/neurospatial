"""Tests for improved CompositeEnvironment dimension mismatch error messages.

This test suite ensures that CompositeEnvironment provides helpful error messages
with diagnostics and guidance when users try to combine environments with different
dimensionalities.
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.composite import CompositeEnvironment


class TestCompositeDimensionMismatchError:
    """Test that dimension mismatch errors include helpful diagnostics."""

    def test_error_shows_actual_dimensions(self):
        """Test that error message shows actual dimension values for both environments."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Should show the actual dimensions
        assert "2" in error_msg  # 2D environment
        assert "3" in error_msg  # 3D environment

    def test_error_shows_environment_indices(self):
        """Test that error message shows which environments have mismatched dimensions."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Should mention which environments (e.g., "Env 0", "Env 1")
        assert "0" in error_msg or "first" in error_msg.lower()
        assert "1" in error_msg or "second" in error_msg.lower()

    def test_error_explains_common_cause(self):
        """Test that error message explains the common cause of the problem."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Should explain WHY this happens (common causes section)
        assert (
            "common" in error_msg.lower()
            or "cause" in error_msg.lower()
            or "typically" in error_msg.lower()
            or "mixed" in error_msg.lower()
        )

    def test_error_provides_fix_guidance(self):
        """Test that error message provides actionable guidance on how to fix."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Should have "To fix" or similar guidance section
        assert (
            "to fix" in error_msg.lower()
            or "suggestion" in error_msg.lower()
            or "ensure" in error_msg.lower()
            or "check" in error_msg.lower()
        )

    def test_error_mentions_data_samples_check(self):
        """Test that error message suggests checking data_samples dimensionality."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Should mention checking data or samples
        assert "data" in error_msg.lower() or "sample" in error_msg.lower()

    def test_error_is_multiline_and_readable(self):
        """Test that error message is formatted for readability."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Should be multi-line (contains newlines)
        assert "\n" in error_msg
        # Should be sufficiently detailed (not just a one-liner)
        assert len(error_msg) > 100

    def test_error_follows_what_why_how_pattern(self):
        """Test that error message follows WHAT/WHY/HOW pattern from design guidelines."""
        # Create 2D environment
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        # Create 3D environment
        data_3d = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d = Environment.from_samples(data_3d, bin_size=2.0)

        # Try to create composite with mismatched dimensions
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d])

        error_msg = str(exc_info.value)

        # WHAT: Should state the problem clearly
        assert "dimension" in error_msg.lower() or "n_dims" in error_msg.lower()

        # WHY: Should explain cause or context
        has_why = any(
            keyword in error_msg.lower()
            for keyword in ["common", "cause", "typically", "because"]
        )
        assert has_why, "Error message should explain WHY (common causes)"

        # HOW: Should provide actionable guidance
        has_how = any(
            keyword in error_msg.lower()
            for keyword in ["to fix", "ensure", "check", "suggestion"]
        )
        assert has_how, "Error message should explain HOW to fix"

    def test_multiple_dimension_mismatches_reports_first_mismatch(self):
        """Test that error reports the first dimension mismatch found."""
        # Create environments with different dimensions
        data_2d = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d = Environment.from_samples(data_2d, bin_size=2.0)

        data_3d_a = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d_a = Environment.from_samples(data_3d_a, bin_size=2.0)

        data_3d_b = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        env_3d_b = Environment.from_samples(data_3d_b, bin_size=2.0)

        # Try to create composite (should fail on first mismatch: 2D vs 3D)
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([env_2d, env_3d_a, env_3d_b])

        error_msg = str(exc_info.value)
        # Should report env 0 (2D) and env 1 (3D) as the mismatch
        assert "0" in error_msg or "first" in error_msg.lower()
        assert "1" in error_msg or "second" in error_msg.lower()

    def test_matching_dimensions_does_not_raise(self):
        """Test that matching dimensions does not raise an error."""
        # Create two 2D environments
        data_2d_a = np.array([[0, 0], [10, 10], [5, 5]])
        env_2d_a = Environment.from_samples(data_2d_a, bin_size=2.0)

        data_2d_b = np.array([[20, 20], [30, 30], [25, 25]])
        env_2d_b = Environment.from_samples(data_2d_b, bin_size=2.0)

        # Should work without error
        composite = CompositeEnvironment([env_2d_a, env_2d_b])
        assert composite.n_dims == 2
        assert composite.n_bins > 0
