"""Tests for error code system.

This module tests that error codes appear in error messages and link to
documentation for debugging assistance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.composite import CompositeEnvironment
from neurospatial.io import to_file


class TestErrorCodeE1001NoActiveBins:
    """Tests for E1001: No active bins found error."""

    def test_e1001_appears_in_no_active_bins_error(self):
        """Test that E1001 error code appears when no active bins found."""
        rng = np.random.default_rng(42)
        # Create data that will result in no active bins
        # Use high bin_count_threshold that no bins can meet
        positions = rng.standard_normal((100, 2)) * 10  # Data in ~[-30, 30] range

        with pytest.raises(ValueError, match=r"\[E1001\]"):
            Environment.from_samples(
                positions,
                bin_size=5.0,
                bin_count_threshold=1000,  # Way too high - no bin has 1000 samples!
            )

    def test_e1001_error_includes_diagnostics(self):
        """Test that E1001 error includes diagnostic information."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 10

        with pytest.raises(ValueError) as exc_info:
            Environment.from_samples(
                positions,
                bin_size=5.0,
                bin_count_threshold=1000,  # Too high - triggers no active bins
            )

        error_msg = str(exc_info.value)
        # Check for error code
        assert "[E1001]" in error_msg
        # Check for diagnostic sections
        assert "Diagnostics:" in error_msg
        assert "Data range:" in error_msg or "bin_size:" in error_msg


class TestErrorCodeE1002InvalidBinSize:
    """Tests for E1002: Invalid bin_size error."""

    def test_e1002_appears_for_negative_bin_size(self):
        """Test that E1002 error code appears for negative bin_size."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 10

        with pytest.raises(ValueError, match=r"\[E1002\]"):
            Environment.from_samples(
                positions,
                bin_size=-5.0,  # Negative!
            )

    def test_e1002_appears_for_zero_bin_size(self):
        """Test that E1002 error code appears for zero bin_size."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 10

        with pytest.raises(ValueError, match=r"\[E1002\]"):
            Environment.from_samples(
                positions,
                bin_size=0.0,  # Zero!
            )

    def test_e1002_appears_for_nan_bin_size(self):
        """Test that E1002 error code appears for NaN bin_size."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 10

        with pytest.raises(ValueError, match=r"\[E1002\]"):
            Environment.from_samples(
                positions,
                bin_size=np.nan,  # NaN!
            )

    def test_e1002_appears_for_inf_bin_size(self):
        """Test that E1002 error code appears for infinite bin_size."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 10

        with pytest.raises(ValueError, match=r"\[E1002\]"):
            Environment.from_samples(
                positions,
                bin_size=np.inf,  # Infinity!
            )


class TestErrorCodeE1003DimensionMismatch:
    """Tests for E1003: Dimension mismatch error."""

    def test_e1003_appears_for_dimension_mismatch(self):
        """Test that E1003 error code appears when dimensions don't match."""
        rng = np.random.default_rng(42)
        # Create 2D environment
        positions_2d = rng.standard_normal((100, 2)) * 10
        env_2d = Environment.from_samples(positions_2d, bin_size=5.0)

        # Create 3D environment
        positions_3d = rng.standard_normal((100, 3)) * 10
        env_3d = Environment.from_samples(positions_3d, bin_size=5.0)

        # Try to combine them
        with pytest.raises(ValueError, match=r"\[E1003\]"):
            CompositeEnvironment(subenvs=[env_2d, env_3d])

    def test_e1003_error_includes_dimension_info(self):
        """Test that E1003 error includes actual dimension values."""
        rng = np.random.default_rng(42)
        positions_2d = rng.standard_normal((100, 2)) * 10
        env_2d = Environment.from_samples(positions_2d, bin_size=5.0)

        positions_3d = rng.standard_normal((100, 3)) * 10
        env_3d = Environment.from_samples(positions_3d, bin_size=5.0)

        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment(subenvs=[env_2d, env_3d])

        error_msg = str(exc_info.value)
        # Check for error code
        assert "[E1003]" in error_msg
        # Check for dimension information
        assert "2" in error_msg  # Should mention 2D
        assert "3" in error_msg  # Should mention 3D


class TestErrorCodeE1004NotFitted:
    """Tests for E1004: Environment not fitted error."""

    def test_e1004_appears_for_unfitted_environment(self):
        """Test that E1004 error code appears for unfitted environment."""
        # Create minimal unfitted environment
        # (Environment requires layout now, so we use a minimal setup)
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        env = Environment(layout=layout)  # Not fitted yet (no _is_fitted=True)

        # Try to call a method that requires fitted state
        with pytest.raises(RuntimeError, match=r"\[E1004\]"):
            env.bin_at([[10.0, 20.0]])

    def test_e1004_error_includes_usage_example(self):
        """Test that E1004 error includes correct usage example."""
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        env = Environment(layout=layout)  # Not fitted

        with pytest.raises(RuntimeError) as exc_info:
            env.bin_at([[10.0, 20.0]])

        error_msg = str(exc_info.value)
        # Check for error code
        assert "[E1004]" in error_msg
        # Check for helpful example
        assert "from_samples" in error_msg or "factory method" in error_msg


class TestErrorCodeE1005PathTraversal:
    """Tests for E1005: Path traversal detected error."""

    def test_e1005_appears_for_path_traversal(self, tmp_path, grid_env_from_samples):
        """Test that E1005 error code appears for path traversal attempt."""
        # Try to save with path traversal
        malicious_path = tmp_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match=r"\[E1005\]"):
            to_file(grid_env_from_samples, malicious_path)

    def test_e1005_error_includes_security_explanation(
        self, tmp_path, grid_env_from_samples
    ):
        """Test that E1005 error explains the security risk."""
        malicious_path = tmp_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError) as exc_info:
            to_file(grid_env_from_samples, malicious_path)

        error_msg = str(exc_info.value)
        # Check for error code
        assert "[E1005]" in error_msg
        # Check for security explanation
        assert "security" in error_msg.lower() or "traversal" in error_msg.lower()


class TestErrorCodeDocumentation:
    """Tests for error code documentation links."""

    def test_error_codes_are_documented(self):
        """Test that all error codes have documentation entries."""
        import re

        # Read the errors.md file
        errors_doc_path = Path(__file__).parent.parent / "docs" / "errors.md"
        assert errors_doc_path.exists(), "docs/errors.md should exist"

        with errors_doc_path.open(encoding="utf-8") as f:
            content = f.read()

        # Check that all 5 error codes are documented
        error_codes = ["E1001", "E1002", "E1003", "E1004", "E1005"]
        for code in error_codes:
            # Look for header like "### E1001: No active bins found"
            pattern = rf"###\s+{code}:"
            assert re.search(pattern, content), (
                f"Error code {code} should be documented"
            )

    def test_error_documentation_has_solutions(self):
        """Test that error documentation includes solution sections."""

        errors_doc_path = Path(__file__).parent.parent / "docs" / "errors.md"

        with errors_doc_path.open(encoding="utf-8") as f:
            content = f.read()

        # Each error should have "Solutions:" or "Solution" section
        # Count should be at least 5 (one per error code)
        solution_count = content.lower().count("solution")
        assert solution_count >= 10, (
            f"Error documentation should have multiple solution sections, "
            f"found {solution_count}"
        )


class TestErrorCodeConstants:
    """Tests for error code constants in source modules."""

    def test_error_code_constants_defined(self):
        """Test that error code constants are defined in relevant modules.

        This is a placeholder test that will pass once constants are defined.
        For now, we just verify the test structure is correct.
        """
        # This test will be enhanced after constants are added
        # For now, just verify we can import the modules
        import neurospatial.composite
        import neurospatial.environment.decorators
        import neurospatial.io
        import neurospatial.layout.validation  # noqa: F401

        # Future: Check that constants like ERROR_E1001 are defined
        # from neurospatial.layout.validation import ERROR_E1001
        # assert ERROR_E1001 == "E1001"
