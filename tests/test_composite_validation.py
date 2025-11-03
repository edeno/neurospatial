"""Tests for CompositeEnvironment constructor validation (P2.4)."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.composite import CompositeEnvironment


@pytest.fixture
def sample_env_2d():
    """Create a simple 2D environment for testing."""
    data = np.random.randn(100, 2) * 10
    return Environment.from_samples(data, bin_size=2.0)


@pytest.fixture
def sample_env_3d():
    """Create a simple 3D environment for testing."""
    data = np.random.randn(100, 3) * 10
    return Environment.from_samples(data, bin_size=2.0)


class TestTypeValidation:
    """Test type validation in CompositeEnvironment constructor."""

    def test_invalid_subenvs_type_single_env(self, sample_env_2d):
        """Test error for passing single Environment instead of list."""
        with pytest.raises(
            TypeError,
            match=r"subenvs must be a list or tuple.*got Environment",
        ):
            CompositeEnvironment(sample_env_2d)  # Should be [sample_env_2d]

    def test_invalid_subenvs_type_string(self):
        """Test error for passing string instead of list."""
        with pytest.raises(
            TypeError, match=r"subenvs must be a list or tuple.*got str"
        ):
            CompositeEnvironment("not a list")

    def test_invalid_subenvs_type_dict(self):
        """Test error for passing dict instead of list."""
        with pytest.raises(
            TypeError, match=r"subenvs must be a list or tuple.*got dict"
        ):
            CompositeEnvironment({"env": "value"})

    def test_invalid_subenv_element_string(self, sample_env_2d):
        """Test error for non-Environment element in list."""
        with pytest.raises(
            TypeError,
            match=r"subenvs\[1\] must be an Environment instance.*got str",
        ):
            CompositeEnvironment([sample_env_2d, "not an environment"])

    def test_invalid_subenv_element_dict(self, sample_env_2d):
        """Test error for dict element in list."""
        with pytest.raises(
            TypeError,
            match=r"subenvs\[1\] must be an Environment instance.*got dict",
        ):
            CompositeEnvironment([sample_env_2d, {"not": "env"}])

    def test_invalid_subenv_element_none(self, sample_env_2d):
        """Test error for None element in list."""
        with pytest.raises(
            TypeError,
            match=r"subenvs\[0\] must be an Environment instance.*got NoneType",
        ):
            CompositeEnvironment([None, sample_env_2d])

    def test_mixed_invalid_elements(self, sample_env_2d):
        """Test error when multiple invalid elements (reports first)."""
        with pytest.raises(
            TypeError,
            match=r"subenvs\[1\] must be an Environment instance",
        ):
            CompositeEnvironment([sample_env_2d, 42, "invalid", None])


class TestValueValidation:
    """Test value validation in CompositeEnvironment constructor."""

    def test_empty_list(self):
        """Test error for empty subenvs list."""
        with pytest.raises(
            ValueError, match="At least one sub-environment is required"
        ):
            CompositeEnvironment([])

    def test_empty_tuple(self):
        """Test error for empty subenvs tuple."""
        with pytest.raises(
            ValueError, match="At least one sub-environment is required"
        ):
            CompositeEnvironment(())

    def test_dimension_mismatch_2d_3d(self, sample_env_2d, sample_env_3d):
        """Test error for mixing 2D and 3D environments."""
        with pytest.raises(
            ValueError,
            match=r"All sub-environments must share the same n_dims.*2.*3",
        ):
            CompositeEnvironment([sample_env_2d, sample_env_3d])

    def test_dimension_mismatch_multiple_envs(self, sample_env_2d, sample_env_3d):
        """Test dimension mismatch error with multiple environments."""
        env_2d_2 = Environment.from_samples(np.random.randn(50, 2) * 10, bin_size=2.0)
        with pytest.raises(
            ValueError,
            match="All sub-environments must share the same n_dims",
        ):
            CompositeEnvironment([sample_env_2d, env_2d_2, sample_env_3d])


class TestValidConstruction:
    """Test that valid inputs work correctly."""

    def test_valid_single_env_in_list(self, sample_env_2d):
        """Test that single environment in list works."""
        comp = CompositeEnvironment([sample_env_2d])
        assert comp.n_bins == sample_env_2d.n_bins
        assert comp.n_dims == sample_env_2d.n_dims

    def test_valid_multiple_envs(self, sample_env_2d):
        """Test that multiple valid environments work."""
        env2 = Environment.from_samples(np.random.randn(80, 2) * 10 + 50, bin_size=2.0)
        comp = CompositeEnvironment([sample_env_2d, env2])
        assert comp.n_bins > sample_env_2d.n_bins
        assert comp.n_dims == 2

    def test_valid_tuple_input(self, sample_env_2d):
        """Test that tuple of environments works (not just list)."""
        env2 = Environment.from_samples(np.random.randn(80, 2) * 10 + 50, bin_size=2.0)
        comp = CompositeEnvironment((sample_env_2d, env2))
        assert comp.n_bins > sample_env_2d.n_bins

    def test_valid_3d_envs(self, sample_env_3d):
        """Test that multiple 3D environments work."""
        env2 = Environment.from_samples(np.random.randn(80, 3) * 10 + 50, bin_size=2.0)
        comp = CompositeEnvironment([sample_env_3d, env2])
        assert comp.n_dims == 3
        assert comp.n_bins > sample_env_3d.n_bins


class TestErrorMessages:
    """Test that error messages are helpful and actionable."""

    def test_single_env_error_suggests_list_wrapping(self, sample_env_2d):
        """Test error message suggests wrapping in list."""
        with pytest.raises(TypeError, match="Use \\[env\\] to wrap it in a list"):
            CompositeEnvironment(sample_env_2d)

    def test_invalid_element_shows_index(self, sample_env_2d):
        """Test that error shows which element is invalid."""
        with pytest.raises(TypeError, match=r"subenvs\[2\]"):
            CompositeEnvironment([sample_env_2d, sample_env_2d, "invalid"])

    def test_dimension_mismatch_shows_both_dims(self, sample_env_2d, sample_env_3d):
        """Test dimension error shows both dimensionalities."""
        with pytest.raises(ValueError) as exc_info:
            CompositeEnvironment([sample_env_2d, sample_env_3d])

        error_msg = str(exc_info.value)
        assert "2" in error_msg
        assert "3" in error_msg
        assert "n_dims" in error_msg
