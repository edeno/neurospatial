"""Tests for validate_environment function."""

import numpy as np
import pytest

from neurospatial import Environment, validate_environment
from neurospatial.layout.validation import GraphValidationError


class TestValidateEnvironment:
    """Test validate_environment function."""

    @pytest.fixture
    def valid_env(self):
        """Create a valid environment."""
        np.random.seed(42)
        data = np.random.randn(200, 2) * 10
        env = Environment.from_samples(data, bin_size=3.0, name="valid")
        env.units = "cm"
        env.frame = "world"
        return env

    def test_valid_environment_passes(self, valid_env):
        """Test that a valid environment passes validation."""
        # Should not raise any errors
        validate_environment(valid_env)

    def test_unfitted_environment_raises_error(self):
        """Test that unfitted environment raises RuntimeError."""
        # Create a minimal layout that hasn't been built
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        # Don't call build(), so it remains unfitted
        env = Environment(name="unfitted", layout=layout)

        with pytest.raises(RuntimeError, match="not fitted"):
            validate_environment(env)

    def test_strict_mode_warns_missing_units(self):
        """Test that strict mode warns about missing units."""
        np.random.seed(42)
        data = np.random.randn(100, 2) * 10
        env = Environment.from_samples(data, bin_size=5.0, name="no_units")
        # Don't set units
        env.frame = "world"

        with pytest.warns(UserWarning, match="no units specified"):
            validate_environment(env, strict=True)

    def test_strict_mode_warns_missing_frame(self):
        """Test that strict mode warns about missing frame."""
        np.random.seed(42)
        data = np.random.randn(100, 2) * 10
        env = Environment.from_samples(data, bin_size=5.0, name="no_frame")
        env.units = "cm"
        # Don't set frame

        with pytest.warns(UserWarning, match="no coordinate frame"):
            validate_environment(env, strict=True)

    def test_non_strict_mode_no_warnings(self):
        """Test that non-strict mode doesn't warn about missing metadata."""
        np.random.seed(42)
        data = np.random.randn(100, 2) * 10
        env = Environment.from_samples(data, bin_size=5.0, name="no_metadata")

        # Should not warn in non-strict mode
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            validate_environment(env, strict=False)

        # Filter out any unrelated warnings
        relevant_warnings = [
            w
            for w in warning_list
            if "units" in str(w.message) or "frame" in str(w.message)
        ]
        assert len(relevant_warnings) == 0

    def test_validates_graph_structure(self, valid_env):
        """Test that graph validation is called."""
        # Corrupt the graph by removing required attribute
        node_id = next(iter(valid_env.connectivity.nodes))
        del valid_env.connectivity.nodes[node_id]["pos"]

        with pytest.raises(GraphValidationError, match="missing required attributes"):
            validate_environment(valid_env)

    def test_validates_bin_centers_consistency(self, valid_env):
        """Test that bin_centers/graph consistency is checked."""
        # Corrupt bin_centers by changing shape
        valid_env.bin_centers = valid_env.bin_centers[:10]

        with pytest.raises(ValueError, match="inconsistent"):
            validate_environment(valid_env)

    def test_environment_with_regions(self, valid_env):
        """Test validation works with regions."""
        valid_env.regions.add("goal", point=np.array([5.0, 5.0]))

        # Should pass validation
        validate_environment(valid_env)
