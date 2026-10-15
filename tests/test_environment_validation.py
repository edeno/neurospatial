"""Tests for validate_environment function."""

import numpy as np
import pytest

from neurospatial import Environment, Regions
from neurospatial.layout.validation import GraphValidationError, validate_environment


class TestValidateEnvironment:
    """Test validate_environment function."""

    @pytest.fixture
    def valid_env(self):
        """Create a valid environment."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 2)) * 10
        env = Environment.from_samples(data, bin_size=3.0, name="valid")
        env = env.with_metadata(units="cm")
        env = env.with_metadata(frame="world")
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
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=5.0, name="no_units")
        # Don't set units
        env = env.with_metadata(frame="world")

        with pytest.warns(UserWarning, match="no units specified"):
            validate_environment(env, strict=True)

    def test_strict_mode_warns_missing_frame(self):
        """Test that strict mode warns about missing frame."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=5.0, name="no_frame")
        env = env.with_metadata(units="cm")
        # Don't set frame

        with pytest.warns(UserWarning, match="no coordinate frame"):
            validate_environment(env, strict=True)

    def test_non_strict_mode_no_warnings(self):
        """Test that non-strict mode doesn't warn about missing metadata."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2)) * 10
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
        import dataclasses

        # Simulate corrupted published geometry using a detached graph;
        # the public connectivity graph rejects attribute edits.
        malformed_graph = valid_env.connectivity.copy()
        node_id = next(iter(malformed_graph.nodes))
        del malformed_graph.nodes[node_id]["pos"]
        valid_env._geometry = dataclasses.replace(
            valid_env._geometry, connectivity=malformed_graph
        )

        with pytest.raises(GraphValidationError, match="missing required attributes"):
            validate_environment(valid_env)

    def test_validates_bin_centers_consistency(self, valid_env):
        """Test that bin_centers/graph consistency is checked."""
        # Simulate corrupted geometry (unreachable through the public API,
        # which publishes geometry read-only).
        import dataclasses

        valid_env._geometry = dataclasses.replace(
            valid_env._geometry,
            bin_centers=np.frombuffer(
                valid_env.bin_centers[:10].tobytes(), dtype=np.float64
            ).reshape(10, valid_env.n_dims),
        )

        with pytest.raises(ValueError, match="inconsistent"):
            validate_environment(valid_env)

    def test_environment_with_regions(self, valid_env):
        """Test validation works with regions."""
        region_builder = Regions(valid_env.regions.values())
        region_builder.add("goal", point=np.array([5.0, 5.0]))
        valid_env = valid_env.with_regions(region_builder)

        # Should pass validation
        validate_environment(valid_env)
