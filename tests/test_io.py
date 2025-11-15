"""Tests for neurospatial.io serialization functionality."""

import json
from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.io import from_dict, from_file, to_dict, to_file


class TestSerialization:
    """Test Environment serialization to/from files."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        np.random.seed(42)
        data = np.random.randn(500, 2) * 10
        env = Environment.from_samples(data, bin_size=3.0, name="test_env")
        env.units = "cm"
        env.frame = "world"
        return env

    @pytest.fixture
    def env_with_regions(self, simple_env):
        """Add regions to environment."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))
        return simple_env

    def test_to_file_creates_both_files(self, simple_env, tmp_path):
        """Test that to_file creates both JSON and npz files."""
        output_path = tmp_path / "test_env"
        to_file(simple_env, output_path)

        assert (tmp_path / "test_env.json").exists()
        assert (tmp_path / "test_env.npz").exists()

    def test_to_file_json_structure(self, simple_env, tmp_path):
        """Test that JSON file has expected structure."""
        output_path = tmp_path / "test_env"
        to_file(simple_env, output_path)

        with (tmp_path / "test_env.json").open() as f:
            data = json.load(f)

        assert "schema_version" in data
        assert data["schema_version"] == "Environment-v1"
        assert "library_version" in data
        assert "created_at" in data
        assert data["name"] == "test_env"
        assert data["n_dims"] == 2
        assert data["n_bins"] > 0
        assert data["units"] == "cm"
        assert data["frame"] == "world"
        assert "graph" in data
        assert "layout_type" in data

    def test_from_file_reconstructs_environment(self, simple_env, tmp_path):
        """Test that from_file correctly reconstructs environment."""
        output_path = tmp_path / "test_env"
        to_file(simple_env, output_path)

        loaded_env = from_file(output_path)

        assert loaded_env.name == simple_env.name
        assert loaded_env.n_bins == simple_env.n_bins
        assert loaded_env.n_dims == simple_env.n_dims
        assert np.allclose(loaded_env.bin_centers, simple_env.bin_centers)
        assert (
            loaded_env.connectivity.number_of_nodes()
            == simple_env.connectivity.number_of_nodes()
        )
        assert (
            loaded_env.connectivity.number_of_edges()
            == simple_env.connectivity.number_of_edges()
        )
        assert loaded_env.units == simple_env.units
        assert loaded_env.frame == simple_env.frame

    def test_roundtrip_preserves_regions(self, env_with_regions, tmp_path):
        """Test that regions survive serialization roundtrip."""
        output_path = tmp_path / "test_env"
        to_file(env_with_regions, output_path)

        loaded_env = from_file(output_path)

        assert len(loaded_env.regions) == len(env_with_regions.regions)
        assert "goal" in loaded_env.regions
        assert np.allclose(loaded_env.regions["goal"].data, [5.0, 5.0])

    def test_env_to_file_method(self, simple_env, tmp_path):
        """Test Environment.to_file() method."""
        output_path = tmp_path / "test_env"
        simple_env.to_file(output_path)

        assert (tmp_path / "test_env.json").exists()
        assert (tmp_path / "test_env.npz").exists()

    def test_env_from_file_classmethod(self, simple_env, tmp_path):
        """Test Environment.from_file() classmethod."""
        output_path = tmp_path / "test_env"
        simple_env.to_file(output_path)

        loaded_env = Environment.from_file(output_path)
        assert loaded_env.n_bins == simple_env.n_bins

    def test_to_dict_creates_valid_dict(self, simple_env):
        """Test that to_dict creates a JSON-serializable dict."""
        data = to_dict(simple_env)

        assert isinstance(data, dict)
        assert "schema_version" in data
        assert "bin_centers" in data
        assert isinstance(data["bin_centers"], list)

        # Ensure it's JSON serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_from_dict_reconstructs_environment(self, simple_env):
        """Test that from_dict correctly reconstructs environment."""
        data = to_dict(simple_env)
        loaded_env = from_dict(data)

        assert loaded_env.name == simple_env.name
        assert loaded_env.n_bins == simple_env.n_bins
        assert np.allclose(loaded_env.bin_centers, simple_env.bin_centers)

    def test_env_to_dict_method(self, simple_env):
        """Test Environment.to_dict() method."""
        data = simple_env.to_dict()
        assert isinstance(data, dict)
        assert "bin_centers" in data

    def test_env_from_dict_classmethod(self, simple_env):
        """Test Environment.from_dict() classmethod."""
        data = simple_env.to_dict()
        loaded_env = Environment.from_dict(data)
        assert loaded_env.n_bins == simple_env.n_bins

    def test_missing_files_raise_error(self, tmp_path):
        """Test that from_file raises error when files don't exist."""
        with pytest.raises(FileNotFoundError):
            from_file(tmp_path / "nonexistent")

    def test_serialization_without_units_frame(self, tmp_path):
        """Test serialization when units/frame are not set."""
        np.random.seed(42)
        data = np.random.randn(100, 2) * 10
        env = Environment.from_samples(data, bin_size=5.0, name="no_metadata")

        output_path = tmp_path / "test_env"
        to_file(env, output_path)

        loaded_env = from_file(output_path)
        assert loaded_env.n_bins == env.n_bins
        # units and frame should be None or not set
        assert not hasattr(loaded_env, "units") or loaded_env.units is None


class TestSecurityPathTraversal:
    """Test security measures against path traversal attacks."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 2) * 10
        return Environment.from_samples(data, bin_size=3.0, name="test_env")

    def test_to_file_rejects_path_traversal(self, simple_env, tmp_path):
        """Test that to_file() rejects paths with parent directory traversal."""
        # Try various path traversal attack vectors
        attack_paths = [
            tmp_path / ".." / ".." / "etc" / "passwd",
            tmp_path / ".." / "sensitive_file",
            "../../../etc/passwd",
            "valid_dir/../../../etc/passwd",
        ]

        for attack_path in attack_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                to_file(simple_env, attack_path)

    def test_to_file_rejects_symlink_attacks(self, simple_env, tmp_path):
        """Test that to_file() validates against symlink attacks."""
        # Skip if symlink creation not supported (Windows without admin)
        try:
            external_dir = tmp_path.parent / "external"
            external_dir.mkdir(exist_ok=True)
            symlink_path = tmp_path / "symlink_to_external"
            symlink_path.symlink_to(external_dir)

            # Should reject paths through symlinks that escape tmp_path
            with pytest.raises(ValueError, match="Path traversal detected"):
                to_file(simple_env, symlink_path / ".." / ".." / "etc" / "passwd")
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not supported on this platform")

    def test_from_file_rejects_path_traversal(self, tmp_path):
        """Test that from_file() rejects paths with parent directory traversal."""
        # Try various path traversal attack vectors
        attack_paths = [
            tmp_path / ".." / ".." / "etc" / "passwd",
            tmp_path / ".." / "sensitive_file",
            "../../../etc/passwd",
        ]

        for attack_path in attack_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                from_file(attack_path)

    def test_to_file_accepts_safe_paths(self, simple_env, tmp_path):
        """Test that to_file() accepts legitimate paths without '..'."""
        # These should all work
        safe_paths = [
            tmp_path / "env",
            tmp_path / "subdir" / "env",
            tmp_path / "deep" / "nested" / "path" / "env",
        ]

        for safe_path in safe_paths:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            to_file(simple_env, safe_path)
            assert safe_path.with_suffix(".json").exists()
            assert safe_path.with_suffix(".npz").exists()

    def test_absolute_paths_without_traversal_accepted(self, simple_env, tmp_path):
        """Test that absolute paths without '..' are accepted."""
        absolute_path = tmp_path.resolve() / "safe_env"
        to_file(simple_env, absolute_path)
        assert absolute_path.with_suffix(".json").exists()


class TestPathlibSupport:
    """Test that I/O functions accept both str and pathlib.Path objects."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 2) * 10
        env = Environment.from_samples(data, bin_size=3.0, name="test_env")
        env.units = "cm"
        return env

    def test_to_file_accepts_str_path(self, simple_env, tmp_path):
        """Test that to_file() accepts string paths."""
        # Use string path
        output_path = str(tmp_path / "test_env_str")
        to_file(simple_env, output_path)

        # Verify files created
        assert Path(output_path).with_suffix(".json").exists()
        assert Path(output_path).with_suffix(".npz").exists()

    def test_to_file_accepts_path_object(self, simple_env, tmp_path):
        """Test that to_file() accepts pathlib.Path objects."""
        # Use Path object
        output_path = tmp_path / "test_env_path"
        to_file(simple_env, output_path)

        # Verify files created
        assert output_path.with_suffix(".json").exists()
        assert output_path.with_suffix(".npz").exists()

    def test_from_file_accepts_str_path(self, simple_env, tmp_path):
        """Test that from_file() accepts string paths."""
        # Save with Path
        output_path = tmp_path / "test_env"
        to_file(simple_env, output_path)

        # Load with string
        loaded_env = from_file(str(output_path))
        assert loaded_env.n_bins == simple_env.n_bins

    def test_from_file_accepts_path_object(self, simple_env, tmp_path):
        """Test that from_file() accepts pathlib.Path objects."""
        # Save with string
        output_path = str(tmp_path / "test_env")
        to_file(simple_env, output_path)

        # Load with Path
        loaded_env = from_file(Path(output_path))
        assert loaded_env.n_bins == simple_env.n_bins

    def test_roundtrip_with_mixed_types(self, simple_env, tmp_path):
        """Test save with str, load with Path and vice versa."""
        # Save with str, load with Path
        str_path = str(tmp_path / "test_str")
        to_file(simple_env, str_path)
        env1 = from_file(Path(str_path))
        assert env1.n_bins == simple_env.n_bins

        # Save with Path, load with str
        path_obj = tmp_path / "test_path"
        to_file(simple_env, path_obj)
        env2 = from_file(str(path_obj))
        assert env2.n_bins == simple_env.n_bins

    def test_relative_path_support(self, simple_env, tmp_path, monkeypatch):
        """Test that relative paths work correctly."""
        # Change to tmp_path directory
        monkeypatch.chdir(tmp_path)

        # Use relative path
        relative_path = Path("subdir") / "test_env"
        relative_path.parent.mkdir(exist_ok=True)

        to_file(simple_env, relative_path)
        assert relative_path.with_suffix(".json").exists()

        loaded_env = from_file(relative_path)
        assert loaded_env.n_bins == simple_env.n_bins

    def test_home_directory_expansion(self, simple_env, tmp_path):
        """Test that Path.home() works correctly."""
        # Create a subdirectory in tmp_path (can't actually write to real home)
        # But we can verify Path objects work
        output_path = tmp_path / "home_test" / "env"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        to_file(simple_env, output_path)
        assert output_path.with_suffix(".json").exists()

    def test_env_to_file_method_with_path_object(self, simple_env, tmp_path):
        """Test Environment.to_file() works with Path objects."""
        output_path = tmp_path / "test_env"
        simple_env.to_file(output_path)
        assert output_path.with_suffix(".json").exists()

    def test_env_from_file_classmethod_with_path_object(self, simple_env, tmp_path):
        """Test Environment.from_file() works with Path objects."""
        output_path = tmp_path / "test_env"
        simple_env.to_file(output_path)

        loaded_env = Environment.from_file(output_path)
        assert loaded_env.n_bins == simple_env.n_bins
