"""Tests for neurospatial.io serialization functionality."""

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tomllib

import neurospatial
from neurospatial import Environment
from neurospatial.io import from_dict, from_file, to_dict, to_file


def test_nwb_docstring_only_references_real_extras() -> None:
    """The io.nwb install hint must name extras that actually exist.

    Regression: the docstring advertised ``neurospatial[nwb-full]``, but only an
    ``nwb`` extra is defined -- following that hint gives a bad-extra error.
    """
    import neurospatial.io.nwb as nwb_module

    root = Path(neurospatial.__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    defined = set(pyproject["project"]["optional-dependencies"])

    referenced = set(
        re.findall(r"neurospatial\[([a-z0-9-]+)\]", nwb_module.__doc__ or "")
    )
    assert referenced, "expected the io.nwb docstring to document an install extra"
    assert referenced <= defined, (
        f"io.nwb docstring references undefined extra(s): "
        f"{sorted(referenced - defined)}"
    )


class TestSerialization:
    """Test Environment serialization to/from files."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((500, 2)) * 10
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

    def test_missing_files_raise_error(self, tmp_path):
        """Test that from_file raises error when files don't exist."""
        with pytest.raises(FileNotFoundError):
            from_file(tmp_path / "nonexistent")

    def test_serialization_without_units_frame(self, tmp_path):
        """Test serialization when units/frame are not set."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2)) * 10
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
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2)) * 10
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
    """``to_file`` / ``from_file`` accept both ``str`` and ``pathlib.Path``."""

    @pytest.fixture
    def simple_env(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=3.0, name="test_env")
        env.units = "cm"
        return env

    @pytest.mark.parametrize("save_as_str", [True, False])
    @pytest.mark.parametrize("load_as_str", [True, False])
    def test_path_roundtrip(self, simple_env, tmp_path, save_as_str, load_as_str):
        """All four combinations of str / Path on save and load round-trip."""
        base = tmp_path / "test_env"
        to_file(simple_env, str(base) if save_as_str else base)
        assert base.with_suffix(".json").exists()
        assert base.with_suffix(".npz").exists()

        loaded = from_file(str(base) if load_as_str else base)
        assert loaded.n_bins == simple_env.n_bins

    def test_relative_path_support(self, simple_env, tmp_path, monkeypatch):
        """Relative paths resolve against the cwd at call time."""
        monkeypatch.chdir(tmp_path)
        relative_path = Path("subdir") / "test_env"
        relative_path.parent.mkdir(exist_ok=True)
        to_file(simple_env, relative_path)
        assert relative_path.with_suffix(".json").exists()
        assert from_file(relative_path).n_bins == simple_env.n_bins


class TestAtomicWriteAndDtypes:
    """Tests for atomic to_file writes and dtype guarantees on load."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple 2D environment for testing."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((300, 2)) * 8
        env = Environment.from_samples(data, bin_size=3.0, name="atomic_env")
        env.units = "cm"
        env.frame = "world"
        return env

    def test_npz_write_failure_leaves_no_partial_json(
        self, simple_env, tmp_path, monkeypatch
    ):
        """If the npz write fails, no dangling .json may be left behind.

        Previously to_file wrote the .json first and the .npz second, so a
        failure in between produced a .json with no matching .npz, which
        from_file then reported as a confusing 'array file not found' error.
        """
        import neurospatial.io.files as files_mod

        output_path = tmp_path / "atomic_env"

        def _boom(*_args, **_kwargs):
            raise OSError("simulated npz write failure")

        monkeypatch.setattr(files_mod.np, "savez_compressed", _boom)

        with pytest.raises(OSError, match="simulated npz write failure"):
            to_file(simple_env, output_path)

        # Neither the final .json nor the final .npz must exist.
        assert not (tmp_path / "atomic_env.json").exists()
        assert not (tmp_path / "atomic_env.npz").exists()
        # No temp leftovers either.
        leftovers = list(tmp_path.iterdir())
        assert leftovers == [], f"Unexpected leftover files: {leftovers}"

    def test_from_file_forces_bin_centers_float64(self, simple_env, tmp_path):
        """from_file must coerce bin_centers to float64 even if stored float32.

        The in-memory from_dict path enforces float64; from_file used to load
        whatever dtype was in the .npz, bypassing that guarantee.
        """
        import json

        import networkx as nx

        output_path = tmp_path / "f32_env"
        to_file(simple_env, output_path)

        # Rewrite the .npz with bin_centers stored as float32.
        npz_path = tmp_path / "f32_env.npz"
        with np.load(npz_path) as arrays:
            data = {k: arrays[k] for k in arrays.files}
        data["bin_centers"] = data["bin_centers"].astype(np.float32)
        np.savez_compressed(str(npz_path), **data)

        # Sanity: the stored array really is float32 now.
        with np.load(npz_path) as arrays:
            assert arrays["bin_centers"].dtype == np.float32

        # Keep json/graph valid (untouched) and load.
        with (tmp_path / "f32_env.json").open() as f:
            meta = json.load(f)
        assert "graph" in meta
        nx.node_link_graph(meta["graph"], edges="links")  # validates structure

        loaded = from_file(output_path)
        assert loaded.bin_centers.dtype == np.float64

    def test_string_list_layout_param_survives_round_trip(self, simple_env, tmp_path):
        """A list-of-strings layout parameter must not be coerced to an object array.

        _convert_lists_to_arrays used to blindly np.array() every list, turning
        a list of strings into a dtype=object numpy array on load.
        """
        from neurospatial.io.files import _convert_lists_to_arrays

        params = {
            "labels": ["north", "south", "east"],
            "numeric": [1.0, 2.0, 3.0],
            "nested": {"more_labels": ["a", "b"]},
        }
        restored = _convert_lists_to_arrays(params)

        # String lists stay as plain Python lists of strings.
        assert restored["labels"] == ["north", "south", "east"]
        assert all(isinstance(x, str) for x in restored["labels"])
        assert restored["nested"]["more_labels"] == ["a", "b"]
        # Numeric lists are still converted to numeric arrays.
        assert isinstance(restored["numeric"], np.ndarray)
        assert np.issubdtype(restored["numeric"].dtype, np.floating)


def _make_graph_env() -> Environment:
    """Build a 1D linearized-track (Graph layout) environment."""
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (0, {"pos": (0.0,)}),
            (1, {"pos": (10.0,)}),
            (2, {"pos": (20.0,)}),
            (3, {"pos": (30.0,)}),
        ]
    )
    graph.add_edge(0, 1, distance=10.0)
    graph.add_edge(1, 2, distance=10.0)
    graph.add_edge(2, 3, distance=10.0)
    return Environment.from_graph(
        graph, [(0, 1), (1, 2), (2, 3)], edge_spacing=0.0, bin_size=2.0
    )


def _make_polygon_env() -> Environment:
    """Build a ShapelyPolygon-layout environment."""
    shapely_geom = pytest.importorskip("shapely.geometry")
    return Environment.from_polygon(shapely_geom.box(0, 0, 20, 20), bin_size=4.0)


def _make_hexagonal_env() -> Environment:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (300, 2))
    return Environment.from_samples(positions, bin_size=5.0, layout="hexagonal")


def _make_masked_env() -> Environment:
    active_mask = np.ones((5, 5), dtype=bool)
    active_mask[0, 0] = False
    grid_edges = (np.arange(6.0), np.arange(6.0))
    return Environment.from_grid_mask(active_mask, grid_edges)


def _make_image_mask_env() -> Environment:
    image = np.zeros((8, 8), dtype=bool)
    image[2:6, 2:6] = True
    return Environment.from_pixel_mask(image, pixel_size=1.0)


def _make_3d_env() -> Environment:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (400, 3))
    return Environment.from_samples(positions, bin_size=10.0)


_LAYOUT_FACTORIES = {
    "Graph": _make_graph_env,
    "Polygon": _make_polygon_env,
    "Hexagonal": _make_hexagonal_env,
    "Masked": _make_masked_env,
    "ImageMask": _make_image_mask_env,
    "3D": _make_3d_env,
}


class TestAllLayoutRoundTrip:
    """Every layout factory must survive a to_file/from_file round-trip.

    Previously only RegularGrid was exercised; Graph and Polygon crashed at
    write time because layout_parameters held a non-JSON-serializable
    networkx.Graph / shapely geometry.
    """

    @pytest.mark.parametrize("layout_name", list(_LAYOUT_FACTORIES))
    def test_to_file_roundtrip_all_layouts(self, layout_name, tmp_path):
        env = _LAYOUT_FACTORIES[layout_name]()

        output_path = tmp_path / f"env_{layout_name}"
        to_file(env, output_path)
        loaded = from_file(output_path)

        assert loaded.n_bins == env.n_bins
        assert loaded.n_dims == env.n_dims
        assert loaded.is_linearized_track == env.is_linearized_track
        np.testing.assert_allclose(
            np.sort(loaded.bin_centers, axis=0),
            np.sort(env.bin_centers, axis=0),
        )
        # Edge set is preserved (compare as undirected frozenset pairs).
        orig_edges = {frozenset(e) for e in env.connectivity.edges()}
        loaded_edges = {frozenset(e) for e in loaded.connectivity.edges()}
        assert loaded_edges == orig_edges
        if env.active_mask is not None:
            assert loaded.active_mask is not None
            np.testing.assert_array_equal(
                np.sort(loaded.active_mask.ravel()),
                np.sort(env.active_mask.ravel()),
            )

    def test_to_dict_roundtrip_graph_layout(self):
        """to_dict(graph_env) is json.dumps-able and from_dict reconstructs it."""
        env = _make_graph_env()

        env_dict = to_dict(env)
        json.dumps(env_dict)  # must not raise

        restored = from_dict(env_dict)
        assert restored.n_bins == env.n_bins
        assert restored.is_linearized_track == env.is_linearized_track
        np.testing.assert_allclose(
            np.sort(restored.bin_centers, axis=0),
            np.sort(env.bin_centers, axis=0),
        )

    def test_to_file_graph_layout_parameters_roundtrip(self, tmp_path):
        """layout_parameters['graph_definition'] decodes back to an nx.Graph."""
        env = _make_graph_env()

        output_path = tmp_path / "graph_env"
        to_file(env, output_path)
        loaded = from_file(output_path)

        graph_def = loaded.layout_parameters["graph_definition"]
        assert isinstance(graph_def, nx.Graph)
        assert not isinstance(graph_def, dict)


class TestAnisotropicPixelMaskRoundTrip:
    """Anisotropic from_pixel_mask envs must survive serialization.

    Regression: ``to_dict`` serialized ``pixel_size`` as a list, ``from_dict``
    rebuilt it as an ndarray, and the scalar ``pixel_size <= 0`` validation in
    ImageMaskLayout.build then raised "The truth value of an array with more
    than one element is ambiguous". The validation is now array-safe.
    """

    @staticmethod
    def _make_anisotropic_env() -> Environment:
        image = np.zeros((5, 4), dtype=bool)
        image[1:4, 1:3] = True
        return Environment.from_pixel_mask(image, pixel_size=(3.0, 1.0))

    def test_to_dict_from_dict_roundtrip(self):
        env = self._make_anisotropic_env()

        env_dict = to_dict(env)
        json.dumps(env_dict)  # must not raise

        # Fail-before/pass-after: from_dict previously raised the
        # ambiguous-truth-value ValueError here.
        restored = from_dict(env_dict)

        assert restored.grid_shape == env.grid_shape
        np.testing.assert_allclose(
            np.sort(restored.bin_centers, axis=0),
            np.sort(env.bin_centers, axis=0),
        )

    def test_to_file_from_file_roundtrip(self, tmp_path):
        env = self._make_anisotropic_env()

        output_path = tmp_path / "anisotropic_image_mask"
        to_file(env, output_path)
        loaded = from_file(output_path)

        assert loaded.grid_shape == env.grid_shape
        np.testing.assert_allclose(
            np.sort(loaded.bin_centers, axis=0),
            np.sort(env.bin_centers, axis=0),
        )
