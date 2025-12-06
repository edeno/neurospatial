"""
Tests for NWB environment functions.

Tests for:
- environment_from_position() - creates an Environment from NWB Position data
- write_environment() - writes Environment to NWB scratch space
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class TestEnvironmentFromPosition:
    """Tests for environment_from_position() function."""

    def test_basic_environment_creation(self, sample_nwb_with_position):
        """Test basic Environment creation from Position data."""
        from neurospatial.io.nwb import environment_from_position

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should create a valid environment
        assert env is not None
        assert env.n_bins > 0
        assert env.bin_centers.shape[1] == 2  # 2D environment

    def test_environment_matches_position_data_bounds(self, sample_nwb_with_position):
        """Test Environment extent matches Position data bounds."""
        from neurospatial.io.nwb import environment_from_position, read_position

        # Get position data for comparison
        positions, _ = read_position(sample_nwb_with_position)

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Environment should cover the position data extent
        # bin_centers should fall within the position data range
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)

        # All bin centers should be near the position data range
        for dim in range(2):
            assert env.bin_centers[:, dim].min() >= pos_min[dim] - 5.0
            assert env.bin_centers[:, dim].max() <= pos_max[dim] + 5.0

    def test_units_parameter_propagation(self, sample_nwb_with_position):
        """Test units parameter is set on Environment."""
        from neurospatial.io.nwb import environment_from_position

        env = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, units="cm"
        )

        assert env.units == "cm"

    def test_units_defaults_from_spatial_series(self, sample_nwb_with_position):
        """Test units are auto-detected from SpatialSeries when not specified."""
        from neurospatial.io.nwb import environment_from_position

        # The fixture has unit="cm" in the SpatialSeries
        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should auto-detect units from SpatialSeries
        assert env.units == "cm"

    def test_frame_parameter_propagation(self, sample_nwb_with_position):
        """Test frame parameter is set on Environment."""
        from neurospatial.io.nwb import environment_from_position

        env = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, frame="session_001"
        )

        assert env.frame == "session_001"

    def test_infer_active_bins_parameter(self, sample_nwb_with_position):
        """Test infer_active_bins parameter is forwarded."""
        from neurospatial.io.nwb import environment_from_position

        # With infer_active_bins=True, should only include visited bins
        env_active = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, infer_active_bins=True
        )

        # With infer_active_bins=False (default), should include all bins in extent
        env_all = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, infer_active_bins=False
        )

        # Active bins environment should have same or fewer bins
        assert env_active.n_bins <= env_all.n_bins

    def test_kwargs_forwarded_to_from_samples(self, sample_nwb_with_position):
        """Test additional kwargs are forwarded to Environment.from_samples()."""
        from neurospatial.io.nwb import environment_from_position

        # Test bin_count_threshold parameter
        env = environment_from_position(
            sample_nwb_with_position,
            bin_size=5.0,
            infer_active_bins=True,
            bin_count_threshold=5,  # Require at least 5 samples per bin
        )

        # Should create environment with fewer bins due to threshold
        assert env is not None
        assert env.n_bins > 0

    def test_processing_module_parameter(self, sample_nwb_with_position):
        """Test processing_module parameter is forwarded to read_position."""
        from neurospatial.io.nwb import environment_from_position

        # Should work when specifying the correct module
        env = environment_from_position(
            sample_nwb_with_position,
            bin_size=5.0,
            processing_module="behavior",
        )

        assert env is not None
        assert env.n_bins > 0

    def test_position_name_parameter(self, sample_nwb_with_position_multiple_series):
        """Test position_name parameter selects specific SpatialSeries."""
        from neurospatial.io.nwb import environment_from_position

        # Select 'head' position (one of the two available)
        env = environment_from_position(
            sample_nwb_with_position_multiple_series,
            bin_size=5.0,
            position_name="head",
        )

        assert env is not None
        assert env.n_bins > 0

    def test_error_when_position_not_found(self, empty_nwb):
        """Test KeyError when Position not found in NWB file."""
        from neurospatial.io.nwb import environment_from_position

        with pytest.raises(KeyError, match="No Position data found"):
            environment_from_position(empty_nwb, bin_size=5.0)

    def test_error_when_processing_module_not_found(self, sample_nwb_with_position):
        """Test KeyError when specified processing module not found."""
        from neurospatial.io.nwb import environment_from_position

        with pytest.raises(KeyError, match="Processing module 'nonexistent' not found"):
            environment_from_position(
                sample_nwb_with_position,
                bin_size=5.0,
                processing_module="nonexistent",
            )

    def test_bin_size_required(self, sample_nwb_with_position):
        """Test that bin_size parameter is required."""
        from neurospatial.io.nwb import environment_from_position

        with pytest.raises(TypeError):
            environment_from_position(sample_nwb_with_position)  # Missing bin_size

    def test_different_bin_sizes(self, sample_nwb_with_position):
        """Test Environment creation with different bin sizes."""
        from neurospatial.io.nwb import environment_from_position

        env_small = environment_from_position(sample_nwb_with_position, bin_size=2.0)
        env_large = environment_from_position(sample_nwb_with_position, bin_size=10.0)

        # Smaller bins should result in more bins
        assert env_small.n_bins > env_large.n_bins

    def test_environment_is_fitted(self, sample_nwb_with_position):
        """Test that returned Environment is fitted and ready to use."""
        from neurospatial.io.nwb import environment_from_position

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should be able to call methods that require fitted state
        assert hasattr(env, "_is_fitted")
        assert env._is_fitted

        # Should be able to use spatial query methods
        point = env.bin_centers[0]
        bin_idx = env.bin_at(point)
        assert bin_idx >= 0

    def test_connectivity_graph_created(self, sample_nwb_with_position):
        """Test that connectivity graph is properly created."""
        from neurospatial.io.nwb import environment_from_position

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should have connectivity graph
        assert env.connectivity is not None
        assert env.connectivity.number_of_nodes() == env.n_bins
        assert env.connectivity.number_of_edges() > 0


class TestWriteEnvironment:
    """Tests for write_environment() function."""

    def test_basic_environment_writing(self, empty_nwb, sample_environment):
        """Test basic Environment writing to scratch/."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment, name="test_env")

        # Should exist in scratch
        assert "test_env" in empty_nwb.scratch
        scratch_data = empty_nwb.scratch["test_env"]
        assert scratch_data is not None

    def test_bin_centers_dataset(self, empty_nwb, sample_environment):
        """Test bin_centers dataset stored with correct shape (n_bins, n_dims)."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Access the bin_centers from the DynamicTable columns
        assert "bin_centers" in scratch_data.colnames
        bin_centers = scratch_data["bin_centers"][:]
        # Data is padded, so check first n_bins rows
        n_bins = sample_environment.n_bins
        np.testing.assert_array_almost_equal(
            bin_centers[:n_bins], sample_environment.bin_centers
        )

    def test_edges_dataset_as_edge_list(self, empty_nwb, sample_environment):
        """Test edges dataset stored as edge list (n_edges, 2)."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert "edges" in scratch_data.colnames
        edges = scratch_data["edges"][:]

        # Should have shape (n_rows, 2) - padded to n_rows
        assert edges.ndim == 2
        assert edges.shape[1] == 2

        # Get actual edges (non-zero rows)
        expected_n_edges = sample_environment.connectivity.number_of_edges()
        # Check that the first n_edges rows contain valid edges
        actual_edges = edges[:expected_n_edges]
        assert actual_edges.shape[0] == expected_n_edges

    def test_edge_weights_dataset(self, empty_nwb, sample_environment):
        """Test edge_weights dataset stored with correct shape (n_edges,)."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert "edge_weights" in scratch_data.colnames
        edge_weights = scratch_data["edge_weights"][:]

        # Data is padded, get first n_edges values
        expected_n_edges = sample_environment.connectivity.number_of_edges()
        actual_weights = edge_weights[:expected_n_edges]

        # Weights should be non-negative (distances)
        assert np.all(actual_weights >= 0)

    def test_dimension_ranges_dataset(self, empty_nwb, sample_environment):
        """Test dimension_ranges dataset stored with shape (n_dims, 2)."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert "dimension_ranges" in scratch_data.colnames
        dim_ranges = scratch_data["dimension_ranges"][:]

        # Data is padded, get first n_dims rows
        n_dims = sample_environment.bin_centers.shape[1]

        # Check values match environment dimension_ranges
        np.testing.assert_array_almost_equal(
            dim_ranges[:n_dims], sample_environment.dimension_ranges
        )

    def test_group_attributes_units(self, empty_nwb, sample_environment):
        """Test units attribute is stored on group."""
        from neurospatial.io.nwb import write_environment

        sample_environment.units = "cm"
        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Access via description or comments which hold metadata
        assert (
            "cm" in scratch_data.description or "units=cm" in scratch_data.description
        )

    def test_group_attributes_frame(self, empty_nwb, sample_environment):
        """Test frame attribute is stored on group."""
        from neurospatial.io.nwb import write_environment

        sample_environment.frame = "session_001"
        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert (
            "session_001" in scratch_data.description
            or "frame=session_001" in scratch_data.description
        )

    def test_group_attributes_n_dims(self, empty_nwb, sample_environment):
        """Test n_dims attribute is stored."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # n_dims should be in the description or metadata
        n_dims = sample_environment.bin_centers.shape[1]
        assert f"n_dims={n_dims}" in scratch_data.description

    def test_group_attributes_layout_type(self, empty_nwb, sample_environment):
        """Test layout_type attribute is stored."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Layout type should be stored in description
        layout_type = sample_environment.layout._layout_type_tag
        assert layout_type in scratch_data.description

    def test_default_name(self, empty_nwb, sample_environment):
        """Test default name is 'spatial_environment'."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        assert "spatial_environment" in empty_nwb.scratch

    def test_custom_name(self, empty_nwb, sample_environment):
        """Test custom name parameter."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment, name="linear_track")

        assert "linear_track" in empty_nwb.scratch
        assert "spatial_environment" not in empty_nwb.scratch

    def test_duplicate_name_error(self, empty_nwb, sample_environment):
        """Test ValueError when writing duplicate name without overwrite."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        with pytest.raises(ValueError, match="already exists"):
            write_environment(empty_nwb, sample_environment)

    def test_overwrite_replaces_existing(self, empty_nwb, sample_environment):
        """Test overwrite=True replaces existing environment."""
        from neurospatial.io.nwb import write_environment

        # Write initial environment
        write_environment(empty_nwb, sample_environment)

        # Modify environment
        modified_env = sample_environment
        modified_env.units = "meters"

        # Should succeed with overwrite=True
        write_environment(empty_nwb, modified_env, overwrite=True)

        # Check new data is present
        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert "meters" in scratch_data.description

    def test_point_regions_stored(self, empty_nwb, sample_environment):
        """Test point regions are stored correctly."""
        from neurospatial.io.nwb import write_environment

        # Ensure we have point regions
        sample_environment.regions.add("center", point=(50.0, 50.0))

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Regions should be stored as column
        assert "regions" in scratch_data.colnames

    def test_polygon_regions_stored(self, empty_nwb):
        """Test polygon regions are stored correctly with ragged vertices."""
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        # Create environment with polygon region
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Add polygon region
        triangle = Polygon([(10, 10), (30, 10), (20, 30)])
        env.regions.add("reward_zone", polygon=triangle)

        write_environment(empty_nwb, env)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert "regions" in scratch_data.colnames

    def test_empty_regions(self, empty_nwb):
        """Test environment with no regions."""
        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        # No regions added

        # Should work without error
        write_environment(empty_nwb, env)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert scratch_data is not None

    def test_metadata_json_stored(self, empty_nwb, sample_environment):
        """Test metadata.json stored for extra attributes."""
        from neurospatial.io.nwb import write_environment

        sample_environment.name = "test_arena"

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Metadata JSON should be stored as column
        assert "metadata" in scratch_data.colnames

    def test_data_integrity_bin_centers(self, empty_nwb, sample_environment):
        """Test bin_centers data integrity after write."""
        from neurospatial.io.nwb import write_environment

        original_bin_centers = sample_environment.bin_centers.copy()

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        stored_bin_centers = scratch_data["bin_centers"][:]
        n_bins = sample_environment.n_bins

        np.testing.assert_array_equal(stored_bin_centers[:n_bins], original_bin_centers)

    def test_data_integrity_edges(self, empty_nwb, sample_environment):
        """Test edges data integrity after write."""
        from neurospatial.io.nwb import write_environment

        # Get original edges from connectivity graph
        original_edges = np.array(list(sample_environment.connectivity.edges()))

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        stored_edges = scratch_data["edges"][:]
        n_edges = sample_environment.connectivity.number_of_edges()

        # Get only the actual edges (first n_edges rows)
        stored_edges = stored_edges[:n_edges]

        # Edges should match (potentially different order, so sort)
        original_sorted = np.sort(original_edges, axis=1)
        stored_sorted = np.sort(stored_edges, axis=1)
        original_sorted = original_sorted[np.lexsort(original_sorted.T)]
        stored_sorted = stored_sorted[np.lexsort(stored_sorted.T)]

        np.testing.assert_array_equal(stored_sorted, original_sorted)

    def test_alternative_2d_environment(self, empty_nwb):
        """Test writing environment created with different parameters."""
        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        # Create 2D environment with different parameters
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Should work without error
        write_environment(empty_nwb, env)

        assert "spatial_environment" in empty_nwb.scratch

    def test_error_on_unfitted_environment(self, empty_nwb, sample_environment):
        """Test ValueError when writing unfitted Environment."""
        from neurospatial.io.nwb import write_environment

        # Manually mark environment as unfitted to test validation
        sample_environment._is_fitted = False

        with pytest.raises(ValueError, match="must be fitted"):
            write_environment(empty_nwb, sample_environment)


class TestReadEnvironment:
    """Tests for read_environment() function."""

    def test_basic_environment_reading(self, empty_nwb, sample_environment):
        """Test basic Environment reading from scratch/."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        assert loaded_env is not None
        assert loaded_env.n_bins == sample_environment.n_bins

    def test_bin_centers_reconstruction(self, empty_nwb, sample_environment):
        """Test bin_centers are correctly reconstructed."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        np.testing.assert_array_almost_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_connectivity_graph_reconstruction(self, empty_nwb, sample_environment):
        """Test connectivity graph is reconstructed from edge list."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        # Same number of nodes and edges
        assert (
            loaded_env.connectivity.number_of_nodes()
            == sample_environment.connectivity.number_of_nodes()
        )
        assert (
            loaded_env.connectivity.number_of_edges()
            == sample_environment.connectivity.number_of_edges()
        )

    def test_edge_weights_applied(self, empty_nwb, sample_environment):
        """Test edge weights (distances) are restored on graph."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        # Check that edges have distance attribute
        for _u, _v, data in loaded_env.connectivity.edges(data=True):
            assert "distance" in data
            assert data["distance"] >= 0

    def test_dimension_ranges_reconstruction(self, empty_nwb, sample_environment):
        """Test dimension_ranges are correctly reconstructed."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        np.testing.assert_array_almost_equal(
            loaded_env.dimension_ranges, sample_environment.dimension_ranges
        )

    def test_units_attribute_restored(self, empty_nwb, sample_environment):
        """Test units attribute is restored."""
        from neurospatial.io.nwb import read_environment, write_environment

        sample_environment.units = "cm"
        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        assert loaded_env.units == "cm"

    def test_frame_attribute_restored(self, empty_nwb, sample_environment):
        """Test frame attribute is restored."""
        from neurospatial.io.nwb import read_environment, write_environment

        sample_environment.frame = "session_001"
        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        assert loaded_env.frame == "session_001"

    def test_point_regions_restored(self, empty_nwb, sample_environment):
        """Test point regions are correctly restored."""
        from neurospatial.io.nwb import read_environment, write_environment

        # sample_environment already has point regions from fixture
        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        # Check regions exist
        assert "start" in loaded_env.regions
        assert "goal" in loaded_env.regions

        # Check point region data
        assert loaded_env.regions["start"].kind == "point"
        np.testing.assert_array_almost_equal(
            loaded_env.regions["start"].data, [10.0, 10.0]
        )

    def test_polygon_regions_restored(self, empty_nwb):
        """Test polygon regions are correctly restored."""
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create environment with polygon region
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        triangle = Polygon([(10, 10), (30, 10), (20, 30)])
        env.regions.add("reward_zone", polygon=triangle)

        write_environment(empty_nwb, env)
        loaded_env = read_environment(empty_nwb)

        # Check polygon region exists and has correct kind
        assert "reward_zone" in loaded_env.regions
        assert loaded_env.regions["reward_zone"].kind == "polygon"

        # Check polygon coordinates are restored
        loaded_coords = list(loaded_env.regions["reward_zone"].data.exterior.coords)
        original_coords = list(triangle.exterior.coords)
        np.testing.assert_array_almost_equal(loaded_coords, original_coords)

    def test_error_when_environment_not_found(self, empty_nwb):
        """Test KeyError when environment not found in scratch/."""
        from neurospatial.io.nwb import read_environment

        with pytest.raises(KeyError, match="not found"):
            read_environment(empty_nwb, name="nonexistent")

    def test_custom_name_parameter(self, empty_nwb, sample_environment):
        """Test reading environment with custom name."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment, name="linear_track")
        loaded_env = read_environment(empty_nwb, name="linear_track")

        assert loaded_env is not None
        assert loaded_env.n_bins == sample_environment.n_bins

    def test_environment_is_fitted(self, empty_nwb, sample_environment):
        """Test loaded Environment is fitted and ready to use."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        # Should be fitted
        assert loaded_env._is_fitted

        # Should be able to use spatial queries
        point = loaded_env.bin_centers[0]
        bin_idx = loaded_env.bin_at(point)
        assert bin_idx >= 0

    def test_empty_regions_handled(self, empty_nwb):
        """Test environment with no regions loads correctly."""
        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        # No regions added

        write_environment(empty_nwb, env)
        loaded_env = read_environment(empty_nwb)

        assert loaded_env is not None
        assert len(loaded_env.regions) == 0

    def test_name_attribute_restored(self, empty_nwb, sample_environment):
        """Test environment name attribute is restored from metadata."""
        from neurospatial.io.nwb import read_environment, write_environment

        sample_environment.name = "test_arena"
        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        assert loaded_env.name == "test_arena"

    def test_graph_node_attributes(self, empty_nwb, sample_environment):
        """Test graph nodes have required attributes after loading."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        # Check required node attributes exist
        for node_id, node_data in loaded_env.connectivity.nodes(data=True):
            assert "pos" in node_data
            # pos should match bin_centers
            np.testing.assert_array_almost_equal(
                node_data["pos"], loaded_env.bin_centers[node_id]
            )


def _create_nwb_for_test():
    """Create a minimal NWB file for testing."""
    from datetime import datetime
    from uuid import uuid4

    from pynwb import NWBFile

    return NWBFile(
        session_description="Test session for round-trip",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )


class TestEnvironmentRoundTrip:
    """File-based round-trip tests: write to disk, read back."""

    def test_full_roundtrip_to_file(self, tmp_path, sample_environment):
        """Test Environment survives NWB write/read cycle to actual file."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "test_roundtrip.nwb"

        # Write to file
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        # Read from file
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Verify round-trip
        assert loaded_env.n_bins == sample_environment.n_bins
        np.testing.assert_array_almost_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_roundtrip_bin_centers_exact(self, tmp_path, sample_environment):
        """Test bin_centers are exactly preserved through file round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "test_bin_centers.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Exact match (no floating point drift)
        np.testing.assert_array_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_roundtrip_connectivity_preserved(self, tmp_path, sample_environment):
        """Test connectivity graph structure is preserved through file round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "test_connectivity.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Same structure
        assert (
            loaded_env.connectivity.number_of_nodes()
            == sample_environment.connectivity.number_of_nodes()
        )
        assert (
            loaded_env.connectivity.number_of_edges()
            == sample_environment.connectivity.number_of_edges()
        )

        # Same edges (sorted for comparison)
        original_edges = sorted(sample_environment.connectivity.edges())
        loaded_edges = sorted(loaded_env.connectivity.edges())
        assert original_edges == loaded_edges

    def test_roundtrip_edge_weights_preserved(self, tmp_path, sample_environment):
        """Test edge weights (distances) are preserved through file round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "test_edge_weights.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Compare edge weights
        for u, v in sample_environment.connectivity.edges():
            original_dist = sample_environment.connectivity[u][v]["distance"]
            loaded_dist = loaded_env.connectivity[u][v]["distance"]
            np.testing.assert_almost_equal(loaded_dist, original_dist)

    def test_roundtrip_regions_preserved(self, tmp_path, sample_environment):
        """Test regions are preserved through file round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "test_regions.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Same regions
        assert set(loaded_env.regions.keys()) == set(sample_environment.regions.keys())

        # Check region data
        for name in sample_environment.regions:
            original = sample_environment.regions[name]
            loaded = loaded_env.regions[name]
            assert loaded.kind == original.kind
            np.testing.assert_array_almost_equal(loaded.data, original.data)

    def test_roundtrip_metadata_preserved(self, tmp_path, sample_environment):
        """Test metadata (units, frame, name) is preserved through file round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        sample_environment.name = "test_arena"
        sample_environment.units = "cm"
        sample_environment.frame = "session_001"

        nwb_path = tmp_path / "test_metadata.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert loaded_env.name == "test_arena"
        assert loaded_env.units == "cm"
        assert loaded_env.frame == "session_001"

    def test_roundtrip_polygon_regions(self, tmp_path):
        """Test polygon regions are preserved through file round-trip."""
        from pynwb import NWBHDF5IO
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create environment with polygon region
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        triangle = Polygon([(10, 10), (30, 10), (20, 30)])
        env.regions.add("reward_zone", polygon=triangle)

        nwb_path = tmp_path / "test_polygon.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Check polygon coordinates
        loaded_coords = list(loaded_env.regions["reward_zone"].data.exterior.coords)
        original_coords = list(triangle.exterior.coords)
        np.testing.assert_array_almost_equal(loaded_coords, original_coords)

    def test_roundtrip_spatial_queries_work(self, tmp_path, sample_environment):
        """Test loaded Environment can perform spatial queries."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "test_queries.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Test bin_at works
        test_point = loaded_env.bin_centers[0]
        bin_idx = loaded_env.bin_at(test_point)
        assert bin_idx == 0

        # Test neighbors works
        neighbors = loaded_env.neighbors(0)
        assert len(neighbors) > 0

        # Test distance_between works
        if len(neighbors) > 0:
            dist = loaded_env.distance_between(0, neighbors[0])
            assert dist > 0

    def test_roundtrip_3d_environment(self, tmp_path):
        """Test 3D environment round-trip preserves all dimensions."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create 3D environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 50, (500, 3))
        env = Environment.from_samples(positions, bin_size=5.0)

        nwb_path = tmp_path / "test_3d.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Verify 3D structure preserved
        assert loaded_env.n_dims == 3
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

    def test_roundtrip_hexagonal_layout(self, tmp_path):
        """Test hexagonal layout environment round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create hexagonal layout environment
        env = Environment.from_layout(
            kind="Hexagonal",
            layout_params={
                "hexagon_width": 5.0,  # Hexagonal uses hexagon_width
                "dimension_ranges": [(0, 50), (0, 50)],
            },
        )

        nwb_path = tmp_path / "test_hex.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Core data should match exactly
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

        # Layout type should be stored (though layout can't be fully restored)
        assert loaded_env._layout_type_used == "Hexagonal"

    def test_roundtrip_polygon_boundary_environment(self, tmp_path):
        """Test polygon-bounded environment round-trip."""
        from pynwb import NWBHDF5IO
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create L-shaped polygon boundary
        boundary = Polygon([(0, 0), (50, 0), (50, 25), (25, 25), (25, 50), (0, 50)])
        env = Environment.from_polygon(boundary, bin_size=5.0)

        nwb_path = tmp_path / "test_polygon_boundary.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Core data should match exactly
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)

        # Connectivity should be preserved
        original_edges = sorted(env.connectivity.edges())
        loaded_edges = sorted(loaded_env.connectivity.edges())
        assert original_edges == loaded_edges

    def test_roundtrip_1d_graph_layout(self, tmp_path):
        """Test 1D GraphLayout (linearized track) environment round-trip."""
        import networkx as nx
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create a simple linear track graph
        # Track: node 0 -- node 1 -- node 2 -- node 3 -- node 4
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(i, pos=(i * 20.0, 0.0))  # Nodes at 0, 20, 40, 60, 80
        for i in range(4):
            # GraphLayout requires edges with 'distance' attribute
            graph.add_edge(i, i + 1, distance=20.0)

        edge_order = [(i, i + 1) for i in range(4)]
        env = Environment.from_graph(
            graph, edge_order=edge_order, edge_spacing=20.0, bin_size=5.0
        )

        assert env.is_1d, "GraphLayout should create a 1D environment"

        nwb_path = tmp_path / "test_1d_graph.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Core data should match exactly
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)

        # Connectivity should be preserved
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

        original_edges = sorted(env.connectivity.edges())
        loaded_edges = sorted(loaded_env.connectivity.edges())
        assert original_edges == loaded_edges

        # is_1d property should be preserved (Graph layouts store 2D bin_centers
        # but are conceptually 1D linearized tracks)
        assert loaded_env.is_1d is True, "is_1d should be restored for Graph layouts"
        assert env.is_1d is True, "Original env should be 1D"

        # n_dims is based on bin_centers shape (2D for projected coordinates)
        assert loaded_env.n_dims == env.n_dims

        # Layout type should be stored
        assert loaded_env._layout_type_used == "Graph"

    def test_roundtrip_masked_grid_layout(self, tmp_path):
        """Test MaskedGrid layout environment round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create a MaskedGrid layout with active region in the center
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True  # Active region in the center

        # Grid edges define the spatial extent
        x_edges = np.linspace(0, 50, 11)  # 10 bins from 0 to 50
        y_edges = np.linspace(0, 50, 11)

        env = Environment.from_mask(
            active_mask=mask,
            grid_edges=(x_edges, y_edges),
        )

        nwb_path = tmp_path / "test_masked_grid.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Core data should match exactly
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

        # Layout type should be stored
        assert loaded_env._layout_type_used == "MaskedGrid"

    def test_roundtrip_image_mask_layout(self, tmp_path):
        """Test ImageMask layout environment round-trip."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create a binary image mask (circular arena)
        y, x = np.ogrid[:50, :50]
        center = (25, 25)
        radius = 20
        mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius**2

        env = Environment.from_image(
            image_mask=mask,  # Boolean mask
            bin_size=2.0,
        )

        nwb_path = tmp_path / "test_image_mask.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Core data should match exactly
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

        # Layout type should be stored
        assert loaded_env._layout_type_used == "ImageMask"

    def test_roundtrip_triangular_mesh_layout(self, tmp_path):
        """Test TriangularMesh layout environment round-trip."""
        from pynwb import NWBHDF5IO
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create a triangular mesh layout with a simple boundary
        boundary = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])

        env = Environment.from_layout(
            kind="TriangularMesh",
            layout_params={
                "boundary_polygon": boundary,
                "point_spacing": 5.0,  # Controls mesh density
            },
        )

        nwb_path = tmp_path / "test_triangular_mesh.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Core data should match exactly
        assert loaded_env.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        )

        # Layout type should be stored
        assert loaded_env._layout_type_used == "TriangularMesh"


# =============================================================================
# Parametrized tests for all layout types
# =============================================================================


def _create_regular_grid_env():
    """Create a 2D RegularGrid environment for testing."""
    from neurospatial import Environment

    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (1000, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    return env


def _create_hexagonal_env():
    """Create a Hexagonal layout environment for testing."""
    from neurospatial import Environment

    env = Environment.from_layout(
        kind="Hexagonal",
        layout_params={
            "hexagon_width": 5.0,
            "dimension_ranges": [(0, 50), (0, 50)],
        },
    )
    return env


def _create_graph_env():
    """Create a 1D Graph layout environment for testing."""
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    for i in range(5):
        graph.add_node(i, pos=(i * 20.0, 0.0))
    for i in range(4):
        graph.add_edge(i, i + 1, distance=20.0)

    edge_order = [(i, i + 1) for i in range(4)]
    env = Environment.from_graph(
        graph, edge_order=edge_order, edge_spacing=20.0, bin_size=5.0
    )
    return env


def _create_masked_grid_env():
    """Create a MaskedGrid layout environment for testing."""
    from neurospatial import Environment

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    x_edges = np.linspace(0, 50, 11)
    y_edges = np.linspace(0, 50, 11)
    env = Environment.from_mask(active_mask=mask, grid_edges=(x_edges, y_edges))
    return env


def _create_image_mask_env():
    """Create an ImageMask layout environment for testing."""
    from neurospatial import Environment

    y, x = np.ogrid[:50, :50]
    center = (25, 25)
    radius = 20
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius**2
    env = Environment.from_image(image_mask=mask, bin_size=2.0)
    return env


def _create_polygon_env():
    """Create a ShapelyPolygon layout environment for testing."""
    from shapely.geometry import Polygon

    from neurospatial import Environment

    boundary = Polygon([(0, 0), (50, 0), (50, 25), (25, 25), (25, 50), (0, 50)])
    env = Environment.from_polygon(boundary, bin_size=5.0)
    return env


def _create_triangular_mesh_env():
    """Create a TriangularMesh layout environment for testing."""
    from shapely.geometry import Polygon

    from neurospatial import Environment

    boundary = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    env = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={"boundary_polygon": boundary, "point_spacing": 5.0},
    )
    return env


def _create_3d_grid_env():
    """Create a 3D RegularGrid environment for testing."""
    from neurospatial import Environment

    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 50, (500, 3))
    env = Environment.from_samples(positions, bin_size=5.0)
    return env


# Layout factory registry for parametrized tests
ALL_LAYOUT_FACTORIES = [
    ("RegularGrid", _create_regular_grid_env),
    ("Hexagonal", _create_hexagonal_env),
    ("Graph", _create_graph_env),
    ("MaskedGrid", _create_masked_grid_env),
    ("ImageMask", _create_image_mask_env),
    ("ShapelyPolygon", _create_polygon_env),
    ("TriangularMesh", _create_triangular_mesh_env),
    ("3D_RegularGrid", _create_3d_grid_env),
]


class TestAllLayoutsRoundTrip:
    """Parametrized tests verifying all properties for all layout types."""

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_bin_centers_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test bin_centers are exactly preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_bin_centers.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        np.testing.assert_array_equal(
            loaded_env.bin_centers,
            env.bin_centers,
            err_msg=f"bin_centers mismatch for {layout_name}",
        )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_connectivity_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test connectivity graph is preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_connectivity.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert (
            loaded_env.connectivity.number_of_nodes()
            == env.connectivity.number_of_nodes()
        ), f"Node count mismatch for {layout_name}"
        assert (
            loaded_env.connectivity.number_of_edges()
            == env.connectivity.number_of_edges()
        ), f"Edge count mismatch for {layout_name}"

        original_edges = sorted(env.connectivity.edges())
        loaded_edges = sorted(loaded_env.connectivity.edges())
        assert original_edges == loaded_edges, f"Edge set mismatch for {layout_name}"

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_edge_weights_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test edge weights (distances) are preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_edge_weights.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        for u, v in env.connectivity.edges():
            original_dist = env.connectivity[u][v]["distance"]
            loaded_dist = loaded_env.connectivity[u][v]["distance"]
            np.testing.assert_almost_equal(
                loaded_dist,
                original_dist,
                err_msg=f"Edge ({u},{v}) weight mismatch for {layout_name}",
            )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_metadata_all_layouts(self, tmp_path, layout_name, env_factory):
        """Test metadata (units, frame, name) is preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        env.name = f"test_{layout_name}"
        env.units = "cm"
        env.frame = "session_001"

        nwb_path = tmp_path / f"test_{layout_name}_metadata.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert loaded_env.name == f"test_{layout_name}", (
            f"Name mismatch for {layout_name}"
        )
        assert loaded_env.units == "cm", f"Units mismatch for {layout_name}"
        assert loaded_env.frame == "session_001", f"Frame mismatch for {layout_name}"

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_point_regions_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test point regions are preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()

        # Add point regions at locations within the environment bounds
        # Use bin centers to ensure points are within valid range
        if env.n_bins >= 2:
            env.regions.add("start", point=tuple(env.bin_centers[0]))
            env.regions.add("goal", point=tuple(env.bin_centers[-1]))

        nwb_path = tmp_path / f"test_{layout_name}_point_regions.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert set(loaded_env.regions.keys()) == set(env.regions.keys()), (
            f"Region keys mismatch for {layout_name}"
        )

        for name in env.regions:
            original = env.regions[name]
            loaded = loaded_env.regions[name]
            assert loaded.kind == original.kind, (
                f"Region {name} kind mismatch for {layout_name}"
            )
            np.testing.assert_array_almost_equal(
                loaded.data,
                original.data,
                err_msg=f"Region {name} data mismatch for {layout_name}",
            )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_polygon_regions_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test polygon regions are preserved for all layout types."""
        from pynwb import NWBHDF5IO
        from shapely.geometry import Polygon

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()

        # Skip 1D layouts for polygon regions (they use 2D projected coordinates)
        if env.n_dims >= 2:
            # Create a small polygon region within the environment bounds
            center = env.bin_centers.mean(axis=0)[:2]  # Use first 2 dims
            size = 5.0  # Small region
            triangle = Polygon(
                [
                    (center[0] - size, center[1] - size),
                    (center[0] + size, center[1] - size),
                    (center[0], center[1] + size),
                ]
            )
            env.regions.add("reward_zone", polygon=triangle)

        nwb_path = tmp_path / f"test_{layout_name}_polygon_regions.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        if "reward_zone" in env.regions:
            assert "reward_zone" in loaded_env.regions, (
                f"Polygon region missing for {layout_name}"
            )
            assert loaded_env.regions["reward_zone"].kind == "polygon", (
                f"Polygon region kind wrong for {layout_name}"
            )

            loaded_coords = list(loaded_env.regions["reward_zone"].data.exterior.coords)
            original_coords = list(env.regions["reward_zone"].data.exterior.coords)
            np.testing.assert_array_almost_equal(
                loaded_coords,
                original_coords,
                err_msg=f"Polygon coordinates mismatch for {layout_name}",
            )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_spatial_queries_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test spatial queries work after round-trip for all layout types.

        For grid-based layouts (RegularGrid, MaskedGrid, ImageMask, ShapelyPolygon),
        bin_at() uses proper grid-based geometric containment.
        For non-grid layouts (Graph, Hexagonal, TriangularMesh), bin_at() uses
        KDTree-based nearest neighbor mapping as a fallback.
        """
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_queries.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Test bin_at works - each bin_center should map to its own index
        test_points = loaded_env.bin_centers[:3]  # Test first 3 bins
        bin_indices = loaded_env.bin_at(test_points)

        # Verify each point maps to its own index
        np.testing.assert_array_equal(
            bin_indices,
            np.array([0, 1, 2]),
            err_msg=f"bin_at failed for {layout_name}",
        )

        # Test neighbors works on a node with edges
        # Find a node with at least one neighbor
        test_node = None
        for node in range(loaded_env.n_bins):
            if loaded_env.connectivity.degree(node) > 0:
                test_node = node
                break

        if test_node is not None:
            neighbors = loaded_env.neighbors(test_node)
            assert len(neighbors) > 0, f"neighbors failed for {layout_name}"

            # Test distance_between works (takes points, not node indices)
            point1 = loaded_env.bin_centers[test_node]
            point2 = loaded_env.bin_centers[neighbors[0]]
            dist = loaded_env.distance_between(point1, point2)
            assert dist > 0, f"distance_between failed for {layout_name}"

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_dimension_ranges_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test dimension_ranges are preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_dim_ranges.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        np.testing.assert_array_almost_equal(
            loaded_env.dimension_ranges,
            env.dimension_ranges,
            err_msg=f"dimension_ranges mismatch for {layout_name}",
        )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_layout_type_stored_all_layouts(
        self, tmp_path, layout_name, env_factory
    ):
        """Test layout type is stored and retrievable for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_layout_type.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Layout type should be stored (though exact type depends on factory)
        assert hasattr(loaded_env, "_layout_type_used"), (
            f"Missing _layout_type_used for {layout_name}"
        )
        assert loaded_env._layout_type_used is not None, (
            f"_layout_type_used is None for {layout_name}"
        )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_n_dims_all_layouts(self, tmp_path, layout_name, env_factory):
        """Test n_dims is preserved for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_n_dims.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert loaded_env.n_dims == env.n_dims, (
            f"n_dims mismatch for {layout_name}: got {loaded_env.n_dims}, expected {env.n_dims}"
        )

    @pytest.mark.parametrize("layout_name,env_factory", ALL_LAYOUT_FACTORIES)
    def test_roundtrip_is_fitted_all_layouts(self, tmp_path, layout_name, env_factory):
        """Test loaded environment is fitted and usable for all layout types."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment, write_environment

        env = env_factory()
        nwb_path = tmp_path / f"test_{layout_name}_fitted.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert loaded_env._is_fitted, f"Environment not fitted for {layout_name}"


# =============================================================================
# Tests for Environment class methods (M3.3)
# =============================================================================


class TestEnvironmentFromNwb:
    """Tests for Environment.from_nwb() classmethod."""

    def test_from_nwb_with_scratch_name(self, tmp_path, sample_environment):
        """Test loading Environment from scratch using scratch_name parameter."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        nwb_path = tmp_path / "test_from_nwb_scratch.nwb"

        # Write environment to scratch
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment, name="my_environment")
            io.write(nwbfile)

        # Load using from_nwb with scratch_name
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = Environment.from_nwb(nwbfile, scratch_name="my_environment")

        assert loaded_env.n_bins == sample_environment.n_bins
        np.testing.assert_array_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_from_nwb_with_bin_size(self, sample_nwb_with_position):
        """Test creating Environment from position data using bin_size parameter."""
        from neurospatial import Environment

        # Load using from_nwb with bin_size (creates from position data)
        env = Environment.from_nwb(sample_nwb_with_position, bin_size=5.0)

        assert env is not None
        assert env.n_bins > 0
        assert env._is_fitted

    def test_from_nwb_error_when_neither_parameter_provided(self, empty_nwb):
        """Test ValueError when neither scratch_name nor bin_size provided."""
        from neurospatial import Environment

        with pytest.raises(ValueError, match="Either scratch_name or bin_size"):
            Environment.from_nwb(empty_nwb)

    def test_from_nwb_kwargs_forwarded_to_from_position(self, sample_nwb_with_position):
        """Test kwargs are forwarded to environment_from_position."""
        from neurospatial import Environment

        # With infer_active_bins=True
        env_active = Environment.from_nwb(
            sample_nwb_with_position, bin_size=5.0, infer_active_bins=True
        )

        # Without infer_active_bins
        env_all = Environment.from_nwb(
            sample_nwb_with_position, bin_size=5.0, infer_active_bins=False
        )

        assert env_active.n_bins <= env_all.n_bins

    def test_from_nwb_units_parameter_propagated(self, sample_nwb_with_position):
        """Test units parameter is propagated when creating from position."""
        from neurospatial import Environment

        env = Environment.from_nwb(
            sample_nwb_with_position, bin_size=5.0, units="meters"
        )

        assert env.units == "meters"

    def test_from_nwb_frame_parameter_propagated(self, sample_nwb_with_position):
        """Test frame parameter is propagated when creating from position."""
        from neurospatial import Environment

        env = Environment.from_nwb(
            sample_nwb_with_position, bin_size=5.0, frame="session_001"
        )

        assert env.frame == "session_001"

    def test_from_nwb_scratch_name_takes_precedence(
        self, tmp_path, sample_environment, sample_nwb_with_position
    ):
        """Test scratch_name takes precedence when both parameters provided."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        nwb_path = tmp_path / "test_precedence.nwb"

        # Create NWB with both position data and stored environment
        with NWBHDF5IO(str(nwb_path), "w") as io:
            # Start from position NWB and add stored environment
            nwbfile = sample_nwb_with_position
            write_environment(nwbfile, sample_environment, name="stored_env")
            io.write(nwbfile)

        # Load with both parameters - scratch_name should take precedence
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = Environment.from_nwb(
                nwbfile, scratch_name="stored_env", bin_size=10.0
            )

        # Should match stored environment, not newly created one
        assert loaded_env.n_bins == sample_environment.n_bins


class TestEnvironmentToNwb:
    """Tests for Environment.to_nwb() method."""

    def test_to_nwb_basic(self, empty_nwb, sample_environment):
        """Test basic writing to NWB file using to_nwb method."""
        sample_environment.to_nwb(empty_nwb)

        assert "spatial_environment" in empty_nwb.scratch

    def test_to_nwb_custom_name(self, empty_nwb, sample_environment):
        """Test custom name parameter for to_nwb."""
        sample_environment.to_nwb(empty_nwb, name="my_env")

        assert "my_env" in empty_nwb.scratch
        assert "spatial_environment" not in empty_nwb.scratch

    def test_to_nwb_overwrite_parameter(self, empty_nwb, sample_environment):
        """Test overwrite parameter for to_nwb."""
        # Write first time
        sample_environment.to_nwb(empty_nwb)

        # Should fail without overwrite
        with pytest.raises(ValueError, match="already exists"):
            sample_environment.to_nwb(empty_nwb)

        # Should succeed with overwrite=True
        sample_environment.units = "meters"
        sample_environment.to_nwb(empty_nwb, overwrite=True)

        assert "meters" in empty_nwb.scratch["spatial_environment"].description

    def test_to_nwb_roundtrip(self, tmp_path, sample_environment):
        """Test that to_nwb and from_nwb work together."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment

        nwb_path = tmp_path / "test_method_roundtrip.nwb"

        # Write using to_nwb method
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            sample_environment.to_nwb(nwbfile, name="test_env")
            io.write(nwbfile)

        # Read using from_nwb classmethod
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = Environment.from_nwb(nwbfile, scratch_name="test_env")

        assert loaded_env.n_bins == sample_environment.n_bins
        np.testing.assert_array_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_to_nwb_preserves_metadata(self, empty_nwb, sample_environment):
        """Test that to_nwb preserves environment metadata."""
        from neurospatial.io.nwb import read_environment

        sample_environment.name = "test_arena"
        sample_environment.units = "cm"
        sample_environment.frame = "session_001"

        sample_environment.to_nwb(empty_nwb)

        # Read back and verify metadata
        loaded = read_environment(empty_nwb)

        assert loaded.name == "test_arena"
        assert loaded.units == "cm"
        assert loaded.frame == "session_001"
