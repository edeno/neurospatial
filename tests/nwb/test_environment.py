"""
Tests for NWB environment functions.

Tests for:
- environment_from_position() - creates an Environment from NWB Position data
- write_environment() - writes Environment to NWB scratch space
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurospatial import Regions
from neurospatial.layout import create_layout

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


def _stored_array(record, key):
    """Read one shaped numerical array through the persisted column mapping."""
    metadata = json.loads(record.description)
    spec = metadata["array_specs"][key]
    return np.asarray(record[spec["column"]][0]).reshape(spec["shape"])


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("n_angle", [1, 2, 6])
def test_polar_circularity_survives_nwb_round_trip(
    empty_nwb, tmp_path, circular: bool, n_angle: int
) -> None:
    """NWB preserves circularity even when seam edges cannot distinguish it."""
    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.from_polar_egocentric(
        distance_range=(0, 3),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=2 * np.pi / n_angle,
        circular_angle=circular,
    )
    write_environment(empty_nwb, env)
    path = tmp_path / "polar.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_environment(io.read())
    assert restored._angular_is_circular is circular
    np.testing.assert_allclose(
        restored.compute_kernel(bandwidth=1.0), env.compute_kernel(bandwidth=1.0)
    )


@pytest.mark.parametrize("kind", ["translation", "rotation", "shear_and_scaling"])
@pytest.mark.integration
def test_transformed_grid_nwb_round_trip_preserves_binning(empty_nwb, kind):
    """Derived data geometry can be restored without a layout wrapper."""
    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment
    from neurospatial.ops.transforms import Affine2D

    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    env = Environment.from_grid_mask(
        mask,
        grid_edges=(np.arange(4.0) * 2, np.arange(4.0)),
        name="arena",
    )
    env = env.with_metadata(units="cm")
    env = env.with_metadata(frame="recording")
    matrices = {
        "translation": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
        "rotation": [[0, -1, 10], [1, 0, 20], [0, 0, 1]],
        "shear_and_scaling": [[2, 1, 10], [0, 3, 20], [0, 0, 1]],
    }
    aligned = env.apply_transform(Affine2D(np.array(matrices[kind], dtype=float)))
    write_environment(empty_nwb, aligned)
    restored = read_environment(empty_nwb)
    np.testing.assert_array_equal(restored.bin_centers, aligned.bin_centers)
    np.testing.assert_array_equal(restored.bin_sizes, aligned.bin_sizes)
    np.testing.assert_array_equal(
        restored.bin_at(aligned.bin_centers), np.arange(aligned.n_bins)
    )
    assert list(restored.connectivity.edges) == list(aligned.connectivity.edges)
    assert restored.frame == aligned.frame
    assert restored.units == aligned.units
    assert (restored._binning.rectilinear is None) == (
        aligned._binning.rectilinear is None
    )


@pytest.mark.parametrize("scale", [3.0, 1e-5], ids=["ordinary_scale", "small_scale"])
def test_restored_track_transforms_keep_segment_lengths(
    empty_nwb,
    scale,
):
    """Stored tracks keep similarity lengths and refuse non-similarity maps."""
    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment
    from neurospatial.ops.transforms import scale_2d

    env = Environment.linear_track(endpoints=((0.0, 0.0), (10.0, 0.0)), bin_size=2.0)
    write_environment(empty_nwb, env)
    restored = read_environment(empty_nwb)
    scaled = restored.apply_transform(scale_2d(scale))
    np.testing.assert_allclose(scaled.bin_sizes, scale * env.bin_sizes)
    with pytest.raises(ValueError, match=r"Environment\.from_graph"):
        restored.apply_transform(scale_2d(scale, 2 * scale))


@pytest.mark.integration
def test_subset_of_read_four_connected_grid_keeps_its_connectivity(empty_nwb, tmp_path):
    """A 4-connected grid read from NWB stays 4-connected through subset."""
    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 30, (2000, 2)),
        bin_size=3.0,
        connect_diagonal_neighbors=False,
    )
    write_environment(empty_nwb, env)
    path = tmp_path / "grid4.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_environment(io.read())

    for mask in (
        np.ones(env.n_bins, dtype=bool),
        np.arange(env.n_bins) < env.n_bins // 2,
    ):
        expected, result = env.subset(mask=mask), restored.subset(mask=mask)
        assert sorted(result.connectivity.edges()) == sorted(
            expected.connectivity.edges()
        )
        np.testing.assert_array_equal(result.boundary_bins, expected.boundary_bins)


@pytest.mark.parametrize("kind", ["samples", "mask", "pixels", "polygon"])
@pytest.mark.integration
def test_read_grid_preserves_linear_operations(empty_nwb, tmp_path, kind):
    """Persisted grid geometry gives exactly the same interpolation and time."""
    from shapely.geometry import Point

    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment

    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    if kind == "samples":
        env = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 30, (300, 2)), bin_size=3.0
        )
    elif kind == "mask":
        env = Environment.from_grid_mask(
            mask, grid_edges=(np.arange(6.0), np.arange(6.0))
        )
    elif kind == "pixels":
        env = Environment.from_pixel_mask(mask, pixel_size=1.0)
    else:
        env = Environment.from_polygon(Point(15, 15).buffer(15), bin_size=2.0)
    write_environment(empty_nwb, env)
    path = tmp_path / "grid.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_environment(io.read())
    field = np.sin(np.arange(env.n_bins))
    points = np.vstack([env.bin_centers[::5], env.bin_centers.max(axis=0) + 100])
    np.testing.assert_array_equal(
        restored.interpolate(field, points, mode="linear"),
        env.interpolate(field, points, mode="linear"),
    )
    times = np.arange(10) * 0.1
    positions = env.bin_centers[np.linspace(0, env.n_bins - 1, 10, dtype=int)]
    np.testing.assert_array_equal(
        restored.occupancy(times, positions, time_allocation="linear"),
        env.occupancy(times, positions, time_allocation="linear"),
    )


@pytest.mark.integration
def test_read_track_preserves_linearization_graph(empty_nwb):
    """Both coordinate conversions use the saved track substrate."""
    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.linear_track(endpoints=((0.0, 0.0), (10.0, 0.0)), bin_size=1.0)
    write_environment(empty_nwb, env)
    restored = read_environment(empty_nwb)
    points = np.array([[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]])
    linear = env.to_linear(points)
    np.testing.assert_array_equal(restored.to_linear(points), linear)
    np.testing.assert_array_equal(
        restored.linear_to_nd(linear), env.linear_to_nd(linear)
    )


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
        position_data = read_position(sample_nwb_with_position)
        positions = position_data.positions

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

    def test_different_bin_sizes(self, sample_nwb_with_position):
        """Test Environment creation with different bin sizes."""
        from neurospatial.io.nwb import environment_from_position

        env_small = environment_from_position(sample_nwb_with_position, bin_size=2.0)
        env_large = environment_from_position(sample_nwb_with_position, bin_size=10.0)

        # Smaller bins should result in more bins
        assert env_small.n_bins > env_large.n_bins

    def test_binning_options_mean_what_they_mean_in_from_samples(
        self, sample_nwb_with_position
    ):
        from neurospatial import Environment, Region, Regions
        from neurospatial.io.nwb import environment_from_position, read_position

        regions = Regions([Region("goal", "point", [10.0, 10.0])])
        options = {
            "bin_size": 5.0,
            "name": "arena",
            "units": "cm",
            "regions": regions,
            "fill_holes": True,
            "connect_diagonal_neighbors": False,
        }
        env = environment_from_position(sample_nwb_with_position, **options)
        positions = read_position(sample_nwb_with_position).positions
        expected = Environment.from_samples(positions, **options)
        np.testing.assert_array_equal(env.bin_centers, expected.bin_centers)
        np.testing.assert_array_equal(env.edge_pairs, expected.edge_pairs)
        assert (env.name, env.units) == ("arena", "cm")
        assert env.regions.list_names() == ["goal"]
        assert env.layout_parameters == expected.layout_parameters

    def test_every_binning_option_is_forwarded_unchanged(
        self, sample_nwb_with_position, monkeypatch
    ):
        """Each option reaches the from_samples implementation as passed."""
        import inspect

        from neurospatial.environment import _factories
        from neurospatial.io.nwb import environment_from_position

        received = {}

        def record(cls, positions, **options):
            received.update(options)

        monkeypatch.setattr(_factories, "_from_samples", record)
        not_options = {"nwbfile", "processing_module", "position_name"}
        names = [
            name
            for name in inspect.signature(environment_from_position).parameters
            if name not in not_options
        ]
        # A distinct object per option: forwarding must not drop, swap or copy.
        passed = {name: object() for name in names}
        environment_from_position(sample_nwb_with_position, **passed)
        assert received.pop("factory") == "environment_from_position"
        assert received.keys() == passed.keys()
        assert all(received[name] is passed[name] for name in names)

    def test_binning_options_match_from_samples(self):
        """Every from_samples option exists here with the same default."""
        import inspect

        from neurospatial import Environment
        from neurospatial.io.nwb import environment_from_position

        def options(function, data):
            parameters = inspect.signature(function).parameters
            return {
                name: str(getattr(p.default, "value", p.default))
                for name, p in parameters.items()
                if name not in data
            }

        builder = options(
            environment_from_position,
            {"nwbfile", "processing_module", "position_name"},
        )
        assert builder == options(Environment.from_samples, {"positions"})

    def test_inapplicable_option_is_refused(self, sample_nwb_with_position):
        from neurospatial.io.nwb import environment_from_position

        with pytest.raises(
            ValueError,
            match=r"environment_from_position\(layout='Hexagonal'\) cannot apply dilate=True",
        ):
            environment_from_position(
                sample_nwb_with_position, bin_size=5.0, layout="Hexagonal", dilate=True
            )
        with pytest.raises(TypeError, match="positional") as caught:
            environment_from_position(sample_nwb_with_position, 5.0)
        assert (
            "pass bin_size=5.0 by keyword, as in "
            "environment_from_position(nwbfile, bin_size=5.0)"
        ) in str(caught.value)

    def test_from_nwb_points_building_options_to_environment_from_position(
        self, sample_nwb_with_position
    ):
        from neurospatial import Environment

        with pytest.raises(TypeError, match="does not take bin_size") as caught:
            Environment.from_nwb(sample_nwb_with_position, bin_size=4.0)
        assert (
            "neurospatial.io.nwb.environment_from_position(nwbfile, bin_size=4.0)"
            in str(caught.value).split("Fix: ")[1]
        )
        # Keywords that no builder takes keep Python's own message.
        with pytest.raises(TypeError, match="unexpected keyword argument 'colour'"):
            Environment.from_nwb(sample_nwb_with_position, colour="red")

    def test_from_nwb_without_saved_environment_points_to_the_builder(
        self, sample_nwb_with_position
    ):
        from neurospatial import Environment

        with pytest.raises(KeyError) as caught:
            Environment.from_nwb(sample_nwb_with_position)
        message = str(caught.value)
        assert "No environment named 'spatial_environment'" in message
        assert "saved environments: none" in message
        fix = message.split("Fix: ")[1]
        assert "Position data at processing/behavior/" in fix
        assert "neurospatial.io.nwb.environment_from_position(nwbfile, bin_size=" in fix

    def test_default_options_are_accepted_with_any_layout(
        self, sample_nwb_with_position
    ):
        from neurospatial.io.nwb import environment_from_position

        env = environment_from_position(
            sample_nwb_with_position,
            bin_size=5.0,
            layout="Hexagonal",
            dilate=False,
            connect_diagonal_neighbors=True,
            bin_count_threshold=0,
        )
        assert env.layout_type == "Hexagonal"


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
        """Centers retain their exact shape in a typed ragged column."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)
        table = empty_nwb.scratch["spatial_environment"]
        centers = _stored_array(table, "binning/bins/centers")
        assert centers.shape == (sample_environment.n_bins, sample_environment.n_dims)
        np.testing.assert_array_equal(centers, sample_environment.bin_centers)

    def test_edges_dataset_as_edge_list(self, empty_nwb, sample_environment):
        """Pairs retain graph iteration order and orientation without padding."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)
        table = empty_nwb.scratch["spatial_environment"]
        edges = _stored_array(table, "binning/bins/pairs")
        np.testing.assert_array_equal(
            edges, list(sample_environment.connectivity.edges)
        )

    def test_edge_weights_dataset(self, empty_nwb, sample_environment):
        """Navigation distances stay aligned with the stored pairs."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)
        table = empty_nwb.scratch["spatial_environment"]
        actual = _stored_array(table, "binning/bins/pair_distances")
        expected = [
            d["distance"]
            for _, _, d in sample_environment.connectivity.edges(data=True)
        ]
        np.testing.assert_array_equal(actual, expected)

    def test_dimension_ranges_dataset(self, empty_nwb, sample_environment):
        """Coordinate extents are stored in the shared JSON metadata."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)
        metadata = json.loads(empty_nwb.scratch["spatial_environment"].description)
        np.testing.assert_array_equal(
            metadata["dimension_ranges"], sample_environment.dimension_ranges
        )

    def test_group_attributes_units(self, empty_nwb, sample_environment):
        """Test units attribute is stored on group."""
        from neurospatial.io.nwb import write_environment

        sample_environment = sample_environment.with_metadata(units="cm")
        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Access via description or comments which hold metadata
        assert (
            "cm" in scratch_data.description or "units=cm" in scratch_data.description
        )

    def test_group_attributes_frame(self, empty_nwb, sample_environment):
        """Test frame attribute is stored on group."""
        from neurospatial.io.nwb import write_environment

        sample_environment = sample_environment.with_metadata(frame="session_001")
        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert (
            "session_001" in scratch_data.description
            or "frame=session_001" in scratch_data.description
        )

    def test_group_attributes_n_dims(self, empty_nwb, sample_environment):
        """Coordinate dimensionality is explicit JSON metadata."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)
        metadata = json.loads(empty_nwb.scratch["spatial_environment"].description)
        assert metadata["n_dims"] == sample_environment.n_dims

    def test_group_attributes_layout_type(self, empty_nwb, sample_environment):
        """Test layout_type attribute is stored."""
        from neurospatial.io.nwb import write_environment

        write_environment(empty_nwb, sample_environment)

        scratch_data = empty_nwb.scratch["spatial_environment"]
        # Layout type should be stored in description
        layout_type = sample_environment.layout_type
        assert layout_type in scratch_data.description

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
        modified_env = modified_env.with_metadata(units="meters")

        # Should succeed with overwrite=True
        write_environment(empty_nwb, modified_env, overwrite=True)

        # Check new data is present
        scratch_data = empty_nwb.scratch["spatial_environment"]
        assert "meters" in scratch_data.description

    def test_point_regions_stored(self, empty_nwb, sample_environment):
        """Point regions use the canonical Region JSON schema."""
        from neurospatial.io.nwb import write_environment

        region_builder = Regions(sample_environment.regions.values())
        region_builder.add("center", point=(50.0, 50.0))
        sample_environment = sample_environment.with_regions(region_builder)
        write_environment(empty_nwb, sample_environment)
        metadata = json.loads(empty_nwb.scratch["spatial_environment"].description)
        expected = sample_environment.regions["center"].to_dict()
        assert expected in metadata["regions"]

    def test_polygon_regions_stored(self, empty_nwb):
        """Polygon region coordinates remain canonical JSON geometry."""
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        env = Environment.from_samples(
            np.random.default_rng(42).uniform(0, 100, (1000, 2)), bin_size=5.0
        )
        triangle = Polygon([(10, 10), (30, 10), (20, 30)])
        region_builder = Regions(env.regions.values())
        region_builder.add("reward_zone", polygon=triangle)
        env = env.with_regions(region_builder)
        write_environment(empty_nwb, env)
        metadata = json.loads(empty_nwb.scratch["spatial_environment"].description)
        expected = json.loads(json.dumps(env.regions["reward_zone"].to_dict()))
        assert expected in metadata["regions"]

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
        """The description stores versioned JSON metadata, separate from arrays."""
        from neurospatial.io.nwb import write_environment

        sample_environment = sample_environment.with_metadata(name="test_arena")
        write_environment(empty_nwb, sample_environment)
        metadata = json.loads(empty_nwb.scratch["spatial_environment"].description)
        assert metadata["name"] == "test_arena"
        assert metadata["schema_version"] == "3.0"

    def test_data_integrity_bin_centers(self, empty_nwb, sample_environment):
        """Writing preserves every center value exactly."""
        from neurospatial.io.nwb import write_environment

        expected = sample_environment.bin_centers.copy()
        write_environment(empty_nwb, sample_environment)
        table = empty_nwb.scratch["spatial_environment"]
        np.testing.assert_array_equal(
            _stored_array(table, "binning/bins/centers"), expected
        )

    def test_data_integrity_edges(self, empty_nwb, sample_environment):
        """Writing preserves pair order and orientation exactly."""
        from neurospatial.io.nwb import write_environment

        expected = np.array(list(sample_environment.connectivity.edges))
        write_environment(empty_nwb, sample_environment)
        table = empty_nwb.scratch["spatial_environment"]
        np.testing.assert_array_equal(
            _stored_array(table, "binning/bins/pairs"), expected
        )


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

        sample_environment = sample_environment.with_metadata(units="cm")
        write_environment(empty_nwb, sample_environment)
        loaded_env = read_environment(empty_nwb)

        assert loaded_env.units == "cm"

    def test_frame_attribute_restored(self, empty_nwb, sample_environment):
        """Test frame attribute is restored."""
        from neurospatial.io.nwb import read_environment, write_environment

        sample_environment = sample_environment.with_metadata(frame="session_001")
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
        region_builder = Regions(env.regions.values())
        region_builder.add("reward_zone", polygon=triangle)
        env = env.with_regions(region_builder)

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

        with pytest.raises(KeyError, match="No environment named 'nonexistent'"):
            read_environment(empty_nwb, name="nonexistent")

    def test_custom_name_parameter(self, empty_nwb, sample_environment):
        """Test reading environment with custom name."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment, name="linear_track")
        loaded_env = read_environment(empty_nwb, name="linear_track")

        assert loaded_env is not None
        assert loaded_env.n_bins == sample_environment.n_bins

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

        sample_environment = sample_environment.with_metadata(name="test_arena")
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

        sample_environment = sample_environment.with_metadata(name="test_arena")
        sample_environment = sample_environment.with_metadata(units="cm")
        sample_environment = sample_environment.with_metadata(frame="session_001")

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

    def test_roundtrip_region_metadata_preserved(self, tmp_path):
        """Region metadata survives the NWB write/read cycle.

        The NWB region (de)serialization delegates to Region.to_dict /
        Region.from_dict, so per-region metadata (which the old bespoke schema
        dropped) is preserved through the round-trip.
        """
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        rng = np.random.default_rng(3)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        region_builder = Regions(env.regions.values())
        region_builder.add(
            "goal",
            point=(50.0, 50.0),
            metadata={"color": "red", "reward": 1.0, "tags": ["target"]},
        )
        env = env.with_regions(region_builder)

        nwb_path = tmp_path / "region_metadata.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        loaded = loaded_env.regions["goal"]
        assert loaded.metadata["color"] == "red"
        assert loaded.metadata["reward"] == 1.0
        # Lists freeze to tuples in stored metadata; to_dict thaws them back.
        assert loaded.to_dict()["metadata"]["tags"] == ["target"]

    def test_roundtrip_frame_literal_unknown_preserved(self, tmp_path):
        """A frame literally named 'unknown' survives the round-trip.

        The frame fallback uses DEFAULT_FRAME ("") rather than DEFAULT_UNITS
        ("unknown"), so a legitimate frame value of "unknown" is not dropped on
        read.
        """
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        rng = np.random.default_rng(4)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        env = env.with_metadata(units="cm")
        env = env.with_metadata(
            frame="unknown"
        )  # legitimate frame name that collides with old default

        nwb_path = tmp_path / "frame_unknown.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        assert loaded_env.frame == "unknown"

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
        region_builder = Regions(env.regions.values())
        region_builder.add("reward_zone", polygon=triangle)
        env = env.with_regions(region_builder)

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

    @pytest.mark.slow
    def test_environment_with_polygon_regions_survives_disk_roundtrip(self, tmp_path):
        """Polygon geometry survives disk round-trip (verified via shapely.equals).

        Complements test_roundtrip_polygon_regions (which checks exterior coords
        of a single polygon) by mixing a point region with a polygon region and
        asserting geometric equality with shapely.equals -- Shapely-to-HDF5
        serialization is non-trivial and is a likely silent-loss spot.
        """
        import shapely
        from pynwb import NWBHDF5IO
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        rng = np.random.default_rng(7)
        positions = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        env = env.with_metadata(units="cm")

        # One point region and one polygon region in the same environment.
        region_builder = Regions(env.regions.values())
        region_builder.add("home", point=(5.0, 5.0))
        env = env.with_regions(region_builder)
        arm = Polygon([(10, 10), (40, 10), (40, 40), (10, 40)])
        region_builder = Regions(env.regions.values())
        region_builder.add("arm", polygon=arm)
        env = env.with_regions(region_builder)

        nwb_path = tmp_path / "polygon_regions.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, env)
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = read_environment(nwbfile)

        # Both regions survive with their kinds.
        assert set(loaded_env.regions.keys()) == {"home", "arm"}
        assert loaded_env.regions["home"].kind == "point"
        assert loaded_env.regions["arm"].kind == "polygon"

        # Polygon geometry is preserved exactly (shapely.equals is topological).
        assert shapely.equals(loaded_env.regions["arm"].data, arm)

        # Point region survives too.
        np.testing.assert_array_almost_equal(
            np.asarray(loaded_env.regions["home"].data, dtype=float), [5.0, 5.0]
        )

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
        neighbors = loaded_env.neighbors(bins=0)
        assert len(neighbors) > 0

        # Test distance_between works. distance_between expects coordinates,
        # not bin indices, so pass the bin centers of bin 0 and a neighbor.
        if len(neighbors) > 0:
            dist = loaded_env.distance_between(
                points=[loaded_env.bin_centers[0], loaded_env.bin_centers[neighbors[0]]]
            )
            assert np.isfinite(dist)
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
            create_layout(
                "Hexagonal", hexagon_width=5.0, dimension_ranges=[(0, 50), (0, 50)]
            )
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
        assert loaded_env.layout_type == "Hexagonal"

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

        assert env.is_linearized_track, "GraphLayout should create a 1D environment"

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

        # is_linearized_track property should be preserved (Graph layouts store 2D bin_centers
        # but are conceptually 1D linearized tracks)
        assert loaded_env.is_linearized_track is True, (
            "is_linearized_track should be restored for Graph layouts"
        )
        assert env.is_linearized_track is True, "Original env should be 1D"

        # The flag is derived from the stored binning's track linearization.
        assert loaded_env.is_linearized_track is True, (
            "Reconstructed layout must report is_linearized_track=True so "
            "env.is_linearized_track and env.is_linearized_track agree"
        )

        # n_dims is based on bin_centers shape (2D for projected coordinates)
        assert loaded_env.n_dims == env.n_dims

        # Layout type should be stored
        assert loaded_env.layout_type == "Graph"

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

        env = Environment.from_grid_mask(
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
        assert loaded_env.layout_type == "MaskedGrid"

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

        env = Environment.from_pixel_mask(
            image_mask=mask,  # Boolean mask
            pixel_size=2.0,
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
        assert loaded_env.layout_type == "ImageMask"

    def test_roundtrip_triangular_mesh_layout(self, tmp_path):
        """Test TriangularMesh layout environment round-trip."""
        from pynwb import NWBHDF5IO
        from shapely.geometry import Polygon

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        # Create a triangular mesh layout with a simple boundary
        boundary = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])

        env = Environment.from_layout(
            create_layout(
                "TriangularMesh", boundary_polygon=boundary, point_spacing=5.0
            )
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
        assert loaded_env.layout_type == "TriangularMesh"


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
        create_layout(
            "Hexagonal", hexagon_width=5.0, dimension_ranges=[(0, 50), (0, 50)]
        )
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
    env = Environment.from_grid_mask(active_mask=mask, grid_edges=(x_edges, y_edges))
    return env


def _create_image_mask_env():
    """Create an ImageMask layout environment for testing."""
    from neurospatial import Environment

    y, x = np.ogrid[:50, :50]
    center = (25, 25)
    radius = 20
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius**2
    env = Environment.from_pixel_mask(image_mask=mask, pixel_size=2.0)
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
        create_layout("TriangularMesh", boundary_polygon=boundary, point_spacing=5.0)
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
        env = env.with_metadata(name=f"test_{layout_name}")
        env = env.with_metadata(units="cm")
        env = env.with_metadata(frame="session_001")

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
            region_builder = Regions(env.regions.values())
            region_builder.add("start", point=tuple(env.bin_centers[0]))
            region_builder.add("goal", point=tuple(env.bin_centers[-1]))
            env = env.with_regions(region_builder)

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

        # Polygon regions have exactly two coordinates.
        if env.n_dims == 2:
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
            region_builder = Regions(env.regions.values())
            region_builder.add("reward_zone", polygon=triangle)
            env = env.with_regions(region_builder)

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
            neighbors = loaded_env.neighbors(bins=test_node)
            assert len(neighbors) > 0, f"neighbors failed for {layout_name}"

            # Test distance_between works (takes points, not node indices)
            point1 = loaded_env.bin_centers[test_node]
            point2 = loaded_env.bin_centers[neighbors[0]]
            dist = loaded_env.distance_between(points=[point1, point2])
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
        assert loaded_env.layout_type == env.layout_type, (
            f"layout_type differs for {layout_name}"
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
    def test_roundtrip_bins_all_layouts(self, tmp_path, layout_name, env_factory):
        """Test the loaded environment keeps its bins for all layout types."""
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

        assert loaded_env.n_bins == env.n_bins, layout_name
        np.testing.assert_array_equal(loaded_env.bin_centers, env.bin_centers)


# =============================================================================
# Tests for Environment class methods
# =============================================================================


class TestEnvironmentFromNwb:
    """Tests for Environment.from_nwb() classmethod."""

    def test_from_nwb_reads_a_named_record(self, tmp_path, sample_environment):
        """from_nwb loads the scratch record with the given name."""
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import write_environment

        nwb_path = tmp_path / "test_from_nwb_scratch.nwb"

        # Write environment to scratch
        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_environment, name="my_environment")
            io.write(nwbfile)

        # Load the named record
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded_env = Environment.from_nwb(nwbfile, name="my_environment")

        assert loaded_env.n_bins == sample_environment.n_bins
        np.testing.assert_array_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_from_nwb_reads_the_default_record(self, empty_nwb, sample_environment):
        from neurospatial import Environment

        sample_environment.to_nwb(empty_nwb)
        loaded = Environment.from_nwb(empty_nwb)
        np.testing.assert_array_equal(
            loaded.bin_centers, sample_environment.bin_centers
        )

    def test_from_nwb_does_not_bin_position_data(self, sample_nwb_with_position):
        """Building from Position data is environment_from_position's job."""
        from neurospatial import Environment

        with pytest.raises(TypeError, match="bin_size"):
            Environment.from_nwb(sample_nwb_with_position, bin_size=5.0)
        with pytest.raises(KeyError, match="spatial_environment"):
            Environment.from_nwb(sample_nwb_with_position)

    def test_from_nwb_reports_missing_pynwb(self, empty_nwb, monkeypatch):
        import sys

        from neurospatial import Environment

        monkeypatch.setitem(sys.modules, "pynwb", None)
        with pytest.raises(ImportError, match="pynwb is required"):
            Environment.from_nwb(empty_nwb, name="spatial_environment")

    def test_from_nwb_does_not_relabel_other_import_errors(
        self, empty_nwb, monkeypatch
    ):
        import sys

        import neurospatial.io.nwb as nwb_io
        from neurospatial import Environment

        # A broken reader module, with pynwb installed, is not a missing pynwb.
        for name in ("read_environment", "environment_from_position"):
            monkeypatch.delattr(nwb_io, name, raising=False)
        monkeypatch.setitem(sys.modules, "neurospatial.io.nwb._environment", None)
        with pytest.raises(ImportError, match="_environment") as caught:
            Environment.from_nwb(empty_nwb, name="spatial_environment")
        assert "pynwb is required" not in str(caught.value)

    @pytest.mark.parametrize("direction", ["from_nwb", "to_nwb"])
    def test_broken_pynwb_dependency_is_not_reported_as_missing_pynwb(
        self, direction, empty_nwb, sample_environment, monkeypatch
    ):
        import sys

        from neurospatial import Environment

        # pynwb is installed but one of its dependencies cannot be imported.
        for module in [m for m in sys.modules if m.split(".")[0] == "pynwb"]:
            monkeypatch.delitem(sys.modules, module)
        monkeypatch.setitem(sys.modules, "hdmf", None)
        with pytest.raises(ModuleNotFoundError) as caught:
            if direction == "from_nwb":
                Environment.from_nwb(empty_nwb)
            else:
                sample_environment.to_nwb(empty_nwb)
        assert caught.value.name == "hdmf"
        assert "pynwb is required" not in str(caught.value)

    def test_to_nwb_reports_missing_pynwb(
        self, empty_nwb, sample_environment, monkeypatch
    ):
        import sys

        monkeypatch.setitem(sys.modules, "pynwb", None)
        with pytest.raises(ImportError, match="pynwb is required"):
            sample_environment.to_nwb(empty_nwb)


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
        sample_environment = sample_environment.with_metadata(units="meters")
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
            loaded_env = Environment.from_nwb(nwbfile, name="test_env")

        assert loaded_env.n_bins == sample_environment.n_bins
        np.testing.assert_array_equal(
            loaded_env.bin_centers, sample_environment.bin_centers
        )

    def test_to_nwb_preserves_metadata(self, empty_nwb, sample_environment):
        """Test that to_nwb preserves environment metadata."""
        from neurospatial.io.nwb import read_environment

        sample_environment = sample_environment.with_metadata(name="test_arena")
        sample_environment = sample_environment.with_metadata(units="cm")
        sample_environment = sample_environment.with_metadata(frame="session_001")

        sample_environment.to_nwb(empty_nwb)

        # Read back and verify metadata
        loaded = read_environment(empty_nwb)

        assert loaded.name == "test_arena"
        assert loaded.units == "cm"
        assert loaded.frame == "session_001"


class TestCoordinateKindRoundTrip:
    """The polar environment type must survive the NWB write/read cycle."""

    def test_nwb_roundtrip_preserves_coordinate_kind(
        self, empty_nwb, sample_polar_environment
    ):
        """A polar environment reads back as Environment."""
        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        assert isinstance(sample_polar_environment, Environment)

        write_environment(empty_nwb, sample_polar_environment, name="polar_env")
        loaded = read_environment(empty_nwb, name="polar_env")

        assert isinstance(loaded, Environment)
        assert loaded.coordinate_kind == "polar"
        assert (
            loaded.is_linearized_track == sample_polar_environment.is_linearized_track
        )

    def test_nwb_roundtrip_cartesian_unaffected(self, empty_nwb, sample_environment):
        """A Cartesian environment round-trips as a (non-polar) Environment."""
        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        assert isinstance(sample_environment, Environment)

        write_environment(empty_nwb, sample_environment, name="cart_env")
        loaded = read_environment(empty_nwb, name="cart_env")

        assert isinstance(loaded, Environment)
        assert loaded.coordinate_kind == "cartesian"

    def test_read_environment_missing_polar_coordinate_kind_is_rejected(
        self, empty_nwb, sample_polar_environment
    ):
        """Polar measures cannot be interpreted as Cartesian cell areas."""
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_polar_environment, name="polar_env")
        record = empty_nwb.scratch["polar_env"]
        metadata = json.loads(record.description)
        metadata.pop("coordinate_kind")
        record.fields["description"] = json.dumps(metadata)
        with pytest.raises(ValueError, match=r"sizes|measures|coordinate_kind"):
            read_environment(empty_nwb, name="polar_env")

    def test_read_environment_missing_cartesian_coordinate_kind_defaults(
        self, empty_nwb, sample_environment
    ):
        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        write_environment(empty_nwb, sample_environment)
        record = empty_nwb.scratch["spatial_environment"]
        metadata = json.loads(record.description)
        metadata.pop("coordinate_kind")
        record.fields["description"] = json.dumps(metadata)
        loaded = read_environment(empty_nwb)
        assert isinstance(loaded, Environment)
        np.testing.assert_array_equal(loaded.bin_sizes, sample_environment.bin_sizes)

    def test_polar_nwb_roundtrip_type(self, tmp_path, sample_polar_environment):
        """A polar env survives a real NWB file round-trip as the polar type.

        Writes to disk via NWBHDF5IO and reads back, asserting the restored
        object is an Environment (not a Cartesian Environment)
        and that the physical polar edge geometry is preserved.
        """
        from pynwb import NWBHDF5IO

        from neurospatial import Environment
        from neurospatial.io.nwb import read_environment, write_environment

        nwb_path = tmp_path / "polar_roundtrip.nwb"

        with NWBHDF5IO(str(nwb_path), "w") as io:
            nwbfile = _create_nwb_for_test()
            write_environment(nwbfile, sample_polar_environment, name="polar_env")
            io.write(nwbfile)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            loaded = read_environment(nwbfile, name="polar_env")

            assert isinstance(loaded, Environment)
            assert loaded.coordinate_kind == "polar"
            assert loaded.n_bins == sample_polar_environment.n_bins
            # The corrected physical polar edge distances survive the round-trip.
            orig_weights = sorted(
                d["distance"]
                for _, _, d in sample_polar_environment.connectivity.edges(data=True)
            )
            loaded_weights = sorted(
                d["distance"] for _, _, d in loaded.connectivity.edges(data=True)
            )
            np.testing.assert_allclose(loaded_weights, orig_weights, rtol=1e-9)


def _y_track_env():
    """Y-shaped linearized track (from_graph, bin 3, edge spacing 10)."""
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    for node, pos in enumerate(
        [(0.0, 0.0), (0.0, 100.0), (-50.0, 150.0), (50.0, 150.0)]
    ):
        graph.add_node(node, pos=pos)
    for edge_id, (u, v) in enumerate([(0, 1), (1, 2), (1, 3)]):
        distance = float(
            np.linalg.norm(np.subtract(graph.nodes[v]["pos"], graph.nodes[u]["pos"]))
        )
        graph.add_edge(u, v, distance=distance, edge_id=edge_id)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2), (1, 3)], edge_spacing=10.0, bin_size=3.0
    )


def _roundtrip_through_file(env, path):
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import read_environment, write_environment

    with NWBHDF5IO(str(path), "w") as io:
        nwbfile = _create_nwb_for_test()
        write_environment(nwbfile, env)
        io.write(nwbfile)
    with NWBHDF5IO(str(path), "r") as io:
        return read_environment(io.read())


class TestGraphGeometryRoundTrip:
    """Non-grid (graph) environments keep their geometry through NWB."""

    def test_graph_env_roundtrip_preserves_geometry(self, tmp_path):
        env = _y_track_env()
        loaded = _roundtrip_through_file(env, tmp_path / "ytrack.nwb")

        assert loaded.n_bins == env.n_bins
        np.testing.assert_array_equal(loaded.bin_sizes, env.bin_sizes)
        assert loaded.grid_edges is not None
        np.testing.assert_array_equal(loaded.grid_edges[0], env.grid_edges[0])

        flipped = [
            (u, v)
            for u, v, data in env.connectivity.edges(data=True)
            if not np.allclose(
                loaded.connectivity.edges[u, v]["vector"], data["vector"]
            )
        ]
        assert flipped == []

    def test_refuses_schema_1_0_file(self):
        """Old records are refused instead of guessing missing bin measures."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_environment

        path = Path(__file__).parent / "data/environment_schema_1_0.nwb"
        with (
            NWBHDF5IO(str(path), "r") as io,
            pytest.raises(ValueError, match=r"1\.0") as caught,
        ):
            read_environment(io.read())
        assert "Fix:" in str(caught.value)


def test_environment_from_position_uses_converted_meters(make_scaled_position_nwb):
    """Positions are converted to meters, and NWB's "meters" becomes "m".

    The converted extent is 1 m, so a cm-scale reading would give one bin.
    """
    import warnings

    from neurospatial.io.nwb import environment_from_position

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = environment_from_position(make_scaled_position_nwb(), bin_size=0.05)

    assert env.units == "m"
    assert not [w for w in caught if "standard registry" in str(w.message)]
    assert env.bin_centers.min() >= 0.1 - 0.05
    assert env.bin_centers.max() <= 1.1 + 0.05


def test_empty_unit_warns_and_assumes_cm(make_scaled_position_nwb):
    """A series that declares no unit warns once, then assumes 'cm'."""
    import warnings

    from neurospatial.io.nwb import environment_from_position

    nwbfile = make_scaled_position_nwb(unit="", conversion=1.0, offset=0.0)

    with pytest.warns(UserWarning, match="declares no unit") as record:
        env = environment_from_position(nwbfile, bin_size=50.0)
    messages = [str(w.message) for w in record if "declares no unit" in str(w.message)]
    assert len(messages) == 1
    assert "Fix: pass units=" in messages[0]
    assert env.units == "cm"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = environment_from_position(nwbfile, bin_size=50.0, units="cm")
    assert env.units == "cm"


@pytest.fixture
def hexagonal_env_from_nwb(empty_nwb):
    """A hexagonal environment after an in-memory NWB write/read round trip."""
    from neurospatial import Environment
    from neurospatial.io.nwb import read_environment, write_environment

    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=5.0, dimension_ranges=[(0, 50), (0, 50)]
        )
    )
    write_environment(empty_nwb, env)
    return env, read_environment(empty_nwb)


@pytest.mark.parametrize("operation", ["gradient", "divergence", "smooth", "basis"])
def test_finite_volume_ops_survive_nwb_round_trip(hexagonal_env_from_nwb, operation):
    """Stored faces preserve calculus, smoothing and heat-kernel bases exactly."""
    from neurospatial.ops.basis import heat_kernel_wavelet_basis
    from neurospatial.ops.calculus import divergence, gradient

    original, loaded = hexagonal_env_from_nwb

    def evaluate(env):
        field = np.sin(np.arange(env.n_bins))
        if operation == "gradient":
            return gradient(env, field)
        if operation == "divergence":
            return divergence(
                env, np.sin(np.arange(env.connectivity.number_of_edges()))
            )
        if operation == "smooth":
            return env.smooth(field, bandwidth=5.0)
        return heat_kernel_wavelet_basis(env, centers=np.array([0]))

    np.testing.assert_array_equal(evaluate(loaded), evaluate(original))


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("labels", {"start", "reward"}, "set value"),
        ("baseline", float("nan"), "non-finite number"),
    ],
    ids=["set", "nan"],
)
def test_nwb_writer_names_a_json_only_track_attribute(empty_nwb, key, value, reason):
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.io.nwb import write_environment
    from neurospatial.io.nwb._environment import DEFAULT_ENVIRONMENT_NAME

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0)
    clean = Environment.from_graph(graph, [(0, 1)], edge_spacing=0.0, bin_size=2.0)
    graph.nodes[0][key] = value
    track = Environment.from_graph(graph, [(0, 1)], edge_spacing=0.0, bin_size=2.0)

    with pytest.raises(
        ValueError, match=f"track graph node 0 attribute '{key}'"
    ) as error:
        write_environment(empty_nwb, track)

    assert reason in str(error.value)
    assert DEFAULT_ENVIRONMENT_NAME not in empty_nwb.scratch
    # The same check sees a record the writer does store under that name.
    write_environment(empty_nwb, clean)
    assert DEFAULT_ENVIRONMENT_NAME in empty_nwb.scratch
