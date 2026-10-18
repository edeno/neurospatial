"""Environment transports retain arena semantics and explicit legacy uncertainty."""

from __future__ import annotations

import numpy as np
import pytest
from environment_helpers import publish_environment

from neurospatial import Environment
from neurospatial.io import to_dict


def test_environment_payload_uses_arena_schema():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    payload = to_dict(env)
    assert payload["schema_version"] == "Environment-v3"
    assert isinstance(payload["arena"], dict)
    assert payload["arena"]["codec_version"] == 2


def test_builtin_arena_round_trip(built_environment, round_trip_environment):
    from neurospatial.geometry._codecs import encode_arena

    original = built_environment
    loaded = round_trip_environment(original)
    original_metadata, original_arrays = encode_arena(original.arena)
    loaded_metadata, loaded_arrays = encode_arena(loaded.arena)
    assert loaded_metadata == original_metadata
    assert loaded_arrays.keys() == original_arrays.keys()
    for name, array in original_arrays.items():
        np.testing.assert_array_equal(loaded_arrays[name], array, strict=True)
    assert loaded.arena.domain_known == original.arena.domain_known
    assert (
        loaded.arena.physical_footprint_known == original.arena.physical_footprint_known
    )


@pytest.mark.parametrize("physical_footprint_known", [False, True])
def test_open_arena_walls_and_provenance_round_trip(
    round_trip_environment, physical_footprint_known
):
    from shapely.geometry import LineString, box

    from neurospatial.arena import OpenArena

    base = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    domain = box(0, 0, 2, 2)
    wall = LineString([(1, 0), (1, 1)])
    original = publish_environment(
        base.name,
        base,
        base.layout_type,
        base.layout_parameters,
        arena=OpenArena(
            domain, walls=(wall,), physical_footprint_known=physical_footprint_known
        ),
    )
    loaded = round_trip_environment(original)
    assert loaded.arena.polygonal_domain().equals_exact(domain, 0)
    assert loaded.arena.physical_footprint_known is physical_footprint_known
    assert len(loaded.arena.walls) == 1
    assert loaded.arena.walls[0].equals_exact(wall, 0)
    np.testing.assert_array_equal(loaded.bin_centers, original.bin_centers)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_nested_transformed_mask_arena_round_trip(round_trip_environment, n_dims):
    from neurospatial.geometry._codecs import encode_arena
    from neurospatial.ops.transforms import AffineND

    mask = np.ones((2,) * n_dims, dtype=bool)
    mask[(0,) * n_dims] = False
    env = Environment.from_grid_mask(mask, grid_edges=(np.arange(3.0),) * n_dims)
    shear = np.eye(n_dims + 1)
    shear[0, 1] = 0.5
    shift = np.eye(n_dims + 1)
    shift[:-1, -1] = np.arange(1, n_dims + 1)
    env = env.apply_transform(AffineND(shear)).apply_transform(AffineND(shift))
    loaded = round_trip_environment(env)
    original_metadata, original_arrays = encode_arena(env.arena)
    loaded_metadata, loaded_arrays = encode_arena(loaded.arena)
    assert loaded_metadata == original_metadata
    for name, array in original_arrays.items():
        np.testing.assert_array_equal(loaded_arrays[name], array, strict=True)
    points = np.concatenate([env.bin_centers, np.full((1, n_dims), -10.0)])
    np.testing.assert_array_equal(
        loaded.arena.contains(points), env.arena.contains(points)
    )
    assert loaded.arena.physical_footprint_known is False


def _make_legacy(metadata, arrays):
    metadata["schema_version"] = (
        "2.0" if metadata["schema_version"] == "3.0" else "Environment-v2"
    )
    metadata.pop("arena")
    for name in list(metadata["array_specs"]):
        if name.startswith("arena/"):
            metadata["array_specs"].pop(name)
            arrays.pop(name)


def test_v2_restores_unknown_arena_or_refuses_polar(
    built_environment, corrupted_environment_round_trip, request
):
    from neurospatial.arena import UnknownArena

    request.getfixturevalue("forbid_layout_builds")
    original = built_environment
    if original.coordinate_kind == "polar":
        # Without a stored arena, nothing declares whether the angle wraps.
        with pytest.raises(ValueError, match="EgocentricSurround"):
            corrupted_environment_round_trip(_make_legacy)
        return
    loaded = corrupted_environment_round_trip(_make_legacy)
    assert isinstance(loaded.arena, UnknownArena)
    assert loaded.arena.domain_known is False
    assert loaded.arena.physical_footprint_known is False
    assert loaded.arena.n_dims == original.n_dims
    assert loaded.arena.coordinate_kind == original.arena.coordinate_kind
    assert not loaded.arena.walls
    np.testing.assert_array_equal(loaded.bin_centers, original.bin_centers)
    np.testing.assert_array_equal(loaded.bin_sizes, original.bin_sizes)


@pytest.mark.parametrize("built_environment", ["polygon"], indirect=True)
@pytest.mark.parametrize(
    "corruption",
    ["missing", "not_record", "codec", "version", "dimensions", "coordinate_kind"],
)
def test_malformed_arena_names_source(corrupted_environment_round_trip, corruption):
    from neurospatial.arena import UnknownArena
    from neurospatial.geometry._codecs import encode_arena

    def corrupt(metadata, arrays):
        if corruption == "missing":
            metadata.pop("arena")
        elif corruption == "not_record":
            metadata["arena"] = None
        elif corruption == "codec":
            metadata["arena"]["codec"] = "unknown-arena-kind"
        elif corruption == "version":
            metadata["arena"]["codec_version"] = 100
        else:
            arena = (
                UnknownArena(3)
                if corruption == "dimensions"
                else UnknownArena(2, "polar")
            )
            metadata["arena"], _ = encode_arena(arena)

    with pytest.raises(
        ValueError,
        match=r"environment dictionary|corrupt_environment\.json|NWB scratch/spatial_environment",
    ) as error:
        corrupted_environment_round_trip(corrupt)
    assert "arena" in str(error.value).lower()


@pytest.mark.parametrize("built_environment", ["polygon"], indirect=True)
def test_required_arena_array_must_be_registered(corrupted_environment_round_trip):
    def corrupt(metadata, arrays):
        name = next(
            name for name in metadata["array_specs"] if name.startswith("arena/")
        )
        metadata["array_specs"].pop(name)

    with pytest.raises(ValueError, match="arena"):
        corrupted_environment_round_trip(corrupt)


@pytest.mark.parametrize("built_environment", ["polygon"], indirect=True)
def test_malformed_arena_wkb_names_source(corrupted_environment_round_trip):
    def corrupt(metadata, arrays):
        arrays["arena/polygon_wkb"][:] = 0

    with pytest.raises(
        ValueError,
        match=r"environment dictionary|corrupt_environment\.json|NWB scratch/spatial_environment",
    ) as error:
        corrupted_environment_round_trip(corrupt)
    assert "arena" in str(error.value).lower()
