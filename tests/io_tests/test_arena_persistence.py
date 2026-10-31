"""Environment transports retain arena semantics and refuse records without one."""

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
    assert payload["schema_version"] == "Environment-v4"
    assert isinstance(payload["arena"], dict)
    assert payload["arena"]["codec_version"] == 3


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
    assert loaded.arena.outline == original.arena.outline
    assert loaded.arena.wall_kinds == original.arena.wall_kinds


@pytest.mark.parametrize("outline", [None, "wall", "transparent_wall", "drop"])
@pytest.mark.parametrize("wall_kind", ["wall", "transparent_wall", "drop"])
def test_open_arena_walls_and_provenance_round_trip(
    round_trip_environment, outline, wall_kind
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
            domain, outline=outline, walls={"divider": wall}, wall_kinds=wall_kind
        ),
    )
    loaded = round_trip_environment(original)
    assert loaded.arena.polygonal_domain().equals_exact(domain, 0)
    assert loaded.arena.outline == outline
    assert loaded.arena.wall_names == ("divider",)
    assert loaded.arena.wall_kinds == (wall_kind,)
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
    assert loaded.arena.outline is None


def _make_version_2(metadata, arrays):
    metadata["schema_version"] = (
        "2.0" if metadata["schema_version"] == "4.0" else "Environment-v2"
    )
    metadata.pop("arena")
    for name in list(metadata["array_specs"]):
        if name.startswith("arena/"):
            metadata["array_specs"].pop(name)
            arrays.pop(name)


def test_version_2_records_are_refused_with_recreation_instructions(
    built_environment, corrupted_environment_round_trip, request
):
    """Version 2 stored no Arena; reading one must not invent an unknown domain."""
    request.getfixturevalue("forbid_layout_builds")
    with pytest.raises(ValueError, match=r"is no longer readable") as error:
        corrupted_environment_round_trip(_make_version_2)
    message = str(error.value)
    assert "'2.0'" in message or "'Environment-v2'" in message
    assert "Fix: recreate the environment with an Environment factory" in message


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


def _make_version_3(metadata, arrays):
    metadata["schema_version"] = (
        "3.0" if metadata["schema_version"] == "4.0" else "Environment-v3"
    )


@pytest.mark.parametrize("built_environment", ["grid8"], indirect=True)
def test_version_3_records_are_refused_with_recreation_instructions(
    built_environment, corrupted_environment_round_trip
):
    """Version 3 stored a connectivity graph; it is refused, never converted."""
    with pytest.raises(ValueError, match=r"is no longer readable") as error:
        corrupted_environment_round_trip(_make_version_3)
    message = str(error.value)
    assert "'3.0'" in message or "'Environment-v3'" in message
    assert "Fix: recreate the environment with an Environment factory" in message
