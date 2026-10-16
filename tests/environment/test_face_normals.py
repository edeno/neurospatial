"""Face directions preserve physical geometry without changing solver behavior."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest
from environment_helpers import publish_binning
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.binning import FaceNormals, KnownFaces
from neurospatial.environment._fields import face_geometry
from neurospatial.geometry._codecs import decode, encode
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.io import from_dict, to_dict
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import Affine2D, AffineND


def _grid(n_dims=2, diagonal=True):
    return Environment.from_grid_mask(
        np.ones((3,) * n_dims, dtype=bool),
        grid_edges=tuple(np.arange(4.0) * (d + 1) for d in range(n_dims)),
        connect_diagonal_neighbors=diagonal,
    )


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("diagonal", [False, True])
def test_native_grid_normals_match_shared_faces(n_dims, diagonal):
    env = _grid(n_dims, diagonal)
    values = env._binning.face_normals.values
    faces = face_geometry(env._cache, env._binning)
    delta = (
        env.bin_centers[env.edge_pairs[:, 1]] - env.bin_centers[env.edge_pairs[:, 0]]
    )
    active = faces.A > 0
    np.testing.assert_array_equal(values[~active], 0)
    np.testing.assert_array_equal(np.count_nonzero(values[active], axis=1), 1)
    np.testing.assert_array_equal(values[active], np.sign(delta[active]))
    # A face normal's axis is omitted from its transverse area product.
    widths = np.arange(1, n_dims + 1)
    axes = np.argmax(np.abs(values[active]), axis=1)
    np.testing.assert_array_equal(faces.A[active], np.prod(widths) / widths[axes])


@pytest.mark.parametrize(
    "built_environment", ["hex", "mesh", "mesh_boundary"], indirect=True
)
def test_planar_normals_are_perpendicular_to_true_shared_edges(built_environment):
    env = built_environment
    normals = env._binning.face_normals.values
    cells = materialize_cells(env._binning)
    for index, (u, v) in enumerate(env.edge_pairs):
        shared = cells[u].boundary.intersection(cells[v].boundary)
        if face_geometry(env._cache, env._binning).A[index] == 0:
            np.testing.assert_array_equal(normals[index], 0)
            continue
        coordinates = np.asarray(shared.coords)
        tangent = coordinates[-1] - coordinates[0]
        assert abs(tangent @ normals[index]) < 1e-12 * np.linalg.norm(tangent)
        assert (env.bin_centers[v] - env.bin_centers[u]) @ normals[index] > 0


@pytest.mark.parametrize(
    "built_environment", ["grid8", "grid4", "mask", "hex", "mesh"], indirect=True
)
def test_subset_normal_rows_keep_edge_orientation(built_environment):
    env = built_environment
    keep = np.arange(env.n_bins) % 3 != 0
    child = env.subset(mask=keep)
    edges = keep[env.edge_pairs].all(axis=1)
    np.testing.assert_array_equal(
        child._binning.face_normals.values, env._binning.face_normals.values[edges]
    )


def test_clipping_recomputes_certified_normals():
    env = _grid(diagonal=False).clip(box(0.25, 0.25, 2.75, 5.75))
    assert isinstance(env._binning.faces, KnownFaces)
    values = env._binning.face_normals.values
    cells = materialize_cells(env._binning)
    for index, (u, v) in enumerate(env.edge_pairs):
        shared = cells[u].boundary.intersection(cells[v].boundary)
        tangent = np.diff(np.asarray(shared.coords)[[0, -1]], axis=0)[0]
        assert abs(tangent @ values[index]) < 1e-12


@pytest.mark.parametrize(
    "linear",
    [
        [[0.0, -1.0], [1.0, 0.0]],
        [[-1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 3.0]],
        [[1.0, 0.3], [0.0, 1.0]],
    ],
)
def test_transformed_normals_follow_actual_face_tangents(linear):
    env = _grid(diagonal=False)
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, -1] = [3.0, -2.0]
    child = env.apply_transform(Affine2D(matrix))
    values = child._binning.face_normals.values
    source = env._binning.face_normals.values
    tangents = np.column_stack((-source[:, 1], source[:, 0])) @ np.asarray(linear).T
    np.testing.assert_allclose(np.sum(values * tangents, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0)
    delta = (
        child.bin_centers[child.edge_pairs[:, 1]]
        - child.bin_centers[child.edge_pairs[:, 0]]
    )
    assert np.all(np.sum(delta * values, axis=1) > 0)


def test_3d_normals_follow_cross_product_oracle():
    env = _grid(3, diagonal=False)
    linear = np.array([[1.0, 0.3, 0.1], [0.0, 2.0, 0.2], [0.0, 0.0, -1.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    child = env.apply_transform(AffineND(matrix))
    for n, actual in zip(
        env._binning.face_normals.values,
        child._binning.face_normals.values,
        strict=True,
    ):
        axis = np.argmax(np.abs(n))
        other = [d for d in range(3) if d != axis]
        cross = np.cross(linear[:, other[0]], linear[:, other[1]])
        cross /= np.linalg.norm(cross)
        cross *= np.sign(cross @ (linear @ n))
        np.testing.assert_allclose(actual, cross, atol=1e-12)


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))]
)
def test_standalone_normal_ownership(restore):
    source = np.array([[1.0, 0.0], [0.0, 0.0]])
    normals = FaceNormals(source)
    source[:] = 0
    np.testing.assert_array_equal(normals.values, [[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="read-only"):
        normals.values[0, 0] = 0
    restored = restore(normals)
    assert not restored.values.flags.writeable
    assert not np.shares_memory(restored.values, normals.values)


@pytest.mark.parametrize(
    "invalid",
    [
        np.ones(3),
        np.ones((2, 0)),
        [[1.0, 1.0]],
        [[np.nan, 0.0]],
        [[np.inf, 0.0]],
        [[True, False]],
        [[1j, 0j]],
        np.full((1, 3), np.finfo(float).max),
    ],
)
def test_invalid_normal_records_refused(invalid):
    with pytest.raises(ValueError, match="face normals"):
        FaceNormals(invalid)


@pytest.mark.parametrize(
    "corruption",
    ["reverse", "tangent", "active_zero", "path_unit", "wrong_shape", "header"],
)
def test_explicit_bins_reject_inconsistent_normals(corruption):
    env = _grid()
    binning = replace(
        env._binning, faces=KnownFaces(face_geometry(env._cache, env._binning))
    )
    values = binning.face_normals.values.copy()
    active = np.flatnonzero(face_geometry(env._cache, env._binning).A > 0)[0]
    if corruption == "reverse":
        values[active] *= -1
    elif corruption == "tangent":
        values[active] = [1.0, 1.0]
        values[active] /= np.sqrt(2)
    elif corruption == "active_zero":
        values[active] = 0
    elif corruption == "path_unit":
        values[np.flatnonzero(face_geometry(env._cache, env._binning).A == 0)[0]] = [
            1.0,
            0.0,
        ]
    elif corruption == "wrong_shape":
        values = values[:-1]
    normals = FaceNormals(values)
    if corruption == "header":
        normals.values.shape = (values.size, 1)
    with pytest.raises(ValueError, match="face normals"):
        Environment.from_bins(replace(binning, face_normals=normals))


@pytest.mark.parametrize(
    "built_environment", ["grid8", "hex", "mesh", "mask"], indirect=True
)
@pytest.mark.parametrize(
    "environment_round_trip",
    ["dict", "file", pytest.param("nwb", marks=pytest.mark.nwb)],
    indirect=True,
)
def test_environment_round_trip_keeps_normal_ownership(environment_round_trip):
    env, restore = environment_round_trip
    restored = restore()
    expected = env._binning.face_normals.values
    actual = restored._binning.face_normals.values
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert not np.shares_memory(actual, expected)


def test_legacy_binning_keeps_old_transform_capabilities():
    env = _grid()
    payload = to_dict(env)
    payload["binning"]["codec_version"] = 1
    del payload["binning"]["face_normals"]
    # Legacy records cannot contain a hidden newer optional component.
    for key in list(payload["arrays"]):
        if key.startswith("binning/face_normals/"):
            del payload["arrays"][key]
            del payload["array_specs"][key]
    loaded = from_dict(payload)
    assert loaded._binning.face_normals is None
    matrix = np.eye(3)
    matrix[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    original = env.apply_transform(Affine2D(matrix))
    legacy = loaded.apply_transform(Affine2D(matrix))
    for actual, expected in zip(
        face_geometry(legacy._cache, legacy._binning),
        face_geometry(original._cache, original._binning),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_current_binning_requires_normal_field_and_other_codecs_stay_version_one():
    env = _grid()
    meta, arrays = encode(env._binning, "binning")
    assert meta["codec_version"] == 3
    assert meta["face_normals"]["codec_version"] == 1
    del meta["face_normals"]
    with pytest.raises(ValueError, match="face_normals"):
        decode(meta, arrays, "binning")
    meta, arrays = encode(env._binning.face_normals, "normals")
    meta["codec_version"] = 2
    with pytest.raises(ValueError, match="codec_version"):
        decode(meta, arrays, "normals")


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9])
def test_rotated_hex_normals_use_source_lattice_at_large_offsets(offset):
    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0.0, 12.0), (0.0, 12.0)]
        )
    )
    angle = 0.3
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), offset],
            [np.sin(angle), np.cos(angle), -offset],
            [0.0, 0.0, 1.0],
        ]
    )
    child = env.apply_transform(Affine2D(matrix))
    expected = env._binning.face_normals.values @ matrix[:2, :2].T
    np.testing.assert_allclose(child._binning.face_normals.values, expected, atol=1e-12)
    loaded = from_dict(to_dict(child))
    np.testing.assert_array_equal(
        loaded._binning.face_normals.values, child._binning.face_normals.values
    )


def test_physical_wall_removes_the_matching_normal_rows():
    from shapely.geometry import LineString

    from neurospatial.arena import OpenArena

    parent = _grid()
    wall = LineString([(1.0, 0.0), (1.0, 6.0)])
    child = Environment.from_grid_mask(
        parent.active_mask,
        grid_edges=parent.grid_edges,
        arena=OpenArena(box(0.0, 0.0, 3.0, 6.0), walls=(wall,)),
    )
    expected = {
        tuple(pair): normal
        for pair, normal in zip(
            parent.edge_pairs, parent._binning.face_normals.values, strict=True
        )
    }
    assert len(child.edge_pairs) < len(parent.edge_pairs)
    np.testing.assert_array_equal(
        child._binning.face_normals.values,
        [expected[tuple(pair)] for pair in child.edge_pairs],
    )


def test_explicit_pair_reordering_reverses_only_its_normal():
    from neurospatial.binning import FaceGeometry, SpatialBins

    env = _grid(diagonal=False)
    order = np.arange(len(env.edge_pairs))[::-1]
    pairs = env.edge_pairs[order, ::-1]
    bins = SpatialBins(
        env.bin_centers,
        env.bin_sizes,
        2,
        pairs,
        env._binning.bins.pair_distances[order],
    )
    old = face_geometry(env._cache, env._binning)
    faces = KnownFaces(
        FaceGeometry(pairs, old.A[order], old.distance[order], env.bin_sizes)
    )
    normals = -env._binning.face_normals.values[order]
    child = Environment.from_bins(
        replace(env._binning, bins=bins, faces=faces, face_normals=FaceNormals(normals))
    )
    np.testing.assert_array_equal(child.edge_pairs, pairs)
    np.testing.assert_array_equal(child._binning.face_normals.values, normals)


def test_native_capture_does_not_materialize_faces(monkeypatch):
    from neurospatial import Environment
    from neurospatial.geometry._native_faces import NativeFaces

    def refuse(self):
        pytest.fail("native construction must retain lazy face building")

    monkeypatch.setattr(NativeFaces, "build", refuse)
    env = Environment.from_samples(
        np.array([[0.0, 0.0], [3.0, 4.0]]), bin_size=1.0, infer_active_bins=False
    )
    assert env._binning.face_normals is not None
    bridge = publish_binning(
        env._binning,
        connectivity=env.to_networkx(),
        dimension_ranges=env.dimension_ranges,
        layout_type=env.layout_type,
        category=env._provenance.category,
        build_params=env.layout_parameters,
        is_linearized_track=False,
    )
    assert bridge._binning.face_normals is not None


@pytest.mark.parametrize("built_environment", ["mesh"], indirect=True)
def test_restored_mesh_normal_validation_keeps_lookup_lazy(built_environment):
    loaded = from_dict(to_dict(built_environment))
    assert loaded._binning.locator._triangle_lookup is None


def test_storage_refuses_normal_downgrade_and_boolean_arrays():
    env = _grid()
    meta, arrays = encode(env._binning, "binning")
    meta["codec_version"] = 1
    with pytest.raises(ValueError, match="requires Binning codec_version 2"):
        decode(meta, arrays, "binning")
    meta, arrays = encode(env._binning.face_normals, "normals")
    arrays["normals/values"] = arrays["normals/values"].astype(bool)
    with pytest.raises(ValueError, match="face normals"):
        decode(meta, arrays, "normals")


def test_non_geometric_bins_do_not_invent_normals():
    from neurospatial.binning import (
        Binning,
        MissingFaces,
        NearestBinLocator,
        SpatialBins,
    )

    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    bins = SpatialBins(centers, np.ones(2), 2, np.array([[0, 1]]), np.ones(1))
    env = Environment.from_bins(
        Binning(
            "sites",
            bins,
            NearestBinLocator(centers),
            MissingFaces("No physical faces."),
        )
    )
    assert env._binning.face_normals is None


def test_zero_area_contact_stays_navigation_only_after_solver_refusal():
    env = _grid()
    old = face_geometry(env._cache, env._binning)
    # Diagonal neighbors share only a corner, so their physical face really
    # has zero area while their navigation connection remains present.
    contact = np.flatnonzero(old.A == 0)[0]
    parent = Environment.from_bins(replace(env._binning, faces=KnownFaces(old)))
    matrix = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    child = from_dict(to_dict(parent.apply_transform(Affine2D(matrix))))
    np.testing.assert_array_equal(child.edge_pairs, parent.edge_pairs)
    np.testing.assert_array_equal(child._binning.face_normals.values[contact], 0.0)
    assert not child._binning.faces.available
    with pytest.raises(NotImplementedError):
        child.smooth(np.ones(child.n_bins), bandwidth=1.0)


@pytest.mark.parametrize("built_environment", ["polar_open"], indirect=True)
def test_polar_records_refuse_cartesian_normal_components(built_environment):
    env = built_environment
    delta = (
        env.bin_centers[env.edge_pairs[:, 1]] - env.bin_centers[env.edge_pairs[:, 0]]
    )
    normals = delta / np.linalg.norm(delta, axis=1)[:, None]
    with pytest.raises(ValueError, match="Cartesian"):
        publish_binning(
            replace(env._binning, face_normals=FaceNormals(normals)),
            connectivity=env.to_networkx(),
            dimension_ranges=env.dimension_ranges,
            layout_type=env.layout_type,
            category=env._provenance.category,
            build_params=env.layout_parameters,
            is_linearized_track=False,
            coordinate_kind="polar",
        )
