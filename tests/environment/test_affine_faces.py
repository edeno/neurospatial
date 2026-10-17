"""Independent geometry and scientific invariants for affine numerical support."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import FaceGeometry, KnownFaces
from neurospatial.environment._fields import face_geometry
from neurospatial.io import from_dict, to_dict
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.ops.transforms import AffineND


def grid(dims=2, *, diagonal=True):
    return Environment.from_grid_mask(
        np.ones((3,) * dims, dtype=bool),
        grid_edges=tuple(np.arange(4.0) * (d + 1) for d in range(dims)),
        connect_diagonal_neighbors=diagonal,
    )


def affine(linear, offset=0.0):
    result = np.eye(len(linear) + 1)
    result[:-1, :-1] = linear
    result[:-1, -1] = offset
    return AffineND(result)


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("reflection", [False, True])
def test_rotated_unequal_scaling_uses_independent_face_area_oracles(dims, reflection):
    env = grid(dims)
    angle = 0.37
    rotation = np.eye(dims)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    scales = np.arange(2.0, dims + 2)
    if reflection:
        scales[0] *= -1
    linear = rotation @ np.diag(scales)
    child = env.apply_transform(affine(linear))
    old, actual = (
        face_geometry(env._cache, env._binning),
        face_geometry(child._cache, child._binning),
    )
    np.testing.assert_allclose(
        actual.volumes, old.volumes * abs(np.prod(scales)), rtol=1e-12
    )
    positive = old.A > 0
    for index in np.flatnonzero(positive):
        normal = env._binning.face_normals.values[index]
        axis = int(np.argmax(np.abs(normal)))
        if dims == 2:
            tangent = np.array([-normal[1], normal[0]]) * old.A[index]
            area = np.linalg.norm(linear @ tangent)
        else:
            other = [d for d in range(3) if d != axis]
            area = (
                np.linalg.norm(np.cross(linear[:, other[0]], linear[:, other[1]]))
                * old.A[index]
            )
        assert actual.A[index] == pytest.approx(area, rel=1e-12)
        u, v = env.edge_pairs[index]
        distance = np.linalg.norm(linear @ (env.bin_centers[v] - env.bin_centers[u]))
        assert actual.distance[index] == pytest.approx(distance, rel=1e-12)
    np.testing.assert_array_equal(actual.A[~positive], 0.0)
    field = np.random.default_rng(417).uniform(size=env.n_bins)
    result = child.diffuse(field * child.bin_sizes, bandwidth=0.7, mode="density")
    assert result @ child.bin_sizes == pytest.approx(field @ child.bin_sizes, rel=1e-12)
    np.testing.assert_allclose(
        child.smooth(np.ones(env.n_bins), bandwidth=0.7), 1.0, atol=1e-12
    )
    flux = gradient(child, np.arange(env.n_bins, dtype=float))
    assert divergence(child, flux) @ child.bin_sizes == pytest.approx(0.0, abs=1e-12)
    loaded = from_dict(to_dict(child))
    np.testing.assert_array_equal(
        face_geometry(loaded._cache, loaded._binning).A, actual.A
    )


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_unequal_scaling_along_rotated_source_axes_is_certified(offset):
    env = grid(diagonal=False)
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    parent = env.apply_transform(affine(rotation, offset))
    linear = rotation @ np.diag([2.0, 3.0]) @ rotation.T
    child = parent.apply_transform(affine(linear))
    assert child._binning.faces.available
    expected = env.apply_transform(affine(rotation @ np.diag([2.0, 3.0])))
    np.testing.assert_allclose(
        face_geometry(child._cache, child._binning).A,
        face_geometry(expected._cache, expected._binning).A,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        face_geometry(child._cache, child._binning).distance,
        face_geometry(expected._cache, expected._binning).distance,
        rtol=1e-12,
    )


@pytest.mark.parametrize("operation", ["smooth", "diffuse", "gradient", "divergence"])
def test_skew_cells_refuse_all_numerical_consumers_but_retain_geometry(operation):
    env = grid().apply_transform(affine(np.array([[1.0, 0.3], [0.0, 1.0]])))
    assert env._binning.grid_frame is not None
    assert np.all(env.bin_at(env.bin_centers) == np.arange(env.n_bins))
    with pytest.raises(NotImplementedError, match=r"face [0-9]+"):
        if operation in ("smooth", "diffuse"):
            getattr(env, operation)(np.ones(env.n_bins), bandwidth=0.7)
        elif operation == "gradient":
            gradient(env, np.ones(env.n_bins))
        else:
            divergence(env, np.zeros(len(env.edge_pairs)))


def test_new_support_requires_located_flux_distances_and_normal_authority():
    env = grid(diagonal=False)
    old = face_geometry(env._cache, env._binning)
    angle = 0.37
    linear = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ) @ np.diag([2.0, 3.0])
    source = Environment.from_bins(
        replace(env._binning, faces=KnownFaces(old), face_normals=None)
    )
    child = source.apply_transform(affine(linear))
    assert not child._binning.faces.available

    # Explicit grid faces must already describe their physical cells before
    # any transform can certify their numerical geometry.
    with pytest.raises(ValueError, match=r"grid faces.distance"):
        Environment.from_bins(
            replace(
                env._binning,
                faces=KnownFaces(
                    FaceGeometry(old.edges, old.A, old.distance * 0.5, old.volumes)
                ),
            )
        )


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_bridge_free_combined_cells_retain_normal_authority(offset):
    left = grid(diagonal=False)
    right = left.apply_transform(affine(np.eye(2), 10.0)).with_metadata(
        frame=left.frame
    )
    joined = Environment.combine({"left": left, "right": right}, auto_bridge=False)
    angle = 0.37
    linear = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ) @ np.diag([2.0, 3.0])
    rotation = linear @ np.diag([0.5, 1 / 3])
    parent = joined.apply_transform(affine(rotation, offset))
    moved = parent.apply_transform(affine(rotation @ np.diag([2.0, 3.0]) @ rotation.T))
    assert moved._binning.faces.available
    np.testing.assert_allclose(
        moved.smooth(np.ones(moved.n_bins), bandwidth=0.7), 1.0, atol=1e-12
    )
    loaded = from_dict(to_dict(moved))
    np.testing.assert_array_equal(
        face_geometry(loaded._cache, loaded._binning).A,
        face_geometry(moved._cache, moved._binning).A,
    )


@pytest.mark.parametrize("derivation", ["native", "transformed", "subset"])
def test_combined_normal_duplicates_must_match_children(derivation):
    from neurospatial.binning import FaceNormals

    left = grid(diagonal=False)
    right = left.apply_transform(affine(np.eye(2), 10.0)).with_metadata(
        frame=left.frame
    )
    env = Environment.combine({"left": left, "right": right}, auto_bridge=False)
    if derivation != "native":
        env = env.apply_transform(affine(np.array([[0.0, -1.0], [1.0, 0.0]]), 1e9))
    if derivation == "subset":
        env = env.subset(mask=np.arange(env.n_bins) % 3 != 1)
    values = env._binning.face_normals.values.copy()
    first = values[0].copy()
    values[0] += 0.1 * np.array([-first[1], first[0]])
    values[0] /= np.linalg.norm(values[0])
    with pytest.raises(ValueError, match=r"combined.*normals"):
        Environment.from_bins(
            replace(env._binning, face_normals=FaceNormals(values)), arena=env.arena
        )


def test_explicit_combined_normals_keep_zero_bridge_rows_through_derivation():
    from neurospatial.geometry._composition import bridge_values, combined_normals

    left = grid(diagonal=False)
    right = left.apply_transform(affine(np.eye(2), 10.0)).with_metadata(
        frame=left.frame
    )
    joined = Environment.combine({"left": left, "right": right})
    normals = combined_normals(joined._binning.locator)
    env = Environment.from_bins(
        replace(joined._binning, face_normals=normals), arena=joined.arena
    )
    for child in (
        env.apply_transform(affine(np.eye(2))),
        env.apply_transform(affine(np.array([[0.0, -2.0], [2.0, 0.0]]))),
        env.subset(mask=np.arange(env.n_bins) != 1),
    ):
        loaded = from_dict(to_dict(child))
        np.testing.assert_array_equal(
            loaded._binning.face_normals.values, child._binning.face_normals.values
        )
        bridges = bridge_values(child._binning)
        rows = [
            i for i, pair in enumerate(child.edge_pairs) if frozenset(pair) in bridges
        ]
        np.testing.assert_array_equal(child._binning.face_normals.values[rows], 0.0)


def test_missing_faces_are_not_resurrected_by_inverse_shear():
    env = grid()
    linear = np.array([[1.0, 0.3], [0.0, 1.0]])
    child = env.apply_transform(affine(linear)).apply_transform(
        affine(np.linalg.inv(linear))
    )
    assert not child._binning.faces.available
    np.testing.assert_allclose(child.bin_centers, env.bin_centers, atol=1e-12)


@pytest.mark.parametrize("dims", [2, 3])
def test_area_vectors_match_seeded_edge_and_cross_product_oracles(dims):
    from neurospatial.binning import FaceNormals
    from neurospatial.geometry._face_transform import transform_normals

    rng = np.random.default_rng(417)
    for _ in range(100):
        linear = rng.normal(size=(dims, dims)) + np.eye(dims)
        if abs(np.linalg.det(linear)) < 0.01:
            continue
        normal = rng.normal(size=dims)
        normal /= np.linalg.norm(normal)
        area = rng.uniform(0.1, 10.0)
        directions, stretches = transform_normals(FaceNormals(normal[None]), linear)
        actual_area = np.exp(np.linalg.slogdet(linear)[1] + stretches[0]) * area
        if dims == 2:
            tangent = np.array([-normal[1], normal[0]]) * area
            transformed = linear @ tangent
            expected_area = np.linalg.norm(transformed)
            expected_normal = (
                np.array([-transformed[1], transformed[0]]) / expected_area
            )
        else:
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            first = np.cross(normal, axis)
            first /= np.linalg.norm(first)
            second = np.cross(normal, first) * area
            cross = np.cross(linear @ first, linear @ second)
            expected_area = np.linalg.norm(cross)
            expected_normal = cross / expected_area
        expected_normal *= np.sign(expected_normal @ (linear @ normal))
        assert actual_area == pytest.approx(expected_area, rel=1e-12)
        np.testing.assert_allclose(directions.values[0], expected_normal, atol=1e-12)


def test_positive_area_underflow_is_refused_in_legacy_similarity_path():
    from neurospatial.binning import Binning, NearestBinLocator, SpatialBins

    bins = SpatialBins(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.full(2, 1e300),
        2,
        np.array([[0, 1]]),
        np.ones(1),
    )
    faces = KnownFaces(
        FaceGeometry(bins.pairs, np.array([1e-200]), np.ones(1), bins.sizes)
    )
    env = Environment.from_bins(
        Binning("declared", bins, NearestBinLocator(bins.centers), faces)
    )
    with pytest.raises(ValueError, match=r"underflow.*rescale"):
        env.apply_transform(affine(np.eye(2) * 1e-150))


def test_affine_isolated_cell_has_zero_flux_and_constant_field():
    env = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(np.arange(2.0),) * 2
    )
    child = env.apply_transform(affine(np.array([[2.0, -3.0], [2.0, 3.0]])))
    assert child._binning.faces.available
    np.testing.assert_allclose(
        child.smooth(np.array([3.0]), bandwidth=0.7), [3.0], atol=1e-12
    )
    assert gradient(child, np.array([3.0])).shape == (0,)
    np.testing.assert_array_equal(divergence(child, np.empty(0)), [0.0])
