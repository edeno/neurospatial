"""Binning derivation preserves restricted geometry without consulting builders."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.geometry._binning import KnownFaces, MissingFaces
from neurospatial.geometry._components import TriangleShapes
from neurospatial.geometry._derive import subset_binning, transform_binning
from neurospatial.geometry._locators import (
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.ops.transforms import AffineND


@pytest.mark.parametrize("built_environment", ["mesh", "mesh_boundary"], indirect=True)
@pytest.mark.parametrize(
    "linear",
    [
        np.array([[1.0, 0.25], [0.0, 1.0]]),
        np.diag([2.0, 0.5]),
        np.diag([-1.0, 1.0]),
    ],
    ids=["shear", "unequal-scaling", "reflection"],
)
def test_affine_triangle_shapes_keep_exact_vertices_and_connectivity(
    built_environment, linear
):
    parent = built_environment._binning
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = (1.25, -2.5)
    transform = AffineND(matrix)
    original_points = parent.shapes.points.copy()
    original_triangles = parent.shapes.triangles.copy()

    child = transform_binning(parent, transform)

    assert isinstance(child.shapes, TriangleShapes)
    np.testing.assert_array_equal(child.shapes.points, transform(original_points))
    np.testing.assert_array_equal(child.shapes.triangles, original_triangles)
    np.testing.assert_array_equal(parent.shapes.points, original_points)
    np.testing.assert_array_equal(parent.shapes.triangles, original_triangles)
    assert not child.shapes.points.flags.writeable
    assert not child.shapes.triangles.flags.writeable
    assert not np.shares_memory(child.shapes.points, parent.shapes.points)


def test_subset_binning_restricts_arrays_and_pair_order(built_environment):
    parent = built_environment._binning
    keep = np.arange(parent.bins.n_bins) % 3 != 1
    keep[-1] = True
    selected = np.flatnonzero(keep)
    child_of = np.full(len(keep), -1, dtype=np.intp)
    child_of[selected] = np.arange(len(selected))
    paired = keep[parent.bins.pairs].all(axis=1)
    child = subset_binning(parent, keep)
    np.testing.assert_array_equal(child.bins.centers, parent.bins.centers[keep])
    np.testing.assert_array_equal(child.bins.sizes, parent.bins.sizes[keep])
    np.testing.assert_array_equal(child.bins.pairs, child_of[parent.bins.pairs[paired]])
    np.testing.assert_array_equal(
        child.bins.pair_distances, parent.bins.pair_distances[paired]
    )
    assert child.bins.measure_dim == parent.bins.measure_dim
    assert isinstance(child.faces, KnownFaces)
    expected_faces = parent.faces.build()
    actual_faces = child.faces.build()
    np.testing.assert_array_equal(actual_faces.edges, child.bins.pairs)
    np.testing.assert_array_equal(actual_faces.A, expected_faces.A[paired])
    np.testing.assert_array_equal(
        actual_faces.distance, expected_faces.distance[paired]
    )
    np.testing.assert_array_equal(actual_faces.volumes, parent.bins.sizes[keep])
    for array in (
        child.bins.centers,
        child.bins.sizes,
        child.bins.pairs,
        child.bins.pair_distances,
        *actual_faces,
    ):
        assert not array.flags.writeable
    assert keep.flags.writeable
    assert child.outline is None
    if parent.rectilinear is not None:
        assert isinstance(child.locator, RectilinearLocator)
        expected_mask = parent.rectilinear.active_mask.copy()
        expected_mask.ravel()[np.flatnonzero(expected_mask)] = keep
        np.testing.assert_array_equal(child.rectilinear.active_mask, expected_mask)
    else:
        assert isinstance(child.locator, SubsetLocator)
        np.testing.assert_array_equal(child.locator.parent_to_child, child_of)
    np.testing.assert_array_equal(child.locator.locate(parent.bins.centers), child_of)
    if parent.shapes is not None:
        if hasattr(parent.shapes, "centers"):
            np.testing.assert_array_equal(
                child.shapes.centers, parent.shapes.centers[keep]
            )
        else:
            np.testing.assert_array_equal(child.shapes.points, parent.shapes.points)
            np.testing.assert_array_equal(
                child.shapes.triangles, parent.shapes.triangles[keep]
            )
    if parent.linearization is not None:
        original_mask = parent.linearization.active_mask
        expected_mask = original_mask.copy()
        expected_mask[np.flatnonzero(original_mask)] = keep
        np.testing.assert_array_equal(child.linearization.active_mask, expected_mask)
        for actual, expected in zip(
            child.linearization.grid_edges, parent.linearization.grid_edges, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        points = parent.bins.centers
        np.testing.assert_array_equal(
            child.linearization.to_linear(points),
            parent.linearization.to_linear(points),
        )


def test_subset_keeps_missing_face_reason(built_environment):
    parent = replace(
        built_environment._binning,
        faces=MissingFaces("Face normals were not provided."),
    )
    keep = np.arange(parent.bins.n_bins) % 2 == 0
    child = subset_binning(parent, keep)
    assert not child.faces.available
    assert child.faces.reason == parent.faces.reason


@pytest.mark.parametrize(
    "keep",
    [
        np.zeros(4, dtype=bool),
        np.ones(3, dtype=bool),
        np.ones(4, dtype=int),
        np.ones((2, 2), dtype=bool),
    ],
)
def test_subset_binning_refuses_invalid_selections(keep):
    parent = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )._binning
    with pytest.raises(ValueError) as caught:
        subset_binning(parent, keep)
    assert "Why:" in str(caught.value)
    assert "Fix:" in str(caught.value)
    assert keep.flags.writeable


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([0.0, 1.0, 1.0]),
        np.array([[1.0, 0.0, np.inf], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.eye(4),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]]),
    ],
)
def test_transform_binning_refuses_invalid_affine_maps(matrix):
    parent = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )._binning
    with pytest.raises(ValueError) as caught:
        transform_binning(parent, AffineND(matrix))
    assert "Why:" in str(caught.value)
    assert "Fix:" in str(caught.value)
    assert matrix.flags.writeable


def test_three_dimensional_axis_scaling_keeps_physical_faces():
    parent = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 3,
        connect_diagonal_neighbors=False,
    )
    transform = AffineND(np.diag([2.0, 3.0, 4.0, 1.0]))
    child = parent.apply_transform(transform)
    expected = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        grid_edges=tuple(np.arange(3.0) * scale for scale in [2.0, 3.0, 4.0]),
        connect_diagonal_neighbors=False,
    )
    np.testing.assert_allclose(child.bin_sizes, parent.bin_sizes * 24.0, rtol=1e-12)
    for actual, wanted in zip(
        child._face_geometry, expected._face_geometry, strict=True
    ):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        child.compute_kernel(2.0), expected.compute_kernel(2.0), rtol=1e-12, atol=1e-12
    )


def test_reflection_uses_inverse_locator_and_preserves_face_geometry():
    parent = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    child = transform_binning(parent._binning, AffineND(np.diag([-1.0, 1.0, 1.0])))
    assert isinstance(child.locator, TransformedLocator)
    assert child.rectilinear is None
    for actual, expected in zip(
        child.faces.build(), parent._face_geometry, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
