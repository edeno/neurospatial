"""Derived environments retain bin identity and the geometry their data supports."""

from __future__ import annotations

import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neurospatial import Environment
from neurospatial.io import from_dict, to_dict
from neurospatial.layout._binning import MissingFaces
from neurospatial.layout._binning_layout import BinningLayout
from neurospatial.layout._codecs import encode
from neurospatial.layout._components import TriangleShapes
from neurospatial.ops.transforms import Affine2D

_CARTESIAN_CASES = [
    "grid8",
    "grid4",
    "polygon",
    "mask",
    "pixels",
    "hex",
    "track",
    "mesh",
    "mesh_boundary",
]


def _affine(linear=None, offset=(0.0, 0.0)):
    matrix = np.eye(3)
    if linear is not None:
        matrix[:2, :2] = linear
    matrix[:2, 2] = offset
    return Affine2D(matrix)


def _keep_bins(env):
    keep = np.arange(env.n_bins) % 3 != 1
    keep[-1] = True
    return keep


def _queries(env):
    low = np.min(env.bin_centers, axis=0)
    high = np.max(env.bin_centers, axis=0)
    margin = np.maximum(high - low, 1.0) * 0.2
    return np.vstack(
        (
            env.bin_centers,
            np.random.default_rng(75).uniform(
                low - margin, high + margin, (100, env.n_dims)
            ),
        )
    )


@pytest.mark.parametrize("built_environment", _CARTESIAN_CASES, indirect=True)
class TestCartesianDerivedGeometry:
    def test_subset_keeps_sizes_and_faces(self, built_environment):
        parent = built_environment
        keep = _keep_bins(parent)
        child = parent.subset(bins=keep)
        np.testing.assert_array_equal(child.bin_sizes, parent.bin_sizes[keep])
        child_of = np.full(parent.n_bins, -1, dtype=np.intp)
        child_of[keep] = np.arange(keep.sum())
        faces = parent._face_geometry
        selected = keep[faces.edges].all(axis=1)
        actual = child._face_geometry
        np.testing.assert_array_equal(actual.edges, child_of[faces.edges[selected]])
        np.testing.assert_array_equal(actual.A, faces.A[selected])
        np.testing.assert_array_equal(actual.distance, faces.distance[selected])
        np.testing.assert_array_equal(actual.volumes, faces.volumes[keep])
        for array in (*actual, child.bin_sizes):
            assert not array.flags.writeable
        assert isinstance(child.layout, BinningLayout)

    def test_subset_lookup_uses_parent_bin_ownership(self, built_environment):
        parent = built_environment
        keep = _keep_bins(parent)
        child = parent.subset(bins=keep)
        points = np.vstack((_queries(parent), [[np.nan, 0.0], [np.inf, 0.0]]))
        child_of = np.full(parent.n_bins, -1, dtype=np.intp)
        child_of[keep] = np.arange(keep.sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            parent_ids = parent.bin_at(points)
            actual = child.bin_at(points)
        expected = np.full(len(points), -1, dtype=np.intp)
        inside = parent_ids >= 0
        expected[inside] = child_of[parent_ids[inside]]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(child.bin_at(parent.bin_centers[~keep]), -1)

    def test_translation_preserves_diffusion_lookup_and_grid_edges(
        self, built_environment
    ):
        parent = built_environment
        offset = np.array([1.25, -2.5])
        transform = _affine(offset=offset)
        child = parent.apply_transform(transform)
        field = np.sin(np.arange(parent.n_bins))
        for mode in ("average", "transition", "density"):
            np.testing.assert_allclose(
                child.smooth(field, 2.0, mode=mode),
                parent.smooth(field, 2.0, mode=mode),
                rtol=1e-12,
                atol=1e-12,
            )
        np.testing.assert_array_equal(
            child.get_differential_operator().toarray(),
            parent.get_differential_operator().toarray(),
        )
        points = _queries(parent)
        np.testing.assert_array_equal(
            child.bin_at(transform(points)), parent.bin_at(points)
        )
        np.testing.assert_array_equal(child.bin_sizes, parent.bin_sizes)
        np.testing.assert_array_equal(child.boundary_bins, parent.boundary_bins)
        if parent._binning.rectilinear is not None:
            for axis, edges in enumerate(parent._binning.rectilinear.edges):
                np.testing.assert_array_equal(
                    child.grid_edges[axis], edges + offset[axis]
                )
        if parent.is_linearized_track:
            np.testing.assert_allclose(
                child.to_linear(transform(points)),
                parent.to_linear(points),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_subset_of_transformed_and_back(self, built_environment):
        parent = built_environment
        transform = _affine([[0.0, -1.5], [1.5, 0.0]], (2.5, -3.0))
        keep = _keep_bins(parent)
        actual = (
            parent.apply_transform(transform)
            .subset(bins=keep)
            .apply_transform(transform.inverse())
        )
        expected = parent.subset(bins=keep)
        points = parent.bin_centers
        np.testing.assert_array_equal(actual.bin_at(points), expected.bin_at(points))
        field = np.sin(np.arange(expected.n_bins))
        np.testing.assert_allclose(
            actual.smooth(field, 2.0),
            expected.smooth(field, 2.0),
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.mark.parametrize("built_environment", ["grid8", "grid4"], indirect=True)
def test_grid_subset_equals_mask_build(built_environment):
    parent = built_environment
    child = parent.subset(bins=_keep_bins(parent))
    diagonal = any(
        np.count_nonzero(parent.bin_centers[u] != parent.bin_centers[v]) > 1
        for u, v in parent.connectivity.edges
    )
    expected = Environment.from_grid_mask(
        child.active_mask,
        grid_edges=child.grid_edges,
        connect_diagonal_neighbors=diagonal,
    )
    field = np.sin(np.arange(child.n_bins))
    np.testing.assert_array_equal(child.smooth(field, 2.0), expected.smooth(field, 2.0))
    np.testing.assert_array_equal(
        child.compute_kernel(2.0), expected.compute_kernel(2.0)
    )
    points = _queries(parent)
    np.testing.assert_array_equal(
        child.interpolate(field, points, mode="linear"),
        expected.interpolate(field, points, mode="linear"),
    )


@pytest.mark.parametrize("built_environment", ["grid8", "grid4", "mask"], indirect=True)
@pytest.mark.parametrize("diagonal", [(-1.0, 1.0), (1.0, -1.0)])
def test_reflection_keeps_bin_identity_and_boundary_owners(built_environment, diagonal):
    parent = built_environment
    transform = _affine(np.diag(diagonal))
    child = parent.apply_transform(transform)
    np.testing.assert_array_equal(
        child.bin_at(transform(parent.bin_centers)), np.arange(parent.n_bins)
    )
    grid = parent._binning.rectilinear
    points = np.array(np.meshgrid(*grid.edges, indexing="ij")).reshape(2, -1).T
    np.testing.assert_array_equal(
        child.bin_at(transform(points)), parent.bin_at(points)
    )
    np.testing.assert_array_equal(child.boundary_bins, parent.boundary_bins)
    assert child._binning.rectilinear is None
    with pytest.raises(NotImplementedError, match="axis-aligned"):
        child.interpolate(
            np.ones(child.n_bins), transform(parent.bin_centers), mode="linear"
        )


@pytest.mark.parametrize("origin", [0.0, 1e9])
@pytest.mark.parametrize("scales", [(2.0, 0.5), (1.3, 0.7)])
def test_positive_axis_scaling_fast_path_boundaries(origin, scales):
    parent = Environment.from_grid_mask(
        np.ones((3, 2), dtype=bool),
        grid_edges=(origin + np.arange(4.0) * 0.1, origin + np.arange(3.0) * 0.2),
    )
    transform = _affine(np.diag(scales), (0.1, 0.2))
    child = parent.apply_transform(transform)
    points = np.array(np.meshgrid(*parent.grid_edges, indexing="ij")).reshape(2, -1).T
    np.testing.assert_array_equal(
        child.bin_at(transform(points)), parent.bin_at(points)
    )
    assert child._binning.rectilinear is not None


@pytest.mark.parametrize(
    "built_environment", ["grid8", "hex", "mesh", "track"], indirect=True
)
@settings(derandomize=True, max_examples=6, deadline=None)
@given(
    angle=st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False),
    scale=st.floats(min_value=0.25, max_value=4.0, allow_nan=False),
    offset=st.tuples(
        st.floats(-1e3, 1e3, allow_nan=False), st.floats(-1e3, 1e3, allow_nan=False)
    ),
)
def test_similarity_scales_smoothing(built_environment, angle, scale, offset):
    parent = built_environment
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transform = _affine(scale * rotation, offset)
    child = parent.apply_transform(transform)
    field = np.sin(np.arange(parent.n_bins)) + 2.0
    for mode in ("average", "transition", "density"):
        expected = parent.smooth(field, 2.0, mode=mode)
        if mode == "density":
            expected = expected / scale**parent._binning.bins.measure_dim
        np.testing.assert_allclose(
            child.smooth(field, scale * 2.0, mode=mode),
            expected,
            rtol=1e-10,
            atol=1e-11,
        )
    np.testing.assert_array_equal(
        child.bin_at(transform(parent.bin_centers)), np.arange(parent.n_bins)
    )


@pytest.mark.parametrize("built_environment", ["grid8", "grid4", "mask"], indirect=True)
def test_axis_scaling_grid_equals_scaled_build(built_environment):
    parent = built_environment
    transform = _affine(np.diag([2.0, 0.5]))
    child = parent.apply_transform(transform)
    grid = parent._binning.rectilinear
    diagonal = any(
        np.count_nonzero(parent.bin_centers[u] != parent.bin_centers[v]) > 1
        for u, v in parent.connectivity.edges
    )
    expected = Environment.from_grid_mask(
        grid.active_mask,
        grid_edges=(grid.edges[0] * 2.0, grid.edges[1] * 0.5),
        connect_diagonal_neighbors=diagonal,
    )
    field = np.sin(np.arange(parent.n_bins))
    np.testing.assert_allclose(
        child.smooth(field, 2.0), expected.smooth(field, 2.0), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("built_environment", ["hex", "mesh"], indirect=True)
def test_axis_scaling_non_grid_refuses_smoothing(built_environment):
    parent = built_environment
    transform = _affine(np.diag([2.0, 0.5]))
    child = parent.apply_transform(transform)
    with pytest.raises(NotImplementedError) as caught:
        child.smooth(np.ones(child.n_bins), 2.0)
    text = str(caught.value).lower()
    assert "axis" in text
    assert "right angle" in text
    assert "fix:" in text
    np.testing.assert_array_equal(
        child.bin_at(transform(parent.bin_centers)), np.arange(parent.n_bins)
    )


@pytest.mark.parametrize("built_environment", ["grid8", "hex", "mesh"], indirect=True)
def test_shear_refuses_smoothing_and_scales_measures(built_environment):
    parent = built_environment
    transform = _affine([[1.0, 0.25], [0.0, 2.0]])
    child = parent.apply_transform(transform)
    np.testing.assert_allclose(child.bin_sizes, parent.bin_sizes * 2.0)
    with pytest.raises(NotImplementedError) as caught:
        child.smooth(np.ones(child.n_bins), 2.0)
    text = str(caught.value).lower()
    assert "shear" in text or "affine" in text
    assert "right angle" in text
    assert "fix:" in text


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
@pytest.mark.parametrize(
    "linear",
    [np.diag([2.0, 1.0]), np.diag([2e-5, 1e-5]), np.array([[1.0, 0.25], [0.0, 1.0]])],
)
def test_track_non_similarity_refused(built_environment, linear):
    with pytest.raises(ValueError) as caught:
        built_environment.apply_transform(_affine(linear))
    assert "Environment.from_graph" in str(caught.value)
    assert "Why:" in str(caught.value)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize(
    "built_environment", ["grid8", "hex", "mesh", "track"], indirect=True
)
@pytest.mark.parametrize("angle", [0.3, 0.7, np.pi / 4, 4.144551617385651e-260])
def test_ordinary_rotations_keep_similarity_faces(built_environment, angle):
    parent = built_environment
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    child = parent.apply_transform(_affine(rotation))
    assert child._binning.faces.available
    np.testing.assert_allclose(child.bin_sizes, parent.bin_sizes, rtol=1e-12)
    np.testing.assert_array_equal(child.boundary_bins, parent.boundary_bins)


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_hexagonal_subset_boundary_matches_native_support(built_environment):
    parent = built_environment
    keep = _keep_bins(parent)
    child = parent.subset(bins=keep)
    expected = Environment.from_layout(
        "Hexagonal",
        {
            **parent.layout_parameters,
            "positions": parent.bin_centers[keep],
            "infer_active_bins": True,
            "bin_count_threshold": 0,
        },
    )
    np.testing.assert_array_equal(expected.bin_centers, child.bin_centers)
    np.testing.assert_array_equal(child.boundary_bins, expected.boundary_bins)
    translated = child.apply_transform(_affine(offset=(2.0, -3.0)))
    np.testing.assert_array_equal(translated.boundary_bins, expected.boundary_bins)


def _derive_environment(parent, derivation):
    if derivation == "subset":
        return parent.subset(bins=_keep_bins(parent))
    if derivation == "translation":
        return parent.apply_transform(_affine(offset=(1.25, -2.5)))
    if derivation == "uniform_scaling":
        return parent.apply_transform(_affine(np.diag([1.5, 1.5])))
    if derivation == "reflection":
        return parent.apply_transform(_affine(np.diag([-1.0, 1.0])))
    if derivation == "axis_scaling":
        return parent.apply_transform(_affine(np.diag([2.0, 0.5]), (0.1, 0.2)))
    if derivation == "shear":
        return parent.apply_transform(_affine([[1.0, 0.25], [0.0, 1.0]]))
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    child = parent.apply_transform(_affine(rotation, (2.0, -3.0)))
    if derivation == "nested":
        return child.subset(bins=_keep_bins(parent)).apply_transform(
            _affine(offset=(-0.5, 1.5))
        )
    return child


_ROUND_TRIP_CASES = [
    (derivation, case)
    for derivation in (
        "subset",
        "translation",
        "rotation",
        "axis_scaling",
        "shear",
        "nested",
    )
    for case in _CARTESIAN_CASES
    if case != "track" or derivation not in {"axis_scaling", "shear"}
]


@pytest.mark.parametrize(
    "derivation,built_environment", _ROUND_TRIP_CASES, indirect=["built_environment"]
)
def test_derived_round_trip(
    built_environment, derivation, round_trip_environment, forbid_layout_builds
):
    child = _derive_environment(built_environment, derivation)
    loaded = round_trip_environment(child)
    expected_meta, expected_arrays = encode(child._binning, "binning")
    actual_meta, actual_arrays = encode(loaded._binning, "binning")
    assert actual_meta == expected_meta
    assert actual_arrays.keys() == expected_arrays.keys()
    for name, expected in expected_arrays.items():
        assert actual_arrays[name].dtype == expected.dtype
        np.testing.assert_array_equal(actual_arrays[name], expected, err_msg=name)
    points = _queries(child)
    np.testing.assert_array_equal(loaded.bin_at(points), child.bin_at(points))
    np.testing.assert_array_equal(loaded.boundary_bins, child.boundary_bins)
    assert loaded.units == child.units
    assert loaded.frame == child.frame
    if child._binning.faces.available:
        field = np.sin(np.arange(child.n_bins))
        for mode in ("average", "transition", "density"):
            np.testing.assert_array_equal(
                loaded.smooth(field, 2.0, mode=mode),
                child.smooth(field, 2.0, mode=mode),
            )
    else:
        assert loaded._binning.faces.reason == child._binning.faces.reason
        with pytest.raises(NotImplementedError):
            loaded.smooth(np.ones(loaded.n_bins), 2.0)
    copied = loaded.copy()
    np.testing.assert_array_equal(copied.bin_at(points), child.bin_at(points))
    assert not copied.bin_centers.flags.writeable
    assert not copied.bin_sizes.flags.writeable


@pytest.mark.parametrize("origin", [0.0, 1e9])
def test_mapped_center_roundoff_survives_save_load(origin, round_trip_environment):
    parent = Environment.from_grid_mask(
        np.ones((3, 2), dtype=bool),
        grid_edges=(origin + np.arange(4) * 0.1, origin + np.arange(3) * 0.2),
    )
    transform = _affine(np.diag([1.3, 0.7]), (0.1, 0.2))
    child = parent.apply_transform(transform)
    grid = child._binning.rectilinear
    axes = [(edges[:-1] + edges[1:]) / 2 for edges in grid.edges]
    midpoints = np.stack(
        [axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=1
    )
    assert not np.array_equal(child.bin_centers, midpoints)
    loaded = round_trip_environment(child)
    np.testing.assert_array_equal(loaded.bin_centers, transform(parent.bin_centers))
    np.testing.assert_array_equal(
        loaded.bin_at(transform(parent.bin_centers)), np.arange(parent.n_bins)
    )
    np.testing.assert_array_equal(loaded.bin_sizes, child.bin_sizes)


@pytest.mark.parametrize("built_environment", ["hex", "mesh", "track"], indirect=True)
@pytest.mark.parametrize("corruption", ["centers", "sizes", "component", "inverse"])
def test_wrapped_geometry_corruption_is_rejected(
    built_environment, corruption, corrupted_environment_round_trip
):
    child = _derive_environment(built_environment, "rotation")

    def corrupt(metadata, arrays):
        if corruption == "sizes":
            arrays["binning/bins/sizes"] *= 2.0
            arrays["binning/faces/volumes"] *= 2.0
        elif corruption == "inverse":
            arrays["binning/locator/inverse"][0, -1] += 0.01
        elif corruption == "component":
            if "binning/shapes/centers" in arrays:
                arrays["binning/shapes/centers"][0, 0] += 0.01
            elif "binning/shapes/points" in arrays:
                arrays["binning/shapes/points"][0, 0] += 0.01
            else:
                arrays["binning/linearization/edges_0"][1] += 0.01
        else:
            arrays["binning/bins/centers"] += 0.01
            arrays["bin_centers"] += 0.01
            for node in metadata["graph"]["nodes"]:
                node["pos"] = (np.asarray(node["pos"]) + 0.01).tolist()
            if "binning/shapes/centers" in arrays:
                arrays["binning/shapes/centers"] += 0.01
            elif "binning/shapes/points" in arrays:
                arrays["binning/shapes/points"] += 0.01
            else:
                for node in metadata["binning"]["linearization"]["graph"]["nodes"]:
                    node["pos"] = (np.asarray(node["pos"]) + 0.01).tolist()

    with pytest.raises(ValueError):
        corrupted_environment_round_trip(corrupt, environment=child)


@pytest.mark.parametrize("built_environment", ["mesh", "mesh_boundary"], indirect=True)
@pytest.mark.parametrize("derivation", ["shear", "axis_scaling", "reflection"])
def test_affine_triangle_shapes_survive_nested_subset_and_round_trip(
    built_environment, derivation, round_trip_environment, forbid_layout_builds
):
    parent = built_environment
    mapped = _derive_environment(parent, derivation)
    assert isinstance(mapped._binning.shapes, TriangleShapes)
    keep = _keep_bins(mapped)
    offset = _affine(offset=(-0.5, 1.5))
    child = mapped.subset(bins=keep).apply_transform(offset)
    second_keep = _keep_bins(child)
    child = child.subset(bins=second_keep)
    expected_points = offset(mapped._binning.shapes.points)
    expected_triangles = parent._binning.shapes.triangles[keep][second_keep]
    loaded = round_trip_environment(child)

    for current in (child, loaded):
        shapes = current._binning.shapes
        assert isinstance(shapes, TriangleShapes)
        np.testing.assert_array_equal(shapes.points, expected_points)
        np.testing.assert_array_equal(shapes.triangles, expected_triangles)
        assert not shapes.points.flags.writeable
        assert not shapes.triangles.flags.writeable
        np.testing.assert_array_equal(
            current.bin_at(child.bin_centers), np.arange(child.n_bins)
        )
        field = np.arange(current.n_bins, dtype=float)
        for operation, renderer in (
            ("plot", "PatchCollection"),
            ("plot_field", "PolyCollection"),
        ):
            figure, axes = plt.subplots()
            try:
                if operation == "plot":
                    current.plot(ax=axes)
                else:
                    current.plot_field(field, ax=axes, colorbar=False)
                figure.canvas.draw()
                assert type(axes.collections[0]).__name__ == renderer
                assert len(axes.collections[0].get_paths()) == current.n_bins
            finally:
                plt.close(figure)
        if derivation == "reflection":
            assert current._binning.faces.available
            np.testing.assert_array_equal(
                current.smooth(field, 2.0), child.smooth(field, 2.0)
            )
        else:
            assert not current._binning.faces.available
            with pytest.raises(NotImplementedError, match="right angle"):
                current.smooth(field, 2.0)


@pytest.mark.parametrize("built_environment", ["mesh", "mesh_boundary"], indirect=True)
@pytest.mark.parametrize("corruption", ["points", "triangles"])
def test_retained_affine_triangle_corruption_is_rejected(
    built_environment, corruption, corrupted_environment_round_trip
):
    child = _derive_environment(built_environment, "shear")
    child = child.subset(bins=_keep_bins(child))
    assert isinstance(child._binning.shapes, TriangleShapes)

    def corrupt(metadata, arrays):
        if corruption == "points":
            arrays["binning/shapes/points"][0, 0] += 0.125
        else:
            triangles = arrays["binning/shapes/triangles"]
            triangles[[0, 1]] = triangles[[1, 0]]

    with pytest.raises(ValueError, match=rf"shapes\.{corruption}"):
        corrupted_environment_round_trip(corrupt, environment=child)


@pytest.mark.parametrize(
    "built_environment,derivation,renderer",
    [
        ("grid8", "translation", "QuadMesh"),
        ("grid8", "uniform_scaling", "QuadMesh"),
        ("grid8", "axis_scaling", "QuadMesh"),
        ("grid8", "subset", "QuadMesh"),
        ("hex", "translation", "PatchCollection"),
        ("hex", "rotation", "PatchCollection"),
        ("hex", "uniform_scaling", "PatchCollection"),
        ("hex", "axis_scaling", "PatchCollection"),
        ("hex", "reflection", "PatchCollection"),
        ("mesh", "translation", "PolyCollection"),
        ("mesh", "rotation", "PolyCollection"),
        ("mesh", "uniform_scaling", "PolyCollection"),
        ("mesh", "reflection", "PolyCollection"),
        ("mesh", "axis_scaling", "PolyCollection"),
        ("mesh", "shear", "PolyCollection"),
        ("track", "translation", "FillBetweenPolyCollection"),
    ],
    indirect=["built_environment"],
)
def test_derived_plot_field_keeps_supported_renderer(
    built_environment, derivation, renderer
):
    child = _derive_environment(built_environment, derivation)
    figure, axes = plt.subplots()
    try:
        child.plot_field(np.arange(child.n_bins, dtype=float), ax=axes, colorbar=False)
        assert type(axes.collections[0]).__name__ == renderer
        if renderer == "FillBetweenPolyCollection":
            assert axes.lines
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "built_environment",
    [case for case in _CARTESIAN_CASES if case != "track"],
    indirect=True,
)
def test_translated_plot_paths_follow_cell_geometry(built_environment):
    parent = built_environment
    offset = np.array([1.25, -2.5])
    child = parent.apply_transform(_affine(offset=offset))
    before, before_axes = plt.subplots()
    after, after_axes = plt.subplots()
    try:
        field = np.arange(parent.n_bins, dtype=float)
        parent.plot_field(field, ax=before_axes, colorbar=False)
        child.plot_field(field, ax=after_axes, colorbar=False)
        before.canvas.draw()
        after.canvas.draw()
        assert len(after_axes.collections) == len(before_axes.collections)
        for actual, expected in zip(
            after_axes.collections, before_axes.collections, strict=True
        ):
            assert type(actual) is type(expected)
            assert len(actual.get_paths()) == len(expected.get_paths())
            for child_path, parent_path in zip(
                actual.get_paths(), expected.get_paths(), strict=True
            ):
                np.testing.assert_allclose(
                    child_path.vertices,
                    parent_path.vertices + offset,
                    rtol=1e-12,
                    atol=1e-12,
                )
            np.testing.assert_array_equal(
                actual.get_facecolors(), expected.get_facecolors()
            )
    finally:
        plt.close(before)
        plt.close(after)


@pytest.mark.parametrize(
    "faces_available", [True, False], ids=["known_faces", "missing_faces"]
)
def test_extreme_axis_reflection_does_not_hide_wrong_measures(
    round_trip_environment, faces_available
):
    parent = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(np.arange(2.0),) * 2
    )
    child = parent.apply_transform(_affine(np.diag([-1e10, 1e-10])))
    if not faces_available:
        child._geometry = replace(
            child._geometry,
            binning=replace(
                child._binning, faces=MissingFaces("Cell faces are unavailable.")
            ),
        )
    loaded = round_trip_environment(child)
    np.testing.assert_array_equal(loaded.bin_sizes, child.bin_sizes)
    np.testing.assert_array_equal(loaded.bin_at(child.bin_centers), [0])
    payload = to_dict(child)
    payload["arrays"]["binning/bins/sizes"][0] = 1.2
    if faces_available:
        payload["arrays"]["binning/faces/volumes"][0] = 1.2
    with pytest.raises(ValueError, match=r"sizes|volumes|measures"):
        from_dict(payload)
