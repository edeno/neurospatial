"""Custom layouts require owned-data migration; derived geometry stays exact."""

from __future__ import annotations

import copy
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from environment_helpers import publish_binning, publish_environment
from matplotlib.patches import RegularPolygon

from neurospatial import Environment
from neurospatial.io import from_dict, to_dict
from neurospatial.ops.transforms import Affine2D


def _owned_layout(parent, tag):
    return publish_binning(
        parent._binning,
        connectivity=parent.connectivity,
        dimension_ranges=parent.dimension_ranges,
        layout_type=tag,
        category=parent._provenance.category,
        build_params={},
        is_linearized_track=parent.is_linearized_track,
    )


class _ExternalGridLayout:
    """A third-party layout with delegated containment and finite-volume faces."""

    def __init__(self, parent):
        self.parent = parent
        self.bin_centers = parent.bin_centers
        self.connectivity = parent.connectivity
        self.dimension_ranges = parent.dimension_ranges
        self.grid_edges = ()
        self.grid_shape = (parent.n_bins,)
        self.active_mask = np.ones(parent.n_bins, dtype=bool)
        self._layout_type_tag = "ExternalGrid"
        self._build_params_used = {}
        self.is_linearized_track = False
        self.layout_type = "custom"
        self.is_grid_compatible = False

    def build(self, **kwargs):
        pass

    def bin_sizes(self):
        return self.parent.bin_sizes

    def finite_volume_geometry(self):
        return self.parent.finite_volume_geometry()

    def point_to_bin_index(self, points):
        return self.parent.bin_at(points)

    def plot(self, ax=None, **kwargs):
        return self.parent.plot(ax=ax, **kwargs)


class _ExternalTrackLayout:
    """A custom track exposes conversions without a built-in track component."""

    def __init__(self, parent):
        self.parent = parent
        self.bin_centers = parent.bin_centers
        self.connectivity = parent.connectivity
        self.dimension_ranges = parent.dimension_ranges
        self.grid_edges = tuple(edge.copy() for edge in parent.grid_edges)
        self.grid_shape = parent.grid_shape
        self.active_mask = parent.active_mask.copy()
        self._layout_type_tag = "ExternalTrack"
        self._build_params_used = {}
        self.is_linearized_track = True
        self.layout_type = "graph"
        self.is_grid_compatible = False

    def build(self, **kwargs):
        pass

    def bin_sizes(self):
        return self.parent.bin_sizes

    def point_to_bin_index(self, points):
        return self.parent.bin_at(points)

    def to_linear(self, points):
        return self.parent.to_linear(points)

    def linear_to_nd(self, coordinates):
        return self.parent.linear_to_nd(coordinates)

    def plot(self, ax=None, **kwargs):
        return self.parent.plot(ax=ax, **kwargs)


@pytest.mark.parametrize("operation", ["subset", "translation", "similarity", "chain"])
@pytest.mark.parametrize(
    "has_linear_grid", [True, False], ids=["linear-grid", "conversions-only"]
)
def test_custom_track_derivations_preserve_absolute_linear_coordinates(
    operation, has_linear_grid
):
    """A custom track migrates to explicit owned data before derivation."""
    native = Environment.linear_track(endpoints=[[0.0, 0.0], [10.0, 0.0]], bin_size=2.0)
    layout = _ExternalTrackLayout(native)
    if not has_linear_grid:
        layout.grid_edges = ()
    with pytest.raises(
        TypeError,
        match=r"No installed snapshot adapter|owned immutable snapshot.*Migrate custom",
    ):
        publish_environment("external_track", layout, "ExternalTrack", {})
    parent = publish_environment(
        "external_track", _owned_layout(native, "ExternalTrack"), "ExternalTrack", {}
    )
    source_writeability = tuple(
        array.flags.writeable for array in (*layout.grid_edges, layout.active_mask)
    )
    translation = Affine2D(
        np.array([[1.0, 0.0, 1.25], [0.0, 1.0, -2.5], [0.0, 0.0, 1.0]])
    )
    similarity = Affine2D(
        np.array([[0.0, -2.0, 1.25], [2.0, 0.0, -2.5], [0.0, 0.0, 1.0]])
    )
    keep = np.array([True, False, True, False, True])
    indices = np.arange(parent.n_bins)
    linear_scale = 1.0
    forward = Affine2D(np.eye(3))
    if operation == "subset":
        child = parent.subset(bins=keep)
        indices = indices[keep]
    elif operation == "translation":
        child = parent.apply_transform(translation)
        forward = translation
    elif operation == "similarity":
        child = parent.apply_transform(similarity)
        forward = similarity
        linear_scale = 2.0
    else:
        child = (
            parent.apply_transform(similarity)
            .subset(bins=keep)
            .apply_transform(similarity.inverse())
            .subset(bins=np.array([True, False, True]))
        )
        indices = indices[keep][[0, 2]]

    expected_centers = forward(native.bin_centers[indices])
    expected_linear = native.bin_centers[indices, 0] * linear_scale
    expected_edges = native.grid_edges[0] * linear_scale
    expected_mask = np.zeros(parent.n_bins, dtype=bool)
    expected_mask[indices] = True
    query = np.array([[0.5, 0.2], [4.5, -0.3], [9.5, 0.1]])
    for restored in (child, copy.deepcopy(child), pickle.loads(pickle.dumps(child))):
        assert restored.is_linearized_track
        assert restored.is_linearized_track
        np.testing.assert_array_equal(restored.bin_centers, expected_centers)
        np.testing.assert_array_equal(
            restored.bin_sizes, native.bin_sizes[indices] * linear_scale
        )
        np.testing.assert_array_equal(
            restored.bin_at(expected_centers), np.arange(len(indices))
        )
        np.testing.assert_allclose(
            restored.to_linear(expected_centers), expected_linear, rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            restored.linear_to_nd(expected_linear),
            expected_centers,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            restored.to_linear(forward(query)),
            query[:, 0] * linear_scale,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_array_equal(restored.grid_edges[0], expected_edges)
        np.testing.assert_array_equal(restored.active_mask, expected_mask)
        component = restored._binning.linearization
        for array in (*component.grid_edges, component.active_mask):
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array.flat[0] = array.flat[0]
        if has_linear_grid:
            assert not np.shares_memory(component.grid_edges[0], layout.grid_edges[0])
        assert not np.shares_memory(component.active_mask, layout.active_mask)
        loaded = from_dict(to_dict(restored))
        np.testing.assert_allclose(loaded.to_linear(expected_centers), expected_linear)

    assert (
        tuple(
            array.flags.writeable for array in (*layout.grid_edges, layout.active_mask)
        )
        == source_writeability
    )


@pytest.mark.parametrize("operation", ["subset", "translation", "chain"])
def test_custom_layout_migrates_to_owned_data_for_derivations_and_save(
    operation,
):
    native = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0) * 10.0,) * 2
    )
    with pytest.raises(
        TypeError,
        match=r"No installed snapshot adapter|owned immutable snapshot.*Migrate custom",
    ):
        publish_environment("external", _ExternalGridLayout(native), "ExternalGrid", {})
    parent = publish_environment(
        "external", _owned_layout(native, "ExternalGrid"), "ExternalGrid", {}
    )
    transform = Affine2D(
        np.array([[1.0, 0.0, 1.25], [0.0, 1.0, -2.5], [0.0, 0.0, 1.0]])
    )
    keep = np.arange(parent.n_bins) % 3 != 1
    if operation == "subset":
        child, expected = parent.subset(bins=keep), native.subset(bins=keep)
    elif operation == "translation":
        child, expected = (
            parent.apply_transform(transform),
            native.apply_transform(transform),
        )
    else:
        child = (
            parent.apply_transform(transform)
            .subset(bins=keep)
            .apply_transform(transform.inverse())
        )
        expected = native.subset(bins=keep)
    points = np.vstack((expected.bin_centers, [[-100.0, -100.0]]))
    np.testing.assert_array_equal(child.bin_at(points), expected.bin_at(points))
    np.testing.assert_array_equal(child.bin_sizes, expected.bin_sizes)
    field = np.sin(np.arange(child.n_bins))
    np.testing.assert_allclose(
        child.smooth(field, 5.0), expected.smooth(field, 5.0), rtol=1e-12, atol=1e-12
    )
    loaded = from_dict(to_dict(child))
    np.testing.assert_array_equal(loaded.bin_at(points), expected.bin_at(points))
    np.testing.assert_array_equal(loaded.bin_sizes, expected.bin_sizes)


@pytest.mark.parametrize("built_environment", ["grid8", "hex", "mesh"], indirect=True)
@pytest.mark.parametrize("selection", ["single", "outer_cells"])
@pytest.mark.parametrize("operation", ["plot", "plot_field"])
def test_subset_plot_bounds_include_whole_retained_cells(
    built_environment, selection, operation
):
    parent = built_environment
    keep = np.zeros(parent.n_bins, dtype=bool)
    keep[0] = True
    if selection == "outer_cells":
        keep[-1] = True
    child = parent.subset(bins=keep)
    if child._binning.rectilinear is not None:
        grid = child._binning.rectilinear
        low = np.array([axis[0] for axis in grid.edges])
        high = np.array([axis[-1] for axis in grid.edges])
    else:
        shapes = child._binning.shapes
        if hasattr(shapes, "radius"):
            vertices = np.concatenate(
                [
                    RegularPolygon(
                        center, 6, radius=shapes.radius, orientation=shapes.orientation
                    ).get_verts()
                    for center in shapes.centers
                ]
            )
        else:
            vertices = shapes.points[shapes.triangles].reshape(-1, 2)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
    figure, axes = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if operation == "plot":
                child.plot(ax=axes)
            else:
                child.plot_field(np.ones(child.n_bins), ax=axes, colorbar=False)
            figure.canvas.draw()
        for axis, limits in enumerate((axes.get_xlim(), axes.get_ylim())):
            assert limits[0] <= low[axis]
            assert limits[1] >= high[axis]
        assert not [
            warning
            for warning in caught
            if "identical low and high" in str(warning.message)
        ]
    finally:
        plt.close(figure)
