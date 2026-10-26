"""Plots preserve artists while reading geometry from the published binning."""

from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from environment_helpers import native_layout
from matplotlib.colors import to_rgba
from shapely.geometry import MultiPolygon, Polygon

from neurospatial import Environment
from neurospatial.layout import create_layout


def record_path(data, prefix, path):
    data[f"{prefix}/vertices"] = np.array(path.vertices, copy=True)
    data[f"{prefix}/codes"] = np.array(
        [] if path.codes is None else path.codes, dtype=np.uint8
    )
    data[f"{prefix}/has_codes"] = np.asarray(path.codes is not None)


def record_artist(data, prefix, artist):
    data[f"{prefix}/class"] = np.asarray(type(artist).__name__)
    data[f"{prefix}/zorder"] = np.asarray(artist.get_zorder())
    data[f"{prefix}/visible"] = np.asarray(artist.get_visible())
    data[f"{prefix}/transform"] = np.array(
        artist.get_transform().get_matrix(), copy=True
    )
    if hasattr(artist, "get_paths"):
        paths = artist.get_paths()
    elif hasattr(artist, "get_path"):
        paths = [artist.get_path()]
    else:
        paths = []
    data[f"{prefix}/n_paths"] = np.asarray(len(paths))
    for index, path in enumerate(paths):
        record_path(data, f"{prefix}/path_{index}", path)
    for name in (
        "facecolor",
        "edgecolor",
        "linewidth",
        "linewidths",
        "offsets",
        "sizes",
    ):
        getter = getattr(artist, f"get_{name}", None)
        if getter is not None:
            value = getter()
            if value is not None:
                data[f"{prefix}/{name}"] = np.asarray(value).copy()
                if np.ma.isMaskedArray(value):
                    data[f"{prefix}/{name}_mask"] = np.ma.getmaskarray(value)
    get_array = getattr(artist, "get_array", None)
    if get_array is not None and get_array() is not None:
        value = get_array()
        data[f"{prefix}/array"] = np.asarray(value).copy()
        data[f"{prefix}/array_mask"] = np.ma.getmaskarray(value)
    alpha = artist.get_alpha()
    if alpha is not None:
        data[f"{prefix}/alpha"] = np.asarray(alpha)
    if hasattr(artist, "get_xydata"):
        data[f"{prefix}/xy"] = np.asarray(artist.get_xydata()).copy()
        data[f"{prefix}/color"] = np.asarray(to_rgba(artist.get_color()))
        data[f"{prefix}/linestyle"] = np.asarray(str(artist.get_linestyle()))
        data[f"{prefix}/marker"] = np.asarray(str(artist.get_marker()))
        data[f"{prefix}/markersize"] = np.asarray(artist.get_markersize())
    if hasattr(artist, "get_text"):
        data[f"{prefix}/text"] = np.asarray(artist.get_text())
        data[f"{prefix}/position"] = np.asarray(artist.get_position())
        data[f"{prefix}/color"] = np.asarray(to_rgba(artist.get_color()))
        data[f"{prefix}/fontsize"] = np.asarray(artist.get_fontsize())
        data[f"{prefix}/rotation"] = np.asarray(artist.get_rotation())
        data[f"{prefix}/horizontalalignment"] = np.asarray(
            artist.get_horizontalalignment()
        )
        data[f"{prefix}/verticalalignment"] = np.asarray(artist.get_verticalalignment())
    if hasattr(artist, "get_extent"):
        data[f"{prefix}/extent"] = np.asarray(artist.get_extent())


def record_figure(figure):
    figure.canvas.draw()
    data = {"axes_count": np.asarray(len(figure.axes))}
    for index, axes in enumerate(figure.axes):
        prefix = f"axes_{index}"
        for name, value in (
            ("class", type(axes).__name__),
            ("title", axes.get_title()),
            ("xlabel", axes.get_xlabel()),
            ("ylabel", axes.get_ylabel()),
            ("xlim", axes.get_xlim()),
            ("ylim", axes.get_ylim()),
            ("aspect", str(axes.get_aspect())),
            ("position", axes.get_position().bounds),
        ):
            data[f"{prefix}/{name}"] = np.asarray(value)
        groups = {
            "collections": list(axes.collections),
            "patches": list(axes.patches),
            "lines": list(axes.lines),
            "images": list(axes.images),
            "texts": [
                *axes.texts,
                axes.title,
                axes.xaxis.label,
                axes.yaxis.label,
                *axes.get_xticklabels(),
                *axes.get_yticklabels(),
            ],
            "spines": list(axes.spines.values()),
            "background": [axes.patch],
        }
        for group, artists in groups.items():
            data[f"{prefix}/{group}/count"] = np.asarray(len(artists))
            for artist_index, artist in enumerate(artists):
                record_artist(data, f"{prefix}/{group}/{artist_index}", artist)
    return data


def render_case(env, operation):
    # Isolate plots from process-wide style changes made by earlier tests.
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        return _render_case(env, operation)


def _render_case(env, operation):
    figure, axes = plt.subplots(figsize=(4, 3), dpi=72)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if operation == "plot":
            env.plot(ax=axes)
        elif operation == "plot_field":
            field = np.linspace(-1, 1, env.n_bins)
            field[len(field) // 2] = np.nan
            env.plot_field(
                field, ax=axes, cmap="viridis", vmin=-1, vmax=1, colorbar=True
            )
        elif operation == "plot_1d":
            env.plot_1d(ax=axes)
        else:
            raise ValueError(operation)
        data = record_figure(figure)
    data["warnings"] = np.asarray(
        [f"{warning.category.__name__}: {warning.message}" for warning in caught],
        dtype=str,
    )
    return figure, data


@pytest.fixture(scope="module")
def plot_artists():
    """Numerical artists recorded before drawing moved from the layouts."""
    with np.load(Path(__file__).parent / "data/plot_artists.npz") as recorded:
        records = json.loads(recorded["records"].item())
    return {
        name: np.frombuffer(base64.b64decode(value["data"]), dtype=value["dtype"])
        .reshape(value["shape"])
        .copy()
        for name, value in records.items()
    }


def _assert_artists(recorded, prefix, actual):
    def retained(key):
        # Tick placement belongs to Matplotlib and varies across versions.
        # Keep the title and axis labels; compare the plotted data and styles.
        if "/texts/" in key:
            index = key.split("/texts/", 1)[1].split("/", 1)[0]
            return index != "count" and int(index) < 3
        return True

    expected_keys = {
        key.removeprefix(prefix)
        for key in recorded
        if key.startswith(prefix) and retained(key.removeprefix(prefix))
    }
    actual = {key: value for key, value in actual.items() if retained(key)}
    assert actual.keys() == expected_keys
    for key, value in actual.items():
        expected = recorded[f"{prefix}{key}"]
        if value.dtype.kind == "f":
            np.testing.assert_allclose(
                value, expected, rtol=1e-12, atol=1e-12, err_msg=f"{prefix}{key}"
            )
        else:
            np.testing.assert_array_equal(value, expected, err_msg=f"{prefix}{key}")


def _assert_mesh_artists(env, figure, operation):
    """Check exact cell/value alignment without assuming Qhull's bin order."""
    axes = figure.axes[0]
    shapes = env._binning.shapes
    collection = axes.collections[0]
    assert type(collection).__name__ == "PolyCollection"
    paths = collection.get_paths()
    assert len(paths) == env.n_bins
    for path, triangle in zip(paths, shapes.points[shapes.triangles], strict=True):
        np.testing.assert_array_equal(path.vertices, np.vstack([triangle, triangle[0]]))
    if operation == "plot_field":
        field = np.linspace(-1, 1, env.n_bins)
        field[len(field) // 2] = np.nan
        values = collection.get_array()
        np.testing.assert_array_equal(np.ma.getmaskarray(values), np.isnan(field))
        np.testing.assert_array_equal(values.compressed(), field[np.isfinite(field)])
        assert len(figure.axes) == 2  # colorbar
    else:
        np.testing.assert_array_equal(
            axes.collections[1].get_offsets(), env.bin_centers
        )
        outline = env._binning.outline.geometry
        rings = [outline.exterior, *outline.interiors]
        assert len(axes.lines) == len(rings)
        for line, ring in zip(axes.lines, rings, strict=True):
            np.testing.assert_array_equal(line.get_xydata(), np.asarray(ring.coords))
        # The connections are one collection, a segment per graph edge.
        segments = axes.collections[2].get_segments()
        assert len(segments) == env.connectivity.number_of_edges()
        for segment, (u, v) in zip(segments, env.connectivity.edges(), strict=True):
            np.testing.assert_array_equal(segment, env.bin_centers[[u, v]])
    assert axes.get_xlabel() == "X Position"
    assert axes.get_ylabel() == "Y Position"


@pytest.mark.slow
@pytest.mark.parametrize("operation", ["plot", "plot_field"])
def test_plot_artists_unchanged(built_environment, request, plot_artists, operation):
    """Every axes retains collection paths, colors, labels and draw order."""
    case = request.node.callspec.params["built_environment"]
    if not case.startswith("mesh"):
        np.testing.assert_array_equal(
            built_environment.bin_centers, plot_artists[f"{case}/centers"]
        )
    figure, actual = render_case(built_environment, operation)
    try:
        if case.startswith("mesh"):
            _assert_mesh_artists(built_environment, figure, operation)
        else:
            _assert_artists(plot_artists, f"{case}/{operation}/", actual)
    finally:
        plt.close(figure)


def _detached_native_layout(env, monkeypatch):
    """Inject a detached builder for tests deliberately corrupting layout data."""
    layout = native_layout(env)
    monkeypatch.setattr(env, "_layout", layout, raising=False)
    return layout


def test_plot_field_renderer_by_component(built_environment, request, monkeypatch):
    """Unknown descriptive tags must retain the same geometric renderer."""
    expected_classes = {
        "grid8": "QuadMesh",
        "grid4": "QuadMesh",
        "polygon": "QuadMesh",
        "mask": "QuadMesh",
        "pixels": "QuadMesh",
        "polar": "QuadMesh",
        "polar_open": "QuadMesh",
        "hex": "PolyCollection",
        "mesh": "PolyCollection",
        "mesh_boundary": "PolyCollection",
        "track": "FillBetweenPolyCollection",
    }
    case = request.node.callspec.params["built_environment"]
    layout = _detached_native_layout(built_environment, monkeypatch)
    layout.layout_type = "unknown_layout"
    figure, axes = plt.subplots(figsize=(4, 3), dpi=72)
    try:
        built_environment.plot_field(
            np.ones(built_environment.n_bins), ax=axes, colorbar=False
        )
        assert type(axes.collections[0]).__name__ == expected_classes[case]
        if case == "track":
            assert axes.lines
    finally:
        plt.close(figure)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
def test_plot_1d_uses_linearization(built_environment, plot_artists):
    """The one-dimensional track keeps its original labels and segment artists."""
    figure, actual = render_case(built_environment, "plot_1d")
    try:
        assert actual["warnings"].size == 0
        _assert_artists(plot_artists, "track/plot_1d/", actual)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_hexagon_plots_are_framed_by_the_cell_extent(built_environment):
    """plot() and plot_field() show the hexagons, corners included, and no more."""
    env = built_environment
    limits = []
    for draw in (env.plot, env.plot_field):
        figure, axes = plt.subplots()
        try:
            if draw == env.plot:
                draw(ax=axes)
            else:
                draw(np.ones(env.n_bins), ax=axes, colorbar=False)
            limits.append((axes.get_xlim(), axes.get_ylim()))
        finally:
            plt.close(figure)
    assert limits[0] == limits[1] == tuple(tuple(r) for r in env.dimension_ranges)


@pytest.mark.parametrize("built_environment", ["hex"], indirect=True)
def test_plot_field_draws_the_published_cells(built_environment, monkeypatch):
    """Hexagons come from the published cell geometry, not the layout engine."""
    from shapely.geometry import Polygon as ShapelyPolygon

    from neurospatial.geometry import cell_polygons

    env = built_environment
    layout = _detached_native_layout(env, monkeypatch)
    layout.hex_radius_ = 999.0
    figure, axes = plt.subplots(figsize=(4, 3), dpi=72)
    try:
        env.plot_field(np.ones(env.n_bins), ax=axes, colorbar=False)
        paths = axes.collections[0].get_paths()
        assert len(paths) == env.n_bins
        for path, cell in zip(paths, cell_polygons(env._binning), strict=True):
            assert ShapelyPolygon(path.vertices).equals(cell)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("built_environment", ["grid8", "hex", "mesh"], indirect=True)
def test_plot_field_does_not_read_private_layout_geometry(
    built_environment, request, monkeypatch
):
    """A drawing can use the snapshot after layout-only geometry is unavailable."""
    attrs = {
        "grid8": ("grid_edges", "grid_shape", "active_mask"),
        "hex": ("hex_radius_",),
        "mesh": ("_full_delaunay_tri", "_active_original_simplex_indices"),
    }
    case = request.node.callspec.params["built_environment"]
    layout = _detached_native_layout(built_environment, monkeypatch)
    for name in attrs[case]:
        setattr(layout, name, None)
    figure, axes = plt.subplots(figsize=(4, 3), dpi=72)
    try:
        built_environment.plot_field(
            np.ones(built_environment.n_bins), ax=axes, colorbar=False
        )
        assert axes.collections
    finally:
        plt.close(figure)


def test_three_dimensional_grid_keeps_plot_field_refusal():
    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 3
    )
    figure, axes = plt.subplots()
    try:
        with pytest.raises(
            NotImplementedError, match="Cannot plot 3D fields spatially"
        ):
            env.plot_field(np.ones(env.n_bins), ax=axes, colorbar=False)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("drawing", ["triangles", "outline"])
def test_plot_helpers_draw_every_multipolygon_component(drawing):
    """An outline with separated arenas draws each exterior and one legend label."""
    from neurospatial.geometry._components import Outline, TriangleShapes
    from neurospatial.layout._plotting import _draw_outline, plot_triangles

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [4.0, 0.0], [6.0, 0.0], [4.0, 2.0]]
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    polygons = [Polygon(vertices) for vertices in points[triangles]]
    outline = Outline(MultiPolygon(polygons))
    figure, axes = plt.subplots(figsize=(4, 3), dpi=72)
    try:
        if drawing == "triangles":
            plot_triangles(
                axes,
                title="Separated arenas",
                shapes=TriangleShapes(points, triangles),
                outline=outline,
                pairs=np.empty((0, 2), dtype=np.intp),
                dimension_ranges=[(0.0, 6.0), (0.0, 2.0)],
                bin_centers=points[triangles].mean(axis=1),
                show_connectivity=False,
                show_centroids=False,
            )
            assert len(axes.lines) == 2
            for line, polygon in zip(axes.lines, polygons, strict=True):
                np.testing.assert_array_equal(
                    line.get_xydata(), np.asarray(polygon.exterior.coords)
                )
            assert axes.get_legend_handles_labels()[1] == ["Boundary"]
        else:
            _draw_outline(axes, outline)
            assert len(axes.patches) == 2
            for patch, polygon in zip(axes.patches, polygons, strict=True):
                assert {tuple(p) for p in patch.get_path().vertices} == {
                    tuple(p) for p in polygon.exterior.coords
                }
    finally:
        plt.close(figure)


def test_drawn_hexagons_are_the_environment_cells():
    from shapely.geometry import Polygon as ShapelyPolygon

    from neurospatial.geometry._polygon import materialize_cells

    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 12), (0, 12)]
        )
    )
    figure, axes = plt.subplots()
    try:
        env.plot_field(np.arange(env.n_bins, dtype=float), ax=axes, colorbar=False)
        paths = axes.collections[0].get_paths()
        cells = materialize_cells(env.binning)
        assert len(paths) == env.n_bins
        for path, cell in zip(paths, cells, strict=True):
            assert ShapelyPolygon(path.vertices).equals(cell)
    finally:
        plt.close(figure)


def _drawn_vertices(axes) -> set[tuple[float, float]]:
    """Every vertex of the patches and lines on ``axes``, rounded."""
    paths = [patch.get_path().vertices for patch in axes.patches]
    paths += [line.get_xydata() for line in axes.lines]
    return {tuple(np.round(point, 9)) for path in paths for point in path}


@pytest.mark.parametrize("angle", [0.0, 0.4])
def test_plot_draws_the_arena_outline_with_its_holes(angle):
    """A polygon arena's outline is drawn with its holes, also once rotated.

    The polygon grid drew only the exterior ring, so an arena with an
    obstacle looked solid, and a rotated arena drew no outline at all.
    """
    from neurospatial.ops.transforms import Affine2D

    arena = Polygon(
        [(0, 0), (40, 0), (40, 30), (0, 30)], [[(15, 10), (25, 10), (25, 20), (15, 20)]]
    )
    env = Environment.from_polygon(arena, bin_size=2.0)
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        env = env.apply_transform(
            Affine2D(np.array([[c, -s, 5], [s, c, 1], [0, 0, 1]]))
        )
    outline = env._binning.outline.geometry
    figure, axes = plt.subplots()
    try:
        env.plot(ax=axes)
        drawn = _drawn_vertices(axes)
        for ring in (outline.exterior, *outline.interiors):
            assert {tuple(np.round(p, 9)) for p in ring.coords} <= drawn
    finally:
        plt.close(figure)
