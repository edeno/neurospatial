"""Draw a binning's cells, or one value per bin, with one renderer per cell shape.

:func:`_cell_shape` names the cells a binning describes, and ``_RENDERERS``
pairs each shape with the function that draws the cells and the function that
colors them by a field. ``Environment.plot`` and ``Environment.plot_field``
both dispatch through this one table, so the outline and the field of one
environment are always drawn from the same cell description.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
    HexagonShapes,
    PolygonCells,
    TriangleShapes,
    require_grid_context,
)
from neurospatial.layout._plotting import (
    _grid_mesh,
    _hexagon_patches,
    _polygon_patches,
    plot_grid,
    plot_hexagons,
    plot_polygon_cells,
    plot_polygon_grid,
    plot_track,
    plot_triangles,
)
from neurospatial.layout.helpers.utils import map_active_data_to_grid

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

CellShape = Literal["track", "hexagons", "triangles", "polygons", "grid", "points"]


def _cell_shape(binning: Binning) -> CellShape:
    """Name the cells that ``binning`` describes, for drawing.

    A published binning carries at most one cell description: a track, cell
    shapes, or an axis-aligned or framed grid (each needs its own kind of
    locator, which validation checks). The order below therefore only fixes
    one answer for every drawing; bins with no cell description are points.

    Parameters
    ----------
    binning : Binning
        Bins to draw.

    Returns
    -------
    {"track", "hexagons", "triangles", "polygons", "grid", "points"}
    """
    if binning.linearization is not None:
        return "track"
    if isinstance(binning.shapes, HexagonShapes):
        return "hexagons"
    if isinstance(binning.shapes, TriangleShapes):
        return "triangles"
    if isinstance(binning.shapes, PolygonCells):
        return "polygons"
    if binning.rectilinear is not None or binning.grid_frame is not None:
        return "grid"
    return "points"


@dataclass(frozen=True)
class _FieldStyle:
    """How a field's values are colored and its axes labeled.

    Parameters
    ----------
    cmap : str or Colormap
        Colormap for the bin values.
    vmin, vmax : float
        Color limits.
    nan_color : str or None
        Color of NaN bins; None leaves them undrawn.
    rasterized : bool
        Rasterize the drawn cells.
    units : str or None
        Coordinate units for axis labels.
    coordinates : CartesianSystem or EgocentricPolarSystem
        Axis names and aspect of the bin coordinates.
    colorbar_label : str
        Name of the field's values.
    """

    cmap: str | Colormap
    vmin: float
    vmax: float
    nan_color: str | None
    rasterized: bool
    units: str | None
    coordinates: CartesianSystem | EgocentricPolarSystem
    colorbar_label: str


# Cells: draw the bins and their connectivity graph.


def _track_cells(
    binning: Binning, connectivity: nx.Graph, title: str, ax: Axes | None, **kwargs: Any
) -> Axes:
    linearization = binning.linearization
    assert linearization is not None
    return plot_track(
        ax,
        title=title,
        linearization=linearization,
        bin_centers=binning.bins.centers,
        connectivity=connectivity,
        grid_edges=linearization.grid_edges,
        **kwargs,
    )


def _hexagon_cells(
    binning: Binning, connectivity: nx.Graph, title: str, ax: Axes | None, **kwargs: Any
) -> Axes:
    assert isinstance(binning.shapes, HexagonShapes)
    return plot_hexagons(
        ax,
        title=title,
        shapes=binning.shapes,
        connectivity=connectivity,
        dimension_ranges=binning.dimension_ranges,
        **kwargs,
    )


def _triangle_cells(
    binning: Binning, connectivity: nx.Graph, title: str, ax: Axes | None, **kwargs: Any
) -> Axes:
    assert isinstance(binning.shapes, TriangleShapes)
    ranges = binning.dimension_ranges
    if ranges is None:
        ranges = tuple(
            (float(axis.min()), float(axis.max())) for axis in binning.bins.centers.T
        )
    return plot_triangles(
        ax,
        title=title,
        shapes=binning.shapes,
        outline=binning.outline,
        connectivity=connectivity,
        dimension_ranges=ranges,
        bin_centers=binning.bins.centers,
        **kwargs,
    )


def _polygon_cells(
    binning: Binning, connectivity: nx.Graph, title: str, ax: Axes | None, **kwargs: Any
) -> Axes:
    assert isinstance(binning.shapes, PolygonCells)
    return plot_polygon_cells(
        ax,
        shapes=binning.shapes,
        title=title,
        connectivity=connectivity,
        dimension_ranges=binning.dimension_ranges,
        **kwargs,
    )


def _grid_cells(
    binning: Binning, connectivity: nx.Graph, title: str, ax: Axes | None, **kwargs: Any
) -> Axes:
    common: dict[str, Any] = {
        "title": title,
        "connectivity": connectivity,
        "dimension_ranges": binning.dimension_ranges,
        "bin_centers": binning.bins.centers,
    }
    if binning.grid_frame is not None:
        return plot_grid(
            ax,
            rectilinear=binning.grid_frame.source,
            frame=binning.grid_frame,
            **common,
            **kwargs,
        )
    assert binning.rectilinear is not None
    if binning.outline is not None:
        return plot_polygon_grid(
            ax,
            rectilinear=binning.rectilinear,
            outline=binning.outline,
            **common,
            **kwargs,
        )
    return plot_grid(ax, rectilinear=binning.rectilinear, **common, **kwargs)


def _point_cells(
    binning: Binning, connectivity: nx.Graph, title: str, ax: Axes | None, **kwargs: Any
) -> Axes:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    centers = binning.bins.centers
    if centers.shape[1] == 2:
        ax.scatter(centers[:, 0], centers[:, 1], **kwargs)
    return ax


# Fields: color each bin by its value and return the color mappable.


def _grid_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Plot field on grid layout using pcolormesh."""
    import matplotlib.pyplot as plt

    grid, frame = require_grid_context(
        binning, "plot_field()", fix="supply rectangular cells"
    )

    # Validate 2D grid
    if len(grid.shape) != 2:
        raise NotImplementedError(
            f"pcolormesh requires 2D grids, got grid_shape={grid.shape}"
        )

    # Convert to grid
    grid_data = map_active_data_to_grid(
        grid.shape, grid.active_mask, field, fill_value=np.nan
    )

    # Setup colormap with NaN handling
    cmap_obj = plt.get_cmap(style.cmap).copy()
    if style.nan_color is not None:
        cmap_obj.set_bad(color=style.nan_color, alpha=1.0)

    x, y = _grid_mesh(grid, frame)
    mesh = ax.pcolormesh(
        x,
        y,
        grid_data.T,
        cmap=cmap_obj,
        vmin=style.vmin,
        vmax=style.vmax,
        rasterized=style.rasterized,
        shading="flat",
        **kwargs,
    )

    return mesh


def _polygon_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Render exact polygon rings with the standard missing-value colors."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    shapes = binning.shapes
    assert isinstance(shapes, PolygonCells)
    cmap_obj = plt.get_cmap(style.cmap).copy()
    cmap_obj.set_bad(style.nan_color or "none")
    collection = PatchCollection(
        _polygon_patches(shapes.polygons),
        cmap=cmap_obj,
        rasterized=style.rasterized,
        **kwargs,
    )
    collection.set_array(np.ma.masked_invalid(field))
    collection.set_clim(style.vmin, style.vmax)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def _hexagon_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Plot field on hexagonal layout using PatchCollection."""
    from matplotlib.collections import PatchCollection

    shapes = binning.shapes
    assert isinstance(shapes, HexagonShapes)

    # Note: Unlike grid layouts which use colormap's built-in NaN handling,
    # PatchCollection requires manual filtering of NaN values to avoid creating
    # empty patches that matplotlib doesn't handle well.
    # Skip NaN bins if nan_color is None
    drawn = (
        np.ones(len(field), dtype=bool)
        if style.nan_color is not None
        else ~np.isnan(field)
    )
    if not np.any(drawn):
        return None

    # Create collection
    collection = PatchCollection(
        _hexagon_patches(shapes, shapes.centers[drawn]),
        cmap=style.cmap,
        rasterized=style.rasterized,
        **kwargs,
    )
    collection.set_array(np.asarray(field)[drawn])
    collection.set_clim(style.vmin, style.vmax)
    collection.set_edgecolor("none")

    ax.add_collection(collection)
    return collection


def _triangle_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Plot field on triangular mesh layout using tripcolor."""
    import matplotlib.pyplot as plt

    shapes = binning.shapes
    assert isinstance(shapes, TriangleShapes)
    triangles = shapes.triangles
    mesh_points = shapes.points

    # Setup colormap with NaN handling
    cmap_obj = plt.get_cmap(style.cmap).copy()
    if style.nan_color is not None:
        cmap_obj.set_bad(color=style.nan_color, alpha=1.0)

    # Plot using tripcolor
    mesh = ax.tripcolor(
        mesh_points[:, 0],
        mesh_points[:, 1],
        triangles,
        facecolors=field,
        cmap=cmap_obj,
        vmin=style.vmin,
        vmax=style.vmax,
        rasterized=style.rasterized,
        **kwargs,
    )

    return mesh


def _track_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Plot a track field in physical display coordinates without bridging gaps."""
    linearization = binning.linearization
    assert linearization is not None
    edges, active_mask = linearization.grid_edges, linearization.active_mask
    x_positions = (edges[0][:-1] + edges[0][1:]) / 2
    values = np.full(x_positions.shape, np.nan, dtype=np.float64)
    values[active_mask.ravel()] = field

    # Keep NaNs in the line and fill: dropping them invents interpolation over
    # unobserved bins and the layout's artificial inter-segment display gaps.
    kwargs.setdefault("linewidth", 2)
    (line,) = ax.plot(x_positions, values, **kwargs)
    ax.fill_between(x_positions, 0, values, alpha=0.3, color=line.get_color())

    unit_label = f" ({style.units})" if style.units else ""
    ax.set_xlabel(f"Linearized track coordinate{unit_label}", fontsize=12)
    ax.set_ylabel(style.colorbar_label or "Field Value", fontsize=12)
    ax.set_title("1D Field Plot", fontsize=14)

    return line


def _point_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Plot field using scatter plot (fallback for bins without cell shapes)."""
    import matplotlib.pyplot as plt

    cmap = style.cmap
    nan_color = style.nan_color
    bin_centers = binning.bins.centers
    n_bins = len(bin_centers)
    # Filter NaN values if nan_color is None
    if nan_color is None:
        valid_mask = ~np.isnan(field)
        plot_centers = bin_centers[valid_mask]
        plot_values = field[valid_mask]
    else:
        plot_centers = bin_centers
        plot_values = field.copy()

    if len(plot_centers) == 0:
        return None

    # Scatter sizes are areas in display points, never physical coordinate
    # units. Estimate visible spacing after accounting for the eventual axes
    # limits and aspect ratio. A cap keeps sparse layouts readable as points.
    if "s" not in kwargs and n_bins > 1:
        ranges = (
            np.asarray(binning.dimension_ranges, dtype=float)
            if binning.dimension_ranges is not None
            else np.column_stack((bin_centers.min(axis=0), bin_centers.max(axis=0)))
        )
        spans = ranges[:2, 1] - ranges[:2, 0]
        axes_points = np.asarray(ax.get_window_extent().size) * 72.0 / ax.figure.dpi
        points_per_unit = np.divide(axes_points, spans, out=np.ones(2), where=spans > 0)
        if style.coordinates.equal_aspect and np.any(spans > 0):
            points_per_unit[:] = np.min(points_per_unit[spans > 0])
        display_centers = (bin_centers[:, :2] - ranges[:2, 0]) * points_per_unit
        tree = cKDTree(display_centers)
        # Find nearest neighbor distance for sample
        sample_size = min(100, n_bins)
        distances, _ = tree.query(display_centers[:sample_size], k=2)
        typical_spacing = np.median(distances[:, 1])
        kwargs["s"] = (
            min(100.0, (0.8 * typical_spacing) ** 2) if typical_spacing > 0 else 100.0
        )
    elif "s" not in kwargs:
        kwargs["s"] = 100.0

    # Keep caller-supplied per-bin sizes aligned when skipping NaN bins.
    if nan_color is None:
        sizes = np.asarray(kwargs["s"])
        if sizes.ndim > 0 and sizes.size == n_bins:
            kwargs["s"] = sizes.ravel()[valid_mask]
    else:
        # Matplotlib otherwise drops nonfinite colors instead of showing the
        # documented missing-support color. Copy so global colormaps stay intact.
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(nan_color)
        kwargs.setdefault("plotnonfinite", True)
    kwargs.setdefault("edgecolors", "none")

    # Plot
    scatter = ax.scatter(
        plot_centers[:, 0],
        plot_centers[:, 1],
        c=plot_values,
        cmap=cmap,
        vmin=style.vmin,
        vmax=style.vmax,
        rasterized=style.rasterized,
        **kwargs,
    )

    return scatter


@dataclass(frozen=True)
class _Renderer:
    """The two drawings of one cell shape: its cells, and a field over them."""

    cells: Callable[..., Axes]
    field: Callable[..., Any]


_RENDERERS: dict[CellShape, _Renderer] = {
    "track": _Renderer(_track_cells, _track_field),
    "hexagons": _Renderer(_hexagon_cells, _hexagon_field),
    "triangles": _Renderer(_triangle_cells, _triangle_field),
    "polygons": _Renderer(_polygon_cells, _polygon_field),
    "grid": _Renderer(_grid_cells, _grid_field),
    "points": _Renderer(_point_cells, _point_field),
}


def _plot_binning(
    binning: Binning,
    connectivity: nx.Graph,
    title: str,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw the bins' cells and connectivity graph.

    Parameters
    ----------
    binning : Binning
        Bins to draw; ``binning.dimension_ranges`` frames the axes.
    connectivity : networkx.Graph
        Bin graph whose nodes carry ``pos``.
    title : str
        Layout label for the title.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; a new figure when None.
    **kwargs
        Passed to the renderer of the binning's cell shape.

    Returns
    -------
    matplotlib.axes.Axes
    """
    renderer = _RENDERERS[_cell_shape(binning)]
    return renderer.cells(binning, connectivity, title, ax, **kwargs)


def _plot_binning_field(
    binning: Binning,
    field: NDArray[np.float64],
    ax: Axes,
    style: _FieldStyle,
    **kwargs: Any,
) -> Any:
    """Color each bin's cell by its value in ``field``.

    Parameters
    ----------
    binning : Binning
        Bins to draw.
    field : ndarray of float64, shape (n_bins,)
        One value per bin; NaN marks missing values.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    style : _FieldStyle
        Colors and labels.
    **kwargs
        Passed to the Matplotlib artist of the binning's cell shape.

    Returns
    -------
    matplotlib artist or None
        The color mappable, or None when nothing was drawn.
    """
    renderer = _RENDERERS[_cell_shape(binning)]
    return renderer.field(binning, field, ax, style, **kwargs)
