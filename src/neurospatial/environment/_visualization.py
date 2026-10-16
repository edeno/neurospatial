"""Rendering adapters for owned environment components and published regions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import PatchCollection
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._components import HexagonShapes, TriangleShapes
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem
from neurospatial.geometry._grid_frame import require_grid_context
from neurospatial.geometry._polygon import PolygonCells
from neurospatial.layout._plotting import (
    hexagon_patches,
    plot_track_linear,
    polygon_patches,
)
from neurospatial.layout._plotting_binning import plot_binning
from neurospatial.layout.helpers.utils import map_active_data_to_grid
from neurospatial.regions import Regions
from neurospatial.regions.plot import plot_regions

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from neurospatial.layout._scale_bar import ScaleBarConfig


# Minimum extent threshold for scale bar (avoids issues with very small extents)
_MIN_EXTENT_FOR_SCALE_BAR = 1e-6


@dataclass(frozen=True)
class PlotComponents:
    """Owned environment components that the plotting functions read.

    Parameters
    ----------
    binning : Binning
        Published bins, cell shapes and grid structure.
    connectivity : networkx.Graph
        Navigation graph over the bins.
    dimension_ranges : sequence of (float, float) or None
        Plot extent per axis.
    grid_edges : tuple of ndarray or None
        Grid edges of the published grid index.
    active_mask : ndarray of bool or None
        Active cells of the published grid index.
    coordinates : CartesianSystem or EgocentricPolarSystem
        Axis labels and aspect of the bin coordinates.
    units : str or None
        Coordinate units for axis labels and scale bars.
    name : str
        Environment name for the layout title.
    layout_type : str
        Layout label for titles.
    regions : Regions
        Named regions drawn on request.
    is_linearized_track : bool
        Whether bins are positions along a linearized track.
    """

    binning: Binning
    connectivity: nx.Graph
    dimension_ranges: Sequence[tuple[float, float]] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    active_mask: NDArray[np.bool_] | None
    coordinates: CartesianSystem | EgocentricPolarSystem
    units: str | None
    name: str
    layout_type: str
    regions: Regions
    is_linearized_track: bool


def _add_scale_bar_if_requested(
    parts: PlotComponents,
    ax: matplotlib.axes.Axes,
    scale_bar: bool | ScaleBarConfig,
) -> None:
    """Add scale bar to axes if requested and valid.

    Helper function to reduce code duplication between plot() and plot_field().
    Handles validation of dimension_ranges and emits warnings for invalid cases.

    Parameters
    ----------
    parts : PlotComponents
        Plotted components; supplies dimension_ranges and units.
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    scale_bar : bool or ScaleBarConfig
        If truthy, adds a scale bar. If ScaleBarConfig, uses that configuration.
    """
    if not scale_bar:
        return

    from neurospatial.layout._scale_bar import ScaleBarConfig as _ScaleBarConfig
    from neurospatial.layout._scale_bar import add_scale_bar_to_axes

    config = scale_bar if isinstance(scale_bar, _ScaleBarConfig) else _ScaleBarConfig()

    # Validate extent is available and reasonable
    dimension_ranges = parts.dimension_ranges
    if not dimension_ranges:
        warnings.warn(
            "Cannot add scale bar: dimension_ranges not available",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    elif not np.isfinite(dimension_ranges[0]).all():
        warnings.warn(
            "Cannot add scale bar: dimension_ranges contains non-finite values",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    else:
        extent_x = dimension_ranges[0][1] - dimension_ranges[0][0]
        if extent_x < _MIN_EXTENT_FOR_SCALE_BAR:
            warnings.warn(
                f"Cannot add scale bar: extent too small ({extent_x})",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        else:
            add_scale_bar_to_axes(ax, extent_x, parts.units, config)


# Helper functions for layout-specific rendering


def _plot_grid_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str | Colormap,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field on grid layout using pcolormesh."""
    grid, frame = require_grid_context(
        parts.binning, "plot_field()", fix="supply rectangular cells"
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
    cmap_obj = plt.get_cmap(cmap).copy()
    if nan_color is not None:
        cmap_obj.set_bad(color=nan_color, alpha=1.0)

    # Plot
    x, y = grid.edges
    if frame is not None:
        xx, yy = np.meshgrid(x, y, indexing="xy")
        corners = frame.to_world(np.column_stack((xx.ravel(), yy.ravel())))
        x, y = (corners[:, axis].reshape(xx.shape) for axis in range(2))
    mesh = ax.pcolormesh(
        x,
        y,
        grid_data.T,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        shading="flat",
        **kwargs,
    )

    return mesh


def _plot_polygon_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str | Colormap,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Render exact polygon rings with the standard missing-value colors."""
    shapes = parts.binning.shapes
    assert isinstance(shapes, PolygonCells)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color or "none")
    collection = PatchCollection(
        polygon_patches(shapes), cmap=cmap_obj, rasterized=rasterized, **kwargs
    )
    collection.set_array(np.ma.masked_invalid(field))
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def _plot_hex_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str | Colormap,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field on hexagonal layout using PatchCollection."""
    shapes = parts.binning.shapes
    if not isinstance(shapes, HexagonShapes):
        raise RuntimeError("Hexagonal field plotting requires hexagon cell shapes.")

    # Note: Unlike grid layouts which use colormap's built-in NaN handling,
    # PatchCollection requires manual filtering of NaN values to avoid creating
    # empty patches that matplotlib doesn't handle well.
    # Skip NaN bins if nan_color is None
    drawn = (
        np.ones(len(field), dtype=bool) if nan_color is not None else ~np.isnan(field)
    )
    if not np.any(drawn):
        return None

    # Create collection
    collection = PatchCollection(
        hexagon_patches(shapes, shapes.centers[drawn]),
        cmap=cmap,
        rasterized=rasterized,
        **kwargs,
    )
    collection.set_array(np.asarray(field)[drawn])
    collection.set_clim(vmin, vmax)
    collection.set_edgecolor("none")

    ax.add_collection(collection)
    return collection


def _plot_trimesh_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str | Colormap,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field on triangular mesh layout using tripcolor."""
    shapes = parts.binning.shapes
    if not isinstance(shapes, TriangleShapes):
        raise RuntimeError("Mesh field plotting requires triangular cell shapes.")
    triangles = shapes.triangles
    mesh_points = shapes.points

    # Setup colormap with NaN handling
    cmap_obj = plt.get_cmap(cmap).copy()
    if nan_color is not None:
        cmap_obj.set_bad(color=nan_color, alpha=1.0)

    # Plot using tripcolor
    mesh = ax.tripcolor(
        mesh_points[:, 0],
        mesh_points[:, 1],
        triangles,
        facecolors=field,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        **kwargs,
    )

    return mesh


def _plot_1d_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    ylabel: str,
    **kwargs: Any,
) -> Any:
    """Plot a track field in physical display coordinates without bridging gaps."""
    edges: tuple[NDArray[np.float64], ...] | None
    active_mask: NDArray[np.bool_] | None
    linearization = parts.binning.linearization
    if isinstance(linearization, TrackLinearization):
        edges, active_mask = linearization.grid_edges, linearization.active_mask
    else:
        edges, active_mask = parts.grid_edges, parts.active_mask
    if edges is None or active_mask is None:
        raise RuntimeError("Track field plotting requires grid_edges and active_mask.")
    x_positions = (edges[0][:-1] + edges[0][1:]) / 2
    values = np.full(x_positions.shape, np.nan, dtype=np.float64)
    values[active_mask.ravel()] = field

    # Keep NaNs in the line and fill: dropping them invents interpolation over
    # unobserved bins and the layout's artificial inter-segment display gaps.
    kwargs.setdefault("linewidth", 2)
    (line,) = ax.plot(x_positions, values, **kwargs)
    ax.fill_between(x_positions, 0, values, alpha=0.3, color=line.get_color())

    unit_label = f" ({parts.units})" if parts.units else ""
    ax.set_xlabel(f"Linearized track coordinate{unit_label}", fontsize=12)
    ax.set_ylabel(ylabel if ylabel else "Field Value", fontsize=12)
    ax.set_title("1D Field Plot", fontsize=14)

    return line


def _plot_scatter_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str | Colormap,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field using scatter plot (fallback for unknown layouts)."""
    bin_centers = parts.binning.bins.centers
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
            np.asarray(parts.dimension_ranges, dtype=float)
            if parts.dimension_ranges is not None
            else np.column_stack((bin_centers.min(axis=0), bin_centers.max(axis=0)))
        )
        spans = ranges[:2, 1] - ranges[:2, 0]
        axes_points = np.asarray(ax.get_window_extent().size) * 72.0 / ax.figure.dpi
        points_per_unit = np.divide(axes_points, spans, out=np.ones(2), where=spans > 0)
        if parts.coordinates.equal_aspect and np.any(spans > 0):
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
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        **kwargs,
    )

    return scatter


def _plot(
    parts: PlotComponents,
    ax: matplotlib.axes.Axes | None = None,
    show_regions: bool = False,
    layout_plot_kwargs: dict[str, Any] | None = None,
    regions_plot_kwargs: dict[str, Any] | None = None,
    scale_bar: bool | ScaleBarConfig = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Draw the bins and graph, optionally with regions and a scale bar."""
    l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
    l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

    ax = plot_binning(
        parts.binning,
        parts.connectivity,
        parts.dimension_ranges,
        parts.layout_type,
        ax=ax,
        **l_kwargs,
    )
    # layout.plot() always returns an Axes object, never None
    assert ax is not None

    if show_regions:
        r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
        plot_regions(parts.regions, ax=ax, **r_kwargs)

    plot_title = parts.name
    if parts.layout_type:
        plot_title += f" ({parts.layout_type})"

    # Only set title if layout.plot didn't set one or user didn't pass one via kwargs to layout.plot
    if ax.get_title() == "":
        ax.set_title(plot_title)

    # Add scale bar if requested
    _add_scale_bar_if_requested(parts, ax, scale_bar)

    return ax


def _plot_1d(
    parts: PlotComponents,
    ax: matplotlib.axes.Axes | None = None,
    layout_plot_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes | None:
    """Draw a linearized track in its 1D coordinate, or warn for other layouts."""
    l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
    l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
    if parts.is_linearized_track:
        linearization = parts.binning.linearization
        if isinstance(linearization, TrackLinearization):
            ax = plot_track_linear(
                ax=ax,
                title=parts.layout_type,
                linearization=linearization,
                grid_edges=linearization.grid_edges,
                **l_kwargs,
            )
        else:
            warnings.warn(
                f"Layout '{parts.layout_type}' is 1D but does not "
                "have a 'plot_linear_layout' method. Skipping 1D plot.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
    else:
        warnings.warn(
            "Environment is not 1D. Skipping 1D plot. Use regular plot() method.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    return ax


def _plot_field(
    parts: PlotComponents,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes | None = None,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    nan_color: str | None = "lightgray",
    rasterized: bool = True,
    scale_bar: bool | ScaleBarConfig = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Draw a bin-valued field with the renderer that matches the cell shapes."""
    binning = parts.binning
    n_bins, n_dims = binning.bins.n_bins, binning.bins.n_dims
    is_linearized_track = parts.is_linearized_track
    if field.ndim != 1 or field.shape[0] != n_bins:
        raise ValueError(
            f"field must be 1D array with length n_bins={n_bins}, "
            f"got shape {field.shape}"
        )

    # Check dimensionality for spatial plots
    if not is_linearized_track and n_dims > 2:
        raise NotImplementedError(
            f"Cannot plot {n_dims}D fields spatially. "
            "Only 1D and 2D environments are supported. "
            "Consider using marginal plots, slicing, or 3D scatter plots instead."
        )

    # Polar envs render via the same dispatch but with polar-aware
    # axis labels and aspect ratio: bin_centers[:, 0] is
    # distance (env units), bin_centers[:, 1] is angle in radians.
    # The label switch happens at the bottom of this method, after
    # dispatch.

    # Create axes if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    # Auto vmin/vmax
    if vmin is None:
        vmin = np.nanmin(field)
        if np.isnan(vmin) or np.isinf(vmin):
            vmin = 0.0

    if vmax is None:
        vmax = np.nanmax(field)
        if np.isnan(vmax) or np.isinf(vmax):
            vmax = 1.0

    # Ensure vmin < vmax
    if vmin >= vmax:
        vmax = vmin + 1.0

    # Choose rendering from the structure published with the bins.
    mappable = None
    if n_dims == 1 and not is_linearized_track:
        ax.plot(binning.bins.centers[:, 0], field, **kwargs)
        unit_label = f" ({parts.units})" if parts.units else ""
        ax.set_xlabel(f"Position{unit_label}")
        ax.set_ylabel(colorbar_label or "Field Value")
    elif binning.rectilinear is not None or binning.grid_frame is not None:
        mappable = _plot_grid_field(
            parts, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif isinstance(binning.shapes, HexagonShapes):
        mappable = _plot_hex_field(
            parts, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif isinstance(binning.shapes, TriangleShapes):
        mappable = _plot_trimesh_field(
            parts, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif isinstance(binning.shapes, PolygonCells):
        mappable = _plot_polygon_field(
            parts, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif binning.linearization is not None or is_linearized_track:
        # Legacy saved tracks retain their previous missing-grid refusal
        # until loading restores their full linearization data.
        mappable = _plot_1d_field(parts, field, ax, colorbar_label, **kwargs)
    else:
        mappable = _plot_scatter_field(
            parts, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )

    # Add colorbar (not for 1D)
    # Note: mappable can be None if all field values are NaN and nan_color=None
    if colorbar and mappable is not None and not is_linearized_track:
        cbar = plt.colorbar(mappable, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Format axes. Polar envs (from from_polar_egocentric) carry
    # bin_centers as (distance, angle in radians), not (x, y), so
    # the default "X Position" / "Y Position" labels are silently
    # wrong; relabel and skip the equal-aspect lock (a polar map's
    # angular extent is fixed by bin_centers[:, 1] but its distance
    # extent is independent and physical).
    if not is_linearized_track and n_dims != 1:
        system = parts.coordinates
        xlabel, ylabel = system.axis_labels(parts.units)
        if system.equal_aspect:
            ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        dimension_ranges = parts.dimension_ranges
        if dimension_ranges and len(dimension_ranges) >= 2:
            ax.set_xlim(dimension_ranges[0])
            ax.set_ylim(dimension_ranges[1])

    # Add scale bar if requested
    _add_scale_bar_if_requested(parts, ax, scale_bar)

    return ax
