"""Rendering adapters for owned environment components and published regions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
)
from neurospatial.layout import (
    _cell_shape,
    _FieldStyle,
    _plot_binning,
    _plot_binning_field,
    _plot_track_linear,
)
from neurospatial.regions import Region, Regions
from neurospatial.regions.plot import plot_regions

if TYPE_CHECKING:
    import matplotlib.axes
    from matplotlib.colors import Colormap

    from neurospatial.geometry import Arena
    from neurospatial.landmarks import Landmark, Landmarks
    from neurospatial.layout import ScaleBarConfig


# Minimum extent threshold for scale bar (avoids issues with very small extents)
_MIN_EXTENT_FOR_SCALE_BAR = 1e-6


@dataclass(frozen=True)
class PlotLabels:
    """The names a plot shows, which the bins themselves do not carry.

    Parameters
    ----------
    name : str
        Environment name for the layout title.
    layout_type : str
        Layout label for titles.
    units : str or None
        Coordinate units for axis labels and scale bars.
    coordinates : CartesianSystem or EgocentricPolarSystem
        Axis names and aspect of the bin coordinates.
    """

    name: str
    layout_type: str
    units: str | None
    coordinates: CartesianSystem | EgocentricPolarSystem


def _add_scale_bar_if_requested(
    dimension_ranges: Sequence[tuple[float, float]] | None,
    units: str | None,
    ax: matplotlib.axes.Axes,
    scale_bar: bool | ScaleBarConfig,
) -> None:
    """Add scale bar to axes if requested and valid.

    Helper function to reduce code duplication between plot() and plot_field().
    Handles validation of dimension_ranges and emits warnings for invalid cases.

    Parameters
    ----------
    dimension_ranges : sequence of (float, float) or None
        Plot extent per axis; the first axis sets the scale bar length.
    units : str or None
        Coordinate units shown on the scale bar.
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    scale_bar : bool or ScaleBarConfig
        If truthy, adds a scale bar. If ScaleBarConfig, uses that configuration.
    """
    if not scale_bar:
        return

    from neurospatial.layout import ScaleBarConfig as _ScaleBarConfig
    from neurospatial.layout import _add_scale_bar_to_axes

    config = scale_bar if isinstance(scale_bar, _ScaleBarConfig) else _ScaleBarConfig()

    # Validate extent is available and reasonable
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
            _add_scale_bar_to_axes(ax, extent_x, units, config)


def _plot(
    binning: Binning,
    connectivity: nx.Graph,
    regions: Regions,
    labels: PlotLabels,
    ax: matplotlib.axes.Axes | None = None,
    show_regions: bool = True,
    arena: Arena | None = None,
    landmarks: Landmarks | None = None,
    layout_plot_kwargs: dict[str, Any] | None = None,
    regions_plot_kwargs: dict[str, Any] | None = None,
    scale_bar: bool | ScaleBarConfig = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Draw the bins and graph, with walls, landmarks, regions and a scale bar.

    ``arena`` and ``landmarks`` are drawn when given; None skips them.
    """
    l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
    l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

    ax = _plot_binning(binning, connectivity, labels.layout_type, ax=ax, **l_kwargs)
    # layout.plot() always returns an Axes object, never None
    assert ax is not None
    if binning.bins.n_dims == 2:
        # Name the axes by what the coordinates mean, as plot_field does:
        # (distance, angle) on a polar environment is not (x, y).
        xlabel, ylabel = labels.coordinates.axis_labels(labels.units)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not labels.coordinates.equal_aspect:
            ax.set_aspect("auto")

    if arena is not None and arena.walls:
        _draw_walls(ax, arena)
    if show_regions and len(regions):
        r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
        plot_regions(regions, ax=ax, **r_kwargs)
        # The renderers fix the limits to the bins; a region drawn past the
        # bins still belongs in the picture.
        _widen_to_show(ax, [_region_geometry(region) for region in regions.values()])
    if landmarks:
        _draw_landmarks(ax, landmarks)

    plot_title = labels.name
    if labels.layout_type:
        plot_title += f" ({labels.layout_type})"

    # Only set title if layout.plot didn't set one or user didn't pass one via kwargs to layout.plot
    if ax.get_title() == "":
        ax.set_title(plot_title)

    # Add scale bar if requested
    _add_scale_bar_if_requested(binning.dimension_ranges, labels.units, ax, scale_bar)

    return ax


def _draw_walls(ax: matplotlib.axes.Axes, arena: Arena) -> None:
    """Draw each declared wall as a thick line, its name at its middle."""
    from matplotlib.collections import LineCollection

    ax.add_collection(
        LineCollection(
            [np.asarray(wall.coords) for wall in arena.walls],
            colors="firebrick",
            linewidths=3,
            zorder=4,
        )
    )
    labeled: set[str] = set()
    for name, wall in zip(arena.wall_names, arena.walls, strict=True):
        if name in labeled:
            continue
        labeled.add(name)
        x, y = wall.interpolate(0.5, normalized=True).coords[0]
        ax.annotate(
            name,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            color="firebrick",
            fontsize=9,
            zorder=6,
        )


def _draw_landmarks(ax: matplotlib.axes.Axes, landmarks: Landmarks) -> None:
    """Draw each landmark by its kind, labeled, and widen the axes to show it."""
    from neurospatial.layout._plotting import _polygon_patches

    color = "darkorange"
    for landmark in landmarks.values():
        data = landmark.data
        if isinstance(data, np.ndarray):
            ax.plot(*data[:2], marker="*", markersize=14, color=color, zorder=5)
        elif data.geom_type == "LineString":
            ax.plot(*np.asarray(data.coords).T, color=color, linewidth=4, zorder=5)
        else:
            # A compound path keeps interior rings open: a ring-shaped object
            # is drawn with its hole, as the spatial calculations see it.
            (patch,) = _polygon_patches([data])
            patch.set(facecolor=color, edgecolor=color, alpha=0.4, zorder=5)
            ax.add_patch(patch)
        ax.annotate(
            landmark.name,
            tuple(landmark.position[:2]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=9,
            zorder=6,
        )
    # The renderers fix the limits to the bins; a cue outside the arena
    # still belongs in the picture.
    _widen_to_show(ax, [_landmark_geometry(item) for item in landmarks.values()])


def _widen_to_show(
    ax: matplotlib.axes.Axes, geometries: list[shapely.Geometry]
) -> None:
    """Grow the axis limits, never shrink them, so every geometry is visible."""
    bounds = np.array([shapely.bounds(geometry) for geometry in geometries])
    (x_low, x_high), (y_low, y_high) = ax.get_xlim(), ax.get_ylim()
    margin = 0.05 * max(x_high - x_low, y_high - y_low)
    ax.set_xlim(
        min(x_low, bounds[:, 0].min() - margin),
        max(x_high, bounds[:, 2].max() + margin),
    )
    ax.set_ylim(
        min(y_low, bounds[:, 1].min() - margin),
        max(y_high, bounds[:, 3].max() + margin),
    )


def _region_geometry(region: Region) -> shapely.Geometry:
    data = region.data
    return shapely.points(data[:2]) if isinstance(data, np.ndarray) else data


def _landmark_geometry(landmark: Landmark) -> shapely.Geometry:
    data = landmark.data
    return shapely.points(data[:2]) if isinstance(data, np.ndarray) else data


def _plot_1d(
    binning: Binning,
    layout_type: str,
    ax: matplotlib.axes.Axes | None = None,
    layout_plot_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes | None:
    """Draw a linearized track in its 1D coordinate, or warn for other layouts."""
    l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
    l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
    linearization = binning.linearization
    if linearization is not None:
        ax = _plot_track_linear(
            ax=ax,
            title=layout_type,
            linearization=linearization,
            grid_edges=linearization.grid_edges,
            **l_kwargs,
        )
    else:
        warnings.warn(
            "Environment is not 1D. Skipping 1D plot. Use regular plot() method.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    return ax


def check_field(field: NDArray[np.float64], n_bins: int) -> None:
    """Refuse a field that is not one value per bin."""
    if field.ndim != 1 or field.shape[0] != n_bins:
        raise ValueError(
            f"field must be 1D array with length n_bins={n_bins}, "
            f"got shape {field.shape}"
        )


def field_color_limits(
    field: NDArray[np.float64], vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    """Color limits of a field: the given ones, else its finite range.

    An all-missing or infinite end falls back to 0 or 1, and equal limits
    widen by one, as every ``plot_field`` backend draws them.
    """
    if vmin is None:
        vmin = np.nanmin(field)
        if np.isnan(vmin) or np.isinf(vmin):
            vmin = 0.0
    if vmax is None:
        vmax = np.nanmax(field)
        if np.isnan(vmax) or np.isinf(vmax):
            vmax = 1.0
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_field(
    binning: Binning,
    labels: PlotLabels,
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
    import matplotlib.pyplot as plt

    n_dims = binning.bins.n_dims
    is_linearized_track = _cell_shape(binning) == "track"

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

    vmin, vmax = field_color_limits(field, vmin, vmax)

    # Choose rendering from the structure published with the bins.
    mappable = None
    if n_dims == 1 and not is_linearized_track:
        ax.plot(binning.bins.centers[:, 0], field, **kwargs)
        unit_label = f" ({labels.units})" if labels.units else ""
        ax.set_xlabel(f"Position{unit_label}")
        ax.set_ylabel(colorbar_label or "Field Value")
    else:
        style = _FieldStyle(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            nan_color=nan_color,
            rasterized=rasterized,
            units=labels.units,
            coordinates=labels.coordinates,
            colorbar_label=colorbar_label,
        )
        mappable = _plot_binning_field(binning, field, ax, style, **kwargs)

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
        system = labels.coordinates
        xlabel, ylabel = system.axis_labels(labels.units)
        if system.equal_aspect:
            ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        dimension_ranges = binning.dimension_ranges
        if dimension_ranges and len(dimension_ranges) >= 2:
            ax.set_xlim(dimension_ranges[0])
            ax.set_ylim(dimension_ranges[1])

    # Add scale bar if requested
    _add_scale_bar_if_requested(binning.dimension_ranges, labels.units, ax, scale_bar)

    return ax
