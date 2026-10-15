"""Visualization methods for Environment.

This module contains the EnvironmentVisualization mixin class that provides
plotting capabilities for Environment instances.

Classes
-------
EnvironmentVisualization
    Mixin class providing plot() and plot_1d() methods.

Notes
-----
This module uses TYPE_CHECKING guards to avoid circular imports with the
Environment class. Type hints use string annotations ("Environment") for
forward references.

"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._protocols import VisualizationConsumer
from neurospatial.layout._binning import TrackLinearization
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._polygon import PolygonCells

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from neurospatial.animation.config import ScaleBarConfig
    from neurospatial.animation.overlays import OverlayProtocol
else:
    # Runtime import handled lazily in methods
    ScaleBarConfig = "ScaleBarConfig"  # For type annotations only


# Minimum extent threshold for scale bar (avoids issues with very small extents)
_MIN_EXTENT_FOR_SCALE_BAR = 1e-6


def _add_scale_bar_if_requested(
    env: VisualizationConsumer,
    ax: matplotlib.axes.Axes,
    scale_bar: bool | ScaleBarConfig,
) -> None:
    """Add scale bar to axes if requested and valid.

    Helper function to reduce code duplication between plot() and plot_field().
    Handles validation of dimension_ranges and emits warnings for invalid cases.

    Parameters
    ----------
    env : Environment
        The environment instance (provides dimension_ranges and units).
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    scale_bar : bool or ScaleBarConfig
        If truthy, adds a scale bar. If ScaleBarConfig, uses that configuration.
    """
    if not scale_bar:
        return

    from neurospatial.animation.config import ScaleBarConfig as _ScaleBarConfig
    from neurospatial.animation.config import add_scale_bar_to_axes

    config = scale_bar if isinstance(scale_bar, _ScaleBarConfig) else _ScaleBarConfig()

    # Validate extent is available and reasonable
    if not env.dimension_ranges:
        warnings.warn(
            "Cannot add scale bar: dimension_ranges not available",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    elif not np.isfinite(env.dimension_ranges[0]).all():
        warnings.warn(
            "Cannot add scale bar: dimension_ranges contains non-finite values",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    else:
        extent_x = env.dimension_ranges[0][1] - env.dimension_ranges[0][0]
        if extent_x < _MIN_EXTENT_FOR_SCALE_BAR:
            warnings.warn(
                f"Cannot add scale bar: extent too small ({extent_x})",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        else:
            add_scale_bar_to_axes(ax, extent_x, env.units, config)


# Helper functions for layout-specific rendering


def _plot_grid_field(
    env: VisualizationConsumer,
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
    from neurospatial.layout._grid_frame import require_grid_context
    from neurospatial.layout.helpers.utils import map_active_data_to_grid

    grid, frame = require_grid_context(
        env._binning, "plot_field()", fix="supply rectangular cells"
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
    env: VisualizationConsumer,
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
    from neurospatial.layout._plotting import polygon_patches

    shapes = env._binning.shapes
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
    env: VisualizationConsumer,
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
    shapes = env._binning.shapes
    if not isinstance(shapes, HexagonShapes):
        raise RuntimeError("Hexagonal field plotting requires hexagon cell shapes.")

    # Create hexagon patches
    # Note: Unlike grid layouts which use colormap's built-in NaN handling,
    # PatchCollection requires manual filtering of NaN values to avoid creating
    # empty patches that matplotlib doesn't handle well.
    patches = []
    colors = []

    for center, value in zip(shapes.centers, field, strict=True):
        # Skip NaN bins if nan_color is None
        if np.isnan(value) and nan_color is None:
            continue

        patch = RegularPolygon(
            xy=center,
            numVertices=6,
            radius=shapes.radius,
            orientation=shapes.orientation,
        )
        patches.append(patch)
        colors.append(value)

    if not patches:
        return None

    # Create collection
    collection = PatchCollection(patches, cmap=cmap, rasterized=rasterized, **kwargs)
    collection.set_array(np.array(colors))
    collection.set_clim(vmin, vmax)
    collection.set_edgecolor("none")

    ax.add_collection(collection)
    return collection


def _plot_trimesh_field(
    env: VisualizationConsumer,
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
    shapes = env._binning.shapes
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
    env: VisualizationConsumer,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    ylabel: str,
    **kwargs: Any,
) -> Any:
    """Plot a track field in physical display coordinates without bridging gaps."""
    edges: tuple[NDArray[np.float64], ...] | None
    active_mask: NDArray[np.bool_] | None
    linearization = env._binning.linearization
    if isinstance(linearization, TrackLinearization):
        edges, active_mask = linearization.grid_edges, linearization.active_mask
    else:
        edges, active_mask = env.grid_edges, env.active_mask
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

    unit_label = f" ({env.units})" if env.units else ""
    ax.set_xlabel(f"Linearized track coordinate{unit_label}", fontsize=12)
    ax.set_ylabel(ylabel if ylabel else "Field Value", fontsize=12)
    ax.set_title("1D Field Plot", fontsize=14)

    return line


def _plot_scatter_field(
    env: VisualizationConsumer,
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
    # Filter NaN values if nan_color is None
    if nan_color is None:
        valid_mask = ~np.isnan(field)
        plot_centers = env.bin_centers[valid_mask]
        plot_values = field[valid_mask]
    else:
        plot_centers = env.bin_centers
        plot_values = field.copy()

    if len(plot_centers) == 0:
        return None

    # Scatter sizes are areas in display points, never physical coordinate
    # units. Estimate visible spacing after accounting for the eventual axes
    # limits and aspect ratio. A cap keeps sparse layouts readable as points.
    if "s" not in kwargs and env.n_bins > 1:
        from scipy.spatial import cKDTree

        ranges = (
            np.asarray(env.dimension_ranges, dtype=float)
            if env.dimension_ranges is not None
            else np.column_stack(
                (env.bin_centers.min(axis=0), env.bin_centers.max(axis=0))
            )
        )
        spans = ranges[:2, 1] - ranges[:2, 0]
        axes_points = np.asarray(ax.get_window_extent().size) * 72.0 / ax.figure.dpi
        points_per_unit = np.divide(axes_points, spans, out=np.ones(2), where=spans > 0)
        if env._coordinate_system.equal_aspect and np.any(spans > 0):
            points_per_unit[:] = np.min(points_per_unit[spans > 0])
        display_centers = (env.bin_centers[:, :2] - ranges[:2, 0]) * points_per_unit
        tree = cKDTree(display_centers)
        # Find nearest neighbor distance for sample
        sample_size = min(100, env.n_bins)
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
        if sizes.ndim > 0 and sizes.size == env.n_bins:
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
    env: VisualizationConsumer,
    ax: matplotlib.axes.Axes | None = None,
    show_regions: bool = False,
    layout_plot_kwargs: dict[str, Any] | None = None,
    regions_plot_kwargs: dict[str, Any] | None = None,
    scale_bar: bool | ScaleBarConfig = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Implement Environment.plot using explicit visualization inputs."""
    l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
    l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

    from neurospatial.layout._plotting_binning import plot_binning

    ax = plot_binning(
        env._binning,
        env.connectivity,
        env.dimension_ranges,
        env.layout_type or "",
        ax=ax,
        **l_kwargs,
    )
    # layout.plot() always returns an Axes object, never None
    assert ax is not None

    if show_regions and hasattr(env, "regions") and env.regions is not None:
        from neurospatial.regions.plot import plot_regions

        r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
        plot_regions(env.regions, ax=ax, **r_kwargs)

    plot_title = env.name
    if env.layout_type:
        plot_title += f" ({env.layout_type})"

    # Only set title if layout.plot didn't set one or user didn't pass one via kwargs to layout.plot
    if ax.get_title() == "":
        ax.set_title(plot_title)

    # Add scale bar if requested
    _add_scale_bar_if_requested(env, ax, scale_bar)

    return ax


def _plot_1d(
    env: VisualizationConsumer,
    ax: matplotlib.axes.Axes | None = None,
    layout_plot_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes | None:
    """Implement Environment.plot_1d using explicit visualization inputs."""
    l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
    l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
    if env.is_linearized_track:
        linearization = env._binning.linearization
        if isinstance(linearization, TrackLinearization):
            from neurospatial.layout._plotting import plot_track_linear

            ax = plot_track_linear(
                ax=ax,
                title=env.layout_type or "",
                linearization=linearization,
                grid_edges=linearization.grid_edges,
                **l_kwargs,
            )
        else:
            warnings.warn(
                f"Layout '{env._layout_type_used}' is 1D but does not "
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
    env: VisualizationConsumer,
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
    """Implement Environment.plot_field using explicit visualization inputs."""
    if field.ndim != 1 or field.shape[0] != env.n_bins:
        raise ValueError(
            f"field must be 1D array with length n_bins={env.n_bins}, "
            f"got shape {field.shape}"
        )

    # Check dimensionality for spatial plots
    if not env.is_linearized_track and env.n_dims > 2:
        raise NotImplementedError(
            f"Cannot plot {env.n_dims}D fields spatially. "
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
    binning = env._binning
    mappable = None
    if env.n_dims == 1 and not env.is_linearized_track:
        ax.plot(env.bin_centers[:, 0], field, **kwargs)
        unit_label = f" ({env.units})" if env.units else ""
        ax.set_xlabel(f"Position{unit_label}")
        ax.set_ylabel(colorbar_label or "Field Value")
    elif binning.rectilinear is not None or binning.grid_frame is not None:
        mappable = _plot_grid_field(
            env, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif isinstance(binning.shapes, HexagonShapes):
        mappable = _plot_hex_field(
            env, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif isinstance(binning.shapes, TriangleShapes):
        mappable = _plot_trimesh_field(
            env, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif isinstance(binning.shapes, PolygonCells):
        mappable = _plot_polygon_field(
            env, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )
    elif binning.linearization is not None or env.is_linearized_track:
        # Legacy saved tracks retain their previous missing-grid refusal
        # until loading restores their full linearization data.
        mappable = _plot_1d_field(env, field, ax, colorbar_label, **kwargs)
    else:
        mappable = _plot_scatter_field(
            env, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
        )

    # Add colorbar (not for 1D)
    # Note: mappable can be None if all field values are NaN and nan_color=None
    if colorbar and mappable is not None and not env.is_linearized_track:
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
    if not env.is_linearized_track and env.n_dims != 1:
        system = env._coordinate_system
        xlabel, ylabel = system.axis_labels(env.units)
        if system.equal_aspect:
            ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if env.dimension_ranges and len(env.dimension_ranges) >= 2:
            ax.set_xlim(env.dimension_ranges[0])
            ax.set_ylim(env.dimension_ranges[1])

    # Add scale bar if requested
    _add_scale_bar_if_requested(env, ax, scale_bar)

    return ax


def _animate_fields(
    env: VisualizationConsumer,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    frame_times: NDArray[np.float64],
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    overwrite: bool = False,
    speed: float = 1.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: Sequence[str] | None = None,
    overlay_trajectory: NDArray[np.float64] | None = None,
    title: str = "Spatial Field Animation",
    dpi: int = 100,
    codec: str = "h264",
    bitrate: int = 5000,
    n_workers: int | None = None,
    dry_run: bool = False,
    image_format: Literal["png", "jpeg"] = "png",
    max_html_frames: int = 500,
    contrast_limits: tuple[float, float] | None = None,
    show_colorbar: bool = False,
    colorbar_label: str = "",
    overlays: list[OverlayProtocol] | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    scale_bar: bool | ScaleBarConfig = False,
    **kwargs: Any,
) -> Any:
    """Implement Environment.animate_fields using explicit visualization inputs."""
    from neurospatial.animation.core import animate_fields as _animate

    # frame_times is now required (no synthesis from fps)
    # The core animate_fields() handles validation
    return _animate(
        env=env,
        fields=fields,
        backend=backend,
        save_path=save_path,
        overwrite=overwrite,
        speed=speed,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        frame_labels=frame_labels,
        overlay_trajectory=overlay_trajectory,
        title=title,
        dpi=dpi,
        codec=codec,
        bitrate=bitrate,
        n_workers=n_workers,
        dry_run=dry_run,
        image_format=image_format,
        max_html_frames=max_html_frames,
        contrast_limits=contrast_limits,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label,
        overlays=overlays,
        frame_times=frame_times,
        show_regions=show_regions,
        region_alpha=region_alpha,
        scale_bar=scale_bar,
        **kwargs,  # Forward backend-specific parameters (e.g., layout, layer_names for napari)
    )
