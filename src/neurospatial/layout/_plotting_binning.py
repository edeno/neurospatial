"""Plot retained owned cell components without a layout facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx

from neurospatial.layout._binning import Binning, TrackLinearization
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._context import grid_index
from neurospatial.layout._plotting import (
    plot_grid,
    plot_hexagons,
    plot_polygon_cells,
    plot_polygon_grid,
    plot_track,
    plot_track_linear,
    plot_triangles,
)
from neurospatial.layout._polygon import PolygonCells

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_binning(
    binning: Binning,
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    title: str,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw retained track, shape or rectangular geometry by component."""
    data = binning
    common: dict[str, Any] = {
        "title": title,
        "connectivity": connectivity,
        "dimension_ranges": dimension_ranges,
    }
    if isinstance(data.linearization, TrackLinearization):
        return plot_track(
            ax,
            title=title,
            linearization=data.linearization,
            bin_centers=binning.bins.centers,
            connectivity=connectivity,
            grid_edges=grid_index(binning).edges,
            **kwargs,
        )
    if isinstance(data.shapes, HexagonShapes):
        return plot_hexagons(ax, shapes=data.shapes, **common, **kwargs)
    if isinstance(data.shapes, TriangleShapes):
        ranges = dimension_ranges
        if ranges is None:
            ranges = tuple(
                (float(axis.min()), float(axis.max()))
                for axis in binning.bins.centers.T
            )
        return plot_triangles(
            ax,
            title=title,
            shapes=data.shapes,
            outline=data.outline,
            connectivity=connectivity,
            dimension_ranges=ranges,
            bin_centers=binning.bins.centers,
            **kwargs,
        )
    if isinstance(data.shapes, PolygonCells):
        return plot_polygon_cells(ax, shapes=data.shapes, **common, **kwargs)
    if data.grid_frame is not None:
        return plot_grid(
            ax,
            rectilinear=data.grid_frame.source,
            frame=data.grid_frame,
            bin_centers=binning.bins.centers,
            **common,
            **kwargs,
        )
    if data.rectilinear is not None:
        if data.outline is not None:
            return plot_polygon_grid(
                ax,
                rectilinear=data.rectilinear,
                outline=data.outline,
                bin_centers=binning.bins.centers,
                **common,
                **kwargs,
            )
        return plot_grid(
            ax,
            rectilinear=data.rectilinear,
            bin_centers=binning.bins.centers,
            **common,
            **kwargs,
        )
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    if binning.bins.centers.shape[1] == 2:
        ax.scatter(binning.bins.centers[:, 0], binning.bins.centers[:, 1], **kwargs)
    return ax


def plot_linear_binning(
    binning: Binning, title: str, ax: Axes | None = None, **kwargs: Any
) -> Axes:
    """Draw a retained track in its absolute linear coordinate system."""
    track = binning.linearization
    if not isinstance(track, TrackLinearization):
        raise NotImplementedError(
            "plot_linear_layout needs stored track linearization."
        )
    return plot_track_linear(
        ax,
        title=title,
        linearization=track,
        grid_edges=track.grid_edges,
        **kwargs,
    )
