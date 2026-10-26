"""Draw layout geometry from binning data without retaining a layout engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from shapely import Polygon

    from neurospatial.geometry import (
        FramedGrid,
        Outline,
        PolygonCells,
        RectilinearGrid,
        TrackLinearization,
        TriangleShapes,
    )


def _draw_connections(
    ax: Axes, centers: NDArray[np.float64], pairs: NDArray[np.intp], **style: Any
) -> None:
    """Draw every connection between bin centers as one line collection.

    One collection draws in about the time of one line; a line per connection
    takes seconds at ten thousand bins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    centers : ndarray of float64, shape (n_bins, 2)
        Bin centers.
    pairs : ndarray of intp, shape (n_pairs, 2)
        Bin indices of each connected pair.
    **style
        Passed to :class:`matplotlib.collections.LineCollection`.
    """
    from matplotlib.collections import LineCollection

    # An (n, 2, 2) array is a sequence of segments; the stubs ask for a list.
    segments = cast("Sequence[NDArray[np.float64]]", centers[pairs])
    ax.add_collection(LineCollection(segments, **style), autolim=False)


def _polygon_corners(polygons: Sequence[Polygon]) -> NDArray[np.float64]:
    """Corners of hole-free polygons that all have the same number of corners.

    One array of corners draws as a single ``PolyCollection``, much faster
    than a patch per polygon.

    Parameters
    ----------
    polygons : sequence of shapely.Polygon
        Polygons without interior rings, each with ``n_corners`` corners.

    Returns
    -------
    ndarray of float64, shape (n_polygons, n_corners, 2)
        Each exterior ring counterclockwise, without its closing repeat: the
        vertices :func:`_polygon_patches` draws.
    """
    import shapely

    rings = shapely.get_coordinates(
        shapely.get_exterior_ring(np.asarray(polygons, dtype=object))
    ).reshape(len(polygons), -1, 2)
    x, y = rings[..., 0], rings[..., 1]
    clockwise = np.sum(x[:, :-1] * y[:, 1:] - x[:, 1:] * y[:, :-1], axis=1) < 0
    rings[clockwise] = rings[clockwise, ::-1]
    corners: NDArray[np.float64] = rings[:, :-1]
    return corners


def _polygon_patches(polygons: Sequence[Polygon]) -> list[Any]:
    """Build compound paths whose interior rings remain unfilled."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    from shapely.geometry.polygon import orient

    patches = []
    for cell in polygons:
        polygon = orient(cell, sign=1.0)
        paths = []
        for ring in (polygon.exterior, *polygon.interiors):
            vertices = np.asarray(ring.coords)
            codes = np.full(len(vertices), Path.LINETO, dtype=np.uint8)
            codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
            paths.append(Path(vertices, codes))
        patches.append(PathPatch(Path.make_compound_path(*paths)))
    return patches


def plot_polygon_cells(
    ax: Axes | None = None,
    *,
    shapes: PolygonCells,
    title: str,
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    figsize: tuple[float, float] = (7, 7),
    cmap: str = "bone_r",
    alpha: float = 0.7,
    show_connectivity: bool = True,
    node_size: float = 20,
    node_color: str = "blue",
    **kwargs: Any,
) -> Axes:
    """Draw exact polygon cells, retaining holes and disconnected pieces."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    collection = PatchCollection(
        _polygon_patches(shapes.polygons), cmap=cmap, alpha=alpha, **kwargs
    )
    collection.set_array(np.ones(len(shapes.polygons)))
    ax.add_collection(collection)
    ax.autoscale_view()
    if show_connectivity:
        positions = nx.get_node_attributes(connectivity, "pos")
        nx.draw_networkx_nodes(
            connectivity, positions, ax=ax, node_size=node_size, node_color=node_color
        )
        nx.draw_networkx_edges(connectivity, positions, ax=ax)
    if dimension_ranges is not None:
        ax.set_xlim(dimension_ranges[0])
        ax.set_ylim(dimension_ranges[1])
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def _grid_mesh(
    rectilinear: RectilinearGrid, frame: FramedGrid | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cell corner coordinates of a 2-D grid for ``pcolormesh``.

    Parameters
    ----------
    rectilinear : RectilinearGrid
        Two-dimensional grid; with ``frame``, the grid in source coordinates.
    frame : FramedGrid or None
        Map from source to world coordinates of a rotated, reflected or
        sheared grid.

    Returns
    -------
    x, y : ndarray of float64
        The axis edges, shapes ``(n_x + 1,)`` and ``(n_y + 1,)``, or with a
        frame the world corners, each of shape ``(n_y + 1, n_x + 1)``.
    """
    x, y = rectilinear.edges
    if frame is not None:
        xx, yy = np.meshgrid(x, y, indexing="xy")
        corners = frame.to_world(np.column_stack((xx.ravel(), yy.ravel())))
        x, y = (corners[:, axis].reshape(xx.shape) for axis in range(2))
    return x, y


def plot_grid(
    ax: Axes | None = None,
    *,
    title: str,
    bin_centers: NDArray[np.float64],
    pairs: NDArray[np.intp],
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    rectilinear: RectilinearGrid,
    frame: FramedGrid | None = None,
    figsize: tuple[float, float] = (7, 7),
    cmap: str = "bone_r",
    alpha: float = 0.7,
    show_connectivity: bool = True,
    node_size: float = 20,
    node_color: str = "blue",
) -> Axes:
    """Draw active rectangular cells and their connectivity in two dimensions."""
    import matplotlib.pyplot as plt

    grid_edges = rectilinear.edges
    if len(rectilinear.shape) != 2 or len(grid_edges) != 2:
        raise NotImplementedError(
            "Plotting for non-2D grid layouts is not implemented yet."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    x, y = _grid_mesh(rectilinear, frame)
    # Fill only the active cells, light whatever the mix of active and
    # inactive cells, so the same bin looks the same in every grid.
    active = rectilinear.active_mask.T
    ax.pcolormesh(
        x,
        y,
        np.ma.masked_where(~active, np.ones(active.shape)),
        cmap=cmap,
        vmin=0.0,
        vmax=4.0,
        alpha=alpha,
        shading="flat",
    )
    if frame is None:
        # Every cell edge as one line collection, independent of the ticks: a
        # label per edge is unreadable and slow, and Matplotlib drops minor
        # ticks (and their gridlines) where they fall on major ones.
        from matplotlib.collections import LineCollection

        x_edges, y_edges = grid_edges
        columns = np.stack(
            [
                np.column_stack((x_edges, np.full_like(x_edges, y_edges[0]))),
                np.column_stack((x_edges, np.full_like(x_edges, y_edges[-1]))),
            ],
            axis=1,
        )
        rows = np.stack(
            [
                np.column_stack((np.full_like(y_edges, x_edges[0]), y_edges)),
                np.column_stack((np.full_like(y_edges, x_edges[-1]), y_edges)),
            ],
            axis=1,
        )
        edges = LineCollection(
            cast("Sequence[NDArray[np.float64]]", np.concatenate([columns, rows])),
            colors="gray",
            linewidths=0.5,
            zorder=1.5,  # where gridlines draw: over the cells, under the graph
        )
        ax.add_collection(edges, autolim=False)
    ax.set_aspect("equal")
    ax.set_title(f"{title} (2D Grid)")
    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    if dimension_ranges:
        ax.set_xlim(dimension_ranges[0])
        ax.set_ylim(dimension_ranges[1])
    if show_connectivity:
        node_position = nx.get_node_attributes(connectivity, "pos")
        nx.draw_networkx_nodes(
            connectivity,
            node_position,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
        )
        _draw_connections(ax, bin_centers, pairs, colors="black", zorder=-1)
    return ax


def plot_polygon_grid(
    ax: Axes | None = None, *, outline: Outline, **grid_kwargs: Any
) -> Axes:
    """Draw a masked rectangular grid and the outline of its defining polygon.

    ``grid_kwargs`` are the options of :func:`plot_grid`.
    """
    ax = plot_grid(ax, **grid_kwargs)
    polygon = outline.geometry
    poly_patch_kwargs = {"fc": "none", "ec": "black", "lw": 1.5}
    if hasattr(polygon, "geoms"):
        for geom in polygon.geoms:
            if hasattr(geom, "exterior"):
                x, y = geom.exterior.xy
                ax.fill(x, y, **poly_patch_kwargs)
    elif hasattr(polygon, "exterior"):
        x, y = polygon.exterior.xy
        ax.fill(x, y, **poly_patch_kwargs)
    return ax


def _generic_graph_plot(
    graph: nx.Graph,
    name: str,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a graph whose nodes have 2-D ``pos`` coordinates.

    Parameters
    ----------
    graph : nx.Graph
        The graph to plot. Nodes must have a 2-D ``pos`` attribute.
    name : str
        A name for the graph, used in the plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None (default), a new figure and axes are created.
    **kwargs : Any
        - ``figsize`` (tuple of float): Size of the figure if created.
        - ``node_kwargs`` (dict): Keyword arguments for ``nx.draw_networkx_nodes``.
        - ``edge_kwargs`` (dict): Keyword arguments for ``nx.draw_networkx_edges``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes on which the graph was plotted.

    Raises
    ------
    ValueError
        If `graph` is empty, or its nodes have no ``pos``.

    """
    import matplotlib.pyplot as plt

    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty or None. Cannot plot an empty graph.")

    node_positions = nx.get_node_attributes(graph, "pos")
    if not node_positions:  # No 'pos' attributes
        raise ValueError("Nodes in graph are missing 'pos' attribute for plotting.")

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))
    node_kwargs = {"node_size": 20, "ax": ax, **kwargs.get("node_kwargs", {})}
    nx.draw_networkx_nodes(graph, pos=node_positions, **node_kwargs)
    edge_kwargs = {
        "alpha": 0.5,
        "edge_color": "gray",
        **kwargs.get("edge_kwargs", {}),
        "ax": ax,
    }
    nx.draw_networkx_edges(graph, pos=node_positions, **edge_kwargs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name} Graph")
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    return ax


def plot_hexagons(
    ax: Axes | None = None,
    *,
    title: str,
    centers: NDArray[np.float64],
    cells: NDArray[np.float64],
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    **kwargs: Any,
) -> Axes:
    """Draw hexagonal cells and the graph of active bin centers.

    ``cells``, shape ``(n_bins, 6, 2)``, are the corners of the bins' own
    polygons (``geometry.cell_polygons``), so the drawing shows the cells the
    analyses use.
    """
    from matplotlib.collections import PolyCollection

    ax = _generic_graph_plot(ax=ax, graph=connectivity, name=title, **kwargs)
    if kwargs.get("show_hexagons", True):
        hex_kws = kwargs.get(
            "hexagon_kwargs",
            {
                "edgecolor": "gray",
                "facecolor": "none",
                "alpha": 0.5,
                "linewidth": 0.5,
            },
        )
        ax.scatter(centers[:, 0], centers[:, 1], s=1, label="hexagonal grid")
        corners = cast("Sequence[NDArray[np.float64]]", cells)
        ax.add_collection(PolyCollection(corners, **hex_kws))
        ax.plot(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            markersize=1,
            color="blue",
            linestyle="None",
            label="midpoint",
        )
        ax.set_title(f"{title} Layout")
        # The ranges are the cells' extent, corners included, as in plot_field.
        if dimension_ranges is not None:
            ax.set_xlim(dimension_ranges[0])
            ax.set_ylim(dimension_ranges[1])
        ax.set_aspect("equal", adjustable="box")
    return ax


def plot_triangles(
    ax: Axes | None = None,
    *,
    title: str,
    shapes: TriangleShapes,
    outline: Outline | None,
    pairs: NDArray[np.intp],
    dimension_ranges: Sequence[tuple[float, float]],
    bin_centers: NDArray[np.float64],
    show_triangles: bool = True,
    show_centroids: bool = True,
    show_connectivity: bool = True,
    show_boundary: bool = True,
    triangle_kwargs: dict[str, Any] | None = None,
    centroid_kwargs: dict[str, Any] | None = None,
    connectivity_kwargs: dict[str, Any] | None = None,
    boundary_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Draw active mesh triangles, boundary rings, centroids and graph edges."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    _triangle_kwargs: dict[str, Any] = {
        "alpha": 0.4,
        "facecolor": "lightblue",
        "edgecolor": "gray",
        "linewidth": 0.5,
    }
    if triangle_kwargs:
        _triangle_kwargs.update(triangle_kwargs)
    _centroid_kwargs = {"color": "blue", "s": 10, "zorder": 3}
    if centroid_kwargs:
        _centroid_kwargs.update(centroid_kwargs)
    _connectivity_kwargs = {
        "color": "black",
        "alpha": 0.5,
        "linewidth": 0.75,
        "zorder": 2,
    }
    if connectivity_kwargs:
        _connectivity_kwargs.update(connectivity_kwargs)
    _boundary_kwargs = {
        "color": "black",
        "linewidth": 1.5,
        "linestyle": "--",
        "zorder": 4,
    }
    if boundary_kwargs:
        _boundary_kwargs.update(boundary_kwargs)
    boundary = outline.geometry if outline is not None else None
    if show_boundary and boundary:
        plot_func = cast("Any", ax.plot)
        polygons = boundary.geoms if hasattr(boundary, "geoms") else (boundary,)
        for index, polygon in enumerate(polygons):
            xb, yb = polygon.exterior.xy
            plot_func(
                xb, yb, label="Boundary" if index == 0 else None, **_boundary_kwargs
            )
            for interior in polygon.interiors:
                xbi, ybi = interior.xy
                plot_func(xbi, ybi, **_boundary_kwargs)
    if show_triangles:
        corners = cast("Sequence[NDArray[np.float64]]", shapes.points[shapes.triangles])
        ax.add_collection(PolyCollection(corners, **_triangle_kwargs))
    if show_centroids and len(bin_centers):
        scatter_func = cast("Any", ax.scatter)
        scatter_func(bin_centers[:, 0], bin_centers[:, 1], **_centroid_kwargs)
    if show_connectivity:
        line_style = dict(_connectivity_kwargs)
        line_style["colors"] = line_style.pop("color", "black")
        if "linewidth" in line_style:
            line_style["linewidths"] = line_style.pop("linewidth")
        _draw_connections(ax, bin_centers, pairs, **line_style)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(dimension_ranges[0])
    ax.set_ylim(dimension_ranges[1])
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(title)
    if show_boundary and boundary:
        ax.legend()
    return ax


def plot_track(
    ax: Axes | None = None,
    *,
    title: str,
    linearization: TrackLinearization,
    bin_centers: NDArray[np.float64],
    pairs: NDArray[np.intp],
    connectivity: nx.Graph,
    grid_edges: tuple[NDArray[np.float64], ...] | None,
    figsize: tuple[float, float] = (7, 7),
) -> Axes:
    """Draw the track graph, active bin centers and projected linear bin edges."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"{title} Layout")
    original_node_pos = nx.get_node_attributes(linearization.graph, "pos")
    nx.draw_networkx_nodes(
        linearization.graph,
        original_node_pos,
        ax=ax,
        node_size=300,
        node_color="#1f77b4",
    )
    for node_id1, node_id2 in linearization.graph.edges:
        pos = np.stack((original_node_pos[node_id1], original_node_pos[node_id2]))
        ax.plot(pos[:, 0], pos[:, 1], color="gray", zorder=-1, label="original edges")
    for node_id, pos in original_node_pos.items():
        ax.text(pos[0], pos[1], str(node_id), ha="center", va="center", zorder=10)
    bin_node_pos = nx.get_node_attributes(connectivity, "pos")
    nx.draw_networkx_nodes(
        connectivity, bin_node_pos, ax=ax, node_size=30, node_color="black"
    )
    _draw_connections(ax, bin_centers, pairs, colors="black", zorder=-1)
    if grid_edges is not None:
        edge_points = linearization.linear_to_nd(grid_edges[0])
        ax.plot(
            edge_points[:, 0],
            edge_points[:, 1],
            color="gray",
            marker="+",
            linestyle="none",
            alpha=0.8,
            label="bin edges",
        )
    return ax


def _plot_track_linear(
    ax: Axes | None = None,
    *,
    title: str,
    linearization: TrackLinearization,
    grid_edges: tuple[NDArray[np.float64], ...] | None,
    **kwargs: Any,
) -> Axes:
    """Draw the absolute linear track coordinate system and its bin edges."""
    import matplotlib.pyplot as plt
    from track_linearization import plot_graph_as_1D

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 3)))
    spacing = linearization.edge_spacing
    plot_graph_as_1D(
        linearization.graph,
        list(linearization.edge_order),
        spacing if isinstance(spacing, (int, float)) else list(spacing),
        ax=ax,
        **kwargs,
    )
    if grid_edges is not None:
        ax.vlines(
            grid_edges[0],
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors="gray",
            linestyles="--",
            alpha=0.5,
        )
    ax.set_title(f"{title} Layout")
    ax.set_xlabel("Linearized Position")
    ax.set_ylabel("Bin Index")
    return ax
