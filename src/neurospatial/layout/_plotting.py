"""Draw layout geometry from binning data without retaining a layout engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import RegularPolygon

    from neurospatial.geometry._binning import TrackLinearization
    from neurospatial.geometry._capabilities import RectilinearGrid
    from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
    from neurospatial.geometry._grid_frame import FramedGrid
    from neurospatial.geometry._polygon import PolygonCells


def polygon_patches(shapes: PolygonCells) -> list[Any]:
    """Build compound paths whose interior rings remain unfilled."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    from shapely.geometry.polygon import orient

    patches = []
    for cell in shapes.polygons:
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
        polygon_patches(shapes), cmap=cmap, alpha=alpha, **kwargs
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


def plot_grid(
    ax: Axes | None = None,
    *,
    title: str,
    bin_centers: NDArray[np.float64],
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
    x, y = grid_edges
    if frame is not None:
        xx, yy = np.meshgrid(x, y, indexing="xy")
        corners = frame.to_world(np.column_stack((xx.ravel(), yy.ravel())))
        x, y = (corners[:, axis].reshape(xx.shape) for axis in range(2))
    ax.pcolormesh(
        x,
        y,
        rectilinear.active_mask.T,
        cmap=cmap,
        alpha=alpha,
        shading="auto",
    )
    if frame is None:
        ax.set_xticks(grid_edges[0])
        ax.set_yticks(grid_edges[1])
        ax.grid(True, ls="-", lw=0.5, c="gray")
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
        for node_id1, node_id2 in connectivity.edges:
            pos = np.stack((node_position[node_id1], node_position[node_id2]))
            ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)
    return ax


def plot_polygon_grid(
    ax: Axes | None = None,
    *,
    title: str,
    bin_centers: NDArray[np.float64],
    connectivity: nx.Graph,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    rectilinear: RectilinearGrid,
    outline: Outline,
    figsize: tuple[float, float] = (7, 7),
    cmap: str = "bone_r",
    alpha: float = 0.7,
    show_connectivity: bool = True,
    node_size: float = 20,
    node_color: str = "blue",
) -> Axes:
    """Draw a masked rectangular grid and fill the defining polygon exterior."""
    ax = plot_grid(
        ax,
        title=title,
        bin_centers=bin_centers,
        connectivity=connectivity,
        dimension_ranges=dimension_ranges,
        rectilinear=rectilinear,
        figsize=figsize,
        cmap=cmap,
        alpha=alpha,
        show_connectivity=show_connectivity,
        node_size=node_size,
        node_color=node_color,
    )
    polygon = outline.geometry
    poly_patch_kwargs = {"alpha": 0.3, "fc": "gray", "ec": "black"}
    if hasattr(polygon, "geoms"):
        for geom in polygon.geoms:
            if hasattr(geom, "exterior"):
                x, y = geom.exterior.xy
                ax.fill(x, y, **poly_patch_kwargs)
    elif hasattr(polygon, "exterior"):
        x, y = polygon.exterior.xy
        ax.fill(x, y, **poly_patch_kwargs)
    return ax


def hexagon_patches(
    shapes: HexagonShapes, centers: NDArray[np.float64]
) -> list[RegularPolygon]:
    """Build one hexagon patch per center, with the cells' radius and orientation.

    Parameters
    ----------
    shapes : HexagonShapes
        Hexagonal cells supplying the radius and orientation.
    centers : ndarray of float64, shape (n_patches, 2)
        Centers of the hexagons to draw.

    Returns
    -------
    list of matplotlib.patches.RegularPolygon
    """
    from matplotlib.patches import RegularPolygon

    return [
        RegularPolygon(
            (x, y),
            numVertices=6,
            radius=shapes.radius,
            orientation=shapes.orientation,
        )
        for x, y in centers
    ]


def plot_hexagons(
    ax: Axes | None = None,
    *,
    title: str,
    shapes: HexagonShapes | None,
    connectivity: nx.Graph | None,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    **kwargs: Any,
) -> Axes:
    """Draw hexagonal cells and the graph of active bin centers."""
    from matplotlib.collections import PatchCollection

    from neurospatial.layout.helpers.utils import _generic_graph_plot

    ax = _generic_graph_plot(ax=ax, graph=connectivity, name=title, **kwargs)
    if kwargs.get("show_hexagons", True) and shapes is not None and len(shapes.centers):
        hex_kws = kwargs.get(
            "hexagon_kwargs",
            {
                "edgecolor": "gray",
                "facecolor": "none",
                "alpha": 0.5,
                "linewidth": 0.5,
            },
        )
        ax.scatter(
            shapes.centers[:, 0], shapes.centers[:, 1], s=1, label="hexagonal grid"
        )
        collection = PatchCollection(hexagon_patches(shapes, shapes.centers), **hex_kws)
        ax.add_collection(collection)
        ax.plot(
            shapes.centers[:, 0],
            shapes.centers[:, 1],
            marker="o",
            markersize=1,
            color="blue",
            linestyle="None",
            label="midpoint",
        )
        ax.set_title(f"{title} Layout")
        padding = 1.1 * shapes.radius
        if dimension_ranges is not None:
            ax.set_xlim(
                (dimension_ranges[0][0] - padding, dimension_ranges[0][1] + padding)
            )
            ax.set_ylim(
                (dimension_ranges[1][0] - padding, dimension_ranges[1][1] + padding)
            )
        ax.set_aspect("equal", adjustable="box")
    return ax


def plot_triangles(
    ax: Axes | None = None,
    *,
    title: str,
    shapes: TriangleShapes,
    outline: Outline | None,
    connectivity: nx.Graph,
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
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    _triangle_kwargs = {
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
        patches = [
            MplPolygon(vertices, closed=True)
            for vertices in shapes.points[shapes.triangles]
        ]
        pc_constructor = cast("Any", PatchCollection)
        pc = pc_constructor(patches, **_triangle_kwargs)
        ax.add_collection(pc)
    if show_centroids and len(bin_centers):
        scatter_func = cast("Any", ax.scatter)
        scatter_func(bin_centers[:, 0], bin_centers[:, 1], **_centroid_kwargs)
    if show_connectivity:
        plot_func2 = cast("Any", ax.plot)
        for u, v in connectivity.edges():
            pos_u = connectivity.nodes[u]["pos"]
            pos_v = connectivity.nodes[v]["pos"]
            plot_func2(
                [pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], **_connectivity_kwargs
            )
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
    connectivity: nx.Graph,
    grid_edges: tuple[NDArray[np.float64], ...] | None,
    **kwargs: Any,
) -> Axes:
    """Draw the track graph, active bin centers and projected linear bin edges."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
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
    for node_id1, node_id2 in connectivity.edges:
        pos = np.stack((bin_node_pos[node_id1], bin_node_pos[node_id2]))
        ax.plot(pos[:, 0], pos[:, 1], color="black", zorder=-1)
    if grid_edges is not None:
        grid_line_nd = linearization.linear_to_nd(grid_edges[0])
        for grid_line in grid_line_nd:
            ax.plot(
                grid_line[0],
                grid_line[1],
                color="gray",
                marker="+",
                alpha=0.8,
                label="bin edges",
            )
    return ax


def plot_track_linear(
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
        for grid_line in grid_edges[0]:
            ax.axvline(grid_line, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"{title} Layout")
    ax.set_xlabel("Linearized Position")
    ax.set_ylabel("Bin Index")
    return ax
