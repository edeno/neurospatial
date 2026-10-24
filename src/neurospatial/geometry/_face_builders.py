"""Pure native cell-face formulas shared by builders and owned lazy sources."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

_MESH_SKEW_ANGLE_DEG = 30.0
_MESH_SKEW_FRACTION = 0.05


def grid_geometry(
    graph: nx.Graph,
    centers: NDArray[np.float64],
    volumes: NDArray[np.float64],
    widths: NDArray[np.float64],
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Compute Cartesian grid faces from axis widths and ordered graph geometry."""
    g = graph.copy()
    n_dims = centers.shape[1]
    for u, v, data in g.edges(data=True):
        offset = centers[u] - centers[v]
        # Native grid centers share exactly stored coordinates along every
        # unchanged axis. An absolute threshold would depend on units.
        moved = offset != 0.0
        if moved.sum() != 1:  # diagonal / Moore edge -> shares only a corner
            data["A"] = 0.0
            continue
        axis = int(np.flatnonzero(moved)[0])
        data["A"] = float(np.prod([widths[d] for d in range(n_dims) if d != axis]))
    return g, volumes


def hex_geometry(
    graph: nx.Graph, volumes: NDArray[np.float64]
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Compute regular hexagon shared sides from navigation distances."""
    g = graph.copy()
    for _u, _v, data in g.edges(data=True):
        data["A"] = float(data["distance"]) / np.sqrt(3.0)
    return g, volumes


def mesh_geometry(
    graph: nx.Graph,
    volumes: NDArray[np.float64],
    points: NDArray[np.float64],
    simplices: NDArray[np.intp],
    active_to_original: NDArray[np.intp],
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Compute shared triangle edges as face areas."""
    g = graph.copy()
    for u, v, data in g.edges(data=True):
        verts_u = set(simplices[active_to_original[u]].tolist())
        verts_v = set(simplices[active_to_original[v]].tolist())
        shared = sorted(verts_u & verts_v)
        if len(shared) != 2:
            # Adjacent triangles must share an edge; if not, no diffusion face.
            data["A"] = 0.0
            continue
        data["A"] = float(np.linalg.norm(points[shared[1]] - points[shared[0]]))
    return g, volumes


def triangle_skew_message(
    points: NDArray[np.float64],
    triangles: NDArray[np.intp],
    centers: NDArray[np.float64],
    pairs: NDArray[np.intp],
) -> str | None:
    """The skew warning text for a triangle mesh, or None when well shaped.

    An interior edge is skewed when the line between the two triangles'
    centroids is more than ``_MESH_SKEW_ANGLE_DEG`` from the shared edge's
    normal (0 = K-orthogonal). Finite-volume smoothing then gives only an
    approximate bandwidth.

    Parameters
    ----------
    points : ndarray of float64, shape (n_points, 2)
        Mesh vertices.
    triangles : ndarray of intp, shape (n_bins, 3)
        Vertex indices of each bin's triangle.
    centers : ndarray of float64, shape (n_bins, 2)
        Bin centers (triangle centroids).
    pairs : ndarray of intp, shape (n_pairs, 2)
        Neighbor bin pairs; pairs that share two vertices are interior edges.

    Returns
    -------
    str or None
        The warning text when more than ``_MESH_SKEW_FRACTION`` of interior
        edges are skewed, otherwise None.
    """
    first = triangles[pairs[:, 0]]
    second = triangles[pairs[:, 1]]
    shared = (first[:, :, None] == second[:, None, :]).any(axis=2)
    interior = shared.sum(axis=1) == 2
    n_interior = int(interior.sum())
    if n_interior == 0:
        return None
    # Row-major order keeps each edge's two shared vertices in one row.
    vertices = first[interior][shared[interior]].reshape(-1, 2)
    edges = points[vertices[:, 1]] - points[vertices[:, 0]]
    lines = centers[pairs[interior, 1]] - centers[pairs[interior, 0]]
    edge_lengths = np.linalg.norm(edges, axis=1)
    line_lengths = np.linalg.norm(lines, axis=1)
    measured = (edge_lengths > 0.0) & (line_lengths > 0.0)
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])[measured]
    normals /= edge_lengths[measured, None]
    units = lines[measured] / line_lengths[measured, None]
    cos_angles = np.abs(np.sum(units * normals, axis=1))
    angles = np.arccos(np.clip(cos_angles, 0.0, 1.0))
    n_skewed = int(np.count_nonzero(angles > np.deg2rad(_MESH_SKEW_ANGLE_DEG)))
    if n_skewed / n_interior <= _MESH_SKEW_FRACTION:
        return None
    return (
        f"Triangular mesh is non-K-orthogonal: {n_skewed}/{n_interior} "
        f"({100.0 * n_skewed / n_interior:.1f}%) of interior edges exceed "
        f"{_MESH_SKEW_ANGLE_DEG:.0f} degrees of non-orthogonality "
        f"(threshold {100.0 * _MESH_SKEW_FRACTION:.0f}%). The diffusion "
        f"bandwidth is only approximate on this mesh; refine toward "
        f"well-shaped (near-equilateral) triangles for an exact physical "
        f"bandwidth."
    )


def track_geometry(
    graph: nx.Graph,
    centers: NDArray[np.float64],
    volumes: NDArray[np.float64],
    junctions: NDArray[np.float64],
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Compute unit track faces with along-junction flux distances."""
    g = graph.copy()
    for index, (u, v, data) in enumerate(g.edges(data=True)):
        data["A"] = 1.0
        j_pos = junctions[index]
        if not np.all(np.isfinite(j_pos)):
            continue
        d = float(np.linalg.norm(centers[u] - j_pos)) + float(
            np.linalg.norm(centers[v] - j_pos)
        )
        data["distance"] = d
    return g, volumes


def track_junctions(
    pairs: NDArray[np.intp],
    segment_ids: NDArray[np.int_],
    track_graph: nx.Graph,
    n_dims: int,
) -> NDArray[np.float64]:
    """Capture physical junction positions without retaining the track builder.

    Parameters
    ----------
    pairs : ndarray of int, shape (n_pairs, 2)
        Neighbor bin pairs.
    segment_ids : ndarray of int, shape (n_bins,)
        Track-graph edge ID (position in ``track_graph.edges()``) of each bin.
    track_graph : networkx.Graph
        Track graph whose nodes carry ``"pos"``.
    n_dims : int
        Number of coordinate dimensions.

    Returns
    -------
    ndarray of float64, shape (n_pairs, n_dims)
        Position of the track node shared by the segments of a pair's two
        bins; NaN for pairs on one segment or without exactly one shared node.
    """
    result = np.full((len(pairs), n_dims), np.nan)
    track_edges = list(track_graph.edges())
    node_pos = nx.get_node_attributes(track_graph, "pos")
    segments = np.asarray(segment_ids)[np.asarray(pairs, dtype=np.intp)].reshape(-1, 2)
    for index in np.flatnonzero(segments[:, 0] != segments[:, 1]):
        a, b = int(segments[index, 0]), int(segments[index, 1])
        if not (0 <= a < len(track_edges) and 0 <= b < len(track_edges)):
            continue
        shared = set(track_edges[a]) & set(track_edges[b])
        if len(shared) == 1:
            result[index] = node_pos[next(iter(shared))]
    result.setflags(write=False)
    return result


def polar_geometry(
    graph: nx.Graph,
    centers: NDArray[np.float64],
    volumes: NDArray[np.float64],
    widths: NDArray[np.float64],
    n_angle: int,
    circular: bool,
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Compute annular-sector arcs and summed two-bin seam interfaces."""
    graph, volumes = grid_geometry(graph, centers, volumes, widths)
    radius = centers[:, 0]
    for u, v, data in graph.edges(data=True):
        # Bins of one ring store the same radius exactly, as grid_geometry
        # finds a pair's axis, so the test needs no unit-dependent threshold.
        if radius[u] != radius[v]:
            data["A"] *= 0.5 * (radius[u] + radius[v])
        elif n_angle == 2 and circular:
            data["A"] *= 2.0
    return graph, volumes
