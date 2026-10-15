"""Pure native cell-face formulas shared by builders and owned lazy sources."""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel

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
    centers: NDArray[np.float64],
    volumes: NDArray[np.float64],
    points: NDArray[np.float64],
    simplices: NDArray[np.intp],
    active_to_original: NDArray[np.intp],
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Compute shared triangle edges and report the original centroid skew guard."""
    g = graph.copy()
    skew_angle_rad = np.deg2rad(_MESH_SKEW_ANGLE_DEG)
    n_interior = 0
    n_skewed = 0

    for u, v, data in g.edges(data=True):
        verts_u = set(simplices[active_to_original[u]].tolist())
        verts_v = set(simplices[active_to_original[v]].tolist())
        shared = sorted(verts_u & verts_v)
        if len(shared) != 2:
            # Adjacent triangles must share an edge; if not, no diffusion face.
            data["A"] = 0.0
            continue
        p1 = points[shared[0]]
        p2 = points[shared[1]]
        edge_vec = np.asarray(p2 - p1, dtype=float)
        data["A"] = float(np.linalg.norm(edge_vec))

        # Skew: angle between the centroid-connection line and the shared-edge
        # normal (0 = K-orthogonal).
        n_interior += 1
        centroid_line = np.asarray(centers[v] - centers[u], dtype=float)
        edge_len = np.linalg.norm(edge_vec)
        line_len = np.linalg.norm(centroid_line)
        if edge_len > 0.0 and line_len > 0.0:
            normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            cos_angle = abs(float(np.dot(centroid_line / line_len, normal)))
            angle = float(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
            if angle > skew_angle_rad:
                n_skewed += 1

    if n_interior > 0 and (n_skewed / n_interior) > _MESH_SKEW_FRACTION:
        warnings.warn(
            f"Triangular mesh is non-K-orthogonal: {n_skewed}/{n_interior} "
            f"({100.0 * n_skewed / n_interior:.1f}%) of interior edges exceed "
            f"{_MESH_SKEW_ANGLE_DEG:.0f} degrees of non-orthogonality "
            f"(threshold {100.0 * _MESH_SKEW_FRACTION:.0f}%). The diffusion "
            f"bandwidth (sigma) is only approximate on this mesh; refine toward "
            f"well-shaped (near-equilateral) triangles for an exact physical sigma.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return g, volumes


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
    graph: nx.Graph, track_graph: nx.Graph, n_dims: int
) -> NDArray[np.float64]:
    """Capture physical junction positions without retaining the track builder."""
    result = np.full((graph.number_of_edges(), n_dims), np.nan)
    track_edges = list(track_graph.edges())
    node_pos = nx.get_node_attributes(track_graph, "pos")
    for index, (u, v) in enumerate(graph.edges()):
        seg_u = graph.nodes[u].get("source_edge_id")
        seg_v = graph.nodes[v].get("source_edge_id")
        if seg_u is None or seg_v is None or seg_u == seg_v:
            continue
        a, b = int(seg_u), int(seg_v)
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
        if abs(radius[u] - radius[v]) > 1e-9:
            data["A"] *= 0.5 * (radius[u] + radius[v])
        elif n_angle == 2 and circular:
            data["A"] *= 2.0
    return graph, volumes
