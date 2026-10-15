"""Internal finite-volume diffusion data and the layout geometry contract.

This foundation module knows no Environment or solver. A layout that supports
diffusion and differential operators defines ``finite_volume_geometry()``: a
copy of its connectivity graph with face measures (``A``) and edge distances,
plus cell measures. ``FaceGeometry.from_graph`` validates that into per-edge
arrays in the supplied graph's order. Explicit binning data can supply another
edge order; ``env.edge_pairs`` defines gradient and flux alignment in either
case. ``DiffusionGeometry.from_faces`` assembles the conductance snapshot every
diffusion consumer shares.

Native construction captures owned face inputs once. Published binnings
retain numerical capability as an explicit faces source. Unavailable stored
geometry uses ``MissingFaces`` with the reason an operation cannot proceed.
"""

from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class FaceGeometry(NamedTuple):
    """Validated, read-only cell faces in authoritative edge order.

    ``edges[k] = (u, v)`` defines the orientation of edge field entry ``k``.
    Published faces align exactly with ``env.edge_pairs``. ``from_graph``
    captures the supplied graph's iteration order; explicit faces may use a
    different order from NetworkX iteration.
    """

    edges: NDArray[np.intp]
    A: NDArray[np.float64]
    distance: NDArray[np.float64]
    volumes: NDArray[np.float64]

    @classmethod
    def from_graph(cls, graph: nx.Graph, volumes: NDArray[np.float64]) -> FaceGeometry:
        """Validate and snapshot face measures, distances and cell measures.

        Parameters
        ----------
        graph : networkx.Graph
            Undirected simple graph with integer nodes ``0..n_bins-1``.
            Each edge supplies finite ``A >= 0`` and ``distance > 0``.
        volumes : ndarray, shape (n_bins,)
            Finite, strictly positive cell measures in node-index order.

        Returns
        -------
        FaceGeometry
            Independent snapshot with read-only arrays of shape ``(n_edges, 2)``,
            ``(n_edges,)``, ``(n_edges,)`` and ``(n_bins,)``.

        Raises
        ------
        ValueError
            If the graph is directed or a multigraph, its nodes are not
            ``0..n_bins-1``, the cell measures are invalid, or an edge lacks a
            valid ``A`` or ``distance``.
        """
        n_bins = graph.number_of_nodes()
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError(
                f"finite-volume geometry needs an undirected simple nx.Graph, got "
                f"{type(graph).__name__}. Each edge is one shared face, so reciprocal "
                "or parallel edges would sum into a doubled conductance. Collapse "
                "them first, e.g. nx.Graph(graph) after merging parallel edges."
            )
        nodes = list(graph.nodes)
        if not all(
            isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in nodes
        ) or {int(x) for x in nodes} != set(range(n_bins)):
            raise ValueError(
                "graph nodes must be contiguous integer labels 0..n-1 (no float/bool)."
            )
        measures = _validate_volumes(volumes, n_bins).copy()

        edges: list[tuple[int, int]] = []
        areas: list[float] = []
        distances: list[float] = []
        for u, v, data in graph.edges(data=True):
            if "A" not in data or "distance" not in data:
                raise ValueError(
                    f"edge ({u},{v}) is missing 'A' and/or 'distance' attribute."
                )
            edges.append((int(u), int(v)))
            areas.append(float(data["A"]))
            distances.append(float(data["distance"]))
        edge_array = np.array(edges, dtype=np.intp).reshape(-1, 2)
        area = np.array(areas, dtype=np.float64)
        distance = np.array(distances, dtype=np.float64)
        bad = ~(np.isfinite(area) & (area >= 0.0))
        if bad.any():
            k = int(np.argmax(bad))
            u, v = edge_array[k]
            raise ValueError(f"edge ({u},{v}) has invalid face measure A={area[k]}.")
        bad = ~(np.isfinite(distance) & (distance > 0.0))
        if bad.any():
            k = int(np.argmax(bad))
            u, v = edge_array[k]
            raise ValueError(f"edge ({u},{v}) has invalid distance d={distance[k]}.")
        for array in (edge_array, area, distance, measures):
            array.setflags(write=False)
        return cls(edge_array, area, distance, measures)


class DiffusionGeometry(NamedTuple):
    """Validated finite-volume diffusion data in active-bin order.

    ``W[i, j] = A[i, j] / distance[i, j]`` holds symmetric nonnegative
    conductances; ``volumes`` holds positive cell measures (lengths, areas, or
    volumes). Components are derived from positive conductances, not from a
    layout's navigation graph: corner-touching bins (``A = 0``) are separate
    components, which the truncated ``env.diffuse`` eigenbasis and the support
    gates rely on. Arrays are read-only, so cached operators cannot change
    through aliases to layout data.

    Environment owns the snapshot's lifetime and invalidates it together with
    its eigencaches when geometry changes.
    """

    W: sparse.csr_matrix
    volumes: NDArray[np.float64]
    n_components: int
    labels: NDArray[np.int_]

    @classmethod
    def from_faces(cls, faces: FaceGeometry) -> DiffusionGeometry:
        """Assemble conductances ``A / distance`` and their components.

        Parameters
        ----------
        faces : FaceGeometry
            Validated cell faces and measures.

        Returns
        -------
        DiffusionGeometry
            Read-only snapshot sharing ``faces.volumes``.
        """
        n_bins = faces.volumes.size
        conductance = faces.A / faces.distance
        flows = conductance > 0.0  # zero-area faces carry no diffusion
        u, v = faces.edges[flows, 0], faces.edges[flows, 1]
        w = conductance[flows]
        weights = sparse.csr_matrix(
            (np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(n_bins, n_bins),
            dtype=np.float64,
        )
        n_components, labels = connected_components(weights, directed=False)
        for array in (weights.data, weights.indices, weights.indptr, labels):
            array.setflags(write=False)
        return cls(weights, faces.volumes, int(n_components), labels)

    @classmethod
    def from_graph(
        cls, graph: nx.Graph, volumes: NDArray[np.float64]
    ) -> DiffusionGeometry:
        """Validate faces from ``graph`` (see ``FaceGeometry.from_graph``) and assemble."""
        return cls.from_faces(FaceGeometry.from_graph(graph, volumes))


def _validate_volumes(volumes: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Validate cell measures before any division or matrix construction."""
    measures = np.asarray(volumes, dtype=np.float64)
    if measures.shape != (n_bins,):
        raise ValueError(
            f"volumes must have shape ({n_bins},), but got {measures.shape}."
        )
    if not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise ValueError(
            "volumes must be finite and strictly positive (per-bin cell volume)."
        )
    return measures
