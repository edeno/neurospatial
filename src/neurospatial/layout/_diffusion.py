"""Internal diffusion data and optional geometry-provider contracts.

This foundation module knows no Environment or solver. Cell measures and
conductances suffice for diffusion; finite-volume faces are a separate
capability used by gradient and divergence. Geometry is a snapshot in active
bin order, constructed with ``from_weights`` or ``from_graph``.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class DiffusionGeometry(NamedTuple):
    """Coordinate-independent, validated diffusion data in active-bin order.

    ``W`` contains symmetric nonnegative conductances; ``volumes`` contains
    positive cell measures (lengths, areas, or volumes). Components are derived
    from positive conductances, not from a layout's navigation graph. Factory
    methods copy their inputs and make the array buffers read-only, so cached
    operators cannot change through aliases to layout data.

    Environment owns the snapshot's lifetime and invalidates it together with
    its eigencaches when geometry changes. No eigensolver cache lives here.
    """

    W: sparse.csr_matrix
    volumes: NDArray[np.float64]
    n_components: int
    labels: NDArray[np.int_]

    @classmethod
    def from_weights(
        cls, weights: sparse.spmatrix, volumes: NDArray[np.float64]
    ) -> DiffusionGeometry:
        """Validate and snapshot conductances and cell measures.

        Parameters
        ----------
        weights : scipy.sparse matrix, shape (n_bins, n_bins)
            Symmetric, finite, nonnegative conductances.
        volumes : ndarray, shape (n_bins,)
            Finite, strictly positive cell measures in the same bin order.

        Returns
        -------
        DiffusionGeometry
            Independent snapshot including its diffusion components.

        Raises
        ------
        ValueError
            If weights or measures violate the diffusion contract.
        """
        if (
            not sparse.issparse(weights)
            or weights.ndim != 2
            or weights.shape[0] != weights.shape[1]
        ):
            raise ValueError("diffusion weights must be a square sparse matrix.")
        if np.issubdtype(weights.dtype, np.complexfloating):
            raise ValueError("diffusion weights must be real.")
        matrix = sparse.csr_matrix(weights, dtype=np.float64, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
            raise ValueError("diffusion weights must be finite and nonnegative.")
        if (matrix != matrix.T).nnz:
            raise ValueError("diffusion weights must be symmetric.")
        measures = _validate_volumes(volumes, matrix.shape[0]).copy()
        n_components, labels = _components_from_W(matrix)
        for array in (matrix.data, matrix.indices, matrix.indptr, measures, labels):
            array.setflags(write=False)
        return cls(matrix, measures, n_components, labels)

    @classmethod
    def from_graph(
        cls, graph: nx.Graph, volumes: NDArray[np.float64]
    ) -> DiffusionGeometry:
        """Build a snapshot from finite-volume face measures and distances.

        Parameters
        ----------
        graph : networkx.Graph
            Undirected simple graph with integer nodes ``0..n_bins-1``.
            Each edge supplies finite ``A >= 0`` and ``distance > 0``.
            Zero-area edges do not connect diffusion components.
        volumes : ndarray, shape (n_bins,)
            Positive cell measures in node-index order.

        Returns
        -------
        DiffusionGeometry
            Validated snapshot with ``W[i, j] = A[i, j] / distance[i, j]``.
        """
        n_bins = graph.number_of_nodes()
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError(
                "finite-volume geometry requires an undirected simple graph."
            )
        nodes = list(graph.nodes)
        if not all(
            isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in nodes
        ) or {int(x) for x in nodes} != set(range(n_bins)):
            raise ValueError(
                "graph nodes must be contiguous integer labels 0..n-1 (no float/bool)."
            )
        measures = _validate_volumes(volumes, n_bins)
        return cls.from_weights(_assemble_W(graph, n_bins), measures)


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


@runtime_checkable
class DiffusionGeometryProvider(Protocol):
    """Optional layout capability; requires no coordinate representation."""

    def diffusion_geometry(self) -> DiffusionGeometry:
        """Return validated diffusion data in active-bin order."""
        ...


@runtime_checkable
class FiniteVolumeGeometryProvider(Protocol):
    """Additional capability for edge gradients and flux divergence."""

    def finite_volume_geometry(self) -> tuple[nx.Graph, NDArray[np.float64]]:
        """Return a graph copy with face measures and distances, plus cell measures."""
        ...


class _FiniteVolumeDiffusion:
    """Assemble diffusion data for providers with finite-volume faces."""

    def finite_volume_geometry(self) -> tuple[nx.Graph, NDArray[np.float64]]:
        """Return the provider's face geometry in active-bin order."""
        raise NotImplementedError("This provider has no finite-volume cell geometry.")

    def diffusion_geometry(self) -> DiffusionGeometry:
        """Build validated diffusion data from this provider's cell faces."""
        return DiffusionGeometry.from_graph(*self.finite_volume_geometry())


def _components_from_W(W: sparse.spmatrix) -> tuple[int, NDArray[np.int_]]:
    """Connected components of the diffusion-weight matrix ``W`` (nonzero ``A``).

    Component structure for any component-aware step is derived from ``W``, NOT
    ``env.connectivity``: corner-touching 8-connected bins have ``A = 0`` and so
    are separate diffusion components. Load-bearing for the truncated
    ``env.diffuse`` apply-path (component-local eigenvectors + the W-component
    support gates).

    Parameters
    ----------
    W : scipy.sparse matrix, shape (n_bins, n_bins)
        Finite-volume weight matrix.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : NDArray[np.int_], shape (n_bins,)
        Component label per node.
    """
    n_components, labels = connected_components(W, directed=False)
    return int(n_components), labels


def _assemble_W(graph: nx.Graph, n_bins: int) -> sparse.csr_matrix:
    """Finite-volume weight matrix ``W[i, j] = A[i, j] / d[i, j]`` from a graph.

    ``A`` (shared-face measure) and ``distance`` (center-to-center) are per-edge
    contract fields; a missing or invalid value raises rather than silently
    degrading the operator. An explicit ``A == 0`` means no diffusion across that
    edge (e.g. corner-touching Cartesian bins) and is skipped. This is the single
    W-assembly used by both the dense ``compute_diffusion_kernels`` path and the
    ``env.diffuse`` eigenbasis, so the two operate on an identical ``W``.

    Parameters
    ----------
    graph : nx.Graph
        Nodes are contiguous integers ``0..n_bins-1``; each edge carries ``"A"``
        (finite, >= 0) and ``"distance"`` (finite, > 0).
    n_bins : int
        Number of nodes (matrix dimension).

    Returns
    -------
    W : sparse.csr_matrix, shape (n_bins, n_bins)
        Symmetric finite-volume weight matrix.

    Raises
    ------
    ValueError
        If an edge is missing ``"A"``/``"distance"`` or carries a non-finite /
        negative ``"A"`` or a non-finite / non-positive ``"distance"``.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        if "A" not in data or "distance" not in data:
            raise ValueError(
                f"edge ({u},{v}) is missing 'A' and/or 'distance' attribute."
            )
        area = float(data["A"])
        d = float(data["distance"])
        if not (np.isfinite(area) and area >= 0.0):
            raise ValueError(f"edge ({u},{v}) has invalid face measure A={area}.")
        if not (np.isfinite(d) and d > 0.0):
            raise ValueError(f"edge ({u},{v}) has invalid distance d={d}.")
        if area == 0.0:
            continue  # explicit A == 0 => no diffusion across this edge
        w = area / d
        rows += [int(u), int(v)]
        cols += [int(v), int(u)]
        vals += [w, w]
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_bins, n_bins), dtype=np.float64
    )
