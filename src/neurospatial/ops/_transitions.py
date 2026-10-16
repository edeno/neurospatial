"""Transition-matrix estimators on the bins of a navigation graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse
from numpy.typing import NDArray


def _row_normalize(
    matrix: scipy.sparse.csr_matrix, n_bins: int
) -> scipy.sparse.csr_matrix:
    """Scale each row to sum to one; rows without entries stay zero."""
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    nonzero_rows = row_sums > 0
    inv_row_sums = np.zeros(n_bins)
    inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    normalizer = scipy.sparse.diags(inv_row_sums, format="csr")
    return normalizer @ matrix


def count_transitions(
    source_bins: NDArray[np.int32],
    target_bins: NDArray[np.int32],
    n_bins: int,
    *,
    graph: nx.Graph | None,
    normalize: bool,
) -> scipy.sparse.csr_matrix:
    """Count observed steps between bins.

    Parameters
    ----------
    source_bins, target_bins : ndarray of int32, shape (n_steps,)
        Bin of each step's start and end, all in ``[0, n_bins)``.
    n_bins : int
        Number of bins.
    graph : networkx.Graph or None
        When given, only steps between graph neighbors, or within one bin,
        are counted; other jumps (such as tracking errors) are dropped.
    normalize : bool
        Return row-stochastic probabilities instead of counts.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        ``T[i, j]`` is the number of ``i -> j`` steps, or with ``normalize``
        ``P(next=j | current=i)``; rows without steps sum to zero.
    """
    if graph is not None:
        adjacency_set = set()
        for u, v in graph.edges():
            adjacency_set.add((u, v))
            adjacency_set.add((v, u))
        for node in graph.nodes():
            adjacency_set.add((node, node))
        is_adjacent = np.array(
            [
                (src, tgt) in adjacency_set
                for src, tgt in zip(source_bins, target_bins, strict=True)
            ],
            dtype=bool,
        )
        source_bins = source_bins[is_adjacent]
        target_bins = target_bins[is_adjacent]

    transition_counts = np.ones(len(source_bins), dtype=float)
    transition_matrix = scipy.sparse.coo_matrix(
        (transition_counts, (source_bins, target_bins)),
        shape=(n_bins, n_bins),
        dtype=float,
    ).tocsr()
    # Repeated steps between the same bins add up.
    transition_matrix.sum_duplicates()
    if normalize:
        transition_matrix = _row_normalize(transition_matrix, n_bins)
    return transition_matrix


def random_walk_transitions(
    graph: nx.Graph, *, normalize: bool
) -> scipy.sparse.csr_matrix:
    """Return the transitions of a walk that moves to a uniform random neighbor.

    Parameters
    ----------
    graph : networkx.Graph
        Navigation graph whose nodes are the bins ``0 .. n_bins - 1``.
    normalize : bool
        Return ``1 / degree(i)`` for each neighbor instead of the adjacency.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Row-stochastic transitions, or the adjacency matrix when not
        normalized. Isolated bins have zero rows.
    """
    n_bins = graph.number_of_nodes()
    adjacency = nx.adjacency_matrix(graph, nodelist=range(n_bins))
    transition_matrix = adjacency.astype(float).tocsr()
    if normalize:
        transition_matrix = _row_normalize(transition_matrix, n_bins)
    return transition_matrix


def diffusion_transitions(
    transition_kernel: NDArray[np.float64] | scipy.sparse.spmatrix,
) -> scipy.sparse.csr_matrix:
    """Return the row-stochastic heat-diffusion transitions.

    The mass-conserving heat kernel ``Hᵀ`` (from
    ``compute_kernel(mode="transition")``) is column-stochastic, so the
    transition matrix ``T[i, j] = P(next=j | current=i)`` is its transpose.
    On a non-uniform mass matrix (polar, mesh) ``Hᵀ != H``, so this
    transposes instead of assuming symmetry.

    Parameters
    ----------
    transition_kernel : ndarray or sparse matrix, shape (n_bins, n_bins)
        Column-stochastic heat kernel.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Row-stochastic transitions.
    """
    kernel = transition_kernel.T
    if not scipy.sparse.issparse(kernel):
        kernel = scipy.sparse.csr_matrix(kernel)
    return kernel
