"""Neighbor pairs of layout bins, computed as arrays in authoritative order.

Every built-in layout engine lists its neighboring bins as an integer pair
array during ``build()``. The pair order is the order of ``env.edge_pairs``,
which indexes edge fields in gradients and divergence, so each engine
produces it by one rule: pairs are proposed bin by bin, each unordered pair
is kept once where it is first proposed, it is oriented from the lower to
the higher bin ID, and pairs are grouped by their lower bin while keeping the
proposal order inside each group (:func:`order_neighbor_pairs`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def order_neighbor_pairs(
    first: NDArray[np.intp],
    second: NDArray[np.intp],
    distances: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Order proposed neighbor pairs by the authoritative pair rule.

    Keeps the first proposal of each unordered pair, orients it from the
    lower to the higher bin ID, then stably sorts by the lower bin. This is
    the order in which a ``networkx.Graph`` whose nodes were added as
    ``0..n_bins - 1`` iterates edges inserted in proposal order.

    Parameters
    ----------
    first, second : ndarray of int, shape (n_proposed,)
        Bin IDs of each proposed pair, in proposal order.
    distances : ndarray of float64, shape (n_proposed,)
        Navigation distance of each proposed pair.

    Returns
    -------
    pairs : ndarray of intp, shape (n_pairs, 2)
        Distinct pairs, ``pairs[:, 0] <= pairs[:, 1]``, in authoritative order.
    pair_distances : ndarray of float64, shape (n_pairs,)
        Distance of each pair, taken from its first proposal.

    Examples
    --------
    >>> pairs, distances = order_neighbor_pairs(
    ...     np.array([2, 0, 1, 0]),
    ...     np.array([0, 1, 0, 2]),
    ...     np.array([3.0, 1.0, 1.0, 3.0]),
    ... )
    >>> pairs.tolist()
    [[0, 2], [0, 1]]
    >>> distances.tolist()
    [3.0, 1.0]
    """
    low = np.minimum(first, second).astype(np.intp, copy=False)
    high = np.maximum(first, second).astype(np.intp, copy=False)
    distances = np.asarray(distances, dtype=np.float64)
    if len(low):
        _, kept = np.unique(np.column_stack((low, high)), axis=0, return_index=True)
        kept.sort()
        low, high, distances = low[kept], high[kept], distances[kept]
    order = np.argsort(low, kind="stable")
    return np.column_stack((low[order], high[order])), distances[order]


def lattice_neighbor_pairs(
    active_flat: NDArray[np.intp],
    grid_shape: tuple[int, ...],
    offsets: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Pair each active lattice cell with its active neighbors at given offsets.

    Bins are the active cells numbered in ``active_flat`` order. Each bin
    proposes the cells at ``offsets`` in order; a proposal is kept when the
    cell lies inside the lattice, is active and has a higher bin ID, so every
    pair is proposed once, by its lower bin, and the result is already in
    authoritative order.

    Parameters
    ----------
    active_flat : ndarray of int, shape (n_bins,)
        Increasing row-major flat indices of the active cells.
    grid_shape : tuple of int
        Shape of the full lattice, ``(n_0, ..., n_{d-1})``.
    offsets : ndarray of int, shape (n_offsets, n_bins or 1, n_dims)
        Index steps from each bin to its candidate neighbors, in proposal
        order; the middle axis broadcasts when every bin uses the same steps.

    Returns
    -------
    ndarray of intp, shape (n_pairs, 2)
        Pairs ``(lower, higher)`` grouped by lower bin, in offset order.

    Examples
    --------
    >>> active = np.array([0, 1, 3])  # 2 x 2 lattice without cell 2
    >>> steps = np.array([[[0, 1]], [[1, 0]]])  # right, then down
    >>> lattice_neighbor_pairs(active, (2, 2), steps).tolist()
    [[0, 1], [1, 2]]
    """
    active_flat = np.asarray(active_flat, dtype=np.intp)
    n_bins = len(active_flat)
    shape = np.asarray(grid_shape, dtype=np.intp)
    bin_of_cell = np.full(int(np.prod(shape)), -1, dtype=np.intp)
    bin_of_cell[active_flat] = np.arange(n_bins, dtype=np.intp)
    cells = np.stack(np.unravel_index(active_flat, grid_shape), axis=-1)
    bins = np.arange(n_bins, dtype=np.intp)
    first: list[NDArray[np.intp]] = []
    second: list[NDArray[np.intp]] = []
    for step in offsets:
        neighbor = cells + step
        inside = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
        flat = np.ravel_multi_index(tuple(neighbor[inside].T), grid_shape)
        other = np.full(n_bins, -1, dtype=np.intp)
        other[inside] = bin_of_cell[flat]
        keep = other > bins
        first.append(bins[keep])
        second.append(other[keep])
    if not first:
        return np.empty((0, 2), dtype=np.intp)
    lower = np.concatenate(first)
    # Proposals are concatenated offset by offset; a stable sort by the lower
    # bin restores bin-by-bin proposal order.
    order = np.argsort(lower, kind="stable")
    return np.column_stack((lower[order], np.concatenate(second)[order]))


def euclidean_pair_distances(
    centers: NDArray[np.float64], pairs: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Return the Euclidean length between the centers of each pair.

    Parameters
    ----------
    centers : ndarray of float64, shape (n_bins, n_dims)
        Bin centers.
    pairs : ndarray of int, shape (n_pairs, 2)
        Neighbor pairs.

    Returns
    -------
    ndarray of float64, shape (n_pairs,)
        ``|centers[pairs[:, 1]] - centers[pairs[:, 0]]|``.
    """
    lengths: NDArray[np.float64] = np.linalg.norm(
        centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1
    )
    return lengths
