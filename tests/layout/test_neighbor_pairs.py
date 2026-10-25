"""Neighbor pairs are ordered by one rule shared by every layout engine."""

import networkx as nx
import numpy as np
import pytest

from neurospatial.layout.helpers.neighbor_pairs import (
    euclidean_pair_distances,
    lattice_neighbor_pairs,
    order_neighbor_pairs,
)


@pytest.mark.parametrize("seed", range(5))
def test_order_matches_networkx_edge_iteration(seed):
    """The rule reproduces how a graph built from the proposals lists its edges."""
    rng = np.random.default_rng(seed)
    n_bins = 12
    first = rng.integers(0, n_bins, 60)
    second = rng.integers(0, n_bins, 60)
    distinct = first != second
    first, second = first[distinct], second[distinct]
    distances = rng.uniform(0.5, 2.0, len(first))

    graph = nx.Graph()
    graph.add_nodes_from(range(n_bins))
    seen = set()
    for u, v, distance in zip(first, second, distances, strict=True):
        if frozenset((u, v)) not in seen:
            seen.add(frozenset((u, v)))
            graph.add_edge(int(u), int(v), distance=float(distance))

    pairs, pair_distances = order_neighbor_pairs(first, second, distances)

    assert pairs.dtype == np.intp
    assert pairs.tolist() == [list(edge) for edge in graph.edges()]
    np.testing.assert_array_equal(
        pair_distances, [d for _, _, d in graph.edges(data="distance")]
    )


def test_order_keeps_first_distance_of_a_repeated_pair():
    pairs, distances = order_neighbor_pairs(
        np.array([1, 0]), np.array([0, 1]), np.array([2.0, 5.0])
    )
    assert pairs.tolist() == [[0, 1]]
    assert distances.tolist() == [2.0]


def test_order_of_no_proposals_is_empty():
    pairs, distances = order_neighbor_pairs(
        np.array([], dtype=np.intp), np.array([], dtype=np.intp), np.array([])
    )
    assert pairs.shape == (0, 2)
    assert distances.shape == (0,)


def test_lattice_pairs_are_already_in_authoritative_order():
    """Proposing each pair once from its lower bin needs no reordering."""
    mask = np.random.default_rng(1).random((6, 7)) < 0.75
    active = np.flatnonzero(mask)
    steps = np.array(
        [[[-1, 1]], [[0, 1]], [[1, -1]], [[1, 0]], [[1, 1]]], dtype=np.intp
    )

    pairs = lattice_neighbor_pairs(mask, steps)

    reordered, _ = order_neighbor_pairs(pairs[:, 1], pairs[:, 0], np.zeros(len(pairs)))
    np.testing.assert_array_equal(reordered, pairs)
    cells = np.column_stack(np.unravel_index(active, mask.shape))
    assert np.all(np.abs(cells[pairs[:, 1]] - cells[pairs[:, 0]]).max(axis=1) == 1)


def test_euclidean_pair_distances():
    centers = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    pairs = np.array([[0, 1], [0, 2]])
    np.testing.assert_array_equal(euclidean_pair_distances(centers, pairs), [5, 3])
