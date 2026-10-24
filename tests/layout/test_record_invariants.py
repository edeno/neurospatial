"""Geometry records refuse inconsistent states when they are constructed."""

import networkx as nx
import numpy as np
import pytest

from neurospatial.arena import MaskArena
from neurospatial.binning import (
    FramedGrid,
    RectilinearGrid,
    RectilinearLocator,
    TrackLinearization,
)


def _track_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(1.0, 0.0))
    graph.add_edge(0, 1, edge_id=0)
    return graph


GRID_RECORDS = {
    "RectilinearGrid": lambda edges, shape, mask: RectilinearGrid(edges, shape, mask),
    "RectilinearLocator": lambda edges, shape, mask: RectilinearLocator(
        edges, shape, mask, None
    ),
    "MaskArena": lambda edges, shape, mask: MaskArena(edges, mask),
    "TrackLinearization": lambda edges, shape, mask: TrackLinearization(
        _track_graph(), [(0, 1)], 0.0, edges, mask
    ),
}


@pytest.mark.parametrize("record", GRID_RECORDS)
@pytest.mark.parametrize(
    ("edges", "shape", "mask"),
    [
        ((np.array([3.0, 2.0, 1.0]),), (2,), np.ones(2, bool)),  # decreasing
        ((np.array([0.0, np.nan, 2.0]),), (2,), np.ones(2, bool)),  # not finite
        ((np.array([0.0, 1.0, 2.0]),), (2,), np.ones((2, 2), bool)),  # mask shape
        ((np.array([0.0, 1.0, 2.0]),), (2,), np.ones(2, int)),  # not boolean
    ],
    ids=["decreasing", "nan", "mask-shape", "int-mask"],
)
def test_grid_records_refuse_inconsistent_cells(record, edges, shape, mask):
    with pytest.raises(ValueError, match=record):
        GRID_RECORDS[record](edges, shape, mask)


@pytest.mark.parametrize("record", ["RectilinearGrid", "RectilinearLocator"])
def test_grid_records_refuse_a_shape_that_disagrees_with_the_edges(record):
    with pytest.raises(ValueError, match="edges\\[0\\]"):
        GRID_RECORDS[record]((np.array([0.0, 1.0, 2.0]),), (5,), np.ones(5, bool))


def test_grid_records_own_frozen_copies():
    edges, mask = np.array([0.0, 1.0, 2.0]), np.ones(2, bool)
    grid = RectilinearGrid((edges,), (np.int64(2),), mask)
    assert grid.shape == (2,) and type(grid.shape[0]) is int
    assert not grid.edges[0].flags.writeable and not grid.active_mask.flags.writeable
    assert edges.flags.writeable and grid.active_mask is not mask


def test_framed_grid_refuses_a_one_dimensional_source():
    grid = RectilinearGrid((np.array([0.0, 1.0]),), (1,), np.ones(1, bool))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        FramedGrid(grid, np.eye(2))
