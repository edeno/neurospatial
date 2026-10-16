"""Project authoritative binning data into standard navigation graph attributes."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout._binning import Binning, Locator, TrackLinearization
from neurospatial.layout._locators import (
    HexLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import ClippedLocator


def _source_grid(locator: Locator) -> tuple[tuple[int, ...], NDArray[np.intp]] | None:
    """Recover logical cell IDs in active-bin order without fitting geometry."""
    if isinstance(locator, ClippedLocator):
        # Clipping creates new cells. Immediate-parent ancestry lives in the
        # separate ClipMapping, rather than in logical grid coordinates.
        count = len(locator.cells.polygons)
        return (count,), np.arange(count, dtype=np.intp)
    if isinstance(locator, (TransformedLocator, SubsetLocator)):
        source = _source_grid(locator.base)
        if source is None:
            return None
        shape, indices = source
        if isinstance(locator, SubsetLocator):
            selected = np.flatnonzero(locator.parent_to_child >= 0)
            selected = selected[np.argsort(locator.parent_to_child[selected])]
            indices = indices[selected]
        return shape, indices
    if isinstance(locator, RectilinearLocator):
        shape = locator.shape
        mask = locator.active_mask
        active = (
            np.arange(np.prod(shape), dtype=np.intp)
            if mask is None
            else np.flatnonzero(mask.ravel())
        )
        mapping = locator.flat_to_active
    elif isinstance(locator, HexLocator):
        shape = locator.centers_shape
        mapping = locator.flat_to_active
        active = np.flatnonzero(mapping >= 0)
    else:
        return None
    if mapping is not None:
        active = active[np.argsort(mapping[active])]
    return shape, active.astype(np.intp, copy=False)


def _graph_from_binning(binning: Binning) -> nx.Graph:
    """Build navigation attributes without changing bin or neighbor-pair order.

    Graph vectors follow graph edge orientation. Edge IDs identify the original
    pair-array rows; source-grid IDs preserve logical indexing when available.
    Derivations preserving existing graph metadata keep that separate from this
    projection of newly constructed bin data.
    """
    graph = nx.Graph()
    for index, center in enumerate(binning.bins.centers):
        graph.add_node(
            index,
            pos=tuple(float(value) for value in center),
            source_grid_flat_index=index,
            original_grid_nd_index=(index,),
        )
    for index, ((a, b), distance) in enumerate(
        zip(binning.bins.pairs, binning.bins.pair_distances, strict=True)
    ):
        # Graph nodes iterate in bin ID order. Keep its vector in that graph
        # orientation; edge_id links it to the explicit pair-array authority.
        u, v = sorted((int(a), int(b)))
        try:
            with np.errstate(over="raise", invalid="raise"):
                vector = binning.bins.centers[v] - binning.bins.centers[u]
        except FloatingPointError as exc:
            raise ValueError(
                "Pair displacement exceeds floating-point range; recenter coordinates."
            ) from exc
        graph.add_edge(
            u,
            v,
            distance=float(distance),
            vector=tuple(float(value) for value in vector),
            edge_id=index,
        )
        if binning.bins.n_dims == 2:
            graph.edges[u, v]["angle_2d"] = float(np.arctan2(vector[1], vector[0]))
    source = _source_grid(binning.locator)
    if source is None and isinstance(binning.linearization, TrackLinearization):
        mask = binning.linearization.active_mask
        source = mask.shape, np.flatnonzero(mask)
    if source is not None:
        shape, indices = source
        for index, flat in enumerate(indices):
            graph.nodes[index].update(
                source_grid_flat_index=int(flat),
                original_grid_nd_index=tuple(
                    int(value) for value in np.unravel_index(flat, shape)
                ),
            )
    return graph
