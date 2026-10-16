"""Private environment metrics computed from owned geometry components."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.geometry._binning import Binning
from neurospatial.geometry._composition import bridge_values, composition_view
from neurospatial.geometry._locators import (
    HexLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon import PolygonCells, _boundary_cell_indices
from neurospatial.layout.helpers.utils import find_boundary_nodes


def _copy_export(value: Any) -> Any:
    """Copy nested export values, including immutable mapping snapshots."""
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _copy_export(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_export(item) for item in value)
    if isinstance(value, list):
        return [_copy_export(item) for item in value]
    return copy.deepcopy(value)


def _copy_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    """Detach frame buffers, labels and nested object-column values."""
    result = frame.copy(deep=True)
    result.index = frame.index.copy(deep=True)
    result.columns = frame.columns.copy(deep=True)
    for column in frame.select_dtypes(include="object"):
        result[column] = pd.Series(
            [_copy_export(value) for value in frame[column]],
            index=result.index,
            dtype=object,
        )
    return result


def _cartesian_boundary_bins(
    binning: Binning,
    graph: nx.Graph,
    grid_shape: tuple[int, ...] | None,
    active_mask: NDArray[np.bool_] | None,
) -> NDArray[np.intp]:
    """Boundary IDs from retained logical indexing, including transformed subsets."""
    shapes = binning.shapes
    if isinstance(shapes, PolygonCells):
        return _boundary_cell_indices(shapes.polygons)
    locator = binning.locator
    selections = []
    while isinstance(locator, (SubsetLocator, TransformedLocator)):
        if isinstance(locator, SubsetLocator):
            selections.append(locator.parent_to_child)
        locator = locator.base
    if isinstance(locator, HexLocator):
        grid_shape = locator.centers_shape
        mapping = locator.flat_to_active.copy()
    elif isinstance(locator, RectilinearLocator):
        grid_shape = locator.shape
        mask = locator.active_mask
        mask = np.ones(grid_shape, dtype=bool) if mask is None else mask
        mapping = np.full(mask.size, -1, dtype=np.intp)
        mapping[mask.ravel()] = np.arange(np.count_nonzero(mask))
    else:
        mapping = None
    if mapping is not None:
        # Cell adjacency survives coordinate transforms. Compose subset
        # maps before deriving support, without indexing the -1 sentinel.
        for selection in reversed(selections):
            inside = mapping >= 0
            mapping[inside] = selection[mapping[inside]]
        active_mask = (mapping >= 0).reshape(grid_shape)
    return find_boundary_nodes(
        graph=graph,
        grid_shape=grid_shape,
        active_mask=active_mask,
    ).astype(np.intp, copy=False)


def _bin_attributes(graph: nx.Graph) -> pd.DataFrame:
    """Node attributes of the graph, one row per bin, positions split by axis."""
    node_attrs = []
    for node_id in graph.nodes():
        attrs = graph.nodes[node_id].copy()
        attrs["node_id"] = node_id
        node_attrs.append(attrs)

    df = pd.DataFrame(node_attrs)

    # Expand position tuple into separate columns
    if "pos" in df.columns:
        pos_array = np.array(df["pos"].tolist())
        for dim in range(pos_array.shape[1]):
            df[f"pos_dim{dim}"] = pos_array[:, dim]
        df = df.drop(columns=["pos"])

    return df.set_index("node_id")


def _edge_attributes(graph: nx.Graph, binning: Binning) -> pd.DataFrame:
    """Edge attributes in bin-pair order, vectors oriented from source to target."""
    edge_attrs = []
    centers = binning.bins.centers
    graph_orientation = {
        frozenset((source, target)): (source, target)
        for source, target in graph.edges()
    }
    for source, target in binning.bins.pairs:
        attrs = graph.edges[source, target].copy()
        if graph_orientation[frozenset((source, target))] != (source, target):
            # Undirected graph insertion order does not establish the
            # orientation of retained attributes. Derive it from the pair
            # instead of blindly negating a possibly already aligned vector.
            vector = centers[target] - centers[source]
            vector[vector == 0.0] = 0.0
            attrs["vector"] = tuple(vector)
            if "angle_2d" in attrs:
                attrs["angle_2d"] = float(np.arctan2(vector[1], vector[0]))
        attrs["source"] = source
        attrs["target"] = target
        edge_attrs.append(attrs)

    df = pd.DataFrame(edge_attrs)

    # Expand vector tuple into separate columns
    if "vector" in df.columns and len(df) > 0:
        vector_array = np.array(df["vector"].tolist())
        for dim in range(vector_array.shape[1]):
            df[f"vector_dim{dim}"] = vector_array[:, dim]
        # Keep original vector column for backward compatibility

    return df


def _get_bridge_attributes(binning: Binning) -> pd.DataFrame:
    """Export retained navigation bridge data as an independent table."""
    columns = [
        "source_component",
        "source_local_bin",
        "target_component",
        "target_local_bin",
        "source",
        "target",
        "distance",
        "conductance",
    ]
    result = composition_view(binning.locator)
    if result is None:
        return pd.DataFrame(columns=columns)
    root, indices, _scale = result

    by_pair = bridge_values(binning)
    offsets = root.offsets
    records = []
    for (source, target), distance in zip(
        binning.bins.pairs, binning.bins.pair_distances, strict=True
    ):
        pair = frozenset((int(source), int(target)))
        if pair not in by_pair:
            continue
        owners = np.searchsorted(
            offsets[1:], [indices[source], indices[target]], side="right"
        )
        records.append(
            (
                root.keys[owners[0]],
                int(indices[source] - offsets[owners[0]]),
                root.keys[owners[1]],
                int(indices[target] - offsets[owners[1]]),
                int(source),
                int(target),
                float(distance),
                float(by_pair[pair]),
            )
        )
    return pd.DataFrame(records, columns=columns)
