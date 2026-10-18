"""Private environment metrics computed from owned geometry components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from neurospatial.geometry import (
    Arena,
    Binning,
    CoordinateSystem,
    blocking_walls,
    bridge_values,
    composition_view,
)

if TYPE_CHECKING:
    import pandas as pd


def _copy_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    """Detach frame buffers and labels from the cached table.

    Object columns hold tuples of scalars (``graph_from_binning``), which are
    immutable, so their values need no copy.
    """
    result = frame.copy(deep=True)
    result.index = frame.index.copy(deep=True)
    result.columns = frame.columns.copy(deep=True)
    return result


def _bin_attributes(graph: nx.Graph) -> pd.DataFrame:
    """Node attributes of the graph, one row per bin, positions split by axis."""
    import pandas as pd

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


def _edge_attributes(
    graph: nx.Graph, binning: Binning, coordinates: CoordinateSystem
) -> pd.DataFrame:
    """Edge attributes in bin-pair order, vectors oriented from source to target."""
    import pandas as pd

    edge_attrs = []
    places = coordinates.cartesian_embedding(binning.bins.centers)
    graph_orientation = {
        frozenset((source, target)): (source, target)
        for source, target in graph.edges()
    }
    for source, target in binning.connections.pairs:
        attrs = graph.edges[source, target].copy()
        if graph_orientation[frozenset((source, target))] != (source, target):
            # Undirected graph insertion order does not establish the
            # orientation of retained attributes. Derive it from the pair
            # instead of blindly negating a possibly already aligned vector.
            vector = places[target] - places[source]
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


def _blocked_edges(binning: Binning, arena: Arena) -> pd.DataFrame:
    """Closed neighbor pairs with their path length and the wall closing each."""
    import pandas as pd

    links = binning.connections
    closed = ~links.is_open
    pairs = links.all_pairs[closed]
    return pd.DataFrame(
        {
            "source": pairs[:, 0].copy(),
            "target": pairs[:, 1].copy(),
            "distance": links.all_distances[closed].copy(),
            "wall": pd.Series(blocking_walls(binning, arena), dtype=object),
        }
    )


def _get_bridge_attributes(binning: Binning) -> pd.DataFrame:
    """Export retained navigation bridge data as an independent table."""
    import pandas as pd

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
        binning.connections.pairs, binning.connections.pair_distances, strict=True
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
