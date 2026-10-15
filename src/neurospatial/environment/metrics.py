"""Private environment metrics using owned geometry and narrow consumers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.environment._protocols import MetricsConsumer
from neurospatial.layout._binning import Binning
from neurospatial.layout._locators import (
    HexLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
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


def _cartesian_boundary_bins(env: MetricsConsumer) -> NDArray[np.intp]:
    """Boundary IDs from retained logical indexing, including transformed subsets."""
    grid_shape, active_mask = env.grid_shape, env.active_mask
    locator = env._binning.locator
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
        graph=env.connectivity,
        grid_shape=grid_shape,
        active_mask=active_mask,
        layout_kind=env._layout_type_used,
    ).astype(np.intp, copy=False)


def _boundary_bins(env: MetricsConsumer) -> NDArray[np.intp]:
    """Implement Environment.boundary_bins using explicit metrics inputs."""
    cached = env._boundary_bins_cached
    return cast("NDArray[np.intp]", cached.copy())


def _linearization_properties(env: MetricsConsumer) -> dict[str, Any]:
    """Implement Environment.linearization_properties using explicit metrics inputs."""
    return cast(
        "dict[str, Any]",
        _copy_export(env._linearization_properties_cached),
    )


def _linearization_properties_cached(env: MetricsConsumer) -> dict[str, Any]:
    """Implement Environment._linearization_properties_cached using explicit metrics inputs."""
    if not env._is_linearized_track_env:
        return {"is_linearized_track": False}

    # The owned public value never exposed mutable builder-only fields.
    return {"is_linearized_track": True}


def _get_bin_attributes(env: MetricsConsumer) -> pd.DataFrame:
    """Implement Environment.get_bin_attributes using explicit metrics inputs."""
    return _copy_attributes(env._bin_attributes_cached)


def _bin_attributes_cached(env: MetricsConsumer) -> pd.DataFrame:
    """Implement Environment._bin_attributes_cached using explicit metrics inputs."""
    node_attrs = []
    for node_id in env.connectivity.nodes():
        attrs = env.connectivity.nodes[node_id].copy()
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


def _get_edge_attributes(env: MetricsConsumer) -> pd.DataFrame:
    """Implement Environment.get_edge_attributes using explicit metrics inputs."""
    return _copy_attributes(env._edge_attributes_cached)


def _edge_attributes_cached(env: MetricsConsumer) -> pd.DataFrame:
    """Implement Environment._edge_attributes_cached using explicit metrics inputs."""
    edge_attrs = []
    centers = env.bin_centers
    graph_orientation = {
        frozenset((source, target)): (source, target)
        for source, target in env.connectivity.edges()
    }
    for source, target in env._binning.bins.pairs:
        attrs = env.connectivity.edges[source, target].copy()
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

    from neurospatial.layout._composition import composition_view

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
    from neurospatial.layout._composition import bridge_values

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
