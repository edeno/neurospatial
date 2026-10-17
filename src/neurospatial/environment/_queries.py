"""Private graph and spatial query implementations for Environment entry points."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._targets import validate_bin_indices
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
    TrackLinearization,
    require_cartesian,
)
from neurospatial.ops.distance import distance_field


def _track_linearization(
    binning: Binning, operation: str, alternative: str
) -> TrackLinearization:
    """Return the track linearization, or refuse ``operation`` off a track."""
    linearization = binning.linearization
    if linearization is None:
        raise TypeError(
            f"{operation} is only available for 1D environments (GraphLayout). "
            f"This environment is {binning.bins.n_dims}D. {alternative}"
        )
    return linearization


def _to_linear(binning: Binning, positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project positions onto a track's centerline, as linear coordinates."""
    linearization = _track_linearization(
        binning,
        "to_linear()",
        "Use bin_at() to map positions to bins for N-D environments.",
    )
    return linearization.to_linear(positions)


def _linear_to_nd(
    binning: Binning, linear_position: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map linear track coordinates back to positions on the centerline."""
    linearization = _track_linearization(
        binning,
        "linear_to_nd()",
        "Use bin_center_of() to get bin centers for N-D environments.",
    )
    return linearization.linear_to_nd(linear_position)


def _bin_center_of(
    bin_centers: NDArray[np.float64],
    bin_indices: int | Sequence[int] | NDArray[np.integer],
    coordinates: CartesianSystem | EgocentricPolarSystem,
) -> NDArray[np.float64]:
    """Return the centers of validated active-bin indices."""
    indices = validate_bin_indices(
        bin_indices,
        n_bins=len(bin_centers),
        coordinates=coordinates,
        operation="bin_center_of()",
        takes_points=False,
    )
    return np.asarray(bin_centers[indices.reshape(np.shape(bin_indices))])


def _path_between(graph: nx.Graph, source_idx: int, target_idx: int) -> list[int]:
    """Shortest distance-weighted path between two resolved active bins."""
    if source_idx == target_idx:
        return [source_idx]

    try:
        path = nx.shortest_path(
            graph,
            source=source_idx,
            target=target_idx,
            weight="distance",
        )
        return list(path)
    except nx.NetworkXNoPath:
        warnings.warn(
            f"No path found between active bin {source_idx} and {target_idx}.",
            category=UserWarning,
            stacklevel=_external_stacklevel(),
        )
        return []
    except nx.NodeNotFound as e:
        # Re-raise if the user provides an invalid node index for active bins
        raise nx.NodeNotFound(
            f"Node not found in connectivity graph: {e}. "
            "Ensure source/target indices are valid active bin indices.",
        ) from e


def _distance_to(
    targets: NDArray[np.intp],
    *,
    metric: Literal["euclidean", "geodesic"],
    bin_centers: NDArray[np.float64],
    graph: nx.Graph,
    coordinates: CartesianSystem | EgocentricPolarSystem,
) -> NDArray[np.float64]:
    """Distance from every bin to the nearest of the resolved target bins."""
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")
    # Compute distances based on metric
    if metric == "euclidean":
        require_cartesian(
            coordinates,
            "Environment.distance_to(metric='euclidean')",
            why="The straight-line distance between bin centers treats them as "
            "Cartesian positions.",
            fix="use metric='geodesic', which follows the connectivity graph and "
            "its physical edge lengths",
            example="env.distance_to(bins=[0], metric='geodesic')",
        )
        # Euclidean distance: straight-line distance to nearest target
        # Vectorized implementation using broadcasting for performance
        target_positions = bin_centers[targets]  # (n_targets, n_dims)

        # Broadcasting: (n_bins, 1, n_dims) - (1, n_targets, n_dims) -> (n_bins, n_targets)
        diffs = bin_centers[:, np.newaxis, :] - target_positions[np.newaxis, :, :]
        dists_to_targets = np.linalg.norm(diffs, axis=2)  # (n_bins, n_targets)
        distances_result: NDArray[np.float64] = np.min(
            dists_to_targets, axis=1
        )  # (n_bins,)

    else:  # metric == "geodesic"
        # Geodesic distance: graph-based shortest path
        distances_result = np.asarray(
            distance_field(graph, sources=targets.tolist()),
            dtype=np.float64,
        )

    return distances_result


def _reachable_from(
    graph: nx.Graph,
    source_bin: int,
    *,
    radius: int | float | None,
    metric: Literal["hops", "geodesic"],
) -> NDArray[np.bool_]:
    """Bins reachable from a resolved active bin, optionally within a radius."""
    if radius is not None and radius < 0:
        raise ValueError(f"radius must be non-negative or None. Got radius={radius}")

    if metric not in ("hops", "geodesic"):
        raise ValueError(f"metric must be 'hops' or 'geodesic'. Got metric='{metric}'")

    # Initialize result mask
    reachable = np.zeros(graph.number_of_nodes(), dtype=bool)

    # Case 1: No radius limit - find entire connected component
    if radius is None:
        # Directly fetch the component containing source_bin instead of
        # scanning every component in Python.
        try:
            component_nodes = nx.node_connected_component(graph, source_bin)
        except KeyError:
            # source_bin is isolated / not in the graph: only itself.
            reachable[source_bin] = True
            return reachable
        reachable[list(component_nodes)] = True
        return reachable

    # Case 2: Radius-limited search
    if metric == "hops":
        # Breadth-first search to specified depth
        # Use NetworkX's single_source_shortest_path_length with cutoff
        distances = nx.single_source_shortest_path_length(
            graph, source_bin, cutoff=int(radius)
        )
        # Mark all nodes within radius as reachable
        for node in distances:
            reachable[node] = True

    else:  # metric == 'geodesic'
        # Dijkstra's algorithm with distance cutoff
        # Use NetworkX's single_source_dijkstra_path_length
        try:
            distances = nx.single_source_dijkstra_path_length(
                graph, source_bin, cutoff=radius, weight="distance"
            )
            # Mark all nodes within radius as reachable
            for node in distances:
                reachable[node] = True
        except nx.NetworkXError:
            # If source_bin has no edges, only mark itself as reachable
            reachable[source_bin] = True

    return reachable


def _components(graph: nx.Graph, *, largest_only: bool) -> list[NDArray[np.intp]]:
    """Connected components of the graph as sorted bin arrays, largest first."""
    component_sets = nx.connected_components(graph)

    # Convert sets to arrays and sort by size (largest first)
    components = [np.asarray(sorted(comp), dtype=np.intp) for comp in component_sets]
    components.sort(key=len, reverse=True)

    # Return only largest if requested
    if largest_only:
        return components[:1]

    return components


def _rings(graph: nx.Graph, center_bin: int, *, hops: int) -> list[NDArray[np.intp]]:
    """Bins grouped by hop distance from ``center_bin``, up to ``hops``."""
    # Type validation for hops
    if not isinstance(hops, (int, np.integer)):
        raise TypeError(f"hops must be an integer, got {type(hops).__name__}")

    # Validate hops is non-negative
    if hops < 0:
        raise ValueError(f"hops must be non-negative (>= 0), got {hops}")

    # Perform BFS to get shortest path lengths
    try:
        # nx.single_source_shortest_path_length returns dict: {node: distance}
        distances = nx.single_source_shortest_path_length(
            graph, center_bin, cutoff=hops
        )
    except nx.NetworkXError:
        # If center_bin has no edges (isolated), only ring 0 exists
        distances = {center_bin: 0}

    # Organize bins into rings by hop distance
    # Note: cutoff parameter already ensures dist <= hops, so all nodes are valid
    rings_lists: list[list[int]] = [[] for _ in range(hops + 1)]
    for node, dist in distances.items():
        rings_lists[dist].append(node)

    # Convert to numpy arrays
    rings_arrays: list[NDArray[np.intp]] = [
        np.array(ring, dtype=np.intp) for ring in rings_lists
    ]

    return rings_arrays
