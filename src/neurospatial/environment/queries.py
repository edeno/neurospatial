"""Private graph and spatial query implementations for Environment entry points."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import (
    BinIndexOutOfRangeError,
    RegionNotFoundError,
    _format_error,
)
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._protocols import QueryConsumer


def _resolve_point_or_index(
    env: QueryConsumer, point_or_index: int | ArrayLike, *, operation: str
) -> int:
    """Resolve a coordinate point OR a bin index to an active-bin index.

    Integers (Python ``int`` / NumPy integer) are treated as bin indices;
    anything else (a coordinate array, list, or float) is treated as a point in
    the environment's coordinate space and mapped to its bin via ``bin_at``. This
    lets the graph-query methods accept either form -- like the geometric methods
    -- instead of silently conflating a coordinate with a bin index.

    Raises
    ------
    BinIndexOutOfRangeError
        If an integer index is outside ``[0, n_bins)``.
    ValueError
        If a coordinate point falls outside every active bin.
    """
    if isinstance(point_or_index, (int, np.integer)):
        idx = int(point_or_index)
        if not 0 <= idx < env.n_bins:
            raise BinIndexOutOfRangeError(idx, n_bins=env.n_bins)
        return idx
    point = np.asarray(point_or_index, dtype=float)
    if point.shape == (1,) and env.n_dims > 1:
        raise ValueError(
            _format_error(
                f"Got a length-1 array with shape {point.shape} in a {env.n_dims}-D environment. bin_at returns an array, even for one point.",
                why="Why: this array is neither a scalar bin index nor a complete coordinate point.",
                fix=f"unwrap the bin_at result with env.neighbors(int(bin_idx[0])), or pass a complete coordinate point of shape ({env.n_dims},)",
            )
        )
    resolved = np.asarray(
        env._require_cartesian(f"{operation} from a coordinate point").bin_at(
            np.atleast_2d(point)
        )
    ).reshape(-1)
    if resolved.size != 1:
        raise ValueError(
            _format_error(
                f"expected a single coordinate point of shape (n_dims,), but got {resolved.size} points (input shape {point.shape}); pass one point or a single bin index (int).",
                fix="pass one point, e.g. env.neighbors([x, y]), or a single bin index: env.neighbors(int(bin_idx[0]))",
                why="Why: a graph query starts from one active bin.",
            )
        )
    idx = int(resolved[0])
    if idx < 0:
        raise ValueError(
            _format_error(
                f"Point {point.tolist()} is not inside any active bin; pass a point within the environment, or a bin index (int).",
                fix="pass a point inside the environment, or an integer bin index from env.bin_at([point])[0]",
                why="Why: graph queries need a point mapped to an active bin.",
            )
        )
    return idx


def _bin_center_of(
    env: QueryConsumer, bin_indices: int | Sequence[int] | NDArray[np.int_]
) -> NDArray[np.float64]:
    """Implement Environment.bin_center_of using explicit queries inputs."""
    indices = np.asarray(bin_indices, dtype=int)
    out_of_range = (indices < 0) | (indices >= env.n_bins)
    if np.any(out_of_range):
        raise BinIndexOutOfRangeError(
            int(indices[out_of_range].flat[0]), n_bins=env.n_bins
        )
    return np.asarray(env.bin_centers[indices], dtype=np.float64)


def _neighbors(env: QueryConsumer, bin_index: int | ArrayLike) -> list[int]:
    """Implement Environment.neighbors using explicit queries inputs."""
    bin_index = _resolve_point_or_index(env, bin_index, operation="neighbors()")
    return list(env.connectivity.neighbors(bin_index))


def _bin_sizes(env: QueryConsumer) -> NDArray[np.float64]:
    """Implement Environment.bin_sizes using explicit queries inputs."""
    sizes: NDArray[np.float64] = env._binning.bins.sizes.copy()
    sizes.setflags(write=False)
    return sizes


def _path_between(
    env: QueryConsumer,
    source_active_bin_idx: int | ArrayLike,
    target_active_bin_idx: int | ArrayLike,
) -> list[int]:
    """Implement Environment.path_between using explicit queries inputs."""
    graph = env.connectivity

    source_idx = _resolve_point_or_index(
        env, source_active_bin_idx, operation="path_between()"
    )
    target_idx = _resolve_point_or_index(
        env, target_active_bin_idx, operation="path_between()"
    )

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
    env: QueryConsumer,
    targets: Sequence[int] | str,
    *,
    metric: Literal["euclidean", "geodesic"] = "geodesic",
) -> NDArray[np.float64]:
    """Implement Environment.distance_to using explicit queries inputs."""
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")

    # Handle region name targets
    if isinstance(targets, str):
        region_name = targets
        if region_name not in env.regions:
            raise RegionNotFoundError(
                region_name, available=list(env.regions.keys()), argument="targets"
            )

        # Get bins in region via membership
        membership = env.region_membership()
        region_idx = list(env.regions.keys()).index(region_name)
        targets = np.where(membership[:, region_idx])[0].tolist()

        if len(targets) == 0:
            warnings.warn(
                f"Region '{region_name}' contains no bins. All distances will be inf.",
                category=UserWarning,
                stacklevel=_external_stacklevel(),
            )

    # Convert to numpy array for validation
    try:
        target_array = np.asarray(targets, dtype=np.int32)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"targets must be a sequence of integers or a string (region name), "
            f"got {type(targets).__name__}"
        ) from e

    # Validate targets not empty
    if len(target_array) == 0:
        raise ValueError(
            "targets cannot be empty. Provide at least one target bin index "
            "or a region name containing bins."
        )

    # Validate target indices in range
    if np.any(target_array < 0) or np.any(target_array >= env.n_bins):
        invalid = target_array[(target_array < 0) | (target_array >= env.n_bins)]
        raise ValueError(
            f"Target bin indices must be in range [0, {env.n_bins}), "
            f"got invalid indices: {invalid.tolist()}"
        )

    # Compute distances based on metric
    if metric == "euclidean":
        env._require_cartesian("distance_to(metric='euclidean')")
        # Euclidean distance: straight-line distance to nearest target
        # Vectorized implementation using broadcasting for performance
        target_positions = env.bin_centers[target_array]  # (n_targets, n_dims)

        # Broadcasting: (n_bins, 1, n_dims) - (1, n_targets, n_dims) -> (n_bins, n_targets)
        diffs = env.bin_centers[:, np.newaxis, :] - target_positions[np.newaxis, :, :]
        dists_to_targets = np.linalg.norm(diffs, axis=2)  # (n_bins, n_targets)
        distances_result: NDArray[np.float64] = np.min(
            dists_to_targets, axis=1
        )  # (n_bins,)

    else:  # metric == "geodesic"
        # Geodesic distance: graph-based shortest path
        from neurospatial.ops.distance import distance_field

        distances_result = np.asarray(
            distance_field(env.connectivity, sources=target_array.tolist()),
            dtype=np.float64,
        )

    return distances_result


def _reachable_from(
    env: QueryConsumer,
    source_bin: int | ArrayLike,
    *,
    radius: int | float | None = None,
    metric: Literal["hops", "geodesic"] = "hops",
) -> NDArray[np.bool_]:
    """Implement Environment.reachable_from using explicit queries inputs."""
    source_bin = _resolve_point_or_index(env, source_bin, operation="reachable_from()")

    if radius is not None and radius < 0:
        raise ValueError(f"radius must be non-negative or None. Got radius={radius}")

    if metric not in ("hops", "geodesic"):
        raise ValueError(f"metric must be 'hops' or 'geodesic'. Got metric='{metric}'")

    # Initialize result mask
    reachable = np.zeros(env.n_bins, dtype=bool)

    # Case 1: No radius limit - find entire connected component
    if radius is None:
        # Directly fetch the component containing source_bin instead of
        # scanning every component in Python.
        try:
            component_nodes = nx.node_connected_component(env.connectivity, source_bin)
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
            env.connectivity, source_bin, cutoff=int(radius)
        )
        # Mark all nodes within radius as reachable
        for node in distances:
            reachable[node] = True

    else:  # metric == 'geodesic'
        # Dijkstra's algorithm with distance cutoff
        # Use NetworkX's single_source_dijkstra_path_length
        try:
            distances = nx.single_source_dijkstra_path_length(
                env.connectivity, source_bin, cutoff=radius, weight="distance"
            )
            # Mark all nodes within radius as reachable
            for node in distances:
                reachable[node] = True
        except nx.NetworkXError:
            # If source_bin has no edges, only mark itself as reachable
            reachable[source_bin] = True

    return reachable


def _components(
    env: QueryConsumer, *, largest_only: bool = False
) -> list[NDArray[np.int32]]:
    """Implement Environment.components using explicit queries inputs."""
    component_sets = nx.connected_components(env.connectivity)

    # Convert sets to arrays and sort by size (largest first)
    components = [np.asarray(sorted(comp), dtype=np.int32) for comp in component_sets]
    components.sort(key=len, reverse=True)

    # Return only largest if requested
    if largest_only:
        return components[:1]

    return components


def _rings(
    env: QueryConsumer, center_bin: int, *, hops: int
) -> list[NDArray[np.int32]]:
    """Implement Environment.rings using explicit queries inputs."""
    if not isinstance(center_bin, (int, np.integer)):
        raise TypeError(
            f"center_bin must be an integer, got {type(center_bin).__name__}"
        )

    # Range validation for center_bin
    if center_bin < 0 or center_bin >= env.n_bins:
        raise ValueError(
            f"center_bin must be in range [0, {env.n_bins}), got {center_bin}"
        )

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
            env.connectivity, center_bin, cutoff=hops
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
    rings_arrays: list[NDArray[np.int32]] = [
        np.array(ring, dtype=np.int32) for ring in rings_lists
    ]

    return rings_arrays
