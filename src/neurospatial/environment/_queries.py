"""Private graph and spatial query implementations for Environment entry points."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._targets import validate_bin_indices
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
    TrackLinearization,
    require_cartesian,
)
from neurospatial.ops._types import DistanceMetric
from neurospatial.ops.distance import distance_field

if TYPE_CHECKING:
    from neurospatial.environment import Environment

_BUILD_A_TRACK = (
    "Environment.linear_track(endpoints=[(0.0, 0.0), (100.0, 0.0)], "
    "bin_size=2.0, track_width=8.0)"
)


def _track_linearization(
    binning: Binning, operation: str, why: str, alternative: str
) -> TrackLinearization:
    """Return the track linearization, or refuse ``operation`` off a track."""
    linearization = binning.linearization
    if linearization is None:
        raise TypeError(
            _format_error(
                f"{operation} needs a linearized track "
                "(env.is_linearized_track is True), but this is a "
                f"{binning.kind} environment, which has no track.",
                why=f"Why: {why}, and only environments built from a track "
                "have a centerline.",
                fix=f"build the track, for example {_BUILD_A_TRACK}, or "
                "Environment.maze(...) or Environment.from_graph(...) for "
                f"other tracks; {alternative}",
            )
        )
    return linearization


def _track_extents(
    env: Environment,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Where each bin starts and ends along a track; None off a track.

    Linearized tracks (``linear_track``, ``from_graph``, ``maze``) measure
    distance along the track, with the spacing between segments left empty;
    1-D environments use their own coordinate. Both arrays have shape
    ``(n_bins,)`` and follow bin numbering, not track order.
    """
    linearization = env._binning.linearization
    if linearization is not None:
        edges = np.asarray(linearization.grid_edges[0], dtype=np.float64)
        active = linearization.active_mask
        return edges[:-1][active], edges[1:][active]
    if env.n_dims == 1:
        centers, half = env.bin_centers[:, 0], env.bin_sizes / 2
        return centers - half, centers + half
    return None


def _require_track_extents(
    env: Environment, operation: str, *, alternative: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return :func:`_track_extents`, or refuse ``operation`` off a track.

    ``alternative`` names what works on other environments.
    """
    extents = _track_extents(env)
    if extents is None:
        raise ValueError(
            _format_error(
                f"{operation} needs a position along a track, but this is a "
                f"{env.n_dims}-D {env._binning.kind} environment that is not "
                "a linearized track.",
                why="it orders bins by their distance along one axis; bin "
                "numbers have no spatial order, and a 2-D or 3-D arena has no "
                "single axis to order them by",
                fix=f"use a 1-D environment or build the track, for example "
                f"{_BUILD_A_TRACK}, or Environment.maze(...) or "
                f"Environment.from_graph(...) for other tracks; {alternative}",
            )
        )
    return extents


def _to_linear(binning: Binning, positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project positions onto a track's centerline, as linear coordinates."""
    linearization = _track_linearization(
        binning,
        "to_linear()",
        "it measures how far along a track's centerline each position lies",
        "on this environment, map positions to bins with env.bin_at(positions)",
    )
    return linearization.to_linear(positions)


def _linear_to_nd(
    binning: Binning, linear_position: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map linear track coordinates back to positions on the centerline."""
    linearization = _track_linearization(
        binning,
        "linear_to_nd()",
        "it maps distances along a track's centerline back to positions on "
        "that centerline",
        "on this environment, get bin centers with env.bin_center_of(bins)",
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


def _distance_to(
    targets: NDArray[np.intp],
    *,
    metric: DistanceMetric,
    bin_centers: NDArray[np.float64],
    graph: nx.Graph,
    coordinates: CartesianSystem | EgocentricPolarSystem,
) -> NDArray[np.float64]:
    """Distance from every bin to the nearest of the resolved target bins."""
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")
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
    return distance_field(
        graph, sources=targets, metric=metric, bin_centers=bin_centers
    )


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

    # An isolated bin is its own component and lies within any radius of
    # itself, so none of these searches needs a special case for it.
    if radius is None:
        reached = nx.node_connected_component(graph, source_bin)
    elif metric == "hops":
        reached = nx.single_source_shortest_path_length(
            graph, source_bin, cutoff=int(radius)
        )
    else:
        reached = nx.single_source_dijkstra_path_length(
            graph, source_bin, cutoff=radius, weight="distance"
        )
    reachable[list(reached)] = True
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

    # {node: hop distance}; an isolated center yields only ring 0.
    distances = nx.single_source_shortest_path_length(graph, center_bin, cutoff=hops)

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
