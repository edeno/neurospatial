"""Routes that lap trajectories follow between neighboring bins.

A lap visits a chain of neighboring bins. The straight line between two
neighboring bin centers on a track cuts the corner where two track segments
meet and leaves the track. On a track the route instead runs along the
centerline: two bins on one segment are joined by that straight segment, and
bins on two segments are joined through the track node the segments share.
Other bins are joined by straight lines between their centers.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.environment import Environment
from neurospatial.geometry import track_junctions


def lap_route(env: Environment, path: Sequence[int]) -> NDArray[np.float64]:
    """Return the waypoints of a lap along a chain of neighboring bins.

    Parameters
    ----------
    env : Environment
        The environment the bins belong to.
    path : sequence of int
        Bin indices; consecutive bins are neighbors or the same bin.

    Returns
    -------
    ndarray of float64, shape (n_waypoints, n_dims)
        Starts at the first bin's center and ends at the last bin's center.
        The lap runs in straight lines between consecutive waypoints.
    """
    bins = np.asarray(path, dtype=np.intp)
    bins = bins[np.r_[True, bins[1:] != bins[:-1]]]
    centers = env.bin_centers
    if len(bins) == 1:
        return centers[bins]
    steps = np.column_stack([bins[:-1], bins[1:]])
    if env.binning.linearization is not None:
        return _track_route(env, steps)
    return centers[bins]


def _track_route(env: Environment, steps: NDArray[np.intp]) -> NDArray[np.float64]:
    """Waypoints along a track's centerline, through the junction nodes."""
    linearization = env.binning.linearization
    assert linearization is not None
    segment_of = nx.get_node_attributes(env.connectivity, "source_edge_id")
    segments = np.array([segment_of[i] for i in range(env.n_bins)])
    # NaN where both bins lie on one straight segment: no turn between them.
    junctions = track_junctions(steps, segments, linearization.graph, env.n_dims)
    centers = env.bin_centers
    waypoints = [centers[steps[0, 0]]]
    for (_a, b), junction in zip(steps, junctions, strict=True):
        if np.all(np.isfinite(junction)):
            waypoints.append(junction)
        waypoints.append(centers[b])
    return np.asarray(waypoints, dtype=np.float64)
