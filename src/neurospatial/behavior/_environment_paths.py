"""Shortest path lengths inside an environment, between continuous positions.

The geodesic distance between two positions is the length of the shortest
path that stays inside the environment. Measuring it along the bin graph
(bin center to bin center) quantizes it: a straight oblique run in an open
field reads as a staircase of grid steps, up to ~8% longer on an 8-connected
grid, and positions inside one bin all read the same distance.

:class:`EnvironmentPaths` measures it in continuous space instead:

1. When the straight segment between the two positions stays inside the
   environment, the distance is the segment's Euclidean length. The segment
   is probed at a quarter of the shortest graph edge (a quarter bin on a
   regular grid): every probe must lie in a bin, and consecutive probes in
   the same bin or in two bins joined by an open connection, so a segment
   across a wall (which closes the connection between the bins on its two
   sides) or an outline slit is not straight. It may cut across a missing
   bin's corner by less than a quarter bin.
2. Otherwise the path goes around walls: take the bin-graph shortest path
   between the positions' bins and pull it taut, replacing runs of waypoints
   by straight segments that stay inside the environment (the post-smoothing
   step of any-angle path planning; Botea et al. 2004, Nash et al. 2007).
   The waypoints are bin centers, so the taut path rounds each corner about
   half a bin inside the wall and overestimates the true shortest path by a
   fraction of a bin per corner.

Positions outside the environment have no distance (NaN); positions in
disconnected parts of the environment are ``inf`` apart.

References
----------
.. [1] Botea, A., Müller, M., & Schaeffer, J. (2004). Near optimal
       hierarchical path-finding. J. Game Development 1(1), 7-28.
.. [2] Nash, A., Daniel, K., Koenig, S., & Felner, A. (2007). Theta*:
       Any-angle path planning on grids. Proc. AAAI 22, 1177-1183.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import ragged_arange

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# Probe a segment every quarter of the shortest graph edge (a quarter bin on a
# regular grid): a missing bin is at least one edge wide, so a segment that
# crosses it is caught unless it only clips a corner.
_PROBE_FRACTION = 0.25
# Probe points located per call to ``Environment.bin_at``.
_PROBE_CHUNK = 1_000_000


class EnvironmentPaths:
    """Shortest path lengths inside one environment, sharing graph searches.

    Parameters
    ----------
    env : Environment
        An allocentric Cartesian environment (the caller checks).
    """

    def __init__(self, env: Environment) -> None:
        from neurospatial.ops.distance import _bin_adjacency

        self._env = env
        self._centers = np.asarray(env.bin_centers, dtype=float)
        self._adjacency = _bin_adjacency(
            env.connectivity, weight="distance", operation="geodesic path length"
        )
        # Each bin's open neighbors, padded with -1, for connection tests.
        degree = np.diff(self._adjacency.indptr)
        self._neighbors = np.full(
            (self._centers.shape[0], int(degree.max(initial=0))), -1, np.intp
        )
        self._neighbors[
            np.repeat(np.arange(degree.size), degree), ragged_arange(degree)
        ] = self._adjacency.indices
        lengths = self._adjacency.data[self._adjacency.data > 0]
        self._spacing = (
            _PROBE_FRACTION * float(lengths.min()) if lengths.size else np.inf
        )
        self._trees: dict[int, tuple[NDArray[np.float64], NDArray[np.int32]]] = {}

    def bins(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Bin of each point, -1 outside the environment or for NaN rows."""
        if points.shape[0] == 0:
            return np.zeros(0, dtype=np.intp)
        return np.asarray(self._env.bin_at(points), dtype=np.intp)

    def visible(
        self, starts: NDArray[np.float64], ends: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Whether each straight segment stays inside the environment.

        Parameters
        ----------
        starts, ends : ndarray, shape (n_segments, n_dims)
            Finite endpoints, both inside the environment.

        Returns
        -------
        ndarray of bool, shape (n_segments,)
            True when every probe point along the segment lies in a bin and
            consecutive probes (endpoints included) lie in the same bin or
            in two bins joined by an open connection.
        """
        n_segments = starts.shape[0]
        clear = np.ones(n_segments, dtype=bool)
        if n_segments == 0 or not np.isfinite(self._spacing):
            return clear
        lengths = np.linalg.norm(ends - starts, axis=1)
        # Points at k / (n_interior + 1) of the way, k = 0 .. n_interior + 1.
        n_interior = np.ceil(lengths / self._spacing).astype(np.intp) - 1
        size = np.maximum(n_interior, 0) + 2
        bounds = np.searchsorted(
            np.cumsum(size), np.arange(_PROBE_CHUNK, int(size.sum()), _PROBE_CHUNK)
        )
        for chunk in np.split(np.arange(n_segments), bounds):
            if not chunk.size:
                continue
            counts = size[chunk]
            owner = np.repeat(np.arange(chunk.size), counts)
            fraction = ragged_arange(counts) / (counts - 1)[owner]
            segment = chunk[owner]
            located = self.bins(
                starts[segment] + fraction[:, None] * (ends[segment] - starts[segment])
            )
            # Consecutive points of one segment.
            same = owner[1:] == owner[:-1]
            left, right = located[:-1][same], located[1:][same]
            bad = (left < 0) | (right < 0)
            moved = ~bad & (left != right)
            bad[moved] = ~np.any(
                self._neighbors[left[moved]] == right[moved][:, None], axis=1
            )
            clear[chunk[np.unique(owner[1:][same][bad])]] = False
        return clear

    def to_point(
        self, points: NDArray[np.float64], target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Shortest path length inside the environment from each point to ``target``.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.
        target : ndarray, shape (n_dims,)
            A finite position inside the environment.

        Returns
        -------
        ndarray, shape (n_points,)
            Path lengths; ``inf`` for points disconnected from the target.
        """
        target_bin = int(self.bins(target[None, :])[0])
        point_bins = self.bins(points)
        out = np.full(points.shape[0], np.nan)
        located = np.flatnonzero(point_bins >= 0)
        if not located.size:
            return out
        repeated = np.broadcast_to(target, points[located].shape)
        clear = self.visible(points[located], repeated)
        out[located[clear]] = np.linalg.norm(points[located[clear]] - target, axis=1)
        hidden = located[~clear]
        if not hidden.size:
            return out
        graph_distance, predecessors = self._tree(target_bin)
        memo: dict[int, tuple[float, list[tuple[NDArray[np.float64], float]]]] = {}
        for bin_index in np.unique(point_bins[hidden]):
            rows = hidden[point_bins[hidden] == bin_index]
            if not np.isfinite(graph_distance[bin_index]):
                out[rows] = np.inf
                continue
            pulled, chain = self._pull(int(bin_index), target, predecessors, memo)
            out[rows] = self._via_chain(
                points[rows], self._centers[bin_index], pulled, chain
            )
        return out

    def between(
        self, starts: NDArray[np.float64], ends: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Shortest path length inside the environment for each pair of points.

        Parameters
        ----------
        starts, ends : ndarray, shape (n_pairs, n_dims)
            Positions; a pair with a NaN row or a point outside the
            environment gives NaN.

        Returns
        -------
        ndarray, shape (n_pairs,)
            Path lengths; ``inf`` between disconnected parts.
        """
        out = np.full(starts.shape[0], np.nan)
        ok = np.flatnonzero((self.bins(starts) >= 0) & (self.bins(ends) >= 0))
        clear = self.visible(starts[ok], ends[ok])
        out[ok[clear]] = np.linalg.norm(ends[ok[clear]] - starts[ok[clear]], axis=1)
        for row in ok[~clear]:
            out[row] = self.to_point(starts[row : row + 1], ends[row])[0]
        return out

    def _tree(self, target_bin: int) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Graph distances and shortest-path predecessors toward ``target_bin``."""
        if target_bin not in self._trees:
            from scipy.sparse.csgraph import dijkstra

            distance, predecessors = dijkstra(
                self._adjacency,
                directed=False,
                indices=target_bin,
                return_predecessors=True,
            )
            self._trees[target_bin] = (distance, predecessors)
        return self._trees[target_bin]

    def _pull(
        self,
        start_bin: int,
        target: NDArray[np.float64],
        predecessors: NDArray[np.int32],
        memo: dict[int, tuple[float, list[tuple[NDArray[np.float64], float]]]],
    ) -> tuple[float, list[tuple[NDArray[np.float64], float]]]:
        """Taut path length from a bin center to ``target`` along the graph path.

        From the bin center, jump to the farthest waypoint of its graph path
        (bin centers, then ``target``) that a straight segment inside the
        environment reaches, and continue from there. The graph neighbour is
        always reachable, so every jump makes progress.

        Returns
        -------
        length : float
            Taut path length from the bin center to ``target``.
        chain : list of (point, remaining)
            The taut path's later waypoints, each with its remaining length
            to ``target``; the last is ``(target, 0.0)``.
        """
        pending: list[tuple[int, int]] = []
        bin_index = start_bin
        while bin_index not in memo:
            path = [bin_index]
            while predecessors[path[-1]] >= 0:
                path.append(int(predecessors[path[-1]]))
            waypoints = np.vstack([self._centers[path[1:]], target[None, :]])
            here = self._centers[bin_index]
            clear = self.visible(np.broadcast_to(here, waypoints.shape), waypoints)
            reach = int(np.flatnonzero(clear)[-1]) if clear.any() else 0
            if reach == len(waypoints) - 1:  # the target itself is in reach
                memo[bin_index] = (
                    float(np.linalg.norm(target - here)),
                    [(target, 0.0)],
                )
                break
            anchor = path[1 + reach]
            pending.append((bin_index, anchor))
            bin_index = anchor
        for bin_index, anchor in reversed(pending):
            anchor_length, anchor_chain = memo[anchor]
            point = self._centers[anchor]
            leg = float(np.linalg.norm(point - self._centers[bin_index]))
            memo[bin_index] = (
                leg + anchor_length,
                [(point, anchor_length), *anchor_chain],
            )
        return memo[start_bin]

    def _via_chain(
        self,
        points: NDArray[np.float64],
        center: NDArray[np.float64],
        center_length: float,
        chain: list[tuple[NDArray[np.float64], float]],
    ) -> NDArray[np.float64]:
        """Shortest of: straight to a reachable chain waypoint, then along it.

        The bin center is always a candidate (each point lies in its bin).
        """
        anchors = np.vstack([center[None, :], *(p[None, :] for p, _ in chain)])
        remaining = np.array([center_length, *(r for _, r in chain)])
        n_points, n_anchors = points.shape[0], anchors.shape[0]
        starts = np.repeat(points, n_anchors, axis=0)
        ends = np.tile(anchors, (n_points, 1))
        clear = self.visible(starts, ends).reshape(n_points, n_anchors)
        clear[:, 0] = True
        legs = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2)
        total = np.where(clear, legs + remaining[None, :], np.inf)
        best: NDArray[np.float64] = total.min(axis=1)
        return best
