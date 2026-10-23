"""Routes that lap trajectories follow between neighboring bins.

A lap visits a chain of neighboring bins. The straight line between two
neighboring bin centers can leave the environment: it cuts the corner where
two track segments meet, crosses outside a curved or slanted arena wall
between two grid cells that touch only at a corner, or crosses a wall. Each
step instead follows the shortest route between the two centers that stays
in the environment:

- On a track the route runs along the centerline. Two bins on one segment
  are joined by that straight segment; bins on two segments are joined
  through the track node the segments share.
- On planar cells (grids, hexagons, triangles, clipped or wall-split cells)
  the route is the shortest path between the two centers inside the two
  bins' cells, cut to the arena's outline and kept off its walls. When the
  two cells meet only at a point, or not at all inside the arena, the cells
  of the bins neighboring both are added. The region is shrunk by a hair
  (a millionth of a cell width) so that no route grazes the outline, where
  a position's bin is decided by rounding.
- Other bins (N-D grids) are joined by straight lines, which stay inside
  their two closed cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import shapely
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.environment import Environment
from neurospatial.geometry import cell_polygons, track_junctions

# Fraction of a cell width by which a route stays away from the outline.
_CLEARANCE = 1e-6


class LapRouter:
    """Routes laps along chains of neighboring bins of one environment.

    One router serves every path of a lap call, so a step routed for the
    outbound path is reused, reversed, by the inbound path.

    Parameters
    ----------
    env : Environment
        The environment the bins belong to.
    call : str
        The public call the scientist made, named in errors.
    """

    def __init__(self, env: Environment, call: str) -> None:
        self.env = env
        self.call = call
        self.segments: NDArray[np.intp] | None = None
        self.cells: _CellRouter | None = None
        if env.binning.linearization is not None:
            self.segments = _track_segments(env)
        elif env.n_dims == 2 and env.binning.bins.has_measure_dim(2):
            self.cells = _CellRouter(env)

    def route(
        self, path: Sequence[int], path_name: str, argument: str
    ) -> NDArray[np.float64]:
        """Return the waypoints of a lap along ``path``.

        Parameters
        ----------
        path : sequence of int
            Bin indices; consecutive bins are neighbors or the same bin.
        path_name : str
            What the path is, such as ``"outbound_path"`` or ``"the reversed
            outbound path"``, named in errors.
        argument : str
            The argument that sets the path, ``"outbound_path"`` or
            ``"inbound_path"``, named in the fix of errors.

        Returns
        -------
        ndarray of float64, shape (n_waypoints, n_dims)
            Starts at the first bin's center and ends at the last bin's
            center. The lap runs in straight lines between consecutive
            waypoints.

        Raises
        ------
        ValueError
            If no route between two consecutive bins stays inside the
            environment, such as a bridge between separate parts of a combined
            environment.
        """
        bins = np.asarray(path, dtype=np.intp)
        bins = bins[np.r_[True, bins[1:] != bins[:-1]]]
        centers = self.env.bin_centers
        if len(bins) == 1:
            return centers[bins]
        steps = np.column_stack([bins[:-1], bins[1:]])
        if self.segments is not None:
            return _track_route(self.env, self.segments, steps)
        if self.cells is not None:
            legs = []
            for a, b in steps.tolist():
                leg = self.cells.route(a, b)
                if leg is None:
                    raise ValueError(
                        _no_route_message(self.call, path_name, argument, a, b)
                    )
                legs.append(leg)
            return np.vstack([centers[bins[:1]], *(leg[1:] for leg in legs)])
        return centers[bins]


def _no_route_message(call: str, path_name: str, argument: str, a: int, b: int) -> str:
    """The error for a step between two bins that no route can take."""
    return _format_error(
        f"{call}: {path_name} steps from bin {a} to bin {b}, "
        "but every route between them would leave the "
        "environment.",
        why="Why: the two bins are neighbors in env.connectivity, "
        "but their cells do not touch inside the environment, "
        "for example across a bridge between separate parts of "
        "a combined environment.",
        fix=f"pass {argument}= as a chain of bins whose cells "
        "touch, or add bins for the space between the two parts",
    )


def _track_segments(env: Environment) -> NDArray[np.intp]:
    """The track-graph edge each bin of a track lies on."""
    segment_of = nx.get_node_attributes(env.connectivity, "source_edge_id")
    if len(segment_of) != env.n_bins:
        raise ValueError(
            _format_error(
                "Lap trajectories cannot follow this track: its bins do not "
                "record which track segment each one lies on.",
                why="Why: a lap turns at the node where two segments meet, and "
                "the stored linear cells of this track do not match its "
                "segments, so the junctions between bins are unknown.",
                fix="rebuild the track with Environment.linear_track, "
                "Environment.maze or Environment.from_graph",
            )
        )
    return np.array([segment_of[i] for i in range(env.n_bins)], dtype=np.intp)


def _track_route(
    env: Environment, segments: NDArray[np.intp], steps: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Waypoints along a track's centerline, through the junction nodes."""
    linearization = env.binning.linearization
    assert linearization is not None
    # NaN where both bins lie on one straight segment: no turn between them.
    junctions = track_junctions(steps, segments, linearization.graph, env.n_dims)
    centers = env.bin_centers
    waypoints = [centers[steps[0, 0]]]
    for (_a, b), junction in zip(steps, junctions, strict=True):
        if np.all(np.isfinite(junction)):
            waypoints.append(junction)
        waypoints.append(centers[b])
    return np.asarray(waypoints, dtype=np.float64)


class _CellRouter:
    """Shortest routes between neighboring planar cells, inside the arena."""

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.cells = cell_polygons(env.binning)
        self.domain = env.arena.polygonal_domain()
        self.clearance = _CLEARANCE * float(np.sqrt(np.min(env.bin_sizes)))
        walls = env.arena.walls
        self.walls = shapely.union_all(list(walls)) if walls else None
        # Routes keep this far from walls in every try.
        self.wall_strips = (
            None
            if self.walls is None
            else self.walls.buffer(self.clearance, quad_segs=1)
        )
        self.supports: dict[int, shapely.Geometry] = {}
        self.routes: dict[tuple[int, int], NDArray[np.float64]] = {}

    def support(self, index: int) -> shapely.Geometry:
        """The part of a bin's cell inside the arena's outline."""
        if index not in self.supports:
            cell = self.cells[index]
            self.supports[index] = (
                cell if self.domain is None else cell.intersection(self.domain)
            )
        return self.supports[index]

    def region(self, bins: Sequence[int], clearance: float) -> shapely.Geometry:
        """The bins' cells inside the outline, shrunk off it and off walls.

        A negative ``clearance`` grows the cells instead.
        """
        union = shapely.union_all([self.support(i) for i in bins])
        region = union.buffer(-clearance, join_style="mitre")
        if self.walls is not None and self.walls.intersects(union):
            region = region.difference(self.wall_strips)
        shapely.prepare(region)
        return region

    def route(self, a: int, b: int) -> NDArray[np.float64] | None:
        """Waypoints from bin ``a``'s center to bin ``b``'s center, or None.

        None means every route between the two bins leaves the environment.
        """
        if (b, a) in self.routes:
            return self.routes[b, a][::-1]
        if (a, b) not in self.routes:
            start, end = self.env.bin_centers[[a, b]]
            # Cells that meet only at a point (a corner no third bin covers)
            # come apart when shrunk. The second try adds the bins neighboring
            # both. The last tries keep the closed cells, then grow them by the
            # clearance, for cells and an outline that a rotation moved apart
            # at that point by rounding; either route is accepted only if the
            # environment holds every bend.
            for with_shared, clearance, verify_contains in (
                (False, self.clearance, False),
                (True, self.clearance, False),
                (False, 0.0, True),
                (False, -self.clearance, True),
            ):
                bins = [a, b, *self.shared(a, b)] if with_shared else [a, b]
                route = _shortest_route(self.region(bins, clearance), start, end)
                if route is not None and (
                    not verify_contains or np.all(self.env.contains(route))
                ):
                    break
            else:
                return None
            self.routes[a, b] = route
        return self.routes[a, b]

    def shared(self, a: int, b: int) -> list[int]:
        """The bins neighboring both ``a`` and ``b``, in index order."""
        graph = self.env.connectivity
        return sorted(set(graph[a]) & set(graph[b]))


def _shortest_route(
    region: shapely.Geometry,
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Shortest polyline from ``start`` to ``end`` inside ``region``, or None.

    A shortest path inside a polygon bends only at the polygon's vertices, so
    it is the shortest path through the graph of straight legs between the
    two ends and the vertices that lie inside the polygon.
    """
    if region.is_empty:
        return None
    if region.covers(shapely.linestrings([start, end])):
        return np.array([start, end])
    rings = [
        shapely.get_coordinates(ring)[:-1]
        for ring in shapely.get_rings(shapely.get_parts(region))
    ]
    points = np.vstack([start, end, *rings])
    first, second = np.triu_indices(len(points), k=1)
    legs = shapely.linestrings(np.stack([points[first], points[second]], axis=1))
    inside = shapely.covers(region, legs)
    first, second = first[inside], second[inside]
    lengths = np.linalg.norm(points[second] - points[first], axis=1)
    graph = scipy.sparse.coo_matrix(
        (lengths, (first, second)), shape=(len(points), len(points))
    ).tocsr()
    distances, predecessors = scipy.sparse.csgraph.dijkstra(
        graph, directed=False, indices=0, return_predecessors=True
    )
    if not np.isfinite(distances[1]):
        return None
    order = [1]
    while order[-1] != 0:
        order.append(int(predecessors[order[-1]]))
    return points[order[::-1]]
