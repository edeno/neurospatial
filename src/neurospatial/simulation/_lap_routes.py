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
  the route is the shortest path between the two centers inside the parts
  of the two bins' cells that each bin stands for (the piece inside the
  arena holding its center, :func:`~neurospatial.geometry.center_pieces`),
  kept off the arena's walls. Neighboring bins' pieces meet along part of
  their shared side or at a shared corner, so the route passes there. The
  pieces are widened, and the walls thickened, by geometry's on-boundary
  tolerance (:func:`~neurospatial.geometry.coordinate_roundoff`), so pieces
  that meet at a single corner, or that a rotation moved apart by rounding,
  still join, and every waypoint is a position the environment holds.
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
from neurospatial.geometry import (
    cell_polygons,
    center_pieces,
    coordinate_roundoff,
    track_junctions,
)


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
        self.cells = np.asarray(cell_polygons(env.binning), dtype=object)
        self.domain = env.arena.polygonal_domain() if env.arena.domain_known else None
        # Rounding of a position on the outline, at the environment's scale.
        reference = float(np.max(np.abs(env.bin_centers)))
        self.tolerance = float(
            coordinate_roundoff(np.array([[reference]]), reference)[0]
        )
        walls = env.arena.walls
        self.walls = (
            shapely.union_all(list(walls)).buffer(self.tolerance, quad_segs=1)
            if walls
            else None
        )
        self.routes: dict[tuple[int, int], NDArray[np.float64] | None] = {}

    def region(self, a: int, b: int) -> shapely.Geometry:
        """The two bins' pieces, widened by the tolerance and off the walls."""
        cells = self.cells[[a, b]]
        if self.domain is not None:
            cells = center_pieces(cells, self.env.bin_centers[[a, b]], self.domain)
        region = shapely.union_all(cells).buffer(self.tolerance, join_style="mitre")
        if self.walls is not None and self.walls.intersects(region):
            region = region.difference(self.walls)
        shapely.prepare(region)
        return region

    def route(self, a: int, b: int) -> NDArray[np.float64] | None:
        """Waypoints from bin ``a``'s center to bin ``b``'s center, or None.

        None means every route between the two bins leaves the environment,
        such as across a bridge between separate parts of a combined
        environment.
        """
        if (b, a) in self.routes:
            reverse = self.routes[b, a]
            return None if reverse is None else reverse[::-1]
        if (a, b) not in self.routes:
            start, end = self.env.bin_centers[[a, b]]
            self.routes[a, b] = _shortest_route(self.region(a, b), start, end)
        return self.routes[a, b]


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
