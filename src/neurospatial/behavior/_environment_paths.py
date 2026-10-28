"""Shortest path lengths inside an environment, between continuous positions.

Every behavior function with ``metric="geodesic"`` measures distance with this
module, so the word has one meaning across :mod:`neurospatial.behavior`:

    the length of the shortest path between two positions that stays inside
    the environment.

The path never crosses a closed bin connection (a wall declared on the arena,
an outline slit, a concave corner) and never leaves the environment. A
function that takes bins rather than positions uses each bin's center as its
position. :func:`environment_paths` returns the shared, cached
:class:`EnvironmentPaths` of an environment.

**Open environments (grids, polygons, meshes).** Bin-graph distances (bin
center to bin center along the connectivity graph, as
``Environment.distance_between`` measures) read an oblique straight run as a
staircase of grid steps, up to ~8% too long on an 8-connected grid, and give
every position in a bin the same distance. This module measures in continuous
space instead:

1. A straight segment between two bin centers is *visible* when it stays
   inside the environment: probed every half of the shortest bin-graph edge
   (never exactly on a lattice edge), every probe lies in a bin's cell, and
   every cell it crosses between two probes is a bin joined to the last by an
   open connection (on a rectilinear grid found exactly from the order of its
   axis crossings). A wall closes the connection between the bins on its two
   sides, so a segment across a wall is not visible, and the test is the same
   for a segment and its mirror image.
2. From the center of the target's bin, the any-angle shortest path to every
   bin center is grown outward in order of bin-graph distance (Theta*; Nash
   et al. 2007): each bin connects straight to the target when visible, else
   straight to a waypoint (the "anchor") of a nearer neighbor's path when
   visible, else along the bin-graph edge to that neighbor. Waypoints are bin
   centers, so a path keeps about half a bin away from each corner it turns
   around and can exceed the true shortest path by up to about one bin size
   per corner (the end of a thin wall is two corners). The field of each
   target bin is kept, so later queries to it cost a lookup.
3. A position ``p`` in bin ``a`` is joined to that field: when ``a``'s center
   sees the target's bin center, the distance is the straight
   ``|p - target|``; otherwise it is ``|p - anchor|`` plus the anchor's path,
   with the path's last leg ending at the target itself rather than at its
   bin center. Which of the two is used is decided between the bin centers,
   so near a wall end a position can be up to a bin's width from where the
   straight line actually becomes blocked; but a leg that starts or ends at
   a position (the straight line, the leg to the anchor, the last leg) is
   used only where it is itself visible. Where it is not, the path goes
   through the bin's own center or ends through the target bin's center,
   whichever is shortest, so no path cuts a corner that the bin centers
   clear. Two positions in one bin are always the straight line apart, and
   two in connected bins are when that line is visible. Toward several
   source bins at once, a position leaves for the anchor of its own bin or
   of a neighboring bin, whichever it sees, or else through its bin's
   center.

**Track environments** (``Environment.from_graph``, ``linear_track``,
``maze``) are one-dimensional: the distance is measured along the track graph
(its centerline) between each position's projection onto the track, so it
does not depend on ``track_width`` or ``bin_size``. A bin's position is its
center, which lies on the centerline.

Positions outside the environment have no distance (NaN); positions in
disconnected parts of the environment are ``inf`` apart.

References
----------
.. [1] Nash, A., Daniel, K., Koenig, S., & Felner, A. (2007). Theta*:
       Any-angle path planning on grids. Proc. AAAI 22, 1177-1183.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import ragged_arange
from neurospatial.environment._cache import CacheKey
from neurospatial.geometry import require_cartesian
from neurospatial.geometry._hex_lookup import cartesian_to_fractional_cube

if TYPE_CHECKING:
    from scipy.sparse import csr_array

    from neurospatial.environment import Environment
    from neurospatial.geometry._locators import HexLocator, LookupIndexes

# Probe a segment every half of the shortest graph edge. Consecutive probes
# must land in the same or connected bins, so a probe step shorter than a bin
# cannot skip a missing bin or a closed connection.
_PROBE_FRACTION = 0.5
# Probes sit at (k - 1 + _PROBE_PHASE) / n of the way, an irrational phase
# (2 - sqrt(2)), so no probe lands exactly on an edge of a regular lattice
# between bin centers, where half-open cells would break mirror symmetry.
_PROBE_PHASE = 2.0 - 2.0**0.5
# A stretch between two probes that are not in nearest-neighbor bins is
# split at this fraction (sqrt(2) - 1) until its pieces are, or until a piece
# is shorter than _SPLIT_RESOLUTION probe spacings.
_SPLIT_AT = 2.0**0.5 - 1.0
_SPLIT_RESOLUTION = 1e-2
# Runs of at most this many probe stretches that are not clean are probed.
_TRACE_RUN = 8
# Probe points located per call to ``Environment.bin_at``.
_PROBE_CHUNK = 1_000_000
# Soft memory budget of the per-target fields an environment keeps.
_FIELD_CACHE_BYTES = 256_000_000
# Targets whose fields are grown together, bounding the working memory.
_FIELD_BATCH = 128
# Coordinates of center differences held at once while building pair
# tables, bounding that (targets, n_bins, n_dims) temporary.
_PAIR_TABLE_VALUES = 4_000_000
# Multi-bin source sets whose fields are kept.
_SOURCE_CACHE_LIMIT = 8
# Relative tolerance below which two bin-graph distances are equal.
_GRAPH_RTOL = 1e-9


def environment_paths(env: Environment, *, call: str) -> EnvironmentPaths:
    """Shared shortest-path engine of ``env``, kept in its analysis cache.

    Parameters
    ----------
    env : Environment
        An allocentric Cartesian environment.
    call : str
        Public function name, used in the error message.

    Returns
    -------
    EnvironmentPaths
        The track engine for a track environment, else the open-environment
        engine.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is an egocentric polar environment, whose bins are
        (distance, angle), not positions a path can run between.
    """
    require_cartesian(
        env.coordinate_system,
        f"{call}(metric='geodesic')",
        why=(
            "a geodesic distance is the length of a path through allocentric "
            "space; an egocentric polar environment's bins are (distance, "
            "angle) in the animal's frame."
        ),
        fix="use the allocentric environment the positions were recorded in, "
        "or metric='euclidean'",
    )
    return env._cache.get(
        _PATHS_KEY,
        lambda: (
            _TrackPaths(env)
            if env.binning.linearization is not None
            else _OpenPaths(env)
        ),
    )


_PATHS_KEY: CacheKey[EnvironmentPaths] = CacheKey("environment paths")


def _bin_graph(env: Environment) -> csr_array:
    """Symmetric bin-graph edge lengths with sorted, unique entries per row."""
    from neurospatial.ops.distance import _bin_adjacency

    adjacency = _bin_adjacency(
        env.connectivity, weight="distance", operation="geodesic path length"
    )
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    return adjacency


def _rate_weighted(adjacency: csr_array, rate: NDArray[np.float64]) -> csr_array:
    """Edge costs: each edge's length times the mean rate of its two bins."""
    weighted = adjacency.copy()
    rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    weighted.data = adjacency.data * 0.5 * (rate[rows] + rate[adjacency.indices])
    return weighted


class EnvironmentPaths(ABC):
    """Shortest path lengths inside one environment.

    Use :func:`environment_paths` to get the shared instance of an
    environment: the open-environment engine (grids, polygons, meshes) or the
    track engine, which answer the same queries. Each query handles missing
    positions (NaN rows and positions outside the environment give NaN) and
    passes the located rest to its engine.

    Parameters
    ----------
    env : Environment
        An allocentric Cartesian environment.
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._centers = np.asarray(env.bin_centers, dtype=float)
        self._n_bins = self._centers.shape[0]
        self._adjacency: csr_array = _bin_graph(env)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def bins(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Bin of each point, -1 outside the environment or for NaN rows."""
        if points.shape[0] == 0:
            return np.zeros(0, dtype=np.intp)
        return np.asarray(self._env.bin_at(points), dtype=np.intp)

    def to_point(
        self, points: NDArray[np.float64], target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Shortest path length inside the environment from each point to ``target``.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.
        target : ndarray, shape (n_dims,)
            A position; outside the environment (or NaN) every length is NaN.

        Returns
        -------
        ndarray, shape (n_points,)
            Path lengths; ``inf`` for points disconnected from the target.
        """
        out = np.full(points.shape[0], np.nan)
        root = int(self.bins(target[None, :])[0])
        if root < 0:
            return out
        point_bins = self.bins(points)
        rows = np.flatnonzero(point_bins >= 0)
        if rows.size:
            out[rows] = self._to_point(points[rows], point_bins[rows], target, root)
        return out

    def between(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        *,
        start_bins: NDArray[np.intp] | None = None,
        end_bins: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Shortest path length inside the environment for each pair of points.

        Parameters
        ----------
        starts, ends : ndarray, shape (n_pairs, n_dims)
            Positions; a pair with a NaN row or a point outside the
            environment gives NaN.
        start_bins, end_bins : ndarray of int, shape (n_pairs,), optional
            The positions' bins (``-1`` outside), when the caller already
            has them; located with :meth:`bins` otherwise.

        Returns
        -------
        ndarray, shape (n_pairs,)
            Path lengths; ``inf`` between disconnected parts.
        """
        if start_bins is None:
            start_bins = self.bins(starts)
        if end_bins is None:
            end_bins = self.bins(ends)
        out = np.full(starts.shape[0], np.nan)
        rows = np.flatnonzero((start_bins >= 0) & (end_bins >= 0))
        if rows.size:
            out[rows] = self._between(
                starts[rows], ends[rows], start_bins[rows], end_bins[rows]
            )
        return out

    def between_bins(
        self, start_bins: NDArray[np.intp], end_bins: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Shortest path length between bin centers, pair by pair.

        Parameters
        ----------
        start_bins, end_bins : ndarray of int, shape (n_pairs,)
            Bin indices; a pair with a ``-1`` gives NaN.

        Returns
        -------
        ndarray, shape (n_pairs,)
            Path lengths; ``inf`` between disconnected parts.
        """
        start_bins = np.asarray(start_bins, dtype=np.intp)
        end_bins = np.asarray(end_bins, dtype=np.intp)
        return self.between(
            self._centers[start_bins],
            self._centers[end_bins],
            start_bins=start_bins,
            end_bins=end_bins,
        )

    @abstractmethod
    def pairwise(self, points: NDArray[np.float64]) -> PairDistances:
        """Distances between many pairs of the same positions, ``(i, j)``.

        Locates the positions once and prepares what every pair needs, so
        each later call is a lookup. For analyses that compare every sample
        with many others, such as mean square displacement.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.

        Returns
        -------
        PairDistances
        """

    def prepare(self, bins: NDArray[np.intp]) -> None:  # noqa: B027
        """Get ready to measure paths to every target bin in ``bins``.

        Callers that will measure paths to many targets (every pair of a
        trajectory) call this once, so an engine that keeps per-target work
        does it for all of them together. The track engine keeps none.
        """

    def directions(
        self, points: NDArray[np.float64], target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Direction in which the shortest path from each point leaves it.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.
        target : ndarray, shape (n_dims,)
            A position; outside the environment (or NaN) every direction is
            NaN.

        Returns
        -------
        ndarray, shape (n_points, n_dims)
            Unit vectors toward the path's first waypoint: the target itself
            when the straight line is the path, else the first corner it
            goes around. On a track, the direction along the track toward
            the target. NaN where it is undefined: at the target, at a point
            disconnected from it, or outside the environment.
        """
        out = np.full(points.shape, np.nan)
        root = int(self.bins(target[None, :])[0])
        if root < 0:
            return out
        point_bins = self.bins(points)
        rows = np.flatnonzero(point_bins >= 0)
        if rows.size:
            out[rows] = self._directions(points[rows], point_bins[rows], target, root)
        return out

    @abstractmethod
    def from_bins(self, source_bins: NDArray[np.intp]) -> NDArray[np.float64]:
        """Distance from every bin center to the nearest source bin center.

        Parameters
        ----------
        source_bins : ndarray of int, shape (n_sources,)
            Valid, distinct bin indices (the caller checks).

        Returns
        -------
        ndarray, shape (n_bins,)
            ``inf`` for bins that cannot reach any source.
        """

    def from_bins_at(
        self, source_bins: NDArray[np.intp], points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Distance from each point to the nearest source bin center.

        With several sources in an open environment, a point's path leaves
        straight for the anchor of its own bin or, when visible from the
        point, of a neighboring bin, whichever is shortest.

        Parameters
        ----------
        source_bins : ndarray of int, shape (n_sources,)
            Valid, distinct bin indices.
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.

        Returns
        -------
        ndarray, shape (n_points,)
        """
        source_bins = np.asarray(source_bins, dtype=np.intp)
        out = np.full(points.shape[0], np.nan)
        point_bins = self.bins(points)
        rows = np.flatnonzero(point_bins >= 0)
        if rows.size:
            out[rows] = self._from_bins_at(source_bins, points[rows], point_bins[rows])
        return out

    @abstractmethod
    def costs_from_bins(
        self, source_bins: NDArray[np.intp], rate: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Least travel cost from every bin center to the nearest source bin.

        The cost of a path is the integral of ``rate`` (cost per unit
        length, one value per bin) along it, taking each stretch's rate from
        the bin it runs through. With ``rate`` all ones it is
        :meth:`from_bins`.

        Parameters
        ----------
        source_bins : ndarray of int, shape (n_sources,)
        rate : ndarray, shape (n_bins,)
            Finite, positive cost per unit length.

        Returns
        -------
        ndarray, shape (n_bins,)
            ``inf`` for bins that cannot reach any source.
        """

    # ------------------------------------------------------------------
    # Engine hooks: every point is inside the environment, in its bin
    # ------------------------------------------------------------------

    @abstractmethod
    def _to_point(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        target: NDArray[np.float64],
        root: int,
    ) -> NDArray[np.float64]:
        """:meth:`to_point` for points in ``point_bins``, target in ``root``."""

    @abstractmethod
    def _between(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """:meth:`between` for pairs whose two points are both located."""

    @abstractmethod
    def _directions(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        target: NDArray[np.float64],
        root: int,
    ) -> NDArray[np.float64]:
        """:meth:`directions` for points in ``point_bins``, target in ``root``."""

    @abstractmethod
    def _from_bins_at(
        self,
        source_bins: NDArray[np.intp],
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """:meth:`from_bins_at` for points in ``point_bins``."""


class _OpenPaths(EnvironmentPaths):
    """Any-angle shortest paths in an open environment, with cached fields."""

    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        n_bins = self._n_bins
        self._locator = env.binning.locator
        adjacency = self._adjacency
        # Each bin's open neighbors, padded with -1: a membership test reads
        # one short row instead of searching every connection.
        degree = np.diff(adjacency.indptr)
        self._neighbors = np.full((n_bins, int(degree.max(initial=0))), -1, np.intp)
        self._neighbors[np.repeat(np.arange(n_bins), degree), ragged_arange(degree)] = (
            adjacency.indices
        )
        lengths = adjacency.data[adjacency.data > 0]
        self._spacing = (
            _PROBE_FRACTION * float(lengths.min()) if lengths.size else np.inf
        )
        self._grid = _GridCells.of(self._locator, self._centers)
        # Deep interiors of a mesh's triangles, where a point's bin is known
        # without the locator.
        self._cores = _MeshCores.of(self._locator, n_bins, env._cache)
        # The lattice whose clean boxes skip probing (:meth:`_trace_boxes`).
        self._boxes: _BoxCells | None = (
            self._grid
            if self._grid is not None
            else _HexCells.of(self._locator, self._centers)
        )
        # Bins this close share a face: a straight stretch between them
        # stays in the two.
        self._nearest = float(lengths.min()) * (1 + 1e-6) if lengths.size else np.inf
        self._fields = _RootFields(n_bins)
        self._source_fields: dict[
            tuple[int, ...], tuple[NDArray[np.float64], NDArray[np.int32]]
        ] = {}

    def pairwise(self, points: NDArray[np.float64]) -> PairDistances:
        return _OpenPairDistances(self, points)

    def prepare(self, bins: NDArray[np.intp]) -> None:
        # As many fields as the cache keeps; the rest grow when queried.
        self._fields.ensure(self, np.unique(bins[bins >= 0])[: self._fields.limit])

    def from_bins(self, source_bins: NDArray[np.intp]) -> NDArray[np.float64]:
        source_bins = np.asarray(source_bins, dtype=np.intp)
        if source_bins.size == 1:
            # One source: the same field as a target position in that bin.
            self._fields.ensure(self, source_bins)
            row = self._fields.row_of_root[source_bins[0]]
            single: NDArray[np.float64] = self._fields.length[row].copy()
            return single
        distances: NDArray[np.float64] = self._source_field(source_bins)[0].copy()
        return distances

    def costs_from_bins(
        self, source_bins: NDArray[np.intp], rate: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        source_bins = np.asarray(source_bins, dtype=np.intp)
        length, _, _ = self._grow(
            [source_bins], rate=rate, single_roots=source_bins.size == 1
        )
        costs: NDArray[np.float64] = length[0]
        return costs

    def _to_point(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        target: NDArray[np.float64],
        root: int,
    ) -> NDArray[np.float64]:
        fields = self._fields
        fields.ensure(self, np.array([root]))
        return self._join(
            points,
            point_bins,
            np.broadcast_to(target, points.shape),
            np.full(points.shape[0], root),
            (fields.length, fields.anchor, fields.lead),
            np.full(points.shape[0], fields.row_of_root[root]),
        )[0]

    def _between(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        out = np.empty(starts.shape[0])
        # Points in one bin see each other (cells are convex); points in two
        # connected bins do when the segment between them stays inside.
        straight = start_bins == end_bins
        joined = np.flatnonzero(~straight & self._connected(start_bins, end_bins))
        if joined.size:
            straight[joined] = self._sees(
                starts[joined], ends[joined], start_bins[joined], end_bins[joined]
            )
        out[straight] = np.linalg.norm(ends[straight] - starts[straight], axis=1)
        rest = np.flatnonzero(~straight)
        if not rest.size:
            return out
        a, b = start_bins[rest], end_bins[rest]
        fields = self._fields
        kept = fields.row_of_root[b] >= 0
        if kept.any():
            out[rest[kept]] = self._join(
                starts[rest[kept]],
                a[kept],
                ends[rest[kept]],
                b[kept],
                (fields.length, fields.anchor, fields.lead),
                fields.row_of_root[b[kept]],
            )[0]
        # Other targets: grow each field only as far as its start bins.
        grow = np.flatnonzero(~kept)
        if grow.size:
            roots, field_rows = np.unique(b[grow], return_inverse=True)
            field_rows = field_rows.ravel()
            # Start bins of each target, grouped by its field.
            stop_bins = np.split(
                a[grow][np.argsort(field_rows, kind="stable")],
                np.cumsum(np.bincount(field_rows))[:-1],
            )
            parts = [
                self._grow(
                    list(roots[lo : lo + _FIELD_BATCH, None]),
                    single_roots=True,
                    stop_bins=stop_bins[lo : lo + _FIELD_BATCH],
                )
                for lo in range(0, roots.size, _FIELD_BATCH)
            ]
            grown = tuple(np.vstack([part[k] for part in parts]) for k in range(3))
            out[rest[grow]] = self._join(
                starts[rest[grow]],
                a[grow],
                ends[rest[grow]],
                b[grow],
                grown,  # type: ignore[arg-type]
                field_rows,
            )[0]
        return out

    def _directions(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        target: NDArray[np.float64],
        root: int,
    ) -> NDArray[np.float64]:
        fields = self._fields
        fields.ensure(self, np.array([root]))
        out = np.full(points.shape, np.nan)
        length, waypoint = self._join(
            points,
            point_bins,
            np.broadcast_to(target, points.shape),
            np.full(points.shape[0], root),
            (fields.length, fields.anchor, fields.lead),
            np.full(points.shape[0], fields.row_of_root[root]),
            waypoints=True,
        )
        assert waypoint is not None
        vector = waypoint - points
        norm = np.linalg.norm(vector, axis=1)
        defined = np.isfinite(length) & (norm > 0)
        out[defined] = vector[defined] / norm[defined, None]
        return out

    def _from_bins_at(
        self,
        source_bins: NDArray[np.intp],
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        if source_bins.size == 1:
            source = int(source_bins[0])
            return self._to_point(points, point_bins, self._centers[source], source)
        length, anchor = self._source_field(source_bins)
        n_points = points.shape[0]
        # Candidate corners: the anchor of the point's bin and the anchors of
        # the bin's open neighbors, each where the point sees it, so the
        # nearest source is chosen from the point itself. Where the point
        # does not see its bin's anchor, it goes through its bin's center,
        # which it sees.
        own = anchor[point_bins].astype(np.intp)
        check = np.flatnonzero((own >= 0) & (own != point_bins))
        if check.size:
            hidden = ~self._sees(
                points[check], self._centers[own[check]], point_bins[check], own[check]
            )
            own[check[hidden]] = point_bins[check[hidden]]
        neighbors = self._neighbors[point_bins]
        open_neighbor = neighbors >= 0
        owner = np.repeat(np.arange(n_points), open_neighbor.sum(axis=1))
        corner = anchor[neighbors[open_neighbor]].astype(np.intp)
        keep = (corner >= 0) & (corner != own[owner])
        keys = np.unique(owner[keep] * self._n_bins + corner[keep])
        owner, corner = keys // self._n_bins, keys % self._n_bins
        visible, _ = self._trace(
            points[owner], self._centers[corner], point_bins[owner], corner
        )
        owner = np.r_[np.arange(n_points), owner[visible]]
        corner = np.r_[own, corner[visible]]
        found = corner >= 0
        owner, corner = owner[found], corner[found]
        total = (
            np.linalg.norm(points[owner] - self._centers[corner], axis=1)
            + (length[corner])
        )
        best = np.full(n_points, np.inf)
        np.minimum.at(best, owner, total)
        return best

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cells(
        self,
        points: NDArray[np.float64],
        near: tuple[NDArray[np.intp], ...] = (),
    ) -> NDArray[np.intp]:
        """Bin whose cell holds each point, -1 in no bin's cell.

        Probes use the bins' own cells: a straight segment between bin
        centers stays inside the environment when it stays inside the union
        of the bins' cells. On a mesh, a point deep inside a triangle
        (:class:`_MeshCores`), one of the ``near`` bins of each point or
        found from a table of cells, is that triangle's without the locator,
        exactly as the locator would find it.
        """
        cores = self._cores
        if cores is None or not points.shape[0]:
            return np.asarray(
                self._locator.locate(points, self._env._cache), dtype=np.intp
            )
        located = np.full(points.shape[0], -1, dtype=np.intp)
        for bins in near:
            rest = np.flatnonzero(located < 0)
            found = cores.deep(points[rest], bins[rest])
            located[rest[found]] = bins[rest[found]]
        rest = np.flatnonzero(located < 0)
        if rest.size:
            located[rest] = cores.lookup(points[rest])
        rest = np.flatnonzero(located < 0)
        if rest.size:
            located[rest] = self._locator.locate(points[rest], self._env._cache)
        return located

    def _grid_cells(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """:meth:`_cells` on a grid: a point's half-open cell when it is a bin.

        Only points outside every bin's half-open cell go to the locator,
        which may give them a bin on the outline.
        """
        grid = self._grid
        assert grid is not None
        located = grid.bin_of(grid.half_open(points))
        rest = np.flatnonzero(located < 0)
        if rest.size:
            located[rest] = self._cells(points[rest])
        return located

    def _probe_cells(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """:meth:`_cells` of probes, from the half-open cells on a grid."""
        if self._grid is not None:
            return self._grid_cells(points)
        return self._cells(points)

    def _connected(self, a: NDArray[np.intp], b: NDArray[np.intp]) -> NDArray[np.bool_]:
        """Whether each bin pair is joined by an open connection."""
        found: NDArray[np.bool_] = np.any(self._neighbors[a] == b[:, None], axis=1)
        return found

    def _flanked(self, a: NDArray[np.intp], b: NDArray[np.intp]) -> NDArray[np.bool_]:
        """Whether two connected bins have two open, face-sharing common neighbors.

        A straight stretch from ``a`` to a diagonal neighbor ``b`` passes
        through one of the two cells flanking their shared corner (or the
        corner itself); when both flanking cells are bins joined to ``a``
        and ``b``, every such stretch stays inside, so it need not be split.
        """
        neighbors_a = self._neighbors[a]
        common = (neighbors_a[:, :, None] == self._neighbors[b][:, None, :]).any(
            axis=2
        ) & (neighbors_a >= 0)
        cell = np.maximum(neighbors_a, 0)
        centers = self._centers
        near_a = np.linalg.norm(centers[cell] - centers[a][:, None, :], axis=2)
        near_b = np.linalg.norm(centers[cell] - centers[b][:, None, :], axis=2)
        flank = common & (near_a <= self._nearest) & (near_b <= self._nearest)
        flanked: NDArray[np.bool_] = (flank.sum(axis=1) >= 2) & self._connected(a, b)
        return flanked

    def _join(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        targets: NDArray[np.float64],
        roots: NDArray[np.intp],
        fields: tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]],
        field_rows: NDArray[np.intp],
        *,
        waypoints: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Join each point to its target through the target bin's field.

        Parameters
        ----------
        points : ndarray, shape (n, n_dims)
            Positions, each in bin ``point_bins``.
        point_bins : ndarray of int, shape (n,)
        targets : ndarray, shape (n, n_dims)
            Target positions, each in bin ``roots``.
        roots : ndarray of int, shape (n,)
        fields : (length, anchor, lead)
            Single-root fields, as from :meth:`_grow`, of shape
            (n_fields, n_bins); each point's bin is reached in its field.
        field_rows : ndarray of int, shape (n,)
            Row of each target's field.
        waypoints : bool
            Also return each path's first waypoint.

        Returns
        -------
        length : ndarray, shape (n,)
        waypoint : ndarray, shape (n, n_dims), or None
            See :meth:`_route`.
        """
        length_of, anchor_of, lead_of = fields
        anchor = anchor_of[field_rows, point_bins].astype(np.intp)
        length = length_of[field_rows, point_bins]
        reached = np.isfinite(length)
        corner = np.maximum(anchor, 0)
        corner_length = np.where(reached, length_of[field_rows, corner], np.inf)
        lead = lead_of[field_rows, corner].astype(np.intp)
        straight = (point_bins == roots) | (anchor == roots)
        around = np.full(points.shape[0], np.inf)
        rows = np.flatnonzero(~straight & reached)
        if rows.size:
            lead_center = self._centers[lead[rows]]
            root_center = self._centers[roots[rows]]
            # The field's last leg ends at the root bin's center; end it at
            # the target instead.
            around[rows] = (
                np.linalg.norm(points[rows] - self._centers[corner[rows]], axis=1)
                + length_of[field_rows[rows], corner[rows]]
                - np.linalg.norm(lead_center - root_center, axis=1)
                + np.linalg.norm(lead_center - targets[rows], axis=1)
            )
        return self._route(
            points,
            point_bins,
            targets,
            roots,
            straight,
            corner,
            corner_length,
            lead,
            around,
            waypoints=waypoints,
        )

    def _route(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        targets: NDArray[np.float64],
        roots: NDArray[np.intp],
        straight: NDArray[np.bool_],
        corner: NDArray[np.intp],
        corner_length: NDArray[np.float64],
        lead: NDArray[np.intp],
        around: NDArray[np.float64],
        *,
        waypoints: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Path from each point to its target, with the legs at positions checked.

        A field's line of sight is decided between bin centers. The legs of
        a path that start or end at a position (the point straight to the
        target, the point to its bin's anchor, the last waypoint to the
        target) are taken only where they stay inside (:meth:`_sees`), so a
        path never cuts a corner its bin centers clear. Where one is
        blocked, the path is the shortest of the routes whose legs stay
        inside: from the point to its bin's anchor or to its bin's own
        center (inside the bin's convex cell), then along the field, ending
        at the target from the path's last waypoint or from the root bin's
        center. The bin's own path has length ``|center - anchor| +
        length[anchor]``, as the field sums it.

        Parameters
        ----------
        points, targets : ndarray, shape (n, n_dims)
            Positions, in ``point_bins`` and ``roots``.
        point_bins, roots : ndarray of int, shape (n,)
        straight : ndarray of bool, shape (n,)
            The field takes the point straight to its target.
        corner : ndarray of int, shape (n,)
            Anchor of each point's bin.
        corner_length : ndarray, shape (n,)
            Field length from the anchor's center; ``inf`` where the point's
            bin is not reached.
        lead : ndarray of int, shape (n,)
            Last waypoint before the root on the anchor's path.
        around : ndarray, shape (n,)
            Length through the anchor, ending at the target from ``lead``,
            where the path is not straight.
        waypoints : bool
            Also return each path's first waypoint.

        Returns
        -------
        length : ndarray, shape (n,)
            ``inf`` where the point's bin is not reached.
        waypoint : ndarray, shape (n, n_dims), or None
            The first point the path heads for after the point itself (the
            target, a bin center), NaN where it is not reached.
        """
        n = points.shape[0]
        centers = self._centers
        out = np.full(n, np.inf)
        waypoint = np.full(points.shape, np.nan) if waypoints else None
        direct = straight.copy()
        check = np.flatnonzero(straight & (point_bins != roots))
        if check.size:
            direct[check] = self._sees(
                points[check], targets[check], point_bins[check], roots[check]
            )
        out[direct] = np.linalg.norm(targets[direct] - points[direct], axis=1)
        if waypoint is not None:
            waypoint[direct] = targets[direct]
        rest = np.flatnonzero(~direct & np.isfinite(corner_length))
        if not rest.size:
            return out, waypoint
        p, a, t, r = points[rest], point_bins[rest], targets[rest], roots[rest]
        c, c_length, c_lead = corner[rest], corner_length[rest], lead[rest]
        # The last waypoint on the own center's path: the bin itself when
        # its anchor is the root, else the anchor's.
        own_lead = np.where(c == r, a, c_lead)
        # Legs to check: point to anchor (never in the point's own bin
        # here), and last waypoint to target unless it is the root's center.
        m = rest.size
        index = np.arange(m)
        last_from = np.flatnonzero(c_lead != r)
        own_from = np.flatnonzero(c == r)
        legs = np.r_[index, last_from, own_from]
        leg_start = np.r_[c, c_lead[last_from], a[own_from]]
        sees = self._sees(
            np.r_[p, centers[leg_start[m:]]],
            np.r_[centers[c], t[legs[m:]]],
            np.r_[a, leg_start[m:]],
            np.r_[c, r[legs[m:]]],
        )
        first = sees[:m]
        last = c_lead == r
        last[last_from] = sees[m : m + last_from.size]
        own_last = last.copy()
        own_last[own_from] = sees[m + last_from.size :]
        tail = np.linalg.norm(centers[r] - t, axis=1)
        via = np.linalg.norm(p - centers[c], axis=1) + c_length
        own = np.linalg.norm(p - centers[a], axis=1) + (
            np.linalg.norm(centers[a] - centers[c], axis=1) + c_length
        )
        options = np.stack(
            [
                np.where(
                    first & last,
                    via
                    - np.linalg.norm(centers[c_lead] - centers[r], axis=1)
                    + np.linalg.norm(centers[c_lead] - t, axis=1),
                    np.inf,
                ),
                np.where(first, via + tail, np.inf),
                np.where(
                    own_last,
                    own
                    - np.linalg.norm(centers[own_lead] - centers[r], axis=1)
                    + np.linalg.norm(centers[own_lead] - t, axis=1),
                    np.inf,
                ),
                own + tail,
            ]
        )
        choice = np.argmin(options, axis=0)
        value = options[choice, index]
        # Where the field's own legs stay inside, its length is kept.
        kept = first & last & ~straight[rest]
        value[kept] = around[rest[kept]]
        out[rest] = value
        if waypoint is not None:
            to_corner = kept | (choice <= 1)
            first_point = np.where(to_corner[:, None], centers[c], centers[a])
            # From the own center itself, the path heads on to the target
            # (the center sees it) or to the anchor.
            at_center = np.flatnonzero(~to_corner & np.all(p == centers[a], axis=1))
            onward = (choice[at_center] == 2) & (own_lead[at_center] == a[at_center])
            first_point[at_center] = np.where(
                onward[:, None], t[at_center], centers[c[at_center]]
            )
            waypoint[rest] = first_point
        return out, waypoint

    def _sees(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        """Whether each straight segment between two points stays inside.

        Parameters
        ----------
        starts, ends : ndarray, shape (n, n_dims)
            Points in ``start_bins`` and ``end_bins``.
        start_bins, end_bins : ndarray of int, shape (n,)
        """
        visible, _ = self._trace(starts, ends, start_bins, end_bins)
        return visible

    def _source_field(
        self, source_bins: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Kept field from a set of source bins: (length, anchor) per bin."""
        from neurospatial._lru import lru_get, lru_put

        key = tuple(sorted(int(b) for b in source_bins))
        cached = lru_get(self._source_fields, key)
        if cached is None:
            length, anchor, _ = self._grow([np.asarray(key, dtype=np.intp)])
            cached = (length[0], anchor[0])
            lru_put(self._source_fields, key, cached, limit=_SOURCE_CACHE_LIMIT)
        return cached

    def _trace(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
        rate: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Visibility and travel cost of straight segments between bins.

        Probes are located in rounds of doubling length from the start, and
        a segment stops being probed at its first obstruction, so a blocked
        segment costs about as much as its distance to the obstruction. On a
        grid or a hexagonal lattice, visibility alone skips the probes of
        clean stretches (:meth:`_trace_boxes`).

        Parameters
        ----------
        starts, ends : ndarray, shape (n_segments, n_dims)
            Endpoints, inside ``start_bins`` and ``end_bins``.
        start_bins, end_bins : ndarray of int, shape (n_segments,)
        rate : ndarray, shape (n_bins,), optional
            Cost per unit length in each bin; None gives the length.

        Returns
        -------
        visible : ndarray of bool, shape (n_segments,)
            True when every probe lies in a bin and consecutive probes lie
            in the same bin or in bins joined by an open connection.
        cost : ndarray, shape (n_segments,)
            Segment length, or (visible segments) the trapezoidal integral
            of ``rate`` over the probes when ``rate`` is given.
        """
        n_segments = starts.shape[0]
        visible = np.ones(n_segments, dtype=bool)
        lengths = np.linalg.norm(ends - starts, axis=1)
        cost = lengths.copy()
        if n_segments == 0:
            return visible, cost
        n_steps = np.maximum(np.ceil(lengths / self._spacing).astype(np.intp), 1)
        if rate is None and self._boxes is not None and self._boxes.boxes(self):
            visible = self._trace_boxes(starts, ends, start_bins, end_bins, n_steps)
            return visible, cost
        if rate is not None:
            cost[:] = 0.0
        previous = start_bins.astype(np.intp, copy=True)
        done = np.zeros(n_segments, dtype=np.intp)  # points located so far
        active = np.arange(n_segments)
        batch = 8
        while active.size:
            # Next points of each active segment: probes 1 .. n_steps, then
            # the end (point n_steps + 1).
            take = np.minimum(
                n_steps[active] + 1 - done[active],
                max(1, min(batch, _PROBE_CHUNK // active.size)),
            )
            owner = np.repeat(np.arange(active.size), take)
            step = done[active][owner] + ragged_arange(take) + 1
            segment = active[owner]
            fraction = self._probe_fraction(step, n_steps[segment])
            at_end = step == n_steps[segment] + 1
            located = np.empty(step.size, dtype=np.intp)
            located[at_end] = end_bins[segment[at_end]]
            probe = ~at_end
            if probe.any():
                located[probe] = self._cells(
                    starts[segment[probe]]
                    + fraction[probe, None]
                    * (ends[segment[probe]] - starts[segment[probe]])
                )
            # Pairs: (previous point, first new), then consecutive new points.
            first = np.cumsum(take) - take
            left = np.empty(step.size, dtype=np.intp)
            left[1:] = located[:-1]
            left[first] = previous[active]
            bad = self._crossing_blocked(
                left,
                located,
                segment,
                self._probe_fraction(step - 1, n_steps[segment]),
                fraction,
                starts,
                ends,
            )
            blocked = np.bincount(owner[bad], minlength=active.size) > 0
            visible[active[blocked]] = False
            if rate is not None:
                ok = ~bad
                stretch = np.zeros(step.size)
                stretch[ok] = (
                    0.5
                    * (rate[left[ok]] + rate[located[ok]])
                    * (
                        fraction[ok]
                        - self._probe_fraction(step[ok] - 1, n_steps[segment[ok]])
                    )
                )
                cost[active] += (
                    np.bincount(owner, weights=stretch, minlength=active.size)
                    * lengths[active]
                )
            previous[active] = located[first + take - 1]
            done[active] += take
            active = active[~blocked & (done[active] < n_steps[active] + 1)]
            batch *= 2
        if rate is None:
            return visible, cost
        cost[~visible] = np.inf
        return visible, cost

    def _trace_boxes(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
        n_steps: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        """:meth:`_trace` visibility on a lattice, skipping clean stretches.

        A run of probes whose lattice cells span a box of bins, every two
        neighbors in it joined, has no blocked stretch (:meth:`_BoxCells.clean`
        says why): the cells of the run's points lie in the box spanned by
        those of its two ends, since lattice coordinates are linear along a
        segment. Runs are halved until they are clean or short, and only
        short runs that are not clean are probed, with the same probes and
        stretch tests as :meth:`_trace`, so the result is the same.
        """
        cells = self._boxes
        assert cells is not None
        visible = np.ones(starts.shape[0], dtype=bool)
        geometry = (starts, ends, start_bins, end_bins, n_steps)
        # Runs of points lo .. hi of a segment (0 = start, n_steps + 1 = end),
        # with the cell range each end of the run may touch.
        seg = np.arange(starts.shape[0])
        lo = np.zeros(seg.size, dtype=np.intp)
        hi = n_steps + 1
        # The start is point 0 itself; the end, as probed, is
        # starts + 1.0 * (ends - starts).
        lo_low, lo_high = self._end_cells(starts, start_bins)
        hi_low, hi_high = self._end_cells(starts + (ends - starts), end_bins)
        while seg.size:
            dirty = ~cells.clean(
                np.minimum(lo_low, hi_low), np.maximum(lo_high, hi_high)
            )
            short = dirty & (hi - lo <= _TRACE_RUN)
            if short.any():
                bad = self._run_blocked(seg[short], lo[short], hi[short], *geometry)
                visible[seg[short][bad]] = False
            keep = dirty & ~short
            keep[keep] = visible[seg[keep]]
            seg, lo, hi = seg[keep], lo[keep], hi[keep]
            lo_low, lo_high, hi_low, hi_high = (
                x[keep] for x in (lo_low, lo_high, hi_low, hi_high)
            )
            mid = (lo + hi) // 2
            # The middle is a probe: in no bin, its segment is blocked.
            mid_points = self._point_coordinates(seg, mid, starts, ends, n_steps)
            mid_low, mid_high = cells.point_cells(mid_points)
            missing = cells.missing(mid_points, self)
            if missing.size:
                visible[seg[missing]] = False
                keep = visible[seg]
                seg, lo, hi, mid = seg[keep], lo[keep], hi[keep], mid[keep]
                lo_low, lo_high, hi_low, hi_high, mid_low, mid_high = (
                    x[keep]
                    for x in (lo_low, lo_high, hi_low, hi_high, mid_low, mid_high)
                )
            seg, lo, hi = np.r_[seg, seg], np.r_[lo, mid], np.r_[mid, hi]
            lo_low, hi_low = np.r_[lo_low, mid_low], np.r_[mid_low, hi_low]
            lo_high, hi_high = np.r_[lo_high, mid_high], np.r_[mid_high, hi_high]
        return visible

    def _end_cells(
        self, points: NDArray[np.float64], bins: NDArray[np.intp]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Cell range a run may touch at a segment's end, in ``bins``.

        The cell of the bin, joined with the cells of the point where it is
        not the bin's center (a center lies in its own cell).
        """
        cells = self._boxes
        assert cells is not None
        low = cells.cell[bins]
        high = low.copy()
        off = np.flatnonzero(np.any(points != self._centers[bins], axis=1))
        if off.size:
            point_low, point_high = cells.point_cells(points[off])
            low[off] = np.minimum(low[off], point_low)
            high[off] = np.maximum(high[off], point_high)
        return low, high

    def _point_coordinates(
        self,
        seg: NDArray[np.intp],
        step: NDArray[np.intp],
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        n_steps: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Coordinates of point ``step`` of each segment, as :meth:`_trace` probes."""
        fraction = self._probe_fraction(step, n_steps[seg])
        coordinates: NDArray[np.float64] = starts[seg] + fraction[:, None] * (
            ends[seg] - starts[seg]
        )
        return coordinates

    def _run_blocked(
        self,
        seg: NDArray[np.intp],
        lo: NDArray[np.intp],
        hi: NDArray[np.intp],
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
        n_steps: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        """Whether any stretch of each run of points ``lo .. hi`` is blocked."""
        count = hi - lo + 1
        owner = np.repeat(np.arange(seg.size), count)
        step = np.repeat(lo, count) + ragged_arange(count)
        segment = seg[owner]
        located = np.empty(step.size, dtype=np.intp)
        at_start = step == 0
        at_end = step == n_steps[segment] + 1
        located[at_start] = start_bins[segment[at_start]]
        located[at_end] = end_bins[segment[at_end]]
        probe = ~(at_start | at_end)
        if probe.any():
            located[probe] = self._probe_cells(
                self._point_coordinates(
                    segment[probe], step[probe], starts, ends, n_steps
                )
            )
        # Stretches end at every point but each run's first.
        right = np.ones(step.size, dtype=bool)
        right[np.cumsum(count) - count] = False
        right_at = np.flatnonzero(right)
        fraction = self._probe_fraction(step, n_steps[segment])
        bad = self._crossing_blocked(
            located[right_at - 1],
            located[right_at],
            segment[right_at],
            fraction[right_at - 1],
            fraction[right_at],
            starts,
            ends,
        )
        blocked: NDArray[np.bool_] = (
            np.bincount(owner[right_at][bad], minlength=seg.size) > 0
        )
        return blocked

    @staticmethod
    def _probe_fraction(
        step: NDArray[np.intp], n_steps: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Fraction of the way along a segment of point ``step`` (0 = start).

        Probe ``k`` of ``n`` sits at ``(k - 1 + _PROBE_PHASE) / n``, so probes
        are ``1 / n`` apart and never land exactly on a bin edge of a regular
        lattice between bin centers; the end is point ``n + 1``.
        """
        fraction = (step - 1 + _PROBE_PHASE) / n_steps
        fraction = np.where(step <= 0, 0.0, fraction)
        result: NDArray[np.float64] = np.where(step > n_steps, 1.0, fraction)
        return result

    def _crossing_blocked(
        self,
        left: NDArray[np.intp],
        right: NDArray[np.intp],
        segment: NDArray[np.intp],
        left_fraction: NDArray[np.float64],
        right_fraction: NDArray[np.float64],
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Whether the stretch between consecutive points leaves the bins.

        Points in one bin are joined inside it (cells are convex), and two
        nearest neighbors (a shared face) must be connected. Between bins
        farther apart, as when a stretch passes near a corner, the cells it
        crosses must all be bins joined in turn: on a rectilinear grid they
        are found exactly from the order of its axis crossings; otherwise
        the stretch is split until its pieces join nearest neighbors, so a
        corner it cuts through a missing or closed bin is caught.
        """
        bad = (left < 0) | (right < 0)
        origin = np.flatnonzero(~bad & (left != right))
        a, b, seg = left[origin], right[origin], segment[origin]
        f0, f1 = left_fraction[origin], right_fraction[origin]
        if self._grid is not None and origin.size:
            span = ends[seg] - starts[seg]
            walked, blocked = self._grid.walk(
                a,
                b,
                starts[seg] + f0[:, None] * span,
                starts[seg] + f1[:, None] * span,
                self._connected,
            )
            bad[origin[walked & blocked]] = True
            origin, a, b, seg = (x[~walked] for x in (origin, a, b, seg))
            f0, f1 = f0[~walked], f1[~walked]
        while origin.size:
            gap = np.linalg.norm(self._centers[a] - self._centers[b], axis=1)
            piece = (f1 - f0) * np.linalg.norm(ends[seg] - starts[seg], axis=1)
            split = (gap > self._nearest) & (piece > _SPLIT_RESOLUTION * self._spacing)
            split[split] = ~self._flanked(a[split], b[split])
            joined = ~split
            # Several pieces can share an origin: mark, never unmark.
            bad[origin[joined][~self._connected(a[joined], b[joined])]] = True
            keep = split & ~bad[origin]
            origin, a, b, seg, f0, f1 = (
                origin[keep],
                a[keep],
                b[keep],
                seg[keep],
                f0[keep],
                f1[keep],
            )
            if not origin.size:
                break
            # Split at an irrational fraction, so the split point does not
            # land on a bin edge either.
            fm = f0 + _SPLIT_AT * (f1 - f0)
            m = self._cells(
                starts[seg] + fm[:, None] * (ends[seg] - starts[seg]), near=(a, b)
            )
            outside = m < 0
            bad[origin[outside]] = True
            inside = ~outside
            origin, a, b, seg, m = (x[inside] for x in (origin, a, b, seg, m))
            f0, f1, fm = f0[inside], f1[inside], fm[inside]
            origin = np.r_[origin, origin]
            a, b = np.r_[a, m], np.r_[m, b]
            seg = np.r_[seg, seg]
            f0, f1 = np.r_[f0, fm], np.r_[fm, f1]
            # A piece within one bin stays inside it.
            differ = a != b
            origin, a, b, seg = (x[differ] for x in (origin, a, b, seg))
            f0, f1 = f0[differ], f1[differ]
        return bad

    def _grow(
        self,
        source_sets: list[NDArray[np.intp]],
        *,
        rate: NDArray[np.float64] | None = None,
        single_roots: bool = False,
        stop_bins: list[NDArray[np.intp]] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
        """Grow the any-angle shortest path field of each source set.

        Bins join the field in layers of bin-graph distance one shortest
        edge thick, so a bin's nearer neighbors in earlier layers are final
        when it is reached (a bin with none waits one layer). Ties in graph
        distance orient no edge, so the result does not depend on bin
        numbering. Each bin takes the cheapest of: straight to the source
        (``single_roots``; always, when visible), straight to a visible
        anchor of a final nearer neighbor, or along the graph edge to that
        neighbor.

        Parameters
        ----------
        source_sets : list of ndarray of int
            Bins of each field's sources (cost 0).
        rate : ndarray, shape (n_bins,), optional
            Cost per unit length; None measures length.
        single_roots : bool
            Each set is one bin, and a bin whose center sees it connects
            straight to it.
        stop_bins : list of ndarray of int, optional
            Per set, the bins it is needed for: it grows only through the
            layer of the farthest of them and stops once they are reached.
            Bins it did not reach are left ``inf``/-1. A bin's value depends
            only on bins in its own and earlier layers, so it never depends
            on where growth stopped.

        Returns
        -------
        length : ndarray, shape (n_sets, n_bins)
            Path length (or cost) from each bin center to its field's
            nearest source; ``inf`` when unreachable.
        anchor : ndarray of int32, shape (n_sets, n_bins)
            The waypoint each bin center connects to straight (itself for a
            source); -1 when unreachable.
        lead : ndarray of int32, shape (n_sets, n_bins)
            The bin whose center is the last waypoint before the source on
            the bin's path (the bin itself when it sees the source).
        """
        from scipy.sparse.csgraph import dijkstra

        n_bins, n_sets = self._n_bins, len(source_sets)
        adjacency = (
            self._adjacency if rate is None else _rate_weighted(self._adjacency, rate)
        )
        if single_roots and stop_bins is not None and rate is None:
            graph = self._graph_through(
                np.array([s[0] for s in source_sets], dtype=np.intp), stop_bins
            )
        elif single_roots:
            graph = np.atleast_2d(
                dijkstra(
                    adjacency,
                    directed=False,
                    indices=np.array([s[0] for s in source_sets]),
                )
            )
        else:
            graph = np.vstack(
                [
                    dijkstra(adjacency, directed=False, indices=s, min_only=True)
                    for s in source_sets
                ]
            )
        n_nodes = n_sets * n_bins
        length = np.full(n_nodes, np.inf)
        anchor = np.full(n_nodes, -1, dtype=np.int32)
        lead = np.full(n_nodes, -1, dtype=np.int32)
        is_source = np.zeros(n_nodes, dtype=bool)
        for f, sources in enumerate(source_sets):
            ids = f * n_bins + np.asarray(sources, dtype=np.intp)
            is_source[ids] = True
            length[ids] = 0.0
            anchor[ids] = sources
            lead[ids] = sources
        settled = is_source.copy()
        flat = graph.ravel()
        # Layers of graph distance, one shortest edge thick (inf unreached).
        layer_of = np.floor(flat / (self._spacing / _PROBE_FRACTION))
        needed = None
        if stop_bins is None:
            grown = np.flatnonzero(np.isfinite(flat))
        else:
            needed = np.zeros(n_nodes, dtype=bool)
            for f, stop in enumerate(stop_bins):
                needed[f * n_bins + np.asarray(stop, dtype=np.intp)] = True
            needed &= ~is_source & np.isfinite(flat)
            # Each field grows through the layer of its farthest needed bin.
            last = np.full(n_sets, -1.0)
            need = np.flatnonzero(needed)
            np.maximum.at(last, need // n_bins, layer_of[need])
            grown = np.flatnonzero(layer_of <= np.repeat(last, n_bins))
        # In-edges v <- u of each grown node v from its nearer neighbors u
        # (g[u] < g[v]), as global ids f * n + bin. The bin graph is
        # symmetric with sorted rows, so reading row v as v's in-edges lists
        # them grouped by (field, v) and then by u, with no sort. A nearer
        # neighbor is in the same or an earlier layer, so it is grown too.
        adjacency_rows = self._adjacency.indptr
        grown_bin = grown % n_bins
        degree = adjacency_rows[grown_bin + 1] - adjacency_rows[grown_bin]
        dst = np.repeat(grown, degree)
        src = (dst - np.repeat(grown_bin, degree)) + self._adjacency.indices[
            np.repeat(adjacency_rows[grown_bin], degree) + ragged_arange(degree)
        ]
        g_v = flat[dst]
        nearer = flat[src] < g_v - _GRAPH_RTOL * g_v
        in_src, dst = src[nearer], dst[nearer]
        in_ptr = np.r_[0, np.cumsum(np.bincount(dst, minlength=n_nodes))]

        root_of_field = (
            np.array([s[0] for s in source_sets], dtype=np.intp)
            if single_roots
            else None
        )
        reached = grown[~is_source[grown]]
        layer = layer_of[reached].astype(np.intp)
        by_layer = np.argsort(layer, kind="stable")
        reached, layer = reached[by_layer], layer[by_layer]
        bounds = np.flatnonzero(np.diff(layer)) + 1
        waves = np.split(reached, bounds)
        if needed is not None:
            # Needed bins each field has yet to settle.
            remaining = np.bincount(need // n_bins, minlength=n_sets)
            open_fields = remaining > 0
        waiting = np.zeros(0, dtype=np.intp)
        for wave in [*waves, np.zeros(0, dtype=np.intp)]:
            nodes = np.r_[waiting, wave] if waiting.size else wave
            if needed is not None:
                nodes = nodes[open_fields[nodes // n_bins]]
            if not nodes.size:
                waiting = nodes
                continue
            waiting = self._settle(
                nodes,
                in_src,
                in_ptr,
                settled,
                length,
                anchor,
                lead,
                is_source,
                rate,
                root_of_field,
            )
            if needed is not None:
                done = nodes[settled[nodes] & needed[nodes]]
                needed[done] = False
                remaining -= np.bincount(done // n_bins, minlength=n_sets)
                open_fields = remaining > 0
                if not open_fields.any():
                    break
        while waiting.size:
            # Only bins whose every nearer neighbor sits in their own layer
            # remain; settle them in turn.
            before = waiting.size
            waiting = self._settle(
                waiting,
                in_src,
                in_ptr,
                settled,
                length,
                anchor,
                lead,
                is_source,
                rate,
                root_of_field,
            )
            if waiting.size == before:
                break
        shape = (n_sets, n_bins)
        return length.reshape(shape), anchor.reshape(shape), lead.reshape(shape)

    def _graph_through(
        self, roots: NDArray[np.intp], stop_bins: list[NDArray[np.intp]]
    ) -> NDArray[np.float64]:
        """Bin-graph distance from each root, as far as its growth needs.

        Dijkstra stops at a distance limit, below which its distances are
        exactly those of an unlimited search. The limit starts near the
        straight-line reach of the root's stop bins and is raised until it
        covers the layer of the farthest of them (or every bin, when a stop
        bin is not found), so every bin :meth:`_grow` reads has its exact
        distance; bins beyond are ``inf`` or exact.

        Parameters
        ----------
        roots : ndarray of int, shape (n_roots,)
        stop_bins : list of ndarray of int
            Per root, the bins its field is needed for.

        Returns
        -------
        ndarray, shape (n_roots, n_bins)
        """
        from scipy.sparse.csgraph import dijkstra

        width = self._spacing / _PROBE_FRACTION
        centers = self._centers
        stops = [np.asarray(stop, dtype=np.intp) for stop in stop_bins]
        stops = [stop[stop != root] for stop, root in zip(stops, roots, strict=True)]
        reach = np.array(
            [
                float(np.linalg.norm(centers[stop] - centers[root], axis=1).max())
                if stop.size
                else 0.0
                for stop, root in zip(stops, roots, strict=True)
            ]
        )
        limit = 1.5 * reach + 2.0 * width
        graph = np.full((roots.size, self._n_bins), np.inf)
        pending = np.arange(roots.size)
        for attempt in range(4):
            if attempt == 3:
                limit[pending] = np.inf
            used = np.empty(roots.size)
            # Roots with limits within a factor of two share one search.
            with np.errstate(divide="ignore"):
                group_of = np.floor(np.log2(limit[pending]))
            for group in np.unique(group_of):
                rows = pending[group_of == group]
                used[rows] = limit[rows].max()
                graph[rows] = dijkstra(
                    self._adjacency,
                    directed=False,
                    indices=roots[rows],
                    limit=used[rows[0]],
                )
            still = []
            for row in pending:
                found = graph[row, stops[row]]
                if not np.all(np.isfinite(found)):
                    if np.isfinite(used[row]):
                        limit[row] = 4.0 * used[row]
                        still.append(row)
                    continue
                farthest = float(found.max(initial=0.0))
                need = (np.floor(farthest / width) + 1.0) * width * (1.0 + 1e-9)
                if need > used[row]:
                    limit[row] = need
                    still.append(row)
            pending = np.array(still, dtype=np.intp)
            if not pending.size:
                break
        return graph

    def _settle(
        self,
        nodes: NDArray[np.intp],
        in_src: NDArray[np.intp],
        in_ptr: NDArray[np.intp],
        settled: NDArray[np.bool_],
        length: NDArray[np.float64],
        anchor: NDArray[np.int32],
        lead: NDArray[np.int32],
        is_source: NDArray[np.bool_],
        rate: NDArray[np.float64] | None,
        root_of_field: NDArray[np.intp] | None,
    ) -> NDArray[np.intp]:
        """Choose the path of each node from its final nearer neighbors.

        Returns
        -------
        ndarray of int
            The nodes that have no final nearer neighbor yet (they wait).
        """
        n_bins = self._n_bins
        centers = self._centers
        nodes = np.sort(nodes)
        starts = in_ptr[nodes]
        counts = in_ptr[nodes + 1] - starts
        # Each in-edge (node, parent), with the node's slot in ``nodes``.
        slot = np.repeat(np.arange(nodes.size), counts)
        parent = in_src[np.repeat(starts, counts) + ragged_arange(counts)]
        final = settled[parent]
        slot, parent = slot[final], parent[final]
        ready = np.zeros(nodes.size, dtype=bool)
        ready[slot] = True
        waiting: NDArray[np.intp] = nodes[~ready]
        if root_of_field is not None and slot.size:
            # A visible root is the straight line, which no chain beats even
            # by rounding: it is chosen outright, so a node that sees its
            # root needs no other candidate.
            ready_at = np.flatnonzero(ready)
            straight = nodes[ready_at]
            straight_field = straight // n_bins
            straight_bin = straight - straight_field * n_bins
            root = root_of_field[straight_field]
            sees, cost = self._trace(
                centers[straight_bin], centers[root], straight_bin, root, rate
            )
            straight, root = straight[sees], root[sees]
            length[straight] = cost[sees] + length[straight_field[sees] * n_bins + root]
            anchor[straight] = root
            lead[straight] = straight_bin[sees]
            settled[straight] = True
            ready[ready_at[sees]] = False
            keep = ready[slot]
            slot, parent = slot[keep], parent[keep]
        if not slot.size:
            return waiting
        node = nodes[slot]
        field = node // n_bins
        node_bin = node - field * n_bins
        parent_bin = parent - field * n_bins
        # Along the graph edge to a nearer neighbor: always allowed.
        edge = np.linalg.norm(centers[node_bin] - centers[parent_bin], axis=1)
        if rate is not None:
            edge = edge * 0.5 * (rate[node_bin] + rate[parent_bin])
        edge_total = edge + length[parent]
        # Straight to a nearer neighbor's anchor, when visible (not the
        # root: these nodes do not see it).
        keys = np.unique(node * n_bins + anchor[parent].astype(np.intp))
        cand_node, cand_bin = keys // n_bins, keys % n_bins
        cand_field = cand_node // n_bins
        cand_node_bin = cand_node - cand_field * n_bins
        keep = cand_bin != cand_node_bin
        if root_of_field is not None:
            keep &= cand_bin != root_of_field[cand_field]
        cand_node, cand_bin, cand_field, cand_node_bin = (
            cand_node[keep],
            cand_bin[keep],
            cand_field[keep],
            cand_node_bin[keep],
        )
        visible, cost = self._trace(
            centers[cand_node_bin], centers[cand_bin], cand_node_bin, cand_bin, rate
        )
        cand_node, cand_bin = cand_node[visible], cand_bin[visible]
        cand_total = cost[visible] + length[cand_field[visible] * n_bins + cand_bin]
        # Rank per node: the lowest total, anchors before graph edges, then
        # the lower bin.
        all_node = np.r_[cand_node, node]
        all_bin = np.r_[cand_bin, parent_bin]
        all_total = np.r_[cand_total, edge_total]
        via_edge = np.r_[np.zeros(cand_node.size), np.ones(node.size)]
        order = np.lexsort((all_bin, via_edge, all_total, all_node))
        order = order[np.r_[True, np.diff(all_node[order]) != 0]]
        chosen_node = all_node[order]
        chosen_bin = all_bin[order]
        length[chosen_node] = all_total[order]
        anchor[chosen_node] = chosen_bin
        chosen_field = chosen_node // n_bins
        via = chosen_field * n_bins + chosen_bin
        lead[chosen_node] = np.where(
            is_source[via], chosen_node - chosen_field * n_bins, lead[via]
        )
        settled[chosen_node] = True
        return waiting


class _BoxCells(ABC):
    """Bins as the cells of an integer lattice, with box-sum tables of defects.

    :meth:`clean` says whether a box of lattice cells holds only bins, every
    two in neighboring cells joined. Then no stretch of a run of probes whose
    points all lie in the box's cells is blocked, provided a stretch (at most
    one probe spacing long) cannot reach past a neighboring cell
    (:meth:`steps_stay_near`): every point :meth:`_OpenPaths._crossing_blocked`
    locates lies on the stretch, so in a bin of the box, and every pair of
    bins it requires to be joined holds two points of the stretch, so they
    are one cell or two neighboring cells.

    Parameters
    ----------
    flat_to_active : ndarray of int, shape (prod(shape),)
        Active bin of each lattice cell in row-major order, -1 for none.
    shape : tuple of int
        Lattice cells along each axis.
    cell : ndarray of int, shape (n_bins, n_dims)
        Lattice cell of each bin.
    """

    def __init__(
        self,
        flat_to_active: NDArray[np.intp],
        shape: tuple[int, ...],
        cell: NDArray[np.intp],
    ) -> None:
        self.flat_to_active = flat_to_active
        self.shape = shape
        self.cell = cell
        # Box-sum tables for :meth:`clean`, built on first use.
        self._defects: NDArray[np.int32] | None = None
        self._defect_extents = np.zeros((0, len(shape)), dtype=np.intp)
        self._usable: bool | None = None

    def boxes(self, paths: _OpenPaths) -> bool:
        """Whether :meth:`clean` can stand in for probing ``paths``'s stretches."""
        if self._usable is None:
            self._usable = self.steps_stay_near(paths._spacing)
            if self._usable:
                self._build_defects(paths._connected)
        return self._usable

    @abstractmethod
    def steps_stay_near(self, spacing: float) -> bool:
        """Whether two points ``spacing`` apart are in one or neighboring cells."""

    @staticmethod
    @abstractmethod
    def neighbor_offsets(n_dims: int) -> list[tuple[int, ...]]:
        """Offsets to neighboring cells, one of each pair of opposites."""

    @abstractmethod
    def point_cells(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """First and last cell, per axis, of the cells a point may be located in.

        Returns
        -------
        low, high : ndarray of int, shape (n_points, n_dims)
        """

    @abstractmethod
    def missing(
        self, points: NDArray[np.float64], paths: _OpenPaths
    ) -> NDArray[np.intp]:
        """Indices of the probes that lie in no bin."""

    def _build_defects(
        self,
        connected: Callable[[NDArray[np.intp], NDArray[np.intp]], NDArray[np.bool_]],
    ) -> None:
        """Box-sum tables of inactive cells and of closed neighbor pairs.

        A closed pair is two active bins in neighboring cells (one of
        :meth:`neighbor_offsets` apart) not joined by an open connection. It
        lies in a box when its lowest corner cell ``m`` does and
        ``m + extent`` too, where the extent is 0 or 1 per axis, so one table
        per extent counts closed pairs at ``m`` (extent 0: inactive cells). Each table is padded
        with a leading zero on every axis and summed cumulatively along
        every axis.
        """
        shape = self.shape
        n_dims = len(shape)
        bins = self.flat_to_active.reshape(shape)
        by_extent = {(0,) * n_dims: bins < 0}
        for offset in self.neighbor_offsets(n_dims):
            here = tuple(
                slice(max(0, -o), n - max(0, o))
                for o, n in zip(offset, shape, strict=True)
            )
            there = tuple(
                slice(max(0, o), n - max(0, -o))
                for o, n in zip(offset, shape, strict=True)
            )
            u, v = bins[here], bins[there]
            both = (u >= 0) & (v >= 0)
            closed = np.zeros(u.shape, dtype=bool)
            closed[both] = ~connected(u[both], v[both])
            if not closed.any():
                continue
            extent = tuple(abs(o) for o in offset)
            table = by_extent.setdefault(extent, np.zeros(shape, dtype=bool))
            table[
                tuple(slice(0, n - e) for e, n in zip(extent, shape, strict=True))
            ] |= closed
        self._defects = self._summed(np.stack(list(by_extent.values())))
        self._defect_extents = np.array(list(by_extent), dtype=np.intp)

    @staticmethod
    def _summed(tables: NDArray[np.bool_]) -> NDArray[np.int32]:
        """Padded cumulative sums of stacked tables, flattened per table."""
        summed = np.zeros(
            (tables.shape[0], *(n + 1 for n in tables.shape[1:])), dtype=np.int32
        )
        summed[(slice(None), *(slice(1, None) for _ in tables.shape[1:]))] = tables
        for axis in range(1, summed.ndim):
            np.cumsum(summed, axis=axis, out=summed)
        return summed.reshape(tables.shape[0], -1)

    def clean(self, low: NDArray[np.intp], high: NDArray[np.intp]) -> NDArray[np.bool_]:
        """Whether each box of cells is all active bins, all joined pairwise.

        Parameters
        ----------
        low, high : ndarray of int, shape (n, n_dims)
            First and last cell of each box on every axis.

        Returns
        -------
        ndarray of bool, shape (n,)
            True when the box lies in the lattice, every cell in it is an
            active bin, and every two of them in neighboring cells are joined
            by an open connection.
        """
        assert self._defects is not None
        clean = np.zeros(low.shape[0], dtype=bool)
        inside = np.flatnonzero(
            np.all((low >= 0) & (high < np.asarray(self.shape)), axis=1)
        )
        if inside.size:
            totals = self._box_sums(
                self._defects, self._defect_extents, low[inside], high[inside]
            )
            clean[inside] = ~np.any(totals, axis=1)
        return clean

    def _box_sums(
        self,
        summed: NDArray[np.int32],
        extents: NDArray[np.intp],
        low: NDArray[np.intp],
        high: NDArray[np.intp],
    ) -> NDArray[np.int64]:
        """Per box and table, the count in cells ``low .. high - extent``.

        Returns
        -------
        ndarray of int, shape (n_boxes, n_tables)
        """
        shape = np.asarray(self.shape)
        strides = np.cumprod(np.r_[shape[1:] + 1, 1][::-1])[::-1]
        n_dims = shape.size
        first = [low[:, axis, None] * strides[axis] for axis in range(n_dims)]
        last = [
            (high[:, axis, None] + 1 - extents[:, axis]) * strides[axis]
            for axis in range(n_dims)
        ]
        tables = np.arange(summed.shape[0])
        total = np.zeros((low.shape[0], summed.shape[0]), dtype=np.int64)
        for corner in itertools.product((False, True), repeat=n_dims):
            index = sum(
                last[axis] if upper else first[axis]
                for axis, upper in enumerate(corner)
            )
            sign = (-1) ** (n_dims - sum(corner))
            total += sign * summed[tables, index]
        return total


class _GridCells(_BoxCells):
    """Grid cell of each bin on a rectilinear grid, for exact cell walks."""

    def __init__(
        self,
        edges: tuple[NDArray[np.float64], ...],
        flat_to_active: NDArray[np.intp],
        shape: tuple[int, ...],
        cell: NDArray[np.intp],
    ) -> None:
        super().__init__(flat_to_active, shape, cell)
        self.edges = edges

    def steps_stay_near(self, spacing: float) -> bool:
        # A stretch shorter than every cell cannot step over one, so the
        # cell walk resolves every stretch between bins.
        narrowest = min(float(np.diff(edge).min()) for edge in self.edges)
        return spacing * (1 + 1e-6) < narrowest

    @staticmethod
    def neighbor_offsets(n_dims: int) -> list[tuple[int, ...]]:
        # Cells that differ by at most one on every axis.
        return [
            offset
            for offset in itertools.product((-1, 0, 1), repeat=n_dims)
            if offset > (0,) * n_dims
        ]

    def point_cells(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        cells = self.half_open(points)
        return cells, cells

    def missing(
        self, points: NDArray[np.float64], paths: _OpenPaths
    ) -> NDArray[np.intp]:
        outside = np.flatnonzero(self.bin_of(self.half_open(points)) < 0)
        missing: NDArray[np.intp] = outside[paths._cells(points[outside]) < 0]
        return missing

    def half_open(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Half-open grid cell of each point per axis, as the locator finds it.

        Returns
        -------
        ndarray of int, shape (n_points, n_dims)
            ``-1`` below an axis's first edge, ``shape[d]`` at or above its
            last.
        """
        return np.column_stack(
            [
                np.digitize(points[:, axis], edge) - 1
                for axis, edge in enumerate(self.edges)
            ]
        ).astype(np.intp, copy=False)

    def bin_of(self, cells: NDArray[np.intp]) -> NDArray[np.intp]:
        """Active bin of each grid cell, -1 for an inactive cell or off the grid."""
        inside = np.all((cells >= 0) & (cells < np.asarray(self.shape)), axis=1)
        bins = np.full(cells.shape[0], -1, dtype=np.intp)
        bins[inside] = self.flat_to_active[
            np.ravel_multi_index(tuple(cells[inside].T), self.shape)
        ]
        return bins

    @classmethod
    def of(cls, locator: object, centers: NDArray[np.float64]) -> _GridCells | None:
        """The grid of a rectilinear locator whose bins are its active cells."""
        from neurospatial.geometry._locators import RectilinearLocator

        if not isinstance(locator, RectilinearLocator) or not centers.size:
            return None
        cell = np.column_stack(
            [
                np.searchsorted(edge, centers[:, axis], side="right") - 1
                for axis, edge in enumerate(locator.edges)
            ]
        )
        shape = np.asarray(locator.shape)
        if np.any((cell < 0) | (cell >= shape)):
            return None
        flat = np.ravel_multi_index(tuple(cell.T), locator.shape)
        if not np.array_equal(locator.flat_to_active[flat], np.arange(len(centers))):
            return None
        return cls(locator.edges, locator.flat_to_active, locator.shape, cell)

    def walk(
        self,
        a: NDArray[np.intp],
        b: NDArray[np.intp],
        p0: NDArray[np.float64],
        p1: NDArray[np.float64],
        connected: Callable[[NDArray[np.intp], NDArray[np.intp]], NDArray[np.bool_]],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Walk the cells a stretch crosses from bin ``a`` to bin ``b``.

        The stretch from ``p0`` (in ``a``) to ``p1`` (in ``b``) crosses each
        cell boundary between them in the order of its crossing times; two
        boundaries crossed at the same time (through an edge or a vertex)
        are one step. Each step must enter a bin joined to the last.

        Returns
        -------
        walked : ndarray of bool
            The pair is resolved here (its cells differ by at most one on
            every axis).
        blocked : ndarray of bool
            A walked stretch leaves the bins or crosses a closed connection.
        """
        start, stop = self.cell[a], self.cell[b]
        step = stop - start
        walked = np.all(np.abs(step) <= 1, axis=1)
        blocked = np.zeros(a.size, dtype=bool)
        rows = np.flatnonzero(walked)
        start, step, p0, p1 = start[rows], step[rows], p0[rows], p1[rows]
        n_dims = start.shape[1]
        # Crossing time of each boundary on the way, inf on axes not crossed.
        boundary = np.column_stack(
            [
                self.edges[axis][
                    np.maximum(start[:, axis], start[:, axis] + step[:, axis])
                ]
                for axis in range(n_dims)
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            time = np.where(step != 0, (boundary - p0) / (p1 - p0), np.inf)
        order = np.argsort(time, axis=1, kind="stable")
        times = np.take_along_axis(time, order, axis=1)
        # Crossings this close in time are one step through an edge or a
        # vertex, so mirror-image stretches take the same steps.
        tied = np.diff(times, axis=1) <= 1e-9
        committed = a[rows].copy()
        pending = start.copy()
        index = np.arange(rows.size)
        bad = np.zeros(rows.size, dtype=bool)
        for k in range(n_dims):
            moving = np.isfinite(times[:, k])
            axis = order[:, k]
            pending[index[moving], axis[moving]] += step[index[moving], axis[moving]]
            last = ~np.isfinite(times[:, k + 1]) if k + 1 < n_dims else True
            commit = moving & (last | ~tied[:, k] if k + 1 < n_dims else moving)
            if not commit.any():
                continue
            enter = self.flat_to_active[
                np.ravel_multi_index(tuple(pending[commit].T), self.shape)
            ]
            fails = enter < 0
            fails[~fails] = ~connected(committed[commit][~fails], enter[~fails])
            bad[commit] |= fails
            committed[commit] = np.where(enter >= 0, enter, committed[commit])
        blocked[rows] = bad
        return walked, blocked


class _HexCells(_BoxCells):
    """Axial lattice cell of each bin of a hexagonal layout, for clean boxes.

    Cells are indexed by axial coordinates ``(q - q_min, r)`` of the pointy-top
    lattice of :class:`~neurospatial.geometry._locators.HexLocator`, so a
    segment's fractional axial coordinates are linear along it.
    """

    # The locator gives a point a hexagon whose center is as near as the
    # nearest one, within its rounding tolerance, so the point lies in that
    # hexagon, whose points are at most 2/3 from its center in fractional
    # axial coordinates (its corners, on one axis). The margin covers the
    # tolerance, which is far smaller on lattices of fewer than _HEX_SPAN
    # hexagons across.
    _REACH = 2.0 / 3.0 + 1e-6
    _HEX_SPAN = 1e6

    def __init__(
        self,
        locator: HexLocator,
        flat_to_active: NDArray[np.intp],
        shape: tuple[int, ...],
        cell: NDArray[np.intp],
        q_min: int,
    ) -> None:
        super().__init__(flat_to_active, shape, cell)
        self._locator = locator
        self._q_min = q_min

    def steps_stay_near(self, spacing: float) -> bool:
        # Hexagons that are not neighbors are at least one side length (the
        # radius) apart: the two common neighbors of a pair two steps apart
        # share a side between them.
        return spacing * (1 + 1e-6) < self._locator.radius

    @staticmethod
    def neighbor_offsets(n_dims: int) -> list[tuple[int, ...]]:
        return [(0, 1), (1, -1), (1, 0)]

    def point_cells(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        locator = self._locator
        q, r, _ = cartesian_to_fractional_cube(
            points[:, 0] - locator.offset_x,
            points[:, 1] - locator.offset_y,
            locator.radius,
        )
        axial = np.column_stack([q - self._q_min, r])
        low = np.ceil(axial - self._REACH).astype(np.intp)
        high = np.floor(axial + self._REACH).astype(np.intp)
        return low, high

    def missing(
        self, points: NDArray[np.float64], paths: _OpenPaths
    ) -> NDArray[np.intp]:
        return np.flatnonzero(paths._cells(points) < 0)

    @classmethod
    def of(cls, locator: object, centers: NDArray[np.float64]) -> _HexCells | None:
        """The lattice of a hexagonal locator whose bins are its active cells."""
        from neurospatial.geometry._hex_lookup import (
            axial_to_offset_bin_indices,
            round_fractional_cube_to_integer_axial,
        )
        from neurospatial.geometry._locators import HexLocator, SubsetLocator

        # Subsets (cells clipped by an arena) map the lattice's bins on.
        subsets = []
        while isinstance(locator, SubsetLocator):
            subsets.append(locator.parent_to_child)
            locator = locator.base
        if not isinstance(locator, HexLocator) or not centers.size:
            return None
        flat_to_active = locator.flat_to_active
        for parent_to_child in reversed(subsets):
            flat_to_active = np.where(
                flat_to_active >= 0,
                parent_to_child[np.maximum(flat_to_active, 0)],
                -1,
            )
        n_rows, n_cols = locator.centers_shape
        width = np.sqrt(3.0) * locator.radius
        extent = max(abs(locator.offset_x), abs(locator.offset_y)) + width * (
            max(n_rows, n_cols) + 1
        )
        if extent > cls._HEX_SPAN * locator.radius:
            return None
        q, r, s = cartesian_to_fractional_cube(
            centers[:, 0] - locator.offset_x,
            centers[:, 1] - locator.offset_y,
            locator.radius,
        )
        q_bin, r_bin = round_fractional_cube_to_integer_axial(q, r, s)
        flat = axial_to_offset_bin_indices(q_bin, r_bin, n_cols, n_rows)
        if np.any(flat < 0) or not np.array_equal(
            flat_to_active[flat], np.arange(len(centers))
        ):
            return None
        # Every hexagon of the lattice, in axial coordinates (odd-r offset
        # rows are shifted half a hexagon right).
        row, col = np.divmod(np.arange(n_rows * n_cols), n_cols)
        q_all = col - (row - (row & 1)) // 2
        q_min = int(q_all.min())
        shape = (int(q_all.max()) - q_min + 1, n_rows)
        table = np.full(shape, -1, dtype=np.intp)
        table[q_all - q_min, row] = flat_to_active
        cell = np.column_stack([q_bin - q_min, r_bin]).astype(np.intp)
        return cls(locator, table.ravel(), shape, cell, q_min)


class _MeshCores:
    """Deep interiors of a mesh's triangles, where the locator's answer is known.

    :class:`~neurospatial.geometry._locators.MeshLocator` gives a point the
    lowest-numbered stored triangle whose barycentric test (within a
    tolerance of 100 machine epsilons) accepts it, or -1 off its boundary
    polygon. A point at least ``4 * margin`` inside every edge of a triangle
    (``margin`` a billionth of the mesh's coordinate scale, far above that
    tolerance and the rounding of the coordinates) is the triangle's when:

    - every other stored triangle whose lookup box meets the triangle's is
      separated from it with room to spare: an edge line of the triangle
      has all of the other's vertices within ``margin`` of its outside, or
      an edge line of the other has all of the triangle's ``2 * margin``
      core outside it by ``2 * margin``, so no point of the deep interior
      passes the other's test;
    - the boundary polygon covers the ``2 * margin`` core;
    - the triangle is not much thinner than the mesh's rounding (its
      altitudes differ by less than ``_ASPECT``).

    Triangles that fail any check have no deep interior here. A table of
    square cells, each inside one deep interior with room for rounding,
    gives most points their bin with one lookup.

    Parameters
    ----------
    entry : ndarray of int, shape (n_bins,)
        Row of each bin's triangle in the transforms, -1 for no deep interior.
    inverse : ndarray, shape (n_rows, 2, 2)
    origin : ndarray, shape (n_rows, 2)
        The locator's barycentric transforms.
    threshold : ndarray, shape (n_rows, 3)
        Least barycentric coordinate of the deep interior.
    """

    _ASPECT = 1e5
    # Cells of the lookup table per smallest inradius, and at most this many.
    _CELLS_PER_INRADIUS = 8
    _MAX_CELLS = 4_000_000

    def __init__(
        self,
        entry: NDArray[np.intp],
        inverse: NDArray[np.float64],
        origin: NDArray[np.float64],
        threshold: NDArray[np.float64],
    ) -> None:
        self.entry = entry
        self.inverse = inverse
        self.origin = origin
        self.threshold = threshold
        self._table = np.full((0, 0), -1, dtype=np.intp)
        self._low = np.zeros(2)
        self._width = 1.0

    def coordinates(
        self, points: NDArray[np.float64], rows: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Barycentric coordinates of each point in triangle ``rows``, shape (n, 3)."""
        coords = np.einsum("kij,kj->ki", self.inverse[rows], points - self.origin[rows])
        return np.column_stack((coords, 1.0 - coords[:, 0] - coords[:, 1]))

    def deep(
        self, points: NDArray[np.float64], bins: NDArray[np.intp]
    ) -> NDArray[np.bool_]:
        """Whether each point lies in the deep interior of its bin's triangle."""
        rows = np.where(bins >= 0, self.entry[np.maximum(bins, 0)], -1)
        found = np.zeros(points.shape[0], dtype=bool)
        known = np.flatnonzero(rows >= 0)
        if known.size:
            coords = self.coordinates(points[known], rows[known])
            found[known] = np.all(coords >= self.threshold[rows[known]], axis=1)
        return found

    def lookup(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Bin of each point from the table of cells, -1 where it is not known."""
        cell = np.floor((points - self._low) / self._width)
        shape = np.asarray(self._table.shape)
        inside = np.all((cell >= 0) & (cell < shape), axis=1)
        located = np.full(points.shape[0], -1, dtype=np.intp)
        index = cell[inside].astype(np.intp)
        located[inside] = self._table[index[:, 0], index[:, 1]]
        return located

    @classmethod
    def of(
        cls, locator: object, n_bins: int, indexes: LookupIndexes | None
    ) -> _MeshCores | None:
        """The deep interiors of a mesh locator's bins, or None for other locators."""
        import shapely

        from neurospatial.geometry._locators import (
            MeshLocator,
            _lookup_index,
            _triangle_index,
        )

        if not isinstance(locator, MeshLocator) or not n_bins:
            return None
        index = _lookup_index(
            locator, indexes, lambda: _triangle_index(locator.points, locator.simplices)
        )
        vertices = locator.points[locator.simplices[index.triangles]]
        # Edge j is opposite vertex j; the locator's coordinates are those
        # of vertices 0, 1 and 2.
        edge = np.linalg.norm(vertices[:, [2, 0, 1]] - vertices[:, [1, 2, 0]], axis=2)
        first = vertices[:, 0] - vertices[:, 2]
        second = vertices[:, 1] - vertices[:, 2]
        area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
        altitude = 2.0 * area[:, None] / edge
        span = np.ptp(vertices, axis=1).sum(axis=1)
        margin = 1e-9 * (float(np.abs(locator.points).max()) + float(span.max()))
        threshold = 4.0 * margin / altitude
        core = 2.0 * margin / altitude
        safe = (altitude.max(axis=1) < cls._ASPECT * altitude.min(axis=1)) & (
            threshold.sum(axis=1) < 0.5
        )
        # Vertex j of the 2 * margin core: coordinate j takes what the other
        # two leave.
        weights = np.repeat(core[:, None, :], 3, axis=1)
        diagonal = np.arange(3)
        weights[:, diagonal, diagonal] = 0.0
        weights[:, diagonal, diagonal] = 1.0 - weights.sum(axis=2)
        core_vertices = np.einsum("nkj,njd->nkd", weights, vertices)
        boxes = index.tree.geometries
        left, right = index.tree.query(boxes)
        other = left != right
        left, right = left[other], right[other]

        def inside(rows, points, row_scale):
            # Signed distance inside each edge of triangle ``rows``,
            # shape (n, n_points, 3).
            flat = points.reshape(-1, 2)
            rep = np.repeat(rows, points.shape[1])
            coords = np.einsum(
                "kij,kj->ki", index.inverse[rep], flat - index.origin[rep]
            )
            coords = np.column_stack((coords, 1.0 - coords.sum(axis=1)))
            return (
                coords.reshape(points.shape[0], points.shape[1], 3)
                * row_scale[:, None, :]
            )

        beyond_own = inside(left, vertices[right], altitude[left]).max(axis=1)
        beyond_other = inside(right, core_vertices[left], altitude[right]).max(axis=1)
        separated = np.any(beyond_own <= margin, axis=1) | np.any(
            beyond_other <= -2.0 * margin, axis=1
        )
        safe[left[~separated]] = False
        if locator.boundary is not None:
            rows = np.flatnonzero(safe)
            safe[rows] = shapely.covers(
                locator.boundary, shapely.polygons(core_vertices[rows])
            )
        active = locator.simplex_to_active[index.triangles]
        entry = np.full(n_bins, -1, dtype=np.intp)
        keep = np.flatnonzero(safe & (active >= 0))
        entry[active[keep]] = keep
        cores = cls(entry, index.inverse, index.origin, threshold)
        cores._build_table(vertices, keep, area, edge)
        return cores

    def _build_table(
        self,
        vertices: NDArray[np.float64],
        rows: NDArray[np.intp],
        area: NDArray[np.float64],
        edge: NDArray[np.float64],
    ) -> None:
        """Cells whose four corners are twice as deep as a deep interior."""
        if not rows.size:
            return
        low = vertices[rows].min(axis=(0, 1))
        high = vertices[rows].max(axis=(0, 1))
        inradius = 2.0 * area[rows] / edge[rows].sum(axis=1)
        width = float(inradius.min()) / self._CELLS_PER_INRADIUS
        extent = high - low
        width = max(width, float(np.sqrt(np.prod(extent) / self._MAX_CELLS)))
        shape = np.maximum(np.ceil(extent / width).astype(np.intp), 1)
        table = np.full(tuple(shape), -1, dtype=np.intp)
        first = np.floor((vertices[rows].min(axis=1) - low) / width).astype(np.intp)
        last = np.minimum(
            np.floor((vertices[rows].max(axis=1) - low) / width).astype(np.intp),
            shape - 1,
        )
        count = last - first + 1
        n_cells = count[:, 0] * count[:, 1]
        owner = np.repeat(np.arange(rows.size), n_cells)
        k = ragged_arange(n_cells)
        cell = first[owner] + np.column_stack(
            (k // count[owner, 1], k % count[owner, 1])
        )
        corners = np.stack(
            [
                low + (cell + np.array(offset)) * width
                for offset in itertools.product((0, 1), repeat=2)
            ],
            axis=1,
        )
        tri = np.repeat(rows[owner], 4)
        coords = self.coordinates(corners.reshape(-1, 2), tri).reshape(-1, 4, 3)
        deep = np.all(
            coords >= 2.0 * self.threshold[rows[owner]][:, None, :], axis=(1, 2)
        )
        bin_of_row = np.full(self.inverse.shape[0], -1, dtype=np.intp)
        bin_of_row[self.entry[self.entry >= 0]] = np.flatnonzero(self.entry >= 0)
        table[cell[deep, 0], cell[deep, 1]] = bin_of_row[rows[owner[deep]]]
        self._table, self._low, self._width = table, low, width


class PairDistances(ABC):
    """Shortest path lengths between pairs of one set of positions.

    Made by :meth:`EnvironmentPaths.pairwise`, which locates the positions
    once and prepares what every pair needs; results equal
    :meth:`EnvironmentPaths.between` up to rounding (an open environment's
    tables sum the same terms in another order).
    """

    def __init__(self, paths: EnvironmentPaths, points: NDArray[np.float64]) -> None:
        self._points = points
        self._bins = paths.bins(points)

    def between(
        self, first: NDArray[np.intp], second: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Path length from ``points[first[k]]`` to ``points[second[k]]``.

        Parameters
        ----------
        first, second : ndarray of int, shape (n_pairs,)
            Row indices into the positions.

        Returns
        -------
        ndarray, shape (n_pairs,)
            NaN where either position is missing or outside the
            environment; ``inf`` between disconnected parts.
        """
        out = np.full(first.size, np.nan)
        ok = np.flatnonzero((self._bins[first] >= 0) & (self._bins[second] >= 0))
        if ok.size:
            out[ok] = self._between(first[ok], second[ok])
        return out

    @abstractmethod
    def _between(
        self, first: NDArray[np.intp], second: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """:meth:`between` for pairs whose two positions are both located."""


class _OpenPairDistances(PairDistances):
    """Pair distances in an open environment, from per-bin-pair tables.

    The fields of every bin the positions visit are reduced to tables once,
    so each pair is a few lookups.
    """

    def __init__(self, paths: _OpenPaths, points: NDArray[np.float64]) -> None:
        super().__init__(paths, points)
        self._paths = paths
        centers = self._centers = paths._centers
        n_bins, n_dims = centers.shape
        roots = np.unique(self._bins[self._bins >= 0])
        # Table row of each target bin.
        self._row_of = np.full(n_bins, -1, dtype=np.intp)
        self._row_of[roots] = np.arange(roots.size)
        shape = (roots.size, n_bins)
        # Straight: same bin, connected bins, or the bin center sees the
        # target bin center.
        self._straight = np.empty(shape, dtype=bool)
        self._corner = np.empty(shape, dtype=np.int32)
        self._lead = np.empty(shape, dtype=np.int32)
        # Path length from the corner on, without its last leg to the target
        # bin center: d = |p - corner| + rest + |lead - target|.
        self._rest = np.empty(shape)
        # Targets whose fields are kept at once, and whose center
        # differences are taken at once.
        fields = paths._fields
        step = max(1, _PAIR_TABLE_VALUES // max(n_bins * n_dims, 1))
        for lo in range(0, roots.size, fields.limit):
            part = roots[lo : lo + fields.limit]
            fields.ensure(paths, part)
            rows = fields.row_of_root[part]
            table = slice(lo, lo + part.size)
            anchor = fields.anchor[rows]
            straight = self._straight[table]
            straight[:] = anchor == part[:, None]
            straight[np.arange(part.size), part] = True
            neighbors = paths._neighbors[part]
            has = neighbors >= 0
            straight[np.nonzero(has)[0], neighbors[has]] = True
            corner = self._corner[table]
            np.maximum(anchor, 0, out=corner)
            lead = self._lead[table]
            np.maximum(fields.lead[rows[:, None], corner], 0, out=lead)
            rest = self._rest[table]
            rest[:] = fields.length[rows[:, None], corner]
            for k in range(0, part.size, step):
                rest[k : k + step] -= np.linalg.norm(
                    centers[lead[k : k + step]] - centers[part[k : k + step], None, :],
                    axis=2,
                )
            rest[anchor < 0] = np.inf

    def _between(
        self, first: NDArray[np.intp], second: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        a, b = self._bins[first], self._bins[second]
        row = self._row_of[b]
        starts, ends = self._points[first], self._points[second]
        centers = self._centers
        corner = self._corner[row, a].astype(np.intp)
        lead = self._lead[row, a].astype(np.intp)
        rest = self._rest[row, a]
        around = (
            np.linalg.norm(starts - centers[corner], axis=1)
            + rest
            + np.linalg.norm(centers[lead] - ends, axis=1)
        )
        # The tables hold each path between bin centers; the legs at the
        # positions are checked as for single pairs.
        distances, _ = self._paths._route(
            starts,
            a,
            ends,
            b,
            self._straight[row, a],
            corner,
            rest + np.linalg.norm(centers[lead] - centers[b], axis=1),
            lead,
            around,
        )
        return distances


class _TrackPairDistances(PairDistances):
    """Pair distances along a track, from each position's projection."""

    def __init__(self, paths: _TrackPaths, points: NDArray[np.float64]) -> None:
        super().__init__(paths, points)
        self._paths = paths
        located = np.flatnonzero(self._bins >= 0)
        self._segment = np.zeros(points.shape[0], dtype=np.intp)
        self._offset = np.full(points.shape[0], np.nan)
        self._segment[located], self._offset[located] = paths.project(points[located])

    def _between(
        self, first: NDArray[np.intp], second: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        return self._paths._pair(
            self._segment[first],
            self._offset[first],
            self._segment[second],
            self._offset[second],
        )


class _RootFields:
    """Kept single-bin fields, stacked so pairs gather from many at once.

    The fields fill the first ``n_kept`` rows of arrays whose capacity
    doubles as needed, up to ``limit`` rows (``_FIELD_CACHE_BYTES``).
    """

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.limit = max(1, _FIELD_CACHE_BYTES // (16 * max(n_bins, 1)))
        self.row_of_root = np.full(n_bins, -1, dtype=np.intp)
        self.n_kept = 0
        self.length = np.empty((0, n_bins))
        self.anchor = np.empty((0, n_bins), dtype=np.int32)
        self.lead = np.empty((0, n_bins), dtype=np.int32)

    def ensure(self, paths: _OpenPaths, roots: NDArray[np.intp]) -> None:
        """Grow and keep the field of every root not yet kept.

        Parameters
        ----------
        paths : _OpenPaths
            The engine that grows the fields.
        roots : ndarray of int, shape (n_roots,)
            Distinct bins, at most ``limit`` of them. When the kept fields
            and the missing ones would exceed ``limit``, only the kept
            fields of ``roots`` stay.
        """
        missing = roots[self.row_of_root[roots] < 0]
        if not missing.size:
            return
        if self.n_kept + missing.size > self.limit:
            # Start over, keeping only what this request needs.
            needed = roots[self.row_of_root[roots] >= 0]
            rows = self.row_of_root[needed]
            for stack in (self.length, self.anchor, self.lead):
                stack[: needed.size] = stack[rows]
            self.row_of_root[:] = -1
            self.row_of_root[needed] = np.arange(needed.size)
            self.n_kept = needed.size
        size = self.n_kept + missing.size
        if size > self.length.shape[0]:
            capacity = min(self.limit, max(size, 2 * self.length.shape[0]))
            n = self.n_kept
            length = np.empty((capacity, self.n_bins))
            anchor = np.empty((capacity, self.n_bins), dtype=np.int32)
            lead = np.empty((capacity, self.n_bins), dtype=np.int32)
            length[:n], anchor[:n], lead[:n] = (
                self.length[:n],
                self.anchor[:n],
                self.lead[:n],
            )
            self.length, self.anchor, self.lead = length, anchor, lead
        for lo in range(0, missing.size, _FIELD_BATCH):
            batch = missing[lo : lo + _FIELD_BATCH]
            start, stop = self.n_kept, self.n_kept + batch.size
            self.length[start:stop], self.anchor[start:stop], self.lead[start:stop] = (
                paths._grow(list(batch[:, None]), single_roots=True)
            )
            self.row_of_root[batch] = np.arange(start, stop)
            self.n_kept = stop


class _TrackPaths(EnvironmentPaths):
    """Distances along a track graph between projections onto the track."""

    def __init__(self, env: Environment) -> None:
        from scipy.sparse import csr_array
        from scipy.sparse.csgraph import dijkstra

        super().__init__(env)
        linearization = env.binning.linearization
        assert linearization is not None  # environment_paths dispatches on it
        graph = linearization.graph
        nodes = list(graph.nodes)
        index = {node: i for i, node in enumerate(nodes)}
        position = np.array([graph.nodes[n]["pos"] for n in nodes], dtype=float)
        # Segments in linear order, oriented as the graph stores each edge
        # (the orientation the linear coordinate uses).
        stored = {frozenset(edge): edge for edge in graph.edges()}
        edges = [stored[frozenset(edge)] for edge in linearization.edge_order]
        self._u = np.array([index[e[0]] for e in edges], dtype=np.intp)
        self._v = np.array([index[e[1]] for e in edges], dtype=np.intp)
        self._first = position[self._u]
        self._direction = position[self._v] - self._first
        self._length = np.linalg.norm(self._direction, axis=1)
        self._scale = float(np.max(np.abs(position))) if position.size else 1.0
        n_nodes = len(nodes)
        matrix = csr_array(
            (
                np.r_[self._length, self._length],
                (np.r_[self._u, self._v], np.r_[self._v, self._u]),
            ),
            shape=(n_nodes, n_nodes),
        )
        self._node_distance = dijkstra(matrix, directed=False)

    def pairwise(self, points: NDArray[np.float64]) -> PairDistances:
        return _TrackPairDistances(self, points)

    def from_bins(self, source_bins: NDArray[np.intp]) -> NDArray[np.float64]:
        source_bins = np.asarray(source_bins, dtype=np.intp)
        return self._from_bins_at(source_bins, self._centers, np.arange(self._n_bins))

    def costs_from_bins(
        self, source_bins: NDArray[np.intp], rate: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Least travel cost along the track's bin graph.

        A track's bins lie on its centerline and its bin graph runs along
        it, so the bin-graph path is the along-track path.
        """
        from scipy.sparse.csgraph import dijkstra

        cost: NDArray[np.float64] = dijkstra(
            _rate_weighted(self._adjacency, rate),
            directed=False,
            indices=source_bins,
            min_only=True,
        )
        return cost

    def _to_point(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        target: NDArray[np.float64],
        root: int,
    ) -> NDArray[np.float64]:
        return self._along(points, np.broadcast_to(target, points.shape))

    def _between(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
        end_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        return self._along(starts, ends)

    def _directions(
        self,
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
        target: NDArray[np.float64],
        root: int,
    ) -> NDArray[np.float64]:
        """Unit vector along the track toward ``target`` at each point.

        NaN at the target and at a point whose projection is a node from
        which the route leaves along another segment.
        """
        out = np.full(points.shape, np.nan)
        seg, off = self.project(points)
        seg_t, off_t = self.project(target[None, :])
        nodes, reach = self._ends(seg, off)
        nodes_t, reach_t = self._ends(seg_t, off_t)
        # Cost of leaving through each end of the point's segment.
        via = reach + (self._node_distance[nodes][:, :, nodes_t[0]] + reach_t[0]).min(
            axis=2
        )
        toward_v = via[:, 1] < via[:, 0]
        same = seg == seg_t[0]
        direct = np.abs(off - off_t[0])
        along_edge = same & (direct <= via.min(axis=1))
        toward_v = np.where(along_edge, off_t[0] > off, toward_v)
        sign = np.where(toward_v, 1.0, -1.0)
        remaining = np.where(toward_v, reach[:, 1], reach[:, 0])
        if along_edge.any():
            remaining[along_edge] = direct[along_edge]
        unit = (
            self._direction[seg]
            / np.where(self._length[seg] > 0, self._length[seg], np.inf)[:, None]
        )
        defined = (remaining > 0) & (self._length[seg] > 0)
        out[defined] = sign[defined, None] * unit[defined]
        return out

    def _from_bins_at(
        self,
        source_bins: NDArray[np.intp],
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        seg_p, off_p = self.project(points)
        seg_s, off_s = self.project(self._centers[source_bins])
        n_points, n_sources = points.shape[0], source_bins.size
        nearest: NDArray[np.float64] = (
            self._pair(
                np.repeat(seg_p, n_sources),
                np.repeat(off_p, n_sources),
                np.tile(seg_s, n_points),
                np.tile(off_s, n_points),
            )
            .reshape(n_points, n_sources)
            .min(axis=1)
        )
        return nearest

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Nearest segment and distance along it from its first node.

        A point as near, within rounding, to several segments takes the one
        first in the track's linear order, as ``Environment.to_linear`` does.
        """
        from neurospatial.geometry._locators import _coordinate_roundoff

        squared = self._length**2
        with np.errstate(divide="ignore", invalid="ignore"):
            along = (
                np.einsum(
                    "sd,psd->ps", self._direction, points[:, None, :] - self._first
                )
                / squared
            )
        along = np.clip(np.where(squared > 0, along, 0.0), 0.0, 1.0)
        projected = self._first + along[:, :, None] * self._direction
        distance = np.linalg.norm(points[:, None, :] - projected, axis=2)
        tolerance = _coordinate_roundoff(points, self._scale)
        tied = distance <= (distance.min(axis=1) + tolerance)[:, None]
        segment = np.argmax(tied, axis=1)
        rows = np.arange(points.shape[0])
        return segment, along[rows, segment] * self._length[segment]

    def _ends(
        self, segment: NDArray[np.intp], offset: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """End nodes of each segment, shape (n, 2), and the distance to each."""
        nodes = np.c_[self._u[segment], self._v[segment]]
        reach = np.c_[offset, self._length[segment] - offset]
        return nodes, reach

    def _pair(
        self,
        seg_a: NDArray[np.intp],
        off_a: NDArray[np.float64],
        seg_b: NDArray[np.intp],
        off_b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Along-track distance between two projected positions."""
        nodes_a, reach_a = self._ends(seg_a, off_a)
        nodes_b, reach_b = self._ends(seg_b, off_b)
        via = (
            reach_a[:, :, None]
            + self._node_distance[nodes_a[:, :, None], nodes_b[:, None, :]]
            + reach_b[:, None, :]
        ).reshape(seg_a.size, 4)
        best: NDArray[np.float64] = via.min(axis=1)
        same = seg_a == seg_b
        best[same] = np.minimum(best[same], np.abs(off_a[same] - off_b[same]))
        return best

    def _along(
        self, starts: NDArray[np.float64], ends: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Along-track distance between the projections of each pair."""
        seg_a, off_a = self.project(starts)
        seg_b, off_b = self.project(ends)
        return self._pair(seg_a, off_a, seg_b, off_b)
