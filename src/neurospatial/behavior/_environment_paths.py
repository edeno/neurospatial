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
   bin center. Visibility is decided between the two bin centers, so near a
   wall end a position can be up to a bin's width from where the straight
   line actually becomes blocked. Two positions in one bin or in two
   connected bins are always the straight line apart. Toward several source
   bins at once, a position leaves for the anchor of its own bin or, when
   visible from it, of a neighboring bin.

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

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import ragged_arange
from neurospatial.environment._cache import CacheKey
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from scipy.sparse import csr_array

    from neurospatial.environment import Environment
    from neurospatial.geometry._binning import TrackLinearization

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
# Probe points located per call to ``Environment.bin_at``.
_PROBE_CHUNK = 1_000_000
# Soft memory budget of the per-target fields an environment keeps.
_FIELD_CACHE_BYTES = 256_000_000
# Targets whose fields are grown together, bounding the working memory.
_FIELD_BATCH = 128
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
    return env._cache.get(_PATHS_KEY, lambda: EnvironmentPaths(env))


_PATHS_KEY: CacheKey[EnvironmentPaths] = CacheKey("environment paths")


class EnvironmentPaths:
    """Shortest path lengths inside one environment, with cached fields.

    Use :func:`environment_paths` to get the shared instance of an
    environment.

    Parameters
    ----------
    env : Environment
        An allocentric Cartesian environment.
    """

    def __init__(self, env: Environment) -> None:
        from neurospatial.ops.distance import _bin_adjacency

        self._env = env
        self._centers = np.asarray(env.bin_centers, dtype=float)
        n_bins = self._centers.shape[0]
        self._n_bins = n_bins
        binning = env.binning
        self._locator = binning.locator
        self._track = (
            _TrackPaths(binning.linearization)
            if binning.linearization is not None
            else None
        )
        adjacency = _bin_adjacency(
            env.connectivity, weight="distance", operation="geodesic path length"
        )
        adjacency.sum_duplicates()
        adjacency.sort_indices()
        self._adjacency: csr_array = adjacency
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
        # Bins this close share a face: a straight stretch between them
        # stays in the two.
        self._nearest = float(lengths.min()) * (1 + 1e-6) if lengths.size else np.inf
        self._fields = _RootFields(n_bins)
        self._source_fields: dict[
            tuple[int, ...], tuple[NDArray[np.float64], NDArray[np.int32]]
        ] = {}

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def bins(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Bin of each point, -1 outside the environment or for NaN rows."""
        if points.shape[0] == 0:
            return np.zeros(0, dtype=np.intp)
        return np.asarray(self._env.bin_at(points), dtype=np.intp)

    def _cells(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Bin whose cell holds each point, -1 in no bin's cell.

        Probes use the bins' own cells: a straight segment between bin
        centers stays inside the environment when it stays inside the union
        of the bins' cells.
        """
        return np.asarray(self._locator.locate(points, self._env._cache), dtype=np.intp)

    def to_point(
        self, points: NDArray[np.float64], target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Shortest path length inside the environment from each point to ``target``.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.
        target : ndarray, shape (n_dims,)
            A finite position inside the environment (the caller checks).

        Returns
        -------
        ndarray, shape (n_points,)
            Path lengths; ``inf`` for points disconnected from the target.
        """
        if self._track is not None:
            return self._track.between(
                points, np.broadcast_to(target, points.shape), self.bins(points)
            )
        root = int(self.bins(target[None, :])[0])
        point_bins = self.bins(points)
        out = np.full(points.shape[0], np.nan)
        rows = np.flatnonzero(point_bins >= 0)
        fields = self._fields
        fields.ensure(self, np.array([root]))
        out[rows] = self._join(
            points[rows],
            point_bins[rows],
            np.broadcast_to(target, (rows.size, target.size)),
            np.full(rows.size, root),
            (fields.length, fields.anchor, fields.lead),
            np.full(rows.size, fields.row_of_root[root]),
        )
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
        if self._track is not None:
            out = self._track.between(starts, ends, start_bins)
            out[end_bins < 0] = np.nan
            return out
        out = np.full(starts.shape[0], np.nan)
        rows = np.flatnonzero((start_bins >= 0) & (end_bins >= 0))
        a, b = start_bins[rows], end_bins[rows]
        # Points in one bin or in two connected bins see each other.
        straight = (a == b) | self._connected(a, b)
        out[rows[straight]] = np.linalg.norm(
            ends[rows[straight]] - starts[rows[straight]], axis=1
        )
        rest = rows[~straight]
        if not rest.size:
            return out
        a, b = start_bins[rest], end_bins[rest]
        fields = self._fields
        kept = fields.row_of_root[b] >= 0
        if kept.any():
            rows_kept = fields.row_of_root[b[kept]]
            out[rest[kept]] = self._join(
                starts[rest[kept]],
                a[kept],
                ends[rest[kept]],
                b[kept],
                (fields.length, fields.anchor, fields.lead),
                rows_kept,
            )
        # Other targets: grow each field only as far as its start bins.
        grow = np.flatnonzero(~kept)
        if grow.size:
            roots, field_rows = np.unique(b[grow], return_inverse=True)
            field_rows = field_rows.ravel()
            parts = []
            for lo in range(0, roots.size, _FIELD_BATCH):
                batch = np.arange(lo, min(lo + _FIELD_BATCH, roots.size))
                parts.append(
                    self._grow(
                        [roots[i : i + 1] for i in batch],
                        single_roots=True,
                        stop_bins=[np.unique(a[grow][field_rows == i]) for i in batch],
                    )
                )
            grown = tuple(np.vstack([part[k] for part in parts]) for k in range(3))
            out[rest[grow]] = self._join(
                starts[rest[grow]],
                a[grow],
                ends[rest[grow]],
                b[grow],
                grown,  # type: ignore[arg-type]
                field_rows,
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

    def pairwise(self, points: NDArray[np.float64]) -> PairDistances:
        """Distances between many pairs of the same positions, ``(i, j)``.

        Locates the positions once and grows the field of every bin they
        lie in together, so each later call is a lookup. For analyses that
        compare every sample with many others, such as mean square
        displacement.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.

        Returns
        -------
        PairDistances
        """
        return PairDistances(self, points)

    def prepare(self, bins: NDArray[np.intp]) -> None:
        """Grow and keep the fields of every target bin in ``bins`` at once.

        Callers that will measure paths to many targets (every pair of a
        trajectory) call this once so the fields are grown together.
        """
        if self._track is None:
            self._fields.ensure(self, np.unique(bins[bins >= 0]))

    def directions(
        self, points: NDArray[np.float64], target: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Direction in which the shortest path from each point leaves it.

        Parameters
        ----------
        points : ndarray, shape (n_points, n_dims)
            Positions; NaN rows and rows outside the environment give NaN.
        target : ndarray, shape (n_dims,)
            A finite position inside the environment.

        Returns
        -------
        ndarray, shape (n_points, n_dims)
            Unit vectors toward the path's first waypoint: the target itself
            when the straight line is the path, else the first corner it
            goes around. On a track, the direction along the track toward
            the target. NaN where it is undefined: at the target, at a point
            disconnected from it, or outside the environment.
        """
        point_bins = self.bins(points)
        if self._track is not None:
            return self._track.directions(points, target, point_bins)
        root = int(self.bins(target[None, :])[0])
        self._fields.ensure(self, np.array([root]))
        out = np.full(points.shape, np.nan)
        rows = np.flatnonzero(point_bins >= 0)
        a = point_bins[rows]
        field_row = self._fields.row_of_root[root]
        anchor = self._fields.anchor[field_row, a]
        reachable = np.isfinite(self._fields.length[field_row, a])
        straight = (a == root) | (anchor == root)
        waypoint = np.where(
            straight[:, None], target[None, :], self._centers[np.maximum(anchor, 0)]
        )
        vector = waypoint - points[rows]
        norm = np.linalg.norm(vector, axis=1)
        defined = reachable & (norm > 0)
        out[rows[defined]] = vector[defined] / norm[defined, None]
        return out

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
        source_bins = np.asarray(source_bins, dtype=np.intp)
        if self._track is not None:
            return self._track.from_points(
                self._centers[source_bins], self._centers, np.arange(self._n_bins)
            )
        if source_bins.size == 1:
            # One source: the same field as a target position in that bin.
            self._fields.ensure(self, source_bins)
            row = self._fields.row_of_root[source_bins[0]]
            single: NDArray[np.float64] = self._fields.length[row].copy()
            return single
        distances: NDArray[np.float64] = self._source_field(source_bins)[0].copy()
        return distances

    def from_bins_at(
        self, source_bins: NDArray[np.intp], points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Distance from each point to the nearest source bin center.

        With several sources, a point's path leaves straight for the anchor
        of its own bin or, when visible from the point, of a neighboring
        bin, whichever is shortest.

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
        point_bins = self.bins(points)
        if self._track is not None:
            return self._track.from_points(
                self._centers[source_bins], points, point_bins
            )
        if source_bins.size == 1:
            return self.to_point(points, self._centers[source_bins[0]])
        length, anchor = self._source_field(source_bins)
        out = np.full(points.shape[0], np.nan)
        rows = np.flatnonzero(point_bins >= 0)
        out[rows] = np.inf
        # Candidate corners: the anchor of the point's bin (its center sees
        # it) and, when the point sees them, the anchors of the bin's open
        # neighbors, so the nearest source is chosen from the point itself.
        own = anchor[point_bins[rows]].astype(np.intp)
        degree = (self._neighbors[point_bins[rows]] >= 0).sum(axis=1)
        owner = np.repeat(np.arange(rows.size), degree)
        neighbor = self._neighbors[point_bins[rows]][
            self._neighbors[point_bins[rows]] >= 0
        ]
        corner = anchor[neighbor].astype(np.intp)
        keep = (corner >= 0) & (corner != own[owner])
        keys = np.unique(owner[keep] * self._n_bins + corner[keep])
        owner, corner = keys // self._n_bins, keys % self._n_bins
        visible, _ = self._trace(
            points[rows[owner]],
            self._centers[corner],
            point_bins[rows[owner]],
            corner,
        )
        owner = np.r_[np.arange(rows.size), owner[visible]]
        corner = np.r_[own, corner[visible]]
        found = corner >= 0
        owner, corner = owner[found], corner[found]
        total = (
            np.linalg.norm(points[rows[owner]] - self._centers[corner], axis=1)
            + length[corner]
        )
        best = np.full(rows.size, np.inf)
        np.minimum.at(best, owner, total)
        out[rows] = best
        return out

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
        if self._track is not None:
            return self._track.costs_from_bins(source_bins, rate, self._adjacency)
        source_bins = np.asarray(source_bins, dtype=np.intp)
        length, _, _ = self._grow(
            [source_bins], rate=rate, single_roots=source_bins.size == 1
        )
        costs: NDArray[np.float64] = length[0]
        return costs

    # ------------------------------------------------------------------
    # Internals: open environments
    # ------------------------------------------------------------------

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
    ) -> NDArray[np.float64]:
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
        """
        length_of, anchor_of, lead_of = fields
        anchor = anchor_of[field_rows, point_bins].astype(np.intp)
        length = length_of[field_rows, point_bins]
        out = np.full(points.shape[0], np.inf)
        straight = (point_bins == roots) | (anchor == roots)
        out[straight] = np.linalg.norm(targets[straight] - points[straight], axis=1)
        around = np.flatnonzero(~straight & np.isfinite(length))
        if around.size:
            corner = anchor[around]
            lead_center = self._centers[lead_of[field_rows[around], corner]]
            root_center = self._centers[roots[around]]
            # The field's last leg ends at the root bin's center; end it at
            # the target instead.
            out[around] = (
                np.linalg.norm(points[around] - self._centers[corner], axis=1)
                + length_of[field_rows[around], corner]
                - np.linalg.norm(lead_center - root_center, axis=1)
                + np.linalg.norm(lead_center - targets[around], axis=1)
            )
        return out

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
        segment costs about as much as its distance to the obstruction.

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
            m = self._cells(starts[seg] + fm[:, None] * (ends[seg] - starts[seg]))
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
            Per set, the bins it is needed for: its growth stops once they
            are reached. Bins it did not reach are left ``inf``/-1. A bin's
            value never depends on whether growth stopped later.

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
        adjacency = self._adjacency
        if rate is not None:
            weighted = adjacency.copy()
            rows = np.repeat(np.arange(n_bins), np.diff(adjacency.indptr))
            weighted.data = (
                adjacency.data * 0.5 * (rate[rows] + rate[adjacency.indices])
            )
            adjacency = weighted
        if single_roots:
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
        # Oriented edges u -> v with g[u] < g[v], as global ids f * n + bin.
        u_bin = np.repeat(np.arange(n_bins), np.diff(self._adjacency.indptr))
        v_bin = self._adjacency.indices.astype(np.intp)
        g_u, g_v = graph[:, u_bin], graph[:, v_bin]
        with np.errstate(invalid="ignore"):
            oriented = np.isfinite(g_v) & (g_u < g_v - _GRAPH_RTOL * g_v)
        field_of, edge_of = np.nonzero(oriented)
        src = field_of * n_bins + u_bin[edge_of]
        dst = field_of * n_bins + v_bin[edge_of]
        n_nodes = n_sets * n_bins
        order_in = np.argsort(dst, kind="stable")
        in_src = src[order_in]
        in_ptr = np.searchsorted(dst[order_in], np.arange(n_nodes + 1))

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
        root_of_field = (
            np.array([s[0] for s in source_sets], dtype=np.intp)
            if single_roots
            else None
        )
        # Layers of graph distance, one shortest edge thick.
        flat = graph.ravel()
        reached = np.flatnonzero(np.isfinite(flat) & ~is_source)
        layer = np.floor(flat[reached] / (self._spacing / _PROBE_FRACTION)).astype(
            np.intp
        )
        by_layer = np.argsort(layer, kind="stable")
        reached, layer = reached[by_layer], layer[by_layer]
        bounds = np.flatnonzero(np.diff(layer)) + 1
        waves = np.split(reached, bounds)
        needed = None
        if stop_bins is not None:
            needed = np.zeros(n_nodes, dtype=bool)
            for f, stop in enumerate(stop_bins):
                needed[f * n_bins + np.asarray(stop, dtype=np.intp)] = True
            needed &= ~settled & np.isfinite(graph.ravel())
            open_fields = (
                np.bincount(np.flatnonzero(needed) // n_bins, minlength=n_sets) > 0
            )
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
                open_fields = (
                    np.bincount(np.flatnonzero(needed) // n_bins, minlength=n_sets) > 0
                )
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
        nodes = np.sort(nodes)
        starts = in_ptr[nodes]
        counts = in_ptr[nodes + 1] - starts
        node = np.repeat(nodes, counts)
        parent = in_src[np.repeat(starts, counts) + ragged_arange(counts)]
        final = settled[parent]
        node, parent = node[final], parent[final]
        ready = np.zeros(nodes.size, dtype=bool)
        ready[np.searchsorted(nodes, node)] = True
        waiting: NDArray[np.intp] = nodes[~ready]
        nodes = nodes[ready]
        if not nodes.size:
            return waiting
        field = node // n_bins
        node_bin = node - field * n_bins
        parent_bin = parent - field * n_bins
        # Along the graph edge to a nearer neighbor: always allowed.
        edge = np.linalg.norm(
            self._centers[node_bin] - self._centers[parent_bin], axis=1
        )
        if rate is not None:
            edge = edge * 0.5 * (rate[node_bin] + rate[parent_bin])
        edge_total = edge + length[parent]
        # Straight to a nearer neighbor's anchor, and (single roots) to the
        # root, when visible.
        corner_node = node
        corner_bin = anchor[parent].astype(np.intp)
        if root_of_field is not None:
            corner_node = np.r_[corner_node, nodes]
            corner_bin = np.r_[corner_bin, root_of_field[nodes // n_bins]]
        keys = np.unique(corner_node * n_bins + corner_bin)
        cand_node, cand_bin = keys // n_bins, keys % n_bins
        cand_node_bin = cand_node % n_bins
        keep = cand_bin != cand_node_bin
        cand_node, cand_bin, cand_node_bin = (
            cand_node[keep],
            cand_bin[keep],
            cand_node_bin[keep],
        )
        visible, cost = self._trace(
            self._centers[cand_node_bin],
            self._centers[cand_bin],
            cand_node_bin,
            cand_bin,
            rate,
        )
        cand_node, cand_bin = cand_node[visible], cand_bin[visible]
        cand_field = cand_node // n_bins
        cand_total = cost[visible] + length[cand_field * n_bins + cand_bin]
        # Rank per node: a visible root first (it is the straight line, which
        # no chain beats even by rounding), then the lowest total, anchors
        # before graph edges, then the lower bin.
        all_node = np.r_[cand_node, node]
        all_bin = np.r_[cand_bin, parent_bin]
        all_total = np.r_[cand_total, edge_total]
        via_edge = np.r_[np.zeros(cand_node.size), np.ones(node.size)]
        not_root = np.ones(all_node.size)
        if root_of_field is not None:
            not_root[: cand_node.size] = cand_bin != root_of_field[cand_field]
        order = np.lexsort((all_bin, via_edge, all_total, not_root, all_node))
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


class _GridCells:
    """Grid cell of each bin on a rectilinear grid, for exact cell walks."""

    def __init__(
        self,
        edges: tuple[NDArray[np.float64], ...],
        flat_to_active: NDArray[np.intp],
        shape: tuple[int, ...],
        cell: NDArray[np.intp],
    ) -> None:
        self.edges = edges
        self.flat_to_active = flat_to_active
        self.shape = shape
        self.cell = cell

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


class PairDistances:
    """Shortest path lengths between pairs of one set of positions.

    Made by :meth:`EnvironmentPaths.pairwise`. The fields of every bin the
    positions visit are reduced to per-bin-pair tables once, so each pair
    is a few lookups; results equal :meth:`EnvironmentPaths.between`.
    """

    def __init__(self, paths: EnvironmentPaths, points: NDArray[np.float64]) -> None:
        self._paths = paths
        self._points = points
        self._bins = paths.bins(points)
        if paths._track is not None:
            located = np.flatnonzero(self._bins >= 0)
            self._segment = np.zeros(points.shape[0], dtype=np.intp)
            self._offset = np.full(points.shape[0], np.nan)
            self._segment[located], self._offset[located] = paths._track.project(
                points[located]
            )
            return
        roots = np.unique(self._bins[self._bins >= 0])
        paths._fields.ensure(paths, roots)
        fields = paths._fields
        rows = fields.row_of_root[roots]
        n_bins = paths._n_bins
        # Table row of each target bin.
        self._row_of = np.full(n_bins, -1, dtype=np.intp)
        self._row_of[roots] = np.arange(roots.size)
        anchor = fields.anchor[rows].astype(np.intp)
        length = fields.length[rows]
        table_rows = np.arange(roots.size)[:, None]
        # Straight: same bin, connected bins, or the bin center sees the
        # target bin center.
        straight = anchor == roots[:, None]
        straight[np.arange(roots.size), roots] = True
        neighbors = paths._neighbors[roots]
        has = neighbors >= 0
        straight[np.nonzero(has)[0], neighbors[has]] = True
        corner = np.maximum(anchor, 0)
        lead = fields.lead[rows[:, None], corner].astype(np.intp)
        lead = np.maximum(lead, 0)
        centers = paths._centers
        # Path length from the corner on, without its last leg to the target
        # bin center: d = |p - corner| + rest + |lead - target|.
        rest = length[table_rows, corner] - np.linalg.norm(
            centers[lead] - centers[roots][:, None, :], axis=2
        )
        rest[anchor < 0] = np.inf
        self._straight = straight
        self._corner = corner
        self._lead = lead
        self._rest = rest

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
        paths = self._paths
        out = np.full(first.size, np.nan)
        start_bins, end_bins = self._bins[first], self._bins[second]
        ok = np.flatnonzero((start_bins >= 0) & (end_bins >= 0))
        if paths._track is not None:
            out[ok] = paths._track._pair(
                self._segment[first[ok]],
                self._offset[first[ok]],
                self._segment[second[ok]],
                self._offset[second[ok]],
            )
            return out
        a, row = start_bins[ok], self._row_of[end_bins[ok]]
        starts, ends = self._points[first[ok]], self._points[second[ok]]
        straight = self._straight[row, a]
        centers = paths._centers
        corner, lead = self._corner[row, a], self._lead[row, a]
        around = (
            np.linalg.norm(starts - centers[corner], axis=1)
            + self._rest[row, a]
            + np.linalg.norm(centers[lead] - ends, axis=1)
        )
        out[ok] = np.where(straight, np.linalg.norm(ends - starts, axis=1), around)
        return out


class _RootFields:
    """Kept single-bin fields, stacked so pairs gather from many at once."""

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.limit = max(1, _FIELD_CACHE_BYTES // (16 * max(n_bins, 1)))
        self.row_of_root = np.full(n_bins, -1, dtype=np.intp)
        self.length = np.empty((0, n_bins))
        self.anchor = np.empty((0, n_bins), dtype=np.int32)
        self.lead = np.empty((0, n_bins), dtype=np.int32)

    def ensure(self, paths: EnvironmentPaths, roots: NDArray[np.intp]) -> None:
        """Grow and keep the field of every root not yet kept."""
        missing = roots[self.row_of_root[roots] < 0]
        if not missing.size:
            return
        if self.length.shape[0] + missing.size > self.limit:
            # Start over, keeping only what this request needs.
            needed = roots[self.row_of_root[roots] >= 0]
            rows = self.row_of_root[needed]
            self.length = self.length[rows]
            self.anchor = self.anchor[rows]
            self.lead = self.lead[rows]
            self.row_of_root[:] = -1
            self.row_of_root[needed] = np.arange(needed.size)
        parts = [
            paths._grow(
                [
                    missing[i : i + 1]
                    for i in range(lo, min(lo + _FIELD_BATCH, missing.size))
                ],
                single_roots=True,
            )
            for lo in range(0, missing.size, _FIELD_BATCH)
        ]
        n_kept = self.length.shape[0]
        self.length = np.vstack([self.length, *(p[0] for p in parts)])
        self.anchor = np.vstack([self.anchor, *(p[1] for p in parts)])
        self.lead = np.vstack([self.lead, *(p[2] for p in parts)])
        self.row_of_root[missing] = n_kept + np.arange(missing.size)


class _TrackPaths:
    """Distances along a track graph between projections onto the track."""

    def __init__(self, linearization: TrackLinearization) -> None:
        from scipy.sparse import csr_array
        from scipy.sparse.csgraph import dijkstra

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

    def project(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Nearest segment and distance along it from its first node.

        A point as near, within rounding, to several segments takes the one
        first in the track's linear order, as ``Environment.to_linear`` does.
        Rows with a non-finite coordinate get segment 0 (callers mask them).
        """
        from neurospatial.geometry._locators import _coordinate_roundoff

        finite = np.all(np.isfinite(points), axis=1)
        values = np.where(finite[:, None], points, 0.0)
        squared = self._length**2
        with np.errstate(divide="ignore", invalid="ignore"):
            along = (
                np.einsum(
                    "sd,psd->ps", self._direction, values[:, None, :] - self._first
                )
                / squared
            )
        along = np.clip(np.where(squared > 0, along, 0.0), 0.0, 1.0)
        projected = self._first + along[:, :, None] * self._direction
        distance = np.linalg.norm(values[:, None, :] - projected, axis=2)
        tolerance = _coordinate_roundoff(values, self._scale)
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

    def between(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        start_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Along-track distance per pair; NaN where ``start_bins`` is -1.

        ``ends`` must be finite where the start is in the track.
        """
        out = np.full(starts.shape[0], np.nan)
        rows = np.flatnonzero((start_bins >= 0) & np.all(np.isfinite(ends), axis=1))
        if rows.size:
            seg_a, off_a = self.project(starts[rows])
            seg_b, off_b = self.project(ends[rows])
            out[rows] = self._pair(seg_a, off_a, seg_b, off_b)
        return out

    def from_points(
        self,
        sources: NDArray[np.float64],
        points: NDArray[np.float64],
        point_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Distance from each point to the nearest of ``sources``."""
        out = np.full(points.shape[0], np.nan)
        rows = np.flatnonzero(point_bins >= 0)
        if not rows.size:
            return out
        seg_p, off_p = self.project(points[rows])
        seg_s, off_s = self.project(sources)
        n_points, n_sources = rows.size, sources.shape[0]
        out[rows] = (
            self._pair(
                np.repeat(seg_p, n_sources),
                np.repeat(off_p, n_sources),
                np.tile(seg_s, n_points),
                np.tile(off_s, n_points),
            )
            .reshape(n_points, n_sources)
            .min(axis=1)
        )
        return out

    def directions(
        self,
        points: NDArray[np.float64],
        target: NDArray[np.float64],
        point_bins: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Unit vector along the track toward ``target`` at each point.

        NaN at the target, at a point whose projection is a node from
        which the route leaves along another segment, and outside the track.
        """
        out = np.full(points.shape, np.nan)
        rows = np.flatnonzero(point_bins >= 0)
        if not rows.size:
            return out
        seg, off = self.project(points[rows])
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
        out[rows[defined]] = sign[defined, None] * unit[defined]
        return out

    def costs_from_bins(
        self,
        source_bins: NDArray[np.intp],
        rate: NDArray[np.float64],
        adjacency: csr_array,
    ) -> NDArray[np.float64]:
        """Least travel cost along the track's bin graph.

        A track's bins lie on its centerline and its bin graph runs along
        it, so the bin-graph path is the along-track path.
        """
        from scipy.sparse.csgraph import dijkstra

        weighted = adjacency.copy()
        rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
        weighted.data = adjacency.data * 0.5 * (rate[rows] + rate[adjacency.indices])
        cost: NDArray[np.float64] = dijkstra(
            weighted, directed=False, indices=source_bins, min_only=True
        )
        return cost
