"""Egocentric polar environment type.

This module defines :class:`EgocentricPolarEnvironment`, a distinct
environment type for egocentric *polar* space. Unlike :class:`Environment`,
whose ``bin_centers`` hold Cartesian ``(x, y[, z])`` coordinates, a polar
environment's ``bin_centers[:, 0]`` is **distance** (radius, in physical
length units such as cm) and ``bin_centers[:, 1]`` is **angle in radians**
(egocentric: 0 = ahead, +π/2 = left, -π/2 = right, ±π = behind).

``EgocentricPolarEnvironment`` is a *sibling* of ``Environment`` — both
subclass the shared ``_BaseEnvironment`` — but it is **not** a subclass of
``Environment``. This is deliberate: any inherited method that consumes
Cartesian ``(x, y[, z])`` coordinates or operates on a Cartesian grid would
silently misinterpret ``(distance, angle)`` pairs and return geometric
nonsense, so each is overridden here to raise ``NotImplementedError`` rather
than inherited. These fall into two groups — Cartesian-coordinate methods
(``bin_at``, ``contains``, ``distance_between``, Euclidean ``distance_to``,
``apply_transform``, ``interpolate``, ``occupancy``, ``bin_sequence``,
``bin_sequence_with_runs``) and Cartesian-grid methods (``to_linear``,
``linear_to_nd``, ``rebin``, ``subset``).

Geodesic operations remain well-defined because they read only the
connectivity graph, whose edge ``distance`` weights are physical polar
lengths (arc ``r·Δθ`` for angular steps, ``Δr`` for radial steps, and
``sqrt(Δr² + (r·Δθ)²)`` for diagonals).

The public entry point is :meth:`Environment.from_polar_egocentric`, which
delegates to :meth:`EgocentricPolarEnvironment.create`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, NoReturn, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import shapely

from neurospatial.environment.core import _BaseEnvironment


class EgocentricPolarEnvironment(_BaseEnvironment):
    """A discretized egocentric polar environment.

    ``bin_centers[:, 0]`` is distance (radius) in physical length units and
    ``bin_centers[:, 1]`` is angle in radians. Build one via
    :meth:`Environment.from_polar_egocentric` (the public factory) or
    :meth:`create` (the underlying constructor).

    This type is **not** a subclass of :class:`Environment`. Every inherited
    method that assumes ``(x, y[, z])`` bin centers or a Cartesian grid is
    overridden to raise ``NotImplementedError`` rather than silently
    misinterpret ``(distance, angle)`` pairs:

    *Cartesian-coordinate methods* (would read ``(distance, angle)`` as
    ``(x, y)``):

    - :meth:`bin_at`, :meth:`contains` — geometric containment in ``(x, y)``.
    - :meth:`distance_between`, ``distance_to(metric="euclidean")`` —
      straight-line distance on ``(x, y)``.
    - :meth:`apply_transform` — affine transforms on ``(x, y)``.
    - :meth:`interpolate` — bypasses the overridden :meth:`bin_at` and would
      otherwise return geometric nonsense without raising.
    - :meth:`occupancy`, :meth:`bin_sequence`, :meth:`bin_sequence_with_runs`
      — map continuous ``(x, y)`` trajectories onto bins.

    *Cartesian-grid methods* (operate on a rectangular grid that has no polar
    meaning):

    - :meth:`to_linear`, :meth:`linear_to_nd` — 1D track linearization.
    - :meth:`rebin`, :meth:`subset` — grid coarsening / sub-selection
      (and would return a Cartesian :class:`Environment`).

    Use connectivity-graph operations instead — :meth:`neighbors`,
    :meth:`path_between`, :meth:`reachable_from`, :meth:`smooth`, and
    ``distance_to(metric="geodesic")`` — which respect the physical polar
    edge geometry.

    Coordinate convention
    ---------------------
    - Angle 0 = directly ahead (egocentric forward direction)
    - Angle +π/2 = left
    - Angle -π/2 = right
    - Angle ±π = behind

    See Also
    --------
    Environment.from_polar_egocentric : Public factory for this type.
    neurospatial.ops.egocentric : Allocentric↔egocentric array transforms.
    """

    _POLAR: ClassVar[bool] = True

    @classmethod
    def create(
        cls,
        distance_range: tuple[float, float],
        angle_range: tuple[float, float],
        distance_bin_size: float,
        angle_bin_size: float,
        *,
        circular_angle: bool = True,
        connect_diagonal_neighbors: bool = True,
        name: str = "",
    ) -> EgocentricPolarEnvironment:
        """Build an egocentric polar environment.

        Creates a 2D polar grid where the first dimension is distance from
        the observer and the second is angle relative to heading.

        Parameters
        ----------
        distance_range : tuple of (float, float)
            The (min, max) range of distances in physical units (e.g., cm).
            Must have min < max.
        angle_range : tuple of (float, float)
            The (min, max) range of angles in radians. For full circle
            coverage use (-π, π) or (0, 2π). Must have min < max.
        distance_bin_size : float
            Size of each distance bin, same units as ``distance_range``.
            Must be positive.
        angle_bin_size : float
            Size of each angle bin in radians. Must be positive.
        circular_angle : bool, default=True
            If True, the angle dimension wraps circularly, connecting the
            first and last angle bins at each distance ring. Appropriate
            when ``angle_range`` spans a full circle.
        connect_diagonal_neighbors : bool, default=True
            Whether to connect diagonally adjacent (distance, angle) bins.
        name : str, default=""
            Optional name for the environment.

        Returns
        -------
        EgocentricPolarEnvironment
            A fitted polar environment with physically correct edge
            geometry (arc length ``r·Δθ`` for angular steps, ``Δr`` for
            radial steps, ``sqrt(Δr² + (r·Δθ)²)`` for diagonals).

        Raises
        ------
        ValueError
            If ``distance_bin_size`` or ``angle_bin_size`` is not positive,
            or if ``distance_range`` / ``angle_range`` has min >= max.

        Notes
        -----
        Coordinate convention: angle 0 = ahead, +π/2 = left, -π/2 = right,
        ±π = behind.
        """
        # Validate parameters
        if distance_bin_size <= 0:
            raise ValueError(
                f"distance_bin_size must be positive, got {distance_bin_size}"
            )
        if angle_bin_size <= 0:
            raise ValueError(f"angle_bin_size must be positive, got {angle_bin_size}")
        if distance_range[0] >= distance_range[1]:
            raise ValueError(
                f"distance_range must have min < max, got {distance_range}"
            )
        if angle_range[0] >= angle_range[1]:
            raise ValueError(f"angle_range must have min < max, got {angle_range}")

        # Calculate number of bins in each dimension
        n_distance = max(
            1,
            int(np.ceil((distance_range[1] - distance_range[0]) / distance_bin_size)),
        )
        n_angle = max(
            1, int(np.ceil((angle_range[1] - angle_range[0]) / angle_bin_size))
        )

        # Create grid edges and an all-active mask
        distance_edges = np.linspace(
            distance_range[0], distance_range[1], n_distance + 1
        )
        angle_edges = np.linspace(angle_range[0], angle_range[1], n_angle + 1)
        grid_edges = (distance_edges, angle_edges)
        active_mask = np.ones((n_distance, n_angle), dtype=bool)

        # Build the polar env via the shared from_layout machinery. Because
        # this is called on EgocentricPolarEnvironment (cls), from_layout
        # constructs the polar type, NOT a Cartesian Environment. (Going
        # through Environment.from_grid_mask would return an Environment.)
        env = cast(
            "EgocentricPolarEnvironment",
            cls.from_layout(
                kind="MaskedGrid",
                layout_params={
                    "active_mask": active_mask,
                    "grid_edges": grid_edges,
                    "connect_diagonal_neighbors": connect_diagonal_neighbors,
                },
                name=name,
            ),
        )

        # Overwrite the grid edge "distance" weights with proper physical
        # polar lengths. The masked-grid layout computed them as the Euclidean
        # norm of (Δdistance, Δangle), collapsing cm and radians; here we use
        # arc/radial/diagonal polar lengths instead. Done BEFORE adding the
        # circular wrap edges so the wrap edges (which span a single
        # angular-bin step across the seam, not the full angular range) keep
        # the correct arc length set by _add_circular_connectivity.
        distance_centers = 0.5 * (distance_edges[:-1] + distance_edges[1:])
        _fix_polar_edge_geometry(env.connectivity)

        # Add circular wrap edges between first and last angle bins.
        if circular_angle and n_angle > 1:
            _add_circular_connectivity(
                env.connectivity, n_distance, n_angle, distance_centers, angle_bin_size
            )

        # The mutations above happened after _setup_from_layout finalized
        # _state_version; bump it so any versioned cache recomputes against
        # the corrected connectivity graph.
        env._state_version += 1

        return env

    # ------------------------------------------------------------------
    # Cartesian-only methods: unavailable on polar environments.
    # ------------------------------------------------------------------
    def _raise_cartesian_only(
        self, method_name: str, *, reason: str | None = None
    ) -> NoReturn:
        """Raise ``NotImplementedError`` for a Cartesian-only method.

        The error names the offending method and points the user at the
        connectivity-graph alternatives that *are* well-defined on polar
        environments.

        Parameters
        ----------
        method_name : str
            Name of the unsupported method (used verbatim in the message).
        reason : str, optional
            Short qualifier describing *why* the method is unsupported,
            inserted in parentheses after the method name (e.g.
            ``"Cartesian-coordinate operation"`` or
            ``"Cartesian-grid operation"``). Defaults to
            ``"Cartesian-coordinate operation"``.
        """
        if reason is None:
            reason = "Cartesian-coordinate operation"
        raise NotImplementedError(
            f"EgocentricPolarEnvironment.{method_name}() is not available "
            f"({reason}): this is a polar environment whose bin_centers[:, 0] "
            "is distance and bin_centers[:, 1] is angle in radians, so "
            "operations that assume Cartesian (x, y[, z]) coordinates or a "
            "Cartesian grid would silently return geometric nonsense. Use "
            "connectivity-graph operations instead (neighbors, path_between, "
            "reachable_from, smooth, or distance_to(metric='geodesic')), or "
            "build an allocentric Cartesian env via Environment.from_samples / "
            "from_polygon / from_grid_mask."
        )

    def bin_at(self, points_nd: NDArray[np.float64]) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("bin_at")

    def contains(self, points_nd: NDArray[np.float64]) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("contains")

    def distance_between(
        self,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("distance_between")

    def distance_to(
        self,
        targets: object,
        *,
        metric: str = "geodesic",
    ) -> NDArray[np.float64]:
        """Distance from every bin to the nearest target.

        Only ``metric="geodesic"`` (graph shortest path) is well-defined on a
        polar environment; ``metric="euclidean"`` would compute straight-line
        distance on ``(distance, angle)`` pairs and is rejected.

        Parameters
        ----------
        targets : sequence of int or str
            Target bin indices or a region name. Forwarded to the base
            geodesic implementation.
        metric : {"geodesic"}, default="geodesic"
            Only ``"geodesic"`` is supported; ``"euclidean"`` raises.

        Returns
        -------
        ndarray of shape (n_bins,)
            Geodesic distance from each bin to the nearest target.

        Raises
        ------
        NotImplementedError
            If ``metric="euclidean"``.
        """
        if metric == "euclidean":
            self._raise_cartesian_only("distance_to(metric='euclidean')")
        return super().distance_to(targets, metric="geodesic")

    def apply_transform(
        self, transform: object, *, name: str | None = None
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("apply_transform")

    # ------------------------------------------------------------------
    # Inherited methods that consume Cartesian coordinates or assume a
    # Cartesian grid: also unavailable on polar environments. These would
    # otherwise misread (distance, angle) pairs as (x, y) (interpolate,
    # occupancy, bin_sequence) or operate on a Cartesian grid that has no
    # polar meaning (to_linear, linear_to_nd, rebin, subset). interpolate is
    # especially dangerous because it bypasses the overridden bin_at and would
    # silently return geometric nonsense rather than raising.
    # ------------------------------------------------------------------
    def interpolate(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
        *,
        mode: Literal["nearest", "linear"] = "nearest",
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("interpolate")

    def occupancy(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        bandwidth: float | None = None,
        time_allocation: Literal["start", "linear"] = "start",
        return_seconds: bool = True,
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("occupancy")

    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        outside_value: int | None = -1,
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("bin_sequence")

    def bin_sequence_with_runs(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("bin_sequence_with_runs")

    def to_linear(self, nd_position: NDArray[np.float64]) -> NoReturn:
        """Unavailable on polar environments (Cartesian-grid operation)."""
        self._raise_cartesian_only("to_linear", reason="Cartesian-grid operation")

    def linear_to_nd(self, linear_position: NDArray[np.float64]) -> NoReturn:
        """Unavailable on polar environments (Cartesian-grid operation)."""
        self._raise_cartesian_only("linear_to_nd", reason="Cartesian-grid operation")

    def rebin(self, factor: int | tuple[int, ...]) -> NoReturn:
        """Unavailable on polar environments (Cartesian-grid operation)."""
        self._raise_cartesian_only("rebin", reason="Cartesian-grid operation")

    def subset(
        self,
        *,
        bins: NDArray[np.bool_] | None = None,
        region_names: Sequence[str] | None = None,
        polygon: shapely.Polygon | None = None,
        invert: bool = False,
    ) -> NoReturn:
        """Unavailable on polar environments (Cartesian-grid operation)."""
        self._raise_cartesian_only("subset", reason="Cartesian-grid operation")

    @property
    def bin_sizes(self) -> NDArray[np.float64]:
        """Physical area of each polar bin (annular sector).

        The base implementation would return the layout's Cartesian cell area
        ``Δr · Δθ``, treating the angular extent (radians) as a length. The
        physically correct area of the annular sector spanning ``[r0, r1]`` in
        radius and ``Δθ`` in angle is ``0.5 · (r1² - r0²) · Δθ``, which equals
        ``r̄ · Δr · Δθ`` with ``r̄`` the mean radius.

        Returns
        -------
        ndarray of shape (n_bins,)
            Annular-sector area of each active bin, in (length unit)².
        """
        grid_edges = self.grid_edges
        if grid_edges is None or len(grid_edges) != 2:
            # Fall back to the layout area if grid edges are unavailable.
            return self.layout.bin_sizes()

        distance_edges, angle_edges = grid_edges
        r0 = distance_edges[:-1]
        r1 = distance_edges[1:]
        d_theta = np.diff(angle_edges)
        # Annular-sector area per (distance_idx, angle_idx) cell.
        sector_area = 0.5 * (r1**2 - r0**2)[:, None] * d_theta[None, :]
        sector_flat = sector_area.ravel()

        active_mask = self.active_mask
        if active_mask is not None:
            return np.asarray(sector_flat[active_mask.ravel()], dtype=np.float64)
        return np.asarray(sector_flat, dtype=np.float64)


def _add_circular_connectivity(
    connectivity: nx.Graph,
    n_distance: int,
    n_angle: int,
    distance_centers: NDArray[np.float64],
    angle_bin_size: float,
) -> None:
    """Add circular wrap edges between first and last angle bins.

    Modifies ``connectivity`` in-place, adding an edge between angle index 0
    and angle index ``n_angle - 1`` for each distance ring. Nodes are indexed
    in row-major order: ``node_id = distance_idx * n_angle + angle_idx``.

    The wrap edge's ``distance`` attribute is the arc length of a single
    ``angle_bin_size`` step at that ring's radius
    (``distance_center * angle_bin_size``), not the Euclidean chord between
    the first/last angle-bin centers (which span the full angular range).

    Parameters
    ----------
    connectivity : nx.Graph
        Connectivity graph to modify.
    n_distance : int
        Number of distance bins.
    n_angle : int
        Number of angle bins.
    distance_centers : ndarray of shape (n_distance,)
        Center radius of each distance ring, in physical units.
    angle_bin_size : float
        Angular step between adjacent angle bins, in radians.
    """
    max_edge_id = max(
        (data.get("edge_id", -1) for _, _, data in connectivity.edges(data=True)),
        default=-1,
    )

    for d_idx in range(n_distance):
        first_angle_node = d_idx * n_angle + 0
        last_angle_node = d_idx * n_angle + (n_angle - 1)

        if not connectivity.has_edge(first_angle_node, last_angle_node):
            pos_first = connectivity.nodes[first_angle_node]["pos"]
            pos_last = connectivity.nodes[last_angle_node]["pos"]

            vector = np.array(pos_last) - np.array(pos_first)
            distance = float(distance_centers[d_idx]) * float(angle_bin_size)
            angle_2d = np.arctan2(vector[1], vector[0]) if len(vector) >= 2 else 0.0

            max_edge_id += 1
            connectivity.add_edge(
                first_angle_node,
                last_angle_node,
                distance=float(distance),
                vector=vector.tolist(),
                angle_2d=float(angle_2d),
                edge_id=int(max_edge_id),
            )


def _fix_polar_edge_geometry(connectivity: nx.Graph) -> None:
    """Rewrite all edge ``distance`` weights with physical polar lengths.

    The masked-grid layout computes each edge ``distance`` as the Euclidean
    norm of the displacement in ``(distance, angle)`` coordinates, which
    treats an angular step ``Δθ`` (radians) as if it were a length and so
    collapses cm and radians into one scalar. Replace each edge weight with
    the correct physical polar length:

    - pure angular step (same distance ring): arc length ``r̄ · Δθ`` where
      ``r̄`` is the mean of the two endpoints' radii;
    - pure radial step (same angle): ``|Δr|``;
    - diagonal step: ``sqrt(Δr² + (r̄ · Δθ)²)``.

    This is called BEFORE the circular wrap edges are added, so it only sees
    the grid edges (single-step radial/angular/diagonal moves). Node positions
    are read from the ``"pos"`` node attribute, where ``pos[0]`` is distance
    and ``pos[1]`` is angle.

    Parameters
    ----------
    connectivity : nx.Graph
        Connectivity graph to modify in-place.
    """
    for u, v, data in connectivity.edges(data=True):
        pos_u = np.asarray(connectivity.nodes[u]["pos"], dtype=np.float64)
        pos_v = np.asarray(connectivity.nodes[v]["pos"], dtype=np.float64)

        r_u, theta_u = float(pos_u[0]), float(pos_u[1])
        r_v, theta_v = float(pos_v[0]), float(pos_v[1])

        d_r = r_v - r_u
        d_theta = theta_v - theta_u
        r_mean = 0.5 * (r_u + r_v)

        arc = r_mean * d_theta
        distance = float(np.hypot(d_r, arc))
        data["distance"] = distance
