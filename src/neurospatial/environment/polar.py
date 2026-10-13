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
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import shapely

    from neurospatial.layout.base import LayoutEngine
    from neurospatial.layout.factories import LayoutType

from neurospatial.environment.core import _BaseEnvironment
from neurospatial.environment.decorators import check_fitted, versioned_cached_property
from neurospatial.layout.engines._polar import _PolarGridLayout
from neurospatial.layout.helpers.utils import find_boundary_nodes


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
    def _create_layout(
        cls, kind: LayoutType | str, layout_params: dict[str, Any]
    ) -> LayoutEngine:
        """Store polar bins on the masked grid whose cells are annular sectors.

        :meth:`create` and the file and NWB readers build through
        ``from_layout``; copies keep the layout class.

        Raises
        ------
        ValueError
            If ``kind`` does not name the masked grid.
        """
        from neurospatial.layout.factories import _normalize_name

        if _normalize_name(str(getattr(kind, "value", kind))) != "maskedgrid":
            raise ValueError(
                f"Polar environments are stored on a 'MaskedGrid' layout, got "
                f"{kind!r}. Build one with Environment.from_polar_egocentric(...)."
            )
        layout = _PolarGridLayout()
        layout.build(**layout_params)
        # Like every grid engine, the layout satisfies LayoutEngine at runtime;
        # mypy rejects its richer plot() signature.
        return cast("LayoutEngine", layout)

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

        # from_layout on cls builds the polar type on _PolarGridLayout.
        env = cast(
            "EgocentricPolarEnvironment",
            cls.from_layout(
                kind="MaskedGrid",
                layout_params={
                    "active_mask": active_mask,
                    "grid_edges": grid_edges,
                    "connect_diagonal_neighbors": connect_diagonal_neighbors,
                    "circular_angle": circular_angle,
                },
                name=name,
            ),
        )

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
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> NoReturn:
        """Unavailable on polar environments (see :meth:`_raise_cartesian_only`)."""
        self._raise_cartesian_only("bin_sequence")

    def bin_sequence_with_runs(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
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

    @cached_property
    @check_fitted
    def boundary_bins(self) -> NDArray[np.intp]:
        """Bins on the edge of the active polar region.

        A bin is on the boundary when a radial or angular neighbor is outside
        the grid or inactive. A circular angular axis wraps, so bins beside the
        ±π seam are interior; on a full, circular grid only the innermost and
        outermost rings are boundary.

        Returns
        -------
        NDArray[np.intp], shape (n_boundary_bins,)
            Bin IDs on the boundary.
        """
        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
            layout_kind=self._layout_type_used,
            periodic_axes=(1,) if self._angular_is_circular else (),
        ).astype(np.intp, copy=False)

    @versioned_cached_property
    def _angular_is_circular(self) -> bool:
        """Whether the angular axis wraps circularly across the ±π seam.

        Derived from the connectivity graph rather than stored as a flag so
        it is always consistent with the actual edges and survives every
        serialization path (``to_file``/``from_file``, NWB round-trip,
        :meth:`copy`) that persists the connectivity graph. When
        :meth:`create` is called with ``circular_angle=True`` (and there is
        more than one angle bin) it adds seam edges connecting the first and
        last angle bins of each distance ring; with ``circular_angle=False``
        no such edges are added, leaving an *open* angular axis whose bins at
        ``-π`` and ``+π`` are genuinely far apart.

        A seam edge is detected as an edge between two bins on the same
        distance ring whose absolute angular separation is markedly larger
        than the smallest angular step in the graph (a regular angular edge
        spans one step; a seam edge spans ``(n_angle - 1)`` steps "the wrong
        way around"). Down-stream smoothing (``gaussian_kde``) reads this to
        decide whether to wrap ``Δθ`` into ``[-π, π]`` (circular) or treat the
        raw angular difference as-is (open), so an open boundary is respected
        and smoothing does not leak across it.

        Returns
        -------
        bool
            True if the angular axis is circular (seam edges present), False
            for an open angular axis.
        """
        connectivity = self.connectivity
        if connectivity.number_of_edges() == 0:
            return False

        # Collect per-edge (Δr, |Δθ|) using node "pos" attributes
        # (pos[0]=distance, pos[1]=angle). Same-ring edges have Δr ~ 0.
        same_ring_dtheta: list[float] = []
        min_step = np.inf
        for u, v in connectivity.edges():
            pos_u = connectivity.nodes[u]["pos"]
            pos_v = connectivity.nodes[v]["pos"]
            d_r = abs(float(pos_v[0]) - float(pos_u[0]))
            d_theta = abs(float(pos_v[1]) - float(pos_u[1]))
            if d_theta > 0:
                min_step = min(min_step, d_theta)
            # Same-ring edge: distance unchanged (radial component ~ 0).
            if d_r <= 1e-9 and d_theta > 0:
                same_ring_dtheta.append(d_theta)

        if not np.isfinite(min_step) or not same_ring_dtheta:
            return False

        # A seam edge spans many regular steps; a regular angular edge spans
        # one. Use a threshold of 1.5x the smallest step: seam edges (which
        # span (n_angle - 1) steps for n_angle >= 3) clear it comfortably,
        # while regular angular edges do not.
        threshold = 1.5 * min_step
        return any(dt > threshold for dt in same_ring_dtheta)
