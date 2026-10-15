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
nonsense. Cartesian methods are absent, and accessing one raises
``CartesianOnlyError``, both a ``NotImplementedError`` and an ``AttributeError``.
These fall into two groups: Cartesian-coordinate methods (``bin_at``,
``contains``, ``distance_between``, ``apply_transform``, ``interpolate``,
``occupancy``, ``bin_sequence``, ``bin_sequence_with_runs``) and Cartesian-grid
methods (``to_linear``, ``linear_to_nd``, ``rebin``, ``subset``). Shared methods
refuse their Cartesian modes, including Euclidean ``distance_to``.

Geodesic operations remain well-defined because they read only the
connectivity graph, whose edge ``distance`` weights are physical polar
lengths (arc ``r·Δθ`` for angular steps, ``Δr`` for radial steps, and
``sqrt(Δr² + (r·Δθ)²)`` for diagonals).

The public entry point is :meth:`Environment.from_polar_egocentric`, which
delegates to :meth:`EgocentricPolarEnvironment.create`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.layout.base import LayoutEngine
    from neurospatial.layout.factories import LayoutType

from neurospatial._exceptions import CartesianOnlyError
from neurospatial.environment._coordinates import EgocentricPolarSystem
from neurospatial.environment.cartesian import (
    CARTESIAN_GRID_OPERATIONS,
    CARTESIAN_OPERATIONS,
)
from neurospatial.environment.core import _BaseEnvironment
from neurospatial.environment.decorators import check_fitted, versioned_cached_property
from neurospatial.layout.engines._polar import _PolarGridLayout
from neurospatial.layout.helpers.utils import find_boundary_nodes

_POLAR_EXPLANATION = (
    "this is a polar environment whose bin_centers[:, 0] is distance and "
    "bin_centers[:, 1] is angle in radians, so operations that assume Cartesian "
    "(x, y[, z]) coordinates or a Cartesian grid would silently return "
    "geometric nonsense. Use connectivity-graph operations instead (neighbors, "
    "path_between, reachable_from, smooth, or distance_to(metric='geodesic')), "
    "or build an allocentric Cartesian env via Environment.from_samples / "
    "from_polygon / from_grid_mask."
)


class EgocentricPolarEnvironment(_BaseEnvironment):
    """A discretized egocentric polar environment.

    ``bin_centers[:, 0]`` is distance (radius) in physical length units and
    ``bin_centers[:, 1]`` is angle in radians. Build one via
    :meth:`Environment.from_polar_egocentric` (the public factory) or
    :meth:`create` (the underlying constructor).

    This type is a sibling of :class:`Environment`. Cartesian methods are
    absent: accessing one raises ``CartesianOnlyError``, both a
    ``NotImplementedError`` and an ``AttributeError``, so ``hasattr`` reports
    them as absent and type checkers flag calls.

    *Cartesian-coordinate methods* (would read ``(distance, angle)`` as
    ``(x, y)``):

    - ``bin_at``, ``contains`` — geometric containment in ``(x, y)``.
    - ``distance_between`` — straight-line distance on ``(x, y)``.
    - ``apply_transform`` — affine transforms on ``(x, y)``.
    - ``interpolate`` — evaluates fields at Cartesian query points.
    - ``occupancy``, ``bin_sequence``, ``bin_sequence_with_runs``
      — map continuous ``(x, y)`` trajectories onto bins.

    *Cartesian-grid methods* (operate on a rectangular grid that has no polar
    meaning):

    - ``to_linear``, ``linear_to_nd`` — 1D track linearization.
    - ``rebin``, ``subset`` — grid coarsening / sub-selection
      (and would return a Cartesian :class:`Environment`).

    Shared methods keep working but refuse their Cartesian modes with the same
    error: ``distance_to(metric="euclidean")``, a coordinate point passed to
    ``neighbors``, ``path_between`` or ``reachable_from``, point regions in
    ``bins_in_region`` and ``region_mask``, and ``transitions`` computed from
    times and positions.

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

    @property
    def _coordinate_system(self) -> EgocentricPolarSystem:
        """Polar distance and axis semantics, with the graph's circularity."""
        return cast("EgocentricPolarSystem", self._geometry.coordinates)

    def _require_cartesian(self, operation: str) -> NoReturn:
        """Refuse ``operation``: polar bin centers are ``(distance, angle)``."""
        raise CartesianOnlyError(
            type(self).__name__,
            operation,
            reason="Cartesian-coordinate operation",
            explanation=_POLAR_EXPLANATION,
        )

    if not TYPE_CHECKING:
        # Hidden from type checkers so ``polar_env.bin_at`` is a static error;
        # at runtime each Cartesian method name raises CartesianOnlyError.
        def __getattr__(self, name: str) -> NoReturn:
            if name in CARTESIAN_OPERATIONS:
                grid = name in CARTESIAN_GRID_OPERATIONS
                raise CartesianOnlyError(
                    type(self).__name__,
                    f"{name}()",
                    reason=(
                        "Cartesian-grid operation"
                        if grid
                        else "Cartesian-coordinate operation"
                    ),
                    explanation=_POLAR_EXPLANATION,
                )
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=self,
            )

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
        units: str | None = None,
        frame: str | None = None,
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
        units : str, optional
            Unit label for distance coordinates; angles remain in radians.
            Declaring units does not scale either coordinate.
        frame : str, optional
            Coordinate frame label for the supplied coordinates.

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
                units=units,
                frame=frame,
            ),
        )

        return env

    @property
    @check_fitted
    def boundary_bins(self) -> NDArray[np.intp]:
        """Boundary-bin indices, exported independently from the private cache."""
        return cast("NDArray[np.intp]", self._boundary_bins_cached.copy())

    @versioned_cached_property
    @check_fitted
    def _boundary_bins_cached(self) -> NDArray[np.intp]:
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

        Derived from the connectivity graph when there are at least three
        angular bins, so it is consistent with the actual edges and survives every
        serialization path (``to_file``/``from_file``, NWB round-trip,
        :meth:`copy`) that persists the connectivity graph. When
        :meth:`create` is called with ``circular_angle=True`` (and there is
        more than one angle bin) it adds seam edges connecting the first and
        last angle bins of each distance ring; with ``circular_angle=False``
        no such edges are added, leaving an *open* angular axis whose bins at
        ``-π`` and ``+π`` are genuinely far apart.

        With one or two angular bins, open and circular axes have identical
        simple graphs. Use the layout's saved ``circular_angle`` option in
        that case. Legacy NWB records without this option retain the previous
        open-axis interpretation for these indistinguishable graphs.

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
            True if the angular axis is circular, False for an open axis.
        """
        return self._coordinate_system.circular
