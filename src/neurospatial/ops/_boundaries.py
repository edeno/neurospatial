"""Exact planar wall evidence shared by border and visibility consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import shapely
from shapely.ops import unary_union

if TYPE_CHECKING:
    from neurospatial.arena import Arena
    from neurospatial.ops._protocols import BoundaryConsumer

BoundarySource = Literal["physical", "analysis"]


def boundary_geometry(
    env: BoundaryConsumer, source: BoundarySource
) -> shapely.Geometry:
    """Return declared physical walls or explicitly selected analysis boundaries.

    Physical mode uses exact planar footprint rings and explicit internal walls.
    It never infers physical walls from bins, sampling masks or unknown domains.
    Analysis mode uses the exact union of retained planar analysis cells.
    """
    if source not in ("physical", "analysis"):
        raise ValueError("boundary_source must be 'physical' or 'analysis'.")
    if env.arena.coordinate_kind != "cartesian" or env.arena.n_dims != 2:
        raise NotImplementedError(
            "Wall geometry requires planar Cartesian coordinates."
        )
    if source == "analysis":
        from neurospatial.layout._polygon import materialize_cells

        try:
            return unary_union(materialize_cells(env._binning)).boundary
        except ValueError as exc:
            raise NotImplementedError(
                "Analysis-boundary geometry needs exact planar cells; supply supported grid or polygon cells."
            ) from exc
    return physical_walls(env.arena)


def physical_walls(arena: Arena) -> shapely.Geometry:
    """Read asserted planar wall evidence independently of analysis bins."""
    from neurospatial.arena import _CombinedArena, _mapped_geometry, _TransformedArena

    def collect(value: Arena) -> list[shapely.Geometry]:
        if isinstance(value, _CombinedArena):
            return [part for child in value.children for part in collect(child)]
        if isinstance(value, _TransformedArena):
            return [
                _mapped_geometry(part, value.forward) for part in collect(value.base)
            ]
        parts: list[shapely.Geometry] = list(value.walls)
        if value.physical_footprint_known:
            domain = value.polygonal_domain()
            if domain is None:
                raise NotImplementedError(
                    "Physical footprint has no exact planar polygon boundary; curved tracks require a separate wall representation. Supply explicit polygonal geometry or choose boundary_source='analysis' when supported."
                )
            parts.append(domain.boundary)
        return parts

    parts = collect(arena)
    if not parts:
        raise NotImplementedError(
            "Physical walls are unknown: positions and analysis masks do not establish a measured footprint. Supply an asserted OpenArena or MaskArena(physical_footprint_known=True), or explicitly choose boundary_source='analysis'."
        )
    return unary_union(parts)
