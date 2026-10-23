"""Immutable pixel-space outlines shared by assistance preview and acceptance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

from neurospatial._results import read_only_copy
from neurospatial.annotation.segmentation import SegmentationProposal, mask_to_polygons


def _tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Simplification tolerance must be finite and nonnegative in pixels."
        ) from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "Simplification tolerance must be finite and nonnegative in pixels."
        )
    return tolerance


@dataclass(frozen=True)
class PolygonOutline:
    """A proposal and the exact polygon geometry currently shown for acceptance.

    Polygons use pixel (x, y) coordinates, including every component and hole.
    The proposal retains the original mask and provider provenance. Shapely
    geometries are immutable; ring arrays returned for rendering are independent.
    """

    proposal: SegmentationProposal
    polygons: tuple[Polygon, ...]
    simplify_tolerance_px: float = 0.0

    def __post_init__(self) -> None:
        polygons = tuple(self.polygons)
        if not polygons or any(
            not isinstance(polygon, Polygon) or polygon.is_empty or not polygon.is_valid
            for polygon in polygons
        ):
            raise ValueError("An outline needs nonempty, valid polygons.")
        if not MultiPolygon(polygons).is_valid:
            raise ValueError("Outline components must form a valid MultiPolygon.")
        object.__setattr__(self, "polygons", polygons)
        object.__setattr__(
            self, "simplify_tolerance_px", _tolerance(self.simplify_tolerance_px)
        )

    @property
    def vertex_count(self) -> int:
        """Count vertices in every exterior and hole, excluding closure copies."""
        return sum(
            len(ring.coords) - 1
            for polygon in self.polygons
            for ring in (polygon.exterior, *polygon.interiors)
        )

    @property
    def rings(self) -> tuple[NDArray[np.float64], ...]:
        """Return closed exterior and interior paths in pixel (x, y) order."""
        arrays = []
        for polygon in self.polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                coords = read_only_copy(ring.coords, np.float64)
                arrays.append(coords)
        return tuple(arrays)


def polygon_outline(proposal: SegmentationProposal) -> PolygonOutline:
    """Vectorize the original mask once, preserving exact foreground support."""
    return PolygonOutline(proposal, tuple(mask_to_polygons(proposal.mask)))


def simplify_polygon_outline(
    exact_outline: PolygonOutline, tolerance_px: float
) -> PolygonOutline:
    """Derive a preview from cached exact geometry without changing its mask.

    Simplifying the complete MultiPolygon together preserves relationships
    between components as well as holes. No dilation or hole filling occurs.
    """
    tolerance = _tolerance(tolerance_px)
    if exact_outline.simplify_tolerance_px != 0.0:
        raise ValueError("Simplify the cached exact outline, not an earlier preview.")
    if tolerance == 0.0:
        return exact_outline
    geometry = MultiPolygon(exact_outline.polygons).simplify(
        tolerance, preserve_topology=True
    )
    if not isinstance(geometry, Polygon | MultiPolygon) or not geometry.is_valid:
        raise ValueError("Simplification did not preserve valid outline geometry.")
    # GEOS may unwrap a one-component MultiPolygon into a Polygon.
    polygons = (geometry,) if isinstance(geometry, Polygon) else tuple(geometry.geoms)
    if len(polygons) != len(exact_outline.polygons) or sorted(
        len(polygon.interiors) for polygon in polygons
    ) != sorted(len(polygon.interiors) for polygon in exact_outline.polygons):
        raise ValueError("Simplification did not preserve every component and hole.")
    return PolygonOutline(exact_outline.proposal, polygons, tolerance)
