"""Derive exact polygon clips for publication as ordinary environments."""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon

from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import (
    Arena,
    Binning,
    bridge_values,
    clip_binning,
)


def _clip(
    binning: Binning, arena: Arena, polygon: Polygon | MultiPolygon
) -> _Derivation:
    """Intersect represented cells with an analysis polygon and known domain."""
    if binning.bins.n_dims != 2:
        raise NotImplementedError(
            "Exact polygon clipping requires Cartesian 2-D cells."
        )
    if bridge_values(binning):
        raise NotImplementedError(
            "Clipping navigation bridge endpoints needs an explicit reconnection rule. "
            "Combine with auto_bridge=False before clipping, or clip the children first."
        )
    domain = arena.polygonal_domain()
    if arena.domain_known and domain is None:
        raise NotImplementedError(
            "Exact clipping requires a polygonal representation of the known "
            "arena domain. Provide polygon geometry or use whole-bin subset()."
        )
    clipped, _ = clip_binning(
        binning,
        polygon,
        domain=domain,
        walls=arena.walls,
    )
    # Publication records the clipped cells' extent as the ranges.
    return _Derivation(clipped, "clip")
