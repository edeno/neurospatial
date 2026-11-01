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
from neurospatial.geometry._lookup import outline_sliver_regions


def _clip(
    binning: Binning, arena: Arena, polygon: Polygon | MultiPolygon
) -> _Derivation:
    """Intersect represented cells with an analysis polygon and known domain."""
    if binning.bins.n_dims != 2:
        msg = "Exact polygon clipping requires Cartesian 2-D cells."
        raise NotImplementedError(msg)
    if bridge_values(binning):
        msg = (
            "Clipping navigation bridge endpoints needs an explicit reconnection rule. "
            "Combine with auto_bridge=False before clipping, or clip the children first."
        )
        raise NotImplementedError(msg)
    domain = arena.polygonal_domain()
    if arena.domain_known and domain is None:
        msg = (
            "Exact clipping requires a polygonal representation of the known "
            "arena domain. Provide polygon geometry or use whole-bin subset()."
        )
        raise NotImplementedError(msg)
    clipped, _ = clip_binning(
        binning,
        polygon,
        domain=domain,
        walls=arena.walls,
        slivers=outline_sliver_regions(binning, arena),
    )
    # Publication records the clipped cells' extent as the ranges.
    return _Derivation(clipped, "clip")
