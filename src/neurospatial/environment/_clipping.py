"""Derive exact polygon clips for publication as ordinary environments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from neurospatial.environment._publication import _Derivation
from neurospatial.geometry._arena import Arena
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._composition import bridge_values
from neurospatial.geometry._polygon import clip_binning
from neurospatial.geometry._polygon_cells import PolygonCells


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
    clipped, mapping = clip_binning(
        binning,
        polygon,
        domain=domain,
        walls=arena.walls,
    )
    shapes = clipped.shapes
    assert isinstance(shapes, PolygonCells)
    bounds = np.array([cell.bounds for cell in shapes.polygons])
    ranges = tuple(
        (float(bounds[:, axis].min()), float(bounds[:, axis + 2].max()))
        for axis in range(2)
    )
    return _Derivation(
        replace(clipped, clip_mapping=mapping, dimension_ranges=ranges), "clip"
    )
