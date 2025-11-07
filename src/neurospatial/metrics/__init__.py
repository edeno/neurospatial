"""
Neuroscience metrics for spatial analysis.

This module provides standard neuroscience metrics for place cells, boundary
cells, and population-level analyses validated against field-standard packages
(opexebo, neurocode, buzcode).

Modules
-------
place_fields
    Place field detection and single-cell spatial metrics.

"""

from __future__ import annotations

from neurospatial.metrics.place_fields import (
    detect_place_fields,
    field_centroid,
    field_size,
    field_stability,
    skaggs_information,
    sparsity,
)

__all__ = [
    "detect_place_fields",
    "field_centroid",
    "field_size",
    "field_stability",
    "skaggs_information",
    "sparsity",
]
