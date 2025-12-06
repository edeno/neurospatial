"""Backward-compatibility shim for alignment module.

This module re-exports all symbols from the new location at
`neurospatial.ops.alignment`. New code should import from
`neurospatial.ops.alignment` directly.

Examples
--------
Old import (still works):
    >>> from neurospatial.alignment import map_probabilities

New import (recommended):
    >>> from neurospatial.ops.alignment import map_probabilities
"""

from neurospatial.ops.alignment import (
    ProbabilityMappingParams,
    apply_similarity_transform,
    get_2d_rotation_matrix,
    map_probabilities,
)

__all__ = [
    "ProbabilityMappingParams",
    "apply_similarity_transform",
    "get_2d_rotation_matrix",
    "map_probabilities",
]
