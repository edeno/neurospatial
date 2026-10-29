"""Neural models for simulating spatial firing patterns.

This module provides implementations of various spatially tuned neural models:
- PlaceCellModel: Gaussian place fields
- BorderCellModel: Firing from the distance to the nearest wall
- BoundaryVectorCellModel: Walls at a preferred distance and direction
- GridCellModel: Hexagonal grid patterns
- ObjectVectorCellModel: Object-distance and direction tuned cells
- SpatialViewCellModel: Gaze-based spatial view cells
- HeadDirectionCellModel: Von Mises directional tuning

All models implement the NeuralModel protocol:
``firing_rate(times, positions, *, headings=None)``, ``requires_headings``
and ``ground_truth``.
"""

# Base protocol
from neurospatial.simulation.models.base import NeuralModel

# Border and boundary-vector cells
from neurospatial.simulation.models.boundary_cells import (
    BorderCellModel,
    BoundaryVectorCellModel,
)

# Grid cells
from neurospatial.simulation.models.grid_cells import GridCellModel

# Head direction cells
from neurospatial.simulation.models.head_direction_cells import HeadDirectionCellModel

# Object-vector cells
from neurospatial.simulation.models.object_vector_cells import ObjectVectorCellModel

# Place cells
from neurospatial.simulation.models.place_cells import PlaceCellModel

# Spatial view cells
from neurospatial.simulation.models.spatial_view_cells import SpatialViewCellModel

__all__ = [
    "BorderCellModel",
    "BoundaryVectorCellModel",
    "GridCellModel",
    "HeadDirectionCellModel",
    "NeuralModel",
    "ObjectVectorCellModel",
    "PlaceCellModel",
    "SpatialViewCellModel",
]
