"""Neural models for simulating spatial firing patterns.

This module provides implementations of various spatially tuned neural models:
- PlaceCellModel: Gaussian place fields
- BoundaryCellModel: Boundary-distance tuned cells
- GridCellModel: Hexagonal grid patterns
- ObjectVectorCellModel: Object-distance and direction tuned cells
- SpatialViewCellModel: Gaze-based spatial view cells
- HeadDirectionCellModel: Von Mises directional tuning
- SpeedCellModel: Speed-modulated firing (planned)

All models implement the NeuralModel protocol.
"""

# Base protocol
from neurospatial.simulation.models.base import NeuralModel

# Boundary cells
from neurospatial.simulation.models.boundary_cells import BoundaryCellModel

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

# Planned: Additional cell types
# from neurospatial.simulation.models.speed_cells import SpeedCellModel

__all__ = [
    "BoundaryCellModel",
    "GridCellModel",
    "HeadDirectionCellModel",
    "NeuralModel",
    "ObjectVectorCellModel",
    "PlaceCellModel",
    "SpatialViewCellModel",
]
