"""Neural models for simulating spatial firing patterns.

This module provides implementations of various spatially tuned neural models:
- PlaceCellModel: Gaussian place fields
- BoundaryCellModel: Boundary-distance tuned cells
- GridCellModel: Hexagonal grid patterns
- ObjectVectorCellModel: Object-distance and direction tuned cells
- HeadDirectionCellModel: Directional tuning (planned)
- SpeedCellModel: Speed-modulated firing (planned)

All models implement the NeuralModel protocol.
"""

# Base protocol
from neurospatial.simulation.models.base import NeuralModel

# Boundary cells
from neurospatial.simulation.models.boundary_cells import BoundaryCellModel

# Grid cells
from neurospatial.simulation.models.grid_cells import GridCellModel

# Object-vector cells
from neurospatial.simulation.models.object_vector_cells import ObjectVectorCellModel

# Place cells
from neurospatial.simulation.models.place_cells import PlaceCellModel

# Planned: Additional cell types
# from neurospatial.simulation.models.head_direction import HeadDirectionCellModel
# from neurospatial.simulation.models.speed_cells import SpeedCellModel

__all__ = [
    "BoundaryCellModel",
    "GridCellModel",
    "NeuralModel",
    "ObjectVectorCellModel",
    "PlaceCellModel",
]
