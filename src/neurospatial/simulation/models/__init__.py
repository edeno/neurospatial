"""Neural models for simulating spatial firing patterns.

This module provides implementations of various spatially tuned neural models:
- PlaceCellModel: Gaussian place fields
- BoundaryCellModel: Boundary-distance tuned cells
- GridCellModel: Hexagonal grid patterns
- HeadDirectionCellModel: Directional tuning
- SpeedCellModel: Speed-modulated firing

All models implement the NeuralModel protocol.
"""

# Milestone 1: Place cells
from neurospatial.simulation.models.base import NeuralModel
from neurospatial.simulation.models.place_cells import PlaceCellModel

# Milestone 2: Boundary cells
# from neurospatial.simulation.models.boundary_cells import BoundaryCellModel

# Milestone 3: Grid cells
# from neurospatial.simulation.models.grid_cells import GridCellModel

# Optional Phase 4: Additional cell types
# from neurospatial.simulation.models.head_direction import HeadDirectionCellModel
# from neurospatial.simulation.models.speed_cells import SpeedCellModel

__all__ = [
    "NeuralModel",
    "PlaceCellModel",
]
