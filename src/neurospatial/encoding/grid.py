"""
Grid cell analysis.

This module provides tools for analyzing grid cells, including spatial
autocorrelation, grid score (hexagonal periodicity), grid scale (spacing),
grid orientation, and periodicity score for irregular topologies.

Imports
-------
All functions can be imported directly from this module:

>>> from neurospatial.encoding.grid import (
...     grid_score,
...     spatial_autocorrelation,
...     grid_scale,
...     grid_orientation,
...     grid_properties,
...     periodicity_score,
...     GridProperties,
... )

Or via the encoding package:

>>> from neurospatial.encoding import grid_score, GridProperties

Functions
---------
spatial_autocorrelation
    Compute spatial autocorrelation of a firing rate map (FFT or graph-based).
grid_score
    Compute grid score (hexagonal periodicity) from 2D autocorrelation.
grid_scale
    Compute grid spacing from 2D autocorrelogram.
grid_orientation
    Compute grid orientation from 2D autocorrelogram.
grid_properties
    Compute all grid cell metrics from 2D autocorrelogram.
periodicity_score
    Compute periodicity score from distance-correlation profile (graph-based).

Classes
-------
GridProperties
    Container dataclass for grid cell metrics.

References
----------
Sargolini, F., Fyhn, M., Hafting, T., McNaughton, B. L., Witter, M. P., Moser, M. B.,
    & Moser, E. I. (2006). Conjunctive representation of position, direction, and
    velocity in entorhinal cortex. Science, 312(5774), 758-762.
"""

from neurospatial.metrics.grid_cells import (
    GridProperties,
    grid_orientation,
    grid_properties,
    grid_scale,
    grid_score,
    periodicity_score,
    spatial_autocorrelation,
)

__all__ = [
    # Dataclass
    "GridProperties",
    # Functions
    "grid_orientation",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "periodicity_score",
    "spatial_autocorrelation",
]
