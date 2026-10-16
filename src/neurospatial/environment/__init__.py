"""Spatial environments with owned geometry and explicit coordinate meaning.

Both Cartesian and polar factories return Environment. Public methods delegate
scientific operations to domain functions; coordinate-specific calls are guarded.
Use ``from neurospatial import Environment`` or this package's equivalent export.
``BinSequenceWithRuns`` is the result of ``Environment.bin_sequence_with_runs``."""

from neurospatial.environment._trajectory import BinSequenceWithRuns
from neurospatial.environment.core import Environment
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem

__all__ = [
    "BinSequenceWithRuns",
    "CartesianSystem",
    "EgocentricPolarSystem",
    "Environment",
]
