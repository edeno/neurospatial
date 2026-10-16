"""Spatial environments with owned geometry and explicit coordinate meaning.

Both Cartesian and polar factories return Environment. Public methods delegate
scientific operations to domain functions; coordinate-specific calls are guarded.
Use ``from neurospatial import Environment`` or this package's equivalent export."""

from neurospatial.environment.core import Environment
from neurospatial.environment.decorators import EnvironmentNotFittedError, check_fitted
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem

__all__ = [
    "CartesianSystem",
    "EgocentricPolarSystem",
    "Environment",
    "EnvironmentNotFittedError",
    "check_fitted",
]
