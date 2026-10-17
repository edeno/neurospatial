"""Spatial environments with owned geometry and explicit coordinate meaning.

Both Cartesian and polar factories return Environment. Public methods delegate
scientific operations to domain functions; coordinate-specific calls are guarded.
Use ``from neurospatial import Environment`` or this package's equivalent export.
``BinSequenceWithRuns`` is the result of ``Environment.bin_sequence_with_runs``."""

from neurospatial.environment._factories import build_from_samples as build_from_samples
from neurospatial.environment._publication import Provenance as Provenance
from neurospatial.environment._publication import declared_metadata as declared_metadata
from neurospatial.environment._publication import publish as publish
from neurospatial.environment._targets import (
    validate_sample_targets as validate_sample_targets,
)
from neurospatial.environment._targets import (
    validate_target_bins as validate_target_bins,
)
from neurospatial.environment._trajectory import BinSequenceWithRuns
from neurospatial.environment.core import Environment
from neurospatial.geometry import CartesianSystem, EgocentricPolarSystem

# Names imported here but not listed in ``__all__`` are shared with other
# neurospatial packages. They are internal, not public API.
__all__ = [
    "BinSequenceWithRuns",
    "CartesianSystem",
    "EgocentricPolarSystem",
    "Environment",
]
