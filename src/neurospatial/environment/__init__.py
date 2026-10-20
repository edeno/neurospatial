"""Spatial environments with owned geometry and explicit coordinate meaning.

Both Cartesian and polar factories return Environment. Public methods delegate
scientific operations to domain functions; coordinate-specific calls are guarded.
Use ``from neurospatial import Environment`` or this package's equivalent export.
``BinSequenceWithRuns`` is the result of ``Environment.bin_sequence_with_runs``
and ``BinMapping`` the result of ``Environment.bin_mapping_from``."""

from neurospatial.environment._bin_mapping import BinMapping
from neurospatial.environment._comparison import (
    _environment_differences as _environment_differences,
)
from neurospatial.environment._comparison import (
    _EnvironmentDifferences as _EnvironmentDifferences,
)
from neurospatial.environment._comparison import (
    _mapped_bin_difference as _mapped_bin_difference,
)
from neurospatial.environment._comparison import (
    _mapped_topology_difference as _mapped_topology_difference,
)
from neurospatial.environment._factories import _from_samples as _from_samples
from neurospatial.environment._publication import _build_record as _build_record
from neurospatial.environment._publication import (
    _declared_metadata as _declared_metadata,
)
from neurospatial.environment._publication import _Provenance as _Provenance
from neurospatial.environment._publication import _publish as _publish
from neurospatial.environment._targets import (
    _validate_sample_targets as _validate_sample_targets,
)
from neurospatial.environment._targets import (
    _validate_target_bins as _validate_target_bins,
)
from neurospatial.environment._trajectory import BinSequenceWithRuns
from neurospatial.environment.core import Environment
from neurospatial.geometry import CartesianSystem, EgocentricPolarSystem

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
__all__ = [
    "BinMapping",
    "BinSequenceWithRuns",
    "CartesianSystem",
    "EgocentricPolarSystem",
    "Environment",
]
