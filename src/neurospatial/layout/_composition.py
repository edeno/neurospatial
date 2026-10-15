"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._composition import (
    CombinedLocator as CombinedLocator,
)
from neurospatial.geometry._composition import (
    _bin_sources as _bin_sources,
)
from neurospatial.geometry._composition import (
    _component_keys as _component_keys,
)
from neurospatial.geometry._composition import (
    bridge_values as bridge_values,
)
from neurospatial.geometry._composition import (
    combined_diffusion as combined_diffusion,
)
from neurospatial.geometry._composition import (
    combined_normals as combined_normals,
)
from neurospatial.geometry._composition import (
    composition_view as composition_view,
)
from neurospatial.geometry._composition import (
    expected_composition_binning as expected_composition_binning,
)
