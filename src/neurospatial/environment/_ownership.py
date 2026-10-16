"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._ownership import (
    _check_binning as _check_binning,
)
from neurospatial.geometry._ownership import (
    _check_locator as _check_locator,
)
from neurospatial.geometry._ownership import (
    _migration as _migration,
)
from neurospatial.geometry._ownership import (
    _snapshot_binning as _snapshot_binning,
)
from neurospatial.geometry._ownership import (
    _snapshot_locator as _snapshot_locator,
)
from neurospatial.geometry._ownership import (
    own_arena as own_arena,
)
from neurospatial.geometry._ownership import (
    own_binning as own_binning,
)
