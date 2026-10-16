"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._binning import (
    Binning as Binning,
)
from neurospatial.geometry._binning import (
    FacesSource as FacesSource,
)
from neurospatial.geometry._binning import (
    KnownFaces as KnownFaces,
)
from neurospatial.geometry._binning import (
    Locator as Locator,
)
from neurospatial.geometry._binning import (
    MissingFaces as MissingFaces,
)
from neurospatial.geometry._binning import (
    SpatialBins as SpatialBins,
)
from neurospatial.geometry._binning import (
    TrackLinearization as TrackLinearization,
)
from neurospatial.geometry._binning import (
    _ReadOnlyTrackAttributes as _ReadOnlyTrackAttributes,
)
