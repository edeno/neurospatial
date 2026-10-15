"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._locators import (
    AffineMatrix as AffineMatrix,
)
from neurospatial.geometry._locators import (
    HexLocator as HexLocator,
)
from neurospatial.geometry._locators import (
    MeshLocator as MeshLocator,
)
from neurospatial.geometry._locators import (
    NearestBinLocator as NearestBinLocator,
)
from neurospatial.geometry._locators import (
    RectilinearLocator as RectilinearLocator,
)
from neurospatial.geometry._locators import (
    SubsetLocator as SubsetLocator,
)
from neurospatial.geometry._locators import (
    TransformedLocator as TransformedLocator,
)
from neurospatial.geometry._locators import (
    _check_homogeneous as _check_homogeneous,
)
from neurospatial.geometry._locators import (
    _LocatorState as _LocatorState,
)
from neurospatial.geometry._locators import (
    _owned_array as _owned_array,
)
