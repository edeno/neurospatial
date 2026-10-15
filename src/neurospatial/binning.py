"""Explicit owned geometry for :meth:`Environment.from_bins`.

``SpatialBins`` supplies bin IDs, physical measures and ordered neighbor pairs.
``Binning`` adds an explicit locator and either ``KnownFaces`` or
``MissingFaces``. Optional components establish grid, track and drawing
capabilities. Nearest-center lookup and scalar measures do not define cells.

Environment construction takes an independent snapshot of these records.
Their array exports can be copied and edited to prepare a new binning without
changing an existing environment. Delegated layout callbacks are not accepted.
"""

from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.layout._capabilities import RectilinearGrid
from neurospatial.layout._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.layout._diffusion import FaceGeometry
from neurospatial.layout._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import ClipMapping, ClippedLocator, PolygonCells

__all__ = [
    "Binning",
    "ClipMapping",
    "ClippedLocator",
    "FaceGeometry",
    "HexLocator",
    "HexagonShapes",
    "KnownFaces",
    "MeshLocator",
    "MissingFaces",
    "NearestBinLocator",
    "Outline",
    "PolygonCells",
    "RectilinearGrid",
    "RectilinearLocator",
    "SpatialBins",
    "SubsetLocator",
    "TrackLinearization",
    "TransformedLocator",
    "TriangleShapes",
]
