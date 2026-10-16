"""Explicit owned geometry for :meth:`Environment.from_bins`.

``SpatialBins`` supplies bin IDs, physical measures and ordered neighbor pairs.
``Binning`` adds an explicit locator and either ``KnownFaces`` or
``MissingFaces``. Optional components establish grid, track and drawing
capabilities. Nearest-center lookup and scalar measures do not define cells.

Environment construction takes an independent snapshot of these records.
Their array exports can be copied and edited to prepare a new binning without
changing an existing environment. Delegated layout callbacks are not accepted.
"""

from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.geometry._diffusion import DiffusionGeometry, FaceGeometry
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._grid_frame import FramedGrid
from neurospatial.geometry._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon import ClipMapping, ClippedLocator, PolygonCells
from neurospatial.geometry._rebin_spec import GridBinningSpec

__all__ = [
    "Binning",
    "ClipMapping",
    "ClippedLocator",
    "DiffusionGeometry",
    "FaceGeometry",
    "FaceNormals",
    "FramedGrid",
    "GridBinningSpec",
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
