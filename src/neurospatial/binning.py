"""Explicit owned geometry for :meth:`Environment.from_bins`.

``SpatialBins`` supplies bin IDs and physical measures, and ``BinConnections``
their ordered neighbor pairs. ``Binning`` adds an explicit locator and either
``KnownFaces`` or ``MissingFaces``. Optional components establish grid, track and drawing
capabilities. Nearest-center lookup and scalar measures do not define cells.

Environment construction takes an independent snapshot of these records.
Their array exports can be copied and edited to prepare a new binning without
changing an existing environment. Delegated layout callbacks are not accepted.
"""

from neurospatial.geometry import (
    BinConnections,
    Binning,
    ClipMapping,
    ClippedLocator,
    DiffusionGeometry,
    FaceGeometry,
    FaceNormals,
    FramedGrid,
    GridBinningSpec,
    HexagonShapes,
    HexLocator,
    KnownFaces,
    MeshLocator,
    MissingFaces,
    NearestBinLocator,
    Outline,
    PolygonCells,
    RectilinearGrid,
    RectilinearLocator,
    SpatialBins,
    SubsetLocator,
    TrackLinearization,
    TransformedLocator,
    TriangleShapes,
)

__all__ = [
    "BinConnections",
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
