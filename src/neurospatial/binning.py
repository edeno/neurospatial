"""Build an environment from your own bins with :meth:`Environment.from_bins`.

Use this module when another tool already defines your spatial bins, for
example reward wells, recording sites or the cells of a lab's own maze
model, and you want neurospatial's analyses on exactly those bins. You
describe the bins with four records and pass them to ``from_bins``:

- ``SpatialBins``: bin centers and each bin's size (area, length or volume,
  in your units).
- ``BinConnections``: which bins are neighbors, and the distance an animal
  travels between them.
- a locator, such as ``NearestBinLocator``: how a tracked position is
  assigned to a bin. ``max_distance`` is how far from a center a position
  can be and still belong to that bin.
- ``KnownFaces`` or ``MissingFaces``: the shared edges between bins, which
  smoothing and other numerical operations need. ``MissingFaces`` says they
  were not measured; lookup and graph distances still work, and operations
  that need faces raise.

The other classes describe grid, hexagon, mesh and track bins and are what
the built-in factories produce; ``env.binning`` returns them for inspection.
The environment keeps its own copy of these records, so editing arrays you
passed in later does not change it.

Examples
--------
Three reward wells, 2 cm apart, with measured path lengths between them:

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.binning import (
...     BinConnections,
...     Binning,
...     MissingFaces,
...     NearestBinLocator,
...     SpatialBins,
... )
>>> centers = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])  # cm
>>> env = Environment.from_bins(
...     Binning(
...         kind="reward_wells",
...         bins=SpatialBins.from_arrays(
...             centers=centers, sizes=np.ones(3), measure_dim=2
...         ),
...         connections=BinConnections.from_arrays(
...             pairs=np.array([[0, 1], [1, 2]]),
...             pair_distances=np.array([2.5, 3.0]),  # path lengths, cm
...         ),
...         locator=NearestBinLocator(centers, max_distance=0.5),
...         faces=MissingFaces("Only well centers were measured."),
...     ),
...     units="cm",
... )
>>> env.bin_at(np.array([[0.1, 0.0], [10.0, 0.0]])).tolist()
[0, -1]
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
