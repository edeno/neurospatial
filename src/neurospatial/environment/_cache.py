"""One memo of the analyses derived from an environment's published geometry.

Every value here depends only on geometry that never changes after
publication, so a value computed once stays correct for the environment's
lifetime. ``with_metadata`` and ``with_regions`` copies keep the same geometry
and therefore share their parent's cache; every derivation, reader and
unpickled environment starts with an empty one. The cache and key classes
are defined in :mod:`neurospatial._cache`, below every package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neurospatial import _constants
from neurospatial._cache import AnalysisCache as AnalysisCache
from neurospatial._cache import CacheKey as CacheKey
from neurospatial._cache import RecentKey as RecentKey

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from scipy import sparse
    from scipy.spatial import cKDTree
    from shapely import Polygon

    from neurospatial.geometry import DiffusionGeometry, FaceGeometry
    from neurospatial.ops import KernelMode, _DiffusionEigenbasis, _MRFEigenbasis

KDTREE: CacheKey[cKDTree] = CacheKey("kdtree")
CELL_POLYGONS: CacheKey[tuple[Polygon, ...]] = CacheKey("cell_polygons")
# The KD-tree over the bin centers' Cartesian embedding, for polar nearest_bin.
EMBEDDED_KDTREE: CacheKey[tuple[NDArray[Any], cKDTree]] = CacheKey("embedded_kdtree")
BIN_SPACING: CacheKey[float] = CacheKey("bin_spacing")
# Each bin's longest shortest path to a connected bin, along the bin graph.
BIN_GRAPH_ECCENTRICITIES: CacheKey[NDArray[Any]] = CacheKey("bin_graph_eccentricities")
# The longest shortest path between two connected bins, along the bin graph.
BIN_GRAPH_DIAMETER: CacheKey[float] = CacheKey("bin_graph_diameter")
# The same for the any-angle (geodesic) path between bin centers.
GEODESIC_DIAMETER: CacheKey[float] = CacheKey("geodesic_diameter")
FACE_GEOMETRY: CacheKey[FaceGeometry] = CacheKey("face_geometry")
DIFFUSION_GEOMETRY: CacheKey[DiffusionGeometry] = CacheKey("diffusion_geometry")
# Records that the mesh skew check ran (and its text, unread), so smoothing
# warns once per cached geometry.
MESH_SKEW_WARNING: CacheKey[str | None] = CacheKey("mesh_skew_warning")
DIFFUSION_EIGENBASIS: CacheKey[_DiffusionEigenbasis] = CacheKey("diffusion_eigenbasis")
MRF_EIGENBASIS: CacheKey[_MRFEigenbasis] = CacheKey("mrf_eigenbasis")
DIFFERENTIAL_OPERATOR: CacheKey[sparse.csc_matrix] = CacheKey("differential_operator")
DIVERGENCE_OPERATOR: CacheKey[sparse.csc_matrix] = CacheKey("divergence_operator")
BOUNDARY_BINS: CacheKey[NDArray[Any]] = CacheKey("boundary_bins")
BIN_ATTRIBUTES: CacheKey[pd.DataFrame] = CacheKey("bin_attributes")
EDGE_ATTRIBUTES: CacheKey[pd.DataFrame] = CacheKey("edge_attributes")
DIFFUSION_KERNELS: RecentKey[tuple[float, KernelMode], NDArray[Any]] = RecentKey(
    "diffusion_kernels", limit=_constants.MAX_CACHED_KERNELS
)
GAUSSIAN_KERNELS: RecentKey[float, NDArray[Any]] = RecentKey(
    "gaussian_kernels", limit=_constants.MAX_CACHED_KERNELS
)
