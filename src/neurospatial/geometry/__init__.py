"""Owned geometry records and pure geometry operations, independent of builders.

This package is internal. Other neurospatial packages import what they need
from here, never from its ``_`` modules. Users reach the records through
``neurospatial.arena``, ``neurospatial.binning`` and ``neurospatial.extensions``.
"""

from neurospatial.geometry._adapters import (
    SnapshotAdapter,
)
from neurospatial.geometry._arena import (
    Arena,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
    _CombinedArena,
    _TransformedArena,
    mapped_geometry,
)
from neurospatial.geometry._arena_fit import (
    apply_walls,
    infer_arena,
    validate_arena,
)
from neurospatial.geometry._arrays import (
    get_centers,
)
from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
    require_grid_context,
    require_rectilinear,
)
from neurospatial.geometry._boundaries import (
    analysis_interior,
    batch_line_of_sight_clear,
    boundary_geometry,
    boundary_view,
    line_of_sight_clear,
    physical_ray_hits,
    physical_walls,
    ray_hits,
    search_distance,
    valid_observers,
    wall_hit_bins,
)
from neurospatial.geometry._boundary_bins import (
    boundary_bins,
)
from neurospatial.geometry._capabilities import (
    RectilinearGrid,
)
from neurospatial.geometry._codecs import (
    decode,
    decode_arena,
    encode,
    encode_arena,
    graph_to_node_link,
    snapshot_provider,
)
from neurospatial.geometry._combine_geometry import (
    BridgeKey,
    arena_pieces,
    cell_boxes,
    combine_binnings,
)
from neurospatial.geometry._components import (
    HexagonShapes,
    Outline,
    TriangleShapes,
)
from neurospatial.geometry._composition import (
    CombinedLocator,
    bridge_values,
    combined_diffusion,
    composition_bin_sources,
    composition_component_keys,
    composition_view,
)
from neurospatial.geometry._context import (
    GridIndex,
    grid_index,
)
from neurospatial.geometry._coordinates import (
    CartesianSystem,
    CoordinateSystem,
    EgocentricPolarSystem,
    require_cartesian,
)
from neurospatial.geometry._derive import (
    select_bins,
    transform_binning,
)
from neurospatial.geometry._diffusion import (
    DiffusionGeometry,
    FaceGeometry,
)
from neurospatial.geometry._explicit import (
    binning_ranges,
    cell_extent,
    component_extent,
    prepare_bins,
)
from neurospatial.geometry._face_builders import (
    track_junctions,
)
from neurospatial.geometry._face_normals import (
    FaceNormals,
    geometric_normals,
)
from neurospatial.geometry._graph import (
    graph_from_binning,
)
from neurospatial.geometry._grid_frame import (
    FramedGrid,
)
from neurospatial.geometry._hex_lookup import (
    MIN_HEX_RADIUS,
    axial_to_offset_bin_indices,
    cartesian_to_fractional_cube,
    get_hex_grid_neighbor_deltas,
    points_to_hex_bin_ind,
    round_fractional_cube_to_integer_axial,
)
from neurospatial.geometry._locators import (
    CallIndexes,
    HexLocator,
    Locator,
    LookupIndexes,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._lookup import (
    locate,
    require_track_width,
)
from neurospatial.geometry._native_faces import (
    NativeFaces,
)
from neurospatial.geometry._ownership import (
    own_arena,
    own_binning,
)
from neurospatial.geometry._polygon import (
    cells_from_locator,
    clip_binning,
    materialize_cells,
)
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
)
from neurospatial.geometry._rebin_spec import (
    GridBinningSpec,
)
from neurospatial.geometry._region_bins import (
    polygon_bin_mask,
    region_bin_mask,
)
from neurospatial.geometry._regular_lookup import (
    points_to_regular_grid_bin_ind,
)
from neurospatial.geometry._track import (
    project_1d_to_2d,
)
from neurospatial.geometry._validation import (
    validate_components,
)
from neurospatial.geometry._wall_frame import (
    wall_source_frame,
)

__all__ = [
    "MIN_HEX_RADIUS",
    "Arena",
    "Binning",
    "BridgeKey",
    "CallIndexes",
    "CartesianSystem",
    "ClipMapping",
    "ClippedLocator",
    "CombinedLocator",
    "CoordinateSystem",
    "DiffusionGeometry",
    "EgocentricPolarSystem",
    "EgocentricSurround",
    "FaceGeometry",
    "FaceNormals",
    "FramedGrid",
    "GridBinningSpec",
    "GridIndex",
    "HexLocator",
    "HexagonShapes",
    "KnownFaces",
    "Locator",
    "LookupIndexes",
    "MaskArena",
    "MeshLocator",
    "MissingFaces",
    "NativeFaces",
    "NearestBinLocator",
    "OpenArena",
    "Outline",
    "PolygonCells",
    "RectilinearGrid",
    "RectilinearLocator",
    "SnapshotAdapter",
    "SpatialBins",
    "SubsetLocator",
    "TrackArena",
    "TrackLinearization",
    "TransformedLocator",
    "TriangleShapes",
    "UnknownArena",
    "_CombinedArena",
    "_TransformedArena",
    "analysis_interior",
    "apply_walls",
    "arena_pieces",
    "axial_to_offset_bin_indices",
    "batch_line_of_sight_clear",
    "binning_ranges",
    "boundary_bins",
    "boundary_geometry",
    "boundary_view",
    "bridge_values",
    "cartesian_to_fractional_cube",
    "cell_boxes",
    "cell_extent",
    "cells_from_locator",
    "clip_binning",
    "combine_binnings",
    "combined_diffusion",
    "component_extent",
    "composition_bin_sources",
    "composition_component_keys",
    "composition_view",
    "decode",
    "decode_arena",
    "encode",
    "encode_arena",
    "geometric_normals",
    "get_centers",
    "get_hex_grid_neighbor_deltas",
    "graph_from_binning",
    "graph_to_node_link",
    "grid_index",
    "infer_arena",
    "line_of_sight_clear",
    "locate",
    "mapped_geometry",
    "materialize_cells",
    "own_arena",
    "own_binning",
    "physical_ray_hits",
    "physical_walls",
    "points_to_hex_bin_ind",
    "points_to_regular_grid_bin_ind",
    "polygon_bin_mask",
    "prepare_bins",
    "project_1d_to_2d",
    "ray_hits",
    "region_bin_mask",
    "require_cartesian",
    "require_grid_context",
    "require_rectilinear",
    "require_track_width",
    "round_fractional_cube_to_integer_axial",
    "search_distance",
    "select_bins",
    "snapshot_provider",
    "track_junctions",
    "transform_binning",
    "valid_observers",
    "validate_arena",
    "validate_components",
    "wall_hit_bins",
    "wall_source_frame",
]
