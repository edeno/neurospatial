"""Owned geometry records and pure geometry operations, independent of builders.

This package is internal. Other neurospatial packages import what they need
from here, never from its ``_`` modules. Users reach the records through
``neurospatial.arena``, ``neurospatial.binning`` and ``neurospatial.extensions``.
"""

from neurospatial.geometry._adapters import SnapshotAdapter as SnapshotAdapter
from neurospatial.geometry._arena import Arena as Arena
from neurospatial.geometry._arena import EgocentricSurround as EgocentricSurround
from neurospatial.geometry._arena import MaskArena as MaskArena
from neurospatial.geometry._arena import OpenArena as OpenArena
from neurospatial.geometry._arena import TrackArena as TrackArena
from neurospatial.geometry._arena import UnknownArena as UnknownArena
from neurospatial.geometry._arena import _CombinedArena as _CombinedArena
from neurospatial.geometry._arena import _TransformedArena as _TransformedArena
from neurospatial.geometry._arena import mapped_geometry as mapped_geometry
from neurospatial.geometry._arena_fit import apply_walls as apply_walls
from neurospatial.geometry._arena_fit import fit_walls as fit_walls
from neurospatial.geometry._arena_fit import infer_arena as infer_arena
from neurospatial.geometry._arena_fit import validate_arena as validate_arena
from neurospatial.geometry._arrays import get_centers as get_centers
from neurospatial.geometry._binning import Binning as Binning
from neurospatial.geometry._binning import KnownFaces as KnownFaces
from neurospatial.geometry._binning import MissingFaces as MissingFaces
from neurospatial.geometry._binning import SpatialBins as SpatialBins
from neurospatial.geometry._binning import TrackLinearization as TrackLinearization
from neurospatial.geometry._binning import require_grid_context as require_grid_context
from neurospatial.geometry._binning import require_rectilinear as require_rectilinear
from neurospatial.geometry._boundaries import analysis_interior as analysis_interior
from neurospatial.geometry._boundaries import (
    batch_line_of_sight_clear as batch_line_of_sight_clear,
)
from neurospatial.geometry._boundaries import boundary_geometry as boundary_geometry
from neurospatial.geometry._boundaries import boundary_view as boundary_view
from neurospatial.geometry._boundaries import line_of_sight_clear as line_of_sight_clear
from neurospatial.geometry._boundaries import physical_ray_hits as physical_ray_hits
from neurospatial.geometry._boundaries import physical_walls as physical_walls
from neurospatial.geometry._boundaries import ray_hits as ray_hits
from neurospatial.geometry._boundaries import search_distance as search_distance
from neurospatial.geometry._boundaries import valid_observers as valid_observers
from neurospatial.geometry._boundaries import wall_hit_bins as wall_hit_bins
from neurospatial.geometry._boundary_bins import boundary_bins as boundary_bins
from neurospatial.geometry._capabilities import RectilinearGrid as RectilinearGrid
from neurospatial.geometry._codecs import decode as decode
from neurospatial.geometry._codecs import decode_arena as decode_arena
from neurospatial.geometry._codecs import encode as encode
from neurospatial.geometry._codecs import encode_arena as encode_arena
from neurospatial.geometry._codecs import graph_to_node_link as graph_to_node_link
from neurospatial.geometry._codecs import snapshot_provider as snapshot_provider
from neurospatial.geometry._combine_geometry import BridgeKey as BridgeKey
from neurospatial.geometry._combine_geometry import arena_pieces as arena_pieces
from neurospatial.geometry._combine_geometry import cell_boxes as cell_boxes
from neurospatial.geometry._combine_geometry import combine_binnings as combine_binnings
from neurospatial.geometry._components import HexagonShapes as HexagonShapes
from neurospatial.geometry._components import Outline as Outline
from neurospatial.geometry._components import TriangleShapes as TriangleShapes
from neurospatial.geometry._composition import CombinedLocator as CombinedLocator
from neurospatial.geometry._composition import bridge_values as bridge_values
from neurospatial.geometry._composition import combined_diffusion as combined_diffusion
from neurospatial.geometry._composition import (
    composition_bin_sources as composition_bin_sources,
)
from neurospatial.geometry._composition import (
    composition_component_keys as composition_component_keys,
)
from neurospatial.geometry._composition import composition_view as composition_view
from neurospatial.geometry._context import GridIndex as GridIndex
from neurospatial.geometry._context import grid_index as grid_index
from neurospatial.geometry._coordinates import CartesianSystem as CartesianSystem
from neurospatial.geometry._coordinates import CoordinateSystem as CoordinateSystem
from neurospatial.geometry._coordinates import (
    EgocentricPolarSystem as EgocentricPolarSystem,
)
from neurospatial.geometry._coordinates import require_cartesian as require_cartesian
from neurospatial.geometry._derive import select_bins as select_bins
from neurospatial.geometry._derive import transform_binning as transform_binning
from neurospatial.geometry._diffusion import DiffusionGeometry as DiffusionGeometry
from neurospatial.geometry._diffusion import FaceGeometry as FaceGeometry
from neurospatial.geometry._explicit import binning_ranges as binning_ranges
from neurospatial.geometry._explicit import cell_extent as cell_extent
from neurospatial.geometry._explicit import component_extent as component_extent
from neurospatial.geometry._explicit import prepare_bins as prepare_bins
from neurospatial.geometry._face_builders import track_junctions as track_junctions
from neurospatial.geometry._face_normals import FaceNormals as FaceNormals
from neurospatial.geometry._face_normals import geometric_normals as geometric_normals
from neurospatial.geometry._graph import graph_from_binning as graph_from_binning
from neurospatial.geometry._grid_frame import FramedGrid as FramedGrid
from neurospatial.geometry._hex_lookup import MIN_HEX_RADIUS as MIN_HEX_RADIUS
from neurospatial.geometry._hex_lookup import (
    axial_to_offset_bin_indices as axial_to_offset_bin_indices,
)
from neurospatial.geometry._hex_lookup import (
    cartesian_to_fractional_cube as cartesian_to_fractional_cube,
)
from neurospatial.geometry._hex_lookup import (
    get_hex_grid_neighbor_deltas as get_hex_grid_neighbor_deltas,
)
from neurospatial.geometry._hex_lookup import (
    points_to_hex_bin_ind as points_to_hex_bin_ind,
)
from neurospatial.geometry._hex_lookup import (
    round_fractional_cube_to_integer_axial as round_fractional_cube_to_integer_axial,
)
from neurospatial.geometry._locators import CallIndexes as CallIndexes
from neurospatial.geometry._locators import HexLocator as HexLocator
from neurospatial.geometry._locators import Locator as Locator
from neurospatial.geometry._locators import LookupIndexes as LookupIndexes
from neurospatial.geometry._locators import MeshLocator as MeshLocator
from neurospatial.geometry._locators import NearestBinLocator as NearestBinLocator
from neurospatial.geometry._locators import RectilinearLocator as RectilinearLocator
from neurospatial.geometry._locators import SubsetLocator as SubsetLocator
from neurospatial.geometry._locators import TransformedLocator as TransformedLocator
from neurospatial.geometry._lookup import locate as locate
from neurospatial.geometry._lookup import (
    locate_outline_points as locate_outline_points,
)
from neurospatial.geometry._lookup import (
    require_position_columns as require_position_columns,
)
from neurospatial.geometry._lookup import require_track_width as require_track_width
from neurospatial.geometry._lookup import (
    wrong_position_columns as wrong_position_columns,
)
from neurospatial.geometry._native_faces import NativeFaces as NativeFaces
from neurospatial.geometry._ownership import own_arena as own_arena
from neurospatial.geometry._ownership import own_binning as own_binning
from neurospatial.geometry._polygon import cells_from_locator as cells_from_locator
from neurospatial.geometry._polygon import clip_binning as clip_binning
from neurospatial.geometry._polygon import materialize_cells as materialize_cells
from neurospatial.geometry._polygon_cells import ClipMapping as ClipMapping
from neurospatial.geometry._polygon_cells import ClippedLocator as ClippedLocator
from neurospatial.geometry._polygon_cells import PolygonCells as PolygonCells
from neurospatial.geometry._rebin_spec import GridBinningSpec as GridBinningSpec
from neurospatial.geometry._region_bins import polygon_bin_mask as polygon_bin_mask
from neurospatial.geometry._region_bins import region_bin_mask as region_bin_mask
from neurospatial.geometry._region_bins import (
    region_bin_masks as region_bin_masks,
)
from neurospatial.geometry._regular_lookup import (
    points_to_regular_grid_bin_ind as points_to_regular_grid_bin_ind,
)
from neurospatial.geometry._track import project_1d_to_2d as project_1d_to_2d
from neurospatial.geometry._validation import validate_components as validate_components
from neurospatial.geometry._wall_frame import wall_source_frame as wall_source_frame
