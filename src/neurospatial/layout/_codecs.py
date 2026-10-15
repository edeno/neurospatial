"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._codecs import (
    _CODECS as _CODECS,
)
from neurospatial.geometry._codecs import (
    _LOCATOR_TYPES as _LOCATOR_TYPES,
)
from neurospatial.geometry._codecs import (
    _RESERVED_GRAPH_ATTRIBUTES as _RESERVED_GRAPH_ATTRIBUTES,
)
from neurospatial.geometry._codecs import (
    Codec as Codec,
)
from neurospatial.geometry._codecs import (
    Payload as Payload,
)
from neurospatial.geometry._codecs import (
    _active_map as _active_map,
)
from neurospatial.geometry._codecs import (
    _array as _array,
)
from neurospatial.geometry._codecs import (
    _arrays as _arrays,
)
from neurospatial.geometry._codecs import (
    _decode_binning as _decode_binning,
)
from neurospatial.geometry._codecs import (
    _decode_bins as _decode_bins,
)
from neurospatial.geometry._codecs import (
    _decode_clip_mapping as _decode_clip_mapping,
)
from neurospatial.geometry._codecs import (
    _decode_clipped_locator as _decode_clipped_locator,
)
from neurospatial.geometry._codecs import (
    _decode_combined as _decode_combined,
)
from neurospatial.geometry._codecs import (
    _decode_face_normals as _decode_face_normals,
)
from neurospatial.geometry._codecs import (
    _decode_faces as _decode_faces,
)
from neurospatial.geometry._codecs import (
    _decode_grid as _decode_grid,
)
from neurospatial.geometry._codecs import (
    _decode_grid_frame as _decode_grid_frame,
)
from neurospatial.geometry._codecs import (
    _decode_grid_spec as _decode_grid_spec,
)
from neurospatial.geometry._codecs import (
    _decode_hex_locator as _decode_hex_locator,
)
from neurospatial.geometry._codecs import (
    _decode_hex_shapes as _decode_hex_shapes,
)
from neurospatial.geometry._codecs import (
    _decode_mesh_locator as _decode_mesh_locator,
)
from neurospatial.geometry._codecs import (
    _decode_missing as _decode_missing,
)
from neurospatial.geometry._codecs import (
    _decode_nearest_locator as _decode_nearest_locator,
)
from neurospatial.geometry._codecs import (
    _decode_outline as _decode_outline,
)
from neurospatial.geometry._codecs import (
    _decode_polygon_cells as _decode_polygon_cells,
)
from neurospatial.geometry._codecs import (
    _decode_rectilinear_locator as _decode_rectilinear_locator,
)
from neurospatial.geometry._codecs import (
    _decode_subset_locator as _decode_subset_locator,
)
from neurospatial.geometry._codecs import (
    _decode_track as _decode_track,
)
from neurospatial.geometry._codecs import (
    _decode_transformed_locator as _decode_transformed_locator,
)
from neurospatial.geometry._codecs import (
    _decode_triangle_shapes as _decode_triangle_shapes,
)
from neurospatial.geometry._codecs import (
    _encode_binning as _encode_binning,
)
from neurospatial.geometry._codecs import (
    _encode_bins as _encode_bins,
)
from neurospatial.geometry._codecs import (
    _encode_clip_mapping as _encode_clip_mapping,
)
from neurospatial.geometry._codecs import (
    _encode_clipped_locator as _encode_clipped_locator,
)
from neurospatial.geometry._codecs import (
    _encode_combined as _encode_combined,
)
from neurospatial.geometry._codecs import (
    _encode_face_normals as _encode_face_normals,
)
from neurospatial.geometry._codecs import (
    _encode_faces as _encode_faces,
)
from neurospatial.geometry._codecs import (
    _encode_grid as _encode_grid,
)
from neurospatial.geometry._codecs import (
    _encode_grid_frame as _encode_grid_frame,
)
from neurospatial.geometry._codecs import (
    _encode_grid_spec as _encode_grid_spec,
)
from neurospatial.geometry._codecs import (
    _encode_hex_locator as _encode_hex_locator,
)
from neurospatial.geometry._codecs import (
    _encode_hex_shapes as _encode_hex_shapes,
)
from neurospatial.geometry._codecs import (
    _encode_mesh_locator as _encode_mesh_locator,
)
from neurospatial.geometry._codecs import (
    _encode_missing as _encode_missing,
)
from neurospatial.geometry._codecs import (
    _encode_nearest_locator as _encode_nearest_locator,
)
from neurospatial.geometry._codecs import (
    _encode_outline as _encode_outline,
)
from neurospatial.geometry._codecs import (
    _encode_polygon_cells as _encode_polygon_cells,
)
from neurospatial.geometry._codecs import (
    _encode_rectilinear_locator as _encode_rectilinear_locator,
)
from neurospatial.geometry._codecs import (
    _encode_subset_locator as _encode_subset_locator,
)
from neurospatial.geometry._codecs import (
    _encode_track as _encode_track,
)
from neurospatial.geometry._codecs import (
    _encode_transformed_locator as _encode_transformed_locator,
)
from neurospatial.geometry._codecs import (
    _encode_triangle_shapes as _encode_triangle_shapes,
)
from neurospatial.geometry._codecs import (
    _finite as _finite,
)
from neurospatial.geometry._codecs import (
    _flag as _flag,
)
from neurospatial.geometry._codecs import (
    _grid_edges as _grid_edges,
)
from neurospatial.geometry._codecs import (
    _json_safe as _json_safe,
)
from neurospatial.geometry._codecs import (
    _mesh_arrays as _mesh_arrays,
)
from neurospatial.geometry._codecs import (
    _node_id as _node_id,
)
from neurospatial.geometry._codecs import (
    _nonnegative_integer as _nonnegative_integer,
)
from neurospatial.geometry._codecs import (
    _shape as _shape,
)
from neurospatial.geometry._codecs import (
    decode as decode,
)
from neurospatial.geometry._codecs import (
    encode as encode,
)
from neurospatial.geometry._codecs import (
    graph_from_node_link as graph_from_node_link,
)
from neurospatial.geometry._codecs import (
    graph_to_node_link as graph_to_node_link,
)
