"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._arena import (
    Arena as Arena,
)
from neurospatial.geometry._arena import (
    EgocentricSurround as EgocentricSurround,
)
from neurospatial.geometry._arena import (
    MaskArena as MaskArena,
)
from neurospatial.geometry._arena import (
    OpenArena as OpenArena,
)
from neurospatial.geometry._arena import (
    TrackArena as TrackArena,
)
from neurospatial.geometry._arena import (
    UnknownArena as UnknownArena,
)
from neurospatial.geometry._arena import (
    __all__ as __all__,
)
from neurospatial.geometry._arena import (
    _affine as _affine,
)
from neurospatial.geometry._arena import (
    _ArrayState as _ArrayState,
)
from neurospatial.geometry._arena import (
    _boolean as _boolean,
)
from neurospatial.geometry._arena import (
    _check_arena_nesting as _check_arena_nesting,
)
from neurospatial.geometry._arena import (
    _CombinedArena as _CombinedArena,
)
from neurospatial.geometry._arena import (
    _contains_with_error as _contains_with_error,
)
from neurospatial.geometry._arena import (
    _CoordinateKind as _CoordinateKind,
)
from neurospatial.geometry._arena import (
    _count as _count,
)
from neurospatial.geometry._arena import (
    _error as _error,
)
from neurospatial.geometry._arena import (
    _feature_cap as _feature_cap,
)
from neurospatial.geometry._arena import (
    _mapped as _mapped,
)
from neurospatial.geometry._arena import (
    _mapped_geometry as _mapped_geometry,
)
from neurospatial.geometry._arena import (
    _polygon as _polygon,
)
from neurospatial.geometry._arena import (
    _query as _query,
)
from neurospatial.geometry._arena import (
    _read_wkb as _read_wkb,
)
from neurospatial.geometry._arena import (
    _real_array as _real_array,
)
from neurospatial.geometry._arena import (
    _segment_distance as _segment_distance,
)
from neurospatial.geometry._arena import (
    _track_distances as _track_distances,
)
from neurospatial.geometry._arena import (
    _TransformedArena as _TransformedArena,
)
from neurospatial.geometry._arena import (
    _unknown as _unknown,
)
from neurospatial.geometry._arena import (
    _wkb as _wkb,
)
from neurospatial.geometry._arena import (
    decode_arena as decode_arena,
)
from neurospatial.geometry._arena import (
    encode_arena as encode_arena,
)
