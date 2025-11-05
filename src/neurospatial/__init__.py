import logging

from neurospatial.alignment import (
    get_2d_rotation_matrix,
    map_probabilities_to_nearest_target_bin,
)
from neurospatial.distance import distance_field, pairwise_distances
from neurospatial.environment import Environment
from neurospatial.field_ops import (
    clamp,
    combine_fields,
    divergence,
    normalize_field,
)
from neurospatial.kernels import compute_diffusion_kernels
from neurospatial.layout.factories import (
    get_layout_parameters,
    list_available_layouts,
)
from neurospatial.layout.validation import validate_environment
from neurospatial.spatial import TieBreakStrategy, map_points_to_bins
from neurospatial.transforms import (
    apply_transform_to_environment,
    estimate_transform,
)

# Add NullHandler to prevent "No handler found" warnings if user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Environment",
    "TieBreakStrategy",
    "apply_transform_to_environment",
    "clamp",
    "combine_fields",
    "compute_diffusion_kernels",
    "distance_field",
    "divergence",
    "estimate_transform",
    "get_2d_rotation_matrix",
    "get_layout_parameters",
    "list_available_layouts",
    "map_points_to_bins",
    "map_probabilities_to_nearest_target_bin",
    "normalize_field",
    "pairwise_distances",
    "validate_environment",
]
