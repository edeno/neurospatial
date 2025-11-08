import logging

from neurospatial.alignment import (
    get_2d_rotation_matrix,
    map_probabilities_to_nearest_target_bin,
)
from neurospatial.composite import CompositeEnvironment
from neurospatial.differential import divergence, gradient
from neurospatial.distance import distance_field, neighbors_within, pairwise_distances
from neurospatial.environment import Environment
from neurospatial.field_ops import (
    clamp,
    combine_fields,
    kl_divergence,
    normalize_field,
)
from neurospatial.kernels import apply_kernel, compute_diffusion_kernels
from neurospatial.layout.factories import (
    LayoutType,
    get_layout_parameters,
    list_available_layouts,
)
from neurospatial.layout.validation import validate_environment
from neurospatial.primitives import convolve, neighbor_reduce
from neurospatial.reward import goal_reward_field, region_reward_field
from neurospatial.spatial import (
    TieBreakStrategy,
    map_points_to_bins,
    regions_to_mask,
    resample_field,
)
from neurospatial.spike_field import compute_place_field, spikes_to_field
from neurospatial.transforms import (
    apply_transform_to_environment,
    estimate_transform,
)

# Add NullHandler to prevent "No handler found" warnings if user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CompositeEnvironment",
    "Environment",
    "LayoutType",
    "TieBreakStrategy",
    "apply_kernel",
    "apply_transform_to_environment",
    "clamp",
    "combine_fields",
    "compute_diffusion_kernels",
    "compute_place_field",
    "convolve",
    "distance_field",
    "divergence",
    "estimate_transform",
    "get_2d_rotation_matrix",
    "get_layout_parameters",
    "goal_reward_field",
    "gradient",
    "kl_divergence",
    "list_available_layouts",
    "map_points_to_bins",
    "map_probabilities_to_nearest_target_bin",
    "neighbor_reduce",
    "neighbors_within",
    "normalize_field",
    "pairwise_distances",
    "region_reward_field",
    "regions_to_mask",
    "resample_field",
    "spikes_to_field",
    "validate_environment",
]
