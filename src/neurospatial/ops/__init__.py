"""
Low-level operations for power users.

This module provides primitive operations for spatial analysis
that can be composed into higher-level functionality.

Submodules
----------
binning : Point-to-bin mapping, region masks
distance : Distance fields, pairwise distances
normalize : Field normalization, clamping
smoothing : Diffusion kernels, kernel application
graph : Graph convolution, neighborhood reduction
calculus : Spatial gradient, divergence
transforms : Affine transforms, calibration
alignment : Probability mapping, similarity transforms
egocentric : Heading computation, allocentric/egocentric transforms
visibility : Viewshed, gaze, line-of-sight
basis : GLM spatial basis functions
"""

# Binning operations
from neurospatial.ops.binning import (
    TieBreakStrategy,
    clear_kdtree_cache,
    map_points_to_bins,
    regions_to_mask,
    resample_field,
)

__all__ = [
    # binning
    "TieBreakStrategy",
    "clear_kdtree_cache",
    "map_points_to_bins",
    "regions_to_mask",
    "resample_field",
]
