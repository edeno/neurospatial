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

# Re-exports will be added as modules are moved here
