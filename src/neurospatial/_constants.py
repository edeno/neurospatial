"""Numerical constants and tolerances for neurospatial.

This module defines all numerical tolerances used throughout the package
for consistent behavior across geometric operations, equality checks, and
numerical stability.

All magic numbers should be imported from this module to ensure consistency
and make it easier to tune parameters globally.
"""

import numpy as np

# Geometric tolerances
# --------------------
# Used for: point-in-polygon tests, boundary detection, polygon simplification

#: Geometric tolerance for distance comparisons (meters/cm)
#: Points within this distance are considered coincident
GEOMETRIC_TOLERANCE = 1e-9

#: Relative tolerance for np.isclose() comparisons
RELATIVE_TOLERANCE = 1e-9

#: Absolute tolerance for np.isclose() comparisons
ABSOLUTE_TOLERANCE = 1e-9

#: Point tolerance for region operations (polygon vertices, boundary detection)
#: Used to determine if two points should be considered the same location
POINT_TOLERANCE = 1e-8

# Numerical stability
# -------------------
# Used for: division by zero prevention, log(0) prevention

#: Small epsilon to prevent division by zero
EPSILON = 1e-10

#: Minimum denominator for inverse distance weighting
#: Prevents division by zero when query point equals a data point
IDW_MIN_DISTANCE = 1e-8

# Angle conventions
# -----------------

#: Angle range for 2D edge angles (radians)
ANGLE_2D_RANGE = (-np.pi, np.pi)

#: Tolerance for angle wraparound comparisons
ANGLE_TOLERANCE = 1e-8

# KDTree configuration
# --------------------

#: Default leaf size for scipy/sklearn KDTree queries
#: Smaller values = faster queries, slower construction
#: Larger values = slower queries, faster construction
#: 16 is a good balance for typical spatial queries
KDTREE_LEAF_SIZE = 16

#: Leaf size for composite environment KDTrees (larger for merged data)
#: Composite environments have more bins, so we use a larger leaf size
KDTREE_COMPOSITE_LEAF_SIZE = 40

# Distance thresholds
# -------------------

#: Maximum distance multiplier for bin containment checks
#: Points further than (bin_size * BIN_CONTAINMENT_FACTOR) from bin center
#: are considered outside the bin
#: Conservative value of 0.7 works well for rectangular grid cells
BIN_CONTAINMENT_FACTOR = 0.7

#: Maximum distance multiplier for nearest neighbor queries
#: Used to detect when a point is outside all bins
NEAREST_BIN_DISTANCE_THRESHOLD = 2.0
