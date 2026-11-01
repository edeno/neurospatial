"""Numerical constants and tolerances shared across neurospatial modules.

A value lives here when more than one module needs it or when it is a tuning
choice worth naming in one place; values specific to one function stay with
that function.
"""

# Equality tolerances
# -------------------

#: Relative tolerance for np.isclose() comparisons
RELATIVE_TOLERANCE = 1e-9

#: Absolute tolerance for np.isclose() comparisons, in the compared values' units
ABSOLUTE_TOLERANCE = 1e-9

# Numerical stability
# -------------------

#: Minimum denominator for inverse distance weighting, in environment units.
#: Prevents division by zero when query point equals a data point
IDW_MIN_DISTANCE = 1e-8

# KDTree configuration
# --------------------

#: Default leaf size for scipy/sklearn KDTree queries
#: Smaller values = faster queries, slower construction
#: Larger values = slower queries, faster construction
#: 16 is a good balance for typical spatial queries
KDTREE_LEAF_SIZE = 16

#: Dense smoothing kernels one environment keeps per cache; the least recently
#: used is dropped first. Each is an (n_bins, n_bins) float64 matrix, so without
#: a bound a bandwidth sweep would hold one per bandwidth for the environment's
#: lifetime.
MAX_CACHED_KERNELS = 8
