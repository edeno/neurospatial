"""Shared numerical-operation limits, with no package dependencies."""

# Dense kernels and near-full-rank bases warn above this bin count. The
# threshold is shared by diffusion and Gaussian smoothing; it is not a limit.
_LARGE_KERNEL_THRESHOLD = 3000
