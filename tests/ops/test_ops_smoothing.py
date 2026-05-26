"""End-to-end check that ``Environment.smooth`` / ``Environment.compute_kernel``
still work through the ``neurospatial.ops.smoothing`` module after the
ops reorganization.

The functional / numerical coverage for ``compute_diffusion_kernels``
and ``apply_kernel`` lives in ``tests/ops/test_kernels.py`` (richer
edge cases: density vs. transition mode, mass conservation, impulse
spreading, bandwidth limits, disconnected components, large-graph
warning, sparse-mass equivalence). The earlier version of this file
re-tested a four-test subset of those properties plus a stack of
``__all__`` / private-helper / re-export audits — all removed.
"""

import numpy as np

from neurospatial import Environment


def test_environment_compute_kernel():
    """``Environment.compute_kernel`` returns a square ``(n_bins, n_bins)`` kernel."""
    data = np.array([[i, j] for i in range(11) for j in range(11)])
    env = Environment.from_samples(data, bin_size=2.0)

    kernel = env.compute_kernel(bandwidth=1.0, mode="transition")

    assert kernel.shape == (env.n_bins, env.n_bins)
    assert kernel.dtype == np.float64


def test_environment_smooth_preserves_mass():
    """``Environment.smooth`` in transition mode preserves total mass."""
    data = np.array([[i, j] for i in range(11) for j in range(11)])
    env = Environment.from_samples(data, bin_size=2.0)

    field = np.ones(env.n_bins) / env.n_bins
    smoothed = env.smooth(field, bandwidth=1.0, mode="transition")

    assert smoothed.shape == (env.n_bins,)
    np.testing.assert_allclose(smoothed.sum(), field.sum(), atol=1e-10)
