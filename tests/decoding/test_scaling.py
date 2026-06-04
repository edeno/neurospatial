"""Scale-safety asserts for memory-safe summary decoding (Task 2.1).

These tests use ``tracemalloc`` to assert that ``decode_position_summary``
never allocates a full ``(n_time, n_bins)`` posterior, and that
``decode_position(..., time_chunk=...)`` reduces the transient peak relative to
the unchunked path.
"""

from __future__ import annotations

import tracemalloc

import numpy as np


def _inputs(env, *, n_time, n_neurons=10, seed=0):
    rng = np.random.default_rng(seed)
    spike_counts = rng.poisson(1.5, (n_time, n_neurons)).astype(np.int64)
    encoding_models = rng.uniform(0.5, 12.0, (n_neurons, env.n_bins))
    return spike_counts, encoding_models


def test_summary_never_allocates_full_posterior(medium_2d_env):
    """decode_position_summary peak memory << full (n_time, n_bins) posterior."""
    from neurospatial.decoding import decode_position_summary

    env = medium_2d_env  # 625 bins
    n_time = 2000
    spike_counts, encoding_models = _inputs(env, n_time=n_time)

    full_posterior_bytes = n_time * env.n_bins * 8  # float64

    tracemalloc.start()
    tracemalloc.reset_peak()
    summ = decode_position_summary(
        env, spike_counts, encoding_models, dt=0.025, time_chunk=128
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Sanity: it actually produced per-time reductions.
    assert summ.map_position.shape[0] == n_time

    # Streaming must stay well below a full dense posterior. Use a clear
    # margin (< 0.5x) so the assertion is robust to small allocator overhead.
    assert peak < 0.5 * full_posterior_bytes, (
        f"peak={peak} bytes is not well below full posterior "
        f"{full_posterior_bytes} bytes"
    )


def test_time_chunk_reduces_transient_peak(medium_2d_env):
    """decode_position(time_chunk=k) peaks below the unchunked decode."""
    from neurospatial.decoding import decode_position

    env = medium_2d_env
    n_time = 2000
    spike_counts, encoding_models = _inputs(env, n_time=n_time)

    tracemalloc.start()
    tracemalloc.reset_peak()
    decode_position(env, spike_counts, encoding_models, dt=0.025)
    _, peak_unchunked = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    decode_position(env, spike_counts, encoding_models, dt=0.025, time_chunk=128)
    _, peak_chunked = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Chunked path must not peak higher than the unchunked path; it should be
    # meaningfully lower since it avoids a full-size log-likelihood .copy().
    assert peak_chunked < peak_unchunked
