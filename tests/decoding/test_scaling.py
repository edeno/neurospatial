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
    """decode_position_summary peaks far below a full decode_position.

    The streaming property under test is "never materializes the full
    ``(n_time, n_bins)`` posterior". An *absolute* byte threshold on the
    summary peak is flaky because the transient tracemalloc peak also counts
    per-block likelihood/scipy intermediates, which swing run-to-run inside
    the allocation-noise band. Instead we measure the peak of a *full*
    ``decode_position`` (which DOES materialize the dense posterior) on the
    SAME fixture and assert the summary peaks comfortably below it, plus a
    hard cap at the dense array's own size. Both comparands are measured here,
    so the assertion is stable whether this file runs alone or alongside the
    rest of the decoding suite.
    """
    from neurospatial.decoding import decode_position, decode_position_summary

    env = medium_2d_env  # 625 bins
    n_time = 2000
    spike_counts, encoding_models = _inputs(env, n_time=n_time)

    full_posterior_bytes = n_time * env.n_bins * 8  # float64 dense array size

    # Peak of the full decode (materializes the dense (n_time, n_bins) array).
    tracemalloc.start()
    tracemalloc.reset_peak()
    decode_position(env, spike_counts, encoding_models, dt=0.025)
    _, full_decode_peak = tracemalloc.get_traced_memory()

    # Peak of the streaming summary on the same inputs.
    tracemalloc.reset_peak()
    summ = decode_position_summary(
        env, spike_counts, encoding_models, dt=0.025, time_chunk=128
    )
    _, summary_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Sanity: it actually produced per-time reductions.
    assert summ.map_position.shape[0] == n_time

    # The streaming path must peak well below the full materialized decode
    # (clear margin), and below the dense array's own footprint. Both are
    # robust to small per-block allocator overhead.
    assert summary_peak < 0.5 * full_decode_peak, (
        f"summary_peak={summary_peak} bytes is not well below full decode "
        f"peak {full_decode_peak} bytes"
    )
    assert summary_peak < full_posterior_bytes, (
        f"summary_peak={summary_peak} bytes is not below the dense posterior "
        f"size {full_posterior_bytes} bytes"
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
