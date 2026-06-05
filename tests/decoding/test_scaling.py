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


def test_session_summary_streams_binning_under_full_matrix(small_2d_env):
    """decode_session_summary peaks far below materializing the count matrix.

    This is the v0.6 DoD evidence for R5: ``decode_session_summary`` streams the
    time-binning, so the dense ``(n_time, n_neurons)`` count matrix is NEVER
    materialized. We measure, in-process and back-to-back:

      * ``reference_peak`` -- a path that FIRST materializes the full count
        matrix (``bin_spikes_in_time`` over the whole session) and THEN streams
        the decode (``decode_position_summary``). This is exactly what
        ``decode_session_summary`` did before R5.
      * ``streamed_peak`` -- the R5 ``decode_session_summary``, which bins
        block-by-block.

    The streamed path's peak must sit well below both the dense count matrix's
    own size and the reference peak, with a clear margin so the assertion is
    robust to allocator noise (same pattern as
    ``test_summary_never_allocates_full_posterior``).

    Extrapolation to the DoD target (1 hr / 25 ms / 5000 bins / <500 MB): the
    streamed peak is ``O(time_chunk * max(n_neurons, n_bins))`` plus the
    ``(n_neurons, n_bins)`` encoding model -- i.e. INDEPENDENT of n_time. For
    1000 neurons x 5000 bins the encoding model is ~40 MB (float64) and each
    block (time_chunk=1024) holds ~1024 x max(1000, 5000) x 8 B ~= 41 MB, so the
    summary peak stays well under 500 MB regardless of the 144k time bins in a
    1 hr session. The full ``(144000, 1000)`` count matrix alone would be
    ~1.15 GB, which this streaming path never allocates.
    """
    from neurospatial.decoding import (
        bin_spikes_in_time,
        decode_position_summary,
        decode_session_summary,
    )

    # Use a small-bin env so the per-block posterior (time_chunk x n_bins) stays
    # tiny and the dense (n_time, n_neurons) count matrix is the dominant
    # comparand. This isolates the property under test: the streamed binning
    # never allocates that count matrix.
    env = small_2d_env  # ~36 bins
    rng = np.random.default_rng(0)
    n_neurons = 400
    # Long session: n_time is large so the dense count matrix is sizable (the
    # quantity the streamed path must NOT allocate), but n_bins is small so the
    # test runs fast.
    dt = 0.025
    duration = 500.0  # 500 s / 25 ms = 20000 time bins
    times = np.arange(0.0, duration, dt / 2.0)  # 2x oversampled trajectory
    lo = env.bin_centers.min(axis=0)
    hi = env.bin_centers.max(axis=0)
    positions = rng.uniform(lo, hi, (len(times), env.n_dims))
    # Precompute small encoding models so both paths skip the (slow) KDE encode
    # and we isolate the binning+decode memory behavior.
    encoding_models = rng.uniform(0.5, 12.0, (n_neurons, env.n_bins))
    # ~3 Hz spike trains across the whole window.
    spike_times = [
        np.sort(rng.uniform(0.0, duration, int(3.0 * duration)))
        for _ in range(n_neurons)
    ]

    n_time = int(np.floor((times.max() - times.min()) / dt + 1e-9))
    full_counts_bytes = n_time * n_neurons * 8  # dense int64 count matrix size

    # --- Reference: materialize the FULL count matrix, then stream-decode. ---
    tracemalloc.start()
    tracemalloc.reset_peak()
    counts, centers = bin_spikes_in_time(
        spike_times, dt, t_start=times.min(), t_stop=times.max()
    )
    ref = decode_position_summary(
        env, counts, encoding_models, dt, times=centers, time_chunk=512
    )
    _, reference_peak = tracemalloc.get_traced_memory()
    del counts, centers, ref

    # --- Streamed: decode_session_summary bins block-by-block. ---
    tracemalloc.reset_peak()
    summ = decode_session_summary(
        env,
        spike_times,
        times,
        positions,
        dt=dt,
        encoding_models=encoding_models,
        time_chunk=512,
        warn_on_drop=False,
    )
    _, streamed_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Sanity: it produced the full per-time reductions.
    assert summ.map_position.shape[0] == n_time

    # The streamed path must peak well below the dense count matrix's own size
    # (it never allocates it) ...
    assert streamed_peak < 0.5 * full_counts_bytes, (
        f"streamed_peak={streamed_peak} bytes is not well below the dense "
        f"count-matrix size {full_counts_bytes} bytes"
    )
    # ... and clearly below the materialize-then-stream reference peak.
    assert streamed_peak < 0.5 * reference_peak, (
        f"streamed_peak={streamed_peak} bytes is not well below the "
        f"materialize-then-stream reference peak {reference_peak} bytes"
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

    # Chunked path must peak meaningfully below the unchunked path (it avoids
    # the full-size ll_shifted temporary). Use a margin for tracemalloc-noise
    # robustness, consistent with the sibling test's reasoning.
    assert peak_chunked < 0.9 * peak_unchunked
