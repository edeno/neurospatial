"""Tests for speed filtering in the batch/single encode path (Task 2.6).

A firing rate map is spike_counts (numerator) / occupancy (denominator) per
bin. A ``min_speed`` filter MUST apply ONE shared per-interval speed gate to
BOTH the spike numerator and the occupancy denominator, or rates are silently
biased. These tests exercise:

1. Default (no ``min_speed``) is byte-for-byte unchanged (single + batch, all
   three smoothing methods).
2. The SAME interval set is dropped from spikes and occupancy (alignment), by
   comparing against a hand-masked reference.
3. Explicit ``speed`` equals the auto-derived forward-difference speed.
4. Wrong-length ``speed`` raises ``ValueError`` naming ``speed``.
5. ``decode_session`` forwards ``min_speed`` to the encoder.
6. A regression that would FAIL under a "filter occupancy only" bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._binning import (
    bin_spike_train,
    compute_occupancy,
    resolve_speed,
)
from neurospatial.encoding.spatial import compute_spatial_rate, compute_spatial_rates

SMOOTHING_METHODS = ["diffusion_kde", "gaussian_kde", "binned"]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def env_1d() -> Environment:
    """A simple 1D environment spanning 0-100."""
    positions = np.linspace(0, 100, 101).reshape(-1, 1)
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def two_speed_trajectory() -> dict:
    """Trajectory with a separable slow segment then a fast segment.

    Sampled at 10 Hz (dt = 0.1 s) so every interval is under ``env.occupancy``'s
    default ``max_gap`` of 0.5 s (otherwise occupancy would be all-zero).

    - Slow segment: samples 0..50, animal creeps up from x=10 with 0.1-unit
      steps → speed 1.0 unit/s.
    - Fast segment: samples 50..70, animal sweeps with 3.0-unit steps from
      ~15 up to ~75 → speed 30.0 unit/s.

    dt = 0.1 everywhere, so the forward-difference speed equals
    (per-step displacement) / 0.1.
    """
    n_slow = 51
    n_fast = 20
    dt = 0.1

    # Slow: 0.1-unit steps → speed 1.0 unit/s. x: 10.0 .. 15.0
    slow_x = 10.0 + np.arange(n_slow) * 0.1
    # Fast: 3.0-unit steps → speed 30.0 unit/s, staying inside [0, 100].
    slow_end = slow_x[-1]
    fast_x = slow_end + np.arange(1, n_fast + 1) * 3.0  # ends near 75

    x = np.concatenate([slow_x, fast_x])
    assert x.min() >= 0 and x.max() <= 100  # stay inside the environment
    positions = x.reshape(-1, 1)
    n = len(x)
    times = np.arange(n, dtype=np.float64) * dt
    return {"times": times, "positions": positions, "n_slow": n_slow}


# ==============================================================================
# 1. Default unchanged (byte-for-byte)
# ==============================================================================


def _assert_array_equal_nan(a: np.ndarray, b: np.ndarray) -> None:
    """Assert equality treating NaN as equal in matching positions."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    np.testing.assert_array_equal(nan_a, nan_b)
    np.testing.assert_array_equal(a[~nan_a], b[~nan_b])


@pytest.mark.parametrize("method", SMOOTHING_METHODS)
def test_single_default_unchanged(method: str, env_1d, two_speed_trajectory) -> None:
    """compute_spatial_rate with no min_speed == prior behavior, byte-for-byte."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = np.array([0.5, 2.5, 4.0, 5.5, 6.5])

    baseline = compute_spatial_rate(
        env_1d, spike_times, times, positions, smoothing_method=method
    )
    # Explicitly default min_speed=None should match (and also the new param
    # path with speed=None, min_speed=None).
    with_param = compute_spatial_rate(
        env_1d,
        spike_times,
        times,
        positions,
        smoothing_method=method,
        speed=None,
        min_speed=None,
    )
    _assert_array_equal_nan(baseline.firing_rate, with_param.firing_rate)
    _assert_array_equal_nan(baseline.occupancy, with_param.occupancy)


@pytest.mark.parametrize("method", SMOOTHING_METHODS)
def test_batch_default_unchanged(method: str, env_1d, two_speed_trajectory) -> None:
    """compute_spatial_rates with no min_speed == prior behavior, byte-for-byte."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = [
        np.array([0.5, 2.5, 5.5]),
        np.array([1.0, 4.0, 6.5]),
        np.array([3.0, 6.0]),
    ]

    baseline = compute_spatial_rates(
        env_1d, spike_times, times, positions, smoothing_method=method
    )
    with_param = compute_spatial_rates(
        env_1d,
        spike_times,
        times,
        positions,
        smoothing_method=method,
        speed=None,
        min_speed=None,
    )
    _assert_array_equal_nan(baseline.firing_rates, with_param.firing_rates)
    _assert_array_equal_nan(baseline.occupancy, with_param.occupancy)


# ==============================================================================
# 2. Identical sample set (the core correctness / alignment test)
# ==============================================================================


def test_alignment_spikes_and_occupancy_same_intervals(
    env_1d, two_speed_trajectory
) -> None:
    """min_speed drops EXACTLY the same intervals from spikes and occupancy.

    Compares the public-path spike_counts/occupancy against a hand-masked
    reference that keeps only the high-speed intervals' spikes and passes the
    same speed/min_speed to env.occupancy.
    """
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]

    # Spikes in both segments (slow: t<5.0, fast: t in (5.0, 7.0)).
    spike_times = np.array([0.5, 2.0, 4.5, 5.5, 6.2, 6.8])

    # Threshold between the slow (1.0) and fast (30.0) speeds.
    min_speed = 10.0

    # --- Public path ---
    resolved = resolve_speed(times, positions, None, min_speed)
    counts_public = bin_spike_train(
        env_1d, spike_times, times, positions, speed=resolved, min_speed=min_speed
    )
    occ_public = compute_occupancy(
        env_1d, times, positions, speed=resolved, min_speed=min_speed
    )

    # --- Hand-masked reference ---
    # Occupancy: pass speed/min_speed straight to env.occupancy.
    occ_ref = env_1d.occupancy(
        times, positions, speed=resolved, min_speed=min_speed, return_seconds=True
    )
    # Spikes: keep only spikes whose interval speed >= min_speed, then bin them
    # over the FULL trajectory (interpolated positions unchanged).
    interval = np.searchsorted(times, spike_times, side="right") - 1
    interval = np.clip(interval, 0, len(times) - 1)
    keep = resolved[interval] >= min_speed
    counts_ref = bin_spike_train(env_1d, spike_times[keep], times, positions)

    np.testing.assert_array_equal(counts_public, counts_ref)
    np.testing.assert_array_equal(occ_public, occ_ref)

    # Sanity: the slow spikes (t=0.5,2.0,4.5) were dropped; the fast spikes
    # (t=5.5,6.2,6.8) retained → 3 spikes counted.
    assert counts_public.sum() == 3
    # Occupancy from slow intervals (~50s) excluded; only fast intervals remain.
    full_occ = compute_occupancy(env_1d, times, positions)
    assert occ_public.sum() < full_occ.sum()


# ==============================================================================
# 3. Explicit speed == auto-derived
# ==============================================================================


def test_explicit_speed_equals_auto_derived(env_1d, two_speed_trajectory) -> None:
    """Passing the forward-difference speed equals auto-derivation."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = [np.array([0.5, 5.5, 6.8]), np.array([2.0, 6.2])]
    min_speed = 10.0

    # Compute forward-difference speed exactly as resolve_speed documents.
    n = len(times)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(times)
    interval_speed = step / dt
    explicit_speed = np.empty(n)
    explicit_speed[:-1] = interval_speed
    explicit_speed[-1] = interval_speed[-1]

    auto = compute_spatial_rates(
        env_1d, spike_times, times, positions, min_speed=min_speed
    )
    explicit = compute_spatial_rates(
        env_1d,
        spike_times,
        times,
        positions,
        speed=explicit_speed,
        min_speed=min_speed,
    )
    _assert_array_equal_nan(auto.firing_rates, explicit.firing_rates)
    _assert_array_equal_nan(auto.occupancy, explicit.occupancy)


def test_resolve_speed_forward_difference_last_sample(
    env_1d, two_speed_trajectory
) -> None:
    """resolve_speed uses forward difference and repeats the last value."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    resolved = resolve_speed(times, positions, None, min_speed=1.0)
    assert resolved is not None
    # speed == per-step displacement / dt.
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(times)
    np.testing.assert_allclose(resolved[:-1], step / dt)
    # Last sample repeats the prior interval speed.
    assert resolved[-1] == resolved[-2]


def test_resolve_speed_none_when_min_speed_none(env_1d, two_speed_trajectory) -> None:
    """resolve_speed returns None (no filtering) when min_speed is None."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    assert resolve_speed(times, positions, None, None) is None


# ==============================================================================
# 4. Length validation
# ==============================================================================


def test_wrong_length_speed_raises(env_1d, two_speed_trajectory) -> None:
    """A wrong-length speed array raises ValueError naming `speed`."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    bad_speed = np.ones(len(times) - 3)

    with pytest.raises(ValueError, match="speed"):
        resolve_speed(times, positions, bad_speed, min_speed=1.0)

    with pytest.raises(ValueError, match="speed"):
        compute_spatial_rate(
            env_1d,
            np.array([5.0]),
            times,
            positions,
            speed=bad_speed,
            min_speed=1.0,
        )

    with pytest.raises(ValueError, match="speed"):
        compute_spatial_rates(
            env_1d,
            [np.array([5.0])],
            times,
            positions,
            speed=bad_speed,
            min_speed=1.0,
        )


# ==============================================================================
# 5. decode_session forwarding
# ==============================================================================


def test_decode_session_forwards_min_speed(env_1d, two_speed_trajectory) -> None:
    """decode_session(min_speed=...) uses speed-filtered encoding."""
    from neurospatial.decoding import decode_session

    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    # One neuron firing ONLY during the slow segment (t < 5.0).
    spike_times = [np.array([0.5, 1.5, 2.5, 3.5, 4.5])]

    result_unfiltered = decode_session(
        env_1d, spike_times, times, positions, dt=0.5, warn_on_drop=False
    )
    result_filtered = decode_session(
        env_1d,
        spike_times,
        times,
        positions,
        dt=0.5,
        min_speed=10.0,
        warn_on_drop=False,
    )
    # Filtering removes the slow-segment spikes from the encoding model, so the
    # encoding model (and hence the posterior) differs.
    assert not np.allclose(
        result_unfiltered.posterior, result_filtered.posterior, equal_nan=True
    )


def test_decode_session_default_unchanged(env_1d, two_speed_trajectory) -> None:
    """decode_session with no min_speed is unchanged."""
    from neurospatial.decoding import decode_session

    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = [np.array([0.5, 5.5, 6.8])]

    baseline = decode_session(
        env_1d, spike_times, times, positions, dt=0.5, warn_on_drop=False
    )
    with_param = decode_session(
        env_1d,
        spike_times,
        times,
        positions,
        dt=0.5,
        min_speed=None,
        warn_on_drop=False,
    )
    np.testing.assert_array_equal(baseline.posterior, with_param.posterior)


# ==============================================================================
# 6. Numerator/denominator alignment regression
# ==============================================================================


def test_filter_occupancy_only_bug_regression(env_1d, two_speed_trajectory) -> None:
    """Rate stays correct: spikes are removed too, not just occupancy.

    Construct a fixture where the animal lingers in the slow segment (large
    occupancy at the slow bin) and fires there. Under a buggy "filter occupancy
    only" implementation, removing the slow occupancy while KEEPING the slow
    spikes would inflate the rate at the slow bin (spikes / small occupancy).
    With the correct shared gate, the slow spikes are removed too, so the slow
    bin has zero spikes counted → zero raw rate there.
    """
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    # All spikes in the slow segment (t < 5.0), all near the same slow bin.
    spike_times = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    min_speed = 10.0

    resolved = resolve_speed(times, positions, None, min_speed)
    counts = bin_spike_train(
        env_1d, spike_times, times, positions, speed=resolved, min_speed=min_speed
    )
    occ = compute_occupancy(
        env_1d, times, positions, speed=resolved, min_speed=min_speed
    )

    # Correct behavior: ALL slow spikes removed (numerator zero everywhere).
    assert counts.sum() == 0

    # The slow bins (around x=10..35) have zero occupancy after filtering, so
    # the raw rate there is 0/0 -> NaN/0, never an inflated finite rate.
    raw_rate = np.divide(counts, occ, out=np.full_like(counts, np.nan), where=occ > 0)
    # No finite positive rate anywhere (no spikes survived).
    assert not np.any(raw_rate > 0)


# ==============================================================================
# 6b. Batch n_jobs parity under an active min_speed filter
# ==============================================================================


def test_batch_njobs_parity_under_min_speed(env_1d, two_speed_trajectory) -> None:
    """n_jobs=2 == n_jobs=1 (NaN-aware) with an active min_speed filter.

    Pins the contract that the once-resolved speed array is threaded into the
    joblib workers, so parallel and serial batch paths produce identical
    firing_rates AND occupancy when min_speed gates a separable slow/fast
    trajectory with spikes in both segments.
    """
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    # Spikes in both the slow (t < 5.0) and fast (t > 5.0) segments, per neuron.
    spike_times = [
        np.array([0.5, 2.0, 5.5, 6.5]),
        np.array([1.0, 4.5, 5.8, 6.8]),
        np.array([3.0, 6.2]),
    ]
    min_speed = 10.0  # between slow (1.0) and fast (30.0) speeds.

    serial = compute_spatial_rates(
        env_1d, spike_times, times, positions, min_speed=min_speed, n_jobs=1
    )
    parallel = compute_spatial_rates(
        env_1d, spike_times, times, positions, min_speed=min_speed, n_jobs=2
    )

    _assert_array_equal_nan(serial.firing_rates, parallel.firing_rates)
    _assert_array_equal_nan(serial.occupancy, parallel.occupancy)


# ==============================================================================
# 7. speed without min_speed raises (symmetry with env.occupancy)
# ==============================================================================


def test_speed_without_min_speed_raises_resolve(env_1d, two_speed_trajectory) -> None:
    """resolve_speed raises when speed is passed without min_speed."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    speed = np.ones(len(times))
    with pytest.raises(ValueError, match="min_speed"):
        resolve_speed(times, positions, speed, None)


def test_speed_without_min_speed_raises_single(env_1d, two_speed_trajectory) -> None:
    """compute_spatial_rate(speed=...) without min_speed raises ValueError."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    speed = np.ones(len(times))
    with pytest.raises(ValueError, match="speed"):
        compute_spatial_rate(
            env_1d,
            np.array([5.0]),
            times,
            positions,
            speed=speed,
        )


def test_speed_without_min_speed_raises_batch(env_1d, two_speed_trajectory) -> None:
    """compute_spatial_rates(speed=...) without min_speed raises ValueError."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    speed = np.ones(len(times))
    with pytest.raises(ValueError, match="speed"):
        compute_spatial_rates(
            env_1d,
            [np.array([5.0])],
            times,
            positions,
            speed=speed,
        )


# ==============================================================================
# 8. min_speed excluding ALL intervals warns once (empty rate map)
# ==============================================================================


def test_all_excluded_warns_single(recwarn, env_1d, two_speed_trajectory) -> None:
    """A min_speed above the max speed warns exactly once about an empty map."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = np.array([0.5, 5.5, 6.8])
    # Max speed is 30 unit/s; this excludes every interval.
    compute_spatial_rate(env_1d, spike_times, times, positions, min_speed=1e6)
    empty_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning) and "min_speed" in str(w.message)
    ]
    assert len(empty_warnings) == 1


def test_all_excluded_warns_batch(env_1d, two_speed_trajectory) -> None:
    """The all-excluded warning fires for the batch path too."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = [np.array([0.5, 5.5]), np.array([2.0, 6.2])]
    with pytest.warns(UserWarning, match="min_speed"):
        compute_spatial_rates(env_1d, spike_times, times, positions, min_speed=1e6)


def test_all_excluded_warns_once_batch(recwarn, env_1d, two_speed_trajectory) -> None:
    """The all-excluded warning fires exactly once for a multi-neuron batch."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = [np.array([0.5]), np.array([2.0]), np.array([6.2])]
    compute_spatial_rates(env_1d, spike_times, times, positions, min_speed=1e6)
    empty_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning)
        and "min_speed" in str(w.message)
        and "empty" in str(w.message)
    ]
    assert len(empty_warnings) == 1


def test_all_excluded_suppressed_when_warn_off(
    recwarn, env_1d, two_speed_trajectory
) -> None:
    """warn_on_drop=False suppresses the all-excluded warning (single + batch)."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]

    compute_spatial_rate(
        env_1d,
        np.array([0.5]),
        times,
        positions,
        min_speed=1e6,
        warn_on_drop=False,
    )
    compute_spatial_rates(
        env_1d,
        [np.array([0.5])],
        times,
        positions,
        min_speed=1e6,
        warn_on_drop=False,
    )

    empty_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning) and "empty" in str(w.message)
    ]
    assert empty_warnings == []


# ==============================================================================
# 8b. The all-excluded warning also fires for max_gap / out-of-bounds (R-follow)
# ==============================================================================
#
# Before this fix only the min_speed gate warned when it emptied the rate map;
# the SAME interval-valid mask also drops intervals for max_gap (large gaps)
# and out-of-bounds start samples, but those emptied the map SILENTLY. The
# generalized guard now fires uniformly for ALL THREE gates.


@pytest.fixture
def all_gap_trajectory() -> dict:
    """A trajectory where EVERY interval exceeds the default max_gap of 0.5 s.

    Sampled at dt=1.0 s, so every dt > 0.5 and the default max_gap excludes all
    intervals -> occupancy is all-zero and the rate map is empty. This emptied
    the map SILENTLY before the generalized warning.
    """
    dt = 1.0
    x = 10.0 + np.arange(10) * 1.0  # 10..19, all in-bounds
    positions = x.reshape(-1, 1)
    times = np.arange(len(x), dtype=np.float64) * dt
    return {"times": times, "positions": positions}


def _empty_map_warnings(recwarn) -> list:
    """Warnings matching the generalized empty-rate-map message."""
    return [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning)
        and "excluded ALL trajectory intervals" in str(w.message)
        and "the rate map is empty" in str(w.message)
    ]


def test_max_gap_excludes_all_warns_single(recwarn, env_1d, all_gap_trajectory) -> None:
    """max_gap (default 0.5) excluding every interval now WARNS (was silent)."""
    times = all_gap_trajectory["times"]
    positions = all_gap_trajectory["positions"]
    spike_times = np.array([0.5, 3.5, 6.5])

    res = compute_spatial_rate(env_1d, spike_times, times, positions)

    warns = _empty_map_warnings(recwarn)
    assert len(warns) == 1
    msg = str(warns[0].message)
    assert "max_gap=0.5" in msg  # names the active gate
    # The map is genuinely empty.
    assert np.nansum(res.occupancy) == 0.0
    fr = res.firing_rate
    assert np.all(np.isnan(fr)) or np.nansum(fr) == 0.0


def test_max_gap_excludes_all_warns_batch(recwarn, env_1d, all_gap_trajectory) -> None:
    """Batch path warns once when max_gap empties the rate map."""
    times = all_gap_trajectory["times"]
    positions = all_gap_trajectory["positions"]
    spike_times = [np.array([0.5, 3.5]), np.array([2.5, 6.5])]

    compute_spatial_rates(env_1d, spike_times, times, positions)
    assert len(_empty_map_warnings(recwarn)) == 1


def test_max_gap_excludes_all_suppressed_when_warn_off(
    recwarn, env_1d, all_gap_trajectory
) -> None:
    """warn_on_drop=False suppresses the max_gap empty-map warning."""
    times = all_gap_trajectory["times"]
    positions = all_gap_trajectory["positions"]

    compute_spatial_rate(
        env_1d,
        np.array([0.5]),
        times,
        positions,
        warn_on_drop=False,
    )
    compute_spatial_rates(
        env_1d,
        [np.array([0.5])],
        times,
        positions,
        warn_on_drop=False,
    )
    assert _empty_map_warnings(recwarn) == []


def test_oob_start_excludes_all_warns(recwarn, env_1d) -> None:
    """Every interval starting out of bounds empties the map and WARNS."""
    # Every START sample is out of bounds (x=-50), so every interval is dropped
    # by the start_bin < 0 gate. dt=0.1 keeps max_gap inactive and there is no
    # min_speed, so out-of-bounds is the SOLE cause.
    x = np.full(10, -50.0)
    positions = x.reshape(-1, 1)
    times = np.arange(len(x), dtype=np.float64) * 0.1
    assert env_1d.bin_at(np.array([[-50.0]]))[0] < 0
    spike_times = np.array([0.15, 0.45, 0.75])

    compute_spatial_rate(env_1d, spike_times, times, positions)
    warns = _empty_map_warnings(recwarn)
    assert len(warns) == 1
    # No max_gap / min_speed named; only the out-of-bounds possibility.
    msg = str(warns[0].message)
    assert "out of bounds" in msg


def test_min_speed_excludes_all_still_warns_and_names_min_speed(
    recwarn, env_1d, two_speed_trajectory
) -> None:
    """The min_speed case STILL warns under the generalized guard and names it."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = np.array([0.5, 5.5, 6.8])

    compute_spatial_rate(env_1d, spike_times, times, positions, min_speed=1e6)
    warns = _empty_map_warnings(recwarn)
    assert len(warns) == 1
    assert "min_speed=1000000.0" in str(warns[0].message)


def test_empty_map_warning_fires_once_for_batch(
    recwarn, env_1d, all_gap_trajectory
) -> None:
    """Exactly ONE empty-map warning for a multi-neuron batch (not per neuron)."""
    times = all_gap_trajectory["times"]
    positions = all_gap_trajectory["positions"]
    spike_times = [
        np.array([0.5]),
        np.array([3.5]),
        np.array([6.5]),
        np.array([2.5]),
    ]
    compute_spatial_rates(env_1d, spike_times, times, positions)
    assert len(_empty_map_warnings(recwarn)) == 1


def test_normal_session_does_not_warn_empty_map(
    recwarn, env_1d, two_speed_trajectory
) -> None:
    """A normal (non-empty) session emits NO empty-map warning (single + batch)."""
    times = two_speed_trajectory["times"]
    positions = two_speed_trajectory["positions"]
    spike_times = np.array([0.5, 2.5, 5.5, 6.5])

    compute_spatial_rate(env_1d, spike_times, times, positions)
    compute_spatial_rates(env_1d, [spike_times, spike_times], times, positions)
    assert _empty_map_warnings(recwarn) == []


# ==============================================================================
# 9. Final-sample gate symmetry for caller-supplied speed
# ==============================================================================


def test_final_sample_gate_matches_last_occupancy_interval(env_1d) -> None:
    """A spike exactly at times[-1] is gated by speed[n-2], like occupancy.

    Uses a caller-supplied speed whose last two elements DIFFER: speed[n-2]
    (the last occupancy interval) is above min_speed, while speed[n-1] is
    below. Occupancy keeps the last interval; the t_max spike must be kept too
    (gated by speed[n-2]), not dropped by the never-consulted speed[n-1].
    """
    # Dense slow ramp so each interval is under env.occupancy's default max_gap.
    x = np.linspace(10.0, 30.0, 11)
    positions = x.reshape(-1, 1)
    times = np.arange(len(x), dtype=np.float64) * 0.1
    n = len(times)
    min_speed = 5.0

    # Caller-supplied speed: last interval (n-2) HIGH (kept), final sample
    # (n-1) LOW (must be ignored for gating the t_max spike).
    speed = np.full(n, 10.0)
    speed[n - 1] = 0.0  # would drop the t_max spike under the old n-1 clip

    # A single spike landing exactly on the final timestamp.
    spike_times = np.array([times[-1]])

    counts = bin_spike_train(
        env_1d, spike_times, times, positions, speed=speed, min_speed=min_speed
    )
    # Occupancy keeps the last interval (speed[n-2]=10 >= 5), so the spike must
    # survive the gate (gated by speed[n-2], not the low speed[n-1]).
    assert counts.sum() == 1

    # Mirror: occupancy's last interval is non-empty (kept).
    occ = compute_occupancy(env_1d, times, positions, speed=speed, min_speed=min_speed)
    assert occ.sum() > 0
