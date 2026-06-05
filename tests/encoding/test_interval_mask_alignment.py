"""Tests for FULL interval-mask alignment between spikes and occupancy (R1).

A firing rate map is ``spike_counts (numerator) / occupancy (denominator)``
per bin. ``env.occupancy`` drops an interval ``k`` (spanning ``[t_k, t_{k+1})``,
``time_allocation="start"``) for THREE reasons:

  * ``dt[k] > max_gap``           (large tracking gap; default max_gap=0.5 s)
  * ``speed[k] < min_speed``      (low-speed filtering)
  * ``start_bin[k] < 0``          (interval's start sample out of bounds)

Before this fix the spike binner only filtered by the time window and (since
task 2.6) by speed — NOT by ``max_gap`` and NOT by the out-of-bounds-start
rule. A spike inside a dropped interval (a tracking gap, or an out-of-bounds
excursion) was therefore COUNTED in the numerator while occupancy EXCLUDED its
time from the denominator, inflating the rate.

These tests pin that the spike numerator and the occupancy denominator now drop
the IDENTICAL set of intervals via the single shared ``interval_valid_mask``
helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._binning import (
    bin_spike_train,
    bin_spike_trains,
    compute_occupancy,
)
from neurospatial.encoding.spatial import compute_spatial_rate, compute_spatial_rates
from neurospatial.environment.trajectory import interval_valid_mask

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
def gap_trajectory() -> dict:
    """Dense 10 Hz trajectory with ONE deliberate large time gap.

    Samples 0..29 step at dt=0.1 s from x=10 to x=12.9 (all in bin 1). Between
    sample 29 and 30 there is a 1.0 s time gap (dt=1.0 > max_gap=0.5) while the
    animal sits near the same location; then samples 30..59 resume at dt=0.1 s
    from x=13.0 to x=15.9 (bins 1..1), so a spike interpolated into the gap
    interval lands in an OCCUPIED bin — the precise condition under which the
    pre-fix bug inflated the rate.

    Interval index 29 (spanning [t_29, t_30)) is the lone gap interval. A spike
    placed inside [t_29, t_30) must be dropped from the count AND its time
    excluded from occupancy.
    """
    dt = 0.1
    n_a = 30
    n_b = 30

    xa = 10.0 + np.arange(n_a) * 0.1  # 10.0 .. 12.9 -> bin 1
    xb = 13.0 + np.arange(n_b) * 0.1  # 13.0 .. 15.9 -> bin 1
    x = np.concatenate([xa, xb])
    positions = x.reshape(-1, 1)

    ta = np.arange(n_a) * dt
    # Insert a 1.0 s gap between segment a and b.
    tb = ta[-1] + 1.0 + np.arange(n_b) * dt
    times = np.concatenate([ta, tb])

    gap_interval = n_a - 1  # interval [t_29, t_30)
    return {
        "times": times,
        "positions": positions,
        "gap_interval": gap_interval,
    }


@pytest.fixture
def oob_trajectory(env_1d) -> dict:
    """Dense 10 Hz trajectory where ONE interval starts OUT OF BOUNDS.

    The environment spans 0-100. We drive the animal briefly to x=-50 (outside
    the active mask) for a single sample so that the interval STARTING at that
    out-of-bounds sample is dropped by env.occupancy (start_bin < 0). A spike
    inside that interval must be excluded too.
    """
    dt = 0.1
    # samples: in, in, OUT, in, in, ... all dt=0.1 so max_gap never triggers.
    x = np.array(
        [10.0, 11.0, -50.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        dtype=np.float64,
    )
    positions = x.reshape(-1, 1)
    times = np.arange(len(x)) * dt

    # interval starting at the OOB sample (index 2) spans [t_2, t_3).
    oob_interval = 2
    # Sanity: env.bin_at(-50) is -1.
    assert env_1d.bin_at(np.array([[-50.0]]))[0] < 0
    return {
        "times": times,
        "positions": positions,
        "oob_interval": oob_interval,
    }


# ==============================================================================
# 1. max_gap alignment (the reproduced bug)
# ==============================================================================


def test_max_gap_spike_in_gap_excluded_from_count(env_1d, gap_trajectory) -> None:
    """A spike inside a >max_gap interval is NOT counted (numerator)."""
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]
    gap_interval = gap_trajectory["gap_interval"]

    # A spike inside the gap interval [t_29, t_30).
    t_gap = 0.5 * (times[gap_interval] + times[gap_interval + 1])
    spike_times = np.array([t_gap])

    counts_default = bin_spike_train(
        env_1d, spike_times, times, positions, warn_on_drop=False
    )
    # With default max_gap=0.5 the spike falls in a dropped interval -> 0 count.
    assert counts_default.sum() == 0.0

    # With max_gap=None the gap is no longer dropped -> the spike IS counted.
    counts_no_gap = bin_spike_train(
        env_1d, spike_times, times, positions, max_gap=None, warn_on_drop=False
    )
    assert counts_no_gap.sum() == 1.0


def test_max_gap_occupancy_excludes_gap(env_1d, gap_trajectory) -> None:
    """env.occupancy excludes the gap interval's 1.0 s (denominator)."""
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]

    occ_default = compute_occupancy(env_1d, times, positions)
    occ_no_gap = compute_occupancy(env_1d, times, positions, max_gap=None)

    # The no-gap occupancy includes the 1.0 s gap interval; default excludes it.
    assert occ_no_gap.sum() == pytest.approx(occ_default.sum() + 1.0)


def test_max_gap_alignment_against_hand_masked_reference(
    env_1d, gap_trajectory
) -> None:
    """Numerator and denominator drop the IDENTICAL gap interval.

    Strong assertion: compare the public spike_counts/occupancy against a
    hand-masked reference that removes the gap interval from BOTH spikes and
    occupancy. The rate in the bin the gap would have populated is not
    inflated.
    """
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]
    gap_interval = gap_trajectory["gap_interval"]

    # Spikes: some in the dense in-bounds segments, ONE inside the gap.
    t_gap = 0.5 * (times[gap_interval] + times[gap_interval + 1])
    spike_times = np.array([0.5, 1.5, t_gap, 3.5, 4.2])

    counts = bin_spike_train(env_1d, spike_times, times, positions, warn_on_drop=False)
    occ = compute_occupancy(env_1d, times, positions)

    # --- Hand-masked reference: drop the gap interval from BOTH sides. ---
    mask = interval_valid_mask(times, positions, env_1d)
    # The gap interval is the dropped one.
    assert not mask[gap_interval]
    assert mask.sum() == len(mask) - 1

    # Reference spike count: keep only spikes whose interval is valid.
    interval = np.clip(
        np.searchsorted(times, spike_times, side="right") - 1, 0, len(times) - 2
    )
    keep = mask[interval]
    ref_spike_pos = np.interp(spike_times[keep], times, positions[:, 0]).reshape(-1, 1)
    ref_bins = env_1d.bin_at(ref_spike_pos)
    ref_counts = np.bincount(ref_bins[ref_bins >= 0], minlength=env_1d.n_bins).astype(
        np.float64
    )

    # Reference occupancy: accumulate dt for valid intervals at start bins.
    dt = np.diff(times)
    start_bin = env_1d.bin_at(positions).astype(np.int64)[:-1]
    ref_occ = np.zeros(env_1d.n_bins)
    for k in np.where(mask)[0]:
        ref_occ[start_bin[k]] += dt[k]

    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(occ, ref_occ)


def test_max_gap_rate_not_inflated(env_1d, gap_trajectory) -> None:
    """The full rate-map path does not inflate the gap bin's rate."""
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]
    gap_interval = gap_trajectory["gap_interval"]

    t_gap = 0.5 * (times[gap_interval] + times[gap_interval + 1])
    spike_times = np.array([t_gap])

    # binned smoothing keeps a clean numerator/denominator relationship.
    res_default = compute_spatial_rate(
        env_1d,
        spike_times,
        times,
        positions,
        smoothing_method="binned",
        bandwidth=1.0,
        warn_on_drop=False,
    )
    res_no_gap = compute_spatial_rate(
        env_1d,
        spike_times,
        times,
        positions,
        smoothing_method="binned",
        bandwidth=1.0,
        max_gap=None,
        warn_on_drop=False,
    )
    # Default: the single gap spike is dropped -> rate map has no positive rate.
    fr_default = np.nan_to_num(res_default.firing_rate, nan=0.0)
    assert fr_default.sum() == 0.0
    # max_gap=None: the gap interval's time IS now in occupancy and the spike
    # IS counted, so the bin the spike maps to gains a positive rate (the
    # previously-inflated path). The key contrast is default(0) vs None(>0).
    fr_no_gap = np.nan_to_num(res_no_gap.firing_rate, nan=0.0)
    assert fr_no_gap.sum() > 0.0


# ==============================================================================
# 2. invalid-start-bin alignment
# ==============================================================================


def test_oob_start_spike_excluded(env_1d, oob_trajectory) -> None:
    """A spike inside an out-of-bounds-START interval is excluded, matching occupancy."""
    times = oob_trajectory["times"]
    positions = oob_trajectory["positions"]
    oob_interval = oob_trajectory["oob_interval"]

    # Spike inside the OOB-start interval [t_2, t_3).
    t_oob = 0.5 * (times[oob_interval] + times[oob_interval + 1])
    spike_times = np.array([t_oob])

    counts = bin_spike_train(env_1d, spike_times, times, positions, warn_on_drop=False)
    # The interval is dropped by the shared mask -> spike not counted.
    assert counts.sum() == 0.0

    # And occupancy excludes that interval's start bin (it is -1).
    mask = interval_valid_mask(times, positions, env_1d)
    assert not mask[oob_interval]


def test_oob_alignment_against_reference(env_1d, oob_trajectory) -> None:
    """Spikes and occupancy drop the IDENTICAL out-of-bounds interval."""
    times = oob_trajectory["times"]
    positions = oob_trajectory["positions"]
    oob_interval = oob_trajectory["oob_interval"]

    t_oob = 0.5 * (times[oob_interval] + times[oob_interval + 1])
    spike_times = np.array([0.05, 0.15, t_oob, 0.45, 0.55])

    counts = bin_spike_train(env_1d, spike_times, times, positions, warn_on_drop=False)
    occ = compute_occupancy(env_1d, times, positions)

    mask = interval_valid_mask(times, positions, env_1d)
    assert not mask[oob_interval]

    interval = np.clip(
        np.searchsorted(times, spike_times, side="right") - 1, 0, len(times) - 2
    )
    keep = mask[interval]
    ref_pos = np.interp(spike_times[keep], times, positions[:, 0]).reshape(-1, 1)
    ref_bins = env_1d.bin_at(ref_pos)
    ref_counts = np.bincount(ref_bins[ref_bins >= 0], minlength=env_1d.n_bins).astype(
        np.float64
    )

    dt = np.diff(times)
    start_bin = env_1d.bin_at(positions).astype(np.int64)[:-1]
    ref_occ = np.zeros(env_1d.n_bins)
    for k in np.where(mask)[0]:
        ref_occ[start_bin[k]] += dt[k]

    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(occ, ref_occ)


# ==============================================================================
# 3. Combined: max_gap + min_speed + invalid-start-bin
# ==============================================================================


def test_combined_mask_alignment(env_1d) -> None:
    """max_gap, min_speed, and invalid-start-bin together drop the identical set."""
    dt = 0.1
    # Build a trajectory exercising all three drop reasons:
    #  - a low-speed creep (speed < min_speed)
    #  - a 1.0 s gap (dt > max_gap)
    #  - an out-of-bounds start sample
    x = np.array(
        [
            10.0,
            10.05,  # slow step (speed 0.5/s)
            10.10,  # slow step
            -50.0,  # OOB start sample
            30.0,  # back in bounds
            33.0,  # fast step (speed 30/s)
            36.0,
            39.0,
            42.0,
            45.0,
        ],
        dtype=np.float64,
    )
    positions = x.reshape(-1, 1)
    t = np.arange(len(x)) * dt
    # Insert a gap between index 7 and 8.
    t[8:] += 1.0
    times = t
    min_speed = 5.0

    speed = None  # let resolve_speed auto-derive (forward difference)

    # Spikes: at least one inside each kind of dropped interval, plus valid ones.
    spike_times = np.array(
        [
            0.05,  # slow interval 0 (dropped by min_speed)
            0.25,  # OOB-start interval 3 (dropped by start_bin<0)
            0.45,  # fast valid interval 4
            0.55,  # fast valid interval 5
        ]
    )
    # gap interval index 7 spans [t_7, t_8); add a spike there too.
    t_gap = 0.5 * (times[7] + times[8])
    spike_times = np.append(spike_times, t_gap)

    from neurospatial.encoding._binning import resolve_speed

    resolved = resolve_speed(times, positions, speed, min_speed)
    counts = bin_spike_train(
        env_1d,
        spike_times,
        times,
        positions,
        speed=resolved,
        min_speed=min_speed,
        warn_on_drop=False,
    )
    occ = compute_occupancy(
        env_1d, times, positions, speed=resolved, min_speed=min_speed
    )

    mask = interval_valid_mask(
        times, positions, env_1d, speed=resolved, min_speed=min_speed
    )

    # Reference numerator and denominator from the SAME mask.
    interval = np.clip(
        np.searchsorted(times, spike_times, side="right") - 1, 0, len(times) - 2
    )
    keep = mask[interval]
    ref_pos = np.interp(spike_times[keep], times, positions[:, 0]).reshape(-1, 1)
    ref_bins = env_1d.bin_at(ref_pos)
    ref_counts = np.bincount(ref_bins[ref_bins >= 0], minlength=env_1d.n_bins).astype(
        np.float64
    )

    dt_arr = np.diff(times)
    start_bin = env_1d.bin_at(positions).astype(np.int64)[:-1]
    ref_occ = np.zeros(env_1d.n_bins)
    for k in np.where(mask)[0]:
        ref_occ[start_bin[k]] += dt_arr[k]

    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(occ, ref_occ)


# ==============================================================================
# 4. max_gap=None keeps both un-gapped and still aligned
# ==============================================================================


def test_max_gap_none_aligned_and_matches_pre_fix(env_1d, gap_trajectory) -> None:
    """max_gap=None disables gap gating on BOTH sides and stays aligned."""
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]
    gap_interval = gap_trajectory["gap_interval"]

    t_gap = 0.5 * (times[gap_interval] + times[gap_interval + 1])
    spike_times = np.array([0.5, t_gap, 3.5])

    counts = bin_spike_train(
        env_1d, spike_times, times, positions, max_gap=None, warn_on_drop=False
    )
    occ = compute_occupancy(env_1d, times, positions, max_gap=None)

    mask = interval_valid_mask(times, positions, env_1d, max_gap=None)
    # No interval dropped (all in-bounds, no speed filter).
    assert mask.all()

    interval = np.clip(
        np.searchsorted(times, spike_times, side="right") - 1, 0, len(times) - 2
    )
    keep = mask[interval]
    ref_pos = np.interp(spike_times[keep], times, positions[:, 0]).reshape(-1, 1)
    ref_bins = env_1d.bin_at(ref_pos)
    ref_counts = np.bincount(ref_bins[ref_bins >= 0], minlength=env_1d.n_bins).astype(
        np.float64
    )

    dt = np.diff(times)
    start_bin = env_1d.bin_at(positions).astype(np.int64)[:-1]
    ref_occ = np.zeros(env_1d.n_bins)
    for k in np.where(mask)[0]:
        ref_occ[start_bin[k]] += dt[k]

    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(occ, ref_occ)
    # And the gap spike IS counted now.
    assert counts.sum() == 3.0


# ==============================================================================
# 5. Default behavior on gap-free, in-bounds data is unchanged
# ==============================================================================


def _clean_trajectory() -> dict:
    """A dense, gap-free, fully-in-bounds 10 Hz trajectory (no exclusions)."""
    dt = 0.1
    x = 10.0 + np.arange(80) * 1.0  # 10..89, all in-bounds, all dt=0.1
    positions = x.reshape(-1, 1)
    times = np.arange(len(x)) * dt
    return {"times": times, "positions": positions}


@pytest.mark.parametrize("method", SMOOTHING_METHODS)
def test_clean_data_default_unchanged_single(env_1d, method: str) -> None:
    """On clean data, the default (max_gap=0.5) introduces NO new exclusions.

    Pin against max_gap=None: on gap-free in-bounds data both must be identical
    (the gap gate never fires), so the default behavior is the pre-fix
    behavior.
    """
    traj = _clean_trajectory()
    times = traj["times"]
    positions = traj["positions"]
    spike_times = np.array([0.5, 2.5, 4.0, 5.5, 6.5])

    res_default = compute_spatial_rate(
        env_1d, spike_times, times, positions, smoothing_method=method
    )
    res_no_gap = compute_spatial_rate(
        env_1d,
        spike_times,
        times,
        positions,
        smoothing_method=method,
        max_gap=None,
    )
    nan_a = np.isnan(res_default.firing_rate)
    nan_b = np.isnan(res_no_gap.firing_rate)
    np.testing.assert_array_equal(nan_a, nan_b)
    np.testing.assert_array_equal(
        res_default.firing_rate[~nan_a], res_no_gap.firing_rate[~nan_b]
    )
    np.testing.assert_array_equal(res_default.occupancy, res_no_gap.occupancy)


@pytest.mark.parametrize("method", SMOOTHING_METHODS)
def test_clean_data_default_unchanged_batch(env_1d, method: str) -> None:
    """Batch: default == max_gap=None on clean data, all methods."""
    traj = _clean_trajectory()
    times = traj["times"]
    positions = traj["positions"]
    spike_times = [
        np.array([0.5, 2.5, 5.5]),
        np.array([1.0, 4.0, 6.5]),
        np.array([3.0, 6.0]),
    ]

    res_default = compute_spatial_rates(
        env_1d, spike_times, times, positions, smoothing_method=method
    )
    res_no_gap = compute_spatial_rates(
        env_1d,
        spike_times,
        times,
        positions,
        smoothing_method=method,
        max_gap=None,
    )
    nan_a = np.isnan(res_default.firing_rates)
    nan_b = np.isnan(res_no_gap.firing_rates)
    np.testing.assert_array_equal(nan_a, nan_b)
    np.testing.assert_array_equal(
        res_default.firing_rates[~nan_a], res_no_gap.firing_rates[~nan_b]
    )
    np.testing.assert_array_equal(res_default.occupancy, res_no_gap.occupancy)


# ==============================================================================
# 6. batch (n_jobs=2) vs single parity under active max_gap
# ==============================================================================


def test_batch_njobs_parity_under_max_gap(env_1d, gap_trajectory) -> None:
    """n_jobs=2 == n_jobs=1 with an active (default) max_gap drop."""
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]
    gap_interval = gap_trajectory["gap_interval"]
    t_gap = 0.5 * (times[gap_interval] + times[gap_interval + 1])

    # Segment a: t in [0, 2.9]; gap: [2.9, 3.9]; segment b: [3.9, 6.8].
    spike_times = [
        np.array([0.5, t_gap, 4.5]),  # valid, GAP, valid -> 2
        np.array([1.0, 5.0, t_gap]),  # valid, valid, GAP -> 2
        np.array([2.0]),  # valid -> 1
    ]

    c1, occ1 = bin_spike_trains(
        env_1d, spike_times, times, positions, n_jobs=1, warn_on_drop=False
    )
    c2, occ2 = bin_spike_trains(
        env_1d, spike_times, times, positions, n_jobs=2, warn_on_drop=False
    )
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(occ1, occ2)
    # The gap spikes are dropped from every neuron.
    assert c1[0].sum() == 2.0
    assert c1[1].sum() == 2.0
    assert c1[2].sum() == 1.0


# ==============================================================================
# 7. Performance: interval-valid mask is computed ONCE per batch, not per neuron
# ==============================================================================


def _make_many_neurons(times, n_neurons: int, seed: int = 0):
    """A list of ``n_neurons`` random spike trains spanning the time window."""
    rng = np.random.default_rng(seed)
    t0, t1 = float(times.min()), float(times.max())
    return [np.sort(rng.uniform(t0, t1, size=20)) for _ in range(n_neurons)]


def test_interval_mask_not_recomputed_per_neuron_batch(
    monkeypatch, env_1d, gap_trajectory
) -> None:
    """``interval_valid_mask`` call count is INDEPENDENT of n_neurons (batch).

    The interval-valid mask depends only on ``(times, positions, env, speed,
    min_speed, max_gap)`` — none vary per neuron — so the batch path must
    compute it ONCE and reuse it across neurons. We spy on the single shared
    helper and assert its call count does not grow with the neuron count (the
    pre-fix code called it once per neuron inside the loop).
    """
    import neurospatial.environment.trajectory as traj_mod

    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]

    real_mask = traj_mod.interval_valid_mask
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real_mask(*args, **kwargs)

    monkeypatch.setattr(traj_mod, "interval_valid_mask", _spy)

    # 5 neurons.
    calls["n"] = 0
    bin_spike_trains(
        env_1d,
        _make_many_neurons(times, 5),
        times,
        positions,
        n_jobs=1,
        warn_on_drop=False,
    )
    calls_5 = calls["n"]

    # 50 neurons — same trajectory, same gate params.
    calls["n"] = 0
    bin_spike_trains(
        env_1d,
        _make_many_neurons(times, 50),
        times,
        positions,
        n_jobs=1,
        warn_on_drop=False,
    )
    calls_50 = calls["n"]

    # Constant regardless of neuron count (the recompute-per-neuron is gone).
    assert calls_5 == calls_50
    # Concretely: once for env.occupancy's denominator mask + once for the
    # shared spike-side mask = 2 per batch call (NOT n_neurons + 1).
    assert calls_5 == 2


def test_compute_spatial_rates_mask_computed_once(
    monkeypatch, env_1d, gap_trajectory
) -> None:
    """A `compute_spatial_rates` call computes the shared mask a constant # of times."""
    import neurospatial.environment.trajectory as traj_mod

    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]

    real_mask = traj_mod.interval_valid_mask
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real_mask(*args, **kwargs)

    monkeypatch.setattr(traj_mod, "interval_valid_mask", _spy)

    n_neurons = 8
    spike_trains = _make_many_neurons(times, n_neurons, seed=1)

    calls["n"] = 0
    compute_spatial_rates(env_1d, spike_trains, times, positions, warn_on_drop=False)
    # Independent of n_neurons: NOT n_neurons-scaled. (2 = occupancy + spikes.)
    assert calls["n"] == 2


def test_interval_mask_precompute_results_unchanged(env_1d, gap_trajectory) -> None:
    """Precomputing the mask once yields byte-for-byte identical batch results.

    Cross-checks the once-computed batch path against the single-neuron path
    (which also precomputes-and-passes), so the optimization cannot have
    changed any count.
    """
    times = gap_trajectory["times"]
    positions = gap_trajectory["positions"]
    spike_trains = _make_many_neurons(times, 6, seed=2)

    batch_counts, _ = bin_spike_trains(
        env_1d, spike_trains, times, positions, n_jobs=1, warn_on_drop=False
    )
    for i, spikes in enumerate(spike_trains):
        single = bin_spike_train(env_1d, spikes, times, positions, warn_on_drop=False)
        np.testing.assert_array_equal(batch_counts[i], single)
