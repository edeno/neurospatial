"""Spatial rate maps warn when they silently lose a meaningful part of the input."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior import compute_speed
from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates


def _shift_out(spike_times, times, fraction):
    """Spike trains shifted later so ``fraction`` of all spikes pass ``times[-1]``."""
    pooled = np.sort(np.concatenate(spike_times))
    shift = times[-1] - pooled[int(np.ceil((1.0 - fraction) * pooled.size))]
    shifted = [train + shift for train in spike_times]
    # Independent count of the spikes the analysis cannot place on the clock.
    pooled = np.concatenate(shifted)
    n_outside = int(np.count_nonzero((pooled < times[0]) | (pooled >= times[-1])))
    return shifted, n_outside, pooled.size


def test_partial_spike_drop_warns_above_5pct(open_field_120s):
    session = open_field_120s
    times, positions = session.times, session.positions

    shifted, n_outside, n_total = _shift_out(session.spike_times, times, 0.28)
    assert 0.25 < n_outside / n_total < 0.31
    with pytest.warns(
        UserWarning, match="fell outside the position time window"
    ) as record:
        rates = compute_spatial_rates(session.env, shifted, times, positions)
    assert len(record) == 1
    assert f"{n_outside}/{n_total} spike_times" in str(record[0].message)
    # The dropped spikes still do not count; the warning only reports them.
    assert rates.n_spikes_outside_intervals.sum() == n_outside

    shifted, n_outside, n_total = _shift_out(session.spike_times, times, 0.02)
    assert 0 < n_outside / n_total <= 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_rates(session.env, shifted, times, positions)


def test_tiny_spatial_bandwidth_warns(open_field_120s):
    session = open_field_120s
    env, spikes = session.env, session.spike_times[0]
    with pytest.warns(UserWarning, match=r"bandwidth=0.05 is far below the 2 cm bin"):
        tiny = compute_spatial_rate(
            env, spikes, session.times, session.positions, bandwidth=0.05
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binned = compute_spatial_rate(
            env, spikes, session.times, session.positions, method="binned", bandwidth=0
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = compute_spatial_rate(env, spikes, session.times, session.positions)
    # What the message claims: a 0.05 cm kernel on 2 cm bins barely smooths.
    # It moves the map by well under 1% (L1), where the default 5 cm kernel
    # moves it by most of its mass.
    finite = (
        np.isfinite(tiny.firing_rate)
        & np.isfinite(binned.firing_rate)
        & np.isfinite(default.firing_rate)
    )
    reference = binned.firing_rate[finite]

    def change(rates):
        return np.abs(rates[finite] - reference).sum() / np.abs(reference).sum()

    assert change(tiny.firing_rate) < 0.01
    assert change(default.firing_rate) > 0.5


def test_tiny_bandwidth_warns_for_a_population_with_no_units(open_field_120s):
    """The bandwidth is wrong whether or not there are units to smooth."""
    session = open_field_120s
    with pytest.warns(UserWarning, match=r"bandwidth=0.05 is far below the 2 cm bin"):
        rates = compute_spatial_rates(
            session.env, [], session.times, session.positions, bandwidth=0.05
        )
    assert rates.firing_rates.shape == (0, session.env.n_bins)


def test_tiny_bandwidth_cutoff_is_a_quarter_bin(open_field_120s):
    """The cutoff is 0.25 of the 2 cm bin length (0.5 cm), inclusive."""
    session = open_field_120s
    env, spikes = session.env, session.spike_times[0]
    assert np.allclose(env.bin_sizes, 4.0)  # 2 cm x 2 cm bins: length 2 cm
    for bandwidth in (0.5, 0.501):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_spatial_rate(
                env, spikes, session.times, session.positions, bandwidth=bandwidth
            )
    with pytest.warns(UserWarning, match=r"bandwidth=0.499 is far below the 2 cm"):
        compute_spatial_rate(
            env, spikes, session.times, session.positions, bandwidth=0.499
        )


@pytest.mark.parametrize("single", [True, False], ids=["single", "population"])
def test_input_errors_come_before_the_tiny_bandwidth_warning(open_field_120s, single):
    """With warnings as errors, an input error is still the error raised."""
    from neurospatial import InputError

    session = open_field_120s
    env, times, positions = session.env, session.times, session.positions
    compute = compute_spatial_rate if single else compute_spatial_rates
    spikes = session.spike_times[0] if single else session.spike_times
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InputError, match="times and positions must have"):
            compute(env, spikes, times[:-1], positions, bandwidth=0.05)
        with pytest.raises(InputError, match=r"bandwidth='5' is not a finite number"):
            compute(env, spikes, times, positions, bandwidth="5")


def test_view_input_error_comes_before_the_tiny_bandwidth_warning(open_field_120s):
    from neurospatial import InputError
    from neurospatial.encoding import compute_view_rate

    session = open_field_120s
    env, times, positions = session.env, session.times, session.positions
    headings = np.zeros(times.size)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InputError, match="times and positions must have"):
            compute_view_rate(
                env,
                session.spike_times[0],
                times[:-1],
                positions,
                headings,
                bandwidth=0.05,
            )


def test_brief_tracking_dropout_is_silent(open_field_120s):
    session = open_field_120s
    env, times, spikes = session.env, session.times, session.spike_times[0]
    # Ten 0.3 s tracking losses (30 samples at 100 Hz), 2.5% of the session:
    # long enough to count as removed runs, too few to look masked.
    rng = np.random.default_rng(5)
    starts = 100 + 1000 * np.arange(10) + rng.integers(0, 900, 10)
    lost = np.zeros(times.size, dtype=bool)
    for start in starts:
        lost[start : start + 30] = True
    assert 0.02 < lost.mean() < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deleted = compute_spatial_rate(
            env, spikes, times[~lost], session.positions[~lost]
        )
    with_nan = np.array(session.positions, copy=True)
    with_nan[lost] = np.nan
    marked = compute_spatial_rate(env, spikes, times, with_nan)
    # Deleted rows and NaN rows are the same missing samples.
    np.testing.assert_allclose(deleted.occupancy, marked.occupancy, atol=1e-12)


def test_boolean_masked_samples_warn(open_field_120s):
    session = open_field_120s
    env, times, positions = session.env, session.times, session.positions
    spikes = session.spike_times[0]
    moving = compute_speed(times, positions) > 5.0
    with pytest.warns(UserWarning, match="boolean-masked") as record:
        masked = compute_spatial_rate(env, spikes, times[moving], positions[moving])
    assert len(record) == 1
    assert "min_speed=" in str(record[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gated = compute_spatial_rate(env, spikes, times, positions, min_speed=5.0)
    # What the warning claims: the bridged rest periods count as occupancy.
    assert masked.occupancy.sum() > 1.05 * gated.occupancy.sum()


def test_occlusion_dropped_frames_warn_with_both_causes():
    """Occlusions of 0.3-0.8 s dropped as low-likelihood frames, 8% of the time.

    Deleted rows and NaN rows give the same occupancy, but only deleted rows
    can come from ``times[mask]``, so only they warn. The warning cannot tell
    a mask from dropped tracking frames, so it names both, each with its fix.
    """
    rng = np.random.default_rng(1)
    times = np.arange(0.0, 600.0, 1 / 30)
    positions = np.c_[50 + 40 * np.sin(times / 7.0), 50 + 40 * np.cos(times / 11.0)]
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    dropped = np.zeros(times.size, dtype=bool)
    starts = np.flatnonzero(rng.random(times.size) < 0.006)
    for start in starts:
        dropped[start : start + int(rng.uniform(0.3, 0.8) * 30)] = True
    assert 0.05 < dropped.mean() < 0.15
    with pytest.warns(UserWarning, match="dropped tracking frames") as record:
        deleted = env.occupancy(times[~dropped], positions[~dropped])
    message = str(record[0].message)
    assert "boolean-masked samples" in message
    assert "min_speed=" in message
    assert "bridging them is intended up to max_gap" in message
    assert "pass a max_gap below the shortest removed run" in message
    with_nan = positions.copy()
    with_nan[dropped] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        marked = env.occupancy(times, with_nan)
    np.testing.assert_allclose(deleted, marked, atol=1e-9)
    # The fix for dropped frames: a max_gap below the shortest occlusion
    # (9 frames, 0.3 s) excludes them, and nothing warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        excluded = env.occupancy(times[~dropped], positions[~dropped], max_gap=0.25)
    np.testing.assert_allclose(
        excluded.sum(), np.count_nonzero(~dropped[:-1] & ~dropped[1:]) / 30, rtol=1e-6
    )


def test_dropped_frames_are_silent():
    rng = np.random.default_rng(11)
    times = np.arange(0.0, 300.0, 1 / 30)
    positions = np.c_[50 + 40 * np.sin(times / 7.0), 50 + 40 * np.cos(times / 11.0)]
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    kept = rng.random(times.size) >= 0.005
    spikes = np.sort(rng.uniform(times[0], times[-1], 600))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_rate(env, spikes, times[kept], positions[kept])


def test_irregular_sampling_is_not_called_masked():
    """Steps spread over 0.001-0.45 s have no sampling interval to be a multiple of.

    Read against their shortest steps (0.02 s), these 15 steps lean toward
    whole multiples by chance (mean phase cosine 0.15, above
    ``MASKED_REGULAR_COHERENCE``), and six of them would count as removed
    runs, 69% of the analysed time. Irregular steps reach that cosine with
    probability 0.21, so the chance check keeps it silent.
    """
    rng = np.random.default_rng(491)
    positions = rng.uniform((-3.0, 11.0), (-1.0, 13.0), size=(16, 2))
    positions[::7] = np.where(rng.random((3, 2)) < 0.5, (-3.0, 11.0), (-1.0, 13.0))
    times = 10.0 + np.cumsum(rng.uniform(0.001, 0.45, 16))
    edges = (np.arange(3.0) - 3.0, np.arange(3.0) + 11.0)
    env = Environment.from_grid_mask(np.ones((2, 2), dtype=bool), edges)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        occupancy = env.occupancy(times, positions, max_gap=0.5)
    np.testing.assert_allclose(occupancy.sum(), times[-1] - times[0], rtol=1e-12)


_FS = 30.0


def _removed_share(sampled, fs, max_gap=1.0):
    """The share of the analysed time a removed-runs check should report.

    Counted from which samples of a regular ``fs`` clock were kept
    (``sampled``), independently of the timestamps: a step across ``g - 1``
    removed samples spans ``g / fs`` seconds and is bridged when that is at
    most ``max_gap``; its removed samples count when they last at least
    ``MASKED_RUN_MIN_DURATION``.
    """
    from neurospatial._intervals import MASKED_RUN_MIN_DURATION

    spanned = np.diff(np.flatnonzero(sampled))
    bridged = spanned / fs <= max_gap
    removed = (spanned - 1) / fs
    counted = bridged & (removed >= MASKED_RUN_MIN_DURATION)
    return removed[counted].sum() / (spanned[bridged].sum() / fs)


def _speed_masked(rng, fs, jitter_sd, *, uniform=False, tick=None):
    """A camera at ``fs`` for 600 s, 0.4 s of samples removed every 2 s.

    Timestamps jitter with SD ``jitter_sd`` (or uniformly within
    ``±jitter_sd``) and are rounded to ``tick`` seconds when given. Returns
    the sorted timestamps of the kept samples and which samples of the
    camera's clock were kept.
    """
    nominal = np.arange(0.0, 600.0, 1 / fs)
    sampled = nominal % 2.0 >= 0.4
    if uniform:
        jitter = rng.uniform(-jitter_sd, jitter_sd, nominal.size)
    else:
        jitter = rng.normal(0.0, jitter_sd, nominal.size)
    times = np.sort((nominal + jitter)[sampled])
    if tick is not None:
        times = np.round(times / tick) * tick
    return times, sampled


def _fragmented(rng):
    """``times[speed > thr]`` for a mostly resting animal at 30 Hz.

    Kept runs of 1-4 frames between removed runs of 10-40 frames. Returns the
    kept times and which frames were kept.
    """
    keep = rng.integers(1, 5, 3000)
    drop = rng.integers(10, 41, 3000)
    sampled = np.concatenate(
        [
            np.r_[np.ones(k, bool), np.zeros(m, bool)]
            for k, m in zip(keep, drop, strict=True)
        ]
    )
    return (np.arange(sampled.size) / _FS)[sampled], sampled


def _spike_timed(rng):
    """Positions read at spike times (as in DANDI 001754): steps spread over
    1 ms to 0.65 s, log-uniformly, on a 30 kHz clock."""
    steps = np.exp(rng.uniform(np.log(0.001), np.log(0.65), 20000))
    return np.unique(np.round(np.cumsum(steps) * 30000) / 30000)


def _pooled_spike_times(rng, rate, tick):
    """Positions read at pooled spike times, rounded to ``tick`` seconds.

    Bursts of activity at ``rate`` Hz for 0.5-2 s, between silences of
    0.3-0.8 s, for 600 s.
    """
    bursts, start = [], 0.0
    while start < 600.0:
        duration = rng.uniform(0.5, 2.0)
        bursts.append(start + rng.uniform(0.0, duration, rng.poisson(rate * duration)))
        start += duration + rng.uniform(0.3, 0.8)
    return np.unique(np.round(np.concatenate(bursts) / tick) * tick)


def _occupancy_warnings(times):
    positions = np.c_[50 + 40 * np.sin(times / 7.0), 50 + 40 * np.cos(times / 11.0)]
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        env.occupancy(times, positions)
    return [str(w.message) for w in record]


@pytest.mark.parametrize(
    ("fs", "jitter_sd"),
    [(30.0, 0.004), (30.0, 0.006), (60.0, 0.003), (120.0, 0.002)],
    ids=["30Hz-4ms", "30Hz-6ms", "60Hz-3ms", "120Hz-2ms"],
)
def test_masked_samples_warn_with_camera_jitter(fs, jitter_sd):
    """A camera with Gaussian timestamp jitter, 0.4 s removed every 2 s.

    Fewer than half the steps are within 10% of the median step, but the
    sampling is regular, so the removed runs warn, with their share of the
    analysed time.
    """
    times, sampled = _speed_masked(np.random.default_rng(0), fs, jitter_sd)
    steps = np.diff(times) / np.median(np.diff(times))
    assert np.mean(np.abs(steps - 1.0) < 0.1) < 0.5
    messages = _occupancy_warnings(times)
    assert len(messages) == 1
    assert "boolean-masked" in messages[0]
    reported = float(messages[0].split("About ")[1].split("%")[0]) / 100
    np.testing.assert_allclose(reported, _removed_share(sampled, fs), rtol=0.05)


@pytest.mark.parametrize("tick", [1e-3, 1e-4], ids=["1ms", "0.1ms"])
def test_rounded_irregular_times_are_not_called_masked(tick):
    """Pooled spike times rounded to a clock tick are still irregular.

    At 1 ms, 14% of the steps are exactly one tick, and every step is a whole
    number of ticks; the silences between bursts are not removed samples.
    """
    times = _pooled_spike_times(np.random.default_rng(0), 150.0, tick)
    if tick == 1e-3:
        assert 0.1 < np.mean(np.isclose(np.diff(times), tick)) < 0.2
    assert _occupancy_warnings(times) == []


# name: simulate(rng) -> (kept times, which samples of the 30 Hz clock were kept)
_REGULAR_CASES = {
    "speed mask": lambda rng: _speed_masked(rng, _FS, 0.0),
    "speed mask, 10 ms jitter window": lambda rng: _speed_masked(
        rng, _FS, 0.005, uniform=True
    ),
    "speed mask, 4 ms SD jitter": lambda rng: _speed_masked(rng, _FS, 0.004),
    "speed mask, 6 ms SD jitter": lambda rng: _speed_masked(rng, _FS, 0.006),
    "speed mask, 6 ms SD jitter, 1 ms clock": lambda rng: _speed_masked(
        rng, _FS, 0.006, tick=1e-3
    ),
    "fragmented mask": _fragmented,
}

# name: simulate(rng) -> times
_IRREGULAR_CASES = {
    "uniform steps": lambda rng: np.cumsum(rng.uniform(0.001, 0.45, 5000)),
    "exponential steps": lambda rng: np.cumsum(rng.exponential(1 / _FS, 5000)),
    "spike-timed": _spike_timed,
    "pooled spikes, 1 ms clock": lambda rng: _pooled_spike_times(rng, 150.0, 1e-3),
    "pooled spikes, 0.1 ms clock": lambda rng: _pooled_spike_times(rng, 150.0, 1e-4),
    "slower pooled spikes, 1 ms clock": lambda rng: _pooled_spike_times(
        rng, 50.0, 1e-3
    ),
}


def _bridged_fraction(times, max_gap=1.0):
    from neurospatial._intervals import _masked_sample_fraction

    return _masked_sample_fraction(times, np.diff(times) <= max_gap, max_gap)


@pytest.mark.parametrize("case", list(_REGULAR_CASES))
def test_masking_check_reports_the_removed_share(case):
    """Masked regular sampling warns, reporting the share it removed.

    The share is counted from which camera frames were kept, not from the
    timestamps. Five seeds per case.
    """
    from neurospatial._constants import DISCARD_WARN_FRACTION

    for seed in range(5):
        times, sampled = _REGULAR_CASES[case](np.random.default_rng(seed))
        truth = _removed_share(sampled, _FS)
        assert truth > 2 * DISCARD_WARN_FRACTION
        np.testing.assert_allclose(_bridged_fraction(times), truth, rtol=0.05)


@pytest.mark.parametrize("case", list(_IRREGULAR_CASES))
def test_masking_check_is_silent_on_irregular_sampling(case):
    """Irregular sampling has no removed samples to report. Five seeds per case."""
    for seed in range(5):
        times = _IRREGULAR_CASES[case](np.random.default_rng(100 + seed))
        assert _bridged_fraction(times) == 0.0


@pytest.mark.parametrize(
    "draw",
    [
        lambda rng, n: rng.uniform(0.001, 0.45, n),
        lambda rng, n: rng.exponential(0.15, n),
        lambda rng, n: np.exp(rng.uniform(np.log(0.001), np.log(0.65), n)),
    ],
    ids=["uniform", "exponential", "log-uniform"],
)
def test_masking_check_rarely_calls_short_irregular_runs_masked(draw):
    """2000 short runs (11-60 steps) of irregular steps rarely warn.

    A few random steps can look locked to an interval by chance; the
    chance check (``MASKED_REGULAR_CHANCE``) keeps that below 0.5% of runs
    (at most 1 of 2000 measured). Without it, 889-997 of the 2000 runs of
    each kind warn.
    """
    from neurospatial._constants import DISCARD_WARN_FRACTION

    rng = np.random.default_rng(7)
    n_warned = 0
    for _ in range(2000):
        times = 10.0 + np.cumsum(np.r_[0.0, draw(rng, rng.integers(11, 61))])
        n_warned += _bridged_fraction(times) > DISCARD_WARN_FRACTION
    assert n_warned <= 10


def test_fragmented_mask_warns_and_inflates_occupancy():
    """Kept runs of 1-4 frames between removed runs of 10-40 frames at 30 Hz.

    Most of the bridged time is in removed runs, so occupancy is several
    times the time of the kept frames.
    """
    times, _ = _fragmented(np.random.default_rng(7))
    positions = np.c_[50 + 40 * np.sin(times / 7.0), 50 + 40 * np.cos(times / 11.0)]
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    with pytest.warns(UserWarning, match="boolean-masked"):
        occupancy = env.occupancy(times, positions)
    kept_time = times.size / _FS
    assert occupancy.sum() > 4.0 * kept_time


def test_polar_bandwidth_is_compared_with_the_ring_width(
    ou_2min_env, ou_2min, obj, noise_trains
):
    """Polar arcs grow with distance; the radial ring width is the bin length."""
    from neurospatial.encoding import compute_object_vector_rate

    times, positions, _ = ou_2min
    spikes = noise_trains(120)[0]
    kwargs = {"distance_range": (0.0, 50.0), "distance_bin_size": 5.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        smoothed = compute_object_vector_rate(
            ou_2min_env,
            spikes,
            times,
            positions,
            obj,
            method="diffusion_kde",
            bandwidth=3.0,
            **kwargs,
        )
        binned = compute_object_vector_rate(
            ou_2min_env,
            spikes,
            times,
            positions,
            obj,
            method="binned",
            bandwidth=0.0,
            **kwargs,
        )
    # A 3 cm kernel on 5 cm rings does smooth the map.
    finite = np.isfinite(smoothed.firing_rate) & np.isfinite(binned.firing_rate)
    change = np.abs(smoothed.firing_rate[finite] - binned.firing_rate[finite]).sum()
    assert change / np.abs(binned.firing_rate[finite]).sum() > 0.05
    with pytest.warns(UserWarning, match=r"bandwidth=0.05 is far below the 5 cm"):
        compute_object_vector_rate(
            ou_2min_env,
            spikes,
            times,
            positions,
            obj,
            method="diffusion_kde",
            bandwidth=0.05,
            **kwargs,
        )


def test_masking_and_spike_drop_warn_together(open_field_120s):
    """One warning per call, naming every problem found."""
    session = open_field_120s
    env, times, positions = session.env, session.times, session.positions
    moving = compute_speed(times, positions) > 5.0
    shifted, _, _ = _shift_out(session.spike_times, times, 0.29)
    with pytest.warns(UserWarning, match="boolean-masked") as record:
        compute_spatial_rates(env, shifted, times[moving], positions[moving])
    assert len(record) == 1
    message = str(record[0].message)
    assert "boolean-masked" in message
    assert "fell outside the position time window" in message
