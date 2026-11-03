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

    By chance 3 of these 15 steps are within 10% of three times the median
    step, about 10% of the analysed time, which reads as removed samples
    unless most steps are the median step.
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


def test_masked_samples_warn_with_camera_jitter():
    """A 30 Hz camera with 4 ms timestamp jitter, 0.4 s removed every 2 s.

    Fewer than half the steps are within 10% of the median step, but the
    sampling is regular and the removed runs still warn.
    """
    rng = np.random.default_rng(0)
    times = np.sort(np.arange(0.0, 600.0, 1 / 30) + rng.normal(0.0, 0.004, 18000))
    steps = np.diff(times) / np.median(np.diff(times))
    assert np.mean(np.abs(steps - 1.0) < 0.1) < 0.5
    positions = np.c_[50 + 40 * np.sin(times / 7.0), 50 + 40 * np.cos(times / 11.0)]
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    kept = times % 2.0 >= 0.4
    with pytest.warns(UserWarning, match="boolean-masked"):
        env.occupancy(times[kept], positions[kept])


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
