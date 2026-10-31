"""A missing sample is a deleted row, in every analysis.

Video and pose tracking lose samples two ways: a dropped frame removes the
row, and a low-confidence keypoint leaves the row with NaN coordinates. Both
are the same missing observation, so every analysis must give the same result
for either. A gap between consecutive valid samples no longer than
``max_gap`` is bridged; a longer one is a tracking loss.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._sampling import observed_support, sample_at_times
from neurospatial.behavior import compute_speed
from neurospatial.behavior.segmentation import segment_by_velocity
from neurospatial.encoding import compute_directional_rate, compute_spatial_rate
from neurospatial.simulation import generate_poisson_spikes

FS = 30.0


@pytest.fixture(scope="module")
def tracking():
    """60 s at 30 Hz: running laps with pauses, DeepLabCut-style NaN bursts.

    Returns the clean recording and the indices of rows lost to isolated
    dropped frames and to low-confidence bursts of 1-6 frames.
    """
    rng = np.random.default_rng(20)
    times = np.arange(0, 60, 1 / FS)
    phase = 2 * np.pi * times / 12.0
    x = 50 + 40 * np.sin(phase) * (np.abs(np.sin(phase)) > 0.2)
    positions = np.column_stack([x, 50 + 10 * np.cos(phase)])
    positions += rng.normal(0, 0.3, positions.shape)
    lost = np.zeros(times.size, dtype=bool)
    lost[rng.random(times.size) < 0.03] = True
    for start in rng.choice(np.arange(5, times.size - 10), 25, replace=False):
        lost[start : start + rng.integers(1, 7)] = True
    lost[[0, -1]] = False
    headings = np.unwrap(
        np.arctan2(np.gradient(positions[:, 1]), 1e-9 + np.gradient(x))
    )
    spikes = generate_poisson_spikes(
        2 + 10 * np.exp(-((x - 70) ** 2) / 200), times, seed=21
    )
    return times, positions, np.angle(np.exp(1j * headings)), spikes, lost


@pytest.fixture(scope="module")
def env():
    grid = np.stack(np.meshgrid(np.arange(0, 101.0), np.arange(30, 71.0)), -1)
    return Environment.from_samples(grid.reshape(-1, 2), bin_size=4.0, units="cm")


def _with_nan(values, lost):
    values = np.array(values, dtype=float, copy=True)
    values[lost] = np.nan
    return values


# Rows also lost at the start and end of the recording, where no valid sample
# bounds the gap on one side.
LOST_ENDS = {"interior": [], "start": [0, 1, 2], "end": [-3, -2, -1]}
LOST_ENDS["both"] = LOST_ENDS["start"] + LOST_ENDS["end"]


def _lose_ends(lost, ends):
    lost = lost.copy()
    lost[LOST_ENDS[ends]] = True
    return lost


@pytest.mark.parametrize("ends", sorted(LOST_ENDS))
@pytest.mark.parametrize("time_allocation", ["start", "linear"])
def test_occupancy_with_nan_rows_equals_deleted_rows(
    env, tracking, time_allocation, ends
):
    times, positions, _, _, lost = tracking
    lost = _lose_ends(lost, ends)
    kept = ~lost
    with_nan = env.occupancy(
        times, _with_nan(positions, lost), time_allocation=time_allocation
    )
    deleted = env.occupancy(
        times[kept], positions[kept], time_allocation=time_allocation
    )
    np.testing.assert_allclose(with_nan, deleted, rtol=0, atol=1e-12)
    # Every gap is shorter than max_gap, so the time from the first to the
    # last valid sample is kept, and nothing outside it.
    observed = times[kept]
    np.testing.assert_allclose(with_nan.sum(), observed[-1] - observed[0], rtol=1e-12)


@pytest.mark.parametrize("ends", sorted(LOST_ENDS))
def test_spatial_rate_with_nan_rows_equals_deleted_rows(env, tracking, ends):
    times, positions, _, spikes, lost = tracking
    lost = _lose_ends(lost, ends)
    kept = ~lost
    with_nan = compute_spatial_rate(
        env, spikes, times, _with_nan(positions, lost), method="binned"
    )
    deleted = compute_spatial_rate(
        env, spikes, times[kept], positions[kept], method="binned"
    )
    np.testing.assert_allclose(with_nan.occupancy, deleted.occupancy, atol=1e-12)
    np.testing.assert_allclose(
        with_nan.firing_rate, deleted.firing_rate, rtol=1e-12, atol=0
    )
    assert (
        with_nan.analysis_settings["observed_intervals"]
        == deleted.analysis_settings["observed_intervals"]
    )


@pytest.mark.parametrize("ends", sorted(LOST_ENDS))
def test_directional_rate_with_nan_headings_equals_deleted_rows(tracking, ends):
    times, _, headings, spikes, lost = tracking
    lost = _lose_ends(lost, ends)
    kept = ~lost
    with_nan = compute_directional_rate(spikes, times, _with_nan(headings, lost))
    deleted = compute_directional_rate(spikes, times[kept], headings[kept])
    np.testing.assert_allclose(with_nan.occupancy, deleted.occupancy, atol=1e-12)
    np.testing.assert_allclose(with_nan.firing_rate, deleted.firing_rate, rtol=1e-12)


@pytest.mark.parametrize("ends", sorted(LOST_ENDS))
def test_speed_and_running_bouts_with_nan_rows_equal_deleted_rows(tracking, ends):
    times, positions, _, _, lost = tracking
    lost = _lose_ends(lost, ends)
    kept = ~lost
    with_nan = _with_nan(positions, lost)
    speed = compute_speed(times, with_nan)
    np.testing.assert_allclose(speed[kept], compute_speed(times[kept], positions[kept]))
    assert np.isnan(speed[lost]).all()
    bouts = segment_by_velocity(times, with_nan, min_speed=10.0)
    expected = segment_by_velocity(times[kept], positions[kept], min_speed=10.0)
    np.testing.assert_array_equal(bouts, expected)
    assert len(bouts) > 4


def test_speed_across_a_bridged_gap_uses_the_elapsed_time():
    times = np.arange(5) / 10
    positions = np.column_stack([10.0 * np.arange(5), np.zeros(5)])  # 100 cm/s
    positions[2] = np.nan
    speed = compute_speed(times, positions, smooth_window=0.0)
    np.testing.assert_allclose(speed[[0, 1, 3, 4]], 100.0)
    assert np.isnan(speed[2])


def test_gap_longer_than_max_gap_is_a_tracking_loss(env, tracking):
    times, positions, _, _, lost = tracking
    with_nan = _with_nan(positions, lost)
    occupancy = env.occupancy(times, with_nan, max_gap=1.5 / FS)
    # Every gap that lost a frame is excluded at this max_gap.
    valid = np.flatnonzero(~lost)
    gaps = np.diff(times[valid])
    np.testing.assert_allclose(occupancy.sum(), gaps[gaps <= 1.5 / FS].sum())


@pytest.mark.parametrize("analysis", ["occupancy", "spatial_rate", "directional_rate"])
def test_all_gaps_too_long_warning_names_the_missing_samples(env, tracking, analysis):
    """Sparse valid samples: the warning blames the gaps, not the coordinates.

    Every analysis warns with the gates that built its interval mask, so a
    selection emptied by gaps across missing samples says so instead of
    suggesting the positions are out of bounds.
    """
    times, positions, headings, spikes, _ = tracking
    sparse = np.arange(times.size) % 60 != 0  # one valid sample every 2 s
    calls = {
        "occupancy": lambda: env.occupancy(times, _with_nan(positions, sparse)),
        "spatial_rate": lambda: compute_spatial_rate(
            env, spikes, times, _with_nan(positions, sparse)
        ),
        "directional_rate": lambda: compute_directional_rate(
            spikes, times, _with_nan(headings, sparse)
        ),
    }
    with pytest.warns(UserWarning, match="lie in gaps across missing samples"):
        calls[analysis]()


def test_animation_samples_a_bridged_gap_between_its_valid_rows(tracking):
    times, positions, _, _, lost = tracking
    kept = ~lost
    rows = _with_nan(positions, lost)
    valid, intervals = observed_support(times, rows, max_gap=1.0)
    query = np.linspace(times[0], times[-1], 997)
    sampled = sample_at_times(
        times,
        rows,
        query,
        interp="linear",
        sample_valid=valid,
        interval_valid=intervals,
    )
    expected = np.column_stack(
        [np.interp(query, times[kept], positions[kept, d]) for d in range(2)]
    )
    np.testing.assert_allclose(sampled, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("metric", ["euclidean", "geodesic"])
@pytest.mark.parametrize("lost_rows", [[0], [-1], [0, 1, -2, -1]])
def test_path_efficiency_with_missing_end_rows_equals_deleted_rows(
    env, tracking, metric, lost_rows
):
    """A trial whose first or last rows are lost starts or ends where observed."""
    from neurospatial.behavior import compute_path_efficiency

    times, positions, _, _, _ = tracking
    trial = slice(0, 90)  # 3 s running toward the goal
    times, positions = times[trial], positions[trial]
    lost = np.zeros(times.size, dtype=bool)
    lost[lost_rows] = True
    goal = positions[-1] + np.array([2.0, 0.0])
    missing, deleted = (
        compute_path_efficiency(env, t, p, goal, metric=metric, reference_speed=30.0)
        for t, p in (
            (times, _with_nan(positions, lost)),
            (times[~lost], positions[~lost]),
        )
    )
    assert np.isfinite(deleted.efficiency)
    for name in (
        "traveled_length",
        "shortest_length",
        "efficiency",
        "time_efficiency",
        "angular_efficiency",
    ):
        assert getattr(missing, name) == pytest.approx(
            getattr(deleted, name), rel=1e-12
        ), name
    np.testing.assert_array_equal(missing.start_position, deleted.start_position)


@pytest.mark.parametrize("summary", [False, True])
def test_decoding_skips_lost_tracking_like_deleted_rows(env, tracking, summary):
    """A 10 s tracking loss is not decoded, with NaN rows or deleted rows.

    The decode clock is the tracked clock: spikes are counted where position
    was recorded, as without ``spike_window``.
    """
    from neurospatial.decoding import decode_session, decode_session_summary

    times, positions, _, spikes, lost = tracking
    lost = lost.copy()
    lost[(times >= 20) & (times < 30)] = True
    decode = decode_session_summary if summary else decode_session
    missing, deleted = (
        decode(env, spikes, t, p, dt=0.25, bandwidth=8.0)
        for t, p in (
            (times, _with_nan(positions, lost)),
            (times[~lost], positions[~lost]),
        )
    )
    assert not np.any((deleted.times > 20) & (deleted.times < 30))
    np.testing.assert_array_equal(missing.times, deleted.times)
    if summary:
        np.testing.assert_allclose(missing.map_position, deleted.map_position)
    else:
        np.testing.assert_allclose(missing.posterior, deleted.posterior, atol=1e-12)


def test_break_declared_on_a_missing_row_splits_like_on_the_next_kept_row(env):
    """A break flagged at a NaN row still separates the two recordings.

    Rows 100-102 are lost and the new recording is flagged at row 101. The
    result equals that of the recording with those rows deleted and the break
    on the first kept row, and differs from the same rows without a break.
    """
    from shapely.geometry import box

    from neurospatial.behavior import detect_region_visits
    from neurospatial.regions import Regions

    builder = Regions(env.regions.values())
    builder.add("east", polygon=box(78, 30, 100, 70))
    region_env = env.with_regions(builder)
    times = np.arange(200) / 10
    positions = np.column_stack(
        [50 + 40 * np.sin(2 * np.pi * times / 8), np.full(200, 50.0)]
    )
    lost = np.arange(100, 103)
    with_nan = _with_nan(positions, lost)
    breaks = np.zeros(200, dtype=bool)
    breaks[101] = True
    kept = np.setdiff1d(np.arange(200), lost)
    deleted_breaks = np.zeros(kept.size, dtype=bool)
    deleted_breaks[100] = True  # original row 103, the first kept row after

    def analyses(t, p, break_before):
        visits = detect_region_visits(
            region_env.bin_at(p),
            t,
            region_env,
            region_names=["east"],
            break_before=break_before,
        )
        return (
            region_env.occupancy(t, p, break_before=break_before),
            region_env.transitions(
                times=t, positions=p, break_before=break_before, normalize=False
            ).toarray(),
            [(v.start_time, v.end_time, v.run_id) for v in visits],
        )

    flagged = analyses(times, with_nan, breaks)
    deleted = analyses(times[kept], positions[kept], deleted_breaks)
    unbroken = analyses(times, with_nan, None)
    np.testing.assert_allclose(flagged[0], deleted[0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(flagged[1], deleted[1])
    assert flagged[2] == deleted[2]
    # The animal is in "east" across the lost rows (t = 10 s), so the break
    # removes the 0.4 s gap's time and transition and splits that visit.
    assert flagged[0].sum() == pytest.approx(unbroken[0].sum() - 0.4)
    assert flagged[1].sum() == unbroken[1].sum() - 1
    assert len(flagged[2]) == len(unbroken[2]) + 1
