from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _record import (
    Range,
    check_expectations,
    heading_vs_movement,
    validate_recording,
)


def test_valid_simulated_recording_passes(simulated_recording):
    validate_recording(simulated_recording)


def test_validation_lists_every_problem(simulated_recording):
    rec = simulated_recording
    broken = replace(
        rec,
        times=rec.times[::-1].copy(),
        headings=np.degrees(np.linspace(-np.pi, np.pi, rec.times.size)),
        spike_times=(rec.spike_times[0][::-1].copy(), *rec.spike_times[1:]),
    )
    with pytest.raises(ValueError, match="breaks the Recording contract") as info:
        validate_recording(broken)
    message = str(info.value)
    assert "strictly increasing" in message
    assert "radians in [-pi, pi]" in message
    assert "spike train 0 must be finite and sorted" in message


def test_headings_without_positions_are_valid(simulated_recording):
    rec = simulated_recording
    headings_only = replace(
        rec, positions=None, position_units=None, headings=np.zeros(rec.times.size)
    )
    validate_recording(headings_only)
    with pytest.raises(ValueError, match="times must be set exactly"):
        validate_recording(replace(headings_only, headings=None))


def test_expectations_report_failures_without_raising(simulated_recording):
    rows = check_expectations(
        simulated_recording,
        {"n_units": 7, "sampling_rate_hz": Range(1000.0, 2000.0), "n_dims": 2},
    )
    by_key = {row["key"]: row for row in rows}
    assert not by_key["n_units"]["passed"]
    assert "got 50" in by_key["n_units"]["detail"]
    assert not by_key["sampling_rate_hz"]["passed"]
    assert by_key["n_dims"]["passed"]
    with pytest.raises(KeyError, match="Unknown expectation 'n_neurons'"):
        check_expectations(simulated_recording, {"n_neurons": 50})


def test_range_expectation_is_elementwise(simulated_recording):
    rows = check_expectations(
        simulated_recording,
        {
            "position_min": Range((-1.0, -1.0), (5.0, 5.0)),
            "position_max": Range((95.0, 200.0), (101.0, 201.0)),
        },
    )
    assert [row["passed"] for row in rows] == [True, False]


def test_two_element_values_are_exact_not_ranges(simulated_recording):
    exact = tuple(simulated_recording.positions.min(axis=0))
    rows = check_expectations(
        simulated_recording, {"position_min": exact, "epoch_names": ("sleep", "wake")}
    )
    assert [row["passed"] for row in rows] == [True, False]


def test_inverted_range_is_refused():
    with pytest.raises(ValueError, match="above high"):
        Range(5.0, 1.0)


def test_check_that_raises_fails_without_raising(simulated_recording):
    rows = check_expectations(
        simulated_recording, {"heading_span": lambda rec: rec.headings.min() < -3}
    )
    assert not rows[0]["passed"]
    assert rows[0]["detail"].startswith("check raised AttributeError")


def test_dt_required_when_decoding_is_possible(simulated_recording):
    params = {k: v for k, v in simulated_recording.params.items() if k != "dt"}
    with pytest.raises(ValueError, match=r"params missing \['dt'\]"):
        validate_recording(replace(simulated_recording, params=params))


def test_three_dimensional_recording_validates(simulated_recording):
    rng = np.random.default_rng(3)
    steps = rng.normal(0.0, 0.02, (simulated_recording.times.size, 3))
    positions = np.clip(np.cumsum(steps, axis=0), -2.0, 2.0)
    rec = replace(simulated_recording, positions=positions, position_units="m")
    validate_recording(rec)
    rows = check_expectations(rec, {"n_dims": 3})
    assert rows[0]["passed"]


def _one_step_offset(times, positions, headings, min_speed):
    # The lag-1 rule written out: direction of the step to the next sample.
    step = np.diff(positions, axis=0)
    speed = np.linalg.norm(step, axis=1) / np.diff(times)
    running = (speed > min_speed) & np.isfinite(headings[:-1])
    diff = np.angle(
        np.exp(1j * (headings[:-1] - np.arctan2(step[:, 1], step[:, 0])))[running]
    )
    mean = np.angle(np.mean(np.exp(1j * diff)))
    return mean, np.median(np.abs(np.angle(np.exp(1j * (diff - mean)))))


@pytest.fixture
def pixel_circle():
    """A head running a 60 px circle at 30 Hz, tracked in whole pixels; heading along the path."""
    times = np.arange(0.0, 60.0, 1 / 30)
    angle = 0.8 * times  # 48 px/s, about 1.6 px per frame
    exact = 60.0 * np.column_stack([np.cos(angle), np.sin(angle)])
    headings = np.angle(np.exp(1j * (angle + np.pi / 2)))
    return times, np.round(exact), headings


def test_heading_vs_movement_default_is_the_one_step_rule(pixel_circle):
    times, positions, headings = pixel_circle
    rng = np.random.default_rng(0)
    noisy = np.angle(np.exp(1j * (headings + rng.normal(0, 0.3, headings.size))))
    noisy[::17] = np.nan
    expected = _one_step_offset(times, positions, noisy, min_speed=10.0)
    np.testing.assert_allclose(
        heading_vs_movement(times, positions, noisy, min_speed=10.0), expected
    )
    np.testing.assert_allclose(
        heading_vs_movement(times, positions, noisy, min_speed=10.0, lag=1),
        expected,
    )


def test_heading_vs_movement_lag_removes_pixel_quantisation(pixel_circle):
    times, positions, headings = pixel_circle
    _, one_step = heading_vs_movement(times, positions, headings, min_speed=10.0)
    mean, longer = heading_vs_movement(
        times, positions, headings, min_speed=10.0, lag=10
    )
    # One-pixel steps snap the direction to multiples of 45 deg.
    assert one_step > np.radians(10)
    # Over 10 samples the chord lags the tangent by half the turn (0.13 rad).
    assert abs(mean + 0.8 * 10 / 30 / 2) < 0.03
    assert longer < np.radians(3)


def test_heading_vs_movement_refuses_a_lag_below_one(pixel_circle):
    times, positions, headings = pixel_circle
    with pytest.raises(ValueError, match="positive number of samples"):
        heading_vs_movement(times, positions, headings, min_speed=10.0, lag=0)


def test_mask_lost_tracking_removes_held_runs_and_bridges():
    from _record import mask_lost_tracking

    rng = np.random.default_rng(0)
    times = np.arange(300) / 30.0
    positions = np.cumsum(rng.normal(0, 0.3, (300, 2)), axis=0)
    # Held for 20 frames after frame 99, then a jump of 50 cm in one frame.
    positions[100:120] = positions[99]
    positions[120:] += 50.0
    # A short repeat (3 frames) is kept: below the run length.
    positions[200:203] = positions[199]
    masked, line = mask_lost_tracking(times, positions, min_frozen=10, max_speed=100.0)
    lost = ~np.isfinite(masked).all(axis=1)
    assert np.flatnonzero(lost).tolist() == list(range(100, 120))
    assert np.isfinite(positions).all()  # the input is not changed
    assert "20 samples" in line
    assert "1 runs" in line
    # A one-frame jump between two real fixes removes the later fix.
    jumped = np.cumsum(rng.normal(0, 0.3, (300, 2)), axis=0)
    jumped[150:] += 50.0
    masked, line = mask_lost_tracking(times, jumped, min_frozen=10, max_speed=100.0)
    assert np.flatnonzero(~np.isfinite(masked).all(axis=1)).tolist() == [150]
    assert "1 samples reached faster than 100 cm/s" in line
