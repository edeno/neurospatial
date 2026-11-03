from __future__ import annotations

import re
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


def _still(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """``n`` frames at 10 Hz moving 1 cm a frame along x (10 cm/s).

    A sub-pixel jitter of under 0.01 cm keeps the positions off a grid.
    """
    times = np.arange(n) / 10.0
    jitter = np.random.default_rng(0).uniform(0.0, 0.01, (n, 2))
    return times, np.column_stack([np.arange(n, dtype=float), np.zeros(n)]) + jitter


def _lost(masked: np.ndarray) -> list[int]:
    return np.flatnonzero(~np.isfinite(masked).all(axis=1)).tolist()


def test_mask_lost_tracking_removes_held_runs():
    from _record import mask_lost_tracking

    times, positions = _still()
    # Held for 12 frames after frame 19 (the animal walks on meanwhile, so
    # the next fix is where it would be); a 3-frame repeat is kept.
    positions[20:32] = positions[19]
    positions[45:48] = positions[44]
    positions[48:] -= 3.0
    masked, line = mask_lost_tracking(times, positions, min_frozen=10, max_speed=100.0)
    assert _lost(masked) == list(range(20, 32))
    assert np.isfinite(positions).all()  # the input is not changed
    assert "12 samples" in line
    assert "1 runs" in line


def test_mask_lost_tracking_refuses_quantized_positions():
    from _record import mask_lost_tracking, position_grid

    times, positions = _still()
    assert not position_grid(positions)[0]
    pixels = np.round(positions * 3.0) / 3.0  # a 1/3 cm grid
    quantized, line = position_grid(pixels)
    assert quantized
    assert "positions quantized" in line
    with pytest.raises(ValueError, match="assumes sub-pixel positions"):
        mask_lost_tracking(times, pixels, min_frozen=10, max_speed=100.0)


def test_mask_lost_tracking_checks_the_step_after_a_held_run():
    from _record import mask_lost_tracking

    times, positions = _still()
    # Held frames 20-31, then frame 32 jumps 200 cm: from the held run's
    # kept first sample (frame 19, 1.3 s earlier) that is 13 + 200 cm in
    # 1.3 s, about 164 cm/s > 100. Frame 33 is 14 cm from frame 19 in 1.4 s.
    positions[20:32] = positions[19]
    positions[32, 1] += 200.0
    masked, line = mask_lost_tracking(times, positions, min_frozen=10, max_speed=100.0)
    assert _lost(masked) == [*range(20, 32), 32]
    assert "1 samples reached faster than 100 cm/s" in line


def test_mask_lost_tracking_removes_only_a_one_frame_glitch():
    from _record import mask_lost_tracking

    times, positions = _still()
    positions[30, 1] += 50.0  # 50 cm in 0.1 s; frame 31 is 2 cm from 29 in 0.2 s
    masked, _ = mask_lost_tracking(times, positions, min_frozen=10, max_speed=100.0)
    assert _lost(masked) == [30]


def test_mask_lost_tracking_removes_a_whole_excursion_and_keeps_the_return():
    from _record import mask_lost_tracking

    times, positions = _still()
    # Frames 30-34 sit 80 cm off the path; from frame 29 the farthest
    # (frame 34) is within 80.3 cm of it after 0.5 s, 161 cm/s > 100.
    positions[30:35, 1] += 80.0
    masked, line = mask_lost_tracking(times, positions, min_frozen=10, max_speed=100.0)
    assert _lost(masked) == [30, 31, 32, 33, 34]
    assert "5 samples reached faster" in line


def test_mask_lost_tracking_skips_input_nan_without_moving_the_anchor():
    from _record import mask_lost_tracking

    times, positions = _still()
    positions[30] = np.nan
    # Frame 31 jumps 50 cm: from frame 29 (the anchor; frame 30 is NaN) that
    # is 50 cm in 0.2 s, 250 cm/s. Frame 32 is 3 cm from frame 29 in 0.3 s.
    positions[31, 1] += 50.0
    masked, line = mask_lost_tracking(times, positions, min_frozen=10, max_speed=100.0)
    assert _lost(masked) == [30, 31]
    assert "1 samples reached faster" in line


@pytest.mark.parametrize(
    ("start", "stop", "length"),
    [(1.0, 1.3, 0.1), (0.0, 0.7, 0.1), (7674.0, 9941.0, 60.0), (0.0, 1.0, 0.5)],
)
def test_alternating_blocks_tile_the_span_without_slivers(start, stop, length):
    from _record import alternating_blocks

    train, test = alternating_blocks(start, stop, length)
    blocks = np.vstack([train, test])
    blocks = blocks[np.argsort(blocks[:, 0])]
    widths = np.diff(blocks, axis=1).ravel()
    assert blocks[0, 0] == start
    assert blocks[-1, 1] == stop
    np.testing.assert_array_equal(blocks[1:, 0], blocks[:-1, 1])
    # Every block is a full one but the last, which is above half a block
    # or longer than a block by less than one.
    assert np.all(widths > 0)
    np.testing.assert_allclose(widths[:-1], length, rtol=1e-9)
    assert widths[-1] > 1e-6 * length


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"epochs": {"bad": np.array([[0.0, np.nan]])}}, "epochs['bad']"),
        ({"events": {"pokes": np.array([1.0, np.nan])}}, "events['pokes']"),
        ({"events": {"runs": np.array([[2.0, 1.0]])}}, "events['runs']"),
        ({"events": {"cube": np.zeros((2, 2, 2))}}, "events['cube']"),
    ],
)
def test_validation_checks_epochs_and_events(simulated_recording, change, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        validate_recording(replace(simulated_recording, **change))


@pytest.mark.parametrize("name", ["train_epochs", "test_epochs", "analysis_epochs"])
@pytest.mark.parametrize(
    "table", [np.array([[0.0, np.nan]]), np.array([[5.0, 1.0]]), np.zeros(3)]
)
def test_validation_checks_epoch_params(simulated_recording, name, table):
    params = {**simulated_recording.params, name: table}
    with pytest.raises(ValueError, match=re.escape(f"params[{name!r}]")):
        validate_recording(replace(simulated_recording, params=params))


@pytest.mark.parametrize(
    ("train", "test", "problem"),
    [
        ([[10, 20], [0, 5]], [[30, 40]], "params['train_epochs'] must be sorted"),
        ([[0, 10], [5, 20]], [[30, 40]], "params['train_epochs'] must be sorted"),
        (
            [[0, 10]],
            [[5, 20]],
            "params['train_epochs'] and params['test_epochs'] overlap",
        ),
        ("even laps", [[5, 20]], "params['train_epochs'] must be finite"),
    ],
)
def test_validation_checks_train_and_test_are_held_out(
    simulated_recording, train, test, problem
):
    params = {
        **simulated_recording.params,
        "train_epochs": train if isinstance(train, str) else np.array(train, float),
        "test_epochs": np.array(test, dtype=float),
    }
    with pytest.raises(ValueError, match=re.escape(problem)):
        validate_recording(replace(simulated_recording, params=params))
