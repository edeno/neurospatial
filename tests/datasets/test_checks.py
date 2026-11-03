"""The automatic checks: clock and units refuse, the rest describe."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest

from neurospatial import InputError
from neurospatial.datasets import DatasetCheckError
from neurospatial.datasets._checks import Stated, run_checks
from neurospatial.datasets._summary import summary_text


def _shift_spikes(rec, stated):
    return {"spike_times": tuple(train + 1000.0 for train in rec.spike_times)}, stated


def _spikes_in_ms(rec, stated):
    return {"spike_times": tuple(train * 1000.0 for train in rec.spike_times)}, stated


def _event_off_clock(rec, stated):
    return {"events": {"reward": np.array([10.0, 20.0, 20_000.0])}}, stated


def _overlapping_epochs(rec, stated):
    return {"epochs": {"run": np.array([[0.0, 40.0], [30.0, 60.0]])}}, stated


def _epochs_off_clock(rec, stated):
    # A second clock, with another zero.
    return {"epochs": {"run": np.array([[0.0, 60.0]]) + 100_000.0}}, stated


def _positions_in_mm(rec, stated):
    return {"positions": rec.positions * 10.0}, stated


def _wrong_rate(rec, stated):
    return {}, dataclasses.replace(stated, sampling_rate=50.0)


REFUSALS = {
    "spikes shifted": ("clock", _shift_spikes, "spikes (100.00%) lie outside"),
    "spikes in ms": (
        "clock",
        _spikes_in_ms,
        "lie outside the tracked span 0 to 59.99 s",
    ),
    "event off the clock": (
        "clock",
        _event_off_clock,
        "events['reward'] has 1 event of 3 outside",
    ),
    "epochs overlap": (
        "clock",
        _overlapping_epochs,
        "epochs['run'] has 1 row starting before",
    ),
    "epochs off the clock": ("clock", _epochs_off_clock, "no epoch holds a tracking"),
    "positions in mm": ("units", _positions_in_mm, "largest position span is 999.1 cm"),
    "sampling rate": ("units", _wrong_rate, "not the stated 50 Hz"),
}


def _origin_zeros(rec, stated):
    positions = np.array(rec.positions)
    positions[::100] = 0.0  # 1% of samples, never two in a row
    return (
        {"positions": positions},
        stated,
        "60 position samples (1.00%) are exactly at (0, 0)",
    )


def _held_run(rec, stated):
    positions = np.array(rec.positions)
    positions[1000:1030] = positions[1000]
    return {"positions": positions}, stated, "1 run of 11 or more identical"


def _fast_step(rec, stated):
    positions = np.array(rec.positions)
    positions[2000] += 50.0
    limited = dataclasses.replace(
        stated, speed_limit=100.0, speed_limit_source="a stated 1 m/s limit"
    )
    return {"positions": positions}, limited, "1 sample farther"


def _silent_unit(rec, stated):
    trains = list(rec.spike_times)
    trains[2] = np.empty(0)
    return {"spike_times": tuple(trains)}, stated, "1 of 6 units has no spikes"


WARNINGS = {
    "origin zeros": _origin_zeros,
    "held run": _held_run,
    "fast step": _fast_step,
    "silent unit": _silent_unit,
}


@pytest.mark.parametrize(
    "name", ["square_recording", "track_recording", "heading_recording"]
)
def test_synthetic_sessions_pass(name, request):
    rec = request.getfixturevalue(name)
    stated = request.getfixturevalue(name.replace("recording", "stated"))
    checked = run_checks(rec, stated=stated)
    assert checked.failed_checks == ()
    if name == "track_recording":
        # The simulated animal holds still for 0.5 s at each of three arm ends.
        (held,) = checked.watch_outs
        assert held.startswith("3 runs of 11 or more identical position samples")
    else:
        assert checked.watch_outs == ()


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_each_refusing_check_fails_alone(square_recording, square_stated, case):
    check, make, numbers = REFUSALS[case]
    fields, stated = make(square_recording, square_stated)
    rec = dataclasses.replace(square_recording, **fields)
    with pytest.raises(DatasetCheckError) as caught:
        run_checks(rec, stated=stated)
    error = caught.value
    assert error.checks == (check,)
    message = str(error)
    assert f"failed the '{check}' check" in message
    assert f"- {check}: " in message
    assert numbers in message
    assert f"accept_failed_checks=('{check}',)" in message


@pytest.mark.parametrize("case", sorted(WARNINGS))
def test_each_warning_check_fails_alone(square_recording, square_stated, case):
    fields, stated, numbers = WARNINGS[case](square_recording, square_stated)
    rec = dataclasses.replace(square_recording, **fields)
    checked = run_checks(rec, stated=stated)
    (finding,) = checked.watch_outs
    assert numbers in finding
    assert checked.failed_checks == ()
    assert np.array_equal(checked.times, rec.times)
    assert np.array_equal(checked.positions, rec.positions, equal_nan=True)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(checked.spike_times, rec.spike_times, strict=True)
    )


def test_both_refusing_checks_reported_together(square_recording, square_stated):
    rec = dataclasses.replace(
        square_recording,
        positions=square_recording.positions * 10.0,
        spike_times=tuple(t + 1000.0 for t in square_recording.spike_times),
    )
    with pytest.raises(DatasetCheckError) as caught:
        run_checks(rec, stated=square_stated)
    assert caught.value.checks == ("clock", "units")
    assert "accept_failed_checks=('clock', 'units')" in str(caught.value)
    # Accepting one still refuses the other, and names only it.
    with pytest.raises(DatasetCheckError) as caught:
        run_checks(rec, stated=square_stated, accept_failed_checks=("clock",))
    assert caught.value.checks == ("units",)


def test_accepted_failure_travels(square_recording, square_stated):
    loader_note = "the dataset's own pitfall"
    rec = dataclasses.replace(
        square_recording,
        spike_times=tuple(t + 1000.0 for t in square_recording.spike_times),
        watch_outs=(loader_note,),
    )
    checked = run_checks(rec, stated=square_stated, accept_failed_checks=("clock",))
    assert checked.failed_checks == ("clock",)
    assert checked.watch_outs[0].startswith("failed the clock check, loaded with")
    assert checked.watch_outs[1] == loader_note
    watch_out = summary_text(checked).split("\nWatch out", 1)[1].splitlines()
    assert watch_out[1].startswith("  ! failed the clock check")
    # Accepting a check that did not fail changes nothing.
    plain = run_checks(square_recording, stated=square_stated)
    accepted = run_checks(
        square_recording, stated=square_stated, accept_failed_checks=("clock", "units")
    )
    assert accepted.failed_checks == plain.failed_checks == ()
    assert accepted.watch_outs == plain.watch_outs


def test_untracked_sleep_passes(square_recording, square_stated):
    """Spikes in a sleep epoch with no tracking samples are on the clock."""
    sleep = np.array([[60.0, 600.0]])
    rng = np.random.default_rng(1)
    trains = tuple(
        np.sort(np.concatenate([train, rng.uniform(60.0, 600.0, 50)]))
        for train in square_recording.spike_times
    )
    rec = dataclasses.replace(
        square_recording,
        spike_times=trains,
        epochs={"run": np.array([[0.0, 60.0]]), "sleep": sleep},
    )
    assert run_checks(rec, stated=square_stated).failed_checks == ()
    # A sleep that starts at the last tracking sample is untracked too.
    last = float(square_recording.times[-1])
    touching = dataclasses.replace(
        rec,
        epochs={"run": np.array([[0.0, last]]), "sleep": np.array([[last, 600.0]])},
    )
    assert run_checks(touching, stated=square_stated).failed_checks == ()
    # The same spikes without the sleep epoch are off the tracked span.
    no_sleep = dataclasses.replace(rec, epochs={"run": np.array([[0.0, 60.0]])})
    with pytest.raises(DatasetCheckError, match="clock"):
        run_checks(no_sleep, stated=square_stated)


def test_spikes_just_past_tracking_pass(square_recording, square_stated):
    """Ephys running a little past the camera is within the clock's slack."""
    trains = list(square_recording.spike_times)
    trains[0] = np.append(trains[0], 60.4)  # 0.41 s past the last sample
    rec = dataclasses.replace(square_recording, spike_times=tuple(trains))
    assert run_checks(rec, stated=square_stated).failed_checks == ()


@pytest.mark.parametrize(("every", "warns"), [(100, True), (2000, False)])
def test_origin_zeros_warn(square_recording, square_stated, every, warns):
    """1% of samples at (0, 0) is reported; 0.05% is not. Nothing is changed."""
    positions = np.array(square_recording.positions)
    positions[::every] = 0.0
    rec = dataclasses.replace(square_recording, positions=positions)
    checked = run_checks(rec, stated=square_stated)
    assert np.array_equal(checked.positions, positions)
    zeros = [w for w in checked.watch_outs if "exactly at (0, 0)" in w]
    if warns:
        (finding,) = zeros
        assert finding.startswith("60 position samples (1.00%)")
    else:
        assert zeros == []


@pytest.mark.parametrize(("identical", "runs"), [(10, 0), (11, 1)])
def test_held_run_is_ten_repeats(square_recording, square_stated, identical, runs):
    """A held run is 10 repeats of the sample before: 11 identical samples."""
    positions = np.array(square_recording.positions)
    positions[1000 : 1000 + identical] = positions[1000]
    rec = dataclasses.replace(square_recording, positions=positions)
    found = run_checks(rec, stated=square_stated).watch_outs
    held = [w for w in found if "identical position samples" in w]
    if runs:
        (finding,) = held
        assert finding.startswith("1 run of 11 or more identical position samples")
        assert "10 repeats, " in finding
    else:
        assert held == []


def test_fast_steps_need_a_stated_limit(square_recording, square_stated):
    positions = np.array(square_recording.positions)
    positions[2000] += 50.0  # one glitched frame, then back on the path
    rec = dataclasses.replace(square_recording, positions=positions)
    # No stated limit: a glitch alone gives no finding.
    assert run_checks(rec, stated=square_stated).watch_outs == ()
    # With one, the glitch counts once, not its return to the path too.
    limited = dataclasses.replace(
        square_stated, speed_limit=100.0, speed_limit_source="a stated 1 m/s limit"
    )
    (finding,) = run_checks(rec, stated=limited).watch_outs
    assert finding.startswith("1 sample farther")
    assert "identical position samples" not in finding
    assert "(a stated 1 m/s limit)" in finding


def test_held_run_says_no_speed_limit(track_recording, track_stated):
    (finding,) = run_checks(track_recording, stated=track_stated).watch_outs
    assert "No speed limit is stated for this dataset" in finding


def test_accept_failed_checks_names_known_checks(square_recording, square_stated):
    for bad in [("clok",), "clock", ("missing_is_zero",)]:
        with pytest.raises(InputError, match="accept_failed_checks"):
            run_checks(square_recording, stated=square_stated, accept_failed_checks=bad)


def test_positions_need_a_stated_span(square_recording):
    with pytest.raises(InputError, match="position_span"):
        run_checks(square_recording, stated=Stated(sampling_rate=100.0))


def test_stated_speed_limit_needs_a_source():
    with pytest.raises(InputError, match="speed_limit_source"):
        Stated(sampling_rate=100.0, speed_limit=100.0)


def test_check_error_pickles():
    error = DatasetCheckError("demo_2026", "s1", {"clock": "3 spikes outside"})
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is DatasetCheckError
    assert str(restored) == str(error)
    assert restored.checks == ("clock",)
    assert isinstance(restored, ValueError)


@pytest.mark.parametrize("events", [np.array([]), np.empty((0, 2))])
def test_empty_events_pass(square_recording, square_stated, events):
    rec = dataclasses.replace(square_recording, events={"reward": events})
    assert run_checks(rec, stated=square_stated).failed_checks == ()


@pytest.mark.parametrize(("overrun", "refuses"), [(11.0, True), (6.0, False)])
def test_clock_slack_is_capped(square_recording, square_stated, overrun, refuses):
    """On a 20 min session 1% is 12 s; the 10 s cap still refuses 11 s."""
    times = square_recording.times * 20.0
    trains = [train * 20.0 for train in square_recording.spike_times]
    trains[0] = np.append(trains[0], times[-1] + overrun)
    rec = dataclasses.replace(
        square_recording,
        times=times,
        spike_times=tuple(trains),
        epochs={"run": np.array([[0.0, 1200.0]])},
    )
    stated = dataclasses.replace(square_stated, sampling_rate=5.0)
    if refuses:
        with pytest.raises(DatasetCheckError, match="give or take 10 s"):
            run_checks(rec, stated=stated)
    else:
        assert run_checks(rec, stated=stated).failed_checks == ()


def test_overlaps_count_every_row_inside_a_long_one(square_recording, square_stated):
    rec = dataclasses.replace(
        square_recording,
        epochs={"run": np.array([[0.0, 50.0], [1.0, 2.0], [3.0, 4.0]])},
    )
    with pytest.raises(DatasetCheckError, match="has 2 rows starting before"):
        run_checks(rec, stated=square_stated)


def test_origin_zeros_skip_one_dimensional_tracks(square_recording, square_stated):
    """On a 1-D track, 0 is a place on the track, not lost tracking."""
    from neurospatial import Environment

    positions = np.array(square_recording.positions[:, :1])
    positions[::50] = 0.0  # 2% of samples at the track's start
    env = Environment.from_samples(positions, bin_size=2.0, units="cm")
    rec = dataclasses.replace(
        square_recording, positions=positions, env=env, headings=None
    )
    assert not any(
        "exactly at" in w for w in run_checks(rec, stated=square_stated).watch_outs
    )


def test_checking_twice_refuses(
    square_recording, square_stated, track_recording, track_stated
):
    shifted = dataclasses.replace(
        square_recording,
        spike_times=tuple(t + 1000.0 for t in square_recording.spike_times),
    )
    accepted = run_checks(
        shifted, stated=square_stated, accept_failed_checks=("clock",)
    )
    with pytest.raises(InputError, match="checked already"):
        run_checks(accepted, stated=square_stated, accept_failed_checks=("clock",))
    held = run_checks(track_recording, stated=track_stated)
    with pytest.raises(InputError, match="checked already"):
        run_checks(held, stated=track_stated)
