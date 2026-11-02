from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _record import check_expectations, validate_recording


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
        {"n_units": 7, "sampling_rate_hz": (1000.0, 2000.0), "n_dims": 2},
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
            "position_min": ((-1.0, -1.0), (5.0, 5.0)),
            "position_max": ((95.0, 200.0), (101.0, 201.0)),
        },
    )
    assert [row["passed"] for row in rows] == [True, False]
