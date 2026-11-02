from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from _calibration import captured, hazards_for, probe_hazards
from operations import Operation, Outcome
from run import render_warnings


def test_captured_groups_by_template():
    def noisy():
        warnings.warn("3 bins were dropped", UserWarning, stacklevel=2)
        warnings.warn("5 bins were dropped", UserWarning, stacklevel=2)
        msg = "then it failed"
        raise ValueError(msg)

    value, error, rows = captured(noisy)
    assert value is None
    assert isinstance(error, ValueError)
    assert rows == [
        {
            "category": "UserWarning",
            "origin": "neurospatial",
            "template": "# bins were dropped",
            "example": "3 bins were dropped",
            "count": 2,
        }
    ]


def _probe(name, run):
    return Operation(name, ("positions",), run)


def test_hazard_responses_classified(simulated_recording, loader_module):
    def raises(rec, cache):
        if rec.positions.max() < 2:  # the length-unit hazard divides cm by 100
            msg = "arena is 1 cm wide"
            raise ValueError(msg)
        return Outcome("ok", {"n": 1})

    def warns(rec, cache):
        warnings.warn("suspicious positions", UserWarning, stacklevel=2)
        return Outcome("ok", {"n": 2})

    def silent(rec, cache):
        return Outcome("ok", {"max": float(rec.positions.max())})

    operations = (
        _probe("p.raise", raises),
        _probe("p.warn", warns),
        _probe("p.silent", silent),
    )
    clean = {"p.silent": {"summary": {"max": 100.0}}}
    rows = probe_hazards(
        loader_module,
        simulated_recording,
        clean,
        operations=operations,
        probe_ops=("p.raise", "p.warn", "p.silent"),
    )
    by_op = {
        row["op"]: row for row in rows if row["hazard"] == "length_unit_off_by_100"
    }
    assert by_op["p.raise"]["response"] == "raised"
    assert by_op["p.raise"]["message"] == "ValueError: arena is 1 cm wide"
    assert by_op["p.warn"]["response"] == "warned"
    assert by_op["p.warn"]["message"] == "suspicious positions"
    assert by_op["p.silent"]["response"] == "silent"
    assert by_op["p.silent"]["clean_summary"] == {"max": 100.0}
    assert by_op["p.silent"]["hazard_summary"]["max"] < 1.01


def test_generic_hazards_apply_only_when_relevant(simulated_recording):
    no_extras = SimpleNamespace()
    names = set(hazards_for(no_extras, simulated_recording))
    assert names == {"spike_times_in_ms", "length_unit_off_by_100"}

    positions = simulated_recording.positions.copy()
    positions[10:20] = np.nan
    with_gaps = replace(
        simulated_recording, positions=positions, headings=np.zeros(positions.shape[0])
    )
    own = SimpleNamespace(HAZARDS={"sentinel_left_in": lambda rec: rec})
    names = set(hazards_for(own, with_gaps))
    assert names == {
        "headings_in_degrees",
        "missing_positions_as_zero",
        "spike_times_in_ms",
        "length_unit_off_by_100",
        "sentinel_left_in",
    }


def test_unreviewed_rows_rendered():
    warning = {"category": "UserWarning", "origin": "neurospatial", "count": 1}
    record = {
        "dataset": "demo",
        "load_warnings": [],
        "operations": {
            "rates.spatial": {
                "warnings": [
                    {
                        **warning,
                        "template": "# spikes dropped",
                        "example": "4 spikes dropped",
                    },
                    {
                        **warning,
                        "template": "bandwidth small",
                        "example": "bandwidth small",
                    },
                ]
            }
        },
        "hazards": [
            {
                "hazard": "spike_times_in_ms",
                "op": "rates.spatial",
                "response": "silent",
                "message": "",
                "clean_summary": {},
                "hazard_summary": {},
            }
        ],
    }
    verdicts = {
        "UserWarning: # spikes dropped": {
            "verdict": "warranted",
            "gap": None,
            "reason": "real",
        }
    }
    text = render_warnings([record], verdicts)
    lines = text.splitlines()
    assert any("4 spikes dropped" in line and "| warranted |" in line for line in lines)
    assert any("bandwidth small" in line and "| unreviewed |" in line for line in lines)
    assert any(
        line.startswith("| demo | spike_times_in_ms") and "unreviewed" in line
        for line in lines
    )
