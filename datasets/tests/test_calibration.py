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
    # Spike times in ms change no field these position-only operations read.
    spikes = [row for row in rows if row["hazard"] == "spike_times_in_ms"]
    assert [row["response"] for row in spikes] == ["not probed"]


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


def test_warning_already_in_clean_run_is_not_a_response(
    simulated_recording, loader_module
):
    def always_warns(rec, cache):
        warnings.warn("12 spikes dropped", UserWarning, stacklevel=2)
        return Outcome("ok", {})

    clean = {
        "p.same": {
            "summary": {},
            "warnings": [
                {
                    "category": "UserWarning",
                    "template": "# spikes dropped",
                    "origin": "neurospatial",
                }
            ],
        }
    }
    rows = probe_hazards(
        loader_module,
        simulated_recording,
        clean,
        operations=(_probe("p.same", always_warns),),
        probe_ops=("p.same",),
    )
    row = next(r for r in rows if r["hazard"] == "length_unit_off_by_100")
    assert row["response"] == "silent"
    assert row["repeated_clean_warnings"] == 1


def test_hazard_reaching_no_operation_is_reported(simulated_recording):
    def needs_headings(rec, cache):
        return Outcome("ok", {})

    module = SimpleNamespace(
        HAZARDS={"spikes_shifted": lambda rec: replace(rec, spike_times=())}
    )
    rows = probe_hazards(
        module,
        simulated_recording,
        {},
        operations=(Operation("p.heading", ("headings",), needs_headings),),
        probe_ops=("p.heading",),
    )
    row = next(r for r in rows if r["hazard"] == "spikes_shifted")
    assert row["response"] == "not probed"
    assert "spike_times" in row["message"]


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
        "warnings": {
            "neurospatial UserWarning: # spikes dropped": {
                "verdict": "warranted",
                "gap": None,
                "reason": "real",
            }
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


def test_operation_failing_on_clean_data_is_not_probed(
    simulated_recording, loader_module
):
    def fails(rec, cache):
        msg = "needs speed="
        raise ValueError(msg)

    rows = probe_hazards(
        loader_module,
        simulated_recording,
        {"p.fails": {"status": "error", "summary": {}}},
        operations=(_probe("p.fails", fails),),
        probe_ops=("p.fails",),
    )
    assert {row["response"] for row in rows} == {"not probed"}
    assert {row["message"] for row in rows if row["op"]} == {"the clean run was error"}
