from __future__ import annotations

from studies.declared_vs_inferred import cell, rows_for


def test_cell_shows_values_and_statuses():
    ok = {"status": "ok", "summary": {"n_laps": 3, "median_error": 12.345}}
    assert cell(ok, "n_laps", "d") == "3"
    assert cell(ok, "median_error", ".1f") == "12.3"
    assert cell(ok, "absent", "d") == ""
    assert cell({"status": "error", "summary": {}}, "n_laps", "d") == "error"
    assert cell({"status": "n/a"}, "n_laps", "d") == "–"
    workaround = {"status": "workaround", "summary": {"davidson_2009_rate": 0.25}}
    assert cell(workaround, "davidson_2009_rate", ".3f") == "0.250 (workaround)"
    assert cell(None, "n_laps", "d") == ""


def test_rows_pair_the_two_runs():
    declared = {
        "operations": {
            "env.declared": {"status": "ok", "summary": {"n_bins": 45}},
            "behavior.laps": {
                "status": "ok",
                "summary": {"n_laps": 1, "n_laps_region": 20},
            },
        }
    }
    inferred = {
        "operations": {
            "env.from_samples": {"status": "ok", "summary": {"n_bins": 667}},
            "behavior.laps": {"status": "error", "summary": {}},
        }
    }
    rows = rows_for(declared, inferred)
    assert rows[0] == ("bins", "45", "667")
    assert ("laps (default)", "1", "error") in rows
    assert ("laps (region)", "20", "error") in rows
