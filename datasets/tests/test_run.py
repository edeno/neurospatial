from __future__ import annotations

import json

import pytest
from operations import OPERATIONS, Operation
from run import render_matrix, run_recording


@pytest.fixture(scope="session")
def simulated_record(simulated_recording, tmp_path_factory):
    from types import SimpleNamespace

    module = SimpleNamespace(EXPECTATIONS={"n_units": 50}, HAZARDS={})
    return run_recording(
        simulated_recording, module, results=tmp_path_factory.mktemp("results")
    )


def test_driver_statuses_on_simulated_recording(simulated_record):
    statuses = {
        op: entry["status"] for op, entry in simulated_record["operations"].items()
    }
    assert statuses.pop("hd.tuning") == "n/a"
    assert simulated_record["operations"]["hd.tuning"]["note"] == "needs headings"
    # An outline inferred from samples has no declared walls, so the physical
    # border score refuses it; the operation records that as an error.
    border = simulated_record["operations"]["cells.border_score"]
    assert statuses.pop("cells.border_score") == "error"
    assert border["note"].startswith(
        "NotImplementedError: Physical boundaries are unknown"
    )
    # Operations that need inputs this recording lacks decide n/a themselves.
    for op_id in (
        "env.from_graph",
        "behavior.segment_by_velocity",
        "behavior.turn_sequence",
        "replay.compare_methods",
        "io.nwb_readers",
        "io.read_pose",
        "imaging.rates",
        "imaging.rates_workaround",
    ):
        assert statuses.pop(op_id) == "n/a", op_id
        assert simulated_record["operations"][op_id]["note"]
    assert set(statuses.values()) == {"ok"}, statuses
    assert simulated_record["expectations"][0]["passed"]


def test_failing_operation_is_recorded_as_error(
    simulated_recording, loader_module, tmp_path
):
    def broken(rec, cache):
        msg = "bins disagree"
        raise RuntimeError(msg)

    operations = (OPERATIONS[0], Operation("broken.op", ("positions",), broken))
    record = run_recording(
        simulated_recording, loader_module, operations=operations, results=tmp_path
    )
    entry = record["operations"]["broken.op"]
    assert entry["status"] == "error"
    assert entry["note"] == "RuntimeError: bins disagree"
    assert (
        json.loads((tmp_path / "simulated_open_field.json").read_text())["operations"][
            "broken.op"
        ]["status"]
        == "error"
    )


def test_simulated_place_cells_recovered(simulated_record):
    # Ground truth is known: 50 place cells, so information and decoding must work.
    operations = simulated_record["operations"]
    assert (
        operations["rates.spatial_information"]["summary"]["bits_per_spike_median"]
        > 0.5
    )
    assert operations["decode.position"]["summary"]["median_error"] < 10.0
    shifts = operations["rates.information_vs_shifts"]["summary"]
    assert shifts["frac_above_every_shift"] > 0.8
    assert shifts["median_shifted_bits_per_spike"] < shifts["median_bits_per_spike"] / 2


def test_matrix_render(simulated_record):
    other = {
        "dataset": "other",
        "expectations": [{"passed": False}],
        "operations": {
            "env.from_samples": {
                "status": "workaround",
                "gap": "G-003",
                "warnings": [{}, {}],
            },
            "hd.tuning": {"status": "n/a"},
        },
    }
    text = render_matrix([simulated_record, other])
    rows = [line for line in text.splitlines() if line.startswith("| other")]
    header = next(line for line in text.splitlines() if line.startswith("| dataset"))
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    assert cells[columns.index("expectations")] == "0/1"
    assert cells[columns.index("env.from_samples")] == "wk G-003 ⚠2"
    assert cells[columns.index("hd.tuning")] == "–"
    assert "| simulated_open_field | 1/1 | ok" in text


def test_error_cell_shows_its_gap(simulated_record):
    text = render_matrix(
        [simulated_record],
        {"simulated_open_field/cells.border_score": {"gap": "G-005"}},
    )
    assert "err G-005" in text


def test_subset_run_keeps_other_operations(
    simulated_recording, loader_module, tmp_path
):
    full = run_recording(
        simulated_recording, loader_module, operations=OPERATIONS[:2], results=tmp_path
    )
    assert set(full["operations"]) == {"env.from_samples", "env.occupancy"}
    partial = run_recording(
        simulated_recording,
        loader_module,
        op_ids=["env.occupancy"],
        operations=OPERATIONS[:2],
        results=tmp_path,
    )
    assert set(partial["operations"]) == {"env.from_samples", "env.occupancy"}
    assert (
        partial["operations"]["env.from_samples"]
        == full["operations"]["env.from_samples"]
    )
