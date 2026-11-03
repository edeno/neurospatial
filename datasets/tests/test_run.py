from __future__ import annotations

import json

import numpy as np
import pytest
from operations import OPERATIONS, Operation
from run import load_fingerprint, render_matrix, run_recording


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
    for op_id in ("hd.tuning", "rates.egocentric_boundary"):
        assert statuses.pop(op_id) == "n/a"
        assert simulated_record["operations"][op_id]["note"] == "needs headings"
    # An outline inferred from samples has no declared walls, so the physical
    # border score refuses it; the operation records that as an error.
    border = simulated_record["operations"]["cells.border_score"]
    assert statuses.pop("cells.border_score") == "error"
    assert border["note"].startswith(
        "NotImplementedError: Physical boundaries are unknown"
    )
    # Operations that need inputs this recording lacks decide n/a themselves.
    for op_id in (
        "env.declared",
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
    assert "| simulated_open_field | 1/1 | inferred | ok" in text


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


def _warning_op(rec, cache):
    import warnings

    from operations import Outcome

    warnings.warn("bins without support: 3", UserWarning, stacklevel=1)
    return Outcome("ok")


def _probe_setup():
    import dataclasses
    from types import SimpleNamespace

    from operations import Outcome

    # A probed operation that warns on clean and hazardous data alike: its
    # hazard response is silent, because the warning is in the clean run.
    ops = (
        Operation("env.from_samples", ("positions",), _warning_op),
        Operation("other.op", ("positions",), lambda r, c: Outcome("ok")),
    )
    module = SimpleNamespace(
        EXPECTATIONS={},
        HAZARDS={
            "copied_positions": lambda r: dataclasses.replace(
                r, positions=r.positions.copy()
            )
        },
    )
    return ops, module


def test_subset_run_probes_only_against_its_fresh_clean_results(
    simulated_recording, tmp_path
):
    ops, module = _probe_setup()
    full = run_recording(simulated_recording, module, operations=ops, results=tmp_path)
    rows = [(h["hazard"], h["op"], h["response"]) for h in full["hazards"]]
    assert ("copied_positions", "env.from_samples", "silent") in rows
    # A subset run that does not rerun the probed operation keeps its row;
    # one with no earlier results probes nothing it did not rerun.
    key = ("copied_positions", "env.from_samples")
    for results, expected in ((tmp_path, "silent"), (tmp_path / "fresh", None)):
        partial = run_recording(
            simulated_recording,
            module,
            op_ids=["other.op"],
            operations=ops,
            results=results,
        )
        responses = {(h["hazard"], h["op"]): h["response"] for h in partial["hazards"]}
        assert responses.get(key) == expected
    rerun = run_recording(
        simulated_recording,
        module,
        op_ids=["env.from_samples"],
        operations=ops,
        results=tmp_path,
    )
    row = next(h for h in rerun["hazards"] if h["op"] == "env.from_samples")
    assert row["response"] == "silent"


@pytest.mark.parametrize("changed", ["conversions", "env_kind"])
def test_subset_run_refuses_a_changed_load(simulated_recording, tmp_path, changed):
    import dataclasses

    ops, module = _probe_setup()
    run_recording(simulated_recording, module, operations=ops, results=tmp_path)
    path = tmp_path / f"{simulated_recording.dataset}.json"
    stored = json.loads(path.read_text())
    if changed == "conversions":
        rec = dataclasses.replace(
            simulated_recording,
            conversions=(*simulated_recording.conversions, "a new conversion"),
        )
    else:
        rec = simulated_recording
        stored["env_kind"] = "declared"
        path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="rerun the recording in full"):
        run_recording(
            rec, module, op_ids=["other.op"], operations=ops, results=tmp_path
        )
    # The stored record is left as it was.
    assert json.loads(path.read_text())["operations"] == stored["operations"]


def _changed_load(rec, part):
    import dataclasses

    if part == "params":
        train = rec.params["train_epochs"].copy()
        train[0, 1] -= 1.0  # one train epoch a second shorter
        return dataclasses.replace(rec, params={**rec.params, "train_epochs": train})
    if part == "positions":
        positions = rec.positions.copy()
        positions[5] = np.nan
        return dataclasses.replace(rec, positions=positions)
    if part == "times":
        return dataclasses.replace(rec, times=rec.times + 1.0)
    if part == "spike_counts":
        trains = (rec.spike_times[0][1:], *rec.spike_times[1:])
        return dataclasses.replace(rec, spike_times=trains)
    if part == "unit_ids":
        return dataclasses.replace(rec, unit_ids=rec.unit_ids[::-1].copy())
    return dataclasses.replace(rec, geometry={**rec.geometry, "size_cm": 100.0})


@pytest.mark.parametrize(
    "part",
    ["params", "positions", "times", "spike_counts", "unit_ids", "geometry"],
)
def test_subset_run_refuses_a_load_with_another_fingerprint(
    simulated_recording, tmp_path, part
):
    ops, module = _probe_setup()
    run_recording(simulated_recording, module, operations=ops, results=tmp_path)
    path = tmp_path / f"{simulated_recording.dataset}.json"
    stored = json.loads(path.read_text())
    with pytest.raises(ValueError, match=f"the load's {part} differ"):
        run_recording(
            _changed_load(simulated_recording, part),
            module,
            op_ids=["other.op"],
            operations=ops,
            results=tmp_path,
        )
    assert json.loads(path.read_text()) == stored


def test_subset_run_passes_the_same_load_and_skips_callables(
    simulated_recording, tmp_path
):
    import dataclasses

    ops, module = _probe_setup()
    rec = dataclasses.replace(
        simulated_recording, geometry={"size_cm": 100.0, "environment": lambda: 1}
    )
    run_recording(rec, module, operations=ops, results=tmp_path)
    # The same load, read again: new arrays, a new callable.
    again = dataclasses.replace(
        rec,
        positions=rec.positions.copy(),
        params={
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in rec.params.items()
        },
        geometry={"size_cm": 100.0, "environment": lambda: 2},
    )
    partial = run_recording(
        again, module, op_ids=["other.op"], operations=ops, results=tmp_path
    )
    assert partial["load_fingerprint"]["params"] == load_fingerprint(rec)["params"]


def test_subset_run_refuses_a_record_without_a_fingerprint(
    simulated_recording, tmp_path
):
    ops, module = _probe_setup()
    run_recording(simulated_recording, module, operations=ops, results=tmp_path)
    path = tmp_path / f"{simulated_recording.dataset}.json"
    stored = json.loads(path.read_text())
    del stored["load_fingerprint"]
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="rerun the recording in full once"):
        run_recording(
            simulated_recording,
            module,
            op_ids=["other.op"],
            operations=ops,
            results=tmp_path,
        )


def test_subset_run_stamps_only_the_cells_it_reran(simulated_recording, tmp_path):
    ops, module = _probe_setup()
    run_recording(simulated_recording, module, operations=ops, results=tmp_path)
    path = tmp_path / f"{simulated_recording.dataset}.json"
    record = json.loads(path.read_text())
    record["versions"] = {"git_commit": "old"}
    path.write_text(json.dumps(record))
    partial = run_recording(
        simulated_recording,
        module,
        op_ids=["other.op"],
        operations=ops,
        results=tmp_path,
    )
    assert partial["versions"] == {"git_commit": "old"}
    assert "versions" not in partial["operations"]["env.from_samples"]
    assert partial["operations"]["other.op"]["versions"]["git_commit"] != "old"
    kept = [h for h in partial["hazards"] if h["op"] == "env.from_samples"]
    assert kept
    assert all("versions" not in h for h in kept)


def test_every_verdict_describes_a_result_row():
    # Each verdicts.json entry must match a row of the committed results
    # with that status: a non-ok cell, a probed hazard row, a clean warning.
    from run import RESULTS, VERDICTS, stale_verdicts

    def records(folder):
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]

    # The inferred run's JSON is not committed; check it only where it exists.
    stale = stale_verdicts(
        records(RESULTS),
        records(RESULTS / "inferred_env") or None,
        json.loads(VERDICTS.read_text()),
    )
    assert stale == []


def test_stale_verdicts_finds_entries_without_their_row():
    from run import stale_verdicts

    record = {
        "dataset": "d",
        "load_warnings": [],
        "operations": {"a": {"status": "ok"}, "b": {"status": "error"}},
        "hazards": [
            {"hazard": "h", "op": "a", "response": "not probed"},
            {"hazard": "h", "op": "b", "response": "silent"},
        ],
    }
    verdicts = {
        "cells": {"d/a": {"gap": "G-1"}, "d/b": {"gap": "G-2"}},
        "hazards": {"d/h/a": {"verdict": "x"}, "d/h/b": {"verdict": "y"}},
    }
    assert stale_verdicts([record], [], verdicts) == ["cells: d/a", "hazards: d/h/a"]
