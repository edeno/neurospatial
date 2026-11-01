"""The published NWB recipe runs on an actual file through every data handoff."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.path import Path as MplPath

pynwb = pytest.importorskip("pynwb")
pytestmark = pytest.mark.nwb
GUIDE = Path(__file__).resolve().parents[2] / "docs/user-guide/interoperability.md"


def _recipe(marker: str) -> str:
    """The NWB recipe in the guide that uses ``marker``."""
    blocks = re.findall(
        r"<!-- nwb-docs-test: run -->\s*```python\n(.*?)\n```",
        GUIDE.read_text(),
        re.DOTALL,
    )
    matching = [block for block in blocks if marker in block]
    assert len(matching) == 1, f"Keep the NWB recipe using {marker} executable."
    return matching[0]


def _run_recipe(marker: str) -> dict:
    namespace = {"__name__": "__main__"}
    exec(compile(_recipe(marker), str(GUIDE), "exec"), namespace)
    return namespace


@pytest.fixture
def component_file(empty_nwb, tmp_path):
    from pynwb.behavior import Position, SpatialSeries

    times = np.concatenate([np.arange(101) / 10, 20 + np.arange(101) / 10])
    positions = np.column_stack(
        [
            20 + 10 * np.sin(2 * np.pi * times / 5),
            20 + 10 * np.cos(2 * np.pi * times / 5),
        ]
    )
    pos = Position(name="Position")
    pos.add_spatial_series(
        SpatialSeries(
            name="xy",
            data=positions,
            timestamps=times,
            reference_frame="arena origin",
            unit="centimeters",
        )
    )
    empty_nwb.create_processing_module("behavior", "tracking").add(pos)
    for label, mask, coverage in (
        (7, positions[:, 0] > 24, [[0.0, 30.0]]),
        (11, positions[:, 0] < 16, [[1.0, 9.0], [20.0, 29.0]]),
    ):
        empty_nwb.add_unit(
            id=label, spike_times=times[mask][::2], obs_intervals=coverage
        )
    empty_nwb.add_epoch(start_time=0.0, stop_time=10.0, tags=["run"])
    empty_nwb.add_epoch(start_time=10.0, stop_time=20.0, tags=["sleep"])
    empty_nwb.add_epoch(start_time=20.0, stop_time=30.0, tags=["run"])
    path = tmp_path / "session.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    return path, positions, times


def test_documented_components_to_decode_after_file_close(component_file, monkeypatch):
    path, expected_positions, expected_times = component_file
    monkeypatch.chdir(path.parent)
    namespace = _run_recipe("read_units")
    try:
        pos = namespace["pos"]
        units = namespace["units"]
        rates = namespace["rates"]
        decoder = namespace["decoder"]
        result = namespace["result"]
        assert pos.units == namespace["env"].units == "cm"
        np.testing.assert_array_equal(pos.positions, expected_positions)
        np.testing.assert_array_equal(pos.times, expected_times)
        np.testing.assert_array_equal(namespace["epochs"].windows, [[0, 10], [20, 30]])
        np.testing.assert_array_equal(units.spike_window, [[1, 9], [20, 29]])
        np.testing.assert_array_equal(rates.unit_ids, [7, 11])
        np.testing.assert_array_equal(namespace["rate_table"].index, [7, 11])
        np.testing.assert_array_equal(decoder.unit_ids, [7, 11])
        np.testing.assert_array_equal(decoder.spike_window, units.spike_window)
        assert not rates.spike_window_assumed
        assert not result.spike_window_assumed
        assert rates.occupancy.sum() == pytest.approx(17.0)
        inside = ((result.times >= 1) & (result.times < 9)) | (
            (result.times >= 20) & (result.times < 29)
        )
        assert np.all(inside)
        assert np.any(result.times < 10)
        assert np.any(result.times >= 20)
        np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0)
        ax = namespace["ax"]
        actual_line = next(
            line for line in ax.lines if line.get_label() == "Actual position"
        )
        map_line = next(line for line in ax.lines if line.get_label() == "MAP")
        np.testing.assert_array_equal(actual_line.get_xdata(), map_line.get_xdata())
        actual = np.column_stack(
            [
                np.interp(result.times, expected_times, expected_positions[:, dim])
                for dim in range(expected_positions.shape[1])
            ]
        )
        finite = np.isfinite(actual_line.get_ydata())
        np.testing.assert_array_equal(
            actual_line.get_ydata()[finite], namespace["env"].bin_at(actual)
        )
        assert (
            sum(
                code == MplPath.MOVETO
                for _, code in actual_line.get_path().iter_segments()
            )
            == 2
        )
    finally:
        plt.close("all")


def test_documented_trials_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = _run_recipe("write_trials")
    trials, loaded = namespace["trials"], namespace["loaded"]
    assert loaded.interval_ids.tolist() == [1, 2, 3, 4]
    assert loaded.fields == trials.fields
    for name in trials.fields:
        assert getattr(loaded, name).tolist() == getattr(trials, name).tolist()
    assert loaded.sel(interval_id=2).end_region is None
    assert namespace["left"].interval_ids.tolist() == [1, 3]


WORKFLOWS = GUIDE.parent / "workflows.md"


def test_route_workflow_curves_survive_an_nwb_round_trip():
    """Workflow 9's trials, written to NWB and read back, give the same curves."""
    from datetime import datetime, timezone

    from pynwb import NWBFile

    from neurospatial.encoding import compute_tuning_curves
    from neurospatial.io.nwb import read_trials, write_trials

    text = WORKFLOWS.read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)

    def block(marker: str) -> str:
        matching = [code for code in blocks if marker in code]
        assert len(matching) == 1, f"{len(matching)} workflow blocks contain {marker!r}"
        return matching[0]

    namespace = {"__name__": "__main__"}
    for marker in (
        "make_w_maze().env_track",
        "def run(start, goal",
        "trials = detect_trials(track",
        "progress = path_progress(",
        "curves = {}",
    ):
        exec(compile(block(marker), str(WORKFLOWS), "exec"), namespace)
    plt.close("all")

    nwbfile = NWBFile(
        session_description="W-track alternation",
        identifier="w-track-session",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    write_trials(nwbfile, namespace["trials"])
    reloaded = read_trials(nwbfile)
    chosen = reloaded[(reloaded.instructed_goal == "well_1") & reloaded.correct]
    again = compute_tuning_curves(
        namespace["spike_times"],
        namespace["times"],
        namespace["progress"],
        bin_edges=namespace["bin_edges"],
        variable="progress",
        epochs=chosen,
    )
    expected = namespace["curves"]["A: left, correct"]
    np.testing.assert_array_equal(again.firing_rates, expected.firing_rates)
    assert (
        again.analysis_settings["epoch_interval_ids"]
        == expected.analysis_settings["epoch_interval_ids"]
    )
