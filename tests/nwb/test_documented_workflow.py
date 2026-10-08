"""The published NWB recipe runs on an actual file through every data handoff."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")
pytestmark = pytest.mark.nwb
GUIDE = Path(__file__).resolve().parents[2] / "docs/user-guide/interoperability.md"


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
    empty_nwb.add_epoch(start_time=20.0, stop_time=30.0, tags=["run"])
    path = tmp_path / "session.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    return path, positions, times


def test_documented_components_to_decode_after_file_close(component_file, monkeypatch):
    path, expected_positions, expected_times = component_file
    monkeypatch.chdir(path.parent)
    blocks = re.findall(
        r"<!-- nwb-docs-test: run -->\s*```python\n(.*?)\n```",
        GUIDE.read_text(),
        re.S,
    )
    assert len(blocks) == 1, "Keep the complete NWB recipe executable."
    namespace = {"__name__": "__main__"}
    try:
        exec(compile(blocks[0], str(GUIDE), "exec"), namespace)
        pos = namespace["pos"]
        units = namespace["units"]
        rates = namespace["rates"]
        decoder = namespace["decoder"]
        result = namespace["result"]
        assert pos.units == namespace["env"].units == "cm"
        np.testing.assert_array_equal(pos.positions, expected_positions)
        np.testing.assert_array_equal(pos.times, expected_times)
        np.testing.assert_array_equal(namespace["epochs"], [[0, 10], [20, 30]])
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
        assert np.any(result.times < 10) and np.any(result.times >= 20)
        np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0)
        actual_line = namespace["actual_line"]
        map_line = namespace["ax"].lines[0]
        np.testing.assert_array_equal(actual_line.get_xdata(), map_line.get_xdata())
        np.testing.assert_array_equal(
            actual_line.get_ydata(), namespace["env"].bin_at(namespace["actual"])
        )
    finally:
        plt.close("all")
