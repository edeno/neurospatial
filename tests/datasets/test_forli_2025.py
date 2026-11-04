"""Forli et al. (2025), Zenodo 15738988: synthetic members in the record's layout, and the real session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurospatial.datasets import _load, load
from neurospatial.datasets._checks import run_checks

ROOM = np.array([[-2.9, 2.9], [-2.6, 2.6], [0.0, 2.3]])


def _flight_path(rate=120.0):
    """600 s: rest at a perch, a flight across the room every 60 s, rest again."""
    rng = np.random.default_rng(0)
    times = -20.0 + np.arange(int(600 * rate)) / rate
    positions = np.tile([-2.5, 0.0, 1.5], (times.size, 1))
    flying = np.zeros(times.size, dtype=bool)
    flights = []
    for start in np.arange(10.0, 590.0, 60.0):
        index = np.flatnonzero((times >= start) & (times < start + 4.0))
        share = np.linspace(0.0, 1.0, index.size)
        sign = 1.0 if len(flights) % 2 == 0 else -1.0
        positions[index, 0] = -2.5 * sign + 5.0 * sign * share
        positions[index, 1] = 1.5 * np.sin(np.pi * share)
        positions[index[-1] + 1 :, 0] = 2.5 * sign  # rests where it landed
        flying[index] = True
        flights.append((index[0] + 1, index[-1] + 1))  # 1-based, as MATLAB
    positions += rng.normal(0, 0.002, positions.shape)
    return times, positions, flying, np.asarray(flights)


def _write_members(folder, *, outside_y=None):
    """The behavior file and merge list as the record stores them."""
    from scipy.io import savemat

    times, positions, flying, flights = _flight_path()
    if outside_y is not None:
        positions[outside_y] = [0.0, -3.0, 0.5]  # past the -y bound
    savemat(
        folder / "Extracted_Behavior_240419.mat",
        {
            "t": times,
            "r": positions,
            "r_lim": ROOM,
            "f_smp": flights,
            "f_num": len(flights),
            "Fs": 120,
            "v_th": 0.5,
            "bflying": flying.astype(float),
            "options": {
                "unique_ID": np.array(["Dataset_3", "14543", "240419"], object)
            },
        },
    )
    merges = np.empty((3, 1), dtype=object)
    merges[0, 0] = np.empty((0, 1), dtype=object)
    merges[1, 0] = np.empty((1, 1), dtype=object)
    merges[1, 0][0, 0] = np.array([[7, 9]], dtype=float)
    merges[2, 0] = np.empty((0, 1), dtype=object)
    savemat(folder / "merges_Dataset_3_14543_240419.mat", {"merges": merges})
    for probe in (1, 2, 3):
        (folder / f"SU_kilosort4_outdir_probe{probe}.mat").write_bytes(b"")
    return times, positions, flying, flights


#: Spike times in µs per probe and cluster; probe 2's clusters 7 and 9 are
#: one unit split in two, sharing one spike.
TABLES = {
    1: {3: [0.0, 1_000_000.0, 1_000_500.0, 2_000_000.0]},
    2: {7: [5_000_000.0, 7_000_000.0], 9: [6_000_000.0, 7_000_200.0], 11: [9e6]},
    3: {4: [-10e6, 500e6]},
}


@pytest.fixture
def members(tmp_path, monkeypatch):
    """Synthetic members; the MATLAB unit tables are read as their DataFrames."""
    from neurospatial.datasets import forli_2025

    def good_units(path):
        probe = int(path.stem.removeprefix("SU_kilosort4_outdir_probe"))
        return pd.DataFrame(
            {
                "cluster_id": list(TABLES[probe]),
                "spikeTimes_usec": [np.array(v) for v in TABLES[probe].values()],
            }
        )

    monkeypatch.setattr(forli_2025, "_good_units", good_units)
    return tmp_path, _write_members(tmp_path)


def test_loads_the_members_into_a_3d_session(members, session_entry):
    from neurospatial.datasets import forli_2025

    folder, (times, positions, _, flights) = members
    entry = session_entry("forli_2025")
    rec = run_checks(forli_2025.load_session(entry, folder), stated=entry.info.stated)
    assert rec.failed_checks == ()
    np.testing.assert_array_equal(rec.positions, positions)
    assert rec.position_units == "m"
    # µs to s; 7 and 9 pooled under 7 and sorted; the 1 ms rule drops the
    # spike 0.5 ms after its predecessor (probe 1) and 0.2 ms (the pool).
    assert list(rec.units.index) == [
        "probe1-3",
        "probe2-7",
        "probe2-11",
        "probe3-4",
    ]
    np.testing.assert_array_equal(rec.spike_times[0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(rec.spike_times[1], [5.0, 6.0, 7.0])
    assert list(rec.units["probe"]) == [1, 2, 2, 3]
    # Flights at the 1-based f_smp; analysis flights clipped to the spikes.
    np.testing.assert_array_equal(
        rec.epochs["flight"], np.column_stack([times[f - 1] for f in flights.T])
    )
    np.testing.assert_array_equal(rec.epochs["neural_recording"], [[-10.0, 500.0]])
    analysis = rec.params["analysis_epochs"]
    assert analysis.min() >= -10.0
    assert analysis.max() <= 500.0
    assert rec.params["min_speed"] == 0.5
    # The room: a grid tiling r_lim exactly, every voxel active.
    assert rec.env.n_bins == 29 * 26 * 11
    np.testing.assert_allclose(
        [rec.env.dimension_ranges[axis] for axis in range(3)], ROOM, atol=1e-9
    )
    dropped = next(c for c in rec.conversions if "1 ms" in c.text)
    assert "(2 spikes)" in dropped.text


def test_samples_past_the_room_are_described_once(members, session_entry):
    """One watch-out counts them; the room's Conversion says where they lie."""
    from neurospatial.datasets import forli_2025

    folder, _ = members
    _write_members(folder, outside_y=slice(1000, 1100))
    entry = session_entry("forli_2025")
    rec = run_checks(forli_2025.load_session(entry, folder), stated=entry.info.stated)
    outside = [w for w in rec.watch_outs if "outside" in w and "r_lim" in w]
    counted = [w for w in rec.watch_outs if "lie outside the declared" in w]
    assert outside == []
    (finding,) = counted
    assert finding.startswith("100 tracked position samples of ")
    assert "of stopped (below 0.5 m/s) samples and" in finding
    room = next(c for c in rec.conversions if c.text.startswith("environment"))
    assert "(100 past -y, up to 0.40 m past a bound" in room.evidence


def test_cells_per_axis_take_the_nearest_count_ties_to_fewer():
    from neurospatial.datasets.forli_2025 import cell_counts

    # 2.3 / 0.2 is 11.499999999999998 in floats: a tie, which goes to 11.
    np.testing.assert_array_equal(
        cell_counts(np.array([5.8, 5.2, 2.3, 2.35, 2.25, 0.05]), 0.2),
        [29, 26, 11, 12, 11, 1],
    )


def test_merge_of_a_cluster_missing_from_the_units_names_it(members):
    from neurospatial.datasets import forli_2025

    folder, _ = members
    # Probe 1's table has cluster 3 only; a merge list naming 9 is another sorting.
    with pytest.raises(ValueError, match=r"probe 1: .*no good unit \[9\]"):
        forli_2025._probe_units(folder / "SU_kilosort4_outdir_probe1.mat", 1, [[3, 9]])


def test_drop_close_spikes_compares_each_with_its_predecessor():
    from neurospatial.datasets.forli_2025 import drop_close_spikes

    # 0.0004 s after 0.0 and 0.0008 s after 0.0004: both dropped, as the
    # script's s(find(diff(s) < 0.001) + 1) drops them.
    spikes = np.array([0.0, 0.0004, 0.0012, 0.5, 0.5009])
    np.testing.assert_array_equal(drop_close_spikes(spikes), [0.0, 0.5])


def test_python_310_says_mat_io_needs_311(
    monkeypatch, tmp_path, block_import, no_download
):
    """mat-io installs only on Python 3.11+: on 3.10 loading the bat says so."""
    block_import("matio")
    monkeypatch.setattr(_load, "_python_version", lambda: (3, 10))
    with pytest.raises(ImportError, match=r"mat-io.*only on Python 3\.11 or later"):
        load("forli_2025", cache=tmp_path)


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """The demo session as the survey expected it: 3-D, 120 Hz, 38 flights."""
    rec = load("forli_2025")
    assert_default_session(rec, session_entry("forli_2025"))
    assert rec.n_units == 289
    assert rec.n_dims == 3
    assert 119.0 <= rec.sampling_rate <= 121.0
    low = np.nanmin(rec.positions, axis=0)
    high = np.nanmax(rec.positions, axis=0)
    assert low[0] >= -3.1
    assert low[2] >= -0.2
    assert high[0] <= 3.1
    assert high[2] <= 2.5
    assert (rec.units["probe"] == 1).sum() <= 24
    assert len(rec.epochs["flight"]) == 38
    assert rec.env.n_bins == 29 * 26 * 11
