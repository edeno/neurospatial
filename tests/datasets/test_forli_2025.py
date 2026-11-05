"""Forli et al. (2025), Zenodo 15738988: synthetic members in the record's layout, and the real session."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from neurospatial import compute_spatial_rates
from neurospatial._intervals import _merge
from neurospatial.datasets import _load, forli_2025, load
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
    savemat(
        folder / "BPC_Dataset_3_14543_240419.mat",
        {"best_prb_ch": np.array([[2, 41]], dtype=np.uint8)},
    )
    for probe in (1, 2, 3):
        (folder / f"SU_kilosort4_outdir_probe{probe}.mat").write_bytes(b"")
        (folder / f"RPL_probe{probe}.mat").write_bytes(b"")
    return times, positions, flying, flights


#: Spike times in µs per probe and cluster; probe 2's clusters 7 and 9 are
#: one unit split in two, sharing one spike.
TABLES = {
    1: {3: [0.0, 1_000_000.0, 1_000_500.0, 2_000_000.0]},
    2: {7: [5_000_000.0, 7_000_000.0], 9: [6_000_000.0, 7_000_200.0], 11: [9e6]},
    3: {4: [-10e6, 500e6]},
}
#: Channel ids per probe; each probe's first is its best channel for SWRs.
LFP_CHANNELS = {1: [271, 273], 2: [388, 390], 3: [500]}
#: Ripple peaks and correlations per probe: on probe 2, a row at exactly the
#: paper's 0.2 is kept, 0.19 is not, and the one at 12 s is in a flight.
RIPPLES = {
    1: {"t": [3.0, 4.0], "corr": [0.5, 0.1]},
    2: {"t": [5.0, 6.0, 7.0, 12.0], "corr": [0.2, 0.19, 0.9, 0.4]},
    3: {"t": [8.0], "corr": [0.05]},
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

    def ripple_file(path):
        probe = int(path.stem.removeprefix("RPL_probe"))
        channels = LFP_CHANNELS[probe]
        return forli_2025._RippleFile(
            table=pd.DataFrame(RIPPLES[probe]),
            probe_info=pd.DataFrame(
                {
                    "ch": channels,
                    "xPos": [8.0] * len(channels),
                    "yPos": [100.0 * (i + 1) for i in range(len(channels))],
                }
            ),
            channel=channels[0],
        )

    monkeypatch.setattr(forli_2025, "_good_units", good_units)
    monkeypatch.setattr(forli_2025, "_ripple_file", ripple_file)
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


def test_ripple_tables_become_events(members, session_entry):
    """Every row, and the rows at the paper's correlation; the best probe named."""
    from neurospatial.datasets import forli_2025

    folder, _ = members
    entry = session_entry("forli_2025")
    rec = run_checks(forli_2025.load_session(entry, folder), stated=entry.info.stated)
    assert rec.failed_checks == ()
    assert rec.params["ripple_probe"] == 2
    np.testing.assert_array_equal(
        rec.events["ripple_candidates_probe2"], [5.0, 6.0, 7.0, 12.0]
    )
    np.testing.assert_array_equal(rec.events["swr_probe2"], [5.0, 7.0, 12.0])
    np.testing.assert_array_equal(rec.events["swr_probe1"], [3.0])
    assert rec.events["swr_probe3"].size == 0
    text = " ".join(rec.watch_outs)
    assert "1 of the 3 SWRs on probe 2 lie in the file's flight samples" in text
    assert "Of 4 rows of that probe's table, 3 reach" in text


def test_lfp_is_described_at_load_and_fetched_only_on_request(members, session_entry):
    """Every channel named with its site, the best probe's ripple channel chosen."""
    from neurospatial.datasets import forli_2025

    folder, _ = members
    entry = session_entry("forli_2025")
    rec = forli_2025.load_session(entry, folder)
    assert list(rec.lfp.channels) == [
        "probe1-ch271",
        "probe1-ch273",
        "probe2-ch388",
        "probe2-ch390",
        "probe3-ch500",
    ]
    assert rec.lfp.ripple_channels == ("probe2-ch388",)
    assert "for this probe's SWRs" in rec.lfp.channels["probe2-ch388"]
    assert "for this probe's SWRs" not in rec.lfp.channels["probe2-ch390"]
    assert rec.lfp.channels["probe2-ch390"].startswith("probe 2, channel 390, 8 um")
    assert rec.lfp.unit == "arbitrary"
    assert [file.channels for file in rec.lfp.files] == [
        ("probe1-ch271", "probe1-ch273"),
        ("probe2-ch388", "probe2-ch390"),
        ("probe3-ch500",),
    ]
    assert rec.lfp.cache == folder.parents[1]
    assert not list(folder.glob("LFP_probe*"))
    assert any("voltage_scaling" in c.text for c in rec.conversions)


def _write_lfp(path, channels, times, values):
    """An LFP_probe*.mat as MATLAB v7.3 writes it: columns stored as HDF5 rows."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as file:
        out = file.create_group("red_out")
        out["lfp"] = np.asarray(values, dtype=np.int16)
        out["channelMap"] = np.asarray(channels, dtype=np.float64)[:, None]
        out["t_ds"] = np.asarray(times, dtype=np.float64)[:, None]
        out["voltage_scaling"] = np.array([[0.5]])


def test_read_lfp_scales_and_orders_the_channels_asked_for(tmp_path):
    from neurospatial.datasets.forli_2025 import read_lfp

    times = np.array([0.0, 0.0008, 0.0016])
    first, second = tmp_path / "LFP_probe1.mat", tmp_path / "LFP_probe2.mat"
    _write_lfp(first, [10, 11], times, [[2, 4, 6], [8, 10, 12]])
    _write_lfp(second, [20], times, [[100, 200, 300]])
    samples, clock = read_lfp((first, second), ("probe2-ch20", "probe1-ch10"))
    np.testing.assert_array_equal(samples, [[50, 1], [100, 2], [150, 3]])
    np.testing.assert_array_equal(clock, times)


def test_read_lfp_refuses_probes_on_different_time_vectors(tmp_path):
    from neurospatial.datasets.forli_2025 import read_lfp

    first, second = tmp_path / "LFP_probe1.mat", tmp_path / "LFP_probe2.mat"
    _write_lfp(first, [10], [0.0, 0.1], [[1, 2]])
    _write_lfp(second, [20], [0.0, 0.2], [[1, 2]])
    with pytest.raises(ValueError, match="its own time vector"):
        read_lfp((first, second), ("probe1-ch10", "probe2-ch20"))


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


@pytest.mark.realdata
def test_default_session_counts_steps_past_the_stated_flight_speed(session_entry):
    """Tsoar et al.'s 15.3 m/s, no head margin: the tracking jumps are counted."""
    from neurospatial.datasets._checks import _fast_samples, _held_samples

    rec = load("forli_2025")
    stated = session_entry("forli_2025").info.stated
    assert stated.speed_limit == 15.3
    assert "No margin is added" in stated.speed_limit_source
    held = _held_samples(rec.times, rec.positions)[0]
    n_fast = int(_fast_samples(rec.times, rec.positions, held, 15.3).sum())
    assert n_fast == 121
    (finding,) = [w for w in rec.watch_outs if "farther from the last believable" in w]
    assert f"{n_fast} samples farther from the last believable sample" in finding
    assert "than 15.3 m/s allows" in finding
    assert "Positions are kept as stored" in finding
    # The relative heuristic is a fallback, not run with a stated limit.
    assert not any("likely tracking jumps" in w for w in rec.watch_outs)
    # Kept as stored: a step of 22 m/s is still in the positions.
    steps = np.linalg.norm(np.diff(rec.positions, axis=0), axis=1) / np.diff(rec.times)
    assert np.nanmax(steps) > 21.0


@pytest.mark.realdata
def test_ripple_events_sit_on_the_spikes_clock():
    """The population rate peaks in the 10 ms before the table's times."""
    rec = load("forli_2025")
    spikes = np.sort(np.concatenate(rec.spike_times))
    peaks = rec.events[f"swr_probe{rec.params['ripple_probe']}"]
    assert peaks.size == 3350
    edges = np.arange(-0.5, 0.5001, 0.01)
    start, stop = np.searchsorted(spikes, [peaks - 0.5, peaks + 0.5])
    counts = np.zeros(edges.size - 1)
    for first, last, peak in zip(start, stop, peaks, strict=True):
        counts += np.histogram(spikes[first:last] - peak, edges)[0]
    rate = counts / (peaks.size * 0.01)
    assert edges[np.argmax(rate)] == pytest.approx(-0.01)
    assert rate.max() > 1.5 * np.median(rate)


@pytest.mark.realdata
def test_lfp_of_the_ripple_channel_shows_the_ripples():
    """rec.lfp is on the events' clock: ripple power peaks at the table's times."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    rec = load("forli_2025")
    samples, times = rec.lfp.fetch(rec.lfp.ripple_channels)
    assert samples.shape == (times.size, 1)
    assert times[0] == pytest.approx(rec.epochs["neural_recording"][0, 0], abs=0.01)
    assert times[-1] == pytest.approx(rec.epochs["neural_recording"][0, 1], abs=0.01)
    sos = butter(4, [100, 200], btype="band", fs=rec.lfp.sampling_rate, output="sos")
    power = np.abs(hilbert(sosfiltfilt(sos, samples[:, 0])))
    peaks = rec.events[f"swr_probe{rec.params['ripple_probe']}"]
    peaks = peaks[(peaks > times[0] + 1.0) & (peaks < times[-1] - 1.0)]
    half = 188  # 150 ms of samples
    centres = np.searchsorted(times, peaks)
    average = np.mean([power[c - half : c + half + 1] for c in centres], axis=0)
    assert abs(np.argmax(average) - half) <= 13  # within 10 ms
    assert average.max() > 1.5 * np.median(average)


def _ripple_windows(rec):
    """Peak +/- 0.15 s of the SWRs, merged, in the recording, clear of flights."""
    peaks = rec.events[f"swr_probe{rec.params['ripple_probe']}"]
    half = forli_2025.OWN_3D_ANALYSIS["window_half_width"]
    rows = _merge(np.column_stack([peaks - half, peaks + half]))
    ((start, stop),) = rec.epochs["neural_recording"]
    flights = rec.epochs["flight"]
    in_flight = np.any(
        (rows[:, 0, None] < flights[:, 1]) & (rows[:, 1, None] > flights[:, 0]), axis=1
    )
    return rows[(rows[:, 0] >= start) & (rows[:, 1] <= stop) & ~in_flight]


def _decoding_units(rec):
    """Flight-active units below the script's interneuron rate, as spike trains."""
    flights = rec.params["analysis_epochs"]
    kept = []
    for train in rec.spike_times:
        n_flight = int(
            np.sum(np.searchsorted(train, flights[:, 1]))
            - np.sum(np.searchsorted(train, flights[:, 0]))
        )
        rate = 1 / np.mean(np.diff(train)) if train.size > 1 else 0.0
        if n_flight > len(flights) and rate <= forli_2025.INTERNEURON_RATE:
            kept.append(train)
    return kept


def _n_judged_trajectories(rec, trains, windows, rates):
    """How many windows the rule judges trajectories, decoding each in sliding bins."""
    from neurospatial.decoding import bin_spikes_in_time, decode_position
    from neurospatial.replay import ReplayCriterion, ReplaySegmentation, detect_replay
    from neurospatial.replay.methods import PFEIFFER_FOSTER_2013

    record = forli_2025.OWN_3D_ANALYSIS
    time_bin, step = record["time_bin"], record["time_step"]
    method = dataclasses.replace(
        PFEIFFER_FOSTER_2013,
        name="bat_room",
        units="m",
        segmentation=ReplaySegmentation(
            {"max_jump": record["max_jump"], "method": "map", "min_time_bins": 11}
        ),
        criteria=(
            ReplayCriterion(
                "segment_start_to_end_distance", ">=", record["min_distance"]
            ),
        ),
    )
    width = round(time_bin / step)
    n_judged = 0
    for start, stop in windows:
        counts, _ = bin_spikes_in_time(
            trains, dt=step, t_start=float(start), t_stop=float(stop)
        )
        counts = np.asarray(counts)
        running = np.vstack([np.zeros((1, counts.shape[1]), counts.dtype), counts])
        running = np.cumsum(running, axis=0)
        summed = running[width:] - running[:-width]
        centres = start + time_bin / 2 + step * np.arange(summed.shape[0])
        decoded = decode_position(rec.env, summed, rates, dt=time_bin, times=centres)
        verdict = detect_replay(
            rec.env, decoded.posterior, centres, replay_method=method
        )
        n_judged += bool(verdict.is_replay)
    return n_judged


def _shifted(trains, span, draw, *, common):
    """Spikes circularly shifted over ``span``, by at least 10 s: the fixed nulls."""
    start, stop = span
    length = stop - start
    rng = np.random.default_rng(np.random.SeedSequence(2026, spawn_key=(draw, common)))
    shared = rng.uniform(10.0, length - 10.0)
    return [
        np.sort(
            np.mod(
                train
                - start
                + (shared if common else rng.uniform(10.0, length - 10.0)),
                length,
            )
            + start
        )
        for train in trains
    ]


@pytest.fixture(scope="module")
def own_analysis():
    """The session, its windows, its units and their 3-D flight fields."""
    rec = load("forli_2025")
    trains = _decoding_units(rec)
    rates = compute_spatial_rates(
        rec.env,
        trains,
        rec.times,
        rec.positions,
        epochs=rec.params["analysis_epochs"],
        bandwidth=forli_2025.OWN_3D_ANALYSIS["bandwidth"],
    )
    return rec, _ripple_windows(rec), trains, rates


@pytest.mark.realdata
def test_own_3d_analysis_recomputes_its_record(own_analysis):
    """The recorded count, windows and first null draws, recomputed.

    The record's verdict was fixed before the run. Every draw of both nulls takes
    about a minute and a half, so this reruns the first of each; the
    record keeps all twenty.
    """
    rec, windows, trains, rates = own_analysis
    record = forli_2025.OWN_3D_ANALYSIS
    assert len(windows) == record["n_windows"]
    assert len(trains) == record["n_units"]
    assert _n_judged_trajectories(rec, trains, windows, rates) == record["n_judged"]
    span = tuple(rec.epochs["neural_recording"][0])
    for common, recorded in ((0, "null_own_offsets"), (1, "null_common_offset")):
        shifted = _shifted(trains, span, 0, common=common)
        assert (
            _n_judged_trajectories(rec, shifted, windows, rates) == record[recorded][0]
        )


def test_own_3d_analysis_states_what_it_is_not():
    """It is ours: no claim on the paper's replay and no number of the paper's."""
    record = forli_2025.OWN_3D_ANALYSIS
    assert "not the paper's" in record["what"]
    assert "1-D" in record["watch_out"]
    assert "0.15" in record["watch_out"]
    verdict = record["n_judged"] > max(
        np.percentile(record["null_own_offsets"], 95),
        np.percentile(record["null_common_offset"], 95),
    )
    assert record["status"] == ("met" if verdict else "not met")


@pytest.mark.realdata
def test_held_out_flights_decode_barely_better_than_chance(own_analysis):
    """Why the verdicts mean little: the decoder hardly follows the bat in flight."""
    from neurospatial.decoding import bin_spikes_in_time, decode_position, map_position

    rec, _, trains, _ = own_analysis
    record = forli_2025.OWN_3D_ANALYSIS["held_out_flights"]
    rates = compute_spatial_rates(
        rec.env,
        trains,
        rec.times,
        rec.positions,
        epochs=rec.params["train_epochs"],
        bandwidth=forli_2025.OWN_3D_ANALYSIS["bandwidth"],
    )
    decoded, actual = [], []
    for start, stop in rec.params["test_epochs"]:
        if stop - start < 0.5:
            continue
        counts, centres = bin_spikes_in_time(
            trains, dt=0.1, t_start=float(start), t_stop=float(stop)
        )
        result = decode_position(
            rec.env, np.asarray(counts), rates, dt=0.1, times=centres
        )
        decoded.append(map_position(rec.env, result.posterior))
        actual.append(
            np.column_stack(
                [np.interp(centres, rec.times, axis) for axis in rec.positions.T]
            )
        )
    decoded, actual = np.vstack(decoded), np.vstack(actual)
    shuffled = np.random.default_rng(0).permutation(actual)
    assert np.median(np.linalg.norm(decoded - actual, axis=1)) == pytest.approx(
        record["median_error_m"], abs=0.005
    )
    assert np.median(np.linalg.norm(decoded - shuffled, axis=1)) == pytest.approx(
        record["shuffled_error_m"], abs=0.005
    )
