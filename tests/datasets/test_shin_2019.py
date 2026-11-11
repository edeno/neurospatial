"""Shin et al. (2019), DANDI 000978: synthetic NWB files and the real sessions."""

from __future__ import annotations

import datetime
import io
import itertools
import urllib.error
import warnings

import numpy as np
import pytest

from neurospatial.datasets._checks import run_checks

#: The synthetic W-maze's nodes, cm: arms at x = 30, 70 and 110, wells at y = 100.
NODES = {
    "well_1": (70.0, 100.0),
    "well_2": (30.0, 100.0),
    "well_3": (110.0, 100.0),
    "base_1": (70.0, 10.0),
    "base_2": (30.0, 10.0),
    "base_3": (110.0, 10.0),
}
#: Rest, run, rest, s.
EPOCHS = ((0.0, 100.0), (110.0, 410.0), (420.0, 520.0))
#: The run's well sequence: centre, side, centre, other side, ...
WELLS = (1, 2, 1, 3, 1, 2, 1, 3, 1)
DWELL = 6.0
SPEED = 40.0
RATE = 30.0
#: Tetrodes (number, location); units sit on tetrodes 1, 3 and 4.
TETRODES = ((1, "CA1"), (2, "CA1"), (3, "PFC"), (4, "PFC"))
UNIT_TETRODES = (1, 1, 3, 4)
LFP_RATE = 1500.0


def _route(a: int, b: int) -> np.ndarray:
    """Corner points from well ``a`` to well ``b`` along the W."""
    return np.array(
        [NODES[f"well_{a}"], NODES[f"base_{a}"], NODES[f"base_{b}"], NODES[f"well_{b}"]]
    )


def _run_path(start: float):
    """Times, positions and trial rows of the run: dwell at a well, then go."""
    times, xy, rows = [], [], []
    now = start
    leave_times = []
    for a, b in itertools.pairwise(WELLS):
        dwell = np.arange(now, now + DWELL, 1 / RATE)
        times.append(dwell)
        xy.append(np.tile(NODES[f"well_{a}"], (dwell.size, 1)))
        now += DWELL
        leave_times.append(now)
        corners = _route(a, b)
        lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        travel = np.arange(0.0, lengths.sum() / SPEED, 1 / RATE)
        along = travel * SPEED
        cumulative = np.r_[0, np.cumsum(lengths)]
        segment = np.clip(np.searchsorted(cumulative, along, side="right") - 1, 0, 2)
        share = (along - cumulative[segment]) / lengths[segment]
        points = corners[segment] + share[:, None] * (
            corners[segment + 1] - corners[segment]
        )
        times.append(now + travel)
        xy.append(points)
        now += travel[-1] + 1 / RATE
    last = np.arange(now, now + DWELL, 1 / RATE)
    times.append(last)
    xy.append(np.tile(NODES[f"well_{WELLS[-1]}"], (last.size, 1)))
    leave_times.append(now + DWELL)
    # A trial runs from leaving one well to leaving the next, 0.5 s late.
    stops = np.array(leave_times) + 0.5
    rows = [(start, stops[1])] + [
        (stops[k], stops[k + 1]) for k in range(1, len(stops) - 1)
    ]
    return np.concatenate(times), np.concatenate(xy), rows


def _write_session(path, *, comments=None):
    """An NWB file laid out as DANDI 000978's."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.ecephys import LFP, ElectricalSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    nwb = pynwb.NWBFile(
        "Single Day WTrack Learning Experiment",
        "SYN_01",
        datetime.datetime(2024, 5, 1, tzinfo=datetime.timezone.utc),
        subject=Subject(subject_id="JDS-SingleDay-JS14", species="Rattus norvegicus"),
    )
    run_times, run_xy, trial_rows = _run_path(EPOCHS[1][0])
    rest = [np.arange(a, b, 1 / RATE) for a, b in (EPOCHS[0], EPOCHS[2])]
    times = np.concatenate([rest[0], run_times, rest[1]])
    box = [
        np.column_stack([60 + 5 * np.sin(r / 7.0), 30 + 5 * np.cos(r / 5.0)])
        for r in rest
    ]
    xy = np.concatenate([box[0], run_xy, box[1]])
    speed = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1) * RATE]
    behavior = nwb.create_processing_module("behavior", "contains behavioral data")
    position = Position(name="Position")
    position.add_spatial_series(
        SpatialSeries(
            name="SpatialSeries",
            data=np.column_stack([xy, speed]),
            timestamps=times,
            reference_frame="center well",
            unit="centimeters; centimeters/second",
            comments=comments
            or "Video frames taken at 30fps - row1:xpos row2:ypos row3:velocity",
        )
    )
    behavior.add(position)
    device = nwb.create_device("array")
    nwb.add_electrode_column("label", "whether the channel is referenced")
    for number, location in TETRODES:
        group = nwb.create_electrode_group(
            f"Tetrode{number}", "tetrode", location=location, device=device
        )
        for label in ("Ref OFF", "Ref ON"):
            nwb.add_electrode(
                group=group,
                group_name=f"tetrode{number}",
                label=label,
                location=location,
            )
    region = nwb.create_electrode_table_region(list(range(2 * len(TETRODES))), "all")
    lfp_times = np.concatenate([np.arange(a, b, 1 / LFP_RATE) for a, b in EPOCHS])
    columns = np.arange(2 * len(TETRODES))
    voltage = (np.sin(lfp_times)[:, None] * 10 + columns[None, :]).astype(np.float64)
    nwb.add_acquisition(
        ElectricalSeries(
            name="ElectricalSeries",
            data=voltage,
            electrodes=region,
            timestamps=lfp_times,
        )
    )
    ecephys = nwb.create_processing_module("ecephys", "ecephys")
    lfp = LFP()
    ecephys.add(lfp)
    lfp.add_electrical_series(
        ElectricalSeries(
            name="ElectricalSeries",
            data=voltage,
            electrodes=nwb.create_electrode_table_region(
                list(range(2 * len(TETRODES))), "lfp"
            ),
            timestamps=lfp_times,
        )
    )
    epochs = TimeIntervals(
        name="epoch intervals", description="Start and end times of all epochs"
    )
    for start, stop in EPOCHS:
        epochs.add_interval(start_time=start, stop_time=stop)
    nwb.add_time_intervals(epochs)
    for column, text in (
        ("correct", "correct"),
        ("trajectory_type", "type"),
        ("start_well", "start"),
        ("end_well", "end"),
    ):
        nwb.add_trial_column(column, text)
    for (start, stop), a, b in zip(trial_rows, WELLS[:-1], WELLS[1:], strict=True):
        nwb.add_trial(
            start_time=start,
            stop_time=stop,
            correct=1.0,
            trajectory_type=0.0 if a == 1 else 1.0,
            start_well=float(a),
            end_well=float(b),
        )
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        # pynwb warns while the units table is not yet in the file.
        warnings.filterwarnings("ignore", message="The linked table for")
        for tetrode in UNIT_TETRODES:
            # The file's 'electrodes' value is the tetrode number, not a table row.
            nwb.add_unit(
                spike_times=np.sort(rng.uniform(1.0, 515.0, 400)), electrodes=[tetrode]
            )
        with pynwb.NWBHDF5IO(str(path), "w") as nwb_io:
            nwb_io.write(nwb)
    import h5py

    # The files store one value per unit, not the ragged list NWB defines.
    with h5py.File(path, "r+") as h5:
        del h5["units/electrodes_index"]
    return times, xy, np.array(trial_rows), lfp_times, voltage


@pytest.fixture
def entry(session_entry):
    return session_entry("shin_2019")


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "session.nwb"
    return path, *_write_session(path)


def test_loads_positions_epochs_trials_and_units(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, times, xy, trial_rows, _, _ = synthetic
    rec = run_checks(shin_2019.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_array_equal(rec.positions, xy)
    np.testing.assert_array_equal(rec.epochs["run"], [EPOCHS[1]])
    np.testing.assert_array_equal(rec.epochs["rest"], [EPOCHS[0], EPOCHS[2]])
    np.testing.assert_array_equal(rec.epochs["trials"], trial_rows)
    np.testing.assert_array_equal(rec.params["analysis_epochs"], [EPOCHS[1]])
    assert len(rec.trials) == len(WELLS) - 1
    # Brain areas by tetrode number: 1 is CA1, 3 and 4 are PFC.
    assert list(rec.units["brain_region"]) == ["CA1", "CA1", "PFC", "PFC"]
    assert list(rec.units["tetrode"]) == list(UNIT_TETRODES)
    assert rec.units["label"].isna().all()
    assert rec.animal == "rat JS14"


def test_trials_alternate_train_and_test_within_each_path(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    # Paths in order: 1-2, 2-1, 1-3, 3-1, 1-2, 2-1, 1-3, 3-1; the first of each
    # path trains, the second tests.
    train, test = rec.params["train_epochs"], rec.params["test_epochs"]
    trials = rec.epochs["trials"]
    np.testing.assert_array_equal(train, trials[:4])
    np.testing.assert_array_equal(test, trials[4:])


def test_w_maze_is_fitted_with_the_wells_at_the_arms_ends(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    for k in (1, 2, 3):
        well = rec.env.regions[f"well_{k}"].data
        assert np.linalg.norm(well - NODES[f"well_{k}"]) < 3.0, k
    on_track = rec.env.bin_at(np.array([NODES["base_1"], (50.0, 10.0), (110.0, 55.0)]))
    assert np.all(on_track >= 0)
    assert rec.env.bin_at(np.array([[50.0, 60.0]]))[0] == -1  # between two arms


def test_watch_outs_count_units_pynwb_would_place_in_the_wrong_area(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    # As rows, tetrode numbers 1, 1, 3, 4 name rows 1 (CA1), 1, 3 (CA1) and 4
    # (PFC): unit 3 lands in the wrong area.
    text = next(t for t in rec.watch_outs if "tetrode number" in t)
    assert "for 1 of the 4 units here the wrong brain area" in text
    region = next(c for c in rec.conversions if c.text.startswith("units:"))
    assert "(CA1 2, PFC 2)" in region.text
    assert "Read as rows here: CA1 3, PFC 1" in region.evidence
    assert any("not one of the paper's six rats" in t for t in rec.watch_outs) is False


def test_trial_stops_are_reported_as_departures(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, *_ = synthetic
    rec = shin_2019.load_session(entry, path)
    trials = next(c for c in rec.conversions if c.text.startswith("trials:"))
    # 0.5 s after leaving at 40 cm/s the rat is 20 cm down the arm.
    assert "the rat is a median 20.0 cm from the end of its end well's arm" in (
        trials.evidence
    )


def test_lfp_source_reads_columns_and_windows(synthetic, entry):
    from neurospatial.datasets import shin_2019

    path, _, _, _, lfp_times, voltage = synthetic
    rec = shin_2019.load_session(entry, path)
    assert list(rec.lfp.channels)[:2] == ["tetrode1 Ref OFF", "tetrode1 Ref ON"]
    assert rec.lfp.channels["tetrode3 Ref ON"].startswith("PFC, tetrode 3, referenced")
    samples, times = rec.lfp.reader((path,), ("tetrode2 Ref ON", "tetrode1 Ref OFF"))
    np.testing.assert_array_equal(times, lfp_times)
    np.testing.assert_array_equal(samples, voltage[:, [3, 0]])
    ((window, at),) = shin_2019._read_lfp_windows(
        path, ("tetrode4 Ref ON",), np.array([EPOCHS[1]])
    )
    inside = (lfp_times >= EPOCHS[1][0]) & (lfp_times < EPOCHS[1][1])
    np.testing.assert_array_equal(at, lfp_times[inside])
    np.testing.assert_array_equal(window[:, 0], voltage[inside, 7])


def test_byte_ranges_are_retried_and_reassembled(monkeypatch):
    """Rows read by ranged requests: a reset is retried, columns are picked."""
    from neurospatial.datasets import shin_2019

    array = np.arange(40 * 3, dtype="<f8").reshape(40, 3)
    blob = b"x" * 16 + array.tobytes()
    failures = {"left": 1}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        if failures["left"]:
            failures["left"] -= 1
            raise urllib.error.URLError(ConnectionResetError(54, "reset"))
        low, high = request.headers["Range"].removeprefix("bytes=").split("-")
        return _Response(blob[int(low) : int(high) + 1])

    monkeypatch.setattr(shin_2019, "_LFP_BLOCK_BYTES", 8 * 3 * 7)
    monkeypatch.setattr(shin_2019.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(shin_2019.urllib.request, "urlopen", urlopen)
    rows = shin_2019._ranged_rows("https://x", 16, 3, 5, 33, np.array([2, 0]))
    np.testing.assert_array_equal(rows, array[5:33][:, [2, 0]])


def test_refuses_a_position_series_without_its_column_names(tmp_path, entry):
    from neurospatial.datasets import shin_2019

    path = tmp_path / "session.nwb"
    _write_session(path, comments="x and y")
    with pytest.raises(ValueError, match="not the x, y and velocity columns"):
        shin_2019.load_session(entry, path)


def test_load_streams_a_stream_only_session_that_is_not_cached(
    monkeypatch, tmp_path, entry
):
    """load() reads such a session by byte range, never downloading it."""
    from neurospatial.datasets import _load, shin_2019

    assert entry.stream_only
    seen = {}
    monkeypatch.setattr(_load, "_stream_url", lambda e: "https://streamed")
    monkeypatch.setattr(
        _load,
        "fetch_session",
        lambda *a, **k: pytest.fail("a stream-only session was downloaded"),
    )

    def read(e, source):
        seen["source"] = source
        msg = "stop"
        raise RuntimeError(msg)

    monkeypatch.setattr(shin_2019, "load_session", read)
    with pytest.raises(RuntimeError, match="stop"):
        _load.load("shin_2019", cache=tmp_path)
    assert seen["source"] == "https://streamed"


# --------------------------------------------------------------------------
# Real sessions (DANDI 000978, read by byte range)
# --------------------------------------------------------------------------

#: "positions within 15 cm of reward wells" (STAR Methods): the template cut,
#: and here also the radius of a well visit's arrival and departure.
WELL_RADIUS = 15.0
#: The paper's immobility (<= 4 cm/s) and movement (> 5 cm/s) speeds, cm/s.
IMMOBILE = 4.0
MOVING = 5.0
#: Unit inclusion and place cells: >= 100 spikes in each session, peak >= 3 Hz.
MIN_SPIKES = 100
PLACE_PEAK = 3.0
#: Linearized maps: Gaussian 4 cm SD, occupancy > 20 ms (2 cm bins: params).
MAP_BANDWIDTH = 4.0
MIN_OCCUPANCY = 0.02
#: Replay: 10 ms bins; the four W-maze trajectories by start and end well.
REPLAY_BIN = 0.01
PATHS = {"1-2": (1, 2), "2-1": (2, 1), "1-3": (1, 3), "3-1": (3, 1)}
SEED = 2019
#: The lab code's sampling: 10,000 samples, R-squared averaged over 500
#: draws, 1,500 shuffles.
LAB_SAMPLES = 10_000
LAB_RESAMPLES = 500
LAB_SHUFFLES = 1500


def _runs(mask):
    """[first, last] index of each run of True, as a list of pairs."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True)
    )


def _well_visits(rec, speed):
    """The immobility period at the start well of each trial (our reading).

    The infrared beam times are not in the file. Arrival and departure are the
    first and last sample of the last run of samples within 15 cm of the well
    before the trial starts (from the previous trial's start, or the run
    epoch's); entry is the first sample from arrival with speed <= 4 cm/s
    ("the first time point when the speed fell below 4 cm/s before the arrival
    trigger"), exit the first sample of the last stretch above 4 cm/s that
    reaches departure, or the first above 4 cm/s after it ("the first time point
    when the speed rose above 4 cm/s after the departure trigger").

    Returns
    -------
    pandas.DataFrame
        One row per trial: ``epoch``, ``well``, ``correct``, ``entry``, ``exit``
        (s, NaN where the rat is never within 15 cm of the well) and
        ``duration``.
    """
    import pandas as pd

    t, pos = rec.times, rec.positions
    table = rec.trials.intervals_table().sort_values("start_time")
    table = table.reset_index(drop=True)
    run = np.asarray(rec.epochs["run"])
    starts = table["start_time"].to_numpy()
    epoch_of = np.searchsorted(run[:, 0], starts, side="right") - 1
    rows = []
    for j, trial in table.iterrows():
        well = int(trial["start_well"])
        same = j > 0 and epoch_of[j - 1] == epoch_of[j]
        first = starts[j - 1] if same else run[epoch_of[j], 0]
        k = np.flatnonzero((t >= first) & (t < starts[j]))
        entry = leave = np.nan
        centre = rec.env.regions[f"well_{well}"].data
        near = np.linalg.norm(pos[k] - centre, axis=1) <= WELL_RADIUS
        if k.size and (stretches := _runs(near)):
            a, d = k[stretches[-1][0]], k[stretches[-1][1]]
            still = np.flatnonzero(speed[a : d + 1] <= IMMOBILE)
            if not still.size:
                entry = leave = t[a]
            else:
                begin = a + still[0]
                if speed[d] <= IMMOBILE:
                    after = np.flatnonzero(speed[d:] > IMMOBILE)
                    end = d + after[0] if after.size else t.size - 1
                else:
                    end = begin + _runs(speed[begin : d + 1] > IMMOBILE)[-1][0]
                entry, leave = t[begin], t[end]
        rows.append(
            {
                "epoch": int(epoch_of[j]),
                "well": well,
                "correct": bool(trial["correct"] == 1),
                "entry": entry,
                "exit": leave,
            }
        )
    visits = pd.DataFrame(rows)
    visits["duration"] = visits["exit"] - visits["entry"]
    return visits


def _trajectory_envs(rec):
    """Each trajectory from well to well on the loader's W-maze, cut 15 cm at each end."""
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.datasets import shin_2019
    from neurospatial.datasets._recording import tracked_mask
    from neurospatial.datasets._tracks import nearest_segment

    run = np.asarray(rec.epochs["run"])
    samples = rec.positions[
        shin_2019._within(rec.times, run) & tracked_mask(rec.positions)
    ]
    nodes = shin_2019._fit_w_maze(samples)  # the loader's fit, deterministic
    distance = nearest_segment(
        samples,
        np.array([nodes[a] for a, _ in shin_2019._EDGES]),
        np.array([nodes[b] for _, b in shin_2019._EDGES]),
    ).distance
    width = max(2 * float(np.quantile(distance, 0.99)), shin_2019.TRACK_WIDTH)
    envs = {}
    for name, (start, end) in PATHS.items():
        a, b = nodes[f"well_{start}"], nodes[f"base_{start}"]
        c, d = nodes[f"base_{end}"], nodes[f"well_{end}"]
        corners = [
            a + (b - a) * WELL_RADIUS / np.linalg.norm(b - a),
            b,
            c,
            d + (c - d) * WELL_RADIUS / np.linalg.norm(c - d),
        ]
        graph = nx.Graph()
        for k, xy in enumerate(corners):
            graph.add_node(k, pos=tuple(float(v) for v in xy))
        edges = [(0, 1), (1, 2), (2, 3)]
        for u, v in edges:
            graph.add_edge(
                u, v, distance=float(np.linalg.norm(corners[u] - corners[v]))
            )
        envs[name] = Environment.from_graph(
            graph, edges, bin_size=rec.params["bin_size"], units="cm", track_width=width
        )
    return envs


def _stable_ca1(rec):
    """Positions of CA1 units with >= 100 spikes in every run epoch."""
    run = np.asarray(rec.epochs["run"])
    kept = [
        k
        for k, (train, area) in enumerate(
            zip(rec.spike_times, rec.units["brain_region"], strict=True)
        )
        if area == "CA1"
        and np.all(
            np.searchsorted(train, run[:, 1]) - np.searchsorted(train, run[:, 0])
            >= MIN_SPIKES
        )
    ]
    return np.array(kept, dtype=int)


def _templates(rec, envs, speed, cells, epoch):
    """Each trajectory's linearized rate maps in one run epoch, where it was run."""
    from neurospatial import compute_spatial_rates

    table = rec.trials.intervals_table()
    start, stop = np.asarray(rec.epochs["run"])[epoch]
    rates = {}
    for name, (a, b) in PATHS.items():
        rows = table[
            (table["start_well"] == a)
            & (table["end_well"] == b)
            & (table["start_time"] >= start)
            & (table["start_time"] < stop)
        ][["start_time", "stop_time"]].to_numpy(dtype=float)
        if not len(rows):  # not run in this session: no template
            continue
        rates[name] = compute_spatial_rates(
            envs[name],
            [rec.spike_times[k] for k in cells],
            rec.times,
            rec.positions,
            epochs=rows,
            method="binned",
            bandwidth=MAP_BANDWIDTH,
            min_occupancy=MIN_OCCUPANCY,
            speed=speed,
            min_speed=MOVING,
        )
    return rates


def _lab_code_p(posterior, positions, counts, rng):
    """The line-fit p-value of the Jadhav lab's code, for one trajectory.

    ``ProcessData/jds_replay_decoding_CA1_M.m`` of JadhavLab/PrefrontalRipples
    at 72f0f3e841117c5979d0f68624ed8f4a1a17b550 (written for Shin & Jadhav
    2024; no code of the 2019 paper is public), read for comparison and
    reproduced here only to measure how its choices change verdicts: the
    posterior of the bins with spikes, normalized within the trajectory;
    10,000 samples in all, each time bin drawn in proportion to its spike
    count and a position from its posterior; R-squared of position on bin
    number averaged over 500 such draws; 1,500 shuffles of the order of the
    bins with spikes, each drawn once; p the share of shuffles with a larger
    R-squared, with no +1. Drawing the samples is drawing how many fall in
    each time bin, then in each position bin; R-squared needs only their sums.

    Parameters
    ----------
    posterior : ndarray, shape (n_time, n_bins)
        The trajectory's conditional posterior.
    positions : ndarray, shape (n_bins,)
        Each bin's position along the trajectory, cm.
    counts : ndarray, shape (n_time,)
        Spikes per time bin.
    rng : numpy.random.Generator

    Returns
    -------
    float
    """
    keep = (counts > 0) & (posterior.sum(axis=1) > 0)
    if keep.sum() < 2:
        return 1.0
    rows = posterior[keep] / posterior[keep].sum(axis=1, keepdims=True)
    weights = counts[keep] / counts[keep].sum()
    times = np.flatnonzero(keep).astype(float)

    def r2(n_draws, order):
        samples = rng.multinomial(LAB_SAMPLES, weights, size=n_draws)
        placed = rng.multinomial(samples, rows)
        sy, syy = placed @ positions, placed @ positions**2
        sx, sxx = (samples * order).sum(axis=1), (samples * order**2).sum(axis=1)
        sxy = (sy * order).sum(axis=1)
        sy, syy = sy.sum(axis=1), syy.sum(axis=1)
        cov = sxy - sx * sy / LAB_SAMPLES
        spread = (sxx - sx**2 / LAB_SAMPLES) * (syy - sy**2 / LAB_SAMPLES)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(cov**2 / spread)

    observed = float(np.mean(r2(LAB_RESAMPLES, times[None, :])))
    orders = np.array([rng.permutation(times) for _ in range(LAB_SHUFFLES)])
    return float(np.mean(r2(LAB_SHUFFLES, orders) > observed))


def _judge(job):
    """SHIN_2019 on one candidate's joint decode, and the lab code's verdict."""
    from neurospatial.decoding import decode_jointly
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import SHIN_2019

    envs, rates, counts, centers, seed = job
    joint = decode_jointly(envs, counts, rates, dt=REPLAY_BIN, times=centers)
    result = detect_replay(
        envs,
        joint,
        centers,
        replay_method=SHIN_2019,
        spike_counts=counts,
        rng=np.random.default_rng(seed),
    )
    rng = np.random.default_rng([*seed, 72])
    lab = [
        _lab_code_p(
            np.asarray(joint.conditional[name].posterior),
            env.to_linear(env.bin_centers),
            counts.sum(axis=1),
            rng,
        )
        for name, env in envs.items()
        if name in rates
    ]
    return bool(result.is_replay), min(lab) < SHIN_2019.alpha


def _per_visit(session, *, workers=6):
    """Shin et al. 2019's per-visit counts on one rat, as the worked result defines them.

    Returns
    -------
    dict
        ``visits`` (the correct trials' visits, with ``swrs``, ``candidates``
        and ``replay`` counted by event start), ``n_place_cells`` per run
        epoch, and the session's ``n_swrs``, ``n_candidates`` and ``n_judged``.
    """
    import multiprocessing
    import warnings
    from concurrent.futures import ProcessPoolExecutor

    import ripple_detection as rd
    from ripple_detection.literature_methods import Recording as Signals
    from ripple_detection.literature_methods import bounds, shin_2019

    from neurospatial.behavior import compute_speed
    from neurospatial.datasets import load
    from neurospatial.datasets import shin_2019 as loader
    from neurospatial.datasets._load import _session_entry, _stream_url
    from neurospatial.datasets._ripple import event_rows, spike_counts

    rec = load("shin_2019", session)
    speed = np.asarray(
        compute_speed(
            rec.times, rec.positions, smooth_window=rec.params["speed_smooth_window"]
        )
    )
    visits = _well_visits(rec, speed)
    visits = visits[visits["correct"] & visits["entry"].notna()].reset_index(drop=True)
    envs = _trajectory_envs(rec)
    stable = _stable_ca1(rec)
    ca1 = rec.units[rec.units["brain_region"] == "CA1"]
    channels = tuple(f"tetrode{k} Ref ON" for k in sorted(set(ca1["tetrode"])))
    run = np.asarray(rec.epochs["run"])
    windows = loader._read_lfp_windows(
        _stream_url(_session_entry("shin_2019", session)), channels, run
    )
    swrs, candidates, unbounded, jobs, n_place, missing = [], [], [], [], [], []
    for epoch, (samples, times) in enumerate(windows):
        rates = _templates(rec, envs, speed, stable, epoch)
        peak = np.nanmax(
            [np.nanmax(r.firing_rates, axis=1) for r in rates.values()], axis=0
        )
        place = np.flatnonzero(peak >= PLACE_PEAK)
        n_place.append(int(place.size))
        missing.extend((epoch, name) for name in PATHS if name not in rates)
        counts = spike_counts(
            [rec.spike_times[k] for k in stable], times, sampling_rate=loader.LFP_RATE
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals = Signals.from_arrays(
                times,
                loader.LFP_RATE,
                lfps=samples,
                multiunit=counts,
                speed=np.interp(times, rec.times, speed),
                place_cells=place,
            )
            found = bounds(shin_2019(signals))
            chosen = bounds(shin_2019(signals, stage="decoding_candidates"))
            # The paper states no minimum time above 3 SD; the detector's 15 ms
            # is Karlsson & Frank 2009's.
            unbounded.append(
                bounds(
                    rd.Karlsson_ripple_detector(
                        signals.time,
                        signals.filtered((150.0, 250.0)),
                        signals.speed,
                        signals.fs,
                        speed_threshold=IMMOBILE,
                        minimum_duration=0.0,
                    )
                )
            )
        del signals, counts
        swrs.append(found)
        candidates.append(chosen)
        inside = np.zeros(len(chosen), dtype=bool)
        mine = visits[visits["epoch"] == epoch]
        for entry, leave in zip(mine["entry"], mine["exit"], strict=True):
            inside |= (chosen[:, 0] >= entry) & (chosen[:, 0] < leave)
        place_rates = _templates(rec, envs, speed, stable[place], epoch)
        trains = [rec.spike_times[stable[k]] for k in place]
        for index, (start, stop) in enumerate(chosen[inside]):
            spikes, centers = event_rows(
                trains, start, stop, time_bin=REPLAY_BIN, step=REPLAY_BIN
            )
            jobs.append(
                (
                    {name: envs[name] for name in place_rates},
                    place_rates,
                    spikes,
                    centers,
                    [SEED, epoch, index],
                    start,
                )
            )
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        verdicts = list(pool.map(_judge, [job[:5] for job in jobs]))
    starts = np.array([job[5] for job in jobs])
    ours = np.array([verdict[0] for verdict in verdicts], dtype=bool)
    lab = np.array([verdict[1] for verdict in verdicts], dtype=bool)

    def count(starts):
        return [
            int(np.sum((starts >= a) & (starts < b)))
            for a, b in zip(visits["entry"], visits["exit"], strict=True)
        ]

    visits["swrs"] = count(np.concatenate(swrs)[:, 0])
    visits["candidates"] = count(np.concatenate(candidates)[:, 0])
    visits["replay"] = count(starts[ours])
    visits["replay_lab_code"] = count(starts[lab])
    visits["swrs_no_minimum"] = count(np.concatenate(unbounded)[:, 0])
    return {
        "visits": visits,
        "ours_only": int(np.sum(ours & ~lab)),
        "lab_code_only": int(np.sum(lab & ~ours)),
        "n_place_cells": n_place,
        "n_swrs": int(sum(len(e) for e in swrs)),
        "n_candidates": int(sum(len(e) for e in candidates)),
        "n_judged": len(jobs),
        "untraveled": missing,
        "swr_rows": np.concatenate(swrs),
        "rec": rec,
    }


def _swr_modulation(rec, swrs):
    """Each unit's rate inside the SWRs over its rate in the run epochs."""
    run = np.asarray(rec.epochs["run"])
    inside = float(np.sum(swrs[:, 1] - swrs[:, 0]))
    total = float(np.sum(run[:, 1] - run[:, 0]))

    def spikes_in(train, rows):
        return int(
            np.sum(
                np.searchsorted(train, rows[:, 1]) - np.searchsorted(train, rows[:, 0])
            )
        )

    return np.array(
        [
            (spikes_in(train, swrs) / inside)
            / max(spikes_in(train, run) / total, 1e-12)
            for train in rec.spike_times
        ]
    )


# The files' devices carry the empty 'manufacturer' field pynwb now deprecates.
NWB_DEVICE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_loads_default_session(session_entry, assert_default_session):
    """JS14 by byte range: 72 units by tetrode, 298 trials, 8 run epochs, the W-maze."""
    from neurospatial.datasets import load

    rec = load("shin_2019")
    assert_default_session(rec, session_entry("shin_2019"))
    assert rec.provenance.url.startswith("https://api.dandiarchive.org/")
    assert rec.units["brain_region"].value_counts().to_dict() == {"CA1": 52, "PFC": 20}
    assert rec.times.size == 501_461
    assert rec.sampling_rate == pytest.approx(32.5733, rel=1e-5)
    assert len(rec.trials) == 298
    assert int(rec.trials.intervals_table()["correct"].sum()) == 258
    assert rec.epochs["run"].shape == (8, 2)
    assert rec.epochs["rest"].shape == (9, 2)
    assert len(rec.lfp.channels) == 62
    text = {c.text.split(":")[0]: c for c in rec.conversions}
    assert "a median 1.005 times that speed" in text["speed"].evidence
    assert "96.3% of samples at the paper's 4 cm/s" in text["speed"].evidence
    assert "Read as rows here: CA1 31, PFC 41" in text["units"].evidence
    assert "a median 16.0 cm/s" in text["trials"].evidence
    assert "a median 18.2 cm from the end of its end well's arm" in (
        text["trials"].evidence
    )
    assert "track_width 34.7 cm" in text["environment"].text
    assert any("for 35 of the 72 units here" in t for t in rec.watch_outs)
    assert any("75% of the tracked samples of the rest" in t for t in rec.watch_outs)


def _summary(visits):
    """The per-visit means and SDs and the shares with two or more events."""
    return {
        "n_visits": len(visits),
        **{
            name: (
                round(float(visits[column].mean()), 3),
                round(float(visits[column].std()), 3),
            )
            for name, column in (
                ("immobility_s", "duration"),
                ("swrs", "swrs"),
                ("candidates", "candidates"),
                ("replay", "replay"),
                ("replay_lab_code", "replay_lab_code"),
                ("swrs_no_minimum", "swrs_no_minimum"),
            )
        },
        "share_two_swrs": round(float(np.mean(visits["swrs"] >= 2)), 3),
        "share_two_candidates": round(float(np.mean(visits["candidates"] >= 2)), 3),
    }


def _criterion(summary):
    """Which of the six fixed conditions hold, by name."""
    from neurospatial.datasets import shin_2019

    paper = shin_2019.PAPER_PER_VISIT
    held = {
        name: abs(summary[name][0] - paper[name][0]) <= paper[name][1] / 2
        for name in ("immobility_s", "swrs", "candidates", "replay")
    }
    for name in ("share_two_swrs", "share_two_candidates"):
        held[name] = abs(summary[name] - paper[name]) <= 0.10
    return held


@pytest.fixture(scope="module")
def js14():
    pytest.importorskip("ripple_detection.literature_methods")
    return _per_visit("sub-JDS-SingleDay-JS14")


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_per_visit_counts_on_js14(js14):
    """JS14's per-visit numbers, as recorded; reads 5.5 GB of LFP by byte range."""
    from neurospatial.datasets import shin_2019

    recorded = shin_2019.PER_VISIT["JS14"]
    summary = _summary(js14["visits"])
    assert summary == recorded["summary"]
    assert js14["n_place_cells"] == recorded["n_place_cells"]
    assert js14["n_swrs"] == recorded["n_swrs"]
    assert js14["n_candidates"] == recorded["n_candidates"]
    assert js14["n_judged"] == recorded["n_judged"]
    assert _criterion(summary) == recorded["criterion"]


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lab_code_changes_verdicts_both_ways_on_js14(js14):
    """The Jadhav lab's line-fit p-value against SHIN_2019 on the same candidates."""
    from neurospatial.datasets import shin_2019

    recorded = shin_2019.PER_VISIT["JS14"]
    assert js14["ours_only"] == recorded["ours_only"]
    assert js14["lab_code_only"] == recorded["lab_code_only"]


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ca1_units_fire_in_swrs_and_pfc_units_do_not(js14):
    """Evidence for reading the units' electrodes as tetrodes: SWR modulation by area."""
    from neurospatial.datasets import shin_2019

    rec = js14["rec"]
    ratio = _swr_modulation(rec, js14["swr_rows"])
    area = rec.units["brain_region"].to_numpy()
    recorded = shin_2019.PER_VISIT["JS14"]["swr_modulation"]
    assert np.median(ratio[area == "CA1"]) == pytest.approx(recorded["CA1"], abs=1e-3)
    assert np.median(ratio[area == "PFC"]) == pytest.approx(recorded["PFC"], abs=1e-3)
    assert np.max(ratio[area == "PFC"]) < 2.0


@pytest.mark.realdata
@pytest.mark.slow
@NWB_DEVICE_DEPRECATION
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_per_visit_counts_of_the_six_rats():
    """The worked result: the paper's six rats pooled; reads about 40 GB by byte range."""
    import pandas as pd

    from neurospatial.datasets import shin_2019

    pytest.importorskip("ripple_detection.literature_methods")
    frames = []
    for rat in shin_2019.PAPER_RATS:
        result = _per_visit(f"sub-JDS-SingleDay-{rat}")
        recorded = shin_2019.PER_VISIT[rat]
        summary = _summary(result["visits"])
        assert summary == recorded["summary"], rat
        assert _criterion(summary) == recorded["criterion"], rat
        for key in (
            "n_place_cells",
            "n_swrs",
            "n_candidates",
            "n_judged",
            "ours_only",
            "lab_code_only",
        ):
            assert result[key] == recorded[key], (rat, key)
        ratio = _swr_modulation(result["rec"], result["swr_rows"])
        area = result["rec"].units["brain_region"].to_numpy()
        for region in ("CA1", "PFC"):
            assert np.median(ratio[area == region]) == pytest.approx(
                recorded["swr_modulation"][region], abs=1e-3
            ), (rat, region)
        frames.append(result["visits"])
    summary = _summary(pd.concat(frames))
    recorded = shin_2019.PER_VISIT["pooled"]
    assert summary == recorded["summary"]
    assert _criterion(summary) == recorded["criterion"]
    assert shin_2019.WORKED_RESULT["attempted"]["value"] == summary["replay"][0]
