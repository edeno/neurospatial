"""Huszár et al. (2022), DANDI 000552: synthetic NWB files and the real sessions."""

from __future__ import annotations

import datetime
import pickle
import warnings

import numpy as np
import pytest

from neurospatial import compute_spatial_rates
from neurospatial.datasets import huszar_2022, load
from neurospatial.datasets._checks import run_checks

#: The figure eight's corners in the synthetic camera frame, cm.
LEFT, RIGHT, BOTTOM, TOP = 15.0, 85.0, 10.0, 80.0
MIDDLE = 0.5 * (BOTTOM + TOP)
RATE = 30.0
#: The two recordings, s.
FIRST = (1000.0, 1300.0)
SECOND = (4000.0, 4100.0)
END = 5000.0


def _loop(fraction, *, upper):
    """A point on one loop of the figure eight, ``fraction`` of the way round it.

    Starts at the right end of the centre bar, along the bar, then around the
    upper or lower loop. The linearized position is ``fraction * 222``.
    """
    y = TOP if upper else BOTTOM
    corners = np.array(
        [
            (RIGHT, MIDDLE),
            (LEFT, MIDDLE),
            (LEFT, y),
            (RIGHT, y),
            (RIGHT, MIDDLE),
        ]
    )
    lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    edges = np.r_[0.0, np.cumsum(lengths)] / lengths.sum()
    return np.column_stack(
        [np.interp(fraction, edges, corners[:, axis]) for axis in (0, 1)]
    )


def _path(times):
    """Laps of 20 s, one on each loop in turn, with a small seeded wobble."""
    laps = (times - times[0]) / 20.0
    fraction = np.mod(laps, 1.0)
    upper = np.floor(laps).astype(int) % 2 == 0
    xy = np.where(
        upper[:, None], _loop(fraction, upper=True), _loop(fraction, upper=False)
    )
    wobble = np.random.default_rng(0).normal(0.0, 0.5, xy.shape)
    return xy + wobble, 222.0 * fraction


def _write_session(
    path,
    *,
    unit="cm",
    second_recording=False,
    with_epochs=True,
    nan_trial=False,
    zero_state=False,
    recordings="a, l, t, e, r, n, a, t, i, o, n",
):
    """An NWB file laid out as DANDI 000552's."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    nwb = pynwb.NWBFile(
        session_description="synthetic",
        identifier="synthetic",
        session_start_time=datetime.datetime(
            2020, 10, 19, tzinfo=datetime.timezone.utc
        ),
        subject=Subject(subject_id="e14_2m2", species="Mus musculus"),
    )
    times = np.arange(*FIRST, 1 / RATE)
    xy, linear = _path(times)
    if second_recording:
        later = np.arange(*SECOND, 1 / RATE)
        straight = np.column_stack(
            [np.full(later.size, 500.0), 60.0 + 50.0 * np.sin(later / 7.0)]
        )
        xy = np.vstack([xy, straight])
        linear = np.r_[linear, 55.0 + 50.0 * np.sin(later / 7.0)]
        times = np.r_[times, later]
    behavior = nwb.create_processing_module(
        name="behavior",
        description=(
            f"The behavior of the subject for the following recordings: {recordings}"
        ),
    )
    for container, name, data, what in (
        ("SubjectPosition", "SpatialSeries", xy, "(x,y) coordinates"),
        ("LinearizedPosition", "LinearizedSpatialSeries", linear, "Linearization"),
    ):
        position = Position(name=container)
        position.add_spatial_series(
            SpatialSeries(
                name=name,
                description=what,
                data=data,
                timestamps=times,
                reference_frame="Arbitrary, camera",
                unit=unit,
            )
        )
        behavior.add(position)
    states = TimeIntervals(name="SleepStates", description="Sleep state of the animal.")
    states.add_column(name="label", description="Sleep state.")
    rows = [(0.0, 400.0, "Awake"), (401.0, 800.0, "Non-REM"), (801.0, 900.0, "REM")]
    rows += [(901.0, 3000.0, "Awake"), (3001.0, 3500.0, "Non-REM")]
    rows += [(3501.0, END, "Awake")]
    for start, stop, label in rows:
        states.add_interval(start_time=start, stop_time=stop, label=label)
    if zero_state:
        states.add_interval(start_time=950.0, stop_time=950.0, label="Awake")
    behavior.add(states)
    ecephys = nwb.create_processing_module(
        name="ecephys", description="No description."
    )
    ripples = TimeIntervals(name="Ripples", description="Ripples and their metrics")
    ripples.add_column(name="peaks", description="Peak of the ripple.")
    for start in (500.0, 1200.0, 3100.0):
        ripples.add_interval(
            start_time=start, stop_time=start + 0.04, peaks=start + 0.02
        )
    ecephys.add(ripples)
    if with_epochs:
        for start, stop in ((0.0, 990.0), (990.0, 1310.0), (1310.0, END)):
            nwb.add_epoch(start_time=start, stop_time=stop)
    nwb.add_trial_column(name="visited_arm", description="arm")
    nwb.add_trial_column(name="recordings", description="recording")
    laps = np.arange(FIRST[0], FIRST[1] - 20.0, 20.0)
    for k, start in enumerate(laps):
        nwb.add_trial(
            start_time=float(start),
            stop_time=float(start) + 19.0,
            visited_arm="left" if k % 2 == 0 else "right",
            recordings=1,
        )
    if nan_trial:
        nwb.add_trial(
            start_time=np.nan, stop_time=np.nan, visited_arm="left", recordings=1
        )
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(
        name="Group 1", description="shank", location="unknown", device=device
    )
    nwb.add_electrode_column(name="channel_name", description="name")
    nwb.add_electrode_column(name="group_name", description="group")
    nwb.add_electrode_column(name="rel_x", description="x")
    nwb.add_electrode_column(name="rel_y", description="y")
    for k in range(2):
        nwb.add_electrode(
            location="unknown",
            group=group,
            channel_name=str(k + 1),
            group_name="Group 1",
            rel_x=10.0,
            rel_y=-20.0 * k,
        )
    nwb.add_unit_column(name="unit_name", description="name")
    nwb.add_unit_column(name="clu_id", description="cluster")
    nwb.add_unit_column(name="group_id", description="group")
    rng = np.random.default_rng(1)
    for k in range(4):
        nwb.add_unit(
            spike_times=np.sort(rng.uniform(0.0, END, 2000)),
            unit_name=str(k + 1),
            clu_id=float(k),
            group_id="Group1",
        )
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    return path


@pytest.fixture
def entry(monkeypatch, session_entry):
    """The registry's default session, with the reward table stubbed.

    The files' ``LabeledEvents`` type belongs to an ndx-events version that can
    no longer write one, so ``test_loads_default_session`` covers that table.
    """
    monkeypatch.setattr(
        huszar_2022,
        "_rewards",
        lambda nwb: (
            np.array([1100.0, 1120.0]),
            np.array(["left_reward", "right_reward"]),
        ),
    )
    return session_entry("huszar_2022")


@pytest.fixture
def synthetic(tmp_path):
    return _write_session(tmp_path / "session.nwb")


def test_loads_positions_epochs_events_and_trials(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    assert rec.positions.shape == (rec.times.size, 2)
    assert rec.position_units == "cm"
    assert rec.linearized.shape == rec.times.shape
    np.testing.assert_array_equal(rec.epochs["maze"], [[990.0, 1310.0]])
    np.testing.assert_array_equal(rec.params["analysis_epochs"], [[990.0, 1310.0]])
    assert rec.epochs["trials"].shape[0] == len(rec.trials)
    np.testing.assert_array_equal(
        rec.epochs["nrem"], [[401.0, 800.0], [3001.0, 3500.0]]
    )
    np.testing.assert_allclose(
        rec.events["ripples"], [[500, 500.04], [1200, 1200.04], [3100, 3100.04]]
    )
    np.testing.assert_allclose(rec.events["ripple_peaks"], [500.02, 1200.02, 3100.02])
    np.testing.assert_array_equal(rec.events["left_rewards"], [1100.0])
    assert rec.units["brain_region"].eq("CA1").all()
    assert rec.units["label"].isna().all()
    assert {"unit_name", "clu_id", "group_id"} <= set(rec.units.columns)
    assert rec.lfp is None
    assert rec.env.units == "cm"


def test_off_maze_runs_around_the_tracked_stretches(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    first, last = rec.times[0], rec.times[-1]
    np.testing.assert_allclose(rec.epochs["off_maze"], [[0.0, first], [last, END]])


def test_train_and_test_hold_both_arms(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    train, test = rec.params["train_epochs"], rec.params["test_epochs"]
    # Trials alternate left, right, so even and odd rows would split them by arm;
    # alternating within each arm takes rows 0, 1 then 4, 5.
    np.testing.assert_array_equal(train[:2, 0], [1000.0, 1020.0])
    np.testing.assert_array_equal(test[:2, 0], [1040.0, 1060.0])


def test_fitted_track_recovers_the_figure_eight(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    graph = rec.env.binning.linearization.graph
    nodes = np.array([graph.nodes[name]["pos"] for name in graph.nodes])
    for corner in ((LEFT, TOP), (RIGHT, TOP), (LEFT, BOTTOM), (RIGHT, BOTTOM)):
        assert np.linalg.norm(nodes - corner, axis=1).min() < 2.0
    for end in ((LEFT, MIDDLE), (RIGHT, MIDDLE)):
        assert np.linalg.norm(nodes - end, axis=1).min() < 2.0
    assert graph.number_of_edges() == 7


def test_seam_of_the_linear_position_is_continuous_on_the_environment(synthetic, entry):
    """The two sides of the file's 0/222 seam are neighbours on the track graph."""
    rec = huszar_2022.load_session(entry, synthetic)
    tracked = rec.env.bin_at(rec.positions) >= 0
    just_before = np.flatnonzero(tracked & (rec.linearized > 216.0))[0]
    just_after = np.flatnonzero(tracked & (rec.linearized < 6.0))[0]
    before, after = rec.env.bin_at(rec.positions[[just_before, just_after]])
    assert abs(rec.linearized[just_before] - rec.linearized[just_after]) > 200.0
    apart = rec.env.distance_between(bins=(int(before), int(after)))
    assert apart < 20.0


@pytest.mark.parametrize(
    ("description", "names"),
    [
        ("recordings: a, l, t, e, r, n, a, t, i, o, n", ["alternation"]),
        ("recordings: alternation, linearMaze", ["alternation", "linearMaze"]),
        ("recordings: alternation, alternation", ["alternation", "alternation"]),
    ],
)
def test_recordings_named_reads_the_description(description, names):
    assert huszar_2022._recordings_named(description) == names


def test_second_recording_keeps_its_samples_and_says_how_many_lie_outside(
    tmp_path, entry
):
    path = _write_session(
        tmp_path / "two.nwb",
        second_recording=True,
        recordings="alternation, linearMaze",
    )
    rec = huszar_2022.load_session(entry, path)
    np.testing.assert_array_equal(rec.epochs["alternation"], [[990.0, 1310.0]])
    assert rec.epochs["linear_maze"][0, 0] == pytest.approx(1310.0)
    assert rec.positions[-1, 0] == 500.0  # kept as stored
    assert any(
        "2 recordings (alternation, linearMaze)" in text
        and "100% of the samples of the later" in text
        for text in rec.watch_outs
    )


def test_file_without_an_epochs_table_uses_the_tracked_span(tmp_path, entry):
    rec = huszar_2022.load_session(
        entry, _write_session(tmp_path / "none.nwb", with_epochs=False)
    )
    np.testing.assert_allclose(
        rec.epochs["maze"], [[rec.times[0], rec.times[-1]]], atol=1e-9
    )
    assert any("no epochs table" in text for text in rec.watch_outs)


def test_trials_without_a_time_are_left_out_and_counted(tmp_path, entry):
    rec = huszar_2022.load_session(
        entry, _write_session(tmp_path / "nan.nwb", nan_trial=True)
    )
    assert np.isfinite(rec.epochs["trials"]).all()
    n_rows = rec.epochs["trials"].shape[0]
    assert any(
        f"1 of the file's {n_rows + 1} trials have no start or stop" in text
        for text in rec.watch_outs
    )


def test_zero_length_sleep_rows_are_left_out_and_counted(tmp_path, entry):
    rec = huszar_2022.load_session(
        entry, _write_session(tmp_path / "zero.nwb", zero_state=True)
    )
    assert not np.any(rec.epochs["awake"][:, 0] == 950.0)
    assert any("1 SleepStates rows start where" in text for text in rec.watch_outs)


def test_watch_outs_count_the_seam_and_yang_inclusion(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    jumps = int(np.sum(np.abs(np.diff(rec.linearized)) > huszar_2022.SEAM_STEP))
    assert jumps > 0
    assert any(f"100 cm {jumps} times" in text for text in rec.watch_outs)
    assert any("has 4 units and" in text for text in rec.watch_outs)


def test_watch_out_says_the_labels_are_not_in_the_file(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    assert any("which units are pyramidal" in text for text in rec.watch_outs)


def test_refuses_positions_not_in_cm(tmp_path, entry):
    path = _write_session(tmp_path / "metres.nwb", unit="m")
    with pytest.raises(ValueError, match="not cm"):
        huszar_2022.load_session(entry, path)


def test_checks_pass_on_the_synthetic_session(synthetic, entry):
    rec = huszar_2022.load_session(entry, synthetic)
    checked = run_checks(rec, stated=entry.info.stated)
    assert checked.failed_checks == ()


def test_recording_with_an_lfp_source_pickles(synthetic, entry, tmp_path):
    from dataclasses import replace

    from neurospatial.datasets._lfp import LFPFile

    with_lfp = replace(
        entry,
        lfp_files=(
            LFPFile(
                path="raw.nwb", sha256="0" * 64, size_bytes=1000, asset_id="asset-id"
            ),
        ),
    )
    rec = huszar_2022.load_session(with_lfp, synthetic)
    assert list(rec.lfp.channels) == ["ch1", "ch2"]
    assert rec.lfp.channels["ch2"] == "Group 1, probe position (10, -20) µm"
    assert rec.lfp.unit == "uV"
    assert rec.lfp.ripple_channels == ()
    assert pickle.loads(pickle.dumps(rec.lfp)).channels == rec.lfp.channels
    assert any("byte range" in text for text in rec.watch_outs)


def test_read_lfp_scales_to_microvolts_in_the_order_asked(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from pynwb.ecephys import LFP, ElectricalSeries

    nwb = pynwb.NWBFile(
        session_description="s",
        identifier="s",
        session_start_time=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(
        name="Group 1", description="shank", location="unknown", device=device
    )
    nwb.add_electrode_column(name="channel_name", description="name")
    for k in range(3):
        nwb.add_electrode(location="unknown", group=group, channel_name=str(k + 1))
    region = nwb.create_electrode_table_region(region=[0, 1, 2], description="all")
    stored = np.arange(30, dtype=np.int16).reshape(10, 3)
    series = ElectricalSeries(
        name="ElectricalSeriesLFP",
        data=stored,
        electrodes=region,
        rate=1250.0,
        starting_time=0.0,
        conversion=1.95e-07,
    )
    lfp = LFP()
    nwb.create_processing_module(name="ecephys", description="lfp").add(lfp)
    lfp.add_electrical_series(series)
    path = tmp_path / "raw.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    samples, times = huszar_2022._read_lfp((path,), ("ch3", "ch1"))
    np.testing.assert_allclose(samples, stored[:, [2, 0]] * 0.195)
    np.testing.assert_allclose(times, np.arange(10) / 1250.0)


# Real sessions -------------------------------------------------------------

NWB_DEVICE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)
#: The dandiset's session with the fewest units: few enough for the
#: full-resolution call, whose float64 counts take 5.4 GB here.
SMALL_SESSION = "sub-e14-2m3_ses-e14-2m3-201017"
#: The session with the smallest sibling LFP file (5.6 GB), and a channel of it.
LFP_SESSION = "sub-e16-3m2_ses-e16-3m2-211211"
LFP_CHANNEL = "ch5"


def _nrem_trigger_events(rec, peaks, *, eligible=None):
    """``yang_2024``, its counts pooled so a session of 300 units fits in memory.

    ``ripple_detection`` keeps one float64 per sample per unit and copies the
    selection: 190 GB for 376 units over 7 hours at 1 kHz, twice that in the copy. Every step here is
    that method's, on one pooled column, except the five-cell rule, which counts
    units from their spike times. ``test_pooled_detection_is_yang_2024`` shows
    the two give the same events.

    Parameters
    ----------
    rec : Recording
    peaks : ndarray, shape (n_ripples,)
    eligible : ndarray, shape (n, 2), optional

    Returns
    -------
    ndarray, shape (n_events, 2)
    """
    import ripple_detection as rd
    from ripple_detection import literature_methods as lm

    from neurospatial.datasets._ripple import spike_counts

    grid = _one_ms_grid(rec)
    pooled = spike_counts(
        [np.sort(np.concatenate(rec.spike_times))], grid, sampling_rate=1000.0
    ).astype(np.float64)
    recording = lm.Recording.from_arrays(
        grid, 1000.0, multiunit=pooled, sleep_intervals=rec.epochs["nrem"]
    )
    trace = lm.population_trace(recording, bin_width=0.001, smoothing_sigma=0.015)
    mask = recording.intervals_to_mask(rec.epochs["nrem"])
    found = trace.detect(
        threshold=3.0,
        normalization_mask=mask[rd.core.nearest_sample_index(grid, trace.time)],
        minimum_duration=0.0,
        minimum_event_duration=0.05,
        maximum_duration=0.5,
        speed_threshold=np.inf,
    )
    events = found[["start_time", "end_time"]].to_numpy()
    # A count belongs to its own millisecond, so the last sample's holds to its end.
    stops = events[:, 1] + 0.001
    active = sum(
        (np.searchsorted(train, stops) > np.searchsorted(train, events[:, 0])).astype(
            int
        )
        for train in rec.spike_times
    )
    events = rd.require_times_inside(events[active >= 5], peaks)
    return events if eligible is None else rd.require_inside(events, eligible)


def _one_ms_grid(rec):
    """A 1 kHz time grid from 0 s past the session's last spike, shape (n_time,)."""
    last = max(float(train[-1]) for train in rec.spike_times if train.size)
    return np.arange(int(np.ceil(last * 1000.0)) + 2) / 1000.0


def _awake_on_maze(rec):
    """The rows of ``epochs['maze']`` cut to the file's Awake rows, sorted, disjoint."""
    maze, awake = rec.epochs["maze"], rec.epochs["awake"]
    starts = np.maximum(maze[:, 0, None], awake[:, 0])
    stops = np.minimum(maze[:, 1, None], awake[:, 1])
    rows = np.column_stack([starts[stops > starts], stops[stops > starts]])
    return rows[np.argsort(rows[:, 0])]


def _overlapped(rows, others):
    """Whether each of ``rows`` touches one of the sorted, disjoint ``others``."""
    left = np.searchsorted(others[:, 0], rows[:, 1], side="right") - 1
    return (left >= 0) & (others[np.clip(left, 0, None), 1] >= rows[:, 0])


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_loads_default_session(session_entry, assert_default_session):
    entry = session_entry("huszar_2022")
    rec = load("huszar_2022")
    assert_default_session(rec, entry)
    assert rec.n_units == 113
    assert len(rec.spike_times) == 113
    assert rec.times.size == 46660
    assert rec.times[0] == pytest.approx(7891.2619)
    np.testing.assert_allclose(rec.epochs["maze"], [[7859.264, 9492.1279667]])
    assert rec.epochs["trials"].shape == (21, 2)
    assert rec.events["ripples"].shape == (4297, 2)
    assert rec.events["rewards"].size == 21
    assert rec.events["left_rewards"].size + rec.events["right_rewards"].size == 21
    assert rec.linearized.min() == pytest.approx(0.3117, abs=1e-3)
    assert rec.linearized.max() == pytest.approx(221.957, abs=1e-3)
    assert rec.lfp is None
    assert rec.env.n_bins == 177


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_rewards_land_at_the_corners_the_arm_names():
    """The file's reward labels name two corners: the clock and labels agree."""
    rec = load("huszar_2022")
    at_left = rec.positions[np.searchsorted(rec.times, rec.events["left_rewards"])]
    at_right = rec.positions[np.searchsorted(rec.times, rec.events["right_rewards"])]
    centres = np.median(at_left, axis=0), np.median(at_right, axis=0)
    assert np.linalg.norm(centres[0] - centres[1]) > 40.0
    assert np.median(np.linalg.norm(at_left - centres[0], axis=1)) < 10.0
    assert np.median(np.linalg.norm(at_right - centres[1], axis=1)) < 10.0


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_ripples_fall_at_the_population_rate_peak():
    """Spikes and the file's ripple table share a clock: firing peaks at the peaks."""
    rec = load("huszar_2022")
    spikes = np.sort(np.concatenate(rec.spike_times))
    peaks = rec.events["ripple_peaks"]
    near = np.searchsorted(spikes, peaks + 0.02) - np.searchsorted(spikes, peaks - 0.02)
    far = np.searchsorted(spikes, peaks - 0.08) - np.searchsorted(spikes, peaks - 0.12)
    assert near.mean() > 2 * far.mean()


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_seam_jumps_are_steps_of_the_line_not_of_the_animal():
    rec = load("huszar_2022")
    jumps = np.flatnonzero(np.abs(np.diff(rec.linearized)) > huszar_2022.SEAM_STEP)
    assert jumps.size == 62
    steps = np.linalg.norm(rec.positions[jumps + 1] - rec.positions[jumps], axis=1)
    assert np.median(steps) < 1.0


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_running_rate_maps_warn_about_the_recording_outside_the_maze():
    rec = load("huszar_2022")
    with pytest.warns(UserWarning, match=r"fell outside the position time"):
        compute_spatial_rates(rec.env, rec.spike_times, rec.times, rec.positions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_rates(
            rec.env,
            rec.spike_times,
            rec.times,
            rec.positions,
            epochs=rec.params["analysis_epochs"],
            min_speed=rec.params["min_speed"],
        )


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_pooled_detection_is_yang_2024():
    """The pooled route finds the same events as ``run_method``, event by event.

    ``ripple_recording`` puts counts on the LFP's grid and this session has no
    LFP; ``yang_2024`` reads none, so the counts go on its own 1 ms grid with
    the adapter's ``spike_counts``.
    """
    pytest.importorskip("ripple_detection.literature_methods")
    from ripple_detection.literature_methods import Recording, run_method

    from neurospatial.datasets._ripple import spike_counts

    rec = load("huszar_2022", SMALL_SESSION)
    eligible = _awake_on_maze(rec)
    peaks = rec.events["ripple_peaks"]
    grid = _one_ms_grid(rec)
    recording = Recording.from_arrays(
        grid,
        1000.0,
        multiunit=spike_counts(rec.spike_times, grid, sampling_rate=1000.0),
        pyramidal=np.ones(len(rec.spike_times), dtype=bool),
        sleep_intervals=rec.epochs["nrem"],
        external_ripples=np.column_stack(
            [rec.events["ripples"], rec.events["ripple_peaks"]]
        ),
    )
    expected = run_method("yang_2024", recording, behavior_intervals=eligible)
    pooled = _nrem_trigger_events(rec, peaks, eligible=eligible)
    assert len(expected) > 0
    np.testing.assert_allclose(
        pooled, expected[["start_time", "end_time"]].to_numpy(), atol=1e-9
    )


@pytest.fixture(scope="module")
def yang_events():
    """Yang et al.'s retained events on the session both archives hold, as rows.

    From the paper's GitHub repository, whose analysis file holds the 20 ms bins
    of the ripple events by event id (MATLAB 7.3, HDF5, no pickle). Each row runs
    from the first bin's timestamp to the end of the last bin, one bin wide
    either way, since the file does not say what a timestamp marks.

    Returns
    -------
    rows : ndarray, shape (n_events, 2)
    ids : ndarray, shape (n_events,)
    """
    h5py = pytest.importorskip("h5py")
    from neurospatial.datasets._fetch import fetch

    path = fetch(
        huszar_2022.YANG_VALIDATION["url"],
        huszar_2022.YANG_VALIDATION["file"],
        sha256=huszar_2022.YANG_VALIDATION["sha256"],
        subdir="huszar_2022/yang_2024",
    )
    with h5py.File(path, "r") as file:
        times = np.asarray(file["timestamps"]).ravel()
        labels = np.asarray(file["ripple_ind"]).ravel().astype(np.int64)
    ids = np.unique(labels[labels > 0])
    first = np.searchsorted(np.sort(labels), ids)
    order = np.argsort(labels, kind="stable")
    last = np.searchsorted(np.sort(labels), ids, side="right") - 1
    rows = np.column_stack([times[order[first]] - 0.01, times[order[last]] + 0.03])
    return rows, ids


@pytest.fixture(scope="module")
def validation_session():
    return load("huszar_2022", huszar_2022.YANG_VALIDATION["session"])


@pytest.fixture(scope="module")
def our_events(validation_session):
    return _nrem_trigger_events(
        validation_session, validation_session.events["ripple_peaks"]
    )


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_yang_analysis_file_is_on_this_clock_with_these_units(validation_session):
    """Its matrix has one column per unit of the file, in the file's order."""
    h5py = pytest.importorskip("h5py")
    from neurospatial.datasets._fetch import fetch

    path = fetch(
        huszar_2022.YANG_VALIDATION["url"],
        huszar_2022.YANG_VALIDATION["file"],
        sha256=huszar_2022.YANG_VALIDATION["sha256"],
        subdir="huszar_2022/yang_2024",
    )
    with h5py.File(path, "r") as file:
        matrix = np.asarray(file["data_noZ"], dtype=np.float64)  # (n_units, n_bins)
        times = np.asarray(file["timestamps"]).ravel()
        labels = np.asarray(file["ripple_ind"]).ravel()
    bins = labels > 0
    assert matrix.shape[0] == validation_session.n_units
    theirs = matrix[:, bins].sum(axis=1)
    ours = np.array(
        [
            np.sum(
                np.searchsorted(train, times[bins] + 0.03)
                - np.searchsorted(train, times[bins] - 0.01)
            )
            for train in validation_session.spike_times
        ],
        dtype=np.float64,
    )
    assert np.corrcoef(theirs, ours)[0, 1] > 0.999


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_agreement_with_yang_event_by_event(yang_events, our_events):
    """The measured agreement, pinned: a change that moves it fails here."""
    record = huszar_2022.YANG_VALIDATION
    rows, _ = yang_events
    assert len(rows) == record["n_theirs"]
    assert len(our_events) == record["n_ours"]
    theirs_found = _overlapped(rows, our_events).mean()
    ours_found = _overlapped(our_events, rows).mean()
    assert round(float(theirs_found), 3) == record["share_of_theirs_we_overlap"]
    assert round(float(ours_found), 3) == record["share_of_ours_they_overlap"]


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_the_gap_in_our_share_is_their_files_selection(yang_events, our_events):
    """Why only half of ours are theirs: their file holds a subset of their own ids.

    Their ids run past the events they kept, and each kept id sits where the
    same rank of our events does, so the ids count a list as long as ours that
    the file then filters; none of our events under 100 ms is in it.
    """
    record = huszar_2022.YANG_VALIDATION
    rows, ids = yang_events
    assert ids.max() == record["largest_id"]
    assert len(ids) / ids.max() < 0.5
    found = _overlapped(rows, our_events)
    rank = np.searchsorted(our_events[:, 0], rows[found, 1], side="right") - 1
    assert np.corrcoef(ids[found], rank)[0, 1] > 0.999
    short = our_events[(our_events[:, 1] - our_events[:, 0]) < 0.1]
    assert _overlapped(short, rows).mean() == 0.0


@pytest.fixture(scope="module")
def replay_analysis():
    """The session, its candidates, the units with a field and their rate maps."""
    pytest.importorskip("ripple_detection.literature_methods")
    from neurospatial.encoding import has_place_field

    record = huszar_2022.REPLAY_ANALYSIS
    rec = load("huszar_2022", record["session"])
    params = rec.params
    windows = _nrem_trigger_events(
        rec, rec.events["ripple_peaks"], eligible=_awake_on_maze(rec)
    )
    trains = [
        train
        for train in rec.spike_times
        if has_place_field(
            rec.env,
            train,
            rec.times,
            rec.positions,
            epochs=params["analysis_epochs"],
            min_speed=params["min_speed"],
            warn_on_drop=False,
        )
    ]
    rates = compute_spatial_rates(
        rec.env,
        trains,
        rec.times,
        rec.positions,
        epochs=params["analysis_epochs"],
        min_speed=params["min_speed"],
    )
    return rec, windows, trains, rates


def _n_judged(rec, trains, windows, rates):
    """How many windows ``PFEIFFER_FOSTER_2013`` judges trajectories."""
    from neurospatial.decoding import bin_spikes_in_time, decode_position
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import PFEIFFER_FOSTER_2013

    time_bin = PFEIFFER_FOSTER_2013.decoding["time_bin"]
    step = PFEIFFER_FOSTER_2013.decoding["time_step"]
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
            rec.env, decoded.posterior, centres, replay_method=PFEIFFER_FOSTER_2013
        )
        n_judged += bool(verdict.is_replay)
    return n_judged


def _shifted(trains, span, draw, *, common):
    """Spikes circularly shifted over ``span`` by at least 10 s: the fixed nulls."""
    start, stop = span
    length = stop - start
    rng = np.random.default_rng(np.random.SeedSequence(2022, spawn_key=(draw, common)))
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


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_awake_candidates_are_judged_trajectories_above_shifted_spikes(replay_analysis):
    """Level B, our analysis: fixed before any event was decoded, and met."""
    rec, windows, trains, rates = replay_analysis
    record = huszar_2022.REPLAY_ANALYSIS
    assert len(windows) == record["n_events"]
    assert len(trains) == record["n_units"]
    observed = _n_judged(rec, trains, windows, rates)
    assert observed == record["n_judged"]
    span = (
        float(rec.params["analysis_epochs"][0, 0]),
        float(rec.params["analysis_epochs"][-1, 1]),
    )
    own = [
        _n_judged(rec, _shifted(trains, span, draw, common=False), windows, rates)
        for draw in range(20)
    ]
    shared = [
        _n_judged(rec, _shifted(trains, span, draw, common=True), windows, rates)
        for draw in range(20)
    ]
    assert tuple(own) == record["null_own_offsets"]
    assert tuple(shared) == record["null_common_offset"]
    assert observed > np.quantile(own, 0.95)
    assert observed > np.quantile(shared, 0.95)
    assert huszar_2022.WORKED_RESULT["ours"] == round(observed / len(windows), 3)


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_held_out_laps_decode_to_the_animal(replay_analysis):
    """The diagnostic fixed with the property: the maps can show a trajectory."""
    from neurospatial.behavior import compute_speed
    from neurospatial.decoding import bin_spikes_in_time, decode_position
    from neurospatial.ops.distance import pairwise_distances

    rec, _, trains, _ = replay_analysis
    params = rec.params
    record = huszar_2022.REPLAY_ANALYSIS["held_out_laps"]
    rates = compute_spatial_rates(
        rec.env,
        trains,
        rec.times,
        rec.positions,
        epochs=params["train_epochs"],
        min_speed=params["min_speed"],
    )
    decoded_bins, true_bins = [], []
    speed = compute_speed(rec.times, rec.positions)
    for start, stop in params["test_epochs"]:
        counts, _ = bin_spikes_in_time(
            trains, dt=params["dt"], t_start=float(start), t_stop=float(stop)
        )
        centres = start + params["dt"] / 2 + params["dt"] * np.arange(len(counts))
        decoded = decode_position(
            rec.env, np.asarray(counts), rates, dt=params["dt"], times=centres
        )
        index = np.clip(np.searchsorted(rec.times, centres), 0, rec.times.size - 1)
        running = speed[index] >= params["min_speed"]
        decoded_bins.append(decoded.posterior.argmax(axis=1)[running])
        true_bins.append(rec.env.bin_at(rec.positions[index[running]]))
    guess, truth = np.concatenate(decoded_bins), np.concatenate(true_bins)
    on_track = truth >= 0
    guess, truth = guess[on_track], truth[on_track]
    matrix = pairwise_distances(rec.env.connectivity, np.arange(rec.env.n_bins))
    error = float(np.median(matrix[guess, truth]))
    rng = np.random.default_rng(2022)
    shuffled = float(np.median(matrix[guess, rng.permutation(truth)]))
    assert error == pytest.approx(record["median_error_cm"], abs=0.01)
    assert shuffled == pytest.approx(record["shuffled_error_cm"], abs=0.01)


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_lfp_is_read_by_byte_range_and_shows_the_ripples():
    """One channel of a sibling file: 130-200 Hz power peaks at the table's peaks."""
    from scipy.signal import butter, sosfiltfilt

    rec = load("huszar_2022", LFP_SESSION)
    samples, times = rec.lfp.fetch([LFP_CHANNEL], stream=True)
    assert times[1] - times[0] == pytest.approx(1 / 1250.0)
    sos = butter(4, (130.0, 200.0), btype="bandpass", fs=1250.0, output="sos")
    power = sosfiltfilt(sos, samples[:, 0]) ** 2
    peaks = rec.events["ripple_peaks"]
    index = np.searchsorted(times, peaks)
    keep = (index > 200) & (index < times.size - 200)
    near = np.array([power[k - 12 : k + 13].mean() for k in index[keep]])
    far = np.array([power[k - 150 : k - 125].mean() for k in index[keep]])
    assert np.median(near) > 3 * np.median(far)
