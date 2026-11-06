"""Bhattarai et al. (2020), Figshare 10032866: synthetic cells in its layout, and the real ones."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from neurospatial.datasets import DatasetCheckError, bhattarai_2020, load
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._fetch import cache_root

#: Trials per block, and the forced trials among them, of the synthetic set.
N_IN_BLOCK = 4
N_FORCED = 1
RATE = 30.0
#: The synthetic trial: a lap along its sequence, then a rest at the reward.
LAP = 6.0
REST = 4.0
CLOCK_DIFFERENCE = 1000.0
#: Blocks of the synthetic set by session, as indices into SEQUENCES.
BLOCKS = ((0, 3), (2, 1))
DEFAULT = "rat2_set2"
#: SI Appendix, Table S1: "Performance (% correct)", in the table's order.
TABLE_PERFORMANCE = (
    81.5, 79.8, 83.2, 91.0, 89.9, 92.4,
    86.6, 78.2, 82.4, 75.6, 63.9, 77.5, 90.8, 75.6, 71.7,
    89.9, 83.3, 84.9, 84.0, 76.5, 89.9, 75.4, 83.2, 79.0, 91.5,
)  # fmt: skip
#: Draws of the level B null, and its seed, as tests/datasets/test_worked_results.py.
N_NULL = 20
NULL_SEED = 2026


def _pixels(sequence):
    """A sequence's waypoints back in VT.mat pixels, shape (n_points, 2)."""
    return (
        bhattarai_2020.sequence_path(sequence)
        * [1.0, -1.0]
        * (bhattarai_2020.PIXELS_PER_CM)
    )


def _lap(sequence, n_samples):
    """Equal steps along a sequence's waypoints, in pixels."""
    points = _pixels(sequence)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.r_[0.0, np.cumsum(lengths)]
    wanted = np.linspace(0.0, along[-1], n_samples)
    return np.column_stack(
        [np.interp(wanted, along, points[:, axis]) for axis in (0, 1)]
    )


def _bearing(xy):
    """Clockwise degrees from the camera's up, y down the image, rounded, 1 to 360."""
    dx, dy = np.gradient(xy, axis=0).T
    degrees = np.degrees(np.arctan2(dx, -dy)) % 360.0
    rounded = np.round(degrees)
    return np.where(rounded == 0.0, 360.0, rounded)


def _cell(*, aborted=0, seed=0):
    """One set's cells: tracking, trains, events and behavior rows."""
    rng = np.random.default_rng(seed)
    rows, events, tracking = [], [], []
    now = 400.0
    n_laps = int(LAP * RATE)
    n_rest = int(REST * RATE)
    for number in range(aborted):
        row = np.zeros(23)
        row[0], row[1] = number + 1, number + 1
        row[11], row[12], row[14] = N_FORCED, 3, 3
        row[21] = (now - 300.0 + number) * 1e3
        rows.append(row)
    for blocks in BLOCKS:
        correct, wrong, count = np.zeros(4), np.zeros(4), 0
        for block in blocks:
            for in_block in range(1, N_IN_BLOCK + 1):
                count += 1
                # The last trial of a block is an error: the rat ran the next
                # sequence and got nothing.
                error = in_block == N_IN_BLOCK
                ran = (block + 1) % 4 if error else block
                if in_block > N_FORCED:
                    if error:
                        wrong[ran] += 1
                    else:
                        correct[ran] += 1
                sequence = bhattarai_2020.SEQUENCES[ran]
                xy = _lap(sequence, n_laps)
                at_reward = np.tile(xy[-1], (n_rest, 1))
                path = np.vstack([xy, at_reward])
                times = now + np.arange(len(path)) / RATE
                headings = np.r_[_bearing(xy), np.full(n_rest, 90.0)]
                # A stored 0 is no direction: every second sample has none.
                headings[::2] = 0.0
                tracking.append(np.column_stack([times, path, headings]))
                start, stop = now, now + (n_laps + n_rest) / RATE
                delay_end, reward = now + 0.4 * LAP, now + LAP
                events.append([start, delay_end, reward, stop])
                row = np.zeros(23)
                row[0], row[1] = in_block, count
                row[2:6] = correct
                row[6] = correct.sum()
                row[7:11] = wrong
                row[11], row[12] = N_FORCED, 3
                row[14] = block
                row[15:22] = (reward + CLOCK_DIFFERENCE) * 1e3
                row[22] = 0 if error else 1
                rows.append(row)
                now = stop
    # beh.mat has a last trial events.mat does not.
    rows.append(rows[-1].copy())
    rows[-1][21] += 12e3
    tracking = np.vstack(tracking)
    span = (tracking[0, 0], tracking[-1, 0])
    trains = [
        np.sort(rng.uniform(*span, 100))[:, None],
        np.sort(np.r_[rng.uniform(*span, 20), rng.uniform(*span, 20)])[:, None],
    ]
    # A burster: pairs 5 ms apart, far below 5 Hz.
    firsts = np.sort(rng.uniform(span[0], span[1] - 1.0, 60))
    trains[1] = np.sort(np.r_[firsts, firsts + 0.005])[:, None]
    return tracking, trains, np.array(events), np.array(rows)


def _write(folder, cells):
    """``VT.mat``, ``spikes.mat``, ``events.mat`` and ``beh.mat`` of ``cells``."""

    def stored(values):
        array = np.empty((1, len(values)), dtype=object)
        for index, value in enumerate(values):
            array[0, index] = value
        return array

    folder.mkdir(parents=True, exist_ok=True)
    savemat(folder / "VT.mat", {"VT": stored([cell[0] for cell in cells])})
    savemat(
        folder / "spikes.mat", {"spikes": stored([stored(cell[1]) for cell in cells])}
    )
    savemat(folder / "events.mat", {"events": stored([cell[2] for cell in cells])})
    savemat(folder / "beh.mat", {"beh": stored([cell[3] for cell in cells])})
    return folder


@pytest.fixture
def archive(tmp_path):
    """A folder in the zip's layout, at ``<cache>/bhattarai_2020/<session>``."""
    return tmp_path / "bhattarai_2020" / "rat1_set1"


def test_positions_are_cm_with_y_negated(archive, session_entry):
    """x / 3.6, -y / 3.6; samples at (0, 0) stay there and the check reports them."""
    cell = _cell()
    tracking = cell[0].copy()
    tracking[5:40, 1:3] = 0.0
    tracking[5:40, 3] = 0.0
    _write(archive, [(tracking, *cell[1:])])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set1"), archive
    )
    np.testing.assert_allclose(rec.positions[:, 0], tracking[:, 1] / 3.6)
    np.testing.assert_allclose(rec.positions[:, 1], -tracking[:, 2] / 3.6)
    assert rec.position_units == "cm"
    assert np.all(rec.positions[5:40] == 0)
    bases = {c.text.split(":")[0]: c.basis for c in rec.conversions}
    assert bases["positions"] == "inferred"
    assert rec.env_kind == "inferred"
    assert rec.env.is_linearized_track


def test_heading_is_a_bearing_turned_into_the_positions_frame(archive, session_entry):
    """Running heading lies along movement; a stored 0 is NaN, and 360 deg is not."""
    _write(archive, [_cell()])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set1"), archive
    )
    tracked = np.isfinite(rec.headings)
    np.testing.assert_array_equal(tracked, _cell()[0][:, 3] != 0)
    step = np.gradient(rec.positions, axis=0)
    moving = np.linalg.norm(step, axis=1) > 0
    difference = np.angle(
        np.exp(1j * (rec.headings - np.arctan2(step[:, 1], step[:, 0])))
    )
    # Turns between waypoints spread it a little; rounding to 1 deg is small.
    assert np.median(np.abs(difference[tracked & moving])) < np.deg2rad(2.0)
    assert any(
        "so the frame is verified by movement" in text for text in rec.watch_outs
    )


def test_a_mirrored_heading_is_reported_not_corrected(archive, session_entry):
    """A heading that does not follow movement stays, and the watch-out says so."""
    tracking, *rest = _cell()
    tracking = tracking.copy()
    held = tracking[:, 3] != 0
    tracking[held, 3] = 361.0 - tracking[held, 3]
    _write(archive, [(tracking, *rest)])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set1"), archive
    )
    assert any("not verified by movement" in text for text in rec.watch_outs)


def test_trials_take_labels_from_the_behavior_rows(archive, session_entry):
    """Session, block, sequence, forced, choice and reward, row by row."""
    _write(archive, [_cell()])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set1"), archive
    )
    table = rec.trials.intervals_table()
    assert len(table) == 4 * N_IN_BLOCK
    assert table["session"].tolist() == [1] * 8 + [2] * 8
    assert table["block"].tolist() == ([1] * 4 + [2] * 4) * 2
    assert table["sequence"].tolist() == (
        ["RR"] * 4 + ["LR"] * 4 + ["RL"] * 4 + ["LL"] * 4
    )
    assert table["forced"].tolist() == [True, False, False, False] * 4
    assert table["rewarded"].tolist() == [True, True, True, False] * 4
    # An error trial's choice is the sequence the counters say the rat ran.
    assert table["choice"].tolist()[:8] == ["RR"] * 3 + ["LL"] + ["LR"] * 3 + ["RR"]
    assert table["end_known"].all()
    assert len(rec.epochs["forced_choice"]) == 4
    assert len(rec.epochs["free_choice"]) == 12
    assert len(rec.epochs["reward"]) == 16
    np.testing.assert_array_equal(rec.events["reward_start"], _cell()[2][:, 2])


def test_rows_before_the_first_trial_are_skipped(tmp_path, session_entry):
    """An aborted start: the events match beh.mat from a later row, found by the sensor."""
    folder = tmp_path / "bhattarai_2020" / "rat1_set2"
    _write(folder, [_cell(seed=1), _cell(aborted=3)])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set2"), folder
    )
    table = rec.trials.intervals_table()
    assert table["session"].tolist() == [1] * 8 + [2] * 8
    assert table["sequence"].tolist()[0] == "RR"
    assert any("the events start at beh.mat's row 4" in text for text in rec.watch_outs)


def test_row_shift_refuses_tables_that_do_not_match():
    _, _, events, behavior = _cell()
    behavior = behavior.copy()
    behavior[:, 21] = np.random.default_rng(0).permutation(behavior[:, 21]) * 50.0
    with pytest.raises(ValueError, match="trials cannot be labelled by block"):
        bhattarai_2020._row_shift(events, behavior)


def test_a_lost_trial_start_moves_only_that_boundary(archive, session_entry):
    """A row starting at its own end: it starts at its end of delay, the last ends there."""
    tracking, trains, events, behavior = _cell()
    events = events.copy()
    original = events.copy()
    events[3, 0] = events[3, 3]
    events[2, 3] = events[3, 3]
    _write(archive, [(tracking, trains, events, behavior)])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set1"), archive
    )
    table = rec.trials.intervals_table()
    assert table["start_time"].iloc[3] == original[3, 1]
    assert table["stop_time"].iloc[2] == original[3, 1]
    assert table["stop_time"].iloc[3] == original[3, 3]
    assert table["end_known"].tolist()[:4] == [True, True, False, True]
    assert len(rec.epochs["reward"]) == 15
    assert any("lost the start of 1 of 16" in text for text in rec.watch_outs)
    # The rows no longer overlap, so the clock check passes.
    run_checks(rec, stated=session_entry("bhattarai_2020", "rat1_set1").info.stated)


def test_a_spike_at_zero_stays_and_fails_the_clock_check(archive, session_entry):
    """The file's zeros are kept, sorted to the front, and the check refuses them."""
    tracking, trains, events, behavior = _cell()
    train = trains[0].ravel()
    mixed = np.insert(train, [10, 20], 0.0)[:, None]
    _write(archive, [(tracking, [mixed, trains[1]], events, behavior)])
    entry = session_entry("bhattarai_2020", "rat1_set1")
    rec = bhattarai_2020.load_session(entry, archive)
    assert rec.spike_times[0][:2].tolist() == [0.0, 0.0]
    assert rec.spike_times[0].size == train.size + 2
    assert any("unit 0: 2" in text for text in rec.watch_outs)
    with pytest.raises(DatasetCheckError, match="outside the tracked span"):
        run_checks(rec, stated=entry.info.stated)
    accepted = run_checks(
        rec, stated=entry.info.stated, accept_failed_checks=("clock",)
    )
    assert accepted.failed_checks == ("clock",)


def test_units_are_labelled_by_the_papers_rule(archive, session_entry):
    """Regular firing is not a complex-spike cell; pairs 5 ms apart are; so is the burst index."""
    _write(archive, [_cell()])
    rec = bhattarai_2020.load_session(
        session_entry("bhattarai_2020", "rat1_set1"), archive
    )
    assert rec.units["label"].tolist() == ["theta", "complex-spike"]
    assert bhattarai_2020._burst_index(np.array([0.0, 0.001, 1.0, 1.001, 2.0])) == 50.0
    # Bursting but above 5 Hz.
    train = np.sort(np.r_[np.arange(0.0, 10.0, 0.1), np.arange(0.0, 10.0, 0.1) + 0.002])
    assert bhattarai_2020._labels((train,), 10.0).tolist() == ["theta"]


def test_sequence_paths_run_from_the_trial_start_to_the_reward():
    """The rat's right is the lower pixel row while it runs toward lower x."""
    paths = {
        name: bhattarai_2020.sequence_path(name) for name in bhattarai_2020.SEQUENCES
    }
    for path in paths.values():
        np.testing.assert_allclose(path[0], paths["RR"][0])
        np.testing.assert_allclose(path[-1], paths["RR"][-1])
        length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        assert abs(length - bhattarai_2020.PAPER_MAZE_LENGTH) < 10.0

    # In the positions' frame y is negated, so "lower in pixels" is greater y.
    def on_first_ring(path):
        x = path[:, 0] * bhattarai_2020.PIXELS_PER_CM
        return path[(x > 500) & (x < 530), 1].mean()

    def on_second_ring(path):
        x = path[:, 0] * bhattarai_2020.PIXELS_PER_CM
        return path[(x > 300) & (x < 360), 1].mean()

    assert on_first_ring(paths["RR"]) > on_first_ring(paths["LL"])
    assert on_second_ring(paths["RR"]) > on_second_ring(paths["LL"])
    assert on_first_ring(paths["RL"]) > on_first_ring(paths["LR"])
    assert on_second_ring(paths["LR"]) > on_second_ring(paths["RL"])


def test_sequence_track_measures_distance_along_the_trial():
    """A 1-D track: the trial's start is at 0 and its reward at the far end."""
    track = bhattarai_2020.sequence_track("RL")
    path = bhattarai_2020.sequence_path("RL")
    start, end = np.asarray(track.to_linear(path[[0, -1]])).ravel()
    assert start < 2.0
    np.testing.assert_allclose(
        end, np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), atol=2.0
    )


def test_read_lfp_takes_one_cell_of_a_v73_file(tmp_path):
    """A cell array of references, each (2, n_time): MATLAB's columns are HDF5's rows."""
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "lfp.mat"
    with h5py.File(path, "w") as file:
        refs = file.create_group("#refs#")
        cells = []
        for index in range(3):
            t = 10.0 * index + np.arange(5) / 2000.0
            cells.append(
                refs.create_dataset(str(index), data=np.vstack([t, t * 2.0])).ref
            )
        file.create_dataset("lfp", data=np.array(cells, dtype=h5py.ref_dtype)[:, None])
    samples, times = bhattarai_2020.read_lfp((path,), ("tetrode",), index=2)
    assert samples.shape == (5, 1)
    np.testing.assert_allclose(times, 20.0 + np.arange(5) / 2000.0)
    np.testing.assert_allclose(samples[:, 0], 2.0 * times)


def test_registry_offers_every_set_in_table_order(session_entry):
    from neurospatial.datasets._catalog import sessions

    entries = sessions("bhattarai_2020")
    assert len(entries) == 25
    assert [bhattarai_2020.set_index(e.session) for e in entries] == list(range(25))
    assert sum(e.n_units for e in entries) == 698
    assert sum(e.default for e in entries) == 1
    # All sessions read the same members and share one lfp.mat.
    assert len({e.lfp_files[0].path for e in entries}) == 1
    assert {m.path for m in entries[0].members} == {
        "VT.mat",
        "spikes.mat",
        "events.mat",
        "beh.mat",
    }


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """rat 2, set 2: 42 units, two sessions of four blocks, and an LFP on request."""
    rec = load("bhattarai_2020")
    assert_default_session(rec, session_entry("bhattarai_2020"))
    assert rec.n_units == 42
    assert rec.position_units == "cm"
    assert 29.5 <= rec.sampling_rate <= 30.0
    table = rec.trials.intervals_table()
    assert len(table) == 159
    blocks = table.groupby(["session", "block"])["sequence"].agg(["first", "nunique"])
    assert len(blocks) == 8
    assert (blocks["nunique"] == 1).all()
    for session in (1, 2):
        assert set(blocks.loc[session, "first"]) == set(bhattarai_2020.SEQUENCES)
    # A rewarded trial is one whose choice was the block's sequence.
    free = table[~table["forced"]]
    assert ((free["choice"] == free["sequence"]) == free["rewarded"]).all()
    assert rec.lfp.ripple_channels == ("tetrode",)


@pytest.mark.realdata
def test_reward_starts_where_the_head_is_at_the_reward_site():
    """The events are on the tracking clock: the head clusters there at their times."""
    rec = load("bhattarai_2020")
    rows = np.searchsorted(rec.times, rec.events["reward_start"])
    distance = np.linalg.norm(
        rec.positions[rows] - bhattarai_2020.sequence_path("RR")[-1], axis=1
    )
    assert np.median(distance) < 10.0


@pytest.mark.realdata
def test_trials_run_the_arms_their_labels_name():
    """L and R mean the rat's own turns: a label names the arm the head went through."""
    rec = load("bhattarai_2020")
    table = rec.trials.intervals_table()
    x_cm = rec.positions[:, 0]
    y_cm = rec.positions[:, 1]
    first, second = [], []
    for row in table.itertuples():
        run = (rec.times >= row.delay_end) & (rec.times < row.reward_start)
        arm_1 = run & (x_cm > 139.0) & (x_cm < 150.0) & (y_cm < 0.0)
        arm_2 = run & (x_cm > 86.0) & (x_cm < 103.0) & (y_cm < 0.0)
        first.append(np.median(y_cm[arm_1]) if arm_1.any() else np.nan)
        second.append(np.median(y_cm[arm_2]) if arm_2.any() else np.nan)
    first, second = np.array(first), np.array(second)
    right_1 = table["choice"].str[0].to_numpy() == "R"
    right_2 = table["choice"].str[1].to_numpy() == "R"
    # In the negated frame the rat's right, heading to lower x, is greater y.
    assert np.nanmin(first[right_1]) > np.nanmedian(first[~right_1])
    assert np.nanmin(second[right_2]) > np.nanmedian(second[~right_2])


def _default_folder():
    """The cached members of the default session, loading it if needed."""
    load("bhattarai_2020")
    return cache_root() / "bhattarai_2020" / DEFAULT


@pytest.mark.realdata
def test_cells_are_in_the_order_of_table_s1():
    """Each cell's rewarded free-choice share tracks Table S1, row by row."""
    folder = _default_folder()
    ours = []
    for index in range(25):
        raw = bhattarai_2020.read_session(folder, index)
        free = raw.behavior[:, 0] > raw.behavior[:, 11]
        ours.append(100.0 * np.mean(raw.behavior[free, 22] == 1))
    difference = np.abs(np.array(ours) - TABLE_PERFORMANCE)
    assert difference.max() < 2.0
    assert np.sort(difference)[-2] < 1.0  # all but the set with an aborted block
    # No other order comes close: the rows differ by up to 28 points.
    assert np.ptp(TABLE_PERFORMANCE) > 25.0


@pytest.mark.realdata
def test_the_papers_rule_gives_its_648_of_698_units():
    """Level A, on all 25 cells: "648 of them (92.8%)" (Results, "Spatial Firing")."""
    folder = _default_folder()
    n_units = n_complex_spike = 0
    for index in range(25):
        raw = bhattarai_2020.read_session(folder, index)
        trains = tuple(np.sort(train) for train in raw.trains)
        duration = raw.tracking[-1, 0] - raw.tracking[0, 0]
        labels = bhattarai_2020._labels(trains, duration)
        n_units += labels.size
        n_complex_spike += int(np.sum(labels == "complex-spike"))
    assert n_units == 698
    assert n_complex_spike == 648


@pytest.mark.realdata
def test_lfp_is_on_the_clock_of_the_spikes():
    """The population rate rises at the peaks of the channel's ripple-band power."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, find_peaks, sosfiltfilt

    from neurospatial.datasets._ripple import _speed_on_grid

    rec = load("bhattarai_2020")
    samples, times = rec.lfp.fetch()
    assert samples.shape == (times.size, 1)
    np.testing.assert_allclose(np.median(np.diff(times)), 1 / 2000.0)
    assert abs(times[0] - rec.times[0]) < 0.1
    # The rails: the file's values never pass them, and often reach them.
    assert np.abs(samples).max() == 327.67
    at_rail = np.abs(samples[:, 0]) == 327.67
    assert 0.2 < at_rail.mean() < 0.25
    # As stored, in a length of whole Neuralynx records, and mostly in running.
    assert samples.dtype == np.float64
    assert times.size % 512 == 0
    speed = _speed_on_grid(rec, times)
    assert at_rail[speed >= 4.0].mean() > 10 * at_rail[speed < 4.0].mean()
    sos = butter(4, (100.0, 250.0), btype="band", fs=2000.0, output="sos")
    power = uniform_filter1d(sosfiltfilt(sos, samples[:, 0]) ** 2, 100)
    z = (power - power.mean()) / power.std()
    peaks, _ = find_peaks(z, height=3.0, distance=400)
    spikes = np.sort(np.concatenate(rec.spike_times))
    edges = np.arange(-0.2, 0.2001, 0.02)
    counts = np.zeros(edges.size - 1)
    for moment in times[peaks]:
        left, right = np.searchsorted(spikes, (moment - 0.2, moment + 0.2))
        counts += np.histogram(spikes[left:right] - moment, edges)[0]
    assert np.abs(edges[np.argmax(counts)] + 0.01) <= 0.03
    assert counts.max() > 1.3 * np.r_[counts[:5], counts[-5:]].mean()


def _replay_events(rec):
    """Each block's candidates by the paper's recipe, and what decoding them needs.

    Complex-spike units; per block, rate maps from forced-choice and correct
    free-choice trials on the block's linear path (2 cm bins, 6 cm bandwidth,
    above 4 cm/s); place cells are those neurospatial finds a place field in.
    One channel stands for the paper's two: ``bhattarai_2020_ripples`` needs two
    columns, and the same one twice averages to the channel's own power.

    Returns
    -------
    list of dict
        One per block: ``track``, ``rates`` (n_place, n_bins), ``prior``,
        ``trains``, and ``events`` (n_events, 2).
    """
    from ripple_detection.literature_methods import run_method

    from neurospatial import compute_spatial_rates
    from neurospatial.datasets._ripple import ripple_recording
    from neurospatial.encoding import detect_place_fields

    samples, times = rec.lfp.fetch()
    table = rec.trials.intervals_table()
    chosen = rec.units["label"].to_numpy() == "complex-spike"
    ids = rec.units.index.to_numpy()[chosen]
    trains = [
        train for train, keep in zip(rec.spike_times, chosen, strict=True) if keep
    ]
    # (0, 0) is the tracker's lost sample: neither the maze nor time on it.
    positions = np.array(rec.positions)
    positions[np.all(positions == 0, axis=1)] = np.nan
    blocks = []
    for _, block in table.groupby(["session", "block"]):
        track = rec.params["sequence_tracks"][block["sequence"].iloc[0]]
        template = block[block["forced"] | block["rewarded"]]
        rates = compute_spatial_rates(
            track,
            trains,
            rec.times,
            positions,
            method="binned",
            bandwidth=rec.params["bandwidth"],
            min_speed=rec.params["min_speed"],
            epochs=template[["start_time", "stop_time"]].to_numpy(),
            unit_ids=ids,
            warn_on_drop=False,
        )
        place = [
            i for i in range(ids.size) if detect_place_fields(track, rates[i]).fields
        ]
        free = block[~block["forced"] & block["end_known"]]
        reward = free[["reward_start", "stop_time"]].to_numpy()
        blocks.append(
            {
                "track": track,
                "rates": np.nan_to_num(np.asarray(rates.firing_rates)[place]),
                "prior": rates.spatial_support * track.bin_sizes,
                "trains": [trains[i] for i in place],
                "ids": ids[place],
                "reward": reward[reward[:, 1] > reward[:, 0]],
            }
        )
    both = np.column_stack([samples[:, 0], samples[:, 0]])
    recording = ripple_recording(rec, both, times)
    for block in blocks:
        recording.place_cells = rec.units.index.isin(block["ids"])
        found = run_method(
            "bhattarai_2020", recording, behavior_intervals=block["reward"]
        )
        block["events"] = found[["start_time", "end_time"]].to_numpy()
    return blocks


def _judged(block, order, *, rng):
    """Each event's verdict under ``BHATTARAI_2020``, the rate maps in ``order``."""
    from neurospatial.datasets._ripple import event_rows
    from neurospatial.decoding import decode_position
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import BHATTARAI_2020

    method = BHATTARAI_2020
    verdicts = []
    for start, stop in block["events"]:
        # An event ends on the sample of its last spike, which counts to the sample's end.
        counts, centers = event_rows(
            block["trains"],
            float(start),
            float(stop) + 1 / bhattarai_2020.LFP_RATE,
            time_bin=method.decoding["time_bin"],
            step=method.decoding["time_step"],
        )
        if centers.size < 2:
            verdicts.append(False)
            continue
        decoded = decode_position(
            block["track"],
            counts,
            block["rates"][order],
            dt=method.decoding["time_bin"],
            prior=block["prior"],
            times=centers,
        )
        verdicts.append(
            detect_replay(
                block["track"],
                np.asarray(decoded.posterior),
                centers,
                replay_method=method,
                spike_counts=counts,
                rng=rng,
            ).is_replay
        )
    return verdicts


def _share_and_null(blocks):
    """Replays and candidates, with the share's null over permuted cell identities."""
    n_events = sum(len(block["events"]) for block in blocks)
    observed = sum(
        sum(_judged(block, np.arange(len(block["trains"])), rng=0)) for block in blocks
    )
    null = []
    for child in np.random.SeedSequence(NULL_SEED).spawn(N_NULL):
        generator = np.random.default_rng(child)
        replays = 0
        for block in blocks:
            order = generator.permutation(len(block["trains"]))
            replays += sum(_judged(block, order, rng=0))
        null.append(replays / n_events)
    return observed, n_events, np.array(null)


@pytest.mark.realdata
def test_replay_on_the_one_channel_pins_the_attempt():
    """The paper's recipe and preset on the file's channel: the criterion is met.

    Fixed before any null ran: the share of candidates judged replay must exceed
    the 95th percentile of the same share with cell identities permuted. It does,
    by two events, and ``WORKED_RESULT`` says why that is not firm. This pins the
    numbers, so a change to the detector, the loader or the preset that moves them
    fails here.
    """
    pytest.importorskip("ripple_detection.literature_methods")
    rec = load("bhattarai_2020")
    blocks = _replay_events(rec)
    replays, n_events, null = _share_and_null(blocks)
    record = bhattarai_2020.WORKED_RESULT
    assert (n_events, replays) == (37, 11)
    assert record["attempted"]["value"] == round(replays / n_events, 3)
    bar = float(np.percentile(null, 95))
    assert round(bar, 3) == 0.246
    assert replays / n_events > bar

    # Reported beside it: the same bursts without the SWR requirement.
    import ripple_detection as rd

    from neurospatial.datasets._ripple import spike_counts

    _, times = rec.lfp.fetch()
    counts = spike_counts(rec.spike_times, times, sampling_rate=rec.lfp.sampling_rate)
    for block in blocks:
        bursts = rd.detect_silence_bounded_events(
            times,
            counts,
            rec.lfp.sampling_rate,
            minimum_silence=0.06 + 1 / rec.lfp.sampling_rate,
            window=0.3,
            window_end_rule="last_spike",
            units=rec.units.index.isin(block["ids"]),
            minimum_active_units=5,
        )
        bounds = bursts[["start_time", "end_time"]].to_numpy()
        inside = [
            np.any((block["reward"][:, 0] <= start) & (stop <= block["reward"][:, 1]))
            for start, stop in bounds
        ]
        block["events"] = bounds[inside]
    replays, n_events, null = _share_and_null(blocks)
    assert (n_events, replays) == (534, 147)
    assert replays / n_events > np.percentile(null, 95)


@pytest.mark.realdata
def test_swrs_per_trial_on_this_set():
    """The like-for-like comparison with the paper's 0.61 and 0.13 per trial."""
    pytest.importorskip("ripple_detection.literature_methods")
    from ripple_detection.literature_methods import run_method

    from neurospatial.datasets._ripple import ripple_recording

    rec = load("bhattarai_2020")
    blocks = _replay_events(rec)
    samples, times = rec.lfp.fetch()
    union = np.unique(np.concatenate([block["ids"] for block in blocks]))
    recording = ripple_recording(
        rec, np.column_stack([samples[:, 0]] * 2), times, place_cells=union
    )
    starts = run_method("bhattarai_2020_ripples", recording)["start_time"].to_numpy()
    table = rec.trials.intervals_table()
    free = table[
        ~table["forced"]
        & table["end_known"]
        & table["reward_start"].notna()
        & (table["stop_time"] > table["reward_start"])
    ]
    in_trial = (starts[:, None] >= free["reward_start"].to_numpy()) & (
        starts[:, None] < free["stop_time"].to_numpy()
    )
    per_trial = in_trial.sum(axis=0)
    correct = free["rewarded"].to_numpy(dtype=bool)
    assert round(per_trial[correct].mean(), 2) == 0.49
    assert round(per_trial[~correct].mean(), 2) == 0.08


def test_worked_result_record_is_complete():
    """The fields of the other records; an attempt that failed says how it differs."""
    record = bhattarai_2020.WORKED_RESULT
    assert set(record) == {
        "status",
        "level",
        "finding",
        "quote",
        "source",
        "paper",
        "ours",
        "criterion",
        "parameters",
        "unmatched",
        "attempted",
        "planted_recall",
        "why",
    }
    assert record["status"] == "not possible yet"
    assert record["level"] is None
    assert record["ours"] is None
    assert record["why"]
    assert set(record["attempted"]) == {"value", "criterion", "parameters", "differs"}
    assert record["attempted"]["differs"]
    # The paper's per-trial rate is the comparison, and the rails are not the cause.
    assert "0.61" in record["attempted"]["differs"]
    assert "do not cause the low count" in record["why"]
    planted = record["planted_recall"]
    assert round(planted["judged"] / planted["planted"], 3) == 0.186
    assert "18.6%" in planted["reading"]
    assert "not evidence of its absence" in planted["reading"]
