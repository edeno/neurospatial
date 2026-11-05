"""Carey et al. (2019), DataLad MotivationalT: synthetic files, and the real session.

The real-data tests hold the validation against the paper's own candidate
events and the worked result. The paper's candidate files are in its code
repository (github.com/vandermeerlab/papers, Git LFS), whose license speaks of
software and not of data, so they are downloaded here, pinned by SHA-256, and
not redistributed. Each stores the paper's joint score sample by sample
(``evt.data``), which lets the score be compared before any event logic.
"""

from __future__ import annotations

import dataclasses
import struct

import numpy as np
import pytest
from scipy.io import savemat

from neurospatial.datasets import carey_2019, load
from neurospatial.datasets._catalog import sessions
from neurospatial.datasets._neuralynx import (
    HEADER_BYTES,
    SAMPLES_PER_RECORD,
    read_video_track,
)

SESSION = "R050-2014-03-29"
CONV = (2.5, 2.0)
VIDEO_HZ = 30.0
#: The paper's candidate file for the default session, at its code's last commit.
PAPERS_COMMIT = "3aebc8e454c3b863b68c20c57421e6c16a6e809b"
#: SHA-256 of each paper session's candidate zip, from its Git LFS pointer.
CANDIDATES_SHA256 = {
    "R042-2013-08-17": "c48aa5667ad786603c9112b7fe8f069a56d4ecbbbad5752d0f63c2d394b46b36",
    "R042-2013-08-18": "b55c56f3ca2e5e94bd3d0fee1c7fa56127dcf2930c3ae0398cf0fdbd188f29fe",
    "R042-2013-08-19": "0c36308706a886276090377be6495361e01353bc23ad42b98b7fe99043c53489",
    "R042-2013-08-20": "f81327c025c1d30ddd146d81ea14ffa57a05e835e39c84a5b9535d2f0ec72b88",
    "R042-2013-08-21": "9e0beed6b0614b19adf7886419383e06d0f59aeba3cbbbc54ec939c45d23519e",
    "R044-2013-12-21": "889cf35b52f4294a2965cff88862d11768d394ba865fe898c9730cb7c3037d4f",
    "R044-2013-12-22": "c3b8cd641992704470354a7550f2bf92ed80bf5f751cd02454d017c6f1998d89",
    "R050-2014-03-29": "567fd61bbb37c012e9ab4d31ccddd1cf4d19f5182190ec9617797fc16fb4776f",
    "R050-2014-03-30": "c55d9f52149df7031843173f5b42a82923f66785da012653f5823eb5c7b4ce4c",
    "R050-2014-03-31": "c60d0b2bbe5c8d72a803e82d40cdb0206e0bec91df69ed830ae39363bf9777f0",
    "R050-2014-04-01": "e8129a1f017a2a5420d6dbfc0e7327bf53fae4d8ad6ea3f31e632b1311605113",
    "R050-2014-04-02": "795e521953f07c30e248835e353ab34e375065cfc0286f6c18a81a13e60ba3a4",
    "R050-2014-04-03": "db1cb57a2834c7a5f9cadb000dcd58ddec3155de1f22232765c37eceaccd685d",
    "R064-2015-04-18": "a009ac912b3ad581165cf4a7170861e456cc6b85436c861fd931cd3fd0d85e2a",
    "R064-2015-04-19": "c0fbc26c8e51d511129fa7901775a9f860631ba7ed9afacd6bb1e9717f203418",
    "R064-2015-04-20": "72fa949566b51040bc9816236b7b9741dd3bc3329b926fcd56edf21d651520bb",
    "R064-2015-04-21": "06e5338780557ce33d310d314557219d78f15681a0ff808646054521cb211d36",
    "R064-2015-04-22": "f02c5f03c9c2551ab7c1fc38602b050542f02ed85f74810135ac79ef78e69856",
    "R064-2015-04-23": "38de21b30968002e733eab2043c11bdc5dd93a63d13d9cb9772ecb5bf4f872e7",
}
#: Draws of each null, and its seed, as tests/datasets/test_worked_results.py.
N_NULL = 20
NULL_SEED = 2026
#: The window around a candidate that is decoded: a sequence must fit in it.
PAD = 0.25
#: Decoding rows: counts in 5 ms bins, summed over five of them.
STEP, N_SUMMED = 0.005, 5
#: Generate_DecSeq.m: fewest gated spikes for a unit, and the track's ends left
#: out of the rate maps, from the start and the end of the trajectory, cm.
MIN_SPIKES = 25
TRACK_ENDS = (20.0, 15.0)

KEYS = """
ExpKeys.RestrictionType = 'food'; % note, 'what' was *withheld*
ExpKeys.Session = 'standard'; % 'pit', 'reversal\xe9
ExpKeys.Layout = 'foodLeft'; % 'foodRight'
ExpKeys.Pedestal = 'R'; % 'L'
ExpKeys.waterVolume = [];
ExpKeys.badTrials = [2]; % interfered with
ExpKeys.pathlength = 334; % in cm; distance from start to finish
ExpKeys.patharms = 254; % in cm
ExpKeys.realTrackDims = [139 185]; % x width and y width
ExpKeys.convFact = [2.5 2]; % x and y
ExpKeys.prerecord = [0 10]; % timestamps for prerecord
ExpKeys.task = [12.5 60]; % timestamps for task
ExpKeys.postrecord = [62 70]; % timestamps for postrecord
ExpKeys.goodSWR = {'R050-2014-03-29-CSC03a.ncs' 'R050-2014-03-29-CSC04a.ncs'}; % best first
ExpKeys.goodTheta = ExpKeys.goodSWR; % same channels
"""


def _iv(rows) -> dict:
    rows = np.asarray(rows, dtype=float)
    return {"tstart": rows[:, 0], "tend": rows[:, 1]}


def _paths():
    """The drawn trajectories, cm: a 100 cm stem and two bent 60 cm arms."""
    stem = np.column_stack([np.linspace(0, 100, 200), np.full(200, 50.0)])
    out = np.column_stack([np.full(100, 100.0), np.linspace(0, 40, 100)])
    across = np.column_stack([np.linspace(100, 80, 60), np.full(60, 40.0)])
    arm = np.vstack([out[1:], across])
    return (
        np.vstack([stem, arm * [1, -1] + np.array([0, 50])]),
        np.vstack([stem, arm + np.array([0, 50])]),
    )


def _write_video(path, xy, times) -> None:
    body = b"".join(
        struct.pack("<HHHQ", 0x800, 0x1000, 1800, round(t * 1e6))
        + bytes(400 * 4)
        + struct.pack("<hiii", 0, int(x), int(y), 0)
        + bytes(50 * 4)
        for (x, y), t in zip(xy, times, strict=True)
    )
    path.write_bytes(b"-SamplingFrequency 29.97\n".ljust(HEADER_BYTES, b"\0") + body)


def _write_events(path, rows) -> None:
    records = b"".join(
        struct.pack("<hhhQhhhhh", 0, 0, 0, round(time * 1e6), 11, 0, 0, 0, 0)
        + bytes(8 * 4)
        + text.encode().ljust(128, b"\0")
        for time, text in rows
    )
    path.write_bytes(bytes(HEADER_BYTES) + records)


def _write_t(path, seconds) -> None:
    ticks = np.round(np.asarray(seconds) / 1e-4).astype(">u4")
    path.write_bytes(b"%%BEGINHEADER\n% T-file\n%%ENDHEADER\n" + ticks.tobytes())


@pytest.fixture
def session(tmp_path):
    """A session's folder and its registry entry."""
    folder = tmp_path / "carey_2019" / SESSION
    folder.mkdir(parents=True)
    (folder / "R050_2014_03_29_keys.m").write_bytes(KEYS.encode("latin-1"))
    left, right = _paths()
    trials = [[15.0, 20.0], [30.0, 35.0], [45.0, 50.0]]
    rests = [[22.0, 28.0], [37.0, 43.0], [52.0, 58.0]]
    centres = np.linspace(1.0, 9.0, 50)
    metadata = {
        "taskvars": {
            "trial_iv": _iv(trials),
            "trial_iv_L": _iv([trials[0], trials[2]]),
            "trial_iv_R": _iv(trials[1:2]),
            "rest_iv": _iv(rests),
            "rest_iv_pedL": _iv([rests[0], rests[2]]),
            "rest_iv_pedR": _iv(rests[1:2]),
            "sequence": np.array(["L", "R", "L"], dtype=object),
        },
        "coord": {
            "coordL_cm": left.T,
            "coordR_cm": right.T,
            "coordL": (left * CONV).T,
            "coordR": (right * CONV).T,
            "chp_cm": left[199],
            "chp": left[199] * CONV,
        },
        "SWRtimes": _iv(np.column_stack([centres - 0.04, centres + 0.04])),
        "SWRfreqs": {"freqs1": np.zeros(60)},
    }
    savemat(folder / f"{SESSION}-metadata.mat", {"metadata": metadata})
    times = np.arange(0.0, 70.0, 1 / VIDEO_HZ)
    frame = np.linspace(0, 1, times.size)[:, None]
    cm = left[0] + frame * (left[-1] - left[0])
    xy = np.round(cm * CONV).astype(int) + 1
    xy[5] = 0  # a frame the tracker missed
    _write_video(folder / "VT1.nvt", xy, times)
    _write_events(
        folder / f"{SESSION}-Events.nev",
        [
            (0.0, "Starting Recording"),
            (15.5, carey_2019.EVENT_TEXTS["center_photobeam"]),
            (19.0, carey_2019.EVENT_TEXTS["food_dispensed"]),
        ],
    )
    _write_t(folder / f"{SESSION}-TT05_2.t", np.arange(1.0, 69.0, 0.5))
    _write_t(folder / f"{SESSION}-TT05_10._t", np.arange(1.25, 69.0, 0.5))
    _write_t(folder / f"{SESSION}-TT04_1.t", np.arange(1.1, 69.0, 0.5))
    default = next(e for e in sessions("carey_2019") if e.default)
    entry = dataclasses.replace(default, session=SESSION, verified=False)
    return folder, entry


def test_keys_are_read_as_text_and_never_run():
    keys = carey_2019.parse_keys(KEYS)
    assert keys["RestrictionType"] == "food"
    assert keys["Session"] == "standard"
    assert keys["pathlength"] == 334.0
    assert keys["convFact"].tolist() == [2.5, 2.0]
    assert keys["waterVolume"].size == 0
    assert keys["goodSWR"][0] == "R050-2014-03-29-CSC03a.ncs"
    assert keys["goodTheta"] == keys["goodSWR"]  # an alias, taken not evaluated
    assert carey_2019.parse_keys("system('rm -rf /')\n") == {}
    assert carey_2019.parse_keys("ExpKeys.day = now();\n") == {"day": None}


def test_positions_divide_by_the_keys_factors_and_lost_frames_are_nan():
    x, y = np.array([250, 0, 100]), np.array([100, 40, 0])
    positions = carey_2019._positions_cm(x, y, np.array(CONV))
    assert positions[0].tolist() == [100.0, 50.0]
    assert np.isnan(positions[1:]).all()


def test_track_graph_is_a_t_of_the_drawn_paths():
    left, right = _paths()
    graph = carey_2019.track_graph(left, right, left[199])
    assert graph.degree("choice") == 3
    total = sum(distance for *_, distance in graph.edges(data="distance"))
    drawn = sum(
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        for path in (left[:200], left[199:], right[199:])
    )
    assert total == pytest.approx(drawn, abs=2.0)


def test_trajectory_environment_runs_from_the_stem_to_the_arms_end():
    left, _ = _paths()
    env = carey_2019.trajectory_environment(left, track_width=10.0, name="left")
    assert env.is_linearized_track
    assert env.units == "cm"
    linear = np.asarray(env.to_linear(left[[10, 300, -1]]))
    assert np.all(np.diff(linear) > 0)
    assert linear[-1] - linear[0] == pytest.approx(
        np.linalg.norm(np.diff(left[10:], axis=0), axis=1).sum(), abs=4.0
    )


@pytest.mark.parametrize(
    ("arms", "bad", "left", "right"),
    [
        ("LRLR", [], [0, 2], [1, 3]),
        ("LRRRL", [2], [0, 4], [2, 3]),  # bad trial 2 is row 1
        ("RRLLLL", [], [2, 3], [0, 1]),  # the nearest left trials to rows 0 and 1
        ("LLLR", [], [2], [3]),
    ],
)
def test_matched_trials_take_the_nearest_of_the_commoner_side(arms, bad, left, right):
    matched = carey_2019.matched_trials(
        np.array(list(arms), dtype=object), np.array(bad, dtype=int)
    )
    assert matched[0].tolist() == left
    assert matched[1].tolist() == right


def test_a_left_arm_on_the_right_of_travel_is_named_mirrored(session):
    left, right = _paths()
    # The stem runs along +x and the left arm to -y: a right turn with y up.
    mirrored = carey_2019.track_graph(left, right, left[199])
    assert carey_2019._left_is_right_of_travel(mirrored)
    assert not carey_2019._left_is_right_of_travel(
        carey_2019.track_graph(right, left, left[199])
    )
    folder, entry = session
    rec = carey_2019.load_session(entry, folder)
    assert any("mirrored" in text for text in rec.watch_outs)


def test_units_are_every_cluster_with_the_files_quality(session):
    folder, entry = session
    rec = carey_2019.load_session(entry, folder)
    assert rec.units.index.tolist() == ["TT04_1", "TT05_2", "TT05_10"]
    assert rec.units["quality"].tolist() == ["good", "good", "questionable"]
    assert rec.spike_times[2][:2] == pytest.approx([1.25, 1.75])


def test_a_session_keeps_its_epochs_events_and_trials(session):
    folder, entry = session
    rec = carey_2019.load_session(entry, folder)
    assert rec.epochs["task"].tolist() == [[12.5, 60.0]]
    assert rec.epochs["rest_left_pedestal"].shape == (2, 2)
    assert rec.trials.arm.tolist() == ["L", "R", "L"]
    assert rec.trials.bad.tolist() == [False, True, False]
    assert rec.epochs["matched_left"].size == 0  # one right trial, and it is bad
    assert rec.events["center_photobeam"].tolist() == [15.5]
    assert rec.events["food_dispensed"].tolist() == [19.0]
    assert rec.events["hand_labelled_swr"].shape == (50, 2)
    assert rec.params["restriction"] == "food"
    assert rec.params["layout"] == "foodLeft"
    assert rec.params["in_paper"]
    assert np.isnan(rec.positions[5]).all()


def test_nothing_of_the_lfp_is_fetched_and_the_ripple_channel_is_the_first_good_one(
    session,
):
    folder, entry = session
    rec = carey_2019.load_session(entry, folder)
    assert rec.lfp.ripple_channels == ("CSC03a", "CSC04a")
    assert rec.lfp.sampling_rate == 2000.0
    assert "the paper's ripple channel" in rec.lfp.channels["CSC03a"]
    assert rec.lfp.channels["HS1R1"].startswith("reference channel")
    assert not list(folder.glob("*.ncs"))


def test_headstage_rows_are_cut_to_the_recording():
    rows = np.array([[0.0, 5.0], [8.0, np.inf], [200.0, np.inf]])
    kept = carey_2019._within(rows, np.array([3.0, 100.0]))
    assert kept.tolist() == [[3.0, 5.0], [8.0, 100.0]]


def test_lfp_reader_scales_each_channel_by_its_own_header(tmp_path):
    def write(name, volts_per_bit, value):
        record = struct.pack("<QIII", 1_000_000, 0, 2000, SAMPLES_PER_RECORD)
        record += np.full(SAMPLES_PER_RECORD, value, dtype="<i2").tobytes()
        header = f"-ADBitVolts {volts_per_bit}\n".encode()
        path = tmp_path / name
        path.write_bytes(header.ljust(HEADER_BYTES, b"\0") + record)
        return path

    first = write("R050-2014-03-29-CSC03a.ncs", 1e-7, 10)
    second = write("R050-2014-03-29-HS1R1.ncs", 2e-7, 10)
    samples, times = carey_2019.read_session_lfp((first, second), ("HS1R1", "CSC03a"))
    assert samples[0] == pytest.approx([2.0, 1.0])
    assert times[0] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def rec():
    return load("carey_2019")


def _paper_candidates(session):
    """The paper's candidates for a session: its ``evt`` struct."""
    import zipfile

    from scipy.io import loadmat

    from neurospatial.datasets._fetch import cache_root, fetch

    name = f"{session}-candidates.zip"
    archive = fetch(
        "https://media.githubusercontent.com/media/vandermeerlab/papers/"
        f"{PAPERS_COMMIT}/Carey_etal_submitted/SWRcandidates/{name}",
        name,
        sha256=CANDIDATES_SHA256[session],
        subdir="carey_2019/paper_candidates",
    )
    folder = cache_root() / "carey_2019" / "paper_candidates"
    member = f"{session}-candidates.mat"
    if not (folder / member).exists():
        with zipfile.ZipFile(archive) as opened:
            opened.extract(member, folder)
    return loadmat(
        folder / member, squeeze_me=True, struct_as_record=False, variable_names=["evt"]
    )["evt"]


@pytest.fixture(scope="module")
def paper():
    return _paper_candidates(SESSION)


def _intervals_of(evt) -> np.ndarray:
    return np.column_stack([evt.tstart, evt.tend])


def _share_overlapped(events, others) -> float:
    """The share of ``events`` that some row of ``others`` overlaps at all."""
    events = events[np.argsort(events[:, 0])]
    others = others[np.argsort(others[:, 0])]
    started = np.searchsorted(others[:, 0], events[:, 1], side="right")
    ended = np.searchsorted(others[:, 1], events[:, 0], side="left")
    return float(np.mean(started > ended))


def _mua_score(rec, times) -> np.ndarray:
    """``amMUA.m``, written from it: the capped spike density above its slow part.

    Each spike goes to its nearest sample; each unit is convolved with a
    Gaussian of 40 samples (5 SD on each side) and capped at 2 spikes' peak.
    The sum, less its 125 ms smooth (capped at 4 cells) and one cell's cap, is
    scaled by the sum's mean and floored at 0.
    """
    from scipy.signal import oaconvolve

    def kernel(radius, sd):
        taps = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2.0 * sd**2))
        return taps / taps.sum()

    n_time = times.size
    cap = 2.0 / 40.0 / np.sqrt(2.0 * np.pi)
    spikes = kernel(200, 40)
    total = np.zeros(n_time)
    for train in rec.spike_times:
        after = np.clip(np.searchsorted(times, train), 1, n_time - 1)
        nearest = after - (times[after] - train > train - times[after - 1])
        counts = np.bincount(nearest, minlength=n_time).astype(np.float64)
        total += np.minimum(oaconvolve(counts, spikes, mode="same"), cap)
    slow = oaconvolve(np.minimum(4.0 * cap, total), kernel(3000, 250), mode="same")
    return np.maximum(0.0, (total - slow - cap) / total.mean())


def _adaptive_speed(times, x, *, window=1.0, smoothing=0.5) -> np.ndarray:
    """``dxdt.m``, written from it, for one coordinate.

    For each sample, the backward slope over 3 to ``window`` s of samples whose
    linear fit has the smallest squared error, then a boxcar of ``smoothing``
    s. Time steps count as their mean, as there.
    """
    step = float(np.nanmean(np.diff(times)))
    n_window = int(np.ceil(window / step))
    best = np.full(x.size, np.inf)
    slope = np.zeros(x.size)

    def lagged(lag):
        return np.r_[np.full(lag, np.nan), x[:-lag]]

    for span in range(3, n_window + 1):
        candidate = (lagged(span) - x) / span
        error = sum((lagged(k) - x + candidate * k) ** 2 for k in range(1, span + 1))
        error = error / span
        better = error < best
        best[better] = error[better]
        slope[better] = candidate[better]
    n_smooth = int(np.ceil(smoothing / step))
    return np.convolve(slope / step, np.ones(n_smooth) / n_smooth, mode="same")


def _papers_speed(times) -> np.ndarray:
    """The paper's speed, pixels/s, on ``times``: dxdt of the tracked frames' pixels."""
    from neurospatial.datasets._fetch import cache_root

    track = read_video_track(cache_root() / "carey_2019" / SESSION / "VT1.nvt")
    tracked = (track.x != 0) & (track.y != 0)
    clock = track.times[tracked]
    x = track.x[tracked].astype(np.float64)
    y = track.y[tracked].astype(np.float64)
    speed = np.hypot(_adaptive_speed(clock, x), _adaptive_speed(clock, y))
    return np.interp(times, clock, speed, left=np.nan, right=np.nan)


def _trial_epochs(rec, side) -> np.ndarray:
    """A side's matched trials, each from 1 s before its next centre photobeam."""
    rows = np.array(rec.epochs[f"matched_{side}"])
    centre = rec.events["center_photobeam"]
    following = np.minimum(np.searchsorted(centre, rows[:, 0]), centre.size - 1)
    rows[:, 0] = centre[following] - 1.0
    return rows[rows[:, 0] < rows[:, 1]]


def _rates_on_side(rec, side, trains):
    """One trajectory's track, the units kept, and their speed-gated rate maps."""
    from neurospatial import compute_spatial_rates

    env = carey_2019.trajectory_environment(
        rec.params[f"{side}_trajectory"],
        track_width=rec.params["track_width"],
        name=side,
    )
    placed = np.zeros(rec.times.size, dtype=bool)
    tracked = np.flatnonzero(np.isfinite(rec.positions).all(axis=1))
    placed[tracked] = np.asarray(env.bin_at(rec.positions[tracked])) >= 0
    linear = np.full(rec.times.size, np.nan)
    linear[placed] = np.asarray(env.to_linear(rec.positions[placed]))
    length = float(np.max(np.asarray(env.to_linear(env.bin_centers))))
    kept = (linear >= TRACK_ENDS[0]) & (linear <= length - TRACK_ENDS[1])
    positions = np.array(rec.positions)
    positions[~kept] = np.nan

    epochs = _trial_epochs(rec, side)
    min_speed = rec.params["min_speed"]
    moving = np.nan_to_num(rec._speed) > min_speed
    usable = kept & moving
    units = []
    for unit, train in enumerate(trains):
        sample = np.clip(np.searchsorted(rec.times, train), 0, rec.times.size - 1)
        row = np.searchsorted(epochs[:, 0], train, side="right") - 1
        in_trial = (row >= 0) & (train < epochs[np.maximum(row, 0), 1])
        if int(np.sum(in_trial & usable[sample])) >= MIN_SPIKES:
            units.append(unit)
    rates = compute_spatial_rates(
        env,
        [trains[unit] for unit in units],
        rec.times,
        positions,
        epochs=epochs,
        min_speed=min_speed,
        method="binned",
        bandwidth=rec.params["bin_size"],
    )
    return env, np.array(units), rates


def _windows_around(events) -> np.ndarray:
    """Each event plus ``PAD`` on each side, overlapping windows merged."""
    starts = np.sort(events[:, 0] - PAD)
    stops = np.sort(events[:, 1] + PAD)
    new = np.r_[True, starts[1:] > stops[:-1]]
    return np.column_stack([starts[new], stops[np.r_[new[1:], True]]])


def _summed_counts(trains, start, stop):
    """Counts in 25 ms windows stepped by 5 ms over [start, stop), and their centres."""
    from neurospatial.decoding import bin_spikes_in_time

    fine, _ = bin_spikes_in_time(
        trains, dt=STEP, t_start=float(start), t_stop=float(stop)
    )
    fine = np.asarray(fine)
    running = np.vstack([np.zeros((1, fine.shape[1]), fine.dtype), fine]).cumsum(axis=0)
    summed = running[N_SUMMED:] - running[:-N_SUMMED]
    centres = start + STEP * N_SUMMED / 2 + STEP * np.arange(summed.shape[0])
    return summed, centres


def _sequences(model, trains, windows, events):
    """Which events a CAREY_2019 sequence of this side overlaps."""
    from neurospatial.decoding import decode_position
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import CAREY_2019

    env, units, rates = model
    chosen = [trains[unit] for unit in units]
    (criterion,) = CAREY_2019.criteria
    hit = np.zeros(len(events), dtype=bool)
    for start, stop in windows:
        summed, centres = _summed_counts(chosen, start, stop)
        decoded = decode_position(env, summed, rates, dt=STEP * N_SUMMED, times=centres)
        verdict = detect_replay(
            env,
            decoded.posterior,
            centres,
            replay_method=CAREY_2019,
            spike_counts=summed,
        )
        segments = verdict.segments
        if segments is None:
            continue
        passing = segments.durations >= criterion.threshold - 1e-9
        for first, last in zip(
            segments.start_times[passing], segments.end_times[passing], strict=True
        ):
            hit |= (events[:, 0] < last) & (events[:, 1] > first)
    return hit


def _judged(rec, events, *, fit_on=None, decode=None):
    """Left and right verdicts per event, the rate maps from ``fit_on`` spikes."""
    fit_on = rec.spike_times if fit_on is None else fit_on
    decode = rec.spike_times if decode is None else decode
    windows = _windows_around(events)
    return tuple(
        _sequences(_rates_on_side(rec, side, fit_on), decode, windows, events)
        for side in ("left", "right")
    )


def _own_offsets(rec, draw):
    """Null A's spikes: each unit shifted alone, circularly within the task."""
    rng = np.random.default_rng(np.random.SeedSequence(NULL_SEED, spawn_key=(draw, 0)))
    start, stop = rec.epochs["task"][0]
    length = stop - start
    shifted = []
    for train in rec.spike_times:
        moved = train.copy()
        in_task = (train >= start) & (train < stop)
        offset = rng.uniform(10.0, length - 10.0)
        moved[in_task] = start + np.mod(train[in_task] - start + offset, length)
        shifted.append(np.sort(moved))
    return tuple(shifted)


def _common_offset(rec, draw):
    """Null B's spikes: every unit by one offset, circularly over the recording."""
    rng = np.random.default_rng(np.random.SeedSequence(NULL_SEED, spawn_key=(draw, 1)))
    start, stop = rec.times[0], rec.times[-1]
    span = stop - start + 1e-3
    offset = rng.uniform(10.0, span - 10.0)
    return tuple(
        np.sort(start + np.mod(train - start + offset, span))
        for train in rec.spike_times
    )


@pytest.fixture(scope="module")
def lfp(rec):
    samples, times = rec.lfp.fetch(("CSC03a",))
    return samples, times


@pytest.fixture(scope="module")
def detected(rec, lfp):
    """``literature_methods.carey_2019`` as packaged, on the paper's channel."""
    pytest.importorskip("ripple_detection.literature_methods")
    from ripple_detection.literature_methods import run_method

    from neurospatial.datasets._ripple import ripple_recording

    samples, times = lfp
    recording = ripple_recording(
        rec, samples, times, example_ripples=rec.events["hand_labelled_swr"]
    )
    table = run_method("carey_2019", recording)
    return table[["start_time", "end_time"]].to_numpy()


@pytest.mark.realdata
def test_default_session(rec, assert_default_session):
    entry = next(e for e in sessions("carey_2019") if e.default)
    assert_default_session(rec, entry)
    assert rec.n_units == 72
    assert int(np.sum(rec.units["quality"] == "good")) == 30
    assert rec.params["restriction"] == "food"
    assert rec.params["layout"] == "foodLeft"
    assert rec.params["in_paper"]
    assert rec.lfp.ripple_channels[0] == "CSC03a"
    assert rec.lfp.sampling_rate == 2000.0


@pytest.mark.realdata
def test_events_land_at_the_behavior_they_name(rec):
    """Food falls in left trials and water in right ones: one clock for events."""
    for name, epochs in (
        ("food_dispensed", "trials_left"),
        ("water_dispensed", "trials_right"),
    ):
        times, rows = rec.events[name], rec.epochs[epochs]
        assert times.size == len(rows)
        assert np.all(carey_2019._inside(times, rows[np.argsort(rows[:, 0])]))


@pytest.mark.realdata
def test_lfp_times_are_those_of_the_papers_candidates(rec, paper, lfp):
    assert np.array_equal(np.round(lfp[1], 6), np.round(paper.tvec, 6))


@pytest.mark.realdata
def test_spikes_share_the_lfps_clock(rec, paper):
    """Population firing peaks inside the paper's events, which come from the LFP."""
    spikes = np.sort(np.concatenate(rec.spike_times))
    centres = (paper.tstart + paper.tend) / 2
    lags = np.searchsorted(spikes, centres[:, None] + np.arange(-0.3, 0.31, 0.02))
    per_bin = np.diff(lags, axis=1).sum(axis=0)
    assert np.argmax(per_bin) in (14, 15)  # the two bins around the centre
    assert per_bin.max() > 3 * np.median(per_bin)


@pytest.mark.realdata
def test_our_joint_score_is_the_papers_stored_score(rec, paper, lfp):
    """The file's 50 examples, its LFP and its spikes give the paper's score.

    This is the score before any event logic. It says the data are read right,
    and that these 50 examples build the template of the published candidates,
    which the metadata's 2017 date left open.
    """
    pytest.importorskip("ripple_detection.literature_methods")
    import ripple_detection as rd

    samples, times = lfp
    ripple = rd.carey_spectral_ripple_score(
        times, samples[:, 0], 2000.0, rec.events["hand_labelled_swr"]
    )
    joint = np.sqrt(ripple * _mua_score(rec, times))
    joint *= 0.5 / joint.mean()
    assert np.corrcoef(joint, paper.data)[0, 1] > 0.99999
    ours, theirs = joint > 4, paper.data > 4
    assert np.sum(ours & theirs) / np.sum(ours | theirs) > 0.9999


def _detector(rec, lfp, *, theta, speed, threshold):
    """The detector with ``carey_2019``'s settings and these two inputs."""
    pytest.importorskip("ripple_detection.literature_methods")
    import ripple_detection as rd

    from neurospatial.datasets._ripple import spike_counts

    samples, times = lfp
    events = rd.Carey_candidate_detector(
        times,
        None,
        spike_counts(rec.spike_times, times, sampling_rate=2000.0).astype(np.float64),
        speed,
        2000.0,
        ripple_score=rd.carey_spectral_ripple_score(
            times, samples[:, 0], 2000.0, rec.events["hand_labelled_swr"]
        ),
        threshold_method="mean",
        low_threshold=4.0,
        high_threshold=4.0,
        theta_lfp=theta,
        speed_threshold=threshold,
    )
    return events[["start_time", "end_time"]].to_numpy()


@pytest.mark.realdata
def test_agreement_of_the_packaged_method_with_the_papers_events(detected, paper):
    """Pinned: 1,328 events; 73.5% of the paper's are found, 91.6% of ours are its.

    A later change that moves these fails here. The cause of the gap is the
    next test.
    """
    theirs = _intervals_of(paper)
    assert len(theirs) == 1654
    assert len(detected) == 1328
    assert _share_overlapped(theirs, detected) == pytest.approx(0.7352, abs=5e-4)
    assert _share_overlapped(detected, theirs) == pytest.approx(0.9157, abs=5e-4)


@pytest.mark.realdata
def test_the_papers_speed_and_theta_channel_close_the_gap(rec, lfp, paper):
    """The named cause of the gap, with the paper's two inputs in its place.

    The paper's speed is dxdt.m's, in pixels/s, smoothed over 0.5 s, against 10
    pixels/s; ``carey_2019`` is given the session's speed (smoothed over
    0.1 s) against 4 cm/s. The paper's theta is ``goodTheta``'s first channel;
    ``carey_2019`` takes theta from the ripple channel.
    """
    theta, _ = rec.lfp.fetch((rec.params["theta_channel"],))
    events = _detector(
        rec, lfp, theta=theta[:, 0], speed=_papers_speed(lfp[1]), threshold=10.0
    )
    theirs = _intervals_of(paper)
    assert len(events) == 1678
    assert _share_overlapped(theirs, events) == pytest.approx(0.9861, abs=5e-4)
    assert _share_overlapped(events, theirs) == pytest.approx(0.9720, abs=5e-4)


@pytest.mark.realdata
def test_decoding_rows_are_25_ms_windows_5_ms_apart(rec):
    """What the worked result decodes: overlapping windows, truly summed."""
    summed, centres = _summed_counts(rec.spike_times, 3100.0, 3101.0)
    assert np.allclose(np.diff(centres), 0.005)
    assert centres[0] == pytest.approx(3100.0125)
    assert summed.shape == (196, rec.n_units)
    for row in (0, 77, 195):
        edges = centres[row] - 0.0125, centres[row] + 0.0125
        direct = [
            np.searchsorted(train, edges[1]) - np.searchsorted(train, edges[0])
            for train in rec.spike_times
        ]
        assert summed[row].tolist() == direct


@pytest.mark.realdata
def test_rate_maps_are_speed_gated_and_exclude_rest(rec):
    """The gate is on: rest spikes and slow spikes do not reach the maps."""
    from neurospatial import compute_spatial_rates

    env, _, rates = _rates_on_side(rec, "left", rec.spike_times)
    assert rates.analysis_settings["min_speed"] == rec.params["min_speed"]
    assert rates.analysis_settings["epochs"] == _trial_epochs(rec, "left").tolist()
    # Trains firing only at rest, or only while slow in a trial, give empty maps.
    epochs = _trial_epochs(rec, "left")
    pedestal = rec.epochs["rest_left_pedestal"]
    in_trial = carey_2019._inside(rec.times, epochs)
    slow = rec.times[in_trial & (np.nan_to_num(rec._speed, nan=99.0) < 3.0)]
    assert slow.size > 100
    maps = compute_spatial_rates(
        env,
        [np.linspace(pedestal[0, 0], pedestal[0, 1], 500), slow],
        rec.times,
        rec.positions,
        epochs=epochs,
        min_speed=rec.params["min_speed"],
        method="binned",
        bandwidth=rec.params["bin_size"],
    )
    assert np.nansum(np.asarray(maps[0].firing_rate)) == 0.0
    assert np.nansum(np.asarray(maps[1].firing_rate)) == 0.0


@pytest.mark.realdata
def test_worked_result_sequences_in_the_candidates(rec, detected, paper):
    """Level B, fixed in writing before any decode: above each of two nulls.

    The count is of candidates a left or right sequence overlaps. Null A fits
    the rate maps on spikes with each unit shifted alone within the task,
    which keeps each event's burst and breaks which place each cell codes.
    Null B decodes all spikes shifted by one offset, which keeps co-firing and
    moves it off the events.
    """
    record = carey_2019.WORKED_RESULT
    left, right = _judged(rec, detected)
    count = int(np.sum(left | right))
    assert count == 65
    assert int(np.sum(left & ~right)) == 15
    assert int(np.sum(right & ~left)) == 35
    assert record["ours"] == round(count / len(detected), 3)
    fitted, moved = [], []
    for draw in range(N_NULL):
        sides = _judged(rec, detected, fit_on=_own_offsets(rec, draw))
        fitted.append(int(np.sum(sides[0] | sides[1])))
        sides = _judged(rec, detected, decode=_common_offset(rec, draw))
        moved.append(int(np.sum(sides[0] | sides[1])))
    assert count > np.percentile(fitted, 95)
    assert count > np.percentile(moved, 95)
    assert (min(fitted), max(fitted)) == (22, 65)
    assert (min(moved), max(moved)) == (2, 14)


@pytest.mark.realdata
def test_the_papers_own_candidates_give_the_same_count(rec, paper):
    left, right = _judged(rec, _intervals_of(paper))
    assert int(np.sum(left | right)) == 64


@pytest.mark.realdata
def test_the_papers_sessions_show_its_shift_in_sequence_content():
    """Exploratory, on the paper's candidates in its 19 sessions: see ACROSS_SESSIONS.

    Events with a sequence on one side only, per rat, beside Supplementary
    Fig. 6, and the left share by what was restricted, beside the paper's 0.35
    and 0.57. CAREY_2019 leaves out the paper's identity-shuffle filter, and
    these are events, not sequences. The shift in the left share is not a
    reproduction: shifted rate maps give the same difference (20 draws took an
    hour, so they are recorded and not rerun here).
    """
    record = carey_2019.ACROSS_SESSIONS
    per_rat: dict[str, int] = {}
    shares, restrictions, left_trials = [], [], []
    for session in sorted(carey_2019.PAPER_SESSIONS):
        rec = load("carey_2019", session)
        evt = _paper_candidates(session)
        left, right = _judged(rec, _intervals_of(evt))
        only_left, only_right = int(np.sum(left & ~right)), int(np.sum(right & ~left))
        rat = session[:4]
        per_rat[rat] = per_rat.get(rat, 0) + only_left + only_right
        shares.append(only_left / (only_left + only_right))
        restrictions.append(rec.params["restriction"])
        left_trials.append(float(np.mean(rec.trials.arm == "L")))
    assert per_rat == record["one_side_per_rat"]
    shares = np.array(shares)
    food = np.array(restrictions) == "food"
    assert round(float(shares[food].mean()), 3) == record["left_share_food"]
    assert round(float(shares[~food].mean()), 3) == record["left_share_water"]
    correlation = np.corrcoef(shares, left_trials)[0, 1]
    assert round(float(correlation), 2) == record["r_with_left_trials"]
