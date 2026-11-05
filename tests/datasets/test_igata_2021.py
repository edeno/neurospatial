"""The Igata 2021 loader: its conversions on small fabricated files, and the real session."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from neurospatial.datasets import igata_2021, load
from neurospatial.datasets._catalog import _read
from neurospatial.datasets.igata_2021 import (
    _flag_rows,
    _optimized,
    _read_spikes,
    learning_point,
    load_session,
    path_track,
    read_lfp,
)

SESSION = "180912_detourG01"
N_COLUMNS = 16
N_TETRODES = 2


def _log(start_ms, n_frames, *, step=40.0, x=100.0, y=500.0):
    """A ``log``: frames ``step`` ms apart; every flag off; sync on."""
    log = np.zeros((n_frames, N_COLUMNS))
    log[:, 0] = start_ms + step * np.arange(n_frames)
    log[:, 1] = x + np.arange(n_frames)
    log[:, 2] = y
    log[:, 3] = 1.0
    log[:, 15] = 1.0
    return log


def _python2_pickle(trains):
    """An object array of spike trains, pickled with the imports the real files have.

    Protocol 2 writes bytes through ``_codecs.encode`` and, with ``fix_imports``,
    names ``builtins`` as ``__builtin__``.
    """
    array = np.empty(len(trains), dtype=object)
    for unit, train in enumerate(trains):
        array[unit] = np.asarray(train, dtype=np.float64)
    return pickle.dumps(array, protocol=2, fix_imports=True)


def _write(path, log, trains, *, string="lUVa", stimulation=None, n_tetrodes=2):
    """One ``.npz`` as the archive stores it, its ``spike`` pickled."""
    span_ms = log[-1, 0] - log[0, 0]
    n_lfp = round(span_ms * 2)
    lfps = np.vstack(
        [log[0, 0] + row * 1000.0 + np.arange(n_lfp) for row in range(n_tetrodes)]
    )
    hist = np.zeros((len(trains), round(span_ms)))
    for unit, train in enumerate(trains):
        hist[unit] = np.histogram(train, bins=np.arange(hist.shape[1] + 1))[0]
    arrays = {
        "log": log,
        "cell": np.zeros((len(log), 3 + len(trains))),
        "velocity": np.zeros((len(log), 7)),
        "lfp": lfps[0],
        "lfps": lfps,
        "ref": -lfps[:1],
        "lfpch": np.array("tet001"),
        "tet_names": np.array([f"tet{k + 1:03d}" for k in range(n_tetrodes)]),
        "spike_hist": hist,
        "locus_string_list": np.array(list(string)),
        "locus_stay_frame_list": np.ones(len(string), dtype=np.int64),
    }
    if stimulation is not None:
        arrays["trial_stim"] = np.atleast_2d(stimulation)
    if not trains:
        arrays["spike"] = np.zeros(0)
        np.savez(path, **arrays)
        return
    # savez refuses nothing here: the object array is written as the archive's is.
    array = np.empty(len(trains), dtype=object)
    for unit, train in enumerate(trains):
        array[unit] = np.asarray(train, dtype=np.float64)
    np.savez(path, spike=array, **arrays)


def _registry(folder, session=SESSION):
    names = sorted(path.name for path in folder.glob("*.npz"))
    return {
        "dataset": "igata_2021",
        "paper": "Igata et al. (2021)",
        "paper_citation": "Igata H. A paper.",
        "species": "rat",
        "brain_regions": ["CA1"],
        "apparatus": "lattice maze",
        "license": "CC-BY-4.0",
        "archive": {"name": "Mendeley", "citation": "The data."},
        "stated": {
            "sampling_rate": {"value": 24.8, "basis": "inferred", "source": "files"},
            "position_span": {
                "value": [10.0, 200.0],
                "basis": "stated",
                "source": "SI",
            },
        },
        "sessions": [
            {
                "session": session,
                "archive": "Mendeley",
                "url": "https://example.org/dataset",
                "file_name": session,
                "sha256": "0" * 64,
                "size_bytes": 1,
                "verified": False,
                "default": True,
                "members": [
                    {
                        "path": name,
                        "sha256": "1" * 64,
                        "size_bytes": 1,
                        "url": f"https://example.org/{name}",
                    }
                    for name in names
                ],
            }
        ],
    }


@pytest.fixture
def session(tmp_path):
    """Two trials, each with its return run, on one clock, with two units.

    Trial 1 runs 1000 to 1160 ms, its return shares that last frame and ends at
    1240 ms; trial 2 starts one frame later.
    """
    folder = tmp_path / "igata_2021" / SESSION
    folder.mkdir(parents=True)
    trial = _log(1000.0, 5)
    trial[1:3, 4] = 1.0  # trial flag
    trial[2, 8] = 1.0  # checkpoint
    returned = _log(1160.0, 3)
    returned[0] = trial[-1]
    second = _log(1280.0, 4)
    second[:2, 5] = 1.0  # nose poke, starting the file
    second[3, 10] = 1.0  # reward port, ending the session
    last = _log(1400.0, 2)
    last[0] = second[-1]
    _write(folder / f"{SESSION}_trial001.npz", trial, [[10.5, 20.0], [100.0]])
    _write(folder / f"{SESSION}_return001.npz", returned, [[5.0], []])
    _write(
        folder / f"{SESSION}_trial002.npz", second, [[1.5], [2.5, 3.5]], string="lUGa"
    )
    _write(folder / f"{SESSION}_return002.npz", last, [[], []])
    return folder


@pytest.fixture(autouse=True)
def pickled_like_the_archive(monkeypatch):
    """Write ``spike`` as Python 2 did: ``np.savez`` alone writes protocol 4+."""
    original = np.lib.format.write_array

    def write_array(fp, array, *args, **kwargs):
        if array.dtype.hasobject:
            np.lib.format.write_array_header_1_0(
                fp, np.lib.format.header_data_from_array_1_0(array)
            )
            fp.write(_python2_pickle(list(array)))
            return
        original(fp, array, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "write_array", write_array)


def _load(folder):
    entry = _read(_registry(folder), "igata_2021.json")[0]
    return load_session(entry, folder)


def test_reads_python_2_spike_pickles(session):
    trains = _read_spikes(session / f"{SESSION}_trial001.npz")
    assert [train.tolist() for train in trains] == [[10.5, 20.0], [100.0]]


def test_refuses_a_pickle_that_imports_anything_else(session, tmp_path):
    import zipfile

    class Evil:
        def __reduce__(self):
            return (print, ("ran",))

    path = tmp_path / "evil.npz"
    with (
        zipfile.ZipFile(session / f"{SESSION}_trial001.npz") as source,
        zipfile.ZipFile(path, "w") as target,
    ):
        for member in source.namelist():
            data = source.read(member)
            if member == "spike.npy":
                header = data[: data.index(b"\n") + 1]
                data = header + pickle.dumps(Evil(), protocol=2)
            target.writestr(member, data)
    with pytest.raises(pickle.UnpicklingError, match=r"refusing evil\.npz"):
        _read_spikes(path)


def test_stitches_files_onto_one_clock(session):
    rec = _load(session)
    # 5 + 3 - 1 shared frames, then 4 + 2 - 1.
    assert rec.times.size == 12
    np.testing.assert_allclose(rec.times[:3], [1.0, 1.04, 1.08])
    assert np.all(np.diff(rec.times) > 0)
    # mm to cm.
    np.testing.assert_allclose(rec.positions[0], [10.0, 50.0])
    assert rec.position_units == "cm"


def test_spike_times_are_ms_from_each_files_first_frame(session):
    rec = _load(session)
    np.testing.assert_allclose(rec.spike_times[0], [1.0105, 1.02, 1.165, 1.2815])
    np.testing.assert_allclose(rec.spike_times[1], [1.1, 1.2825, 1.2835])
    assert rec.n_units == 2


def test_flags_become_epochs_and_events(session):
    rec = _load(session)
    np.testing.assert_allclose(rec.epochs["trial"], [[1.04, 1.12]])
    np.testing.assert_allclose(rec.events["checkpoint"], [1.08])
    # A run from a file's first frame ends at the next frame off.
    np.testing.assert_allclose(rec.events["nose_poke"], [[1.28, 1.36]])
    np.testing.assert_allclose(rec.epochs["return"], [[1.16, 1.24], [1.4, 1.44]])


def test_flag_at_a_gap_closes_on_its_last_frame():
    times = np.array([0.0, 0.04, 0.08, 10.0, 10.04])
    flag = np.array([True, True, True, True, False])
    np.testing.assert_allclose(_flag_rows(times, flag), [[0.0, 0.08], [10.0, 10.04]])


def test_trials_keep_their_lattice_strings(session):
    trials = _load(session).trials
    assert list(trials.lattices) == ["lUVa", "lUGa"]
    assert list(trials.trial) == [1, 2]


def test_overlapping_files_are_refused(session):
    path = session / f"{SESSION}_trial002.npz"
    _write(path, _log(1200.0, 2), [[], []])
    with pytest.raises(ValueError, match="before its predecessor ends"):
        _load(session)


def test_a_shared_frame_that_differs_is_refused(session):
    returned = _log(1160.0, 3, x=500.0)
    _write(session / f"{SESSION}_return001.npz", returned, [[], []])
    with pytest.raises(ValueError, match="with different values"):
        _load(session)


def test_files_with_different_unit_counts_are_refused(session):
    _write(session / f"{SESSION}_return002.npz", _log(1400.0, 2), [[1.0]])
    with pytest.raises(ValueError, match="so ids would not line up"):
        _load(session)


def test_gap_longer_than_a_frame_is_described(session):
    (session / f"{SESSION}_return001.npz").unlink()
    second = _log(9000.0, 3)
    returned = _log(9080.0, 2)
    returned[0] = second[-1]
    _write(session / f"{SESSION}_trial002.npz", second, [[], []])
    _write(session / f"{SESSION}_return002.npz", returned, [[], []])
    rec = _load(session)
    assert any("longer than 1 s between files" in text for text in rec.watch_outs)


def test_session_without_units_says_so(tmp_path):
    folder = tmp_path / "igata_2021" / "180825_detourG01"
    folder.mkdir(parents=True)
    _write(folder / "180825_detourG01_trial001.npz", _log(0.0, 5), [])
    entry = _read(_registry(folder, "180825_detourG01"), "igata_2021.json")[0]
    rec = load_session(entry, folder)
    assert rec.n_units == 0
    assert any("no sorted units" in text for text in rec.watch_outs)
    assert rec.animal == "a rat the archive does not name"


def test_stimulation_times_join_the_clock(tmp_path):
    folder = tmp_path / "igata_2021" / "190523_detourG01"
    folder.mkdir(parents=True)
    _write(
        folder / "190523_detourG01_trial001.npz",
        _log(2000.0, 5),
        [],
        stimulation=[10.0, 50.0],
    )
    entry = _read(_registry(folder, "190523_detourG01"), "igata_2021.json")[0]
    rec = load_session(entry, folder)
    np.testing.assert_allclose(rec.events["stimulation"], [2.01, 2.05])
    assert any("closed-loop experiment" in text for text in rec.watch_outs)


def test_lfp_is_read_on_request_in_the_order_asked(session):
    rec = _load(session)
    assert list(rec.lfp.channels) == ["tet001", "tet002", "ref"]
    assert rec.lfp.ripple_channels == ("tet001",)
    paths = sorted(session.glob("*.npz"))
    samples, times = read_lfp(tuple(paths), ("tet002", "ref"))
    assert samples.shape == (times.size, 2)
    # Each file's values start at its first frame's time; tetrode 2 adds 1000.
    assert samples[0, 0] == 2000.0
    assert samples[0, 1] == -1000.0
    assert times[0] == 1.0
    assert np.all(np.diff(times) > 0)
    # Its last sample is before the frame the next file starts on.
    assert times[times < 1.16].size == 320


def test_lfp_rows_without_names_leave_only_the_representative(tmp_path):
    """Two files name five tetrodes over one row; which one is not said."""
    from neurospatial.datasets.igata_2021 import _representative

    path = tmp_path / "file.npz"
    np.savez(
        path,
        log=_log(0.0, 3),
        tet_names=np.array(["tet002", "tet003", "tet008"]),
        lfps=np.ones((1, 160)),
        lfp=2.0 * np.ones(160),
        ref=np.zeros((1, 160)),
        lfpch=np.array("tet003"),
    )
    assert _representative(path) == (["tet003"], "tet003")
    samples, _ = read_lfp((path,), ("tet003",))
    np.testing.assert_array_equal(samples[:, 0], 2.0)
    with pytest.raises(ValueError, match="holds no channel 'tet002'"):
        read_lfp((path,), ("tet002",))


def test_lfp_of_an_unknown_channel_is_refused(session):
    with pytest.raises(ValueError, match="holds no channel 'tet009'"):
        read_lfp(tuple(session.glob("*.npz")), ("tet009",))


def test_representative_tetrode_is_found_by_its_values(tmp_path):
    """``lfpch`` can drop the suffix a tetrode's name carries."""
    from neurospatial.datasets.igata_2021 import _representative

    path = tmp_path / "file.npz"
    rows = np.arange(12.0).reshape(3, 4)
    np.savez(
        path,
        tet_names=np.array(["tet012_1", "tet013_1", "tet014_1"]),
        lfps=rows,
        lfp=rows[1],
    )
    assert _representative(path) == (["tet012_1", "tet013_1", "tet014_1"], "tet013_1")
    np.savez(path, tet_names=np.array(["a"]), lfps=rows[:1], lfp=rows[1])
    with pytest.raises(ValueError, match="equals 0 rows"):
        _representative(path)


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("lUVQLGHIDEa", True),  # the shortest path through C2
        ("lUVWXSTOJEa", False),  # through C1
        ("lUVQLGFABCDEa", True),  # through C2, 11 field lattices: under 12
        ("lUPKFGBCDEa", True),
        ("lUVQLGFKFABCDEa", False),  # through C2, 13 field lattices
        ("lUVQRSNIHGDEa", False),  # through both
        ("lUVQLHIDEa", False),  # missing C2
    ],
)
def test_optimized_path(string, expected):
    assert _optimized(string) is expected


def test_learning_point_follows_the_si_rule():
    wrong, right = "lUVWXSTOJEa", "lUVQLGHIDEa"
    # The centred average first exceeds 50% at trial 10 (3 of trials 8 to 12).
    # One trial before it, 9, was not optimized, so the next one, 10, counts.
    assert learning_point([wrong] * 10 + [right] * 6) == 10
    # A trial 9 that took the path is the learning point itself.
    assert learning_point([wrong] * 9 + [right] * 7) == 9 - 1 + 1
    assert learning_point([wrong] * 10) is None


def test_track_is_the_papers_lines_between_lattice_centres():
    track = path_track()
    assert track.units == "cm"
    # Every lattice centre lies on the graph.
    for point in ([100.0, 90.0], [70.0, 70.0], [30.0, 30.0], [90.0, 10.0]):
        assert track.bin_at(np.array([point]))[0] >= 0
    # A sample more than 20 cm from every line falls in no bin.
    assert track.bin_at(np.array([[50.0, 50.0]]))[0] == -1


def test_wall_separates_the_field_from_the_return_path(session):
    env = _load(session).env
    field, alley = env.bin_at(np.array([[50.0, 97.5], [50.0, 102.5]]))
    assert field >= 0
    assert alley >= 0
    pairs = {tuple(sorted(pair)) for pair in env.edge_pairs.tolist()}
    assert tuple(sorted((int(field), int(alley)))) not in pairs
    assert set(env.regions) >= {"start_area", "goal_area", "return_path", "C1", "C2"}


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """Rat 1's learning day: 114 units, as the SI states, on 13 tetrodes."""
    rec = load("igata_2021")
    assert_default_session(rec, session_entry("igata_2021"))
    assert rec.n_units == 114
    assert rec.animal == "rat 1"
    assert len(rec.lfp.channels) == 14
    assert rec.lfp.ripple_channels == ("tet013",)
    assert rec.params["learning_point"] == 46
    assert rec.trials.optimized.sum() == 17


@pytest.mark.realdata
def test_spike_field_matches_spike_hist_in_every_file():
    """The two stores of the spikes agree exactly, unit by unit."""
    rec = load("igata_2021")
    folder = Path(rec.lfp.files[0].path)
    from neurospatial.datasets._fetch import cache_root

    root = cache_root(None) / "igata_2021" / folder.parent
    paths = sorted(root.glob("*.npz"))
    assert paths
    for path in paths:
        trains = _read_spikes(path)
        with np.load(path, allow_pickle=False) as data:
            stored = data["spike_hist"]
        edges = np.arange(stored.shape[1] + 1)
        for unit, train in enumerate(trains):
            np.testing.assert_array_equal(
                np.histogram(train, bins=edges)[0], stored[unit]
            )


@pytest.mark.realdata
def test_spike_clock_is_the_frames_clock():
    """The file's per-frame counts hold the spikes around each frame, at no lag."""
    from neurospatial.datasets._fetch import cache_root

    rec = load("igata_2021")
    root = cache_root(None) / "igata_2021" / rec.session
    errors = {}
    for lag in (-40.0, -20.0, 0.0, 20.0, 40.0):
        total = 0.0
        for path in sorted(root.glob("*trial*.npz"))[:20]:
            with np.load(path, allow_pickle=False) as data:
                frames = data["log"][:, 0]
                cell = data["cell"][:, 3:].sum(axis=1)
            spikes = np.concatenate(_read_spikes(path)) + frames[0]
            # A frame owns the half frame either side of it.
            half = np.median(np.diff(frames)) / 2
            edges = np.r_[frames - half, frames[-1] + half]
            total += np.abs(np.histogram(spikes + lag, bins=edges)[0] - cell).sum()
        errors[lag] = total
    assert min(errors, key=errors.get) == 0.0


@pytest.mark.realdata
def test_checkpoint_flag_rises_in_the_lattice_the_session_names(session_entry):
    """The session's S or G is the checkpoint lattice, on every trial."""
    env_letters = {"S": (60.0, 60.0), "G": (20.0, 20.0)}
    for session in ("180912_detourG01", "180912_detourS01"):
        rec = load("igata_2021", session)
        letter = session[len("180912_detour")]
        low = np.array(env_letters[letter])
        where = rec.positions[np.searchsorted(rec.times, rec.events["checkpoint"])]
        assert len(where) == len(rec.trials)
        assert np.all((where >= low) & (where <= low + 20.0))


@pytest.mark.realdata
def test_trials_to_the_learning_point_match_the_paper():
    """Level A on behaviour: '26 +/- 9 trials' to the learning point (n = 5 rats).

    The SI gives data as mean +/- SEM. The rule is the SI's; only the window's
    alignment is unstated, and a centred window is used.
    """
    points = np.array(
        [
            load("igata_2021", f"{day}_detourG01").params["learning_point"]
            for day in igata_2021.LEARNING_RATS
        ],
        dtype=float,
    )
    assert points.size == 5
    assert round(points.mean()) == 26
    assert round(points.std(ddof=1) / np.sqrt(points.size)) == 9


@pytest.mark.realdata
def test_no_sorted_units_where_the_catalog_says():
    rec = load("igata_2021", "180825_detourG01")
    assert rec.n_units == 0
    assert any("no sorted units" in text for text in rec.watch_outs)
