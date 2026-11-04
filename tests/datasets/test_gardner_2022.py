"""Gardner et al. (2022), Figshare 16764508: synthetic recordings and the real default."""

from __future__ import annotations

import collections
import io
import os
import pickle
import zipfile

import numpy as np
import pytest

from neurospatial.datasets import gardner_2022, load
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._pickled import SpikeUnpickler, pickle_imports, read_npz

DEFAULT = "rat_r_day1"
SESSIONS = (
    "#1,  open_field_1,       start=100,  end=700, valid_times=[100,400;450,700]\n"
    "#2,  wagon_wheel_1,      start=800, end=900\n"
)


def _box_walk_m(n: int, seed: int) -> np.ndarray:
    """A reflected random walk filling a 1.46 m square centred at 0, in metres."""
    rng = np.random.default_rng(seed)
    xy = np.mod(rng.normal(0, 0.01, (n, 2)).cumsum(axis=0), 2.92)
    return np.where(xy > 1.46, 2.92 - xy, xy) - 0.73


def _write_recording(folder, *, azimuth=None, seed=0, repeat=None):
    """A recording laid out as the archive's: metres, [0, 2pi) azimuth, pickled spikes."""
    folder.mkdir(parents=True, exist_ok=True)
    t = 50.0 + np.arange(85_000) * 0.01  # 50 to 900 s: before, in and after the box
    xy = _box_walk_m(t.size, seed)
    if azimuth is None:
        step = np.diff(xy, axis=0, append=xy[-1:] + xy[-1:] - xy[-2:-1])
        azimuth = np.mod(np.arctan2(step[:, 1], step[:, 0]), 2 * np.pi)
    rng = np.random.default_rng(seed + 100)
    modules = {}
    for module, n_units in ((1, 2), (2, 1)):
        trains = {
            index: np.sort(rng.uniform(t[0], t[-1], 300)) for index in range(n_units)
        }
        if repeat is not None and module == 1:
            trains[0] = np.sort(np.concatenate([trains[0], [repeat, repeat]]))
        holder = np.empty((), dtype=object)
        holder[()] = trains
        modules[f"spikes_mod{module}"] = holder
    np.savez(
        folder / "rat_r_day1_grid_modules_1_2_3.npz",
        t=t,
        x=xy[:, 0],
        y=xy[:, 1],
        z=np.full(t.size, 0.7),
        azimuth=azimuth,
        **modules,
    )
    (folder / "rat_r_day1_sessions.txt").write_text(SESSIONS)
    return t, xy, modules


@pytest.fixture
def recording(tmp_path, session_entry):
    folder = tmp_path / "members"
    t, xy, modules = _write_recording(folder)
    entry = session_entry("gardner_2022")
    rec = run_checks(gardner_2022.load_session(entry, folder), stated=entry.info.stated)
    return rec, t, xy, modules


def test_keeps_the_open_field_session_in_cm(recording):
    rec, t, xy, modules = recording
    window = (t >= 100.0) & (t <= 700.0)
    np.testing.assert_array_equal(rec.times, t[window])
    np.testing.assert_allclose(rec.positions, xy[window] * 100.0)
    assert rec.position_units == "cm"
    assert rec.failed_checks == ()
    # Spikes of the open field only, in module order, named mod<m>_<index>.
    assert list(rec.units.index) == ["mod1_0", "mod1_1", "mod2_0"]
    assert list(rec.units["module"]) == [1, 1, 2]
    assert list(rec.units["label"]) == ["grid module 1"] * 2 + ["grid module 2"]
    np.testing.assert_array_equal(rec.params["module"], [1, 1, 2])
    expected = modules["spikes_mod1"][()][1]
    np.testing.assert_array_equal(
        rec.spike_times[1], expected[(expected >= 100.0) & (expected <= 700.0)]
    )
    # The valid times are the epochs and the analysis epochs.
    np.testing.assert_array_equal(rec.epochs["open_field"], [[100, 400], [450, 700]])
    np.testing.assert_array_equal(
        rec.params["analysis_epochs"], rec.epochs["open_field"]
    )
    # Train and test blocks lie inside the valid times, never in 400-450 s.
    for blocks in (rec.params["train_epochs"], rec.params["test_epochs"]):
        assert not np.any((blocks[:, 0] < 450.0) & (blocks[:, 1] > 400.0))
    np.testing.assert_array_equal(rec.params["train_epochs"][0], [100.0, 160.0])
    assert rec.params["bin_size"] == 3.0
    assert rec.params["min_speed"] == 2.5


def test_headings_are_the_wrapped_azimuth(recording, tmp_path):
    rec, t, _, _ = recording
    stored = np.load(tmp_path / "members" / "rat_r_day1_grid_modules_1_2_3.npz")
    azimuth = stored["azimuth"][(t >= 100.0) & (t <= 700.0)]
    assert azimuth.max() > np.pi  # stored in [0, 2pi)
    np.testing.assert_allclose(
        np.exp(1j * rec.headings), np.exp(1j * azimuth), atol=1e-12
    )
    assert rec.headings.min() >= -np.pi
    assert rec.headings.max() <= np.pi
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert heading.basis == "inferred"
    assert "so it points along the path" in heading.evidence
    assert not any("not verified" in text for text in rec.watch_outs)


def test_the_time_unit_is_inferred_from_the_agreeing_clocks(recording):
    rec, t, _, _ = recording
    clock = next(c for c in rec.conversions if c.text.startswith("times, spike"))
    assert clock.basis == "inferred"
    assert "no source read states their unit or clock" in clock.evidence
    assert f"the file's t runs from {t[0]:.3f} to {t[-1]:.3f} s" in clock.evidence
    assert "edges that agree to within" in clock.evidence
    window = next(c for c in rec.conversions if c.text.startswith("times, positions"))
    assert window.basis == "stated"
    assert "on the clock of t is inferred" in window.evidence
    trains = next(c for c in rec.conversions if c.text.startswith("spike_times"))
    assert " in s" not in trains.text


def test_names_the_invalid_stretch_and_the_other_sessions(recording):
    rec, t, _, _ = recording
    n_gap = int(np.sum((t > 400.0) & (t < 450.0)))
    text = " ".join(rec.watch_outs)
    assert f"leave out 400 to 450 s ({n_gap} samples) of open_field_1" in text
    assert "also holds wagon_wheel_1 (800 to 900 s)" in text
    assert "outside 100 to 700 s are not in this recording" in text


def test_declares_the_stated_square_on_the_samples(recording):
    rec, *_ = recording
    assert rec.env_kind == "declared"
    assert rec.env.units == "cm"
    placement = rec.conversions[-1]
    assert placement.basis == "inferred"
    assert "the stated 150 x 150 cm square" in placement.text
    (low_x, high_x), (low_y, high_y) = rec.env.dimension_ranges
    # A 150 cm square, centred within a few cm of the walk's centre.
    assert high_x - low_x == pytest.approx(150.0, abs=3.0)
    assert abs((low_x + high_x) / 2) < 3.0
    assert abs((low_y + high_y) / 2) < 3.0
    assert np.all(np.asarray(rec.env.bin_at(rec.positions)) >= 0)


def test_heading_off_the_path_is_kept_and_reported(tmp_path, session_entry):
    folder = tmp_path / "members"
    _write_recording(folder, azimuth=np.mod(np.arange(85_000) * 0.37, 6.28))
    rec = gardner_2022.load_session(session_entry("gardner_2022"), folder)
    assert any(
        text.startswith("The heading frame is not verified by movement")
        for text in rec.watch_outs
    )
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert "its frame is not verified" in heading.evidence


def test_repeated_spikes_are_kept_and_counted(tmp_path, session_entry):
    folder = tmp_path / "members"
    _write_recording(folder, repeat=300.0)
    rec = gardner_2022.load_session(session_entry("gardner_2022"), folder)
    assert int(np.sum(np.diff(rec.spike_times[0]) == 0)) == 1
    assert any(
        text.startswith("1 spike repeats the time of the spike before it")
        for text in rec.watch_outs
    )
    # Kept: the train holds both copies.
    assert int(np.sum(rec.spike_times[0] == 300.0)) == 2


def test_parse_sessions_reads_valid_times_and_refuses_other_lines():
    sessions = gardner_2022.parse_sessions(SESSIONS)
    assert [s.name for s in sessions] == ["open_field_1", "wagon_wheel_1"]
    assert sessions[0].valid.tolist() == [[100.0, 400.0], [450.0, 700.0]]
    assert sessions[1].valid.tolist() == [[800.0, 900.0]]
    with pytest.raises(ValueError, match="does not read as"):
        gardner_2022.parse_sessions("open_field_1 from 1 to 2")


def test_two_open_field_sessions_are_refused(tmp_path, session_entry):
    folder = tmp_path / "members"
    _write_recording(folder)
    (folder / "rat_r_day1_sessions.txt").write_text(
        SESSIONS + "#3,  open_field_2,  start=800, end=850\n"
    )
    with pytest.raises(ValueError, match="2 open-field sessions"):
        gardner_2022.load_session(session_entry("gardner_2022"), folder)


class _Exploit:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def __reduce__(self):
        return (os.system, (f"touch {self.sentinel}",))


def test_pickle_imports_lists_globals_without_running(tmp_path):
    sentinel = tmp_path / "ran"
    blob = pickle.dumps(_Exploit(sentinel), protocol=4)
    assert pickle_imports(blob) == {f"{os.system.__module__}.system"}
    assert not sentinel.exists()


def test_read_npz_refuses_an_exploit_before_running_it(tmp_path):
    sentinel = tmp_path / "ran"
    path = tmp_path / "exploit.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "|O", "fortran_order": False, "shape": ()}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "spikes_mod1.npy",
            header.getvalue() + pickle.dumps(_Exploit(sentinel), protocol=3),
        )
    with pytest.raises(pickle.UnpicklingError, match="system"):
        read_npz(path)
    assert not sentinel.exists()


def test_read_npz_refuses_an_unlisted_global(tmp_path):
    holder = np.empty((), dtype=object)
    holder[()] = collections.OrderedDict({0: np.array([1.0])})
    np.savez(tmp_path / "bad.npz", spikes_mod1=holder)
    with pytest.raises(pickle.UnpicklingError, match="OrderedDict"):
        read_npz(tmp_path / "bad.npz")


def test_unpickler_refuses_an_ndarray_built_from_raw_bytes():
    # REDUCE numpy.ndarray((1,), "O", buffer=<8 bytes>) would read the bytes as
    # an object pointer; the stream may only pass ndarray to _reconstruct.
    def text(value):
        raw = value.encode()
        return pickle.SHORT_BINUNICODE + bytes([len(raw)]) + raw

    blob = (
        pickle.PROTO
        + b"\x04"
        + text("numpy")
        + text("ndarray")
        + pickle.STACK_GLOBAL
        + pickle.MARK
        + pickle.BININT1
        + b"\x01"
        + pickle.TUPLE
        + text("O")
        + pickle.SHORT_BINBYTES
        + b"\x08"
        + id(os.system).to_bytes(8, "little")
        + pickle.TUPLE3
        + pickle.REDUCE
        + pickle.STOP
    )
    with pytest.raises((pickle.UnpicklingError, TypeError)):
        SpikeUnpickler(io.BytesIO(blob)).load()


@pytest.mark.parametrize("protocol", [3, 4])
def test_unpickler_reads_a_dict_of_float_arrays(protocol):
    holder = np.empty((), dtype=object)
    holder[()] = {0: np.array([0.5, 1.5]), 3: np.array([2.0])}
    out = SpikeUnpickler(io.BytesIO(pickle.dumps(holder, protocol=protocol))).load()
    np.testing.assert_array_equal(out[()][0], [0.5, 1.5])
    np.testing.assert_array_equal(out[()][3], [2.0])


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """Rat R day 1's open field: 483 grid-module units, 100 Hz, the 1.5 m box."""
    rec = load("gardner_2022")
    assert_default_session(rec, session_entry("gardner_2022"))
    assert rec.units["module"].value_counts().sort_index().tolist() == [166, 168, 149]
    assert rec.sampling_rate == pytest.approx(100.0, rel=1e-6)
    assert rec.times.size == 858_800
    assert rec.times[0] == pytest.approx(7457.01)
    assert rec.times[-1] == pytest.approx(16045.0)
    np.testing.assert_allclose(np.ptp(rec.positions, axis=0), [147.8, 147.5], atol=0.1)
    np.testing.assert_array_equal(
        rec.params["analysis_epochs"], [[7457.0, 14778.0], [14890.0, 16045.0]]
    )
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert "circular mean of -4.8 deg" in heading.evidence
    assert not any(np.isnan(rec.positions).ravel())
