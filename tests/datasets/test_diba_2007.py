"""Diba & Buzsáki (2007): Diba's rats in CRCNS hc-3, and one session's positions.

Synthetic tests write a tiny session in hc-3's layout (NeuroSuite ``.xml``,
``.res.N``/``.clu.N``, ``.eeg`` and the metadata tables' zip) and a tiny
``fig1.nel`` laid out as Maboudi et al.'s is (nelpy stand-in classes pickled
as dill does). Real-data tests read the archives and pin every number the
loader and the worked result state.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import pickle
import sys
import types
import zipfile

import numpy as np
import pytest
from test_neuroscope import write_session

from neurospatial.datasets import diba_2007, load
from neurospatial.datasets._catalog import _read
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._fetch import crcns_url

SESSION = "2006-1-1_10-0-0"
TOPDIR = "gor01-1-1"
WIDEBAND = 32552.0
N_CHANNELS = 4
N_LFP = 2504  # 2 s at 1252 Hz


def _metadata_zip(path, *, session=SESSION, topdir=TOPDIR):
    """hc-3's metadata tables, with rows for one Diba session and one other."""
    tables = {
        "hc3-session.csv": f"1,{topdir},{session},linearOne,10,2.0\n"
        "2,ec013.38,ec013.669,linear,10,798.3\n",
        "hc3-epos.csv": f'{topdir},gor01,CA3,CA1,"",""\nec013.38,ec013,EC3,EC3,"",""\n',
        # id, topdir, animal, ele, clu, region, 6 counts, 2 rates, type
        "hc3-cell.csv": f"1,{topdir},gor01,1,2,CA3,0,0,0,0,0,0,1.0,1.0,p\n"
        f"2,{topdir},gor01,2,3,CA1,0,0,0,0,0,0,9.0,9.0,i\n"
        f"3,{topdir},gor01,2,4,CA1,0,0,0,0,0,0,1.0,1.0,n\n"
        "4,ec013.38,ec013,1,2,EC3,0,0,0,0,0,0,1.0,1.0,p\n",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in tables.items():
            archive.writestr(f"hc3-metadata-tables/{name}", text)


def _registry(session, eeg_bytes, eeg_sha):
    url = crcns_url(f"hc-3/{TOPDIR}/{session}.tar.gz")
    return {
        "dataset": "diba_2007",
        "paper": "Diba & Buzsáki (2007)",
        "paper_citation": "Diba K. A paper.",
        "species": "rat",
        "brain_regions": ["dorsal CA1", "dorsal CA3"],
        "apparatus": "linear tracks",
        "license": "CRCNS terms of use",
        "archive": {"name": "CRCNS hc-3", "citation": "The data."},
        "stated": {
            "sampling_rate": {"value": 1252.0, "basis": "stated", "source": ".xml"},
            "position_span": None,
        },
        "sessions": [
            {
                "session": session,
                "archive": "CRCNS hc-3",
                "url": url,
                "file_name": f"{session}.tar.gz",
                "sha256": "0" * 64,
                "size_bytes": 1,
                "verified": False,
                "default": True,
                "lfp_files": [
                    {
                        "path": f"{session}/{session}.eeg",
                        "member": f"{TOPDIR}/{session}/{session}.eeg",
                        "sha256": eeg_sha,
                        "size_bytes": eeg_bytes,
                    }
                ],
            }
        ],
    }


@pytest.fixture
def spikes_only(tmp_path):
    """A session without positions: two shanks, an LFP of 2 s."""
    folder = tmp_path / "diba_2007" / SESSION
    rng = np.random.default_rng(0)
    eeg = rng.integers(-200, 200, size=(N_LFP, N_CHANNELS)).astype("<i2")
    write_session(
        folder,
        SESSION,
        n_channels=N_CHANNELS,
        eeg=eeg,
        shanks={
            1: ([100, 30000, 40000, 50000], [2, 0, 2, 1], 3),
            2: ([200, 300, 60000, 61000], [3, 4, 5, 3], 6),
        },
    )
    _metadata_zip(folder / diba_2007.METADATA_FILE)
    payload = (folder / f"{SESSION}.eeg").read_bytes()
    registry = _registry(SESSION, len(payload), hashlib.sha256(payload).hexdigest())
    entry = _read(registry, "diba_2007.json")[0]
    rec = run_checks(diba_2007.load_session(entry, folder), stated=entry.stated)
    return rec, eeg


def test_units_are_clusters_from_two_up_with_hc3_labels(spikes_only):
    rec, _ = spikes_only
    assert list(rec.units.index) == ["1.2", "2.3", "2.4", "2.5"]
    assert rec.units["brain_region"].tolist() == ["CA3", "CA1", "CA1", "CA1"]
    labels = rec.units["label"].tolist()
    assert labels[:3] == ["pyramidal", "interneuron", "unidentified"]
    assert labels[3] != labels[3]  # 2.5 is not in the cell table: missing
    np.testing.assert_array_equal(rec.spike_times[0], np.array([100, 40000]) / WIDEBAND)
    np.testing.assert_array_equal(rec.spike_times[1], np.array([200, 61000]) / WIDEBAND)
    assert any("not in hc-3's cell table" in text for text in rec.watch_outs)


def test_without_positions_times_are_the_lfp_clock(spikes_only):
    rec, _ = spikes_only
    assert rec.positions is None
    assert rec.env is None
    np.testing.assert_array_equal(rec.times, np.arange(N_LFP) / 1252.0)
    assert rec.watch_outs[0].startswith("hc-3 holds no position")
    assert "Not tied to Diba & Buzsáki 2007" in rec.watch_outs[1]


def test_lfp_is_read_in_microvolts_on_the_spikes_clock(spikes_only, tmp_path):
    rec, eeg = spikes_only
    assert list(rec.lfp.channels) == ["ch0", "ch1", "ch2", "ch3"]
    assert rec.lfp.channels["ch2"] == "shank 2, CA1"
    samples, times = rec.lfp.fetch(("ch3", "ch0"), cache=tmp_path)
    step = 20 / 2**16 / 1000 * 1e6  # the .xml's range, bits and gain, in µV
    np.testing.assert_allclose(samples, eeg[:, [3, 0]] * step)
    np.testing.assert_array_equal(times, np.arange(N_LFP) / 1252.0)


def test_a_paper_session_says_how_it_is_tied(tmp_path):
    session = "2006-6-12_16-53-46"
    folder = tmp_path / "diba_2007" / session
    eeg = np.zeros((N_LFP, N_CHANNELS), dtype="<i2")
    write_session(
        folder, session, n_channels=N_CHANNELS, eeg=eeg, shanks={1: ([5], [2], 3)}
    )
    _metadata_zip(folder / diba_2007.METADATA_FILE, session=session)
    payload = (folder / f"{session}.eeg").read_bytes()
    registry = _registry(session, len(payload), hashlib.sha256(payload).hexdigest())
    rec = diba_2007.load_session(_read(registry, "diba_2007.json")[0], folder)
    assert session in diba_2007.PAPER_SESSIONS
    assert rec.watch_outs[1].startswith("Inferred to be one of Diba & Buzsáki")


def test_an_unlisted_session_or_another_wideband_rate_is_refused(spikes_only, tmp_path):
    folder = tmp_path / "diba_2007" / SESSION
    entry = _read(_registry("2006-1-2_9-0-0", 16, "0" * 64), "diba_2007.json")[0]
    with pytest.raises(ValueError, match="does not list it"):
        diba_2007.load_session(entry, folder)
    xml = folder / f"{SESSION}.xml"
    xml.write_text(xml.read_text().replace(">32552<", ">20000<"))
    entry = _read(_registry(SESSION, 16, "0" * 64), "diba_2007.json")[0]
    with pytest.raises(ValueError, match="not the 32552 Hz"):
        diba_2007.load_session(entry, folder)


def test_laps_run_from_one_platform_to_the_other():
    times = np.arange(12.0)
    along = np.array([0, 5, 50, 95, 100, 98, 60, 2, 1, 40, 3, 0], dtype=float)
    laps = diba_2007._laps(times, along, 10.0, 90.0)
    np.testing.assert_array_equal(laps["increasing"], [[1.0, 3.0]])
    np.testing.assert_array_equal(laps["decreasing"], [[5.0, 7.0]])


# ---- a fig1.nel in Maboudi et al.'s layout --------------------------------


class _Attribute:
    """What dill pickles NumPy's ``multiarray.<name>`` as: ``_get_attr(module, name)``."""

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *args):  # pragma: no cover - a reduce callable, never called
        raise AssertionError


class _Module:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def nelpy_classes(monkeypatch):
    """Stand-in nelpy classes, importable by pickle while dumping."""
    classes = {}
    for qualified in (
        "nelpy.auxiliary._results.ResultsContainer",
        "nelpy.core._analogsignalarray.AnalogSignalArray",
        "nelpy.core._epocharray.EpochArray",
    ):
        module_name, name = qualified.rsplit(".", 1)
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            prefix = ".".join(parts[:depth])
            if prefix not in sys.modules:
                monkeypatch.setitem(sys.modules, prefix, types.ModuleType(prefix))
        cls = type(name, (), {"__module__": module_name, "__qualname__": name})
        setattr(sys.modules[module_name], name, cls)
        classes[name] = cls
    dill = types.ModuleType("dill.dill")

    def _import_module(name):  # pragma: no cover - only pickled
        raise AssertionError

    def _get_attr(module, name):  # pragma: no cover - only pickled
        raise AssertionError

    for function in (_import_module, _get_attr):
        function.__module__ = "dill.dill"
        function.__qualname__ = function.__name__
        setattr(dill, function.__name__, function)
    monkeypatch.setitem(sys.modules, "dill", types.ModuleType("dill"))
    monkeypatch.setitem(sys.modules, "dill.dill", dill)
    return classes, dill


def _dumps(obj, dill):
    multiarray = _Module("numpy.core.multiarray")

    class DillLike(pickle.Pickler):
        def reducer_override(self, value):
            if isinstance(value, _Module):
                return dill._import_module, (value.name,)
            if isinstance(value, _Attribute):
                return dill._get_attr, (value.module, value.name)
            if isinstance(value, np.ndarray):
                _, args, state = value.__reduce__()
                return _Attribute(multiarray, "_reconstruct"), args, state
            if isinstance(value, np.generic):
                _, args = value.__reduce__()
                return _Attribute(multiarray, "scalar"), args
            return NotImplemented

    stream = io.BytesIO()
    DillLike(stream, protocol=3).dump(obj)
    return stream.getvalue()


def _nelpy(cls, **state):
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def _fig1(classes, times, xy, support):
    epochs = classes["EpochArray"]
    pos2d = _nelpy(
        classes["AnalogSignalArray"],
        _time=times,
        _ydata=xy,
        _fs=np.float64(60.0),
        _support=_nelpy(epochs, _time=support),
    )
    return _nelpy(
        classes["ResultsContainer"],
        exp_data={"16-40-19": {"pos2d": pos2d}},
        aux_data={},
        description=None,
    )


def test_positions_come_from_the_sample_file_in_two_segments(tmp_path, nelpy_classes):
    """A rat shuttling on a 200 cm track, then on a 140 cm one, at 60 Hz."""
    classes, dill = nelpy_classes
    session = diba_2007.POSITION_SESSION
    folder = tmp_path / "diba_2007" / session
    eeg = np.zeros((N_LFP, N_CHANNELS), dtype="<i2")
    write_session(
        folder, session, n_channels=N_CHANNELS, eeg=eeg, shanks={1: ([5], [2], 3)}
    )
    _metadata_zip(folder / diba_2007.METADATA_FILE, session=session, topdir="gor01-6-7")
    rng = np.random.default_rng(1)
    long_t = np.arange(0, 600, 1 / 60)
    short_t = np.arange(660, 1200, 1 / 60)
    x = np.concatenate(
        [
            100 + 100 * np.sin(2 * np.pi * long_t / 20),
            100 + 70 * np.sin(2 * np.pi * short_t / 15),
        ]
    )
    y = 150 + rng.normal(0, 1.0, x.size)
    times = np.concatenate([long_t, short_t])
    support = np.array([[0.0, long_t[-1]], [660.0, short_t[-1]]])
    with gzip.open(folder / "fig1.nel", "wb") as stream:
        stream.write(_dumps(_fig1(classes, times, np.vstack([x, y]), support), dill))
    payload = (folder / f"{session}.eeg").read_bytes()
    registry = _registry(session, len(payload), hashlib.sha256(payload).hexdigest())
    rec = diba_2007.load_session(_read(registry, "diba_2007.json")[0], folder)
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_array_equal(rec.positions, np.column_stack([x, y]))
    np.testing.assert_array_equal(rec.epochs["long_track"], support[:1])
    np.testing.assert_array_equal(rec.epochs["short_track"], support[1:])
    # 30 sine periods of 20 s, each a lap each way; the first starts mid-track.
    assert len(rec.epochs["long_track_laps_increasing"]) in (29, 30)
    assert len(rec.epochs["long_track_laps_decreasing"]) == 30
    assert rec.env.n_bins == pytest.approx(200 / diba_2007.BIN_SIZE, abs=2)
    assert rec.env_kind == "inferred"


def test_a_position_file_of_another_session_is_refused(tmp_path, nelpy_classes):
    classes, dill = nelpy_classes
    obj = _fig1(classes, np.arange(3.0), np.zeros((2, 3)), np.array([[0.0, 2.0]]))
    obj.exp_data = {"11-26-53": obj.exp_data["16-40-19"]}
    path = tmp_path / "fig1.nel"
    with gzip.open(path, "wb") as stream:
        stream.write(_dumps(obj, dill))
    with pytest.raises(ValueError, match="expected one session '16-40-19'"):
        diba_2007._position_file(path)


# ---- real data ---------------------------------------------------------------


@pytest.mark.realdata
def test_default_session():
    rec = load("diba_2007")
    assert rec.session == diba_2007.POSITION_SESSION
    assert rec.verified
    assert len(rec.units) == 87
    assert rec.units["label"].value_counts().to_dict() == {
        "pyramidal": 76,
        "interneuron": 10,
        "unidentified": 1,
    }
    assert rec.units["brain_region"].value_counts().to_dict() == {"CA3": 73, "CA1": 14}
    assert len(rec.times) == 150_600
    assert {name: len(rows) for name, rows in rec.epochs.items() if "laps" in name} == {
        "long_track_laps_increasing": 19,
        "long_track_laps_decreasing": 20,
        "short_track_laps_increasing": 19,
        "short_track_laps_decreasing": 20,
    }
    np.testing.assert_allclose(rec.epochs["long_track"], [[0.0, 1180.0]])
    np.testing.assert_allclose(rec.epochs["short_track"], [[1250.0, 2580.0]])
    assert "'long_track' spans 254 cm" in " ".join(rec.watch_outs)
    assert "'short_track' 164 cm" in " ".join(rec.watch_outs)
    assert rec.lfp.sampling_rate == 1252.0
    assert len(rec.lfp.channels) == 96


@pytest.mark.realdata
def test_every_session_loads_on_its_own_lfp_clock():
    """27 sessions; spikes end within a sample of hc-3's stated session length."""
    from neurospatial.datasets._catalog import sessions

    entries = sessions("diba_2007")
    assert len(entries) == 27
    assert {e.session for e in entries} >= diba_2007.PAPER_SESSIONS
    for entry in entries:
        rec = load("diba_2007", entry.session)
        assert len(rec.units) == entry.n_units, entry.session
        last = max(float(train[-1]) for train in rec.spike_times if train.size)
        if entry.session == diba_2007.POSITION_SESSION:
            assert last == pytest.approx(2587.804436, abs=1e-6)
            continue
        assert rec.positions is None, entry.session
        # hc-3's session table rounds the length; the .eeg holds it in samples.
        assert rec.times[-1] - last < 0.2, entry.session


@pytest.mark.realdata
def test_position_file_spikes_are_hc3_spikes():
    """The conversion's evidence: Diba's sorting's spikes are hc-3's detected spikes."""
    from neurospatial.datasets._fetch import cache_root
    from neurospatial.datasets._neuroscope import read_res_clu
    from neurospatial.datasets._pickled import inert_fields, read_dill_pickle

    load("diba_2007")
    folder = cache_root() / "diba_2007" / diba_2007.POSITION_SESSION
    root = inert_fields(
        read_dill_pickle(
            folder / "fig1.nel", diba_2007.NELPY_CLASSES, diba_2007.STAND_INS
        )
    )
    trains = inert_fields(root["exp_data"]["16-40-19"]["spikes_noIN"])["_time"]
    theirs = np.sort(np.concatenate([np.asarray(t, float) for t in trains]))
    detected, kept = [], []
    for shank in range(1, 13):
        base = folder / diba_2007.POSITION_SESSION
        samples, clusters = read_res_clu(f"{base}.res.{shank}", f"{base}.clu.{shank}")
        detected.append(samples / diba_2007.WIDEBAND_HZ)
        kept.append(samples[clusters >= 2] / diba_2007.WIDEBAND_HZ)

    def matched(reference, lag):
        reference = np.sort(np.concatenate(reference))
        shifted = theirs + lag
        i = np.clip(np.searchsorted(reference, shifted), 1, reference.size - 1)
        near = np.minimum(
            np.abs(reference[i] - shifted), np.abs(reference[i - 1] - shifted)
        )
        return round(float(np.mean(near < 1e-6)), 3)

    assert len(trains) == 103
    assert theirs.size == 258_427
    assert matched(detected, 0.0) == 0.953
    assert matched(kept, 0.0) == 0.623
    assert matched(detected, 0.1) == matched(detected, -0.1) == 0.0
    assert matched(detected, 1.0) == 0.030


@pytest.mark.realdata
def test_lfp_fetches_one_channel_in_microvolts():
    rec = load("diba_2007")
    samples, times = rec.lfp.fetch(("ch72",))
    assert samples.shape == (len(times), 1)
    assert times[-1] == pytest.approx(
        rec.lfp.files[0].size_bytes / 2 / 96 / 1252 - 1 / 1252
    )
    assert rec.lfp.channels["ch72"] == "shank 10, CA1"
    assert 10 < np.std(samples) < 2000  # µV, an LFP's range


@pytest.fixture(scope="module")
def replay():
    """The worked result's pipeline on hc-3's units and on Diba's sorting (about 30 s)."""
    import dataclasses

    import diba_2007_replay as pipeline
    import pandas as pd

    from neurospatial.datasets._fetch import cache_root
    from neurospatial.datasets._pickled import inert_fields, read_dill_pickle

    rec = load("diba_2007")
    hc3 = pipeline.run(rec)
    path = cache_root() / "diba_2007" / diba_2007.POSITION_SESSION / "fig1.nel"
    root = inert_fields(
        read_dill_pickle(path, diba_2007.NELPY_CLASSES, diba_2007.STAND_INS)
    )
    st = inert_fields(root["exp_data"]["16-40-19"]["spikes_noIN"])
    ids = [f"diba.{unit}" for unit in st["_unit_ids"]]
    theirs = dataclasses.replace(
        rec,
        spike_times=tuple(np.sort(np.asarray(t, float)) for t in st["_time"]),
        units=pd.DataFrame(
            {"brain_region": ["unknown"] * len(ids), "label": ["pyramidal"] * len(ids)},
            index=pd.Index(ids, name="unit_id"),
        ),
    )
    return rec, pipeline, hc3, pipeline.run(theirs)


def _counts(pipeline, events, segment):
    chosen = [e for e in events if segment == "pooled" or e.segment == segment]
    found = pipeline.shares(chosen)
    return {
        "events": found["n_events"],
        "forward": found["n_forward"],
        "reverse": found["n_reverse"],
        "forward_before": found["forward_before"],
        "reverse_after": found["reverse_after"],
    }


@pytest.mark.realdata
def test_replay_check_counts(replay):
    rec, pipeline, hc3, theirs = replay
    record = diba_2007.REPLAY_CHECK
    for segment in ("long_track", "short_track"):
        inputs = pipeline.segment_inputs(rec, segment)
        assert {d: int(mask.sum()) for d, mask in inputs["place"].items()} == record[
            "place_cells"
        ][segment]
    for segment in ("long_track", "short_track", "pooled"):
        assert _counts(pipeline, hc3, segment) == record["hc3_units"][segment]
        assert _counts(pipeline, theirs, segment) == record["diba_sorting"][segment]


@pytest.mark.realdata
def test_worked_result_criterion_is_not_met(replay):
    """Fixed before the run: both shares above one half, one-sided binomial p < 0.05."""
    from scipy.stats import binomtest

    _, pipeline, hc3, _ = replay
    found = pipeline.shares(hc3)
    forward = binomtest(
        found["forward_before"], found["n_forward"], 0.5, alternative="greater"
    )
    reverse = binomtest(
        found["reverse_after"], found["n_reverse"], 0.5, alternative="greater"
    )
    met = (
        found["forward_before_share"] > 0.5
        and forward.pvalue < 0.05
        and found["reverse_after_share"] > 0.5
        and reverse.pvalue < 0.05
    )
    assert not met
    assert forward.pvalue == pytest.approx(0.1325, abs=1e-4)
    assert reverse.pvalue < 1e-7
    record = diba_2007.WORKED_RESULT
    assert record["status"] == "not possible yet"
    assert record["attempted"]["value"] == round(found["forward_before_share"], 3)
    assert (
        "18 of 29 forward before the run (62.1%, p = 0.13)"
        in (record["attempted"]["criterion"])
    )
    assert "40 of 45 reverse after it (88.9%" in record["attempted"]["criterion"]


@pytest.mark.realdata
def test_significant_events_are_chance_with_hc3_units_not_with_diba_sorting(replay):
    """The cause REPLAY_CHECK names: hc-3's sorting, not the forward share."""
    from scipy.stats import binomtest

    _, pipeline, hc3, theirs = replay
    ours, diba = pipeline.shares(hc3), pipeline.shares(theirs)

    def excess(found, key):
        return binomtest(
            found[key], found["n_events"], 0.05, alternative="greater"
        ).pvalue

    assert excess(ours, "n_forward") == pytest.approx(0.89, abs=0.01)
    assert excess(ours, "n_reverse") == pytest.approx(0.069, abs=0.001)
    assert excess(diba, "n_forward") < 1e-3
    assert excess(diba, "n_reverse") < 1e-3
    assert round(diba["forward_before_share"], 3) == 0.574
    assert round(diba["reverse_after_share"], 3) == 0.825
