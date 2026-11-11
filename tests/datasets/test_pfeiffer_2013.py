"""Pfeiffer & Foster (2013), one session in Maboudi et al. 2018's repository.

Synthetic tests write a tiny gzip pickle laid out as ``openfield.nel`` is: nelpy
objects (stand-in classes of the same module and name), NumPy arrays and
scalars reached through ``dill``'s ``_import_module`` and ``_get_attr``, protocol
3. Real-data tests read the deposited file.
"""

from __future__ import annotations

import gzip
import io
import pickle
import sys
import types

import numpy as np
import pytest

from neurospatial.datasets import load, pfeiffer_2013
from neurospatial.datasets._catalog import _read
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._pickled import (
    DillUnpickler,
    InertObject,
    inert_fields,
    pickle_imports,
    read_dill_pickle,
)

SESSION = "rat1_day1"


class _Attribute:
    """What dill pickles NumPy's ``multiarray.<name>`` as: ``_get_attr(module, name)``."""

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *args):  # pragma: no cover - a reduce callable, never called
        raise AssertionError


class _Module:
    """What dill pickles a module as: ``_import_module(name)``."""

    def __init__(self, name):
        self.name = name


@pytest.fixture
def nelpy_and_dill(monkeypatch):
    """Stand-in ``nelpy`` classes and ``dill`` helpers, importable by pickle while dumping."""
    classes = {}
    for qualified in sorted(pfeiffer_2013.NELPY_CLASSES):
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

    def _import_module(name):  # pragma: no cover - only pickled, never called
        raise AssertionError

    def _get_attr(module, name):  # pragma: no cover - only pickled, never called
        raise AssertionError

    for function in (_import_module, _get_attr):
        function.__module__ = "dill.dill"
        function.__qualname__ = function.__name__
        setattr(dill, function.__name__, function)
    monkeypatch.setitem(sys.modules, "dill", types.ModuleType("dill"))
    monkeypatch.setitem(sys.modules, "dill.dill", dill)
    return classes, dill


def _dumps(obj, dill):
    """Pickle ``obj`` as dill does: NumPy's reconstructors through ``_get_attr``."""
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


def _session_object(classes, *, n_units=4, significant=(0, 2)):
    """A ResultsContainer holding one Rat1/Day1 session, as openfield.nel does."""
    rng = np.random.default_rng(0)
    times = 100.0 + np.arange(20_000) / 50.0  # 400 s at 50 Hz
    loop = 2 * np.pi * times / 37.0
    xy = np.vstack([190.0 + 95.0 * np.sin(loop), 100.0 + 95.0 * np.sin(1.3 * loop)])
    trains = np.empty(n_units, dtype=object)
    for unit in range(n_units):
        trains[unit] = np.sort(rng.uniform(times[0], times[-1], 50 + unit))
    windows = np.array([[150.0, 150.2], [200.0, 200.1], [250.0, 250.3]])
    epochs = classes["EpochArray"]
    st = _nelpy(
        classes["SpikeTrainArray"],
        _unit_ids=list(range(1, n_units + 1)),
        _unit_labels=[str(unit) for unit in range(1, n_units + 1)],
        _time=trains,
        _fs=30000,
        _support=_nelpy(epochs, _time=np.array([[times[0], times[-1]]])),
    )
    st.iloc = _nelpy(classes["ItemGetter_iloc"], obj=st)
    pos = _nelpy(
        classes["AnalogSignalArray"],
        _time=times,
        _ydata=xy,
        _fs=np.float64(50.0),
        _support=_nelpy(epochs, _time=np.array([[times[0], times[-1]]])),
    )
    day = {
        "st": st,
        "pos": pos,
        "ExcitatoryNeuronIDs": np.arange(1, n_units, dtype=np.uint8),
        "InhibitoryNeuronIDs": np.array([n_units], dtype=np.uint8),
        "BradRippleStartEndTimes": _nelpy(epochs, _time=windows),
        "BradSignificantRippleIDs": np.array(significant, dtype=np.uint16),
    }
    return _nelpy(
        classes["ResultsContainer"],
        ratdays=[("Rat1", "Day1")],
        data={"Rat1": {"Day1": day}},
        description="example session in open field",
        _index=0,
    )


def _write(path, obj, dill):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        stream.write(_dumps(obj, dill))
    return path


@pytest.fixture
def written(tmp_path, nelpy_and_dill):
    classes, dill = nelpy_and_dill
    path = _write(tmp_path / "openfield.nel", _session_object(classes), dill)
    return path, classes, dill


def _registry():
    return {
        "dataset": "pfeiffer_2013",
        "paper": "Pfeiffer & Foster (2013)",
        "paper_citation": "Pfeiffer BE. A paper.",
        "species": "rat",
        "brain_regions": ["dorsal CA1"],
        "apparatus": "2 m arena",
        "license": "MIT",
        "archive": {"name": "GitHub", "citation": "The data."},
        "stated": {
            "sampling_rate": {"value": 50.0, "basis": "inferred", "source": "file"},
            "position_span": {
                "value": [160.0, 240.0],
                "basis": "stated",
                "source": "Methods",
            },
        },
        "sessions": [
            {
                "session": SESSION,
                "archive": "GitHub",
                "url": "https://example.org/openfield.nel",
                "file_name": "openfield.nel",
                "sha256": "0" * 64,
                "size_bytes": 1,
                "verified": False,
                "default": True,
            }
        ],
    }


def _load(path):
    entry = _read(_registry(), "pfeiffer_2013.json")[0]
    return run_checks(pfeiffer_2013.load_session(entry, path), stated=entry.info.stated)


def test_the_stand_in_pickle_has_the_files_imports(written, nelpy_and_dill):
    """The synthetic file names the globals the real one does, and no others."""
    path, _, _ = written
    expected = {
        "dill.dill._get_attr",
        "dill.dill._import_module",
        "numpy.dtype",
        "numpy.ndarray",
        "nelpy.auxiliary._results.ResultsContainer",
        "nelpy.core._analogsignalarray.AnalogSignalArray",
        "nelpy.core._epocharray.EpochArray",
        "nelpy.core._spiketrain.ItemGetter_iloc",
        "nelpy.core._spiketrain.SpikeTrainArray",
    }
    assert pickle_imports(gzip.decompress(path.read_bytes())) == expected


def test_reads_the_session_as_inert_containers(written):
    path, _, _ = written
    root = read_dill_pickle(path, pfeiffer_2013.NELPY_CLASSES)
    assert isinstance(root, InertObject)
    assert root.qualified_name == "nelpy.auxiliary._results.ResultsContainer"
    day = inert_fields(root)["data"]["Rat1"]["Day1"]
    position = inert_fields(day["pos"])
    assert position["_ydata"].shape == (2, 20_000)
    assert position["_fs"] == np.float64(50.0)
    trains = inert_fields(day["st"])
    # The self-reference through the item getter is kept, not run.
    assert inert_fields(trains["iloc"])["obj"] is day["st"]


def test_loads_spikes_positions_labels_and_events(written):
    path, _, _ = written
    rec = _load(path)
    assert rec.failed_checks == ()
    assert rec.n_units == 4
    assert list(rec.units.index) == [1, 2, 3, 4]
    assert list(rec.units["label"]) == ["excitatory"] * 3 + ["inhibitory"]
    assert rec.position_units == "cm"
    # x and y are the stored rows, as samples.
    loop = 2 * np.pi * rec.times / 37.0
    np.testing.assert_array_equal(rec.positions[:, 0], 190.0 + 95.0 * np.sin(loop))
    np.testing.assert_array_equal(
        rec.positions[:, 1], 100.0 + 95.0 * np.sin(1.3 * loop)
    )
    assert rec.events["ripples"].shape == (3, 2)
    np.testing.assert_array_equal(rec.params["significant"], [True, False, True])
    np.testing.assert_array_equal(
        rec.events["significant_ripples"], rec.events["ripples"][[0, 2]]
    )
    np.testing.assert_array_equal(rec.epochs["session"], [[100.0, rec.times[-1]]])


def test_declares_the_stated_two_metre_square_on_the_samples(written):
    path, _, _ = written
    rec = _load(path)
    assert rec.env.units == "cm"
    # 2 cm bins covering the 2 m square, placed where the samples are.
    np.testing.assert_allclose(rec.env.bin_sizes.sum(), 200.0**2, rtol=0.03)
    assert np.all(rec.env.bin_at(rec.positions) >= 0)
    assert rec.conversions[-1].text.startswith(
        "environment: the stated 200 x 200 cm arena centred at"
    )


def test_position_gaps_are_counted_and_kept(tmp_path, nelpy_and_dill):
    classes, dill = nelpy_and_dill
    obj = _session_object(classes)
    position = obj.data["Rat1"]["Day1"]["pos"]
    keep = np.ones(position._time.size, dtype=bool)
    keep[1000:1010] = False  # one interval of 0.22 s
    position._time = position._time[keep]
    position._ydata = position._ydata[:, keep]
    rec = _load(_write(tmp_path / "gaps.nel", obj, dill))
    assert rec.times.size == keep.sum()
    finding = next(text for text in rec.watch_outs if "position interval" in text)
    assert finding.startswith("1 position interval is longer than 2 median")
    assert "the longest 0.220 s" in finding


def test_unit_lists_that_do_not_partition_the_units_are_refused(
    tmp_path, nelpy_and_dill
):
    classes, dill = nelpy_and_dill
    obj = _session_object(classes)
    obj.data["Rat1"]["Day1"]["InhibitoryNeuronIDs"] = np.array([3], dtype=np.uint8)
    with pytest.raises(ValueError, match="1 in both, 1 in neither"):
        _load(_write(tmp_path / "bad.nel", obj, dill))


def test_significant_ids_beyond_the_windows_are_refused(tmp_path, nelpy_and_dill):
    classes, dill = nelpy_and_dill
    obj = _session_object(classes, significant=(1, 3))  # 1-based would reach 3
    with pytest.raises(ValueError, match="0-based indices into 3 windows"):
        _load(_write(tmp_path / "bad.nel", obj, dill))


class _Exploit:
    def __reduce__(self):
        return (print, ("ran",))


def test_a_malicious_global_is_refused_before_anything_runs(
    tmp_path, nelpy_and_dill, capsys
):
    classes, dill = nelpy_and_dill
    obj = _session_object(classes)
    obj.data["Rat1"]["Day1"]["payload"] = _Exploit()
    path = _write(tmp_path / "evil.nel", obj, dill)
    with pytest.raises(pickle.UnpicklingError, match=r"imports \['builtins\.print'\]"):
        read_dill_pickle(path, pfeiffer_2013.NELPY_CLASSES)
    assert capsys.readouterr().out == ""


def test_a_module_reached_through_dill_is_refused(tmp_path, nelpy_and_dill):
    """``_import_module('os')`` names no global, so only the unpickler can refuse it."""
    _, dill = nelpy_and_dill

    class DillLike(pickle.Pickler):
        def reducer_override(self, value):
            if isinstance(value, _Module):
                return dill._import_module, (value.name,)
            if isinstance(value, _Attribute):
                return dill._get_attr, (value.module, value.name)
            return NotImplemented

    stream = io.BytesIO()
    DillLike(stream, protocol=3).dump({"x": _Attribute(_Module("os"), "system")})
    blob = stream.getvalue()
    assert pickle_imports(blob) == {"dill.dill._get_attr", "dill.dill._import_module"}
    with pytest.raises(pickle.UnpicklingError, match=r"_import_module\('os'\)"):
        DillUnpickler(io.BytesIO(blob), ()).load()
    stream = io.BytesIO()
    DillLike(stream, protocol=3).dump(
        {"x": _Attribute(_Module("numpy.core.multiarray"), "frombuffer")}
    )
    with pytest.raises(pickle.UnpicklingError, match="'frombuffer'"):
        DillUnpickler(io.BytesIO(stream.getvalue()), ()).load()


def test_a_class_outside_the_list_is_refused(written):
    path, _, _ = written
    allowed = pfeiffer_2013.NELPY_CLASSES - {"nelpy.core._epocharray.EpochArray"}
    with pytest.raises(pickle.UnpicklingError, match=r"_epocharray\.EpochArray"):
        read_dill_pickle(path, allowed)


def test_an_object_scalar_is_refused(nelpy_and_dill):
    """``scalar`` reads raw bytes only into a numeric dtype."""
    _, dill = nelpy_and_dill
    blob = _dumps({"x": np.float64(1.5)}, dill)
    assert DillUnpickler(io.BytesIO(blob), ()).load() == {"x": 1.5}
    forged = blob.replace(b"f8", b"O8")
    with pytest.raises(pickle.UnpicklingError, match="refusing scalar"):
        DillUnpickler(io.BytesIO(forged), ()).load()


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """Rat 1 day 1: the paper's 212 units, 203 excitatory and 9 inhibitory."""
    rec = load("pfeiffer_2013")
    assert_default_session(rec, session_entry("pfeiffer_2013"))
    assert rec.units["label"].value_counts().to_dict() == {
        "excitatory": 203,
        "inhibitory": 9,
    }
    assert sum(train.size for train in rec.spike_times) == 1_201_998
    assert rec.events["ripples"].shape == (322, 2)
    assert int(rec.params["significant"].sum()) == 110
    assert round(rec.sampling_rate, 2) == 57.43
    span = np.ptp(rec.positions, axis=0)
    np.testing.assert_allclose(span, [198.6, 191.6], atol=0.05)
    # Every sample lies in the declared arena.
    assert np.all(rec.env.bin_at(rec.positions) >= 0)
