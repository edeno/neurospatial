from __future__ import annotations

import collections
import io
import os
import pickle
import zipfile

import numpy as np
import pytest
from load_gardner_2022_grid import pickle_imports, read_npz


class _Exploit:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def __reduce__(self):
        return (os.system, (f"touch {self.sentinel}",))


def _npz_with_object(path, value, **plain):
    holder = np.empty((), dtype=object)
    holder[()] = value
    np.savez(path, spikes_mod1=holder, **plain)
    return path


def test_pickle_imports_lists_globals_without_running(tmp_path):
    sentinel = tmp_path / "ran"
    blob = pickle.dumps(_Exploit(sentinel), protocol=4)
    assert pickle_imports(blob) == {f"{os.system.__module__}.system"}
    assert not sentinel.exists()

    trains = np.empty((), dtype=object)
    trains[()] = {0: np.array([0.5, 1.5]), 1: np.array([2.0])}
    found = pickle_imports(pickle.dumps(trains, protocol=3))
    assert found
    assert all(name.startswith("numpy") for name in found)


def test_read_npz_reads_spike_dict_and_plain_arrays(tmp_path):
    path = _npz_with_object(
        tmp_path / "ok.npz",
        {0: np.array([0.5, 1.5]), 3: np.array([2.0])},
        t=np.arange(3.0),
    )
    arrays = read_npz(path)
    spikes = arrays["spikes_mod1"][()]
    np.testing.assert_array_equal(spikes[3], [2.0])
    np.testing.assert_array_equal(arrays["t"], [0.0, 1.0, 2.0])


def test_restricted_npz_refuses_unlisted_global(tmp_path):
    path = _npz_with_object(
        tmp_path / "bad.npz", collections.OrderedDict({0: np.array([1.0])})
    )
    with pytest.raises(pickle.UnpicklingError, match="OrderedDict"):
        read_npz(path)


def test_restricted_npz_refuses_exploit_before_running_it(tmp_path):
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
