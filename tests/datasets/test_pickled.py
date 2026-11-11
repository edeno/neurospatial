"""Stand-ins for the functions, types and containers a dill pickle names."""

from __future__ import annotations

import gzip
import io
import pickle

import pytest

from neurospatial.datasets._pickled import (
    DillUnpickler,
    InertStandIn,
    pickle_imports,
    read_dill_pickle,
)

# A dill-style pickle, written opcode by opcode: {"f": a function built by
# _create_function from a code object _load_type("CodeType") makes, with its
# module's namespace; "d": a deque of 1 and 2}.
_STREAM = (
    b"\x80\x02}("
    b"Vf\n"
    b"cdill.dill\n_create_function\n("
    b"cdill.dill\n_load_type\n(VCodeType\ntR(K\x01K\x02tR"
    b"chmmlearn.base\n__dict__\n"
    b"tR"
    b"Vd\n"
    b"ccollections\ndeque\n)R(K\x01K\x02e"
    b"u."
)
_STAND_INS = {
    "dill.dill._create_function",
    "dill.dill._load_type",
    "hmmlearn.base.__dict__",
    "collections.deque",
}


def test_the_stream_names_what_dill_names():
    assert pickle_imports(_STREAM) == _STAND_INS


def test_stand_ins_hold_the_parts_and_run_nothing():
    found = DillUnpickler(io.BytesIO(_STREAM), (), _STAND_INS).load()
    function, deque = found["f"], found["d"]
    assert isinstance(function, InertStandIn)
    assert function.qualified_name == "dill.dill._create_function"
    code, namespace = function.args
    assert isinstance(code, InertStandIn)
    assert code.args == (1, 2)
    assert namespace.qualified_name == "hmmlearn.base.__dict__"
    assert deque.items == [1, 2]


def test_without_stand_ins_the_function_is_refused(tmp_path):
    with pytest.raises(pickle.UnpicklingError, match="_create_function"):
        DillUnpickler(io.BytesIO(_STREAM), ()).load()
    path = tmp_path / "file.nel"
    path.write_bytes(gzip.compress(_STREAM))
    with pytest.raises(pickle.UnpicklingError, match=r"refusing file\.nel"):
        read_dill_pickle(path, ())
    assert read_dill_pickle(path, (), _STAND_INS)["d"].items == [1, 2]


def test_a_stand_in_is_not_a_class_of_the_file():
    """Listing a name as a stand-in does not let another name through."""
    stream = _STREAM.replace(b"collections\ndeque", b"os\nsystem")
    with pytest.raises(pickle.UnpicklingError, match=r"os\.system"):
        DillUnpickler(io.BytesIO(stream), (), _STAND_INS).load()
