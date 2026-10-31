"""Optional dependencies: a missing install names its extra, a broken one does not."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from neurospatial._results import import_xarray
from neurospatial.io import to_pynapple
from neurospatial.io.nwb._core import (
    _require_ndx_events,
    _require_ndx_pose,
    _require_pynwb,
)
from neurospatial.io.pynapple import _require_pynapple

REQUIRERS = [
    pytest.param(_require_pynwb, "pynwb", "nwb", id="pynwb"),
    pytest.param(_require_ndx_pose, "ndx_pose", "nwb", id="ndx_pose"),
    pytest.param(_require_ndx_events, "ndx_events", "nwb", id="ndx_events"),
    pytest.param(_require_pynapple, "pynapple", "pynapple", id="pynapple"),
    pytest.param(import_xarray, "xarray", "xarray", id="xarray"),
]


@pytest.mark.parametrize(("require", "module", "extra"), REQUIRERS)
def test_missing_install_names_the_extra(require, module, extra, monkeypatch):
    monkeypatch.setitem(sys.modules, module, None)  # import raises "not found"
    with pytest.raises(ImportError, match=rf"neurospatial\[{extra}\]") as excinfo:
        require()
    assert f"{module} is required for" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(("require", "module", "extra"), REQUIRERS)
def test_broken_dependency_is_not_reported_as_missing_install(
    require, module, extra, monkeypatch, tmp_path
):
    # An installed package whose own import fails on a missing dependency.
    package = tmp_path / module
    package.mkdir()
    (package / "__init__.py").write_text("import _neurospatial_absent_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        require()
    assert excinfo.value.name == "_neurospatial_absent_dependency"
    assert f"neurospatial[{extra}]" not in str(excinfo.value)


def test_to_pynapple_without_pynapple_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynapple", None)
    times = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ImportError, match=r"neurospatial\[pynapple\]"):
        to_pynapple(times, np.zeros(3))
