"""Public binning snapshots protect the environment's owned geometry."""

import copy
import pickle
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import KnownFaces
from neurospatial.layout._native_faces import NativeFaces


def _arrays(value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _arrays(getattr(value, field.name), seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item, seen)


@pytest.mark.parametrize("restore", ["direct", "copy", "pickle"])
def test_public_binning_has_independent_array_headers_and_buffers(
    built_environment, restore
):
    env = built_environment
    expected = [(array.shape, array.copy()) for array in _arrays(env._binning)]
    exported = env.binning
    if restore == "copy":
        exported = copy.deepcopy(exported)
    elif restore == "pickle":
        exported = pickle.loads(pickle.dumps(exported))
    arrays = list(_arrays(exported))
    assert len(arrays) == len(expected)
    for exported_array, owned_array in zip(arrays, _arrays(env._binning), strict=True):
        assert not np.shares_memory(exported_array, owned_array)
        assert not exported_array.flags.writeable
        exported_array.shape = (exported_array.size,)
        try:
            exported_array.setflags(write=True)
        except ValueError:
            assert not exported_array.flags.writeable
        else:
            if exported_array.size:
                exported_array.flat[0] = 0
    for actual, (shape, values) in zip(_arrays(env.binning), expected, strict=True):
        assert actual.shape == shape
        np.testing.assert_array_equal(actual, values)


def test_public_binning_read_does_not_build_native_faces(monkeypatch):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )

    def refuse_build(self):
        raise AssertionError("Reading binning must not materialize native faces")

    monkeypatch.setattr(NativeFaces, "build", refuse_build)
    exported = env.binning
    assert isinstance(exported.faces, NativeFaces)
    np.testing.assert_array_equal(exported.bins.centers, env.bin_centers)


def test_from_bins_requires_explicit_faces_for_native_binning():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    exported = env.binning
    with pytest.raises(TypeError, match="explicit KnownFaces or MissingFaces"):
        Environment.from_bins(exported)
    rebuilt = Environment.from_bins(
        replace(exported, faces=KnownFaces(exported.faces.build())), arena=env.arena
    )
    np.testing.assert_array_equal(rebuilt.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(rebuilt.compute_kernel(0.5), env.compute_kernel(0.5))
