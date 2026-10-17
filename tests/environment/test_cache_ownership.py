"""Public exports cannot mutate an environment's private analysis caches."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment
from neurospatial.ops.calculus import divergence, gradient


@pytest.fixture
def env():
    """A small unit-cell grid with nontrivial boundary and interior bins."""
    return Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )


def _restore(env, restoration):
    if restoration == "deepcopy":
        return copy.deepcopy(env)
    if restoration == "pickle":
        return pickle.loads(pickle.dumps(env))
    return env


@pytest.mark.parametrize("restoration", ["fresh", "deepcopy", "pickle"])
@pytest.mark.parametrize("mode", ["density", "transition", "average"])
@pytest.mark.parametrize("cache", [False, True])
def test_kernel_exports_do_not_change_later_kernels(env, restoration, mode, cache):
    expected = env.compute_kernel(bandwidth=0.5, mode=mode, cache=cache).copy()
    env = _restore(env, restoration)
    exported = env.compute_kernel(bandwidth=0.5, mode=mode, cache=cache)
    exported[:] = 0.0
    exported.shape = (env.n_bins, env.n_bins, 1)

    np.testing.assert_array_equal(
        env.compute_kernel(bandwidth=0.5, mode=mode, cache=cache), expected
    )


def test_kernel_exports_reuse_computation(env, monkeypatch):
    """Detached exports retain the cache's expensive-computation benefit."""
    from neurospatial.environment import _fields

    original = _fields.heat_kernel_from_W
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_fields, "heat_kernel_from_W", counted)
    first = env.compute_kernel(bandwidth=0.5)
    second = env.compute_kernel(bandwidth=0.5)
    assert len(calls) == 1
    assert first is not second
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "method", ["get_differential_operator", "get_divergence_operator"]
)
@pytest.mark.parametrize("mutation", ["values", "replace_buffer"])
def test_sparse_operator_exports_cannot_change_calculus(env, method, mutation):
    field = np.arange(env.n_bins, dtype=float)
    edge_field = np.arange(env.connectivity.number_of_edges(), dtype=float)
    expected_gradient = gradient(env, field)
    expected_divergence = divergence(env, edge_field)
    operator = getattr(env, method)()
    if mutation == "values":
        operator.data[:] = 0.0
    else:
        operator.data.setflags(write=False)
        operator.data = np.zeros_like(operator.data)

    np.testing.assert_array_equal(gradient(env, field), expected_gradient)
    np.testing.assert_array_equal(divergence(env, edge_field), expected_divergence)


@pytest.mark.parametrize("restoration", ["fresh", "deepcopy", "pickle"])
@pytest.mark.parametrize("method", ["get_bin_attributes", "get_edge_attributes"])
def test_attribute_exports_own_numeric_buffers(env, restoration, method):
    env = env
    expected = getattr(env, method)().copy(deep=True)
    env = _restore(env, restoration)
    exported = getattr(env, method)()
    column = "pos_dim0" if method == "get_bin_attributes" else "distance"
    exported.loc[exported.index[0], column] = 999.0
    exported["caller_column"] = 0.0

    pd.testing.assert_frame_equal(getattr(env, method)(), expected)


@pytest.mark.parametrize("method", ["get_bin_attributes", "get_edge_attributes"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_attribute_exports_own_axis_buffers(env, method, axis):
    expected = getattr(getattr(env, method)(), axis).tolist()
    exported = getattr(env, method)()
    labels = getattr(exported, axis).to_numpy(copy=False)
    labels[0] = 99 if axis == "index" else "caller_column"

    assert getattr(getattr(env, method)(), axis).tolist() == expected


@pytest.mark.parametrize("restoration", ["fresh", "deepcopy", "pickle"])
def test_boundary_exports_are_detached(env, restoration):
    expected = env.boundary_bins.copy()
    env = _restore(env, restoration)
    exported = env.boundary_bins
    exported[:] = -1
    np.testing.assert_array_equal(env.boundary_bins, expected)
    env.clear_cache()
    np.testing.assert_array_equal(env.boundary_bins, expected)


@pytest.mark.parametrize("restoration", ["fresh", "deepcopy", "pickle"])
def test_linearization_metadata_exports_are_detached(env, restoration):
    assert env.linearization_properties == {"is_linearized_track": False}
    env = _restore(env, restoration)
    exported = env.linearization_properties
    exported["is_linearized_track"] = True
    exported["caller_metadata"] = {"label": "edited"}
    assert env.linearization_properties == {"is_linearized_track": False}


def test_bin_sizes_array_metadata_cannot_change_published_sizes(env):
    exported = env.bin_sizes
    assert not exported.flags.writeable
    exported.shape = (3, 3)
    np.testing.assert_array_equal(env.bin_sizes, np.ones(9))


@pytest.mark.parametrize(
    "name", ["boundary_bins", "bin_sizes", "linearization_properties"]
)
@pytest.mark.parametrize("access_first", [False, True])
@pytest.mark.parametrize("operation", ["set", "delete"])
def test_public_cached_properties_cannot_be_rebound(env, name, access_first, operation):
    if access_first:
        getattr(env, name)
    with pytest.raises((AttributeError, TypeError)):
        if operation == "set":
            setattr(env, name, "caller value")
        else:
            delattr(env, name)
