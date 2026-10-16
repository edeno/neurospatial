"""Derived environments and pickle carry geometry without transient analysis work."""

from __future__ import annotations

import pickle
from functools import cached_property

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


class _TransientWork:
    def __deepcopy__(self, memo):
        raise AssertionError("Transient work must not be copied")

    def __reduce__(self):
        raise AssertionError("Transient work must not be pickled")


def _environment():
    return Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(6.0),) * 2
    )


@pytest.mark.parametrize("method", ["derived", "pickle"])
def test_derived_and_pickle_skip_transient_work_without_clearing_source(method):
    env = _environment()
    key = (0.5, "density")
    transient = _TransientWork()
    env._kernel_cache[key] = transient
    if method == "derived":
        restored = env.with_metadata()
    else:
        restored = pickle.loads(pickle.dumps(env))

    assert env._kernel_cache[key] is transient
    assert restored._kernel_cache == {}
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(
        restored.compute_kernel(0.5), _environment().compute_kernel(0.5)
    )


def test_pickle_size_does_not_grow_after_analysis():
    env = _environment()
    cold = pickle.dumps(env)
    kernel = env.compute_kernel(0.5)
    env.get_differential_operator()
    env.get_divergence_operator()
    _ = env.boundary_bins

    warm = pickle.dumps(env)

    assert len(warm) <= len(cold) + 1024
    np.testing.assert_array_equal(env.compute_kernel(0.5), kernel)
    np.testing.assert_array_equal(pickle.loads(warm).compute_kernel(0.5), kernel)


def _copy_environment(env, method):
    if method == "derived":
        return env.with_metadata()
    return pickle.loads(pickle.dumps(env))


@pytest.mark.parametrize("method", ["derived", "pickle"])
def test_computed_properties_are_not_copied_or_pickled(method):
    env = _environment()
    env.get_differential_operator()
    env.get_divergence_operator()
    env.get_bin_attributes()
    env.get_edge_attributes()
    _ = env.boundary_bins, env.bin_sizes, env.linearization_properties
    _ = env._diffusion_geometry, env._diffusion_eigenbasis, env._mrf_eigenbasis
    _ = env._gaussian_kernels, env._source_flat_to_active_node_id_map
    transient = _TransientWork()
    keys = []
    for cls in type(env).__mro__:
        for name, descriptor in vars(cls).items():
            if not isinstance(descriptor, cached_property):
                continue
            assert name in vars(env)
            vars(env)[name] = transient
            keys.append(name)

    restored = _copy_environment(env, method)

    assert all(vars(env)[key] is transient for key in keys)
    assert not any(key in vars(restored) for key in keys)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(restored.bin_sizes, np.ones(env.n_bins))
    assert restored.get_differential_operator().shape == (
        env.n_bins,
        len(env.edge_pairs),
    )


def test_pickle_round_trip_keeps_published_values_immutable():
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        grid_edges=(np.arange(4.0),) * 2,
        name="arena",
        units="cm",
    )
    payload = env.to_dict()
    payload["graph"]["graph"]["recording"] = {"tags": ["original"]}
    env = Environment.from_dict(payload)

    restored = pickle.loads(pickle.dumps(env))

    assert (restored.name, restored.units) == (env.name, env.units)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    for array in (restored.bin_centers, restored.grid_edges[0], restored.active_mask):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(nx.NetworkXError):
        restored.connectivity.add_node(99)
    with pytest.raises(TypeError, match="read-only"):
        restored.connectivity.nodes[0]["pos"] = (99.0, 99.0)
    exported = restored.connectivity.graph
    exported["recording"]["tags"].append("caller")
    assert restored.connectivity.graph["recording"]["tags"] == ["original"]
