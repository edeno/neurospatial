"""Derived environments and pickle carry geometry without transient analysis work."""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment._cache import DIFFUSION_KERNELS


class _TransientWork:
    def __deepcopy__(self, memo):
        raise AssertionError("Transient work must not be copied")

    def __reduce__(self):
        raise AssertionError("Transient work must not be pickled")


def _environment():
    return Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(np.arange(6.0),) * 2
    )


def _refuse_build():
    raise AssertionError("the source cache was cleared")


def _derive_or_pickle(env, method):
    if method == "derived":
        return env.subset(mask=np.ones(env.n_bins, dtype=bool))
    return pickle.loads(pickle.dumps(env))


@pytest.mark.parametrize("method", ["derived", "pickle"])
def test_derived_and_pickle_skip_transient_work_without_clearing_source(method):
    env = _environment()
    key = (0.5, "density")
    transient = _TransientWork()
    env._cache.get_recent(DIFFUSION_KERNELS, key, lambda: transient)

    restored = _derive_or_pickle(env, method)

    assert env._cache.get_recent(DIFFUSION_KERNELS, key, _refuse_build) is transient
    assert len(restored._cache) == 0
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


@pytest.mark.parametrize("method", ["derived", "pickle"])
def test_computed_analyses_are_not_copied_or_pickled(method):
    env = _environment()
    expected_operator = env.get_differential_operator().toarray()
    env.get_divergence_operator()
    env.get_bin_attributes()
    env.get_edge_attributes()
    env.compute_gaussian_kernel(1.0)
    env.nearest_bin(env.bin_centers)
    _ = env.boundary_bins
    env.diffuse(np.ones(env.n_bins), 1.0)
    env.get_mrf_basis(np.ones(env.n_bins), rank=3)
    populated = len(env._cache)
    assert populated >= 10

    restored = _derive_or_pickle(env, method)

    assert len(env._cache) == populated
    assert len(restored._cache) == 0
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(restored.bin_sizes, np.ones(env.n_bins))
    np.testing.assert_array_equal(
        restored.get_differential_operator().toarray(), expected_operator
    )


def test_pickle_round_trip_keeps_published_values_immutable():
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        grid_edges=(np.arange(4.0),) * 2,
        name="arena",
        units="cm",
    )

    restored = pickle.loads(pickle.dumps(env))

    assert (restored.name, restored.units) == (env.name, env.units)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    for array in (restored.bin_centers, restored.grid_edges[0], restored.active_mask):
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0
    with pytest.raises(nx.NetworkXError):
        restored.connectivity.add_node(99)
    with pytest.raises(TypeError, match="read-only"):
        restored.connectivity.nodes[0]["pos"] = (99.0, 99.0)
    with pytest.raises(TypeError, match="read-only"):
        restored.connectivity.graph["caller"] = 1
