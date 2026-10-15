"""Copies and pickle carry geometry without transient analysis work."""

from __future__ import annotations

import copy
import pickle
from functools import cached_property

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


@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
def test_copy_and_pickle_skip_transient_work_without_clearing_source(method):
    env = _environment()
    key = (0.5, "density")
    transient = _TransientWork()
    env._kernel_cache[key] = transient
    if method == "copy":
        restored = env.copy()
    elif method == "deepcopy":
        restored = copy.deepcopy(env)
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
    if method == "shallow":
        return copy.copy(env)
    if method == "copy":
        return env.copy()
    if method == "deepcopy":
        return copy.deepcopy(env)
    return pickle.loads(pickle.dumps(env))


@pytest.mark.parametrize("method", ["shallow", "copy", "deepcopy", "pickle"])
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


@pytest.mark.parametrize("method", ["shallow", "copy", "deepcopy", "pickle"])
def test_legacy_versioned_cache_payloads_are_discarded(method):
    """Retired analyses are disposable even without a current descriptor."""
    env = _environment()
    key = "_versioned_cache___retired_analysis"
    transient = _TransientWork()
    vars(env)[key] = transient
    vars(env)[f"{key}__version"] = 1
    vars(env)["_state_version"] = 1

    restored = _copy_environment(env, method)

    assert vars(env)[key] is transient
    assert vars(env)[f"{key}__version"] == 1
    assert vars(env)["_state_version"] == 1
    assert not any(name.startswith("_versioned_cache__") for name in vars(restored))
    assert "_state_version" not in vars(restored)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


def test_loading_builder_free_pickle_drops_legacy_analysis_state(monkeypatch):
    env = _environment()
    key = "_versioned_cache___retired_analysis"
    vars(env)[key] = np.ones((env.n_bins, env.n_bins))
    vars(env)[f"{key}__version"] = 1
    vars(env)["_state_version"] = 1
    # Emulate the old writer, which included memoized work in instance state.
    with monkeypatch.context() as patch:
        patch.setattr(Environment, "__getstate__", lambda self: self.__dict__.copy())
        saved = pickle.dumps(env)

    restored = pickle.loads(saved)

    assert key in vars(env)
    assert not any(name.startswith("_versioned_cache__") for name in vars(restored))
    assert "_state_version" not in vars(restored)
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )
    np.testing.assert_array_equal(restored.bin_sizes, np.ones(env.n_bins))
