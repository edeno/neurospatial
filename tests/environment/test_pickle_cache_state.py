"""Copies and pickle carry geometry without transient analysis work."""

from __future__ import annotations

import copy
import pickle

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
