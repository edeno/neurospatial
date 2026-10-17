"""Public numerical operations preserve private cache ownership and reuse."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import DiffusionGeometry
from neurospatial.encoding._smoothing import _get_gaussian_kernel
from neurospatial.environment import CartesianSystem, EgocentricPolarSystem
from neurospatial.ops import MRFBasis
from neurospatial.ops.basis import _diffusion_generator
from neurospatial.ops.diffusion import (
    diffusion_kernel,
)


def _grid(n=8):
    return Environment.from_grid_mask(
        np.ones((n, n), dtype=bool), grid_edges=(np.arange(n + 1.0),) * 2
    )


@pytest.mark.parametrize("polar", [False, True])
def test_coordinate_descriptor_is_an_independent_frozen_value(polar):
    env = (
        Environment.from_polar_egocentric(
            distance_range=(0.0, 3.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=1.0,
            angle_bin_size=np.pi / 2,
        )
        if polar
        else _grid()
    )
    first, second = env.coordinate_system, env.coordinate_system
    assert isinstance(first, EgocentricPolarSystem if polar else CartesianSystem)
    assert first == second and first is not second
    with pytest.raises(FrozenInstanceError):
        first.equal_aspect = False
    with pytest.raises(AttributeError):
        env.coordinate_system = first
    if polar:
        vars(first)["circular"] = False
        assert env.coordinate_system.circular


def _geometry_arrays(geometry):
    return (
        geometry.W.data,
        geometry.W.indices,
        geometry.W.indptr,
        geometry.volumes,
        geometry.labels,
    )


def test_diffusion_exports_reuse_buffers_without_exposing_cache_headers(monkeypatch):
    from neurospatial.environment import _fields

    env = _grid()
    build = _fields._build_diffusion_geometry
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(_fields, "_build_diffusion_geometry", counted)
    first = env.get_diffusion_geometry()
    second = env.get_diffusion_geometry()
    assert isinstance(first, DiffusionGeometry)
    assert len(calls) == 1
    assert first.W is not second.W
    expected = [array.copy() for array in _geometry_arrays(first)]
    for exported, other in zip(
        _geometry_arrays(first), _geometry_arrays(second), strict=True
    ):
        assert exported is not other
        assert np.shares_memory(exported, other)
        with pytest.raises(ValueError, match="read-only"):
            exported[...] = 0
    first.W.data = np.zeros_like(first.W.data)
    first.W.indices = np.zeros_like(first.W.indices)
    first.W.indptr = np.zeros_like(first.W.indptr)
    first.volumes.shape = (env.n_bins, 1)
    first.labels.shape = (env.n_bins, 1)
    fresh = env.get_diffusion_geometry()
    for actual, wanted in zip(_geometry_arrays(fresh), expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    assert len(calls) == 1
    env.clear_cache()
    rebuilt = env.get_diffusion_geometry()
    assert len(calls) == 2
    assert not np.shares_memory(rebuilt.W.data, second.W.data)


def test_diffusion_components_follow_conductance_instead_of_navigation():
    env = Environment.from_grid_mask(
        np.eye(2, dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    assert len(env.edge_pairs) == 1
    geometry = env.get_diffusion_geometry()
    assert geometry.n_components == 2
    np.testing.assert_array_equal(geometry.labels, [0, 1])


def test_gaussian_exports_reuse_builds_and_protect_headers(monkeypatch):
    from neurospatial.environment import _fields

    env = _grid()
    build = _fields._gaussian_kernel
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(_fields, "_gaussian_kernel", counted)
    first = env.compute_gaussian_kernel(bandwidth=1.0)
    expected = first.copy()
    second = env.compute_gaussian_kernel(bandwidth=1.0)
    assert len(calls) == 1
    assert first is not second and np.shares_memory(first, second)
    with pytest.raises(ValueError):
        first[:] = 0
    first.shape = (env.n_bins, env.n_bins, 1)
    np.testing.assert_array_equal(env.compute_gaussian_kernel(bandwidth=1.0), expected)
    env.clear_cache()
    rebuilt = env.compute_gaussian_kernel(bandwidth=1.0)
    assert len(calls) == 2
    assert not np.shares_memory(rebuilt, second)
    np.testing.assert_array_equal(rebuilt, expected)


@pytest.mark.parametrize("bandwidth", [0.0, -1.0, np.nan, np.inf])
def test_gaussian_rejects_nonpositive_or_nonfinite_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="finite and positive"):
        _grid().compute_gaussian_kernel(bandwidth=bandwidth)


def test_mrf_exports_are_independent_without_repeating_the_eigensolve(monkeypatch):
    from neurospatial.ops import diffusion

    env = Environment.from_grid_mask(
        np.ones(90, dtype=bool), grid_edges=(np.arange(91.0),)
    )
    occupancy = np.ones(env.n_bins)
    build = diffusion._symmetric_eigenbasis
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(diffusion, "_symmetric_eigenbasis", counted)
    first = env.get_mrf_basis(occupancy, rank=8)
    assert isinstance(first, MRFBasis)
    expected = tuple(array.copy() for array in first[:3])
    n_calls = len(calls)
    assert n_calls > 0
    first.B[:] = 0
    first.d[:] = 999
    first.live_bins[:] = -1
    second = env.get_mrf_basis(occupancy, rank=8)
    assert len(calls) == n_calls
    for actual, wanted in zip(second[:3], expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    env.get_mrf_basis(occupancy, rank=4)
    assert len(calls) == n_calls


def test_accuracy_query_reports_resolution_without_building_a_basis(monkeypatch):
    from neurospatial.ops import diffusion

    env = _grid(2)
    build = diffusion._symmetric_eigenbasis
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(diffusion, "_symmetric_eigenbasis", counted)
    conservative = env.diffusion_accuracy(bandwidth=1.0)
    assert calls == []
    env.diffuse(np.ones(env.n_bins), bandwidth=1.0)
    assert calls
    n_calls = len(calls)
    assert env.diffusion_accuracy(bandwidth=1.0) < conservative
    assert env.diffusion_accuracy(bandwidth=2.0) == conservative
    assert len(calls) == n_calls


@pytest.mark.parametrize("method", ["derived", "pickle"])
def test_derived_and_unpickled_environments_restore_analysis_with_independently_owned_caches(
    method,
):
    env = _grid()
    geometry = env.get_diffusion_geometry()
    gaussian = env.compute_gaussian_kernel(bandwidth=1.0)
    if method == "derived":
        restored = env.subset(mask=np.ones(env.n_bins, dtype=bool))
    else:
        restored = pickle.loads(pickle.dumps(env))
    new_geometry = restored.get_diffusion_geometry()
    new_gaussian = restored.compute_gaussian_kernel(bandwidth=1.0)
    assert not np.shares_memory(new_geometry.W.data, geometry.W.data)
    assert not np.shares_memory(new_gaussian, gaussian)
    assert np.shares_memory(env.get_diffusion_geometry().W.data, geometry.W.data)
    assert np.shares_memory(env.compute_gaussian_kernel(bandwidth=1.0), gaussian)
    for actual, wanted in zip(
        _geometry_arrays(new_geometry), _geometry_arrays(geometry), strict=True
    ):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(new_gaussian, gaussian)


def test_compatibility_functions_need_only_the_public_analysis_surface():
    env = _grid()
    host = SimpleNamespace(
        get_diffusion_geometry=env.get_diffusion_geometry,
        diffusion_accuracy=env.diffusion_accuracy,
        compute_gaussian_kernel=env.compute_gaussian_kernel,
        bin_centers=env.bin_centers,
        bin_center_tree=env.bin_center_tree,
        binning=env.binning,
    )
    np.testing.assert_array_equal(
        diffusion_kernel(host, 1.0), env.compute_kernel(bandwidth=1.0)
    )
    np.testing.assert_array_equal(
        _get_gaussian_kernel(host, 1.0), env.compute_gaussian_kernel(bandwidth=1.0)
    )
    np.testing.assert_array_equal(
        _diffusion_generator(
            host.bin_centers, host.bin_center_tree(), host.get_diffusion_geometry()
        ).toarray(),
        _diffusion_generator(
            env.bin_centers, env.bin_center_tree(), env.get_diffusion_geometry()
        ).toarray(),
    )
