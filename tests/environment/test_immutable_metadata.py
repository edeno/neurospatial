"""Metadata replacements preserve scientific geometry and isolate analyses."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from neurospatial import Environment


def _environment():
    return Environment.from_samples(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        bin_size=1.0,
        infer_active_bins=False,
        name="training",
        units="cm",
        frame="world",
    )


@pytest.mark.parametrize(
    "name,value", [("name", "edited"), ("units", "m"), ("frame", "camera")]
)
def test_metadata_rejects_assignment_and_deletion(name, value):
    env = _environment()
    with pytest.raises(AttributeError, match="read-only"):
        setattr(env, name, value)
    with pytest.raises(AttributeError, match="read-only"):
        delattr(env, name)


def test_metadata_replacement_reuses_geometry_without_republication(monkeypatch):
    env = _environment()
    kernel = env.compute_kernel(bandwidth=0.5, mode="density")
    operator = env._differential_operator_cached
    n_cached = len(env._cache)

    def fail(*args, **kwargs):
        raise AssertionError("metadata replacement must not publish geometry again")

    from neurospatial.environment import _publication

    monkeypatch.setattr(_publication, "_publish", fail)
    updated = env.with_metadata(name="comparison", units="m", frame=None)
    assert type(updated) is type(env)
    assert updated is not env
    assert updated._geometry is env._geometry
    assert updated.name == "comparison"
    assert updated.units == "m"
    assert updated.frame is None
    assert (env.name, env.units, env.frame) == ("training", "cm", "world")
    assert updated._cache is env._cache
    assert len(env._cache) == n_cached
    assert updated._differential_operator_cached is operator
    np.testing.assert_array_equal(
        updated.compute_kernel(bandwidth=0.5, mode="density"), kernel
    )
    np.testing.assert_array_equal(updated.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(updated.bin_sizes, env.bin_sizes)
    np.testing.assert_array_equal(
        updated.bin_at(env.bin_centers), np.arange(env.n_bins)
    )


def test_omitted_metadata_is_preserved_and_none_clears_nullable_labels():
    env = _environment()
    updated = env.with_metadata(units=None)
    assert (updated.name, updated.units, updated.frame) == ("training", None, "world")
    renamed = updated.with_metadata(name="")
    assert (renamed.name, renamed.units, renamed.frame) == ("", None, "world")
    assert env.with_metadata()._geometry is env._geometry


@pytest.mark.parametrize(
    "kwargs", [{"name": None}, {"name": 1}, {"units": 1}, {"frame": 1}]
)
def test_invalid_metadata_is_rejected_without_modifying_source(kwargs):
    env = _environment()
    with pytest.raises(TypeError):
        env.with_metadata(**kwargs)
    assert (env.name, env.units, env.frame) == ("training", "cm", "world")


def test_custom_unit_labels_keep_the_advisory_warning():
    env = _environment()
    with pytest.warns(UserWarning, match="standard"):
        updated = env.with_metadata(units="lab distance")
    assert updated.units == "lab distance"


@pytest.mark.parametrize(
    "restore",
    [
        copy.copy,
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_restoration_preserves_readonly_metadata_and_geometry(restore):
    env = _environment()
    restored = restore(env)
    assert (restored.name, restored.units, restored.frame) == (
        "training",
        "cm",
        "world",
    )
    with pytest.raises(AttributeError, match="read-only"):
        restored.units = "m"
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )


def test_polar_replacement_preserves_concrete_type_and_circularity():
    env = Environment.from_polar_egocentric(
        distance_range=(1.0, 3.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=np.pi,
        circular_angle=True,
        units="cm",
        frame="animal",
    )
    updated = env.with_metadata(name="polar copy")
    assert type(updated) is Environment
    assert updated._geometry is env._geometry
    assert updated.coordinate_system.circular == env.coordinate_system.circular
    assert (updated.units, updated.frame) == ("cm", "animal")


def test_layout_parameters_are_independently_owned_exports():
    env = _environment()
    original = env.layout_parameters
    exported = env.layout_parameters
    exported["bin_size"] = -1000
    exported["custom"] = "changed"
    assert env.layout_parameters == original


@pytest.mark.parametrize(
    "restore", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_copies_and_pickles_keep_name_and_bins(restore):
    env = _environment().with_metadata(name="session")
    expected = env.bin_centers.copy()
    clone = restore(env)
    assert clone.name == "session"
    np.testing.assert_array_equal(clone.bin_centers, expected)


@pytest.mark.parametrize("name", ["graph", "nodes", "edges", "degree", "adj", "name"])
def test_graph_public_bindings_cannot_be_replaced(name):
    env = _environment()
    with pytest.raises(AttributeError, match="read-only"):
        setattr(env.connectivity, name, {})
    with pytest.raises(AttributeError, match="read-only"):
        delattr(env.connectivity, name)
    clone = env.connectivity.copy()
    clone.graph = {"editable": True}
    clone.remove_node(0)
    assert env.n_bins == 9
    assert 0 in env.connectivity


def test_restoration_refuses_custom_arenas_requiring_snapshot_migration():
    from dataclasses import replace

    from shapely.geometry import box

    from neurospatial.arena import OpenArena

    class CustomArena(OpenArena):
        pass

    env = _environment()
    state = env.__getstate__()
    state["_geometry"] = replace(env._geometry, arena=CustomArena(box(-1, -1, 4, 4)))
    restored = Environment.__new__(Environment)
    with pytest.raises(TypeError, match="arena= takes an arena"):
        restored.__setstate__(state)
