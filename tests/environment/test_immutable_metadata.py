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
    env.compute_kernel(0.5)
    env.get_differential_operator()
    source_cache = env._kernel_cache.copy()

    def fail(*args, **kwargs):
        raise AssertionError("metadata replacement must not publish geometry again")

    monkeypatch.setattr(Environment, "_setup_from_layout", fail)
    updated = env.with_metadata(name="comparison", units="m", frame=None)
    assert type(updated) is type(env)
    assert updated is not env
    assert updated._geometry is env._geometry
    assert updated.name == "comparison"
    assert updated.units == "m"
    assert updated.frame is None
    assert (env.name, env.units, env.frame) == ("training", "cm", "world")
    assert not updated._kernel_cache
    assert env._kernel_cache.keys() == source_cache.keys()
    assert all(env._kernel_cache[key] is value for key, value in source_cache.items())
    assert not any(key.startswith("_versioned_cache__") for key in vars(updated))
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
        (1.0, 3.0),
        (-np.pi, np.pi),
        1.0,
        np.pi,
        circular_angle=True,
        units="cm",
        frame="animal",
    )
    updated = env.with_metadata(name="polar copy")
    assert type(updated) is Environment
    assert updated._geometry is env._geometry
    assert updated._angular_is_circular == env._angular_is_circular
    assert (updated.units, updated.frame) == ("cm", "animal")


def test_layout_parameters_are_independently_owned_exports():
    env = _environment()
    original = env.layout_parameters
    exported = env.layout_parameters
    exported["positions"][:] = -1000
    exported["custom"] = "changed"
    assert "custom" not in env.layout_parameters
    np.testing.assert_array_equal(
        env.layout_parameters["positions"], original["positions"]
    )


@pytest.mark.parametrize(
    "restore", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_unfitted_environment_views_and_copies_remain_available(restore):
    from neurospatial.layout import RegularGridLayout

    env = Environment("unfitted", RegularGridLayout())
    assert env.bin_centers.shape == (0, 0)
    clone = restore(env)
    assert clone.name == "unfitted"
    assert not clone._is_fitted
    assert clone.bin_centers.shape == (0, 0)


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
    state = vars(env).copy()
    state["_geometry"] = replace(env._geometry, arena=CustomArena(box(-1, -1, 4, 4)))
    restored = Environment.__new__(Environment)
    with pytest.raises(TypeError, match="snapshot"):
        restored.__setstate__(state)
