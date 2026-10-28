"""Environment region snapshots remain stable across analysis and storage."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.regions import Region, Regions


def _environment(regions=None):
    return Environment.from_samples(
        np.array([[0.0, 0.0], [2.0, 2.0]]),
        bin_size=1.0,
        infer_active_bins=False,
        regions=regions,
    )


def test_constructor_owns_builder_and_replacement_owns_edited_builder():
    builder = Regions()
    builder.add("first", point=(0.25, 0.25), metadata={"labels": ["old"]})
    env = _environment(builder)
    builder.add("later", point=(1.25, 1.25))
    assert list(env.regions) == ["first"]
    edited = Regions(env.regions.values())
    edited.add("second", point=(1.25, 1.25))
    changed = env.with_regions(edited)
    edited.clear()
    assert list(changed.regions) == ["first", "second"]
    assert list(env.regions) == ["first"]
    assert changed._geometry is env._geometry
    assert type(changed.regions) is Regions
    np.testing.assert_array_equal(changed.bins_in_region("first"), [0])
    assert not env.with_regions(Regions()).regions


@pytest.mark.parametrize(
    "restore",
    [
        copy.copy,
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_snapshots_stay_immutable_after_restoration(restore):
    regions = Regions()
    regions.add("first", point=(0.25, 0.25))
    env = restore(_environment(regions))
    with pytest.raises(TypeError, match=r"read-only|immutable"):
        env.regions.clear()
    with pytest.raises(AttributeError, match="read-only"):
        env.regions = Regions()
    with pytest.raises(AttributeError, match="read-only"):
        del env.regions
    np.testing.assert_array_equal(env.regions["first"].data, [0.25, 0.25])


def test_region_dimension_checked_at_publication_and_replacement():
    regions = Regions()
    regions.add("wrong", point=(1.0,))
    message = "Region 'wrong' has 1 coordinates; the environment has 2"
    with pytest.raises(ValueError, match=message):
        _environment(regions)
    env = _environment()
    with pytest.raises(ValueError, match=message):
        env.with_regions(regions)
    assert not env.regions


def test_regions_outside_arena_are_allowed_and_order_is_preserved():
    regions = Regions()
    regions.add("outside", point=(1000.0, 1000.0))
    regions.add("inside", point=(0.25, 0.25))
    env = _environment(regions)
    assert list(env.regions) == ["outside", "inside"]
    assert env.bins_in_region("outside").size == 0


def test_replacement_shares_the_sources_populated_cache():
    env = _environment()
    kernel = env.compute_kernel(bandwidth=0.5, mode="density")
    original_cache = env._cache
    n_cached = len(original_cache)
    replacement = env.with_regions(Regions())
    assert env._cache is original_cache
    assert replacement._cache is original_cache
    assert len(original_cache) == n_cached > 0
    np.testing.assert_array_equal(
        replacement.compute_kernel(bandwidth=0.5, mode="density"), kernel
    )


@pytest.mark.parametrize(
    "source",
    ["builder", "dict", "published"],
)
def test_with_regions_accepts_any_mapping_of_regions_keyed_by_name(source):
    goal = Region("goal", "point", [0.25, 0.25])
    zone = Region("zone", "polygon", box(1.0, 1.0, 2.0, 2.0))
    mapping = {
        "builder": Regions([goal, zone]),
        "dict": {"goal": goal, "zone": zone},
        "published": _environment(Regions([goal, zone])).regions,
    }[source]
    env = _environment().with_regions(mapping)
    assert list(env.regions) == ["goal", "zone"]
    assert type(env.regions) is Regions
    np.testing.assert_array_equal(env.bins_in_region("goal"), [0])
    np.testing.assert_array_equal(
        env.region_membership(mapping), env.region_membership()
    )


def test_with_regions_refuses_values_that_are_not_regions():
    from neurospatial.landmarks import Landmark

    cue = Landmark("cue", "line", [(0.0, 2.0), (2.0, 2.0)])
    with pytest.raises(TypeError, match=r"regions\['cue'\] is a Landmark"):
        _environment().with_regions({"cue": cue})
    with pytest.raises(TypeError, match="must map names to Region values"):
        _environment().with_regions([Region("goal", "point", [0.25, 0.25])])


def test_with_regions_refuses_a_key_that_is_not_the_region_name():
    with pytest.raises(ValueError, match=r"regions\['home'\] holds .* 'goal'"):
        _environment().with_regions({"home": Region("goal", "point", [0.25, 0.25])})
