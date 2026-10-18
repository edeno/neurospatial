"""Environment region snapshots remain stable across analysis and storage."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.regions import Regions


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
    with pytest.raises(ValueError, match="dimensions"):
        _environment(regions)
    env = _environment()
    with pytest.raises(ValueError, match="dimensions"):
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
