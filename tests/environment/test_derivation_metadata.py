"""Derived environments keep the parent's labels, regions and arena."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.geometry._codecs import encode_arena
from neurospatial.ops.transforms import translate
from neurospatial.regions import Regions

SHIFT = (20.0, 0.0)


def _parent(units="cm"):
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        name="arena1",
        units=units,
        frame="room",
    )
    regions = Regions()
    regions.add("goal", point=[2.5, 2.5], metadata={"reward": "sucrose"})
    # Regions are annotations; they may extend outside the retained bins.
    regions.add("outside", polygon=box(8, 8, 14, 14))
    return env.with_regions(regions)


DERIVATIONS = {
    "subset": lambda env, **kw: env.subset(mask=env.bin_centers[:, 0] < 5, **kw),
    "subset_polygon": lambda env, **kw: env.subset(polygon=box(0, 0, 6, 6), **kw),
    "clip": lambda env, **kw: env.clip(box(1, 1, 6, 6), **kw),
    "rebin_factor": lambda env, **kw: env.rebin(2, **kw),
    "rebin_bin_size": lambda env, **kw: env.rebin(bin_size=2.0, **kw),
    "apply_transform": lambda env, **kw: env.apply_transform(translate(*SHIFT), **kw),
}


def _regions(env):
    return {name: region.to_dict() for name, region in env.regions.items()}


@pytest.mark.parametrize(
    "derivation", ["subset", "subset_polygon", "clip", "rebin_factor", "rebin_bin_size"]
)
def test_selection_and_rebinning_keep_every_label(derivation):
    parent = _parent()
    child = DERIVATIONS[derivation](parent)
    assert type(child) is type(parent)
    assert (child.name, child.units, child.frame) == ("arena1", "cm", "room")
    assert _regions(child) == _regions(parent)
    assert encode_arena(child.arena)[0] == encode_arena(parent.arena)[0]


def test_apply_transform_keeps_name_and_units_and_moves_regions():
    parent = _parent()
    child = parent.apply_transform(translate(*SHIFT))
    assert (child.name, child.units) == ("arena1", "cm")
    # Transformed coordinates are no longer in the parent's frame.
    assert child.frame is None
    np.testing.assert_allclose(
        child.regions["goal"].data, parent.regions["goal"].data + SHIFT
    )
    assert child.regions["goal"].metadata["reward"] == "sucrose"
    assert child.regions["outside"].data.equals(box(28, 8, 34, 14))
    np.testing.assert_array_equal(
        child.arena.contains(np.array([[25.0, 5.0], [5.0, 5.0]])), [True, False]
    )


def test_apply_transform_frame_is_the_callers_label():
    parent = _parent()
    child = parent.apply_transform(translate(*SHIFT), frame="session2")
    assert child.frame == "session2"


@pytest.mark.parametrize("derivation", sorted(DERIVATIONS))
def test_every_derivation_accepts_a_new_name(derivation):
    parent = _parent()
    child = DERIVATIONS[derivation](parent, name="renamed")
    assert child.name == "renamed"
    assert child.units == "cm"
    assert parent.name == "arena1"


def test_combine_qualifies_regions_and_starts_unnamed():
    parent = _parent()
    other = parent.apply_transform(translate(*SHIFT), frame="room")
    combined = Environment.combine({"a": parent, "b": other})
    assert (combined.name, combined.units, combined.frame) == ("", "cm", "room")
    assert list(combined.regions) == ["a/goal", "a/outside", "b/goal", "b/outside"]
    np.testing.assert_array_equal(
        combined.regions["b/goal"].data, other.regions["goal"].data
    )
    named = Environment.combine({"a": parent, "b": other}, name="both")
    assert named.name == "both"


@pytest.mark.parametrize("derivation", sorted(DERIVATIONS))
def test_derivations_do_not_repeat_the_units_warning(derivation):
    with pytest.warns(UserWarning, match="registry"):
        parent = _parent(units="centimeters")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*registry.*")
        child = DERIVATIONS[derivation](parent)
    assert child.units == "centimeters"
