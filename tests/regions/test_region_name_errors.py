"""Every lookup of a region by name suggests the closest existing name."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment, RegionNotFoundError, Regions
from neurospatial.behavior import (
    compute_decision_analysis,
    decision_region_entry_time,
    region_reward_field,
)
from neurospatial.encoding import compute_region_coverage


@pytest.fixture(scope="module")
def env() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    arena = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)
    regions = Regions()
    regions.add("reward", point=(15.0, 15.0))
    regions.add("home", point=(3.0, 3.0))
    return arena.with_regions(regions)


def _lookups(env: Environment) -> dict[str, object]:
    """Each call with the misspelled name 'rewrd' and the fix it should give."""
    times = np.arange(10) / 30.0
    positions = np.full((10, 2), 15.0)
    bins = env.bin_at(positions)
    builder = Regions(env.regions.values())
    return {
        "env.regions[...]": (lambda: env.regions["rewrd"], "use 'reward'"),
        "bins_in_region": (
            lambda: env.bins_in_region("rewrd"),
            "pass region_name='reward'",
        ),
        "region_mask": (lambda: env.region_mask("rewrd"), "pass regions='reward'"),
        "distance_to": (
            lambda: env.distance_to(region="rewrd"),
            "pass region='reward'",
        ),
        "subset": (
            lambda: env.subset(region_names=["rewrd"]),
            "pass region_names=['reward']",
        ),
        "Regions.remove": (lambda: builder.remove("rewrd"), "use 'reward'"),
        "Regions.update_region": (
            lambda: builder.update_region("rewrd", point=(1.0, 1.0)),
            "use 'reward'",
        ),
        "Regions.region_center": (
            lambda: builder.region_center("rewrd"),
            "use 'reward'",
        ),
        "region_reward_field": (
            lambda: region_reward_field(env, "rewrd"),
            "pass region_name='reward'",
        ),
        "compute_region_coverage": (
            lambda: compute_region_coverage(env, bins, regions=["rewrd"]),
            "pass regions=['reward']",
        ),
        "decision_region_entry_time": (
            lambda: decision_region_entry_time(bins, times, env, region="rewrd"),
            "pass region='reward'",
        ),
        "compute_decision_analysis": (
            lambda: compute_decision_analysis(
                env, times, positions, decision_region="home", goal_regions=["rewrd"]
            ),
            "pass goal_regions=['reward']",
        ),
    }


_CALLS = [
    "env.regions[...]",
    "bins_in_region",
    "region_mask",
    "distance_to",
    "subset",
    "Regions.remove",
    "Regions.update_region",
    "Regions.region_center",
    "region_reward_field",
    "compute_region_coverage",
    "decision_region_entry_time",
    "compute_decision_analysis",
]


def test_every_lookup_is_checked(env):
    assert sorted(_lookups(env)) == sorted(_CALLS)


@pytest.mark.parametrize("call", _CALLS)
def test_misspelled_region_name_suggests_the_closest_name(env, call):
    lookup, fix = _lookups(env)[call]
    with pytest.raises(RegionNotFoundError) as caught:
        lookup()
    message = str(caught.value)
    assert "Region 'rewrd' not found" in message
    assert message.endswith(f"Fix: {fix}")
    # Callers that catch either built-in type keep working.
    assert isinstance(caught.value, KeyError)
    assert isinstance(caught.value, ValueError)


def test_membership_and_get_do_not_raise(env):
    assert "reward" in env.regions
    assert "rewrd" not in env.regions
    assert env.regions.get("rewrd") is None
    assert env.regions.get("reward").name == "reward"
