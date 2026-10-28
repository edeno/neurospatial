"""Every lookup of a region by name suggests the closest existing name."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neurospatial import Environment, RegionNotFoundError, Regions
from neurospatial.behavior import (
    compute_decision_analysis,
    decision_region_entry_time,
    region_reward_field,
)
from neurospatial.encoding import compute_region_coverage
from neurospatial.regions import regions_containing_points


@pytest.fixture(scope="module")
def env() -> Environment:
    x, y = np.meshgrid(np.arange(0.0, 30.0), np.arange(0.0, 30.0))
    arena = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=3.0)
    regions = Regions()
    regions.add("reward", point=(15.0, 15.0))
    regions.add("home", point=(3.0, 3.0))
    return arena.with_regions(regions)


def _lookups(env: Environment, folder: Path) -> dict[str, tuple[Any, Any, str]]:
    """Each lookup as a call taking the name, its misspelling and the fix.

    ``folder`` receives the animation the ``show_regions`` lookup writes.
    """
    times = np.arange(10) / 30.0
    positions = np.full((10, 2), 15.0)
    bins = env.bin_at(positions)
    builder = Regions(env.regions.values())
    return {
        "env.regions[...]": (
            lambda value: env.regions[value],
            "rewrd",
            "use 'reward'",
        ),
        "bins_in_region": (
            lambda value: env.bins_in_region(value),
            "rewrd",
            "pass region_name='reward'",
        ),
        "region_mask": (
            lambda value: env.region_mask(value),
            "rewrd",
            "pass regions='reward'",
        ),
        "distance_to": (
            lambda value: env.distance_to(region=value),
            "rewrd",
            "pass region='reward'",
        ),
        "subset": (
            lambda value: env.subset(region_names=value),
            ["rewrd"],
            "pass region_names=['reward']",
        ),
        "Regions.remove": (
            lambda value: builder.remove(value),
            "rewrd",
            "use 'reward'",
        ),
        "Regions.update_region": (
            lambda value: builder.update_region(value, point=(1.0, 1.0)),
            "rewrd",
            "use 'reward'",
        ),
        "region_reward_field": (
            lambda value: region_reward_field(env, region_name=value),
            "rewrd",
            "pass region_name='reward'",
        ),
        "compute_region_coverage": (
            lambda value: compute_region_coverage(env, bins, regions=value),
            ["rewrd"],
            "pass regions=['reward']",
        ),
        "decision_region_entry_time": (
            lambda value: decision_region_entry_time(bins, times, env, region=value),
            "rewrd",
            "pass region='reward'",
        ),
        "regions_containing_points": (
            lambda value: regions_containing_points(
                positions, env.regions, region_names=value, point_radius=2.0
            ),
            ["rewrd"],
            "pass region_names=['reward']",
        ),
        "animate_fields(show_regions=)": (
            lambda value: env.animate_fields(
                np.zeros((2, env.n_bins)),
                frame_times=np.array([0.0, 0.1]),
                backend="html",
                save_path=str(folder / "animation.html"),
                show_regions=value,
            ),
            ["rewrd"],
            "pass show_regions=['reward']",
        ),
        "compute_decision_analysis": (
            lambda value: compute_decision_analysis(
                env,
                times,
                positions,
                decision_region="reward",
                goal_regions=value,
                min_speed=5.0,
            ),
            ["rewrd"],
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
    "region_reward_field",
    "compute_region_coverage",
    "decision_region_entry_time",
    "compute_decision_analysis",
    "regions_containing_points",
    "animate_fields(show_regions=)",
]


@pytest.mark.parametrize("call", _CALLS)
def test_misspelled_region_name_suggests_the_closest_name(env, call, tmp_path):
    lookup, misspelled, fix = _lookups(env, tmp_path)[call]
    with pytest.raises(RegionNotFoundError) as caught:
        lookup(misspelled)
    message = str(caught.value)
    assert "Region 'rewrd' not found" in message
    assert message.endswith(f"Fix: {fix}")
    # Callers that catch either built-in type keep working.
    assert isinstance(caught.value, KeyError)
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("call", _CALLS)
def test_suggested_name_resolves(env, call, tmp_path):
    """Passing the name the fix suggests finds the region."""
    lookup, misspelled, _ = _lookups(env, tmp_path)[call]
    with pytest.raises(RegionNotFoundError) as caught:
        lookup(misspelled)
    fix = str(caught.value).splitlines()[-1]
    suggestion = fix.removeprefix("Fix: ").split("=", 1)[-1].removeprefix("use ")
    lookup(ast.literal_eval(suggestion))


def test_animation_refuses_a_misspelled_shown_region_before_any_view(env):
    """Every view and export starts from the scene, which refuses the name."""
    from neurospatial.animation import Animation

    with pytest.raises(RegionNotFoundError, match=r"show_regions=\['reward'\]"):
        Animation(
            env,
            np.zeros((2, env.n_bins)),
            frame_times=[0.0, 0.1],
            show_regions=["rewrd"],
        )


def test_membership_and_get_do_not_raise(env):
    assert "reward" in env.regions
    assert "rewrd" not in env.regions
    assert env.regions.get("rewrd") is None
    assert env.regions.get("reward").name == "reward"
