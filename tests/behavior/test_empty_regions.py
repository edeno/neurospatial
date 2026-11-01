"""Region-based analyses refuse a named region that covers no bins.

A point region outside the arena, or a polygon that contains no bin center,
selects no bins. An analysis that silently returned no trials,
visits or runs for it would read as a behavioral result.
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior import (
    compute_decision_analysis,
    compute_vte_session,
    decision_region_entry_time,
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    detect_region_visits,
    detect_trials,
    distance_to_boundary,
    region_reward_field,
)


@pytest.fixture(scope="module")
def arena():
    """A 40 x 20 cm arena (4 cm bins), a 1.5 s run from left to right at 30 Hz."""
    env = Environment.from_polygon(box(0, 0, 40, 20), bin_size=4.0, units="cm")
    regions = Regions(env.regions.values())
    regions.add("left", polygon=box(0, 0, 8, 20))
    regions.add("right", polygon=box(32, 0, 40, 20))
    regions.add("outside_well", point=[45.0, 10.0])  # beyond the wall: no bin
    regions.add("sliver", polygon=box(0.5, 0.5, 1.5, 19.5))  # no bin center
    env = env.with_regions(regions)
    times = np.arange(45) / 30.0
    positions = np.column_stack([np.linspace(2, 38, 45), np.full(45, 10.0)])
    return env, times, positions


CALLS = {
    "detect_trials": lambda env, t, p, empty: detect_trials(
        env, t, p, regions=["right", empty]
    ),
    "detect_region_crossings": lambda env, t, p, empty: detect_region_crossings(
        env, t, p, region_name=empty
    ),
    "detect_region_visits": lambda env, t, p, empty: detect_region_visits(
        env, t, p, regions=["left", empty]
    ),
    "detect_laps": lambda env, t, p, empty: detect_laps(
        env, t, p, method="region", start_region=empty
    ),
    "detect_goal_directed_runs": lambda env, t, p, empty: (
        detect_goal_directed_runs(env, t, p, goal_region=empty)
    ),
    "decision_region_entry_time": lambda env, t, p, empty: (
        decision_region_entry_time(env, t, p, region=empty)
    ),
    "compute_decision_analysis": lambda env, t, p, empty: (
        compute_decision_analysis(
            env, t, p, decision_region=empty, goal_regions=["left", "right"]
        )
    ),
    "compute_vte_session": lambda env, t, p, empty: compute_vte_session(
        env,
        t,
        p,
        decision_region=empty,
        trials=detect_trials(env, t, p, regions=["left", "right"]),
    ),
    "region_reward_field": lambda env, t, p, empty: region_reward_field(
        env, region_name=empty
    ),
    "distance_to_boundary": lambda env, t, p, empty: distance_to_boundary(
        env, p, boundary_type="region", region_name=empty
    ),
}


@pytest.mark.parametrize("empty", ["outside_well", "sliver"])
@pytest.mark.parametrize("call", list(CALLS), ids=list(CALLS))
def test_region_covering_no_bins_raises(arena, call, empty):
    env, times, positions = arena
    with pytest.raises(ValueError, match=rf"Region '{empty}' covers no bins"):
        CALLS[call](env, times, positions, empty)


def test_point_region_error_names_the_point_and_a_polygon_fix(arena):
    env, times, positions = arena
    with pytest.raises(ValueError, match=r"covers no bins, so detect_trials") as caught:
        detect_trials(env, times, positions, regions=["outside_well", "right"])
    message = str(caught.value)
    assert "(45, 10)" in message
    assert "buffer" in message


def test_region_with_bins_still_works(arena):
    env, times, positions = arena
    trials = detect_trials(env, times, positions, regions=["left", "right"])
    assert trials.outcome.tolist() == ["reached"]
