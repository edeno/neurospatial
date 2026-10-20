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
    detect_runs_between_regions,
    distance_to_boundary,
    region_reward_field,
    running_direction_labels,
    segment_trials,
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
    return env, times, positions, env.bin_at(positions)


CALLS = {
    "segment_trials start": lambda env, t, p, b, empty: segment_trials(
        b, t, env, start_region=empty, end_regions=["right"]
    ),
    "segment_trials end": lambda env, t, p, b, empty: segment_trials(
        b, t, env, start_region="left", end_regions=["right", empty]
    ),
    "detect_region_crossings": lambda env, t, p, b, empty: detect_region_crossings(
        b, t, env, region_name=empty
    ),
    "detect_region_visits": lambda env, t, p, b, empty: detect_region_visits(
        b, t, env, region_names=["left", empty]
    ),
    "detect_runs_between_regions": lambda env, t, p, b, empty: (
        detect_runs_between_regions(b, t, env, source="left", target=empty)
    ),
    "running_direction_labels": lambda env, t, p, b, empty: running_direction_labels(
        b, t, env, start_region=empty, end_regions=["right"]
    ),
    "detect_laps": lambda env, t, p, b, empty: detect_laps(
        b, t, env, method="region", start_region=empty
    ),
    "detect_goal_directed_runs": lambda env, t, p, b, empty: (
        detect_goal_directed_runs(b, t, env, goal_region=empty)
    ),
    "decision_region_entry_time": lambda env, t, p, b, empty: (
        decision_region_entry_time(b, t, env, region=empty)
    ),
    "compute_decision_analysis": lambda env, t, p, b, empty: (
        compute_decision_analysis(
            env, t, p, decision_region=empty, goal_regions=["left", "right"]
        )
    ),
    "compute_vte_session": lambda env, t, p, b, empty: compute_vte_session(
        env,
        t,
        p,
        decision_region=empty,
        trials=segment_trials(b, t, env, start_region="left", end_regions=["right"]),
    ),
    "region_reward_field": lambda env, t, p, b, empty: region_reward_field(
        env, region_name=empty
    ),
    "distance_to_boundary": lambda env, t, p, b, empty: distance_to_boundary(
        env, p, boundary_type="region", region_name=empty
    ),
}


@pytest.mark.parametrize("empty", ["outside_well", "sliver"])
@pytest.mark.parametrize("call", list(CALLS), ids=list(CALLS))
def test_region_covering_no_bins_raises(arena, call, empty):
    env, times, positions, bins = arena
    with pytest.raises(ValueError, match=rf"Region '{empty}' covers no bins"):
        CALLS[call](env, times, positions, bins, empty)


def test_point_region_error_names_the_point_and_a_polygon_fix(arena):
    env, times, _, bins = arena
    with pytest.raises(ValueError) as caught:
        segment_trials(
            bins, times, env, start_region="outside_well", end_regions=["right"]
        )
    message = str(caught.value)
    assert "(45, 10)" in message
    assert "buffer" in message


def test_region_with_bins_still_works(arena):
    env, times, _, bins = arena
    trials = segment_trials(
        bins, times, env, start_region="left", end_regions=["right"], min_duration=0.5
    )
    assert [t.success for t in trials] == [True]
