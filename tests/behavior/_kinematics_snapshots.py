"""Shared gap-free calls for structured kinematic baseline comparisons."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from neurospatial import Regions


def load_kinematic_recording(recording):
    """Load the original recording's exact numeric inputs across platforms."""
    from neurospatial import Environment

    file = Path(__file__).parent / "data" / "kinematics_inputs.npz"
    with np.load(file, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    env = Environment.from_samples(arrays["positions"], bin_size=5.0)
    env = env.with_metadata(units="cm")
    return replace(recording, **arrays, env=env)


def capture_kinematics_outputs(recording, *, legacy_heading=False):
    """Exercise all kinematic entry points using the same recording inputs."""
    from neurospatial.behavior import Trial, heading_from_velocity
    from neurospatial.behavior.decisions import (
        compute_pre_decision_metrics,
        pre_decision_heading_stats,
        pre_decision_speed_stats,
    )
    from neurospatial.behavior.navigation import (
        approach_rate,
        compute_goal_directed_metrics,
        compute_path_efficiency,
        goal_bias,
        heading_direction_labels,
        headings_to_direction_labels,
        instantaneous_goal_alignment,
    )
    from neurospatial.behavior.trajectory import compute_home_range
    from neurospatial.behavior.vte import (
        compute_vte_session,
        compute_vte_trial,
        head_sweep_from_positions,
    )
    from neurospatial.events import add_positions

    r = recording
    goal = r.positions[-1]
    bins = r.env.bin_at(r.positions)
    # Unique visit counts avoid depending on an unstable sort's tied-bin order.
    home_bins = np.repeat(np.arange(20), np.arange(10, 30))
    home_times = r.times[: len(home_bins)]
    env = r.env
    region_builder = Regions(env.regions.values())
    region_builder.add("decision", point=tuple(env.bin_centers[bins[4000]]))
    env = env.with_regions(region_builder)
    trials = [
        Trial(0.0, 99.98, "source", "target", "reached"),
        Trial(100.0, r.times[-1], "source", "target", "reached"),
    ]
    session = compute_vte_session(
        env, r.times, r.positions, decision_region="decision", trials=trials
    )
    assert session.trial_results, "The session golden must exercise real metrics."
    heading_clock = float(np.median(np.diff(r.times))) if legacy_heading else r.times
    return {
        "heading_from_velocity": heading_from_velocity(heading_clock, r.positions),
        "heading_from_velocity_smoothed": heading_from_velocity(
            heading_clock, r.positions, smooth_window=0.04, min_speed=5.0
        ),
        "pre_decision_heading_stats": pre_decision_heading_stats(
            r.times, r.positions, min_speed=5.0
        ),
        "pre_decision_speed_stats": pre_decision_speed_stats(r.times, r.positions),
        "head_sweep_from_positions": head_sweep_from_positions(
            r.times, r.positions, min_speed=5.0
        ),
        "heading_direction_labels": heading_direction_labels(
            r.times, r.positions, min_speed=5.0
        ),
        "heading_direction_precomputed": headings_to_direction_labels(
            r.headings,
            np.ones(len(r.times)) * 10,
            min_speed=5.0,
        ),
        "compute_path_efficiency": compute_path_efficiency(
            r.env, r.times, r.positions, goal, metric="euclidean", reference_speed=10
        ),
        "compute_path_efficiency_geodesic": compute_path_efficiency(
            r.env, r.times, r.positions, goal, reference_speed=10
        ),
        "instantaneous_goal_alignment": instantaneous_goal_alignment(
            r.times,
            r.positions,
            goal,
            min_speed=5.0,
        ),
        "goal_bias": goal_bias(r.times, r.positions, goal, min_speed=5.0),
        "approach_rate": approach_rate(r.times, r.positions, goal),
        "approach_rate_geodesic": approach_rate(
            r.times, r.positions, goal, metric="geodesic", env=r.env
        ),
        "compute_goal_directed_metrics": compute_goal_directed_metrics(
            r.env, r.times, r.positions, goal, goal_radius=5
        ),
        "compute_home_range": compute_home_range(home_bins, times=home_times),
        "compute_home_range_no_times": compute_home_range(home_bins),
        "add_positions": add_positions(
            pd.DataFrame({"timestamp": [0.0, 50.005, 120.01, r.times[-1]]}),
            times=r.times,
            positions=r.positions,
        ).to_dict("list"),
        "compute_pre_decision_metrics": compute_pre_decision_metrics(
            r.times,
            r.positions,
            entry_time=100.0,
            window_duration=2.0,
            min_speed=5.0,
        ),
        "compute_vte_trial": compute_vte_trial(
            r.times,
            r.positions,
            entry_time=100.0,
            window_duration=2.0,
            min_speed=5.0,
        ),
        "compute_vte_session": session,
    }
