"""Shared gap-free calls for structured kinematic baseline comparisons."""

import numpy as np
import pandas as pd


def capture_kinematics_outputs(recording, *, legacy_heading=False):
    """Exercise all kinematic entry points using the same recording inputs."""
    from neurospatial.behavior import Trial
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
        instantaneous_goal_alignment,
    )
    from neurospatial.behavior.trajectory import (
        compute_home_range,
        compute_trajectory_curvature,
    )
    from neurospatial.behavior.vte import (
        compute_vte_session,
        compute_vte_trial,
        head_sweep_from_positions,
    )
    from neurospatial.events import add_positions
    from neurospatial.ops.egocentric import heading_from_velocity

    r = recording
    goal = r.positions[-1]
    bins = r.env.bin_at(r.positions)
    # Avoid changing the session-scoped fixture's regions.
    env = r.env.copy()
    env.regions.add("decision", point=tuple(env.bin_centers[bins[4000]]))
    trials = [
        Trial(0.0, 99.98, "source", "target", True),
        Trial(100.0, r.times[-1], "source", "target", True),
    ]
    session = compute_vte_session(
        env, r.positions, r.times, decision_region="decision", trials=trials
    )
    assert session.trial_results, "The session golden must exercise real metrics."
    heading_clock = float(np.median(np.diff(r.times))) if legacy_heading else r.times
    return {
        "heading_from_velocity": heading_from_velocity(r.positions, heading_clock),
        "heading_from_velocity_smoothed": heading_from_velocity(
            r.positions, heading_clock, bandwidth=2.0, min_speed=5.0
        ),
        "pre_decision_heading_stats": pre_decision_heading_stats(r.positions, r.times),
        "pre_decision_speed_stats": pre_decision_speed_stats(r.positions, r.times),
        "head_sweep_from_positions": head_sweep_from_positions(r.positions, r.times),
        "heading_direction_labels": heading_direction_labels(r.positions, r.times),
        "heading_direction_precomputed": heading_direction_labels(
            speed=np.ones(len(r.times)) * 10, heading=r.headings
        ),
        "compute_path_efficiency": compute_path_efficiency(
            r.env, r.positions, r.times, goal, metric="euclidean", reference_speed=10
        ),
        "compute_path_efficiency_geodesic": compute_path_efficiency(
            r.env, r.positions, r.times, goal, reference_speed=10
        ),
        "instantaneous_goal_alignment": instantaneous_goal_alignment(
            r.positions, r.times, goal
        ),
        "goal_bias": goal_bias(r.positions, r.times, goal),
        "approach_rate": approach_rate(r.positions, r.times, goal),
        "approach_rate_geodesic": approach_rate(
            r.positions, r.times, goal, metric="geodesic", env=r.env
        ),
        "compute_goal_directed_metrics": compute_goal_directed_metrics(
            r.env, r.positions, r.times, goal, goal_radius=5
        ),
        "compute_trajectory_curvature": compute_trajectory_curvature(
            r.positions, times=r.times
        ),
        "compute_trajectory_curvature_no_times": compute_trajectory_curvature(
            r.positions
        ),
        "compute_home_range": compute_home_range(bins, times=r.times),
        "compute_home_range_no_times": compute_home_range(bins),
        "add_positions": add_positions(
            pd.DataFrame({"timestamp": [0.0, 50.005, 120.01, r.times[-1]]}),
            times=r.times,
            positions=r.positions,
        ).to_dict("list"),
        "compute_pre_decision_metrics": compute_pre_decision_metrics(
            r.positions, r.times, entry_time=100.0, window_duration=2.0
        ),
        "compute_vte_trial": compute_vte_trial(
            r.positions, r.times, entry_time=100.0, window_duration=2.0
        ),
        "compute_vte_session": session,
    }
