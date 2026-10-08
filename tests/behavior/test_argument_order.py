"""Argument swaps fail clearly before trajectory calculations."""

import numpy as np
import pytest

from neurospatial import Environment, behavior, ops


@pytest.fixture(scope="module")
def recording():
    times = np.arange(600) / 30
    positions = (
        50
        + 45
        * np.c_[
            np.sin(np.linspace(0, 6 * np.pi, 600)),
            np.cos(np.linspace(0, 6 * np.pi, 600)),
        ]
    )
    env = Environment.from_samples(positions, bin_size=2.0)
    return env, times, positions


@pytest.mark.parametrize(
    "name",
    [
        "approach_rate",
        "compute_decision_analysis",
        "compute_goal_directed_metrics",
        "compute_path_efficiency",
        "compute_pre_decision_metrics",
        "compute_vte_session",
        "compute_vte_trial",
        "extract_pre_decision_window",
        "goal_bias",
        "head_sweep_from_positions",
        "instantaneous_goal_alignment",
        "mean_square_displacement",
        "pre_decision_heading_stats",
        "pre_decision_speed_stats",
        "segment_by_velocity",
        "time_efficiency",
        "heading_from_velocity",
        "compute_trajectory_curvature",
    ],
)
def test_old_order_raises(name, recording):
    env, times, positions = recording
    function = getattr(ops if name == "heading_from_velocity" else behavior, name)
    args = (positions, times)
    kwargs = {}
    if name in {"compute_decision_analysis", "compute_vte_session"}:
        args = (env, positions, times)
        kwargs["decision_region"] = "decision"
        if name == "compute_decision_analysis":
            kwargs["goal_regions"] = ["goal"]
        else:
            kwargs["trials"] = []
    if name in {"compute_path_efficiency", "compute_goal_directed_metrics"}:
        args = (env, positions, times, np.array([95.0, 50.0]))
    if name in {"approach_rate", "goal_bias", "instantaneous_goal_alignment"}:
        args += (np.array([95.0, 50.0]),)
    if name in {
        "compute_pre_decision_metrics",
        "compute_vte_trial",
        "extract_pre_decision_window",
    }:
        kwargs.update(entry_time=5.0, window_duration=2.0)
    if name == "segment_by_velocity":
        kwargs["min_speed"] = 5.0
    if name == "time_efficiency":
        kwargs.update(reference_speed=20.0, optimal_distance=50.0)
    if name == "compute_trajectory_curvature":
        args = (times, positions)
    with pytest.raises(ValueError, match="did you pass"):
        function(*args, **kwargs)
