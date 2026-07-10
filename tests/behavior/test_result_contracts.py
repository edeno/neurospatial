"""Contract tests for the behavior result dataclasses' summary()/str() surface.

When ``summary()`` switched from a formatted string to a flat scalar dict (with
the human-readable form moved to ``__str__``), the existing tests checked the
dict side only with ``isinstance`` and left the VTE/Decision classes' ``str``
side untested entirely. These pin BOTH sides for every retrofitted behavior
result: the dict carries the documented keys with scalar values matching the
attributes, and ``str(result)`` still renders human-readable text -- including
the ``DecisionAnalysisResult`` -> ``DecisionBoundaryMetrics`` composition, which
now interpolates the child via an f-string.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.behavior.decisions import (
    DecisionAnalysisResult,
    DecisionBoundaryMetrics,
    PreDecisionMetrics,
)
from neurospatial.behavior.navigation import (
    GoalDirectedMetrics,
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
)
from neurospatial.behavior.vte import VTESessionResult, VTETrialResult


def _vte_trial() -> VTETrialResult:
    return VTETrialResult(
        head_sweep_magnitude=1.5,
        z_head_sweep=2.0,
        mean_speed=8.0,
        min_speed=1.0,
        z_speed_inverse=1.2,
        vte_index=1.6,
        is_vte=True,
        window_start=10.0,
        window_end=11.0,
    )


def _path_efficiency() -> PathEfficiencyResult:
    return PathEfficiencyResult(
        traveled_length=12.0,
        shortest_length=10.0,
        efficiency=10.0 / 12.0,
        time_efficiency=None,
        angular_efficiency=0.9,
        start_position=np.array([0.0, 0.0]),
        goal_position=np.array([10.0, 0.0]),
        metric="geodesic",
    )


def _boundary() -> DecisionBoundaryMetrics:
    return DecisionBoundaryMetrics(
        goal_labels=np.array([0, 0, 1]),
        distance_to_boundary=np.array([2.0, 1.0, 3.0]),
        crossing_times=[5.0],
        crossing_directions=[(0, 1)],
    )


def _pre_decision() -> PreDecisionMetrics:
    return PreDecisionMetrics(
        mean_speed=8.0,
        min_speed=1.0,
        heading_mean_direction=0.1,
        heading_circular_variance=0.4,
        heading_mean_resultant_length=0.6,
        window_duration=1.0,
        n_samples=30,
    )


# (result, required summary keys, (key, expected value), str substring)
_CASES = [
    (
        _vte_trial(),
        {"head_sweep_magnitude", "mean_speed", "is_vte", "vte_index"},
        ("head_sweep_magnitude", 1.5),
        "VTE",
    ),
    (
        VTESessionResult(
            trial_results=[_vte_trial()],
            mean_head_sweep=1.0,
            std_head_sweep=0.5,
            mean_speed=8.0,
            std_speed=2.0,
            n_vte_trials=1,
            vte_fraction=1.0,
        ),
        {"n_trials", "n_vte_trials", "vte_fraction", "mean_head_sweep"},
        ("n_trials", 1),
        "VTE session",
    ),
    (
        _boundary(),
        {"n_crossings", "mean_distance_to_boundary"},
        ("n_crossings", 1),
        "Decision boundary",
    ),
    (
        DecisionAnalysisResult(
            entry_time=12.0,
            pre_decision=_pre_decision(),
            boundary=_boundary(),
            chosen_goal=1,
        ),
        {
            "entry_time",
            "pre_decision_mean_speed",
            "n_boundary_crossings",
            "chosen_goal",
        },
        ("chosen_goal", 1),
        "Decision analysis",
    ),
    (
        _path_efficiency(),
        {"traveled_length", "shortest_length", "efficiency"},
        ("traveled_length", 12.0),
        "Path:",
    ),
    (
        SubgoalEfficiencyResult(
            segment_results=[_path_efficiency()],
            mean_efficiency=0.83,
            weighted_efficiency=0.85,
            subgoal_positions=np.array([[5.0, 0.0]]),
        ),
        {"n_segments", "mean_efficiency", "weighted_efficiency"},
        ("n_segments", 1),
        "Subgoal path efficiency",
    ),
    (
        GoalDirectedMetrics(
            goal_bias=0.65,
            mean_approach_rate=-8.5,
            time_to_goal=3.2,
            min_distance_to_goal=1.0,
            goal_distance_at_start=20.0,
            goal_distance_at_end=2.0,
            goal_position=np.array([10.0, 0.0]),
            metric="geodesic",
        ),
        {"goal_bias", "mean_approach_rate", "min_distance_to_goal", "time_to_goal"},
        ("goal_bias", 0.65),
        "Goal-directed metrics",
    ),
]


@pytest.mark.parametrize("result, keys, kv, str_sub", _CASES)
def test_summary_is_scalar_dict_with_matching_values(result, keys, kv, str_sub):
    summary = result.summary()
    assert isinstance(summary, dict)
    # The documented keys are present...
    assert keys <= set(summary)
    # ...and every value is a scalar (tabulatable), not an array/nested object.
    for value in summary.values():
        assert value is None or isinstance(value, (int, float, str)), (
            f"non-scalar summary value: {value!r}"
        )
    # A representative value matches the attribute it summarizes.
    key, expected = kv
    assert summary[key] == expected


@pytest.mark.parametrize("result, keys, kv, str_sub", _CASES)
def test_str_renders_human_readable(result, keys, kv, str_sub):
    text = str(result)
    assert isinstance(text, str) and text.strip()
    assert str_sub in text


def test_decision_analysis_str_composes_boundary():
    """DecisionAnalysisResult.__str__ embeds the child boundary's __str__.

    Regression: the composition switched from ``self.boundary.summary()`` to the
    f-string ``f"  {self.boundary}"``, so a break in
    ``DecisionBoundaryMetrics.__str__`` would regress the parent silently. Pin
    that the child's rendered text appears, and that a None boundary is simply
    omitted (no crash).
    """
    boundary = _boundary()
    result = DecisionAnalysisResult(
        entry_time=12.0,
        pre_decision=_pre_decision(),
        boundary=boundary,
        chosen_goal=1,
    )
    assert str(boundary) in str(result)

    no_boundary = DecisionAnalysisResult(
        entry_time=12.0,
        pre_decision=_pre_decision(),
        boundary=None,
        chosen_goal=None,
    )
    assert "Decision analysis" in str(no_boundary)
