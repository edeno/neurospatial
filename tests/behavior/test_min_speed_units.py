"""Speed thresholds have a physical unit, so their defaults follow the arena's units.

``min_speed`` is in position units per second. A function without an
environment cannot know those units, so it requires ``min_speed``. A function
with an environment defaults to 5 cm/s expressed in ``env.units`` and refuses
to guess when the units are not a known length.
"""

import inspect

import numpy as np
import pytest
import shapely

from neurospatial import Environment, behavior

ENV_LESS = [
    "pre_decision_heading_stats",
    "compute_pre_decision_metrics",
    "heading_direction_labels",
    "instantaneous_goal_alignment",
    "goal_bias",
    "head_sweep_from_positions",
    "compute_vte_trial",
]


@pytest.mark.parametrize("name", ENV_LESS)
def test_functions_without_an_environment_require_min_speed(name):
    parameter = inspect.signature(getattr(behavior, name)).parameters["min_speed"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def _run_to_goal(scale, units):
    """A 10 s run toward a goal with a 2 s pause, in cm scaled to ``units``."""
    times = np.arange(0.0, 12.0, 1 / 30)
    x = np.where(times < 5, 8 * times, np.where(times < 7, 40.0, 40 + 8 * (times - 7)))
    positions = np.column_stack([x + 5.0, np.full_like(x, 50.0)]) * scale
    env = Environment.from_polygon(
        shapely.box(0, 0, 100 * scale, 100 * scale), bin_size=5.0 * scale, units=units
    )
    return env, times, positions, np.array([95.0, 50.0]) * scale


@pytest.mark.parametrize(("scale", "units"), [(0.01, "m"), (10.0, "mm")])
def test_goal_metrics_default_threshold_is_five_cm_per_second_in_any_units(
    scale, units
):
    env_cm, times, positions_cm, goal_cm = _run_to_goal(1.0, "cm")
    env, _, positions, goal = _run_to_goal(scale, units)
    reference = behavior.compute_goal_directed_metrics(
        env_cm, times, positions_cm, goal_cm
    )
    scaled = behavior.compute_goal_directed_metrics(env, times, positions, goal)
    np.testing.assert_allclose(scaled.goal_bias, reference.goal_bias)
    explicit = behavior.compute_goal_directed_metrics(
        env, times, positions, goal, min_speed=5.0 * scale
    )
    np.testing.assert_allclose(scaled.goal_bias, explicit.goal_bias)


def test_unknown_units_require_an_explicit_threshold():
    env, times, positions, goal = _run_to_goal(1.0, "cm")
    env = env.with_metadata(units=None)
    with pytest.raises(ValueError, match="min_speed"):
        behavior.compute_goal_directed_metrics(env, times, positions, goal)
    result = behavior.compute_goal_directed_metrics(
        env, times, positions, goal, min_speed=5.0
    )
    assert np.isfinite(result.goal_bias)
