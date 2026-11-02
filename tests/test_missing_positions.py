"""A NaN position is a missing sample, the same as a deleted row, everywhere.

Every public function that takes ``times`` and ``positions`` is run twice on
one simulated session: with some rows set to NaN, and with those rows
deleted. Results must agree. Per-sample outputs agree on the kept rows and
mark the missing rows as missing (NaN, bin or run -1, False).

A new public function with ``times`` and ``positions`` is picked up
automatically; if it needs an argument this module does not know, add a value
to ``VALUES``.
"""

import dataclasses
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from shapely import Point, box
from test_argument_conventions import public_functions

from neurospatial import Environment, Regions
from neurospatial.arena import OpenArena

FS = 30.0  # Hz
TIMES = np.arange(0.0, 60.0, 1 / FS)
N = TIMES.size
MISSING = np.zeros(N, dtype=bool)
MISSING[[100, 401, 777, 1203]] = True  # single lost frames
MISSING[900:909] = True  # 0.3 s of lost tracking, bridged by max_gap=1 s
KEPT = ~MISSING

# Fields that count or index rows of the input arrays: they shift when rows
# are deleted, by design (they slice the caller's arrays).
ROW_INDEX_FIELDS = {
    "n_source_samples",
    "run_lengths",
    "run_sample_bounds",
    "run_starts",
    "source_left",
    "source_right",
    "start_index",
    "step_lengths",  # one entry per pair of consecutive input rows
    "stop_index",
}
# Simulators take positions as the stimulus, not as observations.
NOT_OBSERVATIONS = {"generate_population_spikes"}


def _session():
    """Back-and-forth trips between two corner boxes through the centre."""
    s = 0.5 - 0.5 * np.cos(2 * np.pi * TIMES / 10.0)
    wiggle = 6.0 * np.sin(2 * np.pi * TIMES / 1.3)
    noise = np.cumsum(np.random.default_rng(0).normal(0, 0.15, (N, 2)), axis=0)
    noise -= np.linspace(0, 1, N)[:, None] * noise[-1]
    positions = np.clip(
        np.c_[10 + 80 * s - wiggle / np.sqrt(2), 10 + 80 * s + wiggle / np.sqrt(2)]
        + noise,
        1,
        99,
    )  # cm
    xs, ys = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
    env = Environment.from_samples(
        np.c_[xs.ravel(), ys.ravel()],
        bin_size=4.0,
        units="cm",
        arena=OpenArena(box(0, 0, 100, 100)),
    )
    regions = Regions()
    regions.add("start", polygon=box(0, 0, 18, 18))
    regions.add("goal", polygon=box(82, 82, 100, 100))
    regions.add("center", polygon=Point(50, 50).buffer(8))
    regions.add("left_arm", polygon=box(0, 82, 18, 100))
    return env.with_regions(regions), positions


ENV, POSITIONS = _session()
_VELOCITY = np.gradient(POSITIONS, TIMES, axis=0)
_RATE = 20 * np.exp(-np.sum((POSITIONS - 70) ** 2, axis=1) / (2 * 10.0**2))  # Hz
_SPIKE_RNG = np.random.default_rng(12345)  # apart from the stimulus seed
SPIKE_TIMES = np.sort(
    TIMES[_SPIKE_RNG.random(N) < _RATE / FS] + _SPIKE_RNG.uniform(0, 1 / FS)
)


def _trials():
    from neurospatial.behavior import detect_trials

    return detect_trials(ENV, TIMES, POSITIONS, regions=["start", "goal"])


# Per-sample inputs: deleted with the position rows, missing where they are.
PER_SAMPLE_VALUES = {
    "headings": np.arctan2(_VELOCITY[:, 1], _VELOCITY[:, 0]),  # rad
    "direction_labels": np.where(_VELOCITY.sum(axis=1) >= 0, "out", "in").astype(
        object
    ),
}
VALUES = {
    "spike_times": SPIKE_TIMES,
    "goal": np.array([90.0, 90.0]),
    "start": "start",
    "object_positions": np.array([[30.0, 70.0]]),
    "subgoals": np.array([[50.0, 50.0]]),
    "min_speed": 2.0,
    "entry_time": 25.0,
    "window_duration": 1.0,
    "decision_region": "center",
    "region": "center",
    "regions": ["start", "goal"],
    "goals": ["start", "goal"],
    "goal_regions": ["goal", "left_arm"],
    "trials": _trials(),
    "direction_frame": "allocentric",
    "reward_times": np.array([5.0, 25.0, 45.0]),
    "reference_speed": 20.0,
    "optimal_distance": 113.0,
    "spacing": 2.0,
    "criterion": "threshold",
    "event_times": np.array([12.0, 32.0, 52.0]),
    "window": (-1.0, 1.0),
    "bin_size": 0.1,
    "events": pd.DataFrame({"timestamp": [3.0, 13.4, 30.0, 30.02]}),
}
# Route functions take named regions as start and goal.
FUNCTION_VALUES = {
    ("path_progress", "goal"): "goal",
    ("distance_to_goal", "goal"): "goal",
}
POPULATION_SPIKES = [SPIKE_TIMES, np.sort(_SPIKE_RNG.uniform(0, 60, 300))]


def _call(function, *, deleted):
    """Call with every required argument; NaN rows, or those rows deleted."""
    times, positions = TIMES, POSITIONS.copy()
    if deleted:
        times, positions = TIMES[KEPT], POSITIONS[KEPT]
    else:
        positions[MISSING] = np.nan
    args, kwargs = [], {}
    parameters = inspect.signature(function).parameters
    for name, parameter in parameters.items():
        if name == "n_shuffles" and "criterion" not in parameters:
            kwargs[name] = 20  # fast; the seed makes both calls draw alike
            continue
        if name == "rng" and "criterion" not in parameters:
            kwargs[name] = 0
            continue
        optional = parameter.default is not parameter.empty or parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        )
        if optional and name not in {"times", "positions"}:
            continue
        if name == "times":
            value = times
        elif name == "positions":
            value = positions
        elif name == "env":
            value = ENV
        elif name == "spike_times" and function.__name__.endswith("_rates"):
            value = POPULATION_SPIKES
        elif (function.__name__, name) in FUNCTION_VALUES:
            value = FUNCTION_VALUES[function.__name__, name]
        elif name in PER_SAMPLE_VALUES:
            samples = PER_SAMPLE_VALUES[name]
            if deleted:
                value = samples[KEPT]
            else:
                value = samples.copy()
                value[MISSING] = None if samples.dtype == object else np.nan
        elif name in VALUES:
            value = VALUES[name]
        else:
            pytest.fail(
                f"{function.__name__} needs {name!r}: add a value to VALUES (or PER_SAMPLE_VALUES) in "
                "tests/test_missing_positions.py"
            )
        if parameter.kind == parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _leaves(result, path="result", depth=0):
    """Numeric and label leaves of a result, keyed by attribute path."""
    if depth > 4:
        return {}
    if isinstance(result, (bool, int, float, np.number, np.bool_)):
        return {path: np.asarray(result)}
    if isinstance(result, np.ndarray):
        return {path: result} if result.dtype.kind in "biufUO" else {}
    if isinstance(result, pd.Series):
        return {path: result.to_numpy()}
    if isinstance(result, pd.DataFrame):
        children = {str(c): result[c] for c in result.columns}
    elif hasattr(result, "intervals_table"):
        return _leaves(result.intervals_table().reset_index(), path, depth + 1)
    elif isinstance(result, (list, tuple)):
        children = {f"[{i}]": item for i, item in enumerate(result)}
    elif isinstance(result, dict):
        children = {f"[{k!r}]": v for k, v in result.items()}
    elif dataclasses.is_dataclass(result):
        children = {
            f.name: getattr(result, f.name, None) for f in dataclasses.fields(result)
        }
    elif hasattr(result, "__dict__"):
        children = vars(result)
    else:
        return {}
    leaves = {}
    for name, child in children.items():
        if name.startswith("_") or name in ROW_INDEX_FIELDS:
            continue
        if name in {"analysis_settings", "env"}:
            continue
        separator = "" if name.startswith("[") else "."
        leaves.update(_leaves(child, f"{path}{separator}{name}", depth + 1))
    return leaves


def _is_missing_marker(values):
    if values.dtype.kind == "f":
        return np.isnan(values) | (values == -1)
    if values.dtype.kind in "iu":
        return values == -1
    if values.dtype.kind == "b":
        return ~values
    return pd.isna(values)


def _assert_nan_rows_are_deleted_rows(function):
    bridged = _leaves(_call(function, deleted=False))
    deleted = _leaves(_call(function, deleted=True))
    assert bridged.keys() == deleted.keys()
    for path, values in bridged.items():
        expected = deleted[path]
        if values.ndim and values.shape[0] == N and expected.shape[0] == KEPT.sum():
            at_missing = values[MISSING]
            assert _is_missing_marker(at_missing).all(), (
                f"{path} at missing rows: {at_missing}"
            )
            values = values[KEPT]
        if values.dtype.kind in "fiub":
            np.testing.assert_allclose(
                values.astype(float),
                expected.astype(float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
                err_msg=path,
            )
        else:
            np.testing.assert_array_equal(values, expected, err_msg=path)


SAMPLE_FUNCTIONS = sorted(
    (
        function
        for function, parameters in public_functions()
        if {"times", "positions"} <= parameters.keys()
        and function.__name__ not in NOT_OBSERVATIONS
    ),
    key=lambda function: function.__name__,
)


def test_finds_the_sample_functions():
    names = {function.__name__ for function in SAMPLE_FUNCTIONS}
    assert len(names) >= 60
    assert {"compute_spatial_rate", "detect_trials", "compute_speed"} <= names


@pytest.mark.parametrize(
    "function", SAMPLE_FUNCTIONS, ids=lambda function: function.__name__
)
def test_nan_rows_are_deleted_rows(function):
    _assert_nan_rows_are_deleted_rows(function)


@pytest.mark.parametrize(
    "method",
    sorted(
        name
        for name, member in inspect.getmembers(Environment, inspect.isfunction)
        if not name.startswith("_")
        and {"times", "positions"} <= inspect.signature(member).parameters.keys()
    ),
)
def test_environment_nan_rows_are_deleted_rows(method):
    _assert_nan_rows_are_deleted_rows(getattr(ENV, method))
