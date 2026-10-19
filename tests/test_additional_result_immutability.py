"""Analysis result buffers are owned, read-only, and survive serialization."""

from __future__ import annotations

import copy
import pickle
from dataclasses import fields

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior.decisions import DecisionBoundaryMetrics
from neurospatial.behavior.navigation import (
    GoalDirectedMetrics,
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
)
from neurospatial.behavior.segmentation import Run
from neurospatial.behavior.trajectory import CumulativeDistanceResult, MSDResult
from neurospatial.decoding import DecodingSummary
from neurospatial.decoding.assemblies import AssemblyDetectionResult, AssemblyPattern
from neurospatial.decoding.trajectory import IsotonicFitResult, RadonDetectionResult
from neurospatial.encoding._metrics import BatchScoresResult
from neurospatial.encoding.spatial import DirectionalPlaceFields


@pytest.fixture(scope="module")
def env():
    return Environment.from_samples(
        np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]),
        bin_size=5.0,
    )


CASES = [
    (
        CumulativeDistanceResult,
        {
            "step_lengths": np.ones(1),
            "run_distance": np.array([0.0, 1.0]),
            "cumulative_distance": np.array([0.0, 1.0]),
            "run_ids": np.zeros(2, dtype=np.intp),
            "sample_valid": np.ones(2, dtype=bool),
            "run_sample_bounds": np.array([[0, 1]], dtype=np.intp),
            "run_time_bounds": np.array([[0.0, 0.1]]),
            "run_path_lengths": np.ones(1),
            "max_gap": 0.5,
            "epochs": np.array([[0.0, 1.0]]),
        },
    ),
    (
        DecodingSummary,
        {
            "times": np.array([0.0, 1.0]),
            "map_position": np.ones((2, 2)),
            "mean_position": np.ones((2, 2)),
            "posterior_entropy": np.ones(2),
            "peak_prob": np.ones(2),
            "map_bin": np.array([0, 1]),
        },
    ),
    (BatchScoresResult, {"scores": np.ones(2), "failures": np.array([False, True])}),
    (
        DecisionBoundaryMetrics,
        {
            "goal_labels": np.array([0, 1]),
            "distance_to_boundary": np.ones(2),
            "crossing_times": [],
            "crossing_directions": [],
        },
    ),
    (
        PathEfficiencyResult,
        {
            "traveled_length": 1.0,
            "shortest_length": 1.0,
            "efficiency": 1.0,
            "time_efficiency": None,
            "angular_efficiency": 1.0,
            "start_position": np.ones(2),
            "goal_position": np.ones(2),
            "metric": "euclidean",
        },
    ),
    (
        SubgoalEfficiencyResult,
        {
            "segment_results": [],
            "mean_efficiency": 1.0,
            "weighted_efficiency": 1.0,
            "subgoal_positions": np.ones((2, 2)),
        },
    ),
    (
        GoalDirectedMetrics,
        {
            "goal_bias": 1.0,
            "mean_approach_rate": -1.0,
            "time_to_goal": None,
            "min_distance_to_goal": 1.0,
            "goal_distance_at_start": 2.0,
            "goal_distance_at_end": 1.0,
            "goal_position": np.ones(2),
            "metric": "euclidean",
        },
    ),
    (MSDResult, {"lags": np.ones(2), "msd": np.ones(2)}),
    (
        Run,
        {"start_time": 0.0, "end_time": 1.0, "bins": np.array([0, 1]), "success": True},
    ),
    (
        AssemblyPattern,
        {
            "weights": np.ones(2),
            "member_indices": np.array([0, 1]),
            "explained_variance_ratio": 1.0,
        },
    ),
    (
        AssemblyDetectionResult,
        {
            "patterns": [],
            "activations": np.ones((2, 2)),
            "method": "pca",
            "n_significant": 2,
            "eigenvalues": np.ones(2),
            "threshold": 0.5,
        },
    ),
    (
        IsotonicFitResult,
        {
            "fitted_positions": np.ones(2),
            "r_squared": 1.0,
            "direction": "increasing",
            "residuals": np.zeros(2),
        },
    ),
    (
        RadonDetectionResult,
        {
            "angle_degrees": 0.0,
            "score": 1.0,
            "offset": 0.0,
            "sinogram": np.ones((2, 2)),
        },
    ),
]


@pytest.fixture(params=CASES, ids=lambda case: case[0].__name__)
def result_and_inputs(request, env):
    cls, template = request.param
    inputs = copy.deepcopy(template)
    if cls is DecodingSummary:
        inputs["env"] = env
        inputs["spike_window"] = np.array([[0.0, 2.0]])
    return cls(**inputs), inputs


def _arrays(result):
    return {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if isinstance(getattr(result, field.name), np.ndarray)
    }


def test_arrays_are_owned_and_read_only(result_and_inputs):
    result, inputs = result_and_inputs
    for name, array in _arrays(result).items():
        expected = array.copy()
        assert not np.shares_memory(array, inputs[name]), name
        inputs[name][...] = 0
        np.testing.assert_array_equal(array, expected)
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_restored_arrays_remain_read_only(result_and_inputs, how):
    result, _ = result_and_inputs
    restored = (
        pickle.loads(pickle.dumps(result)) if how == "pickle" else copy.deepcopy(result)
    )
    for name, array in _arrays(restored).items():
        assert not array.flags.writeable, name
        np.testing.assert_array_equal(array, getattr(result, name))


@pytest.mark.parametrize("how", ["original", "pickle", "deepcopy"])
def test_directional_maps_are_owned_and_read_only(env, how):
    rate = np.ones(env.n_bins)
    occupancy = np.ones(env.n_bins)
    rates = {"forward": rate}
    dwell = {"forward": occupancy}
    result = DirectionalPlaceFields(
        rates, dwell, env, ("forward",), spike_window=np.array([[0.0, 2.0]])
    )
    if how == "pickle":
        result = pickle.loads(pickle.dumps(result))
    elif how == "deepcopy":
        result = copy.deepcopy(result)
    rate[:] = 0
    occupancy[:] = 0
    rates.clear()
    dwell.clear()
    for mapping in (result.firing_rates, result.occupancy):
        np.testing.assert_array_equal(mapping["forward"], 1.0)
        with pytest.raises(ValueError, match="read-only"):
            mapping["forward"][0] = 0
        with pytest.raises(TypeError):
            mapping["forward"] = np.zeros(env.n_bins)
    assert not result.spike_window.flags.writeable


def test_summary_xarray_export_is_editable_and_independent(env):
    pytest.importorskip("xarray")
    result = DecodingSummary(
        env=env, **copy.deepcopy(CASES[0][1]), spike_window=np.array([[0.0, 2.0]])
    )
    dataset = result.to_xarray()
    for name in dataset.variables:
        assert dataset[name].values.flags.writeable, name
    dataset["map_x"].values[:] = 9
    dataset["map_bin"].values[:] = 3
    dataset["time"].values[:] = 9
    dataset.attrs["spike_window"][:] = 9
    np.testing.assert_array_equal(result.map_position, 1.0)
    np.testing.assert_array_equal(result.map_bin, [0, 1])
    np.testing.assert_array_equal(result.times, [0, 1])
    np.testing.assert_array_equal(result.spike_window, [[0, 2]])
