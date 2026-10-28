"""Analysis result buffers are owned, read-only, and survive serialization."""

from __future__ import annotations

import copy
import pickle
from dataclasses import fields

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior.decisions import DecisionBoundaryMetrics
from neurospatial.behavior.epochs import LabelEpochsResult
from neurospatial.behavior.navigation import (
    GoalDirectedMetrics,
    PathEfficiencyResult,
    SubgoalEfficiencyResult,
)
from neurospatial.behavior.resampling import ResampledPathResult
from neurospatial.behavior.segmentation import Run
from neurospatial.behavior.trajectory import CumulativeDistanceResult, MSDResult
from neurospatial.decoding import DecodingSummary
from neurospatial.decoding.assemblies import AssemblyDetectionResult, AssemblyPattern
from neurospatial.decoding.trajectory import IsotonicFitResult, RadonDetectionResult
from neurospatial.encoding import RateComparisonResult
from neurospatial.encoding._metrics import BatchScoresResult
from neurospatial.encoding.population import PopulationCoverageResult
from neurospatial.encoding.spatial import DirectionalPlaceFields, PlaceFieldsResult
from neurospatial.events import ContinuousEventResult
from neurospatial.ops import SpatialFeatureResult
from neurospatial.ops.visibility import ViewshedResult
from neurospatial.stats.circular import CircularBasisResult
from neurospatial.stats.shuffle import ShuffleTestResult


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
    (
        MSDResult,
        {
            "lags": np.array([1.0, 2.0]),
            "msd": np.ones(2),
            "metric": "euclidean",
            "epochs": np.array([[0.0, 1.0]]),
        },
    ),
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
            "unit_ids": np.array([5, 9]),
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
            "unit_ids": np.array([5, 9]),
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
    (
        ResampledPathResult,
        {
            "times": np.array([0.0, 0.5]),
            "positions": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "run_distance": np.array([0.0, 1.0]),
            "cumulative_distance": np.array([0.0, 1.0]),
            "run_ids": np.zeros(2, dtype=np.intp),
            "source_left": np.array([0, 0], dtype=np.intp),
            "source_right": np.array([0, 1], dtype=np.intp),
            "weights": np.array([0.0, 0.5]),
            "off_grid_endpoint": np.zeros(2, dtype=bool),
            "run_sample_bounds": np.array([[0, 1]], dtype=np.intp),
            "run_time_bounds": np.array([[0.0, 1.0]]),
            "run_path_lengths": np.array([2.0]),
            "n_source_samples": 2,
            "spacing": 1.0,
            "include_endpoint": False,
            "max_gap": 0.5,
            "epochs": np.array([[0.0, 1.0]]),
        },
    ),
    (
        LabelEpochsResult,
        {
            "labels": ("A", "B"),
            "epoch_bounds": np.array([[0.0, 2.0], [2.0, 3.0]]),
            "epoch_intervals": np.array([[0, 1], [2, 2]], dtype=np.intp),
            "interval_labels": np.array([0, 0, 1], dtype=np.intp),
            "interval_supported": np.ones(3, dtype=bool),
            "unsupported_labels": (),
            "max_gap": 0.5,
            "input_epochs": np.array([[0.0, 10.0]]),
            "missing_label": None,
            "n_declared_breaks": 0,
        },
    ),
    (
        ContinuousEventResult,
        {
            "feature_ids": np.array(["speed", "zone"]),
            "value_kinds": ("linear", "categorical"),
            "interpolation": ("linear", "previous"),
            "feature_units": ("cm/s", None),
            "event_status": np.array(["retained", "outside_recording"]),
            "overlapping": np.zeros(2, dtype=bool),
            "counts": np.ones((2, 2), dtype=np.int64),
            "sums": np.array([[1.0, np.nan], [2.0, np.nan]]),
            "sin_sums": np.full((2, 2), np.nan),
            "cos_sums": np.full((2, 2), np.nan),
            "category_features": np.array([1], dtype=np.intp),
            "category_codes": np.array([0], dtype=np.int64),
            "category_labels": np.array(["center"]),
            "category_counts": np.ones((2, 1), dtype=np.int64),
            "window": (0.0, 1.0),
            "dt": 0.5,
            "window_coverage": "complete",
            "resultant_threshold": 1e-12,
            "max_gap": 0.5,
            "epochs": np.array([[0.0, 10.0]]),
            "values": np.array([[[1.0, 0.0], [2.0, 0.0]]]),
            "support": np.ones((1, 2, 2), dtype=bool),
            "event_ids": np.array([10, 11]),
        },
    ),
    (
        RateComparisonResult,
        {
            "unit_ids": np.array([3, 4]),
            "correlation": np.array([0.5, np.nan]),
            "status": np.array(["scored", "insufficient_common_bins"]),
            "n_common_bins": np.array([3, 1], dtype=np.intp),
            "bins_a": np.array([0, 1, 2], dtype=np.intp),
            "bins_b": np.array([0, 1, 2], dtype=np.intp),
            "firing_rates_a_on_bins": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            "firing_rates_b_on_bins": np.array([[2.0, 1.0, 3.0], [1.0, 2.0, 3.0]]),
            "occupancy_a": np.ones(3),
            "occupancy_b": np.ones(3),
            "occupied_a": np.ones(3, dtype=bool),
            "occupied_b": np.ones(3, dtype=bool),
            "common_support": np.array([[True, True, True], [True, False, False]]),
            "method": "pearson",
            "unit_ids_only_a": np.array([1]),
            "unit_ids_only_b": np.array([], dtype=int),
        },
    ),
    (
        SpatialFeatureResult,
        {
            "values": np.array([[[1.0]], [[np.nan]]]),
            "support": np.array([[[True]], [[False]]]),
            "bins": np.array([[0], [-1]], dtype=np.intp),
            "feature_ids": np.array(["height"]),
            "units": "cm",
            "frame": "room",
            "ego_points": np.array([[5.0, 0.0]]),
            "query_positions": np.array([[[5.0, 0.0]], [[50.0, 0.0]]]),
        },
    ),
    (
        ShuffleTestResult,
        {
            "observed_score": 1.0,
            "null_scores": np.array([0.1, 0.5, 2.0]),
            "p_value": 0.5,
            "z_score": 0.2,
            "shuffle_type": "time_bins",
            "n_shuffles": 3,
        },
    ),
    (
        CircularBasisResult,
        {
            "sin_component": np.sin([0.0, 1.0]),
            "cos_component": np.cos([0.0, 1.0]),
            "angles": np.array([0.0, 1.0]),
        },
    ),
    (
        ViewshedResult,
        {
            "visibility_map": np.array([1.0, 0.0, 1.0]),
            "visible_cues": np.array([1], dtype=np.intp),
            "cue_distances": np.array([3.0]),
            "cue_bearings": np.array([0.5]),
        },
    ),
    (
        PopulationCoverageResult,
        {
            "place_fields": [[np.array([0])]],
            "is_observed": np.array([True, True]),
            "bin_centers": np.array([[0.0, 0.0], [5.0, 0.0]]),
            "bin_sizes": np.array([1.0, 1.0]),
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
    if cls is RateComparisonResult:
        inputs["env"] = env
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
        if name in inputs:  # derived (init=False) arrays have no input
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
        rates, dwell, env, spike_window=np.array([[0.0, 2.0]])
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


@pytest.mark.parametrize("how", ["original", "pickle", "deepcopy"])
def test_place_field_bins_are_owned_and_read_only(how):
    bins = np.array([0, 1, 2])
    given = [bins]
    result = PlaceFieldsResult(fields=given)
    coverage = PopulationCoverageResult(
        place_fields=[result],
        is_observed=np.ones(3, dtype=bool),
        bin_centers=np.zeros((3, 2)),
        bin_sizes=np.ones(3),
    )
    if how == "pickle":
        result, coverage = pickle.loads(pickle.dumps((result, coverage)))
    elif how == "deepcopy":
        result, coverage = copy.deepcopy((result, coverage))
    bins[:] = 9
    given.clear()
    for owned in (result, coverage.place_fields[0]):
        assert len(owned) == 1
        np.testing.assert_array_equal(owned[0], [0, 1, 2])
        with pytest.raises(ValueError, match="read-only"):
            owned[0][0] = 5
    assert isinstance(result.fields, tuple)
    assert isinstance(coverage.place_fields, tuple)


def test_summary_xarray_export_is_editable_and_independent(env):
    pytest.importorskip("xarray")
    inputs = next(inputs for cls, inputs in CASES if cls is DecodingSummary)
    result = DecodingSummary(
        env=env, **copy.deepcopy(inputs), spike_window=np.array([[0.0, 2.0]])
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
