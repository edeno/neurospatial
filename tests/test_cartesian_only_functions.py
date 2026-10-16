"""Domain functions that read bin centers as Cartesian coordinates refuse polar.

Polar bin centers are ``(distance, angle)``. Averaging, differencing or
clustering them as ``(x, y)`` returns plausible but wrong numbers, so these
functions raise :class:`CartesianOnlyError` instead. Graph-based modes of the
same functions use physical edge distances and keep working.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.behavior import (
    detect_laps,
    detect_runs_between_regions,
    distance_to_region,
    graph_turn_sequence,
    path_progress,
    trajectory_similarity,
)
from neurospatial.encoding import (
    compute_field_emd,
    field_shape_metrics,
    field_shift_distance,
    rate_map_centroid,
    spatial_autocorrelation,
)
from neurospatial.ops import (
    compute_egocentric_distance,
    geodesic_rbf_basis,
    map_probabilities,
    resample_field,
    select_basis_centers,
    spatial_basis,
)
from neurospatial.regions import Regions


@pytest.fixture(scope="module")
def polar() -> Environment:
    """Full-circle polar environment with two distance rings and 8 angles."""
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=15.0,
        angle_bin_size=np.pi / 4,
    )


def _rates(env: Environment) -> np.ndarray:
    return np.random.default_rng(0).random(env.n_bins) + 0.1


def _polar_regions_env(env: Environment) -> Environment:
    regions = Regions()
    regions.add("near", polygon=box(0.0, -4.0, 15.0, 4.0))
    regions.add("far", polygon=box(15.0, -4.0, 30.0, 4.0))
    return env.with_regions(regions)


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            lambda env: field_shape_metrics(env, _rates(env), np.arange(4)),
            "field_shape_metrics()",
        ),
        (
            lambda env: field_shift_distance(
                env, _rates(env), np.arange(4), env, _rates(env), np.arange(4)
            ),
            "field_shift_distance()",
        ),
        (
            lambda env: spatial_autocorrelation(env, _rates(env)),
            "spatial_autocorrelation()",
        ),
        (
            lambda env: resample_field(_rates(env), env, env),
            "resample_field()",
        ),
        (
            lambda env: compute_egocentric_distance(
                np.array([[10.0, 0.0]]),
                np.array([0.0]),
                np.array([[20.0, 0.0]]),
                metric="geodesic",
                env=env,
            ),
            "compute_egocentric_distance(metric='geodesic')",
        ),
        (
            lambda env: map_probabilities(
                env, env, np.full(env.n_bins, 1.0 / env.n_bins)
            ),
            "map_probabilities()",
        ),
        (lambda env: spatial_basis(env, n_features=6, rng=0), "spatial_basis()"),
        (
            lambda env: graph_turn_sequence(env, np.array([0, 1, 9, 8]), 0, 8),
            "graph_turn_sequence()",
        ),
        (
            lambda env: detect_laps(np.arange(4), np.arange(4.0), env),
            "detect_laps()",
        ),
        (
            lambda env: rate_map_centroid(env, _rates(env), np.arange(4)),
            "rate_map_centroid(method='euclidean')",
        ),
        (
            lambda env: compute_field_emd(env, _rates(env), _rates(env)[::-1]),
            "compute_field_emd(metric='euclidean')",
        ),
        (
            lambda env: select_basis_centers(env, 3, rng=0),
            "select_basis_centers(method='kmeans')",
        ),
        (
            lambda env: geodesic_rbf_basis(env, n_centers=3, rng=0),
            "select_basis_centers(method='kmeans')",
        ),
        (
            lambda env: path_progress(
                np.array([0, 1]),
                env,
                start_bins=np.array([0, 0]),
                goal_bins=np.array([3, 3]),
                metric="euclidean",
            ),
            "path_progress(metric='euclidean')",
        ),
        (
            lambda env: distance_to_region(
                np.array([0, 1]),
                env,
                target_bins=np.array([3, 3]),
                metric="euclidean",
            ),
            "distance_to_region(metric='euclidean')",
        ),
        (
            lambda env: trajectory_similarity(
                np.array([0, 1, 2]), np.array([2, 3, 4]), env, method="hausdorff"
            ),
            "trajectory_similarity(method='hausdorff')",
        ),
        (
            lambda env: trajectory_similarity(
                np.array([0, 1, 2]), np.array([2, 3, 4]), env, method="dtw"
            ),
            "trajectory_similarity(method='dtw')",
        ),
        (
            lambda env: detect_runs_between_regions(
                np.array([0, 8, 9]),
                np.array([0.0, 0.1, 0.2]),
                _polar_regions_env(env),
                source="near",
                target="far",
                min_speed=1.0,
            ),
            "detect_runs_between_regions(min_speed=...)",
        ),
    ],
)
def test_cartesian_coordinate_functions_refuse_polar(polar, call, operation):
    with pytest.raises(CartesianOnlyError) as info:
        call(polar)
    assert info.value.operation == operation
    assert isinstance(info.value, TypeError)
    what, why, fix = str(info.value).split("\n")
    assert what == f"{operation} does not work on a polar environment."
    assert why.startswith("Why: ") and "(distance, angle)" in why
    assert fix.startswith("Fix: ")


@pytest.mark.parametrize(
    "call",
    [
        lambda env: rate_map_centroid(
            env, _rates(env), np.arange(4), method="geodesic"
        ),
        lambda env: compute_field_emd(
            env, _rates(env), _rates(env)[::-1], metric="geodesic"
        ),
        lambda env: select_basis_centers(env, 3, method="farthest_point", rng=0),
        lambda env: geodesic_rbf_basis(
            env, n_centers=3, center_method="farthest_point", rng=0
        ),
        lambda env: path_progress(
            np.array([0, 1]),
            env,
            start_bins=np.array([0, 0]),
            goal_bins=np.array([3, 3]),
        ),
        lambda env: distance_to_region(
            np.array([0, 1]), env, target_bins=np.array([3, 3])
        ),
        lambda env: trajectory_similarity(
            np.array([0, 1, 2]), np.array([2, 3, 4]), env, method="jaccard"
        ),
        lambda env: detect_runs_between_regions(
            np.array([0, 8, 9]),
            np.array([0.0, 0.1, 0.2]),
            _polar_regions_env(env),
            source="near",
            target="far",
        ),
    ],
)
def test_graph_modes_still_accept_polar(polar, call):
    call(polar)
