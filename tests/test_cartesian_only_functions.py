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
from neurospatial.decoding import (
    DecodingResult,
    decode_position_summary,
    map_position,
    mean_position,
)
from neurospatial.encoding import (
    compute_field_emd,
    field_shape_metrics,
    field_shift_distance,
    rate_map_centroid,
    spatial_autocorrelation,
)
from neurospatial.ops import (
    geodesic_rbf_basis,
    map_points_to_bins,
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


def _behind_bins(env: Environment) -> np.ndarray:
    """Outer-ring bins on either side of the angle seam (theta = +/- 7 pi / 8)."""
    centers = env.bin_centers
    return np.flatnonzero(
        np.isclose(centers[:, 0], 22.5)
        & np.isclose(np.abs(centers[:, 1]), 7 * np.pi / 8)
    )


def test_mean_position_refuses_polar_seam_instead_of_reporting_ahead(polar):
    behind = _behind_bins(polar)
    assert len(behind) == 2
    posterior = np.zeros((1, polar.n_bins))
    posterior[0, behind] = 0.5
    # Averaging (distance, angle) as (x, y) would report angle 0: straight ahead.
    with pytest.raises(CartesianOnlyError, match=r"mean_position\(\)") as info:
        mean_position(polar, posterior)
    message = str(info.value)
    assert "polar environment" in message
    assert "map_position" in message
    # The most probable bin is still well defined in polar coordinates.
    assert np.isclose(abs(map_position(polar, posterior)[0, 1]), 7 * np.pi / 8)


def test_decoding_result_mean_position_refuses_polar(polar):
    posterior = np.full((2, polar.n_bins), 1.0 / polar.n_bins)
    result = DecodingResult(posterior=posterior, env=polar)
    with pytest.raises(CartesianOnlyError, match="mean_position"):
        _ = result.mean_position
    with pytest.raises(CartesianOnlyError, match="mean_position"):
        result.to_dataframe()
    assert result.map_position.shape == (2, 2)


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
            lambda env: decode_position_summary(
                env, np.ones((3, 2)), np.ones((2, env.n_bins)), 0.02
            ),
            "decode_position_summary()",
        ),
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
            lambda env: map_points_to_bins(env, np.array([[10.0, 0.0]])),
            "map_points_to_bins()",
        ),
        (
            lambda env: resample_field(_rates(env), env, env),
            "resample_field()",
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
    assert "polar environment" in str(info.value)


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
