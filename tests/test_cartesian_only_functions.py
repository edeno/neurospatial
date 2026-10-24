"""Domain functions that read bin centers as Cartesian coordinates refuse polar.

Polar bin centers are ``(distance, angle)``. Averaging, differencing or
clustering them as ``(x, y)`` returns plausible but wrong numbers, so these
functions raise :class:`CartesianOnlyError` instead. So do the functions that
read positions as the animal's allocentric positions, although
``env.bin_at`` itself locates ``(distance, angle)`` rows. Graph-based modes of the
same functions use physical edge distances and keep working.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment, compute_spatial_rate
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.behavior import (
    Trial,
    compute_decision_analysis,
    compute_step_lengths,
    compute_vte_session,
    detect_laps,
    detect_runs_between_regions,
    distance_to_boundary,
    distance_to_region,
    distance_to_reward,
    graph_turn_sequence,
    mean_square_displacement,
    path_progress,
    trajectory_similarity,
)
from neurospatial.decoding import decoding_error
from neurospatial.encoding import (
    border_score,
    compute_directional_place_fields,
    compute_field_emd,
    compute_spatial_rates,
    compute_view_rate,
    field_shape_metrics,
    field_shift_distance,
    place_cell_significance,
    rate_map_centroid,
    spatial_autocorrelation,
)
from neurospatial.ops import (
    compute_egocentric_distance,
    geodesic_rbf_basis,
    map_probabilities,
    resample_field,
    sample_egocentric_features,
    sample_spatial_features,
    select_basis_centers,
    spatial_basis,
)
from neurospatial.regions import Regions
from neurospatial.simulation import (
    BoundaryCellModel,
    ObjectVectorCellModel,
    PlaceCellModel,
    SpatialViewCellModel,
    simulate_trajectory_ou,
)


@pytest.fixture(scope="module")
def polar() -> Environment:
    """Full-circle polar environment with two distance rings and 8 angles."""
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=15.0,
        angle_bin_size=np.pi / 4,
    )


def _room() -> Environment:
    """Allocentric 2-D environment paired with the polar one."""
    x, y = np.meshgrid(np.arange(0.0, 20.0, 2.0), np.arange(0.0, 20.0, 2.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=4.0)


# Six (distance, angle) samples inside the 30 cm polar surround.
_TIMES = np.arange(0.0, 3.0, 0.5)
_POLAR_POSITIONS = np.column_stack((np.full(6, 10.0), np.linspace(-1.0, 1.0, 6)))


def _rates(env: Environment) -> np.ndarray:
    return np.random.default_rng(0).random(env.n_bins) + 0.1


def _polar_regions_env(env: Environment) -> Environment:
    # Polygon regions are Cartesian and refused on polar bins; points are
    # (distance, angle) pairs.
    regions = Regions()
    regions.add("near", point=tuple(env.bin_centers[0]))
    regions.add("far", point=tuple(env.bin_centers[8]))
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
            "field_shift_distance(env_1=...)",
        ),
        (
            lambda env: field_shift_distance(
                _room(), _rates(_room()), np.arange(4), env, _rates(env), np.arange(4)
            ),
            "field_shift_distance(env_2=...)",
        ),
        (
            lambda env: spatial_autocorrelation(env, _rates(env)),
            "spatial_autocorrelation()",
        ),
        (
            lambda env: resample_field(_rates(env), env, _room()),
            "resample_field(src_env=...)",
        ),
        (
            lambda env: resample_field(_rates(_room()), _room(), env),
            "resample_field(dst_env=...)",
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
                env, _room(), np.full(env.n_bins, 1.0 / env.n_bins)
            ),
            "map_probabilities(source_env=...)",
        ),
        (
            lambda env: map_probabilities(
                _room(), env, np.full(_room().n_bins, 1.0 / _room().n_bins)
            ),
            "map_probabilities(target_env=...)",
        ),
        (lambda env: spatial_basis(env, n_features=6, rng=0), "spatial_basis()"),
        (
            lambda env: graph_turn_sequence(env, np.array([0, 1, 9, 8])),
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
            lambda env: select_basis_centers(env, n_centers=3, rng=0),
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
        (
            lambda env: border_score(env, _rates(env)),
            "Wall geometry (boundary_source='physical')",
        ),
        (
            lambda env: compute_spatial_rate(
                env,
                np.array([0.5, 1.5]),
                np.arange(0.0, 3.0, 0.5),
                np.column_stack((np.full(6, 10.0), np.linspace(-1.0, 1.0, 6))),
                time_allocation="linear",
            ),
            "time_allocation='linear'",
        ),
        (
            lambda env: compute_spatial_rate(
                env, np.array([0.5, 1.5]), _TIMES, _POLAR_POSITIONS
            ),
            "compute_spatial_rate()",
        ),
        (
            lambda env: compute_spatial_rates(
                env, [np.array([0.5, 1.5])], _TIMES, _POLAR_POSITIONS
            ),
            "compute_spatial_rates()",
        ),
        (
            lambda env: compute_directional_place_fields(
                env,
                np.array([0.5, 1.5]),
                _TIMES,
                _POLAR_POSITIONS,
                np.array(["out"] * _TIMES.size, dtype=object),
            ),
            "compute_directional_place_fields()",
        ),
        (
            lambda env: compute_view_rate(
                env, np.array([0.5, 1.5]), _TIMES, _POLAR_POSITIONS, np.zeros(6)
            ),
            "compute_view_rate()",
        ),
        (
            lambda env: compute_decision_analysis(
                _polar_regions_env(env),
                _TIMES,
                _POLAR_POSITIONS,
                decision_region="near",
                goal_regions=["far"],
            ),
            "compute_decision_analysis()",
        ),
        (
            lambda env: compute_vte_session(
                _polar_regions_env(env),
                _TIMES,
                _POLAR_POSITIONS,
                decision_region="near",
                trials=[Trial(0.0, 2.5, "near", "far", True)],
            ),
            "compute_vte_session()",
        ),
        (
            lambda env: compute_step_lengths(
                _POLAR_POSITIONS, metric="geodesic", env=env
            ),
            "compute_step_lengths(metric='geodesic')",
        ),
        (
            lambda env: mean_square_displacement(
                _TIMES, _POLAR_POSITIONS, metric="geodesic", env=env
            ),
            "mean_square_displacement(metric='geodesic')",
        ),
        (
            lambda env: mean_square_displacement(
                _TIMES[:1], _POLAR_POSITIONS[:1], metric="geodesic", env=env
            ),
            "mean_square_displacement(metric='geodesic')",
        ),
        (
            lambda env: decoding_error(env, _POLAR_POSITIONS, _POLAR_POSITIONS),
            "decoding_error(metric='geodesic')",
        ),
        (
            lambda env: simulate_trajectory_ou(
                env.with_metadata(units="cm"), duration=1.0, speed_units="cm", seed=0
            ),
            "simulate_trajectory_ou()",
        ),
        (lambda env: BoundaryCellModel(env), "BoundaryCellModel()"),
        (
            lambda env: ObjectVectorCellModel(
                env,
                object_positions=np.array([[10.0, 0.0]]),
                preferred_distance=5.0,
                distance_width=2.0,
            ),
            "ObjectVectorCellModel()",
        ),
        (
            lambda env: SpatialViewCellModel(
                env, preferred_view_location=np.array([10.0, 0.0])
            ),
            "SpatialViewCellModel()",
        ),
        (
            lambda env: PlaceCellModel(
                env, center=np.array([10.0, 0.0]), width=5.0, metric="geodesic"
            ),
            "PlaceCellModel(metric='geodesic')",
        ),
        (
            lambda env: distance_to_reward(
                env, _TIMES, _POLAR_POSITIONS, np.array([1.0]), _POLAR_POSITIONS[:1]
            ),
            "distance_to_reward()",
        ),
        (
            lambda env: distance_to_boundary(env, _POLAR_POSITIONS),
            "distance_to_boundary()",
        ),
        (
            lambda env: compute_vte_session(
                _polar_regions_env(env),
                _TIMES,
                _POLAR_POSITIONS,
                decision_region="near",
                trials=[],
            ),
            "compute_vte_session()",
        ),
        (
            lambda env: place_cell_significance(
                env, np.array([0.5, 1.5]), _TIMES, _POLAR_POSITIONS
            ),
            "place_cell_significance()",
        ),
        (
            lambda env: sample_spatial_features(
                env, np.ones(env.n_bins), _POLAR_POSITIONS
            ),
            "sample_spatial_features()",
        ),
        (
            lambda env: sample_egocentric_features(
                env,
                np.ones(env.n_bins),
                _POLAR_POSITIONS,
                np.zeros(len(_POLAR_POSITIONS)),
                np.array([[1.0, 0.0]]),
            ),
            "sample_egocentric_features()",
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
        lambda env: select_basis_centers(
            env, n_centers=3, method="farthest_point", rng=0
        ),
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
