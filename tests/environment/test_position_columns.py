"""Positions need one column per bin dimension, with the same message everywhere.

Every operation that maps positions or points to bins refuses a wrong column
count with an error that names the operation, the shape it got, the shape it
needs and how to fix it, whatever the layout and whether the arena is known.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    RectilinearLocator,
    SpatialBins,
)
from neurospatial.ops.egocentric import compute_egocentric_distance


def _times(positions: np.ndarray) -> np.ndarray:
    return np.arange(len(positions), dtype=np.float64) * 0.1


def _point_bins() -> Environment:
    """Bins known only by their centers: no cells, and no known arena."""
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    bins = SpatialBins.from_arrays(centers, np.ones(4), 2)
    connections = BinConnections.from_arrays(
        [[0, 1], [1, 2], [2, 3], [3, 0]], np.full(4, 2.0)
    )
    locator = NearestBinLocator(centers, max_distance=1.0)
    return Environment.from_bins(
        Binning(
            "explicit",
            bins,
            connections,
            locator,
            MissingFaces("No faces were supplied."),
        )
    )


def _grid_without_arena() -> Environment:
    """Rectangular cells without a known arena, the lookup's direct path."""
    centers = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
    bins = SpatialBins.from_arrays(centers, np.ones(4), 2)
    connections = BinConnections.from_arrays(
        [[0, 1], [0, 2], [1, 3], [2, 3]], np.ones(4)
    )
    locator = RectilinearLocator((np.arange(3.0),) * 2, (2, 2), np.ones((2, 2), bool))
    return Environment.from_bins(
        Binning("explicit_grid", bins, connections, locator, MissingFaces("No faces."))
    )


def _open_field() -> Environment:
    rng = np.random.default_rng(0)
    return Environment.from_samples(rng.uniform(0, 10, (400, 2)), bin_size=2.0)


ENVIRONMENTS = {
    "open field": _open_field,
    "point bins": _point_bins,
    "grid without arena": _grid_without_arena,
}

Call = Callable[[Environment, np.ndarray], object]
ENTRY_POINTS: dict[str, tuple[Call, str]] = {
    "Environment.bin_at": (lambda env, p: env.bin_at(p), "positions"),
    "Environment.contains": (lambda env, p: env.contains(p), "positions"),
    "Environment.occupancy": (
        lambda env, p: env.occupancy(_times(p), p),
        "positions",
    ),
    "Environment.bin_sequence": (
        lambda env, p: env.bin_sequence(_times(p), p),
        "positions",
    ),
    "Environment.bin_sequence_with_runs": (
        lambda env, p: env.bin_sequence_with_runs(_times(p), p),
        "positions",
    ),
    "Environment.transitions": (
        lambda env, p: env.transitions(times=_times(p), positions=p),
        "positions",
    ),
    "Environment.interpolate": (
        lambda env, p: env.interpolate(np.zeros(env.n_bins), p),
        "points",
    ),
    "distance_to()": (lambda env, p: env.distance_to(points=p), "points"),
}


@pytest.fixture(params=list(ENVIRONMENTS), scope="module")
def env(request) -> Environment:
    return ENVIRONMENTS[request.param]()


@pytest.mark.parametrize("operation", list(ENTRY_POINTS))
def test_wrong_column_count_names_the_operation_and_the_fix(env, operation):
    call, name = ENTRY_POINTS[operation]
    positions = np.ones((5, 3))
    expected = (
        rf"^{re.escape(operation)} got {name} with shape \(5, 3\), but this "
        rf"environment's bins are 2-D, so {name} needs shape \(n_{name}, 2\)\.\n"
        rf"Why: each row is one {name[:-1]}, .*\n"
        rf"Fix: pass an array of shape \(n_{name}, 2\), one row per {name[:-1]}$"
    )
    with pytest.raises(ValueError, match=expected):
        call(env, positions)


@pytest.mark.parametrize("operation", list(ENTRY_POINTS))
def test_one_column_on_a_planar_environment_points_at_linearized_positions(
    operation,
):
    call, _ = ENTRY_POINTS[operation]
    with pytest.raises(ValueError, match=r"to_linear\(\)[\s\S]*not linearized"):
        call(_point_bins(), np.ones((5, 1)))


def test_one_flat_point_is_shown_as_one_row():
    with pytest.raises(ValueError, match=r"one point is one row"):
        _open_field().distance_to(points=[5.0, 5.0])


def test_flat_positions_on_a_one_dimensional_environment_suggest_a_column():
    env = Environment.from_samples(np.linspace(0, 10, 50)[:, None], bin_size=1.0)
    x = np.linspace(0, 10, 5)
    with pytest.raises(ValueError, match=r"np\.reshape\(positions, \(-1, 1\)\)"):
        env.occupancy(_times(x), x)
    np.testing.assert_array_equal(
        env.occupancy(_times(x), x[:, None]).sum(), _times(x)[-1]
    )


@pytest.mark.parametrize("method", ["bin_at", "contains"])
def test_flat_positions_are_reported_as_given(method):
    """A flat array of several values is not silently read as one row."""
    with pytest.raises(
        ValueError,
        match=r"got positions with shape \(5,\)[\s\S]*Linearized positions",
    ):
        getattr(_open_field(), method)(np.arange(5.0))


def test_bin_at_still_takes_one_flat_point(env):
    np.testing.assert_array_equal(env.bin_at(env.bin_centers[1]), [1])


def test_checks_run_before_work_that_skips_the_lookup(env):
    """Wrong columns are refused even where no position is looked up."""
    with pytest.raises(ValueError, match=r"^Environment\.occupancy got"):
        env.occupancy(np.array([0.0]), np.ones((1, 3)))
    with pytest.raises(ValueError, match=r"^Environment\.interpolate got"):
        env.interpolate(np.zeros(env.n_bins), np.ones((0, 3)))
    with pytest.raises(ValueError, match=r"^distance_to\(\) got"):
        env.distance_to(points=np.full((1, 3), np.nan))


def test_bin_sequence_refuses_wrong_columns_before_warning_about_gaps():
    times = np.array([0.0, 0.1, 5.0])  # one gap longer than max_gap
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"^Environment\.bin_sequence got"):
            _open_field().bin_sequence(times, np.ones((3, 3)))


def test_geodesic_egocentric_distance_names_itself_on_a_volume():
    rng = np.random.default_rng(1)
    volume = Environment.from_samples(rng.uniform(0, 10, (500, 3)), bin_size=2.5)
    positions = np.full((4, 2), 5.0)
    with pytest.raises(
        ValueError,
        match=r"^compute_egocentric_distance\(metric='geodesic'\) got positions "
        r"with shape \(4, 2\)[\s\S]*bins are 3-D",
    ):
        compute_egocentric_distance(
            positions, None, np.array([[1.0, 1.0]]), metric="geodesic", env=volume
        )
