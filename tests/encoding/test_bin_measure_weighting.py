"""Field statistics weight each bin by the space it covers.

A rate map describes a continuous field: bin ``i`` holds ``rate_i *
measure_i`` of it, where the measure is ``env.bin_sizes[i]`` (area, length or
volume). Three invariants follow, and each test below checks one of them:

1. Splitting a bin into pieces with the same rate leaves the statistic
   unchanged.
2. A uniform field on unequal bins gives the geometric answer.
3. On equal bins the statistic is the unweighted one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)


def _explicit_env(centers, sizes, pairs) -> Environment:
    """Environment with the given bin centers, measures and neighbor pairs."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    pairs = np.asarray(pairs, dtype=int)
    distances = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(centers, sizes, centers.shape[1]),
            BinConnections.from_arrays(pairs, distances),
            NearestBinLocator(centers, max_distance=math.inf),
            MissingFaces("Test bins without faces."),
        ),
        units="cm",
    )


def _chain(n: int) -> np.ndarray:
    return np.column_stack([np.arange(n - 1), np.arange(1, n)])


def strips(x_edges) -> Environment:
    """A 10 cm tall floor cut into vertical strips at ``x_edges``."""
    x_edges = np.asarray(x_edges, dtype=float)
    centers = np.column_stack(
        [(x_edges[:-1] + x_edges[1:]) / 2, np.full(len(x_edges) - 1, 5.0)]
    )
    return _explicit_env(centers, 10.0 * np.diff(x_edges), _chain(len(centers)))


def track(edges) -> Environment:
    """A 1-D track whose bins are the intervals between ``edges``."""
    edges = np.asarray(edges, dtype=float)
    return _explicit_env(
        (edges[:-1] + edges[1:]) / 2, np.diff(edges), _chain(len(edges) - 1)
    )


# A 10 x 10 cm square cut into strips 1, 1 and 8 cm wide, and the same square
# with the wide strip split in two.
SQUARE = [0.0, 1.0, 2.0, 10.0]
SQUARE_SPLIT = [0.0, 1.0, 2.0, 6.0, 10.0]


@pytest.fixture(scope="module")
def square() -> Environment:
    return strips(SQUARE)


@pytest.fixture(scope="module")
def square_split() -> Environment:
    return strips(SQUARE_SPLIT)


@pytest.fixture(scope="module")
def grid() -> Environment:
    rng = np.random.default_rng(3)
    return Environment.from_samples(rng.uniform(0, 20, (2000, 2)), bin_size=2.0)


# --- rate_map_centroid ----------------------------------------------------------


def test_centroid_of_a_uniform_field_is_the_center_of_the_square(square) -> None:
    from neurospatial.encoding import rate_map_centroid

    centroid = rate_map_centroid(square, np.ones(3), np.arange(3))
    np.testing.assert_allclose(centroid, [5.0, 5.0])


def test_centroid_is_unchanged_by_splitting_a_bin(square, square_split) -> None:
    from neurospatial.encoding import rate_map_centroid

    rates = np.array([1.0, 3.0, 7.0])
    split_rates = np.array([1.0, 3.0, 7.0, 7.0])
    np.testing.assert_allclose(
        rate_map_centroid(square, rates, np.arange(3)),
        rate_map_centroid(square_split, split_rates, np.arange(4)),
        rtol=1e-14,
    )


def test_silent_field_centroid_is_its_geometric_center(square) -> None:
    from neurospatial.encoding import rate_map_centroid

    np.testing.assert_allclose(
        rate_map_centroid(square, np.zeros(3), np.arange(3)), [5.0, 5.0]
    )


def test_geodesic_centroid_of_a_uniform_field_holds_the_track_midpoint() -> None:
    from neurospatial.encoding import rate_map_centroid

    env = track([0.0, 1.0, 2.0, 10.0])  # the midpoint, 5 cm, is in bin 2
    centroid = rate_map_centroid(env, np.ones(3), np.arange(3), method="geodesic")
    assert centroid[0] == 6.0


@pytest.mark.parametrize("method", ["euclidean", "geodesic"])
def test_centroid_on_equal_bins_is_the_rate_weighted_one(grid, method) -> None:
    from neurospatial.encoding import rate_map_centroid
    from neurospatial.ops.distance import pairwise_distances

    rng = np.random.default_rng(4)
    rates = rng.uniform(0, 5, grid.n_bins)
    field = np.arange(0, grid.n_bins, 3)
    centers, field_rates = grid.bin_centers[field], rates[field]
    if method == "euclidean":
        expected = np.sum(centers * field_rates[:, None], axis=0) / field_rates.sum()
    else:
        distances = pairwise_distances(grid.connectivity, field)
        expected = centers[np.argmin(field_rates @ distances)]
    np.testing.assert_array_equal(
        rate_map_centroid(grid, rates, field, method=method), expected
    )


def test_field_shift_distance_is_unchanged_by_splitting_a_bin(
    square, square_split
) -> None:
    from neurospatial.encoding import field_shift_distance

    # Field 1 fills the square uniformly (centroid 5 cm); field 2 is the
    # narrow left strips only (centroid 1 cm).
    shift = field_shift_distance(
        square, np.ones(3), np.arange(3), square, np.ones(3), np.arange(2)
    )
    split_shift = field_shift_distance(
        square_split, np.ones(4), np.arange(4), square_split, np.ones(4), np.arange(2)
    )
    assert shift == pytest.approx(4.0, abs=1e-12)
    assert split_shift == pytest.approx(4.0, abs=1e-12)


# --- field_shape_metrics --------------------------------------------------------


def test_field_shape_counts_a_large_bin_like_the_area_it_covers() -> None:
    """Four bins balanced by area: 4 cm^2 at x = +-1 and 1 cm^2 at y = +-2.

    Weighted by area, the spread is 2 * 4 * 1 / 10 = 0.8 cm^2 along both axes,
    a round field. Counted as four equal bins it would be stretched along y.
    """
    from neurospatial.encoding import field_shape_metrics

    env = _explicit_env(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        [4.0, 4.0, 1.0, 1.0],
        [[0, 2], [2, 1], [1, 3], [3, 0]],
    )
    shape = field_shape_metrics(env, np.ones(4), np.arange(4))
    assert shape["eccentricity"] == pytest.approx(0.0, abs=1e-6)
    assert shape["major_axis_length"] == pytest.approx(2 * np.sqrt(0.8))
    assert shape["minor_axis_length"] == pytest.approx(2 * np.sqrt(0.8))


def test_field_shape_on_equal_bins_is_the_rate_weighted_one(grid) -> None:
    from neurospatial.encoding import field_shape_metrics

    rng = np.random.default_rng(5)
    rates = rng.uniform(0, 5, grid.n_bins)
    field = np.arange(0, grid.n_bins, 2)
    weights = rates[field] / rates[field].sum()
    centered = grid.bin_centers[field] - weights @ grid.bin_centers[field]
    eigenvalues = np.linalg.eigvalsh(
        np.einsum("i,ij,ik->jk", weights, centered, centered)
    )
    shape = field_shape_metrics(grid, rates, field)
    assert shape["major_axis_length"] == pytest.approx(
        2 * np.sqrt(eigenvalues[1]), rel=1e-12
    )
    assert shape["minor_axis_length"] == pytest.approx(
        2 * np.sqrt(eigenvalues[0]), rel=1e-12
    )
