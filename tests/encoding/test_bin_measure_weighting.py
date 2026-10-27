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


# --- compute_field_emd ----------------------------------------------------------


def test_emd_is_unchanged_by_splitting_a_bin(square, square_split) -> None:
    from neurospatial.encoding import compute_field_emd

    # A uniform field over the square moves into the narrow left strip.
    emd = compute_field_emd(square, np.ones(3), np.array([1.0, 0.0, 0.0]))
    split_emd = compute_field_emd(
        square_split, np.ones(4), np.array([1.0, 0.0, 0.0, 0.0])
    )
    # 0.1 of the mass moves 1 cm and 0.8 moves 5.5 cm.
    assert emd == pytest.approx(4.5, rel=1e-9)
    assert split_emd == pytest.approx(4.5, rel=1e-9)


@pytest.mark.parametrize(
    "edges", [np.arange(11.0), np.array([0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 10.0])]
)
def test_emd_on_a_track_is_the_wasserstein_distance_of_the_rate_fields(
    edges,
) -> None:
    from scipy.stats import wasserstein_distance

    from neurospatial.encoding import compute_field_emd

    env = track(edges)
    rng = np.random.default_rng(6)
    rates_1, rates_2 = rng.uniform(0, 5, (2, env.n_bins))
    centers, measures = env.bin_centers[:, 0], env.bin_sizes
    expected = wasserstein_distance(
        centers, centers, rates_1 * measures, rates_2 * measures
    )
    assert compute_field_emd(env, rates_1, rates_2) == pytest.approx(expected, rel=1e-9)


def test_unnormalized_emd_moves_rate_times_measure(square) -> None:
    from neurospatial.encoding import compute_field_emd

    # 1 Hz over the 100 cm^2 square and 10 Hz over the 10 cm^2 left strip
    # hold the same firing, 100 Hz cm^2.
    uniform, strip = np.ones(3), np.array([10.0, 0.0, 0.0])
    unnormalized = compute_field_emd(square, uniform, strip, normalize=False)
    normalized = compute_field_emd(square, uniform, strip)
    assert unnormalized == pytest.approx(100.0 * normalized, rel=1e-9)


# --- in_out_field_ratio ---------------------------------------------------------


def test_in_out_ratio_is_unchanged_by_splitting_a_bin(square, square_split) -> None:
    from neurospatial.encoding import in_out_field_ratio

    # Field in the left strip at 10 Hz; outside, 1 Hz on 10 cm^2 and 2 Hz on
    # 80 cm^2, a mean of 170 / 90 Hz over the space outside the field.
    ratio = in_out_field_ratio(square, np.array([10.0, 1.0, 2.0]), np.array([0]))
    split_ratio = in_out_field_ratio(
        square_split, np.array([10.0, 1.0, 2.0, 2.0]), np.array([0])
    )
    assert ratio == pytest.approx(10.0 / (170.0 / 90.0), rel=1e-12)
    assert split_ratio == pytest.approx(ratio, rel=1e-12)


def test_in_out_ratio_on_equal_bins_is_the_ratio_of_bin_means(grid) -> None:
    from neurospatial.encoding import in_out_field_ratio

    rng = np.random.default_rng(7)
    rates = rng.uniform(0, 5, grid.n_bins)
    rates[::7] = np.nan
    field = np.arange(10, 40)
    inside = np.zeros(grid.n_bins, dtype=bool)
    inside[field] = True
    expected = np.nanmean(rates[inside]) / np.nanmean(rates[~inside])
    assert in_out_field_ratio(grid, rates, field) == pytest.approx(expected, rel=1e-12)


def test_in_out_ratio_refuses_a_map_from_another_environment(square) -> None:
    from neurospatial.encoding import in_out_field_ratio

    with pytest.raises(ValueError, match=r"env has 3 bins(.|\n)*rates\.env"):
        in_out_field_ratio(square, np.ones(4), np.array([0]))


# --- compare_spatial_rates mean rates -------------------------------------------


def test_comparison_mean_rate_is_firing_over_time_in_the_common_bins(square) -> None:
    """Spikes over seconds, however the common bins are sized or visited."""
    from environment_helpers import rates_result

    from neurospatial.encoding import compare_spatial_rates

    # Condition A: 6 spikes in 2 s, 2 spikes in 1 s and 20 spikes in 5 s.
    occupancy_a = np.array([2.0, 1.0, 5.0])
    rates_a = np.array([6.0, 2.0, 20.0]) / occupancy_a
    occupancy_b = np.array([1.0, 1.0, 1.0])
    rates_b = np.array([1.0, 2.0, 3.0])
    table = compare_spatial_rates(
        rates_result(square, rates_a, occupancy_a),
        rates_result(square, rates_b, occupancy_b),
    ).summary_table()
    assert table["mean_rate_a"].item() == pytest.approx(28.0 / 8.0)
    assert table["mean_rate_b"].item() == pytest.approx(2.0)
    assert table["mean_rate_difference"].item() == pytest.approx(2.0 - 28.0 / 8.0)
