"""Posterior means on egocentric polar environments.

Polar bin centers are ``(distance, angle)``. The posterior mean is the
probability-weighted centroid in the animal's egocentric plane (x ahead,
y left), reported back as ``(distance, angle)``. Averaging the raw columns
would place two bins just either side of the angle seam directly ahead.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import (
    DecodingResult,
    decode_position,
    decode_position_summary,
    mean_position,
)


def _polar(angle_range: tuple[float, float]) -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=angle_range,
        distance_bin_size=15.0,
        angle_bin_size=np.pi / 4,
    )


@pytest.fixture(scope="module", params=[(-np.pi, np.pi), (0.0, 2 * np.pi)])
def polar(request) -> Environment:
    return _polar(request.param)


def _seam_posterior(env: Environment) -> np.ndarray:
    """Equal mass on the two outer-ring bins nearest directly behind."""
    centers = env.bin_centers
    behind = np.flatnonzero(
        np.isclose(centers[:, 0], 22.5)
        & np.isclose(np.cos(centers[:, 1]), np.cos(7 * np.pi / 8))
    )
    assert len(behind) == 2
    posterior = np.zeros((1, env.n_bins))
    posterior[0, behind] = 0.5
    return posterior


def test_seam_mean_is_behind_not_ahead(polar):
    distance, angle = mean_position(polar, _seam_posterior(polar))[0]
    # Centroid of two points at radius 22.5, angles pi -/+ pi/8.
    assert distance == pytest.approx(22.5 * np.cos(np.pi / 8))
    assert np.cos(angle) == pytest.approx(-1.0)
    assert np.sin(angle) == pytest.approx(0.0, abs=1e-12)
    low, high = polar.dimension_ranges[1]
    assert low <= angle < high


def test_single_bin_posterior_returns_that_bin_center(polar):
    posterior = np.eye(polar.n_bins)
    means = mean_position(polar, posterior)
    np.testing.assert_allclose(means[:, 0], polar.bin_centers[:, 0], rtol=1e-12)
    np.testing.assert_allclose(
        np.exp(1j * means[:, 1]), np.exp(1j * polar.bin_centers[:, 1]), atol=1e-12
    )
    low, high = polar.dimension_ranges[1]
    assert np.all((means[:, 1] >= low) & (means[:, 1] < high))


def test_mean_at_the_animal_has_no_direction(polar):
    # Equal mass around a full ring cancels to rounding-level residue, which
    # is not a direction.
    ring = np.isclose(polar.bin_centers[:, 0], 7.5)
    posterior = np.where(ring, 1.0 / ring.sum(), 0.0)[None, :]
    distance, angle = mean_position(polar, posterior)[0]
    assert distance == 0.0
    assert np.isnan(angle)


def test_missing_posterior_rows_stay_missing(polar):
    posterior = np.vstack([_seam_posterior(polar), np.full((1, polar.n_bins), np.nan)])
    means = mean_position(polar, posterior)
    assert np.all(np.isfinite(means[0]))
    assert np.all(np.isnan(means[1]))


def test_decoding_result_reports_polar_means(polar):
    posterior = np.vstack([_seam_posterior(polar), np.eye(polar.n_bins)[:2]])
    result = DecodingResult(posterior=posterior, env=polar)
    np.testing.assert_allclose(result.mean_position, mean_position(polar, posterior))
    table = result.to_dataframe()
    assert len(table) == 3
    assert np.isfinite(table.filter(like="mean").to_numpy()).all()


def test_streamed_summary_matches_full_decode(polar):
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.5, 20.0, size=(5, polar.n_bins))
    counts = rng.poisson(rates[:, :1].T * 0.2, size=(40, 5))
    full = decode_position(polar, counts, rates, dt=0.2)
    summary = decode_position_summary(polar, counts, rates, dt=0.2, time_chunk=7)
    np.testing.assert_allclose(
        summary.mean_position, mean_position(polar, full.posterior), atol=1e-10
    )


def test_cartesian_mean_is_unchanged():
    env = Environment.from_samples(
        np.random.default_rng(1).uniform(0, 10, size=(500, 2)), bin_size=2.0
    )
    posterior = np.random.default_rng(2).dirichlet(np.ones(env.n_bins), size=4)
    np.testing.assert_array_equal(
        mean_position(env, posterior), posterior @ env.bin_centers
    )


def _polar_counts(env: Environment) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    rates = rng.uniform(0.5, 20.0, size=(5, env.n_bins))
    return rng.poisson(rates[:, :1].T * 0.2, size=(6, 5)), rates


_POLAR_COLUMNS = ["map_distance", "map_angle", "mean_distance", "mean_angle"]


def test_full_decode_exports_name_polar_coordinates(polar):
    counts, rates = _polar_counts(polar)
    result = decode_position(polar, counts, rates, dt=0.2)
    table = result.to_dataframe()
    assert [c for c in table.columns if c.startswith(("map_", "mean_"))] == [
        "map_bin",
        *_POLAR_COLUMNS,
    ]
    np.testing.assert_array_equal(table["map_angle"], result.map_position[:, 1])
    np.testing.assert_array_equal(table["mean_distance"], result.mean_position[:, 0])


def test_summary_exports_name_polar_coordinates(polar):
    import matplotlib

    matplotlib.use("Agg")
    counts, rates = _polar_counts(polar)
    summary = decode_position_summary(polar, counts, rates, dt=0.2)
    table = summary.to_dataframe()
    assert set(_POLAR_COLUMNS) <= set(table.columns)
    assert not {"map_x", "map_y", "mean_x", "mean_y"} & set(table.columns)
    np.testing.assert_array_equal(table["map_angle"], summary.map_position[:, 1])
    labels = [line.get_label() for line in summary.plot(quantity="map").get_lines()]
    assert labels == ["map_distance", "map_angle"]
    xr = pytest.importorskip("xarray")
    dataset = summary.to_xarray()
    assert isinstance(dataset, xr.Dataset)
    assert set(_POLAR_COLUMNS) <= set(dataset.data_vars)
