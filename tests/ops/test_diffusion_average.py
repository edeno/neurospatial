"""Tests for the public intensive-field averaging mode (``mode="average"``).

Covers the row-stochastic ``H`` smoother exposed on ``Environment.compute_kernel``
/ ``Environment.smooth``, and its two internal consumers routed through the
masked (valid-bin-normalized) average: ``smooth_rate_map(method="binned")`` and
``resample_field(method="diffuse")``. The key property is *volume-unbiased*
averaging of an intensive field on a non-uniform mass matrix ``M`` (polar), where
``density`` (``H·M⁻¹``) would introduce a cell-volume bias.

Deterministic construction only.
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding._smoothing import _binned
from neurospatial.environment.polar import EgocentricPolarEnvironment
from neurospatial.ops.binning import (
    TieBreakStrategy,
    map_points_to_bins,
    resample_field,
)
from neurospatial.ops.smoothing import apply_kernel


@pytest.fixture(scope="module")
def polar_env() -> EgocentricPolarEnvironment:
    """Full-circle polar env (non-uniform M = annular-sector area)."""
    return EgocentricPolarEnvironment.create(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=2.0,
        angle_bin_size=np.pi / 12,
    )


@pytest.fixture(scope="module")
def uniform_env() -> Environment:
    """Uniform 2D grid (M is constant)."""
    edges = np.linspace(0.0, 20.0, 21)
    return Environment.from_grid_mask(
        active_mask=np.ones((20, 20), dtype=bool), grid_edges=(edges, edges)
    )


def test_average_is_row_stochastic(uniform_env, polar_env):
    """compute_kernel(mode="average") rows sum to 1 on uniform and polar envs."""
    for env in (uniform_env, polar_env):
        kernel = env.compute_kernel(bandwidth=4.0, mode="average")
        np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1e-9)


def test_average_differs_from_density_nonuniform(polar_env):
    """On a non-uniform-M env the average (H) and density (H·M⁻¹) kernels differ."""
    k_avg = polar_env.compute_kernel(bandwidth=5.0, mode="average")
    k_den = polar_env.compute_kernel(bandwidth=5.0, mode="density")
    assert not np.allclose(k_avg, k_den), (
        "average and density kernels must differ on non-uniform M"
    )


def test_average_smooths_intensive_field(polar_env):
    """A flat (constant) rate map smooths to itself under average on non-uniform M
    (row-stochastic ⇒ constant preserved), whereas density does not."""
    field = np.full(polar_env.n_bins, 7.0)
    smoothed_avg = polar_env.smooth(field, bandwidth=5.0, mode="average")
    np.testing.assert_allclose(smoothed_avg, 7.0, atol=1e-9)

    smoothed_den = polar_env.smooth(field, bandwidth=5.0, mode="density")
    assert not np.allclose(smoothed_den, 7.0, atol=1e-6), (
        "density mode carries a volume bias on non-uniform M (not constant-preserving)"
    )


def _masked_average(kernel, rate, valid):
    """Reference Nadaraya-Watson masked average with an explicit kernel."""
    rate_filled = np.where(valid, rate, 0.0)
    num = kernel @ rate_filled
    den = kernel @ valid.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, num / den, np.nan)


def test_binned_unbiased_on_nonuniform_M(polar_env, uniform_env):
    """binned smoothing equals the row-stochastic H masked-average (volume-unbiased),
    and on non-uniform M differs from the volume-biased density-path result; the
    uniform-grid result is the same either way."""
    for env, expect_bias in ((polar_env, True), (uniform_env, False)):
        n = env.n_bins
        # An intensive rate with a few invalid (missing) bins.
        invalid = np.zeros(n, dtype=bool)
        invalid[:: max(1, n // 7)] = True
        valid = ~invalid
        rate = np.linspace(1.0, 5.0, n)
        rate[invalid] = np.nan

        # Drive _binned via spike_counts/occupancy so raw_rate == rate on valid
        # bins and NaN on invalid (occupancy == 0 there).
        occupancy = valid.astype(np.float64)
        spike_counts = np.where(valid, rate, 0.0)
        got = _binned(env, spike_counts, occupancy, bandwidth=4.0, min_occupancy=0.0)

        k_avg = env.compute_kernel(bandwidth=4.0, mode="average")
        expected_avg = _masked_average(k_avg, np.where(valid, rate, 0.0), valid)
        np.testing.assert_allclose(got[valid], expected_avg[valid], atol=1e-9)

        # The density-path masked average is volume-biased on non-uniform M.
        k_den = env.compute_kernel(bandwidth=4.0, mode="density")
        density_path = _masked_average(k_den, np.where(valid, rate, 0.0), valid)
        if expect_bias:
            assert not np.allclose(got[valid], density_path[valid], atol=1e-6), (
                "binned must NOT match the volume-biased density path on non-uniform M"
            )
        else:
            np.testing.assert_allclose(got[valid], density_path[valid], atol=1e-9)


def _partial_coverage_pair():
    """Source covering x in [0, 10]; destination extending to x = 25 so a band of
    destination bins is genuinely outside the source (``outside_source``)."""
    src = Environment.from_samples(
        np.array([[i, j] for i in np.arange(0, 11) for j in range(3)], dtype=float),
        bin_size=1.0,
    )
    dst = Environment.from_samples(
        np.array([[i, j] for i in np.arange(0, 26) for j in range(3)], dtype=float),
        bin_size=1.0,
    )
    idx = np.asarray(
        map_points_to_bins(
            dst.bin_centers, src, tie_break=TieBreakStrategy.LOWEST_INDEX
        )
    )
    outside = idx < 0
    return src, dst, idx, outside


def test_resample_diffuse_masked_not_biased_down():
    """Covered destination bins adjacent to the uncovered region are NOT pulled
    toward zero (the masked H-average normalizes by valid-neighbour weight), unlike
    the old zero-fill-then-single-smooth; outside-source bins stay NaN."""
    src, dst, idx, outside = _partial_coverage_pair()
    assert outside.any(), "fixture must produce outside-source bins"

    field = np.full(src.n_bins, 5.0)  # constant intensive field
    new = resample_field(field, src, dst, method="diffuse", bandwidth=2.0)

    # Reconstruct the OLD behavior (zero-fill + single mass-conserving smooth).
    resampled = field[np.where(outside, 0, idx)].astype(float)
    resampled[outside] = np.nan
    k_transition = dst.compute_kernel(bandwidth=2.0, mode="transition")
    old = apply_kernel(np.where(outside, 0.0, resampled), k_transition, mode="forward")
    old[outside] = np.nan

    covered = ~outside
    xd = dst.bin_centers[:, 0]
    boundary_x = xd[outside].min()
    near = covered & (xd >= boundary_x - 3.0) & (xd < boundary_x)
    assert near.any()

    # New masked average preserves the constant on every covered bin.
    np.testing.assert_allclose(new[covered], 5.0, atol=1e-6)
    # The old path is measurably biased DOWN near the boundary; new is not.
    assert np.nanmin(old[near]) < 4.9, "old zero-fill should bias down near boundary"
    assert np.all(new[near] >= old[near] - 1e-9), "new must not be more biased than old"
    # Structurally-outside bins remain NaN.
    assert np.all(np.isnan(new[outside]))


def test_resample_diffuse_source_nan_no_propagation():
    """A NaN in the source field does not propagate across reachable bins; a
    source-NaN bin with valid neighbours within the bandwidth is interpolated, and
    a bin with no valid neighbour (den == 0) stays NaN."""
    src, dst, _idx, outside = _partial_coverage_pair()

    field = np.full(src.n_bins, 5.0)
    # Put a NaN at one interior source bin (has valid neighbours).
    mid = int(np.argmin(np.abs(src.bin_centers[:, 0] - 5.0)))
    field[mid] = np.nan

    out = resample_field(field, src, dst, method="diffuse", bandwidth=2.0)

    covered = ~outside
    # No propagation: the source NaN does not wipe out reachable covered bins.
    assert np.all(np.isfinite(out[covered])), (
        "a source NaN must not poison every reachable bin"
    )
    # Covered bins stay at the constant value (the NaN bin is interpolated to ~5).
    np.testing.assert_allclose(out[covered], 5.0, atol=1e-6)

    # A destination fully isolated from any valid bin (no valid neighbour within
    # bandwidth) stays NaN — here, the structurally-outside band.
    assert np.all(np.isnan(out[outside]))
