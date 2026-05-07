"""Parity tests between legacy encoding APIs and the new pipeline.

These tests run each legacy ``compute_*`` function and the corresponding
new ``compute_*_rate`` function on shared synthetic data and check whether
the firing-rate outputs agree numerically. They were written to gate the
M6 milestone in ``docs/plans/encoding-cleanup`` — shimming the legacy
modules to delegate to the new pipeline would have removed ~2000 lines of
duplicated code.

**M6 is on hold.** As of 2026-05-07 every pair below mismatches with
relative differences of 50%+; the legacy modules do not produce the same
numerical output as the new ``compute_*_rate`` functions. Replacing the
legacy bodies with shims would silently change user-facing outputs.
Concrete observed gaps:

* ``compute_head_direction_tuning_curve`` returns ``(firing_rate,
  bin_centers)``; the test unpack order is correct, but the legacy
  bin-center convention may differ from the new directional API.
* ``compute_place_field`` ``"binned"`` and ``"gaussian_kde"`` paths
  return values an order of magnitude smaller than ``compute_spatial_rate``
  — different occupancy normalization or smoothing order.
* ``compute_object_vector_tuning`` returns an ``ObjectVectorMetrics``
  with a different attribute layout than ``EgocentricRateResult``.
* ``compute_spatial_view_field`` returns a ``SpatialViewFieldResult``
  with a different attribute layout than ``ViewRateResult``.

The tests are marked ``xfail`` rather than deleted so they document the
gap and would automatically flip to passing if the underlying pipelines
are aligned in a future change. Tolerances are tight on purpose
(``rtol=1e-6, atol=1e-8``) so any near-fix shows up clearly.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def trajectory() -> dict[str, np.ndarray]:
    """A simple 2D random walk and one head-direction-correlated spike train."""
    rng = np.random.default_rng(0)
    n_samples = 2000
    times = np.linspace(0.0, 200.0, n_samples)
    positions = np.cumsum(rng.normal(0.0, 0.5, size=(n_samples, 2)), axis=0) + 50.0
    positions = np.clip(positions, 0.0, 100.0)
    headings = np.arctan2(np.gradient(positions[:, 1]), np.gradient(positions[:, 0]))
    spike_times = np.sort(rng.uniform(times[0], times[-1], size=300))
    return {
        "times": times,
        "positions": positions,
        "headings": headings,
        "spike_times": spike_times,
    }


@pytest.fixture
def env(trajectory: dict[str, np.ndarray]) -> Environment:
    return Environment.from_samples(trajectory["positions"], bin_size=5.0)


def _equal_with_nan(
    a: np.ndarray, b: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-8
) -> None:
    """Assert two arrays match where both are finite, and NaN-align elsewhere."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    np.testing.assert_array_equal(nan_a, nan_b)
    finite = ~nan_a
    np.testing.assert_allclose(a[finite], b[finite], rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Place field parity
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="Legacy compute_place_field uses a different occupancy normalization "
    "than the new compute_spatial_rate; firing-rate scales differ by ~1.5-25x. "
    "See module docstring."
)
@pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "binned"])
def test_place_field_parity(
    env: Environment, trajectory: dict[str, np.ndarray], method: str
) -> None:
    """Legacy compute_place_field matches new compute_spatial_rate.firing_rate."""
    from neurospatial.encoding.place import compute_place_field
    from neurospatial.encoding.spatial import compute_spatial_rate

    legacy = compute_place_field(
        env,
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["positions"],
        smoothing_method=method,
        bandwidth=5.0,
    )
    result = compute_spatial_rate(
        env,
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["positions"],
        smoothing_method=method,
        bandwidth=5.0,
    )
    _equal_with_nan(legacy, np.asarray(result.firing_rate))


# --------------------------------------------------------------------------- #
# Head-direction tuning curve parity
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="Legacy head_direction tuning curve returns bin-center array first "
    "and uses different bin centers than directional.compute_directional_rate."
)
def test_head_direction_parity(trajectory: dict[str, np.ndarray]) -> None:
    """Legacy compute_head_direction_tuning_curve matches the new directional API."""
    from neurospatial.encoding.directional import compute_directional_rate
    from neurospatial.encoding.head_direction import compute_head_direction_tuning_curve

    bin_size = np.pi / 30
    legacy_rate, _legacy_bin_centers = compute_head_direction_tuning_curve(
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["headings"],
        bin_size=bin_size,
    )
    result = compute_directional_rate(
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["headings"],
        bin_size=bin_size,
    )
    _equal_with_nan(legacy_rate, np.asarray(result.firing_rate))


# --------------------------------------------------------------------------- #
# Object-vector tuning parity
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="Legacy compute_object_vector_tuning has a different signature "
    "(no env arg, returns ObjectVectorMetrics) and a different polar binning."
)
def test_object_vector_parity(
    env: Environment, trajectory: dict[str, np.ndarray]
) -> None:
    """Legacy compute_object_vector_tuning matches the new egocentric API."""
    from neurospatial.encoding.egocentric import compute_egocentric_rate
    from neurospatial.encoding.object_vector import compute_object_vector_tuning

    object_positions = np.array([[50.0, 50.0]])
    legacy_result = compute_object_vector_tuning(
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["positions"],
        trajectory["headings"],
        object_positions,
        distance_range=(0.0, 50.0),
        n_distance_bins=10,
        n_direction_bins=12,
    )
    new = compute_egocentric_rate(
        env,
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["positions"],
        trajectory["headings"],
        object_positions,
        distance_range=(0.0, 50.0),
        n_distance_bins=10,
        n_direction_bins=12,
        smoothing_method="binned",
    )
    _equal_with_nan(
        np.asarray(legacy_result.firing_rate),
        np.asarray(new.firing_rate),
    )


# --------------------------------------------------------------------------- #
# Spatial-view field parity
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="Legacy compute_spatial_view_field returns a SpatialViewFieldResult "
    "with a different attribute layout (no .firing_rate) than ViewRateResult."
)
def test_spatial_view_parity(
    env: Environment, trajectory: dict[str, np.ndarray]
) -> None:
    """Legacy compute_spatial_view_field matches the new view API."""
    from neurospatial.encoding.spatial_view import compute_spatial_view_field
    from neurospatial.encoding.view import compute_view_rate

    legacy_result = compute_spatial_view_field(
        env,
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["positions"],
        trajectory["headings"],
        view_distance=15.0,
        smoothing_method="binned",
        bandwidth=5.0,
    )
    new = compute_view_rate(
        env,
        trajectory["spike_times"],
        trajectory["times"],
        trajectory["positions"],
        trajectory["headings"],
        view_distance=15.0,
        smoothing_method="binned",
        bandwidth=5.0,
    )
    _equal_with_nan(
        np.asarray(legacy_result.firing_rate),
        np.asarray(new.firing_rate),
    )
