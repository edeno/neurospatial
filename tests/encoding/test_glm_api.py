"""Public-API tests for ``method="glm"`` on ``compute_spatial_rate(s)``.

Phase-3 of the MRF-GAM estimator wires the phase-2 penalized-Poisson fit into
``compute_spatial_rate`` / ``compute_spatial_rates``: the unit-major <-> bin-major
transpose, the full active-bin assembly, the GAM result fields, the method-param
validation, and the degenerate-data dispatch.

These tests exercise the public surface (not the fit internals, which live in
``test_glm_fit.py``): the headline finiteness contrast, the result-field shapes /
None-ness, the validation branches, the boundary orientation / dtype / backend
contracts, and the degenerate rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.spatial import (
    _compute_glm_spatial_rates,
    compute_spatial_rate,
    compute_spatial_rates,
)


# ---------------------------------------------------------------------------
# Session builders: turn a set of active-bin targets + place-field centers into
# (times, positions, spike_times) that flow through the public binning layer.
# Positions are placed exactly on active bin centers so binning is deterministic.
# ---------------------------------------------------------------------------
def _grid_session(
    env: Environment,
    centers: list[tuple[float, ...]],
    *,
    bin_indices: np.ndarray | None = None,
    peak_rate: float = 25.0,
    sigma: float = 4.0,
    n_repeats: int = 40,
    dt: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build a session visiting the given active bins with per-unit place fields.

    ``bin_indices`` (default: all active bins) selects which active bins the
    animal visits; the trajectory tiles their centers ``n_repeats`` times, giving
    (approximately) uniform occupancy over exactly those bins. One sorted
    spike-time array is drawn per center from a Gaussian place field.
    """
    rng = np.random.default_rng(seed)
    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    if bin_indices is None:
        bin_indices = np.arange(env.n_bins)
    visited = bin_centers[np.asarray(bin_indices)]
    positions = np.tile(visited, (n_repeats, 1))
    positions = positions[rng.permutation(positions.shape[0])]
    n_samples = positions.shape[0]
    times = np.arange(n_samples, dtype=np.float64) * dt

    spike_times: list[np.ndarray] = []
    for center in centers:
        dist2 = np.sum((positions - np.asarray(center, dtype=np.float64)) ** 2, axis=1)
        rate = peak_rate * np.exp(-dist2 / (2.0 * sigma**2))
        counts = rng.poisson(rate * dt)
        per_unit: list[np.ndarray] = []
        for i, c in enumerate(counts):
            if c:
                per_unit.append(times[i] + rng.uniform(0.0, dt, size=int(c)))
        spike_times.append(
            np.sort(np.concatenate(per_unit)) if per_unit else np.array([], float)
        )
    return times, positions, spike_times


def _component_bins(env: Environment) -> list[np.ndarray]:
    """Active-bin indices grouped by W-component (via the diffusion geometry)."""
    labels = np.asarray(env._diffusion_geometry.labels)
    return [np.flatnonzero(labels == c) for c in np.unique(labels)]


# ---------------------------------------------------------------------------
# Headline (spec §10): finite where the ratio estimator NaNs.
# ---------------------------------------------------------------------------
def test_glm_finite_where_ratio_nans(open_field_env: Environment) -> None:
    """Low-occupancy bins: glm all-finite; diffusion_kde NaN in the same bins.

    ``min_occupancy`` (a ratio-only param) masks low-occupancy bins to NaN in the
    ratio estimator. glm has no such knob -- occupancy enters as a log-offset -- so
    every bin gets a finite rate.
    """
    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=1)

    occupancy_ratio = compute_spatial_rate(
        env, spike_times[0], times, positions, method="diffusion_kde"
    ).occupancy
    occ = np.asarray(occupancy_ratio, dtype=np.float64)
    threshold = float(np.median(occ[occ > 0]))

    ratio = compute_spatial_rate(
        env,
        spike_times[0],
        times,
        positions,
        method="diffusion_kde",
        min_occupancy=threshold,
    )
    glm = compute_spatial_rate(env, spike_times[0], times, positions, method="glm")

    ratio_rate = np.asarray(ratio.firing_rate)
    glm_rate = np.asarray(glm.firing_rate)

    masked = occ < threshold
    assert masked.any(), "test needs at least one low-occupancy bin"
    # diffusion_kde masks EVERY sub-threshold bin, so .all() is the true contract
    # (a regression that masked only some bins would slip past .any()).
    assert np.isnan(ratio_rate[masked]).all(), "diffusion_kde should NaN low-occ bins"
    assert np.all(np.isfinite(glm_rate)), "glm must be finite everywhere"


# ---------------------------------------------------------------------------
# Result fields.
# ---------------------------------------------------------------------------
def test_glm_to_xarray_netcdf_roundtrip(open_field_env: Environment, tmp_path) -> None:
    """A glm result's to_xarray() must be NetCDF-serializable.

    ``bandwidth`` is ``None`` for glm; NetCDF attrs cannot hold ``None``, so it
    must be omitted from the dataset attrs (not stored as null). Ratio results
    keep their float ``bandwidth`` attr.
    """
    import importlib.util

    xr = pytest.importorskip("xarray")
    # to_netcdf needs a backend engine; skip if none is installed.
    if not any(
        importlib.util.find_spec(eng) for eng in ("netCDF4", "h5netcdf", "scipy")
    ):
        pytest.skip("no NetCDF engine available")

    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(4.0, 4.0), (12.0, 12.0)])
    glm = compute_spatial_rates(env, spike_times, times, positions, method="glm")

    ds = glm.to_xarray()
    assert "bandwidth" not in ds.attrs  # omitted (None is not NetCDF-serializable)
    assert ds.attrs["method"] == "glm"

    path = tmp_path / "glm_rates.nc"
    ds.to_netcdf(path)  # must not raise
    with xr.open_dataset(path) as reopened:
        assert reopened.attrs["method"] == "glm"
        assert "bandwidth" not in reopened.attrs

    # Ratio results still carry a float bandwidth attr (unchanged).
    ratio = compute_spatial_rates(
        env, spike_times, times, positions, method="diffusion_kde"
    )
    assert float(ratio.to_xarray().attrs["bandwidth"]) == 5.0


def test_result_invariant_enforced(open_field_env: Environment) -> None:
    """The None-iff-glm invariant is enforced at construction on both classes.

    A ratio result carrying a stray GAM field, a glm result missing a required
    GAM field, and a rank/coefficients shape mismatch each raise ValueError; the
    two legitimate shapes (populated glm, all-None ratio) construct fine.
    """
    from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

    env = open_field_env
    n = env.n_bins
    occ = np.zeros(n)
    rates1 = np.zeros(n)
    ratesN = np.zeros((2, n))

    # Legitimate ratio construction (all GAM None, float bandwidth): no raise.
    SpatialRateResult(
        firing_rate=rates1, occupancy=occ, env=env, method="binned", bandwidth=5.0
    )
    SpatialRatesResult(
        firing_rates=ratesN, occupancy=occ, env=env, method="binned", bandwidth=5.0
    )

    # Ratio result with a stray GAM field -> raise.
    with pytest.raises(ValueError, match="GAM"):
        SpatialRatesResult(
            firing_rates=ratesN,
            occupancy=occ,
            env=env,
            method="binned",
            bandwidth=5.0,
            coefficients=np.zeros((3, 2)),
        )
    with pytest.raises(ValueError, match="GAM"):
        SpatialRateResult(
            firing_rate=rates1,
            occupancy=occ,
            env=env,
            method="binned",
            bandwidth=5.0,
            rank=3,
        )

    # glm result missing a required GAM field -> raise.
    with pytest.raises(ValueError, match="glm"):
        SpatialRatesResult(
            firing_rates=ratesN,
            occupancy=occ,
            env=env,
            method="glm",
            bandwidth=None,
            coefficients=np.zeros((3, 2)),  # rank/etc missing
        )
    # glm result with a non-None bandwidth -> raise.
    with pytest.raises(ValueError, match="bandwidth"):
        SpatialRatesResult(
            firing_rates=ratesN,
            occupancy=occ,
            env=env,
            method="glm",
            bandwidth=5.0,
            coefficients=np.zeros((3, 2)),
            penalty_weights=np.zeros(3),
            rank=3,
            deviance=np.zeros(2),
            converged=True,
            n_iter=1,
        )
    # glm result whose coefficients rows != rank -> raise.
    with pytest.raises(ValueError, match=r"rank|shape"):
        SpatialRatesResult(
            firing_rates=ratesN,
            occupancy=occ,
            env=env,
            method="glm",
            bandwidth=None,
            coefficients=np.zeros((3, 2)),
            penalty_weights=np.zeros(4),
            rank=4,
            deviance=np.zeros(2),
            converged=True,
            n_iter=1,
        )

    def _valid_glm_kwargs(rank: int, n_units: int) -> dict:
        return {
            "coefficients": np.zeros((rank, n_units)),
            "penalty_weights": np.zeros(rank),
            "rank": rank,
            "deviance": np.zeros(n_units),
            "converged": True,
            "n_iter": 1,
        }

    # glm result whose per-unit shapes disagree with n_units (would IndexError on
    # rates[1]) -> raise at construction. Each mis-shape independently rejected.
    for bad in (
        {"coefficients": np.zeros((3, 1))},  # (rank, 1) but 2 units
        {"deviance": np.zeros(1)},  # (1,) but 2 units
        {"penalty_weights": np.zeros(2)},  # (rank-1,)
    ):
        kwargs = _valid_glm_kwargs(3, 2)
        kwargs.update(bad)
        with pytest.raises(ValueError, match="shape"):
            SpatialRatesResult(
                firing_rates=ratesN,
                occupancy=occ,
                env=env,
                method="glm",
                bandwidth=None,
                **kwargs,
            )

    # Unknown method must NOT be silently treated as a ratio method -> raise.
    with pytest.raises(ValueError, match="method"):
        SpatialRatesResult(
            firing_rates=ratesN, occupancy=occ, env=env, method="glmm", bandwidth=5.0
        )

    # A real glm result from the compute path constructs cleanly (guard passes).
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=99)
    glm = compute_spatial_rates(env, spike_times, times, positions, method="glm")
    assert glm.method == "glm"
    # ...and its per-unit slice re-validates through the singular guard.
    _ = glm[0]


def test_glm_result_fields(open_field_env: Environment) -> None:
    env = open_field_env
    times, positions, spike_times = _grid_session(
        env, [(4.0, 4.0), (12.0, 12.0), (8.0, 2.0)], seed=2
    )
    result = compute_spatial_rates(env, spike_times, times, positions, method="glm")

    n_units = len(spike_times)
    rank = result.rank
    assert isinstance(rank, int) and rank >= 1
    assert result.method == "glm"
    assert result.bandwidth is None

    assert np.asarray(result.coefficients).shape == (rank, n_units)
    assert np.asarray(result.penalty_weights).shape == (rank,)
    assert np.asarray(result.deviance).shape == (n_units,)
    assert np.isscalar(result.penalty) or result.penalty is None
    # A well-sampled population fit genuinely converges (guards against the field
    # being stamped None/False -- bool(None) would pass an isinstance-bool check).
    assert result.converged is True
    assert np.isscalar(result.n_iter)
    assert result.reml_objective is None or np.isscalar(result.reml_objective)
    # REML ran on a well-sampled multi-unit population (r > 0, spikes present).
    assert result.reml_objective is not None
    assert result.penalty is not None and float(result.penalty) >= 0.0


def test_ratio_result_gam_fields_none(open_field_env: Environment) -> None:
    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=3)
    result = compute_spatial_rates(
        env, spike_times, times, positions, method="diffusion_kde"
    )
    assert result.method == "diffusion_kde"
    assert isinstance(result.bandwidth, float)
    for name in (
        "coefficients",
        "penalty",
        "penalty_weights",
        "rank",
        "deviance",
        "converged",
        "n_iter",
        "reml_objective",
    ):
        assert getattr(result, name) is None, f"{name} must be None for ratio methods"


def test_indexing_stamps_and_slices(open_field_env: Environment) -> None:
    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0), (8.0, 2.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=4)
    unit_ids = np.array(["a", "b", "c"])
    result = compute_spatial_rates(
        env, spike_times, times, positions, method="glm", unit_ids=unit_ids
    )
    rank = result.rank
    for i in range(len(result)):
        child = result[i]
        assert child.unit_id == unit_ids[i]
        assert child.method == "glm"
        assert child.bandwidth is None
        assert np.asarray(child.coefficients).shape == (rank,)
        assert np.ndim(child.deviance) == 0
        # scalar / shared fields carried through unchanged
        assert np.asarray(child.penalty_weights).shape == (rank,)
        assert child.rank == rank
        np.testing.assert_allclose(
            np.asarray(child.coefficients),
            np.asarray(result.coefficients)[:, i],
        )
        np.testing.assert_allclose(
            float(child.deviance), float(np.asarray(result.deviance)[i])
        )


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def test_mutual_exclusivity(open_field_env: Environment) -> None:
    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=5)
    st = spike_times[0]

    with pytest.raises(ValueError, match="bandwidth"):
        compute_spatial_rate(env, st, times, positions, method="glm", bandwidth=5.0)
    with pytest.raises(ValueError, match="min_occupancy"):
        compute_spatial_rate(env, st, times, positions, method="glm", min_occupancy=1.0)
    with pytest.raises(ValueError, match="fill_value"):
        compute_spatial_rate(env, st, times, positions, method="glm", fill_value=0.0)
    with pytest.raises(ValueError, match="penalty"):
        compute_spatial_rate(env, st, times, positions, method="binned", penalty=1.0)
    with pytest.raises(ValueError, match="rank"):
        compute_spatial_rate(env, st, times, positions, method="diffusion_kde", rank=10)

    # bare glm (no ratio kwargs) must NOT raise on the resolved default bandwidth.
    result = compute_spatial_rate(env, st, times, positions, method="glm")
    assert result.method == "glm"


def test_value_domain(
    open_field_env: Environment, two_component_env: Environment
) -> None:
    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=6)
    st = spike_times[0]

    # penalty domain
    for bad in (True, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="penalty"):
            compute_spatial_rate(env, st, times, positions, method="glm", penalty=bad)
    # rank domain
    for bad in (True, 2.5, 0):
        with pytest.raises(ValueError, match="rank"):
            compute_spatial_rate(env, st, times, positions, method="glm", rank=bad)

    # rank clamps DOWN (> n_live_bins): huge rank -> r_eff == n_live_bins, no raise.
    big = compute_spatial_rate(env, st, times, positions, method="glm", rank=100_000)
    n_live = int(np.asarray(env._mrf_basis(np.ones(env.n_bins), rank=None).B).shape[0])
    assert big.rank == n_live

    # rank clamps UP (< n_live_components): a 2-live-component env, rank=1 -> r_eff == 2.
    two = two_component_env
    comp_bins = _component_bins(two)
    assert len(comp_bins) == 2
    t2, p2, s2 = _grid_session(two, [(2.0, 2.0), (10.0, 2.0)], seed=7, sigma=3.0)
    small = compute_spatial_rate(two, s2[0], t2, p2, method="glm", rank=1)
    assert small.rank == 2


def test_value_domain_overflow(open_field_env: Environment) -> None:
    """Huge ints must not leak OverflowError: penalty rejects cleanly, rank clamps."""
    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=6)
    st = spike_times[0]

    # penalty: too large to represent as a finite float -> clean ValueError (not
    # a raw OverflowError). penalty is NOT clamped.
    with pytest.raises(ValueError, match="penalty"):
        compute_spatial_rate(env, st, times, positions, method="glm", penalty=10**1000)

    # rank: magnitude is CLAMPED, never rejected -- a huge int must not raise and
    # must clamp to the effective rank (n_live_bins here).
    n_live = int(np.asarray(env._mrf_basis(np.ones(env.n_bins), rank=None).B).shape[0])
    big = compute_spatial_rate(env, st, times, positions, method="glm", rank=10**1000)
    assert big.rank == n_live


# ---------------------------------------------------------------------------
# Boundary orientation / dtype / backend.
# ---------------------------------------------------------------------------
def test_glm_orientation(open_field_env: Environment) -> None:
    """firing_rates is (n_units, n_bins) with row k == unit k's map.

    Two units with distinct field centers: each unit's peak bin must land near
    its own center. A transposed assembly would swap the maps and fail this.
    """
    env = open_field_env
    centers = [(2.0, 2.0), (14.0, 14.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=8, sigma=3.0)
    result = compute_spatial_rates(env, spike_times, times, positions, method="glm")

    assert np.asarray(result.firing_rates).shape == (2, env.n_bins)
    peaks = result.peak_locations()  # (n_units, n_dims)
    for k, center in enumerate(centers):
        dist = np.linalg.norm(peaks[k] - np.asarray(center))
        assert dist < 5.0, f"unit {k} peak {peaks[k]} far from center {center}"

    # singular firing_rate is (n_bins,)
    single = compute_spatial_rate(env, spike_times[0], times, positions, method="glm")
    assert np.asarray(single.firing_rate).shape == (env.n_bins,)


def test_glm_dtype(open_field_env: Environment) -> None:
    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=9)

    r64 = compute_spatial_rates(
        env, spike_times, times, positions, method="glm", dtype=np.float64
    )
    r32 = compute_spatial_rates(
        env, spike_times, times, positions, method="glm", dtype=np.float32
    )
    # dtype governs the (n_units, n_bins) rate-map STORAGE only.
    assert np.asarray(r64.firing_rates).dtype == np.float64
    assert np.asarray(r32.firing_rates).dtype == np.float32
    np.testing.assert_allclose(
        np.asarray(r32.firing_rates),
        np.asarray(r64.firing_rates),
        rtol=1e-4,
        atol=1e-4,
    )

    # The GLM diagnostics are the float64 fit result -- they are NOT downcast by
    # dtype (that would lose precision on deviance/coefficients for negligible
    # memory, and make rates[i] inconsistent with compute_spatial_rate, whose
    # diagnostics are always float64). They stay float64 for BOTH dtypes.
    for r in (r32, r64):
        assert np.asarray(r.coefficients).dtype == np.float64
        assert np.asarray(r.penalty_weights).dtype == np.float64
        assert np.asarray(r.deviance).dtype == np.float64
    # And the per-unit slice matches the singular estimator's float64 diagnostics.
    single = compute_spatial_rate(env, spike_times[0], times, positions, method="glm")
    assert (
        np.asarray(r32[0].coefficients).dtype == np.asarray(single.coefficients).dtype
    )


def _jax_available() -> bool:
    from neurospatial.encoding._backend import is_jax_available

    return is_jax_available()


@pytest.mark.skipif(not _jax_available(), reason="JAX not available")
def test_glm_backend_jax_return(open_field_env: Environment) -> None:
    """glm backend='jax' returns the SAME array-type as diffusion_kde backend='jax'.

    The glm fit runs the NumPy core and the assembled rate array is converted to a
    JAX array when the resolved backend is 'jax', matching the ratio path's return
    contract. Values match the backend='numpy' glm result. Both the plural
    (``compute_spatial_rates``) and singular (``compute_spatial_rate``) entry
    points have their own conversion branch, so both are checked.
    """
    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=10)

    # --- plural ---
    ratio_jax = compute_spatial_rates(
        env, spike_times, times, positions, method="diffusion_kde", backend="jax"
    )
    glm_jax = compute_spatial_rates(
        env, spike_times, times, positions, method="glm", backend="jax"
    )
    glm_numpy = compute_spatial_rates(
        env, spike_times, times, positions, method="glm", backend="numpy"
    )
    # Same array type as the ratio path returns for backend="jax" (deleting glm's
    # conversion branch, which returns NumPy, would fail this since ratio is JAX).
    assert type(glm_jax.firing_rates) is type(ratio_jax.firing_rates)
    np.testing.assert_allclose(
        np.asarray(glm_jax.firing_rates),
        np.asarray(glm_numpy.firing_rates),
        rtol=1e-4,
        atol=1e-5,
    )

    # --- singular (its own separate conversion branch) ---
    ratio_single_jax = compute_spatial_rate(
        env, spike_times[0], times, positions, method="diffusion_kde", backend="jax"
    )
    glm_single_jax = compute_spatial_rate(
        env, spike_times[0], times, positions, method="glm", backend="jax"
    )
    glm_single_numpy = compute_spatial_rate(
        env, spike_times[0], times, positions, method="glm", backend="numpy"
    )
    assert type(glm_single_jax.firing_rate) is type(ratio_single_jax.firing_rate)
    np.testing.assert_allclose(
        np.asarray(glm_single_jax.firing_rate),
        np.asarray(glm_single_numpy.firing_rate),
        rtol=1e-4,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Statistical recovery + agreement with the ratio estimator.
# ---------------------------------------------------------------------------
def test_statistical_recovery(open_field_env: Environment) -> None:
    env = open_field_env
    centers = [(3.0, 3.0), (13.0, 13.0), (3.0, 13.0)]
    times, positions, spike_times = _grid_session(
        env, centers, seed=11, sigma=3.0, peak_rate=30.0, n_repeats=60
    )
    result = compute_spatial_rates(env, spike_times, times, positions, method="glm")
    peaks = result.peak_locations()
    for k, center in enumerate(centers):
        assert np.linalg.norm(peaks[k] - np.asarray(center)) < 5.0


def test_agreement_with_ratio(open_field_env: Environment) -> None:
    env = open_field_env
    times, positions, spike_times = _grid_session(
        env, [(8.0, 8.0)], seed=12, sigma=4.0, peak_rate=30.0, n_repeats=60
    )
    ratio = compute_spatial_rate(
        env, spike_times[0], times, positions, method="diffusion_kde", bandwidth=4.0
    )
    glm = compute_spatial_rate(env, spike_times[0], times, positions, method="glm")

    r = np.asarray(ratio.firing_rate)
    g = np.asarray(glm.firing_rate)
    finite = np.isfinite(r) & np.isfinite(g)
    corr = np.corrcoef(r[finite], g[finite])[0, 1]
    assert corr > 0.7, f"glm/ratio correlation too low: {corr:.3f}"
    # Peaks co-locate.
    assert np.linalg.norm(ratio.peak_location() - glm.peak_location()) < 6.0


# ---------------------------------------------------------------------------
# Degenerate-case dispatch (helper level: direct counts/occupancy control).
# ---------------------------------------------------------------------------
def _floor() -> float:
    from neurospatial.encoding._glm import _RATE_FLOOR

    return _RATE_FLOOR


def test_degenerate_no_neurons(open_field_env: Environment) -> None:
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    counts = np.zeros((0, env.n_bins))
    rates, fit = _compute_glm_spatial_rates(
        env,
        counts,
        occ,
        penalty=None,
        rank=None,
        resolved_backend="numpy",
        dtype=np.float64,
    )
    assert np.asarray(rates).shape == (0, env.n_bins)
    assert fit.coefficients.shape == (fit.rank, 0)
    assert fit.deviance.shape == (0,)
    assert fit.penalty is None and fit.reml_objective is None
    assert fit.converged is True


def test_degenerate_zero_occupancy(open_field_env: Environment) -> None:
    env = open_field_env
    occ = np.zeros(env.n_bins)
    counts = np.zeros((2, env.n_bins))
    with pytest.warns(UserWarning, match="occupancy"):
        rates, fit = _compute_glm_spatial_rates(
            env,
            counts,
            occ,
            penalty=None,
            rank=None,
            resolved_backend="numpy",
            dtype=np.float64,
        )
    rates = np.asarray(rates)
    assert rates.shape == (2, env.n_bins)
    assert np.allclose(rates, _floor())
    assert fit.penalty is None and fit.reml_objective is None


def test_degenerate_all_zero_spike_population(open_field_env: Environment) -> None:
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    counts = np.zeros((2, env.n_bins))
    with pytest.warns(UserWarning, match="no unit has any spikes"):
        rates, fit = _compute_glm_spatial_rates(
            env,
            counts,
            occ,
            penalty=None,
            rank=None,
            resolved_backend="numpy",
            dtype=np.float64,
        )
    rates = np.asarray(rates)
    assert np.allclose(rates, _floor())
    assert fit.penalty is None  # auto, no fixed penalty supplied
    assert fit.reml_objective is None
    assert np.all(np.isfinite(fit.deviance))


def test_degenerate_dead_component(two_component_env: Environment) -> None:
    env = two_component_env
    comp_bins = _component_bins(env)
    assert len(comp_bins) == 2
    occ = np.zeros(env.n_bins)
    occ[comp_bins[0]] = 3.0  # only component 0 is live
    rng = np.random.default_rng(0)
    counts = np.zeros((1, env.n_bins))
    counts[0, comp_bins[0]] = rng.poisson(5.0, size=comp_bins[0].size)
    with pytest.warns(UserWarning, match="dead"):
        rates, _fit = _compute_glm_spatial_rates(
            env,
            counts,
            occ,
            penalty=None,
            rank=None,
            resolved_backend="numpy",
            dtype=np.float64,
        )
    rates = np.asarray(rates)[0]
    # dead-component bins floor; live bins are finite and above the floor somewhere
    assert np.allclose(rates[comp_bins[1]], _floor())
    assert np.all(np.isfinite(rates))
    assert rates[comp_bins[0]].max() > _floor()


def test_degenerate_zero_spike_neuron(open_field_env: Environment) -> None:
    env = open_field_env
    occ = np.full(env.n_bins, 3.0)
    bin_centers = np.asarray(env.bin_centers)
    dist2 = np.sum((bin_centers - np.array([8.0, 8.0])) ** 2, axis=1)
    rate = 20.0 * np.exp(-dist2 / (2.0 * 3.0**2))
    rng = np.random.default_rng(1)
    counts = np.zeros((2, env.n_bins))
    counts[0] = rng.poisson(rate * occ)  # informative
    # counts[1] stays all-zero (zero-spike neuron; population has an informative unit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected for this row
        rates, _fit = _compute_glm_spatial_rates(
            env,
            counts,
            occ,
            penalty=None,
            rank=None,
            resolved_backend="numpy",
            dtype=np.float64,
        )
    rates = np.asarray(rates)
    # The zero-spike unit floors to _RATE_FLOOR EVERYWHERE (its intercept is driven
    # to the eta clip); pin that exactly rather than a loose upper bound.
    assert np.allclose(rates[1], _floor())
    # The informative unit is a real field, not floored.
    assert rates[0].max() > 1.0
    assert np.all(np.isfinite(rates))


def test_all_zero_spike_fixed_penalty_public(open_field_env: Environment) -> None:
    """Fixed-penalty contract holds through the public API with no data."""
    env = open_field_env
    times, positions, _ = _grid_session(env, [(8.0, 8.0)], seed=13)
    no_spikes = np.array([], dtype=np.float64)
    with pytest.warns(UserWarning, match="no unit has any spikes"):
        result = compute_spatial_rate(
            env, no_spikes, times, positions, method="glm", penalty=3.0
        )
    assert float(result.penalty) == 3.0
    assert result.reml_objective is None
    assert np.allclose(np.asarray(result.firing_rate), _floor())


def test_glm_no_neurons_public(open_field_env: Environment) -> None:
    """Empty population through the public plural API: (0, n_bins) rates + fields."""
    env = open_field_env
    times, positions, _ = _grid_session(env, [(8.0, 8.0)], seed=17)
    result = compute_spatial_rates(env, [], times, positions, method="glm")
    assert np.asarray(result.firing_rates).shape == (0, env.n_bins)
    assert result.method == "glm"
    assert result.bandwidth is None
    rank = result.rank
    assert isinstance(rank, int) and rank >= 1
    assert np.asarray(result.coefficients).shape == (rank, 0)
    assert np.asarray(result.deviance).shape == (0,)
    assert result.penalty is None and result.reml_objective is None
    assert result.converged is True
    # Occupancy is still computed from the trajectory (independent of neural data).
    assert np.asarray(result.occupancy).shape == (env.n_bins,)


def test_penalty0_identifiability(open_field_env: Environment) -> None:
    """penalty=0 warns IFF the exposed design is rank-deficient (matrix_rank < r_eff)."""
    env = open_field_env
    n_bins = env.n_bins
    bin_centers = np.asarray(env.bin_centers)
    dist2 = np.sum((bin_centers - np.array([8.0, 8.0])) ** 2, axis=1)
    rng = np.random.default_rng(2)

    # (a) all bins exposed, modest rank -> full-rank exposed design -> NO warn.
    occ_full = np.full(n_bins, 3.0)
    counts_full = rng.poisson(15.0 * np.exp(-dist2 / 18.0) * occ_full)[None, :].astype(
        float
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _compute_glm_spatial_rates(
            env,
            counts_full,
            occ_full,
            penalty=0.0,
            rank=8,
            resolved_backend="numpy",
            dtype=np.float64,
        )

    # (b) only half the (single-component) bins exposed but rank = full -> the
    #     exposed design has fewer rows than r_eff -> rank-deficient -> WARN.
    occ_half = np.zeros(n_bins)
    occ_half[: n_bins // 2] = 3.0
    counts_half = np.zeros((1, n_bins))
    counts_half[0, : n_bins // 2] = rng.poisson(10.0, size=n_bins // 2)
    with pytest.warns(UserWarning, match="rank-deficient"):
        _compute_glm_spatial_rates(
            env,
            counts_half,
            occ_half,
            penalty=0.0,
            rank=n_bins,
            resolved_backend="numpy",
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Cross-layout smoke: glm assembly produces finite output across layouts.
# ---------------------------------------------------------------------------
def test_all_layouts_smoke(
    open_field_env: Environment,
    two_component_env: Environment,
    four_component_env: Environment,
    polar_env,
    simulate_place_fields,
) -> None:
    """glm assembly runs on grid / masked-multi-component / 1D / hex / polar / mesh.

    Each layout exercises a distinct connectivity + bin-volume geometry that the
    ``_mrf_basis`` -> ``live_bins`` scatter runs over.
    """
    import importlib.util

    hex_env = Environment.from_layout(
        kind="hexagonal",
        layout_params={
            "hexagon_width": 6.0,
            "dimension_ranges": ((0.0, 24.0), (0.0, 24.0)),
        },
    )
    envs = {
        "open_field": open_field_env,
        "two_component": two_component_env,
        "four_component_1d": four_component_env,
        "hexagonal": hex_env,
        "polar": polar_env,
    }
    # Triangular mesh needs shapely for the boundary polygon; skip that entry if
    # shapely is absent rather than the whole test.
    if importlib.util.find_spec("shapely") is not None:
        from shapely.geometry import Polygon

        envs["triangular_mesh"] = Environment.from_layout(
            kind="TriangularMesh",
            layout_params={
                "boundary_polygon": Polygon([(0, 0), (24, 0), (24, 24), (0, 24)]),
                "point_spacing": 8.0,
            },
        )
    for name, env in envs.items():
        centers = [tuple(np.asarray(env.bin_centers)[0])]
        counts, occ = simulate_place_fields(env, centers, sigma=3.0)
        # counts from the fixture are (n_bins, n_units); the public API is
        # unit-major (n_units, n_bins).
        rates, fit = _compute_glm_spatial_rates(
            env,
            counts.T,
            occ,
            penalty=None,
            rank=50,
            resolved_backend="numpy",
            dtype=np.float64,
        )
        rates = np.asarray(rates)
        assert rates.shape == (1, env.n_bins), name
        assert np.all(np.isfinite(rates)), name
        assert fit.rank >= 1, name


# ---------------------------------------------------------------------------
# Regression: default (ratio) path is byte-identical.
# ---------------------------------------------------------------------------
def test_default_method_unchanged(open_field_env: Environment) -> None:
    env = open_field_env
    times, positions, spike_times = _grid_session(env, [(8.0, 8.0)], seed=14)

    # Omitting method -> "diffusion_kde", omitting bandwidth -> 5.0 resolved.
    default = compute_spatial_rate(env, spike_times[0], times, positions)
    explicit = compute_spatial_rate(
        env, spike_times[0], times, positions, method="diffusion_kde", bandwidth=5.0
    )
    assert default.method == "diffusion_kde"
    assert default.bandwidth == 5.0
    np.testing.assert_array_equal(
        np.asarray(default.firing_rate), np.asarray(explicit.firing_rate)
    )


def test_default_batch_method_unchanged(open_field_env: Environment) -> None:
    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=15)
    default = compute_spatial_rates(env, spike_times, times, positions)
    explicit = compute_spatial_rates(
        env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
    )
    assert default.method == "diffusion_kde"
    assert default.bandwidth == 5.0
    np.testing.assert_array_equal(
        np.asarray(default.firing_rates), np.asarray(explicit.firing_rates)
    )


def test_summary_table_gains_gam_columns(open_field_env: Environment) -> None:
    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0), (8.0, 2.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=16)

    glm = compute_spatial_rates(env, spike_times, times, positions, method="glm")
    df = glm.summary_table()
    for col in ("penalty", "rank", "deviance", "converged", "n_iter", "reml_objective"):
        assert col in df.columns, f"summary_table missing glm column {col}"
    assert len(df) == len(centers)

    ratio = compute_spatial_rates(
        env, spike_times, times, positions, method="diffusion_kde"
    )
    df_ratio = ratio.summary_table()
    for col in ("penalty", "rank", "deviance", "reml_objective"):
        assert col not in df_ratio.columns


def test_glm_nwb_write_roundtrips(open_field_env: Environment) -> None:
    """A glm result writes to NWB and reads back with its GAM diagnostics.

    ``write_spatial_rates`` persists the GAM fields (coefficients, penalty,
    penalty_weights, rank, deviance, converged, n_iter, reml_objective) and
    ``read_place_field`` reconstructs them field-for-field, so the estimator
    output survives a round-trip rather than being dropped.
    """
    pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from pynwb import NWBFile

    from neurospatial.io.nwb._fields import read_place_field, write_spatial_rates

    env = open_field_env
    centers = [(4.0, 4.0), (12.0, 12.0)]
    times, positions, spike_times = _grid_session(env, centers, seed=18)
    glm = compute_spatial_rates(env, spike_times, times, positions, method="glm")

    nwb = NWBFile(
        session_description="t",
        identifier="t",
        session_start_time=datetime.now(timezone.utc),
    )
    write_spatial_rates(nwb, glm, name="glm_rates")
    back = read_place_field(nwb, name="glm_rates", env=env)

    assert back.method == "glm"
    assert back.bandwidth is None
    assert back.rank == glm.rank
    np.testing.assert_allclose(
        np.asarray(back.coefficients), np.asarray(glm.coefficients)
    )
    np.testing.assert_allclose(np.asarray(back.deviance), np.asarray(glm.deviance))
