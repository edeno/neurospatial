"""Smoke tests for the ``neurospatial.stats.circular`` API surface.

The canonical correctness tests for these functions live in
``tests/stats/test_circular_metrics.py`` (richer numerical fixtures,
edge-case / NaN / weighted variants, range checks). This file only
keeps the two pieces that file doesn't cover:

- A handful of basic functional smokes for the broader circular API
  (``circular_basis``, ``is_modulated``, ``wrap_angle``,
  ``plot_circular_basis_tuning``) — these aren't in
  ``test_circular_metrics.py``.
- The plot-helper smoke parametrized over polar vs linear projection.

Everything else (the 13 ``test_X_importable`` tautologies and the
duplicated ``test_rayleigh_test_uniform`` / ``test_*_basic`` tests that
overlap with ``test_circular_metrics.py``) was removed.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.stats.circular import (
    circular_basis,
    circular_basis_metrics,
    circular_mean,
    circular_variance,
    is_modulated,
    mean_resultant_length,
    plot_circular_basis_tuning,
    rayleigh_test,
    wrap_angle,
)


def test_circular_basis_basic():
    """``circular_basis`` builds the sin/cos design matrix at the cardinal angles."""
    angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    result = circular_basis(angles)

    assert result.design_matrix.shape == (4, 2)
    assert_allclose(result.sin_component[0], 0.0, atol=1e-10)
    assert_allclose(result.cos_component[0], 1.0, atol=1e-10)
    assert_allclose(result.sin_component[1], 1.0, atol=1e-10)
    assert_allclose(result.cos_component[1], 0.0, atol=1e-10)


def test_circular_basis_metrics_basic():
    """``circular_basis_metrics`` recovers amplitude / phase from sin/cos coefs."""
    amplitude, phase, pvalue = circular_basis_metrics(1.0, 0.0)
    assert_allclose(amplitude, 1.0)
    assert_allclose(phase, np.pi / 2)
    assert np.isnan(pvalue)  # No cov_matrix provided


def test_circular_basis_metrics_with_cov():
    """``circular_basis_metrics`` returns a p-value when a covariance is given."""
    cov = np.array([[0.01, 0.001], [0.001, 0.01]])
    amplitude, _phase, pvalue = circular_basis_metrics(0.5, 0.3, cov)
    assert_allclose(amplitude, np.sqrt(0.5**2 + 0.3**2))
    assert pvalue is not None
    assert 0 <= pvalue <= 1


def test_is_modulated_strong_signal():
    """``is_modulated`` returns True for a high-amplitude / low-noise signal."""
    cov = np.array([[0.01, 0], [0, 0.01]])
    assert is_modulated(2.0, 2.0, cov)


def test_is_modulated_weak_signal():
    """``is_modulated`` returns False when the signal is dwarfed by noise."""
    cov = np.array([[1.0, 0], [0, 1.0]])
    assert not is_modulated(0.1, 0.1, cov)


def test_wrap_angle_basic():
    """``wrap_angle`` maps a half-turn either way to +pi, the closed end of (-pi, pi]."""
    angles = np.array([0, np.pi, 2 * np.pi, 3 * np.pi, -np.pi, -2 * np.pi, -3 * np.pi])
    wrapped = wrap_angle(angles)

    assert_allclose(wrapped, [0, np.pi, 0, np.pi, np.pi, 0, np.pi], atol=1e-10)


def test_wrap_angle_stays_in_range_next_to_a_half_turn():
    """Rounding next to an odd multiple of pi never lands on the open end -pi."""
    centers = np.array([np.pi, -np.pi, 3 * np.pi, -3 * np.pi])
    neighbors = np.concatenate(
        [np.nextafter(centers, np.inf), np.nextafter(centers, -np.inf), centers]
    )
    wrapped = wrap_angle(neighbors)

    assert np.all(wrapped > -np.pi)
    assert np.all(wrapped <= np.pi)


@pytest.mark.parametrize("projection", ["polar", "linear"])
def test_plot_circular_basis_tuning_smoke(projection):
    """``plot_circular_basis_tuning`` renders polar and linear projections."""
    ax = plot_circular_basis_tuning(0.5, 0.5, intercept=1.0, projection=projection)
    assert ax is not None
    plt.close("all")


def test_plot_circular_basis_tuning_with_data():
    """Data overlay branch renders without raising."""
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rates = np.exp(1 + 0.5 * np.cos(angles))
    ax = plot_circular_basis_tuning(
        0.5, 0.0, intercept=1.0, angles=angles, rates=rates, show_data=True
    )
    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# Weighted circular statistics: weight validation and NaN co-filtering.
# ---------------------------------------------------------------------------


def test_rayleigh_test_length1_weights_raises(concentrated_angles):
    """A length-1 weight array is a mismatch, never broadcast."""
    assert len(concentrated_angles) == 20
    with pytest.raises(ValueError, match="Length mismatch"):
        rayleigh_test(concentrated_angles, weights=np.array([2.0]))


def test_rayleigh_test_negative_weights_raises(concentrated_angles):
    """A negative weight is rejected (would otherwise allow z < 0)."""
    weights = np.ones(len(concentrated_angles))
    weights[3] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        rayleigh_test(concentrated_angles, weights=weights)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rayleigh_test_nonfinite_weights_raises(concentrated_angles, bad):
    """A NaN/Inf weight is rejected (would otherwise make z/p nan)."""
    weights = np.ones(len(concentrated_angles))
    weights[2] = bad
    with pytest.raises(ValueError, match="non-finite"):
        rayleigh_test(concentrated_angles, weights=weights)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_circular_mean_nonfinite_weights_raises(concentrated_angles, bad):
    """``circular_mean`` rejects NaN/Inf weights instead of returning nan."""
    weights = np.ones(len(concentrated_angles))
    weights[5] = bad
    with pytest.raises(ValueError, match="non-finite"):
        circular_mean(concentrated_angles, weights=weights)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_mean_resultant_length_nonfinite_weights_raises(concentrated_angles, bad):
    """``mean_resultant_length`` rejects NaN/Inf weights instead of nan."""
    weights = np.ones(len(concentrated_angles))
    weights[0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        mean_resultant_length(concentrated_angles, weights=weights)


def test_rayleigh_test_all_zero_weights_raises(concentrated_angles):
    """All-zero weights raise a ValueError, not a ZeroDivisionError."""
    weights = np.zeros(len(concentrated_angles))
    with pytest.raises(ValueError, match="zero"):
        rayleigh_test(concentrated_angles, weights=weights)


def test_rayleigh_test_nan_cofilters_weights(concentrated_angles):
    """A NaN angle and its weight are dropped together, keeping alignment."""
    angles = concentrated_angles.copy()
    weights = np.arange(1.0, len(angles) + 1.0)  # counts

    # Insert a NaN at a known index; the matched weight must drop with it.
    nan_idx = 7
    angles_with_nan = angles.copy()
    angles_with_nan[nan_idx] = np.nan

    keep = np.arange(len(angles)) != nan_idx
    angles_manual = angles[keep]
    weights_manual = weights[keep]

    with pytest.warns(UserWarning, match="Removed 1 NaN"):
        z_cofiltered, p_cofiltered = rayleigh_test(angles_with_nan, weights=weights)
    z_manual, p_manual = rayleigh_test(angles_manual, weights=weights_manual)

    assert_allclose(z_cofiltered, z_manual)
    assert_allclose(p_cofiltered, p_manual)


@pytest.mark.parametrize(
    "func", [circular_mean, mean_resultant_length, circular_variance]
)
def test_weighted_metrics_nan_angle_cofilters_weight(func):
    """A NaN angle is dropped with its weight, matching the filtered input.

    Mirrors the weighted Rayleigh co-filter: for
    ``angles=[0, nan, pi/2], weights=[1, 10, 1]`` the NaN sample (and its
    weight) is dropped, so the result equals computing on the explicitly
    filtered ``angles=[0, pi/2], weights=[1, 1]`` -- not ``nan``.
    """
    angles_with_nan = np.array([0.0, np.nan, np.pi / 2])
    weights = np.array([1.0, 10.0, 1.0])

    angles_filtered = np.array([0.0, np.pi / 2])
    weights_filtered = np.array([1.0, 1.0])

    with pytest.warns(UserWarning, match="Removed 1 NaN"):
        result = func(angles_with_nan, weights=weights)
    expected = func(angles_filtered, weights=weights_filtered)

    assert np.isfinite(result)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "func", [circular_mean, mean_resultant_length, circular_variance]
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_weighted_metrics_nonfinite_weight_still_raises(func, bad):
    """A NaN/Inf WEIGHT remains invalid input and raises (not co-filtered)."""
    angles = np.array([0.0, np.pi / 4, np.pi / 2])
    weights = np.array([1.0, bad, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        func(angles, weights=weights)


def test_rayleigh_test_integer_weights_match_replication():
    """Integer weights reproduce physical replication of each angle."""
    angles = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    counts = np.array([1, 2, 1, 3, 1])

    z_weighted, p_weighted = rayleigh_test(angles, weights=counts.astype(float))
    z_repeat, p_repeat = rayleigh_test(np.repeat(angles, counts))

    assert_allclose(z_weighted, z_repeat, rtol=1e-9)
    assert_allclose(p_weighted, p_repeat, rtol=1e-9)


def test_rayleigh_test_concentrated_counts_significant():
    """100 spikes in 2 adjacent bins is strongly tuned, not rejected as NaN.

    The effective sample size for weighted (count) data is sum(weights), not
    the number of occupied angular bins. Two bins carrying 100 spikes must
    therefore yield a small p-value matching physical replication, rather
    than tripping the min-samples gate.
    """
    centers = np.array([0.0, 0.1])
    counts = np.array([60.0, 40.0])  # 100 spikes total in 2 adjacent bins

    z_weighted, p_weighted = rayleigh_test(centers, weights=counts)

    # Significant, not NaN.
    assert np.isfinite(p_weighted)
    assert p_weighted < 1e-10

    # Matches the physical oracle: repeat each angle by its count.
    z_repeat, p_repeat = rayleigh_test(np.repeat(centers, counts.astype(int)))
    assert_allclose(z_weighted, z_repeat, rtol=1e-9)
    assert_allclose(p_weighted, p_repeat, rtol=1e-9)


def test_rayleigh_test_insufficient_total_weight_raises():
    """Too few total counts (sum < 3) is genuinely insufficient and rejected."""
    centers = np.array([0.0, 0.1])
    counts = np.array([1.0, 1.0])  # only 2 spikes total
    with pytest.raises(ValueError, match="3 total weight"):
        rayleigh_test(centers, weights=counts)


def test_mean_resultant_length_length1_weights_raises(concentrated_angles):
    """``mean_resultant_length`` rejects a length-1 weight array."""
    with pytest.raises(ValueError, match="Length mismatch"):
        mean_resultant_length(concentrated_angles, weights=np.array([1.0]))


def test_circular_mean_negative_weights_raises(concentrated_angles):
    """``circular_mean`` rejects negative weights."""
    weights = np.ones(len(concentrated_angles))
    weights[0] = -2.0
    with pytest.raises(ValueError, match="non-negative"):
        circular_mean(concentrated_angles, weights=weights)


def test_circular_variance_weights_validates(concentrated_angles):
    """``circular_variance`` rejects a mismatched-length weight array."""
    weights = np.ones(len(concentrated_angles) - 1)
    with pytest.raises(ValueError, match="Length mismatch"):
        circular_variance(concentrated_angles, weights=weights)


@pytest.mark.parametrize("caller_axis", [False, True], ids=["new axis", "caller axis"])
def test_circular_basis_plot_north_is_up(caller_axis, polar_display_offset):
    """Angles follow the math convention: 0 = East (right), π/2 = North (up).

    A caller-supplied polar axis configured North-up/clockwise is normalized.
    """
    ax = None
    if caller_axis:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    ax = plot_circular_basis_tuning(1.0, 0.0, ax=ax)

    north_dx, north_dy = polar_display_offset(ax, np.pi / 2, 1.0)
    east_dx, east_dy = polar_display_offset(ax, 0.0, 1.0)
    assert abs(north_dx) < 1e-6 * north_dy
    assert north_dy > 0
    assert abs(east_dy) < 1e-6 * east_dx
    assert east_dx > 0
    plt.close("all")


# ---------------------------------------------------------------------------
# Rayleigh p-value: Zar's approximation, checked against Monte Carlo.
# ---------------------------------------------------------------------------


def _uniform_resultant_lengths(n: int, n_draws: int, seed: int) -> np.ndarray:
    """Mean resultant lengths of ``n_draws`` uniform samples of size ``n``."""
    angles = np.random.default_rng(seed).uniform(0, 2 * np.pi, (n_draws, n))
    return np.hypot(np.cos(angles).mean(axis=1), np.sin(angles).mean(axis=1))


def _angles_with_resultant(n: int, r: float) -> np.ndarray:
    """``n`` angles, symmetric about 0, whose mean resultant length is ``r``."""
    from scipy.optimize import brentq

    def resultant(spread: float) -> float:
        return float(np.mean(np.cos(np.linspace(-spread, spread, n)))) - r

    return np.linspace(-1, 1, n) * brentq(resultant, 1e-9, np.pi)


@pytest.mark.parametrize("n", [6, 8, 15, 30])
def test_rayleigh_pvalue_matches_monte_carlo_tail(n):
    """p(R) is the chance that a uniform sample of size n has a resultant >= R.

    Below n = 6 the approximation is conservative in the far tail (p too
    large), so the check starts at 6.
    """
    null_r = _uniform_resultant_lengths(n, 200_000, seed=n)
    for target in (0.05, 0.005):
        r = float(np.quantile(null_r, 1 - target))
        _, p = rayleigh_test(_angles_with_resultant(n, r))
        assert p == pytest.approx(target, rel=0.15)


@pytest.mark.parametrize("n", [3, 6, 8, 10, 12, 20, 49, 50, 200])
def test_rayleigh_pvalue_positive_and_decreasing_in_r(n):
    """Tighter clustering never raises p, and p never reaches zero."""
    p_values = [
        rayleigh_test(_angles_with_resultant(n, r))[1]
        for r in np.linspace(0.05, 0.999, 40)
    ]
    p_values.append(rayleigh_test(np.zeros(n))[1])  # R = 1
    assert np.all(np.array(p_values) > 0)
    assert np.all(np.diff(p_values) <= 1e-15)


def test_rayleigh_pvalue_identical_small_sample_matches_zar():
    """Eight identical angles: Zar's formula, not the old clipped 0.0."""
    n = 8
    _, p = rayleigh_test(np.zeros(n))
    expected = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - n**2)) - (1 + 2 * n))
    assert p == pytest.approx(expected, rel=1e-10)
    assert p > 0


@pytest.mark.parametrize("n", [5, 10, 30])
def test_rayleigh_false_positive_rate_under_uniform_null(n):
    """Under H0, P(p < 0.05) is close to 0.05."""
    rng = np.random.default_rng(100 + n)
    p_values = np.array(
        [rayleigh_test(rng.uniform(0, 2 * np.pi, n))[1] for _ in range(4000)]
    )
    # Binomial SD at 4000 draws is 0.0034; allow a small approximation bias.
    assert np.mean(p_values < 0.05) == pytest.approx(0.05, abs=0.012)


def test_rayleigh_rejects_non_count_weights():
    """Occupancy-like weights would make the p-value depend on their units."""
    angles = np.linspace(0, 1, 8)
    with pytest.raises(ValueError, match="whole-number counts"):
        rayleigh_test(angles, weights=np.full(8, 0.5))


# ---------------------------------------------------------------------------
# Circular-linear and circular-circular correlation.
# ---------------------------------------------------------------------------


def test_circular_linear_r_squared_is_least_squares_r_squared():
    """r**2 is the R**2 of x ~ 1 + cos + sin, and matches Mardia's formula."""
    from scipy.stats import chi2, pearsonr

    from neurospatial.stats.circular import circular_linear_correlation

    rng = np.random.default_rng(3)
    angles = rng.vonmises(1.0, 2.0, 60)  # cos and sin correlated
    x = 2.0 * np.cos(angles - 0.4) + rng.normal(0, 1.5, 60)

    design = np.column_stack([np.ones(60), np.cos(angles), np.sin(angles)])
    fitted = design @ np.linalg.lstsq(design, x, rcond=None)[0]
    r_squared = 1 - np.sum((x - fitted) ** 2) / np.sum((x - x.mean()) ** 2)

    r_xs = pearsonr(x, np.sin(angles))[0]
    r_xc = pearsonr(x, np.cos(angles))[0]
    r_cs = pearsonr(np.cos(angles), np.sin(angles))[0]
    mardia = (r_xs**2 + r_xc**2 - 2 * r_xs * r_xc * r_cs) / (1 - r_cs**2)

    r, p = circular_linear_correlation(angles, x)
    assert abs(r_cs) > 0.1  # the cross term matters
    assert r**2 == pytest.approx(r_squared, rel=1e-10)
    assert r**2 == pytest.approx(mardia, rel=1e-10)
    assert p == pytest.approx(chi2.sf(60 * r_squared, 2), rel=1e-10)


def test_circular_linear_two_valued_angles_use_one_degree_of_freedom():
    """With collinear cos/sin the fit has one predictor: r = |corr(x, cos)|."""
    from scipy.stats import chi2

    from neurospatial.stats.circular import circular_linear_correlation

    angles = np.tile([0.0, 1.0], 10)
    x = np.arange(20.0) + 3.0 * (angles == 1.0)
    r, p = circular_linear_correlation(angles, x)
    expected = abs(np.corrcoef(x, np.cos(angles))[0, 1])
    assert r == pytest.approx(expected, rel=1e-10)
    assert p == pytest.approx(chi2.sf(20 * expected**2, 1), rel=1e-10)


@pytest.mark.parametrize(
    ("angles", "linear"),
    [
        (np.linspace(0, 3, 10), np.full(10, 2.0)),  # constant linear variable
        (np.full(10, 1.3), np.arange(10.0)),  # constant angles
    ],
)
def test_circular_linear_undefined_is_nan(angles, linear):
    """An undefined correlation is NaN, not r=0, p=1 ("no correlation")."""
    from neurospatial.stats.circular import circular_linear_correlation

    with pytest.warns(RuntimeWarning, match="undefined"):
        r, p = circular_linear_correlation(angles, linear)
    assert np.isnan(r) and np.isnan(p)


def test_circular_circular_exact_identities_and_pvalue():
    """r(a, a) = r(a, a + c) = 1 and r(a, -a) = -1; p from the J-S statistic."""
    from scipy.stats import norm

    from neurospatial.stats.circular import circular_circular_correlation

    rng = np.random.default_rng(5)
    a = rng.vonmises(0.3, 1.0, 40)
    assert circular_circular_correlation(a, a)[0] == pytest.approx(1.0, abs=1e-12)
    assert circular_circular_correlation(a, a + 2.0)[0] == pytest.approx(1.0, abs=1e-12)
    assert circular_circular_correlation(a, -a)[0] == pytest.approx(-1.0, abs=1e-12)

    b = a + rng.normal(0, 1.0, 40)
    rho, p = circular_circular_correlation(a, b)
    ma = np.angle(np.mean(np.exp(1j * a)))
    mb = np.angle(np.mean(np.exp(1j * b)))
    sa, sb = np.sin(a - ma), np.sin(b - mb)
    expected_rho = np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    ts = (
        np.sqrt(40 * np.mean(sa**2) * np.mean(sb**2) / np.mean(sa**2 * sb**2))
        * expected_rho
    )
    assert rho == pytest.approx(expected_rho, rel=1e-10)
    assert p == pytest.approx(2 * norm.sf(abs(ts)), rel=1e-8)


# ---------------------------------------------------------------------------
# Descriptive statistics: one NaN rule, NaN when undefined.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func", [circular_mean, circular_variance, mean_resultant_length]
)
def test_nan_angles_dropped_the_same_with_or_without_weights(func):
    """Unit weights must not change how NaN angles are treated."""
    angles = np.array([0.1, np.nan, 0.3, 0.2])
    with pytest.warns(UserWarning, match="Removed 1 NaN"):
        unweighted = func(angles)
    with pytest.warns(UserWarning, match="Removed 1 NaN"):
        weighted = func(angles, weights=np.ones(4))
    expected = func(np.array([0.1, 0.3, 0.2]))
    assert unweighted == pytest.approx(weighted, rel=1e-12)
    assert unweighted == pytest.approx(expected, rel=1e-12)


def test_circular_mean_matches_scipy_and_returns_radians_for_degrees():
    from scipy.stats import circmean

    angles = np.random.default_rng(2).vonmises(2.5, 3.0, 50)
    assert wrap_angle(circular_mean(angles) - circmean(angles)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert circular_mean(np.degrees(angles), angle_unit="deg") == pytest.approx(
        circular_mean(angles), abs=1e-12
    )


@pytest.mark.parametrize(
    "angles",
    [
        np.array([0.0, np.pi]),  # opposite angles: zero resultant
        np.linspace(0, 2 * np.pi, 8, endpoint=False),
        np.array([]),
        np.array([np.nan, np.nan]),
    ],
)
def test_circular_mean_undefined_is_nan(angles):
    """No direction (zero resultant) or no angles gives NaN, not an angle."""
    assert np.isnan(circular_mean(angles))


def test_descriptive_stats_empty_or_all_nan_are_nan():
    for angles in (np.array([]), np.array([np.nan, np.nan])):
        assert np.isnan(mean_resultant_length(angles))
        assert np.isnan(circular_variance(angles))
    assert np.isnan(mean_resultant_length(np.array([1.0, 2.0]), weights=np.zeros(2)))


# ---------------------------------------------------------------------------
# GLM circular basis and Wald test.
# ---------------------------------------------------------------------------


def test_wald_pvalue_with_correlated_covariance():
    """p = chi2.sf(beta' inv(cov) beta, 2), from an explicit 2x2 inverse."""
    from scipy.stats import chi2

    beta = np.array([0.3, 0.1])  # [sin, cos]
    cov = np.array([[0.04, 0.01], [0.01, 0.01]])
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    inverse = np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]]) / det
    expected = chi2.sf(beta @ inverse @ beta, 2)
    _, _, p = circular_basis_metrics(beta[0], beta[1], cov)
    assert p == pytest.approx(expected, rel=1e-12)
    # Swapping the coefficients changes the answer: the order is [sin, cos].
    _, _, swapped = circular_basis_metrics(beta[1], beta[0], cov)
    assert swapped != pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    ("beta_sin", "cov", "match"),
    [
        (0.3, np.array([[-0.01, 0.0], [0.0, -0.01]]), "positive semi-definite"),
        (0.3, np.array([[0.01, 0.02], [0.0, 0.01]]), "symmetric"),
        (0.3, np.array([[np.nan, 0.0], [0.0, 0.01]]), "finite"),
        (np.nan, np.eye(2) * 0.01, "finite"),
    ],
)
def test_wald_rejects_invalid_inputs(beta_sin, cov, match):
    """An invalid covariance or coefficient raises instead of returning p=1."""
    with pytest.raises(ValueError, match=match):
        circular_basis_metrics(beta_sin, 0.2, cov)
    with pytest.raises(ValueError, match=match):
        is_modulated(beta_sin, 0.2, cov)


def test_singular_covariance_gives_nan_and_not_modulated():
    singular = np.array([[0.01, 0.01], [0.01, 0.01]])
    with pytest.warns(UserWarning, match="singular"):
        _, _, p = circular_basis_metrics(0.5, 0.5, singular)
    assert np.isnan(p)
    with pytest.warns(UserWarning, match="singular"):
        assert is_modulated(0.5, 0.5, singular) is False


@pytest.mark.parametrize(
    "angles", [np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0.0, np.nan, 1.0])]
)
def test_circular_basis_rejects_non_1d_or_non_finite(angles):
    with pytest.raises(ValueError, match="1-D array of finite values"):
        circular_basis(angles)


def test_circular_basis_components_derive_from_angles():
    result = circular_basis(np.array([0.0, 90.0, 180.0]), angle_unit="deg")
    assert_allclose(result.angles, [0.0, np.pi / 2, np.pi])
    assert_allclose(result.design_matrix, [[0, 1], [1, 0], [0, -1]], atol=1e-12)
