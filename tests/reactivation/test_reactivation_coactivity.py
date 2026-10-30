"""Member rules, member co-firing, pattern similarity, event responses, the
semipartial correlation, EV contributions and the EV/REV contrast, each
against a hand calculation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from neurospatial.reactivation import (
    AssemblyPattern,
    assembly_event_response,
    assembly_member_cofiring,
    assembly_similarity,
    detect_assemblies,
    ev_rev_contrast,
    explained_variance_contributions,
    explained_variance_reactivation,
    pairwise_correlations,
)
from neurospatial.reactivation.assemblies import _member_masks

# ---------------------------------------------------------------------------
# Member rules
# ---------------------------------------------------------------------------


def _hand_otsu(values: np.ndarray) -> np.ndarray:
    """Otsu's split by trying every split of the sorted values with np.var."""
    ordered = np.sort(values)
    n = ordered.size
    within = [
        (k / n) * np.var(ordered[:k]) + ((n - k) / n) * np.var(ordered[k:])
        for k in range(1, n)
    ]
    return values > ordered[int(np.argmin(within))]


def test_positive_z_uses_signed_weights() -> None:
    """One large negative weight: |w| z-scoring makes it a member and lifts
    the bar for the others; the signed rule follows w > mean + 2 SD, w > 0."""
    weights = np.full(20, 0.05)
    weights[[0, 1]] = 0.6
    weights[2] = -0.5
    weights /= np.linalg.norm(weights)
    patterns = weights[None, :]
    signed = _member_masks(patterns, "positive_z", 2.0, 20)[0]
    expected = (weights > 0) & (weights > weights.mean() + 2 * weights.std())
    assert np.array_equal(signed, expected)
    assert np.flatnonzero(signed).tolist() == [0, 1]
    absolute = _member_masks(patterns, "abs_z", 2.0, 20)[0]
    assert 2 in np.flatnonzero(absolute)


def test_member_rules_match_hand_calculations() -> None:
    rng = np.random.default_rng(0)
    patterns = rng.normal(0.0, 1.0, (5, 40))
    patterns[:, :4] += 4.0
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    otsu = _member_masks(patterns, "otsu", 2.0, 40)
    inverse = _member_masks(patterns, "inverse_sqrt_n", 2.0, 55)
    for row, weights in enumerate(patterns):
        assert np.array_equal(otsu[row], _hand_otsu(np.abs(weights)))
        assert np.array_equal(inverse[row], weights > 1 / np.sqrt(55))


def test_detect_assemblies_applies_the_member_rule() -> None:
    """Members of each rule are the hand rule applied to the returned
    weights (unit length, largest absolute weight positive)."""
    rng = np.random.default_rng(1)
    counts = rng.poisson(1.0, (4000, 24)).astype(float)
    events = rng.choice(4000, 300, replace=False)
    counts[np.ix_(events, [0, 1, 2, 3])] += rng.poisson(3.0, (300, 4))
    for rule in ("positive_z", "otsu", "inverse_sqrt_n"):
        result = detect_assemblies(counts, member_rule=rule, rng=0)
        for pattern in result.patterns:
            w = pattern.weights
            assert np.linalg.norm(w) == pytest.approx(1.0)
            assert w[np.argmax(np.abs(w))] > 0
            if rule == "positive_z":
                hand = (w > 0) & (stats.zscore(w) > 2.0)
            elif rule == "otsu":
                hand = _hand_otsu(np.abs(w))
            else:
                hand = w > 1 / np.sqrt(24)
            assert pattern.member_indices.tolist() == np.flatnonzero(hand).tolist()
    assert detect_assemblies(counts, member_rule="positive_z", rng=0).patterns[
        0
    ].member_indices.tolist() == [0, 1, 2, 3]


def test_unknown_member_rule_is_refused() -> None:
    counts = np.random.default_rng(0).poisson(1.0, (200, 5)).astype(float)
    with pytest.raises(ValueError, match="member_rule must be one of"):
        detect_assemblies(counts, member_rule="largest")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Member co-firing
# ---------------------------------------------------------------------------


@pytest.fixture
def member_pattern() -> AssemblyPattern:
    return AssemblyPattern(
        np.array([0.6, 0.1, 0.6, 0.5, 0.1]),
        np.array([0, 2, 3]),
        0.4,
        unit_ids=[10, 11, 12, 13, 14],
    )


def test_cofiring_matches_np_corrcoef(member_pattern: AssemblyPattern) -> None:
    rng = np.random.default_rng(2)
    before = rng.poisson(2.0, (250, 5)).astype(float)
    after = rng.poisson(2.0, (300, 5)).astype(float)
    after[:, [0, 2, 3]] += rng.poisson(1.5, (300, 1))
    result = assembly_member_cofiring(before, after, member_pattern)
    upper = np.triu_indices(3, k=1)
    hand_before = np.corrcoef(before[:, [0, 2, 3]], rowvar=False)[upper]
    hand_after = np.corrcoef(after[:, [0, 2, 3]], rowvar=False)[upper]
    np.testing.assert_allclose(result.control_correlations, hand_before)
    np.testing.assert_allclose(result.match_correlations, hand_after)
    assert result.control_mean_correlation == pytest.approx(hand_before.mean())
    assert result.match_mean_correlation == pytest.approx(hand_after.mean())
    assert result.mean_correlation_difference == pytest.approx(
        hand_after.mean() - hand_before.mean()
    )
    assert result.member_unit_ids.tolist() == [10, 12, 13]
    assert (result.n_member_pairs, result.n_control_bins, result.n_match_bins) == (
        3,
        250,
        300,
    )


def test_cofiring_means_use_pairs_defined_in_both(
    member_pattern: AssemblyPattern,
) -> None:
    """A member silent in one period has NaN pairs there, left out of both
    means; columns are matched by label."""
    rng = np.random.default_rng(3)
    before = rng.poisson(2.0, (250, 5)).astype(float)
    after = rng.poisson(2.0, (250, 5)).astype(float)
    before[:, 3] = 0.0
    with pytest.warns(UserWarning, match=r"members \[13\]"):
        result = assembly_member_cofiring(
            before[:, ::-1],
            after,
            member_pattern,
            control_unit_ids=[14, 13, 12, 11, 10],
            match_unit_ids=[10, 11, 12, 13, 14],
        )
    assert np.isnan(result.control_correlations[1:]).all()
    assert result.n_member_pairs == 1
    hand_before = np.corrcoef(before[:, 0], before[:, 2])[0, 1]
    hand_after = np.corrcoef(after[:, 0], after[:, 2])[0, 1]
    assert result.control_mean_correlation == pytest.approx(hand_before)
    assert result.match_mean_correlation == pytest.approx(hand_after)


def test_cofiring_of_a_single_member_is_nan() -> None:
    pattern = AssemblyPattern(np.array([1.0, 0.1, 0.1]), np.array([0]), 0.3)
    counts = np.random.default_rng(0).poisson(1.0, (50, 3)).astype(float)
    with pytest.warns(UserWarning, match="no pair of members"):
        result = assembly_member_cofiring(counts, counts, pattern)
    assert np.isnan(result.mean_correlation_difference)
    assert result.n_member_pairs == 0


def test_cofiring_needs_both_label_arguments(member_pattern) -> None:
    counts = np.ones((10, 5))
    with pytest.raises(ValueError, match="together"):
        assembly_member_cofiring(counts, counts, member_pattern, control_unit_ids=[1])


# ---------------------------------------------------------------------------
# Pattern similarity
# ---------------------------------------------------------------------------


def test_similarity_is_the_absolute_cosine_by_label() -> None:
    rng = np.random.default_rng(4)
    first = [
        AssemblyPattern(rng.normal(size=6), np.array([0]), 0.2, unit_ids=list("abcdef"))
        for _ in range(3)
    ]
    order = [5, 3, 1, 0, 2, 4]
    labels = np.array(list("abcdef"))[order]
    second = [
        AssemblyPattern(-first[1].weights[order], np.array([0]), 0.2, unit_ids=labels),
        AssemblyPattern(rng.normal(size=6), np.array([0]), 0.2, unit_ids=labels),
    ]
    result = assembly_similarity(first, second)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            weights_b = np.empty(6)
            weights_b[order] = b.weights
            hand = abs(a.weights @ weights_b) / (
                np.linalg.norm(a.weights) * np.linalg.norm(weights_b)
            )
            assert result.similarity[i, j] == pytest.approx(hand)
    assert result.similarity[1, 0] == pytest.approx(1.0)
    assert result.best_match_indices[1] == 0
    np.testing.assert_allclose(
        result.best_match_similarity, result.similarity.max(axis=1)
    )


def test_similarity_refuses_different_units() -> None:
    a = [AssemblyPattern(np.ones(3), np.array([0]), 0.2, unit_ids=[1, 2, 3])]
    b = [AssemblyPattern(np.ones(3), np.array([0]), 0.2, unit_ids=[1, 2, 4])]
    with pytest.raises(ValueError, match="different units"):
        assembly_similarity(a, b)


# ---------------------------------------------------------------------------
# Event responses
# ---------------------------------------------------------------------------


def _hand_peri_event(times, activation, step, events, lags):
    """Value of the bin [t - step/2, t + step/2) holding each event + lag."""
    values = np.full((events.size, lags.size), np.nan)
    for i, event in enumerate(events):
        for j, lag in enumerate(lags):
            query = event + lag
            inside = (times - step / 2 <= query + 1e-12) & (
                query < times + step / 2 - 1e-12
            )
            if inside.any():
                values[i, j] = activation[np.flatnonzero(inside)[0]]
    return values


@pytest.fixture
def trace() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R(t) every 10 ms in two stretches with a gap, and five events."""
    rng = np.random.default_rng(5)
    times = np.concatenate(
        [np.arange(0.005, 20.0, 0.01), np.arange(30.005, 50.0, 0.01)]
    )
    activation = rng.normal(0.0, 1.0, times.size)
    events = np.array([2.003, 10.0, 19.8, 29.9, 40.017])
    return times, activation, events


def test_event_response_matches_a_hand_alignment(trace) -> None:
    times, activation, events = trace
    result = assembly_event_response(times, activation, events, (-0.5, 0.5))
    lags = np.arange(-50, 50) * 0.01
    np.testing.assert_allclose(result.lags, lags, atol=1e-12)
    hand = _hand_peri_event(times, activation, 0.01, events, lags)
    # The event at 19.8 s runs into the gap and the one at 29.9 s starts in it.
    assert np.isnan(hand[2, -10:]).all() and np.isnan(hand[3, :10]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_allclose(result.peri_event_mean, np.nanmean(hand, axis=0))
        in_response = (lags >= -1e-12) & (lags < 0.1 - 1e-12)
        expected = np.nanmean(hand[:, in_response], axis=1)
    np.testing.assert_allclose(result.event_responses, expected)
    assert result.response == pytest.approx(np.nanmean(expected))
    assert result.n_events == 4 and np.isnan(result.event_responses[3])
    assert np.isnan(result.p_value) and np.isnan(result.baseline_responses).all()


def test_event_response_peak_and_baseline_test(trace) -> None:
    times, activation, events = trace
    result = assembly_event_response(
        times,
        activation,
        events,
        (-0.5, 0.5),
        response_window=(0.0, 0.05),
        baseline_window=(-0.5, -0.1),
        reduce="peak",
    )
    lags = result.lags
    hand = _hand_peri_event(times, activation, 0.01, events, lags)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(hand, axis=0)
        response = (lags >= -1e-12) & (lags < 0.05 - 1e-12)
        baseline = (lags >= -0.5 - 1e-12) & (lags < -0.1 - 1e-12)
        responses = np.nanmean(hand[:, response], axis=1)
        baselines = np.nanmean(hand[:, baseline], axis=1)
    assert result.response == pytest.approx(mean[response].max())
    np.testing.assert_allclose(result.baseline_responses, baselines)
    both = np.isfinite(responses) & np.isfinite(baselines)
    assert result.p_value == pytest.approx(
        stats.wilcoxon(responses[both] - baselines[both]).pvalue
    )


def test_event_response_of_a_constant_is_the_constant(trace) -> None:
    times, _, events = trace
    result = assembly_event_response(
        times, np.full(times.size, 3.5), events, (-0.2, 0.3)
    )
    assert np.allclose(result.peri_event_mean, 3.5)
    assert np.allclose(result.event_responses[[0, 1, 2, 4]], 3.5)
    assert np.isnan(result.event_responses[3])  # 0-100 ms lies in the gap


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"window": (0.5, -0.5)}, "start < stop"),
        ({"response_window": (0.0, 0.9)}, "inside window"),
        ({"reduce": "median"}, "reduce must be"),
    ],
)
def test_event_response_input_errors(trace, options, message) -> None:
    times, activation, events = trace
    arguments = {"window": (-0.5, 0.5), **options}
    window = arguments.pop("window")
    with pytest.raises(ValueError, match=message):
        assembly_event_response(times, activation, events, window, **arguments)


# ---------------------------------------------------------------------------
# Semipartial correlation, contributions and the EV/REV contrast
# ---------------------------------------------------------------------------


@pytest.fixture
def correlation_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(6)
    n_pairs = 45
    control = rng.normal(0.0, 0.1, n_pairs)
    template = 0.5 * control + rng.normal(0.0, 0.1, n_pairs)
    match = 0.4 * control + 0.6 * template + rng.normal(0.0, 0.1, n_pairs)
    return template, match, control


def test_semipartial_is_the_correlation_with_the_regression_residual(
    correlation_vectors,
) -> None:
    template, match, control = correlation_vectors
    result = explained_variance_reactivation(
        template, match, control_correlations=control
    )
    slope, intercept = np.polyfit(control, match, 1)
    residual = match - (slope * control + intercept)
    assert result.semipartial_correlation == pytest.approx(
        np.corrcoef(template, residual)[0, 1]
    )
    # The partial correlation residualizes both sides.
    template_residual = template - np.polyval(np.polyfit(control, template, 1), control)
    assert result.partial_correlation == pytest.approx(
        np.corrcoef(template_residual, residual)[0, 1]
    )
    assert abs(result.semipartial_correlation) < abs(result.partial_correlation)


def test_semipartial_with_two_controls(correlation_vectors) -> None:
    template, match, control = correlation_vectors
    other = np.random.default_rng(7).normal(0.0, 0.1, template.size)
    result = explained_variance_reactivation(
        template, match, control_correlations=[control, other]
    )
    design = np.column_stack([np.ones_like(control), control, other])
    coefficients = np.linalg.lstsq(design, match, rcond=None)[0]
    residual = match - design @ coefficients
    assert result.semipartial_correlation == pytest.approx(
        np.corrcoef(template, residual)[0, 1]
    )


def test_contributions_match_leaving_each_pair_out(correlation_vectors) -> None:
    template, match, control = correlation_vectors
    template = template.copy()
    template[3] = np.nan  # one pair undefined in the template
    result = explained_variance_contributions(
        template, match, control_correlations=control, unit_ids=np.arange(10) + 100
    )
    valid = np.isfinite(template)
    total = explained_variance_reactivation(
        template, match, control_correlations=control
    ).explained_variance
    assert result.explained_variance == pytest.approx(total)
    for pair in np.flatnonzero(valid):
        keep = valid.copy()
        keep[pair] = False
        without = explained_variance_reactivation(
            template, match, control_correlations=control, pair_mask=keep
        ).explained_variance
        assert result.pair_contributions[pair] == pytest.approx(total - without)
    assert np.isnan(result.pair_contributions[3])
    first, second = np.triu_indices(10, k=1)
    for unit in range(10):
        mine = valid & ((first == unit) | (second == unit))
        assert result.unit_contributions[unit] == pytest.approx(
            result.pair_contributions[mine].mean()
        )
    assert result.n_pairs == 44
    assert result.unit_ids.tolist() == list(range(100, 110))


def test_contributions_without_a_control() -> None:
    rng = np.random.default_rng(8)
    counts = rng.poisson(1.0, (2000, 6)).astype(float)
    counts[:, :3] += rng.poisson(0.5, (2000, 1))
    template = pairwise_correlations(counts)
    match = pairwise_correlations(rng.permutation(counts, axis=0) + 0.0)
    result = explained_variance_contributions(template, match)
    total = np.corrcoef(template, match)[0, 1] ** 2
    for pair in range(template.size):
        keep = np.arange(template.size) != pair
        without = np.corrcoef(template[keep], match[keep])[0, 1] ** 2
        assert result.pair_contributions[pair] == pytest.approx(total - without)


def test_ev_rev_contrast() -> None:
    assert ev_rev_contrast(0.75, 0.25) == pytest.approx(0.5)
    values = ev_rev_contrast(np.array([0.2, 0.0, 0.1, np.nan]), [0.2, 0.0, 0.3, 0.1])
    np.testing.assert_allclose(values, [0.0, np.nan, -0.5, np.nan])
