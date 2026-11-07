"""Null data, simulated sessions and detection rates for reactivation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, binomtest

from neurospatial.reactivation import (
    KUDRIMOTI_1999,
    LANSINK_2009,
    PEYRACHE_2010,
    ReactivationWindows,
    compare_reactivation_methods,
    null_reactivation_data,
    reactivation_detection_rates,
    simulated_reactivation_session,
)

GROUPS = {"tetrode": np.arange(30) // 3, "region": np.arange(30) // 15}
# One 10-min window in each period of a 600 s simulation.
KUDRIMOTI_ONE_WINDOW = dataclasses.replace(
    KUDRIMOTI_1999, match_windows=ReactivationWindows(length=600.0, count=1)
)
LANSINK_ONE_WINDOW = dataclasses.replace(
    LANSINK_2009,
    n_shuffles=300,
    control_windows=ReactivationWindows(),
    match_windows=ReactivationWindows(),
)
PEYRACHE_FAST = dataclasses.replace(PEYRACHE_2010, n_shuffles=50)
METHODS = (KUDRIMOTI_ONE_WINDOW, LANSINK_ONE_WINDOW, PEYRACHE_FAST)


def _table(session, spike_times=None, seed: int = 0) -> pd.DataFrame:
    return compare_reactivation_methods(
        session.spike_times if spike_times is None else spike_times,
        session.template_epochs,
        session.match_epochs,
        session.control_epochs,
        reactivation_methods=METHODS,
        unit_groups=GROUPS,
        rng=seed,
    )


@pytest.fixture(scope="module")
def reactivated():
    return simulated_reactivation_session(duration=600.0, rng=0)


# ---------------------------------------------------------------------------
# Null data
# ---------------------------------------------------------------------------


def _split(trains, start: float, stop: float):
    inside = [t[(t >= start) & (t < stop)] for t in trains]
    outside = [t[(t < start) | (t >= stop)] for t in trains]
    return inside, outside


@pytest.mark.parametrize("kind", ["cell_identity", "circular_shift", "rate_poisson"])
def test_spikes_outside_the_match_period_are_kept(reactivated, kind: str) -> None:
    null = null_reactivation_data(
        reactivated.spike_times, reactivated.match_epochs, kind=kind, rng=1
    )
    start, stop = reactivated.match_epochs[0]
    _, before = _split(reactivated.spike_times, start, stop)
    inside, after = _split(null, start, stop)
    for original, kept in zip(before, after, strict=True):
        np.testing.assert_array_equal(original, kept)
    assert all(np.all(np.diff(train) >= 0) for train in null)
    assert sum(t.size for t in inside) > 0
    again = null_reactivation_data(
        reactivated.spike_times, reactivated.match_epochs, kind=kind, rng=1
    )
    for first, second in zip(null, again, strict=True):
        np.testing.assert_array_equal(first, second)


def test_cell_identity_reassigns_whole_match_trains(reactivated) -> None:
    null = null_reactivation_data(
        reactivated.spike_times, reactivated.match_epochs, kind="cell_identity", rng=2
    )
    start, stop = reactivated.match_epochs[0]
    original, _ = _split(reactivated.spike_times, start, stop)
    permuted, _ = _split(null, start, stop)
    sources = []
    for train in permuted:
        matches = [i for i, o in enumerate(original) if np.array_equal(o, train)]
        assert len(matches) == 1
        sources.append(matches[0])
    assert sorted(sources) == list(range(30))
    assert sources != list(range(30))


def test_circular_shift_keeps_counts_and_circular_intervals() -> None:
    rng = np.random.default_rng(3)
    epochs = np.array([[10.0, 20.0], [30.0, 45.0]])
    trains = [np.sort(rng.uniform(0, 50, 60)) for _ in range(4)]
    null = null_reactivation_data(trains, epochs, kind="circular_shift", rng=4)
    for original, shifted in zip(trains, null, strict=True):
        for start, stop in epochs:
            a = original[(original >= start) & (original < stop)]
            b = shifted[(shifted >= start) & (shifted < stop)]
            assert a.size == b.size

            # The intervals around the circle of the epoch, wrap included.
            def circular(spikes, start=start, stop=stop):
                return np.sort(np.diff(np.r_[spikes, spikes[0] + (stop - start)]))

            np.testing.assert_allclose(circular(a), circular(b), atol=1e-9)
            assert not np.allclose(a, b)


def test_rate_poisson_keeps_each_units_rate(reactivated) -> None:
    null = null_reactivation_data(
        reactivated.spike_times, reactivated.match_epochs, kind="rate_poisson", rng=5
    )
    start, stop = reactivated.match_epochs[0]
    original, _ = _split(reactivated.spike_times, start, stop)
    redrawn, _ = _split(null, start, stop)
    counts = np.array([t.size for t in original])
    new_counts = np.array([t.size for t in redrawn])
    # Poisson counts: within 5 standard deviations of each unit's count.
    assert np.all(np.abs(new_counts - counts) < 5 * np.sqrt(counts))


def test_unknown_kind_is_refused(reactivated) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        null_reactivation_data(
            reactivated.spike_times, reactivated.match_epochs, kind="shuffle"
        )


# ---------------------------------------------------------------------------
# Simulated sessions
# ---------------------------------------------------------------------------


def test_simulated_session_layout() -> None:
    session = simulated_reactivation_session(duration=100.0, rng=0)
    assert len(session.spike_times) == 30
    members = [unit for assembly in session.assemblies for unit in assembly]
    assert len(members) == len(set(members)) == 15
    assert session.control_epochs.tolist() == [[0.0, 100.0]]
    assert session.template_epochs.tolist() == [[100.0, 200.0]]
    assert session.match_epochs.tolist() == [[200.0, 300.0]]
    assert all(t.min() >= 0 and t.max() < 300 for t in session.spike_times)


def test_rate_change_raises_only_the_members_rates_after_the_task() -> None:
    session = simulated_reactivation_session(
        duration=2000.0, reactivation=0.0, rate_change=1.0, burst_rate=0.0, rng=1
    )
    members = np.zeros(30, dtype=bool)
    members[[unit for assembly in session.assemblies for unit in assembly]] = True
    rates = {}
    for name, (start, stop) in (
        ("before", session.control_epochs[0]),
        ("after", session.match_epochs[0]),
    ):
        counts = np.array(
            [np.sum((t >= start) & (t < stop)) for t in session.spike_times]
        )
        rates[name] = counts / (stop - start)
    assert rates["after"][members].mean() == pytest.approx(4.0, rel=0.05)
    assert rates["after"][~members].mean() == pytest.approx(2.0, rel=0.05)
    assert rates["before"][members].mean() == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"n_units": 10}, "need 15 units"),
        ({"rate": -1.0}, "rate must be finite and >= 0"),
        ({"reactivation": np.nan}, "reactivation must be finite"),
    ],
)
def test_simulation_refuses_invalid_settings(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulated_reactivation_session(**options)


# ---------------------------------------------------------------------------
# Methods on null data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cell_identity", "circular_shift", "rate_poisson"])
def test_null_match_data_removes_explained_variance(reactivated, kind: str) -> None:
    real = _table(reactivated)
    null = _table(
        reactivated,
        null_reactivation_data(
            reactivated.spike_times, reactivated.match_epochs, kind=kind, rng=7
        ),
    )
    for method in ("kudrimoti_1999", "lansink_2009"):
        before = real[real["method"] == method].iloc[0]
        after = null[null["method"] == method].iloc[0]
        assert before["is_reactivated"]
        assert before["explained_variance"] > 0.5
        assert after["explained_variance"] < 0.05
        assert not after["is_reactivated"]


def test_cell_identity_null_keeps_the_match_periods_own_co_firing(reactivated) -> None:
    """Rest has population bursts in both periods. Permuting identities keeps
    them, so a pattern's mean R(t) after the task falls back to its level
    before it; rotating trains also removes the bursts' co-firing, so R(t)
    after falls below that level for patterns whose weights share a sign
    (on average over the patterns)."""
    tables = {
        kind: _table(
            reactivated,
            null_reactivation_data(
                reactivated.spike_times, reactivated.match_epochs, kind=kind, rng=8
            ),
        )
        for kind in ("cell_identity", "circular_shift")
    }
    identity = tables["cell_identity"].query("method == 'peyrache_2010'")["value"]
    rotated = tables["circular_shift"].query("method == 'peyrache_2010'")["value"]
    assert np.all(np.abs(identity) < 0.1)
    assert rotated.mean() < -0.1


def test_rate_change_alone_is_called_reactivation() -> None:
    """The confound of Tingley & Peyrache (2020): members fire twice as much
    after the task, with no assembly events, during rest whose population
    bursts correlate every pair. Their pair correlations grow, along the
    template's pattern, and the published methods report reactivation."""
    tables = [
        _table(
            simulated_reactivation_session(
                duration=600.0, reactivation=0.0, rate_change=1.0, rng=seed
            ),
            seed=seed,
        )
        for seed in range(3)
    ]
    table = pd.concat(tables, ignore_index=True)
    rates = reactivation_detection_rates(table).set_index("method")
    assert rates.loc["kudrimoti_1999", "rate"] == 1.0
    assert rates.loc["lansink_2009", "rate"] == 1.0
    assert rates.loc["peyrache_2010", "mean_value"] > 0.1


def test_without_reactivation_or_rate_change_the_assembly_difference_is_near_0() -> (
    None
):
    """The verdicts' false-positive rates are calibrated in
    ``test_null_false_positive_rate_is_nominal``."""
    tables = [
        _table(
            simulated_reactivation_session(duration=600.0, reactivation=0.0, rng=seed),
            seed=seed,
        ).assign(session=seed)
        for seed in range(8)
    ]
    rates = reactivation_detection_rates(pd.concat(tables)).set_index("method")
    assert abs(rates.loc["peyrache_2010", "mean_value"]) < 0.05


def _null_false_calls(method, n_sessions: int) -> int:
    """Sessions without reactivation or a rate change that ``method`` calls
    reactivated, one verdict per session."""
    calls = 0
    for seed in range(n_sessions):
        session = simulated_reactivation_session(
            duration=600.0, reactivation=0.0, rng=seed
        )
        table = compare_reactivation_methods(
            session.spike_times,
            session.template_epochs,
            session.match_epochs,
            session.control_epochs,
            reactivation_methods=[method],
            unit_groups=GROUPS,
            rng=seed,
        )
        verdicts = table["is_reactivated"]
        assert len(verdicts) == 1
        assert verdicts.notna().all()
        calls += int(verdicts.sum())
    return calls


# Both verdicts are one-sided at alpha / 2 = 0.025: Kudrimoti's needs the
# two-sided p < 0.05 and a positive partial correlation; Lansink's needs the
# bootstrap's 2.5th percentile of EV - REV above 0. Rest before and rest
# after are the same process here, so a calibrated verdict calls 2.5% of
# sessions. The count is checked against the central 99% binomial interval,
# so a calibrated method fails 1% of seed sets. The session counts make a
# rate of twice the nominal (Kudrimoti, 2000 sessions: interval 33-69 against
# an expected 100) or a tenth of it (Lansink, 1000 sessions: 13-39 against
# 2.5) fall outside the interval.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "n_sessions"),
    [
        pytest.param(
            KUDRIMOTI_ONE_WINDOW,
            2000,
            id="kudrimoti_1999",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "anticonservative: 93 of 2000 null sessions (4.65%) are called "
                    "reactivated against a nominal 2.5% (99% interval 33-69). The "
                    "partial-correlation t test treats pairs as independent, but "
                    "pairs share units, as the recipe's notes say."
                ),
            ),
        ),
        pytest.param(
            LANSINK_ONE_WINDOW,
            1000,
            id="lansink_2009",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "conservative: 1 of 1000 null sessions (0.1%) is called "
                    "reactivated against a nominal 2.5% (99% interval 13-39). "
                    "Not yet explained; a likely cause is that EV and REV are "
                    "both squared partial correlations near 0 under this null, "
                    "where a percentile bootstrap of their difference is too "
                    "wide."
                ),
            ),
        ),
    ],
)
def test_null_false_positive_rate_is_nominal(method, n_sessions: int) -> None:
    nominal = method.alpha / 2
    low, high = binom.ppf([0.005, 0.995], n_sessions, nominal)
    calls = _null_false_calls(method, n_sessions)
    assert low <= calls <= high, f"{calls} of {n_sessions} sessions called"


# ---------------------------------------------------------------------------
# Detection rates
# ---------------------------------------------------------------------------


def test_detection_rates_count_tested_rows_with_exact_intervals() -> None:
    table = pd.DataFrame(
        {
            "method": ["a"] * 5 + ["b"] * 3,
            "session": [0, 0, 1, 1, 1, 0, 1, 1],
            "value": [0.3, 0.1, np.nan, 0.2, 0.0, 1.0, 2.0, np.nan],
            "is_reactivated": pd.array(
                [True, False, pd.NA, True, True, pd.NA, pd.NA, pd.NA], "boolean"
            ),
            "error": ["", "", "too short", "", "", "", "", "too short"],
        }
    )
    rates = reactivation_detection_rates(table, confidence=0.9).set_index("method")
    assert rates.loc["a", "n_rows"] == 5
    assert rates.loc["a", "n_tested"] == 4
    assert rates.loc["a", "n_reactivated"] == 3
    assert rates.loc["a", "n_errors"] == 1
    assert rates.loc["a", "rate"] == 0.75
    interval = binomtest(3, 4).proportion_ci(confidence_level=0.9, method="exact")
    assert rates.loc["a", "ci_low"] == pytest.approx(interval.low)
    assert rates.loc["a", "ci_high"] == pytest.approx(interval.high)
    assert rates.loc["a", "mean_value"] == pytest.approx(0.15)
    assert rates.loc["b", "n_tested"] == 0
    assert np.isnan(rates.loc["b", "rate"])
    assert rates.loc["b", "mean_value"] == pytest.approx(1.5)
    by_session = reactivation_detection_rates(table, groups=["session"])
    assert len(by_session) == 4
    assert (
        by_session.set_index(["method", "session"]).loc[("a", 1), "n_reactivated"] == 2
    )


def test_detection_rates_refuse_bad_input() -> None:
    with pytest.raises(ValueError, match="has no columns"):
        reactivation_detection_rates(pd.DataFrame({"method": ["a"]}))
    table = pd.DataFrame(
        {"method": ["a"], "value": [0.1], "is_reactivated": [True], "error": [""]}
    )
    with pytest.raises(ValueError, match="confidence"):
        reactivation_detection_rates(table, confidence=1.0)
    with pytest.raises(ValueError, match=r"\['session'\]"):
        reactivation_detection_rates(table, groups=["session"])
