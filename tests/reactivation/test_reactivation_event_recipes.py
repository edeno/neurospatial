"""Event-based reactivation recipes: one bin per event, member co-firing,
event-triggered R(t) with joint z-scoring, and the Fernández-Ruiz lab and
El-Gaby recipes, against hand calculations."""

from __future__ import annotations

import dataclasses
import warnings
import zlib

import numpy as np
import pandas as pd
import pytest
from event_session_helpers import EventSession, event_session
from reactivation_helpers import ASSEMBLIES, UNIT_GROUPS
from scipy import stats

from neurospatial.reactivation import (
    CHANG_2025_ASSEMBLY,
    EL_GABY_2021,
    HARVEY_2023_ASSEMBLY,
    LIU_2023_ASSEMBLY,
    LIU_2023_EV,
    LIU_2023_PAIRWISE,
    OLIVA_2020_ASSEMBLY,
    OLIVA_2020_EV,
    ROBINSON_2026_ASSEMBLY,
    ROBINSON_2026_PAIRWISE,
    EventResponse,
    ReactivationMethod,
    ReactivationWindows,
    compare_reactivation_methods,
    detect_assemblies,
    simulated_reactivation_session,
)
from neurospatial.reactivation.methods import _COLUMNS, _event_counts, _period_samples

WHOLE = {
    "control_windows": ReactivationWindows(),
    "match_windows": ReactivationWindows(),
}


@pytest.fixture(scope="module")
def reactivated() -> EventSession:
    return event_session(0)


def _compare(data: EventSession, methods, **options) -> pd.DataFrame:
    arguments = {
        "template_events": data.theta_cycles,
        "control_events": data.control_events,
        "match_events": data.match_events,
        "unit_groups": UNIT_GROUPS,
        "rng": 0,
        **options,
    }
    return compare_reactivation_methods(
        data.spike_times,
        data.template,
        data.match,
        data.control,
        reactivation_methods=methods,
        **arguments,
    )


def _hand_event_counts(spike_times, events: np.ndarray) -> np.ndarray:
    """Spikes in [start, stop) of each event, counted one event at a time."""
    return np.array(
        [
            [np.sum((train >= start) & (train < stop)) for train in spike_times]
            for start, stop in events
        ],
        dtype=float,
    )


def _hand_bins(spike_times, start: float, stop: float, width: float) -> np.ndarray:
    """Counts in whole bins of ``width`` from ``start``, by np.histogram."""
    edges = start + width * np.arange(int(np.floor((stop - start) / width + 1e-9)) + 1)
    return np.column_stack([np.histogram(train, edges)[0] for train in spike_times])


def _pair_vector(counts: np.ndarray) -> np.ndarray:
    return np.corrcoef(counts, rowvar=False)[np.triu_indices(counts.shape[1], k=1)]


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return y - np.polyval(np.polyfit(x, y, 1), x)


# ---------------------------------------------------------------------------
# One bin per event
# ---------------------------------------------------------------------------


def test_event_counts_match_hand_counts_and_keep_short_events() -> None:
    rng = np.random.default_rng(0)
    trains = [np.sort(rng.uniform(0.0, 10.0, 200)) for _ in range(4)]
    # Touching events, a 2 ms event and a spike exactly on a shared edge.
    events = np.array([[1.0, 1.002], [2.0, 2.3], [2.3, 2.6], [5.0, 5.05], [9.0, 9.9]])
    trains[0] = np.sort(np.r_[trains[0], 2.3, 1.001])
    counts = _event_counts(trains, events)
    assert counts.shape == (5, 4)
    np.testing.assert_array_equal(counts, _hand_event_counts(trains, events))
    assert counts[0, 0] >= 1  # the spike in the 2 ms event is counted


def test_event_binned_explained_variance_matches_a_hand_calculation(
    reactivated: EventSession,
) -> None:
    """OLIVA_2020_EV: 100 ms task bins, one bin per SWR, EV - REV; every SWR
    is a bin, including those shorter than 100 ms."""
    data = reactivated
    table = _compare(data, [OLIVA_2020_EV])
    template = _hand_bins(data.spike_times, *data.template[0], 0.1)
    control = _hand_event_counts(data.spike_times, data.control_events)
    match = _hand_event_counts(data.spike_times, data.match_events)
    vectors = [_pair_vector(c) for c in (template, match, control)]
    r = np.corrcoef(vectors)
    r_tm, r_tc, r_mc = r[0, 1], r[0, 2], r[1, 2]
    ev = (r_tm - r_tc * r_mc) ** 2 / ((1 - r_tc**2) * (1 - r_mc**2))
    rev = (r_tc - r_tm * r_mc) ** 2 / ((1 - r_tm**2) * (1 - r_mc**2))
    row = table.iloc[0]
    assert row["explained_variance"] == pytest.approx(ev, rel=1e-9)
    assert row["reversed_explained_variance"] == pytest.approx(rev, rel=1e-9)
    assert row["value"] == pytest.approx(ev - rev, rel=1e-9)
    durations = data.control_events[:, 1] - data.control_events[:, 0]
    assert np.sum(durations < 0.1) > 100
    assert row["n_control_events"] == len(data.control_events)
    assert row["n_match_events"] == len(data.match_events)


def test_theta_cycle_template_and_residual_correlations(
    reactivated: EventSession,
) -> None:
    """LIU_2023_EV, LIU_2023_PAIRWISE (semipartial) and
    ROBINSON_2026_PAIRWISE (partial) against regressions of hand vectors."""
    data = reactivated
    table = _compare(data, [LIU_2023_EV, LIU_2023_PAIRWISE, ROBINSON_2026_PAIRWISE])
    theta = _pair_vector(_hand_event_counts(data.spike_times, data.theta_cycles))
    running = _pair_vector(_hand_bins(data.spike_times, *data.template[0], 0.1))
    control = _pair_vector(_hand_event_counts(data.spike_times, data.control_events))
    match = _pair_vector(_hand_event_counts(data.spike_times, data.match_events))
    values = table.set_index("method")["value"]
    partial = np.corrcoef(_residual(theta, control), _residual(match, control))[0, 1]
    assert values["liu_2023_ev"] == pytest.approx(partial**2, rel=1e-9)
    assert values["liu_2023_pairwise"] == pytest.approx(
        np.corrcoef(theta, _residual(match, control))[0, 1], rel=1e-9
    )
    robinson = np.corrcoef(_residual(running, control), _residual(match, control))[0, 1]
    assert values["robinson_2026_pairwise"] == pytest.approx(robinson, rel=1e-9)


def test_robinson_pairwise_is_the_partial_correlation_of_ev(
    reactivated: EventSession,
) -> None:
    """Same bins as OLIVA_2020_EV: the residual correlation is its partial r,
    whose square is its EV."""
    table = _compare(reactivated, [OLIVA_2020_EV, ROBINSON_2026_PAIRWISE]).set_index(
        "method"
    )
    robinson = table.loc["robinson_2026_pairwise"]
    oliva = table.loc["oliva_2020_ev"]
    assert robinson["value"] == robinson["partial_correlation"]
    assert robinson["value"] == pytest.approx(oliva["partial_correlation"], rel=1e-12)
    assert robinson["value"] ** 2 == pytest.approx(
        oliva["explained_variance"], rel=1e-9
    )


def test_only_events_starting_in_the_window_are_used(reactivated: EventSession) -> None:
    data = reactivated
    extra = np.array([[700.0, 700.1], [900.0, 900.05]])  # inside the task
    with_extra = _compare(
        data,
        [OLIVA_2020_EV],
        control_events=np.vstack([data.control_events, extra]),
    )
    plain = _compare(data, [OLIVA_2020_EV])
    assert with_extra["n_control_events"].iloc[0] == len(data.control_events)
    assert with_extra["value"].iloc[0] == plain["value"].iloc[0]


def test_event_input_errors(reactivated: EventSession) -> None:
    data = reactivated
    overlapping = np.array([[10.0, 10.2], [10.1, 10.3]])
    with pytest.raises(ValueError, match="overlap the next one"):
        _compare(data, [OLIVA_2020_EV], control_events=overlapping)
    with pytest.raises(ValueError, match=r"\['match_events'\] not given"):
        _compare(data, [OLIVA_2020_EV], match_events=None)
    with pytest.raises(ValueError, match="template_events"):
        _compare(data, [LIU_2023_EV], template_events=None)
    with pytest.raises(ValueError, match="stop > start"):
        _compare(data, [OLIVA_2020_EV], control_events=[[5.0, 5.0]])
    # Methods without events need none.
    _compare(
        data,
        [dataclasses.replace(OLIVA_2020_EV, rest_bins="fixed")],
        control_events=None,
        match_events=None,
    )


def test_min_events_gives_an_error_row(reactivated: EventSession) -> None:
    data = reactivated
    n_control, n_match = len(data.control_events), len(data.match_events)
    assert n_match < n_control
    too_many = dataclasses.replace(EL_GABY_2021, min_events=n_match + 1)
    enough = dataclasses.replace(EL_GABY_2021, name="enough", min_events=n_match)
    table = _compare(data, [too_many, enough])
    failed = table[table["method"] == "el_gaby_2021"]
    assert len(failed) == 1
    assert (
        f"match window has {n_match} events, fewer than min_events"
        in (failed["error"].iloc[0])
    )
    assert failed["value"].isna().all()
    assert (table.loc[table["method"] == "enough", "error"] == "").all()


# ---------------------------------------------------------------------------
# Member co-firing (El-Gaby et al. 2021)
# ---------------------------------------------------------------------------


def test_el_gaby_rows_are_member_correlations_over_swrs(
    reactivated: EventSession,
) -> None:
    data = reactivated
    method = dataclasses.replace(EL_GABY_2021, min_events=100)
    table = _compare(data, [method])
    control = _hand_event_counts(data.spike_times, data.control_events)
    match = _hand_event_counts(data.spike_times, data.match_events)
    assert table["n_components"].iloc[0] == len(table) == len(ASSEMBLIES)
    for _, row in table.iterrows():
        members = list(row["member_unit_ids"])
        assert len(members) >= 2
        upper = np.triu_indices(len(members), k=1)
        before = np.corrcoef(control[:, members], rowvar=False)[upper].mean()
        after = np.corrcoef(match[:, members], rowvar=False)[upper].mean()
        assert row["control_member_correlation"] == pytest.approx(before)
        assert row["match_member_correlation"] == pytest.approx(after)
        assert row["value"] == pytest.approx(after - before)
        assert row["n_member_pairs"] == len(upper[0])
    assert sorted(tuple(sorted(r)) for r in table["member_unit_ids"]) == sorted(
        ASSEMBLIES
    )


# ---------------------------------------------------------------------------
# Event-triggered R(t) with joint z-scoring
# ---------------------------------------------------------------------------


def _detection_seed(name: str, rng: int = 0) -> int:
    generator = np.random.default_rng(
        np.random.SeedSequence(
            np.random.SeedSequence(rng).entropy, spawn_key=(zlib.crc32(name.encode()),)
        )
    )
    return int(generator.integers(2**32 - 1))


def _hand_sliding_counts(spike_times, start, stop, width, step):
    """Counts in [left, left + width) for left = start, start + step, ..."""
    lefts = start + step * np.arange(
        int(np.floor((stop - start - width) / step + 1e-9)) + 1
    )
    counts = np.column_stack(
        [
            np.searchsorted(train, lefts + width, side="left")
            - np.searchsorted(train, lefts, side="left")
            for train in spike_times
        ]
    )
    return counts.astype(float), lefts + width / 2


def _hand_responses(times, activation, step, onsets, lags, window):
    """Mean R over the lags in ``window`` after each onset, bin holding t."""
    keep = (lags >= window[0] - 1e-9) & (lags < window[1] - 1e-9)
    responses = []
    for onset in onsets:
        values = []
        for lag in lags[keep]:
            query = onset + lag
            inside = (times - step / 2 <= query + 1e-9) & (
                query < times + step / 2 - 1e-9
            )
            if inside.any():
                values.append(activation[np.flatnonzero(inside)[0]])
        responses.append(np.mean(values) if values else np.nan)
    return np.array(responses)


def test_chang_rows_match_joint_z_scored_sliding_bins(
    reactivated: EventSession,
) -> None:
    """50 ms bins in 10 ms steps, z-scored over both rest periods together,
    each SWR's mean R over 0-100 ms and its -500 to -100 ms baseline."""
    data = reactivated
    table = _compare(data, [CHANG_2025_ASSEMBLY])
    template = _hand_bins(data.spike_times, *data.template[0], 0.05)
    detection = detect_assemblies(
        template,
        member_rule="positive_z",
        member_threshold=3.0,
        rng=_detection_seed("chang_2025_assembly"),
    )
    assert len(table) == len(detection.patterns)
    control, control_times = _hand_sliding_counts(
        data.spike_times, *data.control[0], 0.05, 0.01
    )
    match, match_times = _hand_sliding_counts(
        data.spike_times, *data.match[0], 0.05, 0.01
    )
    stacked = np.vstack([control, match])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    lags = np.arange(-50, 50) * 0.01
    for component, pattern in enumerate(detection.patterns):
        w = pattern.weights / np.linalg.norm(pattern.weights)
        responses = {}
        for name, counts, times, events in (
            ("control", control, control_times, data.control_events),
            ("match", match, match_times, data.match_events),
        ):
            z = (counts - mean) / std
            activation = (z @ w) ** 2 - z**2 @ w**2
            responses[name] = _hand_responses(
                times, activation, 0.01, events[:, 0], lags, (0.0, 0.1)
            )
            if name == "match":
                baseline = _hand_responses(
                    times, activation, 0.01, events[:, 0], lags, (-0.5, -0.1)
                )
        row = table.iloc[component]
        before = np.nanmean(responses["control"])
        after = np.nanmean(responses["match"])
        assert row["control_event_response"] == pytest.approx(before, rel=1e-9)
        assert row["match_event_response"] == pytest.approx(after, rel=1e-9)
        assert row["value"] == pytest.approx((after - before) / before, rel=1e-9)
        both = np.isfinite(responses["match"]) & np.isfinite(baseline)
        differences = responses["match"][both] - baseline[both]
        p_value = stats.wilcoxon(differences).pvalue
        assert row["p_value"] == pytest.approx(p_value, rel=1e-9)
        assert row["is_reactivated"] == (p_value < 0.05 and np.median(differences) > 0)


def test_sliding_samples_match_hand_counts(reactivated: EventSession) -> None:
    data = reactivated
    samples = _period_samples(CHANG_2025_ASSEMBLY, data.spike_times, data.control)
    counts, times = _hand_sliding_counts(data.spike_times, *data.control[0], 0.05, 0.01)
    np.testing.assert_array_equal(samples.values, counts)
    np.testing.assert_allclose(samples.times, times, atol=1e-9)


def test_z_scoring_choice_moves_a_rate_change_in_opposite_directions() -> None:
    """Members firing twice as much after the task without co-firing: their
    SWR response rises with the periods z-scored together (higher z-scores
    after the task) and falls with each z-scored on its own (a larger
    variance after the task shrinks the z-scores). Neither is 0."""
    data = event_session(1, reactivation=0.0, rate_change=1.0)
    period = ReactivationMethod(
        name="period",
        reference="test",
        measure="assembly",
        bin_size=0.025,
        detection="ica",
        event_response=EventResponse(),
        statistic="event_response_difference",
    )
    joint = dataclasses.replace(period, name="joint", zscore_over="control_and_match")
    table = _compare(data, [period, joint]).set_index(["method", "component"])
    assert table.loc["joint", "value"].min() > 0.3
    assert table.loc["period", "value"].max() < -1.0


# ---------------------------------------------------------------------------
# Recipes run and fill their columns
# ---------------------------------------------------------------------------


def test_compare_table_rows_for_the_event_recipes(reactivated: EventSession) -> None:
    methods = [
        OLIVA_2020_EV,
        OLIVA_2020_ASSEMBLY,
        dataclasses.replace(EL_GABY_2021, min_events=100),
        dataclasses.replace(HARVEY_2023_ASSEMBLY, **WHOLE),
        LIU_2023_EV,
        LIU_2023_PAIRWISE,
        dataclasses.replace(LIU_2023_ASSEMBLY, **WHOLE),
        CHANG_2025_ASSEMBLY,
        ROBINSON_2026_PAIRWISE,
        ROBINSON_2026_ASSEMBLY,
    ]
    table = _compare(reactivated, methods)
    assert tuple(table.columns) == _COLUMNS
    assert (table["error"] == "").all(), table["error"].unique()
    counts = table.groupby("method", sort=False).size()
    for name in ("oliva_2020_ev", "liu_2023_ev", "liu_2023_pairwise"):
        assert counts[name] == 1
    for name in ("el_gaby_2021", "harvey_2023_assembly", "chang_2025_assembly"):
        rows = table[table["method"] == name]
        assert len(rows) == rows["n_components"].iloc[0] >= 1
    event_rows = table[table["method"] != "oliva_2020_assembly"]
    assert (event_rows["n_match_events"] == len(reactivated.match_events)).all()
    assert table["n_control_events"].dtype == "Int64"
    harvey = table[table["method"] == "harvey_2023_assembly"]
    assert np.isfinite(harvey[["control_event_response", "match_event_response"]]).all(
        axis=None
    )
    robinson = table[table["method"] == "robinson_2026_assembly"]
    assert set(robinson["region"]) <= {"0", "1", "cross_region"}
    regions = UNIT_GROUPS["region"]
    for _, row in robinson.iterrows():
        member_regions = set(regions[list(row["member_unit_ids"])])
        if row["region"] == "cross_region":
            assert len(member_regions) == 2
        else:
            assert member_regions <= {int(row["region"])}
        for column in (
            "control_significant_event_fraction",
            "match_significant_event_fraction",
        ):
            assert 0.0 <= row[column] <= 1.0
    assert "cross_region" in set(robinson["region"])
    chang = table[table["method"] == "chang_2025_assembly"]
    assert chang["is_reactivated"].notna().all()
    assert table.loc[table["region"].notna(), "method"].unique().tolist() == [
        "robinson_2026_assembly"
    ]


def test_null_data_gives_no_event_reactivation_within_bounds() -> None:
    """Rest after the task without assembly co-firing: the event-based
    statistics stay near 0 over sessions; with co-firing in half the SWRs
    they leave those bounds."""
    methods = [
        OLIVA_2020_EV,
        LIU_2023_EV,
        LIU_2023_PAIRWISE,
        ROBINSON_2026_PAIRWISE,
        dataclasses.replace(EL_GABY_2021, min_events=100),
        dataclasses.replace(LIU_2023_ASSEMBLY, **WHOLE),
        CHANG_2025_ASSEMBLY,
    ]
    # Bounds: EV of about 300 SWRs' pair correlations without reactivation
    # is a squared correlation of noise; the others are differences of noisy
    # means.
    bounds = {
        "oliva_2020_ev": 0.03,
        "liu_2023_ev": 0.03,
        "liu_2023_pairwise": 0.15,
        "robinson_2026_pairwise": 0.15,
        "el_gaby_2021": 0.06,
        "liu_2023_assembly": 1.5,
        "chang_2025_assembly": 0.4,
    }
    means = {}
    for reactivation in (0.0, 0.5):
        tables = []
        for seed in range(4):
            data = event_session(10 + seed, reactivation=reactivation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables.append(_compare(data, methods, rng=seed))
        table = pd.concat(tables)
        assert (table["error"] == "").all()
        means[reactivation] = table.groupby("method")["value"].mean()
        if reactivation == 0.0:
            spread = table.groupby("method")["value"].apply(lambda v: v.abs().max())
            for name, bound in bounds.items():
                assert abs(means[0.0][name]) < bound / 2, name
                assert spread[name] < bound, name
    for name, bound in bounds.items():
        assert means[0.5][name] > bound, name


def test_simulated_session_returns_its_bursts_as_events() -> None:
    session = simulated_reactivation_session(n_units=12, n_assemblies=2, rng=0)
    for events, epochs in (
        (session.control_events, session.control_epochs),
        (session.match_events, session.match_epochs),
    ):
        assert len(events) > 400
        assert np.all(events[1:, 0] > events[:-1, 1])  # merged: disjoint
        assert np.all(events[:, 1] - events[:, 0] >= 0.1 - 1e-12)
        assert events[0, 0] >= epochs[0, 0] and events[-1, 1] <= epochs[0, 1]


# ---------------------------------------------------------------------------
# Recipe validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "options", "message"),
    [
        (LIU_2023_EV, {"bin_size": 0.1}, "bin_size must be None"),
        (OLIVA_2020_EV, {"bin_size": None}, "bin_size must be a positive"),
        (OLIVA_2020_ASSEMBLY, {"rest_bins": "events"}, "does not apply to an assembly"),
        (EL_GABY_2021, {"member_threshold": None}, "needs a finite member_threshold"),
        (HARVEY_2023_ASSEMBLY, {"member_threshold": 2.0}, "member_threshold=None"),
        (OLIVA_2020_ASSEMBLY, {"min_events": 10}, "min_events must be"),
        (
            OLIVA_2020_ASSEMBLY,
            {"statistic": "event_response_difference"},
            "needs event_response",
        ),
        (LIU_2023_ASSEMBLY, {"test": "event_modulation"}, "baseline_window"),
        (CHANG_2025_ASSEMBLY, {"activation_step": 0.03}, "must divide bin_size"),
        (OLIVA_2020_EV, {"zscore_over": "control_and_match"}, "do not apply"),
        (EL_GABY_2021, {"smoothing_bandwidth": 0.01}, "do not apply"),
        (OLIVA_2020_EV, {"statistic": "event_response_ratio"}, "does not fit measure"),
    ],
)
def test_invalid_event_recipes(base, options, message) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **options)


def test_event_response_options_are_checked() -> None:
    with pytest.raises(ValueError, match="inside window"):
        EventResponse(window=(-0.2, 0.2), baseline_window=(-0.5, -0.1))
    with pytest.raises(ValueError, match="start < stop"):
        EventResponse(response_window=(0.1, 0.0))


def test_needs_events_names_the_arguments() -> None:
    assert LIU_2023_EV.needs_events == (
        "template_events",
        "control_events",
        "match_events",
    )
    assert CHANG_2025_ASSEMBLY.needs_events == ("control_events", "match_events")
    assert OLIVA_2020_ASSEMBLY.needs_events == ()


def test_regions_are_required_for_per_region_detection(reactivated) -> None:
    with pytest.raises(ValueError, match=r"unit_groups has no \['region'\]"):
        _compare(reactivated, [ROBINSON_2026_ASSEMBLY], unit_groups={})
