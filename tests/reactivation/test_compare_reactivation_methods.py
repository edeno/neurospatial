"""compare_reactivation_methods: windows, statistics, tests and streams."""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np
import pandas as pd
import pytest
from reactivation_helpers import ASSEMBLIES, N_UNITS, UNIT_GROUPS, session
from scipy import stats

from neurospatial.reactivation import (
    GIRI_2019,
    KUDRIMOTI_1999,
    LANSINK_2009,
    PENNARTZ_2004,
    PEYRACHE_2010,
    TATSUNO_2006,
    VAN_DE_VEN_2016,
    ReactivationMethod,
    ReactivationWindows,
    compare_reactivation_methods,
    detect_assemblies,
)
from neurospatial.reactivation.methods import _COLUMNS, _bootstrap_ev_minus_rev

# Fewer shuffles keep the tests fast; the recipes' statistics are unchanged.
LANSINK_FAST = dataclasses.replace(LANSINK_2009, n_shuffles=500)
PEYRACHE_FAST = dataclasses.replace(PEYRACHE_2010, n_shuffles=100)


@pytest.fixture(scope="module")
def reactivated() -> tuple:
    return session(0)


def _compare(data: tuple, methods, **options):
    spike_times, template, match, control = data
    return compare_reactivation_methods(
        spike_times,
        template,
        match,
        control,
        reactivation_methods=methods,
        unit_groups=UNIT_GROUPS,
        **{"rng": 0, **options},
    )


def _hand_counts(spike_times, window: np.ndarray, bin_size: float) -> np.ndarray:
    """Counts per bin by np.histogram, each window piece tiled on its own."""
    blocks = []
    for start, stop in window:
        n_bins = int(np.floor((stop - start) / bin_size + 1e-9))
        edges = start + bin_size * np.arange(n_bins + 1)
        blocks.append(
            np.column_stack([np.histogram(train, edges)[0] for train in spike_times])
        )
    return np.vstack(blocks).astype(float)


def _hand_ev(template, match, control, keep):
    """Kudrimoti's EV and REV with np.corrcoef on the kept pairs."""
    upper = np.triu_indices(template.shape[1], k=1)
    vectors = [
        np.corrcoef(c, rowvar=False)[upper][keep] for c in (template, match, control)
    ]
    r = np.corrcoef(vectors)
    r_tm, r_tc, r_mc = r[0, 1], r[0, 2], r[1, 2]
    partial = (r_tm - r_tc * r_mc) / np.sqrt((1 - r_tc**2) * (1 - r_mc**2))
    reversed_partial = (r_tc - r_tm * r_mc) / np.sqrt((1 - r_tm**2) * (1 - r_mc**2))
    return partial, reversed_partial**2


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


class TestReactivationWindows:
    epochs = np.array([[0.0, 400.0], [500.0, 900.0]])

    def test_whole_period_is_one_window(self) -> None:
        windows = ReactivationWindows().split(self.epochs)
        assert len(windows) == 1
        np.testing.assert_array_equal(windows[0], self.epochs)

    def test_concatenated_blocks_join_the_epochs(self) -> None:
        windows = ReactivationWindows(length=300.0).split(self.epochs)
        assert [w.tolist() for w in windows] == [
            [[0.0, 300.0]],
            [[300.0, 400.0], [500.0, 700.0]],
        ]

    def test_wall_blocks_keep_the_gap(self) -> None:
        windows = ReactivationWindows(length=300.0, clock="wall").split(self.epochs)
        # Clock 0-900: [0, 300), [300, 600) holding 200 s, [600, 900).
        assert [w.tolist() for w in windows] == [
            [[0.0, 300.0]],
            [[300.0, 400.0], [500.0, 600.0]],
            [[600.0, 900.0]],
        ]

    def test_last_window_from_the_end(self) -> None:
        concatenated = ReactivationWindows(length=300.0, anchor="end", count=1)
        assert [w.tolist() for w in concatenated.split(self.epochs)] == [
            [[600.0, 900.0]]
        ]
        two = ReactivationWindows(length=300.0, anchor="end", count=2)
        # Concatenated time 200-500 s is clock 200-400 and 500-600.
        assert [w.tolist() for w in two.split(self.epochs)] == [
            [[200.0, 400.0], [500.0, 600.0]],
            [[600.0, 900.0]],
        ]

    def test_sliding_windows(self) -> None:
        windows = ReactivationWindows(length=900.0, step=300.0).split([[0.0, 3000.0]])
        starts = [w[0, 0] for w in windows]
        # Whole 15-min windows every 5 min in 50 min: starts 0 ... 2100 s.
        assert starts == [0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0]
        assert all(w[0, 1] - w[0, 0] == 900.0 for w in windows)

    def test_remainder_and_short_periods(self) -> None:
        rule = ReactivationWindows(length=600.0)
        assert len(rule.split([[0.0, 1700.0]])) == 2
        assert rule.split([[0.0, 500.0]]) == []
        assert len(ReactivationWindows(length=600.0, count=3).split([[0, 6000.0]])) == 3

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"length": -1.0}, "length must be a positive"),
            ({"step": 10.0}, "need length"),
            ({"length": 10.0, "anchor": "middle"}, "anchor must be"),
            ({"length": 10.0, "clock": "sleep"}, "clock must be"),
            ({"length": 10.0, "count": 0}, "count must be a positive"),
        ],
    )
    def test_invalid_options(self, options: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            ReactivationWindows(**options)


# ---------------------------------------------------------------------------
# Recipe validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"measure": "rate"}, "measure must be one of"),
        ({"bin_size": 0.0}, "bin_size must be a positive"),
        ({"statistic": "mean_activation_difference"}, "does not fit measure"),
        ({"test": "cell_identity_shuffle"}, "does not fit measure"),
        ({"percentile": 99.0}, "do not apply to explained_variance methods"),
        ({"n_shuffles": 0}, "n_shuffles must be a positive integer"),
        ({"alpha": 1.0}, r"alpha must be in \(0, 1\)"),
    ],
)
def test_invalid_explained_variance_recipes(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(KUDRIMOTI_1999, **options)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"detection": None}, "needs detection"),
        ({"pairs": "different_tetrodes"}, "applies only to EV methods"),
        ({"percentile": None}, "pass percentile"),
    ],
)
def test_invalid_assembly_recipes(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(PEYRACHE_2010, **options)


# ---------------------------------------------------------------------------
# Explained variance against hand calculations
# ---------------------------------------------------------------------------


def test_kudrimoti_rows_match_a_hand_calculation(reactivated) -> None:
    spike_times, template, _match, _control = reactivated
    table = _compare(reactivated, [KUDRIMOTI_1999])
    # 1200 s of rest after: two whole 10-min blocks of the three asked for.
    assert table["window"].tolist() == [0, 1]
    first, second = np.triu_indices(N_UNITS, k=1)
    tetrode = UNIT_GROUPS["tetrode"]
    keep = tetrode[first] != tetrode[second]
    template_counts = _hand_counts(spike_times, template, 0.1)
    control_counts = _hand_counts(spike_times, np.array([[600.0, 1200.0]]), 0.1)
    for window, (start, stop) in enumerate([(2400.0, 3000.0), (3000.0, 3600.0)]):
        match_counts = _hand_counts(spike_times, np.array([[start, stop]]), 0.1)
        partial, reversed_ev = _hand_ev(
            template_counts, match_counts, control_counts, keep
        )
        row = table.iloc[window]
        assert row["window_start"] == start and row["window_stop"] == stop
        assert row["n_pairs"] == keep.sum() == 276 - 12
        assert row["explained_variance"] == pytest.approx(partial**2, rel=1e-9)
        assert row["value"] == row["explained_variance"]
        assert row["reversed_explained_variance"] == pytest.approx(
            reversed_ev, rel=1e-9
        )
        assert row["partial_correlation"] == pytest.approx(partial, rel=1e-9)
        # The partial-correlation t test with n_pairs - 3 degrees of freedom.
        df = keep.sum() - 3
        t = partial * np.sqrt(df / (1 - partial**2))
        assert row["p_value"] == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-6)
        assert row["is_reactivated"]


def test_several_control_windows_average_single_control_evs(reactivated) -> None:
    """Tatsuno et al. (2006): EV(POST(h)) = mean over k of R²(EXP, POST(h) | PRE(k))."""
    spike_times, template, _match, _control = reactivated
    method = dataclasses.replace(
        TATSUNO_2006,
        control_windows=ReactivationWindows(length=400.0, anchor="end", clock="wall"),
        match_windows=ReactivationWindows(length=600.0, clock="wall"),
    )
    table = _compare(reactivated, [method])
    assert table["n_control_windows"].tolist() == [3, 3]
    first, second = np.triu_indices(N_UNITS, k=1)
    keep = UNIT_GROUPS["tetrode"][first] != UNIT_GROUPS["tetrode"][second]
    template_counts = _hand_counts(spike_times, template, 0.25)
    match_counts = _hand_counts(spike_times, np.array([[2400.0, 3000.0]]), 0.25)
    evs, revs = [], []
    for start in (0.0, 400.0, 800.0):
        control_counts = _hand_counts(
            spike_times, np.array([[start, start + 400]]), 0.25
        )
        partial, reversed_ev = _hand_ev(
            template_counts, match_counts, control_counts, keep
        )
        evs.append(partial**2)
        revs.append(reversed_ev)
    row = table.iloc[0]
    assert row["explained_variance"] == pytest.approx(np.mean(evs), rel=1e-9)
    assert row["reversed_explained_variance"] == pytest.approx(np.mean(revs), rel=1e-9)
    # With several controls there is no single partial correlation or p-value.
    assert np.isnan(row["partial_correlation"]) and np.isnan(row["p_value"])
    assert pd.isna(row["is_reactivated"])


def test_pair_rules_choose_the_pairs(reactivated) -> None:
    table = _compare(reactivated, [LANSINK_FAST, GIRI_2019, PENNARTZ_2004])
    n_pairs = table.groupby("method")["n_pairs"].first()
    # 12 x 12 cross-region pairs; pairs on different shanks of 4; all pairs.
    assert n_pairs["lansink_2009"] == 144
    assert n_pairs["giri_2019"] == 276 - 6 * 6
    assert n_pairs["pennartz_2004"] == 276


def test_pair_mask_is_combined_with_the_pair_rule(reactivated) -> None:
    n_pairs = N_UNITS * (N_UNITS - 1) // 2
    mask = np.zeros(n_pairs, dtype=bool)
    mask[:40] = True
    table = _compare(reactivated, [PENNARTZ_2004], pair_mask=mask)
    assert (table["n_pairs"] == 40).all()


def test_bootstrap_matches_resampled_pair_triplets() -> None:
    rng = np.random.default_rng(1)
    template = rng.standard_normal(30)
    control = 0.5 * template + rng.standard_normal(30)
    match = template + 0.3 * control + rng.standard_normal(30)
    differences = _bootstrap_ev_minus_rev(
        template, match, control, 20, np.random.default_rng(5)
    )
    draws = np.random.default_rng(5).multinomial(30, np.full(30, 1 / 30), 20)
    for difference, counts in zip(differences, draws, strict=True):
        index = np.repeat(np.arange(30), counts)
        r = np.corrcoef([template[index], match[index], control[index]])
        r_tm, r_tc, r_mc = r[0, 1], r[0, 2], r[1, 2]
        ev = (r_tm - r_tc * r_mc) ** 2 / ((1 - r_tc**2) * (1 - r_mc**2))
        rev = (r_tc - r_tm * r_mc) ** 2 / ((1 - r_tm**2) * (1 - r_mc**2))
        assert difference == pytest.approx(ev - rev, rel=1e-8)


def test_lansink_verdict_is_the_lower_percentile_of_ev_minus_rev(reactivated) -> None:
    spike_times, template, match, control = reactivated
    table = _compare(reactivated, [LANSINK_FAST])
    row = table.iloc[0]
    assert row["n_shuffles_used"] == 500
    assert row["value"] == pytest.approx(
        row["explained_variance"] - row["reversed_explained_variance"]
    )
    # Replay the method's stream for match window 0 on hand-made
    # correlations of the cross-region pairs.
    first, second = np.triu_indices(N_UNITS, k=1)
    keep = UNIT_GROUPS["region"][first] != UNIT_GROUPS["region"][second]
    upper = np.triu_indices(N_UNITS, k=1)
    vectors = [
        np.corrcoef(_hand_counts(spike_times, period, 0.05), rowvar=False)[upper][keep]
        for period in (template, match, control)
    ]
    generator = np.random.default_rng(
        np.random.SeedSequence(
            table.attrs["rng_entropy"],
            spawn_key=(zlib.crc32(b"lansink_2009"), 0, 0),
        )
    )
    differences = _bootstrap_ev_minus_rev(*vectors, 500, generator)
    assert row["ev_minus_rev_lower"] == pytest.approx(
        np.percentile(differences, 2.5), rel=1e-9
    )
    assert 0 < row["ev_minus_rev_lower"] < row["value"]
    assert row["is_reactivated"]


# ---------------------------------------------------------------------------
# Assembly methods
# ---------------------------------------------------------------------------


def test_mean_activation_is_gamma_minus_one(reactivated) -> None:
    """Peyrache et al. (2010): the average reactivation strength is γ - 1,
    with γ = w' C w for the period's correlation matrix C."""
    spike_times, template, match, control = reactivated
    table = _compare(reactivated, [PEYRACHE_FAST])
    detection = detect_assemblies(
        _hand_counts(spike_times, template, 0.1), method="pca"
    )
    assert table["n_components"].iloc[0] == len(detection.patterns) == 2
    for period, column in (
        (match, "match_mean_activation"),
        (control, "control_mean_activation"),
    ):
        correlation = np.corrcoef(_hand_counts(spike_times, period, 0.1), rowvar=False)
        for component, pattern in enumerate(detection.patterns):
            w = pattern.weights / np.linalg.norm(pattern.weights)
            assert table[column].iloc[component] == pytest.approx(
                w @ correlation @ w - 1, rel=1e-9
            )
    assert np.allclose(
        table["value"],
        table["match_mean_activation"] - table["control_mean_activation"],
    )


def test_peyrache_shares_on_reactivated_data(reactivated) -> None:
    table = _compare(reactivated, [PEYRACHE_FAST])
    # Rest before has no co-firing: about 1% of bins above each bin's
    # shuffle 99th percentile (Peyrache et al. 2010: 1% in PRE).
    assert (table["control_fraction_above_shuffle"] < 0.03).all()
    assert (table["match_fraction_above_shuffle"] > 0.03).all()
    assert (table["fraction_above_control_threshold"] > 0.03).all()
    assert (table["n_shuffles_used"] == 100).all()
    assert table["is_reactivated"].isna().all()


def test_van_de_ven_activations_follow_the_events(reactivated) -> None:
    table = _compare(reactivated, [VAN_DE_VEN_2016])
    assert len(table) == 2
    assert {tuple(sorted(m)) for m in table["member_unit_ids"]} == set(ASSEMBLIES)
    assert (table["value"] > 0.5).all()
    # Two members spiking within ~10 ms by chance also exceed 5, so rest
    # before has activations too; the 1 Hz events add about one per second.
    assert (
        table["match_activation_rate"] > table["control_activation_rate"] + 0.3
    ).all()


# ---------------------------------------------------------------------------
# Without reactivation
# ---------------------------------------------------------------------------


def test_no_reactivation_gives_few_verdicts_and_small_effects() -> None:
    """Rest after running without assembly events: the verdicts stay near
    their level and the effect sizes near 0."""
    calls = {"kudrimoti_1999": 0, "lansink_2009": 0}
    differences, shares = [], []
    lansink = dataclasses.replace(LANSINK_2009, n_shuffles=200)
    peyrache = dataclasses.replace(PEYRACHE_2010, n_shuffles=200)
    n_sessions = 12
    for seed in range(n_sessions):
        table = _compare(
            session(100 + seed, reactivation=0.0, duration=600.0),
            [
                dataclasses.replace(
                    KUDRIMOTI_1999, match_windows=ReactivationWindows(length=600.0)
                ),
                lansink.__class__(
                    **{
                        **dataclasses.asdict(lansink),
                        "control_windows": ReactivationWindows(),
                        "match_windows": ReactivationWindows(),
                    }
                ),
                peyrache,
            ],
        )
        verdicts = table.dropna(subset=["is_reactivated"])
        for name in calls:
            calls[name] += int(
                verdicts.loc[verdicts["method"] == name, "is_reactivated"].sum()
            )
        assembly = table[table["method"] == "peyrache_2010"]
        differences.extend(assembly["value"])
        shares.extend(assembly["match_fraction_above_shuffle"])
    # Binomial(12, 0.05) exceeds 3 with probability below 0.003.
    assert calls["kudrimoti_1999"] <= 3
    assert calls["lansink_2009"] <= 3
    # Mean R(t) of a period without co-firing is 0 up to about sqrt(2 / n)
    # for n = 6000 bins; the difference of two periods within a few times that.
    assert np.max(np.abs(differences)) < 0.15
    # A bin whose R(t) is exchangeable with its 200 shuffles beats their
    # interpolated 99th percentile (between the 3rd and 2nd largest) when it
    # ranks in the top 3 of 201: 3 / 201, about 1.5%.
    assert np.mean(shares) == pytest.approx(3 / 201, abs=0.004)


# ---------------------------------------------------------------------------
# Table, errors and streams
# ---------------------------------------------------------------------------


def test_table_has_one_row_per_method_window_and_component(reactivated) -> None:
    table = _compare(reactivated, [KUDRIMOTI_1999, PEYRACHE_FAST, GIRI_2019])
    assert tuple(table.columns) == _COLUMNS
    counts = table.groupby("method").size()
    # 2 match windows; 2 components x 1 window; 1200 s holds two 15-min
    # windows sliding by 5 min.
    assert counts.to_dict() == {
        "giri_2019": 2,
        "kudrimoti_1999": 2,
        "peyrache_2010": 2,
    }
    assert table.loc[table["method"] == "kudrimoti_1999", "component"].isna().all()
    assert table.loc[table["method"] == "peyrache_2010", "component"].tolist() == [0, 1]
    assert (table["error"] == "").all()


def test_streams_do_not_depend_on_the_other_methods(reactivated) -> None:
    alone = _compare(reactivated, [LANSINK_FAST], rng=3)
    together = _compare(reactivated, [PEYRACHE_FAST, LANSINK_FAST], rng=3)
    np.testing.assert_array_equal(
        alone["ev_minus_rev_lower"],
        together.loc[together["method"] == "lansink_2009", "ev_minus_rev_lower"],
    )
    other_seed = _compare(reactivated, [LANSINK_FAST], rng=4)
    assert (
        other_seed["ev_minus_rev_lower"].iloc[0] != alone["ev_minus_rev_lower"].iloc[0]
    )


def test_rng_entropy_reproduces_the_table(reactivated) -> None:
    first = _compare(reactivated, [LANSINK_FAST, PEYRACHE_FAST], rng=None)
    again = _compare(
        reactivated, [LANSINK_FAST, PEYRACHE_FAST], rng=first.attrs["rng_entropy"]
    )
    columns = ["ev_minus_rev_lower", "match_fraction_above_shuffle"]
    np.testing.assert_array_equal(first[columns], again[columns])


def test_a_period_shorter_than_a_window_is_an_error_row(reactivated) -> None:
    spike_times, template, _match, control = reactivated
    table = compare_reactivation_methods(
        spike_times,
        template,
        [[2400.0, 2700.0]],
        control,
        reactivation_methods=[KUDRIMOTI_1999, PEYRACHE_FAST],
        unit_groups=UNIT_GROUPS,
        rng=0,
    )
    kudrimoti = table[table["method"] == "kudrimoti_1999"]
    assert len(kudrimoti) == 1
    assert "shorter than one 600 s window" in kudrimoti["error"].iloc[0]
    assert np.isnan(kudrimoti["value"].iloc[0])
    assert (table.loc[table["method"] == "peyrache_2010", "error"] == "").all()


def test_two_units_are_an_error_row_for_every_measure() -> None:
    """Two units hold one pair: EV and pattern methods report it in rows."""
    rng = np.random.default_rng(0)
    trains = [np.sort(rng.uniform(0, 3600, 3000)) for _ in range(2)]
    table = compare_reactivation_methods(
        trains,
        [[1200.0, 2400.0]],
        [[2400.0, 3600.0]],
        [[0.0, 1200.0]],
        reactivation_methods=[PEYRACHE_FAST, PENNARTZ_2004],
        rng=0,
    )
    errors = table.set_index("method")["error"]
    assert "at least 3 pairs" in errors["pennartz_2004"]
    assert "2 units hold 1 pair" in errors["pennartz_2004"]
    assert "at least 3 neurons" in errors["peyrache_2010"]


def test_input_errors(reactivated) -> None:
    spike_times, template, match, control = reactivated
    with pytest.raises(ValueError, match="need control_epochs"):
        compare_reactivation_methods(
            spike_times, template, match, reactivation_methods=[PENNARTZ_2004]
        )
    with pytest.raises(ValueError, match=r"unit_groups has no \['tetrode'\]"):
        compare_reactivation_methods(
            spike_times, template, match, control, reactivation_methods=[KUDRIMOTI_1999]
        )
    with pytest.raises(ValueError, match="repeat"):
        compare_reactivation_methods(
            spike_times,
            template,
            match,
            control,
            reactivation_methods=[PENNARTZ_2004, PENNARTZ_2004],
        )
    with pytest.raises(ValueError, match="one group per unit"):
        compare_reactivation_methods(
            spike_times,
            template,
            match,
            control,
            reactivation_methods=[KUDRIMOTI_1999],
            unit_groups={"tetrode": np.arange(3)},
        )
    with pytest.raises(ValueError, match="pair_mask must be a boolean array"):
        compare_reactivation_methods(
            spike_times,
            template,
            match,
            control,
            reactivation_methods=[PENNARTZ_2004],
            pair_mask=np.ones(5, dtype=bool),
        )


def test_assembly_rows_name_members_by_unit_id(reactivated) -> None:
    spike_times, template, match, control = reactivated
    labels = np.arange(100, 100 + N_UNITS)
    table = compare_reactivation_methods(
        spike_times,
        template,
        match,
        control,
        reactivation_methods=[VAN_DE_VEN_2016],
        unit_ids=labels,
        rng=0,
    )
    assert {tuple(sorted(m)) for m in table["member_unit_ids"]} == {
        tuple(100 + unit for unit in members) for members in ASSEMBLIES
    }


def test_a_custom_recipe_runs(reactivated) -> None:
    gridchyn_like = ReactivationMethod(
        name="all_pairs_200ms",
        reference="for illustration",
        measure="explained_variance",
        bin_size=0.2,
    )
    row = _compare(reactivated, [gridchyn_like]).iloc[0]
    assert row["n_pairs"] == 276 and row["value"] > 0.5
