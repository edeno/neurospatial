"""Tests for calibrating replay methods on comparison tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from neurospatial import Environment
from neurospatial.decoding import (
    DecodingResult,
    JointDecodingResult,
    bin_spikes_in_time,
    decode_position,
)
from neurospatial.replay import (
    NullSpikeData,
    compare_replay_methods,
    decode_intervals,
    detection_rates,
    event_properties,
    method_agreement,
    null_events,
    simulated_null_events,
)
from neurospatial.replay.methods import DIBA_BUZSAKI_2007, SILVA_2015, WU_2017
from neurospatial.simulation import simulate_replay_event

DT = 0.02


@pytest.fixture(scope="module")
def track():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )


def _table(verdicts, errors=None, **columns):
    """A comparison table from {method: [verdict per event]}."""
    rows = []
    for method, calls in verdicts.items():
        for event, call in enumerate(calls):
            error = "" if errors is None else errors.get((event, method), "")
            rows.append(
                {"event": event, "method": method, "is_replay": call, "error": error}
            )
    table = pd.DataFrame(rows)
    for name, values in columns.items():
        table[name] = table["event"].map(dict(enumerate(values)))
    return table


def _joint(track_a, track_b, rows_a, rows_b):
    times = np.arange(len(rows_a)) * DT
    return JointDecodingResult(
        template_names=("a", "b"),
        template_probability=np.full((len(rows_a), 2), 0.5),
        conditional={
            "a": DecodingResult(
                posterior=np.eye(track_a.n_bins)[rows_a], env=track_a, times=times
            ),
            "b": DecodingResult(
                posterior=np.eye(track_b.n_bins)[rows_b], env=track_b, times=times
            ),
        },
    )


class TestEventProperties:
    def test_measures_each_event(self, track):
        counts = np.array([[1, 0, 2], [0, 0, 1], [0, 0, 0]])
        event = DecodingResult(
            posterior=np.eye(10)[[2, 5, 8]] * 0.5 + 0.05,
            env=track,
            times=np.arange(3) * DT,
        )
        row = event_properties(track, [event], spike_counts=[counts], dt=DT).iloc[0]
        assert row["n_time_bins"] == 3
        assert row["duration"] == pytest.approx(0.04)
        assert row["window_duration"] == pytest.approx(0.06)
        assert row["mean_peak_probability"] == pytest.approx(0.55)
        assert row["map_range_fraction"] == pytest.approx(0.6)
        assert row["map_path_length"] == pytest.approx(60.0)
        assert (row["n_active_cells"], row["n_spikes"]) == (2, 4)
        assert row["spikes_per_active_cell"] == pytest.approx(2.0)
        assert row["empty_fraction"] == pytest.approx(1 / 3)

    def test_spread_does_not_depend_on_the_bin_size(self):
        """A Gaussian bump of SD 8 cm decoded on 2 cm and 10 cm bins."""
        spreads, peaks = [], []
        for bin_size in (2.0, 10.0):
            env = Environment.linear_track(
                endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=bin_size, units="cm"
            )
            x = env.bin_centers[:, 0]
            bump = np.exp(-0.5 * ((x - 100.0) / 8.0) ** 2)
            event = DecodingResult(
                posterior=np.tile(bump / bump.sum(), (3, 1)),
                env=env,
                times=np.arange(3) * DT,
            )
            row = event_properties(env, [event]).iloc[0]
            spreads.append(row["mean_spread"])
            peaks.append(row["mean_peak_probability"])
        assert spreads[0] == pytest.approx(8.0, rel=0.05)
        assert spreads[1] == pytest.approx(8.0, rel=0.1)
        assert peaks[1] > 3 * peaks[0]

    def test_a_joint_decode_on_one_track_is_summed(self, track):
        joint = _joint(track, track, [1, 2, 3, 4], [1, 2, 3, 4])
        row = event_properties(track, [joint]).iloc[0]
        assert row["mean_peak_probability"] == pytest.approx(1.0)
        assert row["map_range_fraction"] == pytest.approx(0.3)

    def test_separate_tracks_need_a_template(self, track):
        other = Environment.linear_track(
            endpoints=[[0.0, 50.0], [100.0, 50.0]], bin_size=10.0, units="cm"
        )
        joint = _joint(track, other, [0, 1, 2, 3], [9, 8, 7, 6])
        with pytest.raises(ValueError, match="separate"):
            event_properties(track, [joint])
        row = event_properties({"a": track, "b": other}, [joint], template="b").iloc[0]
        assert row["mean_peak_probability"] == pytest.approx(0.5)
        assert row["map_range_fraction"] == pytest.approx(0.3)

    def test_refuses_bad_inputs(self, track):
        with pytest.raises(ValueError, match="no times"):
            event_properties(
                track, [DecodingResult(posterior=np.eye(10)[:3], env=track)]
            )
        timed = DecodingResult(
            posterior=np.eye(10)[:3], env=track, times=np.arange(3) * DT
        )
        with pytest.raises(ValueError, match="has 2 rows"):
            event_properties(track, [timed], spike_counts=[np.ones((2, 4))])
        with pytest.raises(ValueError, match="2 entries"):
            event_properties(track, [timed], spike_counts=[np.ones((3, 4))] * 2)
        with pytest.raises(ValueError, match="non-integer"):
            event_properties(track, [timed], spike_counts=[np.full((3, 2), 0.5)])
        with pytest.raises(ValueError, match="empty"):
            event_properties(track, [])


class TestDetectionRates:
    def test_rates_with_exact_binomial_intervals(self):
        table = _table({"a": [True] * 3 + [False] * 17, "b": [False] * 20})
        rates = detection_rates(table).set_index("method")
        interval = binomtest(3, 20).proportion_ci(0.95, method="exact")
        assert rates.loc["a", "rate"] == pytest.approx(0.15)
        assert rates.loc["a", "ci_low"] == pytest.approx(interval.low)
        assert rates.loc["a", "ci_high"] == pytest.approx(interval.high)
        assert rates.loc["b", "rate"] == 0.0

    def test_errors_count_as_not_replay_unless_excluded(self):
        table = _table(
            {"a": [True, False, True, True]},
            errors={(2, "a"): "too few cells", (3, "a"): "too few cells"},
        )
        same_events = detection_rates(table).iloc[0]
        scored_only = detection_rates(table, errors="exclude").iloc[0]
        assert (same_events["n_events"], same_events["n_replay"]) == (4, 1)
        assert (scored_only["n_events"], scored_only["n_replay"]) == (2, 1)
        assert same_events["n_errors"] == scored_only["n_errors"] == 2

    def test_rates_by_event_property(self):
        table = _table({"a": [True, True, False, False, True, False]})
        properties = pd.DataFrame(
            {"n_active_cells": [4, 5, 6, 10, 15, 30]},
            index=pd.Index(range(6), name="event"),
        )
        rates = detection_rates(
            table,
            properties=properties,
            by={"n_active_cells": [0, 5, 10, 20, np.inf]},
        )
        # Left-closed: 5 falls in [5, 10) and 10 in [10, 20).
        assert rates["n_events"].tolist() == [1, 2, 2, 1]
        assert rates["n_replay"].tolist() == [1, 1, 1, 0]

    def test_two_keys_give_every_combination(self):
        table = _table(
            {"a": [True, False, True, False], "b": [False] * 4},
            n_spikes=[10, 30, 10, 30],
            null_kind=["shuffle", "shuffle", "poisson", "poisson"],
        )
        rates = detection_rates(
            table, by={"n_spikes": [0, 20, np.inf]}, groups=["null_kind"]
        )
        assert len(rates) == 2 * 2 * 2
        cell = rates[
            (rates["method"] == "a")
            & (rates["null_kind"] == "poisson")
            & (rates["n_spikes"].astype(str) == "[0.0, 20.0)")
        ]
        assert cell["n_replay"].item() == 1 and cell["n_events"].item() == 1

    def test_events_outside_every_bin_are_refused(self):
        table = _table({"a": [True, False, True]})
        properties = pd.DataFrame(
            {"map_range_fraction": [0.2, np.nan, 0.5]},
            index=pd.Index(range(3), name="event"),
        )
        with pytest.raises(ValueError, match=r"events \[1\] have no bin"):
            detection_rates(
                table, properties=properties, by={"map_range_fraction": [0, 1]}
            )
        with pytest.raises(ValueError, match="at least two bin edges"):
            detection_rates(
                table, properties=properties, by={"map_range_fraction": [0]}
            )

    def test_concatenated_tables_need_groups(self):
        first = _table({"a": [True, False]}, session=["s1", "s1"])
        second = _table({"a": [False, False]}, session=["s2", "s2"])
        both = pd.concat([first, second])
        with pytest.raises(ValueError, match="repeats 2"):
            detection_rates(both)
        rates = detection_rates(both, groups=["session"]).set_index("session")
        assert rates.loc["s1", "rate"] == 0.5 and rates.loc["s2", "rate"] == 0.0

    def test_refusals(self):
        table = _table({"a": [True]})
        with pytest.raises(ValueError, match="neither a column"):
            detection_rates(table, by={"n_spikes": [0, 10]})
        with pytest.raises(ValueError, match="lacks columns"):
            detection_rates(table.drop(columns="is_replay"))
        with pytest.raises(ValueError, match="boolean"):
            detection_rates(table.assign(is_replay="False"))
        with pytest.raises(ValueError, match="confidence"):
            detection_rates(table, confidence=1.0)
        with pytest.raises(ValueError, match="errors must be"):
            detection_rates(table, errors="drop")


class TestMethodAgreement:
    def test_counts_and_kappa_by_hand(self):
        table = _table(
            {
                "a": [True, False, True, False, True, False],
                "b": [True, False, False, False, True, True],
            }
        )
        row = method_agreement(table).iloc[0]
        assert (row["first"], row["second"]) == ("a", "b")
        assert (row["n_both"], row["n_first_only"]) == (2, 1)
        assert (row["n_second_only"], row["n_neither"]) == (1, 2)
        assert row["agreement"] == pytest.approx(4 / 6)
        # Agreement 4/6; chance 0.5 * 0.5 + 0.5 * 0.5 = 0.5.
        assert row["kappa"] == pytest.approx((4 / 6 - 0.5) / 0.5)

    def test_kappa_matches_a_reference(self):
        rng = np.random.default_rng(4)
        a = rng.random(200) < 0.3
        b = np.where(rng.random(200) < 0.7, a, rng.random(200) < 0.3)
        kappa = method_agreement(_table({"a": list(a), "b": list(b)}))["kappa"].item()
        observed = np.mean(a == b)
        chance = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        assert kappa == pytest.approx((observed - chance) / (1 - chance))

    def test_same_constant_verdicts_have_no_kappa(self):
        assert np.isnan(
            method_agreement(_table({"a": [False] * 4, "b": [False] * 4}))[
                "kappa"
            ].item()
        )
        one_constant = method_agreement(
            _table({"a": [False] * 4, "b": [True, False] * 2})
        )
        assert one_constant["kappa"].item() == pytest.approx(0.0)

    def test_errors_count_as_not_replay_unless_excluded(self):
        table = _table(
            {"a": [True, False, True], "b": [True, False, True]},
            errors={(2, "b"): "too few cells"},
        )
        counted = method_agreement(table).iloc[0]
        dropped = method_agreement(table, errors="exclude").iloc[0]
        assert (counted["n_events"], counted["n_first_only"]) == (3, 1)
        assert (dropped["n_events"], dropped["agreement"]) == (2, 1.0)


def test_false_positive_rates_on_simulated_events(track):
    """Random events rarely pass; sweeps nearly always do; strata add up."""
    x = track.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 20)[:, None]) / 8.0) ** 2
    )
    events, counts, spikes, kinds = [], [], [], []
    offset = 0.0
    for seed in range(16):
        kind = "sweep" if seed % 4 == 0 else "random"
        event = simulate_replay_event(
            track,
            rates,
            kind=kind,
            rate_gain=4.0,
            seed=seed,
            **({"speed": 400.0} if kind == "sweep" else {}),
        )
        event_counts, grid = bin_spikes_in_time(
            list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
        )
        events.append(
            decode_position(track, event_counts, rates, dt=DT, times=grid + offset)
        )
        counts.append(event_counts)
        spikes.append([train + offset for train in event.spike_times])
        kinds.append(kind)
        offset += 10.0
    session_spikes = [
        np.concatenate([event[cell] for event in spikes]) for cell in range(20)
    ]
    table = compare_replay_methods(
        track,
        events,
        replay_methods=[SILVA_2015, WU_2017, DIBA_BUZSAKI_2007],
        spike_counts=counts,
        spike_times=session_spikes,
        encoding_models=rates,
        dt=DT,
        rng=0,
    )
    table["kind"] = table["event"].map(dict(enumerate(kinds)))
    properties = event_properties(track, events, spike_counts=counts, dt=DT)
    by_kind = detection_rates(table, groups=["kind"]).set_index(["method", "kind"])
    assert (by_kind.xs("random", level="kind")["rate"] <= 0.25).all()
    assert (by_kind.xs("sweep", level="kind")["rate"] >= 0.75).all()
    strata = detection_rates(
        table,
        properties=properties,
        by={"n_active_cells": [0, 10, np.inf]},
        groups=["kind"],
    )
    totals = strata.groupby(["method", "kind"])[["n_events", "n_replay"]].sum()
    assert totals.loc[("silva_2015", "random"), "n_events"] == 12
    assert (
        totals["n_replay"].to_numpy()
        == by_kind.loc[totals.index, "n_replay"].to_numpy()
    ).all()
    agreement = method_agreement(table)
    assert len(agreement) == 3 and (agreement["n_events"] == 16).all()


@pytest.fixture(scope="module")
def recording():
    """Three 200 ms events of 6 cells, with spikes outside them too."""
    rng = np.random.default_rng(2)
    intervals = np.array([[1.0, 1.2], [3.0, 3.2], [7.0, 7.2]])
    trains = []
    for cell in range(6):
        inside = [start + rng.uniform(0, 0.2, cell + 1) for start, _ in intervals]
        outside = rng.uniform(0, 10, 5)
        trains.append(np.sort(np.concatenate([*inside, outside])))
    return trains, intervals


def _in_events(trains, intervals):
    """Each event's spikes relative to its start, per cell."""
    return [
        [np.sort(train[(train >= a) & (train < b)] - a) for train in trains]
        for a, b in intervals
    ]


class TestNullEvents:
    @pytest.mark.parametrize("kind", ["cell_identity", "spike_shift", "poisson"])
    def test_layout_and_what_is_kept(self, recording, kind):
        trains, intervals = recording
        null = null_events(trains, intervals, kind=kind, n_copies=3, rng=0)
        assert null.n_events == 9 and null.kind == kind
        np.testing.assert_array_equal(null.source_events, np.repeat([0, 1, 2], 3))
        np.testing.assert_allclose(np.diff(null.intervals, axis=1).ravel(), 0.2)
        # Null events do not overlap, and no spike falls outside them.
        assert np.all(null.intervals[1:, 0] >= null.intervals[:-1, 1] + 1.0 - 1e-9)
        real = _in_events(trains, intervals)
        copies = _in_events(null.spike_times, null.intervals)
        assert sum(t.size for t in null.spike_times) == sum(
            sum(c.size for c in event) for event in copies
        )
        for copy, source in zip(copies, null.source_events, strict=True):
            original = real[source]
            sizes = sorted(c.size for c in copy)
            assert sizes == sorted(c.size for c in original)
            if kind == "cell_identity":
                rounded = [tuple(np.round(c, 9)) for c in copy]
                assert sorted(rounded) == sorted(
                    tuple(np.round(c, 9)) for c in original
                )
            else:
                assert [c.size for c in copy] == [c.size for c in original]
                assert all(np.all((c >= 0) & (c < 0.2)) for c in copy)

    def test_cell_identity_moves_spike_trains_between_cells(self, recording):
        trains, intervals = recording
        null = null_events(trains, intervals, kind="cell_identity", n_copies=5, rng=1)
        copies = _in_events(null.spike_times, null.intervals)
        real = _in_events(trains, intervals)
        moved = sum(
            copy[cell].size != real[source][cell].size
            for copy, source in zip(copies, null.source_events, strict=True)
            for cell in range(6)
        )
        assert moved > 0

    def test_results_are_read_only_and_pickle(self, recording):
        import pickle

        trains, intervals = recording
        null = null_events(trains, intervals, kind="poisson", rng=0)
        assert not null.intervals.flags.writeable
        assert not null.spike_times[0].flags.writeable
        clone = pickle.loads(pickle.dumps(null))
        assert not clone.spike_times[0].flags.writeable
        assert clone.summary() == null.summary()

    def test_refusals(self, recording):
        trains, intervals = recording
        with pytest.raises(ValueError, match="kind must be"):
            null_events(trains, intervals, kind="jitter")
        with pytest.raises(ValueError, match="n_copies"):
            null_events(trains, intervals, kind="poisson", n_copies=0)
        with pytest.raises(ValueError, match="start < stop"):
            null_events(trains, np.array([[1.0, 0.5]]), kind="poisson")

    def test_simulated_events(self, track):
        x = track.bin_centers[:, 0]
        rates = 20.0 * np.exp(-0.5 * ((x - np.linspace(5, 95, 8)[:, None]) / 8.0) ** 2)
        null = simulated_null_events(track, rates, n_events=3, kind="stationary", rng=0)
        assert isinstance(null, NullSpikeData)
        assert null.kind == "simulated_stationary"
        np.testing.assert_array_equal(null.source_events, [-1, -1, -1])
        assert len(null.spike_times) == 8
        with pytest.raises(ValueError, match="kind must be"):
            simulated_null_events(track, rates, n_events=3, kind="sweep")


class TestNullRandomization:
    def test_spike_shift_moves_spikes_but_keeps_their_intervals(self, recording):
        trains, intervals = recording
        null = null_events(trains, intervals, kind="spike_shift", rng=0)
        real = _in_events(trains, intervals)
        copies = _in_events(null.spike_times, null.intervals)
        moved = 0
        for copy, source in zip(copies, null.source_events, strict=True):
            for cell, spikes in enumerate(copy):
                original = real[source][cell]
                moved += not np.allclose(spikes, original)

                # Circular intervals between consecutive spikes are kept.
                def circular(t):
                    return np.sort(np.mod(np.diff(np.r_[t, t[0] + 0.2]), 0.2))

                np.testing.assert_allclose(
                    circular(spikes), circular(original), atol=1e-9
                )
        assert moved > 10

    def test_poisson_redraws_spike_times(self, recording):
        trains, intervals = recording
        null = null_events(trains, intervals, kind="poisson", rng=0)
        real = _in_events(trains, intervals)
        copies = _in_events(null.spike_times, null.intervals)
        same = sum(
            np.allclose(copy[cell], real[source][cell])
            for copy, source in zip(copies, null.source_events, strict=True)
            for cell in range(6)
        )
        assert same == 0

    @pytest.mark.parametrize("kind", ["cell_identity", "spike_shift", "poisson"])
    def test_a_seed_reproduces_and_copies_differ(self, recording, kind):
        trains, intervals = recording
        first = null_events(trains, intervals, kind=kind, n_copies=2, rng=3)
        again = null_events(trains, intervals, kind=kind, n_copies=2, rng=3)
        for a, b in zip(first.spike_times, again.spike_times, strict=True):
            np.testing.assert_array_equal(a, b)
        copies = _in_events(first.spike_times, first.intervals)
        assert any(
            a.size != b.size or not np.allclose(a, b)
            for a, b in zip(copies[0], copies[1], strict=True)
        )

    def test_cells_outside_the_mask_keep_their_spikes(self, recording):
        trains, intervals = recording
        cells = np.array([True, True, True, False, False, False])
        null = null_events(trains, intervals, kind="poisson", cells=cells, rng=0)
        real = _in_events(trains, intervals)
        copies = _in_events(null.spike_times, null.intervals)
        for copy, source in zip(copies, null.source_events, strict=True):
            for cell in range(3, 6):
                np.testing.assert_allclose(copy[cell], real[source][cell])
        with pytest.raises(ValueError, match="one boolean per cell"):
            null_events(trains, intervals, kind="poisson", cells=cells[:3])

    def test_simulated_random_events_are_laid_out(self, track):
        x = track.bin_centers[:, 0]
        rates = 20.0 * np.exp(-0.5 * ((x - np.linspace(5, 95, 8)[:, None]) / 8.0) ** 2)
        null = simulated_null_events(
            track, rates, n_events=3, kind="random", duration=0.3, rng=0
        )
        np.testing.assert_allclose(null.intervals, [[0, 0.3], [1.3, 1.6], [2.6, 2.9]])
        inside = _in_events(null.spike_times, null.intervals)
        assert sum(t.size for t in null.spike_times) == sum(
            sum(c.size for c in event) for event in inside
        )

    def test_stationary_events_sit_along_the_whole_track(self, track):
        x = track.bin_centers[:, 0]
        rates = 0.5 + 40.0 * np.exp(
            -0.5 * ((x - np.linspace(5, 95, 10)[:, None]) / 6.0) ** 2
        )
        null = simulated_null_events(
            track, rates, n_events=20, kind="stationary", rate_gain=3.0, rng=0
        )
        events, _ = decode_intervals(
            track, null.spike_times, null.intervals, rates, dt=DT
        )
        positions = [
            x[np.bincount(event.posterior.argmax(axis=1)).argmax()] for event in events
        ]
        assert np.ptp(positions) >= 50.0


class TestDecodeIntervals:
    def test_decodes_each_interval(self, track, recording):
        trains, intervals = recording
        rates = np.full((6, track.n_bins), 5.0)
        events, counts = decode_intervals(track, trains, intervals, rates, dt=DT)
        assert len(events) == len(counts) == 3
        np.testing.assert_allclose(events[1].times, 3.0 + DT / 2 + DT * np.arange(10))
        assert counts[0].shape == (10, 6)
        assert counts[0].sum() == sum(c.size for c in _in_events(trains, intervals)[0])

    def test_drops_a_trailing_remainder_and_keeps_one_whole_bin(self, track, recording):
        trains, _ = recording
        rates = np.full((6, track.n_bins), 5.0)
        events, counts = decode_intervals(
            track, trains, np.array([[1.0, 1.05], [0.1, 0.12]]), rates, dt=DT
        )
        assert [len(event.posterior) for event in events] == [2, 1]
        assert counts[1].shape == (1, 6)

    def test_refuses_an_interval_shorter_than_a_bin(self, track, recording):
        trains, _ = recording
        with pytest.raises(ValueError, match="smaller than one bin"):
            decode_intervals(
                track, trains, np.array([[1.0, 1.01]]), np.ones((6, 10)), dt=DT
            )


def test_false_positives_on_cell_identity_copies(track):
    """Shuffling cell identities of real sweeps removes their replay."""
    x = track.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 20)[:, None]) / 8.0) ** 2
    )
    trains = [[] for _ in range(20)]
    intervals = []
    for index in range(6):
        event = simulate_replay_event(
            track, rates, kind="sweep", speed=400.0, rate_gain=4.0, seed=index
        )
        start = 2.0 * index
        for cell, train in enumerate(event.spike_times):
            trains[cell].append(np.asarray(train) + start)
        intervals.append((start, start + event.duration))
    trains = [np.concatenate(parts) for parts in trains]
    intervals = np.asarray(intervals)
    real_events, real_counts = decode_intervals(track, trains, intervals, rates, dt=DT)
    null = null_events(trains, intervals, kind="cell_identity", n_copies=3, rng=0)
    null_decoded, null_counts = decode_intervals(
        track, null.spike_times, null.intervals, rates, dt=DT
    )
    tables = []
    for name, events, counts in (
        ("real", real_events, real_counts),
        ("cell_identity", null_decoded, null_counts),
    ):
        table = compare_replay_methods(
            track, events, replay_methods=[SILVA_2015], spike_counts=counts, rng=0
        )
        table["data"] = name
        tables.append(table)
    rates_by_data = detection_rates(pd.concat(tables), groups=["data"]).set_index(
        "data"
    )
    assert rates_by_data.loc["real", "rate"] >= 0.8
    assert rates_by_data.loc["cell_identity", "rate"] <= 0.25
