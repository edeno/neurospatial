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
    compare_replay_methods,
    detection_rates,
    event_properties,
    method_agreement,
)
from neurospatial.replay.methods import SILVA_2015, WU_2017
from neurospatial.simulation import simulate_replay_event

DT = 0.02


@pytest.fixture(scope="module")
def track():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )


def _table(verdicts, errors=None):
    """A comparison table from {method: [verdict per event]}."""
    rows = []
    for method, calls in verdicts.items():
        for event, call in enumerate(calls):
            error = "" if errors is None else errors.get((event, method), "")
            rows.append(
                {"event": event, "method": method, "is_replay": call, "error": error}
            )
    return pd.DataFrame(rows)


class TestEventProperties:
    def test_measures_each_event(self, track):
        counts = np.array([[1, 0, 2], [0, 0, 1], [0, 0, 0]])
        event = DecodingResult(
            posterior=np.eye(10)[[2, 5, 8]] * 0.5 + 0.05,
            env=track,
            times=np.arange(3) * DT,
        )
        row = event_properties(track, [event], spike_counts=[counts]).iloc[0]
        assert row["n_time_bins"] == 3
        assert row["duration"] == pytest.approx(0.04)
        assert row["mean_peak_probability"] == pytest.approx(0.55)
        assert row["coverage"] == pytest.approx(0.6)
        assert (row["n_active_cells"], row["n_spikes"]) == (2, 4)

    def test_a_joint_decode_is_measured_summed(self, track):
        times = np.arange(4) * DT
        joint = JointDecodingResult(
            template_names=("outbound", "inbound"),
            template_probability=np.full((4, 2), 0.5),
            conditional={
                "outbound": DecodingResult(
                    posterior=np.eye(10)[[1, 2, 3, 4]], env=track, times=times
                ),
                "inbound": DecodingResult(
                    posterior=np.eye(10)[[1, 2, 3, 4]], env=track, times=times
                ),
            },
        )
        row = event_properties(track, [joint]).iloc[0]
        assert row["mean_peak_probability"] == pytest.approx(1.0)
        assert row["coverage"] == pytest.approx(0.3)

    def test_refuses_mismatched_inputs(self, track):
        event = DecodingResult(posterior=np.eye(10)[:3], env=track, times=None)
        with pytest.raises(ValueError, match="no times"):
            event_properties(track, [event])
        timed = DecodingResult(
            posterior=np.eye(10)[:3], env=track, times=np.arange(3) * DT
        )
        with pytest.raises(ValueError, match="has 2 rows"):
            event_properties(track, [timed], spike_counts=[np.ones((2, 4))])
        with pytest.raises(ValueError, match="2 entries"):
            event_properties(track, [timed], spike_counts=[np.ones((3, 4))] * 2)


class TestDetectionRates:
    def test_rates_with_exact_binomial_intervals(self):
        table = _table({"a": [True] * 3 + [False] * 17, "b": [False] * 20})
        rates = detection_rates(table).set_index("method")
        interval = binomtest(3, 20).proportion_ci(0.95, method="exact")
        assert rates.loc["a", "rate"] == pytest.approx(0.15)
        assert rates.loc["a", "ci_low"] == pytest.approx(interval.low)
        assert rates.loc["a", "ci_high"] == pytest.approx(interval.high)
        assert rates.loc["b", "rate"] == 0.0

    def test_errors_are_counted_not_rated(self):
        table = _table(
            {"a": [True, False, True]},
            errors={(2, "a"): "fewer than two time bins to score"},
        )
        row = detection_rates(table).iloc[0]
        assert (row["n_events"], row["n_replay"], row["n_errors"]) == (2, 1, 1)
        assert row["rate"] == pytest.approx(0.5)

    def test_rates_by_event_property(self):
        table = _table({"a": [True, True, False, False, True, False]})
        properties = pd.DataFrame(
            {"n_active_cells": [4, 6, 6, 12, 15, 30]},
            index=pd.Index(range(6), name="event"),
        )
        rates = detection_rates(
            table,
            properties=properties,
            by={"n_active_cells": [0, 5, 10, 20, np.inf]},
        )
        assert rates["n_events"].tolist() == [1, 2, 2, 1]
        assert rates["n_replay"].tolist() == [1, 1, 1, 0]
        empty = detection_rates(
            table, properties=properties, by={"n_active_cells": [100, 200]}
        )
        assert empty["n_events"].iloc[0] == 0 and np.isnan(empty["rate"].iloc[0])

    def test_refusals(self):
        table = _table({"a": [True]})
        with pytest.raises(ValueError, match="needs properties"):
            detection_rates(table, by={"n_spikes": [0, 10]})
        with pytest.raises(ValueError, match="lacks columns"):
            detection_rates(table.drop(columns="is_replay"))
        with pytest.raises(ValueError, match="confidence"):
            detection_rates(table, confidence=1.0)


class TestMethodAgreement:
    def test_kappa_by_hand(self):
        table = _table(
            {
                "a": [True, False, True, False, True, False],
                "b": [True, False, False, False, True, True],
            }
        )
        kappa = method_agreement(table)
        # Agreement 4/6; chance 0.5 * 0.5 + 0.5 * 0.5 = 0.5.
        assert kappa.loc["a", "b"] == pytest.approx((4 / 6 - 0.5) / 0.5)
        assert kappa.loc["b", "a"] == kappa.loc["a", "b"]
        assert kappa.loc["a", "a"] == 1.0

    def test_constant_verdicts_have_no_kappa(self):
        kappa = method_agreement(_table({"a": [False] * 4, "b": [True, False] * 2}))
        # a never calls replay: its agreement with b is chance, with itself
        # undefined.
        assert kappa.loc["a", "b"] == pytest.approx(0.0)
        assert np.isnan(kappa.loc["a", "a"])

    def test_events_with_an_error_are_left_out(self):
        table = _table(
            {"a": [True, False, True], "b": [True, False, False]},
            errors={(2, "b"): "fewer than two time bins to score"},
        )
        assert method_agreement(table).loc["a", "b"] == pytest.approx(1.0)


def test_false_positive_rates_on_simulated_events(track):
    """Random events rarely pass; sweeps nearly always do."""
    x = track.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 20)[:, None]) / 8.0) ** 2
    )
    events, counts, kinds = [], [], []
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
        events.append(decode_position(track, event_counts, rates, dt=DT, times=grid))
        counts.append(event_counts)
        kinds.append(kind)
    table = compare_replay_methods(
        track, events, replay_methods=[SILVA_2015, WU_2017], spike_counts=counts, rng=0
    )
    properties = event_properties(track, events, spike_counts=counts)
    properties["kind"] = kinds
    table["kind"] = table["event"].map(properties["kind"])
    null = detection_rates(table[table["kind"] == "random"]).set_index("method")
    found = detection_rates(table[table["kind"] == "sweep"]).set_index("method")
    assert (null["rate"] <= 0.25).all()
    assert (found["rate"] >= 0.75).all()
    assert set(method_agreement(table).index) == {"silva_2015", "wu_2017"}
