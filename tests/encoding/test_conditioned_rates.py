"""Conditioned rate maps: labels -> epochs -> the existing estimators."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.behavior import labels_to_epochs
from neurospatial.decoding import BayesianDecoder
from neurospatial.encoding import compute_spatial_rates

UNITS = ["u1"]
# Below two 0.1 s frames, so the lost frame at 1.2 s is a tracking loss.
MAX_GAP = 0.15


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment.from_samples(
        np.linspace(0, 100, 101)[:, None], bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def times() -> np.ndarray:
    return np.arange(20) / 10


@pytest.fixture(scope="module")
def positions() -> np.ndarray:
    pos = np.linspace(5, 95, 20)[:, None]
    pos[12] = np.nan
    return pos


@pytest.fixture(scope="module")
def conditions(times, positions):
    labels = ["out"] * 10 + ["in"] * 10
    return labels_to_epochs(
        times, labels, valid=np.isfinite(positions).all(axis=1), max_gap=MAX_GAP
    )


def condition(conditions, label):
    """The epochs of one label, to pass as ``epochs=``."""
    return conditions[conditions.label == label]


def rates(env, spikes, times, positions, epochs):
    return compute_spatial_rates(
        env,
        [np.asarray(spikes)],
        times,
        positions,
        epochs=epochs,
        unit_ids=UNITS,
        method="binned",
        bandwidth=1.0,
        max_gap=MAX_GAP,
    )


def test_condition_epochs(conditions, times):
    assert conditions.label.tolist() == ["out", "in", "in"]
    assert_array_equal(condition(conditions, "out").windows, [[0.0, 1.0]])
    assert_array_equal(
        condition(conditions, "in").windows,
        [[times[10], times[11]], [times[13], times[19]]],
    )


@pytest.mark.parametrize(("label", "duration"), [("out", 1.0), ("in", 0.7)])
def test_dwell_equals_labelled_duration(
    env, times, positions, conditions, label, duration
):
    epochs = condition(conditions, label)
    result = rates(env, [0.05], times, positions, epochs)
    assert_allclose(result.occupancy.sum(), epochs.durations.sum(), rtol=1e-12)
    assert_allclose(result.occupancy.sum(), duration, rtol=1e-12)


def test_spikes_in_excluded_intervals_do_not_count(env, times, positions, conditions):
    spikes = np.array([0.05, 0.55, 1.15, 1.25, 1.85])
    inbound, outbound = condition(conditions, "in"), condition(conditions, "out")
    inside = rates(env, spikes, times, positions, inbound)
    without = rates(env, spikes[[0, 1, 4]], times, positions, inbound)
    assert_array_equal(inside.firing_rates, without.firing_rates)
    out = rates(env, spikes, times, positions, outbound)
    out_without = rates(env, spikes[:4], times, positions, outbound)
    assert_array_equal(out.firing_rates, out_without.firing_rates)


def test_estimator_and_labels_agree_on_support(env, times, positions, conditions):
    spikes = np.array([1.05, 1.35, 1.85])
    from_labels = rates(env, spikes, times, positions, condition(conditions, "in"))
    explicit = rates(env, spikes, times, positions, [(1.0, 1.9)])
    assert_array_equal(from_labels.firing_rates, explicit.firing_rates)
    assert_array_equal(from_labels.occupancy, explicit.occupancy)
    assert len(condition(conditions, "in")) == 2


@pytest.mark.parametrize("label", ["out", "in"])
def test_unit_identity_and_epochs_recorded(env, times, positions, conditions, label):
    epochs = condition(conditions, label)
    result = rates(env, [0.05, 1.35], times, positions, epochs)
    assert result.unit_ids.tolist() == UNITS
    assert result.analysis_settings["epochs"] == epochs.windows.tolist()


def test_lost_frame_inside_max_gap_keeps_one_condition_epoch(env, times, positions):
    labels = ["out"] * 10 + ["in"] * 10
    conditions = labels_to_epochs(
        times, labels, valid=np.isfinite(positions).all(axis=1)
    )
    assert conditions.label.tolist() == ["out", "in"]
    assert_array_equal(condition(conditions, "in").windows, [[times[10], times[19]]])
    result = compute_spatial_rates(
        env,
        [np.array([1.05, 1.15, 1.25])],
        times,
        positions,
        epochs=condition(conditions, "in"),
        method="binned",
        bandwidth=1.0,
    )
    assert_allclose(result.occupancy.sum(), 0.9, rtol=1e-12)


def test_decoder_reuses_condition_epochs(env, times, positions, conditions):
    decoder = BayesianDecoder(env, dt=0.1, bandwidth=10.0).fit(
        [np.array([0.05, 0.55])],
        times,
        positions,
        unit_ids=UNITS,
        epochs=condition(conditions, "out"),
    )
    assert decoder.analysis_settings["epochs"] == [[0.0, 1.0]]
