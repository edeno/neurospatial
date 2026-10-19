"""Observation-support resolution: validity, breaks, gaps, epochs and runs."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import behavior
from neurospatial._observation import (
    ObservationSupport,
    resolve_observation_support,
    validate_break_before,
    validate_max_gap,
    validated_observations,
)
from neurospatial._validation import validate_strictly_increasing


def _support(
    times, x=None, *, max_gap=0.5, epochs=None, break_before=None
) -> ObservationSupport:
    times = np.asarray(times, dtype=float)
    x = np.arange(times.size, dtype=float) if x is None else np.asarray(x, float)
    positions = np.c_[x, np.zeros_like(x)]
    return validated_observations(
        times,
        positions,
        call="f",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the result is empty",
    )[2]


def test_explicit_break_splits_runs_despite_short_elapsed_time():
    s = _support([0, 0.1, 0.2, 0.3], break_before=[False, False, True, False])
    np.testing.assert_array_equal(s.interval_mask, [True, False, True])
    np.testing.assert_array_equal(s.run_ids, [0, 0, 1, 1])
    np.testing.assert_array_equal(s.run_sample_bounds, [[0, 1], [2, 3]])
    np.testing.assert_allclose(s.run_time_bounds, [[0, 0.1], [0.2, 0.3]])


def test_break_before_first_sample_is_ignored():
    base = _support([0, 0.1, 0.2, 0.3])
    flagged = _support([0, 0.1, 0.2, 0.3], break_before=[True, False, False, False])
    np.testing.assert_array_equal(base.interval_mask, flagged.interval_mask)
    np.testing.assert_array_equal(base.run_ids, flagged.run_ids)


def test_nan_row_breaks_the_path():
    x = np.arange(5, dtype=float)
    positions = np.c_[x, np.zeros(5)]
    positions[2] = np.nan
    s = validated_observations(
        np.arange(5) * 0.1,
        positions,
        call="f",
        max_gap=0.5,
        epochs=None,
        break_before=None,
        result="r",
    )[2]
    np.testing.assert_array_equal(s.interval_mask, [True, False, False, True])
    np.testing.assert_array_equal(s.run_ids, [0, 0, -1, 1, 1])
    assert not s.sample_valid[2]
    assert not s.isolated.any()


@pytest.mark.filterwarnings("ignore:.*max_gap:UserWarning")
def test_isolated_finite_sample():
    s = _support([0, 0.1, 5, 10, 10.1])
    assert s.run_ids[2] == -1
    assert s.sample_valid[2]
    np.testing.assert_array_equal(s.isolated, [False, False, True, False, False])


def test_stationary_interval_is_usable():
    s = _support([0, 0.1, 0.2, 0.3], x=[1, 1, 1, 1])
    assert s.interval_mask.all()
    np.testing.assert_allclose(s.run_time_bounds, [[0, 0.3]])


def test_epochs_are_half_open():
    s = _support([0, 0.5, 1.0, 1.5], epochs=(0, 1.0))
    np.testing.assert_array_equal(s.interval_mask, [True, True, False])


def test_max_gap_none_keeps_long_interval():
    s = _support([0, 10], max_gap=None)
    np.testing.assert_array_equal(s.interval_mask, [True])
    with pytest.warns(UserWarning, match="max_gap"):
        assert not _support([0, 10]).interval_mask.any()


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_max_gap_raises(bad):
    with pytest.raises(ValueError, match="max_gap must be a positive finite"):
        validate_max_gap(bad)
    with pytest.raises(ValueError, match="max_gap must be a positive finite"):
        _support([0, 1, 2], max_gap=bad)


def test_repeated_timestamp_raises_with_recovery_advice():
    with pytest.raises(ValueError, match="strictly increasing") as exc:
        _support([0, 1, 1, 2])
    assert "analyze each recording" in str(exc.value)
    assert "index 1" in str(exc.value)


def test_decreasing_time_raises():
    with pytest.raises(ValueError, match="times"):
        _support([0, 2, 1])


def test_validate_strictly_increasing_accepts_increasing():
    validate_strictly_increasing(np.array([0.0, 1.0, 2.0]), call="f")


@pytest.mark.parametrize(
    "bad", [[0, 0, 1, 0], [False, True], [False] * 5, np.zeros((4, 1), bool)]
)
def test_invalid_break_before_raises(bad):
    with pytest.raises(ValueError, match="Boolean array"):
        _support([0, 0.1, 0.2, 0.3], break_before=np.asarray(bad))


def test_validate_break_before_none_and_valid():
    assert validate_break_before(None, 3, call="f") is None
    out = validate_break_before([True, False, True], 3, call="f")
    assert out is not None and out.dtype == bool


@pytest.mark.parametrize("n", [0, 1])
def test_zero_and_one_sample(n):
    s = resolve_observation_support(
        np.arange(n) * 0.1,
        np.zeros((n, 2)),
        max_gap=0.5,
        epochs=None,
        break_before=None,
        result="r",
    )
    assert s.interval_mask.shape == (0,)
    np.testing.assert_array_equal(s.run_ids, np.full(n, -1))
    assert s.run_sample_bounds.shape == (0, 2)
    assert s.run_time_bounds.shape == (0, 2)
    assert s.sample_valid.shape == (n,)


def test_one_dimensional_positions_are_accepted():
    _, p, s = validated_observations(
        [0, 0.1, 0.2],
        [0.0, 1.0, 2.0],
        call="f",
        max_gap=0.5,
        epochs=None,
        break_before=None,
        result="r",
    )
    assert p.shape == (3, 1)
    assert s.interval_mask.all()


def test_arrays_are_read_only():
    s = _support([0, 0.1, 0.2])
    for name in (
        "sample_valid",
        "interval_mask",
        "run_ids",
        "run_sample_bounds",
        "run_time_bounds",
    ):
        with pytest.raises(ValueError, match="read-only"):
            getattr(s, name)[...] = 0


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("compute_cumulative_distance", {}),
        ("traveled_path_length", {}),
        ("resample_by_distance", {"spacing": 1.0}),
        ("compute_planar_turn_angles", {}),
        ("compute_planar_curvature", {}),
        ("compute_movement_angular_velocity", {}),
    ],
)
def test_empty_input_raises_actionable_error(name, kwargs):
    with pytest.raises(ValueError) as excinfo:
        getattr(behavior, name)(np.array([]), np.zeros((0, 2)), **kwargs)
    what, why, fix = str(excinfo.value).split("\n")
    assert what == f"{name}() got no samples: times and positions are empty."
    assert why == "there is no observation to measure."
    assert fix.startswith("Fix: pass at least one sample")
    assert "epoch or trial selection" in fix
