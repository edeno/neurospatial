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
    validated_signals,
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
        ("compute_speed", {}),
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


# --- per-feature support -----------------------------------------------------

_T5 = [0.0, 0.1, 0.2, 0.3, 0.4]


def _signals(times=_T5, signals=None, *, valid=None, **kwargs):
    kwargs.setdefault("max_gap", 0.5)
    kwargs.setdefault("epochs", None)
    kwargs.setdefault("break_before", None)
    return validated_signals(
        times,
        signals,
        valid=valid,
        call="f",
        result="the aligned values",
        **kwargs,
    )


def test_signals_nan_in_one_feature_only_splits_that_feature():
    sig = np.ones((5, 2))
    sig[2, 1] = np.nan
    _, _, supports = _signals(signals=sig)
    assert len(supports) == 2
    np.testing.assert_array_equal(supports[0].run_ids, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(supports[1].run_ids, [0, 0, -1, 1, 1])
    assert not supports[1].sample_valid[2]


def test_signals_valid_one_dimensional_applies_to_every_feature():
    valid = np.array([True, True, False, True, True])
    _, _, supports = _signals(signals=np.ones((5, 2)), valid=valid)
    for support in supports:
        np.testing.assert_array_equal(support.run_ids, [0, 0, -1, 1, 1])


def test_signals_valid_two_dimensional_applies_per_feature():
    valid = np.ones((5, 2), dtype=bool)
    valid[2, 0] = False
    _, _, supports = _signals(signals=np.ones((5, 2)), valid=valid)
    np.testing.assert_array_equal(supports[0].run_ids, [0, 0, -1, 1, 1])
    np.testing.assert_array_equal(supports[1].run_ids, [0, 0, 0, 0, 0])


def test_signals_finite_zero_is_valid():
    _, _, supports = _signals(signals=np.zeros((5, 2)))
    for support in supports:
        assert support.sample_valid.all()


def test_signals_break_before_splits_every_feature():
    _, _, supports = _signals(
        signals=np.ones((5, 2)),
        break_before=np.array([False, False, True, False, False]),
    )
    for support in supports:
        np.testing.assert_array_equal(support.run_ids, [0, 0, 1, 1, 1])


def test_signals_one_dimensional_signal_is_one_feature():
    t, sig, supports = _signals(signals=np.arange(5.0))
    assert sig.shape == (5, 1)
    assert sig.dtype == np.float64
    assert t.shape == (5,)
    assert len(supports) == 1


def test_signals_shared_gate_warns_once_for_many_features():
    times = [0.0, 0.1, 0.2, 1.0, 1.1]
    with pytest.warns(UserWarning) as record:
        _signals(times, np.ones((5, 3)), max_gap=0.05)
    assert len(record) == 1


@pytest.mark.parametrize(
    ("times", "signals", "valid"),
    [
        ([0.0, 0.1, 0.1, 0.3, 0.4], np.ones((5, 2)), None),
        (_T5, np.ones((5, 2)), np.ones((5, 2), dtype=int)),
        (_T5, np.ones((5, 2)), np.ones((4, 2), dtype=bool)),
        (_T5, np.ones((5, 2, 1)), None),
        (_T5, np.ones((4, 2)), None),
        ([], np.ones((0, 2)), None),
        ([0.0, 0.1, np.nan, 0.3, 0.4], np.ones((5, 2)), None),
        ([0.4, 0.3, 0.2, 0.1, 0.0], np.ones((5, 2)), None),
        (_T5, np.ones((5, 2), dtype=complex), None),
        (_T5, np.array([["a", "b"]] * 5), None),
        (_T5, np.ones((5, 0)), None),
    ],
    ids=[
        "repeated-time",
        "int-valid",
        "valid-shape",
        "three-dimensional",
        "length-mismatch",
        "empty",
        "nan-time",
        "decreasing-time",
        "complex",
        "non-numeric",
        "zero-features",
    ],
)
def test_signals_errors_name_the_fix(times, signals, valid):
    with pytest.raises(ValueError, match="Fix:"):
        _signals(times, signals, valid=valid)


def test_signals_without_features_say_so():
    with pytest.raises(ValueError, match="at least one feature"):
        _signals(signals=np.ones((5, 0)))


def test_signals_reuse_max_gap_and_break_validation():
    with pytest.raises(ValueError, match="max_gap"):
        _signals(signals=np.ones((5, 2)), max_gap=-1.0)
    with pytest.raises(ValueError, match="break_before"):
        _signals(signals=np.ones((5, 2)), break_before=np.zeros(4, dtype=bool))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_signals_refuse_integers_float64_cannot_hold(dtype):
    sig = np.array([[1, 2**53 + 1]] * 5, dtype=dtype)
    with pytest.raises(ValueError, match=r"column\(s\) \[1\]") as err:
        _signals(signals=sig)
    assert "f()" in str(err.value)
    assert "np.unique" in str(err.value)


def test_signals_accept_integers_at_the_float64_limit():
    sig = np.array([-(2**53), 2**53, 0, 1, 2], dtype=np.int64)
    _signals(signals=sig)
