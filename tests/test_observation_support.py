"""Observation-support resolution: validity, breaks, gaps, epochs and runs."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import behavior
from neurospatial._intervals import validate_max_gap
from neurospatial._observation import (
    ObservationSupport,
    label_codes,
    label_runs,
    resolve_observation_support,
    resolve_sample_support,
    validate_break_before,
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


def _nan_row_support(max_gap):
    positions = np.c_[np.arange(5, dtype=float), np.zeros(5)]
    positions[2] = np.nan
    return validated_observations(
        np.arange(5) * 0.1,
        positions,
        call="f",
        max_gap=max_gap,
        epochs=None,
        break_before=None,
        result="r",
    )[2]


def test_nan_row_is_bridged_like_a_deleted_row():
    s = _nan_row_support(max_gap=0.5)
    deleted = _support(np.delete(np.arange(5) * 0.1, 2), max_gap=0.5)
    # The 0.2 s gap between samples 1 and 3 is bridged; the NaN sample is in
    # no run, but the run spans it.
    np.testing.assert_array_equal(s.interval_mask, [True, True, True, True])
    np.testing.assert_array_equal(s.run_ids, [0, 0, -1, 0, 0])
    np.testing.assert_array_equal(s.run_sample_bounds, [[0, 4]])
    np.testing.assert_allclose(s.run_time_bounds, deleted.run_time_bounds)
    assert not s.sample_valid[2]
    assert not s.isolated.any()


def test_nan_row_gap_longer_than_max_gap_breaks_the_path():
    s = _nan_row_support(max_gap=0.15)
    np.testing.assert_array_equal(s.interval_mask, [True, False, False, True])
    np.testing.assert_array_equal(s.run_ids, [0, 0, -1, 1, 1])
    np.testing.assert_array_equal(s.interval_gates, [True, False, False, True])


@pytest.mark.filterwarnings("ignore:.*max_gap:UserWarning")
def test_bridged_gaps_longer_than_max_gap_warn_once():
    # Each 0.1 s sampling interval passes max_gap; each 0.2 s gap across a
    # missing row does not, so nothing is usable.
    x = np.arange(10.0)
    x[1::2] = np.nan
    with pytest.warns(UserWarning, match="excluded ALL") as record:
        support = _support(np.arange(10) * 0.1, x, max_gap=0.15)
    assert len(record) == 1
    assert "gaps across missing samples" in str(record[0].message)
    assert not support.interval_mask.any()


def test_no_observed_sample_warns_that_nothing_is_usable():
    with pytest.warns(UserWarning, match="excluded ALL.*samples are finite"):
        _support(np.arange(5) * 0.1, np.full(5, np.nan))


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


def test_sample_in_epochs_excludes_the_stop_instant():
    """The interval ending at an epoch's stop is inside; that sample is not."""
    s = _support([0, 0.5, 1.0, 1.5, 2.0], epochs=[(0, 1.0), (1.5, 3.0)])
    np.testing.assert_array_equal(s.sample_in_epochs, [True, True, False, True, True])
    np.testing.assert_array_equal(_support([0, 1.0]).sample_in_epochs, [True, True])


def test_max_gap_none_keeps_long_interval():
    s = _support([0, 10], max_gap=None)
    np.testing.assert_array_equal(s.interval_mask, [True])
    with pytest.warns(UserWarning, match="max_gap"):
        assert not _support([0, 10]).interval_mask.any()


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, "0.5", True])
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
        "interval_gates",
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
    assert why == "Why: there is no observation to measure."
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


def test_signals_nan_in_one_feature_only_affects_that_feature():
    sig = np.ones((5, 2))
    sig[2, 1] = np.nan
    _, _, supports = _signals(signals=sig)
    assert len(supports) == 2
    np.testing.assert_array_equal(supports[0].run_ids, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(supports[1].run_ids, [0, 0, -1, 0, 0])
    assert not supports[1].sample_valid[2]
    _, _, supports = _signals(signals=sig, max_gap=0.15)
    np.testing.assert_array_equal(supports[0].run_ids, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(supports[1].run_ids, [0, 0, -1, 1, 1])


def test_signals_valid_one_dimensional_applies_to_every_feature():
    valid = np.array([True, True, False, True, True])
    _, _, supports = _signals(signals=np.ones((5, 2)), valid=valid, max_gap=0.15)
    for support in supports:
        np.testing.assert_array_equal(support.run_ids, [0, 0, -1, 1, 1])


def test_signals_valid_two_dimensional_applies_per_feature():
    valid = np.ones((5, 2), dtype=bool)
    valid[2, 0] = False
    _, _, supports = _signals(signals=np.ones((5, 2)), valid=valid, max_gap=0.15)
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


def test_signals_warn_once_when_bridged_gaps_exclude_every_feature():
    # Every sampling interval passes max_gap; every gap across a missing row
    # does not, in every feature.
    times = np.arange(20) * 0.1
    sig = np.ones((20, 2))
    sig[1::2] = np.nan
    with pytest.warns(UserWarning, match="excluded ALL") as record:
        _, _, supports = _signals(times, sig, max_gap=0.15)
    assert len(record) == 1
    assert "gaps across missing samples" in str(record[0].message)
    assert not any(support.interval_mask.any() for support in supports)


def test_signals_one_usable_feature_keeps_the_table_silent(recwarn):
    times = np.arange(20) * 0.1
    sig = np.ones((20, 2))
    sig[1::2, 1] = np.nan
    _, _, supports = _signals(times, sig, max_gap=0.15)
    assert supports[0].interval_mask.all()
    assert not supports[1].interval_mask.any()
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    ("times", "signals", "valid", "reason"),
    [
        ([0.0, 0.1, 0.1, 0.3, 0.4], np.ones((5, 2)), None, "strictly increasing"),
        (_T5, np.ones((5, 2)), np.ones((5, 2), dtype=int), "got dtype int64"),
        (_T5, np.ones((5, 2)), np.ones((4, 2), dtype=bool), r"shape \(4, 2\)"),
        (_T5, np.ones((5, 2, 1)), None, r"got shape \(5, 2, 1\)"),
        (_T5, np.ones((4, 2)), None, r"got shape \(4, 2\)"),
        ([], np.ones((0, 2)), None, "got no samples"),
        ([0.0, 0.1, np.nan, 0.3, 0.4], np.ones((5, 2)), None, "non-finite"),
        ([0.4, 0.3, 0.2, 0.1, 0.0], np.ones((5, 2)), None, "decreases"),
        (_T5, np.ones((5, 2), dtype=complex), None, "real numeric signals"),
        (_T5, np.array([["a", "b"]] * 5), None, "real numeric signals"),
        (_T5, np.ones((5, 0)), None, "at least one feature"),
        (list("abcde"), np.ones((5, 2)), None, "numeric values convertible"),
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
        "non-numeric-times",
    ],
)
def test_signals_errors_name_the_reason_and_fix(times, signals, valid, reason):
    with pytest.raises(ValueError, match=reason) as caught:
        _signals(times, signals, valid=valid)
    assert "\nWhy: " in str(caught.value)
    assert "\nFix: " in str(caught.value)


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
    _, values, _ = _signals(signals=sig)
    np.testing.assert_array_equal(
        values[:, 0], np.array([-(2**53), 2**53, 0, 1, 2], dtype=np.float64)
    )


def test_interval_gates_are_evaluated_on_bridged_gaps():
    """``interval_mask == interval_gates`` where both sides have a valid sample."""
    times = np.array([0.0, 0.1, 0.2, 0.3, 1.5, 1.6, 1.7, 1.8])
    positions = np.c_[np.arange(8.0), np.zeros(8)]
    positions[1] = np.nan
    positions[7] = np.nan
    break_before = np.zeros(8, dtype=bool)
    break_before[6] = True
    s = validated_observations(
        times,
        positions,
        call="f",
        max_gap=0.5,
        epochs=None,
        break_before=break_before,
        result="r",
    )[2]
    # gap after sample 3 (1.2 s) and the break before sample 6 close gates;
    # the NaN row inside a 0.2 s gap does not, and the trailing NaN row has
    # no valid sample after it, so its interval is gated on its own samples.
    np.testing.assert_array_equal(
        s.interval_gates, [True, True, True, False, True, False, True]
    )
    np.testing.assert_array_equal(
        s.interval_mask, [True, True, True, False, True, False, False]
    )


@pytest.mark.parametrize("n", [1, 2])
def test_interval_gates_shape(n):
    s = _support(np.arange(n, dtype=float) * 0.1)
    assert s.interval_gates.shape == (max(n - 1, 0),)


# --- sample-validity entry point and label runs ------------------------------


def test_resolve_sample_support_matches_position_entry_point():
    rng = np.random.default_rng(3)
    times = np.cumsum(rng.uniform(0.05, 0.15, size=60))
    times[30:] += 2.0  # one gap longer than max_gap
    positions = rng.normal(size=(60, 2))
    positions[rng.choice(60, size=8, replace=False), rng.integers(0, 2, 8)] = np.nan
    breaks = np.zeros(60, dtype=bool)
    breaks[45] = True
    kwargs = {"max_gap": 0.5, "epochs": None, "break_before": breaks, "result": "r"}
    expected = resolve_observation_support(times, positions, **kwargs)
    actual = resolve_sample_support(times, np.isfinite(positions).all(axis=1), **kwargs)
    assert not expected.sample_valid.all()
    assert expected.run_sample_bounds.shape[0] > 2
    for name in (
        "sample_valid",
        "interval_gates",
        "interval_mask",
        "run_ids",
        "run_sample_bounds",
        "run_time_bounds",
    ):
        a, b = getattr(actual, name), getattr(expected, name)
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_label_runs_merges_only_adjacent_retained_equal_codes():
    codes = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, -1, 1], dtype=np.intp)
    mask = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    first, last, code = label_runs(codes, mask)
    np.testing.assert_array_equal(first, [0, 2, 4, 7, 8, 10])
    np.testing.assert_array_equal(last, [1, 3, 4, 7, 8, 10])
    np.testing.assert_array_equal(code, [0, 1, 0, 0, 1, 1])


def test_label_runs_with_nothing_retained_is_empty():
    for codes, mask in (
        (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=bool)),
        (np.array([-1, 0], dtype=np.intp), np.array([True, False])),
    ):
        first, last, code = label_runs(codes, mask)
        assert first.shape == last.shape == code.shape == (0,)


def test_label_codes_declared_missing_integer():
    codes, values = label_codes(np.array([0, 0, 1, -1, 0]), -1)
    np.testing.assert_array_equal(codes, [0, 0, 1, -1, 0])
    assert codes.dtype == np.intp
    assert values == [0, 1]
    assert type(values[0]) is int


def test_label_codes_none_and_nan_are_missing():
    labels = np.array(["a", None, "b", np.nan, "a"], dtype=object)
    codes, values = label_codes(labels, None)
    np.testing.assert_array_equal(codes, [0, -1, 1, -1, 0])
    assert values == ["a", "b"]


def test_missing_label_that_differs_only_in_case_warns():
    labels = np.array(["rest", "run", "rest"], dtype=object)
    with pytest.warns(UserWarning, match="missing_label='Rest'.*'rest'"):
        _, values = label_codes(labels, "Rest")
    assert values == ["rest", "run"]


def test_absent_missing_label_without_a_near_match_is_silent(recwarn):
    label_codes(np.array(["run", "run"], dtype=object), "stationary")
    assert not recwarn.list


def test_validated_break_before_is_read_only():
    flags = np.array([False, True, False])
    owned = validate_break_before(flags, 3, call="demo")
    assert not owned.flags.writeable
    flags[1] = False
    assert owned[1]
