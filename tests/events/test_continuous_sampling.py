"""Relative query grid and run-bounded sampling of one continuous feature."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial._intervals import as_intervals
from neurospatial._observation import validated_signals
from neurospatial.events._continuous import _relative_times, _sample_feature

# Two runs of dyadic samples separated by a 1 s gap: [0, 1] and [2, 3].
TWO_RUNS = np.r_[np.arange(0, 1.0001, 0.125), np.arange(2, 3.0001, 0.125)]
QUARTERS = np.arange(0, 2.0001, 0.25)


def _sample(
    times,
    values,
    queries,
    *,
    interpolation="linear",
    kind="linear",
    max_gap=0.5,
    epochs=None,
    break_before=None,
):
    """Resolve support for one feature, then sample it at ``queries``."""
    times = np.asarray(times, dtype=float)
    with warnings.catch_warnings():
        # Sparse clocks trip the shared "times are probably not in seconds"
        # warning, which is not what these tests check.
        warnings.filterwarnings(
            "ignore", message=r".*probably not in seconds", category=UserWarning
        )
        t, sig, (support,) = validated_signals(
            times,
            np.asarray(values, dtype=float),
            valid=None,
            call="test",
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
            result="nothing is sampled",
        )
    return _sample_feature(
        t,
        sig[:, 0],
        support,
        np.asarray(queries, dtype=float),
        interpolation=interpolation,
        kind=kind,
        epochs=as_intervals(epochs, name="epochs"),
    )


# ---------------------------------------------------------------- query grid


def test_grid_tenth_second_steps():
    rel = _relative_times((-1.0, 2.0), 0.1)
    assert rel.shape == (30,)
    assert rel.dtype == np.float64
    assert rel[0] == -1.0
    assert rel[-1] == pytest.approx(1.9)


def test_grid_step_not_dividing_window():
    np.testing.assert_allclose(_relative_times((0.0, 1.0), 0.3), [0, 0.3, 0.6, 0.9])


def test_grid_query_at_the_event_is_exactly_zero():
    # -0.33 + 0.03 * 11 is -5.6e-17: a "previous" sample at the event would be
    # read from the sample before it.
    rel = _relative_times((-0.33, 0.33), 0.03)
    assert (rel == 0.0).sum() == 1
    t = np.array([0.0, 0.25, 0.5, 0.75])
    values, support = _sample(
        t, [0.0, 1.0, 2.0, 3.0], 0.5 + rel, interpolation="previous", max_gap=None
    )
    assert support[rel == 0.0].all()
    assert values[rel == 0.0] == 2.0


def test_grid_excludes_stop():
    rel = _relative_times((-0.75, 0.75), 0.25)
    np.testing.assert_array_equal(rel, [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5])


# ---------------------------------------------------------------------- wrap


# ------------------------------------------------------------------ sampling


def test_irregular_sampling_reproduces_linear_signal():
    times = np.array([0, 0.13, 0.31, 0.5, 0.74, 1.0, 1.21, 1.5, 1.77, 2.0])
    queries = 1.0 + _relative_times((-0.75, 0.75), 0.25)
    values, support = _sample(times, 3 * times + 1, queries)
    assert support.all()
    np.testing.assert_allclose(values, 3 * queries + 1, rtol=0, atol=1e-12)


def test_circular_takes_shorter_arc_across_pi():
    values, support = _sample(
        [0.0, 1.0],
        np.deg2rad([179.0, -179.0]),
        [0.5],
        interpolation="circular",
        kind="circular",
        max_gap=None,
    )
    assert support.all()
    assert abs(abs(values[0]) - np.pi) < 1e-12


def test_circular_exact_half_turn_goes_positive():
    values, _ = _sample(
        [0.0, 1.0],
        [0.0, np.pi],
        [0.5],
        interpolation="circular",
        kind="circular",
        max_gap=None,
    )
    assert values[0] == pytest.approx(np.pi / 2, abs=1e-12)


def test_circular_exact_sample_is_wrapped():
    values, support = _sample(
        [0.0, 1.0],
        [3 * np.pi / 2, 0.0],
        [0.0],
        interpolation="circular",
        kind="circular",
    )
    assert support.all()
    assert values[0] == pytest.approx(-np.pi / 2, abs=1e-12)


def test_circular_previous_is_wrapped():
    # Holding the previous sample must still report a wrapped angle.
    values, support = _sample(
        [0.0, 1.0],
        [3 * np.pi / 2, 3 * np.pi / 2],
        [0.0, 0.5],
        interpolation="previous",
        kind="circular",
        max_gap=None,
    )
    assert support.all()
    np.testing.assert_allclose(values, [-np.pi / 2, -np.pi / 2], rtol=0, atol=1e-12)


def test_never_bridges_a_gap():
    values, support = _sample(TWO_RUNS, TWO_RUNS, [1.25, 1.375, 1.5, 1.625, 1.0])
    assert np.isnan(values[:4]).all()
    assert not support[:4].any()
    assert support[4]
    assert values[4] == 1.0


def test_no_extrapolation_past_recording_edges():
    values, support = _sample(TWO_RUNS, TWO_RUNS, [-0.1, 3.1, 0.0, 3.0])
    np.testing.assert_array_equal(support, [False, False, True, True])
    assert np.isnan(values[:2]).all()
    np.testing.assert_array_equal(values[2:], [0.0, 3.0])


def test_invalid_sample_is_bridged_like_a_deleted_row():
    x = TWO_RUNS.copy()
    x[3] = np.nan  # t = 0.375
    queries = [0.26, 0.3, 0.375, 0.45, 0.49, 0.25, 0.5]
    values, support = _sample(TWO_RUNS, x, queries)
    kept = np.delete(np.arange(TWO_RUNS.size), 3)
    deleted, deleted_support = _sample(TWO_RUNS[kept], x[kept], queries)
    assert support.all()
    np.testing.assert_array_equal(support, deleted_support)
    np.testing.assert_allclose(values, deleted)
    np.testing.assert_allclose(values, queries)  # the line x = t, interpolated


def test_invalid_sample_beyond_max_gap_blocks_its_intervals_and_instant():
    x = TWO_RUNS.copy()
    x[3] = np.nan  # t = 0.375
    queries = [0.26, 0.3, 0.375, 0.45, 0.49, 0.25, 0.5]
    values, support = _sample(TWO_RUNS, x, queries, max_gap=0.2)
    np.testing.assert_array_equal(
        support, [False, False, False, False, False, True, True]
    )
    assert np.isnan(values[:5]).all()
    np.testing.assert_array_equal(values[5:], [0.25, 0.5])


def test_infinite_sample_is_missing():
    x = TWO_RUNS.copy()
    x[3] = np.inf
    values, support = _sample(TWO_RUNS, x, [0.3, 0.375], max_gap=0.2)
    assert not support.any()
    assert np.isnan(values).all()


def test_isolated_sample_supports_only_its_instant():
    times = [0, 0.1, 5, 10, 10.1]
    values, support = _sample(times, [1.0, 2.0, 7.0, 4.0, 5.0], [5.0, 5.05])
    np.testing.assert_array_equal(support, [True, False])
    assert values[0] == 7.0
    assert np.isnan(values[1])


def test_single_sample_recording():
    values, support = _sample([2.0], [4.0], [2.0, 2.1, 1.9])
    np.testing.assert_array_equal(support, [True, False, False])
    assert values[0] == 4.0


def test_previous_holds_category_codes():
    values, support = _sample(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 1.99],
        interpolation="previous",
        max_gap=None,
    )
    assert support.all()
    np.testing.assert_array_equal(values, [0, 1, 1])


def test_previous_does_not_hold_past_last_sample():
    values, support = _sample(
        [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [2.5], interpolation="previous", max_gap=None
    )
    assert not support[0]
    assert np.isnan(values[0])


def test_finite_zero_is_supported():
    values, support = _sample([0.0, 1.0], [0.0, 0.0], [0.0, 0.5, 1.0], max_gap=None)
    assert support.all()
    np.testing.assert_array_equal(values, [0.0, 0.0, 0.0])


def test_epoch_stop_instant_is_unsupported():
    values, support = _sample(QUARTERS, QUARTERS, [1.0, 0.75, 0.8], epochs=[[0, 1.0]])
    np.testing.assert_array_equal(support, [False, True, True])
    assert np.isnan(values[0])


def test_epoch_edge_between_samples():
    # The epoch [0.1, 1.6) cuts the intervals [0, 0.25) and [1.5, 1.75), so
    # neither is usable; the sample at 1.5 is still observed inside the epoch.
    values, support = _sample(
        QUARTERS, QUARTERS, [0.2, 1.55, 1.5, 0.3], epochs=[[0.1, 1.6]]
    )
    np.testing.assert_array_equal(support, [False, False, True, True])
    assert np.isnan(values[:2]).all()
    np.testing.assert_allclose(values[2:], [1.5, 0.3], atol=1e-12)


def test_exact_sample_outside_epochs_is_unsupported():
    values, support = _sample(
        QUARTERS, QUARTERS, [0.25, 1.5], epochs=[[0.5, 1.0], [1.25, 2.0]]
    )
    np.testing.assert_array_equal(support, [False, True])
    assert values[1] == 1.5


def test_declared_break_is_not_bridged():
    breaks = np.zeros(QUARTERS.size, dtype=bool)
    breaks[4] = True
    values, support = _sample(
        QUARTERS, QUARTERS, [0.875, 0.75, 1.0], break_before=breaks
    )
    np.testing.assert_array_equal(support, [False, True, True])
    assert np.isnan(values[0])


def test_query_shape_is_preserved():
    queries = np.linspace(-0.5, 2.5, 12).reshape(3, 4)
    values, support = _sample(QUARTERS, QUARTERS, queries)
    assert values.shape == (3, 4)
    assert support.shape == (3, 4)
    assert support.dtype == np.bool_
    inside = (queries >= 0) & (queries <= 2)
    np.testing.assert_array_equal(support, inside)
    np.testing.assert_allclose(values[inside], queries[inside], atol=1e-12)
    assert np.isnan(values[~inside]).all()
