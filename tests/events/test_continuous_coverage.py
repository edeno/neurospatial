"""Event window coverage, overlaps, event sampling and streamed summaries."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial._intervals import as_intervals
from neurospatial._observation import validated_signals
from neurospatial.events._continuous import (
    _circular_means,
    _event_chunks,
    _overlapping,
    _relative_times,
    _resolve_features,
    _sample_events,
    _Summaries,
    _window_status,
)

EIGHTHS_1S = np.arange(0, 1.0001, 0.125)
EIGHTHS_2S = np.arange(0, 2.0001, 0.125)
# Two runs of dyadic samples separated by a 1 s gap: [0, 1] and [2, 3].
TWO_RUNS = np.r_[np.arange(0, 1.0001, 0.125), np.arange(2, 3.0001, 0.125)]


def _spec(kinds=None, labels=None):
    n = len(kinds) if kinds is not None else 1
    return _resolve_features(
        n,
        feature_ids=None,
        feature_units=None,
        value_kinds=None if kinds is None else dict(enumerate(kinds)),
        category_labels=labels,
        interpolation=None,
        call="test",
    )


def _supports(times, signals, *, epochs=None):
    t, sig, supports = validated_signals(
        times,
        signals,
        valid=None,
        call="test",
        max_gap=0.5,
        epochs=epochs,
        break_before=None,
        result="nothing is sampled",
    )
    return t, sig, supports


def _status(
    times, signals, event_times, window, *, epochs=None, window_coverage="complete"
):
    t, _, supports = _supports(times, signals, epochs=epochs)
    return _window_status(
        np.asarray(event_times, dtype=float),
        window,
        t,
        supports,
        as_intervals(epochs, name="epochs"),
        window_coverage,
    )


# ------------------------------------------------------------ complete status


def test_complete_window_judged_on_interval_not_grid():
    # Queries 0.5, 0.75, 1.0 are all observed, but the window [0.5, 1.05)
    # runs past the last sample.
    status = _status(EIGHTHS_1S, EIGHTHS_1S, [0.5], (0.0, 0.55))
    np.testing.assert_array_equal(status, ["outside_recording"])
    rel = _relative_times((0.0, 0.55), 0.25)
    t, sig, supports = _supports(EIGHTHS_1S, EIGHTHS_1S)
    _, support = _sample_events(np.array([0.5]), rel, t, sig, supports, _spec(), None)
    assert support.all()


def test_unsupported_interval_between_queries_rejects_event():
    x = EIGHTHS_2S.copy()
    x[9] = np.nan  # t = 1.125
    rel = _relative_times((-1.0, 1.0), 0.5)
    t, sig, supports = _supports(EIGHTHS_2S, x)
    _, support = _sample_events(np.array([1.0]), rel, t, sig, supports, _spec(), None)
    assert support.all()
    status = _status(EIGHTHS_2S, x, [1.0], (-1.0, 1.0))
    np.testing.assert_array_equal(status, ["incomplete_support"])


def test_complete_window_inside_one_run_is_retained():
    status = _status(EIGHTHS_2S, EIGHTHS_2S, [1.0, 1.0], (-1.0, 1.0))
    np.testing.assert_array_equal(status, ["retained", "retained"])


def test_every_feature_must_cover_the_window():
    signals = np.column_stack([EIGHTHS_2S, EIGHTHS_2S])
    signals[9, 1] = np.nan
    status = _status(EIGHTHS_2S, signals, [1.0, 0.5], (-0.5, 0.5))
    np.testing.assert_array_equal(status, ["incomplete_support", "retained"])


def test_outside_recording_takes_precedence():
    status = _status(EIGHTHS_2S, EIGHTHS_2S, [0.1], (-0.5, 0.5), epochs=[[0, 1.0]])
    np.testing.assert_array_equal(status, ["outside_recording"])


def test_outside_epochs_takes_precedence_over_support():
    status = _status(EIGHTHS_2S, EIGHTHS_2S, [0.9], (-0.25, 0.25), epochs=[[0, 1.0]])
    np.testing.assert_array_equal(status, ["outside_epochs"])


def test_window_ending_at_epoch_stop_is_inside():
    status = _status(EIGHTHS_2S, EIGHTHS_2S, [0.75], (-0.25, 0.25), epochs=[[0, 1.0]])
    np.testing.assert_array_equal(status, ["retained"])


def test_window_ending_at_last_sample_is_retained():
    status = _status(EIGHTHS_1S, EIGHTHS_1S, [0.5], (-0.5, 0.5))
    np.testing.assert_array_equal(status, ["retained"])


# ------------------------------------------------------------- partial status


def test_partial_window_before_recording_is_excluded():
    status = _status(
        EIGHTHS_1S,
        EIGHTHS_1S,
        [-1.0, -0.25, 1.5, 1.0],
        (-0.25, 0.25),
        window_coverage="partial",
    )
    # [-1.25, -0.75) and [-0.5, 0.0) end at or before times[0]; [1.25, 1.75)
    # starts after times[-1]; [0.75, 1.25) overlaps the recording.
    np.testing.assert_array_equal(
        status,
        ["outside_recording", "outside_recording", "outside_recording", "retained"],
    )


def test_partial_window_starting_at_last_sample_is_kept():
    status = _status(
        EIGHTHS_1S, EIGHTHS_1S, [1.25], (-0.25, 0.25), window_coverage="partial"
    )
    np.testing.assert_array_equal(status, ["retained"])


def test_partial_window_in_gap_is_provisionally_retained():
    status = _status(
        TWO_RUNS, TWO_RUNS, [1.5], (-0.25, 0.25), window_coverage="partial"
    )
    np.testing.assert_array_equal(status, ["retained"])
    t, sig, supports = _supports(TWO_RUNS, TWO_RUNS)
    values, support = _sample_events(
        np.array([1.5]),
        _relative_times((-0.25, 0.25), 0.125),
        t,
        sig,
        supports,
        _spec(),
        None,
    )
    assert support.shape == (1, 4, 1)
    assert not support.any()
    assert np.isnan(values).all()


def test_partial_ignores_epochs_and_support():
    status = _status(
        TWO_RUNS,
        TWO_RUNS,
        [0.9],
        (-0.25, 0.25),
        epochs=[[0, 0.5]],
        window_coverage="partial",
    )
    np.testing.assert_array_equal(status, ["retained"])


# ------------------------------------------------------------------- overlaps


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([1.0, 1.2], [True, True]),
        ([1.0, 3.0], [False, False]),
        ([1.0, 1.0], [True, True]),
        ([1.0, 2.0], [False, False]),  # half-open windows that only touch
        ([3.0, 1.0, 1.4], [False, True, True]),  # input order is kept
        ([1.0, 5.0, 1.8], [True, False, True]),
    ],
)
def test_overlapping(events, expected):
    events = np.asarray(events, dtype=float)
    out = _overlapping(events, (-0.5, 0.5), np.ones(events.size, dtype=bool))
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.bool_


def test_wide_earlier_window_overlaps_later_start():
    # Overlap with an earlier window is judged against the latest stop so far,
    # not only the immediately preceding window.
    out = _overlapping(
        np.array([0.0, 0.5, 2.0]), (0.0, 3.0), np.array([True, True, True])
    )
    np.testing.assert_array_equal(out, [True, True, True])


def test_excluded_event_never_overlaps():
    events = np.array([1.0, 1.2, 1.4, 4.0])
    retained = np.array([True, False, False, True])
    out = _overlapping(events, (-0.5, 0.5), retained)
    np.testing.assert_array_equal(out, [False, False, False, False])


# ------------------------------------------------------------- event sampling


def test_sample_events_keeps_input_order():
    times = np.arange(0, 5.0001, 0.125)
    t, sig, supports = _supports(times, times)
    rel = _relative_times((-0.5, 0.5), 0.25)
    values, support = _sample_events(
        np.array([3.0, 1.0, 3.0]), rel, t, sig, supports, _spec(), None
    )
    assert values.shape == (3, 4, 1)
    assert support.all()
    np.testing.assert_array_equal(values[0], values[2])
    assert not np.array_equal(values[0], values[1])
    np.testing.assert_allclose(values[1, :, 0], 1.0 + rel, atol=1e-12)


def test_sample_events_uses_each_feature_kind():
    times = np.array([0.0, 1.0, 2.0])
    signals = np.column_stack(
        [[0.0, 1.0, 2.0], np.deg2rad([179.0, -179.0, -179.0]), [0.0, 1.0, 1.0]]
    )
    spec = _spec(["linear", "circular", "categorical"], labels={2: {0: "a", 1: "b"}})
    t, sig, supports = validated_signals(
        times,
        signals,
        valid=None,
        call="test",
        max_gap=None,
        epochs=None,
        break_before=None,
        result="nothing is sampled",
    )
    values, support = _sample_events(
        np.array([0.5]), np.array([0.0, 1.0]), t, sig, supports, spec, None
    )
    assert support.all()
    # Halfway between samples: linear interpolates, circular takes the shorter
    # arc across pi, categorical holds the previous code.
    assert values[0, 0, 0] == pytest.approx(0.5)
    assert abs(abs(values[0, 0, 1]) - np.pi) < 1e-12
    assert values[0, 0, 2] == 0.0
    np.testing.assert_allclose(
        values[0, 1], [1.5, np.deg2rad(-179.0), 1.0], rtol=0, atol=1e-12
    )


# ----------------------------------------------------------------- chunking


def test_event_chunks_split_by_cells():
    chunks = list(_event_chunks(10, 3, max_cells=7))
    assert chunks == [slice(i, i + 2) for i in range(0, 10, 2)]


def test_event_chunks_uneven_tail():
    chunks = list(_event_chunks(5, 3, max_cells=7))
    assert chunks == [slice(0, 2), slice(2, 4), slice(4, 5)]


def test_event_chunks_one_event_minimum():
    chunks = list(_event_chunks(3, 10, max_cells=7))
    assert chunks == [slice(0, 1), slice(1, 2), slice(2, 3)]


# --------------------------------------------------------------- summaries


def test_empty_summaries_nan_columns_by_kind():
    spec = _spec(["linear", "circular", "categorical"], labels={2: {0: "a", 1: "b"}})
    s = _Summaries.empty(4, spec)
    assert s.counts.shape == (4, 3)
    assert s.counts.dtype == np.int64
    assert s.category_counts.shape == (4, 2)
    assert s.category_counts.dtype == np.int64
    assert (s.counts == 0).all() and (s.category_counts == 0).all()
    np.testing.assert_array_equal(np.isnan(s.sums[0]), [False, True, True])
    np.testing.assert_array_equal(np.isnan(s.sin_sums[0]), [True, False, True])
    np.testing.assert_array_equal(np.isnan(s.cos_sums[0]), [True, False, True])
    assert (s.sums[:, 0] == 0).all()
    assert (s.sin_sums[:, 1] == 0).all()


def test_add_ignores_unsupported_cells():
    spec = _spec(["linear", "circular", "categorical"], labels={2: {0: "a", 1: "b"}})
    values = np.array(
        [[[2.0, 0.0, 1.0]], [[np.nan, np.nan, np.nan]], [[0.0, np.pi / 2, 0.0]]]
    )
    support = ~np.isnan(values)
    support[2, 0, :] = [True, True, False]
    s = _Summaries.empty(1, spec)
    s.add(values, support)
    np.testing.assert_array_equal(s.counts, [[2, 2, 1]])
    assert s.sums[0, 0] == 2.0
    np.testing.assert_allclose(s.sin_sums[0, 1], 1.0, atol=1e-15)
    np.testing.assert_allclose(s.cos_sums[0, 1], 1.0, atol=1e-15)
    np.testing.assert_array_equal(s.category_counts, [[0, 1]])
    assert np.isnan(s.sums[0, 1:]).all()


def test_circular_means_across_pi():
    spec = _spec(["circular"])
    values = np.deg2rad([179.0, -179.0])[:, None, None] * np.ones((2, 3, 1))
    s = _Summaries.empty(3, spec)
    s.add(values, np.ones(values.shape, dtype=bool))
    means, lengths, defined = _circular_means(s.sin_sums, s.cos_sums, s.counts, 1e-12)
    assert defined.all()
    assert (np.abs(np.abs(means) - np.pi) < 1e-12).all()
    np.testing.assert_allclose(lengths, np.cos(np.deg2rad(1.0)), rtol=1e-12)


def test_circular_mean_of_one_half_turn_is_positive_pi():
    # A sampled heading of pi is wrapped to -pi; atan2 then returns -pi.
    spec = _spec(["circular"])
    s = _Summaries.empty(1, spec)
    s.add(np.full((1, 1, 1), -np.pi), np.ones((1, 1, 1), dtype=bool))
    assert np.arctan2(s.sin_sums, s.cos_sums)[0, 0] == -np.pi
    means, _, defined = _circular_means(s.sin_sums, s.cos_sums, s.counts, 1e-12)
    assert defined[0, 0]
    assert means[0, 0] == np.pi


def test_circular_means_opposite_headings_undefined():
    spec = _spec(["circular"])
    values = np.array([0.0, np.pi])[:, None, None]
    s = _Summaries.empty(1, spec)
    s.add(values, np.ones(values.shape, dtype=bool))
    means, lengths, defined = _circular_means(s.sin_sums, s.cos_sums, s.counts, 1e-12)
    assert lengths[0, 0] <= 1e-12
    assert np.isnan(means[0, 0])
    assert not defined[0, 0]


def test_circular_means_no_support():
    spec = _spec(["circular"])
    s = _Summaries.empty(2, spec)
    means, lengths, defined = _circular_means(s.sin_sums, s.cos_sums, s.counts, 1e-12)
    assert np.isnan(means).all()
    assert np.isnan(lengths).all()
    assert not defined.any()


def _parity_inputs():
    rng = np.random.default_rng(7)
    times = np.arange(0, 60.0, 0.01)
    linear = rng.normal(size=times.size).cumsum()
    for start in rng.integers(0, times.size - 200, size=8):
        linear[start : start + rng.integers(20, 150)] = np.nan
    circular = rng.normal(scale=0.2, size=times.size).cumsum()
    codes = np.repeat(rng.integers(0, 3, size=times.size // 100 + 1), 100)
    signals = np.column_stack([linear, circular, codes[: times.size].astype(float)])
    spec = _spec(
        ["linear", "circular", "categorical"],
        labels={2: {0: "a", 1: "b", 2: "c"}},
    )
    events = rng.uniform(2.0, 58.0, size=50)
    return times, signals, spec, events


@pytest.mark.parametrize("events_per_chunk", [1, 7])
def test_chunked_summaries_match_single_pass(events_per_chunk):
    times, signals, spec, events = _parity_inputs()
    t, sig, supports = _supports(times, signals)
    rel = _relative_times((-1.0, 1.0), 0.05)
    cells = rel.size * sig.shape[1]

    chunked = _Summaries.empty(rel.size, spec)
    chunks = list(_event_chunks(events.size, cells, max_cells=events_per_chunk * cells))
    assert len(chunks) == -(-events.size // events_per_chunk)
    if events_per_chunk > 1:  # an uneven last chunk
        assert chunks[-1].stop - chunks[-1].start < events_per_chunk
    for chunk in chunks:
        chunked.add(*_sample_events(events[chunk], rel, t, sig, supports, spec, None))

    values, support = _sample_events(events, rel, t, sig, supports, spec, None)
    whole = _Summaries.empty(rel.size, spec)
    whole.add(values, support)

    # The NaN gaps leave some cells with fewer events.
    assert (whole.counts[:, 0] < events.size).any()
    np.testing.assert_array_equal(chunked.counts, whole.counts)
    np.testing.assert_array_equal(chunked.category_counts, whole.category_counts)
    for name in ("sums", "sin_sums", "cos_sums"):
        whole_sums = getattr(whole, name)
        scale = np.nanmax(np.abs(whole_sums))
        np.testing.assert_allclose(
            getattr(chunked, name), whole_sums, rtol=1e-12, atol=1e-9 * scale
        )
    assert whole.category_counts.sum() == whole.counts[:, 2].sum()

    observed = whole.counts[:, 0] > 0
    np.testing.assert_allclose(
        whole.sums[observed, 0] / whole.counts[observed, 0],
        np.nanmean(values[:, observed, 0], axis=0),
        rtol=1e-12,
    )
