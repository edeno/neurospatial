"""Public alignment of continuous signals to events and its result."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

import neurospatial.events._continuous as continuous_kernels
import neurospatial.events.continuous as continuous_module
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial.events import ContinuousEventResult, align_continuous_to_events
from neurospatial.events._continuous import _relative_times

EIGHTHS = np.arange(0, 10.0001, 0.125)
# Two runs of dyadic samples separated by a 2 s gap: [0, 1] and [3, 4].
TWO_RUNS = np.r_[np.arange(0, 1.0001, 0.125), np.arange(3, 4.0001, 0.125)]


# ------------------------------------------------------------- event cohort


def test_event_cohort_keeps_input_order_and_duplicates():
    result = align_continuous_to_events(
        EIGHTHS, EIGHTHS, [3.0, 1.0, 3.0, 99.0], (-0.5, 0.5), dt=0.25
    )
    np.testing.assert_array_equal(
        result.event_status,
        ["retained", "retained", "retained", "outside_recording"],
    )
    np.testing.assert_array_equal(result.event_indices, [0, 1, 2])
    assert result.event_indices.dtype == np.intp
    assert result.n_events == 3
    assert result.n_events_dropped == 1
    assert result.values.shape == (3, 4, 1)
    np.testing.assert_array_equal(result.values[0], result.values[2])
    np.testing.assert_array_equal(result.values[1, :, 0], 1.0 + result.relative_times)
    assert result.support.all()
    # Duplicate times overlap each other; excluded events never overlap.
    np.testing.assert_array_equal(result.overlapping, [True, False, True, False])
    assert result.event_ids is None


def test_event_ids_label_every_input_event():
    result = align_continuous_to_events(
        EIGHTHS,
        EIGHTHS,
        [3.0, 1.0, 3.0, 99.0],
        (-0.5, 0.5),
        dt=0.25,
        event_ids=["a", "b", "c", "d"],
    )
    np.testing.assert_array_equal(result.event_ids, ["a", "b", "c", "d"])
    np.testing.assert_array_equal(result.event_indices, [0, 1, 2])
    np.testing.assert_array_equal(
        result.event_ids[result.event_indices], ["a", "b", "c"]
    )


def test_result_records_settings():
    result = align_continuous_to_events(
        EIGHTHS,
        EIGHTHS,
        [3.0],
        (-0.5, 0.5),
        dt=0.25,
        feature_ids=["speed"],
        feature_units={"speed": "cm/s"},
        epochs=[[5.0, 9.0], [0.0, 5.0]],
    )
    np.testing.assert_array_equal(result.relative_times, [-0.5, -0.25, 0.0, 0.25])
    np.testing.assert_array_equal(result.feature_ids, ["speed"])
    assert result.value_kinds == ("linear",)
    assert result.interpolation == ("linear",)
    assert result.feature_units == ("cm/s",)
    assert result.window == (-0.5, 0.5)
    assert result.dt == 0.25
    assert result.window_coverage == "complete"
    assert result.resultant_threshold == 1e-12
    assert result.max_gap == DEFAULT_MAX_GAP
    # Epochs are stored normalized: sorted and merged.
    np.testing.assert_array_equal(result.epochs, [[0.0, 9.0]])


# --------------------------------------------------------- window coverage


def test_partial_coverage_keeps_supported_cells():
    # Event 2.0625 sits in the gap: no query lands on a sample or a usable
    # interval. Event 0.75 is observed up to the last sample of run 0, t = 1.
    result = align_continuous_to_events(
        TWO_RUNS,
        TWO_RUNS,
        [0.75, 2.0625, 3.5],
        (-0.5, 0.5),
        dt=0.125,
        window_coverage="partial",
    )
    np.testing.assert_array_equal(
        result.event_status, ["retained", "no_support", "retained"]
    )
    np.testing.assert_array_equal(result.event_indices, [0, 2])
    queries = 0.75 + result.relative_times
    np.testing.assert_array_equal(result.support[0, :, 0], queries <= 1.0)
    assert np.isnan(result.values[0, queries > 1.0, 0]).all()
    assert result.support[1].all()

    for j in range(result.relative_times.size):
        assert result.counts[j, 0] == result.support[:, j, 0].sum()
    observed = result.counts[:, 0] > 0
    np.testing.assert_allclose(
        result.means[observed, 0],
        np.nanmean(result.values[:, observed, 0], axis=0),
        rtol=1e-12,
    )


def test_window_starting_on_last_sample_of_a_run_is_a_point_observation():
    # The first query of event 1.5 is t = 1.0, the last valid sample of run
    # 0: a point observation, so the event is kept with that one cell.
    result = align_continuous_to_events(
        TWO_RUNS, TWO_RUNS, [1.5], (-0.5, 0.5), dt=0.125, window_coverage="partial"
    )
    np.testing.assert_array_equal(result.event_status, ["retained"])
    np.testing.assert_array_equal(
        result.support[0, :, 0], result.relative_times == -0.5
    )
    assert result.values[0, 0, 0] == 1.0


def test_complete_coverage_needs_one_run_over_the_window():
    result = align_continuous_to_events(
        TWO_RUNS, TWO_RUNS, [0.75, 2.0625, 3.5], (-0.5, 0.5), dt=0.125
    )
    np.testing.assert_array_equal(
        result.event_status,
        ["incomplete_support", "incomplete_support", "retained"],
    )
    np.testing.assert_array_equal(result.event_indices, [2])
    assert result.support.all()


def test_window_coverage_outside_epochs():
    result = align_continuous_to_events(
        EIGHTHS, EIGHTHS, [2.0, 6.0], (-0.5, 0.5), dt=0.25, epochs=[[0.0, 5.0]]
    )
    np.testing.assert_array_equal(result.event_status, ["retained", "outside_epochs"])


def test_per_feature_invalid_samples_are_bridged_like_deleted_rows():
    signals = np.column_stack([EIGHTHS, 2 * EIGHTHS])
    valid = np.ones(signals.shape, dtype=bool)
    valid[(EIGHTHS >= 5.75) & (EIGHTHS <= 6.0), 1] = False
    result = align_continuous_to_events(
        EIGHTHS, signals, [3.0, 6.0], (-0.5, 0.5), dt=0.25, valid=valid, max_gap=0.5
    )
    np.testing.assert_array_equal(result.event_status, ["retained", "retained"])
    assert result.support.all()
    # Feature 1 is the line 2 t, interpolated across its missing samples.
    np.testing.assert_allclose(
        result.values[1, :, 1], 2 * (6.0 + result.relative_times)
    )


def test_per_feature_validity_decides_coverage():
    signals = np.column_stack([EIGHTHS, 2 * EIGHTHS])
    valid = np.ones(signals.shape, dtype=bool)
    valid[(EIGHTHS >= 5.75) & (EIGHTHS <= 6.0), 1] = False

    complete = align_continuous_to_events(
        EIGHTHS, signals, [3.0, 6.0], (-0.5, 0.5), dt=0.25, valid=valid, max_gap=0.2
    )
    np.testing.assert_array_equal(
        complete.event_status, ["retained", "incomplete_support"]
    )

    partial = align_continuous_to_events(
        EIGHTHS,
        signals,
        [3.0, 6.0],
        (-0.5, 0.5),
        dt=0.25,
        valid=valid,
        window_coverage="partial",
        max_gap=0.2,
    )
    np.testing.assert_array_equal(partial.event_status, ["retained", "retained"])
    assert partial.support[1, :, 0].all()
    # Queries 5.5, 5.75, 6.0, 6.25: the invalid samples 5.75 and 6.0 block
    # their instants and the intervals around them.
    np.testing.assert_array_equal(partial.support[1, :, 1], [True, False, False, True])
    np.testing.assert_array_equal(partial.counts[:, 1], [2, 1, 1, 2])


# ------------------------------------------------------- summary-only mode


def _mixed_kind_inputs():
    rng = np.random.default_rng(7)
    times = np.arange(0, 60.0, 0.01)
    linear = rng.normal(size=times.size).cumsum()
    for start in rng.integers(0, times.size - 200, size=8):
        linear[start : start + rng.integers(20, 150)] = np.nan
    circular = rng.normal(scale=0.2, size=times.size).cumsum()
    codes = np.repeat(rng.integers(0, 3, size=times.size // 100 + 1), 100)
    signals = np.column_stack([linear, circular, codes[: times.size].astype(float)])
    events = rng.uniform(2.0, 58.0, size=50)
    kwargs = {
        "dt": 0.05,
        "feature_ids": ["speed", "heading", "zone"],
        "value_kinds": {"heading": "circular", "zone": "categorical"},
        "category_labels": {"zone": {0: "a", 1: "b", 2: "c"}},
        "window_coverage": "partial",
    }
    return times, signals, events, (-1.0, 1.0), kwargs


def test_summary_only_agrees_with_retained_events(monkeypatch):
    times, signals, events, window, kwargs = _mixed_kind_inputs()
    # Shrink the chunk budget to 7 events' cells and record the chunks used.
    cells = _relative_times(window, kwargs["dt"]).size * signals.shape[1]
    monkeypatch.setattr(continuous_kernels, "_MAX_CHUNK_CELLS", 7 * cells)
    chunk_sizes = []

    def recording_chunks(*args, **chunk_kwargs):
        for chunk in continuous_kernels._event_chunks(*args, **chunk_kwargs):
            chunk_sizes.append(chunk.stop - chunk.start)
            yield chunk

    monkeypatch.setattr(continuous_module, "_event_chunks", recording_chunks)

    full = align_continuous_to_events(times, signals, events, window, **kwargs)
    summary = align_continuous_to_events(
        times, signals, events, window, retain_aligned=False, **kwargs
    )
    assert full.n_events == events.size
    # Both calls streamed 50 events as 7, ..., 7, 1.
    assert chunk_sizes == 2 * ([7] * 7 + [1])
    assert summary.values is None
    assert summary.support is None
    np.testing.assert_array_equal(summary.event_status, full.event_status)
    np.testing.assert_array_equal(summary.counts, full.counts)
    np.testing.assert_array_equal(summary.category_counts, full.category_counts)
    np.testing.assert_array_equal(summary.mean_status, full.mean_status)
    np.testing.assert_allclose(summary.means, full.means, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        summary.resultant_lengths, full.resultant_lengths, rtol=1e-12, equal_nan=True
    )

    # The streamed statistics equal a single pass over the retained tensor.
    values, support = full.values, full.support
    np.testing.assert_array_equal(summary.counts, support.sum(axis=0))
    assert (summary.counts[:, 0] < events.size).any()  # NaN gaps are exercised
    observed = summary.counts[:, 0] > 0
    linear_means = (
        np.nansum(values[:, observed, 0], axis=0) / summary.counts[observed, 0]
    )
    scale = np.nanmax(np.abs(values[..., 0]))
    np.testing.assert_allclose(
        summary.means[observed, 0], linear_means, rtol=1e-12, atol=1e-9 * scale
    )
    angles = values[..., 1]
    sin_sum = np.nansum(np.sin(angles), axis=0)
    cos_sum = np.nansum(np.cos(angles), axis=0)
    np.testing.assert_allclose(
        summary.resultant_lengths[:, 1],
        np.hypot(sin_sum, cos_sum) / summary.counts[:, 1],
        rtol=1e-12,
        atol=1e-9,
    )
    defined = summary.mean_status[:, 1] == "defined"
    assert defined.all()
    np.testing.assert_allclose(
        summary.means[:, 1], np.arctan2(sin_sum, cos_sum), rtol=1e-12, atol=1e-9 * np.pi
    )
    for row, code in enumerate(summary.category_codes):
        np.testing.assert_array_equal(
            summary.category_counts[:, row], (values[..., 2] == code).sum(axis=0)
        )


# ------------------------------------------------------------- value kinds


def test_circular_mean_across_the_wrap():
    heading = np.where(EIGHTHS < 5.0, np.deg2rad(179.0), np.deg2rad(-179.0))
    result = align_continuous_to_events(
        EIGHTHS,
        heading,
        [2.0, 7.0],
        (-0.5, 0.5),
        dt=0.25,
        value_kinds={0: "circular"},
    )
    assert result.feature_units == ("rad",)
    assert (np.abs(np.abs(result.means) - np.pi) < 1e-9).all()
    assert (result.mean_status == "defined").all()
    np.testing.assert_allclose(
        result.resultant_lengths, np.cos(np.deg2rad(1.0)), rtol=1e-12
    )


def test_circular_mean_at_half_turn_is_positive_pi():
    # Sampled headings and their mean share the (-pi, pi] convention.
    heading = np.full(EIGHTHS.size, np.pi)
    result = align_continuous_to_events(
        EIGHTHS, heading, [3.0], (-0.5, 0.5), dt=0.1, value_kinds={0: "circular"}
    )
    assert (result.values == np.pi).all()
    np.testing.assert_array_equal(result.means, np.pi)


def test_opposite_headings_have_no_direction():
    heading = np.where(EIGHTHS < 5.0, 0.0, np.pi)
    result = align_continuous_to_events(
        EIGHTHS, heading, [2.0, 7.0], (-0.5, 0.5), dt=0.25, value_kinds={0: "circular"}
    )
    assert (result.mean_status == "undefined_direction").all()
    assert np.isnan(result.means).all()


def test_categorical_counts_and_no_numeric_mean():
    zone = (EIGHTHS >= 5.0).astype(float)
    signals = np.column_stack([EIGHTHS, zone])
    result = align_continuous_to_events(
        EIGHTHS,
        signals,
        [2.0, 4.875, 7.0],
        (-0.5, 0.5),
        dt=0.25,
        feature_ids=["speed", "zone"],
        value_kinds={"zone": "categorical"},
        category_labels={"zone": {1: "arm", 0: "center"}},
    )
    assert result.interpolation == ("linear", "previous")
    assert result.feature_units == (None, None)
    assert np.isnan(result.means[:, 1]).all()
    assert np.isnan(result.resultant_lengths).all()
    np.testing.assert_array_equal(result.mean_status[:, 1], "not_numeric")
    np.testing.assert_array_equal(result.mean_status[:, 0], "defined")
    np.testing.assert_array_equal(result.category_features, [1, 1])
    np.testing.assert_array_equal(result.category_codes, [0, 1])
    np.testing.assert_array_equal(result.category_labels, ["center", "arm"])
    np.testing.assert_array_equal(
        result.category_counts.sum(axis=1), result.counts[:, 1]
    )
    # Queries of event 4.875 are 4.375 .. 5.125; the last holds zone 1 from 5.0.
    np.testing.assert_array_equal(result.category_counts[:, 0], [2, 2, 2, 1])
    np.testing.assert_array_equal(result.category_counts[:, 1], [1, 1, 1, 2])


def test_no_support_cell_status():
    # One event, partial coverage, a query after the recording ends.
    result = align_continuous_to_events(
        EIGHTHS, EIGHTHS, [9.875], (-0.5, 0.5), dt=0.25, window_coverage="partial"
    )
    np.testing.assert_array_equal(result.counts[:, 0], [1, 1, 1, 0])
    np.testing.assert_array_equal(
        result.mean_status[:, 0], ["defined", "defined", "defined", "no_support"]
    )
    assert np.isnan(result.means[3, 0])


# --------------------------------------------------------------- ownership


def test_result_owns_read_only_arrays():
    signals = EIGHTHS.copy()
    events = np.array([3.0, 1.0])
    ids = np.array(["a", "b"])
    result = align_continuous_to_events(
        EIGHTHS, signals, events, (-0.5, 0.5), dt=0.25, event_ids=ids
    )
    values = np.array(result.values)
    signals[:] = -1.0
    events[:] = 0.0
    ids[:] = "z"
    np.testing.assert_array_equal(result.values, values)
    np.testing.assert_array_equal(result.event_ids, ["a", "b"])
    with pytest.raises(ValueError, match="read-only"):
        result.values[0, 0, 0] = 1.0
    for restored in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
        assert not restored.values.flags.writeable
        assert not restored.event_ids.flags.writeable
        np.testing.assert_array_equal(restored.values, values)


def test_derived_properties_are_fresh():
    result = align_continuous_to_events(EIGHTHS, EIGHTHS, [3.0], (-0.5, 0.5), dt=0.25)
    result.means[:] = 0.0
    result.event_indices[:] = 5
    assert result.means[0, 0] == 2.5
    np.testing.assert_array_equal(result.event_indices, [0])


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"window": (1.0, 1.0)}, "start"),
        ({"window": (0.0, 1e-12)}, "no query"),
        ({"dt": 0.0}, "dt"),
        ({"dt": np.nan}, "dt"),
        ({"dt": "0.1"}, "dt"),
        ({"window_coverage": "most"}, "window_coverage"),
        ({"resultant_threshold": 1.5}, "resultant_threshold"),
        ({"resultant_threshold": np.nan}, "resultant_threshold"),
        ({"retain_aligned": "yes"}, "retain_aligned"),
        ({"event_times": [3.0, np.nan]}, "event_times"),
        ({"event_times": []}, "event_times"),
        ({"event_times": [[3.0]]}, "event_times"),
        ({"event_ids": ["a"]}, "event_ids"),
        ({"event_ids": "ab"}, "event_ids"),
        ({"event_ids": 5}, "event_ids"),
        ({"dt": np.array([0.1, 0.2])}, "dt"),
        ({"dt": True}, "dt"),
        ({"resultant_threshold": np.array([0.1])}, "resultant_threshold"),
        ({"resultant_threshold": False}, "resultant_threshold"),
        ({"event_times": [99.0, -50.0]}, "All 2 events were excluded"),
    ],
)
def test_actionable_errors(override, match):
    arguments = {
        "times": EIGHTHS,
        "signals": EIGHTHS,
        "event_times": [3.0, 5.0],
        "window": (-0.5, 0.5),
        "dt": 0.25,
    }
    arguments.update(override)
    positional = [arguments.pop(name) for name in ("times", "signals", "event_times")]
    window = arguments.pop("window")
    with pytest.raises(ValueError, match=match) as info:
        align_continuous_to_events(*positional, window, **arguments)
    assert "Fix:" in str(info.value)


def test_all_excluded_error_counts_each_reason():
    with pytest.raises(ValueError) as info:
        align_continuous_to_events(
            TWO_RUNS, TWO_RUNS, [0.75, 99.0, 1.25], (-0.5, 0.5), dt=0.125
        )
    message = str(info.value)
    assert "All 3 events were excluded" in message
    assert "1 outside_recording" in message
    assert "2 incomplete_support" in message
    assert "window_coverage='partial'" in message


# ------------------------------------------------------ result invariants


def _result_inputs():
    return {
        "feature_ids": np.array(["speed"]),
        "value_kinds": ("linear",),
        "interpolation": ("linear",),
        "feature_units": ("cm/s",),
        "event_status": np.array(["retained", "outside_recording"]),
        "overlapping": np.zeros(2, dtype=bool),
        "counts": np.ones((2, 1), dtype=np.int64),
        "sums": np.ones((2, 1)),
        "sin_sums": np.full((2, 1), np.nan),
        "cos_sums": np.full((2, 1), np.nan),
        "category_features": np.zeros(0, dtype=np.intp),
        "category_codes": np.zeros(0, dtype=np.int64),
        "category_labels": np.zeros(0, dtype=np.str_),
        "category_counts": np.zeros((2, 0), dtype=np.int64),
        "window": (0.0, 1.0),
        "dt": 0.5,
        "window_coverage": "complete",
        "resultant_threshold": 1e-12,
        "max_gap": 0.5,
        "epochs": None,
        "values": np.ones((1, 2, 1)),
        "support": np.ones((1, 2, 1), dtype=bool),
        "event_ids": None,
    }


def test_result_accepts_consistent_inputs():
    result = ContinuousEventResult(**_result_inputs())
    assert result.n_events == 1


def test_result_list_fields_become_arrays_and_tuples():
    """A list event_status passed the checks, then counted 0 retained events."""
    inputs = _result_inputs()
    inputs["event_status"] = ["retained", "outside_recording"]
    inputs["value_kinds"] = ["linear"]
    inputs["window"] = [0.0, 1.0]

    result = ContinuousEventResult(**inputs)

    assert result.n_events == 1
    assert result.event_indices.tolist() == [0]
    assert result.value_kinds == ("linear",)
    assert result.window == (0.0, 1.0)
    assert not result.event_status.flags.writeable


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"values": np.ones((2, 2, 1))}, "values and support need shape"),
        ({"support": None}, "both given or both None"),
        ({"counts": np.ones((3, 1), dtype=np.int64)}, ": counts needs shape"),
        ({"sums": np.ones((2, 2))}, ": sums needs shape"),
        ({"event_status": np.array(["retained", "lost"])}, "event_status must be in"),
        ({"overlapping": np.zeros(3, dtype=bool)}, "overlapping needs shape"),
        ({"event_ids": np.array([1, 2, 3])}, "event_ids needs shape"),
        ({"value_kinds": ("linear", "linear")}, "value_kinds needs 1 entries"),
        ({"interpolation": ()}, "interpolation needs 1 entries"),
        ({"feature_units": ("cm/s", "s")}, "feature_units needs 1 entries"),
        ({"window_coverage": "most"}, "window_coverage must be in"),
        ({"resultant_threshold": 2.0}, "resultant_threshold must be finite"),
        (
            {"category_counts": np.zeros((2, 1), dtype=np.int64)},
            "category_counts needs shape",
        ),
        (
            {"category_features": np.array([0], dtype=np.intp)},
            "one entry per category row",
        ),
        ({"dt": 0.0}, "dt must be a positive"),
        ({"window": (1.0, 0.0)}, "start < stop"),
        ({"max_gap": -3.0}, "max_gap must be None"),
        ({"interpolation": ("spline",)}, "not allowed for linear"),
        ({"epochs": np.array([1.0, 2.0, 3.0])}, "epochs must be None"),
        ({"epochs": np.array([[2.0, 3.0], [0.0, 1.0]])}, "epochs must be None"),
        ({"epochs": np.array([[0.0, 2.0], [1.0, 3.0]])}, "epochs must be None"),
        ({"epochs": np.array([[0.0, 1.0], [1.0, 3.0]])}, "epochs must be None"),
        (
            {"counts": np.full((2, 1), 2, dtype=np.int64)},
            r"counts must lie in \[0, 1\]",
        ),
        ({"counts": -np.ones((2, 1), dtype=np.int64)}, "counts must lie in"),
        ({"values": np.full((1, 2, 1), np.nan)}, "finite wherever support"),
        ({"feature_ids": np.array(["s", "s"])}, "feature_ids must be unique"),
        ({"resultant_threshold": "0.1"}, "resultant_threshold must be finite"),
        (
            {"value_kinds": ("circular",), "interpolation": ("circular",)},
            "circular features need unit 'rad'",
        ),
    ],
)
def test_result_rejects_inconsistent_inputs(override, match):
    inputs = _result_inputs()
    inputs.update(override)
    with pytest.raises(ValueError, match=match):
        ContinuousEventResult(**inputs)


def _two_code_alignment(codes, **overrides):
    codes = list(codes)
    times = np.arange(8.0) * 0.25
    signal = np.where(times < 1.0, codes[0], codes[1])
    return align_continuous_to_events(
        times,
        np.asarray(signal),
        [0.5, 1.5],
        (-0.25, 0.25),
        dt=0.25,
        value_kinds={0: "categorical"},
        category_labels={0: {int(c): f"c{i}" for i, c in enumerate(codes)}},
        **overrides,
    )


def test_category_counts_sum_to_supported_events_per_cell():
    result = _two_code_alignment([3, 7])
    np.testing.assert_array_equal(
        result.category_counts.sum(axis=1), result.counts[:, 0]
    )
    np.testing.assert_array_equal(result.counts[:, 0], 2)
    np.testing.assert_array_equal(result.category_counts, [[1, 1]] * 2)


def test_integer_codes_at_the_float64_limit_are_accepted():
    for edge in (2**53, -(2**53)):
        result = _two_code_alignment([edge, 5])
        np.testing.assert_array_equal(
            result.category_counts.sum(axis=1), result.counts[:, 0]
        )


@pytest.mark.parametrize(
    "codes",
    [
        [2**53, 2**53 + 1],
        [-(2**53) - 1, 0],
        [0, 2**62],
    ],
)
def test_integer_codes_beyond_float64_are_refused(codes):
    with pytest.raises(ValueError, match=r"2\*\*53") as err:
        _two_code_alignment(codes)
    assert "align_continuous_to_events" in str(err.value)
    assert "np.unique" in str(err.value)


def test_unsigned_codes_beyond_float64_are_refused():
    times = np.arange(8.0) * 0.25
    signal = np.full(8, 2**63, dtype=np.uint64)
    with pytest.raises(ValueError, match=r"2\*\*53"):
        align_continuous_to_events(
            times,
            signal,
            [0.5],
            (-0.25, 0.25),
            dt=0.25,
            value_kinds={0: "categorical"},
            category_labels={0: {0: "a"}},
        )


def test_category_label_keys_beyond_float64_are_refused():
    times = np.arange(8.0) * 0.25
    with pytest.raises(ValueError, match=r"2\*\*53"):
        align_continuous_to_events(
            times,
            np.zeros(8),
            [0.5],
            (-0.25, 0.25),
            dt=0.25,
            value_kinds={0: "categorical"},
            category_labels={0: {0: "a", 2**53 + 1: "b"}},
        )
