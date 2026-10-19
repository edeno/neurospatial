"""Tables, summary and plot exported by ``ContinuousEventResult``."""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest

from neurospatial.events import align_continuous_to_events

TIMES = np.arange(0, 10.0001, 0.125)


def _mixed_result(**kwargs):
    """3 events, 4 times, 2 linear features."""
    signals = np.column_stack([TIMES, 2.0 * TIMES])
    signals[np.isclose(TIMES, 3.0), 1] = np.nan  # one unsupported cell
    return align_continuous_to_events(
        TIMES,
        signals,
        [2.0, 3.0, 5.0],
        (-0.5, 0.5),
        dt=0.25,
        feature_ids=["a", "b"],
        feature_units={"a": "cm", "b": "cm/s"},
        window_coverage="partial",
        **kwargs,
    )


def _mixed_kinds_result():
    angle = np.angle(np.exp(1j * TIMES))
    codes = (np.floor(TIMES) % 3).astype(float)
    signals = np.column_stack([TIMES, angle, codes])
    return align_continuous_to_events(
        TIMES,
        signals,
        [2.0, 3.5, 5.0, 6.5],
        (-0.5, 0.5),
        dt=0.25,
        feature_ids=["lin", "circ", "cat"],
        value_kinds={"circ": "circular", "cat": "categorical"},
        category_labels={"cat": {0: "zero", 1: "one", 2: "two"}},
        interpolation={"cat": "previous"},
    )


def test_row_counts_keys_and_no_unit_id():
    result = _mixed_result()
    aligned = result.aligned_table()
    keys = ["event_index", "relative_time", "feature_id"]
    assert len(aligned) == 3 * 4 * 2
    assert not aligned.duplicated(keys).any()
    # Nesting order: event, then time, then feature.
    expected = list(
        itertools.product([0, 1, 2], result.relative_times.tolist(), ["a", "b"])
    )
    assert list(aligned[keys].itertuples(index=False, name=None)) == expected
    course = result.time_course_table()
    assert len(course) == 8
    assert list(
        course[["relative_time", "feature_id"]].itertuples(index=False, name=None)
    ) == list(itertools.product(result.relative_times.tolist(), ["a", "b"]))
    for table in (
        aligned,
        result.time_course_table(),
        result.events_table(),
        _mixed_kinds_result().category_counts_table(),
    ):
        assert "unit_id" not in table.columns


def test_aligned_table_round_trips_values_and_support():
    result = _mixed_result()
    aligned = result.aligned_table()
    assert list(aligned.columns) == [
        "event_index",
        "relative_time",
        "feature_id",
        "value",
        "supported",
    ]
    value = aligned.pivot(
        index=["event_index", "relative_time"], columns="feature_id", values="value"
    )
    np.testing.assert_array_equal(
        value.to_numpy().reshape(result.values.shape), result.values
    )
    assert np.isnan(result.values).any()
    # Feature "a" is the ramp t, feature "b" is 2t (NaN at the sample t=3).
    ramp = np.array([2.0, 3.0, 5.0])[:, None] + result.relative_times[None, :]
    np.testing.assert_allclose(result.values[:, :, 0], ramp)
    np.testing.assert_allclose(value["a"].to_numpy().reshape(3, 4), ramp)
    supported = aligned.pivot(
        index=["event_index", "relative_time"],
        columns="feature_id",
        values="supported",
    )
    np.testing.assert_array_equal(
        supported.to_numpy().reshape(result.support.shape), result.support
    )


def test_events_table_status_and_optional_event_id():
    result = _mixed_result()
    table = result.events_table()
    assert list(table.columns) == ["event_index", "status", "overlapping"]
    np.testing.assert_array_equal(table["status"], result.event_status)
    np.testing.assert_array_equal(table["overlapping"], result.overlapping)

    labeled = align_continuous_to_events(
        TIMES,
        TIMES,
        [2.0, 99.0, 5.0],
        (-0.5, 0.5),
        dt=0.25,
        event_ids=["x", "y", "z"],
    )
    table = labeled.events_table()
    assert list(table.columns) == ["event_index", "event_id", "status", "overlapping"]
    np.testing.assert_array_equal(table["event_id"], ["x", "y", "z"])
    np.testing.assert_array_equal(
        table["status"], ["retained", "outside_recording", "retained"]
    )
    # Aligned rows carry the caller's label of the retained events.
    aligned = labeled.aligned_table()
    assert list(aligned.columns)[:2] == ["event_index", "event_id"]
    np.testing.assert_array_equal(aligned["event_index"].unique(), [0, 2])
    np.testing.assert_array_equal(aligned["event_id"].unique(), ["x", "z"])


def test_time_course_table_matches_result_properties():
    result = _mixed_kinds_result()
    table = result.time_course_table()
    assert list(table.columns) == [
        "relative_time",
        "feature_id",
        "value_kind",
        "count",
        "mean",
        "resultant_length",
        "status",
    ]
    assert len(table) == 4 * 3
    np.testing.assert_array_equal(
        table["mean"].to_numpy(), result.means.reshape(-1), strict=True
    )
    np.testing.assert_array_equal(
        table["status"].to_numpy(), result.mean_status.reshape(-1)
    )
    np.testing.assert_array_equal(table["count"].to_numpy(), result.counts.reshape(-1))
    cat = table[table["feature_id"] == "cat"]
    assert cat["mean"].isna().all()
    assert (cat["status"] == "not_numeric").all()
    assert (cat["value_kind"] == "categorical").all()
    assert table.loc[table["feature_id"] == "lin", "resultant_length"].isna().all()


def test_category_table_proportions():
    result = _mixed_kinds_result()
    table = result.category_counts_table()
    assert list(table.columns) == [
        "relative_time",
        "feature_id",
        "category_code",
        "category_label",
        "count",
        "proportion",
    ]
    assert len(table) == 4 * 3
    assert set(table["feature_id"]) == {"cat"}
    assert 0 in set(table["category_code"])
    assert table.loc[table["category_code"] == 0, "category_label"].eq("zero").all()
    sums = table.groupby(["relative_time", "feature_id"])["proportion"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0)


def _missing_at_zero_result():
    """Two events whose categorical signal is missing exactly at relative 0."""
    codes = np.zeros_like(TIMES)
    codes[np.isclose(TIMES, 3.0) | np.isclose(TIMES, 5.0)] = np.nan
    return align_continuous_to_events(
        TIMES,
        codes,
        [3.0, 5.0],
        (-0.5, 0.5),
        dt=0.25,
        value_kinds={0: "categorical"},
        category_labels={0: {0: "a", 1: "b"}},
        interpolation={0: "previous"},
        window_coverage="partial",
    )


def test_category_proportion_nan_without_support(agg):
    result = _missing_at_zero_result()
    zero_cell = np.isclose(result.relative_times, 0.0)
    np.testing.assert_array_equal(result.counts[:, 0] == 0, zero_cell)
    table = result.category_counts_table()
    assert len(table) == 4 * 2
    no_support = np.isclose(table["relative_time"].to_numpy(), 0.0)
    np.testing.assert_array_equal(np.isnan(table["proportion"].to_numpy()), no_support)
    np.testing.assert_array_equal(table["count"].to_numpy()[no_support], 0)
    np.testing.assert_array_equal(table["proportion"].to_numpy()[~no_support][::2], 1.0)
    # The plot shows the same gap.
    ax = result.plot()
    y = ax.lines[0].get_ydata()
    np.testing.assert_array_equal(np.isnan(y), zero_cell)


def test_category_table_empty_without_categorical_features():
    table = _mixed_result().category_counts_table()
    assert len(table) == 0
    assert list(table.columns) == [
        "relative_time",
        "feature_id",
        "category_code",
        "category_label",
        "count",
        "proportion",
    ]


def test_attrs_hold_alignment_settings():
    result = _mixed_result(max_gap=0.5, resultant_threshold=0.1)
    for table in (
        result.events_table(),
        result.aligned_table(),
        result.time_course_table(),
        result.category_counts_table(),
    ):
        assert table.attrs["window"] == (-0.5, 0.5)
        assert table.attrs["dt"] == 0.25
        assert table.attrs["window_coverage"] == "partial"
        assert table.attrs["max_gap"] == 0.5
        assert table.attrs["resultant_threshold"] == 0.1
        assert table.attrs["feature_units"] == {"a": "cm", "b": "cm/s"}


def test_exports_are_editable_copies():
    result = _mixed_result()
    values_before = result.values.copy()
    counts_before = result.counts.copy()
    status_before = result.event_status.copy()
    aligned = result.aligned_table()
    aligned.loc[0, "value"] = -999.0
    aligned.loc[0, "supported"] = False
    course = result.time_course_table()
    course.loc[0, "count"] = 77
    course.loc[0, "mean"] = -1.0
    events = result.events_table()
    events.loc[0, "status"] = "edited"
    events.loc[0, "overlapping"] = True
    np.testing.assert_array_equal(result.values, values_before)
    np.testing.assert_array_equal(result.counts, counts_before)
    np.testing.assert_array_equal(result.event_status, status_before)
    cats = _mixed_kinds_result()
    before = cats.category_counts.copy()
    table = cats.category_counts_table()
    table.loc[0, "count"] = 55
    np.testing.assert_array_equal(cats.category_counts, before)
    # Table columns own writable memory.
    assert aligned["value"].to_numpy().flags.writeable


def test_summary_only_result_refuses_aligned_table_only():
    result = _mixed_result(retain_aligned=False)
    assert result.values is None
    with pytest.raises(ValueError, match="Fix:") as excinfo:
        result.aligned_table()
    assert "retain_aligned=True" in str(excinfo.value)
    assert len(result.events_table()) == 3
    assert len(result.time_course_table()) == 8
    assert len(result.category_counts_table()) == 0
    assert result.summary()["n_events"] == 3


def test_exports_work_without_xarray(monkeypatch):
    monkeypatch.setitem(sys.modules, "xarray", None)
    for result in (_mixed_result(), _mixed_kinds_result()):
        result.events_table()
        result.aligned_table()
        result.time_course_table()
        result.category_counts_table()
        result.summary()
        repr(result)


def test_to_dataframe_is_not_implemented():
    with pytest.raises(NotImplementedError):
        _mixed_result().to_dataframe()


def test_summary_scalars_and_repr():
    signals = TIMES.copy()
    result = align_continuous_to_events(
        TIMES,
        signals,
        [2.0, 2.2, 99.0, -5.0],
        (-0.5, 0.5),
        dt=0.25,
        max_gap=0.5,
    )
    summary = result.summary()
    assert summary["n_events"] == 2
    assert summary["n_events_dropped"] == 2
    assert summary["n_features"] == 1
    assert summary["n_relative_times"] == 4
    assert summary["window_start"] == -0.5
    assert summary["window_stop"] == 0.5
    assert summary["dt"] == 0.25
    assert summary["window_coverage"] == "complete"
    assert summary["n_overlapping_events"] == 2
    assert summary["min_count"] == 2
    assert summary["n_dropped_outside_recording"] == 2
    assert "n_dropped_no_support" not in summary
    for key, value in summary.items():
        assert type(value) in (int, float, str), key
    assert "n_events=" in repr(result)


# ------------------------------------------------------------------- plot


@pytest.fixture
def agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_returns_axes_for_every_kind(agg):
    from matplotlib.axes import Axes

    result = _mixed_kinds_result()
    ax = result.plot(feature_id="lin")
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Time from event (s)"
    assert len(ax.lines) >= 2  # mean line + dashed zero line
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), result.means[:, 0])
    ax = result.plot(feature_id="circ")
    assert isinstance(ax, Axes)
    assert ax.get_ylabel() == "radians"
    assert ax.get_ylim() == pytest.approx((-np.pi, np.pi))
    assert ax.lines[0].get_linestyle() == "None"
    assert ax.lines[0].get_marker() == "o"
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), result.means[:, 1])
    ax = result.plot(feature_id="cat")
    assert isinstance(ax, Axes)
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"zero", "one", "two"}


def test_plot_uses_given_axes_and_unit_label(agg):
    _, ax = agg.subplots()
    result = _mixed_result()
    assert result.plot(ax, feature_id="b") is ax
    assert ax.get_ylabel() == "cm/s"


def test_plot_needs_feature_id_for_several_features(agg):
    result = _mixed_result()
    with pytest.raises(ValueError, match="feature_id") as excinfo:
        result.plot()
    assert "'a'" in str(excinfo.value) and "'b'" in str(excinfo.value)
    with pytest.raises(ValueError, match="not a feature"):
        result.plot(feature_id="zzz")


def test_plot_single_feature_needs_no_feature_id(agg):
    result = align_continuous_to_events(TIMES, TIMES, [2.0, 3.0], (-0.5, 0.5), dt=0.25)
    assert result.plot() is not None
