"""Per-sample labels become supported, half-open epochs, one row per epoch.

Every expected row below was captured from the mapping result that
``labels_to_epochs`` returned before it returned ``Intervals``; selecting a
label (``epochs[epochs.label == x]``) gives exactly that label's old windows.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from neurospatial import Interval, Intervals
from neurospatial._observation import validated_observations
from neurospatial.behavior import Lap, labels_to_epochs

_ABAB = list("AABBAAAABB")


def _rows(epochs: Intervals[Interval]) -> tuple[list[object], list[list[float]]]:
    """The label and ``[start, stop)`` of every row, in time order."""
    return epochs.label.tolist(), np.column_stack([epochs.start, epochs.stop]).tolist()


def _windows(epochs: Intervals[Interval], label: object) -> list[list[float]]:
    """The ``[start, stop)`` rows of one label."""
    return epochs[epochs.label == label].windows.tolist()


@pytest.fixture
def invalid_sample_epochs() -> Intervals[Interval]:
    valid = np.ones(10, dtype=bool)
    valid[6] = False
    # The 2 s gap around the invalid sample exceeds max_gap, so it splits.
    return labels_to_epochs(np.arange(10.0), _ABAB, valid=valid, max_gap=1.5)


def test_invalid_sample_splits_same_label(invalid_sample_epochs):
    r = invalid_sample_epochs
    assert isinstance(r, Intervals)
    assert r.record_type is Interval
    assert r.fields == ("label",)
    assert r.interval_ids.tolist() == [0, 1, 2, 3, 4]
    assert _rows(r) == (
        ["A", "B", "A", "A", "B"],
        [[0, 2], [2, 4], [4, 5], [7, 8], [8, 9]],
    )
    assert _windows(r, "A") == [[0, 2], [4, 5], [7, 8]]
    assert _windows(r, "B") == [[2, 4], [8, 9]]
    durations = {label: r[r.label == label].durations.sum() for label in "AB"}
    assert durations == {"A": 4.0, "B": 3.0}
    assert r.analysis_settings == {
        "labels": ["A", "B"],
        "unsupported_labels": [],
        "max_gap": 1.5,
        "epochs": None,
        "missing_label": None,
        "n_declared_breaks": 0,
    }


def test_declared_break_splits():
    breaks = np.zeros(10, dtype=bool)
    breaks[3] = True
    r = labels_to_epochs(np.arange(10.0), ["A"] * 10, break_before=breaks, max_gap=None)
    assert _rows(r) == (["A", "A"], [[0, 2], [3, 9]])
    assert r.analysis_settings["n_declared_breaks"] == 1


def test_break_before_first_sample_is_not_counted():
    breaks = np.zeros(4, dtype=bool)
    breaks[0] = True
    r = labels_to_epochs(np.arange(4.0), ["A"] * 4, break_before=breaks, max_gap=None)
    assert r.analysis_settings["n_declared_breaks"] == 0
    assert _rows(r) == (["A"], [[0, 3]])


def test_time_gap_splits():
    r = labels_to_epochs([0, 1, 2, 3, 10, 11, 12], ["A"] * 7, max_gap=1.5)
    assert _rows(r) == (["A", "A"], [[0, 3], [10, 12]])
    assert r.analysis_settings["max_gap"] == 1.5


def test_missing_labels_are_excluded():
    labels = ["A", None, "A", "other", "A", np.nan, "A", "A", "A", "A"]
    r = labels_to_epochs(np.arange(10.0), labels, max_gap=None, missing_label="other")
    assert _rows(r) == (["A"] * 4, [[0, 1], [2, 3], [4, 5], [6, 9]])
    settings = r.analysis_settings
    assert settings["missing_label"] == "other"
    assert settings["labels"] == ["A"]
    assert "other" not in settings["unsupported_labels"]


def test_direction_labels_need_no_missing_label():
    """Lap labels feed labels_to_epochs directly: None outside laps is left out."""
    times = np.arange(0, 10, 0.5)
    laps = Intervals(
        [Lap(0.0, 4.0, "clockwise", 1.0), Lap(6.0, 9.5, "counter-clockwise", 1.0)]
    )
    labels = laps[laps.direction != "unknown"].sample_values(times, "direction")
    conditions = labels_to_epochs(times, labels)
    assert _rows(conditions) == (
        ["clockwise", "counter-clockwise"],
        [[0.0, 4.0], [6.0, 9.5]],
    )


def test_final_sample_adds_no_time():
    r = labels_to_epochs([0.0, 1.0], ["A", "B"], max_gap=None)
    assert _rows(r) == (["A"], [[0, 1]])
    assert r.analysis_settings["unsupported_labels"] == []


def test_single_sample_gives_empty_epochs():
    r = labels_to_epochs([0.0], ["A"], max_gap=None)
    assert len(r) == 0
    assert r.fields == ("label",)
    assert r.windows.shape == (0, 2)
    assert r.label.dtype == object
    assert r.analysis_settings["labels"] == []
    assert r.analysis_settings["unsupported_labels"] == []


def test_label_with_no_supported_interval_has_no_row():
    valid = np.array([True, True, False, True])
    r = labels_to_epochs(np.arange(4.0), list("ABAA"), valid=valid, max_gap=1.5)
    assert _rows(r) == (["A"], [[0, 1]])
    assert r.analysis_settings["unsupported_labels"] == ["B"]
    selected = r[r.label == "B"]
    assert len(selected) == 0
    assert selected.fields == ("label",)


def test_unobserved_end_sample_is_bridged_or_a_tracking_loss():
    """Sample 3 is unobserved: bridged by default, a loss at max_gap=0.15.

    At max_gap=0.15 the 0.2 s gap around it is a tracking loss, so both
    intervals of label "b" are unsupported. With the default max_gap the
    same unobserved end sample is bridged and "b" keeps the epoch [0.2, 0.4).
    """
    times = np.arange(6) * 0.1
    labels = ["a", "a", "b", "b", "c", "c"]
    valid = np.array([True, True, True, False, True, True])
    bridged = labels_to_epochs(times, labels, valid=valid)
    assert _rows(bridged) == (
        ["a", "b", "c"],
        [[times[0], times[2]], [times[2], times[4]], [times[4], times[5]]],
    )

    r = labels_to_epochs(times, labels, valid=valid, max_gap=0.15)
    assert _rows(r) == (["a", "c"], [[times[0], times[2]], [times[4], times[5]]])
    assert r.analysis_settings["unsupported_labels"] == ["b"]


def test_trailing_unobserved_samples_leave_their_intervals_out():
    """Intervals after the last observed sample are unsupported."""
    times = np.arange(6) * 0.1
    valid = np.array([True, True, True, True, False, False])
    r = labels_to_epochs(times, ["a", "a", "b", "b", "c", "c"], valid=valid)
    assert _rows(r) == (["a", "b"], [[times[0], times[2]], [times[2], times[3]]])
    assert r.analysis_settings["unsupported_labels"] == ["c"]


def test_valid_flags_error_describes_bridging():
    """The malformed-valid error says unobserved samples are bridged."""
    with pytest.raises(ValueError, match="valid") as info:
        labels_to_epochs(np.arange(4.0), list("aabb"), valid=np.ones(4, dtype=int))
    message = str(info.value)
    assert "both of its end samples" not in message
    assert "as if its row were deleted" in message
    assert "max_gap" in message


def test_labels_are_ordered_by_first_epoch():
    """B B A A B B: epochs B[0, .5), A[.5, 1), B[1, 1.25) -> labels (B, A)."""
    r = labels_to_epochs(np.arange(6) * 0.25, list("BBAABB"))
    assert _rows(r) == (["B", "A", "B"], [[0.0, 0.5], [0.5, 1.0], [1.0, 1.25]])
    assert r.analysis_settings["labels"] == ["B", "A"]


def test_invalid_sample_in_bridged_gap_keeps_the_label():
    valid = np.ones(10, dtype=bool)
    valid[6] = False
    r = labels_to_epochs(np.arange(10.0), _ABAB, valid=valid, max_gap=None)
    kept = np.delete(np.arange(10), 6)
    deleted = labels_to_epochs(
        np.arange(10.0)[kept], [_ABAB[k] for k in kept], max_gap=None
    )
    expected = (["A", "B", "A", "B"], [[0, 2], [2, 4], [4, 8], [8, 9]])
    assert _rows(r) == expected
    assert _rows(deleted) == expected


def test_zero_is_a_label():
    r = labels_to_epochs(np.arange(5.0), np.array([0, 0, 1, 1, 0]), max_gap=None)
    assert _rows(r) == ([0, 1], [[0, 2], [2, 4]])
    assert all(type(label) is int for label in r.label.tolist())
    assert _windows(r, 0) == [[0, 2]]
    assert _windows(r, 1) == [[2, 4]]


def test_negative_one_is_a_label():
    r = labels_to_epochs(np.arange(3.0), [-1, -1, "x"], max_gap=None)
    assert _rows(r) == ([-1], [[0, 2]])


def test_empty_string_label_is_refused_unless_declared_missing():
    """NWB reads "" back as a missing field, so it cannot name a condition."""
    with pytest.raises(ValueError, match='missing_label=""'):
        labels_to_epochs(np.arange(3.0), ["", "", "x"], max_gap=None)
    r = labels_to_epochs(
        np.arange(4.0), ["", "x", "x", ""], max_gap=None, missing_label=""
    )
    assert _rows(r) == (["x"], [[1, 3]])


def test_nan_in_a_list_of_strings_is_missing():
    r = labels_to_epochs(np.arange(4.0), ["A", np.nan, "A", "A"], max_gap=None)
    assert _rows(r) == (["A", "A"], [[0, 1], [2, 3]])


def test_epochs_gate():
    r = labels_to_epochs(np.arange(10.0), ["A"] * 10, epochs=[(2.0, 6.0)], max_gap=None)
    assert _rows(r) == (["A"], [[2, 6]])
    assert r.analysis_settings["epochs"] == [[2.0, 6.0]]
    restored = pickle.loads(pickle.dumps(r))
    assert _rows(restored) == _rows(r)
    assert restored.analysis_settings == r.analysis_settings


def test_labels_ordered_by_first_retained_interval():
    valid = np.array([False, False, True, True])
    r = labels_to_epochs(np.arange(4.0), list("BAAA"), valid=valid, max_gap=None)
    assert _rows(r) == (["A"], [[2, 3]])
    assert r.analysis_settings["labels"] == ["A"]
    assert r.analysis_settings["unsupported_labels"] == ["B"]


def test_epochs_match_the_shared_resolver():
    rng = np.random.default_rng(11)
    n = 200
    times = np.cumsum(rng.uniform(0.05, 0.15, size=n))
    times[rng.choice(n, size=5, replace=False)] += 1.0
    times = np.sort(times)
    valid = rng.random(n) > 0.1
    breaks = rng.random(n) > 0.95
    r = labels_to_epochs(
        times, ["A"] * n, valid=valid, break_before=breaks, max_gap=0.5
    )
    positions = np.where(valid[:, None], 1.0, np.nan) * np.ones((n, 2))
    support = validated_observations(
        times,
        positions,
        call="f",
        max_gap=0.5,
        epochs=None,
        break_before=breaks,
        result="r",
    )[2]
    # Every bound is a sample time; row i covers intervals first[i]..last[i].
    first = np.searchsorted(times, r.start)
    after_last = np.searchsorted(times, r.stop)
    np.testing.assert_array_equal(times[first], r.start)
    np.testing.assert_array_equal(times[after_last], r.stop)
    expected_spans = [
        [0, 18], [20, 32], [36, 37], [39, 53], [55, 55], [57, 86], [88, 96],
        [99, 103], [106, 113], [116, 126], [129, 130], [132, 154], [157, 158],
        [161, 170], [172, 198],
    ]  # fmt: skip
    assert np.column_stack([first, after_last - 1]).tolist() == expected_spans
    kept = np.concatenate(
        [np.arange(a, b) for a, b in zip(first, after_last, strict=True)]
    )
    np.testing.assert_array_equal(kept, np.flatnonzero(support.interval_mask))


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"valid": np.ones(10, dtype=int)}, "valid"),
        ({"valid": np.ones(9, dtype=bool)}, "valid"),
        ({"labels": ["A"] * 9}, "labels"),
        ({"labels": np.full((10, 2), "A")}, "labels"),
        ({"labels": "AAAAAAAAAA"}, "labels"),
        ({"times": np.r_[0.0, 2.0, 1.0, np.arange(3.0, 10.0)]}, "times"),
        ({"times": np.r_[0.0, 0.0, np.arange(2.0, 10.0)]}, "times"),
        ({"times": [], "labels": []}, "times"),
        ({"break_before": np.ones(10)}, "break_before"),
        ({"max_gap": -1.0}, "max_gap"),
    ],
)
def test_invalid_input_names_the_argument(kwargs, name):
    call = {"times": np.arange(10.0), "labels": ["A"] * 10, **kwargs}
    with pytest.raises(ValueError, match=name):
        labels_to_epochs(call.pop("times"), call.pop("labels"), **call)


def test_epochs_own_their_data():
    times = np.arange(10.0)
    labels = np.array(_ABAB)
    valid = np.ones(10, dtype=bool)
    valid[6] = False
    epochs = np.array([[0.0, 9.0]])
    r = labels_to_epochs(times, labels, valid=valid, epochs=epochs, max_gap=None)
    expected = (["A", "B", "A", "B"], [[0, 2], [2, 4], [4, 8], [8, 9]])
    assert _rows(r) == expected
    assert r.analysis_settings["epochs"] == [[0.0, 9.0]]
    times[:] = -1
    labels[:] = "B"
    valid[:] = False
    epochs[:] = 0
    assert _rows(r) == expected
    assert r.analysis_settings["epochs"] == [[0.0, 9.0]]
    with pytest.raises(ValueError, match="read-only"):
        r.start[0] = 1.0
    restored = pickle.loads(pickle.dumps(r))
    assert _rows(restored) == expected
    assert not restored.start.flags.writeable


def test_intervals_table(invalid_sample_epochs):
    df = invalid_sample_epochs.intervals_table()
    assert list(df.columns) == ["start_time", "stop_time", "label"]
    assert df.index.name == "interval_id"
    assert df.index.tolist() == [0, 1, 2, 3, 4]
    assert df["label"].tolist() == ["A", "B", "A", "A", "B"]
    assert (df["stop_time"] - df["start_time"]).sum() == 7.0
    df.loc[0, "start_time"] = 99.0
    assert invalid_sample_epochs.start[0] == 0.0


def test_selected_windows_restrict_the_original_samples(invalid_sample_epochs):
    """Each label's rows are directly usable as half-open ``epochs=`` windows."""
    from neurospatial._intervals import interval_valid_mask

    times = np.arange(10.0)
    epochs = invalid_sample_epochs
    selected = epochs[epochs.label == "A"].windows
    mask = interval_valid_mask(times, max_gap=None, epochs=selected)
    # Intervals [k, k + 1) labelled "A" and supported: 0, 1, 4 and 7.
    np.testing.assert_array_equal(mask, [1, 1, 0, 0, 1, 0, 0, 1, 0])


def test_labels_follow_first_retained_interval():
    valid = np.ones(5, dtype=bool)
    valid[0] = False  # "B" starts at sample 0 but is first retained after "A"
    r = labels_to_epochs(np.arange(5.0), list("BAABB"), valid=valid, max_gap=None)
    assert _rows(r) == (["A", "B"], [[1, 3], [3, 4]])
    assert r.analysis_settings["labels"] == ["A", "B"]


def test_labels_are_plain_python_values():
    r = labels_to_epochs(
        np.arange(5.0), [np.int64(1), np.int64(1), "x", "x", "x"], max_gap=None
    )
    assert _rows(r) == ([1, "x"], [[0, 2], [2, 4]])
    assert type(r.label.iloc[0]) is int
    assert type(r.analysis_settings["labels"][0]) is int


def test_tuple_labels_are_combined_conditions():
    labels = [("arm1", "out")] * 3 + [("arm1", "in")] * 2
    r = labels_to_epochs(np.arange(5.0), labels, max_gap=None)
    assert _rows(r) == ([("arm1", "out"), ("arm1", "in")], [[0, 3], [3, 4]])
    assert _windows(r, ("arm1", "out")) == [[0, 3]]
    assert _windows(r, ("arm1", "in")) == [[3, 4]]
