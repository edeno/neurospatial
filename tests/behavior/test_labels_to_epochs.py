"""Per-sample labels become supported, half-open epochs per label."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from neurospatial._observation import validated_observations
from neurospatial.behavior import LabelEpochsResult, labels_to_epochs

_ABAB = list("AABBAAAABB")


@pytest.fixture
def invalid_sample_result() -> LabelEpochsResult:
    valid = np.ones(10, dtype=bool)
    valid[6] = False
    return labels_to_epochs(np.arange(10.0), _ABAB, valid=valid, max_gap=None)


def test_invalid_sample_splits_same_label(invalid_sample_result):
    r = invalid_sample_result
    assert isinstance(r, LabelEpochsResult)
    assert list(r) == ["A", "B"]
    np.testing.assert_array_equal(r["A"], [[0, 2], [4, 5], [7, 8]])
    np.testing.assert_array_equal(r["B"], [[2, 4], [8, 9]])
    np.testing.assert_array_equal(r.interval_labels, [0, 0, 1, 1, 0, -1, -1, 0, 1])
    assert r.durations == {"A": 4.0, "B": 3.0}
    np.testing.assert_array_equal(
        r.epoch_bounds, [[0, 2], [2, 4], [4, 5], [7, 8], [8, 9]]
    )
    np.testing.assert_array_equal(r.epoch_labels, [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(
        r.epoch_intervals, [[0, 1], [2, 3], [4, 4], [7, 7], [8, 8]]
    )
    np.testing.assert_array_equal(r.interval_supported, [1, 1, 1, 1, 1, 0, 0, 1, 1])
    assert r.unsupported_labels == ()
    assert r.n_samples == 10


def test_declared_break_splits():
    breaks = np.zeros(10, dtype=bool)
    breaks[3] = True
    r = labels_to_epochs(np.arange(10.0), ["A"] * 10, break_before=breaks, max_gap=None)
    np.testing.assert_array_equal(r["A"], [[0, 2], [3, 9]])
    assert r.n_declared_breaks == 1


def test_break_before_first_sample_is_not_counted():
    breaks = np.zeros(4, dtype=bool)
    breaks[0] = True
    r = labels_to_epochs(np.arange(4.0), ["A"] * 4, break_before=breaks, max_gap=None)
    assert r.n_declared_breaks == 0
    np.testing.assert_array_equal(r["A"], [[0, 3]])


def test_time_gap_splits():
    r = labels_to_epochs([0, 1, 2, 3, 10, 11, 12], ["A"] * 7, max_gap=1.5)
    np.testing.assert_array_equal(r["A"], [[0, 3], [10, 12]])
    assert r.max_gap == 1.5


def test_missing_labels_are_excluded():
    labels = ["A", None, "A", "other", "A", np.nan, "A", "A", "A", "A"]
    r = labels_to_epochs(np.arange(10.0), labels, max_gap=None, missing_label="other")
    assert list(r) == ["A"]
    np.testing.assert_array_equal(r["A"], [[0, 1], [2, 3], [4, 5], [6, 9]])
    summary = r.summary()
    assert summary["n_missing_label_intervals"] == 3
    assert summary["n_unsupported_intervals"] == 0
    assert r.missing_label == "other"
    assert "other" not in r.unsupported_labels


def test_final_sample_adds_no_time():
    r = labels_to_epochs([0.0, 1.0], ["A", "B"], max_gap=None)
    assert list(r) == ["A"]
    np.testing.assert_array_equal(r["A"], [[0, 1]])
    assert "B" not in r
    assert "B" not in r.unsupported_labels


def test_single_sample_gives_empty_result():
    r = labels_to_epochs([0.0], ["A"], max_gap=None)
    assert len(r) == 0
    assert r.epoch_bounds.shape == (0, 2)
    assert r.epoch_intervals.shape == (0, 2)
    assert r.interval_labels.shape == (0,)
    assert r.interval_supported.shape == (0,)
    assert r.unsupported_labels == ()
    assert r.summary()["labeled_duration"] == 0.0


def test_label_with_no_supported_interval_is_not_a_key():
    valid = np.array([True, True, False, True])
    r = labels_to_epochs(np.arange(4.0), list("ABAA"), valid=valid, max_gap=None)
    assert "B" in r.unsupported_labels
    assert "B" not in r
    with pytest.raises(KeyError, match="none of its intervals"):
        r["B"]
    assert r.summary()["n_unsupported_labels"] == 1


def test_unknown_label_error_lists_labels(invalid_sample_result):
    with pytest.raises(KeyError, match=r"'A', 'B'"):
        invalid_sample_result["C"]


def test_zero_is_a_label():
    r = labels_to_epochs(np.arange(5.0), np.array([0, 0, 1, 1, 0]), max_gap=None)
    assert list(r) == [0, 1]
    assert all(type(k) is int for k in r)
    np.testing.assert_array_equal(r[0], [[0, 2]])
    np.testing.assert_array_equal(r[1], [[2, 4]])


@pytest.mark.parametrize("label", [-1, ""])
def test_negative_one_and_empty_string_are_labels(label):
    r = labels_to_epochs(np.arange(3.0), [label, label, "x"], max_gap=None)
    assert list(r) == [label]


def test_nan_in_a_list_of_strings_is_missing():
    r = labels_to_epochs(np.arange(4.0), ["A", np.nan, "A", "A"], max_gap=None)
    assert list(r) == ["A"]
    np.testing.assert_array_equal(r["A"], [[0, 1], [2, 3]])


def test_epochs_gate():
    r = labels_to_epochs(np.arange(10.0), ["A"] * 10, epochs=[(2.0, 6.0)], max_gap=None)
    np.testing.assert_array_equal(r["A"], [[2, 6]])
    np.testing.assert_array_equal(r.epochs, [[2.0, 6.0]])


def test_labels_ordered_by_first_retained_interval():
    valid = np.array([False, False, True, True])
    r = labels_to_epochs(np.arange(4.0), list("BAAA"), valid=valid, max_gap=None)
    assert list(r) == ["A"]
    assert r.unsupported_labels == ("B",)


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
    kept = np.concatenate([np.arange(a, b + 1) for a, b in r.epoch_intervals])
    np.testing.assert_array_equal(kept, np.flatnonzero(support.interval_mask))
    np.testing.assert_array_equal(r.interval_supported, support.interval_mask)
    assert r.epoch_bounds.shape[0] > 5
    np.testing.assert_array_equal(r.epoch_bounds[:, 0], times[r.epoch_intervals[:, 0]])
    np.testing.assert_array_equal(
        r.epoch_bounds[:, 1], times[r.epoch_intervals[:, 1] + 1]
    )


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


def test_result_owns_its_arrays():
    times = np.arange(10.0)
    labels = np.array(_ABAB)
    valid = np.ones(10, dtype=bool)
    valid[6] = False
    epochs = np.array([[0.0, 9.0]])
    r = labels_to_epochs(times, labels, valid=valid, epochs=epochs, max_gap=None)
    expected = {name: np.array(getattr(r, name)) for name in _ARRAY_FIELDS}
    times[:] = -1
    labels[:] = "B"
    valid[:] = False
    epochs[:] = 0
    for name, array in expected.items():
        np.testing.assert_array_equal(getattr(r, name), array)
    with pytest.raises(ValueError, match="read-only"):
        r.epoch_bounds[0, 0] = 1.0
    selected = r["A"]
    with pytest.raises(ValueError, match="read-only"):
        selected[0, 0] = 1.0
    restored = pickle.loads(pickle.dumps(r))
    assert list(restored) == ["A", "B"]
    for name in _ARRAY_FIELDS:
        assert not getattr(restored, name).flags.writeable, name
        np.testing.assert_array_equal(getattr(restored, name), getattr(r, name))


_ARRAY_FIELDS = (
    "epoch_bounds",
    "epoch_labels",
    "epoch_intervals",
    "interval_labels",
    "interval_supported",
    "epochs",
)


def test_equality_is_identity(invalid_sample_result):
    r = invalid_sample_result
    assert r == r
    assert r != labels_to_epochs(np.arange(10.0), _ABAB, max_gap=None)


def test_epochs_table_and_summary(invalid_sample_result):
    r = invalid_sample_result
    df = r.epochs_table()
    assert list(df.columns) == [
        "label",
        "start",
        "stop",
        "duration",
        "first_interval",
        "last_interval",
        "n_intervals",
    ]
    assert len(df) == 5
    assert df["duration"].sum() == 7.0
    assert df["label"].tolist() == ["A", "B", "A", "A", "B"]
    assert df["n_intervals"].tolist() == [2, 2, 1, 1, 1]
    assert df.attrs == {
        "max_gap": None,
        "epochs": None,
        "missing_label": None,
        "n_declared_breaks": 0,
        "allocation": "interval_start",
    }
    df.loc[0, "start"] = 99.0
    assert r.epoch_bounds[0, 0] == 0.0
    summary = r.summary()
    assert summary == {
        "n_labels": 2,
        "n_epochs": 5,
        "labeled_duration": 7.0,
        "n_unsupported_intervals": 2,
        "n_missing_label_intervals": 0,
        "n_unsupported_labels": 0,
        "max_gap": None,
    }
    for value in summary.values():
        assert value is None or type(value) in (int, float)


def test_selected_windows_restrict_the_original_samples(invalid_sample_result):
    """Each label's rows are directly usable as half-open ``epochs=`` windows."""
    from neurospatial._intervals import interval_valid_mask

    times = np.arange(10.0)
    windows = invalid_sample_result["A"]
    mask = interval_valid_mask(times, max_gap=None, epochs=windows)
    np.testing.assert_array_equal(mask, invalid_sample_result.interval_labels == 0)
