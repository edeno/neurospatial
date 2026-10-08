"""Null ranks and stable unit-label streams are independent of batch order."""

import warnings

import numpy as np

from neurospatial.encoding._significance import (
    _stream_key,
    run_shuffle_test,
    shuffle_pvalues,
    to_shuffle_results,
)


def test_unit_streams_match_alone_and_reordered():
    trains = [np.array([1.0, 20.0, 70.0]), np.array([2.0, 30.0, 80.0])]
    labels = np.array([10, 20])

    def statistic(shifted):
        return np.array([train.mean() for train in shifted])

    kwargs = {"n_shuffles": 20, "min_shift": 20.0, "rng": 7}
    obs, null = run_shuffle_test(
        statistic, trains, np.array([[0.0, 100.0]]), labels, **kwargs
    )
    single_obs, single_null = run_shuffle_test(
        statistic, trains[1:], np.array([[0.0, 100.0]]), labels[1:], **kwargs
    )
    np.testing.assert_array_equal(single_null[:, 0], null[:, 1])
    reverse_obs, reverse_null = run_shuffle_test(
        statistic, trains[::-1], np.array([[0.0, 100.0]]), labels[::-1], **kwargs
    )
    np.testing.assert_array_equal(reverse_null, null[:, ::-1])
    np.testing.assert_array_equal(single_obs[0], obs[1])
    np.testing.assert_array_equal(reverse_obs, obs[::-1])
    assert _stream_key(3) == _stream_key(np.int64(3))


def test_finite_upper_tail_ranks_and_undefined_observation():
    observed = np.array([[3.0, np.nan, 0.0]])
    null = np.array(
        [
            [[1.0, 1.0, np.nan]],
            [[3.0, 2.0, np.nan]],
            [[4.0, 3.0, np.nan]],
            [[np.nan, 4.0, np.nan]],
        ]
    )
    np.testing.assert_array_equal(
        shuffle_pvalues(observed, null), [[0.75, np.nan, 1.0]]
    )


def test_results_preserve_labels_and_finite_null_zscore():
    observed = np.array([[3.0], [1.0]])
    null = np.array([[[1.0], [1.0]], [[2.0], [1.0]], [[np.nan], [1.0]]])
    results = to_shuffle_results(
        observed, null, shuffle_pvalues(observed, null), np.array([11, 37])
    )
    assert list(results) == [11, 37]
    assert results[11].z_score == 3
    assert np.isnan(results[37].z_score)
    assert results[11].shuffle_type == "circular_time_shift"
    assert results[11].n_shuffles == 3


def test_observed_warning_once_null_warnings_suppressed():
    def statistic(trains):
        warnings.warn("observed-map warning", UserWarning, stacklevel=1)
        return np.array([len(train) for train in trains])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_shuffle_test(
            statistic,
            [np.array([1.0, 2.0])],
            np.array([[0.0, 100.0]]),
            np.array([11]),
            n_shuffles=4,
            min_shift=20,
            rng=7,
        )
    assert len(caught) == 1
