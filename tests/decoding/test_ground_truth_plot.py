"""Posterior overlays and error use the same observed ground-truth support."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.path import Path

from neurospatial import Environment
from neurospatial.decoding import BayesianDecoder, DecodingResult


@pytest.fixture
def env():
    return Environment.from_samples(
        np.arange(0.0, 101.0, 5.0)[:, None], bin_size=5.0, units="cm"
    )


def delta_result(env, times, bins):
    posterior = np.zeros((len(times), env.n_bins))
    posterior[np.arange(len(times)), bins] = 1
    return DecodingResult(posterior, env, np.asarray(times))


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("extent", [None, [2.0, 22.0, -7.0, 77.0]])
@pytest.mark.parametrize("origin", ["lower", "upper"])
def test_truth_and_map_share_image_centers_and_breaks(env, gapped, extent, origin):
    bins = np.array([4, 16, 8, 12])
    times = (
        np.array([10.05, 10.15, 20.05, 20.15]) if gapped else 10.05 + np.arange(4) * 0.1
    )
    positions = env.bin_centers[bins].copy()
    original = positions.copy()
    result = delta_result(env, times, bins)
    kwargs = {"extent": extent} if extent is not None else {}
    ax = result.plot(
        true_times=times,
        true_positions=positions,
        show_map=True,
        origin=origin,
        **kwargs,
    )
    try:
        left, right, bottom, top = ax.images[0].get_extent()
        expected_x = left + (np.arange(4) + 0.5) * (right - left) / 4
        step = (top - bottom) / env.n_bins
        expected_y = (
            (top - (bins + 0.5) * step)
            if origin == "upper"
            else bottom + (bins + 0.5) * step
        )
        for label in ("MAP", "Actual position"):
            line = next(line for line in ax.lines if line.get_label() == label)
            valid = np.isfinite(line.get_ydata())
            np.testing.assert_allclose(line.get_xdata()[valid], expected_x)
            np.testing.assert_allclose(line.get_ydata()[valid], expected_y)
            starts = sum(
                code == Path.MOVETO for _, code in line.get_path().iter_segments()
            )
            assert starts == (2 if gapped else 1)
        if gapped:
            marker = ax.lines[-1]
            np.testing.assert_allclose(
                marker.get_xdata(), [left + (right - left) / 2] * 2
            )
        np.testing.assert_array_equal(positions, original)
        np.testing.assert_array_equal(result.times, times)
        np.testing.assert_array_equal(result.map_estimate, bins)
    finally:
        plt.close(ax.figure)


def test_missing_truth_and_source_gaps_have_nan_errors_and_no_overlay(env):
    times = np.array([-0.1, 0, 0.05, 0.15, 0.2, 0.6, 1.05, 1.2, 1.3])
    result = delta_result(env, times, np.zeros(len(times), dtype=int))
    true_times = np.array([0, 0.1, 0.2, 1, 1.1, 1.2])
    true_positions = np.array([[0.0], [10.0], [np.nan], [30.0], [40.0], [50.0]])
    errors = result.error_against(true_times, true_positions)
    np.testing.assert_allclose(
        errors, [np.nan, 0, 5, np.nan, np.nan, np.nan, 35, 50, np.nan]
    )
    ax = result.plot(true_times=true_times, true_positions=true_positions)
    try:
        truth = ax.lines[0]
        # Recording breaks insert additional NaNs, but every observed truth
        # sample still has the same bin membership as the error alignment.
        np.testing.assert_array_equal(
            truth.get_ydata()[np.isfinite(truth.get_ydata())], [0, 1, 7, 10]
        )
    finally:
        plt.close(ax.figure)


def test_gap_limit_can_be_disabled_without_extrapolating_or_filling_nan(env):
    times = np.array([-0.1, 0, 0.5, 1, 1.1])
    result = delta_result(env, times, np.zeros(len(times), dtype=int))
    truth = np.array([[0.0], [100.0]])
    np.testing.assert_allclose(
        result.error_against([0, 1], truth), [np.nan, 0, np.nan, 100, np.nan]
    )
    np.testing.assert_allclose(
        result.error_against([0, 1], truth, max_gap=None), [np.nan, 0, 50, 100, np.nan]
    )
    # A missing true position is bridged like a deleted row when the gap
    # passes max_gap, and not otherwise.
    source = np.array([[0.0], [np.nan], [100.0]])
    np.testing.assert_allclose(
        result.error_against([0, 0.5, 1], source, max_gap=None),
        result.error_against([0, 1], truth, max_gap=None),
    )
    assert np.isnan(result.error_against([0, 0.5, 1], source, max_gap=0.75)[2])


def test_truth_line_never_joins_distinct_runs_even_with_only_two_decode_rows(env):
    result = delta_result(env, [0, 10], [0, 20])
    ax = result.plot(
        true_times=np.array([0.0, 10.0]), true_positions=np.array([[0.0], [100.0]])
    )
    try:
        line = ax.lines[0]
        assert np.isnan(line.get_ydata()[1])
        np.testing.assert_allclose(result.error_against([0, 10], [[0], [100]]), [0, 0])
    finally:
        plt.close(ax.figure)


def test_undecodable_rows_do_not_invent_bin_zero_in_either_line(env):
    result = delta_result(env, [0, 0.1, 0.2], [4, 0, 8])
    posterior = result.posterior.copy()
    posterior[1] = np.nan
    result = replace(result, posterior=posterior)
    truth = np.array([[20.0], [30.0], [40.0]])
    ax = result.plot(true_times=result.times, true_positions=truth, show_map=True)
    try:
        for line in ax.lines:
            assert np.isnan(line.get_ydata()[1])
        np.testing.assert_allclose(
            result.error_against(result.times, truth), [0, np.nan, 0]
        )
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("which", ["true_times", "true_positions"])
def test_overlay_requires_a_pair(env, which):
    result = delta_result(env, [0, 0.1], [0, 1])
    value = np.array([0, 0.1]) if which == "true_times" else np.array([[0.0], [5.0]])
    with pytest.raises(ValueError, match="together"):
        result.plot(**{which: value})


def test_overlay_requires_decode_times(env):
    result = DecodingResult(np.ones((2, env.n_bins)) / env.n_bins, env)
    with pytest.raises(ValueError, match="decode times"):
        result.plot(true_times=[0, 0.1], true_positions=[[0], [5]])


@pytest.mark.parametrize("max_gap", [0, -1, np.nan, np.inf])
def test_ground_truth_gap_limit_is_validated(env, max_gap):
    result = delta_result(env, [0, 0.1], [0, 1])
    with pytest.raises(ValueError, match="max_gap"):
        result.error_against([0, 0.1], [[0], [5]], max_gap=max_gap)


def test_score_forwards_its_configured_gap_policy(env):
    decoder = replace(
        BayesianDecoder.from_models(env, np.ones((2, env.n_bins)), dt=0.25),
        max_gap=None,
    )
    score = decoder.score(
        [np.empty(0), np.empty(0)],
        [0, 1],
        np.array([[0.0], [100.0]]),
        metric="mean_error",
        distance="euclidean",  # exact straight-line mean of the interpolated track
    )
    assert score == pytest.approx(50.0)


def test_millisecond_truth_clock_warns_instead_of_silent_nan_error(env):
    times = np.arange(0.0, 2.0, 0.1)
    result = delta_result(env, times, np.zeros(len(times), dtype=int))
    true_times = np.arange(0.0, 2.0, 1 / 30) * 1000.0
    true_positions = np.linspace(0.0, 100.0, true_times.size)[:, None]
    with pytest.warns(UserWarning, match="probably not in seconds"):
        result.error_against(true_times, true_positions)
