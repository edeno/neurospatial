"""Posterior heatmaps show recording breaks rather than compressing pauses."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial.decoding import DecodingResult


@pytest.mark.parametrize("kind", ["gapped", "contiguous", "single"])
def test_plot_marks_recording_gaps(continuous_recording, kind):
    env = continuous_recording.env
    if kind == "gapped":
        times = np.r_[
            0.0125 + 0.025 * np.arange(3999), 1100.0125 + 0.025 * np.arange(3999)
        ]
    elif kind == "contiguous":
        times = 0.0125 + 0.025 * np.arange(10)
    else:
        times = np.array([0.0125])
    result = DecodingResult(
        np.full((len(times), env.n_bins), 1 / env.n_bins), env, times
    )
    ax = result.plot(colorbar=False, show_map=True)
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    if kind == "gapped":
        assert ax.get_xlabel().startswith("Time bin")
        assert len(dashed) == 1
        np.testing.assert_array_equal(dashed[0].get_xdata(), [3998.5, 3998.5])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(len(times)))
    else:
        assert ax.get_xlabel() == "Time (s)"
        assert not dashed
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), times)
    plt.close(ax.figure)


@pytest.mark.parametrize("quantity", ["entropy", "map"])
def test_summary_plot_breaks_lines_at_recording_gaps(continuous_recording, quantity):
    """A summary line must not draw a straight segment across a pause."""
    from neurospatial.decoding import decode_position_summary

    env = continuous_recording.env
    times = np.r_[0.0125 + 0.025 * np.arange(5), 1100.0125 + 0.025 * np.arange(5)]
    rng = np.random.default_rng(0)
    summary = decode_position_summary(
        env,
        rng.poisson(1.0, (len(times), 3)),
        rng.uniform(1.0, 10.0, (3, env.n_bins)),
        dt=0.025,
        times=times,
    )
    ax = summary.plot(quantity=quantity)
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        assert np.isnan(x[5]) and np.isnan(y[5])
        np.testing.assert_array_equal(np.delete(x, 5), times)
    plt.close(ax.figure)
