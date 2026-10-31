"""Export of decode results with ``to_pynapple``."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurospatial.behavior import in_epochs
from neurospatial.decoding import decode_session, decode_session_summary
from neurospatial.io import from_pynapple, to_pynapple

HAS_PYNAPPLE = importlib.util.find_spec("pynapple") is not None
needs_pynapple = pytest.mark.skipif(
    not HAS_PYNAPPLE, reason="requires the pynapple extra"
)


def _half_open(closed_rows) -> np.ndarray:
    """The half-open rows holding the same floats as pynapple's ``[start, end]``."""
    rows = np.array(closed_rows, dtype=np.float64).reshape(-1, 2)
    rows[:, 1] = np.nextafter(rows[:, 1], np.inf)
    return rows


@pytest.fixture(scope="module")
def gapped_decode(continuous_recording):
    """A decode over two epochs separated by a 20 s gap, and its spikes."""
    r = continuous_recording
    epochs = np.array([[0.0, 50.0], [70.0, 120.0]])
    result = decode_session(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.25,
        epochs=epochs,
    )
    return result, epochs, r.spike_times


# ---------------------------------------------------------------------------
# Decode results
# ---------------------------------------------------------------------------


@needs_pynapple
@pytest.mark.pynapple
def test_decoded_track_time_support_is_the_decoded_epochs(gapped_decode):
    import pynapple as nap

    result, epochs, spikes = gapped_decode
    track = to_pynapple(result)

    # The 50-70 s gap was not decoded, so it is not part of the time support.
    np.testing.assert_array_equal(_half_open(track.time_support.values), epochs)
    assert list(track.columns) == ["x", "y"]
    np.testing.assert_array_equal(track.t, result.times)
    np.testing.assert_array_equal(track.values, result.map_position)
    kept = nap.Ts(t=spikes).restrict(track.time_support).t
    np.testing.assert_array_equal(kept, spikes[in_epochs(spikes, epochs)])


@needs_pynapple
@pytest.mark.pynapple
def test_posterior_export_keeps_mass_per_bin_and_bin_centers(gapped_decode):
    result, epochs, _ = gapped_decode
    posterior = to_pynapple(result, quantity="posterior")

    assert posterior.shape == result.posterior.shape
    np.testing.assert_array_equal(posterior.values, result.posterior)
    np.testing.assert_array_equal(posterior.columns, np.arange(result.env.n_bins))
    np.testing.assert_allclose(np.nansum(posterior.values, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(posterior.bin_center_x, result.env.bin_centers[:, 0])
    np.testing.assert_array_equal(posterior.bin_center_y, result.env.bin_centers[:, 1])
    np.testing.assert_array_equal(_half_open(posterior.time_support.values), epochs)
    # Selecting bins by place keeps the matching posterior columns.
    west = posterior[:, posterior.bin_center_x < 50.0]
    np.testing.assert_array_equal(
        west.values, result.posterior[:, result.env.bin_centers[:, 0] < 50.0]
    )
    times, values, support = from_pynapple(posterior)
    np.testing.assert_array_equal(times, result.times)
    np.testing.assert_array_equal(values, result.posterior)
    np.testing.assert_array_equal(support, epochs)
    # The export is a copy: editing it never changes the result.
    posterior.values[0, 0] = -1.0
    assert result.posterior[0, 0] >= 0.0


@needs_pynapple
@pytest.mark.pynapple
def test_mean_track_and_summary_track(continuous_recording, gapped_decode):
    result, epochs, _ = gapped_decode
    mean = to_pynapple(result, quantity="mean_position")
    np.testing.assert_array_equal(mean.values, result.mean_position)

    r = continuous_recording
    summary = decode_session_summary(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.25,
        epochs=epochs,
    )
    track = to_pynapple(summary)
    np.testing.assert_array_equal(track.values, summary.map_position)
    np.testing.assert_array_equal(_half_open(track.time_support.values), epochs)


def test_posterior_export_refusals(continuous_recording, gapped_decode):
    result, _, _ = gapped_decode
    r = continuous_recording
    summary = decode_session_summary(
        r.env, [r.spike_times], r.times, r.positions, method="binned", dt=0.25
    )
    with pytest.raises(ValueError, match="DecodingSummary, which keeps no posterior"):
        to_pynapple(summary, quantity="posterior")
    with pytest.raises(TypeError, match="columns= with quantity='posterior'"):
        to_pynapple(result, quantity="posterior", columns=["a"])
    with pytest.raises(ValueError, match="quantity must be one of"):
        to_pynapple(result, quantity="map")
    with pytest.raises(TypeError, match="quantity='posterior' with a values array"):
        to_pynapple(result.times, result.map_position, quantity="posterior")
