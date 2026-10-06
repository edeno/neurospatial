"""Spatial rates recover known firing rates across pauses between recordings."""

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates


@pytest.mark.parametrize("plural", [False, True])
def test_spatial_recovers_true_rate_across_pause(two_epoch_recording, plural):
    r = two_epoch_recording
    function = compute_spatial_rates if plural else compute_spatial_rate
    result = function(
        r.env,
        [r.spike_times] if plural else r.spike_times,
        r.times,
        r.positions,
        method="binned",
    )
    rates = np.asarray(result.firing_rates if plural else result.firing_rate)
    assert np.nansum(
        rates * result.occupancy
    ) / result.occupancy.sum() == pytest.approx(5.0, rel=0.05)
    assert result.occupancy.sum() == pytest.approx(199.96, abs=1e-6)
    assert result.occupancy.max() < 10


@pytest.mark.slow
@pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "glm"])
def test_spatial_all_methods_recover_rate(two_epoch_recording, method):
    r = two_epoch_recording
    result = compute_spatial_rate(
        r.env, r.spike_times, r.times, r.positions, method=method
    )
    pooled = np.nansum(result.firing_rate * result.occupancy) / result.occupancy.sum()
    assert pooled == pytest.approx(5.0, rel=0.1)
    assert result.occupancy.sum() == pytest.approx(199.96, abs=1e-6)


def test_is_place_cell_forwards_time_windows(continuous_recording, monkeypatch):
    import neurospatial.encoding.spatial as spatial

    r = continuous_recording
    original = spatial.compute_spatial_rate
    seen = []

    def track_rate(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append((kwargs, result))
        return result

    monkeypatch.setattr(spatial, "compute_spatial_rate", track_rate)
    answer = spatial.is_place_cell(
        r.env,
        r.spike_times,
        r.times,
        r.positions,
        epochs=[(0, 100)],
        spike_window=(0, 1200),
        max_gap=1.0,
    )
    kwargs, result = seen[0]
    assert kwargs["epochs"] == [(0, 100)]
    assert kwargs["spike_window"] == (0, 1200)
    assert kwargs["max_gap"] == 1.0
    assert result.occupancy.sum() == pytest.approx(100.0, abs=1e-9)
    assert isinstance(answer, bool)
