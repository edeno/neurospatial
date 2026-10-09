"""Spatial rates recover known firing rates across pauses between recordings."""

from dataclasses import replace

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates


@pytest.mark.parametrize("plural", [False, True])
def test_rate_family_recovers_true_rate_across_pause(
    frame_family, two_epoch_recording, plural
):
    f, r = frame_family, two_epoch_recording
    result = (f.plural if plural else f.single)(
        *f.args(r, [r.spike_times] if plural else r.spike_times), **f.defaults
    )
    rates = np.asarray(result.firing_rates if plural else result.firing_rate)
    pooled = np.nansum(rates * result.occupancy) / result.occupancy.sum()
    assert pooled == pytest.approx(5.0, rel=0.05)
    assert result.occupancy.sum() <= 199.96 + 1e-6
    if f.name != "view":
        assert result.occupancy.sum() == pytest.approx(199.96, abs=1e-6)
    assert result.occupancy.max() < 10


def test_predicates_forward_time_windows(
    frame_family, continuous_recording, monkeypatch
):
    f, r = frame_family, continuous_recording
    original = f.single
    seen = []

    def track_rate(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append((kwargs, result))
        return result

    monkeypatch.setattr(f.module, original.__name__, track_rate)
    options = dict(f.defaults)
    if f.name == "egocentric":
        options.pop("bandwidth")
        options.pop("method")
    answer = f.predicate(
        *f.args(r, r.spike_times),
        **options,
        epochs=[(0, 100)],
        spike_window=(0, 1200),
        max_gap=1.0,
    )
    kwargs, result = seen[0]
    assert kwargs["epochs"] == [(0, 100)]
    assert kwargs["spike_window"] == (0, 1200)
    assert kwargs["max_gap"] == 1.0
    assert result.occupancy.sum() <= 100.0 + 1e-9
    keep = r.times <= 100
    sliced = replace(
        r, times=r.times[keep], positions=r.positions[keep], headings=r.headings[keep]
    )
    expected = original(*f.args(sliced, r.spike_times[r.spike_times < 100]), **options)
    np.testing.assert_allclose(result.occupancy, expected.occupancy, rtol=1e-12, atol=0)
    assert isinstance(answer, bool)


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
    # Pooled totals cannot see spikes or occupancy landing in the wrong bin;
    # every well-sampled bin must recover the true rate on its own.
    well_sampled = result.occupancy >= 2.0
    assert well_sampled.sum() >= 5
    np.testing.assert_allclose(
        rates.reshape(-1, rates.shape[-1])[:, well_sampled], 5.0, rtol=0.05
    )


@pytest.mark.parametrize(
    "method",
    [
        "diffusion_kde",
        "gaussian_kde",
        pytest.param("glm", marks=pytest.mark.slow),
    ],
)
def test_spatial_all_methods_recover_rate(two_epoch_recording, method):
    r = two_epoch_recording
    result = compute_spatial_rate(
        r.env, r.spike_times, r.times, r.positions, method=method
    )
    pooled = np.nansum(result.firing_rate * result.occupancy) / result.occupancy.sum()
    assert pooled == pytest.approx(5.0, rel=0.05)
    assert result.occupancy.sum() == pytest.approx(199.96, abs=1e-6)


@pytest.mark.parametrize("max_gap", [0.5, None])
def test_allocentric_object_vector_recovers_rate_across_pause(
    two_epoch_recording, max_gap
):
    """The default gap gate, not explicit epochs, keeps the pause out."""
    from neurospatial.encoding import compute_object_vector_rate

    r = two_epoch_recording
    result = compute_object_vector_rate(
        r.env,
        r.spike_times,
        r.times,
        r.positions,
        np.array([[50.0, 50.0]]),
        distance_range=(0, 100),
        method="binned",
        max_gap=max_gap,
    )
    pooled = np.nansum(result.firing_rate * result.occupancy) / result.occupancy.sum()
    if max_gap is None:
        # The 1000-second pause is charged to one bin, diluting the rate.
        assert pooled < 1.0
    else:
        assert pooled == pytest.approx(5.0, rel=0.05)
        assert result.occupancy.sum() == pytest.approx(199.96, abs=1e-6)
        assert result.occupancy.max() < 100  # no bin absorbs the pause


@pytest.mark.parametrize("name", ["has_place_field", "is_place_cell"])
def test_place_predicates_forward_time_windows(continuous_recording, monkeypatch, name):
    import neurospatial.encoding.spatial as spatial

    r = continuous_recording
    original = spatial.compute_spatial_rate
    seen = []

    def track_rate(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append((kwargs, result))
        return result

    monkeypatch.setattr(spatial, "compute_spatial_rate", track_rate)
    answer = getattr(spatial, name)(
        r.env,
        r.spike_times,
        r.times,
        r.positions,
        epochs=[(0, 100)],
        spike_window=(0, 1200),
        max_gap=1.0,
        **({"criterion": "spatial_info"} if name == "is_place_cell" else {}),
    )
    kwargs, result = seen[0]
    assert kwargs["epochs"] == [(0, 100)]
    assert kwargs["spike_window"] == (0, 1200)
    assert kwargs["max_gap"] == 1.0
    assert result.occupancy.sum() == pytest.approx(100.0, abs=1e-9)
    assert isinstance(answer, bool)


_FREE_PREDICATES = {
    "place": "is_place_cell",
    "head_direction": "is_head_direction_cell",
    "view": "is_spatial_view_cell",
    "object_vector": "is_object_vector_cell",
    "egocentric_object_vector": "is_egocentric_object_vector_cell",
}


def test_shuffle_predicates_forward_time_windows(
    significance_family, significance_recording, monkeypatch
):
    """Dropping a window from a shuffle branch would test the whole session."""
    f, r = significance_family, significance_recording
    significance = f.function
    seen = []

    def track(*args, **kwargs):
        seen.append(kwargs)
        return significance(*args, **kwargs)

    monkeypatch.setattr(f.module, significance.__name__, track)
    predicate = getattr(f.module, _FREE_PREDICATES[f.name])
    windows = {"epochs": [(0.0, 30.0)], "spike_window": (0.0, 50.0), "max_gap": 1.0}
    verdict = predicate(
        *f.args(r, r.trains[0]),
        criterion="shuffle",
        n_shuffles=5,
        min_shift=1.0,
        rng=0,
        **windows,
    )
    assert isinstance(verdict, bool)
    assert len(seen) == 1
    for key, value in windows.items():
        assert seen[0][key] == value
