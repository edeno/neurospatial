"""Object-vector maps report and recover their measured reference frame."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import encoding


def test_allocentric_equals_zero_heading_egocentric(ou_env, ou_2min, noise_trains, obj):
    times, positions, _ = ou_2min
    trains = noise_trains(120)[:2]
    allo = encoding.compute_object_vector_rates(
        ou_env, trains, times, positions, obj, unit_ids=[11, 37]
    )
    ego = encoding.compute_egocentric_rates(
        ou_env, trains, times, positions, np.zeros(len(times)), obj, unit_ids=[11, 37]
    )
    np.testing.assert_array_equal(allo.firing_rates, ego.firing_rates)
    np.testing.assert_array_equal(allo.occupancy, ego.occupancy)
    np.testing.assert_array_equal(allo.unit_ids, [11, 37])
    assert allo[0].unit_id == 11


def test_allocentric_recovers_east_field(
    ou_env, ou_10min, obj, allocentric_field_spikes
):
    times, positions, headings = ou_10min
    allo = encoding.compute_object_vector_rate(
        ou_env, allocentric_field_spikes, times, positions, obj
    )
    ego = encoding.compute_egocentric_rate(
        ou_env, allocentric_field_spikes, times, positions, headings, obj
    )
    angle_error = np.angle(np.exp(1j * (allo.preferred_direction() - np.pi)))
    assert abs(angle_error) <= np.pi / 6
    assert allo.spatial_information() >= 1.3 * ego.spatial_information()


def test_egocentric_recovers_ahead_ovc(ou_env, ou_10min, obj):
    from neurospatial.simulation import ObjectVectorCellModel, generate_poisson_spikes

    times, positions, headings = ou_10min
    model = ObjectVectorCellModel(
        ou_env,
        direction_frame="egocentric",
        object_positions=obj,
        preferred_distance=20,
        distance_width=5,
        preferred_direction=0.0,
        max_rate=10,
    )
    spikes = generate_poisson_spikes(
        model.firing_rate(positions, headings=headings), times, seed=5
    )
    allo = encoding.compute_object_vector_rate(ou_env, spikes, times, positions, obj)
    ego = encoding.compute_egocentric_rate(
        ou_env, spikes, times, positions, headings, obj
    )
    assert abs(ego.preferred_direction()) <= np.pi / 6
    assert ego.spatial_information() >= 1.3 * allo.spatial_information()


@pytest.mark.parametrize("frame", ["allocentric", "egocentric"])
def test_result_records_frame(frame, ou_env, ou_2min, noise_trains, obj):
    times, positions, headings = ou_2min
    extra = () if frame == "allocentric" else (headings,)
    single_fn = getattr(
        encoding, f"compute_{'object_vector' if frame == 'allocentric' else frame}_rate"
    )
    plural_fn = getattr(
        encoding,
        f"compute_{'object_vector' if frame == 'allocentric' else frame}_rates",
    )
    trains = noise_trains(120)[:2]
    single = single_fn(ou_env, trains[0], times, positions, *extra, obj)
    plural = plural_fn(ou_env, trains, times, positions, *extra, obj)
    for result in (single, plural, plural[0]):
        assert result.direction_frame == result.summary()["direction_frame"] == frame
    with pytest.raises(TypeError, match="direction_frame"):
        encoding.ObjectVectorRateResult(
            single.firing_rate,
            single.occupancy,
            single.env,
            single.distance_range,
            single.n_distance_bins,
            single.n_direction_bins,
        )
    with pytest.raises(TypeError, match="direction_frame"):
        encoding.ObjectVectorRatesResult(
            plural.firing_rates,
            plural.occupancy,
            plural.env,
            plural.distance_range,
            plural.n_distance_bins,
            plural.n_direction_bins,
        )


def test_frame_compute_signatures_are_truthful():
    for name in ("compute_object_vector_rate", "compute_object_vector_rates"):
        params = inspect.signature(getattr(encoding, name)).parameters
        assert "headings" not in params and "direction_frame" not in params
    for name in ("compute_egocentric_rate", "compute_egocentric_rates"):
        params = inspect.signature(getattr(encoding, name)).parameters
        assert params["headings"].default is inspect.Parameter.empty
        assert "direction_frame" not in params
    with pytest.raises(TypeError, match="headings"):
        encoding.compute_object_vector_rate(None, [], [], [], [], headings=[])


def test_allocentric_preserves_gap_and_window_gates():
    times = np.array([0, 0.1, 0.2, 100, 100.1, 100.2])
    positions = np.zeros((len(times), 2))
    result = encoding.compute_object_vector_rate(
        None,
        [0.05, 50, 100.05],
        times,
        positions,
        [[10, 0]],
        epochs=[[0, 0.2], [100, 100.2]],
        spike_window=[0, 100.2],
    )
    assert result.occupancy.sum() == pytest.approx(0.4)
    assert np.nansum(result.firing_rate * result.occupancy) == pytest.approx(2)
    assert result.spike_window_assumed is False
    empty = encoding.compute_object_vector_rates(None, [], times, positions, [[10, 0]])
    assert empty.firing_rates.shape == (0, 120)
    assert empty.direction_frame == "allocentric"
    assert empty.occupancy.sum() == pytest.approx(0.4)


def test_egocentric_requires_nonnull_headings():
    with pytest.raises(ValueError, match=r"headings.*required"):
        encoding.compute_egocentric_rate(
            None, [], [0, 0.1], [[0, 0], [0, 0]], None, [[10, 0]]
        )


@pytest.mark.parametrize("frame", ["allocentric", "egocentric"])
@pytest.mark.parametrize("method", ["binned", "gaussian_kde"])
def test_free_predicates_match_methods(
    frame, method, ou_env, ou_2min, noise_trains, obj
):
    times, positions, headings = ou_2min
    extra = () if frame == "allocentric" else (headings,)
    family = "object_vector" if frame == "allocentric" else "egocentric"
    predicate = getattr(
        encoding,
        f"is_{'' if frame == 'allocentric' else 'egocentric_'}object_vector_cell",
    )
    trains = noise_trains(120)[:2]
    kwargs = {
        "method": method,
        "bandwidth": 1,
        "min_occupancy": 0.05,
        "epochs": [0, 100],
        "spike_window": [5, 105],
    }
    plural = getattr(encoding, f"compute_{family}_rates")(
        ou_env, trains, times, positions, *extra, obj, **kwargs
    )
    for i, train in enumerate(trains):
        single = getattr(encoding, f"compute_{family}_rate")(
            ou_env, train, times, positions, *extra, obj, **kwargs
        )
        for threshold in (0.3, 1.0):
            verdict = predicate(
                ou_env,
                train,
                times,
                positions,
                *extra,
                obj,
                min_info=threshold,
                **kwargs,
            )
            assert verdict == single.is_object_vector_cell(min_info=threshold)
            assert verdict == plural.classify(min_info=threshold)[i]


def test_frame_predicate_signatures_are_truthful():
    allocentric = inspect.signature(encoding.is_object_vector_cell).parameters
    assert "headings" not in allocentric and "direction_frame" not in allocentric
    egocentric = inspect.signature(encoding.is_egocentric_object_vector_cell).parameters
    assert egocentric["headings"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="headings"):
        encoding.is_object_vector_cell(None, [], [], [], [], headings=[])


def test_plot_allocentric_north_is_up(ou_env, ou_2min, obj):
    times, positions, _ = ou_2min
    result = encoding.compute_object_vector_rate(ou_env, [], times, positions, obj)
    # A known north-peaked map checks the actual display transform, independent
    # of estimator/trajectory uncertainty.
    rate = np.zeros_like(result.firing_rate)
    centers = result.env.bin_centers
    peak = np.argmin(np.abs(centers[:, 1] - np.pi / 2) + np.abs(centers[:, 0] - 20))
    rate[peak] = 10
    from dataclasses import replace

    result = replace(result, firing_rate=rate)
    ax = encoding.plot_object_vector_tuning(result)
    ax.figure.canvas.draw()
    origin = ax.transData.transform((0, 0))
    target = ax.transData.transform((np.pi / 2, 20))
    dx, dy = target - origin
    assert dy > 0 and abs(dx) < 0.1 * dy
    assert "allocentric" in ax.get_xlabel()
    plt.close(ax.figure)
