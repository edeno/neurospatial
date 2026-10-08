"""Rate-map shuffle tests respect identity, observed intervals and input copies."""

import numpy as np
import pytest

from neurospatial import encoding


def test_shuffle_detects_field_cell_fast(ou_2min_env, ou_2min, strong_field_spikes):
    times, positions, _ = ou_2min
    result = encoding.place_cell_significance(
        ou_2min_env,
        [strong_field_spikes],
        times,
        positions,
        bandwidth=5,
        n_shuffles=20,
        rng=0,
    )[0]
    assert result.p_value == 1 / 21
    assert result.observed_score > np.nanmax(result.null_scores)


def test_threshold_flags_noise_as_object_vector_cell(
    ou_env, ou_10min, noise_trains, obj
):
    times, positions, headings = ou_10min
    trains = noise_trains(600)
    ego = encoding.compute_egocentric_rates(
        ou_env, trains, times, positions, headings, obj
    )
    allo = encoding.compute_object_vector_rates(ou_env, trains, times, positions, obj)
    assert ego.classify().sum() == 20
    assert allo.classify().sum() >= 18


def test_population_labels_and_order(significance_family, significance_recording):
    f, r = significance_family, significance_recording
    ids = np.array([10, 20, 30])
    first = f.function(*f.args(r), unit_ids=ids, n_shuffles=20, rng=0, **f.defaults)
    reverse = f.function(
        *f.args(r, r.trains[::-1]),
        unit_ids=ids[::-1],
        n_shuffles=20,
        rng=0,
        **f.defaults,
    )
    assert list(first) == [10, 20, 30]
    assert list(reverse) == [30, 20, 10]
    for i, uid in enumerate(ids):
        alone = f.function(
            *f.args(r, [r.trains[i]]),
            unit_ids=[uid],
            n_shuffles=20,
            rng=0,
            **f.defaults,
        )[uid]
        assert first[uid].p_value == reverse[uid].p_value == alone.p_value
        np.testing.assert_allclose(
            first[uid].null_scores, alone.null_scores, rtol=0, atol=1e-12
        )
        assert (
            np.min(np.abs(first[uid].null_scores - first[uid].observed_score)) > 1e-12
        )


def test_significance_uses_only_observed_windows(
    significance_family, significance_recording, monkeypatch
):
    f, r = significance_family, significance_recording
    compute = getattr(f.module, f.compute_name)
    recorded = []

    def capture(*args, **kwargs):
        trains = args[0] if f.name == "head_direction" else args[1]
        recorded.append([train.copy() for train in trains])
        return compute(*args, **kwargs)

    monkeypatch.setattr(f.module, f.compute_name, capture)
    f.function(
        *f.args(r),
        epochs=[[0, 20], [30, 60]],
        spike_window=[5, 55],
        n_shuffles=4,
        min_shift=5,
        rng=0,
        **f.defaults,
    )
    assert len(recorded) == 5
    for null_trains in recorded[1:]:
        for train in null_trains:
            assert np.all(
                ((train >= 5) & (train < 20)) | ((train >= 30) & (train < 55))
            )


@pytest.mark.parametrize(
    "significance_family,argument",
    [
        (family, argument)
        for family in (
            "place",
            "head_direction",
            "view",
            "object_vector",
            "egocentric_object_vector",
        )
        for argument in [
            "trains",
            "times",
            "epochs",
            "spike_window",
            "unit_ids",
            "bandwidth",
        ]
        + {
            "place": ["positions", "speed"],
            "head_direction": ["headings"],
            "view": ["positions", "headings", "gaze_offsets"],
            "object_vector": ["positions", "objects"],
            "egocentric_object_vector": ["positions", "headings", "objects"],
        }[family]
    ],
    indirect=["significance_family"],
)
def test_significance_isolated_from_caller_mutation(
    significance_family, significance_recording, monkeypatch, argument
):
    f, r = significance_family, significance_recording
    kwargs = {
        **f.defaults,
        "unit_ids": np.array([10, 20, 30]),
        "epochs": np.array([[0.0, 60.0]]),
        "spike_window": np.array([[0.0, 60.0]]),
    }
    if f.name == "place":
        if argument == "speed":
            # Put the cutoff between original and replacement speeds so
            # caller mutation would change every interval without a copy.
            r.speed[:] = 2e6
            kwargs.update(speed=r.speed, min_speed=1.5e6)
        else:
            kwargs.update(speed=r.speed, min_speed=2.0)
    if f.name == "view":
        kwargs["gaze_offsets"] = r.gaze_offsets
    kwargs["bandwidth"] = np.array(0.2 if f.name == "head_direction" else 1.0)
    if f.name != "head_direction":
        kwargs["method"] = "gaussian_kde"
    clean = f.function(*f.args(r), n_shuffles=4, rng=0, **kwargs)
    target = kwargs[argument] if argument in kwargs else getattr(r, argument)
    compute = getattr(f.module, f.compute_name)
    calls = 0

    def mutate_after_observation(*args, **parameters):
        nonlocal calls
        result = compute(*args, **parameters)
        if calls == 0:
            if argument == "trains":
                target[0][:] += 5
            elif argument == "unit_ids":
                target[:] = [110, 120, 130]
            elif argument == "speed":
                target[:] = 1e6
            else:
                target[...] = 0 if argument not in ("objects", "gaze_offsets") else 5
        calls += 1
        return result

    monkeypatch.setattr(f.module, f.compute_name, mutate_after_observation)
    protected = f.function(*f.args(r), n_shuffles=4, rng=0, **kwargs)
    for uid in clean:
        assert protected[uid].p_value == clean[uid].p_value
        np.testing.assert_array_equal(
            protected[uid].null_scores, clean[uid].null_scores
        )


def test_significance_rejects_duplicate_labels_and_glm(significance_recording):
    r = significance_recording
    with pytest.raises(ValueError, match="repeated"):
        encoding.place_cell_significance(
            r.env, r.trains, r.times, r.positions, unit_ids=[3, 3, 7], n_shuffles=20
        )
    with pytest.raises(ValueError, match="Fix:"):
        encoding.place_cell_significance(
            r.env, r.trains, r.times, r.positions, method="glm", n_shuffles=20
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "place_cell_significance",
        "object_vector_cell_significance",
        "egocentric_object_vector_cell_significance",
    ],
)
def test_shuffle_noise_false_positives(name, ou_env, ou_10min, noise_trains, obj):
    times, positions, headings = ou_10min
    args = (ou_env, noise_trains(600), times, positions)
    if name.startswith("egocentric"):
        args += (headings, obj)
    elif name.startswith("object"):
        args += (obj,)
    kwargs = {"bandwidth": 5} if name.startswith("place") else {}
    result = getattr(encoding, name)(*args, n_shuffles=200, rng=0, **kwargs)
    assert sum(r.p_value < 0.05 for r in result.values()) <= 3


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "place_cell_significance",
        "object_vector_cell_significance",
        "egocentric_object_vector_cell_significance",
    ],
)
def test_shuffle_pooled_false_positive_rate(name, ou_env, ou_10min, noise_trains, obj):
    times, positions, headings = ou_10min
    trains = [train for seed in range(1, 6) for train in noise_trains(600, seed=seed)]
    args = (ou_env, trains, times, positions)
    if name.startswith("egocentric"):
        args += (headings, obj)
    elif name.startswith("object"):
        args += (obj,)
    kwargs = {"bandwidth": 5} if name.startswith("place") else {}
    result = getattr(encoding, name)(*args, n_shuffles=200, rng=0, **kwargs)
    assert sum(r.p_value < 0.05 for r in result.values()) <= 10


@pytest.mark.slow
@pytest.mark.parametrize("egocentric", [False, True])
def test_shuffle_detects_true_cells(
    egocentric, ou_env, ou_10min, obj, allocentric_field_spikes, egocentric_ovc_spikes
):
    times, positions, headings = ou_10min
    if egocentric:
        result = encoding.egocentric_object_vector_cell_significance(
            ou_env,
            [egocentric_ovc_spikes],
            times,
            positions,
            headings,
            obj,
            n_shuffles=200,
            rng=0,
        )[0]
    else:
        result = encoding.object_vector_cell_significance(
            ou_env,
            [allocentric_field_spikes],
            times,
            positions,
            obj,
            n_shuffles=200,
            rng=0,
        )[0]
    assert result.p_value == 1 / 201


def test_actual_tracking_gap_drops_spikes_from_observed_and_null(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    keep = (r.times < 10) | (r.times >= 20)
    r.times, r.positions, r.headings = (
        r.times[keep],
        r.positions[keep],
        r.headings[keep],
    )
    with_gap_spikes = [np.sort(np.r_[train, 15.0]) for train in r.trains]
    clean = [train[(train < 9.9) | (train >= 20)] for train in with_gap_spikes]
    first = f.function(
        *f.args(r, with_gap_spikes), n_shuffles=4, min_shift=5, rng=0, **f.defaults
    )
    second = f.function(
        *f.args(r, clean), n_shuffles=4, min_shift=5, rng=0, **f.defaults
    )
    for uid in first:
        assert first[uid].observed_score == second[uid].observed_score
        np.testing.assert_array_equal(first[uid].null_scores, second[uid].null_scores)


def test_input_group_labels_cannot_be_overridden(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    group = encoding.SpikeTrains(r.trains[:2], unit_ids=[10, 20])
    with pytest.raises(ValueError) as caught:
        f.function(
            *f.args(r, group), unit_ids=[20, 10], n_shuffles=4, rng=0, **f.defaults
        )
    assert "10" in str(caught.value) and "20" in str(caught.value)
    assert "Fix:" in str(caught.value)
