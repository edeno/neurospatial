"""Multi-class spatial labels require their documented information gate."""

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rates


@pytest.fixture(scope="module")
def noise_rates(ou_env, ou_10min, noise_trains):
    times, positions, _ = ou_10min
    return compute_spatial_rates(
        ou_env, noise_trains(600), times, positions, bandwidth=5
    )


def test_label_cell_types_noise_is_unclassified(noise_rates):
    labels = noise_rates.label_cell_types()
    assert np.all(noise_rates.spatial_information() < 0.5)
    assert np.all(labels == "unclassified"), labels


@pytest.mark.parametrize("score_kind", ["border", "grid"])
def test_label_cell_types_border_requires_information(
    ou_env, ou_10min, noise_trains, allocentric_field_spikes, score_kind
):
    times, positions, _ = ou_10min
    rates = compute_spatial_rates(
        ou_env,
        [allocentric_field_spikes, noise_trains(600)[0]],
        times,
        positions,
        bandwidth=5,
    )
    scores = {"grid_scores": np.zeros(2), "border_scores": np.zeros(2)}
    scores[f"{score_kind}_scores"] = np.full(2, 0.9)
    np.testing.assert_array_equal(
        rates.label_cell_types(**scores), [score_kind, "unclassified"]
    )


def test_label_cell_types_short_recording_bias(ou_2min_env, ou_2min, noise_trains):
    times, positions, _ = ou_2min
    rates = compute_spatial_rates(
        ou_2min_env, noise_trains(120), times, positions, bandwidth=5
    )
    labels = rates.label_cell_types()
    tuned = rates.spatial_information() >= 0.5
    np.testing.assert_array_equal(labels == "border", tuned)
    assert np.all(labels[~tuned] == "unclassified")
