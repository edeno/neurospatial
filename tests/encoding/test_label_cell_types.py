"""Spatial cell-type screens judge each label by its own statistic."""

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates


@pytest.fixture(scope="module")
def field_and_noise(ou_env, ou_10min, noise_trains, allocentric_field_spikes):
    times, positions, _ = ou_10min
    rates = compute_spatial_rates(
        ou_env,
        [allocentric_field_spikes, noise_trains(600)[0]],
        times,
        positions,
        bandwidth=5,
    )
    tuned = rates.spatial_information() >= 0.5
    assert tuned.tolist() == [True, False]
    return rates


@pytest.mark.parametrize(
    ("grid", "border", "expected"),
    [
        ([0.9, 0.9], [0.0, 0.0], ["grid", "grid"]),
        ([0.0, 0.0], [0.9, 0.9], ["border", "border"]),
        ([0.9, 0.9], [0.9, 0.9], ["grid+border", "grid+border"]),
        ([0.0, 0.0], [0.0, 0.0], ["place", "unclassified"]),
    ],
)
def test_labels_follow_their_own_statistic(field_and_noise, grid, border, expected):
    """No information gate on grid or border, and no grid-over-border rule."""
    labels = field_and_noise.label_cell_types(
        grid_scores=np.array(grid),
        border_scores=np.array(border),
        boundary_source="analysis",
    )
    np.testing.assert_array_equal(labels, expected)


def test_table_reports_each_screen(field_and_noise):
    table = field_and_noise.summary_table(boundary_source="analysis")
    thresholds = table.attrs["classification_thresholds"]
    for name, cutoff in [
        ("spatial_information", thresholds["min_spatial_info"]),
        ("grid_score", thresholds["min_grid_score"]),
        ("border_score", thresholds["min_border_score"]),
    ]:
        np.testing.assert_array_equal(
            table[f"passes_{name}"].to_numpy(dtype=bool),
            table[name].to_numpy() >= cutoff,
        )
    np.testing.assert_array_equal(
        table["cell_type"],
        field_and_noise.label_cell_types(boundary_source="analysis"),
    )


def test_track_place_cell_is_place():
    """Grid and border cells need a 2-D map; a tuned track cell is a place cell."""
    from neurospatial.simulation import generate_poisson_spikes

    env = Environment.from_samples(np.arange(0.0, 201.0)[:, None], bin_size=2.0)
    times = np.arange(0.0, 600.0, 1 / 30)
    x = 200 * np.abs((times / 20 + 0.37) % 2 - 1)
    field = 0.1 + 12 * np.exp(-0.5 * ((x - 70) / 8) ** 2)
    trains = [
        generate_poisson_spikes(field, times, seed=11),
        generate_poisson_spikes(np.full(len(times), 1.5), times, seed=12),
    ]
    rates = compute_spatial_rates(env, trains, times, x[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = rates.label_cell_types()
    np.testing.assert_array_equal(labels, ["place", "unclassified"])
    table = rates.summary_table()
    np.testing.assert_array_equal(table["cell_type"], labels)
    assert not table["passes_grid_score"].any()
    assert not table["passes_border_score"].isna().any()
    assert set(table.attrs["unavailable_metrics"]) == {"grid_score", "border_score"}


def test_unavailable_border_score_is_undetermined_not_an_error(
    ou_env, ou_10min, allocentric_field_spikes, noise_trains
):
    """Without measured walls any cell could be a border cell."""
    times, positions, _ = ou_10min
    rates = compute_spatial_rates(
        ou_env,
        [allocentric_field_spikes, noise_trains(600)[0]],
        times,
        positions,
        bandwidth=5,
    )
    with pytest.warns(UserWarning, match="border_score cannot be computed"):
        labels = rates.label_cell_types()
    table = rates.summary_table()
    np.testing.assert_array_equal(labels, ["undetermined", "undetermined"])
    np.testing.assert_array_equal(table["cell_type"], labels)
    assert table["passes_border_score"].isna().all()
    assert table["passes_spatial_information"].tolist() == [True, False]
