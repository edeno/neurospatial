"""Headline summaries describe aggregation and stay safe for missing maps."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

FAMILIES = ("spatial", "view", "allocentric", "egocentric", "directional")
HEADLINES = {
    "spatial": {"spatial_info": "spatial_information", "sparsity": "sparsity"},
    "view": {"view_spatial_info": "view_spatial_information"},
    "allocentric": {
        "preferred_distance": "preferred_distance",
        "preferred_direction": "preferred_direction",
    },
    "egocentric": {
        "preferred_distance": "preferred_distance",
        "preferred_direction": "preferred_direction",
    },
    "directional": {
        "preferred_direction": "preferred_direction",
        "mean_vector_length": "mean_vector_length",
    },
}
TABLE_COLUMNS = {
    "spatial": [
        "peak_rate",
        "spatial_info",
        "sparsity",
        "grid_score",
        "border_score",
        "peak_x",
        "peak_y",
        "cell_type",
    ],
    "view": [
        "peak_rate",
        "view_spatial_info",
        "peak_x",
        "peak_y",
        "is_spatial_view_cell",
    ],
    "allocentric": [
        "peak_rate",
        "preferred_distance",
        "preferred_direction_deg",
        "preferred_direction",
        "is_object_vector_cell",
    ],
    "egocentric": [
        "peak_rate",
        "preferred_distance",
        "preferred_direction_deg",
        "preferred_direction",
        "is_object_vector_cell",
    ],
    "directional": [
        "peak_rate",
        "mean_vector_length",
        "preferred_direction_deg",
        "tuning_width_deg",
        "preferred_direction",
        "tuning_width",
        "is_head_direction_cell",
    ],
}


@pytest.mark.parametrize("family", FAMILIES)
def test_summary_table_column_order_and_attrs(rate_family_results, family):
    result = rate_family_results[family]
    frame = result.summary_table()
    assert frame.columns.tolist() == TABLE_COLUMNS[family]
    assert "method" not in frame.columns
    assert frame.attrs["units"]["peak_rate"] == "Hz"
    if family in ("spatial", "view"):
        assert frame.attrs["method"] == result.method
    if family == "spatial":
        assert "spatial_info" in frame.to_string(max_cols=6)
        assert "cell_type" in frame.to_string(max_cols=6)
        assert frame.attrs["units"]["spatial_info"] == "bits/spike"
    if family in ("allocentric", "egocentric"):
        assert frame.attrs["direction_frame"] == family
        assert frame.attrs["units"]["preferred_direction"] == "rad"
        assert frame.attrs["units"]["preferred_direction_deg"] == "deg"


@pytest.mark.parametrize("family", FAMILIES)
def test_thresholds_attrs_match_resolved_defaults(rate_family_results, family):
    from neurospatial.encoding.directional import HEAD_DIRECTION_THRESHOLDS
    from neurospatial.encoding.egocentric import OBJECT_VECTOR_THRESHOLDS
    from neurospatial.encoding.spatial import PLACE_GRID_BORDER_THRESHOLDS
    from neurospatial.encoding.view import VIEW_THRESHOLDS

    constants = {
        "spatial": PLACE_GRID_BORDER_THRESHOLDS,
        "view": VIEW_THRESHOLDS,
        "allocentric": OBJECT_VECTOR_THRESHOLDS,
        "egocentric": OBJECT_VECTOR_THRESHOLDS,
        "directional": HEAD_DIRECTION_THRESHOLDS,
    }
    result = rate_family_results[family]
    defaults = dict(constants[family])
    assert result.summary_table().attrs["classification_thresholds"] == defaults
    assert result[0].summary_table().attrs["classification_thresholds"] == defaults
    classifier = result.label_cell_types if family == "spatial" else result.classify
    np.testing.assert_array_equal(classifier(), classifier(**defaults))


@pytest.mark.parametrize("family", FAMILIES)
def test_population_summary_names_aggregation(rate_family_results, family):
    result = rate_family_results[family]
    summary = result.summary()
    expected = {
        "n_units",
        "n_bins",
        "max_peak_firing_rate",
        "total_occupancy",
        "spike_window_assumed",
        "spike_window",
    }
    if family in ("spatial", "view"):
        expected.add("method")
    if family in ("allocentric", "egocentric"):
        expected.add("direction_frame")
        assert summary["direction_frame"] == family
    assert set(summary) == expected
    assert summary["n_units"] == 3
    assert summary["max_peak_firing_rate"] == np.nanmax(result.peak_firing_rate())
    assert "n_neurons" not in repr(result)
    assert "max_peak_firing_rate=" in repr(result)


@pytest.mark.parametrize("family", ("spatial", "directional"))
def test_total_occupancy_is_shared_not_summed(rate_family_results, family):
    result = rate_family_results[family]
    assert result.occupancy.ndim == 1
    assert result.summary()["total_occupancy"] == pytest.approx(59.967, abs=1e-3)
    assert result.summary()["total_occupancy"] == result[0].summary()["total_occupancy"]


@pytest.mark.parametrize("family", FAMILIES)
def test_singular_summary_headline_metrics(rate_family_results, family):
    result = rate_family_results[family][0]
    summary = result.summary()
    assert "peak_firing_rate" in summary
    assert "max_peak_firing_rate" not in summary
    for key, method in HEADLINES[family].items():
        assert summary[key] == pytest.approx(
            float(getattr(result, method)()), nan_ok=True
        )
        assert f"{key}=" in repr(result)
    dead = replace(result, firing_rate=np.full_like(result.firing_rate, np.nan))
    for key in ("peak_firing_rate", *HEADLINES[family]):
        assert np.isnan(dead.summary()[key])
    assert repr(dead)


@pytest.mark.parametrize("family", FAMILIES)
def test_summary_table_single_matches_batch(rate_family_results, family):
    result = rate_family_results[family]
    batch = result.summary_table()
    for i in range(3):
        single = result[i].summary_table()
        pd.testing.assert_frame_equal(single, batch.iloc[[i]])
        assert single.attrs == batch.attrs


@pytest.mark.parametrize("family", ("view", "allocentric", "egocentric", "directional"))
def test_singular_verdict_matches_table_and_classify(rate_family_results, family):
    result = rate_family_results[family]
    predicate = {
        "view": "is_spatial_view_cell",
        "allocentric": "is_object_vector_cell",
        "egocentric": "is_object_vector_cell",
        "directional": "is_head_direction_cell",
    }[family]
    batch = result.summary_table()
    for i in range(3):
        expected = getattr(result[i], predicate)()
        assert result.classify()[i] == expected
        assert result[i].summary_table()[predicate].iloc[0] == expected
        assert batch[predicate].iloc[i] == expected


@pytest.mark.parametrize("pooled", [True, False])
def test_glm_summary_table_single_matches_batch(rate_family_inputs, pooled):
    from neurospatial import compute_spatial_rates

    env, times, positions, _, spikes, _ = rate_family_inputs
    trains = spikes if pooled else [spikes[0], spikes[1], np.array([])]
    result = compute_spatial_rates(
        env, trains, times, positions, method="glm", pooled=pooled
    )
    batch = result.summary_table()
    for i in range(3):
        pd.testing.assert_frame_equal(result[i].summary_table(), batch.iloc[[i]])
        assert result[i].summary_table().attrs == batch.attrs
    if not pooled:
        assert result[2].penalty_selected_by_reml is False
        assert not result[2].summary_table()["penalty_selected_by_reml"].iloc[0]


def test_standalone_xarray_retains_absent_identity_and_recording_windows(
    rate_family_inputs,
):
    xr = pytest.importorskip("xarray")
    from neurospatial import compute_spatial_rate

    env, times, positions, _, spikes, _ = rate_family_inputs
    result = compute_spatial_rate(
        env, spikes[0], times, positions, spike_window=(0, 60)
    )
    ds = result.to_xarray()
    assert ds.sizes == {"unit_id": 1, "bin": env.n_bins}
    assert pd.isna(ds.unit_id.values[0])
    assert ds.attrs["spike_window_assumed"] == 0
    np.testing.assert_array_equal(ds.attrs["spike_window"], [0, 60])
    xr.testing.assert_equal(
        ds.firing_rate.squeeze("unit_id", drop=True),
        xr.DataArray(
            result.firing_rate,
            dims="bin",
            coords={name: ds.coords[name] for name in ds.coords if name != "unit_id"},
        ),
    )
