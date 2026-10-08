"""Headline summaries describe aggregation and stay safe for missing maps."""

from dataclasses import replace

import numpy as np
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
