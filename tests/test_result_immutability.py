"""Rate and peri-event results own read-only copies of their arrays.

Results are frozen dataclasses, and their arrays are frozen too: editing an
array a result was built from, or one read back from it, must not silently
change the analysis it reports. Exports (``to_dataframe``, ``to_xarray``) are
the caller's to edit, so they are writable copies.
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.directional import compute_directional_rates
from neurospatial.encoding.egocentric import compute_egocentric_rates
from neurospatial.encoding.spatial import SpatialRatesResult, compute_spatial_rates
from neurospatial.encoding.view import compute_view_rates
from neurospatial.events import population_peri_event_histogram

FAMILIES = ["spatial", "directional", "view", "egocentric", "peri_event"]


@pytest.fixture(scope="module")
def populations() -> dict[str, object]:
    """One population result per family from a small seeded session."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    headings = rng.uniform(-np.pi, np.pi, 500)
    trains = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    objects = np.array([[25.0, 25.0]])
    return {
        "spatial": compute_spatial_rates(env, trains, times, positions),
        "directional": compute_directional_rates(trains, times, headings),
        "view": compute_view_rates(
            env,
            trains,
            times,
            positions,
            headings,
            gaze_model="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
        ),
        "egocentric": compute_egocentric_rates(
            None,
            trains,
            times,
            positions,
            headings,
            objects,
            distance_range=(0.0, 50.0),
        ),
        "peri_event": population_peri_event_histogram(
            trains, np.array([10.0, 20.0]), (-1.0, 1.0)
        ),
    }


def _result(populations, family, level):
    population = populations[family]
    return population if level == "population" else population[0]


def _arrays(result) -> dict[str, np.ndarray]:
    return {
        name: value
        for name, value in vars(result).items()
        if isinstance(value, np.ndarray)
    }


@pytest.mark.parametrize("level", ["population", "single"])
@pytest.mark.parametrize("family", FAMILIES)
def test_result_arrays_are_read_only(populations, family, level):
    arrays = _arrays(_result(populations, family, level))
    assert "occupancy" in arrays or "histograms" in arrays or "histogram" in arrays
    for name, value in arrays.items():
        assert value.flags.writeable is False, name


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
@pytest.mark.parametrize("level", ["population", "single"])
@pytest.mark.parametrize("family", FAMILIES)
def test_copied_result_stays_read_only(populations, family, level, how):
    result = _result(populations, family, level)
    copied = (
        pickle.loads(pickle.dumps(result)) if how == "pickle" else copy.deepcopy(result)
    )
    original = _arrays(result)
    for name, value in _arrays(copied).items():
        assert value.flags.writeable is False, name
        np.testing.assert_array_equal(value, original[name])


def test_result_is_isolated_from_constructor_arrays():
    corners = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    env = Environment.from_samples(corners, bin_size=5.0)
    rates = np.ones((2, env.n_bins))
    occupancy = np.ones(env.n_bins)
    result = SpatialRatesResult(
        firing_rates=rates,
        occupancy=occupancy,
        env=env,
        method="binned",
        bandwidth=2.0,
    )
    rates[:] = 7.0
    occupancy[:] = 0.0
    np.testing.assert_array_equal(result.firing_rates, 1.0)
    np.testing.assert_array_equal(result.occupancy, 1.0)


@pytest.mark.parametrize("family", ["spatial", "directional", "view", "egocentric"])
@pytest.mark.parametrize("level", ["population", "single"])
def test_analysis_settings_are_independent_copies(populations, family, level):
    """Editing provenance cannot change stored settings or effective support."""
    settings = copy.deepcopy(_result(populations, family, level).analysis_settings)
    result = replace(_result(populations, family, level), analysis_settings=settings)
    expected = copy.deepcopy(settings)
    support = result.spatial_support.copy()

    settings["min_occupancy"] = 1000.0
    settings["observed_intervals"][0][0] = -10.0
    returned = result.analysis_settings
    returned["min_occupancy"] = 1000.0
    returned["observed_intervals"][0][0] = -20.0

    assert result.analysis_settings == expected
    np.testing.assert_array_equal(result.spatial_support, support)


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_restored_rate_settings_are_independent(populations, how):
    result = populations["spatial"]
    restored = (
        pickle.loads(pickle.dumps(result)) if how == "pickle" else copy.deepcopy(result)
    )
    expected = copy.deepcopy(restored.analysis_settings)
    restored.analysis_settings["min_occupancy"] = 1000.0
    assert restored.analysis_settings == expected


def test_decoder_settings_are_independent(populations):
    from neurospatial.decoding import BayesianDecoder

    decoder = BayesianDecoder.from_rates(populations["spatial"])
    expected = copy.deepcopy(decoder.analysis_settings)
    decoder.analysis_settings["min_occupancy"] = 1000.0
    decoder.analysis_settings["observed_intervals"][0][0] = -10.0
    assert decoder.analysis_settings == expected


@pytest.mark.parametrize("family", FAMILIES)
def test_dataframe_export_is_an_editable_copy(populations, family):
    result = populations[family]
    table = result.to_dataframe()
    column = "firing_rate" if "firing_rate" in table else table.columns[-1]
    before = np.array(
        result.firing_rates if family != "peri_event" else result.histograms
    )
    table.loc[table.index[0], column] = -5.0
    after = result.firing_rates if family != "peri_event" else result.histograms
    np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("family", ["spatial", "directional", "view", "egocentric"])
def test_xarray_export_is_an_editable_copy(populations, family):
    pytest.importorskip("xarray")
    result = populations[family]
    dataset = result.to_xarray()
    dataset["firing_rate"][0, 0] = -5.0
    assert np.asarray(result.firing_rates)[0, 0] != -5.0


def test_array_records_compare_by_identity_instead_of_raising():
    from neurospatial._observation import resolve_sample_support
    from neurospatial.encoding import SpikeTrains

    trains = SpikeTrains([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
    support = resolve_sample_support(
        np.arange(4.0),
        np.ones(4, dtype=bool),
        max_gap=None,
        epochs=None,
        break_before=None,
        result="empty",
    )
    for record in (trains, support):
        assert record == record
        assert record != copy.deepcopy(record)  # equal values, no ValueError
