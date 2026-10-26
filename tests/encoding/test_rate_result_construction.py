"""Construction contract shared by every rate result family.

Spatial, directional, view, object-vector and boundary-vector rate results,
singular and population, check their arrays and settings when built, own
their per-unit metadata and carry unit-label provenance the same way.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import (
    compute_boundary_vector_rates,
    compute_directional_rates,
    compute_egocentric_rates,
    compute_spatial_rates,
    compute_view_rates,
)

N_SAMPLES = 1000
DURATION = 100.0
# Every sampled position is within 57 cm of the object, so no frame is out of range.
OBJECT_RANGE = (0.0, 80.0)


@pytest.fixture(scope="module")
def recording() -> dict[str, Any]:
    """A seeded open-field session with three units and an object."""
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, DURATION, N_SAMPLES)
    positions = rng.uniform(10, 90, (N_SAMPLES, 2))
    headings = rng.uniform(-np.pi, np.pi, N_SAMPLES)
    spike_rng = np.random.default_rng(1)
    spike_times = [np.sort(spike_rng.uniform(0, DURATION, k)) for k in (100, 150, 50)]
    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=10.0, units="cm")
    return {
        "env": env,
        "times": times,
        "positions": positions,
        "headings": headings,
        "spike_times": spike_times,
        "object_positions": np.array([[50.0, 50.0]]),
    }


def _spatial(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_spatial_rates(
        rec["env"], rec["spike_times"], rec["times"], rec["positions"], **kwargs
    )


def _directional(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_directional_rates(
        rec["spike_times"], rec["times"], rec["headings"], **kwargs
    )


def _view(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_view_rates(
        rec["env"],
        rec["spike_times"],
        rec["times"],
        rec["positions"],
        rec["headings"],
        **kwargs,
    )


def _object_vector(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_egocentric_rates(
        rec["env"],
        rec["spike_times"],
        rec["times"],
        rec["positions"],
        rec["headings"],
        rec["object_positions"],
        distance_range=OBJECT_RANGE,
        **kwargs,
    )


def _boundary_vector(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_boundary_vector_rates(
        rec["env"],
        rec["spike_times"],
        rec["times"],
        rec["positions"],
        rec["headings"],
        direction_frame="egocentric",
        **kwargs,
    )


FAMILIES: dict[str, Callable[..., Any]] = {
    "spatial": _spatial,
    "directional": _directional,
    "view": _view,
    "object_vector": _object_vector,
    "boundary_vector": _boundary_vector,
}


@pytest.fixture(scope="module", params=list(FAMILIES))
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def population(family: str, recording: dict[str, Any]) -> Any:
    """Each family's population result with generated unit labels."""
    return FAMILIES[family](recording)


@pytest.fixture(scope="module")
def labelled_population(family: str, recording: dict[str, Any]) -> Any:
    """Each family's population result with caller-supplied unit labels."""
    return FAMILIES[family](recording, unit_ids=["a", "b", "c"])


# ---------------------------------------------------------------------------
# Unit-label provenance
# ---------------------------------------------------------------------------


def test_generated_labels_are_marked_on_every_unit(population: Any) -> None:
    """Indexing without unit_ids marks each unit's label as generated."""
    assert population._unit_ids_generated is True
    for i, unit in enumerate(population):
        assert unit.unit_id == i
        assert unit._unit_ids_generated is True
    assert population.sel(unit_id=1)._unit_ids_generated is True
    assert dataclasses.replace(population)._unit_ids_generated is True


def test_supplied_labels_are_not_marked_generated(labelled_population: Any) -> None:
    assert labelled_population._unit_ids_generated is False
    assert [unit._unit_ids_generated for unit in labelled_population] == [False] * 3
    assert labelled_population[1].unit_id == "b"


# ---------------------------------------------------------------------------
# Estimator metadata
# ---------------------------------------------------------------------------


def test_every_family_reports_its_estimator(population: Any) -> None:
    """method and bandwidth are fields on every family and reach every export."""
    method, bandwidth = population.method, population.bandwidth
    assert method == population.analysis_settings["method"]
    assert bandwidth == population.analysis_settings["bandwidth"]
    assert (population[0].method, population[0].bandwidth) == (method, bandwidth)
    assert population.summary()["method"] == method
    assert population[0].summary()["method"] == method
    table = population.summary_table()
    assert "method" not in table.columns
    assert (table.attrs["method"], table.attrs["bandwidth"]) == (method, bandwidth)
    assert (population.to_dataframe()["method"] == method).all()


@pytest.mark.parametrize("family_name", ["object_vector", "boundary_vector"])
def test_polar_kernel_estimate_records_its_bandwidth(
    family_name: str, recording: dict[str, Any]
) -> None:
    """A smoothed polar map carries the bandwidth it was smoothed with."""
    binned = FAMILIES[family_name](recording)
    smoothed = FAMILIES[family_name](recording, method="gaussian_kde", bandwidth=8.0)
    assert (binned.method, binned.bandwidth) == ("binned", None)
    assert (smoothed.method, smoothed.bandwidth) == ("gaussian_kde", 8.0)
    assert smoothed[2].bandwidth == 8.0


# ---------------------------------------------------------------------------
# Construction checks
# ---------------------------------------------------------------------------


def test_population_refuses_occupancy_of_other_bins(population: Any) -> None:
    with pytest.raises(ValueError, match=r"occupancy has shape \(\d+,\), not"):
        dataclasses.replace(population, occupancy=population.occupancy[:-1])


def test_population_refuses_rates_of_other_bins(population: Any) -> None:
    rates = np.asarray(population.firing_rates)
    with pytest.raises(ValueError, match="firing_rates has shape"):
        dataclasses.replace(population, firing_rates=rates[:, :-1])


def test_population_refuses_one_dimensional_rates(population: Any) -> None:
    """A single map is not a population of one unit per bin."""
    rates = np.asarray(population.firing_rates)
    with pytest.raises(ValueError, match=r"not \(n_units, n_bins\)"):
        dataclasses.replace(population, firing_rates=rates[0])


def test_single_unit_refuses_rates_of_other_bins(population: Any) -> None:
    unit = population[0]
    with pytest.raises(ValueError, match=r"firing_rate has shape .* not \(n_bins,\)"):
        dataclasses.replace(unit, firing_rate=np.asarray(unit.firing_rate)[:-1])
    with pytest.raises(ValueError, match="occupancy has shape"):
        dataclasses.replace(unit, occupancy=np.ones(3))


@pytest.mark.parametrize("bandwidth", [-5.0, 0.0, np.nan, np.inf, "wide"])
def test_refuses_bandwidth_the_estimator_cannot_use(
    population: Any, bandwidth: Any
) -> None:
    """Every default estimator needs a positive bandwidth or none at all."""
    for result in (population, population[0]):
        with pytest.raises(ValueError, match=r"bandwidth=.* is not valid for method"):
            dataclasses.replace(result, bandwidth=bandwidth)


def test_bandwidth_is_stored_as_a_float(population: Any) -> None:
    if population.bandwidth is None:
        pytest.skip("this family's default estimator does not smooth")
    rebuilt = dataclasses.replace(population, bandwidth=np.array(population.bandwidth))
    assert type(rebuilt.bandwidth) is float
    assert rebuilt.bandwidth == population.bandwidth


def test_refuses_unknown_estimator(population: Any) -> None:
    if not any(f.name == "method" for f in dataclasses.fields(population)):
        pytest.skip("directional results derive method from bandwidth")
    with pytest.raises(ValueError, match="method='bogus' is not an estimator"):
        dataclasses.replace(population, method="bogus")


@pytest.mark.parametrize(
    "spike_window",
    [np.ones((2, 2, 2)), [(5.0, 1.0)], [(0.0, np.nan)]],
    ids=["three-d", "inverted", "nan"],
)
def test_refuses_malformed_spike_window(population: Any, spike_window: Any) -> None:
    with pytest.raises(ValueError, match="spike_window"):
        dataclasses.replace(population, spike_window=spike_window)


def test_spike_window_is_normalized_rows(population: Any) -> None:
    """Windows are stored as sorted, merged (n_windows, 2) rows in seconds."""
    rebuilt = dataclasses.replace(population, spike_window=[(50.0, 60.0), (0.0, 55.0)])
    np.testing.assert_array_equal(rebuilt.spike_window, [[0.0, 60.0]])
    assert not rebuilt.spike_window.flags.writeable


def test_list_inputs_become_read_only_arrays(population: Any) -> None:
    unit = population[0]
    rebuilt = dataclasses.replace(
        unit,
        firing_rate=np.asarray(unit.firing_rate).tolist(),
        occupancy=np.asarray(unit.occupancy).tolist(),
    )
    assert isinstance(rebuilt.firing_rate, np.ndarray)
    assert not rebuilt.firing_rate.flags.writeable
    assert not rebuilt.occupancy.flags.writeable


def test_directional_refuses_counts_and_bins_that_disagree() -> None:
    from neurospatial.encoding import DirectionalRateResult, DirectionalRatesResult

    with pytest.raises(ValueError, match="not \\(n_bins,\\) over the result's 3 bins"):
        DirectionalRateResult(
            np.ones(10), np.ones(7), np.linspace(0, 360, 3), 6.0, None
        )
    with pytest.raises(ValueError, match="occupancy has shape \\(7,\\)"):
        DirectionalRateResult(np.ones(3), np.ones(7), np.zeros(3), 2.0, None)
    with pytest.raises(ValueError, match=r"bin_size=12\.0 is not an angular bin width"):
        DirectionalRateResult(np.ones(3), np.ones(3), np.zeros(3), 6.0 * 2, None)
    with pytest.raises(ValueError, match="bin_centers must be a 1-D array"):
        DirectionalRateResult(np.ones(3), np.ones(3), np.zeros((3, 1)), 2.0, None)
    with pytest.raises(ValueError, match="spike_counts has shape"):
        DirectionalRatesResult(
            np.ones((2, 3)), np.ones(3), np.zeros(3), 2.0, None, np.ones((2, 4))
        )


def test_view_refuses_unknown_gaze_model(recording: dict[str, Any]) -> None:
    view = _view(recording)
    with pytest.raises(ValueError, match="gaze_model='nonsense' is not a gaze model"):
        dataclasses.replace(view, gaze_model="nonsense")
    with pytest.raises(ValueError, match="was given with gaze_model='ray_cast'"):
        dataclasses.replace(view[0], view_distance=10.0)
    with pytest.raises(ValueError, match="needs a finite view_distance > 0"):
        dataclasses.replace(view, gaze_model="fixed_distance", view_distance=-1.0)


# ---------------------------------------------------------------------------
# Occupancy threshold
# ---------------------------------------------------------------------------


def test_support_follows_the_typed_min_occupancy(population: Any) -> None:
    """spatial_support is occupancy > 0 and >= min_occupancy, whatever is recorded."""
    occupancy = np.asarray(population.occupancy)
    threshold = float(np.median(occupancy[occupancy > 0]))
    plain = dataclasses.replace(population, analysis_settings=None, min_occupancy=0.0)
    gated = dataclasses.replace(
        population, analysis_settings=None, min_occupancy=threshold
    )
    finite = np.isfinite(np.asarray(population.firing_rates)).all(axis=0)
    np.testing.assert_array_equal(plain.spatial_support, (occupancy > 0) & finite)
    np.testing.assert_array_equal(
        gated.spatial_support, (occupancy > 0) & (occupancy >= threshold) & finite
    )
    assert gated.spatial_support.sum() < plain.spatial_support.sum()
    np.testing.assert_array_equal(gated[0].spatial_support, gated.spatial_support)
    assert gated[0].min_occupancy == threshold


def test_compute_records_the_threshold_it_applied(
    family: str, recording: dict[str, Any]
) -> None:
    if family == "directional":
        pytest.skip("directional rates have no occupancy threshold")
    result = FAMILIES[family](recording, min_occupancy=0.3)
    assert result.min_occupancy == 0.3
    assert result.analysis_settings["min_occupancy"] == 0.3
    assert result[1].min_occupancy == 0.3
    occupancy = np.asarray(result.occupancy)
    assert not result.spatial_support[(occupancy > 0) & (occupancy < 0.3)].any()


@pytest.mark.parametrize("min_occupancy", [-1.0, np.nan, np.inf, "abc", None])
def test_refuses_a_threshold_that_is_not_seconds(
    population: Any, min_occupancy: Any
) -> None:
    for result in (population, population[0]):
        with pytest.raises(ValueError, match="is not a finite number of seconds"):
            dataclasses.replace(result, min_occupancy=min_occupancy)


def test_refuses_a_threshold_that_disagrees_with_the_record(population: Any) -> None:
    settings = {**(population.analysis_settings or {}), "min_occupancy": 0.5}
    with pytest.raises(ValueError, match="disagrees with the recorded"):
        dataclasses.replace(population, analysis_settings=settings, min_occupancy=0.0)
    agreeing = dataclasses.replace(
        population, analysis_settings=settings, min_occupancy=0.5
    )
    assert agreeing.min_occupancy == 0.5


# ---------------------------------------------------------------------------
# Per-unit metadata ownership
# ---------------------------------------------------------------------------


def test_unit_table_edits_through_the_result_do_not_change_it(
    family: str, recording: dict[str, Any]
) -> None:
    """In-place DataFrame edits cannot break the table's alignment to units."""
    import pandas as pd

    table = pd.DataFrame({"unit_id": ["a", "b", "c"], "region": ["CA1", "CA1", "CA3"]})
    labelled = FAMILIES[family](recording, unit_ids=["a", "b", "c"])
    result = dataclasses.replace(labelled, unit_table=table)
    result.unit_table.drop(index=result.unit_table.index[0], inplace=True)
    result.unit_table.loc[:, "region"] = "V1"
    table.loc[:, "region"] = "PFC"
    assert result.unit_table["unit_id"].tolist() == ["a", "b", "c"]
    assert result.unit_table["region"].tolist() == ["CA1", "CA1", "CA3"]
    assert result.unit_table is not result.unit_table
    restored = dataclasses.replace(result)
    assert restored.unit_table["region"].tolist() == ["CA1", "CA1", "CA3"]


def test_spike_trains_unit_table_is_copied_on_read() -> None:
    import pandas as pd

    from neurospatial.encoding import SpikeTrains

    trains = SpikeTrains(
        [np.array([0.1]), np.array([0.2])],
        unit_ids=[7, 9],
        unit_table=pd.DataFrame({"unit_id": [7, 9], "depth": [10.0, 20.0]}),
    )
    trains.unit_table.drop(index=0, inplace=True)
    assert trains.unit_table["depth"].tolist() == [10.0, 20.0]


def test_spike_trains_metadata_reaches_every_family(
    family: str, recording: dict[str, Any]
) -> None:
    """Labels and unit_table on SpikeTrains reach the population result."""
    import pandas as pd

    from neurospatial.encoding import SpikeTrains

    table = pd.DataFrame({"unit_id": [4, 5, 6], "region": ["CA1", "CA3", "CA1"]})
    trains = SpikeTrains(recording["spike_times"], unit_ids=[4, 5, 6], unit_table=table)
    result = FAMILIES[family]({**recording, "spike_times": trains})
    assert result.unit_ids.tolist() == [4, 5, 6]
    assert result._unit_ids_generated is False
    pd.testing.assert_frame_equal(result.unit_table, trains.unit_table)


# ---------------------------------------------------------------------------
# Spike accounting
# ---------------------------------------------------------------------------


def _single(family: str, rec: dict[str, Any], spikes: np.ndarray, **kwargs: Any) -> Any:
    """Each family's single-unit result for one spike train."""
    from neurospatial.encoding import (
        compute_boundary_vector_rate,
        compute_directional_rate,
        compute_egocentric_rate,
        compute_spatial_rate,
        compute_view_rate,
    )

    t, pos, hd = rec["times"], rec["positions"], rec["headings"]
    if family == "spatial":
        return compute_spatial_rate(rec["env"], spikes, t, pos, **kwargs)
    if family == "directional":
        return compute_directional_rate(spikes, t, hd, **kwargs)
    if family == "view":
        return compute_view_rate(rec["env"], spikes, t, pos, hd, **kwargs)
    if family == "object_vector":
        return compute_egocentric_rate(
            rec["env"],
            spikes,
            t,
            pos,
            hd,
            rec["object_positions"],
            distance_range=OBJECT_RANGE,
            **kwargs,
        )
    return compute_boundary_vector_rate(
        rec["env"], spikes, t, pos, hd, direction_frame="egocentric", **kwargs
    )


@pytest.fixture(scope="module")
def spikes_with_known_exclusions() -> tuple[list[np.ndarray], list[int]]:
    """Spike trains with a known number of spikes outside epochs=[(10, 60)].

    Every excluded spike is at least 4 s from a window edge, so the expected
    count does not depend on where the sampled intervals start or stop.
    """
    rng = np.random.default_rng(7)
    trains, outside = [], []
    for n_in, n_before, n_after, n_untracked in (
        (40, 3, 5, 2),
        (25, 0, 0, 0),
        (0, 4, 1, 1),
    ):
        train = np.concatenate(
            [
                rng.uniform(20.0, 50.0, n_in),
                rng.uniform(0.5, 6.0, n_before),
                rng.uniform(64.0, 99.0, n_after),
                rng.uniform(101.0, 120.0, n_untracked),  # after tracking stops
            ]
        )
        trains.append(np.sort(train))
        outside.append(n_before + n_after + n_untracked)
    return trains, outside


def test_spikes_outside_the_analyzed_intervals_are_counted_per_unit(
    family: str,
    recording: dict[str, Any],
    spikes_with_known_exclusions: tuple[list[np.ndarray], list[int]],
) -> None:
    """Population, its units and single-unit calls report the same counts."""
    trains, expected = spikes_with_known_exclusions
    rec = {**recording, "spike_times": trains}
    population = FAMILIES[family](rec, epochs=[(10.0, 60.0)])
    np.testing.assert_array_equal(population.n_spikes_outside_intervals, expected)
    assert population.n_spikes_outside_intervals.dtype == np.int64
    table = population.summary_table()
    assert table["n_spikes_outside_intervals"].dtype == "Int64"
    assert table["n_spikes_outside_intervals"].tolist() == expected
    for i, train in enumerate(trains):
        single = _single(family, rec, train, epochs=[(10.0, 60.0)])
        assert single.n_spikes_outside_intervals == expected[i]
        assert population[i].n_spikes_outside_intervals == expected[i]
        single_table = single.summary_table()
        assert single_table.columns.tolist() == table.columns.tolist()
        assert single_table["n_spikes_outside_intervals"].tolist() == [expected[i]]


def test_accounting_does_not_vary_settings_between_population_and_units(
    family: str,
    recording: dict[str, Any],
    spikes_with_known_exclusions: tuple[list[np.ndarray], list[int]],
) -> None:
    """Counts are per-unit fields; analysis_settings stay shared and identical."""
    trains, _expected = spikes_with_known_exclusions
    population = FAMILIES[family]({**recording, "spike_times": trains})
    assert "n_spikes_outside_intervals" not in population.analysis_settings
    for unit in population:
        assert unit.analysis_settings == population.analysis_settings


def test_unknown_accounting_is_missing_not_zero(population: Any) -> None:
    unknown = dataclasses.replace(population, n_spikes_outside_intervals=None)
    column = unknown.summary_table()["n_spikes_outside_intervals"]
    assert column.isna().all()
    assert unknown[0].n_spikes_outside_intervals is None
    assert unknown[0].summary_table()["n_spikes_outside_intervals"].isna().all()


@pytest.mark.parametrize(
    "counts", [np.array([1, 2]), np.array([1.5, 0.0, 2.0]), np.array([-1, 0, 0])]
)
def test_refuses_counts_that_are_not_one_per_unit(population: Any, counts: Any) -> None:
    with pytest.raises(ValueError, match="is not one spike count >= 0 per unit"):
        dataclasses.replace(population, n_spikes_outside_intervals=counts)
