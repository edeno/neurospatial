"""Result objects carry their units, plot their populations and say what they are.

Each test runs a documented workflow on a seeded simulator session and checks
the result a scientist gets back: physical units in every table, a population
overview from ``plot()``, place fields that keep their map, a population
significance result, VTE verdicts that need head sweeps, and an environment
repr that shows its scale.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt

from neurospatial import encoding
from neurospatial.simulation import (
    open_field_session,
    tmaze_alternation_session,
)


@pytest.fixture(scope="module")
def open_field():
    """A 60 s open-field session with eight place cells."""
    return open_field_session(duration=60.0, n_place_cells=8, rng=0)


@pytest.fixture(scope="module")
def open_field_rates(open_field):
    s = open_field
    return encoding.compute_spatial_rates(s.env, s.spike_times, s.times, s.positions)


@pytest.fixture(scope="module")
def tmaze():
    return tmaze_alternation_session(rng=0)


@pytest.fixture(scope="module")
def tmaze_trials(tmaze):
    from neurospatial.behavior import detect_trials

    return detect_trials(
        tmaze.env,
        tmaze.times,
        tmaze.positions,
        regions=["start", "well_left", "well_right"],
    )


def _colorbar_labels(ax) -> list[str]:
    return [other.get_ylabel() for other in ax.figure.axes if other is not ax]


def test_population_plot_without_idx(open_field, open_field_rates):
    """PSTH, tuning and directional populations draw a heatmap in Hz.

    Rows are the units, sorted so each row's peak bin is no earlier than the
    row above; spatial maps, which are 2-D, raise the shared error instead.
    """
    from neurospatial.events import population_peri_event_histogram

    s = open_field
    rng = np.random.default_rng(1)
    headings = rng.uniform(-np.pi, np.pi, len(s.times))
    events = np.arange(5.0, 55.0, 5.0)
    results = {
        "psth": population_peri_event_histogram(
            s.spike_times, events, (-1.0, 1.0), dt=0.1
        ),
        "tuning": encoding.compute_tuning_curves(
            s.spike_times, s.times, s.positions[:, 0], bin_size=10.0
        ),
        "directional": encoding.compute_directional_rates(
            s.spike_times, s.times, headings
        ),
    }
    for name, result in results.items():
        ax = result.plot()
        assert any("Hz" in label for label in _colorbar_labels(ax)), name
        image = ax.collections[0].get_array()
        rates = np.ma.filled(np.ma.asarray(image), np.nan).reshape(len(result), -1)
        filled = np.where(np.isnan(rates), -np.inf, rates)
        # Silent units (no rate above zero) have no peak and come last.
        peaks = np.where(
            (filled > 0).any(axis=1), np.argmax(filled, axis=1), rates.shape[1]
        )
        assert np.all(np.diff(peaks) >= 0), name
        # Every unit is drawn: the rows are a permutation of the rates.
        drawn = np.sort(np.nan_to_num(rates, nan=-1.0), axis=0)
        expected = np.asarray(
            result.firing_rates if name == "psth" else result._supported_rates(),
            dtype=float,
        )
        np.testing.assert_allclose(
            drawn, np.sort(np.nan_to_num(expected, nan=-1.0), axis=0)
        )
        plt.close("all")

    with pytest.raises(ValueError, match=r"requires a unit index(.|\n)*Fix:"):
        open_field_rates.plot()


def test_population_mean_psth_title_names_the_population(open_field):
    from neurospatial.events import population_peri_event_histogram

    s = open_field
    result = population_peri_event_histogram(
        s.spike_times, np.arange(5.0, 55.0, 5.0), (-1.0, 1.0), dt=0.1
    )
    ax = result.plot(kind="mean")
    assert ax.get_title().startswith(f"Population mean of {len(result)} units")
    plt.close("all")


def test_detect_place_fields_accepts_rate_result(open_field, open_field_rates):
    """A rate result gives its own bandwidth and unit to the fields."""
    s, rates = open_field, open_field_rates
    single = rates[3]
    fields = encoding.detect_place_fields(s.env, single)
    assert fields.bandwidth == rates.bandwidth > 0
    assert fields.unit_id == rates.unit_ids[3]

    by_array = encoding.detect_place_fields(
        s.env, np.asarray(single.firing_rate), bandwidth=rates.bandwidth
    )
    assert [f.tolist() for f in fields] == [f.tolist() for f in by_array]

    population = [encoding.detect_place_fields(s.env, unit) for unit in rates]
    assert [f.unit_id for f in population] == list(rates.unit_ids)
    assert [f.tolist() for f in population[3]] == [f.tolist() for f in fields]
    # One map per call: a population is refused with the loop to write.
    with pytest.raises(TypeError, match=r"for unit in rates"):
        encoding.detect_place_fields(s.env, rates)

    table = fields.to_dataframe()
    assert {"unit_id", "area", "peak_rate"} <= set(table.columns)
    assert (table["unit_id"] == rates.unit_ids[3]).all()
    for row, bins in zip(table.itertuples(), fields, strict=True):
        assert row.area == pytest.approx(s.env.bin_sizes[bins].sum())
        assert row.peak_rate == pytest.approx(np.nanmax(single.firing_rate[bins]))
    assert table.attrs["units"]["area"] == f"{s.env.units}^2"
    assert fields.summary_table().index.tolist() == [rates.unit_ids[3]]
    ax = fields.plot()
    plt.close(ax.figure)


def test_place_fields_plot_on_a_track_marks_linear_positions(tmaze):
    """On a 2-D-embedded track the markers sit on the linearized rate curve."""
    s = tmaze
    rates = encoding.compute_spatial_rates(
        s.env, s.spike_times[:5], s.times, s.positions
    )
    fields = next(
        f for f in (encoding.detect_place_fields(s.env, unit) for unit in rates) if f
    )
    ax = fields.plot()
    marks = ax.collections[-len(fields.fields) :]
    for bins, mark in zip(fields.fields, marks, strict=True):
        expected = np.c_[
            s.env.to_linear(s.env.bin_centers[bins]), fields.firing_rate[bins]
        ]
        np.testing.assert_allclose(np.asarray(mark.get_offsets()), expected)
    assert np.all(np.asarray(marks[0].get_offsets())[:, 1] > 0)
    plt.close(ax.figure)


def test_place_fields_tables_on_a_polar_environment():
    """A polar map's centroid is a field bin; the summary needs no centroid."""
    from neurospatial import Environment

    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 100.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
    )
    rate = np.zeros(env.n_bins)
    peak = int(np.argmin(np.linalg.norm(env.bin_centers - [15.0, -0.4], axis=1)))
    rate[peak] = 5.0
    rate[list(env.connectivity.neighbors(peak))] = 4.0  # a field larger than a bin
    fields = encoding.detect_place_fields(env, rate, bandwidth=0.0)
    assert fields.n_fields == 1
    table = fields.to_dataframe()
    centroid = table[["centroid_distance", "centroid_angle"]].to_numpy()[0]
    assert any(np.allclose(centroid, env.bin_centers[b]) for b in fields.fields[0])
    assert table.attrs["units"]["centroid_angle"] == "rad"
    summary = fields.summary_table()
    assert summary["peak_rate"].iloc[0] == 5.0
    assert summary["largest_field_area"].iloc[0] == pytest.approx(
        env.bin_sizes[fields.fields[0]].sum()
    )


def test_detect_place_fields_array_needs_bandwidth(open_field, open_field_rates):
    with pytest.raises(ValueError, match=r"bandwidth(.|\n)*Fix:"):
        encoding.detect_place_fields(
            open_field.env, np.asarray(open_field_rates[0].firing_rate)
        )
    with pytest.raises(ValueError, match="estimated with bandwidth"):
        encoding.detect_place_fields(open_field.env, open_field_rates[0], bandwidth=1.0)


def test_binned_rates_give_unsmoothed_fields(open_field):
    """A ``method="binned"`` result records bandwidth 0: the peak bin is the floor."""
    s = open_field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        binned = encoding.compute_spatial_rates(
            s.env, s.spike_times, s.times, s.positions, method="binned"
        )
    fields = encoding.detect_place_fields(s.env, binned[0])
    assert fields.bandwidth == 0.0


def test_place_cell_significance_population_result(open_field):
    s = open_field
    # Units 1, 2, 4 and 5 have enough spikes for a known information bias.
    trains = [s.spike_times[i] for i in (1, 2, 4, 5)]
    ids = np.array([11, 3, 7, 20])
    result = encoding.place_cell_significance(
        s.env, trains, s.times, s.positions, unit_ids=ids, n_shuffles=49, rng=0
    )
    assert result.statistic == "spatial_information"
    assert result.statistic_units == "bits/spike"
    table = result.summary_table()
    assert table.index.name == "unit_id"
    assert table.index.tolist() == ids.tolist()
    # The Monte Carlo p-value of each unit, counted by hand from its null.
    null = result.null_scores
    defined = ~np.isnan(null)
    at_least = (null >= result.observed[:, None]) & defined
    expected = (at_least.sum(axis=1) + 1) / (defined.sum(axis=1) + 1)
    np.testing.assert_allclose(table["p_value"], expected)
    np.testing.assert_allclose(table["null_mean"], np.nanmean(null, axis=1))
    np.testing.assert_allclose(table["null_std"], np.nanstd(null, axis=1))
    np.testing.assert_allclose(
        table["z_score"],
        (result.observed - np.nanmean(null, axis=1)) / np.nanstd(null, axis=1),
    )
    # Each unit's test, by label, as the per-unit ShuffleTestResult.
    unit = result.sel(unit_id=7)
    assert unit.p_value == table.loc[7, "p_value"]
    assert unit.statistic_units == "bits/spike"
    assert "statistic_units=bits/spike" in repr(unit)
    # The screen is the shuffle verdict's rule on the same tests.
    debiased = result.observed - result.spatial_information_bias
    np.testing.assert_array_equal(
        result.classify(), (result.p_value < 0.05) & (debiased >= 0.5)
    )
    ax = result.plot(2)
    assert ax.get_xlabel() == "spatial_information (bits/spike)"
    plt.close(ax.figure)


def test_place_cell_significance_matches_the_dict_it_replaced(open_field):
    """p-values pinned from the dict return; classify() is the shuffle verdict.

    The values were recorded from ``place_cell_significance`` before it
    returned a result object, on the same seeded session and shuffles.
    """
    s = open_field
    result = encoding.place_cell_significance(
        s.env, s.spike_times, s.times, s.positions, n_shuffles=49, rng=0
    )
    with pytest.warns(UserWarning, match="information bias is unknown"):
        screen = result.classify()
    np.testing.assert_array_equal(
        result.p_value, [0.02, 0.02, 0.02, 0.12, 0.02, 0.02, 0.02, 0.02]
    )
    # Unit 0 is significant but has too few spikes to estimate its bias.
    with pytest.warns(UserWarning, match="information bias is unknown"):
        verdicts = [
            encoding.is_place_cell(
                s.env, train, s.times, s.positions, unit_id=unit, n_shuffles=49, rng=0
            )
            for unit, train in zip(result.unit_ids, s.spike_times, strict=True)
        ]
    assert screen.tolist() == verdicts
    assert screen.tolist() == [False, True, True, False, True, True, True, True]


def test_tables_carry_units(open_field, tmaze, tmaze_trials):
    from neurospatial.behavior import compute_vte_session
    from neurospatial.decoding import decode_session
    from neurospatial.events import peri_event_histogram

    s = open_field
    decoded = decode_session(s.env, s.spike_times, s.times, s.positions)
    # The simulated runs are straight, so no trial has a head sweep to rank.
    with pytest.warns(UserWarning, match="No variation in head sweep magnitude"):
        vte = compute_vte_session(
            tmaze.env,
            tmaze.times,
            tmaze.positions,
            trials=tmaze_trials,
            region="junction",
        )
    psth = peri_event_histogram(s.spike_times[0], np.arange(5.0, 55.0, 5.0), (-1, 1))
    tables = {
        "decode": decoded.to_dataframe(),
        "psth": psth.to_dataframe(),
        "psth summary": psth.summary_table(),
        "trials": tmaze_trials.intervals_table(),
        "vte trials": vte.trials_table(),
    }
    for name, table in tables.items():
        units = table.attrs.get("units")
        assert isinstance(units, dict), name
    assert tables["decode"].attrs["units"]["map_x"] == s.env.units
    assert tables["decode"].attrs["units"]["time"] == "s"
    assert tables["psth"].attrs["units"]["firing_rate"] == "Hz"
    assert tables["trials"].attrs["units"]["start_time"] == "s"
    assert tables["vte trials"].attrs["units"]["mean_speed"] == "cm/s"
    assert tables["vte trials"].attrs["units"]["head_sweep_magnitude"] == "rad"


def test_standalone_psth_unit_id_is_na():
    from neurospatial.events import peri_event_histogram

    psth = peri_event_histogram(np.array([0.8, 1.2]), np.array([1.0]), (-0.5, 0.5))
    # pd.NA as on the rate results, which pandas shows as <NA>, not None.
    assert psth.summary_table().index[0] is pd.NA
    assert all(label is pd.NA for label in psth.to_dataframe()["unit_id"])


def test_path_efficiency_not_above_one(tmaze, tmaze_trials):
    """Region goals are reached at their edge, up to the trial's excluded last step.

    A trial is ``[start, stop)``: its path ends at the last sample before the
    sample at ``stop``, where the animal enters the goal region. So the
    shortest path to the region's edge can exceed the traveled path by at
    most that excluded step, and efficiency by at most ``traveled /
    (traveled - last_step)``. Measured to the nearest region bin centre
    instead, the excess was half a 2 cm bin on every trial.
    """
    from neurospatial.behavior import compute_path_efficiency

    s, trials = tmaze, tmaze_trials
    goals = trials.intervals_table()["end_region"]
    result = compute_path_efficiency(s.env, s.times, s.positions, trials, goal=goals)
    traveled = np.asarray(result.traveled_length)
    shortest = np.asarray(result.shortest_length)
    # The step each trial leaves out: from its last sample to the one at stop.
    stop = np.searchsorted(s.times, trials.stop)
    assert np.all(s.times[stop] == trials.stop)
    last_step = np.linalg.norm(s.positions[stop] - s.positions[stop - 1], axis=1)

    assert np.all(shortest <= traveled + last_step + 1e-9)
    assert np.all(
        np.asarray(result.efficiency) <= traveled / (traveled - last_step) + 1e-12
    )
    half_bin = s.env.bin_spacing / 2
    assert np.max(shortest - traveled) < 0.1 * half_bin


def test_vte_zero_sweep_is_na(tmaze, tmaze_trials):
    """Straight runs have no head sweep: no trial is called VTE on speed alone."""
    from neurospatial.behavior import compute_vte_session

    with pytest.warns(UserWarning, match="is_vte is <NA>"):
        result = compute_vte_session(
            tmaze.env,
            tmaze.times,
            tmaze.positions,
            trials=tmaze_trials,
            region="junction",
        )
    table = result.trials_table()
    measured = table["status"] == "ok"
    assert measured.sum() > 1
    assert table["is_vte"].isna().all()
    assert result.n_vte_trials == 0
    assert result.mean_head_sweep == 0.0
    # Speeds still vary, so the speed z-scores are kept for inspection.
    assert table.loc[measured, "z_speed_inverse"].std() > 0


def test_environment_repr_shows_units(open_field):
    env = open_field.env
    text = repr(env)
    assert f"@ {env.bin_spacing:.3g} {env.units}" in text
    assert f"{env.n_bins} bins" in text


def test_lap_direction_on_a_linear_track():
    """Round trips on a track have a direction along it, not 'unknown'."""
    from neurospatial.behavior import detect_laps
    from neurospatial.simulation import linear_track_session

    s = linear_track_session(duration=60.0, n_place_cells=2, rng=0)
    x = s.positions[:, 0]
    start = int(np.argmin(x[:2000]))
    stop = start + int(np.argmax(x[start : start + 2000]))
    out = s.env.bin_at(s.positions[start:stop])
    for template, expected in ((out, "increasing"), (out[::-1], "decreasing")):
        laps = detect_laps(
            s.env, s.times, s.positions, method="reference", reference_lap=template
        )
        assert len(laps) > 2
        assert set(laps.direction) == {expected}
    filtered = detect_laps(
        s.env,
        s.times,
        s.positions,
        method="reference",
        reference_lap=out,
        direction="decreasing",
    )
    assert len(filtered) == 0


def test_cell_type_significance_repr_shows_attributes():
    from result_instances import RESULT_FACTORIES

    result = RESULT_FACTORIES["CellTypeSignificanceResult"]()
    assert repr(result) == (
        f"CellTypeSignificanceResult(n_units={result.n_units}, "
        f"n_shuffles={result.n_shuffles})"
    )
    assert isinstance(pd.Series(result.summary())["n_place"], (int, np.integer))
