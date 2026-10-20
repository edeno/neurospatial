"""A combined raster/PSTH must show one retained cohort and bin range."""

from dataclasses import fields, replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import EventCollection
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial.events import (
    align_spikes_to_events,
    peri_event_histogram,
    plot_peri_event_response,
    population_peri_event_histogram,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def raster_rows(ax):
    """Read the spikes actually drawn, including silent event rows."""
    return [
        np.asarray([segment[0, 0] for segment in collection.get_segments()])
        for collection in ax.collections
        if isinstance(collection, EventCollection)
    ]


def test_filtered_gap_cohort_preserves_input_order_and_duplicate_events():
    events = np.array([0.1, 8.0, 2.0, 2.0, 4.9, 12.0, 7.5])
    spikes = np.array([8.2, 1.8, 2.0, 2.3, 7.5, 8.0])  # unsorted
    result = peri_event_histogram(
        spikes,
        events,
        (-0.5, 0.5),
        bin_size=0.25,
        epochs=[[1.0, 4.0], [7.0, 9.0]],
        spike_window=[[0.0, 5.0], [7.0, 10.0]],
    )

    raster_ax, histogram_ax = plot_peri_event_response(spikes, events, result)

    assert_array_equal(result.event_indices, [1, 2, 3, 6])
    expected = [[-0.5, 0.0, 0.2], [-0.2, 0.0, 0.3], [-0.2, 0.0, 0.3], [0.0]]
    rows = raster_rows(raster_ax)
    assert len(rows) == result.n_events == 4
    for row, expected_row in zip(rows, expected, strict=True):
        assert_allclose(row, expected_row)
    assert_array_equal(histogram_ax.lines[0].get_ydata(), result.firing_rate)
    assert raster_ax.get_ylim() == (-0.5, 3.5)
    assert raster_ax.get_shared_x_axes().joined(raster_ax, histogram_ax)


def test_exact_left_edge_is_included_and_right_edge_is_excluded():
    events = np.array([10.0, 20.0])
    spikes = np.concatenate(
        [
            [event - 0.5, event, np.nextafter(event + 0.5, -np.inf), event + 0.5]
            for event in events
        ]
    )
    result = peri_event_histogram(spikes, events, (-0.5, 0.5), bin_size=0.25)

    raster_ax, _ = plot_peri_event_response(spikes, events, result)

    for row in raster_rows(raster_ax):
        assert row.size == 3
        assert row[0] == -0.5
        assert row[-1] < 0.5
    assert result.histogram.sum() == 3
    # The low-level aligner uses the same half-open window as the histogram.
    assert len(align_spikes_to_events(spikes, events[:1], (-0.5, 0.5))[0]) == 3


def test_trailing_partial_bin_is_omitted_from_raster_and_axes():
    events = np.array([10.0, 20.0])
    spikes = np.concatenate(
        [event + np.array([-0.5, -0.2, 0.1, 0.4, 0.45]) for event in events]
    )
    result = peri_event_histogram(spikes, events, (-0.5, 0.5), bin_size=0.3)

    raster_ax, histogram_ax = plot_peri_event_response(spikes, events, result)

    assert [row.size for row in raster_rows(raster_ax)] == [3, 3]
    assert result.histogram.sum() == 3
    assert_allclose(raster_ax.get_xlim(), [-0.5, 0.4])
    assert_allclose(histogram_ax.get_xlim(), [-0.5, 0.4])


def test_partial_tail_exposure_still_uses_original_full_window():
    events = np.array([1.0, 2.0, 3.0])
    spikes = np.array([1.0, 2.0, 3.0])
    # The complete bins end at event + 0.4, but exposure needs event + 0.5.
    result = peri_event_histogram(
        spikes, events, (-0.5, 0.5), bin_size=0.3, spike_window=(0.0, 3.45)
    )

    raster_ax, _ = plot_peri_event_response(spikes, events, result)

    assert_array_equal(result.event_indices, [0, 1])
    assert [row.size for row in raster_rows(raster_ax)] == [1, 1]


def test_large_clock_membership_uses_absolute_bounds_before_subtraction():
    events = np.array([1e9 + 0.9, 1e9 + 10.9])
    spikes = np.concatenate([event + np.array([0.0, 0.2, 0.3]) for event in events])
    # A relative < 0.3 filter would incorrectly include the exact stop spike.
    assert (events[0] + 0.3) - events[0] < 0.3
    result = peri_event_histogram(spikes, events, (0.0, 0.3), bin_size=0.1)

    raster_ax, _ = plot_peri_event_response(spikes, events, result)

    assert [row.size for row in raster_rows(raster_ax)] == [2, 2]
    assert result.histogram.sum() == 2


def test_large_clock_selected_left_edge_remains_visible():
    events = np.array([1e9, 1e9 + 10])
    spikes = events - 0.1
    assert spikes[0] - events[0] < -0.1  # floating-point display rounding
    result = peri_event_histogram(spikes, events, (-0.1, 0.2), bin_size=0.1)

    raster_ax, _ = plot_peri_event_response(spikes, events, result)

    rows = raster_rows(raster_ax)
    assert [row.size for row in rows] == [1, 1]
    assert all(row[0] == raster_ax.get_xlim()[0] for row in rows)
    assert result.histogram.sum() == 1


def test_silent_unit_retains_every_event_row():
    spikes = np.array([])
    events = np.array([1.0, 2.0])
    result = peri_event_histogram(spikes, events, (-0.5, 0.5))

    raster_ax, histogram_ax = plot_peri_event_response(spikes, events, result)

    assert [row.size for row in raster_rows(raster_ax)] == [0, 0]
    assert_array_equal(histogram_ax.lines[0].get_ydata(), np.zeros(40))
    assert raster_ax.get_ylim() == (-0.5, 1.5)


def test_selected_unit_label_and_supplied_axes_are_preserved():
    from neurospatial.encoding import SpikeTrains

    events = np.array([1.0, 2.0])
    trains = SpikeTrains([np.array([1.1, 2.1]), np.array([0.8, 1.8])], unit_ids=[42, 7])
    population = population_peri_event_histogram(
        trains, events, (-0.5, 0.5), bin_size=0.25
    )
    result = population.sel(unit_id=7)
    fig, axes = plt.subplots(2, 1)

    raster_ax, histogram_ax = plot_peri_event_response(
        trains.sel(unit_id=7),
        events,
        result,
        axes=axes,
        show_sem=False,
        as_rate=False,
        color="purple",
        raster_color="orange",
    )

    assert raster_ax is axes[0]
    assert histogram_ax is axes[1]
    assert raster_ax.figure is fig
    assert "Unit 7:" in raster_ax.get_title()
    for row in raster_rows(raster_ax):
        assert_allclose(row, [-0.2])
    assert_array_equal(histogram_ax.lines[0].get_ydata(), result.histogram)
    assert histogram_ax.lines[0].get_color() == "purple"
    assert histogram_ax.get_ylabel() == "Spike count"
    assert not histogram_ax.collections
    assert_allclose(raster_ax.collections[0].get_colors()[0], [1, 0.64705882, 0, 1])


def test_one_event_has_undefined_sem_but_draws_response():
    events = np.array([1.0])
    spikes = np.array([1.0])
    with pytest.warns(UserWarning, match="SEM|sem|one event|single event|1 event"):
        result = peri_event_histogram(spikes, events, (-0.5, 0.5), bin_size=0.25)

    raster_ax, histogram_ax = plot_peri_event_response(spikes, events, result)

    assert len(raster_rows(raster_ax)) == 1
    assert_array_equal(histogram_ax.lines[0].get_ydata(), result.firing_rate)
    assert np.all(np.isnan(result.sem))


def test_unknown_cohort_rejected_instead_of_guessing():
    events = np.array([1.0, 2.0])
    spikes = np.array([1.0])
    result = replace(
        peri_event_histogram(spikes, events, (-0.5, 0.5)), event_indices=None
    )

    with pytest.raises(ValueError, match="event_indices is unknown"):
        plot_peri_event_response(spikes, events, result)


def test_filtered_event_array_rejected_even_when_it_matches_retained_count():
    events = np.array([0.1, 1.0, 2.0])
    spikes = np.array([1.0, 2.0])
    result = peri_event_histogram(spikes, events, (-0.5, 0.5), spike_window=(0, 3))

    with pytest.raises(ValueError, match="complete original event array"):
        plot_peri_event_response(spikes, events[result.event_indices], result)


@pytest.mark.parametrize("argument", ["spike_times", "event_times"])
@pytest.mark.parametrize(
    "invalid",
    [np.array([[1.0, 2.0]]), np.array([1.0, np.nan]), np.array([1.0, np.inf])],
)
def test_nonfinite_or_multidimensional_inputs_rejected(argument, invalid):
    events = np.array([1.0, 2.0])
    spikes = np.array([1.0])
    result = peri_event_histogram(spikes, events, (-0.5, 0.5))
    inputs = {"spike_times": spikes, "event_times": events}
    inputs[argument] = invalid

    with pytest.raises(ValueError, match=f"{argument} must be a finite 1-D"):
        plot_peri_event_response(**inputs, result=result)


def test_population_result_requires_explicit_unit_selection():
    events = np.array([1.0, 2.0])
    spikes = np.array([1.0])
    population = population_peri_event_histogram([spikes, spikes], events, (-0.5, 0.5))

    with pytest.raises(TypeError, match=r"single-unit.*Select a population unit"):
        plot_peri_event_response(spikes, events, population)


@pytest.mark.parametrize("invalid", [(), [None, None], ["raster", "histogram"], None])
def test_invalid_axes_pair_rejected(invalid):
    events = np.array([1.0, 2.0])
    spikes = np.array([1.0])
    result = peri_event_histogram(spikes, events, (-0.5, 0.5))
    if invalid is None:
        _, ax = plt.subplots()
        invalid = (ax, ax)

    with pytest.raises(ValueError, match="two distinct Matplotlib Axes"):
        plot_peri_event_response(spikes, events, result, axes=invalid)


def test_inconsistent_bin_metadata_rejected():
    events = np.array([1.0, 2.0])
    spikes = np.array([1.0])
    result = peri_event_histogram(spikes, events, (-0.5, 0.5))
    altered = replace(result, bin_centers=result.bin_centers + 0.001)

    with pytest.raises(ValueError, match="bin_centers does not match"):
        plot_peri_event_response(spikes, events, altered)


def test_no_complete_bin_rejected_with_actionable_error():
    events = np.array([1.0, 2.0])
    spikes = np.array([1.0])
    result = peri_event_histogram(spikes, events, (0.0, 0.1), bin_size=0.25)

    with pytest.raises(ValueError, match="at least one complete bin"):
        plot_peri_event_response(spikes, events, result)


def test_plotting_does_not_retain_or_mutate_raw_inputs_or_result_arrays():
    events = np.array([2.0, 1.0])
    spikes = np.array([2.1, 1.1])
    result = peri_event_histogram(spikes, events, (-0.5, 0.5))
    before_events, before_spikes = events.copy(), spikes.copy()
    before_arrays = {
        item.name: getattr(result, item.name).copy()
        for item in fields(result)
        if isinstance(getattr(result, item.name), np.ndarray)
    }

    plot_peri_event_response(spikes, events, result)

    assert_array_equal(events, before_events)
    assert_array_equal(spikes, before_spikes)
    for name, values in before_arrays.items():
        assert_array_equal(getattr(result, name), values)
        assert not getattr(result, name).flags.writeable
    assert {item.name for item in fields(result)} == {
        "bin_centers",
        "histogram",
        "sem",
        "n_events",
        "window",
        "bin_size",
        "unit_id",
        "n_events_dropped",
        "event_indices",
        "firing_rate",
    }


def test_optional_pynapple_cohort_and_unit_identity_are_preserved():
    nap = pytest.importorskip("pynapple")
    events = np.array([0.1, 1.0, 2.0, 4.0])
    spikes = np.array([1.1, 2.1, 4.1])
    group = nap.TsGroup({42: nap.Ts(t=spikes), 7: nap.Ts(t=np.array([1.2, 2.2]))})
    epochs = nap.IntervalSet(start=[0.0], end=[3.0])
    population = population_peri_event_histogram(
        group, events, (-0.5, 0.5), epochs=epochs
    )
    result = population.sel(unit_id=42)

    raster_ax, _ = plot_peri_event_response(group[42].index.to_numpy(), events, result)

    assert_array_equal(result.event_indices, [1, 2])
    assert "Unit 42:" in raster_ax.get_title()
    for row in raster_rows(raster_ax):
        assert_allclose(row, [0.1])
