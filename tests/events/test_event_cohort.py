"""PSTH exposure filtering preserves the identity of the events being averaged."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.events import (
    align_spikes_to_events,
    peri_event_histogram,
    population_peri_event_histogram,
)


def test_exposure_filtered_histograms_and_raster_share_original_event_cohort():
    # Keep input order and duplicate timestamps: they remain distinct events.
    events = np.array([0.1, 8.0, 2.0, 2.0, 4.9, 12.0, 7.5])
    spikes = np.array([1.8, 2.0, 2.3, 7.5, 8.0, 8.2])
    kwargs = {
        "window": (-0.5, 0.5),
        "bin_size": 0.25,
        "epochs": [[1.0, 4.0], [7.0, 9.0]],
        "spike_window": [[0.0, 5.0], [7.0, 10.0]],
    }
    single = peri_event_histogram(spikes, events, **kwargs)
    population = population_peri_event_histogram(
        [spikes, spikes + 0.05], events, unit_ids=["a", "b"], **kwargs
    )
    for result in (single, population, population.sel(unit_id="a")):
        assert_array_equal(result.event_indices, [1, 2, 3, 6])
        assert result.n_events == 4
        assert result.n_events_dropped == 3
        assert not result.event_indices.flags.writeable

    cohort = events[single.event_indices]
    aligned = align_spikes_to_events(spikes, cohort, kwargs["window"])
    edges = np.linspace(-0.5, 0.5, 5)
    counts = np.array([np.histogram(train[train < 0.5], edges)[0] for train in aligned])
    assert_array_equal(single.histogram, counts.mean(axis=0))
    assert_array_equal(population[0].histogram, single.histogram)


def test_unfiltered_event_indices_are_original_positions():
    result = peri_event_histogram(
        np.array([1.0, 2.0]), np.array([2.0, 1.0]), (-0.2, 0.2)
    )
    assert_array_equal(result.event_indices, [0, 1])


@pytest.mark.parametrize("indices", [[0], [1, 1], [-1, 1], [0, 4], [0.0, 1.0]])
def test_event_indices_reject_incompatible_cohort_metadata(indices):
    from dataclasses import replace

    result = peri_event_histogram(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), (-0.2, 0.2)
    )
    with pytest.raises(ValueError, match="event_indices"):
        replace(result, event_indices=np.asarray(indices))
