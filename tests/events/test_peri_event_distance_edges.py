"""Bin edges of the peri-event distance histogram."""

from __future__ import annotations

import numpy as np

from neurospatial.events import peri_event_distance_histogram


def test_window_stop_is_left_out():
    """Bins are half-open: the window's upper distance belongs to no bin.

    The animal moves 1 cm per 0.5 s sample from the event, stops for 1 s at
    exactly 5 cm (the window stop) and moves on. Each 1 cm bin gets the 0.5 s
    spent crossing it; the stop at 5 cm gets no bin. Spikes at 2.0 s (4 cm)
    count; spikes at 2.5, 3.0 and 3.25 s, all at 5 cm, do not.
    """
    times = np.arange(0.0, 6.0, 0.5)
    positions = np.array([0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9], dtype=float)[:, None]
    result = peri_event_distance_histogram(
        np.array([2.0, 2.5, 3.0, 3.25]),
        np.array([0.0]),
        times,
        positions,
        (0.0, 5.0),
        bin_size=1.0,
    )
    np.testing.assert_array_equal(result.occupancy, [[0.5, 0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(result.spike_counts, [[0, 0, 0, 0, 1]])


def test_linear_matches_old_time_per_bin():
    """Time per distance bin equals the values the histogram gave before.

    The expected seconds were computed by the histogram's earlier dedicated
    time-per-bin code on these samples, which include a zero-length step, a
    step crossing every bin from beyond both edges, a stop exactly at the
    upper edge and a sample on an inner edge.
    """
    from neurospatial._intervals import linear_allocated_occupancy

    times = np.array(
        [
            0.0263,
            0.0813,
            0.2435,
            0.3641,
            0.392,
            0.4843,
            0.5853,
            0.6257,
            0.7752,
            0.8068,
            0.8912,
            0.9994,
        ]
    )
    distance = np.array(
        [
            2.364,
            8.298,
            14.038,
            14.038,
            -3.2,
            10.645,
            -27.913,
            41.206,
            -13.943,
            20.0,
            20.0,
            5.0,
        ]
    )
    edges = np.arange(-10.0, 21.0, 5.0)
    every = np.ones(times.size - 1, dtype=bool)
    expected = [
        0.034228700450015345,
        0.06074128891735495,
        0.10008670621267365,
        0.19038404352003488,
        0.3044284767935133,
        0.05719821477475555,
    ]
    np.testing.assert_allclose(
        linear_allocated_occupancy(distance, np.diff(times), edges, every),
        expected,
        rtol=0,
        atol=1e-12,
    )
    # Leaving out the zero-length step at 14.038 (0.1206 s in [10, 15)) and
    # the step across every bin (0.0404 s over 69.119 cm).
    some = every.copy()
    some[[2, 6]] = False
    expected_some = [
        0.03130620446483037,
        0.05781879293216998,
        0.09716421022748868,
        0.18746154753484992,
        0.18090598080832837,
        0.054275718789570576,
    ]
    np.testing.assert_allclose(
        linear_allocated_occupancy(distance, np.diff(times), edges, some),
        expected_some,
        rtol=0,
        atol=1e-12,
    )


def test_allocated_time_adds_up_to_the_counted_intervals():
    """Every counted second lands in some bin when the values stay inside.

    With every value inside ``[edges[0], edges[-1])``, each masked interval's
    time is split over bins only, so the bins sum to ``dt[mask].sum()``.
    """
    from neurospatial._intervals import linear_allocated_occupancy

    rng = np.random.default_rng(4)
    edges = np.linspace(-10.0, 10.0, 9)
    values = rng.uniform(-10.0, 10.0, 200)
    values[50:60] = values[50]  # a stop: zero-length steps
    values[80] = edges[3]  # a sample on an inner edge
    dt = rng.uniform(0.01, 0.1, values.size - 1)
    mask = rng.random(dt.size) < 0.7
    seconds = linear_allocated_occupancy(values, dt, edges, mask)
    np.testing.assert_allclose(seconds.sum(), dt[mask].sum(), rtol=1e-12)
    assert (seconds >= 0).all()
