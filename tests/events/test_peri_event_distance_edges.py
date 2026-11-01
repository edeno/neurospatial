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
