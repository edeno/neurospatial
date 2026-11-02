"""detect_laps needs a loop; back-and-forth runs are trials."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from neurospatial.behavior.segmentation import detect_laps


def test_detect_laps_refuses_linear_track(linear_track, continuous_lap_track):
    env, times, positions = linear_track.env, linear_track.times, linear_track.positions
    assert nx.cycle_basis(env.connectivity) == []
    with pytest.raises(ValueError, match=r"has no loop[\s\S]*detect_trials"):
        detect_laps(env, times, positions)

    # A template round trip is the user's own definition of a lap, so it
    # still segments the runs: out to the far end and back to the start.
    x = positions[:, 0]
    far = int(np.argmax(x > x.max() - 5.0))
    back = far + int(np.argmax(x[far:] < x[0] + 5.0))
    first = env.bin_at(positions[: back + 1])
    trips = detect_laps(env, times, positions, method="reference", reference_lap=first)
    assert len(trips) > 1
    assert np.all(trips.overlap_score >= 0.8)

    # On a loop, laps are still detected, one per 40 s circuit, with a direction.
    loop = continuous_lap_track
    assert nx.cycle_basis(loop.env.connectivity) != []
    laps = detect_laps(loop.env, loop.times, loop.positions)
    assert len(laps) >= 4
    assert np.all(np.isin(laps.direction, ["clockwise", "counter-clockwise"]))
