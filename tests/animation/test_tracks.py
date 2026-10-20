"""Shared track geometry preserves spatial gaps and probability support."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._tracks import (
    _checked_track,
    _project,
    _projection,
    track_extents,
    track_position,
    track_segments,
)


def test_projection_preserves_mass_and_missing_support():
    plane = Environment.from_grid_mask(
        np.ones((3, 2), dtype=bool),
        grid_edges=(np.arange(4.0), np.arange(3.0)),
    )
    track = Environment.linear_track(endpoints=[[0.0, 0.5], [3.0, 0.5]], bin_size=1.0)
    assert _checked_track(plane, track) is track
    projection = _projection(plane, track)
    rows = np.array([[0.1, 0.2, 0.0, np.nan, 0.3, 0.4], [np.nan] * 6])
    mapped = _project(rows, projection)
    assert np.nansum(mapped[0]) == pytest.approx(1.0)
    assert np.isnan(mapped[1]).all()
    # True zero observations remain distinct from unsupported bins.
    np.testing.assert_allclose(mapped[0], [0.3, 0.0, 0.7])


def test_track_coordinates_retain_segment_spacing_and_missing_positions():
    track = Environment.linear_track(
        node_positions=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]],
        bin_size=1.0,
        edge_spacing=2.0,
    )
    extents = track_extents(track)
    assert extents is not None
    starts, stops = extents
    np.testing.assert_allclose(starts, [0, 1, 2, 3, 6, 7, 8, 9])
    np.testing.assert_allclose(stops, starts + 1)
    segments = track_segments(track)
    assert len(segments) == 2
    assert segments[0][1:] == (0.0, 4.0)
    assert segments[1][1:] == (6.0, 10.0)
    np.testing.assert_allclose(
        track_position(track, np.array([[1.0, 0.0], [4.0, 3.0], [np.nan, np.nan]])),
        [1.0, 9.0, np.nan],
    )
