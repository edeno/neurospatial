"""Shared track geometry preserves spatial gaps and probability support."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._tracks import (
    _checked_track,
    _project,
    _projection,
    track_position,
    track_segments,
)
from neurospatial.environment import _track_extents


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
    extents = _track_extents(track)
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


@pytest.mark.parametrize(
    "env_metadata, track_metadata, accepted",
    [
        ({}, {}, True),
        (
            {"units": "cm", "frame": "arena"},
            {"units": "centimeters", "frame": "arena"},
            True,
        ),
        ({"units": "cm"}, {"units": "m"}, False),
        ({"frame": "arena"}, {"frame": "camera"}, False),
        ({"units": "cm"}, {}, False),
        ({}, {"units": "cm"}, False),
        ({"frame": "arena"}, {}, False),
        ({}, {"frame": "arena"}, False),
    ],
)
def test_projection_requires_corresponding_units_and_frames(
    env_metadata, track_metadata, accepted
):
    plane = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0), np.arange(2.0)),
        **env_metadata,
    )
    track = Environment.linear_track(
        endpoints=[[0.0, 0.5], [2.0, 0.5]], bin_size=1, **track_metadata
    )
    if accepted:
        assert _checked_track(plane, track) is track
    else:
        with pytest.raises(ValueError, match="never silently rescaled or transformed"):
            _checked_track(plane, track)


def test_projection_checks_both_coordinate_systems():
    from types import SimpleNamespace

    from neurospatial.geometry._coordinates import (
        CartesianSystem,
        EgocentricPolarSystem,
    )

    plane = SimpleNamespace(
        is_linearized_track=False, n_dims=2, coordinate_system=CartesianSystem()
    )
    track = SimpleNamespace(
        is_linearized_track=True,
        n_dims=2,
        coordinate_system=EgocentricPolarSystem(circular=False),
    )
    with pytest.raises(TypeError, match="Cartesian"):
        _checked_track(plane, track)
