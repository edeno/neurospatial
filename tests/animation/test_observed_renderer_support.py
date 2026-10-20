"""Headless renderers never connect trajectories through unobserved intervals."""

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._parallel import (
    OverlayArtistManager,
    _render_head_direction_overlay_matplotlib,
    _render_position_overlay_matplotlib,
)
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)


@pytest.fixture
def env():
    return Environment.from_samples(
        np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]]), bin_size=2.0
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_trails_preserve_segments_on_both_sides_of_recording_break(env, persistent):
    positions = np.column_stack([np.arange(6.0), np.zeros(6)])
    positions[2] = np.nan
    data = PositionData(
        positions,
        "red",
        10.0,
        6,
        interval_valid=np.array([True, False, False, False, True]),
    )
    fig, ax = plt.subplots()
    try:
        if persistent:
            manager = OverlayArtistManager(
                ax, env, OverlayData(positions=[data]), False, 0.3
            )
            manager.initialize(0)
            manager.update_frame(5)
            trail = manager._position_trails[0]
        else:
            _render_position_overlay_matplotlib(ax, data, 5)
            trail = ax.collections[0]
        np.testing.assert_array_equal(
            trail.get_segments(), [positions[[0, 1]], positions[[4, 5]]]
        )
        np.testing.assert_allclose(
            trail.get_colors()[:, 3], np.array([1.0, 5.0]) / 5 * 0.7
        )
    finally:
        plt.close(fig)


def test_napari_trail_ids_keep_new_run_endpoints():
    from neurospatial.animation.backends.napari_backend import (
        _build_position_trail_tracks,
    )

    positions = np.column_stack([np.arange(6.0), np.zeros(6)])
    positions[2] = np.nan
    tracks = _build_position_trail_tracks(
        positions, np.array([True, False, False, False, True])
    )
    np.testing.assert_array_equal(tracks[:, 1], [0, 1, 3, 4, 5])
    assert tracks[0, 0] == tracks[1, 0]
    assert tracks[2, 0] != tracks[3, 0]
    assert tracks[3, 0] == tracks[4, 0]
    np.testing.assert_array_equal(tracks[-2:, 2:], positions[-2:])


def test_napari_pose_filters_missing_points_and_matching_feature_labels(
    env, monkeypatch
):
    from neurospatial.animation.backends import napari_backend

    monkeypatch.setattr(
        napari_backend, "screen_point_options", lambda size: {"size": size}
    )

    body = BodypartData(
        {
            "nose": np.array([[1.0, 1.0], [np.nan, np.nan]]),
            "body": np.array([[2.0, 2.0], [3.0, 3.0]]),
        },
        None,
        {"nose": "red", "body": "blue"},
    )
    viewer = MagicMock()
    napari_backend._render_bodypart_overlay(viewer, body, env)
    points = viewer.add_points.call_args.args[0]
    assert points.shape == (3, 3)
    assert np.isfinite(points).all()
    assert viewer.add_points.call_args.kwargs["features"]["bodypart"] == [
        "nose",
        "body",
        "body",
    ]
    assert viewer.add_points.call_args.kwargs["face_color"] == ["red", "blue", "blue"]


@pytest.mark.parametrize("persistent", [False, True])
def test_missing_position_hides_heading_but_absent_anchor_uses_center(env, persistent):
    positions = PositionData(np.array([[np.nan, np.nan]]), "red", 10.0, None)
    heading = HeadDirectionData(np.array([0.0]), "red", 1.0)
    ax = MagicMock()
    if persistent:
        manager = OverlayArtistManager(ax, env, OverlayData(), False, 0.3)
        assert manager._create_head_direction_quiver(heading, 0, positions) is None
        ax.quiver.assert_not_called()
        manager._create_head_direction_quiver(heading, 0, None)
    else:
        _render_head_direction_overlay_matplotlib(ax, heading, 0, env, positions)
        ax.quiver.assert_not_called()
        _render_head_direction_overlay_matplotlib(ax, heading, 0, env)
    ax.quiver.assert_called_once()
    np.testing.assert_array_equal(
        ax.quiver.call_args.args[:2], env.bin_centers.mean(axis=0)
    )


def test_napari_heading_omits_missing_position_anchor(env):
    from neurospatial.animation.backends.napari_backend import (
        _build_head_direction_tracks,
    )

    positions = PositionData(
        np.array([[np.nan, np.nan], [2.0, 2.0]]), "red", 10.0, None
    )
    heading = HeadDirectionData(np.array([0.0, 0.0]), "red", 1.0)
    tracks = _build_head_direction_tracks(heading, env, positions)
    np.testing.assert_array_equal(tracks[:, 0], [1, 1])
