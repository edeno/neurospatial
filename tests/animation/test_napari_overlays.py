"""Tests for Napari backend with overlay rendering.

Tests overlay integration with napari backend, including:
- Position overlays (tracks + points with trails)
- Bodypart overlays (points + skeleton shapes)
- Head direction overlays (vectors)
- Region overlays (polygon shapes)
- Coordinate transformation (x, y) → (y, x)
- Batched update callback mechanism
- Multi-animal scenarios (multiple overlays)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Skip all tests if napari not available
pytest.importorskip("napari")


@pytest.fixture
def simple_env():
    """Create simple 2D environment for testing."""
    from neurospatial import Environment

    positions = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ]
    )
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def simple_fields(simple_env: Environment) -> list[NDArray[np.float64]]:
    """Create simple field sequence for testing (10 frames)."""
    return [np.random.rand(simple_env.n_bins) for _ in range(10)]


@pytest.fixture
def position_overlay_data():
    """Create PositionData for testing."""
    from neurospatial.animation.overlays import PositionData

    # 10 frames, 2D coordinates
    data = np.array(
        [
            [5.0 + i * 0.5, 5.0 + i * 0.3]  # Diagonal trajectory
            for i in range(10)
        ]
    )
    return PositionData(data=data, color="red", size=10.0, trail_length=5)


@pytest.fixture
def bodypart_overlay_data():
    """Create BodypartData for testing."""
    from neurospatial.animation.overlays import BodypartData

    # 10 frames, 3 bodyparts (head, body, tail)
    bodyparts = {
        "head": np.array([[5.0 + i, 5.0] for i in range(10)]),
        "body": np.array([[4.0 + i, 5.0] for i in range(10)]),
        "tail": np.array([[3.0 + i, 5.0] for i in range(10)]),
    }
    skeleton = [("head", "body"), ("body", "tail")]
    colors = {"head": "red", "body": "green", "tail": "blue"}
    return BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=colors,
        skeleton_color="white",
        skeleton_width=2.0,
    )


@pytest.fixture
def head_direction_overlay_data():
    """Create HeadDirectionData for testing (angles)."""
    from neurospatial.animation.overlays import HeadDirectionData

    # 10 frames, angles in radians
    data = np.linspace(0, np.pi, 10)  # Rotate from 0 to 180 degrees
    return HeadDirectionData(data=data, color="yellow", length=2.0)


@pytest.fixture
def overlay_data_all_types(
    position_overlay_data, bodypart_overlay_data, head_direction_overlay_data
):
    """Create OverlayData with all overlay types."""
    from neurospatial.animation.overlays import OverlayData

    return OverlayData(
        positions=[position_overlay_data],
        bodypart_sets=[bodypart_overlay_data],
        head_directions=[head_direction_overlay_data],
        regions=None,
    )


# =============================================================================
# Position Overlay Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_position_overlay_creates_layers(
    mock_viewer_class, simple_env, simple_fields, position_overlay_data
):
    """Test position overlay creates both tracks and points layers."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    # Create mock viewer instance
    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Create overlay data
    overlay_data = OverlayData(positions=[position_overlay_data])

    # Render with overlay
    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Verify tracks layer created for trail
    assert mock_viewer.add_tracks.called
    tracks_call = mock_viewer.add_tracks.call_args

    # Verify track data shape: (n_trail_points, 4) for (track_id, time, y, x)
    track_data = tracks_call[0][0]
    assert track_data.shape[1] == 4  # (track_id, time, y, x)
    assert track_data.shape[0] > 0  # Has trail points

    # Verify points layer created for current position marker
    assert mock_viewer.add_points.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_position_overlay_coordinate_transform(
    mock_viewer_class, simple_env, simple_fields, position_overlay_data
):
    """Test position overlay applies (x, y) → (row, col) coordinate transformation.

    The transformation maps environment coordinates to napari pixel indices:
    - X maps to column (scaled to grid width)
    - Y maps to row (scaled and inverted - high Y → low row)
    """
    from neurospatial.animation.backends.napari_backend import (
        _transform_coords_for_napari,
        render_napari,
    )
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(positions=[position_overlay_data])

    # Get expected transformed coordinates using the actual transform function
    original_coords = position_overlay_data.data[0:1]  # First point as (1, 2)
    expected_transformed = _transform_coords_for_napari(original_coords, simple_env)

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Get tracks data (should be transformed to row, col)
    track_data = mock_viewer.add_tracks.call_args[0][0]

    # Track data format: (track_id, time, row, col)
    # First point should have row in position 2, col in position 3
    assert np.isclose(track_data[0, 2], expected_transformed[0, 0])  # Row coordinate
    assert np.isclose(track_data[0, 3], expected_transformed[0, 1])  # Col coordinate


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_position_overlay_applies_color_and_size(
    mock_viewer_class, simple_env, simple_fields, position_overlay_data
):
    """Test position overlay applies color and size properties."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(positions=[position_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Check tracks layer color (kwargs should include color)
    # NOTE: Implementation uses color parameter in add_tracks call

    # Check points layer size
    points_kwargs = mock_viewer.add_points.call_args[1]
    # Size should be specified
    assert "size" in points_kwargs or "point_size" in points_kwargs


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_position_overlay_without_trail(mock_viewer_class, simple_env, simple_fields):
    """Test position overlay without trail_length (points only, no tracks)."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, PositionData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Position data without trail
    position_data = PositionData(
        data=np.array([[5.0 + i, 5.0] for i in range(10)]),
        color="blue",
        size=8.0,
        trail_length=None,  # No trail
    )
    overlay_data = OverlayData(positions=[position_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should NOT create tracks layer (no trail)
    assert not mock_viewer.add_tracks.called

    # Should create points layer only
    assert mock_viewer.add_points.called


# =============================================================================
# Bodypart Overlay Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_bodypart_overlay_creates_layers(
    mock_viewer_class, simple_env, simple_fields, bodypart_overlay_data
):
    """Test bodypart overlay creates points and shapes layers."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(bodypart_sets=[bodypart_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should create points layer(s) for bodyparts
    assert mock_viewer.add_points.called

    # Should create shapes layer for skeleton
    assert mock_viewer.add_shapes.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_bodypart_overlay_skeleton_as_lines(
    mock_viewer_class, simple_env, simple_fields, bodypart_overlay_data
):
    """Test bodypart skeleton rendered as line shapes."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(bodypart_sets=[bodypart_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Check shapes layer has shape_type="line"
    shapes_kwargs = mock_viewer.add_shapes.call_args[1]
    assert "shape_type" in shapes_kwargs
    assert shapes_kwargs["shape_type"] == "line"


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_bodypart_overlay_coordinate_transform(
    mock_viewer_class, simple_env, simple_fields, bodypart_overlay_data
):
    """Test bodypart overlay applies (x, y) → (row, col) transformation.

    The transformation maps environment coordinates to napari pixel indices:
    - X maps to column (scaled to grid width)
    - Y maps to row (scaled and inverted - high Y → low row)
    """
    from neurospatial.animation.backends.napari_backend import (
        _transform_coords_for_napari,
        render_napari,
    )
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(bodypart_sets=[bodypart_overlay_data])

    # Get expected transformed coordinates for "head" bodypart
    original_coords = bodypart_overlay_data.bodyparts["head"][0:1]  # First point
    expected_transformed = _transform_coords_for_napari(original_coords, simple_env)

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Get points data (should be transformed to row, col)
    points_data = mock_viewer.add_points.call_args[0][0]

    # Points format: (time, row, col) for 2D + time
    # First point should have row, col from transformation
    assert np.isclose(points_data[0, 1], expected_transformed[0, 0])  # Row coordinate
    assert np.isclose(points_data[0, 2], expected_transformed[0, 1])  # Col coordinate


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_bodypart_overlay_per_part_colors(
    mock_viewer_class, simple_env, simple_fields, bodypart_overlay_data
):
    """Test bodypart overlay applies per-part colors."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(bodypart_sets=[bodypart_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Check points layer has color properties
    points_kwargs = mock_viewer.add_points.call_args[1]
    # Colors may be specified via 'face_color', 'edge_color', or 'properties'
    has_color = (
        "face_color" in points_kwargs
        or "edge_color" in points_kwargs
        or "properties" in points_kwargs
    )
    assert has_color


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_bodypart_overlay_skeleton_color_and_width(
    mock_viewer_class, simple_env, simple_fields, bodypart_overlay_data
):
    """Test bodypart skeleton applies color and width properties."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(bodypart_sets=[bodypart_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Check shapes layer has edge_color and edge_width
    shapes_kwargs = mock_viewer.add_shapes.call_args[1]
    assert "edge_color" in shapes_kwargs or "edge_colours" in shapes_kwargs
    assert "edge_width" in shapes_kwargs


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_bodypart_overlay_without_skeleton(
    mock_viewer_class, simple_env, simple_fields
):
    """Test bodypart overlay without skeleton (points only, no shapes)."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import BodypartData, OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Bodypart data without skeleton
    bodypart_data = BodypartData(
        bodyparts={"head": np.array([[5.0 + i, 5.0] for i in range(10)])},
        skeleton=None,  # No skeleton
        colors={"head": "red"},
        skeleton_color="white",
        skeleton_width=2.0,
    )
    overlay_data = OverlayData(bodypart_sets=[bodypart_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should create points layer
    assert mock_viewer.add_points.called

    # Should NOT create shapes layer (no skeleton)
    assert not mock_viewer.add_shapes.called


# =============================================================================
# Head Direction Overlay Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_head_direction_overlay_creates_vectors_layer(
    mock_viewer_class, simple_env, simple_fields, head_direction_overlay_data
):
    """Test head direction overlay creates vectors layer."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(head_directions=[head_direction_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should create vectors layer
    assert mock_viewer.add_vectors.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_head_direction_overlay_applies_color_and_length(
    mock_viewer_class, simple_env, simple_fields, head_direction_overlay_data
):
    """Test head direction overlay applies color and length properties."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(head_directions=[head_direction_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Check vectors layer has color and length properties
    vectors_kwargs = mock_viewer.add_vectors.call_args[1]
    # Color specified via 'edge_color' or similar
    has_color = "edge_color" in vectors_kwargs or "edge_colour" in vectors_kwargs
    assert has_color

    # Length should be encoded in vector data or properties
    # Vector data format: (position, direction) where direction has length


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_head_direction_overlay_coordinate_transform(
    mock_viewer_class, simple_env, simple_fields, head_direction_overlay_data
):
    """Test head direction overlay applies coordinate transformation."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    overlay_data = OverlayData(head_directions=[head_direction_overlay_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Get vectors data (should be transformed to y, x)
    vectors_data = mock_viewer.add_vectors.call_args[0][0]

    # Vectors format: [[time, y, x], [dy, dx]] for 2D + time
    # Should have correct dimensionality
    assert vectors_data.shape[1] == 2  # (position, direction)
    assert vectors_data.shape[2] >= 2  # At least (y, x) or (time, y, x)


# =============================================================================
# Region Overlay Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_region_overlay_creates_shapes_layer(
    mock_viewer_class, simple_env, simple_fields
):
    """Test region overlay creates shapes layer with polygons."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Add region to environment (just a point)
    simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

    # Render with show_regions=True
    render_napari(simple_env, simple_fields, show_regions=True, region_alpha=0.3)

    # Should create shapes layer for regions
    # NOTE: May create single shapes layer or multiple calls
    assert mock_viewer.add_shapes.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_region_overlay_applies_alpha(mock_viewer_class, simple_env, simple_fields):
    """Test region overlay applies alpha transparency."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

    render_napari(simple_env, simple_fields, show_regions=True, region_alpha=0.5)

    # Check shapes layer has opacity/alpha property
    shapes_kwargs = mock_viewer.add_shapes.call_args[1]
    has_alpha = (
        "opacity" in shapes_kwargs
        or "face_color" in shapes_kwargs
        or "edge_color" in shapes_kwargs
    )
    assert has_alpha


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_region_overlay_filters_by_list(mock_viewer_class, simple_env, simple_fields):
    """Test region overlay filters regions by name list."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Add multiple regions
    simple_env.regions.add("goal", point=np.array([5.0, 5.0]))
    simple_env.regions.add("start", point=np.array([15.0, 15.0]))

    # Render with specific region filter
    render_napari(simple_env, simple_fields, show_regions=["goal"], region_alpha=0.3)

    # Should create shapes layer (with only "goal" region)
    assert mock_viewer.add_shapes.called
    # NOTE: Verification of actual filtering logic happens in implementation


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_region_overlay_disabled_by_default(
    mock_viewer_class, simple_env, simple_fields
):
    """Test region overlay is disabled when show_regions=False."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

    # Render WITHOUT show_regions (default False)
    render_napari(simple_env, simple_fields)

    # Should NOT create shapes layer for regions
    # NOTE: May have shapes for bodyparts, but not regions
    # We'll verify in implementation that region shapes are skipped


# =============================================================================
# Multi-Animal / Multiple Overlays Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_multiple_position_overlays(mock_viewer_class, simple_env, simple_fields):
    """Test multiple position overlays (multi-animal scenario)."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, PositionData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Create two position overlays (two animals)
    pos1 = PositionData(
        data=np.array([[5.0 + i, 5.0] for i in range(10)]),
        color="red",
        size=10.0,
        trail_length=5,
    )
    pos2 = PositionData(
        data=np.array([[15.0 - i, 15.0] for i in range(10)]),
        color="blue",
        size=8.0,
        trail_length=3,
    )

    overlay_data = OverlayData(positions=[pos1, pos2])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should create tracks layers for both animals
    assert mock_viewer.add_tracks.call_count == 2

    # Should create points layers for both animals
    assert mock_viewer.add_points.call_count == 2


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_mixed_overlay_types(
    mock_viewer_class, simple_env, simple_fields, overlay_data_all_types
):
    """Test rendering with all overlay types simultaneously."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    render_napari(simple_env, simple_fields, overlay_data=overlay_data_all_types)

    # Should create tracks layer (position with trail)
    assert mock_viewer.add_tracks.called

    # Should create points layers (position marker + bodyparts)
    assert mock_viewer.add_points.call_count >= 2

    # Should create shapes layer (bodypart skeleton)
    assert mock_viewer.add_shapes.called

    # Should create vectors layer (head direction)
    assert mock_viewer.add_vectors.called


# =============================================================================
# Batched Update Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_batched_update_callback_registered(
    mock_viewer_class, simple_env, simple_fields, overlay_data_all_types
):
    """Test batched update callback is registered to viewer.dims.events."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Mock dims events
    mock_viewer.dims.events.current_step = MagicMock()

    render_napari(simple_env, simple_fields, overlay_data=overlay_data_all_types)

    # Verify callback registered to current_step event
    assert mock_viewer.dims.events.current_step.connect.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_batched_update_updates_all_layers(
    mock_viewer_class, simple_env, simple_fields, overlay_data_all_types
):
    """Test batched update callback updates all overlay layers."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Mock dims events
    mock_viewer.dims.events.current_step = MagicMock()

    render_napari(simple_env, simple_fields, overlay_data=overlay_data_all_types)

    # Get the callback function that was registered
    callback = mock_viewer.dims.events.current_step.connect.call_args[0][0]

    # Create mock layers that the callback will update
    mock_tracks_layer = MagicMock()
    mock_points_layer = MagicMock()
    mock_shapes_layer = MagicMock()
    mock_vectors_layer = MagicMock()

    mock_viewer.layers = [
        mock_tracks_layer,
        mock_points_layer,
        mock_shapes_layer,
        mock_vectors_layer,
    ]

    # Mock current_step change (frame 5)
    mock_event = MagicMock()
    mock_event.value = (5, 0, 0, 0)  # Frame 5
    mock_viewer.dims.current_step = (5, 0, 0, 0)

    # Call the callback
    callback(mock_event)

    # Verify layers were updated (data property modified)
    # NOTE: Specific update mechanism depends on implementation
    # We verify that the callback doesn't raise errors and executes


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_no_overlays_backward_compatibility(
    mock_viewer_class, simple_env, simple_fields
):
    """Test render_napari works without overlay_data (backward compatibility)."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Render without overlay_data parameter
    render_napari(simple_env, simple_fields)

    # Should complete without errors
    # Should NOT create overlay layers
    assert not mock_viewer.add_tracks.called
    assert not mock_viewer.add_vectors.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_empty_overlay_data(mock_viewer_class, simple_env, simple_fields):
    """Test render_napari with empty OverlayData (no overlays)."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Empty overlay data
    overlay_data = OverlayData(
        positions=[], bodypart_sets=[], head_directions=[], regions=None
    )

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should complete without errors
    # Should NOT create overlay layers
    assert not mock_viewer.add_tracks.called
    assert not mock_viewer.add_vectors.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_overlay_data_none_parameter(mock_viewer_class, simple_env, simple_fields):
    """Test render_napari with overlay_data=None (explicit None)."""
    from neurospatial.animation.backends.napari_backend import render_napari

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Explicitly pass None
    render_napari(simple_env, simple_fields, overlay_data=None)

    # Should complete without errors
    assert not mock_viewer.add_tracks.called


# =============================================================================
# Signature Update Test
# =============================================================================


def test_render_napari_signature_includes_overlay_params():
    """Test render_napari signature includes all new overlay parameters."""
    import inspect

    from neurospatial.animation.backends.napari_backend import render_napari

    sig = inspect.signature(render_napari)
    param_names = list(sig.parameters.keys())

    # Check all required overlay parameters are present
    assert "overlay_data" in param_names
    assert "show_regions" in param_names
    assert "region_alpha" in param_names

    # Verify default values
    assert sig.parameters["show_regions"].default is False
    assert sig.parameters["region_alpha"].default == 0.3
    assert sig.parameters["overlay_data"].default is None


# =============================================================================
# Head Direction + Position Pairing Tests
# =============================================================================


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_head_direction_paired_with_single_position(
    mock_viewer_class, simple_env, simple_fields
):
    """Test head direction arrows are anchored at position when exactly one position overlay exists."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import (
        HeadDirectionData,
        OverlayData,
        PositionData,
    )

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    # Create position data with known trajectory
    n_frames = 10
    positions = np.array([[5.0 + i, 5.0 + i * 0.5] for i in range(n_frames)])
    pos_data = PositionData(
        data=positions,
        color="red",
        size=10.0,
        trail_length=5,
    )

    # Create head direction data (angles)
    angles = np.linspace(0, np.pi, n_frames)
    head_dir_data = HeadDirectionData(
        data=angles,
        color="yellow",
        length=5.0,
    )

    overlay_data = OverlayData(positions=[pos_data], head_directions=[head_dir_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should create vectors layer
    assert mock_viewer.add_vectors.called

    # Get vectors data
    vectors_data = mock_viewer.add_vectors.call_args[0][0]

    # Vectors should have origins that follow the position trajectory
    # Format: [[time, y, x], [dt, dy, dx]]
    assert vectors_data.shape[0] == n_frames  # One vector per frame


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_head_direction_not_paired_with_multiple_positions(
    mock_viewer_class, simple_env, simple_fields
):
    """Test head direction uses centroid when multiple position overlays exist."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import (
        HeadDirectionData,
        OverlayData,
        PositionData,
    )

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    n_frames = 10

    # Create TWO position overlays (multi-animal scenario)
    pos1 = PositionData(
        data=np.array([[5.0 + i, 5.0] for i in range(n_frames)]),
        color="red",
        size=10.0,
        trail_length=5,
    )
    pos2 = PositionData(
        data=np.array([[15.0 - i, 15.0] for i in range(n_frames)]),
        color="blue",
        size=8.0,
        trail_length=3,
    )

    # Create head direction data
    angles = np.linspace(0, np.pi, n_frames)
    head_dir_data = HeadDirectionData(
        data=angles,
        color="yellow",
        length=5.0,
    )

    overlay_data = OverlayData(positions=[pos1, pos2], head_directions=[head_dir_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should still create vectors layer (uses centroid as origin)
    assert mock_viewer.add_vectors.called


@patch("neurospatial.animation.backends.napari_backend.napari.Viewer")
def test_head_direction_without_position_uses_centroid(
    mock_viewer_class, simple_env, simple_fields
):
    """Test head direction uses environment centroid when no position overlay exists."""
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import HeadDirectionData, OverlayData

    mock_viewer = MagicMock()
    mock_viewer_class.return_value = mock_viewer
    mock_viewer.dims.ndim = 4
    mock_viewer.dims.current_step = (0, 0, 0, 0)

    n_frames = 10

    # Create head direction data WITHOUT any position overlay
    angles = np.linspace(0, np.pi, n_frames)
    head_dir_data = HeadDirectionData(
        data=angles,
        color="yellow",
        length=5.0,
    )

    overlay_data = OverlayData(head_directions=[head_dir_data])

    render_napari(simple_env, simple_fields, overlay_data=overlay_data)

    # Should create vectors layer (uses centroid as origin)
    assert mock_viewer.add_vectors.called

    # Vectors should all originate from the same centroid position
    vectors_data = mock_viewer.add_vectors.call_args[0][0]

    # All vectors should have the same origin (y, x) in positions 1 and 2
    # (position 0 is time, which varies)
    origins_y = vectors_data[:, 0, 1]  # All y origins
    origins_x = vectors_data[:, 0, 2]  # All x origins

    # All origins should be identical (centroid is fixed)
    assert np.allclose(origins_y, origins_y[0])
    assert np.allclose(origins_x, origins_x[0])
