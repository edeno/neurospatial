"""Tests for EventOverlay and SpikeOverlay animation overlay classes.

This module tests the EventOverlay dataclass, its validation logic,
conversion to EventData, and integration with the animation pipeline.

Tests are organized by milestone tasks from TASKS.md.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy.testing import assert_array_equal

# =============================================================================
# Milestone 1: Core Data Structures Tests
# =============================================================================


class TestEventOverlayBasicCreation:
    """Test basic EventOverlay creation with single event type."""

    def test_trajectory_mode_basic_creation(self):
        """Test creating EventOverlay with trajectory interpolation mode (Mode B)."""
        from neurospatial.animation.overlays import EventOverlay

        # Create trajectory data
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        position_times = np.array([0.0, 1.0, 2.0])
        event_times = np.array([0.5, 1.5])

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
        )

        # event_times is normalized to dict format in __post_init__
        assert isinstance(overlay.event_times, dict)
        assert "event" in overlay.event_times
        assert_array_equal(overlay.event_times["event"], event_times)
        assert_array_equal(overlay.positions, positions)
        assert_array_equal(overlay.position_times, position_times)
        assert overlay.event_positions is None
        assert overlay.interp == "linear"

    def test_explicit_positions_mode_basic_creation(self):
        """Test creating EventOverlay with explicit positions mode (Mode A)."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5, 2.5])
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        # Both are normalized to dict format in __post_init__
        assert isinstance(overlay.event_times, dict)
        assert "event" in overlay.event_times
        assert_array_equal(overlay.event_times["event"], event_times)
        assert isinstance(overlay.event_positions, dict)
        assert "event" in overlay.event_positions
        assert_array_equal(overlay.event_positions["event"], event_positions)
        assert overlay.positions is None
        assert overlay.position_times is None

    def test_default_styling_values(self):
        """Test that default styling values are correct."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5])
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        assert overlay.colors is None  # Auto-assign from tab10
        assert overlay.size == 8.0
        assert overlay.decay_frames is None  # Instant (no decay)
        assert overlay.markers is None  # Default 'o'
        assert overlay.border_color == "white"
        assert overlay.border_width == 0.5

    def test_custom_styling_values(self):
        """Test EventOverlay with custom styling."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5])
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            colors="red",
            size=12.0,
            decay_frames=5,
            markers="s",
            border_color="black",
            border_width=1.0,
        )

        assert overlay.colors == "red"
        assert overlay.size == 12.0
        assert overlay.decay_frames == 5
        assert overlay.markers == "s"
        assert overlay.border_color == "black"
        assert overlay.border_width == 1.0


class TestEventOverlayMultipleEventTypes:
    """Test EventOverlay with multiple event types (dict input)."""

    def test_dict_event_times_trajectory_mode(self):
        """Test EventOverlay with dict of event times in trajectory mode."""
        from neurospatial.animation.overlays import EventOverlay

        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        position_times = np.array([0.0, 1.0, 2.0])
        event_times = {
            "cell_001": np.array([0.5, 1.5]),
            "cell_002": np.array([0.3, 0.8, 1.2]),
        }

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
        )

        assert isinstance(overlay.event_times, dict)
        assert "cell_001" in overlay.event_times
        assert "cell_002" in overlay.event_times
        assert len(overlay.event_times["cell_001"]) == 2
        assert len(overlay.event_times["cell_002"]) == 3

    def test_dict_event_times_explicit_positions_mode(self):
        """Test EventOverlay with dict of event times and explicit positions."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {
            "reward": np.array([1.0, 2.0]),
            "punishment": np.array([1.5]),
        }
        event_positions = {
            "reward": np.array([[50.0, 25.0], [50.0, 25.0]]),  # Same location
            "punishment": np.array([[75.0, 50.0]]),
        }

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        assert isinstance(overlay.event_times, dict)
        assert isinstance(overlay.event_positions, dict)

    def test_dict_custom_colors_per_type(self):
        """Test EventOverlay with custom colors per event type."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {
            "lick": np.array([0.5]),
            "reward": np.array([1.0]),
        }
        event_positions = {
            "lick": np.array([[10.0, 20.0]]),
            "reward": np.array([[50.0, 25.0]]),
        }
        colors = {"lick": "cyan", "reward": "gold"}

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            colors=colors,
        )

        assert overlay.colors == colors
        assert overlay.colors["lick"] == "cyan"
        assert overlay.colors["reward"] == "gold"

    def test_dict_custom_markers_per_type(self):
        """Test EventOverlay with custom markers per event type."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {
            "lick": np.array([0.5]),
            "reward": np.array([1.0]),
        }
        event_positions = {
            "lick": np.array([[10.0, 20.0]]),
            "reward": np.array([[50.0, 25.0]]),
        }
        markers = {"lick": "o", "reward": "s"}

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            markers=markers,
        )

        assert overlay.markers == markers


class TestEventOverlayValidation:
    """Test EventOverlay __post_init__ validation."""

    def test_mutual_exclusion_both_modes_raises(self):
        """Test that providing both position modes raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5])
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        position_times = np.array([0.0, 1.0])
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0]])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                positions=positions,
                position_times=position_times,
                event_positions=event_positions,  # Both modes!
            )

        error_msg = str(exc_info.value).lower()
        # Should mention mutual exclusion
        assert "exclusive" in error_msg or "both" in error_msg

    def test_mutual_exclusion_neither_mode_raises(self):
        """Test that providing neither position mode raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                # No positions or event_positions!
            )

        error_msg = str(exc_info.value).lower()
        # Should mention missing position data
        assert "position" in error_msg

    def test_trajectory_mode_missing_position_times_raises(self):
        """Test that trajectory mode without position_times raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5])
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                positions=positions,
                # Missing position_times!
            )

        error_msg = str(exc_info.value).lower()
        assert "position_times" in error_msg

    def test_trajectory_mode_missing_positions_raises(self):
        """Test that trajectory mode without positions raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.5])
        position_times = np.array([0.0, 1.0])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                position_times=position_times,
                # Missing positions!
            )

        error_msg = str(exc_info.value).lower()
        assert "positions" in error_msg

    def test_event_times_not_1d_raises(self):
        """Test that non-1D event times array raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([[0.5, 1.5], [2.0, 2.5]])  # 2D!
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0]])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                event_positions=event_positions,
            )

        error_msg = str(exc_info.value).lower()
        assert "1d" in error_msg or "dimension" in error_msg

    def test_event_times_with_nan_raises(self):
        """Test that event times with NaN values raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, np.nan, 1.5])
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                event_positions=event_positions,
            )

        error_msg = str(exc_info.value).lower()
        assert "nan" in error_msg or "finite" in error_msg

    def test_event_times_with_inf_raises(self):
        """Test that event times with Inf values raises ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, np.inf, 1.5])
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                event_positions=event_positions,
            )

        error_msg = str(exc_info.value).lower()
        assert "inf" in error_msg or "finite" in error_msg

    def test_event_positions_shape_mismatch_raises(self):
        """Test that mismatched event_positions/event_times shapes raise ValueError."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.0, 1.5, 2.0])  # 4 events
        event_positions = np.array([[10.0, 20.0], [30.0, 40.0]])  # Only 2 positions

        with pytest.raises(ValueError) as exc_info:
            EventOverlay(
                event_times=event_times,
                event_positions=event_positions,
            )

        error_msg = str(exc_info.value).lower()
        assert "incompatible" in error_msg or "shape" in error_msg


class TestEventOverlayPositionBroadcast:
    """Test position broadcasting for explicit positions mode."""

    def test_single_position_broadcast_to_all_events(self):
        """Test that single position (1, n_dims) broadcasts to all events."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 1.0, 1.5, 2.0])  # 4 events
        event_positions = np.array([[50.0, 25.0]])  # Single position (1, 2)

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        # Should create without error - broadcast handled during convert_to_data
        # Normalized to dict format in __post_init__
        assert overlay.event_positions["event"].shape == (1, 2)
        assert len(overlay.event_times["event"]) == 4

    def test_single_position_broadcast_in_dict(self):
        """Test single position broadcast with dict event_times."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {
            "reward": np.array([1.0, 2.0, 3.0]),  # 3 events
        }
        event_positions = {
            "reward": np.array([[50.0, 25.0]]),  # Single position
        }

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        assert overlay.event_positions["reward"].shape == (1, 2)


class TestEventOverlayInterpolationModes:
    """Test interpolation modes for trajectory mode."""

    def test_interp_linear_default(self):
        """Test that interp defaults to 'linear'."""
        from neurospatial.animation.overlays import EventOverlay

        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        position_times = np.array([0.0, 1.0])
        event_times = np.array([0.5])

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
        )

        assert overlay.interp == "linear"

    def test_interp_nearest(self):
        """Test EventOverlay with interp='nearest'."""
        from neurospatial.animation.overlays import EventOverlay

        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        position_times = np.array([0.0, 1.0])
        event_times = np.array([0.5])

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
            interp="nearest",
        )

        assert overlay.interp == "nearest"


class TestSpikeOverlayAlias:
    """Test that SpikeOverlay is an alias for EventOverlay."""

    def test_spike_overlay_is_event_overlay(self):
        """Test that SpikeOverlay is the same as EventOverlay."""
        from neurospatial.animation.overlays import EventOverlay, SpikeOverlay

        assert SpikeOverlay is EventOverlay

    def test_spike_overlay_can_be_instantiated(self):
        """Test that SpikeOverlay can be instantiated like EventOverlay."""
        from neurospatial.animation.overlays import SpikeOverlay

        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        position_times = np.array([0.0, 1.0])
        spike_times = np.array([0.5])

        overlay = SpikeOverlay(
            event_times=spike_times,
            positions=positions,
            position_times=position_times,
            colors="red",
        )

        assert overlay.colors == "red"
        # event_times is normalized to dict format in __post_init__
        assert_array_equal(overlay.event_times["event"], spike_times)


class TestEventDataDataclass:
    """Test EventData internal container."""

    def test_event_data_creation(self):
        """Test creating EventData with all fields."""
        from neurospatial.animation.overlays import EventData

        event_positions = {
            "cell_001": np.array([[10.0, 20.0], [30.0, 40.0]]),
        }
        event_frame_indices = {
            "cell_001": np.array([0, 5]),
        }
        colors = {"cell_001": "#1f77b4"}
        markers = {"cell_001": "o"}

        event_data = EventData(
            event_positions=event_positions,
            event_frame_indices=event_frame_indices,
            colors=colors,
            markers=markers,
            size=8.0,
            decay_frames=0,
            border_color="white",
            border_width=0.5,
        )

        assert "cell_001" in event_data.event_positions
        assert "cell_001" in event_data.event_frame_indices
        assert event_data.colors["cell_001"] == "#1f77b4"
        assert event_data.markers["cell_001"] == "o"
        assert event_data.size == 8.0
        assert event_data.decay_frames == 0

    def test_event_data_is_pickle_safe(self):
        """Test that EventData can be pickled for parallel rendering."""
        import pickle

        from neurospatial.animation.overlays import EventData

        event_data = EventData(
            event_positions={"event": np.array([[10.0, 20.0]])},
            event_frame_indices={"event": np.array([0])},
            colors={"event": "#ff0000"},
            markers={"event": "o"},
            size=8.0,
            decay_frames=0,
            border_color="white",
            border_width=0.5,
        )

        # Should be pickle-able
        pickled = pickle.dumps(event_data)
        unpickled = pickle.loads(pickled)

        assert unpickled.colors["event"] == "#ff0000"
        assert_array_equal(
            unpickled.event_positions["event"], event_data.event_positions["event"]
        )


class TestOverlayDataEventsField:
    """Test that OverlayData has events field."""

    def test_overlay_data_has_events_attribute(self):
        """Test that OverlayData has events attribute."""
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData()

        assert hasattr(overlay_data, "events")
        assert isinstance(overlay_data.events, list)
        assert len(overlay_data.events) == 0

    def test_overlay_data_events_default_empty_list(self):
        """Test that events defaults to empty list."""
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData()

        assert overlay_data.events == []


class TestEventOverlayConvertToData:
    """Test EventOverlay.convert_to_data() method."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims: ClassVar[int] = 2
            dimension_ranges: ClassVar[list[tuple[float, float]]] = [
                (0.0, 100.0),
                (0.0, 100.0),
            ]

        return MockEnv()

    def test_convert_trajectory_mode_linear_interp(self, mock_env):
        """Test convert_to_data with trajectory mode and linear interpolation."""
        from neurospatial.animation.overlays import EventOverlay

        # Trajectory: (0,0) at t=0 -> (100,100) at t=10
        positions = np.array([[0.0, 0.0], [100.0, 100.0]])
        position_times = np.array([0.0, 10.0])
        # Events at t=2.5, t=5.0, t=7.5
        event_times = np.array([2.5, 5.0, 7.5])

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
        )

        frame_times = np.linspace(0.0, 10.0, 11)  # 11 frames, 0-10s
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Should have interpolated positions
        assert "event" in event_data.event_positions
        event_pos = event_data.event_positions["event"]

        # Event at t=2.5 should be at (25, 25) - 25% of way
        assert np.allclose(event_pos[0], [25.0, 25.0])
        # Event at t=5.0 should be at (50, 50) - 50% of way
        assert np.allclose(event_pos[1], [50.0, 50.0])
        # Event at t=7.5 should be at (75, 75) - 75% of way
        assert np.allclose(event_pos[2], [75.0, 75.0])

    def test_convert_trajectory_mode_nearest_interp(self, mock_env):
        """Test convert_to_data with trajectory mode and nearest interpolation."""
        from neurospatial.animation.overlays import EventOverlay

        positions = np.array([[0.0, 0.0], [100.0, 100.0]])
        position_times = np.array([0.0, 10.0])
        event_times = np.array([2.5])

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
            interp="nearest",
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        event_pos = event_data.event_positions["event"]
        # With nearest, t=2.5 should snap to t=0 (closer than t=10)
        assert np.allclose(event_pos[0], [0.0, 0.0])

    def test_convert_explicit_positions_mode(self, mock_env):
        """Test convert_to_data with explicit positions mode."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 5.0, 9.5])
        event_positions = np.array([[10.0, 20.0], [50.0, 60.0], [90.0, 80.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Positions should be used directly
        assert_array_equal(event_data.event_positions["event"], event_positions)

    def test_convert_single_position_broadcast(self, mock_env):
        """Test that single position is broadcast to all events."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([1.0, 2.0, 3.0])  # 3 events
        event_positions = np.array([[50.0, 25.0]])  # Single position

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Should have 3 events all at (50, 25)
        event_pos = event_data.event_positions["event"]
        assert event_pos.shape == (3, 2)
        assert np.allclose(event_pos[0], [50.0, 25.0])
        assert np.allclose(event_pos[1], [50.0, 25.0])
        assert np.allclose(event_pos[2], [50.0, 25.0])

    def test_convert_frame_indices_correct(self, mock_env):
        """Test that frame indices are computed correctly."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([0.5, 2.5, 5.0, 7.5, 9.5])
        event_positions = np.array(
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]]
        )

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        # Frame times at 0, 1, 2, ..., 10 (1s intervals)
        frame_times = np.arange(0.0, 11.0, 1.0)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Event at 0.5s -> frame 0 or 1 (nearest)
        # Event at 2.5s -> frame 2 or 3 (nearest)
        # Event at 5.0s -> frame 5 (exact)
        # Event at 7.5s -> frame 7 or 8 (nearest)
        # Event at 9.5s -> frame 9 or 10 (nearest)
        frame_indices = event_data.event_frame_indices["event"]
        assert len(frame_indices) == 5
        assert frame_indices[2] == 5  # Exact match

    def test_convert_auto_colors_tab10(self, mock_env):
        """Test that auto colors are assigned from tab10 colormap."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {
            "cell_001": np.array([1.0]),
            "cell_002": np.array([2.0]),
            "cell_003": np.array([3.0]),
        }
        event_positions = {
            "cell_001": np.array([[10.0, 10.0]]),
            "cell_002": np.array([[20.0, 20.0]]),
            "cell_003": np.array([[30.0, 30.0]]),
        }

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            colors=None,  # Auto-assign
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Should have distinct colors for each type
        assert "cell_001" in event_data.colors
        assert "cell_002" in event_data.colors
        assert "cell_003" in event_data.colors
        # Colors should be strings (hex format)
        assert isinstance(event_data.colors["cell_001"], str)
        # Colors should be different
        colors = list(event_data.colors.values())
        assert len(set(colors)) == 3  # All distinct

    def test_convert_custom_colors_dict(self, mock_env):
        """Test that custom colors dict is used."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {"a": np.array([1.0]), "b": np.array([2.0])}
        event_positions = {"a": np.array([[10.0, 10.0]]), "b": np.array([[20.0, 20.0]])}
        colors = {"a": "red", "b": "blue"}

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            colors=colors,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert event_data.colors["a"] == "red"
        assert event_data.colors["b"] == "blue"

    def test_convert_single_color_string(self, mock_env):
        """Test that single color string applies to all event types."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = {"a": np.array([1.0]), "b": np.array([2.0])}
        event_positions = {"a": np.array([[10.0, 10.0]]), "b": np.array([[20.0, 20.0]])}

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            colors="green",  # Single color for all
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert event_data.colors["a"] == "green"
        assert event_data.colors["b"] == "green"

    def test_convert_decay_frames_preserved(self, mock_env):
        """Test that decay_frames value is preserved in EventData."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([1.0])
        event_positions = np.array([[10.0, 10.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            decay_frames=5,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert event_data.decay_frames == 5

    def test_convert_decay_frames_none_becomes_zero(self, mock_env):
        """Test that decay_frames=None becomes 0 in EventData."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([1.0])
        event_positions = np.array([[10.0, 10.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
            decay_frames=None,  # Instant
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        assert event_data.decay_frames == 0


class TestEventOverlayEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims: ClassVar[int] = 2
            dimension_ranges: ClassVar[list[tuple[float, float]]] = [
                (0.0, 100.0),
                (0.0, 100.0),
            ]

        return MockEnv()

    def test_empty_event_times_array(self, mock_env):
        """Test EventOverlay with empty event times array."""
        from neurospatial.animation.overlays import EventOverlay

        event_times = np.array([])  # Empty
        event_positions = np.array([]).reshape(0, 2)  # Empty (0, 2)

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Should have empty arrays, not crash
        assert len(event_data.event_positions["event"]) == 0
        assert len(event_data.event_frame_indices["event"]) == 0

    def test_events_outside_frame_times_range_warns(self, mock_env):
        """Test that events outside frame_times range emit warning."""
        from neurospatial.animation.overlays import EventOverlay

        # Events at t=-1 and t=15, but frames only cover 0-10
        event_times = np.array([-1.0, 5.0, 15.0])
        event_positions = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        frame_times = np.linspace(0.0, 10.0, 11)  # 0 to 10
        n_frames = 11

        with pytest.warns(UserWarning, match="outside"):
            event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Only the event at t=5 should be included
        assert len(event_data.event_positions["event"]) == 1

    def test_events_outside_position_times_range_warns(self, mock_env):
        """Test that events outside position_times range emit warning (trajectory mode)."""
        from neurospatial.animation.overlays import EventOverlay

        positions = np.array([[0.0, 0.0], [100.0, 100.0]])
        position_times = np.array([0.0, 10.0])
        # Events at t=-1 and t=15, outside position_times
        event_times = np.array([-1.0, 5.0, 15.0])

        overlay = EventOverlay(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        with pytest.warns(UserWarning, match="outside"):
            event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)

        # Only the event at t=5 should be included
        assert len(event_data.event_positions["event"]) == 1


class TestEventOverlayPerformance:
    """Test performance with high event counts."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims: ClassVar[int] = 2
            dimension_ranges: ClassVar[list[tuple[float, float]]] = [
                (0.0, 100.0),
                (0.0, 100.0),
            ]

        return MockEnv()

    @pytest.mark.slow
    def test_high_event_count_performance(self, mock_env):
        """Test that 10,000+ events can be processed in reasonable time."""
        import time

        from neurospatial.animation.overlays import EventOverlay

        # 10,000 events
        n_events = 10_000
        rng = np.random.default_rng(42)
        event_times = np.sort(rng.uniform(0, 100, n_events))
        event_positions = rng.uniform(0, 100, (n_events, 2))

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        frame_times = np.linspace(0.0, 100.0, 1001)  # 1000 frames
        n_frames = 1001

        start = time.time()
        event_data = overlay.convert_to_data(frame_times, n_frames, mock_env)
        elapsed = time.time() - start

        # Should complete in less than 1 second
        assert elapsed < 1.0, f"convert_to_data took {elapsed:.2f}s, expected < 1s"
        assert len(event_data.event_positions["event"]) == n_events


class TestConvertOverlaysToDataWithEvents:
    """Test _convert_overlays_to_data() with EventOverlay."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims: ClassVar[int] = 2
            dimension_ranges: ClassVar[list[tuple[float, float]]] = [
                (0.0, 100.0),
                (0.0, 100.0),
            ]

        return MockEnv()

    def test_event_overlay_conversion_through_pipeline(self, mock_env):
        """Test that EventOverlay is properly dispatched in conversion pipeline."""
        from neurospatial.animation.overlays import (
            EventOverlay,
            _convert_overlays_to_data,
        )

        event_times = np.array([1.0, 2.0])
        event_positions = np.array([[10.0, 10.0], [20.0, 20.0]])

        overlay = EventOverlay(
            event_times=event_times,
            event_positions=event_positions,
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Should have events in the result
        assert len(overlay_data.events) == 1
        assert "event" in overlay_data.events[0].event_positions

    def test_multiple_event_overlays_conversion(self, mock_env):
        """Test converting multiple EventOverlay instances."""
        from neurospatial.animation.overlays import (
            EventOverlay,
            _convert_overlays_to_data,
        )

        overlay1 = EventOverlay(
            event_times=np.array([1.0]),
            event_positions=np.array([[10.0, 10.0]]),
            colors="red",
        )
        overlay2 = EventOverlay(
            event_times=np.array([2.0]),
            event_positions=np.array([[20.0, 20.0]]),
            colors="blue",
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        overlay_data = _convert_overlays_to_data(
            overlays=[overlay1, overlay2],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Should have 2 event overlays
        assert len(overlay_data.events) == 2

    def test_mixed_overlay_types_with_events(self, mock_env):
        """Test conversion of mixed overlay types including events."""
        from neurospatial.animation.overlays import (
            EventOverlay,
            PositionOverlay,
            _convert_overlays_to_data,
        )

        position_overlay = PositionOverlay(
            data=np.array([[10.0, 20.0], [30.0, 40.0]]),
            times=np.array([0.0, 10.0]),
        )
        event_overlay = EventOverlay(
            event_times=np.array([5.0]),
            event_positions=np.array([[50.0, 50.0]]),
        )

        frame_times = np.linspace(0.0, 10.0, 11)
        n_frames = 11

        overlay_data = _convert_overlays_to_data(
            overlays=[position_overlay, event_overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Should have both types
        assert len(overlay_data.positions) == 1
        assert len(overlay_data.events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
